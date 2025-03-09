import argparse
import math
import os
import time
from collections import OrderedDict
from datetime import datetime
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
from constants import DATA_GROUPS
from fastai.callback.tracker import EarlyStoppingCallback
from imputations import impute_missing_data
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from torch import nn, optim

# from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm

import rclstm.rclstm
from rclstm.rclstm import RNN, LSTMCell, RCLSTMCell


def create_record(metrics, params, prediction_time, training_time, connectivity, column_now=None):
    params = params.__dict__
    params["timestamp"] = datetime.now()
    record = pd.DataFrame(
        [
            {
                "metrics": metrics,
                "Training Time": training_time,
                "Prediction Time": prediction_time,
                "params": params,
                "connectivity": connectivity,
                "file path": f"{main_dir}/output/{group}/{aggregation}/RCLSTM/rclsm/{file_id}.csv",
                "timestamp": datetime.now(),
                "column": column_now,
            }
        ]
    )
    try:
        record.to_csv(f"{main_dir}/output/records/{group}/{aggregation}/RCLSTM/rclsm/record_{PBS_JOB_ID}_{col}.csv")
    except OSError:
        os.makedirs(f"{main_dir}/output/records/{group}/{aggregation}/RCLSTM/rclsm")
        record.to_csv(f"{main_dir}/output/records/{group}/{aggregation}/RCLSTM/rclsm/record_{PBS_JOB_ID}_{col}.csv")


np.random.seed(1000)


def get_args(parser):
    parser.add_argument("--main_dir", type=str, default="./data", help="Main directory for data.")
    parser.add_argument(
        "--aggregation",
        type=str,
        default="agg_1_hour",
        choices=["agg_1_day", "agg_1_hour", "agg_10_minutes"],
        help="Aggregation method.",
    )
    parser.add_argument("--file_id", type=int, default=25, help="File ID.")
    parser.add_argument(
        "--subnets",
        action="store_true",
        default=False,
        help="Flag to indicate if subnets should be used.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        default=False,
        help="Flag to indicate if sample dataset should be used.",
    )
    parser.add_argument("--test-set-size", type=float, default=0.6, help="Test set size.")
    parser.add_argument("--valid-set-size", type=float, default=0.05, help="Validation set size.")
    parser.add_argument("--training-window", type=int, default=24, help="Window size.")
    parser.add_argument(
        "--ts-attributes",
        nargs="+",
        default=[],
        help="Time series attributes to use for training.",
    )
    parser.add_argument("--model-name", type=str, default="LSTM_FCN", help="Model name.")
    parser.add_argument("--impute", type=str, default="mean", help="Imputation method.")
    parser.add_argument("--prediction-window", default=1, type=int, help="The size of output data")

    # parser.add_argument('--connectivity', type=float, default=.5, help='the neural connectivity')
    # parser.add_argument('--hidden-size', type=int, default=300, help='The number of hidden units')
    # parser.add_argument('--batch-size', type=int, default=32, help='The size of each batch')
    # parser.add_argument('--max-iter', type=int, default=100, help='The maximum iteration count')
    # parser.add_argument('--dropout', type=float, default=.5, help='Dropout')
    # parser.add_argument('--num-layers', type=int, default=1, help='The number of RNN layers')
    return parser


parser = argparse.ArgumentParser()
parser = get_args(parser)
args = parser.parse_args()
print(args)

# data_path = args.data
model_name = args.model
hidden_size = args.hidden_size
batch_size = args.batch_size
max_iter = args.max_iter
use_gpu = args.gpu
# connectivity = args.connectivity
training_window = args.time_window
input_size = args.input_size
dropout = args.dropout
num_layers = args.num_layers
is_csv = args.is_csv
column_to_use = [args.column_to_use]
prediction_window = args.output_size
sample = args.sample
subnets = args.subnets
if sample:
    group = DATA_GROUPS[2]
elif subnets:
    group = DATA_GROUPS[0]
else:
    group = DATA_GROUPS[1]

PBS_JOB_ID = os.environ.get("PBS_JOBID")
IMPUTATION = args.impute
main_dir = args.main_dir
aggregation = args.aggregation
file_id = args.file_id

patience = 5


def shufflelists(X, Y):
    ri = np.random.permutation(len(X))
    X_shuffle = [X[i].tolist() for i in ri]
    Y_shuffle = [Y[i].tolist() for i in ri]
    return np.array(X_shuffle), np.array(Y_shuffle)


# define function for create N lags
# def create_lags(df, N):
#     for i in range(N):
#         df['Lag' + str(i + 1)] = df.temp.shift(i + 1)
#     return df


def create_jumping_windows(data, training_window, prediction_window):
    """
    Create dataset such that:
      - Each sample (X) has length `time_window`.
      - Each target (Y) has length `output_size`.
      - The dataset moves ahead by only `output_size` each time,
        so consecutive windows overlap by (time_window - output_size).
    """
    X, Y = [], []
    idx = 0
    while idx + training_window + prediction_window <= len(data):
        X.append(data[idx : idx + training_window])
        Y.append(data[idx + training_window : idx + training_window + prediction_window])
        # Only increment by `output_size` (NOT time_window + output_size).
        idx += prediction_window
    return np.array(X), np.array(Y)


def compute_loss_accuracy(loss_fn, data, label):
    hx = None
    _, (h_n, _) = model[0](input_=data, hx=hx)
    logits = model[1](h_n[-1])
    loss = torch.sqrt(loss_fn(input=logits, target=label))
    return loss, logits


# learning rate decay
def exp_lr_scheduler(optimizer, epoch, init_lr=1e-2, lr_decay_epoch=3):
    lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))
    if epoch % lr_decay_epoch == 0:
        print("LR is set to {}".format(lr))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return optimizer


all_columns = [
    "n_flows",
    "n_packets",
    "n_bytes",
    "sum_n_dest_asn",
    "average_n_dest_asn",
    "std_n_dest_asn",
    "sum_n_dest_ports",
    "average_n_dest_ports",
    "std_n_dest_ports",
    "sum_n_dest_ip",
    "average_n_dest_ip",
    "std_n_dest_ip",
    "tcp_udp_ratio_packets",
    "tcp_udp_ratio_bytes",
    "dir_ratio_packets",
    "dir_ratio_bytes",
    "avg_duration",
    "avg_ttl",
]


if column_to_use != [None]:
    all_columns = column_to_use

for col in all_columns:
    best_loss = float("inf")
    patience_counter = 0

    times = pd.read_csv(f"{main_dir}/data_new/{group}/{aggregation}/times.csv")
    data = pd.read_csv(f"{main_dir}/data_new/{group}/{aggregation}/{file_id}.csv")
    df = data[[col, "id_time"]]
    df = impute_missing_data(df=df, times=times, cols=[col], method=IMPUTATION)

    data = df[col].values

    df = pd.DataFrame({"temp": data})

    X, y = create_jumping_windows(df["temp"].values, training_window, prediction_window)

    # ---------------------------------------------------------
    # Train/val/test split using length of X
    # ---------------------------------------------------------
    n_total = len(X)
    train_idx = self.get_test_set_index(X)
    val_idx = int(n_total * 0.35)

    # create train and test data
    train_X, train_Y, val_X, val_Y, test_X, test_Y = (
        X[:train_idx],
        y[:train_idx],
        X[train_idx:val_idx],
        y[train_idx:val_idx],
        X[val_idx:],
        y[val_idx:],
    )

    max_data = np.max(train_X)
    min_data = np.min(train_X)

    train_X = (train_X - min_data) / (max_data - min_data)
    val_X = (val_X - min_data) / (max_data - min_data)
    test_X = (test_X - min_data) / (max_data - min_data)
    train_Y = (train_Y - min_data) / (max_data - min_data)
    val_Y = (val_Y - min_data) / (max_data - min_data)
    test_Y = (test_Y - min_data) / (max_data - min_data)

    print("the number of train data: ", len(train_X))
    print("the number of test data: ", len(test_X))
    print("the shape of input: ", train_X.shape)
    print("the shape of target: ", train_Y.shape)

    device = "cpu"
    loss_fn = nn.MSELoss()
    num_batch = int(math.ceil(len(train_X) // batch_size))
    num_val_batch = int(math.ceil(len(val_X) // batch_size))

    # train RCLSTM with different neural connection ratio
    for connectivity in [0.01]:
        print("neural connection ratio:", connectivity)
        if model_name in ["lstm", "rclstm"]:
            rnn_model = RNN(
                device=device,
                cell_class=model_name,
                input_size=input_size,
                hidden_size=hidden_size,
                connectivity=connectivity,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        else:
            raise ValueError
        fc2 = nn.Linear(in_features=hidden_size, out_features=output_size)
        model = nn.Sequential(
            OrderedDict(
                [
                    ("rnn", rnn_model),
                    ("fc2", fc2),
                ]
            )
        )

        model.to(device)

        optim_method = optim.Adam(params=model.parameters())

        start = time.time()
        iter_cnt = 0
        while iter_cnt < max_iter:
            train_inputs, train_targets = shufflelists(train_X, train_Y)
            optimizer = exp_lr_scheduler(optim_method, iter_cnt, init_lr=0.01, lr_decay_epoch=3)
            for i in range(num_batch):
                low_index = batch_size * i
                high_index = batch_size * (i + 1)
                if low_index <= len(train_inputs) - batch_size:
                    batch_inputs = (
                        train_inputs[low_index:high_index].reshape(batch_size, time_window, 1).astype(np.float32)
                    )
                    batch_targets = (
                        train_targets[low_index:high_index].reshape((batch_size, output_size)).astype(np.float32)
                    )
                else:
                    batch_inputs = train_inputs[low_index:].astype(float)
                    batch_targets = train_targets[low_index:].astype(float)

                batch_inputs = torch.from_numpy(batch_inputs).to(device)
                batch_targets = torch.from_numpy(batch_targets).to(device)

                model.train(True)
                model.zero_grad()
                train_loss, logits = compute_loss_accuracy(loss_fn=loss_fn, data=batch_inputs, label=batch_targets)
                train_loss.backward()
                optimizer.step()

                if i % 20 == 0:
                    print("the %dth iter, the %dth batch, train loss is %.4f" % (iter_cnt, i, train_loss.item()))

            # Validation Loop
            model.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for i in range(num_val_batch):
                    val_inputs = (
                        val_X[i * batch_size : (i + 1) * batch_size]
                        .reshape(batch_size, time_window, 1)
                        .astype(np.float32)
                    )
                    val_targets = (
                        val_Y[i * batch_size : (i + 1) * batch_size].reshape((batch_size, 1)).astype(np.float32)
                    )

                    val_inputs = torch.from_numpy(val_inputs).to(device)
                    val_targets = torch.from_numpy(val_targets).to(device)

                    val_loss, _ = compute_loss_accuracy(loss_fn=loss_fn, data=val_inputs, label=val_targets)
                    epoch_val_loss += val_loss.item()
            print(f"Epoch val loss is {epoch_val_loss}")
            # Early Stopping
            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                patience_counter = 0  # Reset patience counter
                # Optionally, you can save the model state if it's the best performance so far
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break
            iter_cnt += 1
        training_time = time.time() - start

        model.eval()
        with torch.no_grad():
            start = time.time()
            test_inputs = torch.from_numpy(test_X.reshape(len(test_X), time_window, 1).astype(np.float32)).to(device)
            test_targets = torch.from_numpy(test_Y.reshape(len(test_Y), output_size).astype(np.float32)).to(device)
            test_loss, test_preds = compute_loss_accuracy(loss_fn=loss_fn, data=test_inputs, label=test_targets)
            testing_time = time.time() - start
            from evaluation_metrics import get_prediction_metrics

            metrics = get_prediction_metrics(test_targets.cpu().numpy().flatten(), test_preds.cpu().numpy().flatten())
            create_record(metrics, args, testing_time, training_time, connectivity, col)
            print(metrics)
