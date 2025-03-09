"""
This code is based off of the code from the paper Deep Learning with Long Short-Term Memory for Time Series Prediction,
with a lot of refactoring, extensions and adjustments.
Link: https://github.com/huajay1/RCLSTM

"""

import math
import time
from collections import OrderedDict

import numpy as np
import torch
from torch import nn, optim

from rclstm.rclstm import RNN
from src.RunnerComponent import RunnerComponent
from src.utils.constants import MODEL_PARAMS
from src.utils.evaluation_metrics import get_prediction_metrics


class RclstmRunner(RunnerComponent):

    def __init__(self, ts_attribute: str, args):
        super().__init__(ts_attribute, args)
        self.params = MODEL_PARAMS[args.model_name]
        self.connectivity = self.params["connectivity"]
        self.hidden_size = self.params["hidden_size"]
        self.batch_size = self.params["batch_size"]
        self.dropout = self.params["dropout"]
        self.num_layers = self.params["num_layers"]
        self.epochs = self.params["epochs"]
        self.patience = 5

    def shufflelists(self, X, Y):
        ri = np.random.permutation(len(X))
        X_shuffle = [X[i].tolist() for i in ri]
        Y_shuffle = [Y[i].tolist() for i in ri]
        return np.array(X_shuffle), np.array(Y_shuffle)

    def compute_loss_accuracy(self, model, loss_fn, data, label):
        hx = None
        _, (h_n, _) = model[0](input_=data, hx=hx)
        logits = model[1](h_n[-1])
        loss = torch.sqrt(loss_fn(input=logits, target=label))
        return loss, logits

    def exp_lr_scheduler(self, optimizer, epoch, init_lr=1e-2, lr_decay_epoch=3):
        lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))
        if epoch % lr_decay_epoch == 0:
            print("LR is set to {}".format(lr))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return optimizer

    def train(self, device, loss_fn, X_train, y_train, X_val, y_val):
        num_batch = int(math.ceil(len(X_train) // self.batch_size))
        num_val_batch = int(math.ceil(len(X_val) // self.batch_size))
        best_loss = float("inf")
        patience_counter = 0

        rnn_model = RNN(
            device=device,
            cell_class=self.MODEL_NAME,
            input_size=1,  # Number of attrs
            hidden_size=self.hidden_size,
            connectivity=self.connectivity,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout,
        )
        fc2 = nn.Linear(in_features=self.hidden_size, out_features=self.PREDICTION_WINDOW)
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
        while iter_cnt < self.epochs:
            train_inputs, train_targets = self.shufflelists(X_train, y_train)
            optimizer = self.exp_lr_scheduler(optim_method, iter_cnt, init_lr=0.01, lr_decay_epoch=3)
            for i in range(num_batch):
                low_index = self.batch_size * i
                high_index = self.batch_size * (i + 1)
                if low_index <= len(train_inputs) - self.batch_size:
                    batch_inputs = (
                        train_inputs[low_index:high_index]
                        .reshape(self.batch_size, self.TRAINING_WINDOW, 1)
                        .astype(np.float32)
                    )
                    batch_targets = (
                        train_targets[low_index:high_index]
                        .reshape((self.batch_size, self.PREDICTION_WINDOW))
                        .astype(np.float32)
                    )
                else:
                    batch_inputs = train_inputs[low_index:].astype(float)
                    batch_targets = train_targets[low_index:].astype(float)

                batch_inputs = torch.from_numpy(batch_inputs).to(device)
                batch_targets = torch.from_numpy(batch_targets).to(device)

                model.train(True)
                model.zero_grad()
                train_loss, logits = self.compute_loss_accuracy(
                    model=model, loss_fn=loss_fn, data=batch_inputs, label=batch_targets
                )
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
                        X_val[i * self.batch_size : (i + 1) * self.batch_size]
                        .reshape(self.batch_size, self.TRAINING_WINDOW, 1)
                        .astype(np.float32)
                    )
                    val_targets = (
                        y_val[i * self.batch_size : (i + 1) * self.batch_size]
                        .reshape((self.batch_size, 1))
                        .astype(np.float32)
                    )

                    val_inputs = torch.from_numpy(val_inputs).to(device)
                    val_targets = torch.from_numpy(val_targets).to(device)

                    val_loss, _ = self.compute_loss_accuracy(
                        model=model, loss_fn=loss_fn, data=val_inputs, label=val_targets
                    )
                    epoch_val_loss += val_loss.item()
            print(f"Epoch val loss is {epoch_val_loss}")

            # Early Stopping
            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print("Early stopping triggered")
                    break
            iter_cnt += 1
        training_time = time.time() - start
        return model, training_time

    def run(self):
        df = self.preprocess()
        X, y = self.create_input_data(df)
        X = np.swapaxes(X, 1, 2)

        X_train, X_val, X_test, y_train, y_val, y_test = self.split_and_normalize(X, y)

        device = "cpu"
        loss_fn = nn.MSELoss()

        model, training_time = self.train(device, loss_fn, X_train, y_train, X_val, y_test)

        # Test
        model.eval()
        with torch.no_grad():
            start = time.time()
            test_inputs = torch.from_numpy(X_test.reshape(len(X_test), self.TRAINING_WINDOW, 1).astype(np.float32)).to(
                device
            )
            test_targets = torch.from_numpy(y_test.reshape(len(y_test), self.PREDICTION_WINDOW).astype(np.float32)).to(
                device
            )
            test_loss, test_preds = self.compute_loss_accuracy(
                model=model, loss_fn=loss_fn, data=test_inputs, label=test_targets
            )
            testing_time = time.time() - start
            metrics = get_prediction_metrics(test_targets.cpu().numpy().flatten(), test_preds.cpu().numpy().flatten())
            self.create_record(metrics, self.params, testing_time, training_time)
            print(metrics)
