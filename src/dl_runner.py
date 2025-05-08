import time

import numpy as np
from fastai.callback.tracker import EarlyStoppingCallback
from tsai.all import (
    Learner,
    TSDataLoaders,
    TSDatasets,
    TSRegression,
    combine_split_data,
    create_model,
    mae,
    nn,
    rmse,
)

from src.runner_component import RunnerComponent
from src.utils.constants import MODEL_MAPPER, MODEL_PARAMS
from src.utils.evaluation_metrics import get_prediction_metrics


class DLRunner(RunnerComponent):

    def __init__(self, ts_attribute: str, args):
        super().__init__(ts_attribute, args)

    def train(self, X, y, splits, params) -> (Learner, float):
        batch_size = params["batch_size"]
        tfms = [None, [TSRegression()]]
        dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)
        dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[batch_size, batch_size], num_workers=0)

        # Load relevant params for model set it up.
        arch = MODEL_MAPPER[self.MODEL_NAME]
        if self.MODEL_NAME in ["LSTM_FCN", "GRU_FCN"]:
            k = {
                "rnn_layers": params["rnn_layers"],
                "hidden_size": params["hidden_size"],
                "bidirectional": params["bidirectional"],
                "rnn_dropout": params["rnn_dropout"],
                "conv_layers": params["conv_layers"],
                "kss": params["kss"],
                "fc_dropout": params["fc_dropout"],
            }
        elif self.MODEL_NAME == "InceptionTime":
            k = {"nf": params["nf"], "ks": params["ks"]}
        elif self.MODEL_NAME in ["LSTM", "GRU"]:
            k = {
                "n_layers": params["n_layers"],
                "hidden_size": params["hidden_size"],
                "bidirectional": params["bidirectional"],
            }
        else:
            k = {}
        k["c_out"] = self.PREDICTION_WINDOW
        model = create_model(arch, dls=dls, **k)
        model = nn.Sequential(model, nn.Sigmoid())

        # Set up learner and track time
        learn = Learner(dls, model, metrics=[mae, rmse], opt_func=params["optimizer"])
        start = time.time()
        learn.fit_one_cycle(
            params["epochs"],
            lr_max=params["lr"],
            cbs=EarlyStoppingCallback(monitor="valid_loss", min_delta=0.0, patience=params["patience"]),
        )
        training_time = time.time() - start
        print(f"Training time: {training_time}")

        return learn, training_time

    def run(self):
        df = self.preprocess()
        X, y = self.create_input_data(df)
        X = np.swapaxes(X, 1, 2)

        X_train, X_val, X_test, y_train, y_val, y_test = self.split_and_normalize(X, y)

        # Train
        params = MODEL_PARAMS[self.MODEL_NAME]
        X, y, splits = combine_split_data([X_train, X_val], [y_train, y_val])
        learn, training_time = self.train(X, y, splits, params)

        # Test
        dls = learn.dls
        val_dl = dls.valid
        test_ds = val_dl.dataset.add_test(X_test, y_test)  # use the test data
        test_dl = val_dl.new(test_ds)

        start = time.time()
        _, test_targets, test_preds = learn.get_preds(dl=test_dl, with_decoded=True, save_preds=None, save_targs=None)
        prediction_time = time.time() - start
        print("Prediction time:", prediction_time)

        # Evaluate predictions
        y_true = test_targets.numpy().flatten()
        y_pred = test_preds.numpy().flatten()
        metrics = get_prediction_metrics(y_true, y_pred)
        print(metrics)

        # Create file with record
        self.create_record(
            metrics=metrics,
            params=params,
            prediction_time=prediction_time,
            training_time=training_time,
        )
        anomaly_indices = self.get_anomaly_test_set_indices(y_true, y_pred)
        print(f"Identified {len(anomaly_indices)} anomalies")
        if self.plot_anomalies:
            self.create_plot(y_true, anomaly_indices)
