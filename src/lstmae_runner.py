import math
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from torch import nn, optim

from sklearn.preprocessing import MinMaxScaler
from src.runner_component import RunnerComponent
from src.utils.constants import MODEL_PARAMS
from torch.utils.data import DataLoader, TensorDataset
from src.lstmae.lstmae import LSTMAutoencoder
from src.utils.evaluation_metrics import get_prediction_metrics


class LstmaeRunner(RunnerComponent):

    def __init__(self, ts_attribute: str, args):
        super().__init__(ts_attribute, args)
        self.params = MODEL_PARAMS[args.model_name][self.TRAINING_WINDOW]
        self.hidden_size = self.params["hidden_size"]
        self.batch_size = self.params["batch_size"]
        self.num_layers = self.params["num_layers"]
        self.epochs = self.params["epochs"]
        self.patience = self.params["patience"]
        self.learning_rate = self.params["lr"]

    def get_last_samples_only(self, reconstructed, original):
        """
        Process original data and reconstructed for metric calculation.
        We only calculate metrics using the last sample of the output.
        """
        all_rec_first = [[x[-1] for x in batch] for batch in reconstructed]
        all_recs = []
        for r in all_rec_first:
            for el in r:
                all_recs.append(el.numpy()[0])

        all_org_first = [[x[-1] for x in batch] for batch in original]
        all_org = []
        for r in all_org_first:
            for el in r:
                all_org.append(el.numpy()[0])

        # return np.array(all_recs), np.array(all_org)
        all_recs = [x[-1].numpy()[0] for batch in reconstructed for x in batch]
        all_org = [x[-1].numpy()[0] for batch in original for x in batch]
        return np.array(all_org), np.array(all_recs)

    def create_lags(self, df) -> np.ndarray:
        data = df[self.ts_attribute].values
        df = pd.DataFrame({"temp": data})
        for i in range(self.TRAINING_WINDOW):
            df["Lag" + str(i + 1)] = df.temp.shift(i + 1)
        df = df.dropna()
        x = df.iloc[:, 1:].values
        return x

    def split_and_normalize(self, X, y=None):
        test_set_first_index = self.get_test_set_index(X)
        val_set_first_index = self.get_val_set_index(X)

        X_train_val = X[:test_set_first_index, :]
        X_test = X[test_set_first_index:, :]

        X_train_val_1d = X_train_val.reshape(-1, 1)
        X_test_1d = X_test.reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(X_train_val_1d)

        X_train_val_1d = scaler.transform(X_train_val_1d)
        X_test_1d = scaler.transform(X_test_1d)
        X_train_val = X_train_val_1d.reshape(X_train_val.shape)
        X_test = X_test_1d.reshape(X_test.shape)
        X_train = X_train_val[:val_set_first_index, :]
        X_val = X_train_val[val_set_first_index:, :]

        return X_train, X_val, X_test

    def create_data_loaders(self, train_data, val_data, test_data):
        train_data = train_data[..., np.newaxis]
        val_data = val_data[..., np.newaxis]
        test_data = test_data[..., np.newaxis]

        train_dataset = TensorDataset(torch.Tensor(train_data))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_dataset = TensorDataset(torch.Tensor(val_data))
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        test_dataset = TensorDataset(torch.Tensor(test_data))
        test_loader = DataLoader(test_dataset, batch_size=1)
        return train_loader, val_loader, test_loader

    def train(self, loss_fn, train_loader, val_loader):
        model = LSTMAutoencoder(1, self.hidden_size, num_layers=self.num_layers)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        best_val_loss = float("inf")
        patience_counter = 0

        start = time.time()
        # Training loop
        for epoch in range(self.epochs):
            model.train()
            running_loss = 0.0
            for batch in train_loader:
                inputs = batch[0]

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                reconstructed, _ = model(inputs)

                # Compute reconstruction loss
                loss = loss_fn(reconstructed, inputs)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # Print average loss for each epoch
            avg_train_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {running_loss / len(train_loader)}")

            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch[0]

                    # Forward pass
                    reconstructed, _ = model(inputs)

                    # Compute reconstruction loss
                    loss = loss_fn(reconstructed, inputs)
                    val_loss += loss.item()

            # Calculate average validation loss for this epoch
            avg_val_loss = val_loss / len(val_loader)

            print(
                f"Epoch [{epoch + 1}/{self.epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}"
            )

            # Early stopping checks based on validation loss
            if best_val_loss > avg_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter > self.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        train_time = time.time() - start
        print(f"Training time {train_time}")
        return model, train_time

    def run(self):
        df = self.preprocess()
        X = self.create_lags(df)

        X_train, X_val, X_test = self.split_and_normalize(X)
        train_loader, val_loader, test_loader = self.create_data_loaders(X_train, X_val, X_test)

        loss_fn = nn.MSELoss()

        model, training_time = self.train(loss_fn, train_loader, val_loader)

        # Test loop
        start = time.time()
        model.eval()
        all_reconstructed = []
        all_original = []
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch[0]

                # Forward pass
                reconstructed, _ = model(inputs)
                all_original.append(inputs)
                all_reconstructed.append(reconstructed)

        testing_time = time.time() - start
        original_data, reconstructed_data = self.get_last_samples_only(all_reconstructed, all_original)
        metrics = get_prediction_metrics(original_data, reconstructed_data)
        self.create_record(metrics, self.params, testing_time, training_time)
        print(metrics)
        anomaly_indices = self.get_anomaly_test_set_indices(original_data, reconstructed_data)
        print(f"Identified {len(anomaly_indices)} anomalies")
        if self.plot_anomalies:
            self.create_plot(original_data, anomaly_indices)
