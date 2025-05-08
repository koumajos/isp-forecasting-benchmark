import os
from datetime import datetime
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.utils.constants import DATA_GROUPS
from src.utils.imputations import impute_missing_data


class RunnerComponent:

    def __init__(self, ts_attribute: str, args):
        self.ts_attribute = ts_attribute
        self.MAIN_DIR = args.main_dir
        self.AGGREGATION = args.aggregation
        self.FILE_ID = args.file_id
        self.TEST_SET_SIZE = args.test_set_size
        self.VALID_SET_SIZE = args.valid_set_size
        self.TRAINING_WINDOW = args.training_window
        self.MODEL_NAME = args.model_name
        self.IMPUTATION = args.impute
        self.PBS_JOB_ID = os.environ.get("PBS_JOBID")  # metacentrum variable
        self.PREDICTION_WINDOW = args.prediction_window
        self.anomaly_threshold = args.anomaly_threshold
        self.plot_anomalies = args.plot_anomalies
        if args.sample:
            self.GROUP = DATA_GROUPS[2]
        elif args.subnets:
            self.GROUP = DATA_GROUPS[0]
        else:
            self.GROUP = DATA_GROUPS[1]

    def create_record(self, metrics: dict, params: dict, prediction_time: float, training_time: float):
        record = pd.DataFrame(
            [
                {
                    "METRICS": metrics,
                    "TRAINING_TIME": training_time,
                    "PREDICTION_TIME": prediction_time,
                    "PBS_JOB_ID": self.PBS_JOB_ID,
                    "FILE_ID": self.FILE_ID,
                    "TEST_SET_SIZE": self.TEST_SET_SIZE,
                    "VALID_SET_SIZE": self.VALID_SET_SIZE,
                    "TRAINING_WINDOW": self.TRAINING_WINDOW,
                    "TS_ATTRIBUTE": self.ts_attribute,
                    "MODEL_NAME": self.MODEL_NAME,
                    "GROUP": self.GROUP,
                    "IMPUTATION": self.IMPUTATION,
                    "PREDICTION_WINDOW": self.PREDICTION_WINDOW,
                    "PARAMS": params,
                    "TIMESTAMP": datetime.now(),
                }
            ]
        )
        try:
            record.to_csv(
                f"{self.MAIN_DIR}/output/records/{self.GROUP}/{self.AGGREGATION}/{self.MODEL_NAME}/record_{self.PBS_JOB_ID}_{self.ts_attribute}_{self.TRAINING_WINDOW}_{self.PREDICTION_WINDOW}_{self.FILE_ID}.csv"
            )
        except OSError:
            os.makedirs(f"{self.MAIN_DIR}/output/records/{self.GROUP}/{self.AGGREGATION}/{self.MODEL_NAME}")
            record.to_csv(
                f"{self.MAIN_DIR}/output/records/{self.GROUP}/{self.AGGREGATION}/{self.MODEL_NAME}/record_{self.PBS_JOB_ID}_{self.ts_attribute}_{self.TRAINING_WINDOW}_{self.PREDICTION_WINDOW}_{self.FILE_ID}.csv"
            )

    def get_test_set_index(self, X) -> int:
        return int(len(X) * (1 - self.TEST_SET_SIZE))

    def get_val_set_index(self, X) -> int:
        return self.get_test_set_index(X) - round(len(X) * self.VALID_SET_SIZE)

    def get_anomaly_test_set_indices(self, original, pred) -> np.array:
        anomalies = abs(np.array(original) - np.array(pred)) > self.anomaly_threshold
        anomaly_indices = np.where(anomalies)[0]
        return anomaly_indices

    def create_input_data(self, data: pd.DataFrame) -> (np.array, np.array):
        """
        Convert data to numpy arrays. Sliding window of PREDICTION_WINDOW is applied.
        """
        X, y = [], []
        for i in range(
            self.TRAINING_WINDOW,
            len(data) - self.PREDICTION_WINDOW,
            self.PREDICTION_WINDOW,
        ):
            X.append(data[[self.ts_attribute]].iloc[i - self.TRAINING_WINDOW : i].values)
            y.append(data[[self.ts_attribute]].iloc[i : i + self.PREDICTION_WINDOW].values.flatten())
        X = np.array(X)
        y = np.array(y)
        return X, y

    def preprocess(self) -> pd.DataFrame:
        """
        Read data and impute missing values.
        """
        df = pd.read_csv(f"{self.MAIN_DIR}/{self.GROUP}/{self.AGGREGATION}/{self.FILE_ID}.csv")
        times = pd.read_csv(f"{self.MAIN_DIR}/{self.GROUP}/{self.AGGREGATION}/times.csv")
        df = df[[self.ts_attribute, "id_time"]]
        df = impute_missing_data(df=df, times=times, ts_attr=self.ts_attribute, method=self.IMPUTATION)
        return df

    def split_and_normalize(self, X, y):
        test_set_first_index = self.get_test_set_index(X)
        val_set_first_index = self.get_val_set_index(X)

        # Split data and normalize
        X_train_val = X[:test_set_first_index, :, :]
        X_test = X[test_set_first_index:, :, :]
        y_train_val = y[:test_set_first_index, :]
        y_test = y[test_set_first_index:, :]  # add option for multivariate

        # Reshape for easy fit
        X_train_val_1d = X_train_val.reshape(-1, 1)
        y_train_val_1d = y_train_val.reshape(-1, 1)
        X_test_1d = X_test.reshape(-1, 1)
        y_test_1d = y_test.reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(X_train_val_1d)

        # Transform
        X_train_val_1d = scaler.transform(X_train_val_1d)
        X_test_1d = scaler.transform(X_test_1d)
        y_train_val_1d = scaler.transform(y_train_val_1d)
        y_test_1d = scaler.transform(y_test_1d)

        # Reshape back
        X_train_val = X_train_val_1d.reshape(X_train_val.shape)
        X_test = X_test_1d.reshape(X_test.shape)
        y_train_val = y_train_val_1d.reshape(y_train_val.shape)
        y_test = y_test_1d.reshape(y_test.shape)

        X_train = X_train_val[:val_set_first_index, :, :]
        X_val = X_train_val[val_set_first_index:, :, :]
        y_train = y_train_val[:val_set_first_index, :]
        y_val = y_train_val[val_set_first_index:, :]

        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_test_set_time_frame(self, data_len):
        df_times = pd.read_csv(f"{self.MAIN_DIR}/{self.GROUP}/{self.AGGREGATION}/times.csv")
        df_times["time"] = pd.to_datetime(df_times["time"])
        # Test set is always X last samples in the dataset
        return df_times["time"].iloc[df_times.shape[0] - data_len :]

    def create_plot(self, original_values, anomaly_indices):
        # Set up a wider figure
        plt.figure(figsize=(14, 5))
        ax1 = plt.gca()  # Get current axis

        times = self.get_test_set_time_frame(len(original_values))
        # Plotting values
        ax1.plot(times, original_values, alpha=1)

        # Highlight anomalies
        anomaly_times = times.iloc[anomaly_indices]
        anomaly_values = original_values[anomaly_indices]
        ax1.scatter(anomaly_times, anomaly_values, color="red", label="Anomalies", zorder=5)

        # Add legend, title, and rotate x-axis labels
        ax1.legend([self.ts_attribute, "Anomalies"], fontsize=18)
        ax1.set_title("Anomalies detected", fontsize=18)
        ax1.xaxis.set_tick_params(rotation=0, labelsize=14)
        ax1.tick_params(axis="y", labelsize=14)
        # Final adjustments
        plt.tight_layout()
        plt.savefig(f"{self.MODEL_NAME}-anomalies.png")
