import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def smape(A, F):
    return (100 / len(A)) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# Defined metrics
metrics_functions = {"rmse": rmse, "sMAPE": smape, "R2": r2_score}


def get_prediction_metrics(y_true, y_pred):
    res = {}
    for metric_name, metric_function in metrics_functions.items():
        try:
            res[metric_name] = metric_function(y_true, y_pred)
        except Exception as e:
            res[metric_name] = None

    return res
