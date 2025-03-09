import pandas as pd


def impute_interpolate(df: pd.DataFrame, ts_attr: str, limit: int = None) -> pd.DataFrame:
    df[ts_attr] = df[ts_attr].interpolate(method="linear", limit=limit, limit_direction="forward")
    return df


def impute_average(df: pd.DataFrame, ts_attr: str) -> pd.DataFrame:
    df[ts_attr] = df[ts_attr].fillna(df[ts_attr].mean())
    return df


def impute_zeros(df: pd.DataFrame, ts_attr: str) -> pd.DataFrame:
    df[ts_attr] = df[ts_attr].fillna(0)
    return df


def impute_missing_data(df: pd.DataFrame, ts_attr: str, times: pd.DataFrame, method: str) -> pd.DataFrame:
    df = df.merge(times, on="id_time", how="right")
    if method == "mean":
        df = impute_average(df, ts_attr)
    elif method == "interpolate":
        df = impute_interpolate(df, ts_attr)
    elif method == "zeros":
        df = impute_zeros(df, ts_attr)
    return df
