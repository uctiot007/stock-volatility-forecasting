import numpy as np
import pandas as pd


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns.

    Formula:
        r_t = log(P_t / P_{t-1})
    """
    df = df.copy()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    return df


def compute_volatility(df: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """
    Compute rolling realized volatility (annualized).

    Formula:
        vol = std(log_returns) * sqrt(252)
    """
    df = df.copy()
    df["volatility"] = (
        df["log_return"]
        .rolling(window=window)
        .std()
        * np.sqrt(252)
    )
    return df


def create_features(df: pd.DataFrame, n_lags: int = 5) -> pd.DataFrame:
    """
    Create lagged volatility features: vol_lag_1 ... vol_lag_n
    """
    df = df.copy()

    for i in range(1, n_lags + 1):
        df[f"vol_lag_{i}"] = df["volatility"].shift(i)

    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Target = next day's volatility.
    """
    df = df.copy()
    df["target"] = df["volatility"].shift(-1)
    return df


def prepare_dataset(df: pd.DataFrame, n_lags: int = 5):
    """
    Full feature pipeline: raw prices -> X, y ready for model.

    Returns:
        X (np.ndarray): feature matrix
        y (np.ndarray): target vector
        df (pd.DataFrame): full dataframe with all intermediate columns
    """
    df = compute_log_returns(df)
    df = compute_volatility(df)
    df = create_features(df, n_lags)
    df = create_target(df)

    # Drop rows with NaNs created by shifting/rolling
    df = df.dropna()

    feature_cols = [f"vol_lag_{i}" for i in range(1, n_lags + 1)]

    X = df[feature_cols].values
    y = df["target"].values

    return X, y, df


if __name__ == "__main__":
    from src.data_loader import load_data

    raw = load_data()
    X, y, full_df = prepare_dataset(raw)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print(full_df.head())