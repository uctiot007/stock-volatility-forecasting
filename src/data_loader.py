import os
import pandas as pd
import yfinance as yf

# This builds the path to your project's root folder automatically,
# so the script works no matter where you run it from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")


def fetch_data(ticker: str = "^GSPC", start: str = "2015-01-01", end: str = "2023-01-01") -> pd.DataFrame:
    """
    Download historical price/volume data from Yahoo Finance.
    """
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    assert data is not None, "yf.download returned None — check your internet connection or ticker symbol"

    if data.empty:
        raise ValueError(f"No data returned for ticker '{ticker}' between {start} and {end}.")

    # yfinance sometimes returns MultiIndex columns like ('Close', '^GSPC')
    # This flattens them to just 'Close', 'Volume', etc.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the columns we need and drop any missing rows.
    """
    required_cols = ["Close", "Volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected columns: {missing}")

    df = df.loc[:, required_cols].copy()
    df = df.dropna()

    return df


def save_data(df: pd.DataFrame, filename: str = "sp500_raw.csv") -> None:
    """
    Save a dataframe as CSV into data/raw/.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    path = os.path.join(RAW_DATA_DIR, filename)
    df.to_csv(path)
    print(f"Data saved to {path}")


def load_data(filename: str = "sp500_raw.csv") -> pd.DataFrame:
    """
    Load a previously saved CSV from data/raw/.
    """
    path = os.path.join(RAW_DATA_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist. Run get_data() first.")

    return pd.read_csv(path, index_col=0, parse_dates=True)


def get_data(ticker: str = "^GSPC", start: str = "2015-01-01", end: str = "2023-01-01") -> pd.DataFrame:
    """
    Full pipeline: fetch -> clean -> save -> return.
    """
    df = fetch_data(ticker, start, end)
    df = clean_data(df)
    save_data(df)
    return df


if __name__ == "__main__":
    df = get_data()
    print(df.head())
    print(df.shape)