import pandas as pd
import numpy as np


FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Return",
    "MA_5",
    "MA_20",
    "Volatility_20",
]


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the stock dataset for LSTM training.

    Expected columns:
    Date, Open, High, Low, Close, Volume, Ticker,
    Return, MA_5, MA_20, Volatility_20, Target
    """

    df = df.copy()

    # Convert date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Sort chronologically within each stock
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Make sure numeric columns are numeric
    numeric_columns = FEATURE_COLUMNS + ["Target"]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Remove invalid rows
    df = df.dropna(
        subset=FEATURE_COLUMNS + ["Target"]
    ).reset_index(drop=True)

    return df


def create_sequences(
    df: pd.DataFrame,
    window_length: int = 30
):
    """
    Create sequences for the LSTM.

    Each sample contains the previous window_length
    trading observations.

    Returns:
        X: numpy array
           shape = (samples, window_length, features)

        y: numpy array
           shape = (samples,)
    """

    X = []
    y = []

    # Process each stock separately.
    # This prevents a sequence from crossing
    # from one stock into another.
    for ticker, stock_df in df.groupby("Ticker", sort=False):

        stock_df = stock_df.sort_values("Date").reset_index(drop=True)

        features = stock_df[FEATURE_COLUMNS].values.astype(np.float32)
        targets = stock_df["Target"].values.astype(np.float32)

        for i in range(window_length, len(stock_df)):

            X.append(
                features[i - window_length:i]
            )

            y.append(
                targets[i]
            )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32)
    )


def process_inputs(
    df: pd.DataFrame,
    window_length: int = 30
):
    """
    Compatibility wrapper for creating LSTM inputs.
    """

    df = prepare_dataframe(df)

    X, y = create_sequences(
        df,
        window_length=window_length
    )

    return X, y
