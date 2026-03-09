import pandas as pd
import numpy as np

print("Loading raw data...")

df = pd.read_csv("raw_market_data.csv", low_memory=False)

df["Date"] = pd.to_datetime(df["Date"])

# Convert all numeric columns
for col in df.columns:
    if col != "Date":
        df[col] = pd.to_numeric(df[col], errors="coerce")

stocks = [
    "RELIANCE",
    "TCS",
    "INFY",
    "HDFCBANK",
    "ICICIBANK",
    "SBIN",
    "LT",
    "ITC"
]

print("Generating features...")

for s in stocks:

    close = f"{s}_Close"
    high = f"{s}_High"
    low = f"{s}_Low"
    volume = f"{s}_Volume"

    # Log return
    df[f"{s}_return"] = np.log(df[close]).diff()

    # Volatility
    df[f"{s}_volatility"] = df[f"{s}_return"].rolling(10).std()

    # Normalized volume
    df[f"{s}_volume_norm"] = df[volume] / df[volume].rolling(10).mean()

    # Moving averages
    df[f"{s}_MA5"] = df[close].rolling(5).mean()
    df[f"{s}_MA20"] = df[close].rolling(20).mean()

    # Momentum
    df[f"{s}_momentum"] = df[close] - df[f"{s}_MA20"]

    # Intraday volatility
    df[f"{s}_range"] = (df[high] - df[low]) / df[close]

    # RSI
    delta = df[close].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    df[f"{s}_RSI"] = 100 - (100 / (1 + rs))

    # Z-score
    rolling_mean = df[close].rolling(20).mean()
    rolling_std = df[close].rolling(20).std()

    df[f"{s}_zscore"] = (df[close] - rolling_mean) / rolling_std


print("Generating market features...")

df["index_return"] = np.log(df["index_close"]).diff()
df["index_vol"] = df["index_return"].rolling(10).std()

df["sector_return"] = np.log(df["sector_close"]).diff()
df["sector_vol"] = df["sector_return"].rolling(10).std()

print("Computing market correlation...")

for s in stocks:
    df[f"{s}_market_corr"] = df[f"{s}_return"].rolling(20).corr(df["index_return"])

df = df.dropna()

df.to_csv("engineered_features.csv", index=False)

print("Feature engineering complete")
print("Dataset shape:", df.shape)