# ============================
# DEMONETIZATION + ARTIFICIAL VOLATILITY (FINAL CLEAN VERSION)
# ============================

import matplotlib
matplotlib.use('Agg')  # Fix GUI crash

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

# ============================
# STOCK LIST
# ============================

stocks = [
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS",
    "KOTAKBANK.NS", "BAJFINANCE.NS",
    "INFY.NS", "TCS.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS",
    "RELIANCE.NS", "LT.NS", "ADANIPORTS.NS", "SIEMENS.NS",
    "NTPC.NS", "POWERGRID.NS", "ONGC.NS", "GAIL.NS",
    "HINDUNILVR.NS", "ITC.NS", "DABUR.NS", "BRITANNIA.NS", "ASIANPAINT.NS",
    "MARUTI.NS", "M&M.NS", "TATAMOTORS.NS", "HEROMOTOCO.NS", "BAJAJ-AUTO.NS",
    "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS",
    "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS"
]

# ============================
# CONFIG
# ============================

event_date = pd.to_datetime("2016-11-08")

before_start = "2011-01-01"
after_end = "2021-12-31"

window_size = 15

# ============================
# OUTPUT FOLDER
# ============================

base_dir = "outputs/demonetization_analysis"
os.makedirs(base_dir, exist_ok=True)
os.makedirs(f"{base_dir}/plots", exist_ok=True)
os.makedirs(f"{base_dir}/csv", exist_ok=True)

# ============================
# HELPERS
# ============================

def clean_array(arr):
    arr = np.array(arr).reshape(-1)
    return arr[np.isfinite(arr)]

def process_period(df):
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['volatility'] = df['log_return'].rolling(10).std()
    df['vol_norm'] = df['Volume'] / df['Volume'].rolling(10).mean()
    df = df.dropna()

    if len(df) < window_size:
        return None, None

    windows = []
    for i in range(len(df) - window_size):
        w = df[['volatility','vol_norm']].iloc[i:i+window_size]
        windows.append(w.values)

    features = []
    for w in windows:
        features.append([
            np.mean(w[:,0]),
            np.max(w[:,0]),
            np.std(w[:,0]),
            np.max(w[:,1])
        ])

    features = np.array(features)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features)

    scores = []
    for c in range(3):
        mask = clusters == c
        if np.sum(mask) == 0:
            scores.append(-np.inf)
        else:
            scores.append(np.mean(features[mask][:,1]))

    artificial_cluster = np.argmax(scores)
    labels = (clusters == artificial_cluster).astype(int)

    return df, labels

def compute_growth(df):
    if df is None or len(df) == 0:
        return np.nan

    start_price = df['Close'].iloc[0]
    end_price = df['Close'].iloc[-1]

    # Convert safely
    start_price = float(np.array(start_price).flatten()[0])
    end_price = float(np.array(end_price).flatten()[0])

    if start_price == 0:
        return np.nan

    return (end_price - start_price) / start_price

def compute_returns(data, labels):
    art, nat = [], []

    for i in range(len(labels)-1):
        p1 = float(np.array(data['Close'].iloc[i]).flatten()[0])
        p2 = float(np.array(data['Close'].iloc[i+1]).flatten()[0])

        if p1 <= 0:
            continue

        ret = (p2 - p1) / p1

        if labels[i] == 1:
            art.append(ret)
        else:
            nat.append(ret)

    return clean_array(art), clean_array(nat)

def pad(arr, length):
    return list(arr) + [np.nan]*(length - len(arr))

# ============================
# MAIN LOOP
# ============================

results = []

for stock in stocks:
    print(f"\nProcessing {stock}...")

    df = yf.download(stock, start=before_start, end=after_end, progress=False)

    if df.empty or len(df) < 500:
        print("Skipping (not enough data)")
        continue

    df = df[['Close','Volume']].copy().ffill()

    before_df = df[df.index < event_date].copy()
    after_df  = df[df.index >= event_date].copy()

    before_df, before_labels = process_period(before_df)
    after_df, after_labels   = process_period(after_df)

    if before_df is None or after_df is None:
        print("Skipping (processing issue)")
        continue

    before_growth = compute_growth(before_df)
    after_growth  = compute_growth(after_df)

    b_art, b_nat = compute_returns(before_df, before_labels)
    a_art, a_nat = compute_returns(after_df, after_labels)

    stock_name = stock.replace(".NS","")

    # ============================
    # FIXED CSV (NO LENGTH ERROR)
    # ============================

    max_len = max(len(b_art), len(b_nat), len(a_art), len(a_nat))

    df_out = pd.DataFrame({
        "before_artificial": pad(b_art, max_len),
        "before_natural": pad(b_nat, max_len),
        "after_artificial": pad(a_art, max_len),
        "after_natural": pad(a_nat, max_len)
    })

    df_out.to_csv(f"{base_dir}/csv/{stock_name}.csv", index=False)

    # ============================
    # PLOT
    # ============================

    plt.figure()
    plt.plot(before_df['Close']/before_df['Close'].iloc[0], label="Before")
    plt.plot(after_df['Close']/after_df['Close'].iloc[0], label="After")
    plt.title(stock_name)
    plt.legend()
    plt.savefig(f"{base_dir}/plots/{stock_name}.png")
    plt.close()

    # ============================
    # STORE SUMMARY
    # ============================

    results.append({
        "stock": stock_name,
        "before_growth": before_growth,
        "after_growth": after_growth,
        "before_artificial_mean": np.mean(b_art),
        "before_natural_mean": np.mean(b_nat),
        "after_artificial_mean": np.mean(a_art),
        "after_natural_mean": np.mean(a_nat)
    })

# ============================
# FINAL SUMMARY
# ============================

summary_df = pd.DataFrame(results)
summary_df.to_csv(f"{base_dir}/summary.csv", index=False)

print("\n✅ DONE → outputs/demonetization_analysis/")