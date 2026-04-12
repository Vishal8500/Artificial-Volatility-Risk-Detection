# ============================
# INTERVAL-BASED ARTIFICIAL VOLATILITY ANALYSIS
# ============================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import ttest_ind
import os

# ============================
# CONFIG
# ============================
stocks = [
    "RELIANCE.NS", "ITC.NS", "LT.NS", "SBIN.NS", "TATASTEEL.NS"
]

intervals = [
    ("2000-01-01", "2005-01-01"),
    ("2005-01-01", "2010-01-01"),
    ("2010-01-01", "2015-01-01"),
    ("2015-01-01", "2020-01-01"),
    ("2020-01-01", "2026-01-01"),
]

window_size = 15
future_days = 180

# ============================
# OUTPUT FOLDERS
# ============================
base_dir = "outputs/interval_analysis"
os.makedirs(base_dir, exist_ok=True)
os.makedirs(f"{base_dir}/plots", exist_ok=True)
os.makedirs(f"{base_dir}/csv", exist_ok=True)

# ============================
# HELPERS
# ============================
def clean_array(arr):
    arr = np.array(arr).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return arr

def cohens_d(x, y):
    x = np.array(x)
    y = np.array(y)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    pooled_std = np.sqrt(((nx-1)*np.var(x) + (ny-1)*np.var(y)) / (nx+ny-2))
    return (np.mean(x) - np.mean(y)) / pooled_std

# ============================
# MAIN LOOP
# ============================
all_results = []

for start_date, end_date in intervals:
    print(f"\n===== INTERVAL {start_date[:4]} - {end_date[:4]} =====")

        # Download indices
    market = yf.download("^NSEI", start=start_date, end=end_date, progress=False)
    sector = yf.download("^CNXIT", start=start_date, end=end_date, progress=False)

    # Fallback if empty
    if market.empty or sector.empty:
        print("⚠️ Market/sector data missing → skipping this interval")
        continue

    for stock in stocks:
        print(f"Processing {stock}...")

        df = yf.download(stock, start=start_date, end=end_date, progress=False)

        if df.empty or len(df) < 200:
            print("Skipping (insufficient data)")
            continue

        # ============================
        # PREPROCESSING
        # ============================
        df = df[['Close', 'Volume']].copy()
        df = df.ffill()

        market_df = market[['Close']].rename(columns={'Close': 'market_close'})
        sector_df = sector[['Close']].rename(columns={'Close': 'sector_close'})

        df = df.merge(market_df, left_index=True, right_index=True, how='inner')
        df = df.merge(sector_df, left_index=True, right_index=True, how='inner')

        # ============================
        # FEATURES
        # ============================
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['volatility'] = df['log_return'].rolling(10).std()
        df['vol_norm'] = df['Volume'] / df['Volume'].rolling(10).mean()

        df['market_return'] = np.log(df['market_close'] / df['market_close'].shift(1))
        df['market_vol'] = df['market_return'].rolling(10).std()

        df['sector_return'] = np.log(df['sector_close'] / df['sector_close'].shift(1))
        df['sector_vol'] = df['sector_return'].rolling(10).std()

        df = df.dropna()

        if len(df) < window_size + future_days:
            print("Skipping (not enough processed data)")
            continue

        # ============================
        # WINDOWING
        # ============================
        windows = []
        dates = []

        for i in range(len(df) - window_size):
            w = df[['volatility','vol_norm','market_vol','sector_vol']].iloc[i:i+window_size]
            windows.append(w.values)
            dates.append(df.index[i])

        # ============================
        # FEATURE SUMMARY
        # ============================
        features = []
        for w in windows:
            features.append([
                np.mean(w[:,0]),
                np.max(w[:,0]),
                np.std(w[:,0]),
                np.max(w[:,1]),
                np.mean(w[:,2]),
                np.mean(w[:,3])
            ])

        features = np.array(features)

        # ============================
        # KMEANS
        # ============================
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features)

        cluster_scores = []
        for c in range(3):
            mask = clusters == c
            if np.sum(mask) == 0:
                cluster_scores.append(-np.inf)
                continue
            score = np.mean(features[mask][:,1]) + np.mean(features[mask][:,4])
            cluster_scores.append(score)

        artificial_cluster = np.argmax(cluster_scores)
        labels = (clusters == artificial_cluster).astype(int)

        # ============================
        # RETURNS
        # ============================
        artificial_returns = []
        normal_returns = []

        for i in range(len(dates)):
            if i + future_days >= len(df):
                continue

            p1 = float(df['Close'].iloc[i])
            p2 = float(df['Close'].iloc[i + future_days])

            if p1 <= 0:
                continue

            ret = (p2 - p1) / p1

            if labels[i] == 1:
                artificial_returns.append(ret)
            else:
                normal_returns.append(ret)

        artificial_returns = clean_array(artificial_returns)
        normal_returns = clean_array(normal_returns)

        if len(artificial_returns) < 5 or len(normal_returns) < 5:
            print("Skipping (not enough return samples)")
            continue

        # ============================
        # STATS
        # ============================
        mean_artificial = np.mean(artificial_returns)
        mean_normal = np.mean(normal_returns)

        t_stat, p_value = ttest_ind(artificial_returns, normal_returns, equal_var=False)
        p_value = float(np.array(p_value).flatten()[0])

        effect_size = cohens_d(artificial_returns, normal_returns)

        print(f"Artificial: {mean_artificial:.4f} | Normal: {mean_normal:.4f} | p={p_value:.4f} | d={effect_size:.4f}")

        # ============================
        # SAVE CSV
        # ============================
        interval_name = f"{start_date[:4]}_{end_date[:4]}"
        stock_name = stock.replace(".NS", "")

        result_df = pd.DataFrame({
            'type': ['Artificial']*len(artificial_returns) + ['Natural']*len(normal_returns),
            'returns': list(artificial_returns) + list(normal_returns)
        })

        result_df.to_csv(f"{base_dir}/csv/{stock_name}_{interval_name}.csv", index=False)

        # ============================
        # PLOT
        # ============================
        plt.figure()
        plt.boxplot([artificial_returns, normal_returns])
        plt.xticks([1,2], ['Artificial','Natural'])
        plt.title(f"{stock_name} ({interval_name})")
        plt.savefig(f"{base_dir}/plots/{stock_name}_{interval_name}.png")
        plt.close()

        # ============================
        # STORE SUMMARY
        # ============================
        all_results.append({
            'stock': stock_name,
            'interval': interval_name,
            'mean_artificial': mean_artificial,
            'mean_normal': mean_normal,
            'p_value': p_value,
            'effect_size': effect_size
        })

# ============================
# FINAL SUMMARY
# ============================
summary_df = pd.DataFrame(all_results)
summary_df.to_csv(f"{base_dir}/summary.csv", index=False)

print("\n✅ DONE: Interval analysis saved in outputs/interval_analysis/")