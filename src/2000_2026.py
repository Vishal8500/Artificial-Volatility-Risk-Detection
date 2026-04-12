# ============================
# ARTIFICIAL VOLATILITY PROJECT - FULL PIPELINE
# ============================

import os
import warnings
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import yfinance as yf
from sklearn.cluster import KMeans
from scipy.stats import ttest_ind

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ============================
# CONFIG
# ============================
STOCKS = [
    # Core Industrial
    "RELIANCE.NS", "LT.NS", "TATASTEEL.NS", "GRASIM.NS", "HINDALCO.NS",

    # Banking
    "SBIN.NS", "ICICIBANK.NS", "HDFCBANK.NS", "AXISBANK.NS", "BANKBARODA.NS",

    # IT
    "INFY.NS", "WIPRO.NS", "HCLTECH.NS",

    # Energy
    "ONGC.NS", "NTPC.NS", "GAIL.NS", "BPCL.NS", "HINDPETRO.NS",

    # Auto
    "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS",

    # FMCG
    "ITC.NS", "HINDUNILVR.NS", "DABUR.NS",

    # Capital Goods
    "BHEL.NS", "SIEMENS.NS"
]

START_DATE = "2000-01-01"
END_DATE = "2026-01-01"
WINDOW_SIZE = 15
FUTURE_DAYS = 180
ROLLING_VOL_DAYS = 10
N_CLUSTERS = 3

OUTPUT_DIR = "outputs"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
CSV_DIR = os.path.join(OUTPUT_DIR, "csv")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

# ============================
# HELPERS
# ============================
def safe_download(ticker: str, start: str, end: str, retries: int = 3) -> pd.DataFrame:
    last_err = None
    for _ in range(retries):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=False,
                threads=True
            )
            if df is not None and not df.empty:
                df = df.copy()
                df.index = pd.to_datetime(df.index)
                df.sort_index(inplace=True)
                return df
        except Exception as e:
            last_err = e
    print(f"Skipping {ticker}: download failed or empty. {last_err}")
    return pd.DataFrame()

def clean_1d_array(arr) -> np.ndarray:
    arr = np.asarray(arr, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return arr

def build_reference_series(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    for ticker in tickers:
        ref = safe_download(ticker, start, end)
        if not ref.empty and "Close" in ref.columns:
            return ref[["Close"]].rename(columns={"Close": "ref_close"})
    return pd.DataFrame()

def prepare_data(stock_df: pd.DataFrame, market_df: pd.DataFrame, sector_df: pd.DataFrame) -> pd.DataFrame:
    df = stock_df.copy()

    required_cols = [c for c in ["Close", "Volume"] if c in df.columns]
    if len(required_cols) < 2:
        return pd.DataFrame()

    df = df[["Close", "Volume"]].copy()
    df = df.join(market_df.rename(columns={"ref_close": "market_close"}), how="inner")
    df = df.join(sector_df.rename(columns={"ref_close": "sector_close"}), how="inner")

    df = df.ffill().bfill()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["log_return"] = np.log(out["Close"] / out["Close"].shift(1))
    out["volatility"] = out["log_return"].rolling(ROLLING_VOL_DAYS, min_periods=ROLLING_VOL_DAYS).std()

    out["vol_norm"] = out["Volume"] / out["Volume"].rolling(ROLLING_VOL_DAYS, min_periods=ROLLING_VOL_DAYS).mean()

    out["market_return"] = np.log(out["market_close"] / out["market_close"].shift(1))
    out["market_vol"] = out["market_return"].rolling(ROLLING_VOL_DAYS, min_periods=ROLLING_VOL_DAYS).std()

    out["sector_return"] = np.log(out["sector_close"] / out["sector_close"].shift(1))
    out["sector_vol"] = out["sector_return"].rolling(ROLLING_VOL_DAYS, min_periods=ROLLING_VOL_DAYS).std()

    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out

def make_windows(df: pd.DataFrame, window_size: int) -> Tuple[List[np.ndarray], List[pd.Timestamp], List[int]]:
    feature_cols = ["volatility", "vol_norm", "market_vol", "sector_vol"]
    windows = []
    dates = []
    indices = []

    if len(df) <= window_size:
        return windows, dates, indices

    for i in range(len(df) - window_size):
        w = df[feature_cols].iloc[i:i + window_size].to_numpy(dtype=float)
        windows.append(w)
        dates.append(df.index[i])
        indices.append(i)

    return windows, dates, indices

def summarize_window(w: np.ndarray) -> List[float]:
    stock_vol = w[:, 0]
    vol_norm = w[:, 1]
    market_vol = w[:, 2]
    sector_vol = w[:, 3]

    mean_vol = float(np.mean(stock_vol))
    max_vol = float(np.max(stock_vol))
    std_vol = float(np.std(stock_vol))
    volume_spike_ratio = float(np.max(vol_norm))
    stock_market_div = float(np.mean(np.abs(stock_vol - market_vol)))
    stock_sector_div = float(np.mean(np.abs(stock_vol - sector_vol)))

    return [
        mean_vol,
        max_vol,
        std_vol,
        volume_spike_ratio,
        stock_market_div,
        stock_sector_div
    ]

def label_windows(features: np.ndarray) -> Tuple[np.ndarray, int]:
    n_samples = len(features)
    if n_samples < 3:
        return np.zeros(n_samples, dtype=int), 0

    n_clusters = min(N_CLUSTERS, n_samples)
    if n_clusters < 2:
        return np.zeros(n_samples, dtype=int), 0

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = km.fit_predict(features)

    cluster_scores = []
    for c in range(n_clusters):
        mask = clusters == c
        if not np.any(mask):
            cluster_scores.append(-np.inf)
            continue

        subset = features[mask]
        mean_max_vol = float(np.mean(subset[:, 1]))
        mean_div = float(np.mean(subset[:, 4]) + np.mean(subset[:, 5]))
        score = mean_max_vol + mean_div
        cluster_scores.append(score)

    artificial_cluster = int(np.argmax(cluster_scores))
    labels = (clusters == artificial_cluster).astype(int)
    return labels, artificial_cluster

def future_returns_by_label(
    df: pd.DataFrame,
    dates: List[pd.Timestamp],
    labels: np.ndarray,
    future_days: int
) -> Tuple[List[float], List[float], List[dict]]:
    artificial_returns = []
    normal_returns = []
    event_rows = []

    for i, date in enumerate(dates):
        future_idx = i + future_days
        if future_idx >= len(df):
            continue

        p1 = float(df.iloc[i]["Close"])
        p2 = float(df.iloc[future_idx]["Close"])

        if not np.isfinite(p1) or not np.isfinite(p2) or p1 <= 0:
            continue

        ret = (p2 - p1) / p1

        label = int(labels[i])
        row = {
            "date": date,
            "label": label,
            "future_days": future_days,
            "start_close": p1,
            "future_close": p2,
            "future_return": float(ret)
        }
        event_rows.append(row)

        if label == 1:
            artificial_returns.append(float(ret))
        else:
            normal_returns.append(float(ret))

    return artificial_returns, normal_returns, event_rows

def safe_ttest(a: List[float], b: List[float]) -> Tuple[float, float]:
    a = clean_1d_array(a)
    b = clean_1d_array(b)

    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")

    result = ttest_ind(a, b, equal_var=False, nan_policy="omit")

    t_stat = float(np.asarray(result.statistic).reshape(-1)[0])
    p_value = float(np.asarray(result.pvalue).reshape(-1)[0])
    return t_stat, p_value

def save_boxplot(artificial_returns: List[float], normal_returns: List[float], stock: str) -> None:
    a = clean_1d_array(artificial_returns)
    n = clean_1d_array(normal_returns)

    if len(a) == 0 or len(n) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([a, n])
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Artificial", "Natural"])
    ax.set_title(f"{stock} Return Comparison")
    ax.set_ylabel("Future Return")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"{stock}_boxplot.png"), dpi=200)
    plt.close(fig)

def save_event_study(df: pd.DataFrame, dates: List[pd.Timestamp], labels: np.ndarray, stock: str) -> None:
    half_window = 30
    selected_indices = [i for i, lab in enumerate(labels) if lab == 1][:50]

    curves = []
    for idx in selected_indices:
        if idx - half_window < 0 or idx + half_window >= len(df):
            continue
        segment = df["Close"].iloc[idx - half_window: idx + half_window + 1].to_numpy(dtype=float)
        center = segment[half_window]
        if not np.isfinite(center) or center == 0:
            continue
        segment = segment / center
        curves.append(segment)

    if not curves:
        return

    curves = np.asarray(curves, dtype=float)
    mean_curve = np.nanmean(curves, axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    for c in curves:
        ax.plot(c, alpha=0.15)
    ax.plot(mean_curve, linewidth=2)

    ax.set_title(f"{stock} Event Study (Artificial Events)")
    ax.set_xlabel("Days Around Event")
    ax.set_ylabel("Normalized Price")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"{stock}_event_study.png"), dpi=200)
    plt.close(fig)

def save_growth_curve(df: pd.DataFrame, stock: str) -> None:
    close = df["Close"].astype(float).copy()
    close = close.replace([np.inf, -np.inf], np.nan).dropna()
    if close.empty:
        return

    growth = close / close.iloc[0]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(growth.index, growth.values)
    ax.set_title(f"{stock} Long-Term Growth Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth Index (Normalized)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"{stock}_growth_curve.png"), dpi=200)
    plt.close(fig)

# ============================
# DOWNLOAD REFERENCE DATA
# ============================
print("Downloading reference data...")
market_ref = build_reference_series(["^NSEI"], START_DATE, END_DATE)
sector_ref = build_reference_series(["^CNXIT", "^NIFTYIT"], START_DATE, END_DATE)

if market_ref.empty:
    raise RuntimeError("Market index data could not be downloaded.")
if sector_ref.empty:
    raise RuntimeError("Sector index data could not be downloaded.")

# ============================
# MAIN PIPELINE
# ============================
summary_rows = []
event_rows_all = []

for stock in STOCKS:
    print(f"\nProcessing {stock}...")

    raw = safe_download(stock, START_DATE, END_DATE)
    if raw.empty:
        print(f"Skipping {stock}: no usable stock data.")
        continue

    df = prepare_data(raw, market_ref, sector_ref)
    if df.empty or len(df) < (ROLLING_VOL_DAYS + WINDOW_SIZE + FUTURE_DAYS + 5):
        print(f"Skipping {stock}: not enough aligned data after preprocessing.")
        continue

    df = engineer_features(df)
    if df.empty or len(df) < (WINDOW_SIZE + FUTURE_DAYS + 5):
        print(f"Skipping {stock}: not enough data after feature engineering.")
        continue

    windows, dates, indices = make_windows(df, WINDOW_SIZE)
    if len(windows) == 0:
        print(f"Skipping {stock}: no windows could be created.")
        continue

    features = np.asarray([summarize_window(w) for w in windows], dtype=float)
    labels, artificial_cluster = label_windows(features)

    artificial_returns, normal_returns, event_rows = future_returns_by_label(
        df=df,
        dates=dates,
        labels=labels,
        future_days=FUTURE_DAYS
    )

    artificial_returns = clean_1d_array(artificial_returns)
    normal_returns = clean_1d_array(normal_returns)

    mean_artificial = float(np.mean(artificial_returns)) if len(artificial_returns) else float("nan")
    mean_normal = float(np.mean(normal_returns)) if len(normal_returns) else float("nan")
    t_stat, p_value = safe_ttest(artificial_returns, normal_returns)

    print(f"Mean Artificial Return: {mean_artificial:.4f}" if np.isfinite(mean_artificial) else "Mean Artificial Return: nan")
    print(f"Mean Normal Return: {mean_normal:.4f}" if np.isfinite(mean_normal) else "Mean Normal Return: nan")
    print(f"P-value: {p_value:.4f}" if np.isfinite(p_value) else "P-value: nan")

    stock_name = stock.replace(".NS", "")

    # Save per-stock event returns
    per_stock_event_df = pd.DataFrame(event_rows)
    per_stock_event_df.to_csv(os.path.join(CSV_DIR, f"{stock_name}_event_returns.csv"), index=False)

    # Save per-stock summary
    summary_rows.append({
        "stock": stock_name,
        "ticker": stock,
        "n_windows": int(len(labels)),
        "n_artificial": int(np.sum(labels == 1)),
        "n_natural": int(np.sum(labels == 0)),
        "mean_artificial_return": mean_artificial,
        "mean_normal_return": mean_normal,
        "t_stat": t_stat,
        "p_value": p_value,
        "artificial_cluster": artificial_cluster
    })

    event_rows_all.extend([{**r, "stock": stock_name, "ticker": stock} for r in event_rows])

    # Save plots
    save_boxplot(artificial_returns, normal_returns, stock_name)
    save_event_study(df, dates, labels, stock_name)
    save_growth_curve(df, stock_name)

# ============================
# FINAL CSV OUTPUTS
# ============================
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary_results.csv"), index=False)

all_events_df = pd.DataFrame(event_rows_all)
all_events_df.to_csv(os.path.join(OUTPUT_DIR, "all_event_returns.csv"), index=False)

print("\nDONE")
print(f"Summary saved to: {os.path.join(OUTPUT_DIR, 'summary_results.csv')}")
print(f"All event returns saved to: {os.path.join(OUTPUT_DIR, 'all_event_returns.csv')}")
print(f"Plots saved to: {PLOTS_DIR}")
print(f"Per-stock CSVs saved to: {CSV_DIR}")