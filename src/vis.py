# ============================
# ADVANCED ANALYSIS (7, 8, 9) — FINAL FIXED VERSION
# ============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import os

os.makedirs("outputs", exist_ok=True)

# ============================
# LOAD DATA
# ============================

events_df = pd.read_csv("outputs/all_event_returns.csv")

events_df["date"] = pd.to_datetime(events_df["date"])
events_df["future_return"] = pd.to_numeric(events_df["future_return"], errors="coerce")
events_df = events_df.dropna(subset=["future_return"])

# ============================
# 🔹 7. DISTRIBUTION ANALYSIS
# ============================

art = events_df[events_df["label"] == 1]["future_return"]
nat = events_df[events_df["label"] == 0]["future_return"]

plt.figure()
plt.hist(art, bins=50, alpha=0.6, label="Artificial")
plt.hist(nat, bins=50, alpha=0.6, label="Natural")

plt.title("Return Distribution (Artificial vs Natural)")
plt.xlabel("Return")
plt.ylabel("Frequency")
plt.legend()

plt.savefig("outputs/distribution_plot.png", dpi=200)
plt.close()

print("📈 Plot 4 saved")

# ============================
# 🔹 8. EFFECT SIZE OVER TIME
# ============================

def cohens_d(a, b):
    a = pd.to_numeric(a, errors='coerce').dropna().values.astype(float)
    b = pd.to_numeric(b, errors='coerce').dropna().values.astype(float)

    if len(a) < 2 or len(b) < 2:
        return np.nan

    pooled = np.sqrt((np.var(a) + np.var(b)) / 2)
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0

events_df["year"] = events_df["date"].dt.year

effect_trend = []

for y in sorted(events_df["year"].unique()):
    sub = events_df[events_df["year"] == y]

    art = sub[sub["label"] == 1]["future_return"]
    nat = sub[sub["label"] == 0]["future_return"]

    if len(art) < 5 or len(nat) < 5:
        continue

    d = cohens_d(art, nat)
    effect_trend.append((y, d))

effect_df = pd.DataFrame(effect_trend, columns=["Year", "Effect Size"])

plt.figure()
plt.plot(effect_df["Year"], effect_df["Effect Size"], marker='o')

plt.title("Effect Size Over Time")
plt.xlabel("Year")
plt.ylabel("Cohen's d")
plt.grid()

plt.savefig("outputs/effect_size_trend.png", dpi=200)
plt.close()

print("📈 Plot 5 saved")

# ============================
# 🔹 9. STABILITY & RECOVERY
# ============================

stocks = ["RELIANCE.NS", "HDFCBANK.NS", "TCS.NS", "INFY.NS", "ITC.NS"]

recovery_rows = []
all_curves = []

for stock in stocks:
    print("Processing recovery:", stock)

    df = yf.download(stock, start="2000-01-01", end="2026-01-01", progress=False)
    if df.empty:
        continue

    df = df[['Close']].dropna().copy()
    df["Close"] = df["Close"].astype(float)

    df["return"] = df["Close"].pct_change()
    df = df.dropna()

    threshold = df["return"].std() * 2
    spike_idx = df[np.abs(df["return"]) > threshold].index

    recovery_days = []

    for idx in spike_idx[:20]:
        try:
            pos = df.index.get_loc(idx)

            base_price = float(df["Close"].iloc[pos])

            # look at next few days to detect drop
            window = df["Close"].iloc[pos:pos+10]

            if len(window) < 2:
                continue

            min_price = float(window.min())

            # skip if no drop
            if min_price >= base_price:
                continue

            # recovery target (90% recovery)
            target_price = min_price + 0.9 * (base_price - min_price)

            recovered = False

            for i in range(1, 60):
                if pos + i >= len(df):
                    break

                future_price = float(df["Close"].iloc[pos + i])

                if future_price >= target_price:
                    recovery_days.append(i)
                    recovered = True
                    break

            # if no recovery → assign max window
            if not recovered:
                recovery_days.append(60)

            # build curve
            if pos - 20 >= 0 and pos + 20 < len(df):
                segment = df["Close"].iloc[pos-20:pos+20].values.astype(float)
                segment = segment / segment[20]
                all_curves.append(segment)

        except:
            continue

    avg_recovery = np.mean(recovery_days) if len(recovery_days) > 0 else np.nan

    close = df["Close"].dropna()

    if len(close) < 2:
        continue

    start_price = float(close.iloc[0])
    end_price = float(close.iloc[-1])

    # trend classification
    if end_price > start_price * 2:
        trend = "Strong Upward"
    elif end_price > start_price:
        trend = "Upward"
    else:
        trend = "Stable"

    recovery_rows.append({
        "Company": stock.replace(".NS", ""),
        "Artificial Events": len(spike_idx),
        "Avg Recovery Time": round(avg_recovery, 2),
        "Long-Term Trend": trend
    })

# ============================
# SAVE TABLE 7
# ============================

table7 = pd.DataFrame(recovery_rows)
table7.to_csv("outputs/recovery_analysis.csv", index=False)

print("\n📊 Table 7 saved")
print(table7)

if len(all_curves) > 0:
    curves = np.array(all_curves)

    # 🔥 FIX: Ensure 2D shape (N, 40)
    curves = np.squeeze(curves)

    # If still wrong shape, enforce manually
    if len(curves.shape) > 2:
        curves = curves.reshape(curves.shape[0], -1)

    mean_curve = np.mean(curves, axis=0)
    std_curve = np.std(curves, axis=0)

    # 🔥 Ensure 1D
    mean_curve = mean_curve.flatten()
    std_curve = std_curve.flatten()

    x = np.arange(-20, 20)

    plt.figure(figsize=(8, 5))

    # Confidence band
    plt.fill_between(
        x,
        mean_curve - std_curve,
        mean_curve + std_curve,
        alpha=0.2,
        label="±1 Std Dev"
    )

    # Mean curve
    plt.plot(x, mean_curve, linewidth=2.5, label="Mean Recovery")

    # Event marker
    plt.axvline(0, linestyle="--", linewidth=1.5)
    plt.text(0, max(mean_curve)+0.02, "Event", ha='center')

    plt.title("Recovery Pattern Following Volatility Spikes")
    plt.xlabel("Days Relative to Event")
    plt.ylabel("Normalized Price (Base = 1 at Event)")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/recovery_pattern_publication.png", dpi=300)
    plt.close()

    print("✅ Fixed and saved publication-quality plot")