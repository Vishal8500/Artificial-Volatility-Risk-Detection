import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

print("Loading engineered features...")

df = pd.read_csv("engineered_features.csv")

stocks = [
    "HDFCBANK","ICICIBANK","KOTAKBANK","AXISBANK","SBIN","BAJFINANCE","BAJAJFINSV",
    "TCS","INFY","HCLTECH","WIPRO","TECHM",
    "RELIANCE","ONGC","IOC","BPCL",
    "MARUTI","M&M","HEROMOTOCO",
    "HINDUNILVR","ITC","NESTLEIND","BRITANNIA",
    "LT","ULTRACEMCO","GRASIM",
    "SUNPHARMA","CIPLA",
    "BHARTIARTL","NTPC","POWERGRID"
]

anomaly_rows = []

print("Running DBSCAN for all stocks...")

for s in stocks:

    print("Processing:", s)

    close = f"{s}_Close"
    vol = f"{s}_volatility"
    vol_norm = f"{s}_volume_norm"

    # Skip if missing
    if vol not in df.columns:
        print(f"Skipping missing: {s}")
        continue

    features = df[[vol, vol_norm, "index_vol", "sector_vol"]]

    X = features.values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    dbscan = DBSCAN(eps=2.5, min_samples=15)
    clusters = dbscan.fit_predict(X_scaled)

    # Get anomaly indices
    anomaly_idx = np.where(clusters == -1)[0]

    # Extract clean anomaly dataset
    temp = pd.DataFrame({
        "Date": df.loc[anomaly_idx, "Date"],
        "Stock": s,
        "Close": df.loc[anomaly_idx, close],
        "Volatility": df.loc[anomaly_idx, vol],
        "Volume_Norm": df.loc[anomaly_idx, vol_norm],
        "Index_Vol": df.loc[anomaly_idx, "index_vol"],
        "Sector_Vol": df.loc[anomaly_idx, "sector_vol"],
        "Label": 1
    })

    anomaly_rows.append(temp)

# Combine all anomalies
all_anomalies = pd.concat(anomaly_rows, ignore_index=True)

# Sort by severity (important)
all_anomalies = all_anomalies.sort_values(by="Index_Vol", ascending=False)

# Save final report
all_anomalies.to_csv("artificial_volatility_points.csv", index=False)

print("\n✅ DONE")
print("Total anomaly points:", len(all_anomalies))
print("Saved as: artificial_volatility_points.csv")

print("\nTop anomaly points:")
print(all_anomalies.head(10))