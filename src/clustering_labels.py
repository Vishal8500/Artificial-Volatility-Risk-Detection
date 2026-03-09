import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

print("Loading engineered features...")

df = pd.read_csv("engineered_features.csv")

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

# initialize labels
final_labels = np.zeros(len(df))

print("Running DBSCAN per stock...")

for s in stocks:

    print("Processing:", s)

    features = df[[
        f"{s}_volatility",
        f"{s}_volume_norm",
        "index_vol",
        "sector_vol"
    ]]

    X = features.values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    dbscan = DBSCAN(
        eps=2.5,
        min_samples=15
    )

    clusters = dbscan.fit_predict(X_scaled)

    # DBSCAN outliers = -1
    stock_labels = np.where(clusters == -1, 1, 0)

    # combine anomalies across stocks
    final_labels = np.maximum(final_labels, stock_labels)

df["label"] = final_labels.astype(int)

# remove fragmentation warning
df = df.copy()

df.to_csv("labeled_dataset.csv", index=False)

print("Dataset saved as labeled_dataset.csv")
print("Shape:", df.shape)
print("Class distribution:", np.bincount(df["label"]))