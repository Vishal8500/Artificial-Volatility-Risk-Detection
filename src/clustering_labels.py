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

print("Preparing clustering features...")

cluster_features = []

for s in stocks:

    features = df[[
        f"{s}_volatility",
        f"{s}_volume_norm",
        "index_vol",
        "sector_vol"
    ]]

    cluster_features.append(features)

cluster_features = pd.concat(cluster_features, axis=1)

# convert to numpy
X = cluster_features.values

print("Scaling features...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Running DBSCAN...")

dbscan = DBSCAN(
    eps=1.5,
    min_samples=10
)

clusters = dbscan.fit_predict(X_scaled)

print("Assigning labels...")

# DBSCAN outliers = -1
labels = np.where(clusters == -1, 1, 0)

df["label"] = labels

df.to_csv("labeled_dataset.csv", index=False)

print("Dataset saved as labeled_dataset.csv")
print("Shape:", df.shape)
print("Class distribution:", np.bincount(labels))