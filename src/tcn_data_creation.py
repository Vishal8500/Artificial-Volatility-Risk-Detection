import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

WINDOW = 15

print("Loading labeled dataset...")

df = pd.read_csv("labeled_dataset.csv")

stocks = [
    "RELIANCE","TCS","INFY","HDFCBANK",
    "ICICIBANK","SBIN","LT","ITC"
]

features = ["index_vol","sector_vol"]

for s in stocks:
    features.append(f"{s}_volatility")
    features.append(f"{s}_volume_norm")

X = df[features].values
y = df["label"].values

# scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_windows = []
y_windows = []

print("Creating time windows...")

for i in range(WINDOW, len(X)):

    X_windows.append(X[i-WINDOW:i])
    y_windows.append(y[i])

X_windows = np.array(X_windows)
y_windows = np.array(y_windows)

print("Final dataset shape:", X_windows.shape)

# save dataset
np.savez("tcn_dataset.npz", X=X_windows, y=y_windows)

print("Saved as tcn_dataset.npz")