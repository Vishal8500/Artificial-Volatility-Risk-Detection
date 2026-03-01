import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.linear_model import LinearRegression

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("synthetic_volatility_final_realistic.csv")
num_feature_columns = df.shape[1] - 1
WINDOW_SIZE = num_feature_columns // 4  # make sure this matches generator

# Separate features and label
labels = df["label"].values
feature_df = df.drop(columns=["label"])

X_flat = feature_df.values

# Reshape
X = X_flat.reshape(X_flat.shape[0], WINDOW_SIZE, 4)

num_samples = X.shape[0]

# Feature indices
STOCK_VOL = 0
VOLUME = 1
INDEX_VOL = 2
SECTOR_VOL = 3

# ===============================
# ADVANCED FEATURE ENGINEERING
# ===============================

vol_persistence = []
vol_trend = []
market_divergence = []
sector_divergence = []
vol_volume_elasticity = []
vol_kurtosis = []
vol_skewness = []
max_drawdown = []
regime_concentration = []
lag_corr = []

for i in range(num_samples):
    
    stock_vol = X[i,:,STOCK_VOL]
    volume = X[i,:,VOLUME]
    index_vol = X[i,:,INDEX_VOL]
    sector_vol = X[i,:,SECTOR_VOL]
    
    # 1 Volatility Persistence
    persistence = np.corrcoef(stock_vol[:-1], stock_vol[1:])[0,1]
    vol_persistence.append(persistence)
    
    # 2 Volatility Trend
    time_idx = np.arange(WINDOW_SIZE).reshape(-1,1)
    model = LinearRegression().fit(time_idx, stock_vol)
    vol_trend.append(model.coef_[0])
    
    # 3 Market Divergence
    market_divergence.append(np.mean(stock_vol - index_vol))
    
    # 4 Sector Divergence
    sector_divergence.append(np.mean(stock_vol - sector_vol))
    
    # 5 Volume Elasticity
    elasticity = np.corrcoef(stock_vol, volume)[0,1]
    vol_volume_elasticity.append(elasticity)
    
    # 6 Kurtosis
    vol_kurtosis.append(kurtosis(stock_vol))
    
    # 7 Skewness
    vol_skewness.append(skew(stock_vol))
    
    # 8 Max Drawdown
    peak_idx = np.argmax(stock_vol)
    if peak_idx < WINDOW_SIZE - 1:
        drawdown = stock_vol[peak_idx] - np.min(stock_vol[peak_idx:])
    else:
        drawdown = 0
    max_drawdown.append(drawdown)
    
    # 9 Regime Concentration
    regime_ratio = np.max(stock_vol) / (np.sum(stock_vol) + 1e-6)
    regime_concentration.append(regime_ratio)
    
    # 10 Lag Correlation
    lag_corr_value = np.corrcoef(stock_vol[1:], index_vol[:-1])[0,1]
    lag_corr.append(lag_corr_value)

# Add to dataframe
df["vol_persistence"] = vol_persistence
df["vol_trend"] = vol_trend
df["market_divergence"] = market_divergence
df["sector_divergence"] = sector_divergence
df["vol_volume_elasticity"] = vol_volume_elasticity
df["vol_kurtosis"] = vol_kurtosis
df["vol_skewness"] = vol_skewness
df["max_drawdown"] = max_drawdown
df["regime_concentration"] = regime_concentration
df["lag_corr"] = lag_corr

df.to_csv("synthetic_volatility_hard_engg.csv", index=False)

print("Feature engineering complete.")
print("New shape:", df.shape)