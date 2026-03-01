import numpy as np
import pandas as pd

np.random.seed(42)

WINDOW_SIZE = 20
NUM_NATURAL = 3200
NUM_ARTIFICIAL = 800
LABEL_NOISE_RATIO = 0.10  # 10% ambiguity

# ==========================
# REGIME SWITCHING
# ==========================
def generate_regime_states(window_size):
    states = np.zeros(window_size)
    state = np.random.choice([0, 1])
    
    for t in range(window_size):
        if np.random.rand() < 0.15:   # slightly higher switching
            state = 1 - state
        states[t] = state
    return states

# ==========================
# GARCH WITH REGIMES
# ==========================
def generate_garch_regime(window_size):
    states = generate_regime_states(window_size)
    
    omega_calm = 0.00015
    omega_high = 0.0006
    
    alpha = 0.18
    beta = 0.70
    
    returns = np.zeros(window_size)
    sigma2 = np.zeros(window_size)
    
    sigma2[0] = 0.0005
    
    for t in range(1, window_size):
        omega = omega_high if states[t] == 1 else omega_calm
        returns[t-1] = np.random.normal(0, np.sqrt(sigma2[t-1]))
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
    
    return np.sqrt(sigma2)

# ==========================
# NATURAL SAMPLE
# ==========================
def generate_natural_sample():
    
    stock_vol = generate_garch_regime(WINDOW_SIZE)
    
    # 30% chance of natural shock
    if np.random.rand() < 0.3:
        spike_day = np.random.randint(4, WINDOW_SIZE-4)
        stock_vol[spike_day] += np.random.uniform(0.015, 0.035)
    
    # Strong but noisy correlation
    rho = np.random.uniform(0.35, 0.7)
    index_vol = rho * stock_vol + np.random.normal(0, 0.015, WINDOW_SIZE)
    
    sector_vol = 0.5 * stock_vol + np.random.normal(0, 0.02, WINDOW_SIZE)
    
    volume = 1 + 18 * stock_vol + np.random.normal(0, 0.3, WINDOW_SIZE)
    
    return np.stack([stock_vol, volume, index_vol, sector_vol], axis=1)

# ==========================
# ARTIFICIAL SAMPLE
# ==========================
def generate_artificial_sample():
    
    stock_vol = generate_garch_regime(WINDOW_SIZE)
    
    # Moderate spike (reduced magnitude)
    spike_day = np.random.randint(4, WINDOW_SIZE-4)
    spike = np.random.uniform(0.02, 0.05)
    stock_vol[spike_day] += spike
    
    # 40% behave almost natural (overlap)
    if np.random.rand() < 0.4:
        rho = np.random.uniform(0.35, 0.65)
    else:
        rho = np.random.uniform(0.0, 0.3)
    
    index_vol = rho * stock_vol + np.random.normal(0, 0.02, WINDOW_SIZE)
    
    sector_vol = np.random.normal(np.mean(stock_vol)*0.6, 0.025, WINDOW_SIZE)
    
    volume = 1 + 15 * stock_vol + np.random.normal(0, 0.4, WINDOW_SIZE)
    
    # Only sometimes volume spike
    if np.random.rand() < 0.5:
        volume[spike_day] += np.random.uniform(0.3, 1.0)
    
    return np.stack([stock_vol, volume, index_vol, sector_vol], axis=1)

# ==========================
# GENERATE DATASET
# ==========================
X = []
y = []

for _ in range(NUM_NATURAL):
    X.append(generate_natural_sample())
    y.append(0)

for _ in range(NUM_ARTIFICIAL):
    X.append(generate_artificial_sample())
    y.append(1)

X = np.array(X)
y = np.array(y)

# ==========================
# ADD LABEL NOISE
# ==========================
num_noise = int(len(y) * LABEL_NOISE_RATIO)
noise_indices = np.random.choice(len(y), num_noise, replace=False)
y[noise_indices] = 1 - y[noise_indices]

# ==========================
# SAVE CSV
# ==========================
X_flat = X.reshape(len(X), -1)

columns = []
for feature in ["stock_vol", "volume", "index_vol", "sector_vol"]:
    for t in range(1, WINDOW_SIZE + 1):
        columns.append(f"{feature}_t{t}")

df = pd.DataFrame(X_flat, columns=columns)
df["label"] = y

df.to_csv("synthetic_volatility_final_realistic.csv", index=False)

print("Final realistic dataset generated.")
print("Shape:", df.shape)
print("Class Distribution:", np.bincount(y))