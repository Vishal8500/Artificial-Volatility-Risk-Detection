import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# LOAD RAW SYNTHETIC DATA
# =========================
df = pd.read_csv("synthetic_volatility_final_realistic.csv")

# Remove label
y = df["label"].values
X_flat = df.drop(columns=["label"]).values

# Automatically detect WINDOW_SIZE
num_features = X_flat.shape[1]
WINDOW_SIZE = num_features // 4
SEQ_FEATURES = 4

print("Detected WINDOW_SIZE:", WINDOW_SIZE)

# Reshape to (samples, window, features)
X = X_flat.reshape(-1, WINDOW_SIZE, SEQ_FEATURES)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# NORMALIZATION
# =========================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train.reshape(len(X_train), -1))
X_test = scaler.transform(X_test.reshape(len(X_test), -1))

X_train = X_train.reshape(-1, WINDOW_SIZE, SEQ_FEATURES)
X_test = X_test.reshape(-1, WINDOW_SIZE, SEQ_FEATURES)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

# =========================
# TCN MODEL (Sequence Only)
# =========================
class TCN(nn.Module):
    def __init__(self):
        super(TCN, self).__init__()
        
        self.conv1 = nn.Conv1d(SEQ_FEATURES, 32, kernel_size=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.dropout = nn.Dropout(0.3)
        
        self.fc1 = nn.Linear(64 * (WINDOW_SIZE - 4), 32)
        self.fc2 = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, features, window)
        
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        
        return self.fc2(x)  # logits

model = TCN().to(device)

# =========================
# LOSS (handle imbalance)
# =========================
class_counts = np.bincount(y)
neg, pos = class_counts[0], class_counts[1]

pos_weight = torch.tensor([neg / pos]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = optim.Adam(model.parameters(), lr=0.001)

# =========================
# TRAINING
# =========================
EPOCHS = 15
BATCH_SIZE = 64

for epoch in range(EPOCHS):
    model.train()
    permutation = torch.randperm(X_train.size(0))
    
    total_loss = 0
    
    for i in range(0, X_train.size(0), BATCH_SIZE):
        indices = permutation[i:i+BATCH_SIZE]
        
        batch_X = X_train[indices]
        batch_y = y_train[indices]
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss:.4f}")

# =========================
# EVALUATION
# =========================
model.eval()
with torch.no_grad():
    logits = model(X_test)
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

print("\nClassification Report:\n")
print(classification_report(y_test.cpu(), preds.cpu()))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test.cpu(), preds.cpu()))

# =========================
# SAVE MODEL + SCALER
# =========================
torch.save(model.state_dict(), "tcn_volatility_model.pth")

import joblib
joblib.dump(scaler, "tcn_scaler.pkl")

print("Model and scaler saved successfully.")