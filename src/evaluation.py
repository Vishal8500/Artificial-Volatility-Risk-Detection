import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("synthetic_volatility_final_realistic.csv")

y = df["label"].values
X_flat = df.drop(columns=["label"]).values

WINDOW_SIZE = X_flat.shape[1] // 4
SEQ_FEATURES = 4

X = X_flat.reshape(-1, WINDOW_SIZE, SEQ_FEATURES)

# Same split as training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# LOAD SCALER
# =========================
scaler = joblib.load("tcn_scaler.pkl")

X_test = scaler.transform(X_test.reshape(len(X_test), -1))
X_test = X_test.reshape(-1, WINDOW_SIZE, SEQ_FEATURES)

X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

# =========================
# DEFINE MODEL (Same Architecture!)
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
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# =========================
# LOAD MODEL
# =========================
model = TCN().to(device)
model.load_state_dict(torch.load("tcn_volatility_model.pth"))
model.eval()

print("Model loaded successfully.")

# =========================
# EVALUATION
# =========================
with torch.no_grad():
    logits = model(X_test)
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

print("\nClassification Report:\n")
print(classification_report(y_test.cpu(), preds.cpu()))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test.cpu(), preds.cpu()))