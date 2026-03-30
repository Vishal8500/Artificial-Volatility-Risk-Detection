import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pytorch_tcn import TCN

# -----------------------
# Device
# -----------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# -----------------------
# Load dataset
# -----------------------

data = np.load("tcn_dataset.npz")

X = data["X"]
y = data["y"]

print("Dataset shape:", X.shape)

# -----------------------
# Train Test Split
# -----------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------
# Torch tensors
# -----------------------

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=64,
    shuffle=True,
    pin_memory=True
)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=64,
    pin_memory=True
)

# -----------------------
# TCN Model
# -----------------------

class TCNModel(nn.Module):

    def __init__(self, input_features):

        super().__init__()

        self.tcn = TCN(
            num_inputs=input_features,
            num_channels=[32,64,64],
            kernel_size=3,
            dropout=0.2
        )

        self.fc = nn.Linear(64,1)

    def forward(self,x):

        x = x.transpose(1,2)

        y = self.tcn(x)

        y = y[:,:,-1]

        y = self.fc(y)

        return torch.sigmoid(y)


model = TCNModel(X.shape[2]).to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -----------------------
# Training
# -----------------------

EPOCHS = 10

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for X_batch, y_batch in loop:

        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad()

        preds = model(X_batch).squeeze()

        loss = criterion(preds, y_batch)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        loop.set_postfix(loss=loss.item())

    print("Train Loss:", total_loss/len(train_loader))

    # -----------------------
    # Validation
    # -----------------------

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for X_batch, y_batch in test_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            preds = model(X_batch).squeeze()

            preds = (preds > 0.5).float()

            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    acc = correct/total

    print("Validation Accuracy:", acc)

# -----------------------
# Save model
# -----------------------

torch.save(model.state_dict(),"tcn_risk_model.pth")

print("Model saved successfully")