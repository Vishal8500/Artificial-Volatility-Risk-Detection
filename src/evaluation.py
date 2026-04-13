import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from pytorch_tcn import TCN
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve
# ------------------------
# Device
# ------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# ------------------------
# Load dataset
# ------------------------

data = np.load("tcn_dataset.npz")

X = data["X"]
y = data["y"]

print("Dataset shape:", X.shape)

# same split used in training
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=64
)

# ------------------------
# Model definition
# ------------------------

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


# ------------------------
# Load trained model
# ------------------------

model = TCNModel(X.shape[2]).to(device)

model.load_state_dict(torch.load("tcn_risk_model.pth"))

model.eval()

print("Model loaded successfully")

# ------------------------
# Evaluation
# ------------------------

all_preds = []
all_probs = []
all_labels = []

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        X_batch = X_batch.to(device)

        outputs = model(X_batch).squeeze()

        probs = outputs.cpu().numpy()

        preds = (outputs > 0.5).float().cpu().numpy()

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())


all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# ------------------------
# Metrics
# ------------------------

print("\nClassification Report\n")

print(classification_report(all_labels, all_preds))

print("\nConfusion Matrix\n")

print(confusion_matrix(all_labels, all_preds))

print("\nROC-AUC Score:", roc_auc_score(all_labels, all_probs))
cm = confusion_matrix(all_labels, all_preds)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

plt.figure()
disp.plot()
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()

print("Confusion matrix saved as confusion_matrix.png")

# ------------------------
# Save ROC Curve
# ------------------------

fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
auc_score = roc_auc_score(all_labels, all_probs)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("roc_curve.png", dpi=300, bbox_inches='tight')
plt.close()

print("ROC curve saved as roc_curve.png")