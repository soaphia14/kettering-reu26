import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod
from art.defences.detector.evasion import BeyondDetectorPyTorch

from attacks.basic.utils import create_dataset, SimpleNet

# ----
# Config
data_file = "data/RandomPos_0709.csv"
model_filename = "RandomPos-full"
eps = 0.04

# ----
# Step 1: Load data
(x_train, y_train), (x_test, y_test) = create_dataset(data_file=data_file, divide_by=1)

# BeyondDetector expects 4D input (batch, channels, height, width).
# Reshape tabular (N, 11) -> (N, 1, 1, 11) to satisfy this requirement.
x_train_4d = x_train[:, np.newaxis, np.newaxis, :].astype(np.float32)
x_test_4d  = x_test[:,  np.newaxis, np.newaxis, :].astype(np.float32)

# ----
# Step 2: Build target classifier (the model being protected)
target_net = SimpleNet()
criterion  = nn.CrossEntropyLoss()
optimizer  = optim.Adam(target_net.parameters(), lr=0.01)

target_classifier = PyTorchClassifier(
    model=target_net,
    clip_values=(x_train.min(), x_train.max()),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(11,),
    nb_classes=2,
)
target_net.load_state_dict(torch.load(f"saved_models/{model_filename}.pth", weights_only=True))
target_net.eval()

# ----
# Step 3: Build SSL model required by BEYOND.
# In practice this would be trained with contrastive learning (e.g. SimCLR).
# Here we define the structure; swap in a properly trained model for real results.

class TabularBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(11, 64)
        self.fc2 = nn.Linear(64, 64)

    def forward(self, x):
        x = x.view(x.shape[0], -1)   # (N, 1, 1, 11) -> (N, 11)
        return F.relu(self.fc2(F.relu(self.fc1(x))))

class TabularProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 32)

    def forward(self, x):
        return F.normalize(self.fc(x), dim=-1)

class TabularSSLNet(nn.Module):
    """Wrapper exposing backbone / projector / classifier as top-level attributes."""
    def __init__(self):
        super().__init__()
        self.backbone   = TabularBackbone()
        self.projector  = TabularProjector()
        self.classifier = nn.Linear(64, 2)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

ssl_net = TabularSSLNet()

ssl_classifier = PyTorchClassifier(
    model=ssl_net,
    clip_values=(x_train.min(), x_train.max()),
    loss=nn.CrossEntropyLoss(),
    optimizer=optim.Adam(ssl_net.parameters(), lr=0.01),
    input_shape=(1, 1, 11),
    nb_classes=2,
)

# ----
# Step 4: Define augmentations for tabular data.
# BEYOND applies these to generate neighborhoods around each sample.
def tabular_augment(x: torch.Tensor) -> torch.Tensor:
    """Add small Gaussian noise and clamp to original data range."""
    return (x + torch.randn_like(x) * 0.01).clamp(
        float(x_train.min()), float(x_train.max())
    )

# ----
# Step 5: Build the BEYOND detector
detector = BeyondDetectorPyTorch(
    target_classifier=target_classifier,
    ssl_classifier=ssl_classifier,
    augmentations=tabular_augment,
    aug_num=50,
    alpha=0.8,
    var_K=20,
    percentile=5,
)

# Step 6: Fit detector threshold on clean training samples
print("Fitting BEYOND detector threshold on clean data...")
detector.fit(x_train_4d)
print(f"Threshold set to: {detector.threshold:.6f}")

# ----
# Step 7: Generate adversarial examples (on the flat input, using the target classifier)
attack = FastGradientMethod(estimator=target_classifier, eps=eps)
x_test_adv = attack.generate(x=x_test)
x_test_adv_4d = x_test_adv[:, np.newaxis, np.newaxis, :].astype(np.float32)

# ----
# Step 8: Detect
_, is_adv_on_clean = detector.detect(x_test_4d)
_, is_adv_on_adv   = detector.detect(x_test_adv_4d)

fpr = np.mean(is_adv_on_clean)   # false positive rate on clean inputs
tpr = np.mean(is_adv_on_adv)     # true positive rate on adversarial inputs

print(f"\n=== BEYOND Detector (eps={eps}) ===")
print(f"False positive rate (clean flagged as adv): {fpr:.2%}")
print(f"True positive rate  (adv correctly flagged): {tpr:.2%}")
