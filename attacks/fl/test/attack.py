"""
Testing data loading with this repo's setup (utils.py).
Attempting to make a cleaner way to test the model

"""

import sys
from pathlib import Path
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier

sys.path.append(str(Path(__file__).resolve().parents[3]))

import os

import torch
from attacks.fl.models import OBU
from attacks.fl.utils import get_windowed_data, load_model_checkpoint

# ----
# Input
data_filename = "RandomPos_0709.csv"
ckpt_filename = "mainModelBackup.ckpt"
train_perc = 80

# ----
# Load model
checkpoint_file = f"attacks/fl/checkpoints/{ckpt_filename}"
model = load_model_checkpoint(checkpoint_file, gpu=False)

# ----
# Load data
data_file = f"data/{data_filename}"
(x_train, y_train), (x_test, y_test), fed_dataset = get_windowed_data(data_file, train_perc)

# ----
# Test the model
out = model.test(x_test, y_test, mathy=True)
print(f"Out: {out}")

# ----
# Generate attack
# Step 6: Generate adversarial test examples
attack = FastGradientMethod(model, eps=0.1)

x_test_adv = attack.generate(x=x_test)