"""
Testing data loading with this repo's setup (utils.py).
Attempting to test with adversarial attacks. 

"""

import sys
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod
from sklearn.preprocessing import MinMaxScaler

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

class ARTCfCWrapper(nn.Module):
    def __init__(self, modena_model):
        super().__init__()
        self.modena_model = modena_model

    def forward(self, x):
        logits, _ = self.modena_model(x)

        # Shape:
        # (batch, seq_len, num_classes)
        return logits

class SequenceCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, a, b):
        print("A:", a.shape)
        print("B:", b.shape)

        return self.loss(
            a.permute(0, 2, 1),
            b.long()
        )
        
wrapped_model = ARTCfCWrapper(model.model)
criterion = SequenceCrossEntropy()
optimizer = optim.Adam(
    wrapped_model.parameters(),
    lr=0.001
)

classifier = PyTorchClassifier(
    model=wrapped_model,
    loss=criterion,
    optimizer=optimizer,
    input_shape=(10, 7),
    nb_classes=2,
    clip_values=None,
)

print(type(model))
print(type(model.model))
print(type(wrapped_model))

# ----
# Load data
data_file = f"data/{data_filename}"
(x_train, y_train), (x_test, y_test), fed_dataset = get_windowed_data(data_file, train_perc)

# ----
# # Test the model
# out = model.test(x_test, y_test, mathy=True)
# print(f"Out: {out}")

# Adversarial test
attack = FastGradientMethod(estimator=classifier, eps=0.1)
x_test_adv = attack.generate(x=x_test)

adversarial_predictions = classifier.predict(x_test_adv)
adversarial_accuracy = np.sum(np.argmax(adversarial_predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print(f"Adversarial accuracy: {adversarial_accuracy}")