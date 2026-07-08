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
from art.attacks.evasion import ProjectedGradientDescent
from art.attacks.evasion import FastGradientMethod
from sklearn.preprocessing import MinMaxScaler

sys.path.append(str(Path(__file__).resolve().parents[3]))

import os

import torch
from attacks.fl.models import OBU
from attacks.fl.utils import get_windowed_data, load_model_checkpoint

# ----
# Input
checkpoint_file="saved_models/RandomPos-cenFL.ckpt"
data_filename = "RandomPos_0709.csv"
train_perc = 80

# ----
# Load model
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
        # ART passes one-hot labels; CrossEntropyLoss needs class indices
        if b.dim() == 3:
            b = b.argmax(dim=-1)  # (batch, seq_len, num_classes) → (batch, seq_len)
        return self.loss(
            a.permute(0, 2, 1),  # (batch, seq_len, C) → (batch, C, seq_len)
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
    nb_classes=20,
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
# Test the model
out = model.test(x_test, y_test, mathy=True)
print(f"Out: {out}")

# Benign test
benign_predictions = classifier.predict(x_test)
benign_pred_classes = np.argmax(benign_predictions, axis=-1)  # (N, seq_len)
accuracy = np.sum(benign_pred_classes == y_test) / benign_pred_classes.size
print(f"Benign accuracy: {accuracy}")

# Adversarial test
attack = FastGradientMethod(estimator=classifier, eps=0.2)
x_test_adv = attack.generate(x=x_test)

adversarial_predictions = classifier.predict(x_test_adv)
pred_classes = np.argmax(adversarial_predictions, axis=-1)  # (N, seq_len)
adversarial_accuracy = np.sum(pred_classes == y_test) / pred_classes.size
print(f"Adversarial accuracy: {adversarial_accuracy}")