# Imports
from ncps.wirings import AutoNCP
from ncps.torch import CfC
from basic_attacks.windowed.utils import get_windowed_data
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from numpy import genfromtxt
import numpy as np
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import torch.nn as nn
from random import sample
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.preprocessing import MinMaxScaler

torch.set_float32_matmul_precision("high")

## Constants
batch_size = 64
train_perc = 80
data_file = './data/RandomPos_0709.csv'

## Get training/testing data
(x_train, y_train), (x_test, y_test) = get_windowed_data(data_file, train_perc)

## Define NN model

class Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(7, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x[:, -1, :])
    

# Step 2: Create the model

model = Net()

# Step 2a: Define the loss function and the optimizer

class CrossEntropyLossOH(nn.Module):
    def forward(self, input, target):
        if target.is_floating_point():
            target = target.argmax(dim=-1)
        return F.cross_entropy(input, target)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Step 3: Create the ART classifier

classifier = PyTorchClassifier(
    model=model,
    clip_values=(x_train.min(), x_train.max()),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(10,7),
    nb_classes=2,
)

print("x_train:", x_train.shape)
print("y_train:", y_train.shape)

# Step 4: Train the ART classifier
print("TRAINING")
classifier.fit(x_train, y_train, batch_size=64, nb_epochs=50)
print("DONE TRAINING")

# Step 5: Evaluate the ART classifier on benign test examples

benign_predictions = classifier.predict(x_test)

benign_accuracy = np.sum(np.argmax(benign_predictions, axis=1) == y_test) / len(y_test)
print("Accuracy on benign tests: \t{:.2f}%".format(benign_accuracy * 100))

# Step 6: Generate adversarial test examples
attack = FastGradientMethod(estimator=classifier, eps=0.1)

x_test_adv = attack.generate(x=x_test)

# Step 7: Evaluate the ART classifier on adversarial test examples

adversarial_predictions = classifier.predict(x_test_adv)
adversarial_accuracy = np.sum(np.argmax(adversarial_predictions, axis=1) == y_test) / len(y_test)
print("Accuracy on adversarial tests: \t{:.2f}%".format(adversarial_accuracy * 100))

print("Difference in accuracy: {:.2f}".format(benign_accuracy-adversarial_accuracy))

benign_pred = np.argmax(benign_predictions, axis=1)
adv_pred = np.argmax(adversarial_predictions, axis=1)

changed = np.mean(benign_pred != adv_pred)

print("Prediction change rate:", changed)