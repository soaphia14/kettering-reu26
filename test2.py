# Train a simple model

# =====================================================
# IMPORTS
# =====================================================

from numpy import genfromtxt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import os

torch.set_float32_matmul_precision("high")

# =====================================================
# SETTINGS
# =====================================================

batchSize = 64
trainPerc = 80
epochs = 5

dataFile = "data/ConstPos_0709.csv"
modelFile = f"basic_nn_random_epoch{epochs}.pth"

# =====================================================
# LOAD CSV
# =====================================================

print("Loading dataset...")

dataSet = genfromtxt(
    dataFile,
    delimiter=","
)

# Remove header row
dataSet = np.delete(
    dataSet,
    0,
    axis=0
)

print("Dataset shape:", dataSet.shape)

# =====================================================
# CREATE SEQUENCE WINDOWS
# =====================================================

unq, counts = np.unique(
    dataSet[:, 2],
    return_counts=True
)

sender = 0
lastSenderCount = 0

newData = []

while sender < counts.shape[0]:

    index = 0

    while index < counts[sender] - 10:

        newData.append(
            dataSet[
                lastSenderCount + index :
                lastSenderCount + index + 10
            ]
        )

        index += 5

    sender += 1
    lastSenderCount += counts[sender - 1]

dataSet = torch.tensor(newData)

print("Windowed dataset shape:", dataSet.shape)

# =====================================================
# SPLIT INPUTS / LABELS
# =====================================================

length = dataSet.shape[0]

train_end = int(
    length * (trainPerc / 100)
)

# Inputs:
# Columns 3-9 inclusive
trainDataIn = (
    dataSet[:train_end, :, 3:10]
    .float()
)

testDataIn = (
    dataSet[train_end:, :, 3:10]
    .float()
)

# Labels:
# Column 11

trainDataOut = (
    dataSet[:train_end, :, 11]
    .long()
)

testDataOut = (
    dataSet[train_end:, :, 11]
    .long()
)

# =====================================================
# FIX LABELS IF NEEDED
# =====================================================

unique_labels = torch.unique(trainDataOut)

print("Labels found:", unique_labels)

# Example:
# convert [-1,1] -> [0,1]

if torch.min(trainDataOut) < 0:

    trainDataOut = trainDataOut - torch.min(trainDataOut)
    testDataOut = testDataOut - torch.min(testDataOut)

num_classes = (
    int(torch.max(trainDataOut).item()) + 1
)

print("Number of classes:", num_classes)

# =====================================================
# DATALOADERS
# =====================================================

trainLoader = data.DataLoader(
    data.TensorDataset(
        trainDataIn,
        trainDataOut
    ),
    batch_size=batchSize,
    shuffle=True
)

testLoader = data.DataLoader(
    data.TensorDataset(
        testDataIn,
        testDataOut
    ),
    batch_size=batchSize,
    shuffle=False
)

# =====================================================
# MODEL
# =====================================================

class BasicNN(nn.Module):

    def __init__(
        self,
        input_size=70,
        hidden_size=128,
        num_classes=2
    ):
        super().__init__()

        self.fc1 = nn.Linear(
            input_size,
            hidden_size
        )

        self.fc2 = nn.Linear(
            hidden_size,
            hidden_size
        )

        self.fc3 = nn.Linear(
            hidden_size,
            10 * num_classes
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.2)

        self.num_classes = num_classes

    def forward(self, x):

        batch_size = x.shape[0]

        # (batch,10,7)
        # -> (batch,70)

        x = x.reshape(
            batch_size,
            -1
        )

        x = self.relu(
            self.fc1(x)
        )

        x = self.dropout(x)

        x = self.relu(
            self.fc2(x)
        )

        x = self.fc3(x)

        x = x.reshape(
            batch_size,
            10,
            self.num_classes
        )

        return x

# =====================================================
# DEVICE
# =====================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)

# =====================================================
# INITIALIZE MODEL
# =====================================================

model = BasicNN(
    input_size=70,
    hidden_size=128,
    num_classes=num_classes
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# =====================================================
# TRAIN
# =====================================================

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for inputs, labels in trainLoader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        outputs = outputs.reshape(
            -1,
            num_classes
        )

        labels = labels.reshape(-1)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = (
        running_loss /
        len(trainLoader)
    )

    print(
        f"Epoch {epoch+1}/{epochs} "
        f"Loss: {avg_loss:.4f}"
    )

# =====================================================
# TEST
# =====================================================

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for inputs, labels in testLoader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        predictions = torch.argmax(
            outputs,
            dim=2
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.numel()

accuracy = (
    100 * correct / total
)

print(
    f"Test Accuracy: "
    f"{accuracy:.2f}%"
)

# =====================================================
# SAVE MODEL
# =====================================================

save_dir = "saved_models"

os.makedirs(
    save_dir,
    exist_ok=True
)

save_path = os.path.join(
    save_dir,
    modelFile
)

torch.save(
    {
        "model_state_dict":
            model.state_dict(),
        "optimizer_state_dict":
            optimizer.state_dict(),
        "num_classes":
            num_classes
    },
    save_path
)

print(
    f"Model saved to: {save_path}"
)

