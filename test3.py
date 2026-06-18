# Test the simple model

# =====================================================
# IMPORTS
# =====================================================

from numpy import genfromtxt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

# =====================================================
# SETTINGS
# =====================================================

batchSize = 64
trainPerc = 80

epochs = 5

dataFile = "data/RandomPos_0709.csv"
modelFile = f"saved_models/basic_nn_random_epoch{epochs}.pth"

# =====================================================
# MODEL DEFINITION
# =====================================================

class BasicNN(nn.Module):

    def __init__(
        self,
        input_size=70,
        hidden_size=128,
        num_classes=2
    ):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 10 * num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.num_classes = num_classes

    def forward(self, x):

        batch_size = x.shape[0]

        x = x.reshape(batch_size, -1)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.relu(self.fc2(x))

        x = self.fc3(x)

        x = x.reshape(
            batch_size,
            10,
            self.num_classes
        )

        return x


# =====================================================
# LOAD DATASET
# =====================================================

print("Loading dataset...")

dataSet = genfromtxt(
    dataFile,
    delimiter=","
)

# remove header row
dataSet = np.delete(
    dataSet,
    0,
    axis=0
)

# =====================================================
# CREATE WINDOWS
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
# RECREATE TEST SPLIT
# =====================================================

length = dataSet.shape[0]

train_end = int(
    length * (trainPerc / 100)
)

testDataIn = (
    dataSet[train_end:, :, 3:10]
    .float()
)

testDataOut = (
    dataSet[train_end:, :, 11]
    .long()
)

# same label correction used during training

if torch.min(testDataOut) < 0:
    testDataOut = testDataOut - torch.min(testDataOut)

num_classes = (
    int(torch.max(testDataOut).item()) + 1
)

print("Number of classes:", num_classes)

# =====================================================
# TEST LOADER
# =====================================================

testLoader = data.DataLoader(
    data.TensorDataset(
        testDataIn,
        testDataOut
    ),
    batch_size=batchSize,
    shuffle=False
)

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
# LOAD MODEL
# =====================================================

checkpoint = torch.load(
    modelFile,
    map_location=device
)

model = BasicNN(
    input_size=70,
    hidden_size=128,
    num_classes=checkpoint["num_classes"]
).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model loaded successfully.")

# =====================================================
# TEST MODEL
# =====================================================

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
    100.0 * correct / total
)

print(f"Test Accuracy: {accuracy:.2f}%")

# =====================================================
# SHOW EXAMPLE PREDICTIONS
# =====================================================

# print("\nExample predictions:")

# with torch.no_grad():

#     inputs, labels = next(iter(testLoader))

#     inputs = inputs.to(device)

#     outputs = model(inputs)

#     predictions = torch.argmax(
#         outputs,
#         dim=2
#     )

#     for i in range(min(5, len(predictions))):

#         print("\nSequence", i)

#         print("Actual:")
#         print(labels[i].cpu().numpy())

#         print("Predicted:")
#         print(predictions[i].cpu().numpy())