# Imports
from ncps.wirings import AutoNCP
from ncps.torch import CfC
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

## Get data
data_file = 'data/ConstPos_0709.csv'
raw_msg_data = genfromtxt(data_file, delimiter=',')

## Divide dataset into reciever groups
raw_msg_data = np.delete(raw_msg_data, 0, axis=0) # Remove the labels at the beginning of the dataset
sorted_msg_data = raw_msg_data[np.argsort(raw_msg_data[:, 1])] # Sort dataset by receiver ID (receiver id is the 2nd column)

_, freqs = np.unique(sorted_msg_data[:,1], return_counts=True) # Get the indexes of the change in datasets. (count each time they appear)

acc = 0
split_indices = []
for i in range(len(freqs)): # Accumulating counts so that we can use them as indexes
    acc += freqs[i]
    split_indices.append(acc)

split_msg_data = np.split(sorted_msg_data, split_indices) # Split larger dataset into per vehicle datasets. 1d list -> indcies along which to split

windowed_data = [] 
for vehicle_msgs in split_msg_data: # Go through all vehicle datasets
    veh_windows = []
    index = 0
    while index < len(vehicle_msgs) - 10: # organize the new dataset as a list of chunks of 10 messages 
        msg_window = vehicle_msgs[index:index+10]
        veh_windows.append(msg_window)
        index += 5
    veh_windows = torch.Tensor(veh_windows)

    if len(veh_windows) != 0: # Don't add empty data
        windowed_data.append(veh_windows) # Create tensor from per vehicle dataset and add to list of datas.

per_veh_data = windowed_data # List of windowed msgs for each vehicle (overlap 5)

## Proper formatting for testing datasets (centralized)
# Time sequences are 10 timepoints (Messages) with 7 features per message.
# Organized by car, added to one large list.

unq_msg_data, freqs = np.unique(raw_msg_data[:, 2], return_counts = True) # Split by sender id

sender_index = 0
last_sender_coubt = 0
centr_data = []

## Organize dataset into sets of 10 messages by sender
while sender_index < freqs.shape[0]:
    # Loop through sender
    index = 0
    while index < freqs[sender_index] - 10:
        # Loop through messages from sender
        window = raw_msg_data[last_sender_coubt+index:last_sender_coubt +index+10]
        centr_data.append(window)
        index += 5
    
    last_sender_coubt += freqs[sender_index]
    sender_index += 1

centr_data = np.array(centr_data)

## Create seperate datasets for testing and training, using Train Percentage as metric for split
length = int(centr_data.shape[0]/100)
train_end = int(length*(train_perc/100))

x_train = torch.Tensor(centr_data[:train_end,:,3:10]).float()
y_train = centr_data[:train_end,:,11]

x_test = torch.Tensor(centr_data[train_end:length,:,3:10]).float()
y_test = centr_data[train_end:length,:,11]

print((np.unique([np.sum(row) for row in y_test])))
print(len(np.unique([np.sum(row) for row in y_test])))

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

        return self.fc3(x)
    

# Step 2: Create the model

model = Net()

# Step 2a: Define the loss function and the optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Step 3: Create the ART classifier

classifier = PyTorchClassifier(
    model=model,
    clip_values=(x_train.min(), x_train.max()),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(7,),
    nb_classes=2,
)

print("x_train:", x_train.shape)
print("y_train:", y_train.shape)

# Step 4: Train the ART classifier
print("TRAINING")
classifier.fit(x_train, y_train, batch_size=64, nb_epochs=5)
print("DONE TRAINING")

# Step 5: Evaluate the ART classifier on benign test examples

benign_predictions = classifier.predict(x_test)
benign_accuracy = np.sum(np.argmax(benign_predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign tests: \t{:.2f}%".format(benign_accuracy * 100))

# Step 6: Generate adversarial test examples
attack = FastGradientMethod(estimator=classifier, eps=0.1)

x_test_adv = attack.generate(x=x_test)

# Step 7: Evaluate the ART classifier on adversarial test examples

adversarial_predictions = classifier.predict(x_test_adv)
adversarial_accuracy = np.sum(np.argmax(adversarial_predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on adversarial tests: \t{:.2f}%".format(adversarial_accuracy * 100))

print("Difference in accuracy: {:.2f}".format(benign_accuracy-adversarial_accuracy))

benign_pred = np.argmax(benign_predictions, axis=1)
adv_pred = np.argmax(adversarial_predictions, axis=1)

changed = np.mean(benign_pred != adv_pred)

print("Prediction change rate:", changed)