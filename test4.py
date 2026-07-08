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
import os
import time
import csv
import sys
import json

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
    acc += split_indices[i]
    split_indices[i] = acc

split_msg_data = np.split(sorted_msg_data, split_indices) # Split larger dataset into per vehicle datasets. 1d list -> indcies along which to split

windowed_data = [] 
for vehicle_msgs in split_msg_data: # Go through all vehicle datasets
    veh_windows = []
    index = 0
    while index < len(vehicle_msgs) - 10: # organize the new dataset as a list of chuncks of 10 messages 
        msg_window = vehicle_msgs[index:index+10]
        veh_windows.append(msg_window)
        index += 5
    veh_windows = torch.Tensor(veh_windows)

    if veh_windows.shape[0] != 0: # Don't add empty data
        windowed_data.append(veh_windows) # Create tensor from per vehicle dataset and add to list of datas.

per_veh_data = windowed_data # List of windowed msgs for each vehicle (overlap 5)

## Proper formatting for testing datasets (centralized)
# Time sequences are 10 timepoints (Messages) with 7 features per message.
# Organized by car, added to one large list.

unq_msg_data, freqs = np.unique(raw_msg_data[:, 2], return_counts = True) # Split by sender id

sender_index = 0
last_sender_coubt = 0
centr_data = []

# Organize dataset into sets of 10 messages by sender
while sender_index < freqs.shape[0]:
    # Loop through sender
    index = 0
    while index < freqs[sender_index] - 10:
        # Loop through messages from sender
        window = raw_msg_data[last_sender_coubt+index:last_sender_coubt +index+10]
        centr_data.append(window)
        index += 5
    
    last_sender_coubt += freqs[sender_index]
    sender += 1

centr_data = torch.tensor(centr_data) # centralized data


## Create seperate datasets for testing and training, using Train Percentage as metric for split
leng = centr_data.shape[0]
train_end = int(leng*(centr_data/100))

x_centr_train = torch.Tensor(centr_data[:train_end,:,3:10]).float()
y_centr_train = torch.Tensor(np.int_(centr_data[:train_end,:,11])).long()
x_centr_test = torch.Tensor(centr_data[train_end:,:,3:10]).float()
y_centr_test = torch.Tensor(np.int_(centr_data[train_end:,:,11])).long()

tiny_train_inputs = []
tiny_train_labels = []
tiny_test_inputs = []
tiny_test_labels = []
verification_inputs = []
verification_labels = []

train_end_idx = int(leng * (train_perc / 100))

for seq_idx in range(leng):
    if (seq_idx // 10) % 300 == 0:
        verification_inputs.append(centr_data[seq_idx, :, 3:10])
        verification_labels.append(centr_data[seq_idx, :, 11])

for seq_idx in range(train_end_idx):
    if (seq_idx // 10) % 100 == 0:
        tiny_train_inputs.append(centr_data[seq_idx, :, 3:10])
        tiny_train_labels.append(centr_data[seq_idx, :, 11])

for seq_idx in range(train_end_idx, leng):
    if (seq_idx // 10) % 10 == 0:
        tiny_test_inputs.append(centr_data[seq_idx, :, 3:10])
        tiny_test_labels.append(centr_data[seq_idx, :, 11])

x_tiny_train = torch.tensor(np.array(tiny_train_inputs), dtype=torch.float32)
y_tiny_train = torch.tensor(np.array(tiny_train_labels), dtype=torch.long)
x_tiny_test_small = torch.tensor(np.array(tiny_test_inputs), dtype=torch.float32)
y_tiny_test_small = torch.tensor(np.array(tiny_test_labels), dtype=torch.long)
x_tiny_test = torch.tensor(np.array(verification_inputs), dtype=torch.float32)
y_tiny_test = torch.tensor(np.array(verification_labels), dtype=torch.long)

# Create Dataloaders for all the datasets
centr_train_loader = data.DataLoader(data.TensorDataset(x_centr_train, y_centr_train), batch_size=batch_size, shuffle=False, num_workers=10, persistent_workers = True, drop_last= True)
centr_test_loader = data.DataLoader(data.TensorDataset(x_centr_test, y_centr_test), batch_size=batch_size, shuffle=False, num_workers=10, persistent_workers = True, drop_last= True)
tiny_train_loader = data.DataLoader(data.TensorDataset(x_tiny_train, y_tiny_train), batch_size=batch_size, shuffle = False, num_workers=10, persistent_workers = True, drop_last= True)
tiny_test_loader = data.DataLoader(data.TensorDataset(x_tiny_test, y_tiny_test), batch_size=batch_size, shuffle = False, num_workers = 10, persistent_workers= True)

print("\n=======")