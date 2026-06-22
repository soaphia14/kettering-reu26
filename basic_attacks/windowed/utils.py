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

def get_windowed_data(data_file : str = './data/RandomPos_0709.csv', train_perc : int = 80):
    ## Get data
    raw_msg_data = genfromtxt(data_file, delimiter=',')

    ## Divide dataset into reciever groups
    scaler = MinMaxScaler()
    raw_msg_data = np.delete(raw_msg_data, 0, axis=0) # Remove the labels at the beginning of the dataset
    raw_msg_data = scaler.fit_transform(raw_msg_data)
    sorted_msg_data = raw_msg_data[np.argsort(raw_msg_data[:, 1])] # Sort dataset by receiver ID (receiver id is the 2nd column)

    _, freqs = np.unique(sorted_msg_data[:,1], return_counts=True) # Get the indexes of the change in datasets. (count each time they appear)

    acc = 0
    split_indices = []
    for i in range(len(freqs)): # Accumulating counts so that we can use them as indexes
        acc += freqs[i]
        split_indices.append(acc)

    split_msg_data = np.split(sorted_msg_data, split_indices) # Split larger dataset into per vehicle datasets. 1d list -> indcies along which to split

    windowed_fed_data = [] 
    for vehicle_msgs in split_msg_data: # Go through all vehicle datasets
        veh_windows = []
        index = 0
        while index < len(vehicle_msgs) - 10: # organize the new dataset as a list of chunks of 10 messages 
            msg_window = vehicle_msgs[index:index+10]
            veh_windows.append(msg_window)
            index += 5
        veh_windows = torch.Tensor(veh_windows)

        if len(veh_windows) != 0: # Don't add empty data
            windowed_fed_data.append(veh_windows) # Create tensor from per vehicle dataset and add to list of datas.

    ## Proper formatting for testing datasets (centralized)
    # Time sequences are 10 timepoints (Messages) with 7 features per message.
    # Organized by car, added to one large list.

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
    length = int(centr_data.shape[0])
    train_end = int(length*(train_perc/100))


    x_train = np.array(centr_data[:train_end,:,3:10]).astype(np.float32)
    y_train = (np.array(centr_data[:train_end, 0, 11]) > 0).astype(np.int64)

    x_test = np.array(centr_data[train_end:length,:,3:10]).astype(np.float32)
    y_test = (np.array(centr_data[train_end:length, 0, 11]) > 0).astype(np.int64)

    return (x_train, y_train), (x_test, y_test), windowed_fed_data
