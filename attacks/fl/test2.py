"""
File to test data loading with the original dataset.
For debugging largely.

"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

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
from attacks.fl.models import OBU
 

torch.set_float32_matmul_precision("high")

batchSize = 64

# ----
# Definitions
subEpochs = 10
gpu = False
lr = 0.001
motors = 8
units = 20

data_filename = "RandomPos_0709.csv"
ckpt_filename = "mainModelBackup.ckpt"
train_perc = 80

# ----
data_file = f"data/{data_filename}"
checkpoint_file = f"attacks/fl/checkpoints/{ckpt_filename}"

if not os.path.exists(checkpoint_file):
    print(os.listdir("/"))
    raise ValueError(f"Checkpoint file {checkpoint_file} does not exist.")
else:
    print("Checkpoint path exists!")

# FORMATTING DATASET FOR FED. LEARNING
testName = 'RandPos'
doEvil = False
percEvil = 20
dataFile = 'data/RandomPos_0709.csv'
dataSet = genfromtxt(dataFile, delimiter=',')
dataSet = np.delete(dataSet, 0, axis=0) # Remove the labels at the beginning of the dataset

# Devide dataset into reciever groups
fedDataSet = dataSet[np.argsort(dataSet[:, 1])] # Sort dataset by reciver ID
_, counts = np.unique(fedDataSet[:,1], return_counts=True) # Get the indexes of the change in datasets.
sum = 0
for i in range(len(counts)): # Accumulating counts so that we can use them as indexes
    sum += counts[i]
    counts[i] = sum
fedDataSet = np.split(fedDataSet, counts) # Split larger dataset into per vehicle datasets.

newData = [] 
for reciever in fedDataSet: # Go through all vehicle datasets
    subData = []
    index = 0
    while index < len(reciever) - 10: # organize the new dataset as a list of chuncks of 10 messages 
        subData.append(reciever[index:index+10])
        index += 5
    subData = torch.Tensor(subData)
    if subData.shape[0] != 0:
        newData.append(subData) # Create tensor from per vehicle dataset and add to list of datas.
fedDataSet = newData
# Final output of this cell is fedDataSet, a list of the datasets of each vehicle.


#PROPER FORMATTING FOR TESTING DATASETS
#Time sequences are 10 timepoints (Messages) with 7 features per message.
#Organized by car.
unq, counts = np.unique(dataSet[:, 2], return_counts = True)
sender = 0
lastSenderCount = 0
newData = []
# Organize dataset into sets of 10 messages by sender
while sender < counts.shape[0]:
    # Loop through sender
    index = 0
    while index < counts[sender] - 10:
        # Loop through messages from sender
        newData.append(dataSet[lastSenderCount+index:lastSenderCount +index+10])
        index += 5
    sender += 1
    lastSenderCount += counts[sender-1]
dataSet = torch.tensor(newData)
leng = dataSet.shape[0]
trainPerc = 80
# Create seperate datasets for testing and training, using Train Percentage as metric for split
trainDataIn = torch.Tensor(dataSet[:int(leng*(trainPerc/100)),:,3:10]).float()
trainDataOut = torch.Tensor(np.int_(dataSet[:int(leng*(trainPerc/100)),:,11])).long()
testDataIn = torch.Tensor(dataSet[int(leng*(trainPerc/100)):,:,3:10]).float()
testDataOut = torch.Tensor(np.int_(dataSet[int(leng*(trainPerc/100)):,:,11])).long()



# Testing the model

model = OBU(7, epochs= subEpochs, gpu = gpu, lr = lr, motors = motors, units = units)



checkpoint = torch.load(checkpoint_file)
model.learner.load_state_dict(checkpoint)

model.test(testDataIn, testDataOut)
