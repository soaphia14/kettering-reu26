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

batchSize = 64

# FORMATTING DATASET FOR FED. LEARNING
testName = 'ConstPos'
doEvil = False
percEvil = 20
dataFile = 'data/ConstPos_0709.csv'
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
newsetIn = []
newsetOut = []
testsetIn = []
testsetOut = []
tinyTestIn = []
tinyTestOut = []
# Create tiny dataset to run verification tests
for index in range(leng):
    if not (int(index/10) % 300):
        tinyTestIn.append(dataSet[index, :, 3:10])
        tinyTestOut.append(dataSet[index, :, 11])
# Create dataset of 1/100th of the entries for quicker testing during development
for index in range(0,int((leng) * (trainPerc/100))):
    if not (int(index/10) % 100):
        newsetIn.append(dataSet[index,:,3:10])
        newsetOut.append((dataSet[index,:,11]))
for idx in range(int((leng) * (trainPerc/100)), leng):
    if not (int(idx/10) % 10):
        testsetIn.append(dataSet[idx,:,3:10])
        testsetOut.append((dataSet[idx,:,11]))
testingIn = torch.Tensor(np.array(newsetIn)).float()
testingOut = torch.Tensor(np.array(newsetOut)).long()
inTest = torch.Tensor(np.array(testsetIn)).float()
outTest = torch.Tensor(np.array(testsetOut)).long()
tinyTestIn = torch.Tensor(np.array(tinyTestIn)).float()
tinyTestOut = torch.Tensor(np.array(tinyTestOut)).long()
# Create Dataloaders for all the datasets

# dataset used for training
# 80% of the dataest for training
dataLoaderTrain = data.DataLoader(data.TensorDataset(trainDataIn, trainDataOut), batch_size=batchSize, shuffle=False, num_workers=10, persistent_workers = True, drop_last= True)
# remaining 20% of the dataset
dataLoaderTest = data.DataLoader(data.TensorDataset(testDataIn, testDataOut), batch_size=batchSize, shuffle=False, num_workers=10, persistent_workers = True, drop_last= True)
# quick testing
testingDataLoader = data.DataLoader(data.TensorDataset(testingIn, testingOut), batch_size=batchSize, shuffle = False, num_workers=10, persistent_workers = True, drop_last= True)
# even smaller testing
tinyTestLoader = data.DataLoader(data.TensorDataset(tinyTestIn, tinyTestOut), batch_size=batchSize, shuffle = False, num_workers = 10, persistent_workers= True)

print(len(newsetIn))
print(len(tinyTestIn))
print(len(testsetIn))
print(len(testsetOut))