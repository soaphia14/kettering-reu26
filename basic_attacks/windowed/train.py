# Imports
from ncps.wirings import AutoNCP
from ncps.torch import CfC
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
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
from config import OBU, OutLogger
from utils import get_windowed_data

# Variables need to define

## Constants
batch_size = 64
train_perc = 80
data_file = './data/RandomPos_0709.csv'
percEvil = 20

## Get training/testing data
(x_train, y_train), (x_test, y_test), fed_dataset = get_windowed_data(data_file, train_perc)


# Tested with weights, but weighing off of the loss leads to choosing the models trained by vehicles without attacks.
pl.seed_everything(1000)
results = {}
dataSets = {}
models = {}
histWeights = []
percentages = []
cars = []
rcvrs = []
percs = {}
states = {}
subEpochs = 10 # 50
epochs = 60 # 5
vehicles = 200 # 50
lr = 0.01
motors = 8
units = 20
batchSize = 64
gpu = False
deepTest = False
weighing = False
randomVehicles = False
doValidation = False
avgLossVEpoch = []
avgF1VEpoch = []
avgRecallVEpoch = []
avgPrecisionVEpoch = []

# Create starting models
mainModel = OBU(7, epochs= subEpochs, gpu = gpu, lr = lr, motors = motors, units = units)
nextModel = OBU(7, epochs= subEpochs, gpu = gpu, lr = lr, motors = motors, units = units)
path = f"test/"
if not os.path.exists(f"out/{path}"):
    os.makedirs(f"out/{path}")

log = OutLogger(path)

if not randomVehicles:
    # Divide dataset of recieving vehicles among OBUs
    rcvrs = []
    for vehicle in fed_dataset[500:vehicles+500]: # 10
        rcvrID = int(vehicle[0,0,2].item())
        # Add new OBU for each model
        if np.random.randint(0,100) < percEvil:
            models[rcvrID] = OBU(7, epochs = subEpochs, gpu=gpu, lr = lr, motors = motors, units = units, evil = True)
        else:
            models[rcvrID] = OBU(7, epochs = subEpochs, gpu=gpu, lr = lr, motors = motors, units = units)
        # Create Slice of dataset
        vehicle = data.DataLoader(data.TensorDataset(vehicle[:,:,3:10].float(), vehicle[:,:,11].long()), batch_size=batchSize, shuffle=False, num_workers=16, persistent_workers = True) # type: ignore
        # Add sub - dataset to dataset
        rcvrs.append(rcvrID)
        dataSets[rcvrID]=vehicle
        models[rcvrID].dataset = vehicle

# Train individual models and combine
for epoch in range(epochs):
    i = 0
    if randomVehicles:
        # choose random assortment of vehicles to train on
        cars = []
        rcvrs = []
        for ted in np.random.choice(len(fed_dataset), vehicles, replace = False):
            cars.append(fed_dataset[ted])
        # Divide dataset of recieving vehicles among OBUs
        for vehicle in cars: # 10
            rcvrID = int(vehicle[0,0,2].item())
            # Add new OBU for each model
            if rcvrID not in models:
                if np.random.randint(0,100) < percEvil:
                    models[rcvrID] = OBU(7, epochs = subEpochs, gpu=gpu, lr = lr, motors = motors, units = units, evil = True)
                else:
                    models[rcvrID] = OBU(7, epochs = subEpochs, gpu=gpu, lr = lr, motors = motors, units = units)
            # Create Slice of dataset
            vehicle = data.DataLoader(data.TensorDataset(vehicle[:,:,3:10].float(), vehicle[:,:,11].long()), batch_size=batchSize, shuffle=False, num_workers=16, persistent_workers = True)
            # Add sub - dataset to dataset
            rcvrs.append(rcvrID)
            dataSets[rcvrID]=vehicle
            models[rcvrID].dataset = vehicle

    print(rcvrs, file = open('rcvrs.txt', 'w'))
    # Baseline model to add everything to. !!Do I want this or should it be a completely new model?!! Got 0% on combination before, testing with new model for next model.
    nextModel = OBU(7, epochs= subEpochs, gpu = gpu, lr = lr, motors = motors, units = units)
    # Train models
    weights = []
    for rcvr in rcvrs:
        log.startEpochTimer()
        log.startVehicleTimer()
        # Make multithreaded?
        if doValidation and models[rcvr].prevAccuracy != 0:
            _, _, p, _ = mainModel.test(x_test, y_test)
            print(f"Current Epoch: {epoch}, Reciever: {rcvr}")
            print(f"Tested. main perc: {p}, my perc: {models[rcvr].prevAccuracy}")
            if p > models[rcvr].prevAccuracy:
                print("Updating Model")
                # set model to main model, and train that 
                models[rcvr].setState(mainModel.getState())
        else:
            models[rcvr].setState(mainModel.getState())
        # Reset the trainer (should be unneeded now) to allow for further training
        # mod.resetTrainer()
        # Actually train
        models[rcvr].updateSavedStates()
        modLoss = models[rcvr].step(subEpochs)

        weights.append(1/modLoss)
        if deepTest or doValidation:
            _, _ , perc, _ = models[rcvr].test(x_test, y_test)
            percs[rcvr] = perc
            models[rcvr].prevAccuracy = perc
            # Test individual model
            print(rcvr)
            if rcvr not in results:
                results[rcvr] = ([epoch, i+1, perc, modLoss.item()])
            else:
                results[rcvr].append([epoch, i+1, perc, modLoss.item()])
        states[rcvr] = (models[rcvr].getState())
        i+=1
        log.endEpochTimer()
        log.endVehicleTimer()
    # Create combined model
    # combine models
    weights = np.abs(weights)/np.sum(weights)

    log.updateLogs([models[rcvr] for rcvr in rcvrs], epoch, x_test, y_test)

    y=0
    for w in weights:
        y+= w
    teds = [[float(weights[n]), percs[n]] for n in range(len(percs))]
    percentages.append(teds)
    histWeights.append([weights,y])
    tom = 1/i
    for rcvr in rcvrs:
        if weighing:
            nextState = nextModel.setState(nextModel.getSavedState(), dict((n, states[rcvr].get(n, 0)*weights[rcvr]) for n in states[rcvr])) # Done with weights
        else:
            nextState = nextModel.setState(nextModel.getSavedState(), states[rcvr]) # No Weights
    if weighing:
        mainModel.setState(nextState) #Weights
    else:
        mainModel.setState(dict((n, nextState.get(n, 0)/i) for n in nextState)) # No weights
# Test combined model at end
log.finalLogs([models[rcvr] for rcvr in rcvrs], percEvil)
perc = mainModel.test(x_test, y_test)
results['FINAL'] = [-1, -1, perc]
evils = []
for rcvr in rcvrs: # Create list of evil/bad vehicles
    if models[rcvr].isEvil():
        evils.append(models[rcvr].id)
print(evils, file=open(f'out/{path}VehicleStatus.txt','w'))
print(results, file = open(f'out/{path}results.txt', 'w'))
print(histWeights, file = open(f'out/{path}Weights.txt', 'w'))
print(percentages, file = open(f'out/{path}Percs.txt', 'w'))
log.log()

