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
from sklearn.preprocessing import MinMaxScaler

torch.set_float32_matmul_precision("high")

from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.models import CfCLearner, Modena, OutLogger, OBU

start_time = time.time_ns()


batch_size = 64

# --- FORMATTING DATASET FOR FED. LEARNING
test_name = 'RandPos-Test-Evasion'
do_evil = False
perc_evil = 20
data_file = 'data/RandomPos_0709.csv'

# --- Load the dataset
data_set = genfromtxt(data_file, delimiter=',')
data_set = np.delete(data_set, 0, axis=0)  # Remove the labels at the beginning of the dataset

# --- Scale the data
train_perc = 80
split_idx = int(data_set.shape[0] * (train_perc / 100))

# Fit scaler ONLY on the training portion of the raw rows
# Include last column to make the labels binary
scaler = MinMaxScaler()
scaler.fit(data_set[:split_idx, 3:12])

# Transform the ENTIRE dataset using the scaler fit only on training data
data_set[:, 3:12] = scaler.transform(data_set[:, 3:12])

# Check that the benign/attack labels are binary
if not np.all((data_set[:, 11] == 0) | (data_set[:, 11] == 1)):
    raise ValueError("Benign/Attack labels are not binary")

# --- Divide dataset into receiver groups
fed_data_set = data_set[np.argsort(data_set[:, 1])] # Sort dataset by reciver ID
_, counts = np.unique(fed_data_set[:,1], return_counts=True) # Get the indexes of the change in datasets.
cumulative = 0
for i in range(len(counts)): # Accumulating counts so that we can use them as indexes
    cumulative += counts[i]
    counts[i] = cumulative
fed_data_set = np.split(fed_data_set, counts) # Split larger dataset into per vehicle datasets.

new_data = [] 
for receiver in fed_data_set: # Go through all vehicle datasets
    sub_data = []
    index = 0
    while index < len(receiver) - 10: # organize the new dataset as a list of chuncks of 10 messages 
        sub_data.append(receiver[index:index+10])
        index += 5
    sub_data = torch.Tensor(sub_data)
    if sub_data.shape[0] != 0:
        new_data.append(sub_data) # Create tensor from per vehicle dataset and add to list of datas.
fed_data_set = new_data
# Final output of this cell is fed_data_set, a list of the datasets of each vehicle.

#PROPER FORMATTING FOR TESTING DATASETS
#Time sequences are 10 timepoints (Messages) with 7 features per message.
#Organized by car.
                        # !! dataset is sorted by 3rd col - sender ID !!
_, counts = np.unique(data_set[:, 2], return_counts = True)
sender = 0
last_sender_count = 0
new_data = []
# Organize dataset into sets of 10 messages by sender
while sender < counts.shape[0]:
    # Loop through sender
    index = 0
    while index < counts[sender] - 10:
        # Loop through messages from sender
        new_data.append(data_set[last_sender_count+index:last_sender_count +index+10])
        index += 5
    sender += 1
    last_sender_count += counts[sender-1]
data_set = torch.tensor(new_data)
sequence_count = data_set.shape[0]
train_perc = 80

# Create seperate datasets for testing and training, using Train Percentage as metric for split
test_data_in = torch.Tensor(data_set[int(sequence_count*(train_perc/100)):,:,3:11]).float()
test_data_out = torch.Tensor(np.int_(data_set[int(sequence_count*(train_perc/100)):,:,11])).long()
train_in_list = []
train_out_list = []
val_in_list = []
val_out_list = []
tiny_test_in = []
tiny_test_out = []
# Create tiny dataset to run verification tests
for index in range(sequence_count):
    if not (int(index/10) % 300):
        tiny_test_in.append(data_set[index, :, 3:11])
        tiny_test_out.append(data_set[index, :, 11])
# Create dataset of 1/100th of the entries for quicker testing during development
for index in range(0,int((sequence_count) * (train_perc/100))):
    if not (int(index/10) % 100):
        train_in_list.append(data_set[index,:,3:11])
        train_out_list.append((data_set[index,:,11]))
for idx in range(int((sequence_count) * (train_perc/100)), sequence_count):
    if not (int(idx/10) % 10):
        val_in_list.append(data_set[idx,:,3:11])
        val_out_list.append((data_set[idx,:,11]))
testing_in = torch.Tensor(np.array(train_in_list)).float()
testing_out = torch.Tensor(np.array(train_out_list)).long()
val_in = torch.Tensor(np.array(val_in_list)).float()
val_out = torch.Tensor(np.array(val_out_list)).long()
 
    

# Standard Federated Learning

# With 2 epochs, 10 sub, and 0:3 models ending acc. of 0.1377%
# With 2 epochs, 10 sub, and 0:10 models ending acc. of

# Tested with weights, but weighing off of the loss leads to choosing the models trained by vehicles without attacks.
pl.seed_everything(1000)
results = {}
data_sets = {}
models = {}
hist_weights = []
percentages = []
cars = []
receiver_ids = []
accuracy_by_receiver = {}
state_by_receiver = {}
sub_epochs = 5 # 30
epochs = 5 # 30
vehicle_count = 5 # 200
lr = 0.01
motors = 8
units = 20
batch_size = 64
gpu = True
deep_test = False
weighing = False
random_vehicles = False
do_validation = False
adv_train = True
pgd_eps = 0.05
pgd_steps = 5
avg_loss_by_epoch = []
avg_f1_by_epoch = []
avg_recall_by_epoch = []
avg_precision_by_epoch = []

# Create starting models
main_model = OBU(8, epochs= sub_epochs, gpu = gpu, lr = lr, motors = motors, units = units)
next_model = OBU(8, epochs= sub_epochs, gpu = gpu, lr = lr, motors = motors, units = units)
path = f"FL/{test_name}-{do_evil}-{perc_evil}-{epochs}-{sub_epochs}-{vehicle_count}/"
if not os.path.exists(f"out/{path}"):
    os.makedirs(f"out/{path}")

log = OutLogger(path)

if not random_vehicles:
    # Divide dataset of recieving vehicles among OBUs
    receiver_ids = []
    for vehicle in fed_data_set[500:vehicle_count+500]: # 10
        receiver_id = int(vehicle[0,0,2].item())
        # Add new OBU for each model
        if do_evil:
            if np.random.randint(0,100) < perc_evil:
                models[receiver_id] = OBU(8, epochs = sub_epochs, gpu=gpu, lr = lr, motors = motors, units = units, evil = True, adv_train=adv_train, pgd_eps=pgd_eps, pgd_steps=pgd_steps)
            else:
                models[receiver_id] = OBU(8, epochs = sub_epochs, gpu=gpu, lr = lr, motors = motors, units = units, adv_train=adv_train, pgd_eps=pgd_eps, pgd_steps=pgd_steps)
        else:
            models[receiver_id] = OBU(8, epochs = sub_epochs, gpu=gpu, lr = lr, motors = motors, units = units, adv_train=adv_train, pgd_eps=pgd_eps, pgd_steps=pgd_steps)
        # Create Slice of dataset
        vehicle = data.DataLoader(data.TensorDataset(vehicle[:,:,3:11].float(), vehicle[:,:,11].long()), batch_size=batch_size, shuffle=False, num_workers=16, persistent_workers = True) # type: ignore
        # Add sub - dataset to dataset
        receiver_ids.append(receiver_id)
        data_sets[receiver_id]=vehicle
        models[receiver_id].dataset = vehicle

# Train individual models and combine
for epoch in range(epochs):
    count = 0
    if random_vehicles:
        # choose random assortment of vehicles to train on
        cars = []
        receiver_ids = []
        for random_idx in np.random.choice(len(fed_data_set), vehicle_count, replace = False):
            cars.append(fed_data_set[random_idx])
        # Divide dataset of recieving vehicles among OBUs
        for vehicle in cars: # 10
            receiver_id = int(vehicle[0,0,2].item())
            # Add new OBU for each model
            if receiver_id not in models:
                if do_evil:
                    if np.random.randint(0,100) < perc_evil:
                        models[receiver_id] = OBU(8, epochs = sub_epochs, gpu=gpu, lr = lr, motors = motors, units = units, evil = True, adv_train=adv_train, pgd_eps=pgd_eps, pgd_steps=pgd_steps)
                    else:
                        models[receiver_id] = OBU(8, epochs = sub_epochs, gpu=gpu, lr = lr, motors = motors, units = units, adv_train=adv_train, pgd_eps=pgd_eps, pgd_steps=pgd_steps)
                else:
                    models[receiver_id] = OBU(8, epochs = sub_epochs, gpu=gpu, lr = lr, motors = motors, units = units, adv_train=adv_train, pgd_eps=pgd_eps, pgd_steps=pgd_steps)
            # Create Slice of dataset
            vehicle = data.DataLoader(data.TensorDataset(vehicle[:,:,3:11].float(), vehicle[:,:,11].long()), batch_size=batch_size, shuffle=False, num_workers=16, persistent_workers = True)
            # Add sub - dataset to dataset
            receiver_ids.append(receiver_id)
            data_sets[receiver_id]=vehicle
            models[receiver_id].dataset = vehicle

    print(receiver_ids, file = open('rcvrs.txt', 'w'))
    # Baseline model to add everything to. !!Do I want this or should it be a completely new model?!! Got 0% on combination before, testing with new model for next model.
    next_model = OBU(8, epochs= sub_epochs, gpu = gpu, lr = lr, motors = motors, units = units)
    # Train models
    weights = []
    for receiver_id in receiver_ids:
        log.startEpochTimer()
        log.startVehicleTimer()
        # Make multithreaded?
        if do_validation and models[receiver_id].prevAccuracy != 0:
            _, _, main_accuracy, _ = main_model.test(val_in, val_out)
            print(f"Current Epoch: {epoch}, Reciever: {receiver_id}")
            print(f"Tested. main perc: {main_accuracy}, my perc: {models[receiver_id].prevAccuracy}")
            if main_accuracy > models[receiver_id].prevAccuracy:
                print("Updating Model")
                # set model to main model, and train that
                models[receiver_id].setState(main_model.getState())
        else:
            models[receiver_id].setState(main_model.getState())
        # Reset the trainer (should be unneeded now) to allow for further training
        # mod.resetTrainer()
        # Actually train
        models[receiver_id].updateSavedStates()
        model_loss = models[receiver_id].step(sub_epochs)

        weights.append(1/model_loss)
        if deep_test or do_validation:
            _, _ , accuracy, _ = models[receiver_id].test(val_in, val_out)
            accuracy_by_receiver[receiver_id] = accuracy
            models[receiver_id].prevAccuracy = accuracy
            # Test individual model
            if receiver_id not in results:
                results[receiver_id] = ([epoch, count+1, accuracy, model_loss.item()])
            else:
                results[receiver_id].append([epoch, count+1, accuracy, model_loss.item()])

        state_by_receiver[receiver_id] = (models[receiver_id].getState())
        count+=1
        log.endEpochTimer()
        log.endVehicleTimer()
    # Create combined model
    # combine models
    weights = np.abs(weights)/np.sum(weights)

    log.updateLogs([models[receiver_id] for receiver_id in receiver_ids], epoch, val_in, val_out)

    weight_sum=0
    for weight in weights:
        weight_sum+= weight
    epoch_stats = [[float(weights[idx]), accuracy_by_receiver[idx]] for idx in range(len(accuracy_by_receiver))]
    percentages.append(epoch_stats)
    hist_weights.append([weights,weight_sum])
    inv_count = 1/count
    for receiver_id in receiver_ids:
        if weighing:
            next_state = next_model.setState(next_model.getSavedState(), dict((key, state_by_receiver[receiver_id].get(key, 0)*weights[receiver_id]) for key in state_by_receiver[receiver_id])) # Done with weights
        else:
            next_state = next_model.setState(next_model.getSavedState(), state_by_receiver[receiver_id]) # No Weights
    if weighing:
        main_model.setState(next_state) #Weights
    else:
        main_model.setState(dict((key, next_state.get(key, 0)/count) for key in next_state)) # No weights
# Test combined model at end
log.finalLogs(perc_evil)
accuracy = main_model.test(test_data_in, test_data_out)
results['FINAL'] = [-1, -1, accuracy]
evil_ids = []
for receiver_id in receiver_ids: # Create list of evil/bad vehicles
    if models[receiver_id].isEvil():
        evil_ids.append(models[receiver_id].id)
print(evil_ids, file=open(f'out/{path}VehicleStatus.txt','w'))
print(results, file = open(f'out/{path}results.txt', 'w'))
print(hist_weights, file = open(f'out/{path}Weights.txt', 'w'))
print(percentages, file = open(f'out/{path}Percs.txt', 'w'))
save_path = f'out/{path}mainModelBackup.ckpt'
torch.save(
    main_model.getState(), 
    save_path
)

log.log()

print("Saved backup of main model.")
print(f"SAVE PATH: {save_path}")

# 'Model got 340703/1247740 right. Accuracy: 0.27305608540240756, Precision: 0.27978235144854047, Recall: 0.919080118694362, F1 Score: 0.42897730670851897'
# This was with scheduler, 50, 10, 100. Before accuracy with this was 96.388%, now 27.306%. 
# Running test with 50, 5, 50, w/out scheduler: 'Model got 1170620/1247740 right. Accuracy: 0.9381922515908763, Precision: 0.8443495151097161, Recall: 0.9709495548961424, F1 Score: 0.90323495386345'
# Accuracy of 93.820% after half the vehicles and half the epochs. I think we need to rething the scheduler.


elapsed_time = time.time_ns()-start_time

print("Elapsed time (ns):", elapsed_time)