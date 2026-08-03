# Imports
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from numpy import genfromtxt
import numpy as np
import torch
import torch.utils.data as data
import os
import time
import csv
import json
import sys
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.models import CfCLearner, Modena, OutLogger, OBU

torch.set_float32_matmul_precision("high")


batch_size = 64
total_epochs = 100

# most up to date with DeFL

def defl_pd_detection(do_evil, perc_evil):

    # --- Format data 
    test_name = 'DeConstPos-Test'
    do_evil = False
    perc_evil = 20
    data_file = 'data/ConstPos_0709.csv'

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
    train_in_list = []
    train_out_list = []
    tiny_test_in = []
    tiny_test_out = []

    # Create dataset of 1/100th of the entries for quicker testing during development
    for index in range(sequence_count):
        if not (int(index/10) % 300):
            tiny_test_in.append(data_set[index, :, 3:11])
            tiny_test_out.append(data_set[index, :, 11])
    for index in range(0,int((sequence_count) * (train_perc/100))):
        if not (int(index/10) % 100):
            train_in_list.append(data_set[index,:,3:11])
            train_out_list.append((data_set[index,:,11]))
    testing_in = torch.Tensor(np.array(train_in_list)).float()
    testing_out = torch.Tensor(np.array(train_out_list)).long()
    tiny_test_in = torch.Tensor(np.array(tiny_test_in)).float()
    tiny_test_out = torch.Tensor(np.array(tiny_test_out)).long()


    # DeFTA: Decentralized Federalized Training

    pl.seed_everything(1000)

    vehicle_count = 200 # 50 # 50
    subnet_size = 45 # 15 # 15
    steps_per_epoch = 10 # 5 # 30
    steps_per_testing_epoch = 15
    first_epoch_steps = 50
    min_connected_vehicles = 25 # 10
    backup_threshold = 0.1
    vehicles = []
    selection_weights = {}
    gpu = False
    lr = 0.01
    phi_gain = 1

    path = f"LongerPoison/{test_name}-{do_evil}-{perc_evil}-{vehicle_count}-{subnet_size}-{total_epochs}-{steps_per_epoch}-{steps_per_testing_epoch}-{min_connected_vehicles}-{backup_threshold}-{phi_gain}/"
    if not os.path.exists(f"out/{path}"):
        os.makedirs(f"out/{path}")
    else:
        return

    log = OutLogger(path)

    # --- Trust Update Algorithm
    def update_trust(vehicle):
        sample_counts = [0 for _ in range(vehicle_count)]
        for sampled_id in vehicle.sampling:
            sample_counts[sampled_id] += 1  # define matrix that contains whether the vehicle is in the sampled set, and how many times it is in the set.
        vehicle.curr_f1, _, _, vehicle.curr_acc = vehicle.test(tiny_test_in, tiny_test_out, True)
        if vehicle.testing:
            print("TESTING ROUND")
            vehicle.rounds = 0
            if vehicle.prev_f1:
                if vehicle.curr_f1 >= vehicle.prev_f1-0.05: # If vehicle helps our model
                    for i in range(vehicle_count):
                        if sample_counts[i] != 0:
                            vehicle.good_neighbors.append(i)
                    print(vehicle.good_neighbors)
                # Keep prev_accuracy and backup constant during testing phase
                vehicle.restore_from_backup() # Restore backup weights in order to keep best model during testing
                print("Restored")
            else:
                vehicle.prev_acc = vehicle.curr_acc
                vehicle.prev_f1 = vehicle.curr_f1
                print('Saved')
                vehicle.save_backup() # Save Backup for when we run training.


            if len(vehicle.to_test):
                vehicle.sampling = [vehicle.to_test.pop()]
            else:
                vehicle.testing = False
                vehicle.sampling = vehicle.good_neighbors.copy()
        else:
            vehicle.rounds += 1
            print("PREDICTING ROUND")
            if vehicle.prev_f1: # If we have previous data to go off of
                if vehicle.curr_f1 >= vehicle.prev_f1: # If model gets better or stays the same
                    vehicle.save_backup() # Save model
                    vehicle.prev_f1 = vehicle.curr_f1 # update acc
                    vehicle.prev_acc = vehicle.curr_acc
                elif vehicle.curr_f1 > vehicle.prev_f1 - 0.1: # Update model but do not update saved backup model
                    vehicle.prev_f1 = vehicle.curr_f1
                else: # If model is ruined
                    print(f"Previous f1: {vehicle.prev_f1}, Current f1: {vehicle.curr_f1}")
                    print("Loading From Backup")
                    vehicle.restore_from_backup() # Restore backup weights
                    vehicle.step(steps_per_epoch) # Training Step
                    vehicle.prev_f1 = vehicle.curr_f1 # update acc
                    vehicle.prev_acc = vehicle.curr_acc
                    # Run tests early as we have a bad-actor
                    vehicle.to_test = [i for i in vehicle.nearby_ids] # Only test those that contributed to this model
                    vehicle.sampling = [vehicle.to_test.pop()] # Initialize first sample
                    vehicle.testing = True # Start testing
                    vehicle.good_neighbors = [] # Reset known good
            else:
                vehicle.prev_acc = vehicle.curr_acc
                vehicle.prev_f1 = vehicle.curr_f1
                vehicle.save_backup() # Save Backup for when we run training.

            if vehicle.rounds > 60:
                vehicle.to_test = [i for i in vehicle.nearby_ids] # Test all nearby OBUs
                vehicle.sampling = [vehicle.to_test.pop()] # Sample the first OBU
                vehicle.testing = True # Start test
                vehicle.good_neighbors = [] # reset known good
            else:
                vehicle.sampling = vehicle.good_neighbors # Sample all known good

        if vehicle.vehicle_id in selections:
            selections[vehicle.vehicle_id].append([vehicle.sampling[:]])
        else:
            selections[vehicle.vehicle_id] = [[vehicle.sampling[:]]]


    # --- Update Priority, calculate weight for each vehicle model
    def update_priorities(vehicles):
        for i in range(vehicle_count):
            vehicles[i].out_num = len(vehicles[i].sampling) + 1 # Update useful outnumber of each vehicle in the simulation by having out_num = num sampled vehicles

        for i in range(vehicle_count): # Loop through vehicles and add priority of vehicle, done in separete loop as it requires info from other vehicles
            total_weight = vehicles[i].data_len/vehicles[i].out_num # Start out by including this vehicle's priority
            subgroup_total = vehicles[i].data_len/vehicles[i].out_num
            if len(vehicles[i].sampling) != 0:
                for neighbor_idx in vehicles[i].sampling:
                    total_weight += vehicles[neighbor_idx].data_len/vehicles[neighbor_idx].out_num
                vehicles[i].priority = (vehicles[i].data_len/vehicles[i].out_num)/(total_weight)
            else:
                vehicles[i].priority = 1
            if len(vehicles[i].sampling) != 0:
                for neighbor_idx in vehicles[i].sampling:
                    vehicles[i].other_priorities[neighbor_idx] = (vehicles[neighbor_idx].data_len/vehicles[neighbor_idx].out_num)/total_weight # fill out standard list of other priorities
                    subgroup_total += vehicles[neighbor_idx].data_len/vehicles[neighbor_idx].out_num
                print(f"vehicle {i} priority: {vehicles[i].priority}, total priority of subgroup: {subgroup_total/total_weight}")
            else:
                print(f'Vehicle {i} not sampling any vehicles this cycle.')


    # --- Main Loop
    for i in range(vehicle_count):
        vehicle_data = data.DataLoader(data.TensorDataset(fed_data_set[i][:,:,3:11].float(), fed_data_set[i][:,:,11].long()), batch_size=batch_size, shuffle=False, num_workers=10, persistent_workers = True) # Create datasets

        # If evil, create evil vehicles
        if do_evil:
            if np.random.randint(0,100) < perc_evil:
                vehicles.append(OBU(input_size=8, units=20, motors=8, outputs=2, lr=lr, rand_int=i, gpu=gpu, data_loader=vehicle_data, evil=True)) # Create evil vehicles
            else:
                vehicles.append(OBU(input_size=8, units=20, motors=8, outputs=2, lr=lr, rand_int=i, gpu=gpu, data_loader=vehicle_data)) # Create vehicles
        else:
            vehicles.append(OBU(input_size=8, units=20, motors=8, outputs=2, lr=lr, rand_int=i, gpu=gpu, data_loader=vehicle_data)) # Create vehicles
        
        vehicles[i].prev_weights = vehicles[i].get_state() # Save previous state, so that we can do it in iterations
        vehicles[i].data_len = fed_data_set[i].shape[0]
        vehicles[i].out_num = np.random.randint(min_connected_vehicles, subnet_size) # Get number of vehicles in sub network, at least #x so that vehicle has some use.
        ids_before = np.arange(0, i).tolist()
        ids_after = np.arange(i+1, vehicle_count).tolist()
        vehicles[i].nearby_ids = np.random.choice(ids_before + ids_after, vehicles[i].out_num, replace=False) # Create subnetworks and add them to the vehicle
        vehicles[i].confidences = np.full((vehicle_count), 3.) # Initialize confidence values for all vehicles - can be shifted to a dict later to allow for varying number/discovery of vehicles
        vehicles[i].sampling_weights = np.full((vehicle_count), .5) # Initialize the sampling weights to 0.5 - similarly, can be switched to a dict
        vehicles[i].other_priorities = np.zeros((vehicle_count)) # Initialize list of priorities
        vehicles[i].sampling = [] # Start by sampling no vehicles
        vehicles[i].vehicle_id = i # Save vehicle's id
        vehicles[i].to_test = [nid for nid in vehicles[i].nearby_ids]
        vehicles[i].testing = True
        vehicles[i].good_neighbors = []

    historic_loss = {}
    selections = {}

    for epoch in range(total_epochs):
        print(f"Starting Epoch {epoch} now")
        update_priorities(vehicles)
        for vehicle in vehicles:
            vehicle.update_saved_states() # Save current model as model to send to others, so that they are getting the latest after each loop
        for vehicle in vehicles:
            log.start_epoch_timer()
            log.start_vehicle_timer()
            # Model aggregation - Sum weights of all participating models weighted by their priority
            agg_weights = None
            for i in vehicle.sampling:
                if not agg_weights:
                    state = vehicles[i].get_saved_state()
                    agg_weights = dict((key, state.get(key, 0)*vehicle.other_priorities[i]) for key in state) # Multiplying weights and priority
                else:
                    state = vehicles[i].get_saved_state()
                    weighted_state = dict((key, state.get(key, 0)*vehicle.other_priorities[i]) for key in state) # Multiply weights and priority
                    agg_weights = dict( (key, weighted_state.get(key, 0)+agg_weights.get(key, 0)) for key in agg_weights) # Add this models weights to the sum
            state = vehicle.get_saved_state()
            weighted_state = dict((key, state.get(key, 0)*vehicle.priority) for key in state)
            if agg_weights:
                agg_weights = dict((key, weighted_state.get(key, 0)+agg_weights.get(key, 0)) for key in agg_weights) # Add this vehicles model to the aggregation
            else:
                agg_weights = weighted_state
            vehicle.set_state(agg_weights)
            if epoch == 0:
                loss = vehicle.step(first_epoch_steps)
            elif vehicle.testing:
                loss = vehicle.step(steps_per_testing_epoch) # Run training step to progress model
            else:
                loss = vehicle.step(steps_per_epoch) # Run training step to progress model
            if vehicle.vehicle_id in historic_loss:
                historic_loss[vehicle.vehicle_id].append(loss)
            else:
                historic_loss[vehicle.vehicle_id] = [loss]
            update_trust(vehicle) # Update theta and confidence matrix
            log.end_epoch_timer()
            log.end_vehicle_timer()
        log.update_logs(vehicles, epoch)

    sample_sizes = {}
    for vehicle_id in selections:
        for epoch in selections[vehicle_id]:
            if vehicle_id in sample_sizes:
                sample_sizes[vehicle_id].append(len(epoch))
            else:
                sample_sizes[vehicle_id] = [len(epoch)] # Printing number of vehicles sampled by each vehicle each iteration
    print(sample_sizes, file=open(f'out/{path}SampleSizes.txt', 'w'))
    print(historic_loss, file=open(f'out/{path}HistoricLoss.txt', 'w'))
    print(selections, file=open(f'out/{path}SelectedVehicles.txt', 'w'))
    evil_ids = []
    log.final_logs(vehicles, perc_evil)
    for vehicle in vehicles: # Create list of evil/bad vehicles
        if vehicle.is_evil():
            evil_ids.append(vehicle.vehicle_id)
    print(evil_ids, file=open(f'out/{path}VehicleStatus.txt','w'))
    print(selection_weights, file = open(f'out/{path}SelectionWeights.txt', 'w')) # Print out the chances of selecting each vehicle
    log.log()

    # --- Save models
    for vehicle in vehicles:
        save_path = f'out/{path}vehicle_{vehicle.vehicle_id}_model.ckpt'
        torch.save(vehicle.get_state(), save_path)
    print(f"Saved {len(vehicles)} vehicle models to out/{path}")

defl_pd_detection(False, 20)