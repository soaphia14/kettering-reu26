"""
Model definitions, includes CfCLearner (the training), Modena (the model),
and the OBU (simulating the vehicle).

OutLogger class for logging model outputs.
"""

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

# Creating Learner
class CfCLearner(pl.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.lr = lr
        self.lossFunc = nn.CrossEntropyLoss()
        self.loss = None
    
    def training_step(self, batch, batch_idx):
        # Get in and out from batch
        inputs, target = batch
        # Put input through model
        output, _ = self.model.forward(inputs)
        # Reorganize inputs for use with loss function
        output = output.permute(0, 2, 1)
        # Calculate Loss using Cross Entropy Loss 
        loss = self.lossFunc(output, target)
        self.log("trainLoss", loss, prog_bar=True)
        self.loss = loss
        return loss

    def validation_step(self, batch, batch_idx):
        # Get in and out from batch
        inputs, target = batch
        # Put input through model
        output, _ = self.model.forward(inputs)
        # Reorganize inputs for use with loss function
        output = output.permute(0, 2, 1)
        print(f"output: {output.shape}")
        print(f"target: {target.shape}")
        # Calculate Loss using Cross Entropy Loss 
        loss = self.lossFunc(output, target)
        self.log("valLoss", loss, prog_bar=True)
        self.loss = loss
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        # Using AdamW optomizer based on info from paper
        # optimizer = torch.optim.AdamW(self.model.parameters(), lr = 0.001)
        # return ([optimizer], [torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.6)])
        return torch.optim.AdamW(self.model.parameters(), lr = 0.01) # TESTING REMOVING THE SCHEDULER
    

# Creating Model/Module
class Modena(nn.Module): 
    # CfC with feed-forward layer to classify at end.
    def __init__(self, inputSize, unitNum = None, motorNum = 2, outputDim = 2, batchFirst = True):
        super().__init__()
        #Allow for creation of a copy of another instance
        if isinstance(inputSize, Modena):
            self.inputSize = inputSize.inputSize
            self.unitNum = inputSize.unitNum
            self.motorNum = inputSize.motorNum
            self.outputDim = inputSize.outputDim
            self.batchFirst = inputSize.batchFirst
            # Create NCP wiring for CfC
            wiring = AutoNCP(self.unitNum, self.motorNum)
            # Create CfC model with inputs and wiring
            self.cfc = CfC(self.inputSize, wiring, batch_first=self.batchFirst)
            # Create feed-forward layer
            self.fF = nn.Linear(self.motorNum, self.outputDim)
            self.fF.weight = nn.Parameter(inputSize.fF.weight)
        else:
            self.inputSize = inputSize
            self.unitNum = unitNum
            self.motorNum = motorNum
            self.outputDim = outputDim
            self.batchFirst = batchFirst
            # Create NCP wiring for CfC
            wiring = AutoNCP(unitNum, motorNum)
            # Create CfC model with inputs and wiring
            self.cfc = CfC(inputSize, wiring, batch_first=batchFirst)
            # Create feed-forward layer
            self.fF = nn.Linear(motorNum, outputDim)
        

    def forward(self, batch, hidden = None):
        batch, hidden = self.cfc(batch, hidden) # Pass inputs through CfC
        out = nn.functional.relu(self.fF(batch)) # pass through FeedForward Layer, then make 0 minimum
        return out, hidden # Return the guess and the hidden state
    

# Creating overall model Class
class OBU():
    def __init__(self, inputSize, units = 20, motors = 8, outputs = 20, epochs = 10, lr = 0.01, randInt = 0, gpu = False, dataset = None, evil = False):
        if isinstance(inputSize, OBU):
            self.lr = inputSize.lr
            self.epochs = inputSize.epochs
            self.gpu = inputSize.gpu
            self.model = Modena(inputSize.model)
            self.model.load_state_dict(inputSize.model.state_dict())
            self.learner = CfCLearner(self.model, self.lr)
            self.learner.load_state_dict(inputSize.learner.state_dict())
            self.trainer = pl.Trainer(
                logger = CSVLogger('log/Fed'), # Set ouput destination of logs, logging accuracy every 50 steps
                max_epochs = self.epochs, # Number of epochs to train for
                gradient_clip_val = 1, # This is said to stabilize training, but we should test if that is true
                accelerator = "gpu" if self.gpu else "cpu" # Using the GPU to run training or not
                )

        else:
            self.lr = lr
            self.epochs = epochs
            self.gpu = gpu
            self.model = Modena(inputSize, units, motors, outputs)
            self.learner = CfCLearner(self.model, lr) # tune units, lr
            self.trainer = pl.Trainer(
                logger = CSVLogger('log/Fed'), # Set ouput destination of logs, logging accuracy every 50 steps
                max_epochs = epochs, # Number of epochs to train for
                gradient_clip_val = 1, # This is said to stabilize training, but we should test if that is true
                accelerator = "gpu" if gpu else "cpu" # Using the GPU to run training or not
                )
        # Creating variables needed for DeFL
        self.prevAccuracy = 0
        self.evil = evil
        self.nearbyOBUs = []
        self.id = None
        self.dataset = dataset
        self.outnum = 0
        self.confidences = []
        self.samplingWeights = []
        self.priority = 0
        self.datalen = 0
        self.otherPriorities = []
        self.sampling = []
        self.curr_loss = 0
        self.prev_loss = None
        self.trust_loss = 0
        self.curr_acc = 0
        self.prev_acc = None
        self.trust_acc = 0
        self.backupWeights = self.learner.state_dict()
        self.prevWeights = dict(self.learner.state_dict())
        self.goodNeighbors = []
        self.testing = True
        self.toTest = []
        self.perEpoch = 30
        self.rounds = 0
        self.curr_f1 = 0
        self.prev_f1 = None
    

    # Overloading add function to create fed.avg. model
    def __add__(self, other):
        self.learner.load_state_dict(dict( (n, self.learner.state_dict().get(n, 0)+other.learner.state_dict().get(n, 0)) for n in set(self.learner.state_dict())|set(other.learner.state_dict()) ))
        return self
    
    # Overloading multiplication function to add weights
    def __mul__(self, i):
        self.learner.load_state_dict(dict((n, self.learner.state_dict().get(n, 0)*i) for n in self.learner.state_dict()))
        return self

    # Overloading div. function to average model
    def __truediv__(self, i):
        self.learner.load_state_dict(dict((n, self.learner.state_dict().get(n, 0)/i) for n in self.learner.state_dict()))
        return self
    
    def fit(self, dataLoader):
        # calling built in fit function
        self.trainer.fit(self.learner, dataLoader) 
        return self.learner.loss
    
    # Function to run model through a testing dataset and calculate accuracy. Can be expanded to give more metrics and more useful metrics.
    def test(self, dataIn, dataOut, mathy = False):
        # Ensure model inputs/targets are tensors before inference.
        if isinstance(dataIn, np.ndarray):
            dataIn = torch.from_numpy(dataIn).float()
        elif not isinstance(dataIn, torch.Tensor):
            dataIn = torch.tensor(dataIn, dtype=torch.float32)
        else:
            dataIn = dataIn.float()

        if isinstance(dataOut, np.ndarray):
            dataOut = torch.from_numpy(dataOut)
        elif not isinstance(dataOut, torch.Tensor):
            dataOut = torch.tensor(dataOut)
        dataOut = dataOut.long()

        # Run inference on the model device, then move to CPU for metric loops.
        device = next(self.model.parameters()).device
        dataIn = dataIn.to(device)
        dataOut = dataOut.to(device)

        with torch.no_grad():
            outs, _ = self.model(dataIn)

        outs = outs.detach().cpu()
        dataOut = dataOut.detach().cpu()
        # Get the label with the maximum confidence for determining classification
        print(outs.shape)
        _, res = torch.max(outs, 2)
        Pt = Pf = Nt = Nf = 0
        countR = 0
        numZero = 0
        tot = outs.shape[0]
        total = 0
        for i in range(0, tot):
            # Loop through sequences of 10 each
            for t in range(0, res[i].shape[0]):
                # Loop through the sub-sequences
                if dataOut.dim() == 1:
                    target = dataOut[i]
                elif dataOut.dim() == 2 and dataOut.shape[1] == 1:
                    target = dataOut[i, 0]
                else:
                    target = dataOut[i, t]

                if res[i, t] == target:
                    if res[i, t] == 0:
                        Nt += 1
                        numZero += 1
                    else:
                        Pt += 1
                    # Check if label is correct, and add to count right accordingly
                    countR += 1
                else:
                    if target == 0:
                        Pf += 1
                        numZero += 1
                    else:
                        Nf += 1
                total += 1
        # Mathy determines if we want the program to output just the numbers or a string for easy readability
        if mathy:
            # If we have at least one true positive, we can do all the other calculations.
            if Pt != 0:
                # Calculate accuracy, precision, recall and f1.
                accuracy = (Pt+Nt)/(Pt+Pf+Nf+Nt)
                precision = (Pt)/(Pt+Pf)
                recall = (Pt)/(Pt+Nf)
                f1 = (2*precision*recall)/(precision+recall)
                print(precision)
                print(recall)
                print("Model got " + str(countR) + "/" + str(total) + " right.")
                print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
                print(f"{numZero}, {numZero/total * 100}% Zeroes, {total-numZero} Non Zero entries.")
                return f1, recall, precision, accuracy
            else:
                # At least calculate accuracy when the model predicts zero.
                accuracy = (Pt+Nt)/(Pt+Pf+Nf+Nt)
                print("Model could not complete tests.")
                print(f"Confusion counts -> TP: {Pt}, FP: {Pf}, TN: {Nt}, FN: {Nf}")
                return 0, 0, 0, accuracy 
        else:
            # If we have at least one true positive, we can do all the other calculations.
            if Pt != 0:
                # Calculate accuracy, prcecision, recall and f1.
                accuracy = (Pt+Nt)/(Pt+Pf+Nf+Nt)
                precision = (Pt)/(Pt+Pf)
                recall = (Pt)/(Pt+Nf)
                f1 = (2*precision*recall)/(precision+recall)
                print(precision)
                print(recall)
                print("Model got " + str(countR) + "/" + str(total) + " right.")
                print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
                print(f"{numZero}, {numZero/total * 100}% Zeroes, {total-numZero} Non Zero entries.")
                return f"Model got {countR}/{total} right. Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}"
            else:
                print("Model could not complete tests.")
                print(f"Confusion counts -> TP: {Pt}, FP: {Pf}, TN: {Nt}, FN: {Nf}")
                return f"Model could not complete tests, found 0 of misbehaviour."
    
    # Run a test step of the model.
    def testStep(self, dataLoader):
        self.learner.validation_step(next(iter(dataLoader)), 0)
    
    # Re-define the model.
    def setModel(self, model):
        if not model == None:
            self.model = model

    # Return the current model.
    def getModel(self):
        return self.model
    
    # Get the saved vehicle weights.
    def getSavedState(self):
        return self.prevWeights

    # Update the saved vehicle weights.
    def updateSavedStates(self):
        # If this OBU is acting as malicious, we return the incorrect weights.
        if self.evil:
            self.prevWeights = dict((n, torch.full(self.learner.state_dict()[n].shape,10000000)) for n in self.learner.state_dict()).copy()
            return # Never update weights, so always passing on very large weights
        self.prevWeights = self.learner.state_dict().copy()
    
    # get the current state.
    def getState(self):
        return self.learner.state_dict()
    
    # Restore the OBU from the saved state.
    def restoreFromBackup(self):
        self.trainer.fit_loop.max_epochs = self.trainer.current_epoch - self.perEpoch
        self.trainer.fit(self.learner, self.dataset, ckpt_path=f'log/Fed/{self.id}checkpoint.ckpt')

    # Save the current state as a backup.
    def saveBackup(self):
        self.trainer.save_checkpoint(f'log/Fed/{self.id}checkpoint.ckpt')
    
    # Get malicious status of vehicle.
    def isEvil(self):
        return True if self.evil else False
    
    # Set state of learner and model. One input just sets the state, and two inputs adds them together first.
    def setState(self, one, two = None):
        if two:
            tom = dict((n, one.get(n, 0)+two.get(n, 0)) for n in set(one)|set(two))
        else:
            tom = one
        self.learner.load_state_dict(tom)
        return tom
    
    # Run one sub-epoch of the model.
    def step(self, epochs):
        self.perEpoch = epochs
        self.trainer.fit_loop.max_epochs = self.trainer.current_epoch + epochs
        self.curr_loss = self.fit(self.dataset).item()
        return self.curr_loss

    # Update the vehicles we are sampling for this epoch.
    # Not used by DeFL/DeFTA anymore.
    def updateSelected(self):
        self.sampling = []
        count = 0
        for idx in self.nearbyOBUs:
            rand = np.random.randint(0, 100)
            if rand <= int(100*self.samplingWeights[idx]): # Less than or equal, as we want 1 to be selected every time.
                self.sampling.append(int(idx))
                count += 1
        return self.sampling # Returning how many vehicles were selected for training
        
    # Reset the trainer with a new trainer.
    def resetTrainer(self):
        self.trainer = pl.Trainer(
            logger = CSVLogger('log'), # Set ouput destination of logs, logging accuracy every 50 steps
            max_epochs = self.epochs, # Number of epochs to train for
            gradient_clip_val = 1, # This is said to stabilize training, but we should test if that is true
            accelerator = "gpu" if self.gpu else "cpu" # Using the GPU to run training or not
            )
        

# Define class to monitor all metrics of model to generate graphs and compare.
class OutLogger():
    def __init__(self, path):
        #Helpers
        self.path = path
        self.epochTimes = []
        self.times = []

        #Outs
        self.avgLossVEpoch = []
        self.avgF1VEpoch = []
        self.avgRecallVEpoch = []
        self.avgPrecisionVEpoch = []
        self.avgAccuracyVEpoch = []
        self.lossVPercEvil = None
        self.F1VPercEvil = None
        self.RecallVPercEvil = None
        self.PrecisionVPercEvil = None
        self.AccuracyVPercEvil = None
        self.AvgVehicleTime = None
        self.MaxVehicleTime = None
        self.TotTime = None

    # Define start and end for timing the per vehicle time.
    def startVehicleTimer(self):
        self.startTime = time.time()
    def endVehicleTimer(self):
        self.times.append(time.time()-self.startTime)

    # Define start and end for timing the full epoch time.
    def startEpochTimer(self):
        self.startEpochTime = time.time()
    def endEpochTimer(self):
        self.epochTimes.append(time.time()-self.startEpochTime)

    # Update the logs by running a test per vehicle. 
    def updateLogs(self, vehicles, epoch, x_test, y_test):
        currLoss = 0
        currF1 = 0
        currRecall = 0
        currPrecision = 0
        currAccuracy = 0
        count = 0
        for vehicle in vehicles:
            currLoss += vehicle.curr_loss
            f1, recall, precision, accuracy = vehicle.test(x_test, y_test, True)
            currF1 += f1
            currRecall += recall
            currPrecision += precision
            currAccuracy += accuracy
            count += 1
        self.avgLossVEpoch.append([epoch, currLoss/count])
        self.avgF1VEpoch.append([epoch, currF1/count])
        self.avgRecallVEpoch.append([epoch, currRecall/count])
        self.avgPrecisionVEpoch.append([epoch, currPrecision/count])
        self.avgAccuracyVEpoch.append([epoch, currAccuracy/count])
            
    # Add the final times and values to their respective point in the log.
    def finalLogs(self, vehicles, percEvil):
        self.lossVPercEvil = [percEvil, self.avgLossVEpoch[-1][1]]
        self.F1VPercEvil = [percEvil, self.avgF1VEpoch[-1][1]]
        self.RecallVPercEvil = [percEvil, self.avgRecallVEpoch[-1][1]]
        self.PrecisionVPercEvil = [percEvil, self.avgPrecisionVEpoch[-1][1]]
        self.AccuracyVPercEvil = [percEvil, self.avgAccuracyVEpoch[-1][1]]
        self.AvgVehicleTime = np.sum(self.times)/len(self.times)
        self.MaxVehicleTime = np.max(self.times)
        self.TotTime = np.sum(self.epochTimes)/len(self.epochTimes)

    # Ouput the logged values to files.
    def log(self):
        path = f"out/{self.path}"
        if not os.path.exists(f"out/{self.path}"):
            os.makedirs(f"out/{self.path}")
        with open(f'{path}avgLossVEpoch.csv', 'w', newline='') as filename:
            writer = csv.writer(filename)
            writer.writerow(['epoch', 'avg Loss'])
            writer.writerows(self.avgLossVEpoch)
        with open(f'{path}avgF1VEpoch.csv', 'w', newline='') as filename:
            writer = csv.writer(filename)
            writer.writerow(['epoch', 'avg F1'])
            writer.writerows(self.avgF1VEpoch)
        with open(f'{path}avgRecallVEpoch.csv', 'w', newline='') as filename:
            writer = csv.writer(filename)
            writer.writerow(['epoch', 'avg Recall'])
            writer.writerows(self.avgRecallVEpoch)
        with open(f'{path}avgPrecisionVEpoch.csv', 'w', newline='') as filename:
            writer = csv.writer(filename)
            writer.writerow(['epoch', 'avg Precision'])
            writer.writerows(self.avgPrecisionVEpoch)
        with open(f'{path}avgAccuracyVEpoch.csv', 'w', newline='') as filename:
            writer = csv.writer(filename)
            writer.writerow(['epoch', 'avg Accuracy'])
            writer.writerows(self.avgAccuracyVEpoch)
        others = {'Loss V PercEvil':self.lossVPercEvil, 'F1 V PercEvil':self.F1VPercEvil, 'Recall V PercEvil':self.RecallVPercEvil, 'Precision V PercEvil':self.PrecisionVPercEvil, 
                  'Accuracy V PercEvil':self.AccuracyVPercEvil, 'Max Per-Vehicle Time':self.MaxVehicleTime, 'Avg Per-Vehicle Time':self.AvgVehicleTime, 'Total Time Per Epoch':self.TotTime}
        with open(f'{path}avgAccuracyVEpoch.csv', 'w', newline='') as filename:
            writer = csv.writer(filename)
            writer.writerow(['epoch', 'avg Accuracy'])
            writer.writerows(self.avgAccuracyVEpoch)
        with open(f'{path}ExtraData.json', 'w') as filename:
            json.dump(others, filename)


