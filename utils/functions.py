"""
Utility functions for the project.
Includes functions:
- creating a simple dataset for simple testing
- running and end to end simple attack
- loading and processing data
- loading model checkpoints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from art.estimators.classification import PyTorchClassifier

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
import time
import os

from numpy import genfromtxt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from utils.models import OBU

torch.set_float32_matmul_precision("high")
from utils.models import SimpleNet



# Create dataset based on the provided CSV file and divide it according to the specified ratio
def create_simple_dataset(data_file, divide_by, normalize : bool = False, train_percen : float = 0.8):
    data_set = np.genfromtxt(
        data_file,
        delimiter=","
    )
    data_set = np.delete(
        data_set,
        0,
        axis=0
    )

    # create the base input/output
    x = np.delete(data_set, 11, axis=1)
    if normalize:
        scaler = MinMaxScaler()
        x = scaler.fit_transform(x)
    y = data_set[:, 11:12]
    y = np.array([ [1, 0] if _y[0] == 0 else [0, 1] for _y in y])

    x = x.astype(np.float32)
    y = y.astype(np.float32)

    length = int(x.shape[0]/divide_by)
    print(f"dataset length: {length}")
    train_end = int(length*train_percen)

    x_train = x[:train_end]
    y_train = y[:train_end]

    x_test = x[train_end:length]
    y_test = y[train_end:length]

    print("% of attackers in training dataset:", np.all(y_train == [0,1], axis=1).sum()/train_end)
    print("% of attackers in testing dataset:", np.all(y_test == [0,1], axis=1).sum()/(length-train_end))
    
    return (x_train, y_train), (x_test, y_test)

# Run a simple adversarial ART attack, end to end. model_filename = None to not save the model
def run_simple_full_attack(data_file, divide_by, model_filename, Attack, **attack_kwargs): 
    
    ## Step 1: Get dataset
    (x_train, y_train), (x_test, y_test) = create_simple_dataset(data_file=data_file, divide_by=divide_by)


    ## Step 2: Create the model
    model = SimpleNet()

    # Step 2a: Define the loss function and the optimizer

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    ## Step 3: Create the ART classifier

    classifier = PyTorchClassifier(
        model=model,
        clip_values=(x_train.min(), x_train.max()),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(11,),
        nb_classes=2,
    )

    start = time.time()
    # Step 4: Train the ART classifier
    print("TRAINING")
    classifier.fit(x_train, y_train, batch_size=64, nb_epochs=5)
    print("DONE TRAINING")
    if model_filename:
        torch.save(classifier.model.state_dict(), f"saved_model/{model_filename}.pth")
    print("Model saved to model.pth")
    print("----------")
    # Step 5: Evaluate the ART classifier on benign test examples

    benign_predictions = classifier.predict(x_test)
    benign_pred_classes = np.argmax(benign_predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    benign_accuracy = np.sum(benign_pred_classes == true_classes) / len(y_test)
    benign_f1 = f1_score(true_classes, benign_pred_classes, average="weighted")
    print("Accuracy on benign tests: \t{:.2f}%".format(benign_accuracy * 100))
    print("F1 on benign tests: \t\t{:.4f}".format(benign_f1))

    # Step 6: Generate adversarial test examples
    attack = Attack(classifier, **attack_kwargs)

    x_test_adv = attack.generate(x=x_test)

    # Step 7: Evaluate the ART classifier on adversarial test examples

    adversarial_predictions = classifier.predict(x_test_adv)
    adv_pred_classes = np.argmax(adversarial_predictions, axis=1)
    adversarial_accuracy = np.sum(adv_pred_classes == true_classes) / len(y_test)
    adversarial_f1 = f1_score(true_classes, adv_pred_classes, average="weighted")
    print("Accuracy on adversarial tests: \t{:.2f}%".format(adversarial_accuracy * 100))
    print("F1 on adversarial tests: \t{:.4f}".format(adversarial_f1))

    print("Difference in accuracy: \t{:.2f}".format(benign_accuracy-adversarial_accuracy))

    benign_pred = np.argmax(benign_predictions, axis=1)
    adv_pred = np.argmax(adversarial_predictions, axis=1)

    changed = np.mean(benign_pred != adv_pred)

    print("Prediction change rate: \t{:.2f}".format(changed))
    end = time.time()
    print("Total time (excluding setup): \t{:.2f} s".format((end - start)))
    print("----------")
    
    print("Dataset: {}, Divide by: {}".format(data_file, divide_by))

# Run a simple adversarial ART attack, end to end. model_filename = None to not save the model
def test_simple_model(data_file, divide_by, model_filename, Attack, **attack_kwargs):

    ## Step 1: Get dataset
    _, (x_test, y_test) = create_simple_dataset(data_file=data_file, divide_by=divide_by)


    ## Step 2: Create the model
    model = SimpleNet()

    # Step 2a: Define the loss function and the optimizer

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    ## Step 3: Create the ART classifier

    classifier = PyTorchClassifier(
        model=model,
        clip_values=(x_test.min(), x_test.max()),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(11,),
        nb_classes=2,
    )

    start = time.time()
    # Step 4: Load model
    model.load_state_dict(torch.load(f"saved_models/{model_filename}.pth", weights_only=True))
    model.eval()
    print("----------")
    # Step 5: Evaluate the ART classifier on benign test examples

    benign_predictions = classifier.predict(x_test)
    benign_pred_classes = np.argmax(benign_predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    benign_accuracy = np.sum(benign_pred_classes == true_classes) / len(y_test)
    benign_f1 = f1_score(true_classes, benign_pred_classes, average="weighted")
    print("Accuracy on benign tests: \t{:.2f}%".format(benign_accuracy * 100))
    print("F1 on benign tests: \t\t{:.4f}".format(benign_f1))

    # Step 6: Generate adversarial test examples
    attack = Attack(classifier, **attack_kwargs)

    x_test_adv = attack.generate(x=x_test)

    # Step 7: Evaluate the ART classifier on adversarial test examples

    adversarial_predictions = classifier.predict(x_test_adv)
    adv_pred_classes = np.argmax(adversarial_predictions, axis=1)
    adversarial_accuracy = np.sum(adv_pred_classes == true_classes) / len(y_test)
    adversarial_f1 = f1_score(true_classes, adv_pred_classes, average="weighted")
    print("Accuracy on adversarial tests: \t{:.2f}%".format(adversarial_accuracy * 100))
    print("F1 on adversarial tests: \t{:.4f}".format(adversarial_f1))

    print("Difference in accuracy: \t{:.2f}".format(benign_accuracy-adversarial_accuracy))

    benign_pred = np.argmax(benign_predictions, axis=1)
    adv_pred = np.argmax(adversarial_predictions, axis=1)

    changed = np.mean(benign_pred != adv_pred)

    print("Prediction change rate: \t{:.2f}".format(changed))
    end = time.time()
    print("Total time (excluding setup): \t{:.2f} s".format((end - start)))
    print("----------")
    
    print("Dataset: {}, Divide by: {}".format(data_file, divide_by))

# Get windowed data created from a CSV file
def get_windowed_data(data_file, normalize : bool, train_perc : int = 80, divide_by : int = 1):
    ## Get data
    raw_msg_data = genfromtxt(data_file, delimiter=',')

    ## Divide dataset into reciever groups
    # scaler = MinMaxScaler()
    raw_msg_data = np.delete(raw_msg_data, 0, axis=0) # Remove the labels at the beginning of the dataset
    
    # Normalize the data
    if normalize:
        # Split before scaler
        split_index = int(raw_msg_data.shape[0] * (train_perc / 100))

        # Fit scaler ONLY on the training portion of the raw rows
        scaler = MinMaxScaler()
        scaler.fit(raw_msg_data[:split_index, 3:10])

        # Scale only training data
        raw_msg_data[:, 3:10] = scaler.transform(raw_msg_data[:, 3:10])

    # raw_msg_data = scaler.fit_transform(raw_msg_data)
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
        veh_windows = torch.Tensor(np.array(veh_windows))

        if len(veh_windows) != 0: # Don't add empty data
            windowed_fed_data.append(veh_windows) # Create tensor from per vehicle dataset and add to list of datas.

    ## Proper formatting for testing datasets (centralized)
    # Time sequences are 10 timepoints (Messages) with 7 features per message.
    # Organized by car, added to one large list.

    _, freqs = np.unique(raw_msg_data[:, 2], return_counts = True)

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
    length = int(centr_data.shape[0]/divide_by)
    train_end = int(length*(train_perc/100))

    x_train = torch.Tensor(centr_data[:train_end,:,3:10]).float()
    y_train = (torch.Tensor(centr_data[:train_end, :, 11])).long()

    x_test = torch.Tensor(centr_data[train_end:length,:,3:10]).float() 
    y_test = (torch.Tensor(centr_data[train_end:length, :, 11])).long()

    

    return (x_train, y_train), (x_test, y_test), windowed_fed_data

# Load a FL-trained model from a checkpoint file
def load_model_checkpoint(checkpoint_file : str, gpu : bool = False, lr : float = 0.001, motors : int = 8, units : int = 20, subEpochs : int = 10):    
    model = OBU(7, gpu = gpu, lr = lr, motors = motors, units = units, epochs = subEpochs)

    # Load model
    if not os.path.exists(checkpoint_file):
        print(os.listdir("/"))
        raise ValueError(f"Checkpoint file {checkpoint_file} does not exist.")
    else:
        print("Checkpoint path exists!")

    checkpoint = torch.load(checkpoint_file)
    model.learner.load_state_dict(checkpoint)
    return model
