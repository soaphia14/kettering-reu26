import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from art.estimators.classification import PyTorchClassifier

from sklearn.preprocessing import MinMaxScaler

import time

# Create dataset based on the provided CSV file and divide it according to the specified ratio
def create_dataset(data_file, divide_by):
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
    scaler = MinMaxScaler()
    x = np.delete(data_set, 11, axis=1)
    x = scaler.fit_transform(x)
    y = data_set[:, 11:12]
    y = np.array([ [1, 0] if _y[0] == 0 else [0, 1] for _y in y])

    x = x.astype(np.float32)
    y = y.astype(np.float32)

    length = int(x.shape[0]/divide_by)
    print(f"dataset length: {length}")
    train_percen = 0.8
    train_end = int(length*train_percen)

    x_train = x[:train_end]
    y_train = y[:train_end]

    x_test = x[train_end:length]
    y_test = y[train_end:length]

    print("% of attackers in training dataset:", np.all(y_train == [0,1], axis=1).sum()/train_end)
    print("% of attackers in testing dataset:", np.all(y_test == [0,1], axis=1).sum()/(length-train_end))
    
    return (x_train, y_train), (x_test, y_test)

# Define the neural network model
class SimpleNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(11, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)
    
# Run a simple adversarial ART attack, end to end
def run_simple_full_attack(data_file, divide_by, Attack, **attack_kwargs): 
    
    ## Step 1: Get dataset
    (x_train, y_train), (x_test, y_test) = create_dataset(data_file=data_file, divide_by=divide_by)


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
    print("----------")
    # Step 5: Evaluate the ART classifier on benign test examples

    benign_predictions = classifier.predict(x_test)
    benign_accuracy = np.sum(np.argmax(benign_predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("Accuracy on benign tests: \t{:.2f}%".format(benign_accuracy * 100))

    # Step 6: Generate adversarial test examples
    attack = Attack(classifier, **attack_kwargs)

    x_test_adv = attack.generate(x=x_test)

    # Step 7: Evaluate the ART classifier on adversarial test examples

    adversarial_predictions = classifier.predict(x_test_adv)
    adversarial_accuracy = np.sum(np.argmax(adversarial_predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("Accuracy on adversarial tests: \t{:.2f}%".format(adversarial_accuracy * 100))

    print("Difference in accuracy: \t{:.2f}".format(benign_accuracy-adversarial_accuracy))

    benign_pred = np.argmax(benign_predictions, axis=1)
    adv_pred = np.argmax(adversarial_predictions, axis=1)

    changed = np.mean(benign_pred != adv_pred)

    print("Prediction change rate: \t{:.2f}".format(changed))
    end = time.time()
    print("Total time (excluding setup): \t{:.2f} s".format((end - start)))
    print("----------")
    
    print("Dataset: {}, Divide by: {}".format(data_file, divide_by))