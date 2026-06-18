
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from art.attacks.evasion import DeepFool
from art.estimators.classification import PyTorchClassifier

from sklearn.preprocessing import MinMaxScaler


# Step 0: Define the neural network model, return logits instead of activation in forward method

# minimum amount of change (guaranteed deviation)
# see if attacker data is even in it


class Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(11, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)

# Step 1: Load the MNIST dataset
# load dataset
dataFile = "../data/ConstPos_0709.csv"
dataSet = np.genfromtxt(
    dataFile,
    delimiter=","
)
dataSet = np.delete(
    dataSet,
    0,
    axis=0
)


# Step 1a: create dataset

# create the base input/output
scaler = MinMaxScaler()
x = np.delete(dataSet, 11, axis=1)
x = scaler.fit_transform(x)
y = dataSet[:, 11:12]
y = np.array([ [1, 0] if _y[0] == 0 else [0, 1] for _y in y])

x = x.astype(np.float32)
y = y.astype(np.float32)

length = int(x.shape[0]/100)
print(f"dataset length: {length}")
train_percen = 0.8
train_end = int(length*train_percen)

x_train = x[:train_end]
y_train = y[:train_end]

x_test = x[train_end:length]
y_test = y[train_end:length]

print("% of attackers in training dataset:", np.all(y_train == [0,1], axis=1).sum()/train_end)
print("% of attackers in testing dataset:", np.all(y_test == [0,1], axis=1).sum()/(length-train_end))


# Step 2: Create the model

model = Net()

# Step 2a: Define the loss function and the optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Step 3: Create the ART classifier

classifier = PyTorchClassifier(
    model=model,
    clip_values=(x_train.min(), x_train.max()),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(11,),
    nb_classes=2,
)

# Step 4: Train the ART classifier
print("TRAINING")
classifier.fit(x_train, y_train, batch_size=64, nb_epochs=5)
print("DONE TRAINING")

# Step 5: Evaluate the ART classifier on benign test examples

benign_predictions = classifier.predict(x_test)
benign_accuracy = np.sum(np.argmax(benign_predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(benign_accuracy * 100))

# Step 6: Generate adversarial test examples
attack = DeepFool(classifier, max_iter=100, nb_grads=2)
x_test_adv = attack.generate(x=x_test)

# Step 7: Evaluate the ART classifier on adversarial test examples

adversarial_predictions = classifier.predict(x_test_adv)
adversarial_accuracy = np.sum(np.argmax(adversarial_predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on adversarial test examples: {}%".format(adversarial_accuracy * 100))

print(f"Difference in accuracy: {benign_accuracy-adversarial_accuracy}")

benign_pred = np.argmax(benign_predictions, axis=1)
adv_pred = np.argmax(adversarial_predictions, axis=1)

changed = np.mean(benign_pred != adv_pred)

print("Prediction change rate:", changed)