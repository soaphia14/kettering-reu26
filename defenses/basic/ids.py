import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from art.estimators.classification import PyTorchClassifier

from sklearn.preprocessing import MinMaxScaler

import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from art.attacks.evasion import FastGradientMethod

from attacks.basic.utils import run_simple_full_attack, test_simple_model, create_dataset, SimpleNet

## Step 0: Define constants
data_file = "data/RandomPos_0709.csv"
bn_model_filename = "RandomPos-full"
adv_model_filename = "RandomPos-adv-train"
eps=0.04

(x_train, y_train), (x_test, y_test) = create_dataset(data_file=data_file, divide_by=100)


# Step 2: adversarial train model
# Run a simple adversarial ART attack, end to end. model_filename = None to not save the model
def adv_train(data_file, divide_by, bn_model_filename, adv_model_filename, Attack, **attack_kwargs):

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
    model.load_state_dict(torch.load(f"saved_models/{bn_model_filename}.pth", weights_only=True))
    model.eval()

    # Step 6: Generate adversarial test examples
    attack = Attack(classifier, **attack_kwargs)

    x_train_adv = attack.generate(x=x_train)
    return x_train_adv


x_train_adv = adv_train(data_file=data_file,
          divide_by=2,
          bn_model_filename=bn_model_filename,
          adv_model_filename=adv_model_filename,
          Attack=FastGradientMethod,
          eps=eps)

print("Negative values in x_train:", np.sum(x_train_adv == 1))

