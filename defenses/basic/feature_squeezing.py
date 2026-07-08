import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod
from art.defences.preprocessor import FeatureSqueezing

from utils.functions import create_simple_dataset
from utils.models import SimpleNet

# ----
# Config
data_file = "data/RandomPos_0709.csv"
model_filename = "RandomPos-full"
eps = 0.04
bit_depth = 6 # squeeze to this many bits (lower = more squeezing)

# ----
# Step 1: Load data
(x_train, y_train), (x_test, y_test) = create_simple_dataset(data_file=data_file, divide_by=1)

# ----
# Step 2: Build classifier WITHOUT feature squeezing (baseline)
model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

classifier = PyTorchClassifier(
    model=model,
    clip_values=(x_train.min(), x_train.max()),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(11,),
    nb_classes=2,
)

model.load_state_dict(torch.load(f"saved_models/{model_filename}.pth", weights_only=True))
model.eval()

# ----
# Step 3: Feature squeezing preprocessor (applied manually to avoid float64 dtype mismatch)
squeezer = FeatureSqueezing(bit_depth=bit_depth, clip_values=(x_train.min(), x_train.max()))

# ----
# Step 4: Generate adversarial examples (against the undefended classifier)
attack = FastGradientMethod(estimator=classifier, eps=eps)
x_test_adv = attack.generate(x=x_test)

# ----
# Step 5: Evaluate
def accuracy(preds, labels):
    return np.sum(np.argmax(preds, axis=1) == np.argmax(labels, axis=1)) / len(labels)

print("=== Undefended classifier ===")
print("Benign accuracy:      {:.2f}%".format(accuracy(classifier.predict(x_test), y_test) * 100))
print("Adversarial accuracy: {:.2f}%".format(accuracy(classifier.predict(x_test_adv), y_test) * 100))

x_test_squeezed = squeezer(x_test)[0].astype(np.float32)
x_test_adv_squeezed = squeezer(x_test_adv)[0].astype(np.float32)

print("\n=== Feature squeezing (bit depth={}) ===".format(bit_depth))
print("Benign accuracy:      {:.2f}%".format(accuracy(classifier.predict(x_test_squeezed), y_test) * 100))
print("Adversarial accuracy: {:.2f}%".format(accuracy(classifier.predict(x_test_adv_squeezed), y_test) * 100))
