import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from sklearn.metrics import f1_score
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ZooAttack

from attacks.basic.utils import create_dataset, SimpleNet

# ----
# Config
data_file = "data/RandomPos_0709.csv"
model_filename = "RandomPos-full"

# ZOO config
confidence      = 0.0   # higher = more confident (and more distorted) adversarial examples
max_iter        = 20    # iterations per binary search step
binary_steps    = 5     # binary search steps for the attack constant
nb_parallel     = 11    # coordinate updates per step — set to input size for tabular data
variable_h      = 1e-4  # step size for gradient estimation

# ----
# Step 1: Load data (use a small subset — ZOO is slow on CPU)
(_, _), (x_test, y_test) = create_dataset(data_file=data_file, divide_by=1)
x_test_small = x_test # x_test[:200]
y_test_small = y_test # y_test[:200]

# ----
# Step 2: Build classifier
model     = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

classifier = PyTorchClassifier(
    model=model,
    clip_values=(x_test.min(), x_test.max()),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(11,),
    nb_classes=2,
)

model.load_state_dict(torch.load(f"saved_models/{model_filename}.pth", weights_only=True))
model.eval()

# ----
# Step 3: Evaluate baseline (benign)
def evaluate(preds, labels):
    pred_classes = np.argmax(preds, axis=1)
    true_classes = np.argmax(labels, axis=1)
    acc = np.mean(pred_classes == true_classes)
    f1  = f1_score(true_classes, pred_classes, average="weighted")
    return acc, f1

benign_preds = classifier.predict(x_test_small)
benign_acc, benign_f1 = evaluate(benign_preds, y_test_small)
print("=== Benign ===")
print(f"Accuracy: {benign_acc:.2%}  |  F1: {benign_f1:.4f}")

# ----
# Step 4: Run ZOO black-box attack
# use_resize=False: tabular data has no spatial structure to resize
attack = ZooAttack(
    classifier=classifier,
    confidence=confidence,
    targeted=False,
    max_iter=max_iter,
    binary_search_steps=binary_steps,
    nb_parallel=nb_parallel,
    use_resize=False,
    use_importance=False,
    variable_h=variable_h,
    verbose=True,
)

print("\nRunning ZOO attack...")
x_test_adv = attack.generate(x=x_test_small)

# ----
# Step 5: Evaluate adversarial
adv_preds = classifier.predict(x_test_adv)
adv_acc, adv_f1 = evaluate(adv_preds, y_test_small)
print("\n=== ZOO Adversarial ===")
print(f"Accuracy: {adv_acc:.2%}  |  F1: {adv_f1:.4f}")

print("\n=== Delta ===")
print(f"Accuracy drop: {benign_acc - adv_acc:.2%}")
print(f"F1 drop:       {benign_f1 - adv_f1:.4f}")
print(f"Mean L2 perturbation: {np.mean(np.linalg.norm(x_test_adv - x_test_small, axis=1)):.6f}")
