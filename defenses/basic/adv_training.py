import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from art.attacks.evasion import FastGradientMethod

from attacks.basic.utils import run_simple_full_attack, test_simple_model

## Step 0: Define constants
data_file = "data/RandomPos_0709.csv"
divide_by = 1 # how much dividing the dataset by (1 = use the entire dataset)
model_fileame = "RandomPos-full"

# Step 1: save model
# run_simple_full_attack(data_file, divide_by, model_fileame, FastGradientMethod, eps=0.1)

# Step 2: test model
test_simple_model(data_file, divide_by, model_fileame, FastGradientMethod, eps=0.1)