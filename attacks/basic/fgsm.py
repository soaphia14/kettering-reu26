from art.attacks.evasion import FastGradientMethod

from attacks.basic.utils import run_simple_full_attack

## Step 0: Define constants
data_file = "data/RandomPos_0709.csv"
divide_by = 1 # how much dividing the dataset by (1 = use the entire dataset)

# Step 1: run attack
run_simple_full_attack(data_file, divide_by, FastGradientMethod, eps=0.1)