# Special lil

from art.attacks.evasion import CarliniL2Method

from utils.functions import run_simple_full_attack

## Step 0: Define constants
data_file = "data/RandomPos_0709.csv"
divide_by = 100 # how much dividing the dataset by (1 = use the entire dataset)


run_simple_full_attack(data_file, divide_by, CarliniL2Method, max_iter=15)