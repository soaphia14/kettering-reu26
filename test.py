import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parents[0]))

from utils.functions import get_windowed_data

data_file = "data/RandomPos_0709.csv"
norm_trained = False
train_perc = 80

# (No edit) Load data for og model (very time consuming part)
(x_train, y_train), (x_test, y_test), fed_dataset = get_windowed_data(data_file, 
                                                                      normalize=norm_trained, 
                                                                      train_perc=train_perc)


print(x_test)
print(y_test)

print("input shape:", x_test.shape)
print("output shape:", y_test.shape)