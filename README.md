# Kettering REU 26
Sophia Liu

## Installation

Based on a fresh linux computer.

```
# Install basic git and python modules
sudo apt update && sudo apt install -y git
sudo apt install python3.12-venv # install venv
python3 -m venv .venv # create venv
source .venv/bin/activate # activate venv
sudo apt install python3-pip # install pip

# Install dependencies
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install ncps numpy lightning matplotlib adversarial-robustness-toolbox
pip install -U scikit-learn

# If need to commit/push:
git config --global user.name "<github id>"
git config --global user.email "<github email>"
```


## Abbreviations / Terms

FL = federated learning

cen FL = centralized FL

decen FL = decentralized FL

clean = the original data, aka. data not adversarially perturbed

adv trained = adversarially trained

general training = adv trained, perturbing both benign and attacker messages

targeted training = adv trained, perturbing only attacker messages


## Folders

`/attacks`

Contains code to run attacks on different IDS models

- `/fed` - For running attacks on centralized FL main model. It uses the `TestLoader` class to run FGSM/PGD experiments. JSMA and CW attacks were also ran when the code was more decoupled - though now the `TestLoader` class can be easily configured to include.
- `/defed` - For running attacks on decentralized FL models. It's less developed, but essentially does what `/fed` but on each individual vehicle model.
- `/const` - testing the functionality preserving code

`/data`

Where the `.csv` data files go. Find the files in this [google drive folder](https://drive.google.com/drive/folders/1P-I0NZ9L2_bRBrjmJm21qkmPVjL4Vh_H).

Currently using:
- ConstPos_0709.csv
- RandomPos_0709.csv
- RandomSpeed_0709.csv

`/defenses/fed`

Only one folder in here - `/fed` for testing adversarial training on the FL main model.
The `run_clean.ipynb` file contains code to run the clean data on a clean,
general adv trained, and targeted adv trained model. The `run_train.ipynb` file contains
code to retrain a model using the ART AdversarialTrainer on a specific PGD epsilon, then 
run a FGSM sweep and save the results.

`/plot`

Contains one file - `plot.ipynb` - to plot various data saved in the different folders.
Find more information within the file. 

`/saved_models`

Where the models are saved, with the extension `.ckpt`. The `-final` are cleanly
trained models.

- `/adv_trained` - Adversarially trained models using the Cen FL method
- `/decen` - The individual vehicle models as the output of the decen training.

`/training`

The files used to train the cen main model and decen vehicle models. Parameters at the top
of the files are used to define the type of model trained. the `DecenFL.py` is less 
developed - as decen training wasn't as much of a focus. 

`/utils`

Files with common functions used throughout the repository.

- `functions.py` - General functions used throughout
- `loader.py` - Contains `TestLoader` to make loading models and running adversarial attacks easier
- `models.py` - The model classes used for training
- `notebook.py` - Functions commonly used in the ipynb notebooks
- `paths.py` - Contains a helper function to find the path root (function only used towards end of development, so this is not standard throughout the repo)