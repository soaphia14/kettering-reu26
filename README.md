sudo apt update && sudo apt install -y git


sudo apt install python3.12-venv # install venv


python3 -m venv .venv # create venv


source .venv/bin/activate # activate venv


sudo apt install python3-pip # install pip


pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

pip install ncps numpy lightning matplotlib

pip install -U scikit-learn

pip install adversarial-robustness-toolbox


If need to commit/push:

git config --global user.name "soaphia14"

git config --global user.email "sliu@kettering.edu"