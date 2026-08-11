from dataclasses import dataclass
from dataclasses import asdict
# Imports
import sys, os, torch, json
from pathlib import Path
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, SaliencyMapMethod, CarliniL2Method
from art.defences.trainer import AdversarialTrainer, AdversarialTrainerTRADESPyTorch
import art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_pytorch as _pgd_pt
_pgd_pt.compute_success = lambda *a, **kw: 0.0
from typing import Optional

sys.path.append(str(Path.cwd().parents[0]))

from utils.functions import get_windowed_data
from utils.notebook import get_model_classifier, clean_data_test, adv_test, FilenameLoader, get_filename_from_path, freeze_attack_cols, freeze_benign_and_cols
from utils.paths import find_repo_root

# If you want to define end_index - but currently I'll run it on the full thing, so right now it's not used
# @dataclass
# class AdvParams:
#     end_index : Optional[int] = None

@dataclass
class FgsmParams:
    eps : float

@dataclass
class PgdParams:
    eps : float

@dataclass
class PgdTrainerParams:
    pgd_eps : float
    pgd_iter : int
    ratio : float
    epochs : int

class TestLoader:
    def __init__ (self, ckpt_file, data_file, save_dir):
        self.checkpoint_file = ckpt_file
        self.data_file = data_file
        self.save_dir = save_dir
        self.training_params = None
        self.root = find_repo_root()
        print(f"> Initialized with {self.checkpoint_file} {self.data_file}")
        os.makedirs(self.root / save_dir, exist_ok=True)
    
    def load_data(self):
        print("> Starting loading data")
        (self.x_train, self.y_train), (self.x_test, self.y_test), fed_dataset, self.scaler = get_windowed_data(self.root / "data" / self.data_file, 
                                                                      normalize=True, 
                                                                      train_perc=80)
        self.x_train_np = self.x_train.numpy()
        self.y_train_np = self.y_train.numpy()
        print(f"Loaded data for {self.data_file}")

    def load_model(self):
        print("> Starting loading model")
        self.model, self.classifier = get_model_classifier(self.root / "saved_models" / self.checkpoint_file)
        print(f"Loaded model for {self.checkpoint_file}")

    def test_clean(self, filename : str, save_results : bool = True):
        """
        - filename: str, without the extension (so "clean" instead of "clean.json")
        - save_results: bool (default = True), whether to save results or not
        """
        print("[!!] Running on *current* loaded model! To run on a different model, re-initialize TestLoader")
        clean_out = clean_data_test(
            self.model, self.classifier, self.x_test, self.y_test, 
            checkpoint_file=self.checkpoint_file, data_file=self.data_file,
            save_path=self.root / self.save_dir,
            filename=f"{filename}.json",
            save_results=save_results
        )
        return clean_out 

    def test_adv_fgsm(self, params : FgsmParams):
        """
        - save_dir: str, directory to save to, *goes from root* (don't use ../)
        - params: FgsmParams, params to use for FGSM test
        """
        print("[!!] Running on *current* loaded model! To run on a different model, re-initialize TestLoader")
        
        adv_out = adv_test(
            self.classifier, self.x_test, self.y_test, 
            checkpoint_file=self.checkpoint_file, data_file=self.data_file,
            end_index=len(self.y_test.numpy()),
            path=self.root / self.save_dir / "fgsm",
            filename=f"fgsm_adv_eps_{params.eps}.json",
            Attack=FastGradientMethod,
            eps=params.eps,
        )
        return adv_out

    def test_adv_pgd(self, params : PgdParams):
            """
            - save_dir: str, directory to save to, *goes from root* (don't use ../)
            - params: PgdParams, params to use for PGD test (iter = 10)
            """
            print("[!!] Running on *current* loaded model! To run on a different model, re-initialize TestLoader")
            
            adv_out = adv_test(
                self.classifier, self.x_test, self.y_test, 
                checkpoint_file=self.checkpoint_file, data_file=self.data_file,
                end_index=len(self.y_test.numpy()),
                path=self.root / self.save_dir / "pgd",
                filename=f"pgd_adv_eps_{params.eps}.json",
                Attack=ProjectedGradientDescent,
                eps=params.eps,
                max_iter = 10
            )
            return adv_out

    def train_adv_pgd (self, params : PgdTrainerParams):
        # Run adv training
        # freeze_benign_and_cols keeps rcvTime (col 0) untouched, same threat model as
        # adv_test's freeze_cols default, and additionally only lets the attack perturb
        # timesteps labeled attacker=1 - a real attacker can't manipulate benign
        # messages from other vehicles, so FGSM/PGD shouldn't be allowed to "cheat" by
        # moving benign traffic during training either.
        print("> Running Adv Training")
        self.training_params = params
        attack = freeze_benign_and_cols(
            ProjectedGradientDescent(
                self.classifier, eps=params.pgd_eps, max_iter=params.pgd_iter,
                eps_step=2.5 * params.pgd_eps / params.pgd_iter
            ),
            freeze_cols=(0,3)
        )
        trainer = AdversarialTrainer(self.classifier, attacks=attack, ratio=params.ratio)
        trainer.fit(self.x_train_np, self.y_train_np, nb_epochs=params.epochs)

    def save_model_config(self):
        ckpt_out = self.root / self.save_dir / f"advtrained.ckpt"
        torch.save(self.model.learner.state_dict(), ckpt_out)
        print(f"Saved adversarially-trained checkpoint to {ckpt_out}")

        config = {
            "trainingParams": asdict(self.training_params),
            "checkpointFile": self.checkpoint_file,
            "dataFile": self.data_file,
            "attack": "ProjectedGradientDescent",
        }

        with open(self.root / self.save_dir / f"config.json", "w") as f:
            json.dump(config, f, indent=4)
            print("Config saved:", f"{self.save_dir}/config.json")

def get_loader(ckpt_file : str, data_file : str, save_dir : str):
    loader = TestLoader(ckpt_file=ckpt_file, 
            data_file=data_file, 
            save_dir=save_dir)
    loader.load_model()
    loader.load_data()
    return loader