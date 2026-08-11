# (No edit) Imports
import os
import json
import time
import inspect

import sys
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from art.estimators.classification import PyTorchClassifier
from sklearn.preprocessing import MinMaxScaler

sys.path.append(str(Path.cwd().parents[2]))

import matplotlib.pyplot as plt

import os

import torch
from utils.functions import get_windowed_data, load_model_checkpoint

from sklearn.metrics import precision_score, recall_score, f1_score


class FilenameLoader():
    """
    Functions to load the names of the checkpoint, data files, save folder name

    Return: checkpoint file, data file, save folder name
    """
    
    def const_pos():
        return "ConstantPos-final.ckpt", "ConstPos_0709.csv", "constpos", "ConstPos"
    def rand_pos():
        return "RandomPos-final.ckpt", "RandomPos_0709.csv", "randpos", "RandPos"
    def rand_speed():
        return "RandomSpeed-final.ckpt", "RandomSpeed_0709.csv", "randspeed", "RandSpeed"

class SequenceCrossEntropy(nn.Module):
    """
    Loss class for ART wrapper.
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, a, b):
        if a.dim() == 3:
            # sequence output: (batch, seq_len, num_classes)
            if b.dim() == 3:
                b = b.argmax(dim=-1)
            return self.loss(a.permute(0, 2, 1), b.long())
        else:
            # collapsed output: (batch, num_classes)
            if b.dim() == 2:
                b = b.argmax(dim=-1)
            return self.loss(a, b.long())


class NormalizedCfCWrapper(nn.Module):
    """
    Wrapper for trained normalized model.
    """
    def __init__(self, modena_model, collapsed : bool = False):
        super().__init__()
        self.modena_model = modena_model
        self.collapsed = collapsed

    def forward(self, x_normalized):        
        x_raw = x_normalized
        logits, _ = self.modena_model(x_raw)

        if self.collapsed:
            return logits.mean(dim=1)
        else:
            return logits

# Get classifier to adv testing
def get_model_classifier (checkpoint_file : str, collapsed : bool = False):
    """
    Get the ART wrapper model classifier based on the checkpoint file.

    Input checkpoint_file path : str
    Return: model, classifier
    """
    model = load_model_checkpoint(checkpoint_file, gpu=False)
    wrapped_model = NormalizedCfCWrapper(modena_model=model.model, collapsed = collapsed)
    criterion = SequenceCrossEntropy()
    optimizer = optim.Adam(
        wrapped_model.parameters(),
        lr=0.001
    )

    classifier = PyTorchClassifier(
        model=wrapped_model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(10, 8),
        nb_classes=2,
        clip_values=(0.0, 1.0), # for normalized
        device_type="cpu"
    )
    return model, classifier

# Calculate metrics based on the predictions and what the y_test should be
def calculate_metrics(predictions, in_y_test):
    pred_flat = np.argmax(predictions, axis=-1).flatten()
    true_flat = in_y_test.flatten()

    TP = int(np.sum((true_flat == 1) & (pred_flat == 1)))
    TN = int(np.sum((true_flat == 0) & (pred_flat == 0)))
    FP = int(np.sum((true_flat == 0) & (pred_flat == 1)))
    FN = int(np.sum((true_flat == 1) & (pred_flat == 0)))

    total = TP + TN + FP + FN
    accuracy  = (TP + TN) / total if total > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fnr       = FN / (TP + FN) if (TP + FN) > 0 else 0.0
    fpr       = FP / (FP + TN) if (FP + TN) > 0 else 0.0

    return {"accuracy": accuracy, 
            "precision": precision, 
            "recall": recall, 
            "f1": f1, 
            "falseNegativeRate": fnr, 
            "falsePositiveRate": fpr, 
            "TP": TP, 
            "TN": TN, 
            "FP": FP, 
            "FN": FN}


def clean_data_test(model, classifier,
                    x_test, y_test,
                    checkpoint_file : str, data_file : str,
                    save_path : str, filename : str,
                    save_results : bool,
                    collapsed : bool = False, attacker_code : int = 1):
    """
    Run no-wrapper (original) and wrapper tests on clean data.

    collapsed must match whatever `classifier` was built with (see
    NormalizedCfCWrapper's collapsed flag / get_model_classifier). model.test()
    below always uses the raw, non-wrapped model, which is per-message regardless
    of `collapsed` - so it always needs the original y_test. The wrapper metrics,
    on the other hand, need one label per window when collapsed=True, matching
    classifier's now-collapsed (batch, nb_classes) output - a window counts as an
    attack only if every message in it is attacker_code (same convention as
    windowed_eval.py / MBD_systems/tensor_eval.py).
    """
    # No wrapper - always per-message, regardless of `collapsed`
    no_wrapper_out = model.test(x_test, y_test, mathy=True)

    # Wrapper
    y_test_np = y_test.numpy() if isinstance(y_test, torch.Tensor) else y_test
    if collapsed:
        benign_y_test = (y_test_np == attacker_code).all(axis=1).astype(np.int64)
    else:
        benign_y_test = y_test_np
    benign_predictions = classifier.predict(x_test, batch_size=64)

    wrapper_out = calculate_metrics(benign_predictions, benign_y_test)

    metrics = {
        "noWrapper": no_wrapper_out,
        "wrapper": wrapper_out,
        "files":{
            "checkpointFile": checkpoint_file,
            "dataFile": data_file
        }
    }
    
    # Save it
    if save_results:
        os.makedirs(save_path, exist_ok=True)
        output_path = f"{save_path}/{filename}"
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Saved to {output_path}")
    else:
        print("save=False, Metrics not saved")
        print("Metrics", metrics)
    return metrics


def adv_test(classifier,
             x_test, y_test,
             checkpoint_file : str, data_file : str,
             end_index: int, path: str, filename: str, Attack,
             freeze_cols=(0,), collapsed : bool = False,
             **kwargs):
    print(f"=== Attack: {Attack.__name__}, kwargs: {kwargs} ===")
    start = time.time_ns()

    x_in = x_test.numpy()[:end_index]

    # collapsed must match whatever `classifier` was built with (see
    # NormalizedCfCWrapper's collapsed flag). Its output is (batch, nb_classes) -
    # one label per window, not per message - so both the attack's target labels
    # and the ground truth used for scoring need to be collapsed the same way, or
    # they end up 10x too long compared to the predictions.
    if collapsed:
        y_in = (y_test.numpy()[:end_index] == 1).all(axis=1).astype(np.int64)
    else:
        y_in = y_test.numpy()[:end_index]
    true_flat = y_in.flatten()

    y_attack = 1 - y_in
    attack_kwargs = dict(kwargs)
    if "targeted" not in attack_kwargs and "targeted" in inspect.signature(Attack.__init__).parameters:
        attack_kwargs["targeted"] = True

    generate_kwargs = {}
    if freeze_cols:
        # Broadcastable mask (10, 8): honored by FGSM/PGD, but SaliencyMapMethod
        # and CarliniL2Method ignore `mask` entirely - the restore below after
        # generate() is what actually guarantees these columns (e.g. rcvTime)
        # stay untouched regardless of which Attack is used.
        mask = np.ones(x_in.shape[1:], dtype=np.float32)
        mask[:, freeze_cols] = 0.0
        generate_kwargs["mask"] = mask

    attack = Attack(classifier, **attack_kwargs)
    x_test_adv = attack.generate(x=x_in, y=y_attack, **generate_kwargs)

    if freeze_cols:
        x_test_adv[:, :, freeze_cols] = x_in[:, :, freeze_cols]

    adversarial_predictions = classifier.predict(x_test_adv)
    pred_flat = np.argmax(adversarial_predictions, axis=-1).flatten()

    elapsed_ns = time.time_ns() - start

    # Confusion matrix (binary: 1=attack, 0=benign)
    TP = int(np.sum((true_flat == 1) & (pred_flat == 1)))
    TN = int(np.sum((true_flat == 0) & (pred_flat == 0)))
    FP = int(np.sum((true_flat == 0) & (pred_flat == 1)))
    FN = int(np.sum((true_flat == 1) & (pred_flat == 0)))

    total = TP + TN + FP + FN
    accuracy  = (TP + TN) / total if total > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    asr       = FN / (TP + FN) if (TP + FN) > 0 else 0.0  # false negative rate
    fpr       = FP / (FP + TN) if (FP + TN) > 0 else 0.0

    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Precision:          {precision:.4f}")
    print(f"Recall:             {recall:.4f}")
    print(f"F1:                 {f1:.4f}")
    print(f"ASR (FNR):          {asr:.4f}")
    print(f"False Positive Rate:{fpr:.4f}")
    print(f"TP={TP}, TN={TN}, FP={FP}, FN={FN}")
    print(f"Time elapsed:       {elapsed_ns / 1e9:.2f}s")

    os.makedirs(path, exist_ok=True)
    metrics = {
        "endIndex": end_index,
        "attack": Attack.__name__,
        "timeElapsedSec": elapsed_ns / 1e9,
        "kwargs": kwargs,
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "falseNegativeRate": asr,
            "falsePositiveRate": fpr,
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN
        },
        "files":{
            "checkpointFile": checkpoint_file,
            "dataFile": data_file
        }
    }
    output_path = f"{path}/{filename}"
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {output_path}")

    return metrics


def freeze_attack_cols(attack, freeze_cols=(0,)):
    """Patches an ART evasion attack in place so it can never move the frozen
    feature columns (e.g. rcvTime, MssgCount) - a real attacker can't manipulate
    those, so adversarial training shouldn't let FGSM/PGD touch them either.
    Without this, AdversarialTrainer can satisfy its training objective by
    learning to detect tampering on columns that were never attackable in the
    first place, without ever having to move its decision boundary on the real
    motion features - which costs nothing in clean accuracy but also buys no
    real robustness (see adv_test's freeze_cols default, which applies the same
    restriction at eval time).

    Returns the same attack instance (still a real EvasionAttack, so it still
    passes AdversarialTrainer's isinstance check) with `generate` overridden.

    Usage: attack = freeze_attack_cols(FastGradientMethod(classifier, eps=eps))

    ** Only freezes the receiveTime column
    """
    original_generate = attack.generate

    def generate(x, y=None, **kwargs):
        mask = np.ones(x.shape[1:], dtype=np.float32)
        mask[:, freeze_cols] = 0.0
        x_adv = original_generate(x=x, y=y, mask=mask, **kwargs)
        # Hard restore in case the underlying attack ignores `mask` entirely
        # (SaliencyMapMethod, CarliniL2Method - same belt-and-suspenders as adv_test)
        x_adv[:, :, freeze_cols] = x[:, :, freeze_cols]
        return x_adv

    attack.generate = generate
    return attack


def freeze_benign_and_cols(attack, freeze_cols=(0,), attacker_code : int = 1):
    """Patches an ART evasion attack in place so it can only perturb messages
    labeled attacker_code (default 1), in addition to freezing the given
    feature columns (see freeze_attack_cols).

    A real attacker only controls their own malicious messages, not benign
    traffic from other vehicles, so adversarial training shouldn't let
    FGSM/PGD move benign timesteps either - windows that are entirely benign
    end up with an all-zero mask and pass through generate() unperturbed.

    Usage: attack = freeze_benign_and_cols(ProjectedGradientDescent(classifier, eps=eps), freeze_cols=(0, 3))
    """
    original_generate = attack.generate

    def generate(x, y=None, **kwargs):
        mask = np.ones(x.shape, dtype=np.float32)
        mask[:, :, freeze_cols] = 0.0
        benign = None
        if y is not None:
            benign = (y != attacker_code)
            mask[benign] = 0.0
        x_adv = original_generate(x=x, y=y, mask=mask, **kwargs)
        x_adv[:, :, freeze_cols] = x[:, :, freeze_cols]
        if benign is not None:
            x_adv[benign] = x[benign]
        return x_adv

    attack.generate = generate
    return attack


def get_filename_from_path(file_path : str):
    """
    Get the filename from a filepath.

    Ex: ../../test.ckpt -> test
    """
    return file_path.split("/")[-1].split(".")[0]


def display_plot (title, display_metric, x : tuple[float, float], y : tuple[float, float], x_metric : str = "Epsilon"):
    """
    Provide the rest of the graph after calling plt for 
    graphing data

    Inputs: 
    - title, metric (for axis) - should be in correct format already
    - limit - whether to limit the graph to [0, 1]
    """
    
    plt.title(title)
    plt.xlabel(x_metric)
    plt.ylabel(f"{display_metric}")

    plt.xlim(x[0], x[1])
    plt.ylim(y[0], y[1])
    plt.grid(True)

    # Add flare to the graph
    plt.legend()
    plt.minorticks_on()
    plt.grid(True, which='minor', linewidth=0.5)
    plt.grid(True, which='major', linewidth=1.0)

def format_metric(metric : str):
    """
    Turn metric into a presentable label.
    
    Ex: falsePositiveRate -> False Positive Rate
    """
    words = []
    temp_word = ""
    for i, letter in enumerate(metric): 
        if (letter.isupper()):
            words.append(temp_word)
            temp_word = ""
        temp_word += letter
    else:
        words.append(temp_word)

    words = [word.capitalize() for word in words]
    return " ".join(words)
