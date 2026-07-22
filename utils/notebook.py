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

import os

import torch
from utils.models import OBU
from utils.functions import get_windowed_data, load_model_checkpoint

from sklearn.metrics import precision_score, recall_score, f1_score


class FilenameLoader():
    """
    Functions to load the names of the checkpoint, data files, save folder name

    Return: checkpoint file, data file, save folder name
    """
    
    def const_pos():
        return "ConstantPos-final.ckpt", "ConstPos_0709.csv", "constpos"
    def rand_pos():
        return "RandomPos-final.ckpt", "RandomPos_0709.csv", "randpos"
    def rand_speed():
        return "RandomSpeed-final.ckpt", "RandomSpeed_0709.csv", "randspeed"

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
        nb_classes=2, # the range [0, 3]; WRONG: the number of unique classes in y_test, len(np.unique(y_test.numpy()))
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
    if collapsed:
        benign_y_test = (y_test.numpy() == attacker_code).all(axis=1).astype(np.int64)
    else:
        benign_y_test = y_test.numpy()
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

    # Evasion goal is "make the classifier wrong", not "make it right". FGSM/PGD/CW
    # treat y as the true label to push AWAY from when targeted=False (their
    # default) - but SaliencyMapMethod has no targeted flag at all and always
    # pushes predictions TOWARD y, so passing the true label there tells it to
    # help the classifier get it right instead of attacking it (this is why F1
    # climbed toward 1.0 as gamma increased instead of dropping). Passing the
    # flipped label + targeted=True (where supported) gives every attack the same
    # "push away from truth" semantics - for a binary softmax/CE classifier this is
    # bit-for-bit identical to FGSM/PGD's original targeted=False behavior.
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

def get_filename_from_path(file_path : str):
    """
    Get the filename from a filepath.

    Ex: ../../test.ckpt -> test
    """
    return file_path.split("/")[-1].split(".")[0]