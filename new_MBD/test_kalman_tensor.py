"""Tests for evaluate_kalman_tensor_windows in MBD_systems/tensor_eval.py.

Covers four scenarios:
  1. Synthetic benign/attack windows — checks basic TP/TN logic
  2. Simulated adversarial perturbation — compares clean vs perturbed metrics
  3. Real CSV data (skipped if the file is missing)
  4. Real CSV data with simulated perturbation
  5. Real ART FGSM adversarial examples from a saved model checkpoint —
     compares neural network accuracy vs Kalman filter recall on the same examples

Run from the repo root:
    python new_MBD/test_kalman_tensor.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from MBD_systems.tensor_eval import evaluate_kalman_tensor_windows


# ---------------------------------------------------------------------------
# Helpers to build synthetic 10-message windows
# Feature layout (cols 3:11 from the raw CSV):
#   [0] rcvTime  [1] RelX  [2] RelY  [3] MssgCount
#   [4] dVx      [5] dVy   [6] dAx   [7] dAy
# ---------------------------------------------------------------------------

def make_benign_window(dt=0.1, vx=10.0, vy=5.0, n=10, label=0):
    """Constant-velocity trajectory. Kalman filter should NOT flag this."""
    t = np.arange(n) * dt
    return np.stack([
        t,                   # rcvTime
        vx * t,              # RelX
        vy * t,              # RelY
        np.zeros(n),         # MssgCount (unused by Kalman)
        np.full(n, vx),      # dVx
        np.full(n, vy),      # dVy
        np.zeros(n),         # dAx
        np.zeros(n),         # dAy
    ], axis=1).astype(np.float32), np.full(n, label, dtype=np.int64)


def make_attack_window(dt=0.1, label=3, n=10, seed=42):
    """Random position jumps. Kalman filter SHOULD flag this."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    return np.stack([
        t,
        rng.uniform(-200, 200, n),   # RelX: random jumps
        rng.uniform(-200, 200, n),   # RelY: random jumps
        np.zeros(n),
        rng.uniform(-20, 20, n),     # dVx
        rng.uniform(-20, 20, n),     # dVy
        rng.uniform(-5, 5, n),       # dAx
        rng.uniform(-5, 5, n),       # dAy
    ], axis=1).astype(np.float32), np.full(n, label, dtype=np.int64)


# ---------------------------------------------------------------------------
# Test 1: Synthetic benign + attack windows
# ---------------------------------------------------------------------------

def test_synthetic():
    print("=== Test 1: Synthetic benign + attack windows ===")

    windows, labels = [], []
    for _ in range(5):
        w, l = make_benign_window()
        windows.append(w); labels.append(l)
    for i in range(5):
        w, l = make_attack_window(seed=i)
        windows.append(w); labels.append(l)

    x = np.stack(windows)   # (10, 10, 8)
    y = np.stack(labels)    # (10, 10)

    m = evaluate_kalman_tensor_windows(x, y, scaler=None, attacker_code=3, adversarial=True)
    print(f"  tp={m['tp']}, tn={m['tn']}, fp={m['fp']}, fn={m['fn']}")
    print(f"  accuracy={m['accuracy']:.4f}  recall={m['recall']:.4f}  f1={m['f1']:.4f}")
    print(f"  attackSuccessRate={m['attackSuccessRate']:.4f}")

    assert m['tn'] > 0, "Expected true negatives from constant-velocity benign windows"
    assert m['tp'] > 0, "Expected true positives from random-jump attack windows"
    print("  PASSED\n")


# ---------------------------------------------------------------------------
# Test 2: Simulated adversarial perturbation
# ---------------------------------------------------------------------------

def test_perturbation():
    print("=== Test 2: Simulated adversarial perturbation ===")
    rng = np.random.default_rng(1)

    # 20 attack windows (smooth trajectories labeled as attacks)
    windows, labels = [], []
    for i in range(20):
        w, _ = make_benign_window(vx=float(i), vy=float(i) * 0.5)
        windows.append(w)
        labels.append(np.full(10, 3, dtype=np.int64))  # labeled attack

    x_clean = np.stack(windows)
    y = np.stack(labels)

    # Adversarial: large random jumps added to position features
    x_adv = x_clean.copy()
    x_adv[:, :, 1] += rng.uniform(-100, 100, x_adv[:, :, 1].shape).astype(np.float32)
    x_adv[:, :, 2] += rng.uniform(-100, 100, x_adv[:, :, 2].shape).astype(np.float32)

    clean = evaluate_kalman_tensor_windows(x_clean, y, scaler=None, attacker_code=3, adversarial=True)
    adv   = evaluate_kalman_tensor_windows(x_adv,   y, scaler=None, attacker_code=3, adversarial=True)

    print(f"  Clean     — recall={clean['recall']:.4f}  ASR={clean['attackSuccessRate']:.4f}")
    print(f"  Perturbed — recall={adv['recall']:.4f}  ASR={adv['attackSuccessRate']:.4f}")
    print("  PASSED\n")


# ---------------------------------------------------------------------------
# Test 3: Real CSV data (skipped if file missing)
# ---------------------------------------------------------------------------

def test_real_data():
    data_file = Path(__file__).parent.parent / "data" / "RandomPos_0709.csv"
    if not data_file.exists():
        print(f"=== Test 3: SKIPPED (not found: {data_file}) ===\n")
        return

    print("=== Test 3: Real CSV data (first 500 test windows) ===")
    from utils.functions import get_windowed_data

    (x_train, _), (x_test, y_test), _, scaler = get_windowed_data(
        str(data_file), normalize=False, train_perc=80
    )

    N = 500
    x_np = x_test[:N].numpy()
    y_np = y_test[:N].numpy()

    # norm_trained=False → scaler=None (data is already in raw units)
    m = evaluate_kalman_tensor_windows(x_np, y_np, scaler=None, attacker_code=3, adversarial=True)
    print(f"  tp={m['tp']}, tn={m['tn']}, fp={m['fp']}, fn={m['fn']}")
    print(f"  accuracy={m['accuracy']:.4f}  recall={m['recall']:.4f}  f1={m['f1']:.4f}")
    print(f"  attackSuccessRate={m['attackSuccessRate']:.4f}")
    print("  PASSED\n")


def test_real_data_with_perturbation():
    data_file = Path(__file__).parent.parent / "data" / "RandomPos_0709.csv"
    if not data_file.exists():
        print(f"=== Test 4: SKIPPED (not found: {data_file}) ===\n")
        return

    print("=== Test 4: Real data — clean vs simulated adversarial ===")
    from utils.functions import get_windowed_data

    (x_train, _), (x_test, y_test), _, scaler = get_windowed_data(
        str(data_file), normalize=False, train_perc=80
    )

    N = 500
    rng = np.random.default_rng(42)
    x_np = x_test[:N].numpy()
    y_np = y_test[:N].numpy()

    # Simulate adversarial: add small eps=5.0 noise to position features
    eps = 5.0
    x_adv = x_np.copy()
    x_adv[:, :, 1] += rng.uniform(-eps, eps, x_adv[:, :, 1].shape).astype(np.float32)
    x_adv[:, :, 2] += rng.uniform(-eps, eps, x_adv[:, :, 2].shape).astype(np.float32)

    clean = evaluate_kalman_tensor_windows(x_np,  y_np, scaler=None, attacker_code=3, adversarial=True)
    adv   = evaluate_kalman_tensor_windows(x_adv, y_np, scaler=None, attacker_code=3, adversarial=True)

    print(f"  Clean     — accuracy={clean['accuracy']:.4f}  recall={clean['recall']:.4f}  ASR={clean['attackSuccessRate']:.4f}")
    print(f"  Perturbed — accuracy={adv['accuracy']:.4f}  recall={adv['recall']:.4f}  ASR={adv['attackSuccessRate']:.4f}")
    print("  PASSED\n")


# ---------------------------------------------------------------------------
# Test 5: Real ART FGSM adversarial examples → Kalman filter
# ---------------------------------------------------------------------------

def test_with_art_fgsm(checkpoint_file=None, data_file=None, norm_trained=True,
                        eps=0.1, n_windows=500):
    checkpoint_path = Path(checkpoint_file) if checkpoint_file else \
        Path(__file__).parent.parent / "saved_models" / "RandomPos-final.ckpt"
    data_path = Path(data_file) if data_file else \
        Path(__file__).parent.parent / "data" / "RandomPos_0709.csv"

    if not checkpoint_path.exists():
        print(f"=== Test 5: SKIPPED (not found: {checkpoint_path.name}) ===\n")
        return
    if not data_path.exists():
        print(f"=== Test 5: SKIPPED (not found: {data_path.name}) ===\n")
        return

    print(f"=== Test 5: ART FGSM adversarial examples → Kalman filter ===")
    print(f"  checkpoint={checkpoint_path.name}  norm_trained={norm_trained}  eps={eps}")

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from art.estimators.classification import PyTorchClassifier
    from art.attacks.evasion import FastGradientMethod
    from utils.functions import get_windowed_data, load_model_checkpoint

    # Load data — normalize must match how the checkpoint was trained
    (x_train, _), (x_test, y_test), _, scaler = get_windowed_data(
        str(data_path), normalize=norm_trained, train_perc=80
    )

    model = load_model_checkpoint(str(checkpoint_path))

    # Collapses (N, 10, 2) → (N, 2) so ART sees a flat classifier output
    class CfCWrapper(nn.Module):
        def __init__(self, modena):
            super().__init__()
            self.modena = modena
        def forward(self, x):
            logits, _ = self.modena(x)
            return logits.mean(dim=1)

    class SequenceCrossEntropy(nn.Module):
        def __init__(self):
            super().__init__()
            self.ce = nn.CrossEntropyLoss()
        def forward(self, a, b):
            # a: (N, 2) collapsed; b: one-hot (N, 2) or indices (N,) from ART
            if b.dim() == 2:
                b = b.argmax(dim=-1)
            return self.ce(a, b.long())

    wrapped = CfCWrapper(model.model)
    criterion = SequenceCrossEntropy()
    optimizer = optim.Adam(wrapped.parameters(), lr=0.001)

    clip_lo = 0.0 if norm_trained else float(x_train.min())
    clip_hi = 1.0 if norm_trained else float(x_train.max())

    classifier = PyTorchClassifier(
        model=wrapped, loss=criterion, optimizer=optimizer,
        input_shape=(10, 8), nb_classes=2,
        clip_values=(clip_lo, clip_hi), device_type="cpu",
    )

    # When norm_trained=True, the scaler maps AttkType 3 → 1, so labels in y_test are {0, 1}
    attacker_code = 1 if norm_trained else 3

    N = n_windows
    x_np = x_test[:N].numpy()
    y_np = y_test[:N].numpy()

    # Mask: freeze rcvTime (feature 0) — a real attacker can't manipulate receive timestamps.
    # All motion features (RelX, RelY, dVx, dVy, dAx, dAy) are still perturbed.
    mask = np.ones((10, 8), dtype=np.float32)
    mask[:, 0] = 0.0  # rcvTime

    print(f"  Generating FGSM adversarial examples for {N} windows (rcvTime frozen)...")
    x_adv = FastGradientMethod(classifier, eps=eps).generate(x=x_np, mask=mask)

    # Window-level NN metrics: window is attack if ALL 10 labels == attacker_code
    def nn_window_metrics(x_in, y_labels):
        preds = classifier.predict(x_in)
        pred_cls = np.argmax(preds, axis=-1)
        true_cls = np.array([(row == attacker_code).all() for row in y_labels], dtype=int)
        TP = int(np.sum((true_cls == 1) & (pred_cls == 1)))
        TN = int(np.sum((true_cls == 0) & (pred_cls == 0)))
        FP = int(np.sum((true_cls == 0) & (pred_cls == 1)))
        FN = int(np.sum((true_cls == 1) & (pred_cls == 0)))
        total = TP + TN + FP + FN
        acc = (TP + TN) / total if total > 0 else 0.0
        rec = TP / (TP + FN)  if (TP + FN) > 0 else 0.0
        asr = FN / (TP + FN)  if (TP + FN) > 0 else 0.0
        return {"accuracy": acc, "recall": rec, "attackSuccessRate": asr,
                "TP": TP, "TN": TN, "FP": FP, "FN": FN}

    nn_clean = nn_window_metrics(x_np,  y_np)
    nn_adv   = nn_window_metrics(x_adv, y_np)

    kal_clean = evaluate_kalman_tensor_windows(
        x_np,  y_np, scaler=scaler if norm_trained else None,
        attacker_code=attacker_code, adversarial=True
    )
    kal_adv = evaluate_kalman_tensor_windows(
        x_adv, y_np, scaler=scaler if norm_trained else None,
        attacker_code=attacker_code, adversarial=True
    )

    print(f"\n  Neural network (window-level, {N} windows):")
    print(f"    Clean       — accuracy={nn_clean['accuracy']:.4f}  recall={nn_clean['recall']:.4f}  ASR={nn_clean['attackSuccessRate']:.4f}")
    print(f"    Adversarial — accuracy={nn_adv['accuracy']:.4f}  recall={nn_adv['recall']:.4f}  ASR={nn_adv['attackSuccessRate']:.4f}")
    print(f"\n  Kalman filter (window-level, {N} windows):")
    print(f"    Clean       — accuracy={kal_clean['accuracy']:.4f}  recall={kal_clean['recall']:.4f}  ASR={kal_clean['attackSuccessRate']:.4f}")
    print(f"    Adversarial — accuracy={kal_adv['accuracy']:.4f}  recall={kal_adv['recall']:.4f}  ASR={kal_adv['attackSuccessRate']:.4f}")
    print("\n  PASSED\n")


if __name__ == "__main__":
    test_synthetic()
    test_perturbation()
    test_real_data()
    test_real_data_with_perturbation()
    test_with_art_fgsm()
    print("All tests done.")
