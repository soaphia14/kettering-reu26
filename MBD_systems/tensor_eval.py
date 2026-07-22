"""Runs MBD plausibility checks directly on model-format windowed tensors,
without needing a CSV file. Designed for use with get_windowed_data(pair_windows=True),
where each window in x_test is already from one (sender, receiver) pair.

The 8 features expected (columns 3:11 from the raw CSV):
  [0] rcvTime, [1] RelX, [2] RelY, [3] MssgCount, [4] dVx, [5] dVy, [6] dAx, [7] dAy

Both LegacyChecks and the Kalman filter are available.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_structures import Parameters
from legacy_checks import LegacyChecks
from windowed_eval import evaluate_window, aggregate_window_metrics
from kalman_check import RelativeMotionKalmanFilter, CHI2_6DOF_P99


def evaluate_tensor_windows(x_windows, y_windows, scaler, attacker_code: int = 3,
                             params: Parameters = None, adversarial: bool = False) -> dict:
    """Runs LegacyChecks on pre-windowed normalized tensors and returns window-level metrics.

    Args:
        x_windows:   (N, 10, 7) numpy array or torch tensor, normalized
        y_windows:   (N, 10) or (N,) labels — (N,) is treated as one label per window
        scaler:      fitted MinMaxScaler returned by get_windowed_data
        attacker_code: AttkType value that means "attack" (e.g. 3 for RandomPos)
        params:      Parameters instance with thresholds; None uses defaults
        adversarial: if True, also compute ASR (fraction of malicious windows that evaded detection)

    Returns:
        dict with tp, tn, fp, fn, accuracy, precision, recall, f1, check_counts,
        and optionally attackSuccessRate if adversarial=True
    """
    if params is None:
        params = Parameters()
    checks = LegacyChecks(params)

    if hasattr(x_windows, 'numpy'):
        x_windows = x_windows.numpy()
    if hasattr(y_windows, 'numpy'):
        y_windows = y_windows.numpy()

    N = x_windows.shape[0]
    n_features = x_windows.shape[2]
    # Scaler was fit on 9 cols (3:12) but x_windows has 8 (3:11), so manually
    # inverse-transform using only the first n_features components of the scaler
    x_flat = x_windows.reshape(-1, n_features)
    x_raw = ((x_flat - scaler.min_[:n_features]) / scaler.scale_[:n_features]).reshape(N, 10, n_features)

    scenario_stats = []
    for i in range(N):
        window = x_raw[i]  # (10, n_features)
        labels = y_windows[i] if y_windows.ndim == 2 else np.full(10, y_windows[i])

        window_df = pd.DataFrame({
            'rcvTime': window[:, 0],
            'RelX':    window[:, 1],
            'RelY':    window[:, 2],
            'dVx':     window[:, 4],
            'dVy':     window[:, 5],
            'AttkType': labels,
        })

        outcome = evaluate_window(window_df, checks, attacker_code)

        tp = tn = fp = fn = 0
        if   outcome['ground_truth'] == 1 and outcome['prediction'] == 1:
            tp = 1
        elif outcome['ground_truth'] == 0 and outcome['prediction'] == 0:
            tn = 1
        elif outcome['ground_truth'] == 0 and outcome['prediction'] == 1:
            fp = 1
        else:
            fn = 1

        scenario_stats.append({'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
                                'check_counts': outcome['check_counts']})

    metrics = aggregate_window_metrics(scenario_stats)

    if adversarial:
        total_malicious = metrics['tp'] + metrics['fn']
        metrics['attackSuccessRate'] = metrics['fn'] / total_malicious if total_malicious > 0 else 0

    return metrics


def evaluate_kalman_tensor_windows(x_windows, y_windows, scaler=None, attacker_code: int = 3,
                                   jerk_noise: float = 1.0, measurement_noise: float = 0.5,
                                   chi2_threshold: float = CHI2_6DOF_P99,
                                   adversarial: bool = False) -> dict:
    """Runs a per-window Kalman filter on pre-windowed tensors and returns window-level metrics.

    Each 10-message window gets its own fresh Kalman filter (no state carries across windows).
    The first message in each window initializes the filter; messages 2-10 are checked against
    predictions. A window is predicted "attack" if any message's Mahalanobis distance exceeds
    chi2_threshold. Requires 8 features (columns 3:11) so that dAy (index 7) is present.

    Args:
        x_windows:    (N, 10, 8) numpy array or torch tensor; normalized or raw
        y_windows:    (N, 10) or (N,) labels
        scaler:       fitted MinMaxScaler from get_windowed_data; None if data is already raw
        attacker_code: raw AttkType value for attacks (e.g. 3 for RandomPos, before normalization)
        adversarial:  if True, also compute attackSuccessRate (FN / all malicious windows)

    Returns:
        dict with tp, tn, fp, fn, accuracy, precision, recall, f1,
        and optionally attackSuccessRate if adversarial=True
    """
    if hasattr(x_windows, 'numpy'):
        x_windows = x_windows.numpy()
    if hasattr(y_windows, 'numpy'):
        y_windows = y_windows.numpy()

    N = x_windows.shape[0]
    n_features = x_windows.shape[2]

    if scaler is not None:
        x_flat = x_windows.reshape(-1, n_features)
        x_raw = ((x_flat - scaler.min_[:n_features]) / scaler.scale_[:n_features]).reshape(N, 10, n_features)
    else:
        x_raw = x_windows

    tp = tn = fp = fn = 0
    for i in range(N):
        window = x_raw[i]  # (10, n_features)
        labels = y_windows[i] if y_windows.ndim == 2 else np.full(10, y_windows[i])
        ground_truth = 1 if (labels == attacker_code).all() else 0

        kf = RelativeMotionKalmanFilter(jerk_noise, measurement_noise)
        prev_time = None
        flagged = False
        for j in range(len(window)):
            row = window[j]
            z = np.array([row[1], row[4], row[6], row[2], row[5], row[7]], dtype=np.float64)
            # z = [RelX, dVx, dAx, RelY, dVy, dAy]
            if prev_time is None:
                kf.step(z, 0.0)
            else:
                dt = float(row[0]) - prev_time
                if dt > 0:
                    d2 = kf.step(z, dt)
                    if d2 is not None and d2 > chi2_threshold:
                        flagged = True
            prev_time = float(row[0])

        prediction = 1 if flagged else 0
        if ground_truth == 1 and prediction == 1:
            tp += 1
        elif ground_truth == 0 and prediction == 0:
            tn += 1
        elif ground_truth == 0 and prediction == 1:
            fp += 1
        else:
            fn += 1

    total = tp + tn + fp + fn
    accuracy  = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
               'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

    if adversarial:
        total_malicious = tp + fn
        metrics['attackSuccessRate'] = fn / total_malicious if total_malicious > 0 else 0.0

    return metrics
