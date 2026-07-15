"""Runs LegacyChecks plausibility checks directly on model-format windowed tensors,
without needing a CSV file. Designed for use with get_windowed_data(pair_windows=True),
where each window in x_test is already from one (sender, receiver) pair.

The 7 features expected (columns 3:10 from the raw CSV, as used by get_windowed_data):
  [0] rcvTime, [1] RelX, [2] RelY, [3] MssgCount, [4] dVx, [5] dVy, [6] dAx

Note: dAy (col 10) is absent from the model input, so the Kalman filter cannot run here.
All legacy checks (range, speed, position/speed consistency, sudden appearance) work fine.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_structures import Parameters
from legacy_checks import LegacyChecks
from windowed_eval import evaluate_window, aggregate_window_metrics


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
    x_raw = scaler.inverse_transform(x_windows.reshape(-1, 7)).reshape(N, 10, 7)

    scenario_stats = []
    for i in range(N):
        window = x_raw[i]  # (10, 7)
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
