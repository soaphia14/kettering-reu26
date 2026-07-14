"""Kalman-filter-based plausibility check for the relative-feature CSVs
(RecieverID/SenderID/rcvTime/RelX/RelY/dVx/dVy/dAx/dAy/AttkType).

Tracks a constant-acceleration motion model over one sender's relative position/velocity/
acceleration [x, vx, ax, y, vy, ay], AS SEEN BY ONE FIXED RECEIVER, and flags any message whose
reported state is a statistical outlier (large Mahalanobis distance on the Kalman innovation)
relative to what the filter predicted from the message before it.

This only makes sense on (SenderID, RecieverID)-paired data (see
MBD_systems.windowed_eval.split_csv_by_sender_receiver) - a continuous trajectory in one
consistent relative frame. Running it on sender-only-grouped data would mix multiple receivers'
relative frames into a single supposedly-continuous trajectory and produce meaningless
innovations (see the earlier RelX comparison: sender 9 alternating between receiver 45 and 63).
"""

import os
import sys
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from windowed_eval import split_csv_by_sender_receiver, RAW_CSV_CHUNKSIZE, WINDOW_SIZE, STRIDE

STATE_DIM = 6  # [x, vx, ax, y, vy, ay]

# Chi-square critical value for 6 degrees of freedom at alpha=0.01 (99th percentile).
CHI2_6DOF_P99 = 16.812


class RelativeMotionKalmanFilter:
    """Constant-acceleration Kalman filter over one sender's relative position/velocity/
    acceleration, as seen by one fixed receiver. State: [x, vx, ax, y, vy, ay]. Measurement =
    state directly (RelX/dVx/dAx/RelY/dVy/dAy are all measured, none are inferred), so H = I."""

    def __init__(self, jerk_noise: float = 1.0, measurement_noise: float = 0.5):
        self.jerk_noise = jerk_noise
        self.R = np.eye(STATE_DIM) * measurement_noise
        self.x = None
        self.P = None

    @staticmethod
    def _axis_blocks(dt: float, jerk_noise: float):
        """Per-axis 3x3 constant-acceleration transition and continuous-white-noise-jerk
        process-noise blocks (position, velocity, acceleration)."""
        f = np.array([
            [1.0, dt, 0.5 * dt * dt],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0],
        ])
        q = jerk_noise * np.array([
            [dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
            [dt ** 4 / 8, dt ** 3 / 3, dt ** 2 / 2],
            [dt ** 3 / 6, dt ** 2 / 2, dt],
        ])
        return f, q

    def _transition(self, dt: float):
        f_block, q_block = self._axis_blocks(dt, self.jerk_noise)
        F = np.zeros((STATE_DIM, STATE_DIM))
        Q = np.zeros((STATE_DIM, STATE_DIM))
        F[0:3, 0:3] = f_block
        F[3:6, 3:6] = f_block
        Q[0:3, 0:3] = q_block
        Q[3:6, 3:6] = q_block
        return F, Q

    def step(self, z: np.ndarray, dt: float):
        """Predicts forward by dt, updates with measurement z, and returns the squared
        Mahalanobis distance of the innovation. Returns None on the very first call, since
        there's no prior state yet to predict from (nothing to check the first message against)."""
        if self.x is None:
            self.x = z.copy()
            self.P = self.R * 10.0
            return None

        F, Q = self._transition(dt)
        x_pred = F @ self.x
        P_pred = F @ self.P @ F.T + Q

        innovation = z - x_pred
        S = P_pred + self.R
        S_inv = np.linalg.inv(S)
        K = P_pred @ S_inv

        self.x = x_pred + K @ innovation
        self.P = (np.eye(STATE_DIM) - K) @ P_pred

        return float(innovation.T @ S_inv @ innovation)


def _row_to_axis_major(row: pd.Series) -> np.ndarray:
    return np.array([row['RelX'], row['dVx'], row['dAx'], row['RelY'], row['dVy'], row['dAy']],
                    dtype=np.float64)


def flag_implausible_messages(df: pd.DataFrame, jerk_noise: float = 1.0,
                              measurement_noise: float = 0.5,
                              chi2_threshold: float = CHI2_6DOF_P99) -> pd.Series:
    """Runs the Kalman filter across one (sender, receiver) pair's full chronologically-sorted
    history (df must already be sorted by rcvTime) and flags every message whose innovation
    Mahalanobis distance exceeds chi2_threshold. Returns a boolean Series aligned to df.index -
    the first message is always False, since there's no prior state to check it against."""
    kf = RelativeMotionKalmanFilter(jerk_noise, measurement_noise)
    flags = []
    prev_time = None

    for _, row in df.iterrows():
        z = _row_to_axis_major(row)
        if prev_time is None:
            kf.step(z, 0.0)
            flags.append(False)
        else:
            dt = row['rcvTime'] - prev_time
            if dt <= 0:
                flags.append(False)
            else:
                d2 = kf.step(z, dt)
                flags.append(d2 is not None and d2 > chi2_threshold)
        prev_time = row['rcvTime']

    return pd.Series(flags, index=df.index)


def process_kalman_pair(df: pd.DataFrame, attacker_code: int, jerk_noise: float,
                        measurement_noise: float, chi2_threshold: float,
                        window_size: int = WINDOW_SIZE, stride: int = STRIDE) -> Dict[str, int]:
    """One (sender, receiver) pair's rows -> Kalman-flagged messages -> the same 10-message/
    stride-5 windows as windowed_eval.py -> window-level tp/tn/fp/fn counts."""
    df = df.sort_values('rcvTime').reset_index(drop=True)
    flagged = flag_implausible_messages(df, jerk_noise, measurement_noise, chi2_threshold)

    tp = tn = fp = fn = 0
    flagged_messages = 0
    n = len(df)
    index = 0
    while index < n - window_size:
        window_attk = df['AttkType'].iloc[index:index + window_size]
        window_flags = flagged.iloc[index:index + window_size]

        ground_truth = 1 if (window_attk == attacker_code).all() else 0
        prediction = 1 if window_flags.any() else 0

        if ground_truth == 1 and prediction == 1:
            tp += 1
        elif ground_truth == 0 and prediction == 0:
            tn += 1
        elif ground_truth == 0 and prediction == 1:
            fp += 1
        else:
            fn += 1

        flagged_messages += int(window_flags.sum())
        index += stride

    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'flagged_messages': flagged_messages}


def worker_evaluate_kalman_pair(pair_file: str, attacker_code: int, jerk_noise: float,
                                measurement_noise: float, chi2_threshold: float,
                                window_size: int, stride: int, source_file: str):
    try:
        df = pd.read_csv(pair_file)
        result = process_kalman_pair(df, attacker_code, jerk_noise, measurement_noise,
                                     chi2_threshold, window_size, stride)
        return {'status': 'ok', 'metrics': result, 'source': source_file}
    except Exception as e:
        return {'status': 'error', 'error': str(e), 'source': source_file}


def aggregate_kalman_metrics(scenario_stats: List[dict]) -> dict:
    total_tp = sum(s.get('tp', 0) for s in scenario_stats)
    total_tn = sum(s.get('tn', 0) for s in scenario_stats)
    total_fp = sum(s.get('fp', 0) for s in scenario_stats)
    total_fn = sum(s.get('fn', 0) for s in scenario_stats)
    total_flagged_messages = sum(s.get('flagged_messages', 0) for s in scenario_stats)

    total_windows = total_tp + total_tn + total_fp + total_fn

    metrics = {
        'tp': total_tp, 'tn': total_tn, 'fp': total_fp, 'fn': total_fn,
        'flagged_messages': total_flagged_messages,
        'accuracy': (total_tp + total_tn) / total_windows if total_windows > 0 else 0,
        'precision': total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0,
        'recall': total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0,
    }
    metrics['f1'] = (2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])
                     if (metrics['precision'] + metrics['recall']) > 0 else 0)
    return metrics


def evaluate_kalman_windowed_ids(input_csv: str, attacker_code: int = 3, jerk_noise: float = 1.0,
                                 measurement_noise: float = 0.5, chi2_threshold: float = CHI2_6DOF_P99,
                                 window_size: int = WINDOW_SIZE, stride: int = STRIDE,
                                 workers: int = None, chunksize: int = RAW_CSV_CHUNKSIZE,
                                 cleanup: bool = True) -> dict:
    """Splits input_csv by (SenderID, RecieverID) pair, runs the Kalman-filter plausibility
    check continuously across each pair's full history, then scores the same 10-message/
    stride-5 windows as windowed_eval.evaluate_windowed_ids_by_pair - so the two are directly
    comparable head to head. jerk_noise/measurement_noise/chi2_threshold are tunable: raise
    measurement_noise or chi2_threshold to make the filter less trigger-happy."""
    csv_path = Path(input_csv)
    workers = workers or os.cpu_count() or 4

    tmp_dir = csv_path.parent / f"_pair_split_{csv_path.stem}"
    print(f"Splitting {csv_path} by (SenderID, RecieverID) into {tmp_dir} ...", file=sys.stderr)
    pairs = split_csv_by_sender_receiver(csv_path, tmp_dir, chunksize)
    print(f"Found {len(pairs)} sender/receiver pairs, evaluating with {workers} workers...", file=sys.stderr)

    scenario_stats = []
    count = 0
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for sender_id, receiver_id in pairs:
                pair_file = tmp_dir / f"pair_{sender_id}_{receiver_id}.csv"
                source_name = f"{sender_id}_{receiver_id}"
                f = executor.submit(worker_evaluate_kalman_pair, str(pair_file), attacker_code,
                                    jerk_noise, measurement_noise, chi2_threshold,
                                    window_size, stride, source_name)
                futures[f] = source_name

            for future in as_completed(futures):
                source = futures[future]
                res = future.result()
                if res.get('status') == 'ok':
                    scenario_stats.append(res.get('metrics', {}))
                else:
                    print(f"[ERROR] pair {source}: {res.get('error')}", file=sys.stderr)
                count += 1
                if count % 5000 == 0 or count == len(pairs):
                    print(f"Processed {count}/{len(pairs)} pairs", file=sys.stderr)
    finally:
        if cleanup and tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    return aggregate_kalman_metrics(scenario_stats)
