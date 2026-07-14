"""Calibrates LegacyChecks (and the Kalman filter) thresholds against this dataset's own
benign traffic, instead of reusing Parameters defaults that were tuned for a different
(absolute-position/noise) schema.

Splits the file chronologically into a TRAIN slice (first train_perc%) and a TEST slice (the
rest), mirroring the trainPerc convention used elsewhere in this repo (get_windowed_data,
CenFL.py). Only TRAIN rows with AttkType == 0 (confirmed benign) are used to build empirical
distributions of the raw quantities each check compares against a threshold; a percentile of
each distribution becomes the new threshold - e.g. the 99th percentile of benign relative
distance becomes MAX_PLAUSIBLE_RANGE, so only ~1% of genuine benign traffic would trip that
check. The TEST slice is then used to evaluate default vs. calibrated thresholds head to head,
so the comparison isn't calibrated-and-tested on the same data.
"""

import os
import sys
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_structures import Parameters
from windowed_eval import split_csv_by_sender_receiver, evaluate_windowed_ids_by_pair, RAW_CSV_CHUNKSIZE
from kalman_check import RelativeMotionKalmanFilter, evaluate_kalman_windowed_ids, CHI2_6DOF_P99

STAT_KEYS = ('distances', 'speeds', 'implied_speeds', 'implied_accels', 'implied_decels', 'mahalanobis')


def _time_bounds(csv_path: Path, chunksize: int = RAW_CSV_CHUNKSIZE):
    min_time = max_time = None
    for chunk in pd.read_csv(csv_path, usecols=['rcvTime'], chunksize=chunksize):
        cmin, cmax = chunk['rcvTime'].min(), chunk['rcvTime'].max()
        min_time = cmin if min_time is None else min(min_time, cmin)
        max_time = cmax if max_time is None else max(max_time, cmax)
    return min_time, max_time


def write_test_slice(csv_path: Path, cutoff_time: float, out_path: Path,
                     chunksize: int = RAW_CSV_CHUNKSIZE) -> None:
    """Streams csv_path and writes only rows with rcvTime >= cutoff_time to out_path - a
    held-out evaluation slice that calibration never saw."""
    if out_path.exists():
        out_path.unlink()
    wrote_header = False
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, index_col=0):
        test_chunk = chunk[chunk['rcvTime'] >= cutoff_time]
        if not test_chunk.empty:
            test_chunk.to_csv(out_path, mode='a', header=not wrote_header, index=False)
            wrote_header = True


def _pair_benign_stats(df: pd.DataFrame, cutoff_time: float, jerk_noise: float,
                       measurement_noise: float) -> dict:
    train_benign = df[(df['rcvTime'] < cutoff_time) & (df['AttkType'] == 0)].sort_values('rcvTime')

    distances, speeds = [], []
    implied_speeds, implied_accels, implied_decels, mahalanobis = [], [], [], []
    first_distance = None

    kf = RelativeMotionKalmanFilter(jerk_noise, measurement_noise)
    prev_row = None

    for _, row in train_benign.iterrows():
        distance = float(np.hypot(row['RelX'], row['RelY']))
        speed = float(np.hypot(row['dVx'], row['dVy']))

        distances.append(distance)
        speeds.append(speed)
        if first_distance is None:
            first_distance = distance

        z = np.array([row['RelX'], row['dVx'], row['dAx'], row['RelY'], row['dVy'], row['dAy']],
                     dtype=np.float64)

        if prev_row is None:
            kf.step(z, 0.0)
        else:
            dt = row['rcvTime'] - prev_row['rcvTime']
            if dt > 0:
                implied_speed = float(np.hypot(row['RelX'] - prev_row['RelX'],
                                               row['RelY'] - prev_row['RelY'])) / dt
                implied_speeds.append(implied_speed)

                prev_speed = float(np.hypot(prev_row['dVx'], prev_row['dVy']))
                speed_delta = speed - prev_speed
                if speed_delta > 0:
                    implied_accels.append(speed_delta / dt)
                else:
                    implied_decels.append(abs(speed_delta) / dt)

                d2 = kf.step(z, dt)
                if d2 is not None:
                    mahalanobis.append(d2)

        prev_row = row

    return {
        'distances': distances, 'first_distance': first_distance, 'speeds': speeds,
        'implied_speeds': implied_speeds, 'implied_accels': implied_accels,
        'implied_decels': implied_decels, 'mahalanobis': mahalanobis,
    }


def worker_collect_pair_stats(pair_file: str, cutoff_time: float, jerk_noise: float,
                              measurement_noise: float):
    try:
        df = pd.read_csv(pair_file)
        return {'status': 'ok', 'stats': _pair_benign_stats(df, cutoff_time, jerk_noise, measurement_noise)}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def calibrate_parameters(input_csv: str, train_perc: float = 80, percentile: float = 99,
                         jerk_noise: float = 1.0, measurement_noise: float = 0.5,
                         workers: int = None, chunksize: int = RAW_CSV_CHUNKSIZE,
                         cleanup: bool = True) -> Dict[str, Any]:
    """Returns {'params': <calibrated Parameters>, 'chi2_threshold': <float>, 'cutoff_time':
    <float>, 'summary': {...sample counts and percentile values per statistic...}}, using only
    the first train_perc% of input_csv (by rcvTime) and only its AttkType == 0 rows."""
    csv_path = Path(input_csv)
    workers = workers or os.cpu_count() or 4

    print(f"Scanning {csv_path} for rcvTime bounds...", file=sys.stderr)
    min_time, max_time = _time_bounds(csv_path, chunksize)
    cutoff_time = min_time + (train_perc / 100) * (max_time - min_time)
    print(f"Train/test cutoff rcvTime={cutoff_time:.2f} (train_perc={train_perc}%)", file=sys.stderr)

    tmp_dir = csv_path.parent / f"_pair_split_{csv_path.stem}"
    pairs = split_csv_by_sender_receiver(csv_path, tmp_dir, chunksize)
    print(f"Found {len(pairs)} pairs, collecting benign-only train-slice stats with {workers} workers...",
          file=sys.stderr)

    pooled = {key: [] for key in STAT_KEYS}
    pooled['first_distance'] = []
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(worker_collect_pair_stats, str(tmp_dir / f"pair_{s}_{r}.csv"),
                                       cutoff_time, jerk_noise, measurement_noise) for s, r in pairs]
            count = 0
            for future in as_completed(futures):
                res = future.result()
                count += 1
                if count % 5000 == 0 or count == len(futures):
                    print(f"Processed {count}/{len(futures)} pairs", file=sys.stderr)
                if res.get('status') != 'ok':
                    continue
                stats = res['stats']
                for key in STAT_KEYS:
                    pooled[key].extend(stats[key])
                if stats['first_distance'] is not None:
                    pooled['first_distance'].append(stats['first_distance'])
    finally:
        if cleanup and tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    def pct(key, p):
        arr = pooled[key]
        return float(np.percentile(arr, p)) if arr else None

    kwargs = {}
    for field, key, p in (
        ('MAX_PLAUSIBLE_RANGE', 'distances', percentile),
        ('MAX_SA_RANGE', 'first_distance', 100 - percentile),
        ('MAX_PLAUSIBLE_SPEED', 'speeds', percentile),
        ('MAX_PLAUSIBLE_ACCEL', 'implied_accels', percentile),
        ('MAX_PLAUSIBLE_DECEL', 'implied_decels', percentile),
    ):
        value = pct(key, p)
        if value is not None:
            kwargs[field] = value

    calibrated_params = Parameters(**kwargs)
    chi2_threshold = pct('mahalanobis', percentile) or CHI2_6DOF_P99

    summary = {key: {'n': len(pooled[key]), f'p{percentile}': pct(key, percentile)}
              for key in list(STAT_KEYS) + ['first_distance']}

    return {
        'params': calibrated_params,
        'chi2_threshold': chi2_threshold,
        'cutoff_time': cutoff_time,
        'summary': summary,
    }


def compare_before_after(input_csv: str, attacker_code: int = 3, train_perc: float = 80,
                         percentile: float = 99, workers: int = None,
                         chunksize: int = RAW_CSV_CHUNKSIZE) -> Dict[str, Any]:
    """Calibrates on the train slice, then evaluates LegacyChecks (pair-based) and the Kalman
    filter on the held-out test slice with default vs. calibrated thresholds - a direct,
    train/test-separated before/after comparison."""
    csv_path = Path(input_csv)

    calibration = calibrate_parameters(input_csv, train_perc, percentile, workers=workers,
                                       chunksize=chunksize)

    test_slice_path = csv_path.parent / f"_test_slice_{csv_path.stem}.csv"
    print(f"Writing held-out test slice to {test_slice_path} ...", file=sys.stderr)
    write_test_slice(csv_path, calibration['cutoff_time'], test_slice_path, chunksize)

    try:
        legacy_default = evaluate_windowed_ids_by_pair(str(test_slice_path), attacker_code, workers=workers)
        legacy_calibrated = evaluate_windowed_ids_by_pair(str(test_slice_path), attacker_code,
                                                          workers=workers, params=calibration['params'])
        kalman_default = evaluate_kalman_windowed_ids(str(test_slice_path), attacker_code, workers=workers)
        kalman_calibrated = evaluate_kalman_windowed_ids(str(test_slice_path), attacker_code,
                                                         workers=workers,
                                                         chi2_threshold=calibration['chi2_threshold'])
    finally:
        if test_slice_path.exists():
            test_slice_path.unlink()

    return {
        'calibration': calibration,
        'legacy_default': legacy_default,
        'legacy_calibrated': legacy_calibrated,
        'kalman_default': kalman_default,
        'kalman_calibrated': kalman_calibrated,
    }
