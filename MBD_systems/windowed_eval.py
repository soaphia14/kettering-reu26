"""Evaluates the LegacyChecks pipeline as a window-level attacker detector against the
relative-feature CSVs (RecieverID/SenderID/rcvTime/RelX/RelY/MssgCount/dVx/dVy/dAx/dAy/AttkType),
using the same 10-message / stride-5 windowing as utils.functions.get_windowed_data - without
ever holding a full (very large) CSV in memory at once.

Each window is 10 consecutive messages from the same sender. A window's ground truth is
"attacker" only if every message in it carries the given attacker_code (mirrors
get_windowed_data's per-window label). The IDS's prediction for a window is "attacker" if any
message in it trips a check.
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

from data_structures import Coord, Parameters
from legacy_checks import LegacyChecks

RAW_CSV_CHUNKSIZE = 200_000
WINDOW_SIZE = 10
STRIDE = 5

CHECK_NAMES = [
    'range_plausibility',
    'speed_plausibility',
    'sudden_appearance',
    'position_consistency',
    'speed_consistency',
    'position_speed_consistency',
]


def _empty_check_counts() -> Dict[str, int]:
    return {name: 0 for name in CHECK_NAMES}


def split_csv_by_sender(csv_path: Path, tmp_dir: Path, chunksize: int = RAW_CSV_CHUNKSIZE) -> List[int]:
    """Streams a large relative-feature CSV in chunks and buckets rows into one temp CSV per
    SenderID (windows are built per-sender), so the full file is never held in memory at once."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    written = set()

    for chunk in pd.read_csv(csv_path, chunksize=chunksize, index_col=0):
        for sender_id, group in chunk.groupby('SenderID'):
            out_path = tmp_dir / f"sender_{sender_id}.csv"
            group.to_csv(out_path, mode='a', header=sender_id not in written, index=False)
            written.add(sender_id)

    return sorted(written)


def split_csv_by_sender_receiver(csv_path: Path, tmp_dir: Path,
                                 chunksize: int = RAW_CSV_CHUNKSIZE) -> List[tuple]:
    """Streams a large relative-feature CSV in chunks and buckets rows into one temp CSV per
    (SenderID, RecieverID) pair, so the full file is never held in memory at once. Unlike
    split_csv_by_sender, every row in a resulting file is the same sender AS SEEN BY the same
    receiver - RelX/RelY/dVx/dVy/dAx/dAy stay in one consistent relative frame the whole way
    through, instead of jumping between whichever receivers happened to hear that sender."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    written = set()

    for chunk in pd.read_csv(csv_path, chunksize=chunksize, index_col=0):
        for (sender_id, receiver_id), group in chunk.groupby(['SenderID', 'RecieverID']):
            out_path = tmp_dir / f"pair_{sender_id}_{receiver_id}.csv"
            key = (sender_id, receiver_id)
            group.to_csv(out_path, mode='a', header=key not in written, index=False)
            written.add(key)

    return sorted(written)


def _row_state(row: pd.Series):
    """The receiver is treated as the coordinate origin; the sender's position/speed are
    expressed relative to it (RelX/RelY, and speed as the magnitude of dVx/dVy), since this
    format carries no absolute position or heading."""
    pos = Coord(x=row['RelX'], y=row['RelY'], z=0.0)
    spd = float(np.hypot(row['dVx'], row['dVy']))
    return pos, spd


def evaluate_window(window_df: pd.DataFrame, checks: LegacyChecks, attacker_code: int) -> Dict[str, Any]:
    ground_truth = 1 if (window_df['AttkType'] == attacker_code).all() else 0

    receiver_pos = Coord(0.0, 0.0, 0.0)
    prev_pos = prev_spd = prev_time = None
    flagged = False
    check_counts = _empty_check_counts()

    for i, (_, row) in enumerate(window_df.iterrows()):
        pos, spd = _row_state(row)

        if checks.range_plausibility_check(pos, receiver_pos) == 0:
            flagged = True
            check_counts['range_plausibility'] += 1
        if checks.speed_plausibility_check(spd) == 0:
            flagged = True
            check_counts['speed_plausibility'] += 1
        if i == 0 and checks.sudden_appearance_check(pos, receiver_pos) == 0:
            flagged = True
            check_counts['sudden_appearance'] += 1

        if prev_pos is not None:
            delta_time = row['rcvTime'] - prev_time
            if delta_time > 0:
                if checks.position_consistency_check(pos, prev_pos, delta_time) == 0:
                    flagged = True
                    check_counts['position_consistency'] += 1
                if checks.speed_consistency_check(spd, prev_spd, delta_time) == 0:
                    flagged = True
                    check_counts['speed_consistency'] += 1
                if checks.position_speed_consistency_check(pos, prev_pos, spd, prev_spd, delta_time) == 0:
                    flagged = True
                    check_counts['position_speed_consistency'] += 1

        prev_pos, prev_spd, prev_time = pos, spd, row['rcvTime']

    return {'ground_truth': ground_truth, 'prediction': 1 if flagged else 0, 'check_counts': check_counts}


def process_windowed_sender(df: pd.DataFrame, params: Parameters, attacker_code: int,
                            window_size: int = WINDOW_SIZE, stride: int = STRIDE) -> Dict[str, int]:
    """One sender's worth of rows -> sliding windows -> window-level tp/tn/fp/fn counts."""
    checks = LegacyChecks(params)
    df = df.sort_values('rcvTime').reset_index(drop=True)

    tp = tn = fp = fn = 0
    check_counts = _empty_check_counts()
    n = len(df)
    index = 0
    while index < n - window_size:
        window_df = df.iloc[index:index + window_size]
        outcome = evaluate_window(window_df, checks, attacker_code)

        if outcome['ground_truth'] == 1 and outcome['prediction'] == 1:
            tp += 1
        elif outcome['ground_truth'] == 0 and outcome['prediction'] == 0:
            tn += 1
        elif outcome['ground_truth'] == 0 and outcome['prediction'] == 1:
            fp += 1
        else:
            fn += 1

        for name, count in outcome['check_counts'].items():
            check_counts[name] += count

        index += stride

    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'check_counts': check_counts}


def worker_evaluate_sender(sender_file: str, params_dict: Dict[str, Any], attacker_code: int,
                           window_size: int, stride: int, source_file: str):
    try:
        params = Parameters(**params_dict)
        df = pd.read_csv(sender_file)
        result = process_windowed_sender(df, params, attacker_code, window_size, stride)
        return {'status': 'ok', 'metrics': result, 'source': source_file}
    except Exception as e:
        return {'status': 'error', 'error': str(e), 'source': source_file}


def aggregate_window_metrics(scenario_stats: List[dict]) -> dict:
    total_tp = sum(s.get('tp', 0) for s in scenario_stats)
    total_tn = sum(s.get('tn', 0) for s in scenario_stats)
    total_fp = sum(s.get('fp', 0) for s in scenario_stats)
    total_fn = sum(s.get('fn', 0) for s in scenario_stats)

    total_windows = total_tp + total_tn + total_fp + total_fn

    check_counts = _empty_check_counts()
    for s in scenario_stats:
        for name, count in s.get('check_counts', {}).items():
            check_counts[name] += count

    metrics = {
        'tp': total_tp, 'tn': total_tn, 'fp': total_fp, 'fn': total_fn,
        'accuracy': (total_tp + total_tn) / total_windows if total_windows > 0 else 0,
        'precision': total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0,
        'recall': total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0,
        'check_counts': check_counts,
    }
    metrics['f1'] = (2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])
                     if (metrics['precision'] + metrics['recall']) > 0 else 0)
    return metrics


def evaluate_windowed_ids(input_csv: str, attacker_code: int = 3, window_size: int = WINDOW_SIZE,
                          stride: int = STRIDE, workers: int = None, chunksize: int = RAW_CSV_CHUNKSIZE,
                          cleanup: bool = True, params: Parameters = None) -> dict:
    """Splits input_csv by SenderID (streamed, chunked), evaluates every 10-message/stride-5
    window per sender in parallel, and returns window-level tp/tn/fp/fn/accuracy/precision/
    recall/f1. attacker_code should match the file's AttkType (e.g. 1=ConstPos, 3=RandomPos,
    7=RandomSpeed). Pass params to override the default (e.g. dataset-calibrated) thresholds."""
    csv_path = Path(input_csv)
    workers = workers or os.cpu_count() or 4
    params_dict = vars(params) if params is not None else vars(Parameters())

    tmp_dir = csv_path.parent / f"_sender_split_{csv_path.stem}"
    print(f"Splitting {csv_path} by SenderID into {tmp_dir} ...", file=sys.stderr)
    sender_ids = split_csv_by_sender(csv_path, tmp_dir, chunksize)
    print(f"Found {len(sender_ids)} senders, evaluating windows with {workers} workers...", file=sys.stderr)

    scenario_stats = []
    count = 0
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for sender_id in sender_ids:
                sender_file = tmp_dir / f"sender_{sender_id}.csv"
                f = executor.submit(worker_evaluate_sender, str(sender_file), params_dict,
                                    attacker_code, window_size, stride, str(sender_id))
                futures[f] = sender_id

            for future in as_completed(futures):
                source = futures[future]
                res = future.result()
                if res.get('status') == 'ok':
                    scenario_stats.append(res.get('metrics', {}))
                else:
                    print(f"[ERROR] sender {source}: {res.get('error')}", file=sys.stderr)
                count += 1
                if count % 500 == 0 or count == len(sender_ids):
                    print(f"Processed {count}/{len(sender_ids)} senders", file=sys.stderr)
    finally:
        if cleanup and tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    return aggregate_window_metrics(scenario_stats)


def evaluate_windowed_ids_by_pair(input_csv: str, attacker_code: int = 3, window_size: int = WINDOW_SIZE,
                                  stride: int = STRIDE, workers: int = None,
                                  chunksize: int = RAW_CSV_CHUNKSIZE, cleanup: bool = True,
                                  params: Parameters = None) -> dict:
    """Same as evaluate_windowed_ids, but windows are built per (SenderID, RecieverID) pair
    instead of per SenderID alone - so every window is a continuous trajectory in one
    consistent relative frame (one sender, as seen by one fixed receiver), rather than
    potentially interleaving messages the sender sent to several different receivers.
    Expect far more (and shorter) groups than the sender-only version, since a given
    sender/receiver pair only exchanges messages while they're within range of each other.
    Pass params to override the default (e.g. dataset-calibrated) thresholds."""
    csv_path = Path(input_csv)
    workers = workers or os.cpu_count() or 4
    params_dict = vars(params) if params is not None else vars(Parameters())

    tmp_dir = csv_path.parent / f"_pair_split_{csv_path.stem}"
    print(f"Splitting {csv_path} by (SenderID, RecieverID) into {tmp_dir} ...", file=sys.stderr)
    pairs = split_csv_by_sender_receiver(csv_path, tmp_dir, chunksize)
    print(f"Found {len(pairs)} sender/receiver pairs, evaluating windows with {workers} workers...",
          file=sys.stderr)

    scenario_stats = []
    count = 0
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for sender_id, receiver_id in pairs:
                pair_file = tmp_dir / f"pair_{sender_id}_{receiver_id}.csv"
                source_name = f"{sender_id}_{receiver_id}"
                f = executor.submit(worker_evaluate_sender, str(pair_file), params_dict,
                                    attacker_code, window_size, stride, source_name)
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

    return aggregate_window_metrics(scenario_stats)
