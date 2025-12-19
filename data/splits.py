"""
Centralized data splitting for consistent train/val/test splits.

Both pretraining and RL must use this to ensure they see the same data splits.
"""

import json
import random
from math import floor
from typing import List, Dict, Any, Tuple, DefaultDict

from src.config import DataConfig


def load_rows(path: str) -> List[Dict[str, Any]]:
    """Load all rows from a JSONL file."""
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_split_indices(
    n: int,
    config: DataConfig,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Get train/val/test indices for n examples.

    Uses config.seed to ensure reproducibility and consistency
    across pretraining and RL.
    """
    rng = random.Random(config.seed)
    indices = list(range(n))
    rng.shuffle(indices)

    n_train = floor(config.train_frac * n)
    n_val = floor(config.val_frac * n)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx


def get_splits(
    path: str,
    config: DataConfig,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Load data and split into train/val/test.

    This is the single source of truth for data splitting.
    Both pretraining and RL should call this function.

    Returns:
        (train_rows, val_rows, test_rows)
    """
    rows = load_rows(path)
    train_idx, val_idx, test_idx = get_split_indices(len(rows), config)

    train_rows = [rows[i] for i in train_idx]
    val_rows = [rows[i] for i in val_idx]
    test_rows = [rows[i] for i in test_idx]

    print(f"Split sizes - Total: {len(rows)}, Train: {len(train_rows)}, Val: {len(val_rows)}, Test: {len(test_rows)}")

    return train_rows, val_rows, test_rows


def get_stratified_subset(
    rows: List[Dict[str, Any]],
    n: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Get a stratified subset of rows, uniformly distributed across
    (num_nodes, shortest_path_length) strata.

    Args:
        rows: List of data rows
        n: Target subset size
        seed: Random seed for reproducibility

    Returns:
        Stratified subset of approximately n rows
    """
    from collections import defaultdict

    rng = random.Random(seed)

    # Group by (num_nodes, shortest_path_length)
    strata: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)
    for row in rows:
        key = (row["num_nodes"], row["shortest_path_length"])
        strata[key].append(row)

    # Calculate samples per stratum (uniform across strata)
    num_strata = len(strata)
    base_per_stratum = n // num_strata
    remainder = n % num_strata

    # Sample from each stratum
    subset = []
    stratum_keys = sorted(strata.keys())
    rng.shuffle(stratum_keys)  # Randomize which strata get extra samples

    for i, key in enumerate(stratum_keys):
        stratum_rows = strata[key]
        # First 'remainder' strata get one extra sample
        target = base_per_stratum + (1 if i < remainder else 0)
        # Sample min(target, available) from this stratum
        sample_size = min(target, len(stratum_rows))
        subset.extend(rng.sample(stratum_rows, sample_size))

    rng.shuffle(subset)
    return subset
