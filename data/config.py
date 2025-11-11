"""
Configuration file for dataset generation.

Contains all hyperparameters and constants for generating the graph dataset.
"""

from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class GraphBucketConfig:
    """Configuration for a single graph size bucket."""
    name: str
    min_nodes: int
    max_nodes: int
    num_examples: int
    erdos_renyi_p: float = 0.5  # Edge probability for Erdős-Rényi model
    
    def __str__(self) -> str:
        return f"{self.name} ({self.min_nodes}-{self.max_nodes} nodes, {self.num_examples:,} examples)"


@dataclass
class DatasetConfig:
    """Configuration for the entire dataset generation process."""
    
    # Total dataset size
    total_examples: int = 1_200_000
    
    # Train/Val/Test split ratios
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Graph size buckets
    buckets: Tuple[GraphBucketConfig, ...] = (
        GraphBucketConfig(
            name="small",
            min_nodes=1,
            max_nodes=5,
            num_examples=400_000,
            erdos_renyi_p=0.5  # Fixed density p=0.5 for all buckets
        ),
        GraphBucketConfig(
            name="medium",
            min_nodes=6,
            max_nodes=15,
            num_examples=400_000,
            erdos_renyi_p=0.5  # Fixed density p=0.5 for all buckets
        ),
        GraphBucketConfig(
            name="large",
            min_nodes=16,
            max_nodes=25,
            num_examples=400_000,
            erdos_renyi_p=0.5  # Fixed density p=0.5 for all buckets
        ),
    )
    
    # Output paths
    output_dir: str = "data/output"
    train_file: str = "train.jsonl"
    val_file: str = "val.jsonl"
    test_file: str = "test.jsonl"
    
    # Generation parameters
    max_connection_attempts: int = 100  # Max attempts to generate connected graph
    random_seed: int = 42
    num_graphs_per_bucket: int = 100_000  # Number of unique graphs to generate per bucket
    
    # Logging
    log_every_n_examples: int = 10_000
    log_every_n_graphs: int = 10_000  # Log progress during graph generation
    
    def __post_init__(self):
        """Validate configuration."""
        assert abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"
        assert sum(b.num_examples for b in self.buckets) == self.total_examples, \
            "Bucket examples must sum to total_examples"
    
    def get_split_sizes(self) -> Dict[str, int]:
        """Calculate the number of examples in each split."""
        return {
            "train": int(self.total_examples * self.train_ratio),
            "val": int(self.total_examples * self.val_ratio),
            "test": int(self.total_examples * self.test_ratio),
        }


# Default configuration instance
DEFAULT_CONFIG = DatasetConfig()

