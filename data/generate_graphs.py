"""
Main script for generating the ShortLLM graph reasoning dataset.

This script generates a balanced dataset of graph examples for training
language models on shortest path reasoning tasks.

Usage:
    python generate_graphs.py [options]

Examples:
    # Generate full dataset with default settings
    python generate_graphs.py

    # Generate a small test dataset
    python generate_graphs.py --test-run --num-examples 1000

    # Custom output directory
    python generate_graphs.py --output-dir ./my_dataset
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

from data.config import DatasetConfig, GraphBucketConfig
from data.dataset_builder import DatasetBuilder


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Generate graph reasoning dataset for ShortLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate full dataset (1.2M examples)
  python generate_graphs.py

  # Quick test run with 1000 examples
  python generate_graphs.py --test-run --num-examples 1000

  # Custom configuration
  python generate_graphs.py --num-examples 500000 --output-dir ./custom_data
        """
    )
    
    # Dataset size arguments
    parser.add_argument(
        '--num-examples',
        type=int,
        default=1_200_000,
        help='Total number of examples to generate (default: 1,200,000)'
    )
    
    parser.add_argument(
        '--num-graphs',
        type=int,
        default=100_000,
        help='Number of graphs to generate per bucket (default: 100,000)'
    )
    
    parser.add_argument(
        '--test-run',
        action='store_true',
        help='Run in test mode with reduced examples (1000 total, 100 graphs)'
    )
    
    # Output arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/output',
        help='Directory to save generated dataset (default: data/output)'
    )
    
    # Split ratios
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Proportion of data for training (default: 0.8)'
    )
    
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Proportion of data for validation (default: 0.1)'
    )
    
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.1,
        help='Proportion of data for testing (default: 0.1)'
    )
    
    # Random seed
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    # Logging
    parser.add_argument(
        '--log-every',
        type=int,
        default=10_000,
        help='Log progress every N examples (default: 10,000)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def create_config_from_args(args: argparse.Namespace) -> DatasetConfig:
    """
    Create a DatasetConfig from command line arguments.
    
    Args:
        args: Parsed command line arguments.
    
    Returns:
        Configured DatasetConfig object.
    """
    # Handle test run
    if args.test_run:
        num_examples = 1000
        num_graphs = 100
        logger.info("Running in TEST MODE with 1,000 examples from 100 graphs")
    else:
        num_examples = args.num_examples
        num_graphs = args.num_graphs
    
    # Create bucket configs - ensure they sum to num_examples
    examples_per_bucket = num_examples // 3
    # Add remaining examples to last bucket to ensure exact total
    remainder = num_examples - (examples_per_bucket * 3)
    
    buckets = (
        GraphBucketConfig(
            name="small",
            min_nodes=1,
            max_nodes=5,
            num_examples=examples_per_bucket,
            erdos_renyi_p=0.5  # Fixed density p=0.5
        ),
        GraphBucketConfig(
            name="medium",
            min_nodes=6,
            max_nodes=15,
            num_examples=examples_per_bucket,
            erdos_renyi_p=0.5  # Fixed density p=0.5
        ),
        GraphBucketConfig(
            name="large",
            min_nodes=16,
            max_nodes=25,
            num_examples=examples_per_bucket + remainder,
            erdos_renyi_p=0.5  # Fixed density p=0.5
        ),
    )
    
    # Create config
    config = DatasetConfig(
        total_examples=num_examples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        buckets=buckets,
        output_dir=args.output_dir,
        random_seed=args.seed,
        num_graphs_per_bucket=num_graphs,
        log_every_n_examples=args.log_every,
        log_every_n_graphs=max(1000, num_graphs // 10)  # Log at 10% intervals
    )
    
    return config


def main():
    """Main entry point for dataset generation."""
    # Parse arguments
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Print header
    logger.info("=" * 70)
    logger.info("ShortLLM Graph Reasoning Dataset Generator")
    logger.info("=" * 70)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    
    # Create configuration
    try:
        config = create_config_from_args(args)
    except AssertionError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    
    # Log configuration
    logger.info("Configuration:")
    logger.info(f"  Total examples: {config.total_examples:,}")
    logger.info(f"  Graphs per bucket: {config.num_graphs_per_bucket:,}")
    logger.info(f"  Edge density (p): 0.5 for all buckets")
    logger.info(f"  Random seed: {config.random_seed}")
    logger.info(f"  Output directory: {config.output_dir}")
    logger.info(f"  Split ratios: Train={config.train_ratio}, Val={config.val_ratio}, Test={config.test_ratio}")
    split_sizes = config.get_split_sizes()
    logger.info(f"  Split sizes: Train={split_sizes['train']:,}, Val={split_sizes['val']:,}, Test={split_sizes['test']:,}")
    logger.info("")
    logger.info("Graph size buckets:")
    for bucket in config.buckets:
        logger.info(f"  - {bucket}")
    logger.info("")
    
    # Create output directory
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize dataset builder
        logger.info("Initializing dataset builder...")
        builder = DatasetBuilder(config)
        
        # Generate dataset
        logger.info("=" * 70)
        logger.info("Starting dataset generation...")
        logger.info("=" * 70)
        start_time = datetime.now()
        
        builder.build_dataset()
        
        generation_time = (datetime.now() - start_time).total_seconds()
        logger.info("=" * 70)
        logger.info(f"Dataset generation completed in {generation_time:.2f} seconds")
        logger.info(f"Average: {len(builder.examples) / generation_time:.2f} examples/second")
        logger.info("=" * 70)
        logger.info("")
        
        # Save dataset
        logger.info("Saving dataset to disk...")
        builder.save_dataset()
        
        # Print summary
        logger.info("")
        logger.info("=" * 70)
        logger.info("Dataset Generation Complete!")
        logger.info("=" * 70)
        logger.info(f"Total examples: {len(builder.examples):,}")
        logger.info(f"Output directory: {output_path.absolute()}")
        logger.info(f"Files created:")
        logger.info(f"  - {config.train_file} ({split_sizes['train']:,} examples)")
        logger.info(f"  - {config.val_file} ({split_sizes['val']:,} examples)")
        logger.info(f"  - {config.test_file} ({split_sizes['test']:,} examples)")
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 70)
        
    except KeyboardInterrupt:
        logger.warning("\nGeneration interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during generation: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

