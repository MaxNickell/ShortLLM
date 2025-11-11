"""
Dataset builder module for creating balanced graph reasoning datasets.

This module handles the generation of graph examples with uniform distribution
of shortest path lengths, serialization, and train/val/test splitting.
"""

import json
import random
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from collections import defaultdict, Counter
import logging

import networkx as nx

from data.graph_generator import GraphGenerator
from data.config import DatasetConfig, GraphBucketConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetBuilder:
    """
    Builds a balanced dataset of graph reasoning examples.
    
    New approach:
    1. Generate N connected graphs per bucket
    2. Extract all source-destination pairs from each graph
    3. Group pairs by path length
    4. Sample uniformly to create final dataset
    
    This is more efficient than generating graphs on-the-fly.
    """
    
    def __init__(self, config: DatasetConfig):
        """
        Initialize the dataset builder.
        
        Args:
            config: Dataset configuration object.
        """
        self.config = config
        self.generator = GraphGenerator(random_seed=config.random_seed)
        self.examples = []
        
        # Set random seed for reproducibility
        if config.random_seed is not None:
            random.seed(config.random_seed)
    
    def build_dataset(self) -> List[Dict]:
        """
        Build the complete dataset across all graph size buckets.
        
        Returns:
            List of all generated examples as dictionaries.
        """
        logger.info(f"Starting dataset generation with config:")
        logger.info(f"  Total examples: {self.config.total_examples:,}")
        logger.info(f"  Buckets: {len(self.config.buckets)}")
        logger.info(f"  Graphs per bucket: {self.config.num_graphs_per_bucket:,}")
        
        self.examples = []
        
        for bucket in self.config.buckets:
            logger.info(f"\nGenerating bucket: {bucket}")
            bucket_examples = self._generate_bucket(bucket)
            self.examples.extend(bucket_examples)
            logger.info(f"  Completed: {len(bucket_examples):,} examples generated")
        
        logger.info(f"\nTotal examples generated: {len(self.examples):,}")
        return self.examples
    
    def _generate_bucket(self, bucket: GraphBucketConfig) -> List[Dict]:
        """
        Generate examples for a single graph size bucket.
        
        New approach:
        1. Generate num_graphs_per_bucket connected graphs
        2. Extract all source-destination pairs from each graph
        3. Group pairs by path length
        4. Sample uniformly to reach target number of examples
        
        Args:
            bucket: Configuration for this bucket.
        
        Returns:
            List of examples for this bucket.
        """
        logger.info(f"  Phase 1: Generating {self.config.num_graphs_per_bucket:,} connected graphs...")
        
        # Generate graphs
        graphs = []
        for i in range(self.config.num_graphs_per_bucket):
            num_nodes = random.randint(bucket.min_nodes, bucket.max_nodes)
            
            try:
                graph = self.generator.generate_connected_graph(
                    num_nodes,
                    bucket.erdos_renyi_p,
                    max_attempts=self.config.max_connection_attempts
                )
                graphs.append(graph)
                
                if (i + 1) % self.config.log_every_n_graphs == 0:
                    logger.info(f"    Generated {i + 1:,}/{self.config.num_graphs_per_bucket:,} graphs")
            
            except RuntimeError as e:
                logger.warning(f"    Failed to generate connected graph: {e}")
                continue
        
        logger.info(f"  Generated {len(graphs):,} connected graphs")
        
        # Phase 2: Extract all source-destination pairs grouped by path length
        logger.info(f"  Phase 2: Extracting source-destination pairs from graphs...")
        
        pairs_by_length = defaultdict(list)
        total_pairs = 0
        
        for graph_idx, graph in enumerate(graphs):
            # Relabel nodes randomly
            graph, node_mapping = self.generator.relabel_nodes_randomly(graph)
            
            # Get all nodes
            nodes = list(graph.nodes())
            
            # Skip graphs with only 1 node (no pairs possible)
            if len(nodes) < 2:
                continue
            
            # Compute all-pairs shortest paths
            all_paths = self.generator.compute_all_pairs_shortest_paths(graph)
            
            # Extract all source-destination pairs
            for source in all_paths:
                for destination in all_paths[source]:
                    if source != destination:  # Exclude trivial paths
                        path = all_paths[source][destination]
                        path_length = self.generator.get_path_length(path)
                        
                        # Store the pair with the graph
                        pair_data = {
                            'graph': graph,
                            'source': source,
                            'destination': destination,
                            'path': path,
                            'path_length': path_length,
                            'bucket': bucket.name
                        }
                        pairs_by_length[path_length].append(pair_data)
                        total_pairs += 1
            
            if (graph_idx + 1) % self.config.log_every_n_graphs == 0:
                logger.info(f"    Processed {graph_idx + 1:,}/{len(graphs):,} graphs, {total_pairs:,} pairs extracted")
        
        logger.info(f"  Extracted {total_pairs:,} total source-destination pairs")
        
        # Log distribution of path lengths
        logger.info(f"  Path length distribution in graph pool:")
        for length in sorted(pairs_by_length.keys()):
            count = len(pairs_by_length[length])
            percentage = (count / total_pairs) * 100 if total_pairs > 0 else 0
            logger.info(f"    Length {length}: {count:,} pairs ({percentage:.1f}%)")
        
        # Phase 3: Sample uniformly from path lengths
        logger.info(f"  Phase 3: Sampling {bucket.num_examples:,} examples with uniform path length distribution...")
        
        # Get available path lengths
        available_lengths = sorted(pairs_by_length.keys())
        
        if not available_lengths:
            logger.warning(f"  No pairs available for bucket {bucket.name}")
            return []
        
        # Calculate target per length
        target_per_length = bucket.num_examples // len(available_lengths)
        remainder = bucket.num_examples % len(available_lengths)
        
        logger.info(f"  Available path lengths: {available_lengths}")
        logger.info(f"  Target per length: ~{target_per_length:,}")
        
        # Sample from each path length
        bucket_examples = []
        for idx, length in enumerate(available_lengths):
            available_pairs = pairs_by_length[length]
            
            # Add remainder to last length to hit exact target
            target = target_per_length + (remainder if idx == len(available_lengths) - 1 else 0)
            
            # Sample (with replacement if necessary)
            if len(available_pairs) >= target:
                sampled_pairs = random.sample(available_pairs, target)
            else:
                # If not enough pairs, use all available and sample with replacement
                sampled_pairs = random.choices(available_pairs, k=target)
                if len(available_pairs) < target:
                    logger.warning(f"    Path length {length}: only {len(available_pairs):,} pairs available, needed {target:,} (sampling with replacement)")
            
            # Serialize examples
            for pair_data in sampled_pairs:
                example = self._serialize_example(
                    pair_data['graph'],
                    pair_data['source'],
                    pair_data['destination'],
                    pair_data['path'],
                    pair_data['bucket']
                )
                bucket_examples.append(example)
        
        # Shuffle to mix path lengths
        random.shuffle(bucket_examples)
        
        # Log final distribution
        logger.info(f"  Final path length distribution:")
        final_counts = Counter(ex['path_length'] for ex in bucket_examples)
        for length in sorted(final_counts.keys()):
            count = final_counts[length]
            percentage = (count / len(bucket_examples)) * 100 if bucket_examples else 0
            logger.info(f"    Length {length}: {count:,} ({percentage:.1f}%)")
        
        return bucket_examples
    
    def _serialize_example(
        self,
        graph: nx.Graph,
        source: int,
        destination: int,
        path: List[int],
        bucket_name: str
    ) -> Dict:
        """
        Serialize a graph example into the required format.
        
        Format: edges, origin, destination, path (token sequence)
        
        Args:
            graph: The graph.
            source: Source node.
            destination: Destination node.
            path: Shortest path from source to destination.
            bucket_name: Name of the bucket (small/medium/large).
        
        Returns:
            Dictionary containing the serialized example.
        """
        # Get randomized edge list
        edges = self.generator.get_edge_list_randomized(graph)
        
        # Create serialized format
        example = {
            "edges": edges,
            "source": source,
            "destination": destination,
            "path": path,
            "path_length": len(path) - 1,
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "bucket": bucket_name,
        }
        
        return example
    
    def save_dataset(self, output_dir: Optional[str] = None):
        """
        Save the dataset split into train/val/test sets as JSONL files.
        
        Args:
            output_dir: Directory to save files. Uses config default if None.
        """
        if output_dir is None:
            output_dir = self.config.output_dir
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Shuffle examples before splitting
        shuffled_examples = self.examples.copy()
        random.shuffle(shuffled_examples)
        
        # Calculate split indices
        split_sizes = self.config.get_split_sizes()
        train_end = split_sizes["train"]
        val_end = train_end + split_sizes["val"]
        
        # Split the data
        train_data = shuffled_examples[:train_end]
        val_data = shuffled_examples[train_end:val_end]
        test_data = shuffled_examples[val_end:]
        
        # Save each split
        splits = {
            "train": (train_data, self.config.train_file),
            "val": (val_data, self.config.val_file),
            "test": (test_data, self.config.test_file),
        }
        
        logger.info(f"\nSaving dataset to {output_path}")
        
        for split_name, (data, filename) in splits.items():
            filepath = output_path / filename
            self._save_jsonl(data, filepath)
            logger.info(f"  {split_name}: {len(data):,} examples -> {filepath}")
            
            # Log statistics for this split
            self._log_split_statistics(data, split_name)
    
    def _save_jsonl(self, data: List[Dict], filepath: Path):
        """
        Save data as JSONL (JSON Lines) format.
        
        Args:
            data: List of examples.
            filepath: Path to save file.
        """
        with open(filepath, 'w') as f:
            for example in data:
                json.dump(example, f)
                f.write('\n')
    
    def _log_split_statistics(self, data: List[Dict], split_name: str):
        """
        Log statistics about a data split.
        
        Args:
            data: List of examples in the split.
            split_name: Name of the split (train/val/test).
        """
        if not data:
            return
        
        # Bucket distribution
        bucket_counts = Counter(ex["bucket"] for ex in data)
        logger.info(f"    {split_name} bucket distribution:")
        for bucket, count in sorted(bucket_counts.items()):
            percentage = (count / len(data)) * 100
            logger.info(f"      {bucket}: {count:,} ({percentage:.1f}%)")
        
        # Path length distribution
        path_length_counts = Counter(ex["path_length"] for ex in data)
        logger.info(f"    {split_name} path length range: "
                   f"{min(path_length_counts.keys())}-{max(path_length_counts.keys())}")
