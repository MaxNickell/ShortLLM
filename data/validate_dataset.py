"""
Simple script to validate the generated dataset.

This script checks that the generated examples are well-formed and that
paths are indeed shortest paths.
"""

import json
import sys
from pathlib import Path
import networkx as nx


def validate_example(example: dict) -> bool:
    """
    Validate a single example.
    
    Args:
        example: The example dictionary.
    
    Returns:
        True if valid, False otherwise.
    """
    # Check required fields
    required_fields = ["edges", "source", "destination", "path", "path_length", "num_nodes", "num_edges", "bucket"]
    for field in required_fields:
        if field not in example:
            print(f"Missing field: {field}")
            return False
    
    # Build graph from edges
    graph = nx.Graph()
    graph.add_edges_from(example["edges"])
    
    # Verify graph properties
    if graph.number_of_nodes() != example["num_nodes"]:
        print(f"Node count mismatch: expected {example['num_nodes']}, got {graph.number_of_nodes()}")
        return False
    
    if graph.number_of_edges() != example["num_edges"]:
        print(f"Edge count mismatch: expected {example['num_edges']}, got {graph.number_of_edges()}")
        return False
    
    # Verify path
    path = example["path"]
    source = example["source"]
    destination = example["destination"]
    
    if path[0] != source:
        print(f"Path doesn't start with source: {path[0]} != {source}")
        return False
    
    if path[-1] != destination:
        print(f"Path doesn't end with destination: {path[-1]} != {destination}")
        return False
    
    # Verify path length
    if len(path) - 1 != example["path_length"]:
        print(f"Path length mismatch: expected {example['path_length']}, got {len(path) - 1}")
        return False
    
    # Verify path is valid (consecutive nodes are connected)
    for i in range(len(path) - 1):
        if not graph.has_edge(path[i], path[i + 1]):
            print(f"Invalid path: no edge between {path[i]} and {path[i + 1]}")
            return False
    
    # Verify path is shortest
    try:
        shortest_path = nx.shortest_path(graph, source, destination)
        shortest_length = len(shortest_path) - 1
        if shortest_length != example["path_length"]:
            print(f"Path is not shortest: expected length {shortest_length}, got {example['path_length']}")
            print(f"Correct path: {shortest_path}")
            print(f"Given path: {path}")
            return False
    except nx.NetworkXNoPath:
        print(f"No path exists between {source} and {destination}")
        return False
    
    return True


def validate_file(filepath: Path, max_examples: int = 100) -> tuple[int, int]:
    """
    Validate examples in a file.
    
    Args:
        filepath: Path to JSONL file.
        max_examples: Maximum number of examples to validate (None for all).
    
    Returns:
        Tuple of (valid_count, total_count).
    """
    valid_count = 0
    total_count = 0
    
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if max_examples and i >= max_examples:
                break
            
            example = json.loads(line)
            total_count += 1
            
            if validate_example(example):
                valid_count += 1
            else:
                print(f"Invalid example at line {i + 1}")
    
    return valid_count, total_count


def main():
    """Main validation routine."""
    output_dir = Path("data/output")
    
    if not output_dir.exists():
        print(f"Output directory not found: {output_dir}")
        sys.exit(1)
    
    files = ["train.jsonl", "val.jsonl", "test.jsonl"]
    
    print("=" * 70)
    print("Dataset Validation Report")
    print("=" * 70)
    print()
    
    all_valid = True
    
    for filename in files:
        filepath = output_dir / filename
        if not filepath.exists():
            print(f"File not found: {filepath}")
            all_valid = False
            continue
        
        print(f"Validating {filename}...")
        valid_count, total_count = validate_file(filepath, max_examples=100)
        
        if valid_count == total_count:
            print(f"  ✓ All {valid_count}/{total_count} examples are valid")
        else:
            print(f"  ✗ Only {valid_count}/{total_count} examples are valid")
            all_valid = False
        print()
    
    print("=" * 70)
    if all_valid:
        print("✓ Dataset validation passed!")
    else:
        print("✗ Dataset validation failed!")
    print("=" * 70)
    
    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())

