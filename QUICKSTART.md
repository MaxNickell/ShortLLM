# ShortLLM Quick Start Guide

This guide will help you get started with generating the graph reasoning dataset.

## Setup

### 1. Create the Conda Environment

If you haven't already created the ShortLLM conda environment:

```bash
conda env create -f environment.yml
```

### 2. Activate the Environment

```bash
conda activate ShortLLM
```

## Generating Data

### Test Run (1,000 examples)

For testing and development, generate a small dataset:

```bash
python -m data.generate_graphs --test-run
```

This will create:
- `data/output/train.jsonl` (800 examples)
- `data/output/val.jsonl` (100 examples)
- `data/output/test.jsonl` (100 examples)

Generation time: ~1 second

### Full Dataset (1.2M examples)

To generate the complete dataset:

```bash
python -m data.generate_graphs
```

This will create:
- `data/output/train.jsonl` (960,000 examples)
- `data/output/val.jsonl` (120,000 examples)
- `data/output/test.jsonl` (120,000 examples)

Estimated generation time: ~7-10 minutes (depending on your system)

## Validating Data

To verify the generated dataset is correct:

```bash
python -m data.validate_dataset
```

This will check that:
- All required fields are present
- Paths are valid (consecutive nodes are connected)
- Paths are indeed shortest paths
- Graph properties match the serialized data

## Example Output Format

Each line in the JSONL files contains one example:

```json
{
  "edges": [[0, 1], [1, 2], [0, 3]],
  "source": 0,
  "destination": 2,
  "path": [0, 1, 2],
  "path_length": 2,
  "num_nodes": 4,
  "num_edges": 3,
  "bucket": "small"
}
```

## Command Line Options

### Dataset Size

```bash
# Generate 500,000 examples
python -m data.generate_graphs --num-examples 500000
```

### Custom Output Location

```bash
# Save to a different directory
python -m data.generate_graphs --output-dir ./my_custom_dataset
```

### Custom Data Splits

```bash
# 70% train, 15% val, 15% test
python -m data.generate_graphs --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
```

### Random Seed

```bash
# For reproducibility with different seed
python -m data.generate_graphs --seed 12345
```

### Verbose Logging

```bash
# See detailed debug information
python -m data.generate_graphs --verbose
```

## Loading Data in Python

Here's how to load and use the generated data:

```python
import json

# Load training data
with open('data/output/train.jsonl', 'r') as f:
    train_examples = [json.loads(line) for line in f]

# Access an example
example = train_examples[0]
print(f"Graph has {example['num_nodes']} nodes and {example['num_edges']} edges")
print(f"Shortest path from {example['source']} to {example['destination']}: {example['path']}")
print(f"Path length: {example['path_length']}")
```

## Project Structure

```
ShortLLM/
├── data/
│   ├── config.py              # Configuration and hyperparameters
│   ├── graph_generator.py     # Graph generation using Erdős-Rényi model
│   ├── dataset_builder.py     # Dataset building with uniform distribution
│   ├── generate_graphs.py     # Main CLI script
│   └── validate_dataset.py    # Validation utility
├── environment.yml            # Conda environment
├── requirements.txt           # Python dependencies
├── README.md                  # Full documentation
└── QUICKSTART.md             # This file
```

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError: No module named 'data'`, make sure you're running the script as a module from the project root:

```bash
# ✓ Correct
python -m data.generate_graphs --test-run

# ✗ Wrong
python data/generate_graphs.py --test-run
```

### Conda Environment Issues

If you have issues with the conda environment:

```bash
# Remove and recreate the environment
conda env remove -n ShortLLM
conda env create -f environment.yml
conda activate ShortLLM
```

### NetworkX Not Found

If NetworkX is not installed:

```bash
conda activate ShortLLM
conda install -c conda-forge networkx numpy
```

## Next Steps

1. **Generate the dataset** using one of the commands above
2. **Validate the data** to ensure correctness
3. **Explore the data** to understand distributions and patterns
4. **Train a model** on the shortest path task
5. **Evaluate performance** across different graph sizes and path lengths

For more detailed information, see the full [README.md](README.md).

