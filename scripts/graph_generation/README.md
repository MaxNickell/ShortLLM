# Graph Shortest Path Dataset Generator

This project generates a synthetic dataset for training language models on shortest path finding tasks in graphs. It produces a JSONL file containing graph representations, shortest path questions, and ground truth answers.

## Requirements

- Python 3.8+
- [NetworkX](https://networkx.org/) (`pip install networkx`)
- [nauty](https://pallini.di.uniroma1.it/) (specifically `geng` for small graph generation)
  - The code assumes `geng` is located at `/usr/bin/nauty-geng`. You may need to adjust `GENG_PATH` in `graph_generator.py` if installed elsewhere.

## File Structure

- `graph_generator.py`: Generates connected non-isomorphic graphs. Uses `geng` for small $N$ and a random sampler for large $N$.
- `dataset_generator.py`: Consumes generated graphs to create dataset examples with node relabeling and path length balancing.
- `common.py`: Shared utility functions and data structures.
- `graphs/`: Directory where generated graph pickle files are stored.

## Usage

### 1. Generate Base Graphs

First, generate the base set of non-isomorphic graphs for the desired range of node counts (e.g., 2 to 14).

```bash
mkdir -p graphs
python graph_generator.py 2 14
```

This will create `graphs/graphs_N.pkl` files for each N.

### 2. Generate Dataset Examples

Once the graphs are generated, run the dataset generator to create the final JSONL dataset.

```bash
python dataset_generator.py 2 14
```

This will produce `all_examples_2_14.jsonl` containing the full dataset.

## Data Format

Each line in the output JSONL file is a JSON object with the following fields:

- `id`: Unique example ID.
- `num_nodes`: Number of nodes in the graph.
- `adl`: Adjacency list dictionary.
- `graph_repr`: Serialized string representation of edges (randomized order).
- `origin`: Source node ID.
- `destination`: Target node ID.
- `shortest_path`: List of nodes representing the shortest path.
- `shortest_path_length`: Length of the shortest path.
- `serialized_path`: Tokenized string representation of the path (target for LLM).

## Methodology

- **Small Graphs ($N \le 8$)**: Exhaustive generation of all connected non-isomorphic graphs using `geng`.
- **Large Graphs ($N > 8$)**: Random generation with rejection sampling to ensure diversity and connectivity.
- **Balancing**: The dataset generator attempts to balance the distribution of examples across all possible shortest path lengths for each $N$.
