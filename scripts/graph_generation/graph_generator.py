import sys
import subprocess
import networkx as nx
from typing import List, Dict, Any
from common import (
    save_array_pkl,
    generate_path_graph,
    check_counts,
    current_max_count,
    random_connected_graph,
    graph_key,
    max_shortest_path,
    path_given_length
)

GENG_PATH = "/usr/bin/nauty-geng"

def generate_connected_graphs_geng(n: int, expected_count: int) -> List[nx.Graph]:
    """
    Generate connected, non-isomorphic graphs with n nodes
    using nauty-geng, stopping when expected count is reached.

    Args:
        n (int): Number of nodes
        expected_count (int): Expected number of graphs

    Returns:
        list[nx.Graph]: List of NetworkX graphs
    """
    graphs = []
    
    # Run geng command: -c for connected, n is number of nodes
    try:
        process = subprocess.Popen(
            [GENG_PATH, "-c", str(n)],
            stdout=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        print(f"Error: {GENG_PATH} not found. Please ensure nauty-geng is installed.")
        sys.exit(1)

    for line in process.stdout:
        if len(graphs) >= expected_count:
            break

        # GENG outputs graph6 format
        G = nx.from_graph6_bytes(line.strip().encode())
        graphs.append(G)

    process.wait()
    return graphs

def generate_random_connected_graphs_large(n: int, extra_edge_prob: float = 0.2) -> List[nx.Graph]:
    """
    Generate a diverse set of connected graphs for large N (n >= 9).
    Uses a bucketed rejection sampling approach to find non-isomorphic graphs
    and attempts to balance the distribution of maximum shortest path lengths.
    """
    print(f"Generating random graphs for N={n}...")
    
    # Target counts for different path lengths to ensure diversity
    # These are heuristic targets based on observation
    example_counts = {
        9: 14286,
        10: 12500,
        11: 11111,
        12: 10000,
        13: 9091,
        14: 8333,
        15: 7692
    }
    
    # Set default target if not in map
    target_per_bucket = example_counts.get(n, 5000)

    # Buckets for path lengths. Key is path length, value is count remaining
    # We want to fill buckets for path lengths 1 to n-1
    # Actually code uses 1 to n-2 mostly
    pending_graph_counts = {i: target_per_bucket for i in range(1, n - 1)}
    
    # Buckets for isomorphism check: invariant_key -> list of graphs
    buckets: Dict[Any, List[nx.Graph]] = {}
    
    count = 0
    tries = 0
    
    # Start with a path graph to ensure we have at least one long path
    p_graph = generate_path_graph(n)
    p_key = graph_key(p_graph)
    buckets[p_key] = [p_graph]
    count += 1
    
    no_help_streak = 0
    max_no_help = 10_000_000

    while check_counts(pending_graph_counts):
        tries += 1
        current_target_len = current_max_count(pending_graph_counts)
        
        # Generate a candidate graph
        G = random_connected_graph(n, extra_edge_prob=extra_edge_prob)
        key = graph_key(G)
        bucket = buckets.get(key, [])

        # Check for isomorphism within the bucket
        is_iso = False
        for existing_G in bucket:
            if nx.is_isomorphic(G, existing_G):
                is_iso = True
                break
        
        if is_iso:
            no_help_streak += 1
            if no_help_streak > max_no_help:
                print(f"n={n}: stopping due to no-help streak; may have exhausted useful non-iso graphs.")
                break
            continue
            
        no_help_streak = 0 # Reset streak on success
        
        # Check if this graph helps us fill a pending bucket
        mxp, _ = max_shortest_path(G)
        
        # Strategies to assign this graph to a bucket (decrement count)
        if mxp == current_target_len:
            pending_graph_counts[current_target_len] -= 1
        elif mxp > current_target_len:
            # Can we find a path of length 'current_target_len'?
            # If yes, this graph is useful for that length too, essentially
            # But the logic here seems to be "counting graphs that *support* length X"
            # Or just gathering graphs. 
            # The original logic used this to decrement counts.
            pth = path_given_length(G, current_target_len)
            if pth is not None:
                pending_graph_counts[current_target_len] -= 1
            else:
                # Should not happen if max path > target
                continue
        else:
             # mxp < current_target_len. 
             # Check if it can fill any other bucket
            tmp = mxp
            while tmp > 1:
                if tmp in pending_graph_counts and pending_graph_counts[tmp] > 0:
                    pth = path_given_length(G, tmp)
                    if pth is not None:
                        pending_graph_counts[tmp] -= 1
                        break
                tmp -= 1
    
        # Add to collection
        bucket.append(G)
        buckets[key] = bucket
        count += 1
        
        if count % 100 == 0:
            print(f"Generated {count} examples after {tries} tries...")
            # print(f"Current pending counts: {pending_graph_counts}")

    print(f"Finished N={n}: {count} examples after {tries} tries.")
    
    # Flatten buckets
    all_graphs = []
    for bucket in buckets.values():
        all_graphs.extend(bucket)
    return all_graphs


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python graph_generator.py <start_n> <end_n>")
        sys.exit(1)
        
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    
    # OEIS A001187: Number of connected non-isomorphic graphs
    non_iso_connected_graphs_counts = {
        2: 1,
        3: 2,
        4: 6,
        5: 21,
        6: 112,
        7: 853,
        8: 11117,
    }
    
    for n in range(start, end + 1):
        if n > 8:
            graphs = generate_random_connected_graphs_large(n)
        else:
            if n not in non_iso_connected_graphs_counts:
                 print(f"Warning: Exact count for n={n} not known/supported for GENG in this script.")
                 continue
            graphs = generate_connected_graphs_geng(n, non_iso_connected_graphs_counts[n])
            
        save_path = f"graphs/graphs_{n}.pkl"
        save_array_pkl(graphs, save_path)
        print(f"Saved {len(graphs)} graphs to {save_path}")
