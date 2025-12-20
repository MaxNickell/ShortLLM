import sys
import json
import random
import itertools
import networkx as nx
from typing import List, Dict, Any
from common import (
    load_array_pkl,
    check_counts,
    sort_graphs_by_diameter_desc,
    max_shortest_path,
    path_given_length,
    build_example
)

def generate_examples(start_n: int, end_n: int) -> None:
    """
    Generate dataset examples for graphs with node counts from start_n to end_n.
    """
    global_id = 1
    rows = []
    
    # Heuristic limits for smaller graphs where we might have too few non-iso graphs
    # but lots of relabeling possibilities
    examples_limit_map = {
        2: 105,
        3: 910,
        4: 8190,
        5: 63063
    }
    
    # Hard limit for larger graphs
    default_limit = 100000
    
    for n in range(start_n, end_n + 1):
        try:
            graphs = load_array_pkl(f"graphs/graphs_{n}.pkl")
        except FileNotFoundError:
            print(f"Error: graphs/graphs_{n}.pkl not found. Run graph_generator.py first.")
            continue
            
        print(f"Processing n={n}. Loaded {len(graphs)} base graphs.")
        
        # All possible node combinations from range [1, 15] of size n
        # This is used for relabeling to create diverse node IDs
        all_node_ids = range(1, 16)
        pairs_combinations = list(itertools.combinations(all_node_ids, n))
        random.shuffle(pairs_combinations)
        
        cnt = 1
        target_counts = {}
        
        # Special handling for very small n (n=2)
        if n == 2:
            # For n=2, there is only 1 non-iso graph (two nodes connected)
            # We just iterate through combinations
            for graph in graphs:
                nodes = list(graph.nodes())
                for pair in pairs_combinations:
                    mapping = {nodes[0]: pair[0], nodes[1]: pair[1]}
                    relabeled_graph = nx.relabel_nodes(graph, mapping)
                    # For n=2, path is always between the two nodes
                    ex = build_example(global_id, cnt, relabeled_graph, pair[0], pair[1])
                    rows.append(ex)
                    cnt += 1
                    global_id += 1
                    if cnt % 100 == 0:
                        print(f"Generated {cnt} examples for n={n}")
        else:
            # Determine target count
            if n < 6:
                total_target = examples_limit_map.get(n, default_limit)
            else:
                total_target = default_limit
                
            print(f"Target total samples for n={n}: {total_target}")
            
            # Distribute targets across path lengths
            # We mostly care about lengths 1 to n-1 (diameter dependent)
            if n <= 8:
                per_bucket = total_target // (n - 1)
                target_counts = {k: per_bucket for k in range(1, n)}
            else:
                # For larger n, reserve some for n-1 but mostly 1 to n-2
                per_bucket = total_target // (n - 2)
                target_counts = {k: per_bucket for k in range(1, n - 1)}
                target_counts[n - 1] = 10 # Heuristic explicit small target for max path
                
            print(f"Target counts buckets: {target_counts}")
            
            # Sort graphs to prioritize large diameter ones (helps find long paths)
            sorted_graphs = sort_graphs_by_diameter_desc(graphs)
            
            # Iterate until we fill all buckets or run out of graphs/patience
            # We loop over graphs and try to match them with random node labels
            while check_counts(target_counts):
                
                # Check if we are making progress or looping endlessly?
                # The original code just iterated sorted_graphs repeatedly.
                # To avoid infinite loops if buckets can't be filled, we should ideally track logic. 
                # But original logic relied on breaking if check_counts is false.
                
                processed_in_pass = 0
                
                for graph in sorted_graphs:
                    if not check_counts(target_counts):
                        break
                        
                    nodes = list(graph.nodes())
                    
                    # Randomly pick a set of target node IDs
                    pair_ids = random.choice(pairs_combinations)
                    
                    mapping = {nodes[i]: pair_ids[i] for i in range(n)}
                    relabeled_graph = nx.relabel_nodes(graph, mapping)
                    
                    mxp, (s, t) = max_shortest_path(relabeled_graph)
                    
                    # Try to use the max path first
                    if mxp in target_counts and target_counts[mxp] > 0:
                        ex = build_example(global_id, cnt, relabeled_graph, s, t)
                        rows.append(ex)
                        target_counts[mxp] -= 1
                        cnt += 1
                        global_id += 1
                        processed_in_pass += 1
                    else:
                        # Fallback: look for shorter paths in this graph that we still need
                        current_len = mxp
                        found_fallback = False
                        
                        # Check downwards for needed lengths
                        while current_len > 1:
                            if current_len in target_counts and target_counts[current_len] > 0:
                                pth = path_given_length(relabeled_graph, current_len)
                                if pth is not None:
                                    (s_fb, t_fb) = pth
                                    ex = build_example(global_id, cnt, relabeled_graph, s_fb, t_fb)
                                    rows.append(ex)
                                    target_counts[current_len] -= 1
                                    cnt += 1
                                    global_id += 1
                                    processed_in_pass += 1
                                    found_fallback = True
                                    break
                            current_len -= 1
                        
                        if not found_fallback:
                            continue

                    if cnt % 1000 == 0:
                        print(f"Generated {cnt} examples for n={n}")

                if processed_in_pass == 0:
                    print(f"Warning: Could not generate any more examples for n={n} with remaining targets: {target_counts}")
                    break
                    
            print(f"Finished n={n}. Final remaining targets: {target_counts}")

    out_path = f"all_examples_{start_n}_{end_n}.jsonl"
    with open(out_path, "w") as out_f:
        for row in rows:
            out_f.write(json.dumps(row) + "\n")
            
    print(f"Saved {len(rows)} total examples to {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python dataset_generator.py <start_n> <end_n>")
        sys.exit(1)
        
    start_arg = int(sys.argv[1])
    end_arg = int(sys.argv[2])
    generate_examples(start_arg, end_arg)
