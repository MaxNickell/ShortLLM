import random
import pickle
import json
import networkx as nx
from typing import List, Dict, Any, Tuple, Optional

# Constants for tokens
EDGE_TOKEN = "<EDGE>"
BD_TOKEN = "<BD>"
START_PATH_TOKEN = "<START_PATH>"
END_PATH_TOKEN = "<END_PATH>"
TO_TOKEN = "<TO>"

def save_array_pkl(arr: Any, filepath: str) -> None:
    """Save an object to a pickle file."""
    with open(filepath, "wb") as f:
        pickle.dump(arr, f)

def load_array_pkl(filepath: str) -> Any:
    """Load an object from a pickle file."""
    with open(filepath, "rb") as f:
        return pickle.load(f)

def generate_path_graph(n: int) -> nx.Graph:
    """Generate a path graph with n nodes."""
    return nx.path_graph(n)

def max_shortest_path(G: nx.Graph) -> Tuple[int, Tuple[Any, Any]]:
    """
    Find the maximum shortest path length in the graph (diameter) 
    and the corresponding (source, target) pair.
    """
    max_len = -1
    max_pair = (None, None)
    
    # Check if graph is connected, otherwise diameter is undefined/infinite
    # But for this specific use case, we might just want max path in components
    # The current logic iterates all pairs, so it works for disconnected graphs too (in path length terms)
    # but strictly speaking we only generate connected graphs.
    
    for src, lengths in nx.all_pairs_shortest_path_length(G):
        for dst, dist in lengths.items():
            if dist > max_len:
                max_len = dist
                max_pair = (src, dst)
    return max_len, max_pair

def check_counts(pending_graph_counts: Dict[int, int]) -> bool:
    """Check if there are any pending path counts needed."""
    return any(v > 0 for v in pending_graph_counts.values())

def current_max_count(pending_graph_counts: Dict[int, int]) -> int:
    """Return the path length (key) that currently has the highest demand."""
    if not pending_graph_counts:
        return 0
    return max(pending_graph_counts.items(), key=lambda x: x[1])[0]

def path_given_length(G: nx.Graph, length: int) -> Optional[Tuple[Any, Any]]:
    """Return any (source, target) pair with a shortest path of the given length."""
    for src, lengths in nx.all_pairs_shortest_path_length(G):
        for dst, dist in lengths.items():
            if dist == length:
                return (src, dst)
    return None

def sort_graphs_by_diameter_desc(graph_list: List[nx.Graph]) -> List[nx.Graph]:
    """
    Sorts a list of NetworkX graphs by diameter (descending).
    """
    # Note: nx.diameter requires connected graphs. 
    # Our graphs are assumed connected.
    return sorted(graph_list, key=lambda G: nx.diameter(G), reverse=True)

def graph_key(G: nx.Graph) -> Tuple[int, ...]:
    """
    Cheap invariant key to bucket graphs:
    - Sorted degree sequence.
    """
    deg_seq = tuple(sorted((d for _, d in G.degree()), reverse=True)) # Specific order doesn't matter as long as consistent
    return deg_seq

def random_connected_graph(n: int, extra_edge_prob: float = 0.2) -> nx.Graph:
    """
    Generate a random connected graph on n nodes.
    """
    if random.random() < 0.1:
        G = generate_path_graph(n)
    else:
        G = nx.random_labeled_tree(n)

    # Add extra random edges to make it denser
    for i in range(n):
        for j in range(i + 1, n):
            if not G.has_edge(i, j) and random.random() < extra_edge_prob:
                G.add_edge(i, j)
    return G

def build_example(id: int, nid: int, graph: nx.Graph, source: Any, target: Any) -> Dict[str, Any]:
    """
    Constructs a dataset example dictionary.
    
    Args:
        id: Global example ID.
        nid: ID within the current N or batch.
        graph: NetworkX graph.
        source: Origin node.
        target: Destination node.
        
    Returns:
        Dictionary containing graph representation, path info, etc.
    """
    # Adjacency list
    adl = {str(k): list(v.keys()) for k, v in graph.adjacency()}
    
    # Serialized graph representation (edges)
    # Randomize edges for better generalization
    edges = list(graph.edges())
    random.shuffle(edges)
    
    graph_repr_parts = []
    for u, v in edges:
        graph_repr_parts.append(f"{EDGE_TOKEN}{u}{BD_TOKEN}{v}")
    graph_repr = "".join(graph_repr_parts)
    
    # Shortest path
    try:
        shortest_path = nx.shortest_path(graph, source=source, target=target)
        shortest_path_length = len(shortest_path) - 1
    except nx.NetworkXNoPath:
        # Should not happen for connected graphs
        shortest_path = []
        shortest_path_length = -1
        
    # Serialized path
    path_str_parts = [START_PATH_TOKEN]
    if shortest_path:
        path_str_parts.append(str(shortest_path[0]))
        for node in shortest_path[1:]:
            path_str_parts.append(f"{TO_TOKEN}{node}")
    path_str_parts.append(END_PATH_TOKEN)
    serialized_path = "".join(path_str_parts)
    
    return {
        "id": id,
        "nid": nid,
        "num_nodes": graph.number_of_nodes(),
        "adl": adl,
        "graph_repr": graph_repr,
        "origin": source,
        "destination": target,
        "shortest_path": shortest_path,
        "shortest_path_length": shortest_path_length,
        "serialized_path": serialized_path
    }