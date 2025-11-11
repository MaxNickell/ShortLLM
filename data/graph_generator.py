"""
Graph generation module for creating connected Erdős-Rényi graphs.

This module provides utilities for generating random graphs with guaranteed
connectivity using the Erdős-Rényi model G(n,p).
"""

import random
from typing import List, Tuple, Dict, Set, Optional
import networkx as nx
from collections import deque


class GraphGenerator:
    """
    Generates connected, unweighted graphs using the Erdős-Rényi model.
    
    This class provides methods to generate random graphs and compute
    all-pairs shortest paths using BFS.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize the graph generator.
        
        Args:
            random_seed: Seed for reproducibility. If None, uses random state.
        """
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            # NetworkX uses its own random state
            self._nx_random_state = random.Random(random_seed)
        else:
            self._nx_random_state = None
    
    def generate_connected_graph(
        self,
        num_nodes: int,
        edge_probability: float,
        max_attempts: int = 100
    ) -> nx.Graph:
        """
        Generate a connected Erdős-Rényi graph G(n,p).
        
        Samples graphs from the Erdős-Rényi model until a connected graph
        is found.
        
        Args:
            num_nodes: Number of nodes in the graph.
            edge_probability: Probability of edge creation (p in G(n,p)).
            max_attempts: Maximum number of attempts before raising an error.
        
        Returns:
            A connected NetworkX graph.
        
        Raises:
            RuntimeError: If unable to generate connected graph within max_attempts.
        """
        for attempt in range(max_attempts):
            # Generate Erdős-Rényi graph
            graph = nx.erdos_renyi_graph(
                num_nodes,
                edge_probability,
                seed=self._nx_random_state
            )
            
            # Check if connected
            if nx.is_connected(graph):
                return graph
        
        # If we reach here, couldn't generate connected graph
        raise RuntimeError(
            f"Failed to generate connected graph with {num_nodes} nodes "
            f"and p={edge_probability} after {max_attempts} attempts. "
            f"Consider increasing edge_probability."
        )
    
    def compute_all_pairs_shortest_paths(
        self,
        graph: nx.Graph
    ) -> Dict[int, Dict[int, List[int]]]:
        """
        Compute all-pairs shortest paths using BFS.
        
        Since the graph is unweighted, BFS provides optimal shortest paths.
        
        Args:
            graph: The input graph.
        
        Returns:
            Dictionary mapping source -> destination -> shortest path (as list of nodes).
            The path includes both source and destination nodes.
        """
        all_paths = {}
        
        for source in graph.nodes():
            all_paths[source] = self._bfs_shortest_paths(graph, source)
        
        return all_paths
    
    def _bfs_shortest_paths(
        self,
        graph: nx.Graph,
        source: int
    ) -> Dict[int, List[int]]:
        """
        Compute shortest paths from source to all other nodes using BFS.
        
        Args:
            graph: The input graph.
            source: The source node.
        
        Returns:
            Dictionary mapping destination -> shortest path from source.
        """
        if source not in graph.nodes():
            return {}
        
        # Track visited nodes and their parents
        visited = {source}
        parent = {source: None}
        queue = deque([source])
        
        # BFS traversal
        while queue:
            current = queue.popleft()
            
            for neighbor in graph.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)
        
        # Reconstruct paths for all reachable nodes
        paths = {}
        for destination in visited:
            if destination == source:
                paths[destination] = [source]
            else:
                path = self._reconstruct_path(parent, source, destination)
                paths[destination] = path
        
        return paths
    
    def _reconstruct_path(
        self,
        parent: Dict[int, Optional[int]],
        source: int,
        destination: int
    ) -> List[int]:
        """
        Reconstruct path from source to destination using parent pointers.
        
        Args:
            parent: Dictionary mapping node -> parent node in BFS tree.
            source: The source node.
            destination: The destination node.
        
        Returns:
            List of nodes forming the path from source to destination.
        """
        path = []
        current = destination
        
        while current is not None:
            path.append(current)
            current = parent[current]
        
        path.reverse()
        return path
    
    def get_path_length(self, path: List[int]) -> int:
        """
        Get the length of a path (number of edges).
        
        Args:
            path: List of nodes in the path.
        
        Returns:
            Number of edges in the path.
        """
        return max(0, len(path) - 1)
    
    def relabel_nodes_randomly(self, graph: nx.Graph) -> Tuple[nx.Graph, Dict[int, int]]:
        """
        Relabel graph nodes with random identifiers to prevent structural bias.
        
        Args:
            graph: The input graph.
        
        Returns:
            Tuple of (relabeled_graph, mapping) where mapping is old_node -> new_node.
        """
        nodes = list(graph.nodes())
        new_labels = list(range(len(nodes)))
        random.shuffle(new_labels)
        
        mapping = dict(zip(nodes, new_labels))
        relabeled_graph = nx.relabel_nodes(graph, mapping, copy=True)
        
        return relabeled_graph, mapping
    
    def get_edge_list_randomized(self, graph: nx.Graph) -> List[Tuple[int, int]]:
        """
        Get edge list with randomized order to prevent structural bias.
        
        Args:
            graph: The input graph.
        
        Returns:
            List of edges in random order. Each edge is (node1, node2) where node1 < node2.
        """
        edges = [(min(u, v), max(u, v)) for u, v in graph.edges()]
        random.shuffle(edges)
        return edges

