"""
Taxonomy Tree Management for WikiKG Semantic Graph

Handles loading, processing, and querying the hierarchical taxonomy tree
from parquet files (nodes.parquet and edges.parquet).
"""

import os
import pickle
from collections import defaultdict
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


@dataclass
class NodeInfo:
    """Information about a taxonomy node."""
    id: str
    name: str
    synonyms: List[str] = field(default_factory=list)
    definition: str = ""
    node_type: str = "category"  # 'category' or 'page'
    depth: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "synonyms": self.synonyms,
            "definition": self.definition,
            "node_type": self.node_type,
            "depth": self.depth
        }

    def get_searchable_text(self) -> str:
        """Get text for embedding/search indexing."""
        # Convert underscores to spaces for better text matching
        name_clean = self.name.replace("_", " ")
        parts = [name_clean]
        parts.extend(self.synonyms)
        if self.definition:
            parts.append(self.definition)
        return " ".join(parts)


class TaxonomyTree:
    """
    Manages the WikiKG hierarchical taxonomy.

    Provides efficient access to:
    - Node information (name, type, depth)
    - Parent-child relationships
    - Ancestry paths
    - Descendant subtrees

    Loads from parquet files:
    - nodes.parquet: node_id, title, node_type, depth, parent_id
    - edges.parquet: parent_id, child_id
    """

    def __init__(self):
        self.nodes: Dict[str, NodeInfo] = {}
        self.child_to_parent: Dict[str, Optional[str]] = {}
        self.parent_to_children: Dict[str, List[str]] = defaultdict(list)
        self.root_id: str = "cat_192834"  # Main_topic_classifications
        self._path_cache: Dict[str, List[str]] = {}
        self._descendants_cache: Dict[str, List[str]] = {}

    @classmethod
    def load(
        cls,
        nodes_path: str,
        edges_path: str,
        cache_path: Optional[str] = None
    ) -> "TaxonomyTree":
        """
        Load taxonomy from parquet files.

        Args:
            nodes_path: Path to nodes.parquet
            edges_path: Path to edges.parquet
            cache_path: Optional path to save/load processed cache

        Returns:
            TaxonomyTree instance
        """
        # Try loading from cache first
        if cache_path and os.path.exists(cache_path):
            print(f"Loading taxonomy from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        if not HAS_PANDAS:
            raise ImportError(
                "pandas is required to load parquet files. "
                "Install with: pip install pandas pyarrow"
            )

        print(f"Loading taxonomy from parquet files...")
        print(f"  Nodes: {nodes_path}")
        print(f"  Edges: {edges_path}")

        tree = cls()
        tree._load_from_parquet(nodes_path, edges_path)

        # Save cache if path provided
        if cache_path:
            print(f"Saving taxonomy cache to: {cache_path}")
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(tree, f)

        return tree

    def _load_from_parquet(self, nodes_path: str, edges_path: str):
        """
        Load and process the parquet files.

        nodes.parquet columns: node_id, title, node_type, depth, parent_id
        edges.parquet columns: parent_id, child_id
        """
        # Load dataframes
        nodes_df = pd.read_parquet(nodes_path)
        edges_df = pd.read_parquet(edges_path)

        print(f"  Loaded {len(nodes_df)} nodes and {len(edges_df)} edges")

        # Build node info from nodes dataframe
        for _, row in nodes_df.iterrows():
            node_id = row['node_id']
            title = row['title']
            node_type = row['node_type']
            depth = int(row['depth'])
            parent_id = row['parent_id'] if pd.notna(row['parent_id']) else None

            # Create NodeInfo
            self.nodes[node_id] = NodeInfo(
                id=node_id,
                name=title,
                synonyms=[],  # Could be extended if available
                definition="",  # Could be extended if available
                node_type=node_type,
                depth=depth
            )

            # Store parent relationship from nodes.parquet
            self.child_to_parent[node_id] = parent_id

        # Build parent_to_children from edges dataframe
        for _, row in edges_df.iterrows():
            parent_id = row['parent_id']
            child_id = row['child_id']
            if parent_id and child_id:
                self.parent_to_children[parent_id].append(child_id)

        # Find and set root
        root_nodes = [
            nid for nid, parent in self.child_to_parent.items()
            if parent is None
        ]
        if root_nodes:
            self.root_id = root_nodes[0]
            print(f"  Root node: {self.root_id} ({self.nodes[self.root_id].name})")

        print(f"  Total nodes in tree: {len(self.nodes)}")

    def get_node(self, node_id: str) -> Optional[NodeInfo]:
        """Get node information by ID."""
        return self.nodes.get(node_id)

    def get_path_to_root(self, node_id: str) -> List[NodeInfo]:
        """
        Get the path from a node to the root.

        Returns list from root to node (inclusive).
        """
        if node_id in self._path_cache:
            return [
                self.nodes[nid]
                for nid in self._path_cache[node_id]
                if nid in self.nodes
            ]

        path_ids = []
        current = node_id
        visited = set()

        while current and current not in visited:
            visited.add(current)
            path_ids.append(current)
            current = self.child_to_parent.get(current)

        path_ids.reverse()  # Root first
        self._path_cache[node_id] = path_ids

        return [self.nodes[nid] for nid in path_ids if nid in self.nodes]

    def get_children(self, node_id: str) -> List[NodeInfo]:
        """Get direct children of a node."""
        child_ids = self.parent_to_children.get(node_id, [])
        return [self.nodes[cid] for cid in child_ids if cid in self.nodes]

    def get_descendants(
        self,
        node_id: str,
        max_depth: Optional[int] = None
    ) -> List[str]:
        """
        Get all descendant node IDs.

        Args:
            node_id: Starting node
            max_depth: Maximum depth to traverse (None for unlimited)

        Returns:
            List of descendant node IDs
        """
        if max_depth is None and node_id in self._descendants_cache:
            return self._descendants_cache[node_id]

        descendants = []
        queue = [(node_id, 0)]

        while queue:
            current, depth = queue.pop(0)
            children = self.parent_to_children.get(current, [])
            for child in children:
                if max_depth is None or depth < max_depth:
                    descendants.append(child)
                    queue.append((child, depth + 1))

        if max_depth is None:
            self._descendants_cache[node_id] = descendants

        return descendants

    def get_ancestors(self, node_id: str) -> List[str]:
        """Get all ancestor node IDs (excluding self)."""
        path = self.get_path_to_root(node_id)
        return [n.id for n in path[:-1]]  # Exclude the node itself

    def get_all_nodes(self) -> List[NodeInfo]:
        """Get all nodes in the tree."""
        return list(self.nodes.values())

    def get_statistics(self) -> Dict[str, Any]:
        """Get tree statistics."""
        leaf_nodes = [n for n in self.nodes if n not in self.parent_to_children]
        internal_nodes = [n for n in self.nodes if n in self.parent_to_children]
        category_nodes = [n for n in self.nodes.values() if n.node_type == "category"]
        page_nodes = [n for n in self.nodes.values() if n.node_type == "page"]
        max_depth = max((n.depth for n in self.nodes.values()), default=0)

        return {
            "total_nodes": len(self.nodes),
            "leaf_nodes": len(leaf_nodes),
            "internal_nodes": len(internal_nodes),
            "category_nodes": len(category_nodes),
            "page_nodes": len(page_nodes),
            "max_depth": max_depth,
            "root": self.root_id
        }

    def to_json(self) -> Dict[str, Any]:
        """Export tree as JSON-serializable dict."""
        return {
            "statistics": self.get_statistics(),
            "root": self.root_id,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "edges": self.child_to_parent
        }
