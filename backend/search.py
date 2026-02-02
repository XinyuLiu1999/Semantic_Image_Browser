"""
Search Service

Orchestrates query processing, taxonomy matching via embeddings, and image retrieval.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .taxonomy import TaxonomyTree, NodeInfo
from .embeddings import EmbeddingIndex
from .image_index import ImageIndex, ImageData


@dataclass
class SearchResult:
    """Complete search result including taxonomy context."""
    query: str
    total: int
    page: int
    limit: int
    taxonomy_path: List[Dict[str, Any]]  # Path for primary matched node
    matched_nodes: List[Dict[str, Any]]  # All matched taxonomy nodes
    images: List[Dict[str, Any]]  # Image results with metadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "total": self.total,
            "page": self.page,
            "limit": self.limit,
            "taxonomy_path": self.taxonomy_path,
            "matched_nodes": self.matched_nodes,
            "images": self.images
        }


class SearchService:
    """
    Main search service that coordinates all components.

    Flow:
    1. User query comes in
    2. Query matched to taxonomy nodes via EMBEDDINGS (semantic similarity)
    3. Images retrieved for matched nodes
    4. Results assembled with taxonomy context
    """

    def __init__(
        self,
        taxonomy: TaxonomyTree,
        embedding_index: EmbeddingIndex,
        image_index: ImageIndex
    ):
        self.taxonomy = taxonomy
        self.embedding_index = embedding_index
        self.image_index = image_index

    def search(
        self,
        query: str,
        page: int = 1,
        limit: int = 20,
        include_descendants: bool = True,
        max_taxonomy_matches: int = 5,
        min_similarity: float = 0.3
    ) -> SearchResult:
        """
        Perform a semantic search using embeddings.

        Args:
            query: User search query
            page: Page number (1-indexed)
            limit: Results per page
            include_descendants: Include images from descendant nodes
            max_taxonomy_matches: Maximum taxonomy nodes to match
            min_similarity: Minimum embedding similarity threshold

        Returns:
            SearchResult with images and taxonomy context
        """
        # Step 1: Match query to taxonomy nodes via embeddings
        # Get more candidates than needed so we can filter to nodes with images
        candidate_nodes = self.embedding_index.query(
            query_text=query,
            top_k=max_taxonomy_matches * 5,  # Get extra candidates for filtering
            min_similarity=min_similarity
        )

        if not candidate_nodes:
            return SearchResult(
                query=query,
                total=0,
                page=page,
                limit=limit,
                taxonomy_path=[],
                matched_nodes=[],
                images=[]
            )

        # Also find nodes with exact name match that might not be in embedding results
        query_lower = query.lower().strip()
        candidate_node_ids = {n[0] for n in candidate_nodes}
        for node_id, node_info in self.taxonomy.nodes.items():
            if node_info.name.lower() == query_lower and node_id not in candidate_node_ids:
                # Add exact name match with high similarity score
                candidate_nodes.append((node_id, node_info.name, 0.6))

        # Step 2: Find matched nodes and their image counts (for display)
        matched_nodes: List[Tuple[str, str, float, int]] = []  # (node_id, name, similarity, image_count)

        for node_id, node_name, node_similarity in candidate_nodes:
            # Boost exact name matches to prioritize direct matches
            node_name_lower = node_name.lower()
            if node_name_lower == query_lower:
                # Exact match gets highest boost
                node_similarity = min(1.0, node_similarity + 0.15)

            node_images = self.image_index.get_images_for_node(
                node_id,
                taxonomy=self.taxonomy if include_descendants else None,
                include_descendants=include_descendants,
                max_results=None  # Get all images to ensure consistent total count
            )

            image_count = len(node_images)
            if image_count > 0:
                matched_nodes.append((node_id, node_name, node_similarity, image_count))

        # Sort by similarity (descending) and limit to top matches that have images
        matched_nodes.sort(key=lambda x: x[2], reverse=True)
        matched_nodes = matched_nodes[:max_taxonomy_matches]

        if not matched_nodes:
            return SearchResult(
                query=query,
                total=0,
                page=page,
                limit=limit,
                taxonomy_path=[],
                matched_nodes=[],
                images=[]
            )

        # Step 3: Get images ONLY from the primary (first) matched node
        primary_node_id = matched_nodes[0][0]
        primary_node_similarity = matched_nodes[0][2]

        primary_images = self.image_index.get_images_for_node(
            primary_node_id,
            taxonomy=self.taxonomy if include_descendants else None,
            include_descendants=include_descendants,
            max_results=None
        )

        # Build image list with scores
        all_images: Dict[str, Tuple[ImageData, float, str]] = {}
        for img_data, img_score in primary_images:
            combined_score = primary_node_similarity * img_score
            all_images[img_data.id] = (img_data, combined_score, primary_node_id)

        # Step 4: Get taxonomy path for primary match
        taxonomy_path = self._get_taxonomy_path(primary_node_id)

        # Step 5: Sort and paginate
        sorted_images = sorted(all_images.values(), key=lambda x: x[1], reverse=True)
        total = len(sorted_images)

        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        page_images = sorted_images[start_idx:end_idx]

        # Step 6: Format results
        formatted_images = []
        for img_data, score, node_id in page_images:
            img_dict = img_data.to_dict()
            img_dict["relevance_score"] = round(score, 4)
            img_dict["matched_node"] = node_id
            formatted_images.append(img_dict)

        formatted_nodes = []
        for nid, name, sim, img_count in matched_nodes:
            node_info = {"id": nid, "name": name, "score": round(sim, 4)}
            # Add parent name for disambiguation when nodes have the same name
            parent_id = self.taxonomy.child_to_parent.get(nid)
            if parent_id:
                parent_node = self.taxonomy.get_node(parent_id)
                if parent_node:
                    node_info["parent_name"] = parent_node.name
            formatted_nodes.append(node_info)

        return SearchResult(
            query=query,
            total=total,
            page=page,
            limit=limit,
            taxonomy_path=taxonomy_path,
            matched_nodes=formatted_nodes,
            images=formatted_images
        )

    def _get_taxonomy_path(self, node_id: str) -> List[Dict[str, Any]]:
        """Get formatted taxonomy path for a node."""
        path = self.taxonomy.get_path_to_root(node_id)
        return [{"id": n.id, "name": n.name} for n in path]

    def get_node_details(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a taxonomy node."""
        node = self.taxonomy.get_node(node_id)
        if not node:
            return None

        path = self.taxonomy.get_path_to_root(node_id)
        children = self.taxonomy.get_children(node_id)

        # Count images
        image_count = len(self.image_index.node_to_images.get(node_id, []))

        return {
            "node": node.to_dict(),
            "path_to_root": [{"id": n.id, "name": n.name} for n in path],
            "children": [{"id": c.id, "name": c.name} for c in children],
            "image_count": image_count
        }

    def browse_node(
        self,
        node_id: str,
        page: int = 1,
        limit: int = 20,
        include_descendants: bool = True
    ) -> Dict[str, Any]:
        """
        Browse images under a specific taxonomy node.

        Unlike search, this doesn't score against a query - just returns
        images linked to the node.
        """
        node = self.taxonomy.get_node(node_id)
        if not node:
            return {"error": "Node not found", "node_id": node_id}

        # Get images
        node_images = self.image_index.get_images_for_node(
            node_id,
            taxonomy=self.taxonomy if include_descendants else None,
            include_descendants=include_descendants,
            max_results=None  # Get all images to ensure consistent total count
        )

        total = len(node_images)
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        page_images = node_images[start_idx:end_idx]

        # Format
        formatted_images = []
        for img_data, score in page_images:
            img_dict = img_data.to_dict()
            img_dict["relevance_score"] = round(score, 4)
            formatted_images.append(img_dict)

        return {
            "node": node.to_dict(),
            "path_to_root": self._get_taxonomy_path(node_id),
            "total": total,
            "page": page,
            "limit": limit,
            "images": formatted_images
        }

    def get_suggestions(self, query: str, limit: int = 10) -> List[Dict[str, str]]:
        """
        Get taxonomy node suggestions for autocomplete.

        Uses embedding similarity for semantic matching.

        Args:
            query: Partial query string
            limit: Maximum suggestions

        Returns:
            List of {id, name} suggestions
        """
        if len(query) < 2:
            return []

        matches = self.embedding_index.query(query, top_k=limit, min_similarity=0.2)
        return [{"id": nid, "name": name} for nid, name, _ in matches]
