"""
Image Index Management

Handles loading images from JSONL, linking to taxonomy via embeddings, and retrieval.

Pipeline:
1. Load images from JSONL
2. For each image, use BM25 to rank tags by relevance to generated_caption
3. Use embeddings to match top-ranked tags to taxonomy nodes
4. Store image-to-node links with combined scores
"""

import hashlib
import json
import os
import pickle
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field

from .bm25 import BM25TagScorer
from .embeddings import EmbeddingIndex


@dataclass
class ImageData:
    """Metadata for a single image."""
    id: str
    image_path: str
    generated_caption: str = ""
    tags: List[str] = field(default_factory=list)
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    # Computed fields
    ranked_tags: List[Tuple[str, float]] = field(default_factory=list)  # (tag, bm25_score)
    linked_nodes: List[Tuple[str, str, float]] = field(default_factory=list)  # (node_id, node_name, score)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "id": self.id,
            "image_path": self.image_path,
            "generated_caption": self.generated_caption,
            "tags": self.tags,
            "ranked_tags": self.ranked_tags,
            "linked_nodes": [(nid, name, round(score, 4)) for nid, name, score in self.linked_nodes],
            **self.extra_metadata
        }


class ImageIndex:
    """
    Manages image data and provides retrieval by taxonomy nodes.

    Linking Pipeline:
    1. BM25 scores each tag against generated_caption
    2. Top-ranked tags are matched to taxonomy nodes via embeddings
    3. Combined scores determine final image-node links

    This approach ensures:
    - Tags relevant to the actual image content (via BM25) are prioritized
    - Semantic matching (via embeddings) handles synonyms and related concepts
    """

    def __init__(self):
        self.images: Dict[str, ImageData] = {}  # image_id -> ImageData

        # Inverted indices
        self.tag_to_images: Dict[str, Set[str]] = defaultdict(set)
        self.node_to_images: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

        # Scorers
        self.bm25_scorer = BM25TagScorer()

        # Statistics
        self.total_images: int = 0
        self.avg_caption_length: float = 15.0

    def load_jsonl(self, jsonl_path: str, image_base_path: Optional[str] = None):
        """
        Load images from a JSONL file.

        Expected format per line:
        {
            "id": "img_001",
            "image_path": "path/to/image.jpg",
            "generated_caption": "A dog playing...",
            "tags": ["dog", "outdoor", "playing"],
            ...other fields...
        }

        Args:
            jsonl_path: Path to JSONL file
            image_base_path: Optional base path to prepend to image_path
        """
        print(f"Loading images from: {jsonl_path}")

        if not os.path.exists(jsonl_path):
            print(f"Warning: JSONL file not found: {jsonl_path}")
            return

        count = 0
        total_caption_length = 0

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {count + 1}: {e}")
                    continue

                # Extract required fields
                image_id = data.get("id") or data.get("image_id") or f"img_{count}"
                image_path = data.get("images", "")
                image_path = image_path[0]

                if image_base_path and not os.path.isabs(image_path):
                    image_path = os.path.join(image_base_path, image_path)

                # Extract optional fields
                caption = data.get("generated_caption", "") or data.get("caption", "")
                tags = data.get("tags", [])

                # Handle tags as string (comma-separated) or list
                if isinstance(tags, str):
                    tags = [t.strip() for t in tags.split(",") if t.strip()]

                # Collect extra metadata
                known_fields = {"id", "image_id", "image_path", "generated_caption", "caption", "tags"}
                extra = {k: v for k, v in data.items() if k not in known_fields}

                # Create ImageData
                img = ImageData(
                    id=image_id,
                    image_path=image_path,
                    generated_caption=caption,
                    tags=tags,
                    extra_metadata=extra
                )

                self.images[image_id] = img

                # Update tag index
                for tag in tags:
                    tag_lower = tag.lower()
                    self.tag_to_images[tag_lower].add(image_id)

                # Track caption length for BM25 normalization
                total_caption_length += len(caption.split())
                count += 1

        self.total_images = count
        if count > 0:
            self.avg_caption_length = total_caption_length / count

        print(f"Loaded {count} images with {len(self.tag_to_images)} unique tags")
        print(f"Average caption length: {self.avg_caption_length:.1f} words")

    def link_to_taxonomy(
        self,
        embedding_index: EmbeddingIndex,
        top_k_tags: int = 5,
        top_k_nodes: int = 3,
        min_similarity: float = 0.3
    ):
        """
        Link images to taxonomy nodes using BM25 + embedding pipeline.

        For each image:
        1. Use BM25 to rank tags by relevance to generated_caption
        2. Take top-k tags (most relevant to image content)
        3. Use embeddings to find matching taxonomy nodes for each tag
        4. Combine scores: final_score = bm25_score * embedding_similarity

        Args:
            embedding_index: Initialized EmbeddingIndex with taxonomy
            top_k_tags: Number of top tags to use per image
            top_k_nodes: Number of taxonomy nodes to link per tag
            min_similarity: Minimum embedding similarity threshold
        """
        print(f"Linking {len(self.images)} images to taxonomy nodes...")
        print(f"  - Top {top_k_tags} tags per image (ranked by BM25)")
        print(f"  - Top {top_k_nodes} taxonomy nodes per tag (by embedding similarity)")

        # Clear existing links
        self.node_to_images = defaultdict(list)

        # Step 1: Rank tags by BM25 for all images and collect unique tags
        print("  Step 1: Ranking tags by BM25 relevance...")
        unique_tags: Set[str] = set()
        image_top_tags: Dict[str, List[Tuple[str, float]]] = {}  # img_id -> [(tag, bm25_score)]

        for img_id, img in self.images.items():
            if not img.tags:
                continue

            # Rank tags by BM25 relevance to caption
            if img.generated_caption:
                img.ranked_tags = self.bm25_scorer.score_tags(
                    caption=img.generated_caption,
                    tags=img.tags,
                    avg_caption_length=self.avg_caption_length
                )
            else:
                # If no caption, use tags as-is with equal scores
                img.ranked_tags = [(tag, 1.0) for tag in img.tags]

            # Get top-k tags
            top_tags = img.ranked_tags[:top_k_tags]
            image_top_tags[img_id] = top_tags

            # Collect unique tags
            for tag, _ in top_tags:
                unique_tags.add(tag)

        print(f"  Collected {len(unique_tags)} unique tags from {len(image_top_tags)} images")

        # Step 2: Batch query all unique tags at once
        print("  Step 2: Batch matching tags to taxonomy nodes...")
        tag_list = list(unique_tags)
        tag_to_nodes = embedding_index.batch_query(
            query_texts=tag_list,
            top_k=top_k_nodes,
            min_similarity=min_similarity
        )

        # Step 3: Distribute results back to each image
        print("  Step 3: Computing combined scores and linking images...")
        for img_id, top_tags in image_top_tags.items():
            img = self.images[img_id]
            node_scores: Dict[str, Tuple[str, float]] = {}  # node_id -> (node_name, score)

            for tag, bm25_score in top_tags:
                # Get pre-computed matches for this tag
                matches = tag_to_nodes.get(tag, [])

                for node_id, node_name, emb_similarity in matches:
                    # Combined score: BM25 relevance * embedding similarity
                    # BM25 score normalized to [0, 1] range (cap at 2.0 for typical scores)
                    normalized_bm25 = min(bm25_score / 2.0, 1.0) if bm25_score > 0 else 0.5
                    combined_score = normalized_bm25 * emb_similarity

                    # Keep best score for each node
                    if node_id not in node_scores or node_scores[node_id][1] < combined_score:
                        node_scores[node_id] = (node_name, combined_score)

            # Store linked nodes
            img.linked_nodes = [
                (node_id, name, score)
                for node_id, (name, score) in sorted(
                    node_scores.items(),
                    key=lambda x: x[1][1],
                    reverse=True
                )
            ]

            # Update node_to_images index
            for node_id, node_name, score in img.linked_nodes:
                self.node_to_images[node_id].append((img_id, score))

        # Sort node_to_images by score
        for node_id in self.node_to_images:
            self.node_to_images[node_id].sort(key=lambda x: x[1], reverse=True)

        total_links = sum(len(v) for v in self.node_to_images.values())
        print(f"Created {total_links} image-node links across {len(self.node_to_images)} nodes")

    def get_images_for_node(
        self,
        node_id: str,
        taxonomy=None,
        include_descendants: bool = False,
        max_results: Optional[int] = 100
    ) -> List[Tuple[ImageData, float]]:
        """
        Get images linked to a taxonomy node.

        Args:
            node_id: Taxonomy node ID
            taxonomy: Optional TaxonomyTree for descendant lookup
            include_descendants: Whether to include images from descendant nodes
            max_results: Maximum number of results (None for no limit)

        Returns:
            List of (ImageData, score) tuples
        """
        all_image_scores: Dict[str, float] = {}

        # Get images directly linked to this node
        for img_id, score in self.node_to_images.get(node_id, []):
            all_image_scores[img_id] = max(all_image_scores.get(img_id, 0), score)

        # Optionally include descendants
        if include_descendants and taxonomy:
            descendants = taxonomy.get_descendants(node_id, max_depth=None)
            for desc_id in descendants:
                for img_id, score in self.node_to_images.get(desc_id, []):
                    # Apply decay for descendant matches
                    decayed_score = score * 0.8
                    all_image_scores[img_id] = max(all_image_scores.get(img_id, 0), decayed_score)

        # Sort by score and return
        sorted_images = sorted(all_image_scores.items(), key=lambda x: x[1], reverse=True)

        # Apply max_results limit if specified
        if max_results is not None:
            sorted_images = sorted_images[:max_results]

        results = []
        for img_id, score in sorted_images:
            if img_id in self.images:
                results.append((self.images[img_id], score))

        return results

    def search_by_tags(self, tags: List[str], match_all: bool = False) -> List[ImageData]:
        """
        Search images by tags directly.

        Args:
            tags: List of tags to match
            match_all: If True, image must have all tags. If False, any tag matches.

        Returns:
            List of matching ImageData
        """
        if not tags:
            return []

        tags_lower = [t.lower() for t in tags]

        if match_all:
            result_ids = None
            for tag in tags_lower:
                tag_images = self.tag_to_images.get(tag, set())
                if result_ids is None:
                    result_ids = tag_images.copy()
                else:
                    result_ids &= tag_images
        else:
            result_ids = set()
            for tag in tags_lower:
                result_ids |= self.tag_to_images.get(tag, set())

        return [self.images[img_id] for img_id in result_ids if img_id in self.images]

    def get_image(self, image_id: str) -> Optional[ImageData]:
        """Get a single image by ID."""
        return self.images.get(image_id)

    def get_all_images(self, limit: int = 100, offset: int = 0) -> List[ImageData]:
        """Get paginated list of all images."""
        all_ids = list(self.images.keys())
        return [self.images[img_id] for img_id in all_ids[offset:offset + limit]]

    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "total_images": len(self.images),
            "unique_tags": len(self.tag_to_images),
            "linked_nodes": len(self.node_to_images),
            "total_links": sum(len(v) for v in self.node_to_images.values()),
            "avg_caption_length": round(self.avg_caption_length, 1)
        }

    @staticmethod
    def compute_file_hash(file_path: str) -> str:
        """Compute MD5 hash of a file for cache invalidation."""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            # Read in chunks for large files
            for chunk in iter(lambda: f.read(65536), b''):
                hasher.update(chunk)
        return hasher.hexdigest()

    def save_links(self, cache_path: str, source_file_hash: str):
        """
        Save computed image-node links to cache.

        Args:
            cache_path: Path to save the cache file
            source_file_hash: Hash of the source JSONL file for validation
        """
        cache_data = {
            "version": 1,
            "source_hash": source_file_hash,
            "total_images": self.total_images,
            "avg_caption_length": self.avg_caption_length,
            # Save image data with computed fields
            "images": {
                img_id: {
                    "id": img.id,
                    "image_path": img.image_path,
                    "generated_caption": img.generated_caption,
                    "tags": img.tags,
                    "extra_metadata": img.extra_metadata,
                    "ranked_tags": img.ranked_tags,
                    "linked_nodes": img.linked_nodes
                }
                for img_id, img in self.images.items()
            },
            # Save indices
            "tag_to_images": {k: list(v) for k, v in self.tag_to_images.items()},
            "node_to_images": dict(self.node_to_images)
        }

        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"  Saved image links cache to: {cache_path}")

    def load_links(self, cache_path: str, expected_hash: str) -> bool:
        """
        Load cached image-node links if valid.

        Args:
            cache_path: Path to the cache file
            expected_hash: Expected hash of the source JSONL file

        Returns:
            True if cache was loaded successfully, False otherwise
        """
        if not os.path.exists(cache_path):
            return False

        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)

            # Validate cache version and source hash
            if cache_data.get("version") != 1:
                print(f"  Cache version mismatch, will rebuild")
                return False
            if cache_data.get("source_hash") != expected_hash:
                print(f"  Source file changed, will rebuild")
                return False

            # Restore state
            self.total_images = cache_data["total_images"]
            self.avg_caption_length = cache_data["avg_caption_length"]

            # Restore images
            self.images = {}
            for img_id, img_data in cache_data["images"].items():
                self.images[img_id] = ImageData(
                    id=img_data["id"],
                    image_path=img_data["image_path"],
                    generated_caption=img_data["generated_caption"],
                    tags=img_data["tags"],
                    extra_metadata=img_data["extra_metadata"],
                    ranked_tags=img_data["ranked_tags"],
                    linked_nodes=img_data["linked_nodes"]
                )

            # Restore indices
            self.tag_to_images = defaultdict(set)
            for k, v in cache_data["tag_to_images"].items():
                self.tag_to_images[k] = set(v)

            self.node_to_images = defaultdict(list)
            for k, v in cache_data["node_to_images"].items():
                self.node_to_images[k] = v

            print(f"  Loaded image links from cache: {cache_path}")
            print(f"  {self.total_images} images, {len(self.node_to_images)} linked nodes")
            return True

        except Exception as e:
            print(f"  Failed to load cache: {e}")
            return False
