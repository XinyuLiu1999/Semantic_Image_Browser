"""
BM25 Implementation for Tag Relevance Scoring

BM25 is used to score the relevance of each tag against the generated_caption.
Tags with higher BM25 scores are considered more relevant to the image content.
"""

import math
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Set


def tokenize(text: str) -> List[str]:
    """
    Simple tokenizer: lowercase, split on non-alphanumeric, filter short tokens.
    """
    text = text.lower()
    tokens = re.findall(r'\b[a-z0-9]+\b', text)
    return [t for t in tokens if len(t) > 1]


class BM25TagScorer:
    """
    BM25 scorer for ranking tags by relevance to a caption.

    This is used to determine which tags are most relevant to an image's
    generated_caption. Tags with higher scores should be prioritized when
    linking to the taxonomy tree.

    Usage:
        scorer = BM25TagScorer()
        ranked_tags = scorer.score_tags(
            caption="A golden retriever playing in the park",
            tags=["dog", "golden retriever", "park", "outdoor", "animal"]
        )
        # Returns: [("golden retriever", 0.95), ("park", 0.82), ("dog", 0.75), ...]
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            k1: Term frequency saturation parameter (default 1.5)
            b: Length normalization parameter (default 0.75)
        """
        self.k1 = k1
        self.b = b

    def score_tags(
        self,
        caption: str,
        tags: List[str],
        avg_caption_length: float = 15.0
    ) -> List[Tuple[str, float]]:
        """
        Score each tag's relevance to the caption using BM25.

        The caption is treated as the "document" and each tag as a "query".
        Tags that appear in or relate to the caption get higher scores.

        Args:
            caption: The generated caption text
            tags: List of tags to score
            avg_caption_length: Average caption length for normalization

        Returns:
            List of (tag, score) tuples sorted by score descending
        """
        if not caption or not tags:
            return [(tag, 0.0) for tag in tags]

        # Tokenize caption
        caption_tokens = tokenize(caption)
        if not caption_tokens:
            return [(tag, 0.0) for tag in tags]

        # Build term frequency map for caption
        caption_tf: Dict[str, int] = defaultdict(int)
        for token in caption_tokens:
            caption_tf[token] += 1

        caption_len = len(caption_tokens)

        # Score each tag
        scored_tags = []
        for tag in tags:
            tag_tokens = tokenize(tag)
            if not tag_tokens:
                scored_tags.append((tag, 0.0))
                continue

            score = 0.0
            for term in tag_tokens:
                tf = caption_tf.get(term, 0)
                if tf > 0:
                    # BM25 scoring for this term
                    # IDF is approximated as 1.0 since we're scoring single document
                    idf = 1.0
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (caption_len / avg_caption_length))
                    score += idf * (numerator / denominator)

            # Normalize by number of tag tokens
            if len(tag_tokens) > 0:
                score = score / len(tag_tokens)

            scored_tags.append((tag, score))

        # Sort by score descending
        scored_tags.sort(key=lambda x: x[1], reverse=True)
        return scored_tags

    def get_top_tags(
        self,
        caption: str,
        tags: List[str],
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[Tuple[str, float]]:
        """
        Get the top-k most relevant tags for a caption.

        Args:
            caption: The generated caption
            tags: List of tags to score
            top_k: Maximum number of tags to return
            min_score: Minimum score threshold

        Returns:
            Top-k (tag, score) tuples
        """
        scored = self.score_tags(caption, tags)
        filtered = [(tag, score) for tag, score in scored if score >= min_score]
        return filtered[:top_k]
