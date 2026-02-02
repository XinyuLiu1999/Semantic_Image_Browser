"""
Embedding-based Semantic Matching

Uses sentence embeddings for:
1. Matching tags to taxonomy nodes (tag → taxonomy)
2. Matching user queries to taxonomy nodes (query → taxonomy)

Embeddings capture semantic similarity beyond exact text matching.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count


def _init_worker(model_name: str):
    """Initialize worker process with its own model instance."""
    global _worker_model
    import torch
    torch.set_num_threads(1)
    from sentence_transformers import SentenceTransformer
    _worker_model = SentenceTransformer(model_name, device='cpu')


def _encode_chunk(args):
    """Encode a chunk of texts in worker process."""
    texts, normalize = args
    global _worker_model
    embeddings = _worker_model.encode(
        texts,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
        show_progress_bar=False
    )
    return embeddings


@dataclass
class EmbeddingConfig:
    """Configuration for embedding model."""
    model_name: str = "all-MiniLM-L6-v2"  # Lightweight, fast model
    normalize: bool = True
    batch_size: int = 512  # Batch size for single-process encoding
    num_processes: int = 8  # Number of processes (0 = auto, uses cpu_count)


class EmbeddingIndex:
    """
    Manages embeddings for taxonomy nodes and provides semantic search.

    The index pre-computes embeddings for all taxonomy node texts
    (name + synonyms + definition) and enables fast cosine similarity search.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self.model = None
        self.node_ids: List[str] = []
        self.node_names: Dict[str, str] = {}
        self.embeddings: Optional[np.ndarray] = None
        self._initialized = False
        self._device = None

    def _load_model(self):
        """Lazy load the sentence transformer model."""
        if self.model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            import torch

            # Set 1 thread per process for CPU to avoid contention
            torch.set_num_threads(1)

            # Auto-detect device
            if torch.cuda.is_available():
                self._device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"

            print(f"Loading embedding model: {self.config.model_name} (device: {self._device})")
            self.model = SentenceTransformer(self.config.model_name, device=self._device)
            print("Embedding model loaded successfully")
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for embedding-based matching. "
                "Install with: pip install sentence-transformers"
            )

    def _encode_single_process(self, texts: List[str]) -> np.ndarray:
        """Encode texts using single process."""
        self._load_model()
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=self.config.normalize,
            convert_to_numpy=True
        )
        return embeddings

    def _encode_multi_process(self, texts: List[str]) -> np.ndarray:
        """Encode texts using multiple processes with separate model instances."""
        num_processes = self.config.num_processes if self.config.num_processes > 0 else cpu_count()
        print(f"Using {num_processes} processes for parallel encoding")

        # Split texts into chunks for each process
        chunk_size = (len(texts) + num_processes - 1) // num_processes
        chunks = []
        for i in range(0, len(texts), chunk_size):
            chunks.append((texts[i:i + chunk_size], self.config.normalize))

        print(f"Split {len(texts)} texts into {len(chunks)} chunks")

        # Process chunks in parallel
        with Pool(
            processes=num_processes,
            initializer=_init_worker,
            initargs=(self.config.model_name,)
        ) as pool:
            results = []
            for i, result in enumerate(pool.imap(_encode_chunk, chunks)):
                results.append(result)
                print(f"  Completed chunk {i + 1}/{len(chunks)}")

        # Concatenate results
        embeddings = np.vstack(results)
        return embeddings

    def _encode(self, texts: List[str], use_multiprocess: bool = False) -> np.ndarray:
        """Encode texts to embeddings."""
        # Use multiprocess only for large batches on CPU
        if use_multiprocess and len(texts) > 1000:
            # Check device first
            self._load_model()
            if self._device == "cpu":
                return self._encode_multi_process(texts)

        return self._encode_single_process(texts)

    def index_taxonomy(self, taxonomy) -> None:
        """
        Build embedding index for all taxonomy nodes.

        Args:
            taxonomy: TaxonomyTree instance
        """
        print("Building embedding index for taxonomy nodes...")

        nodes = taxonomy.get_all_nodes()
        self.node_ids = []
        self.node_names = {}
        texts = []

        for node in nodes:
            self.node_ids.append(node.id)
            self.node_names[node.id] = node.name
            # Use searchable text: name + synonyms + definition
            texts.append(node.get_searchable_text())

        print(f"Encoding {len(texts)} taxonomy nodes...")
        # Use multiprocess for large taxonomy encoding
        self.embeddings = self._encode(texts, use_multiprocess=True)
        self._initialized = True

        print(f"Taxonomy embedding index built: {self.embeddings.shape}")

    def query(
        self,
        query_text: str,
        top_k: int = 10,
        min_similarity: float = 0.0
    ) -> List[Tuple[str, str, float]]:
        """
        Find most similar taxonomy nodes to query text.

        Args:
            query_text: Query string (tag or user query)
            top_k: Number of results to return
            min_similarity: Minimum cosine similarity threshold

        Returns:
            List of (node_id, node_name, similarity) tuples
        """
        if not self._initialized:
            raise RuntimeError("Index not initialized. Call index_taxonomy() first.")

        # Encode query (single process for small queries)
        query_embedding = self._encode([query_text], use_multiprocess=False)[0]

        # Compute cosine similarities
        # Since embeddings are normalized, dot product = cosine similarity
        similarities = np.dot(self.embeddings, query_embedding)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            sim = float(similarities[idx])
            if sim >= min_similarity:
                node_id = self.node_ids[idx]
                node_name = self.node_names[node_id]
                results.append((node_id, node_name, sim))

        return results

    def batch_query(
        self,
        query_texts: List[str],
        top_k: int = 5,
        min_similarity: float = 0.0
    ) -> Dict[str, List[Tuple[str, str, float]]]:
        """
        Find similar taxonomy nodes for multiple queries efficiently.

        Args:
            query_texts: List of query strings
            top_k: Number of results per query
            min_similarity: Minimum cosine similarity threshold

        Returns:
            Dict mapping query text to list of (node_id, node_name, similarity)
        """
        if not self._initialized:
            raise RuntimeError("Index not initialized. Call index_taxonomy() first.")

        if not query_texts:
            return {}

        # Encode all queries at once (use multiprocess for large batches)
        query_embeddings = self._encode(query_texts, use_multiprocess=True)

        # Compute all similarities at once
        all_similarities = np.dot(query_embeddings, self.embeddings.T)

        results = {}
        for i, query_text in enumerate(query_texts):
            similarities = all_similarities[i]
            top_indices = np.argsort(similarities)[::-1][:top_k]

            query_results = []
            for idx in top_indices:
                sim = float(similarities[idx])
                if sim >= min_similarity:
                    node_id = self.node_ids[idx]
                    node_name = self.node_names[node_id]
                    query_results.append((node_id, node_name, sim))

            results[query_text] = query_results

        return results

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for a text."""
        return self._encode([text], use_multiprocess=False)[0]

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        emb1 = self._encode([text1], use_multiprocess=False)[0]
        emb2 = self._encode([text2], use_multiprocess=False)[0]
        return float(np.dot(emb1, emb2))

    def get_node_embedding(self, node_id: str) -> Optional[np.ndarray]:
        """Get pre-computed embedding for a taxonomy node."""
        if not self._initialized:
            return None
        try:
            idx = self.node_ids.index(node_id)
            return self.embeddings[idx]
        except ValueError:
            return None

    def save(self, path: str) -> None:
        """Save the embedding index to disk."""
        import pickle
        data = {
            "node_ids": self.node_ids,
            "node_names": self.node_names,
            "embeddings": self.embeddings,
            "config": self.config
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Embedding index saved to: {path}")

    def load(self, path: str) -> None:
        """Load the embedding index from disk."""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.node_ids = data["node_ids"]
        self.node_names = data["node_names"]
        self.embeddings = data["embeddings"]
        self.config = data.get("config", EmbeddingConfig())
        self._initialized = True
        print(f"Embedding index loaded: {self.embeddings.shape}")
