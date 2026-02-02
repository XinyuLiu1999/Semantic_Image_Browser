"""
WikiKG Semantic Image Retrieval API

FastAPI backend for semantic image search using embedding-based taxonomy matching.
Uses WikiKG taxonomy from parquet files (nodes.parquet and edges.parquet).
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.taxonomy import TaxonomyTree
from backend.embeddings import EmbeddingIndex
from backend.image_index import ImageIndex
from backend.search import SearchService


# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

# WikiKG taxonomy parquet files (default paths)
DEFAULT_NODES_PARQUET = "/cephfs/liuxinyu/wikiKG/tests/semantic_graph/nodes.parquet"
DEFAULT_EDGES_PARQUET = "/cephfs/liuxinyu/wikiKG/tests/semantic_graph/edges.parquet"

TAXONOMY_CACHE = DATA_DIR / "taxonomy_cache.pkl"
EMBEDDING_CACHE = DATA_DIR / "embedding_index.pkl"
IMAGE_LINKS_CACHE = DATA_DIR / "image_links_cache.pkl"

# Global service instances
taxonomy: Optional[TaxonomyTree] = None
embedding_index: Optional[EmbeddingIndex] = None
image_index: Optional[ImageIndex] = None
search_service: Optional[SearchService] = None


def get_jsonl_path() -> Optional[Path]:
    """Get JSONL file path from environment variable or default."""
    data_path = os.environ.get("WIKIKG_IMAGE_DATA")
    if data_path:
        return Path(data_path)
    # Default fallback
    default_path = DATA_DIR / "images.jsonl"
    if default_path.exists():
        return default_path
    return None


def get_taxonomy_paths() -> tuple[str, str]:
    """Get taxonomy parquet file paths from environment or defaults."""
    nodes_path = os.environ.get("WIKIKG_NODES_PARQUET", DEFAULT_NODES_PARQUET)
    edges_path = os.environ.get("WIKIKG_EDGES_PARQUET", DEFAULT_EDGES_PARQUET)
    return nodes_path, edges_path


def initialize_services():
    """Initialize all services on startup."""
    global taxonomy, embedding_index, image_index, search_service

    print("=" * 60)
    print("Initializing WikiKG Semantic Image Retrieval System")
    print("=" * 60)

    # Ensure data directory exists
    DATA_DIR.mkdir(exist_ok=True)

    # Step 1: Load taxonomy from parquet files
    print("\n[1/5] Loading taxonomy tree from parquet files...")
    nodes_path, edges_path = get_taxonomy_paths()
    taxonomy = TaxonomyTree.load(
        nodes_path=nodes_path,
        edges_path=edges_path,
        cache_path=str(TAXONOMY_CACHE)
    )
    stats = taxonomy.get_statistics()
    print(f"  Loaded {stats['total_nodes']} nodes (max depth: {stats['max_depth']})")
    print(f"  Categories: {stats.get('category_nodes', 'N/A')}, Pages: {stats.get('page_nodes', 'N/A')}")

    # Step 2: Build or load embedding index for taxonomy
    print("\n[2/5] Building embedding index for taxonomy nodes...")
    embedding_index = EmbeddingIndex()
    if EMBEDDING_CACHE.exists():
        try:
            embedding_index.load(str(EMBEDDING_CACHE))
            print(f"  Loaded from cache: {EMBEDDING_CACHE}")
        except Exception as e:
            print(f"  Cache load failed: {e}")
            embedding_index.index_taxonomy(taxonomy)
            embedding_index.save(str(EMBEDDING_CACHE))
    else:
        embedding_index.index_taxonomy(taxonomy)
        embedding_index.save(str(EMBEDDING_CACHE))

    # Step 3: Load images and link to taxonomy (with caching)
    print("\n[3/5] Loading image data...")
    image_index = ImageIndex()
    jsonl_file = get_jsonl_path()
    cache_loaded = False

    if jsonl_file and jsonl_file.exists():
        print(f"  Dataset: {jsonl_file}")
        # Compute hash of source file for cache validation
        source_hash = ImageIndex.compute_file_hash(str(jsonl_file))
        print(f"  Source hash: {source_hash[:16]}...")

        # Try to load from cache first
        if IMAGE_LINKS_CACHE.exists():
            print("\n[4/5] Loading image links from cache...")
            cache_loaded = image_index.load_links(str(IMAGE_LINKS_CACHE), source_hash)

        if not cache_loaded:
            # Load images and compute links
            print(f"  Loading images from JSONL...")
            image_index.load_jsonl(str(jsonl_file))

            # Link images to taxonomy using embeddings
            print("\n[4/5] Linking images to taxonomy nodes via embeddings...")
            if image_index.total_images > 0:
                image_index.link_to_taxonomy(embedding_index)
                # Save cache for next time
                image_index.save_links(str(IMAGE_LINKS_CACHE), source_hash)
            else:
                print("  Skipped (no images loaded)")
        else:
            print("\n[4/5] Using cached image-taxonomy links")
    elif jsonl_file:
        print(f"  Warning: No image data found at {jsonl_file}")
        print(f"  Create a JSONL file with image metadata to enable search.")
    else:
        print(f"  Warning: No dataset path provided.")
        print(f"  Use --data flag to specify the path to your JSONL dataset.")

    # Step 5: Create search service
    print("\n[5/5] Creating search service...")
    search_service = SearchService(taxonomy, embedding_index, image_index)

    print("\n" + "=" * 60)
    print("System initialized successfully!")
    print("=" * 60)
    print(f"  Taxonomy nodes: {stats['total_nodes']}")
    print(f"  Images loaded:  {image_index.total_images}")
    print(f"  Unique tags:    {len(image_index.tag_to_images)}")
    print("=" * 60 + "\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    initialize_services()
    yield
    # Shutdown
    print("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="WikiKG Semantic Image Retrieval",
    description="Search images using embedding-based taxonomy matching with WikiKG",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/api/search")
async def search(
    q: str = Query(..., min_length=1, description="Search query"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Results per page"),
    include_descendants: bool = Query(True, description="Include descendant nodes")
):
    """
    Perform semantic search over images.

    Maps the query to taxonomy nodes using embeddings and returns matching images.
    """
    if not search_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    result = search_service.search(
        query=q,
        page=page,
        limit=limit,
        include_descendants=include_descendants
    )
    return result.to_dict()


@app.get("/api/suggestions")
async def suggestions(
    q: str = Query(..., min_length=2, description="Partial query"),
    limit: int = Query(10, ge=1, le=50, description="Max suggestions")
):
    """Get taxonomy node suggestions for autocomplete."""
    if not search_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return search_service.get_suggestions(q, limit)


@app.get("/api/taxonomy")
async def get_taxonomy_stats():
    """Get taxonomy statistics."""
    if not taxonomy:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return taxonomy.get_statistics()


@app.get("/api/taxonomy/{node_id}")
async def get_taxonomy_node(node_id: str):
    """Get detailed information about a taxonomy node."""
    if not search_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    result = search_service.get_node_details(node_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")

    return result


@app.get("/api/browse/{node_id}")
async def browse_node(
    node_id: str,
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    include_descendants: bool = Query(True)
):
    """Browse images under a specific taxonomy node."""
    if not search_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    result = search_service.browse_node(
        node_id=node_id,
        page=page,
        limit=limit,
        include_descendants=include_descendants
    )

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    return result


@app.get("/api/images/{image_id}")
async def get_image_metadata(image_id: str):
    """Get metadata for a specific image."""
    if not image_index:
        raise HTTPException(status_code=503, detail="Service not initialized")

    image = image_index.get_image(image_id)
    if not image:
        raise HTTPException(status_code=404, detail=f"Image not found: {image_id}")

    return image.to_dict()


@app.get("/api/images/{image_id}/file")
async def get_image_file(image_id: str):
    """Serve the actual image file."""
    if not image_index:
        raise HTTPException(status_code=503, detail="Service not initialized")

    image = image_index.get_image(image_id)
    if not image:
        raise HTTPException(status_code=404, detail=f"Image not found: {image_id}")

    if not os.path.exists(image.image_path):
        raise HTTPException(status_code=404, detail=f"Image file not found: {image.image_path}")

    return FileResponse(image.image_path)


@app.get("/api/stats")
async def get_stats():
    """Get system statistics."""
    if not taxonomy or not image_index:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return {
        "taxonomy": taxonomy.get_statistics(),
        "images": image_index.get_statistics()
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "taxonomy_loaded": taxonomy is not None,
        "embedding_index_loaded": embedding_index is not None,
        "images_loaded": image_index is not None and image_index.total_images > 0
    }


# ============================================================================
# Static File Serving (Frontend)
# ============================================================================

FRONTEND_DIST = BASE_DIR / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(str(FRONTEND_DIST / "index.html"))

    @app.get("/{path:path}")
    async def serve_frontend_routes(path: str):
        file_path = FRONTEND_DIST / path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(FRONTEND_DIST / "index.html"))


# ============================================================================
# Server Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(
        description="WikiKG Semantic Image Retrieval Server"
    )
    parser.add_argument(
        "--data", "-d",
        type=str,
        help="Path to the JSONL dataset file containing image metadata"
    )
    parser.add_argument(
        "--nodes",
        type=str,
        default=DEFAULT_NODES_PARQUET,
        help=f"Path to nodes.parquet (default: {DEFAULT_NODES_PARQUET})"
    )
    parser.add_argument(
        "--edges",
        type=str,
        default=DEFAULT_EDGES_PARQUET,
        help=f"Path to edges.parquet (default: {DEFAULT_EDGES_PARQUET})"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("HOST", "0.0.0.0"),
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=int(os.environ.get("PORT", 8000)),
        help="Port to bind to (default: 8000)"
    )

    args = parser.parse_args()

    # Set environment variables for paths so they persist across uvicorn reload
    if args.data:
        data_path = Path(args.data)
        if not data_path.is_absolute():
            data_path = Path.cwd() / data_path
        os.environ["WIKIKG_IMAGE_DATA"] = str(data_path)
        if not data_path.exists():
            print(f"Warning: Dataset file not found: {data_path}")

    # Set taxonomy parquet paths
    os.environ["WIKIKG_NODES_PARQUET"] = args.nodes
    os.environ["WIKIKG_EDGES_PARQUET"] = args.edges

    print(f"\nStarting server at http://{args.host}:{args.port}")
    print(f"API docs available at http://{args.host}:{args.port}/docs")
    print(f"Taxonomy: {args.nodes}")
    if args.data:
        print(f"Dataset: {os.environ.get('WIKIKG_IMAGE_DATA')}")
    print()

    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port
    )
