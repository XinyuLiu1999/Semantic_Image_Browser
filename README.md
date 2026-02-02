# Semantic Image Retrieval System

A web-based image visualization tool that performs semantic retrieval over a JSONL dataset using the  hierarchical taxonomy.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    User Query                                │
│                  "golden retriever"                          │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Embedding-based Taxonomy Matching               │
│                                                              │
│   Query → Sentence Embedding → Cosine Similarity Search     │
│                           ↓                                  │
│   Matched Nodes: [golden_retriever, dog, canine, ...]       │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Image Retrieval                           │
│                                                              │
│   Images pre-linked to taxonomy nodes via:                   │
│   1. BM25: Rank tags by relevance to generated_caption      │
│   2. Embeddings: Match top tags to taxonomy nodes           │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Search Results                            │
│                                                              │
│   • Taxonomy path (breadcrumbs)                              │
│   • Matched categories with scores                           │
│   • Images with metadata (hover to see details)              │
└─────────────────────────────────────────────────────────────┘
```

## How It Works

### 1. Tag Relevance Scoring (BM25)

For each image, tags are ranked by their relevance to the `generated_caption` using BM25:

```
Image: {
  caption: "A golden retriever running happily in a sunny park",
  tags: ["dog", "golden retriever", "park", "outdoor", "running", "pet"]
}

BM25 Ranking:
  1. "golden retriever" → 0.95 (exact match in caption)
  2. "running" → 0.82 (appears in caption)
  3. "park" → 0.78 (appears in caption)
  4. "dog" → 0.45 (semantic but not exact)
  ...
```

### 2. Tag-to-Taxonomy Matching (Embeddings)

Top-ranked tags are matched to taxonomy nodes using sentence embeddings:

```
Tag: "golden retriever"
  ↓ Sentence Embedding
  ↓ Cosine Similarity with all taxonomy nodes

Matched Nodes:
  • n02099601 (golden retriever) → 0.95
  • n02099712 (Labrador retriever) → 0.72
  • n02084071 (dog) → 0.68
```

### 3. Query-to-Taxonomy Matching (Embeddings)

User queries are also matched to taxonomy nodes via embeddings:

```
Query: "cute puppy"
  ↓ Sentence Embedding
  ↓ Cosine Similarity Search

Results:
  • n02084071 (dog) → 0.82
  • n02085936 (puppy) → 0.78
  • n02099601 (golden retriever) → 0.65
```

## Project Structure

```
ImageNet21K/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── run.sh                       # Startup script
│
├── backend/
│   ├── main.py                  # FastAPI application
│   ├── taxonomy.py              # Taxonomy tree management
│   ├── embeddings.py            # Sentence embedding index
│   ├── bm25.py                  # BM25 tag scoring
│   ├── image_index.py           # Image data management
│   └── search.py                # Search service
│
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
│       ├── App.tsx              # Main application
│       ├── api.ts               # API client
│       └── components/
│           ├── SearchBar.tsx
│           ├── TaxonomyBreadcrumb.tsx
│           ├── MasonryGrid.tsx
│           └── ImageCard.tsx
│
└── data/
    └── images.jsonl             # Your image metadata
```

## Installation

### Prerequisites

- Python 3.9+
- Node.js 18+ (for frontend)

### Setup

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. Download NLTK WordNet data:

```bash
python -c "import nltk; nltk.download('wordnet')"
```

### Frontend Setup (Optional)

```bash
cd frontend
npm install
npm run build
cd ..
```

## Data Format

Create a JSONL file with your image metadata:

```jsonl
{"id": "img_001", "image_path": "/path/to/image.jpg", "generated_caption": "A golden retriever playing in the park", "tags": ["dog", "golden retriever", "park", "outdoor"]}
{"id": "img_002", "image_path": "/path/to/image2.jpg", "generated_caption": "A red sports car on the highway", "tags": ["car", "sports car", "red", "highway"]}
```

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique image identifier |
| `image_path` | string | Path to the image file |
| `generated_caption` | string | Text description of the image |
| `tags` | list[string] | List of semantic tags |

### Optional Fields

Any additional fields will be preserved and displayed in the image metadata hover card.

## Running the Application

### Quick Start

```bash
./run.sh --data /path/to/your/images.jsonl
```

### Manual Start

```bash
cd backend
python main.py --data /path/to/your/images.jsonl
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--data`, `-d` | Path to the JSONL dataset file |
| `--host` | Host to bind to (default: 0.0.0.0) |
| `--port`, `-p` | Port to bind to (default: 8000) |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/search?q={query}` | GET | Semantic search |
| `/api/suggestions?q={query}` | GET | Autocomplete suggestions |
| `/api/taxonomy` | GET | Taxonomy statistics |
| `/api/taxonomy/{node_id}` | GET | Node details |
| `/api/browse/{node_id}` | GET | Browse images by node |
| `/api/images/{image_id}` | GET | Image metadata |
| `/api/images/{image_id}/file` | GET | Serve image file |
| `/api/stats` | GET | System statistics |
| `/api/health` | GET | Health check |

API documentation available at http://localhost:8000/docs

## Caching

The system caches:
- **Taxonomy tree**: `data/taxonomy_cache.pkl`
- **Embedding index**: `data/embedding_index.pkl`

Delete these files to force regeneration on next startup.

## Performance Notes

- First startup builds the embedding index (~74K nodes)
- Subsequent startups load from cache
- Embedding model: `all-MiniLM-L6-v2` (~90MB, fast inference)
- Memory usage: ~2GB with full taxonomy index

## Frontend Features

- **Search Bar**: Type-ahead with taxonomy suggestions
- **Taxonomy Breadcrumbs**: Interactive path showing hierarchy
- **Masonry Grid**: Responsive image layout
- **Hover Cards**: Show image ID, tags, caption, and metadata
- **Pagination**: Navigate through large result sets

## License

This project is for research and educational purposes.
