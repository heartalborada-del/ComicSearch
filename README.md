# ComicSearch

FastAPI-based manga/comic image search backend with:
- SQLAlchemy ORM (`manga`, `pack`, `keyword`, `pack_keyword`)
- ONNXRuntime image embeddings (CPU-first, CLIP-style preprocessing)
- Qdrant vector retrieval with AND keyword filtering and manga-level ranking

## Requirements

- Python 3.11+
- Qdrant server
- ONNX image encoder model file (e.g. CLIP image encoder)

## Install

```bash
pip install -r requirements.txt
```

## Configuration

Environment variables:

- `ONNX_MODEL_PATH` (default: `models/clip_image_encoder.onnx`)
- `EMBEDDER_INPUT_SIZE` (default: `224`)
- `EMBEDDER_INTRA_THREADS` (default: `4`)
- `QDRANT_HOST` (default: `127.0.0.1`)
- `QDRANT_PORT` (default: `6333`)
- `QDRANT_COLLECTION` (default: `pages`)
- `DATABASE_URL` (default: `sqlite:///./comicsearch.db`)

## Run API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## `/search` endpoint

`POST /search` (multipart/form-data)

Fields:
- `image`: uploaded image file
- `keyword_ids`: optional JSON int array (`[1,2]`) or comma-separated (`1,2`)
- `robust_partial`: bool, default `true`
- `include_corners`: bool, default `true`
- `include_contrast`: bool, default `true`
- `per_view_limit`: int, default `80`
- `top_k_manga`: int, default `10`
- `top_k_packs`: int, default `30`

Response format:

```json
{
  "best_manga": {"manga_id": 101, "score": 0.42, "hits": 4, "top1_score": 0.91},
  "confidence": "high",
  "candidate_manga": [],
  "top_packs_in_best_manga": [
    {"pack_id": 55, "score": 0.88, "hits": 3, "cover_thumb_path": "..."}
  ]
}
```

## Data/indexing expectations

Qdrant collection payload should include:
- `manga_id` (int)
- `pack_id` (int)
- `keyword_ids` (int array)
- `cover_thumb_path` (string)

Vectors should be L2-normalized embeddings from the same ONNX model used at query time.

## ORM and DB

- Models are in `app/models.py`
- DB session/init helpers are in `app/db.py`
- Initialize tables:

```python
from app.db import init_db
init_db()
```
