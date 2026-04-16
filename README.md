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
- `keyword_ids`: optional JSON int array (`[1,2]`), max length 20, each value in `1..2147483647`
- `robust_partial`: bool, default `true`
- `include_corners`: bool, default `true`
- `include_contrast`: bool, default `true`
- `per_view_limit`: int in `[10, 300]`, default `80`
- `top_k_manga`: int in `[1, 50]`, default `10`
- `top_k_packs`: int in `[1, 100]`, default `30`

Upload constraints:
- allowed `image` content type: `image/jpeg`, `image/png`, `image/webp`
- max upload size: `10MB`
- invalid values return clear `4xx` errors (`400/413/415/422`)

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

## Build face-crop subset dataset

Generate cropped character head/face images plus manifest:

```bash
python scripts/build_face_crops.py \
  --input-root /data/manga_pages \
  --output-crop-root /data/face_crops \
  --output-manifest /data/face_crops/manifest.jsonl \
  --confidence-threshold 0.35 \
  --min-crop-size 48 \
  --max-detections-per-image 6 \
  --bbox-expand-ratio 0.15
```

Manifest JSONL fields:
- `original_image_path`
- `crop_image_path`
- `bbox` (`[x1,y1,x2,y2]`)
- `score`

## Index all datasets into Qdrant

Index full pages and optional face-crop subset:

```bash
python scripts/index_all_datasets.py \
  --datasets-root /data/manga_pages \
  --face-crops-manifest /data/face_crops/manifest.jsonl \
  --qdrant-url http://127.0.0.1:6333 \
  --collection pages \
  --onnx-model models/clip_image_encoder.onnx \
  --embed-batch-size 64 \
  --upsert-batch-size 512 \
  --resume-state .cache/index_all_datasets_state.json
```

Payload includes:
- `manga_id`, `pack_id`, `keyword_ids`, `cover_thumb_path`
- `page_no`, `page_path`, `source_type`
- crop metadata when applicable: `crop_bbox`, `crop_score`, `crop_original_path`

Optional config:
- `--keyword-map /abs/path/keyword_map.json` where JSON format is `{"<abs page path>": [1,2]}`
- `--reset-state` to re-index everything from scratch

## ORM and DB

- Models are in `app/models.py`
- DB session/init helpers are in `app/db.py`
- Initialize tables:

```python
from app.db import init_db
init_db()
```
