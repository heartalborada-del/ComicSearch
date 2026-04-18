# ComicSearch

FastAPI-based manga/comic image search backend with:
- SQLAlchemy ORM (`pack`, `keyword`, `pack_keyword`, `tag_id_map`)
- ONNXRuntime image embeddings (CPU-first, CLIP-style preprocessing)
- Qdrant vector retrieval with AND keyword filtering and manga-level ranking

中文说明请查看：[README.zh-CN.md](README.zh-CN.md)

## Requirements

- Python 3.11+
- Qdrant server
- ONNX image encoder model file (e.g. CLIP image encoder)

## Install

```bash
pip install -r requirements.txt
```

Backend-specific alternatives:

- CUDA (NVIDIA):

```bash
pip install -r requirements-cuda.txt
```

- DirectML (Windows GPU):

```bash
pip install -r requirements-directml.txt
```

Do not install `onnxruntime-gpu` and `onnxruntime-directml` in the same environment. Use separate virtual environments for each backend.

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
- `include_contrast`: bool, default `false`
- `per_view_limit`: int in `[10, 300]`, default `80`
- `top_k_manga`: int in `[1, 50]`, default `10`

Upload constraints:
- allowed `image` content type: `image/jpeg`, `image/png`, `image/webp`
- max upload size: `10MB`
- invalid values return clear `4xx` errors (`400/413/415/422`)

Response format:

```json
{
  "best_manga": {"pack_id": 101, "score": 0.42, "hits": 4, "top1_score": 0.91, "top_page_no": 12},
  "confidence": "high",
  "candidate_manga": []
}
```

`top_page_no` is the `page_no` of the highest-scoring hit in that candidate pack. It can be `null` if `page_no` is missing in payload.

## `/info` endpoint

Both forms are supported:
- `GET /info/{id}`
- `GET /info?id=123`

Used to query pack metadata by `pack_id`.

Success response example:

```json
{
  "pack_id": 11,
  "title": "demo pack",
  "source": "https://example.com/demo-pack",
  "keyword_ids": [100, 101],
  "keywords": [
    {"id": 100, "name": "action"},
    {"id": 101, "name": "romance"}
  ]
}
```

Error response:
- Returns `404` when the pack does not exist (`pack not found: {id}`)

## Data/indexing expectations

Qdrant collection payload should include:
- `pack_id` (int)
- `keyword_ids` (int array)
- `cover_thumb_path` (string)

Pack source is stored in SQL as `pack.source` and returned by `GET /info`.

Vectors should be L2-normalized embeddings from the same ONNX model used at query time.

## Build face-crop subset dataset

Generate cropped character head/face images plus manifest:

```bash
python scripts/build_face_crops.py \
  --input-root /data/manga_pages \
  --output-crop-root /data/face_crops \
  --output-manifest /data/face_crops/manifest.jsonl \
  --yolo-model /path/to/yolo_model.pt \
  --yolo-classes 0 \
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

Notes:
- `build_face_crops.py` supports one-level pack directory recursion under `--input-root`.
- Crop images preserve the original directory structure relative to `--input-root`.

## Index all datasets into Qdrant + SQL DB

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
- `pack_id`, `keyword_ids`, `cover_thumb_path`
- `page_no`, `page_path`, `source_type`
- crop metadata when applicable: `crop_bbox`, `crop_score`, `crop_original_path`

Optional config:
- metadata tags are mapped to `keyword_ids` (case-insensitive)
  - if a tag is missing in DB `tag_id_map`, a new `keyword_id` is auto-assigned and persisted into DB `tag_id_map` + `keyword`
  - if DB mappings are duplicated or conflicting, existing DB mappings are treated as authoritative
- `--tag-map-output /abs/path/tag_id_map.json` to export the effective map (DB map + auto-added tags); defaults to project root `tag_id_map.json`
- `--db-url sqlite:///./comicsearch.db` (or any SQLAlchemy URL) for normal DB indexing (`pack/keyword/pack_keyword/tag_id_map`)
- `--reset-state` to re-index everything from scratch

Each dataset root must contain a `ComicInfo.xml` file with this shape:

```xml
<?xml version="1.0" encoding="utf-8"?>
<ComicInfo>
  <Title>album title</Title>
  <Tags>tagA, tagB</Tags>
  <Web>https://example.com/source</Web>
</ComicInfo>
```

The `Title` value is stored in `pack.title`.
The `Tags` value is split by comma, trimmed, and mapped to `keyword_ids` (DB mappings are used first, and missing tags are auto-added).
The `Web` value is stored in `pack.source`; if `Web` is empty, `URL` is used instead.
The effective mapping is exported to `tag_id_map.json` for review and reuse.

Dataset root discovery rules:
- for each `--datasets-root`, if `ComicInfo.xml` exists in that directory, it is treated as a dataset root
- first-level subdirectories with `ComicInfo.xml` are also auto-discovered as dataset roots

## ORM and DB

- Models are in `app/models.py`
- DB session/init helpers are in `app/db.py`
- Initialize tables:

```python
from app.db import init_db
init_db()
```
