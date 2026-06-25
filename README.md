# ComicSearch

FastAPI-based manga/comic image search backend with:
- SQLAlchemy ORM (`pack`, `keyword`, `pack_keyword`, `tag_id_map`)
- ONNXRuntime image embeddings (CPU-first, CLIP-style preprocessing)
- Qdrant vector retrieval with AND keyword filtering and manga-level ranking
- Vue 3 + Vuetify 3 frontend (Material You design, responsive)

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

## Qdrant Setup

ComicSearch depends on a Qdrant vector database for image retrieval. Install and start Qdrant first.

### Option 1: Docker (recommended)

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_data:/qdrant/storage" \
  qdrant/qdrant
```

### Option 2: Native install

```bash
# Download Qdrant binary
wget https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-pc-windows-msvc.zip
# Extract and run
./qdrant.exe
```

### Create Collection

After Qdrant is running, use the bundled script to create the collection:

```bash
python scripts/setup_qdrant.py \
  --qdrant-url http://127.0.0.1:6333 \
  --collection pages \
  --vector-size 512
```

Parameters:
- `--qdrant-url`: Qdrant service URL (default `http://127.0.0.1:6333`)
- `--collection`: Collection name (default `pages`, must match `config.toml`)
- `--vector-size`: Vector dimension (default `512`, for CLIP-ViT-B-16)

The collection host, port, and name must match the `[qdrant]` section in `config.toml`.

## Configuration

You can configure startup parameters with a TOML file.

- Default config file: `config.toml`
- Copy `config.example.toml` to `config.toml` as a starting point
- Override config location with `COMICSEARCH_CONFIG=/path/to/config.toml`
- Environment variables still work and override values from the TOML file

Example `config.toml`:

```toml
[embedder]
onnx_path = "models/clip_image_encoder.onnx"
input_size = 224
intra_threads = 4

[cors]
allow_origins = ["*"]

[qdrant]
host = "127.0.0.1"
port = 6333
collection = "pages"

[database]
url = "sqlite:///./comicsearch.db"

[search_defaults]
robust_partial = true
include_corners = true
include_contrast = false
per_view_limit = 80
top_k_manga = 10

[ehentai]
proxy = "http://127.0.0.1:7890"
is_exhentai = true
archive_extract_root = "comics/origin/ehentai"
allow_archive_fallback = false
face_crop_model = "models/yolov8n.pt"
face_crop_device = "cpu"
face_crop_confidence_threshold = 0.35
face_crop_expand_ratio = 0.15
face_crop_min_size = 48
face_crop_max_detections_per_image = 6
download_timeout_seconds = 60

[ehentai.cookies]
ipb_member_id = "your_member_id"
ipb_pass_hash = "your_pass_hash"
igneous = "your_igneous"
sk = "your_sk"
hath_perks = "your_hath_perks"
```

Environment variables:

- `ONNX_MODEL_PATH` (default: `models/clip_image_encoder.onnx`)
- `EMBEDDER_INPUT_SIZE` (default: `224`)
- `EMBEDDER_INTRA_THREADS` (default: `4`)
- `QDRANT_HOST` (default: `127.0.0.1`)
- `QDRANT_PORT` (default: `6333`)
- `QDRANT_COLLECTION` (default: `pages`)
- `DATABASE_URL` (default: `sqlite:///./comicsearch.db`)

Ehentai submission endpoint:

- Async task submission only (see “Async task mode” below)
- It uses the Ehentai proxy/cookie/crop settings from `[ehentai]` in the TOML config
- If the same `gid + gallerykey` has already been imported, the endpoint returns `status=duplicate`
- If the `gallerykey` changed for the same `gid`, the existing record is updated and ingestion continues

Duplicate/update decision logic (simplified):

- Duplicate:
  - `existing_gallery is not None and existing_gallery.current_token == resolved_token and existing_gallery.current_gid == resolved_gid`
- Gallery key updated (same `gid`, different key):
  - `existing_gallery is not None and existing_gallery.current_token != resolved_token`
- New gallery:
  - `existing_gallery is None`

Notes:

- `existing_gallery` is loaded by `resolved_gid` primary key, so `current_gid == resolved_gid` is normally already guaranteed when `existing_gallery` is not `None`.
- The duplicate branch returns early with `status=duplicate`; the update branch cleans old points/datasets for this `gid` and then continues ingestion.

Async task mode:

- `POST /api/ehentai/import/tasks` submits the same JSON body and returns `202` with `task_id`
- `GET /api/tasks/{task_id}` returns task status (`pending`, `running`, `success`, `failed`) and result/error
- `GET /api/tasks?limit=50&status=running` lists recent tasks with optional status filter
- `POST /api/tasks/{task_id}/cancel` requests cancellation for a pending/running task
- Unfinished tasks (`pending`/`running`) are persisted and resumed automatically after process restart

The `/search` endpoint also reads these optional defaults from `[search_defaults]` in the TOML config when the request omits them:

- `robust_partial`
- `include_corners`
- `include_contrast`
- `per_view_limit`
- `top_k_manga`

## Run API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

CORS is enabled by default (`allow_origins = ["*"]`). Configure allowed origins in `[cors]` section of `config.toml`.

## Frontend

A Vue 3 + Vuetify 3 frontend with Material You (Material Design 3) design language, supporting mobile, tablet, and desktop.

### Tech Stack

- Vue 3 (Composition API, `<script setup>`)
- Vuetify 3 (M3 themes, tonal elevation, responsive navigation)
- TypeScript
- Vite
- Pinia (state management)
- Vue Router
- Native `fetch` (no axios)

### Frontend Setup

```bash
cd frontend
npm install
npm run dev         # Development server (http://localhost:3000)
npm run build       # Production build
npm run preview     # Preview production build
npm run type-check  # TypeScript type checking
npm run lint        # ESLint
```

### Auto-Build & Integrated Serving

The backend can automatically build and serve the frontend — no separate dev server needed.

Configure in `config.toml`:

```toml
[frontend]
enabled = true       # Serve frontend from FastAPI
dist_dir = "frontend/dist"  # Built output directory
auto_build = true    # Auto-run `npm install && npm run build` if dist is missing
source_dir = "frontend"    # Frontend source directory
```

When `enabled = true` and `auto_build = true`, the backend will:
1. Check if `frontend/dist/index.html` exists
2. If not, run `npm install` then `npm run build` automatically
3. Mount the built static files and serve them at the same origin as the API
4. SPA routes (e.g. `/`, `/info/1`, `/tasks`) are served by `index.html` with fallback

This means you can start everything with a single command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then visit `http://localhost:8000` — both API and frontend are served from the same port.

To disable the frontend and run API-only (headless mode), set in `config.toml`:

```toml
[frontend]
enabled = false
```

When disabled, the root path `/` returns API info instead of the frontend. Use this mode when:
- Developing with `npm run dev` as a separate frontend dev server
- Serving the frontend via Nginx or another external web server (see Nginx deployment below)
- Running as a pure API service without a web UI

### Nginx Deployment (Frontend + Backend Separate)

For production with Nginx serving the frontend and reverse-proxying the API:

1. Build the frontend:

```bash
cd frontend && npm run build
```

2. Copy `frontend/dist/` to your server (e.g. `/var/www/comicsearch`)

3. Use the provided Nginx config template (`frontend/nginx.conf.example`):

```nginx
server {
    listen 80;
    server_name your-domain.com;
    root /var/www/comicsearch;
    index index.html;

    # Static assets with long cache (Vite content-hash filenames)
    location /assets/ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # API proxy
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        client_max_body_size 12m;
    }

    # SPA fallback — critical for /tasks, /info/123 etc. to work on refresh
    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

The key line is `try_files $uri $uri/ /index.html;` — it ensures that any route
not matching a real file (like `/tasks`, `/info/123`) falls back to `index.html`,
so the Vue Router can handle the path client-side. Without this, refreshing
`/tasks` would return 404 because Nginx looks for a file at that path.

Set `frontend.enabled = false` in `config.toml` when using Nginx, since Nginx
handles static file serving instead of FastAPI.

### Environment Variables

Create `.env.development` and `.env.production` in `frontend/`:

- `VITE_API_BASE_URL` — API base URL (dev: `/api`, proxied to `http://localhost:8000`)
- `VITE_IMAGE_BASE_URL` — External image server base URL for cover thumbnails and page previews

### Pages

| Route | Page | Description |
|-------|------|-------------|
| `/` | Search | Upload image to search for similar manga |
| `/info/:id` | Detail | View pack metadata, keywords, and cover |
| `/import` | Import | Submit E-Hentai URLs for async import |
| `/tasks` | Tasks | View, filter, and cancel async tasks |

### Responsive Design

- **Mobile** (`<960px`): Bottom navigation bar, single-column grid
- **Tablet** (`960px`+): Side navigation drawer (rail mode), 2-3 column grid
- **Desktop** (`1280px`+): Expandable side navigation, 3-4 column grid
- Touch-friendly: minimum 44×44px tap targets, no hover-dependent logic on mobile

### Theme

- Material You (M3) color system with light/dark themes
- Theme toggle persisted in `localStorage`
- Follows system preference on first visit

## `/search` endpoint

`POST /api/search` (multipart/form-data)

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
- `GET /api/info/{id}`
- `GET /api/info?id=123`

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
- `page_no`, `source_type`
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
