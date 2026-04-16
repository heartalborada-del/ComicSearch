# ComicSearch（中文说明）

基于 FastAPI 的漫画/图像检索后端，包含：
- SQLAlchemy ORM（`manga`、`pack`、`keyword`、`pack_keyword`）
- ONNXRuntime 图像向量（优先 CPU，CLIP 风格预处理）
- Qdrant 向量检索（支持关键词 AND 过滤与漫画级聚合排序）

## 环境要求

- Python 3.11+
- Qdrant 服务
- ONNX 图像编码模型文件（例如 CLIP 图像编码器）

## 安装

```bash
pip install -r requirements.txt
```

## 配置

环境变量：

- `ONNX_MODEL_PATH`（默认：`models/clip_image_encoder.onnx`）
- `EMBEDDER_INPUT_SIZE`（默认：`224`）
- `EMBEDDER_INTRA_THREADS`（默认：`4`）
- `QDRANT_HOST`（默认：`127.0.0.1`）
- `QDRANT_PORT`（默认：`6333`）
- `QDRANT_COLLECTION`（默认：`pages`）
- `DATABASE_URL`（默认：`sqlite:///./comicsearch.db`）

## 启动 API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## `/search` 接口

`POST /search`（`multipart/form-data`）

字段：
- `image`：上传图片文件
- `keyword_ids`：可选 JSON 整数数组（`[1,2]`），最大长度 20，每个值范围 `1..2147483647`
- `robust_partial`：布尔值，默认 `true`
- `include_corners`：布尔值，默认 `true`
- `include_contrast`：布尔值，默认 `true`
- `per_view_limit`：整数，范围 `[10, 300]`，默认 `80`
- `top_k_manga`：整数，范围 `[1, 50]`，默认 `10`
- `top_k_packs`：整数，范围 `[1, 100]`，默认 `30`

上传限制：
- `image` 允许的内容类型：`image/jpeg`、`image/png`、`image/webp`
- 最大上传大小：`10MB`
- 非法参数会返回清晰的 `4xx` 错误（`400/413/415/422`）

响应格式：

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

## 数据与索引要求

Qdrant collection 的 payload 应包含：
- `manga_id`（int）
- `pack_id`（int）
- `keyword_ids`（int 数组）
- `cover_thumb_path`（string）

向量应使用与查询时相同 ONNX 模型生成，并进行 L2 归一化。

## 构建人脸裁剪子数据集

生成角色头部/人脸裁剪图及清单：

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

Manifest（JSONL）字段：
- `original_image_path`
- `crop_image_path`
- `bbox`（`[x1,y1,x2,y2]`）
- `score`

## 将全部数据集索引到 Qdrant + SQL DB

索引全量页面和可选人脸裁剪子集：

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

Payload 包含：
- `manga_id`、`pack_id`、`keyword_ids`、`cover_thumb_path`
- `page_no`、`page_path`、`source_type`
- 裁剪图相关元数据（如有）：`crop_bbox`、`crop_score`、`crop_original_path`

可选参数：
- `--keyword-map /abs/path/keyword_map.json`，格式：`{"<abs page path>": [1,2]}`
- `--dataset-metadata /abs/path/metadata.json`，格式支持：
  - 全局标签：`{"tags": ["tagA", "tagB"]}`
  - 页面级标签：`{"<abs or relative page path>": {"tags": ["tagA"]}}`
- `--tag-id-map /abs/path/tag_id_map.json`，格式：`{"tagA": 1, "tagB": 2}`
  - metadata 中的标签会通过该映射转为 `keyword_ids`
- `--db-url sqlite:///./comicsearch.db`（或任意 SQLAlchemy URL）用于常规 DB 索引（`manga/pack/keyword/pack_keyword`）
- `--reset-state` 从头重新索引

## ORM 与数据库

- 模型定义在 `app/models.py`
- DB 会话/初始化在 `app/db.py`
- 初始化表：

```python
from app.db import init_db
init_db()
```
