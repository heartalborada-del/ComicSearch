# ComicSearch（中文说明）

基于 FastAPI 的漫画/图像检索后端，包含：
- SQLAlchemy ORM（`pack`、`keyword`、`pack_keyword`、`tag_id_map`）
- ONNXRuntime 图像向量（优先 CPU，CLIP 风格预处理）
- Qdrant 向量检索（支持关键词 AND 过滤与漫画级聚合排序）
- Vue 3 + Vuetify 3 前端（Material You 设计，多端适配）

## 环境要求

- Python 3.11+
- Qdrant 服务
- ONNX 图像编码模型文件（例如 CLIP 图像编码器）

## 安装

```bash
pip install -r requirements.txt
```

按推理后端安装（可选）：

- CUDA（NVIDIA）：

```bash
pip install -r requirements-cuda.txt
```

- DirectML（Windows GPU）：

```bash
pip install -r requirements-directml.txt
```

不要在同一个 Python 环境里同时安装 `onnxruntime-gpu` 和 `onnxruntime-directml`。如需两种后端，请分别创建虚拟环境。

## 配置

你可以使用 TOML 配置文件来设置启动参数。

- 默认配置文件：`config.toml`
- 你可以先复制 `config.example.toml` 为 `config.toml`
- 可通过 `COMICSEARCH_CONFIG=/path/to/config.toml` 指定配置文件路径
- 环境变量仍然可用，并且会覆盖 TOML 中的配置

`config.toml` 示例：

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

环境变量：

- `ONNX_MODEL_PATH`（默认：`models/clip_image_encoder.onnx`）
- `EMBEDDER_INPUT_SIZE`（默认：`224`）
- `EMBEDDER_INTRA_THREADS`（默认：`4`）
- `QDRANT_HOST`（默认：`127.0.0.1`）
- `QDRANT_PORT`（默认：`6333`）
- `QDRANT_COLLECTION`（默认：`pages`）
- `DATABASE_URL`（默认：`sqlite:///./comicsearch.db`）

Ehentai 提交接口：

- 仅支持异步任务提交（见下方“异步任务模式”）
- 会使用 TOML 配置中的 `[ehentai]` 段来设置代理、Cookie 和裁切参数
- 如果相同的 `gid + gallerykey` 已经导入过，会直接返回 `status=duplicate`
- 如果同一个 `gid` 的 `gallerykey` 发生变化，会更新已有记录并继续导入

判重/更新逻辑（简化版）：

- 判重：
  - `existing_gallery is not None and existing_gallery.current_token == resolved_token and existing_gallery.current_gid == resolved_gid`
- gallery key 更新（同 `gid`、不同 key）：
  - `existing_gallery is not None and existing_gallery.current_token != resolved_token`
- 新画廊：
  - `existing_gallery is None`

说明：

- `existing_gallery` 是通过 `resolved_gid` 主键读取的，所以当 `existing_gallery is not None` 时，`current_gid == resolved_gid` 通常已天然成立。
- 判重分支会直接返回 `status=duplicate`；更新分支会先清理该 `gid` 的旧向量点和旧数据目录，再继续导入。

异步任务模式：

- `POST /api/ehentai/import/tasks` 提交同样的 JSON，请求会返回 `202` 和 `task_id`
- `GET /api/tasks/{task_id}` 查询任务状态（`pending`、`running`、`success`、`failed`）以及结果/错误
- `GET /api/tasks?limit=50&status=running` 查询最近任务列表（可按状态过滤）
- `POST /api/tasks/{task_id}/cancel` 对 `pending/running` 任务发起取消
- 未完成任务（`pending`/`running`）会持久化，服务重启后自动继续执行

如果请求体里没有传这些字段，`/search` 也会读取 TOML 配置中的 `[search_defaults]` 作为默认值：

- `robust_partial`
- `include_corners`
- `include_contrast`
- `per_view_limit`
- `top_k_manga`

## 启动 API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

CORS 默认开启（`allow_origins = ["*"]`）。可在 `config.toml` 的 `[cors]` 段配置允许的来源。

## 前端

基于 Vue 3 + Vuetify 3 的前端，采用 Material You（Material Design 3）设计语言，支持手机、平板、桌面多端适配。

### 技术栈

- Vue 3（Composition API，`<script setup>`）
- Vuetify 3（M3 主题、tonal elevation、响应式导航）
- TypeScript
- Vite
- Pinia（状态管理）
- Vue Router
- 原生 `fetch`（不使用 axios）

### 前端安装

```bash
cd frontend
npm install
npm run dev         # 开发服务器（http://localhost:3000）
npm run build       # 生产构建
npm run preview     # 预览构建产物
npm run type-check  # TypeScript 类型检查
npm run lint        # ESLint 检查
```

### 自动构建与一体化部署

后端可以自动构建并托管前端——无需单独启动开发服务器。

在 `config.toml` 中配置：

```toml
[frontend]
enabled = true       # 由 FastAPI 托管前端
dist_dir = "frontend/dist"  # 构建产物目录
auto_build = true    # dist 不存在时自动执行 `npm install && npm run build`
source_dir = "frontend"    # 前端源码目录
```

当 `enabled = true` 且 `auto_build = true` 时，后端会：
1. 检查 `frontend/dist/index.html` 是否存在
2. 若不存在，自动执行 `npm install` 然后 `npm run build`
3. 挂载构建产物，与 API 同源提供服务
4. SPA 路由（如 `/`、`/info/1`、`/tasks`）通过 `index.html` 回退提供服务

因此只需一条命令即可启动全部服务：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

然后访问 `http://localhost:8000`——API 和前端从同一端口提供服务。

如需禁用（例如开发时使用 `npm run dev`），设置 `enabled = false`。

### 环境变量

在 `frontend/` 目录下创建 `.env.development` 和 `.env.production`：

- `VITE_API_BASE_URL` — API 基地址（开发环境为 `/api`，通过 Vite 代理转发到 `http://localhost:8000`）
- `VITE_IMAGE_BASE_URL` — 外部图片服务器基地址，用于拼接封面缩略图和页面预览

### 页面

| 路由 | 页面 | 说明 |
|------|------|------|
| `/` | 搜索 | 上传图片以图搜漫 |
| `/info/:id` | 详情 | 查看图包元数据、关键词、封面 |
| `/import` | 导入 | 提交 E-Hentai URL 进行异步导入 |
| `/tasks` | 任务 | 查看、过滤、取消异步任务 |

### 响应式设计

- **移动端**（`<960px`）：底部导航栏，单列栅格
- **平板**（`960px`+）：侧边栏导航（rail 模式），2-3 列栅格
- **桌面**（`1280px`+）：可折叠侧边栏，3-4 列栅格
- 触摸友好：最小点击区域 44×44px，移动端不依赖 hover 逻辑

### 主题

- Material You（M3）色板系统，支持亮色/暗色主题
- 主题切换持久化到 `localStorage`
- 首次访问时跟随系统偏好

## `/search` 接口

`POST /api/search`（`multipart/form-data`）

字段：
- `image`：上传图片文件
- `keyword_ids`：可选 JSON 整数数组（`[1,2]`），最大长度 20，每个值范围 `1..2147483647`
- `robust_partial`：布尔值，默认 `true`
- `include_corners`：布尔值，默认 `true`
- `include_contrast`：布尔值，默认 `false`
- `per_view_limit`：整数，范围 `[10, 300]`，默认 `80`
- `top_k_manga`：整数，范围 `[1, 50]`，默认 `10`

上传限制：
- `image` 允许的内容类型：`image/jpeg`、`image/png`、`image/webp`
- 最大上传大小：`10MB`
- 非法参数会返回清晰的 `4xx` 错误（`400/413/415/422`）

响应格式：

```json
{
  "best_manga": {"pack_id": 101, "score": 0.42, "hits": 4, "top1_score": 0.91, "top_page_no": 12},
  "confidence": "high",
  "candidate_manga": []
}
```

`top_page_no` 表示该候选包中最高分命中的 `page_no`。如果 payload 缺少 `page_no`，该值可能为 `null`。

## `/info` 接口

支持两种形式：
- `GET /api/info/{id}`
- `GET /api/info?id=123`

用于按 `pack_id` 查询图包信息。

成功响应示例：

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

错误响应：
- 找不到对应图包时返回 `404`（`pack not found: {id}`）

## 数据与索引要求

Qdrant collection 的 payload 应包含：
- `pack_id`（int）
- `keyword_ids`（int 数组）
- `cover_thumb_path`（string）

`pack.source` 会写入 SQL，并由 `GET /info` 返回。

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

说明：
- `build_face_crops.py` 支持对 `--input-root` 下的一级图包目录逐个递归处理。
- 裁剪图会保持原有目录结构（相对于 `--input-root` 的路径层级不变）。

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
- `pack_id`、`keyword_ids`、`cover_thumb_path`
- `page_no`、`source_type`
- 裁剪图相关元数据（如有）：`crop_bbox`、`crop_score`、`crop_original_path`

可选参数：
- metadata 中的标签会映射为 `keyword_ids`（标签匹配忽略大小写）
  - 若数据库 `tag_id_map` 中缺失某个标签，会自动分配新的 `keyword_id`，并写入数据库 `tag_id_map` + `keyword`
  - 若数据库存在重复或冲突映射，以数据库中已有映射为准
- `--tag-map-output /abs/path/tag_id_map.json` 导出生效映射（数据库映射 + 自动新增标签）；默认导出到项目根目录 `tag_id_map.json`
- `--db-url sqlite:///./comicsearch.db`（或任意 SQLAlchemy URL）用于常规 DB 索引（`pack/keyword/pack_keyword/tag_id_map`）
- `--reset-state` 从头重新索引

每个图集根目录都需要放一个 `ComicInfo.xml`，格式如下：

```xml
<?xml version="1.0" encoding="utf-8"?>
<ComicInfo>
  <Title>图包名称</Title>
  <Tags>tagA, tagB</Tags>
  <Web>https://example.com/source</Web>
</ComicInfo>
```

其中：
- `Title` 会写入数据库 `pack.title`
- `Tags` 会按英文逗号分割、去空白后映射为 `keyword_ids`（优先使用数据库映射，缺失时自动新增并入库）
- `Web` 会写入数据库 `pack.source`；如果 `Web` 为空，则使用 `URL`
- 本次生效映射会导出为 `tag_id_map.json`（可用于审阅与复用）

目录扫描规则：
- `index_all_datasets.py` 对每个 `--datasets-root` 会先检查自身是否是图包根目录（存在 `ComicInfo.xml`）。
- 同时支持自动发现该目录下一级子目录中的图包根目录（子目录内存在 `ComicInfo.xml`）。

## ORM 与数据库

- 模型定义在 `app/models.py`
- DB 会话/初始化在 `app/db.py`
- 初始化表：

```python
from app.db import init_db
init_db()
```
