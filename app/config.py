from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import tomllib

DEFAULT_CONFIG_PATH = "config.toml"
DEFAULT_ONNX_MODEL_PATH = "models/clip_image_encoder.onnx"
DEFAULT_DB_URL = "sqlite:///./comicsearch.db"
INT32_MAX = 2_147_483_647
MAX_KEYWORD_IDS = 20


@dataclass(frozen=True)
class EmbedderSettings:
    onnx_path: str = DEFAULT_ONNX_MODEL_PATH
    input_size: int = 224
    intra_threads: int = 4


@dataclass(frozen=True)
class QdrantSettings:
    host: str = "127.0.0.1"
    port: int = 6333
    collection: str = "pages"


@dataclass(frozen=True)
class DatabaseSettings:
    url: str = DEFAULT_DB_URL


@dataclass(frozen=True)
class SearchDefaults:
    robust_partial: bool = True
    include_corners: bool = True
    include_contrast: bool = False
    per_view_limit: int = 80
    top_k_manga: int = 10


@dataclass(frozen=True)
class EhentaiSettings:
    proxy: str | None = None
    is_exhentai: bool = False
    cookies: dict[str, str] = field(default_factory=dict)
    face_crop_model: str = "models/yolov8n.pt"
    face_crop_device: str = "cpu"
    face_crop_confidence_threshold: float = 0.35
    face_crop_expand_ratio: float = 0.15
    face_crop_min_size: int = 48
    face_crop_max_detections_per_image: int = 6
    download_timeout_seconds: float = 60.0
    archive_extract_root: str = "comics/origin/ehentai"
    allow_archive_fallback: bool = False


@dataclass(frozen=True)
class AppSettings:
    embedder: EmbedderSettings = field(default_factory=EmbedderSettings)
    qdrant: QdrantSettings = field(default_factory=QdrantSettings)
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    search: SearchDefaults = field(default_factory=SearchDefaults)
    ehentai: EhentaiSettings = field(default_factory=EhentaiSettings)


def _coerce_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    raise ValueError(f"{field_name} must be a boolean")


def _coerce_int(value: Any, *, field_name: str, min_value: int | None = None, max_value: int | None = None) -> int:
    if type(value) is not int:
        raise ValueError(f"{field_name} must be an integer")
    if min_value is not None and value < min_value:
        raise ValueError(f"{field_name} must be >= {min_value}")
    if max_value is not None and value > max_value:
        raise ValueError(f"{field_name} must be <= {max_value}")
    return value


def _coerce_float(value: Any, *, field_name: str, min_value: float | None = None, max_value: float | None = None) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{field_name} must be a number")
    result = float(value)
    if min_value is not None and result < min_value:
        raise ValueError(f"{field_name} must be >= {min_value}")
    if max_value is not None and result > max_value:
        raise ValueError(f"{field_name} must be <= {max_value}")
    return result


def _coerce_str(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    raise ValueError("value must be a string or null")


def _coerce_str_map(value: Any, *, field_name: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a TOML table")
    return {str(key): str(item) for key, item in value.items() if item is not None}


def _env_int(
    name: str,
    fallback: int,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return fallback
    return _coerce_int(int(raw_value), field_name=name, min_value=min_value, max_value=max_value)


def _resolve_config_path(config_path: str | os.PathLike[str] | None) -> Path | None:
    candidate = config_path or os.getenv("COMICSEARCH_CONFIG") or DEFAULT_CONFIG_PATH
    if candidate is None:
        return None
    return Path(candidate).expanduser()


def _load_toml_file(path: Path) -> dict[str, Any]:
    with path.open("rb") as fp:
        loaded = tomllib.load(fp)
    if not isinstance(loaded, dict):
        raise ValueError("config file must contain a TOML table")
    return loaded


def _resolve_relative_path(path_value: str, *, base_dir: Path) -> str:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def _get_section(data: dict[str, Any], name: str) -> dict[str, Any]:
    section = data.get(name, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"[{name}] must be a TOML table")
    return section


def load_settings(config_path: str | os.PathLike[str] | None = None) -> AppSettings:
    resolved_config_path = _resolve_config_path(config_path)
    file_data: dict[str, Any] = {}
    base_dir = Path.cwd()

    if resolved_config_path is not None and resolved_config_path.exists():
        file_data = _load_toml_file(resolved_config_path)
        base_dir = resolved_config_path.parent.resolve()

    embedder_section = _get_section(file_data, "embedder")
    qdrant_section = _get_section(file_data, "qdrant")
    database_section = _get_section(file_data, "database")
    search_section = _get_section(file_data, "search_defaults")
    ehentai_section = _get_section(file_data, "ehentai")

    embedder = EmbedderSettings(
        onnx_path=_resolve_relative_path(
            _coerce_str(embedder_section.get("onnx_path", DEFAULT_ONNX_MODEL_PATH), field_name="embedder.onnx_path"),
            base_dir=base_dir,
        ),
        input_size=_coerce_int(embedder_section.get("input_size", 224), field_name="embedder.input_size", min_value=1),
        intra_threads=_coerce_int(
            embedder_section.get("intra_threads", 4),
            field_name="embedder.intra_threads",
            min_value=1,
        ),
    )
    qdrant = QdrantSettings(
        host=_coerce_str(qdrant_section.get("host", "127.0.0.1"), field_name="qdrant.host"),
        port=_coerce_int(qdrant_section.get("port", 6333), field_name="qdrant.port", min_value=1, max_value=65535),
        collection=_coerce_str(qdrant_section.get("collection", "pages"), field_name="qdrant.collection"),
    )
    database = DatabaseSettings(
        url=_coerce_str(database_section.get("url", DEFAULT_DB_URL), field_name="database.url"),
    )
    search = SearchDefaults(
        robust_partial=_coerce_bool(search_section.get("robust_partial", True), field_name="search_defaults.robust_partial"),
        include_corners=_coerce_bool(search_section.get("include_corners", True), field_name="search_defaults.include_corners"),
        include_contrast=_coerce_bool(search_section.get("include_contrast", False), field_name="search_defaults.include_contrast"),
        per_view_limit=_coerce_int(
            search_section.get("per_view_limit", 80),
            field_name="search_defaults.per_view_limit",
            min_value=10,
            max_value=300,
        ),
        top_k_manga=_coerce_int(
            search_section.get("top_k_manga", 10),
            field_name="search_defaults.top_k_manga",
            min_value=1,
            max_value=50,
        ),
    )
    ehentai = EhentaiSettings(
        proxy=_optional_str(ehentai_section.get("proxy")),
        is_exhentai=_coerce_bool(ehentai_section.get("is_exhentai", False), field_name="ehentai.is_exhentai"),
        cookies=_coerce_str_map(ehentai_section.get("cookies", {}), field_name="ehentai.cookies"),
        face_crop_model=_resolve_relative_path(
            _coerce_str(ehentai_section.get("face_crop_model", "models/yolov8n.pt"), field_name="ehentai.face_crop_model"),
            base_dir=base_dir,
        ),
        face_crop_device=_coerce_str(ehentai_section.get("face_crop_device", "cpu"), field_name="ehentai.face_crop_device"),
        face_crop_confidence_threshold=_coerce_float(
            ehentai_section.get("face_crop_confidence_threshold", 0.35),
            field_name="ehentai.face_crop_confidence_threshold",
            min_value=0.0,
        ),
        face_crop_expand_ratio=_coerce_float(
            ehentai_section.get("face_crop_expand_ratio", 0.15),
            field_name="ehentai.face_crop_expand_ratio",
            min_value=0.0,
        ),
        face_crop_min_size=_coerce_int(
            ehentai_section.get("face_crop_min_size", 48),
            field_name="ehentai.face_crop_min_size",
            min_value=1,
        ),
        face_crop_max_detections_per_image=_coerce_int(
            ehentai_section.get("face_crop_max_detections_per_image", 6),
            field_name="ehentai.face_crop_max_detections_per_image",
            min_value=1,
        ),
        download_timeout_seconds=_coerce_float(
            ehentai_section.get("download_timeout_seconds", 60.0),
            field_name="ehentai.download_timeout_seconds",
            min_value=1.0,
        ),
        archive_extract_root=_resolve_relative_path(
            _coerce_str(
                ehentai_section.get("archive_extract_root", "comics/origin/ehentai"),
                field_name="ehentai.archive_extract_root",
            ),
            base_dir=base_dir,
        ),
        allow_archive_fallback=_coerce_bool(
            ehentai_section.get("allow_archive_fallback", False),
            field_name="ehentai.allow_archive_fallback",
        ),
    )

    env_model_path = os.getenv("ONNX_MODEL_PATH")
    if env_model_path:
        embedder = EmbedderSettings(
            onnx_path=env_model_path,
            input_size=_env_int("EMBEDDER_INPUT_SIZE", embedder.input_size, min_value=1),
            intra_threads=_env_int("EMBEDDER_INTRA_THREADS", embedder.intra_threads, min_value=1),
        )

    qdrant_host = os.getenv("QDRANT_HOST")
    qdrant_port = os.getenv("QDRANT_PORT")
    qdrant_collection = os.getenv("QDRANT_COLLECTION")
    if qdrant_host or qdrant_port or qdrant_collection:
        qdrant = QdrantSettings(
            host=_coerce_str(qdrant_host or qdrant.host, field_name="QDRANT_HOST"),
            port=_env_int("QDRANT_PORT", qdrant.port, min_value=1, max_value=65535),
            collection=_coerce_str(qdrant_collection or qdrant.collection, field_name="QDRANT_COLLECTION"),
        )

    database_url = os.getenv("DATABASE_URL")
    if database_url:
        database = DatabaseSettings(url=_coerce_str(database_url, field_name="DATABASE_URL"))

    return AppSettings(embedder=embedder, qdrant=qdrant, database=database, search=search, ehentai=ehentai)
