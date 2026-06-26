"""修补 ehentai_face_crop 点缺失的 origin_source_path。

问题：页面 point 有 origin_source_path，但人脸裁剪点没有。
当搜索时最高分 point 恰好是裁剪点时，top_page_origin_path 为 null。

此脚本从同 pack_id + page_no 的页面点复制 origin_source_path 到裁剪点。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

# ── 批量大小 ──────────────────────────────────────────────
SCROLL_BATCH_SIZE = 256


def _load_qdrant_config(config_path: str) -> dict[str, Any]:
    """从 config.toml 读取 Qdrant 连接信息（仅需 host, port, collection）。"""
    import os
    import sys
    import tomllib

    config_file = Path(config_path).expanduser()
    if not config_file.exists():
        raise FileNotFoundError(f"config file not found: {config_file}")

    with config_file.open("rb") as fp:
        data = tomllib.load(fp)

    qdrant_section = data.get("qdrant", {}) or {}
    return {
        "host": qdrant_section.get("host", "127.0.0.1"),
        "port": int(qdrant_section.get("port", 6333)),
        "collection": qdrant_section.get("collection", "pages"),
    }


def _find_page_origin_path(
    client: QdrantClient,
    collection: str,
    pack_id: int,
    page_no: int,
) -> str | None:
    """查找同一 pack_id + page_no 的 ehentai_page 点的 origin_source_path。"""
    filter_cond = qm.Filter(
        must=[
            qm.FieldCondition(key="pack_id", match=qm.MatchValue(value=pack_id)),
            qm.FieldCondition(key="page_no", match=qm.MatchValue(value=page_no)),
            qm.FieldCondition(key="source_type", match=qm.MatchValue(value="ehentai_page")),
        ]
    )
    results, _next_offset = client.scroll(
        collection_name=collection,
        scroll_filter=filter_cond,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if results:
        payload = results[0].payload or {}
        return payload.get("origin_source_path")
    return None


def _build_origin_map(
    client: QdrantClient,
    collection: str,
    crop_points: list[Any],
) -> dict[tuple[int, int], str | None]:
    """构建 (pack_id, page_no) → origin_source_path 的映射。"""
    # 去重
    keys: set[tuple[int, int]] = set()
    for point in crop_points:
        payload = point.payload or {}
        pid = payload.get("pack_id")
        pn = payload.get("page_no")
        if pid is not None and pn is not None:
            keys.add((int(pid), int(pn)))

    mapping: dict[tuple[int, int], str | None] = {}
    for pack_id, page_no in sorted(keys):
        origin_path = _find_page_origin_path(client, collection, pack_id, page_no)
        mapping[(pack_id, page_no)] = origin_path
    return mapping


def run(config_path: str, *, dry_run: bool = False) -> None:
    qconfig = _load_qdrant_config(config_path)
    client = QdrantClient(host=qconfig["host"], port=qconfig["port"])
    collection = qconfig["collection"]

    filter_missing = qm.Filter(
        must=[
            qm.FieldCondition(key="source_type", match=qm.MatchValue(value="ehentai_face_crop")),
        ],
        must_not=[
            qm.HasIdCondition(has_id=[]),  # dummy – we filter missing in code
        ],
    )

    total_updated = 0
    total_missing_origin = 0
    offset: str | int | None = None

    while True:
        results, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=filter_missing,
            limit=SCROLL_BATCH_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not results:
            break

        # 筛选出真正缺少 origin_source_path 的点
        missing: list[Any] = []
        for point in results:
            payload = point.payload or {}
            if "origin_source_path" not in payload or payload.get("origin_source_path") is None:
                missing.append(point)

        if missing:
            print(f"批次: 扫描 {len(results)} 个裁剪点, {len(missing)} 个缺少 origin_source_path")
            origin_map = _build_origin_map(client, collection, missing)

            # 批量更新 payload（只改 payload，不影响 vector）
            payload_updates: list[tuple[int | str, dict[str, Any]]] = []
            for point in missing:
                payload = point.payload or {}
                key = (int(payload["pack_id"]), int(payload["page_no"]))
                origin_path = origin_map.get(key)
                if origin_path is None:
                    total_missing_origin += 1
                    print(f"  警告: pack_id={key[0]} page_no={key[1]} 找不到页面点的 origin_source_path")
                    continue

                new_payload = {"origin_source_path": origin_path}
                if dry_run:
                    print(f"  [DRY RUN] 将更新 point_id={point.id} → origin_source_path={origin_path}")
                else:
                    payload_updates.append((point.id, new_payload))

            if payload_updates and not dry_run:
                for point_id, payload_data in payload_updates:
                    client.set_payload(
                        collection_name=collection,
                        payload=payload_data,
                        points=[point_id],
                    )
                total_updated += len(payload_updates)
                print(f"  已更新 {len(payload_updates)} 个点")

        offset = next_offset
        if offset is None:
            break

    print()
    print(f"完成！共更新 {total_updated} 个裁剪点")
    if total_missing_origin > 0:
        print(f"警告：{total_missing_origin} 个裁剪点找不到对应页面点（可能数据不完整）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="修补 ehentai_face_crop 点缺失的 origin_source_path")
    parser.add_argument(
        "--config",
        default="config.toml",
        help="TOML 配置文件路径（默认: config.toml）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅预览将要修改的内容，不实际写入",
    )
    args = parser.parse_args()
    run(args.config, dry_run=args.dry_run)
