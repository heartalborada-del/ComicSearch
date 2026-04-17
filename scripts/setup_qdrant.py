#!/usr/bin/env python
"""Initialize Qdrant collection for manga page indexing."""

import argparse
from qdrant_client import QdrantClient
from qdrant_client.http import models


def setup_qdrant(qdrant_url: str, collection_name: str, vector_size: int = 512):
    """Create Qdrant collection for manga indexing.

    Args:
        qdrant_url: Qdrant service URL (e.g. http://127.0.0.1:6333)
        collection_name: Collection name (default: "pages")
        vector_size: Vector embedding dimension (default: 512 for CLIP-ViT-B)
    """
    client = QdrantClient(url=qdrant_url)

    # 检查 collection 是否已存在
    try:
        info = client.get_collection(collection_name)
        print(f"Collection '{collection_name}' already exists.")
        print(f"  Points: {info.points_count}")
        print(f"  Vector size: {info.config.params.vectors.size}")
        return
    except Exception:
        pass

    # 创建 collection
    print(f"Creating collection '{collection_name}'...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
        payload_schema={
            "pack_id": models.PayloadSchemaType.INTEGER,
            "keyword_ids": models.PayloadSchemaType.INTEGER,
            "page_no": models.PayloadSchemaType.INTEGER,
            "source_type": models.PayloadSchemaType.KEYWORD,
        }
    )

    print(f"✓ Collection '{collection_name}' created successfully!")
    print(f"  Vector size: {vector_size}")
    print(f"  Distance metric: COSINE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Setup Qdrant collection for manga indexing.")
    parser.add_argument(
        "--qdrant-url",
        default="http://127.0.0.1:6333",
        help="Qdrant service URL (default: http://127.0.0.1:6333)",
    )
    parser.add_argument(
        "--collection",
        default="pages",
        help="Collection name (default: pages)",
    )
    parser.add_argument(
        "--vector-size",
        type=int,
        default=512,
        help="Vector embedding dimension (default: 512 for CLIP-ViT-B-16)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_qdrant(args.qdrant_url, args.collection, args.vector_size)

