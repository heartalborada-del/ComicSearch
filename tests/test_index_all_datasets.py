import tempfile
import unittest
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.models import Base, Keyword, Manga, Pack, PackKeyword
from scripts.index_all_datasets import (
    IndexItem,
    build_keyword_ids,
    iter_manifest_original_items,
    load_dataset_metadata,
    load_tag_id_map,
    make_point_id,
    upsert_db_records,
)


def make_session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session_cls = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, class_=Session)
    return session_cls()


class TagMetadataTests(unittest.TestCase):
    def test_load_tag_id_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tag_map.json"
            path.write_text('{"action": 1, "romance": "2"}', encoding="utf-8")
            self.assertEqual(load_tag_id_map(path), {"action": 1, "romance": 2})

    def test_load_dataset_metadata_global_tags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            root.mkdir(parents=True, exist_ok=True)
            path = Path(tmp) / "metadata.json"
            path.write_text('{"tags": ["action", "school"]}', encoding="utf-8")
            tags_by_path, global_tags = load_dataset_metadata(path, [root])
            self.assertEqual(tags_by_path, {})
            self.assertEqual(global_tags, ["action", "school"])

    def test_load_dataset_metadata_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            (root / "manga_1").mkdir(parents=True, exist_ok=True)
            image = root / "manga_1" / "page_001.jpg"
            image.write_bytes(b"x")
            path = Path(tmp) / "metadata.json"
            path.write_text('{"manga_1/page_001.jpg": {"tags": ["action"]}}', encoding="utf-8")
            tags_by_path, global_tags = load_dataset_metadata(path, [root])
            self.assertEqual(global_tags, [])
            self.assertEqual(tags_by_path[str(image.resolve())], ["action"])

    def test_build_keyword_ids_merge(self):
        page = "/tmp/a.jpg"
        ids = build_keyword_ids(
            page_path=page,
            keyword_map={page: [2, 3]},
            metadata_tags_by_path={page: ["action", "romance"]},
            global_tags=[],
            tag_id_map={"action": 1, "romance": 2},
        )
        self.assertEqual(ids, [1, 2, 3])


class DbUpsertTests(unittest.TestCase):
    def test_upsert_db_records(self):
        session = make_session()
        items = [
            IndexItem(
                point_id="p1",
                image_path=Path("/tmp/a.jpg"),
                payload={"manga_id": 1, "pack_id": 10, "keyword_ids": [100, 101]},
            )
        ]
        upsert_db_records(session, items, {100: "action", 101: "romance"})
        self.assertIsNotNone(session.get(Manga, 1))
        self.assertIsNotNone(session.get(Pack, 10))
        self.assertEqual(session.get(Keyword, 100).name, "action")
        self.assertIsNotNone(session.get(PackKeyword, {"pack_id": 10, "keyword_id": 100}))


class ManifestOriginalItemsTests(unittest.TestCase):
    def test_iter_manifest_original_items_includes_unique_original_pages(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            page = tmp_path / "manga_1" / "pack_10" / "page_001.jpg"
            page.parent.mkdir(parents=True, exist_ok=True)
            page.write_bytes(b"page")

            crop = tmp_path / "crops" / "page_001_face_crop_00.jpg"
            crop.parent.mkdir(parents=True, exist_ok=True)
            crop.write_bytes(b"crop")

            manifest = tmp_path / "manifest.jsonl"
            manifest.write_text(
                "\n".join(
                    [
                        '{"original_image_path": "%s", "crop_image_path": "%s", "bbox": [1,2,3,4], "score": 0.9}'
                        % (str(page), str(crop)),
                        '{"original_image_path": "%s", "crop_image_path": "%s", "bbox": [5,6,7,8], "score": 0.8}'
                        % (str(page), str(crop)),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            items = list(
                iter_manifest_original_items(
                    manifest,
                    keyword_map={},
                    metadata_tags_by_path={},
                    global_tags=[],
                    tag_id_map={},
                )
            )

            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].point_id, make_point_id("page", page.resolve()))
            self.assertEqual(items[0].payload["source_type"], "page")
            self.assertIsNone(items[0].payload["crop_original_path"])


if __name__ == "__main__":
    unittest.main()
