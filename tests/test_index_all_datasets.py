import tempfile
import unittest
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.models import Base, Keyword, Manga, Pack, PackKeyword, TagIdMap
from scripts.index_all_datasets import (
    IndexItem,
    build_keyword_ids,
    discover_dataset_roots,
    load_dataset_metadata,
    load_tag_id_map,
    resolve_effective_tag_maps,
    upsert_db_records,
    validate_used_keyword_ids,
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

    def test_load_dataset_metadata_per_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root1 = Path(tmp) / "dataset1"
            root2 = Path(tmp) / "dataset2"
            root1.mkdir(parents=True, exist_ok=True)
            root2.mkdir(parents=True, exist_ok=True)
            (root1 / "metadata.json").write_text('{"title": "album one", "tags": ["action", "school"]}', encoding="utf-8")
            (root2 / "metadata.json").write_text('{"title": "album two", "tags": ["romance"]}', encoding="utf-8")

            metadata_by_root = load_dataset_metadata([root1, root2])

            self.assertEqual(metadata_by_root[root1.resolve()].title, "album one")
            self.assertEqual(metadata_by_root[root1.resolve()].tags, ["action", "school"])
            self.assertEqual(metadata_by_root[root2.resolve()].title, "album two")
            self.assertEqual(metadata_by_root[root2.resolve()].tags, ["romance"])

    def test_build_keyword_ids(self):
        ids = build_keyword_ids(
            metadata_tags=["action", "romance"],
            tag_id_map={"action": 1, "romance": 2},
            context_path="/tmp/a.jpg",
        )
        self.assertEqual(ids, [1, 2])

    def test_build_keyword_ids_case_insensitive(self):
        page = "/tmp/a.jpg"
        ids = build_keyword_ids(
            metadata_tags=["Action", "RoMaNcE"],
            tag_id_map={"action": 1, "romance": 2},
            context_path=page,
        )
        self.assertEqual(ids, [1, 2])

    def test_build_keyword_ids_missing_tag_raises(self):
        with self.assertRaises(ValueError):
            build_keyword_ids(
                metadata_tags=["unknown-tag"],
                tag_id_map={"action": 1},
                context_path="/tmp/a.jpg",
            )


class DatasetDiscoveryTests(unittest.TestCase):
    def test_discover_dataset_roots_one_level(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "datasets"
            pack_a = base / "pack_a"
            pack_b = base / "pack_b"
            pack_a.mkdir(parents=True, exist_ok=True)
            pack_b.mkdir(parents=True, exist_ok=True)
            (pack_a / "metadata.json").write_text('{"title": "a", "tags": ["t1"]}', encoding="utf-8")
            (pack_b / "metadata.json").write_text('{"title": "b", "tags": ["t2"]}', encoding="utf-8")

            roots = discover_dataset_roots([str(base)])

            self.assertEqual(roots, [pack_a.resolve(), pack_b.resolve()])


class TagMapResolutionTests(unittest.TestCase):
    def test_resolve_effective_tag_maps_fallback_to_db(self):
        tag_to_id, id_to_tag = resolve_effective_tag_maps(
            map_tag_to_id={},
            db_tag_to_id={"action": 7},
            db_id_to_tag={7: "Action"},
        )
        self.assertEqual(tag_to_id["action"], 7)
        self.assertEqual(id_to_tag[7], "Action")

    def test_resolve_effective_tag_maps_duplicate_id_uses_db(self):
        tag_to_id, id_to_tag = resolve_effective_tag_maps(
            map_tag_to_id={"romance": 1},
            db_tag_to_id={"action": 1},
            db_id_to_tag={1: "Action"},
        )
        self.assertEqual(tag_to_id["action"], 1)
        self.assertNotIn("romance", tag_to_id)
        self.assertEqual(id_to_tag[1], "Action")

    def test_resolve_effective_tag_maps_duplicate_id_without_db_raises(self):
        with self.assertRaises(ValueError):
            resolve_effective_tag_maps(
                map_tag_to_id={"action": 1, "romance": 1},
                db_tag_to_id={},
                db_id_to_tag={},
            )


class DbUpsertTests(unittest.TestCase):
    def test_upsert_db_records(self):
        session = make_session()
        items = [
            IndexItem(
                point_id="p1",
                image_path=Path("/tmp/a.jpg"),
                payload={"manga_id": 1, "pack_id": 10, "keyword_ids": [100, 101]},
                pack_title="album one",
            )
        ]
        upsert_db_records(session, items, {100: "action", 101: "romance"})
        self.assertIsNotNone(session.get(Manga, 1))
        self.assertEqual(session.get(Pack, 10).title, "album one")
        self.assertEqual(session.get(Keyword, 100).name, "action")
        self.assertEqual(session.get(TagIdMap, "action").keyword_id, 100)
        self.assertIsNotNone(session.get(PackKeyword, {"pack_id": 10, "keyword_id": 100}))

    def test_validate_used_keyword_ids_raises_for_unknown_id(self):
        items = [
            IndexItem(
                point_id="p1",
                image_path=Path("/tmp/a.jpg"),
                payload={"keyword_ids": [999]},
            )
        ]
        with self.assertRaises(ValueError):
            validate_used_keyword_ids(items, {100: "action"})


if __name__ == "__main__":
    unittest.main()
