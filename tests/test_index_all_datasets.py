import json
import tempfile
import unittest
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.models import Base, Keyword, Pack, PackKeyword, TagIdMap
from scripts.index_all_datasets import (
    IndexItem,
    build_keyword_ids,
    discover_dataset_roots,
    ensure_metadata_tags_mapped,
    export_tag_id_map,
    iter_crop_items,
    load_dataset_metadata,
    load_tag_id_map,
    normalize_title,
    normalize_tag,
    parse_page_no,
    resolve_effective_tag_maps,
    upsert_db_records,
    upsert_tag_registry,
    validate_used_keyword_ids,
)


def make_session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session_cls = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, class_=Session)
    return session_cls()


def write_comicinfo(path: Path, *, title: str, tags: list[str], web: str | None = None, url: str | None = None) -> None:
    tags_text = ", ".join(tags)
    parts = ["<?xml version=\"1.0\" encoding=\"utf-8\"?>", "<ComicInfo>", f"  <Title>{title}</Title>", f"  <Tags>{tags_text}</Tags>"]
    if web is not None:
        parts.append(f"  <Web>{web}</Web>")
    if url is not None:
        parts.append(f"  <URL>{url}</URL>")
    parts.append("</ComicInfo>")
    path.write_text("\n".join(parts), encoding="utf-8")


class TagMetadataTests(unittest.TestCase):
    def test_normalize_title_removes_trailing_bracket_tags(self):
        self.assertEqual(normalize_title("作品名 [英訳] [DL版]"), "作品名")

    def test_normalize_title_keeps_fully_bracketed_title(self):
        self.assertEqual(normalize_title("【推しの皮】"), "【推しの皮】")

    def test_normalize_title_keeps_non_trailing_or_non_bracket_parts(self):
        self.assertEqual(normalize_title("[合集] 作品名"), "[合集] 作品名")

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
            write_comicinfo(root1 / "ComicInfo.xml", title="album one", tags=["action", "school"], web="https://example.com/1")
            write_comicinfo(root2 / "ComicInfo.xml", title="album two", tags=["romance"], url="https://example.com/2")

            metadata_by_root = load_dataset_metadata([root1, root2])

            self.assertEqual(metadata_by_root[root1.resolve()].title, "album one")
            self.assertEqual(metadata_by_root[root1.resolve()].tags, ["action", "school"])
            self.assertEqual(metadata_by_root[root1.resolve()].source, "https://example.com/1")
            self.assertEqual(metadata_by_root[root2.resolve()].title, "album two")
            self.assertEqual(metadata_by_root[root2.resolve()].tags, ["romance"])
            self.assertEqual(metadata_by_root[root2.resolve()].source, "https://example.com/2")

    def test_load_dataset_metadata_splits_tags_and_prefers_web_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            root.mkdir(parents=True, exist_ok=True)
            (root / "ComicInfo.xml").write_text(
                "\n".join(
                    [
                        '<?xml version="1.0" encoding="utf-8"?>',
                        "<ComicInfo>",
                        "  <Title>album</Title>",
                        "  <Tags> action, romance , ,artist: yamadori kodi </Tags>",
                        "  <Web>https://example.com/web</Web>",
                        "  <URL>https://example.com/url</URL>",
                        "</ComicInfo>",
                    ]
                ),
                encoding="utf-8",
            )

            metadata = load_dataset_metadata([root])[root.resolve()]

            self.assertEqual(metadata.tags, ["action", "romance", "artist: yamadori kodi"])
            self.assertEqual(metadata.source, "https://example.com/web")

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
            metadata_tags=["Artist: Yamadori Kodi", "female:sex toys"],
            tag_id_map={"artist:yamadori_kodi": 1, "female:sex_toys": 2},
            context_path=page,
        )
        self.assertEqual(ids, [1, 2])

    def test_normalize_tag_rules(self):
        self.assertEqual(normalize_tag("  Artist: Yamadori Kodi  "), "artist:yamadori_kodi")
        self.assertEqual(normalize_tag("female: sex toys"), "female:sex_toys")

    def test_build_keyword_ids_missing_tag_raises(self):
        with self.assertRaises(ValueError):
            build_keyword_ids(
                metadata_tags=["unknown-tag"],
                tag_id_map={"action": 1},
                context_path="/tmp/a.jpg",
            )

    def test_ensure_metadata_tags_mapped_auto_assigns_new_ids(self):
        metadata_by_root = {
            Path("/tmp/a"): type("Meta", (), {"tags": ["Action", "NewTag"]})(),
            Path("/tmp/b"): type("Meta", (), {"tags": ["newtag", "Another"]})(),
        }
        tag_to_id = {"action": 5}
        id_to_tag = {5: "action"}

        added = ensure_metadata_tags_mapped(metadata_by_root, tag_to_id, id_to_tag)

        self.assertEqual(added, 2)
        self.assertEqual(tag_to_id["newtag"], 6)
        self.assertEqual(tag_to_id["another"], 7)
        self.assertEqual(id_to_tag[6], "newtag")
        self.assertEqual(id_to_tag[7], "another")

    def test_export_tag_id_map_orders_by_keyword_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tag_id_map.json"
            export_tag_id_map(path, {"romance": 2, "action": 1})
            loaded = load_tag_id_map(path)
            self.assertEqual(list(loaded.items()), [("action", 1), ("romance", 2)])


class DatasetDiscoveryTests(unittest.TestCase):
    def test_discover_dataset_roots_one_level(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "datasets"
            pack_a = base / "pack_a"
            pack_b = base / "pack_b"
            pack_a.mkdir(parents=True, exist_ok=True)
            pack_b.mkdir(parents=True, exist_ok=True)
            write_comicinfo(pack_a / "ComicInfo.xml", title="a", tags=["t1"])
            write_comicinfo(pack_b / "ComicInfo.xml", title="b", tags=["t2"])

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
                payload={"pack_id": 10, "keyword_ids": [100, 101]},
                pack_title="album one",
                pack_source="https://example.com/album-one",
            )
        ]
        upsert_db_records(session, items, {100: "action", 101: "romance"})
        self.assertEqual(session.get(Pack, 10).title, "album one")
        self.assertEqual(session.get(Pack, 10).source, "https://example.com/album-one")
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

    def test_upsert_tag_registry_creates_keyword_and_tag_map(self):
        session = make_session()
        upsert_tag_registry(session, {101: "new_tag"})
        session.commit()

        self.assertEqual(session.get(Keyword, 101).name, "new_tag")
        tag_row = session.get(TagIdMap, "new_tag")
        self.assertEqual(int(getattr(tag_row, "keyword_id")), 101)

    def test_upsert_db_records_backfills_missing_pack_id_from_existing_title(self):
        session = make_session()
        session.add(Pack(pack_id=22, title="album one"))
        session.commit()

        items = [
            IndexItem(
                point_id="p1",
                image_path=Path("/tmp/a.jpg"),
                payload={"pack_id": None, "keyword_ids": [100]},
                pack_title="album one",
                pack_source="https://example.com/album-one",
            )
        ]

        upsert_db_records(session, items, {100: "action"})

        self.assertEqual(items[0].payload["pack_id"], 22)
        self.assertIsNotNone(session.get(PackKeyword, {"pack_id": 22, "keyword_id": 100}))

    def test_upsert_db_records_backfills_missing_pack_id_by_creating_pack(self):
        session = make_session()
        session.add(Pack(pack_id=10, title="existing"))
        session.commit()

        items = [
            IndexItem(
                point_id="p2",
                image_path=Path("/tmp/b.jpg"),
                payload={"pack_id": None, "keyword_ids": [101]},
                pack_title="new album",
                pack_source="https://example.com/new-album",
            )
        ]

        upsert_db_records(session, items, {101: "romance"})

        new_pack_id = items[0].payload["pack_id"]
        self.assertIsInstance(new_pack_id, int)
        self.assertEqual(session.get(Pack, int(new_pack_id)).title, "new album")
        self.assertIsNotNone(session.get(PackKeyword, {"pack_id": int(new_pack_id), "keyword_id": 101}))


class ParsePageNoTests(unittest.TestCase):
    def test_parse_page_no_numeric_and_prefixed_numeric(self):
        all_names = ["_010.webp", "_009.webp", "_001.webp", "_002.webp"]
        self.assertEqual(parse_page_no("_001.webp", all_names), 1)
        self.assertEqual(parse_page_no("_002.webp", all_names), 2)
        self.assertEqual(parse_page_no("_009.webp", all_names), 3)
        self.assertEqual(parse_page_no("_010.webp", all_names), 4)

    def test_parse_page_no_alphabetic_fallback(self):
        all_names = ["c.webp", "a.webp", "b.webp"]
        self.assertEqual(parse_page_no("a.webp", all_names), 1)
        self.assertEqual(parse_page_no("b.webp", all_names), 2)
        self.assertEqual(parse_page_no("c.webp", all_names), 3)

    def test_parse_page_no_target_not_in_list(self):
        self.assertIsNone(parse_page_no("missing.webp", ["a.webp", "b.webp"]))


class CropItemTests(unittest.TestCase):
    def test_iter_crop_items_keeps_rows_without_original_pack_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = tmp_path / "dataset"
            dataset_root.mkdir(parents=True, exist_ok=True)
            write_comicinfo(dataset_root / "ComicInfo.xml", title="album", tags=[])

            valid_original = dataset_root / "pack_1" / "01.webp"
            valid_original.parent.mkdir(parents=True, exist_ok=True)
            valid_original.write_bytes(b"fake")

            invalid_original = dataset_root / "no_pack" / "01.webp"
            invalid_original.parent.mkdir(parents=True, exist_ok=True)
            invalid_original.write_bytes(b"fake")

            valid_crop = tmp_path / "crop_valid.jpg"
            valid_crop.write_bytes(b"fake")
            invalid_crop = tmp_path / "crop_invalid.jpg"
            invalid_crop.write_bytes(b"fake")

            manifest = tmp_path / "manifest.jsonl"
            manifest.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "original_image_path": str(valid_original),
                                "crop_image_path": str(valid_crop),
                                "bbox": [0, 0, 10, 10],
                                "score": 0.9,
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "original_image_path": str(invalid_original),
                                "crop_image_path": str(invalid_crop),
                                "bbox": [0, 0, 10, 10],
                                "score": 0.8,
                            },
                            ensure_ascii=False,
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            metadata_by_root = load_dataset_metadata([dataset_root])
            items = list(
                iter_crop_items(
                    manifest,
                    [dataset_root],
                    metadata_by_root,
                    {},
                    {dataset_root.resolve(): [str((valid_original).relative_to(dataset_root))]},
                )
            )

            self.assertEqual(len(items), 2)
            self.assertEqual(items[0].payload["pack_id"], 1)
            self.assertIsNone(items[1].payload["pack_id"])

    def test_iter_crop_items_skips_rows_without_metadata_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = tmp_path / "dataset"
            dataset_root.mkdir(parents=True, exist_ok=True)
            write_comicinfo(dataset_root / "ComicInfo.xml", title="album", tags=[])

            valid_original = dataset_root / "pack_1" / "01.webp"
            valid_original.parent.mkdir(parents=True, exist_ok=True)
            valid_original.write_bytes(b"fake")

            orphan_root = tmp_path / "orphan_dataset"
            orphan_root.mkdir(parents=True, exist_ok=True)
            orphan_original = orphan_root / "pack_2" / "01.webp"
            orphan_original.parent.mkdir(parents=True, exist_ok=True)
            orphan_original.write_bytes(b"fake")

            valid_crop = tmp_path / "crop_valid.jpg"
            valid_crop.write_bytes(b"fake")
            orphan_crop = tmp_path / "crop_orphan.jpg"
            orphan_crop.write_bytes(b"fake")

            manifest = tmp_path / "manifest.jsonl"
            manifest.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "original_image_path": str(valid_original),
                                "crop_image_path": str(valid_crop),
                                "bbox": [0, 0, 10, 10],
                                "score": 0.9,
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "original_image_path": str(orphan_original),
                                "crop_image_path": str(orphan_crop),
                                "bbox": [0, 0, 10, 10],
                                "score": 0.8,
                            },
                            ensure_ascii=False,
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            metadata_by_root = load_dataset_metadata([dataset_root])
            items = list(
                iter_crop_items(
                    manifest,
                    [dataset_root],
                    metadata_by_root,
                    {},
                    {dataset_root.resolve(): [str((valid_original).relative_to(dataset_root))]},
                )
            )

            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].payload["pack_id"], 1)


if __name__ == "__main__":
    unittest.main()
