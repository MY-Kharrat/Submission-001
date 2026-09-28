from types import SimpleNamespace

import pytest

from rag.storage import build_rows, make_chunk_id, make_source_id, replace_source_chunks


class FakeTable:
    def __init__(self, existing=None, fail_upsert=False):
        self.existing = existing or []
        self.fail_upsert = fail_upsert
        self.action = None
        self.upserts = []
        self.deletes = []

    def select(self, _columns):
        self.action = "select"
        return self

    def eq(self, _column, _value):
        return self

    def upsert(self, rows, on_conflict):
        if self.fail_upsert:
            raise RuntimeError("database unavailable")
        self.action = "upsert"
        self.upserts.extend(rows)
        assert on_conflict == "id"
        return self

    def delete(self):
        self.action = "delete"
        return self

    def in_(self, _column, values):
        self.deletes.extend(values)
        return self

    def execute(self):
        if self.action == "select":
            return SimpleNamespace(data=[{"id": value} for value in self.existing])
        return SimpleNamespace(data=[])


class FakeClient:
    def __init__(self, table):
        self.fake_table = table

    def table(self, name):
        assert name == "knowledge_chunks"
        return self.fake_table


def make_rows():
    return build_rows(
        [{"content": "first", "metadata": {"record_id": "CV-1"}}],
        [[0.0] * 384],
        doc_type="cv",
        source_id="cv:data/cvs.json",
        source_file="cvs.json",
    )


def test_chunk_ids_are_deterministic():
    assert make_chunk_id("source", 1) == make_chunk_id("source", 1)
    assert make_chunk_id("source", 1) != make_chunk_id("source", 2)


def test_source_id_can_be_explicit(tmp_path):
    assert make_source_id(tmp_path / "x.json", "cv", "internal/cvs") == "cv:internal/cvs"


def test_replace_upserts_before_deleting_stale_rows():
    rows = make_rows()
    table = FakeTable(existing=["old-id"])
    replace_source_chunks(rows, source_id=rows[0]["source_id"], client=FakeClient(table))
    assert len(table.upserts) == 1
    assert table.deletes == ["old-id"]


def test_failed_upsert_never_deletes_old_rows():
    rows = make_rows()
    table = FakeTable(existing=["old-id"], fail_upsert=True)
    with pytest.raises(RuntimeError, match="unavailable"):
        replace_source_chunks(rows, source_id=rows[0]["source_id"], client=FakeClient(table))
    assert table.deletes == []


def test_build_rows_rejects_wrong_vector_size():
    with pytest.raises(ValueError, match="dimensions"):
        build_rows(
            [{"content": "first"}],
            [[0.0] * 10],
            doc_type="cv",
            source_id="cv:data/cvs.json",
            source_file="cvs.json",
        )
