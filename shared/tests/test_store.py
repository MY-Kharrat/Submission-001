"""Storage-layer idempotency.

AGENTS.md requires duplicate ingests to be handled idempotently because n8n
fires triggers more than once. This pins the guarantee at both layers it is
supposed to hold: the application-level hash short-circuit, and the UNIQUE
constraint underneath it.
"""

import pytest

from shared import store
from shared.schemas import ProspectResearch, Tender


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "test.db")
    store.init_db()
    yield


def _tender(tender_id="t-1", title="Rail Signaling", issuer="MetroDOT", status="new"):
    from datetime import date, datetime, timezone

    return Tender(
        id=tender_id,
        title=title,
        issuer=issuer,
        sector="Security Assessment & Penetration Testing",
        requirements=["OWASP top 10 testing"],
        deadline=date(2026, 12, 1),
        raw_text="Procurement notice for rail signaling security review.",
        source="simulated_feed",
        detected_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        status=status,
    )


def test_dedup_hash_is_stable_for_the_same_title_and_issuer():
    from detection.normalize import compute_dedup_hash

    a = compute_dedup_hash("Rail Signaling", "MetroDOT")
    b = compute_dedup_hash("Rail Signaling", "MetroDOT")
    assert a == b


def test_dedup_hash_differs_across_issuers_and_titles():
    from detection.normalize import compute_dedup_hash

    base = compute_dedup_hash("Rail Signaling", "MetroDOT")
    assert base != compute_dedup_hash("Rail Signaling", "OtherDOT")
    assert base != compute_dedup_hash("Road Works", "MetroDOT")


def test_find_by_hash_round_trips_a_saved_tender():
    store.save_tender(_tender(), "hash-1")
    found = store.find_by_hash("hash-1")
    assert found is not None
    assert found.id == "t-1"
    assert found.requirements == ["OWASP top 10 testing"], "JSON column must decode"


def test_find_by_hash_returns_none_for_unknown_hash():
    assert store.find_by_hash("never-seen") is None


def _research(tender_id="t-1", notes="precious"):
    return ProspectResearch(
        tender_id=tender_id, issuer="MetroDOT", sector="Transit",
        estimated_revenue="$1.2B", past_projects=["Signaling"],
        key_partners=["Stellar Civil"], notes=notes, confidence="medium",
        sources=["https://mdot.example"],
    )


def test_resaving_the_same_tender_updates_in_place_without_duplicating():
    """A re-ingest of the same id is an update, not a second row."""
    store.save_tender(_tender(status="new"), "hash-1")
    store.save_tender(_tender(status="processed"), "hash-1")
    assert len(store.list_tenders()) == 1
    assert store.get_tender("t-1").status == "processed"


# --- Re-ingest must never destroy research -------------------------------
# research.tender_id is ON DELETE CASCADE, so any statement that deletes and
# reinserts the tender row silently takes the computed research with it. These
# tests exist because `INSERT OR REPLACE` did exactly that on every re-ingest,
# and the tender row still looked correct, so the loss was invisible.

def test_resaving_a_tender_preserves_its_research():
    """The core regression: a status transition must not wipe research."""
    store.save_tender(_tender(status="new"), "hash-1")
    store.save_research("t-1", _research())

    store.save_tender(_tender(status="processed"), "hash-1")

    assert store.get_tender("t-1").status == "processed"
    kept = store.get_research("t-1")
    assert kept is not None, "re-saving a tender destroyed its research"
    assert kept.key_partners == ["Stellar Civil"]
    assert kept.confidence == "medium"


def test_duplicate_ingest_with_a_new_id_keeps_the_original_id_and_research():
    """Each ingest mints a random uuid, so a duplicate arrives under a new id.

    The first id must win, otherwise research already keyed to it is orphaned
    and a second row would appear for one document.
    """
    store.save_tender(_tender(tender_id="t-original"), "hash-1")
    store.save_research("t-original", _research("t-original"))

    store.save_tender(_tender(tender_id="t-duplicate"), "hash-1")

    assert store.get_tender("t-original") is not None
    assert store.get_tender("t-duplicate") is None, "duplicate created a second row"
    assert len(store.list_tenders()) == 1
    assert store.get_research("t-original") is not None


def test_edited_tender_with_same_id_refreshes_hash_and_keeps_research():
    """A sheet edit changes title -> new hash, same id. Must not 500 or lose data."""
    store.save_tender(_tender(title="Rail Signaling"), "hash-1")
    store.save_research("t-1", _research())

    from detection.normalize import compute_dedup_hash

    new_hash = compute_dedup_hash("Rail Signaling (Revised)", "MetroDOT")
    assert new_hash != "hash-1"
    store.save_tender(_tender(title="Rail Signaling (Revised)"), new_hash)

    updated = store.get_tender("t-1")
    assert updated is not None and updated.title == "Rail Signaling (Revised)"
    assert store.find_by_hash(new_hash) is not None
    assert store.get_research("t-1") is not None, "an edited tender lost its research"


def test_reingest_refreshes_every_mutable_column():
    """REPLACE reset omitted columns to defaults; DO UPDATE SET must be explicit."""
    store.save_tender(_tender(), "hash-1")
    changed = _tender(
        title="New Title", issuer="New Issuer", status="processed",
    )
    changed = changed.model_copy(update={
        "sector": "Custom Software Development",
        "requirements": ["Updated requirement"],
    })
    store.save_tender(changed, "hash-2")

    stored = store.get_tender("t-1")
    assert stored.title == "New Title"
    assert stored.issuer == "New Issuer"
    assert stored.sector == "Custom Software Development"
    assert stored.requirements == ["Updated requirement"]
    assert stored.status == "processed"
    assert stored.raw_text == _tender().raw_text, "untouched column was blanked"


def test_research_is_replaced_not_duplicated_for_one_tender():
    """Re-running research must overwrite, since tender_id is the primary key."""
    research = ProspectResearch(
        tender_id="t-1", issuer="MetroDOT", sector="Transit",
        estimated_revenue="$1.2B", past_projects=["Signaling"],
        key_partners=["Stellar Civil"], notes="", confidence="medium",
        sources=["https://mdot.example"],
    )
    store.save_tender(_tender(), "hash-1")
    store.save_research("t-1", research)
    research.confidence = "high"
    research.notes = "second pass"
    store.save_research("t-1", research)

    stored = store.get_research("t-1")
    assert stored.confidence == "high"
    assert stored.notes == "second pass"
    assert stored.sources == ["https://mdot.example"]


def test_deleting_research_reports_whether_a_row_went_away():
    research = ProspectResearch(
        tender_id="t-1", issuer="MetroDOT", sector=None, estimated_revenue=None,
        past_projects=[], key_partners=[], notes="", confidence="low", sources=[],
    )
    store.save_tender(_tender(), "hash-1")
    store.save_research("t-1", research)
    assert store.delete_research("t-1") is True
    assert store.delete_research("t-1") is False, "second delete must report no-op"
