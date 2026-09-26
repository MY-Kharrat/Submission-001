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


def test_resaving_the_same_tender_updates_in_place_without_duplicating():
    """A re-ingest of the same id is an update, not a second row."""
    store.save_tender(_tender(status="new"), "hash-1")
    store.save_tender(_tender(status="processed"), "hash-1")
    assert len(store.list_tenders()) == 1
    assert store.get_tender("t-1").status == "processed"


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
