import pytest

from rag.chunking import chunk_document, sliding_window, split_into_paragraphs
from rag.structured_chunking import (
    chunk_cv_record,
    chunk_olivesoft_project_record,
    chunk_project_record,
    chunk_tool_record,
)


def test_document_chunking_splits_sections_and_caps_size():
    text = "Introductory paragraph with enough content.\n\n" + ("long text " * 250)
    chunks = chunk_document(text)
    assert len(chunks) >= 3
    assert all(0 < len(chunk) <= 1000 for chunk in chunks)


def test_sliding_window_rejects_non_advancing_overlap():
    with pytest.raises(ValueError, match="overlap"):
        sliding_window("text", max_chars=100, overlap=100)


def test_split_rejects_non_string():
    with pytest.raises(TypeError):
        split_into_paragraphs(None)  # type: ignore[arg-type]


def test_cv_chunk_contains_record_metadata():
    record = {
        "cv_id": "CV-1",
        "name": "Amina",
        "role": "Backend Developer",
        "years_experience": 5,
        "skills": ["Python", "FastAPI"],
        "certifications": ["AWS Certified Developer"],
        "past_projects": [
            {
                "project_name": "Citizen Portal",
                "client_sector": "public administration",
                "description": "Built secure APIs.",
                "tech_stack": ["Python", "FastAPI"],
            }
        ],
    }
    chunks = chunk_cv_record(record)
    assert len(chunks) == 2
    assert all(chunk["record_id"] == "CV-1" for chunk in chunks)
    assert chunks[0]["record_meta"]["skills"] == ["Python", "FastAPI"]


def test_project_chunk_rejects_bad_technology_list():
    record = {
        "project_id": "P-1",
        "summary": "Portal",
        "client_sector": "public",
        "tech_stack": "Python",
        "outcome": "Delivered",
    }
    with pytest.raises(ValueError, match="tech_stack"):
        chunk_project_record(record)


def test_cv_chunk_rejects_negative_experience():
    with pytest.raises(ValueError, match="years_experience"):
        chunk_cv_record(
            {
                "cv_id": "CV-1",
                "name": "Amina",
                "role": "Developer",
                "years_experience": -1,
            }
        )


def test_tool_chunk_contains_capability_and_compliance_metadata():
    chunks = chunk_tool_record({
        "tool_id": "TOOL-1",
        "name": "Healthcare Integration Pack",
        "category": "Custom Software Development",
        "description": "Secure interoperability patterns.",
        "capabilities": ["FHIR integration", "audit trails"],
        "technologies": ["HL7 FHIR", "FastAPI"],
        "use_cases": ["patient portals"],
        "certifications": ["GDPR"],
    })
    assert chunks[0]["record_id"] == "TOOL-1"
    assert "HL7 FHIR" in chunks[0]["content"]
    assert chunks[0]["record_meta"]["certifications"] == ["GDPR"]


def test_olivesoft_project_keeps_status_priority_and_source_evidence():
    chunks = chunk_olivesoft_project_record({
        "id": "olivesoft-027",
        "name": "OLIA agentic platform",
        "relationship": "pfe_internship_project",
        "edition": "2026",
        "domain": ["generative AI", "RAG"],
        "description": "Agentic AI platform.",
        "tech_stack": {"technologies": ["FastAPI", "React"]},
        "outcome": "proposed",
        "sources": [{"file": "pfe2026.pdf", "page": 4, "tier": 1}],
        "confidence": "high",
    })
    metadata = chunks[0]["record_meta"]
    assert metadata["project_origin"] == "olivesoft"
    assert metadata["evidence_status"] == "proposed"
    assert metadata["is_completed"] is False
    assert metadata["matching_priority_boost"] == 0.08
    assert metadata["source_references"][0]["page"] == 4
