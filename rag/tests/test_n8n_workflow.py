"""Guards for the shipped sheet-driven orchestrator.

The workflow under test is ``n8n/workflows/Phase 1 RFP Orchestrator.json``:
Schedule Trigger -> Google Sheets rows -> Normalize -> Loop Over Items ->
Validate -> Ingest -> Research -> parallel RAG (CVs + Projects) ->
coverage scoring -> Append row. There is no webhook and no proposal node
in this workflow; those belong to a different export and must not be
asserted here.
"""

import json
from pathlib import Path


WORKFLOW_PATH = (
    Path(__file__).parents[2]
    / "n8n"
    / "workflows"
    / "Phase 1 RFP Orchestrator.json"
)


def load_workflow() -> dict:
    return json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))


def nodes_by_name(workflow: dict) -> dict:
    return {node["name"]: node for node in workflow["nodes"]}


def test_workflow_is_valid_with_unique_nodes_and_sound_connections():
    workflow = load_workflow()
    names = [node["name"] for node in workflow["nodes"]]
    assert workflow["active"] is False
    assert len(names) == len(set(names)) == len(workflow["nodes"])
    for source, output_groups in workflow["connections"].items():
        assert source in names, f"dangling connection source: {source}"
        for outputs in output_groups.values():
            for group in outputs:
                if not group:
                    continue
                for connection in group:
                    assert connection["node"] in names, (
                        f"dangling connection target: {connection['node']}"
                    )


def test_sheet_trigger_loop_present():
    workflow = load_workflow()
    names = nodes_by_name(workflow)
    assert names["Schedule Trigger"]["type"] == "n8n-nodes-base.scheduleTrigger"
    assert names["Get row(s) in sheet"]["type"] == "n8n-nodes-base.googleSheets"
    assert names["Normalize Sheet Tender"]["type"] == "n8n-nodes-base.code"
    loop = names["Loop Over Items"]
    assert loop["type"] == "n8n-nodes-base.splitInBatches"
    # Second output loops back to per-tender processing.
    loop_targets = [
        connection["node"]
        for group in workflow["connections"]["Loop Over Items"]["main"]
        for connection in group
    ]
    assert "Validate Tender and Configure URLs" in loop_targets


def test_service_chain_nodes_present():
    workflow = load_workflow()
    names = set(nodes_by_name(workflow))
    assert {
        "Validate Tender and Configure URLs",
        "1 - Ingest and Normalize Tender",
        "2 - Run Prospect Research",
        "Prepare RAG Context",
        "3A - Match CVs",
        "Collect CV Matches",
        "3B - Match Projects",
        "Collect Project Matches",
        "Merge RAG Results",
        "Code in JavaScript",
        "Append row in sheet",
    }.issubset(names)


def test_http_nodes_have_timeouts_retries_and_single_token():
    workflow = load_workflow()
    http_nodes = [
        node for node in workflow["nodes"]
        if node["type"] == "n8n-nodes-base.httpRequest"
    ]
    # Exactly the four service calls: ingest, research, CVs, projects.
    assert {node["name"] for node in http_nodes} == {
        "1 - Ingest and Normalize Tender",
        "2 - Run Prospect Research",
        "3A - Match CVs",
        "3B - Match Projects",
    }
    token_ids = set()
    for node in http_nodes:
        assert node["parameters"]["options"]["timeout"] <= 45_000
        assert node["retryOnFail"] is True
        assert 2 <= node["maxTries"] <= 3
        token_ids.add(node["credentials"]["httpHeaderAuth"]["id"])
    # One shared internal-token credential, not four diverging ones.
    assert len(token_ids) == 1


def test_service_urls_centralized_and_docker_native():
    workflow = load_workflow()
    config_node = nodes_by_name(workflow)["Validate Tender and Configure URLs"]
    code = config_node["parameters"]["jsCode"]
    for service, port in (
        ("detection", 8000),
        ("rag", 8001),
        ("research", 8002),
    ):
        assert f"http://{service}:{port}" in code
    assert "host.docker.internal" not in WORKFLOW_PATH.read_text(encoding="utf-8")


def test_export_contains_no_secrets():
    raw = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "dev-secret-token" not in raw
    assert "SUPABASE_SERVICE_KEY" not in raw
    assert "LLM_API_KEY" not in raw
    assert "TAVILY_API_KEY" not in raw


def test_prepare_rag_context_passes_config_through():
    """3A/3B build their URL from ``$json.config`` but consume Prepare's
    output. If Prepare drops ``config``, both RAG nodes throw at runtime
    and no backend change can compensate."""
    workflow = load_workflow()
    prepare = nodes_by_name(workflow)["Prepare RAG Context"]
    returned = prepare["parameters"]["jsCode"].split("return", 1)[1]
    assert "config" in returned


def test_validate_documents_sheet_contract():
    """Validate requires ``id`` and ``title``, so the sheet must carry an
    ``id`` column — detection mints nothing at this stage."""
    workflow = load_workflow()
    code = nodes_by_name(workflow)[
        "Validate Tender and Configure URLs"
    ]["parameters"]["jsCode"]
    assert "tender.id" in code
    assert "tender.title" in code
