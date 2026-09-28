import json
from pathlib import Path


WORKFLOW_PATH = (
    Path(__file__).parents[2]
    / "n8n"
    / "workflows"
    / "olivesoft_phase1_orchestrator.json"
)


def load_workflow() -> dict:
    return json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))


def test_n8n_workflow_is_valid_and_self_contained():
    workflow = load_workflow()
    names = {node["name"] for node in workflow["nodes"]}
    assert workflow["active"] is False
    assert len(names) == len(workflow["nodes"])
    assert {
        "Receive Tender Webhook",
        "1 - Ingest and Normalize Tender",
        "2 - Run Prospect Research",
        "3A - Match CVs",
        "3B - Match Projects",
        "3C - Match Capabilities",
        "Build Matrix and Initial Proposal Brief",
        "4 - Generate PPTX and PDF",
        "Return Orchestration Result",
    }.issubset(names)

    for source, output_groups in workflow["connections"].items():
        assert source in names
        for outputs in output_groups.values():
            for group in outputs:
                if not group:
                    continue
                for connection in group:
                    assert connection["node"] in names


def test_n8n_export_contains_no_real_credentials():
    raw = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "dev-secret-token" not in raw
    assert "SUPABASE_SERVICE_KEY" not in raw
    assert "LLM_API_KEY" not in raw
    assert "TAVILY_API_KEY" not in raw

    workflow = json.loads(raw)
    credential_ids = {
        credential["id"]
        for node in workflow["nodes"]
        for credential in node.get("credentials", {}).values()
    }
    assert credential_ids == {
        "REPLACE_WITH_WEBHOOK_CREDENTIAL_ID",
        "REPLACE_WITH_INTERNAL_CREDENTIAL_ID",
    }


def test_n8n_http_nodes_have_timeouts_and_retries():
    workflow = load_workflow()
    http_nodes = [
        node for node in workflow["nodes"]
        if node["type"] == "n8n-nodes-base.httpRequest"
    ]
    assert len(http_nodes) == 6
    for node in http_nodes:
        assert node["parameters"]["options"]["timeout"] <= 45_000
        assert node["retryOnFail"] is True
        assert 2 <= node["maxTries"] <= 3


def test_n8n_service_urls_are_centralized_and_docker_native():
    workflow = load_workflow()
    config_node = next(
        node for node in workflow["nodes"]
        if node["name"] == "Validate Tender and Configure URLs"
    )
    code = config_node["parameters"]["jsCode"]
    for service, port in (
        ("detection", 8000),
        ("rag", 8001),
        ("research", 8002),
        ("proposal", 8003),
    ):
        assert f"http://{service}:{port}" in code
    assert "host.docker.internal" not in WORKFLOW_PATH.read_text(encoding="utf-8")


def test_n8n_brief_carries_commercial_intelligence_and_review_actions():
    workflow = load_workflow()
    brief_node = next(
        node
        for node in workflow["nodes"]
        if node["name"] == "Build Matrix and Initial Proposal Brief"
    )
    code = brief_node["parameters"]["jsCode"]
    assert "orchestration_version: '1.2.0'" in code
    assert "estimated_budget: research.estimated_budget" in code
    assert "competitor_insights: research.competitor_insights" in code
    assert "Commercial owner validates the sourced budget" in code
