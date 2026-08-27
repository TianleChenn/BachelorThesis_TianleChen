from pathlib import Path
from unittest.mock import Mock, call, patch

import frontend
from frontend import build_llm_selection_label
from llm.model_config import LOCAL_EDGE_GENERATOR_MODEL
from sports.service import _build_pipeline_model_audit


def _audit(route, tier, model):
    return _build_pipeline_model_audit(
        {"route":route}, {"selected_model":f"{tier}_model", "selected_tier":tier, "execution_model":model},
        {"actual_model":model, "requested_model":model, "provider":"openai_compatible"})


def test_cloud_and_local_audit_are_explicit():
    cloud = _audit("cloud", "cloud", "gemini-3.5-flash")
    local = _audit("cloud", "local", "Ministral-3-8B-Local")
    assert cloud["cost_aware_selected_tier"] == "cloud" and cloud["external_cloud_api_used"] is True
    assert local["cost_aware_selected_tier"] == "local" and local["external_cloud_api_used"] is False
    assert local["base_url"] == "http://127.0.0.1:8080/v1"


def test_frontend_uses_one_llm_selection_card():
    source = Path("frontend.py").read_text(encoding="utf-8")
    assert '"Selected LLM"' in source
    assert '"LLM Selection"' not in source


def test_pipeline_overview_uses_backend_local_verification_status():
    columns = [Mock() for _ in range(4)]
    streamlit = Mock(); streamlit.columns.return_value = columns
    response = {
        "generated_code":"result = analysis.table1(filters={})",
        "privacy_decision":{"route":"cloud"},
        "model_decision":{"selected_tier":"local", "execution_model":"Ministral-3-8B-Local"},
        "pipeline_audit":{"cost_aware_router_applicable":True,
            "cost_aware_selected_tier":"local", "actual_generator_model":"Ministral-3-8B-Local"},
        "pipeline_diagnostics":{"validation_passed":True},
    }
    with patch.object(frontend,"st",streamlit):
        frontend.show_pipeline_overview(response)
    assert columns[0].metric.call_args == call("Privacy-aware Router","Cloud")
    assert columns[1].metric.call_args == call("Cost-aware Router","Local")
    selected_call = columns[2].metric.call_args.args
    assert selected_call[0] == "Selected LLM"
    assert selected_call[1].startswith("Local")
    assert "Ministral-3-8B-Local" in selected_call[1]
    assert columns[3].metric.call_args == call("Local Verification","PASS")


def test_pipeline_overview_shows_failed_local_verification():
    columns = [Mock() for _ in range(4)]
    streamlit = Mock(); streamlit.columns.return_value = columns
    with patch.object(frontend,"st",streamlit):
        frontend.show_pipeline_overview({
            "generated_code":"result = analysis.table1(filters={})",
            "privacy_decision":{"route":"local_edge"},
            "model_decision":{"selected_tier":"local", "execution_model":"Ministral-3-8B-Local"},
            "pipeline_audit":{"cost_aware_router_applicable":False},
            "pipeline_diagnostics":{"validation_passed":False},
        })
    columns[3].metric.assert_called_once_with("Local Verification","FAIL")


def test_single_llm_selection_label_covers_all_routes():
    assert build_llm_selection_label("cloud", "cloud", "gemini-3.5-flash", None).startswith("Cloud")
    assert build_llm_selection_label("cloud", "local", None, LOCAL_EDGE_GENERATOR_MODEL).startswith("Local")
    assert build_llm_selection_label("collaboration", "cloud", "gemini-3.5-flash", None).startswith("Cloud")
    assert build_llm_selection_label("local_edge", "local", None, LOCAL_EDGE_GENERATOR_MODEL) == "Local — Ministral-3-8B-Local"
    assert build_llm_selection_label("blocked", "none", None, None) == "None"


def test_local_edge_audit_marks_cost_aware_router_not_applicable():
    audit = _audit("local_edge", "local", LOCAL_EDGE_GENERATOR_MODEL)
    assert audit["cost_aware_router_applicable"] is False
    assert audit["cost_aware_router_status"] == "not_applicable"
    assert audit["cost_aware_selected_tier"] is None
    assert audit["execution_selected_tier"] == "local"
    assert audit["execution_location"] == "local"
