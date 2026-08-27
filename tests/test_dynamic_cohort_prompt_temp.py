from pathlib import Path

import pytest

import sports.service
from ui.cohort_prompts import build_dashboard_prompt
from llm.analysis_request_contracts import parse_figure2_request_size


class PromptCaptured(RuntimeError):
    pass


def _capture_prism_input(monkeypatch, prompt, filters):
    captured = {}

    def fake_prism_route(actual_prompt):
        captured["prompt"] = actual_prompt
        raise PromptCaptured

    monkeypatch.setattr(sports.service, "prism_route", fake_prism_route)
    with pytest.raises(PromptCaptured):
        sports.service.handle_user_request(
            prompt, requested_analysis="table1", analysis_filters=filters, use_openai=False,
        )
    return captured["prompt"]


def test_different_cohort_prompts_reach_prism_exactly(monkeypatch):
    table_tennis = build_dashboard_prompt("table1", "table tennis athletes")
    received_table_tennis = _capture_prism_input(
        monkeypatch, table_tennis, {"sport": "table tennis"},
    )
    female = build_dashboard_prompt("table1", "female athletes")
    received_female = _capture_prism_input(monkeypatch, female, {"sex": "female"})

    print("Table tennis PRISM prompt:", received_table_tennis)
    print("Female PRISM prompt:", received_female)
    assert received_table_tennis == table_tennis
    assert received_female == female
    assert "table tennis athletes" in received_table_tennis.lower()
    assert "female athletes" in received_female.lower()
    assert received_table_tennis != received_female


def test_all_six_prompt_helpers_include_current_cohort():
    for analysis in ("table1", "table2", "figure1", "figure2", "correlation", "variance_analysis"):
        first = build_dashboard_prompt(analysis, "basketball athletes", "20")
        second = build_dashboard_prompt(analysis, "table tennis athletes", "20")
        assert "basketball athletes" in first
        assert "table tennis athletes" in second
        assert first != second
    assert parse_figure2_request_size(build_dashboard_prompt("figure2","female athletes","20"))==20
    assert parse_figure2_request_size(build_dashboard_prompt("figure2","female athletes","50"))==50
    assert parse_figure2_request_size(build_dashboard_prompt("figure2","female athletes","80"))==80
    assert parse_figure2_request_size(build_dashboard_prompt("figure2","female athletes","All")) is None


def test_frontend_uses_helper_and_service_exposes_exact_prompt_metadata():
    frontend = Path("frontend.py").read_text(encoding="utf-8")
    service = Path("sports/service.py").read_text(encoding="utf-8")
    assert "build_dashboard_prompt" in frontend
    assert "analysis_filters=active_filters" in frontend
    assert '"prism_input_prompt": user_prompt' in service
    assert "PRISM Input Prompt" in frontend
