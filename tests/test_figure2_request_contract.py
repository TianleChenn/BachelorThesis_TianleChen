from pathlib import Path

from llm.analysis_request_contracts import build_request_contract,parse_figure2_request_size,render_request_contract
from llm.generated_code_verifier import inspect_generated_code
from ui.cohort_prompts import build_dashboard_prompt


def test_figure2_contract_uses_active_filters():
    contract=render_request_contract(build_request_contract("figure2"))
    assert "filters={}" in contract
    assert "reference_group='selected_cohort'" in contract
    assert "higher_expertise" not in contract


def test_dashboard_size_parser_and_contracts():
    cases = {
        "Generate Figure 2 for all athletes showing 20 athletes.": 20,
        "Generate Figure 2 for all athletes showing 50 athletes.": 50,
        "Generate Figure 2 for all athletes showing 80 athletes.": 80,
        "Generate Figure 2-style z-score profiles for all athletes.": None,
    }
    for prompt, expected in cases.items():
        size = parse_figure2_request_size(prompt)
        assert size == expected
        code=render_request_contract(build_request_contract("figure2",{},prompt))
        validation = inspect_generated_code(code,user_request=prompt,
            requested_analysis="figure2",requested_filters={})
        assert validation.request_match_passed
        assert validation.generated_arguments["filters"] == {}
        assert validation.generated_arguments["reference_group"] == "selected_cohort"
        assert validation.generated_arguments["max_athletes"] == expected


def test_frontend_all_option_has_no_numeric_limit():
    source = Path("frontend.py").read_text(encoding="utf-8")
    assert 'options=["20", "50", "80", "All"]' in source
    assert 'build_dashboard_prompt("figure2",cohort_group,figure2_size_option)' in source
    prompt = build_dashboard_prompt("figure2", "table tennis athletes", "All")
    assert "table tennis athletes" in prompt
    assert "showing all available anonymous athletes" in prompt
    assert parse_figure2_request_size(prompt) is None
