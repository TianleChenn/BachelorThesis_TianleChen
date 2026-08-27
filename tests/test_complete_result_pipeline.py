from pathlib import Path


FRONTEND = Path("frontend.py").read_text(encoding="utf-8")


def _pipeline_renderer():
    start = FRONTEND.index("def show_pipeline_response")
    end = FRONTEND.index("\ndef run_request", start)
    return FRONTEND[start:end]


def test_complete_safe_result_pipeline_is_present():
    renderer = _pipeline_renderer()
    for expected in [
        "_render_final_analysis_result(response)",
        "show_pipeline_overview(response=response)",
        "show_prism_privacy_result(response=response)",
        'render_llm_result(response.get("llm_result"),response=response)',
            '"Local execution diagnostics" if route=="local_edge" else "Technical details"',
    ]:
        assert expected in renderer


def test_each_analysis_keeps_its_own_response():
    for key in [
        "table1_response",
        "table2_response",
        "figure1_response",
        "figure2_response",
        "correlation_response",
        "variance_response",
        "individual_analysis_response",
    ]:
        assert key in FRONTEND


def test_removed_unsafe_ui_stays_removed():
    assert "Ask a protected analysis question..." not in FRONTEND
    assert "st.chat_input" not in FRONTEND
    assert "st.chat_message" not in FRONTEND
    assert "raw CSV viewer" not in FRONTEND
    assert "st.download_button" not in FRONTEND
    assert "Data Generation & Processing" in FRONTEND


def test_only_restricted_generated_call_is_rendered_to_normal_users():
    start = FRONTEND.index("def render_llm_result")
    end = FRONTEND.index("\ndef _render_module_separator", start)
    renderer = FRONTEND[start:end]
    assert 'generated_call=response.get("generated_code")' in renderer
    assert 'generated_arguments=code_execution.get("generated_arguments")' in renderer
    assert "_restricted_call_display_rows(generated_method,generated_arguments)" in renderer
    assert "raw_response" not in renderer
    assert "The selected LLM generates only a Restricted Analysis Call" in renderer
