from pathlib import Path


def test_analysis_chat_removed_but_fixed_controls_remain():
    source = Path("frontend.py").read_text(encoding="utf-8")
    assert "Ask a protected analysis question..." not in source
    assert "st.chat_input" not in source
    assert "st.chat_message" not in source
    assert "analysis_messages" not in source
    for label in [
        "Run Table 1",
        "Run Table 2",
        "Generate Network Analysis",
        "Generate Athlete Profile Visualization",
        "Run correlation",
        "Run variance",
    ]:
        assert label in source
    assert "Final Analysis Result" in source
    assert "Pipeline Overview" in source
    assert "Privacy-aware Router Result" in source


def test_analysis_result_renderer_uses_current_structured_modules_in_order():
    source = Path("frontend.py").read_text(encoding="utf-8")
    start = source.index("def show_pipeline_response")
    end = source.index("\ndef render_header", start)
    renderer = source[start:end]
    assert "show_pipeline_overview(response=response)" in renderer
    assert "show_prism_privacy_result(response=response)" in renderer
    expected=["show_pipeline_overview(response=response)","show_prism_privacy_result(response=response)",
        'render_llm_result(response.get("llm_result"),response=response)']
    assert [renderer.index(item) for item in expected] == sorted(renderer.index(item) for item in expected)
    assert renderer.index("show_pipeline_overview(response=response)") < renderer.index("_render_final_analysis_result(response)") < renderer.index("show_prism_privacy_result(response=response)")


def test_no_free_text_chat_response_before_structured_analysis_result():
    frontend = Path("frontend.py").read_text(encoding="utf-8")
    start = frontend.index("def show_pipeline_response")
    end = frontend.index("\ndef run_request", start)
    renderer = frontend[start:end]
    assert "st.chat_input" not in frontend
    assert renderer.index("show_pipeline_overview(response=response)") < renderer.index("_render_final_analysis_result(response)")

    result_renderer = Path("ui/result_renderer.py").read_text(encoding="utf-8")
    assert 'value.get("privacy_note")' not in result_renderer
