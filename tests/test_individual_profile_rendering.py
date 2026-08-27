from pathlib import Path


def test_safe_renderer_draws_figure_before_eight_row_table():
    source=Path("ui/result_renderer.py").read_text(encoding="utf-8")
    assert source.index("st.pyplot(figure") < source.index("st.dataframe(frame")
    assert "DeltaGenerator" in source


def test_complete_pipeline_modules_remain_after_profile():
    source=Path("frontend.py").read_text(encoding="utf-8")
    for module in ["show_pipeline_overview", "show_prism_privacy_result", "render_llm_result"]:
        assert module in source
