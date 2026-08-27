from pathlib import Path


def test_renderer_preserves_session_figure_and_draws_before_table():
    source=Path("ui/result_renderer.py").read_text(encoding="utf-8")
    assert "clear_figure=False" in source
    assert source.index("st.pyplot(figure")<source.index("st.dataframe(frame")
