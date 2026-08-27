from ui.result_renderer import render_analysis_result
def test_safe_renderer_returns_none_for_suppressed_objects():
    assert render_analysis_result(None) is None
    assert render_analysis_result(lambda:None) is None
    assert render_analysis_result(render_analysis_result) is None
