from frontend import show_result_table
def test_result_renderer_returns_none():
    assert show_result_table(None) is None
