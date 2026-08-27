from pathlib import Path
from ui.result_renderer import render_analysis_result
def test_none_and_unsupported_return_none():
    assert render_analysis_result(None) is None
    assert render_analysis_result(lambda:None) is None
    assert render_analysis_result(type) is None
def test_forbidden_documentation_not_in_frontend_source():
    source=Path("frontend.py").read_text(encoding="utf-8")
    assert "Creator of Delta protobuf messages." not in source
    assert "Get our DeltaGenerator." not in source
    for unsafe in ["st.help(","st.write(st)","st.write(main)","return st.success(","return st.error(","return st.info(","return st.warning(","return st.dataframe(","return st.pyplot("]:
        assert unsafe not in source
