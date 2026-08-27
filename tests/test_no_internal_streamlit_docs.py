from pathlib import Path
def test_no_internal_streamlit_documentation_or_unsafe_render_patterns():
    source=Path("frontend.py").read_text(encoding="utf-8")
    for text in ["Creator of Delta protobuf messages","Get our DeltaGenerator","root_container","add_rows method","altair_chart method","area_chart method","pyplot method","write method"]:assert text not in source
    for pattern in ["st.help(","return st.write(","return st.success(","return st.error(","return st.info(","return st.warning(","return st.dataframe(","return st.pyplot(","st.write(st)","st.write(main)"]:assert pattern not in source
    assert "st.chat_input" not in source
    assert "main();  # Semicolon intentionally suppresses Streamlit magic rendering." in source
