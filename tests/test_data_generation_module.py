from pathlib import Path
def test_navigation_exists_without_removed_visibility_module():
    source=Path("frontend.py").read_text(encoding="utf-8")
    assert '"Data Generation & Processing"' in source
    for removed_copy in (
        "Aggregate Validation",
        "Target vs Observed Correlations",
        "Aggregate Sample Composition",
        "Data Visibility and Privacy Boundaries",
        "Raw measurements are stored and processed locally",
        "Raw Dataset Protection",
        "Inspect Column Provenance",
        "Generate or rebuild the dataset",
    ):
        assert removed_copy not in source
    assert "st.download_button" not in source
    assert "st.dataframe(df.head())" not in source
