from ui.data_generation_helpers import build_domain_construction_rows,build_visibility_matrix,load_generation_metadata,load_generation_report,load_safe_dataset_summary,normalize_correlation_report,validate_no_row_level_exposure
def test_missing_files_are_safe(tmp_path):
    assert load_generation_report(tmp_path/"missing.json") is None
    assert load_generation_metadata(tmp_path/"missing.json")==[]
    assert load_safe_dataset_summary(tmp_path/"missing.csv")=={}
def test_domain_and_visibility_tables():
    assert len(build_domain_construction_rows())==8
    assert all(r["Raw Data Visible to LLM"]=="No" and r["Raw Data Visible to Frontend"]=="No" for r in build_domain_construction_rows())
    raw=next(r for r in build_visibility_matrix() if r["Data Layer"]=="Raw diagnostic measurements");assert raw["Cloud LLM"]=="No" and raw["Frontend"]=="No"
def test_correlation_normalization_is_aggregate_only():
    rows=normalize_correlation_report({"target_correlations":{"mental_health|social_support":{"target":.277,"observed":.25,"absolute_difference":.027}}})
    assert len(rows)==1 and "athlete_id" not in rows[0]
    assert validate_no_row_level_exposure(rows)
