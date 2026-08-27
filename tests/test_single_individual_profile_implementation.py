from pathlib import Path


LEGACY_NAMES = (
    "create_" + "individual_profile_figure",
    "individual_athlete_" + "report",
    "run_" + "individual_profile",
    "run_" + "individual_analysis",
    "run_" + "individual_report",
    "analyze_" + "individual_athlete",
)


def test_only_restricted_api_implements_individual_profiles():
    project_files = [
        *Path("sports").glob("*.py"),
        *Path("llm").glob("*.py"),
        Path("frontend.py"),
    ]
    source = "\n".join(path.read_text(encoding="utf-8") for path in project_files)
    for legacy_name in LEGACY_NAMES:
        assert legacy_name not in source

    restricted_source = Path("sports/restricted_analysis_api.py").read_text(encoding="utf-8")
    assert restricted_source.count("def individual_profile(") == 1


def test_profile_result_and_plot_are_anonymous():
    from llm.generated_code_verifier import verify_and_execute_generated_code
    from llm.analysis_request_contracts import build_request_contract,render_request_contract

    execution = verify_and_execute_generated_code(
        render_request_contract(build_request_contract("individual_profile")),
        user_request="profile",
        requested_analysis="individual_profile",
        subject_reference="Athlete_003",
    )
    assert execution.local_execution_passed
    result = execution.result
    assert "Athlete_003" not in repr(result)
    axis = result["figure"].axes[0]
    assert "Athlete_003" not in axis.get_title()
    assert all("Athlete_003" not in text.get_text() for text in axis.get_legend().get_texts())
