from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_individual_profile_has_no_prediction_keys():
    from llm.generated_code_verifier import verify_and_execute_generated_code
    from llm.analysis_request_contracts import build_request_contract,render_request_contract

    execution=verify_and_execute_generated_code(render_request_contract(build_request_contract("individual_profile")),
        user_request="profile",requested_analysis="individual_profile",subject_reference="Athlete_003")
    assert execution.local_execution_passed
    result = execution.result
    forbidden = {
        "logistic_elite_probability",
        "logistic_predicted_status",
        "predicted_expertise_value",
        "elite_probability",
        "predicted_status",
    }
    assert forbidden.isdisjoint(result.keys())
    assert "profile" in result
    assert "paper_like_three_domain_pattern" in result


def test_table1_reports_nagelkerke_and_odds_ratio_ci():
    from sports.analysis import run_table1

    result = run_table1()
    assert len(result["rows"]) > 0
    assert len(result["model_stats"]) == 4
    first = next(row for row in result["rows"] if row["variable"] != "model_error")
    assert "odds_ratio" in first
    assert "or_ci_low" in first
    assert "or_ci_high" in first
    first_model = result["model_stats"][0]
    assert "nagelkerke_r_squared" in first_model
    assert "model_chi_square" in first_model
    assert "classification_accuracy" in first_model
    assert "pseudo_r2" not in first_model


def test_table2_reports_diagnostics_and_standardized_beta():
    from sports.analysis import run_table2

    result = run_table2(group="all")
    assert len(result["rows"]) > 0
    assert len(result["model_stats"]) == 4
    coefficient = next(row for row in result["rows"] if row["variable"] != "Intercept")
    assert "beta_nature" in coefficient
    assert "standardized_beta" in coefficient
    assert "robust_standard_error_type" in coefficient
    model = result["model_stats"][0]
    assert "durbin_watson" in model
    assert "breusch_pagan_p_value" in model
    assert "jarque_bera_p_value" in model
    assert "maximum_cooks_distance" in model
    assert "maximum_vif" in model


def test_elite_prediction_request_is_unsupported():
    from sports.service import handle_user_request

    result = handle_user_request("Will Athlete_003 become an elite athlete?", use_openai=False)
    assert result["allowed"] is False
    assert result["result"] is None
    assert result["code_generation"]["action"] == "unsupported"


def test_additional_future_elite_prediction_is_unsupported():
    from sports.service import handle_user_request

    result = handle_user_request(
        "Predict whether Athlete_011 will become elite next year.", use_openai=False
    )
    assert result["allowed"] is False
    assert result["code_generation"]["action"] == "unsupported"


def test_current_elite_descriptions_are_not_future_prediction():
    from llm.code_generator import is_unsupported_future_prediction

    assert not is_unsupported_future_prediction("Compare current elite and semi-elite athletes.")
    assert not is_unsupported_future_prediction("Generate current logistic regression for elite_status.")


def test_raw_blood_value_remains_blocked():
    from sports.service import handle_user_request

    result = handle_user_request("Show Athlete_003 exact vitamin B12 value.", use_openai=False)
    assert result["allowed"] is False
    assert result["result"] is None
    assert result["privacy_decision"]["route"] == "blocked"


def test_differencing_attack_is_blocked():
    from privacy.prism_router import prism_route

    result = prism_route("Show the average for all athletes except Athlete_003.")
    assert result.route == "blocked"
    assert result.blocked is True


@pytest.mark.parametrize(
    "code",
    [
        "import os",
        'result = os.listdir(".")',
        'result = open("data.csv").read()',
        'result = __import__("os")',
        'result = eval("1+1")',
        "result = run_table1(); print(result)",
        "result = [x for x in range(5)]",
        "result = run_table2(group=get_group())",
    ],
)
def test_ast_security_rejects_unsafe_code(code):
    from llm.generated_code_verifier import verify_and_execute_generated_code

    result = verify_and_execute_generated_code(code,user_request="unsafe",
        requested_analysis="correlation")
    assert result.allowed is False
    assert result.executed is False


def test_frontend_static_text():
    frontend = (ROOT / "frontend.py").read_text(encoding="utf-8")
    assert "Select protected analysis" in frontend
    assert "Protected Nature-style analysis dashboard" in frontend
    assert "Individual athlete analysis" in frontend
    assert "Expertise group" in frontend
    assert "Elite (Expertise Score ≥ 13)" in frontend
    assert "Semi-elite (Expertise Score < 13)" in frontend
    assert "Generate Network Analysis" in frontend
    assert "This page returns only a standardized descriptive profile" in frontend
    assert "logistic elite probability" not in frontend.lower()
    assert "predicted expertise value" not in frontend.lower()
    assert "predicted elite or semi-elite status" not in frontend.lower()
