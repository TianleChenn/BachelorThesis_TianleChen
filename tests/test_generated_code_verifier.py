import json
from pathlib import Path
from unittest.mock import patch

import pytest

from llm.analysis_request_contracts import build_request_contract, render_request_contract
from llm.generated_code_verifier import inspect_generated_code, verify_and_execute_generated_code


def contract_code(analysis, filters=None, prompt=""):
    return render_request_contract(build_request_contract(analysis, filters, prompt))


def inspect(code, analysis="correlation", filters=None, prompt="request"):
    return inspect_generated_code(
        code,
        user_request=prompt,
        requested_analysis=analysis,
        requested_filters=filters or {},
    )


def test_correct_all_domain_pearson_passes_request_validation():
    result = inspect(contract_code("correlation"))
    assert result.structure_validation_passed
    assert result.request_match_passed


def test_trusted_optional_defaults_are_normalized_before_comparison():
    full=contract_code("correlation")
    without_defaults=full.replace(", filters={}, method='pearson'", "")
    result=inspect(without_defaults)
    assert result.request_match_passed
    assert result.generated_arguments["filters"]=={}
    assert result.generated_arguments["method"]=="pearson"
    assert result.generated_arguments["visualization"] is True


def test_unrestricted_filters_none_normalizes_to_empty_dictionary():
    code=contract_code("correlation").replace("filters={}", "filters=None")
    result=inspect(code)
    assert result.request_match_passed
    assert result.generated_arguments["filters"]=={}


def test_restricted_request_rejects_filters_none_after_normalization():
    filters={"national_team":"Junior"}
    code=contract_code("correlation",filters).replace(
        "filters={'national_team': 'Junior'}", "filters=None")
    result=inspect(code,filters=filters)
    assert result.failure_stage=="request_validation"
    assert result.generated_arguments["filters"]=={}
    assert "Expected filters={'national_team': 'Junior'}" in result.validation_error


def test_controls_none_is_not_semantically_expanded():
    code=contract_code("table1").replace(
        "controls=[[], ['sex'], ['age'], ['sex', 'age']]", "controls=None")
    result=inspect(code,analysis="table1")
    assert result.failure_stage=="request_validation"
    assert result.generated_arguments["controls"] is None
    assert "Expected all four predefined Table 1 control specifications" in result.validation_error


@pytest.mark.parametrize("controls", [
    "[['age'], ['sex'], ['age', 'sex'], []]",
    "[['sex', 'age'], [], ['sex'], ['age']]",
])
def test_table1_control_order_is_semantically_normalized(controls):
    code = contract_code("table1").replace(
        "controls=[[], ['sex'], ['age'], ['sex', 'age']]", f"controls={controls}"
    )
    result = inspect(code, analysis="table1")
    assert result.request_match_passed


@pytest.mark.parametrize("controls", [
    "[[], ['age', 'sex']]",
    "[[], ['sex'], ['sex'], ['age', 'sex']]",
])
def test_table1_incomplete_or_duplicate_controls_still_fail(controls):
    code = contract_code("table1").replace(
        "controls=[[], ['sex'], ['age'], ['sex', 'age']]", f"controls={controls}"
    )
    result = inspect(code, analysis="table1")
    assert result.failure_stage == "request_validation"
    assert "Expected all four predefined Table 1 control specifications" in result.validation_error


@pytest.mark.parametrize(
    "replacement, expected_message",
    [
        ('variables=["mental_health", "social_support"]', "Expected 8 public domains"),
        ('method="spearman"', "Expected correlation method"),
    ],
)
def test_runnable_but_wrong_correlation_is_rejected(replacement, expected_message):
    code = contract_code("correlation")
    if replacement.startswith("variables"):
        start, end = code.index("variables="), code.index(", filters=")
        code = code[:start] + replacement + code[end:]
    else:
        code = code.replace("method='pearson'", replacement)
    result = inspect(code)
    assert result.failure_stage == "request_validation"
    assert not result.local_execution_passed
    assert expected_message in result.validation_error


def test_wrong_filters_are_rejected():
    filters = {"national_team": "Junior"}
    result = inspect(contract_code("correlation"), filters=filters)
    assert result.failure_stage == "request_validation"
    assert "Expected filters" in result.validation_error


@pytest.mark.parametrize(
    ("analysis", "old", "new"),
    [
        ("table1", "controls=[[], ['sex'], ['age'], ['sex', 'age']]", "controls=[['sex']]"),
        ("figure1", "correlation_threshold=0.15", "correlation_threshold=0.9"),
        ("figure1", "variance_iterations=1000", "variance_iterations=100"),
        ("variance_analysis", "iterations=1000", "iterations=100"),
    ],
)
def test_wrong_analysis_configuration_is_rejected(analysis, old, new):
    result = inspect(contract_code(analysis).replace(old, new), analysis=analysis)
    assert result.failure_stage == "request_validation"


def test_wrong_method_is_request_validation_failure():
    result = inspect(contract_code("figure1"), analysis="correlation")
    assert result.structure_validation_passed
    assert result.failure_stage == "request_validation"


@pytest.mark.parametrize(
    "code",
    [
        "import os\n" + contract_code("correlation"),
        "result = analysis.dataframe.correlation(variables=[])",
        contract_code("correlation").replace("filters={}", "filters={'sex': 'Athlete_003'}"),
    ],
)
def test_unsafe_structure_is_rejected(code):
    result = inspect(code)
    assert result.failure_stage == "format_validation"


def test_request_failure_never_instantiates_backend():
    wrong = contract_code("correlation").replace("method='pearson'", "method='spearman'")
    with patch("llm.generated_code_verifier.RestrictedAnalysisAPI") as api:
        result = verify_and_execute_generated_code(
            wrong, user_request="Pearson", requested_analysis="correlation"
        )
    assert result.failure_stage == "request_validation"
    api.assert_not_called()


def test_backend_exception_is_local_execution_failure():
    code = contract_code("correlation")
    with patch("llm.generated_code_verifier.RestrictedAnalysisAPI") as api:
        api.return_value.correlation.side_effect = RuntimeError("private backend detail")
        result = verify_and_execute_generated_code(
            code, user_request="Pearson", requested_analysis="correlation"
        )
    assert result.failure_stage == "local_execution"
    assert not result.fully_correct


def test_malformed_backend_result_is_result_validation_failure():
    code = contract_code("correlation")
    with patch("llm.generated_code_verifier.RestrictedAnalysisAPI") as api:
        api.return_value.correlation.return_value = {"analysis": "wrong"}
        result = verify_and_execute_generated_code(
            code, user_request="Pearson", requested_analysis="correlation"
        )
    assert result.local_execution_passed
    assert result.failure_stage == "result_validation"
    assert not result.fully_correct


def test_all_40_frontend_router_samples_have_valid_canonical_contracts():
    dataset = json.loads(
        Path("evaluation/frontend_realistic_benchmark_60.json").read_text(encoding="utf-8")
    )
    eligible = [sample for sample in dataset["samples"] if sample["llm_router_eligible"]]
    assert len(eligible) == 40
    for sample in eligible:
        contract = build_request_contract(
            sample["requested_analysis"], sample["analysis_filters"], sample["prompt"]
        )
        result = inspect_generated_code(
            render_request_contract(contract),
            user_request=sample["prompt"],
            requested_analysis=sample["requested_analysis"],
            requested_filters=sample["analysis_filters"],
        )
        assert result.request_match_passed, sample["id"]
