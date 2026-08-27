from llm.generated_code_verifier import verify_and_execute_generated_code
from llm.analysis_request_contracts import build_request_contract,render_request_contract
from sports.config import DOMAIN_ORDER


def _result():
    execution=verify_and_execute_generated_code(render_request_contract(build_request_contract("individual_profile")),
        user_request="profile",requested_analysis="individual_profile",subject_reference="Athlete_003")
    assert execution.local_execution_passed
    return execution.result


def test_exactly_eight_domains_in_fixed_order():
    result=_result();rows=result["table"]
    assert len(rows)==8
    assert [row["domain_key"] for row in rows]==DOMAIN_ORDER
    assert "lower_body_dynamics" in [row["domain_key"] for row in rows]
    assert "muscle_power_genetics" in [row["domain_key"] for row in rows]
    assert result["domain_count"]==8


def test_profile_contains_no_identifier_or_raw_fields():
    text=repr(_result())
    assert "Athlete_003" not in text
    for raw in ["body_weight","vitamin_b12","polygenic_score","phq4_score"]:
        assert raw not in text


def test_missing_domain_is_retained_as_unavailable():
    rows=_result()["profile"]
    assert len(rows)==8
    assert all("z_score" in row and "interpretation" in row for row in rows)
