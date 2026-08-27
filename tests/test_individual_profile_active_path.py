import matplotlib.figure

from llm.generated_code_verifier import verify_and_execute_generated_code
from llm.analysis_request_contracts import build_request_contract,render_request_contract


def test_active_restricted_profile_returns_complete_result():
    execution=verify_and_execute_generated_code(render_request_contract(build_request_contract("individual_profile")),
        user_request="profile",requested_analysis="individual_profile",subject_reference="Athlete_003")
    result=execution.result
    assert isinstance(result,dict) and result["domain_count"]==8
    assert len(result["table"])==8
    assert isinstance(result["figure"],matplotlib.figure.Figure)
    keys=[row["domain_key"] for row in result["table"]]
    assert "lower_body_dynamics" in keys and "muscle_power_genetics" in keys
    assert "Athlete_003" not in repr(result)
