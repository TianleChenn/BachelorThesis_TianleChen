import matplotlib.figure

from llm.generated_code_verifier import verify_and_execute_generated_code
from llm.analysis_request_contracts import build_request_contract,render_request_contract
from sports.config import DOMAIN_ORDER


def _profile_result():
    execution=verify_and_execute_generated_code(render_request_contract(build_request_contract("individual_profile")),
        user_request="profile",requested_analysis="individual_profile",subject_reference="Athlete_003")
    assert execution.local_execution_passed
    return execution.result


def test_complete_profile_survives_pipeline():
    result=_profile_result();rows=result["table"]
    assert len(rows)==8 and result["domain_count"]==8
    assert [row["domain_key"] for row in rows]==DOMAIN_ORDER
    assert "lower_body_dynamics" in result["domain_keys"]
    assert "muscle_power_genetics" in result["domain_keys"]
    assert isinstance(result["figure"],matplotlib.figure.Figure)
    assert "Athlete_003" not in repr(result)
    assert result["identifier_exposed"] is False


def test_figure_has_eight_labels_and_reference_lines():
    axis=_profile_result()["figure"].axes[0]
    assert len(axis.get_xticklabels())==8
    assert len(axis.lines)>=4
