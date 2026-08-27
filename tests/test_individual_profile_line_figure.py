import matplotlib.figure
import numpy as np

from llm.generated_code_verifier import verify_and_execute_generated_code
from llm.analysis_request_contracts import build_request_contract,render_request_contract
from sports.config import DOMAIN_LABELS,DOMAIN_ORDER


def _result():
    return verify_and_execute_generated_code(render_request_contract(build_request_contract("individual_profile")),
        user_request="profile",requested_analysis="individual_profile",
        subject_reference="Athlete_003").result


def test_line_figure_has_required_scientific_layout():
    result=_result();figure=result["figure"]
    assert isinstance(figure,matplotlib.figure.Figure)
    ax=figure.axes[0];lines={line.get_label():line for line in ax.lines};profile_line=lines["Anonymous Profile"]
    assert len(profile_line.get_xdata())==8
    assert tuple(round(v) for v in ax.get_ylim())==(-3,3)
    assert list(ax.get_yticks())==list(range(-3,4))
    reference=[line for line in ax.lines if line.get_linestyle()=="--" and len(line.get_ydata())==8]
    assert len(reference)==1
    assert np.allclose(reference[0].get_ydata(),result["elite_mean_profile"])
    assert not np.allclose(reference[0].get_ydata(),np.zeros(8))
    assert reference[0].get_label()=="Elite Mean Profile"
    assert np.allclose(lines["Overall Mean (z = 0)"].get_ydata(),[0.0,0.0])
    assert len(ax.patches)==8
    assert [tick.get_text() for tick in ax.get_xticklabels()]==[DOMAIN_LABELS[key] for key in DOMAIN_ORDER]
    assert "Athlete_003" not in ax.get_title()
    assert result["profile_label"] in ax.get_title()


def test_table_and_figure_values_match():
    result=_result();expected=[row["z_score"] for row in result["table"]]
    lines={line.get_label():line for line in result["figure"].axes[0].lines}
    actual=list(lines["Anonymous Profile"].get_ydata())
    assert np.allclose(actual,expected,equal_nan=True)
