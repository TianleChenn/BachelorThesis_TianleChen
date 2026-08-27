from pathlib import Path

import pandas as pd

from llm.analysis_request_contracts import build_request_contract,render_request_contract
from sports.analysis import load_data
from sports.anonymous_subject import (
    ATHLETE_GROUP_OPTIONS,
    anonymous_candidate_frame,
    select_anonymous_subject,
)
from sports.config import DOMAIN_ORDER
from sports.restricted_analysis_api import RestrictedAnalysisAPI


def test_frontend_has_group_selector_and_no_direct_id_input():
    source = Path("frontend.py").read_text(encoding="utf-8")
    page = source[source.index("def page_individual") : source.index("def main")]
    assert ATHLETE_GROUP_OPTIONS == ("Elite", "Semi-elite", "All athletes")
    assert '"Athlete group"' in page
    assert "ATHLETE_GROUP_LABELS.get(group, group)" in page
    assert '"Elite": "Elite (Expertise Score ≥ 13)"' in source
    assert '"Semi-elite": "Semi-elite (Expertise Score < 13)"' in source
    assert '"Analyze anonymous athlete"' in page
    assert "text_input" not in page
    assert '"Athlete ID"' not in page
    assert "Anonymous standardized profile generated locally." in page


def test_candidate_groups_use_the_one_expertise_threshold():
    full = load_data()
    elite = anonymous_candidate_frame("Elite", full)
    semi = anonymous_candidate_frame("Semi-elite", full)
    all_athletes = anonymous_candidate_frame("All athletes", full)
    assert elite["expertise_value"].ge(13).all()
    assert semi["expertise_value"].lt(13).all()
    assert len(all_athletes) == len(full)
    assert len(elite) + len(semi) == len(full)


def test_consecutive_selection_avoids_previous_and_single_candidate_works():
    frame = pd.DataFrame({"athlete_id": ["internal-one", "internal-two"], "expertise_value": [14, 15]})
    first = select_anonymous_subject("Elite", dataframe=frame, chooser=lambda values: values[0])
    second = select_anonymous_subject("Elite", previous_id=first, dataframe=frame, chooser=lambda values: values[0])
    assert first != second
    single = frame.iloc[:1].copy()
    assert select_anonymous_subject("Elite", previous_id="internal-one", dataframe=single) == "internal-one"


def test_public_profile_generated_code_is_identifier_free():
    internal_id = select_anonymous_subject("Elite", chooser=lambda values: values[0])
    result = RestrictedAnalysisAPI(subject_reference=internal_id).individual_profile(
        subject_token="CURRENT_SUBJECT", variables=list(DOMAIN_ORDER),
        reference_group="all", output_mode="standardized_profile",
    )
    public = {key: value for key, value in result.items() if key != "figure"}
    assert "Athlete_" not in repr(public)
    contract=render_request_contract(build_request_contract("individual_profile"))
    assert "Athlete_" not in contract
    assert "CURRENT_SUBJECT" in contract
    assert result["analysis"] == "Anonymous Athlete Profile"
    assert result["profile_label"] == "Anonymous Profile"
    assert all(row["z_score"] is None or isinstance(row["z_score"], float) for row in result["table"])
    assert all(row["z_score"] is None or -3.0 <= row["z_score"] <= 3.0 for row in result["table"])
    labels = [line.get_label() for line in result["figure"].axes[0].lines]
    assert "Elite Mean Profile" in labels
    assert "Overall Mean (z = 0)" in labels


def test_group_change_callback_clears_stale_local_and_public_state_keys():
    source = Path("frontend.py").read_text(encoding="utf-8")
    callback = source[source.index("def _clear_individual_analysis_on_group_change") : source.index("def page_individual")]
    for key in (
        "individual_analysis_response",
        "last_anonymous_athlete_id", "last_anonymous_athlete_group",
    ):
        assert key in callback
    assert "on_change=_clear_individual_analysis_on_group_change" in source
