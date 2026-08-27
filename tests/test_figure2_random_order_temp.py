"""Temporary diagnostic: verify Figure 2 shuffles before anonymizing."""
from __future__ import annotations

from sports.analysis import load_data, run_figure2
from sports.config import PREDICTORS
from sports.filters import apply_analysis_filters


TEST_FILTERS = {"sport": "table tennis"}
MAX_ATHLETES = 20


def _capture_selected_ids(monkeypatch):
    import sports.figures

    captured = []

    def capture_figure(*, selected_dataframe, **kwargs):
        captured.extend(selected_dataframe["athlete_id"].astype(str).tolist())
        return None

    monkeypatch.setattr(sports.figures, "create_figure2", capture_figure)
    result = run_figure2(
        variables=list(PREDICTORS),
        filters=TEST_FILTERS,
        max_athletes=MAX_ATHLETES,
        reference_group="selected_cohort",
    )
    return captured, result


def test_figure2_randomized_internal_order_is_reproducible_and_publicly_anonymous(monkeypatch):
    filtered = apply_analysis_filters(load_data(), TEST_FILTERS)
    original_order = filtered["athlete_id"].astype(str).tolist()

    first_order, first_result = _capture_selected_ids(monkeypatch)
    second_order, second_result = _capture_selected_ids(monkeypatch)
    expected_length = min(MAX_ATHLETES, len(filtered))

    print("Original first 10 internal IDs:", original_order[:10])
    print("Figure 2 first 10 selected internal IDs:", first_order[:10])
    print("Order differs:", first_order != original_order[:expected_length])
    print("Random-order mode: fixed seed (reproducible)")

    assert len(first_order) == expected_length
    assert first_order != original_order[:expected_length]
    assert first_order[0] != original_order[0]
    assert set(first_order).issubset(set(original_order))
    assert first_order == second_order

    for result in (first_result, second_result):
        public_labels = [row["anonymous_profile_label"] for row in result["table"]]
        assert public_labels == [f"Profile {index:02d}" for index in range(1, len(public_labels) + 1)]
        assert public_labels[0] == "Profile 01"
        assert not any("Athlete_" in label for label in public_labels)
        public_result = {key: value for key, value in result.items() if key != "figure"}
        assert "Athlete_" not in repr(public_result)

