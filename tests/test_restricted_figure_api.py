import pytest
import pandas as pd
from sports.restricted_analysis_api import RestrictedAnalysisAPI
from sports.config import DOMAIN_ORDER, PREDICTORS
from sports.analysis import load_data, run_figure2

def test_figure_parameter_bounds():
    api=RestrictedAnalysisAPI()
    with pytest.raises(ValueError):api.figure1(variables=PREDICTORS,correlation_threshold=2)
    with pytest.raises(ValueError):api.figure2(variables=PREDICTORS,max_athletes=11)
    with pytest.raises(ValueError):api.variance_analysis(variables=PREDICTORS,iterations=10)


def test_variance_analysis_does_not_apply_old_minimum_group_size_rule():
    result = RestrictedAnalysisAPI().variance_analysis(
        variables=list(DOMAIN_ORDER),
        filters={"sport": "3x3 basketball"},
        iterations=100,
        visualization=False,
    )

    assert result["analysis"] == "Dynamic variance analysis"
    assert result["cohort_size"] == 35
    assert result["elite_group_size"] == 4
    assert result["semi_elite_group_size"] == 31
    assert len(result["table"]) == 8
    assert "noise_utility" in result


def test_variance_analysis_requires_two_observations_per_group():
    source = load_data()
    one_elite = source[source["elite_status"].astype(int).eq(1)].head(1)
    two_semi_elite = source[source["elite_status"].astype(int).eq(0)].head(2)
    dataframe = pd.concat([one_elite, two_semi_elite], ignore_index=True)
    api = RestrictedAnalysisAPI(protected_dataframe=dataframe)

    with pytest.raises(
        ValueError,
        match="requires at least two elite and two semi-elite athletes",
    ):
        api.variance_analysis(
            variables=list(DOMAIN_ORDER),
            iterations=100,
            visualization=False,
        )


@pytest.mark.parametrize("size", [20, 50, 80])
def test_figure2_all_athletes_supports_dashboard_sizes(size):
    result = RestrictedAnalysisAPI().figure2(
        variables=PREDICTORS, group="all", max_athletes=size, reference_group="all"
    )
    assert result["profile_count"] == size
    assert len(result["table"]) == size
    title = result["figure"].axes[0].get_title()
    assert "All athletes" in title
    assert f"Showing {size} of {len(load_data())} athletes" in title
    assert result["summary"] == (
        f"Figure 2-style z-score profiles: All athletes\n"
        f"Showing {size} of {len(load_data())} athletes"
    )


def test_figure2_all_option_returns_complete_anonymous_dataset():
    result = RestrictedAnalysisAPI().figure2(
        variables=PREDICTORS, group="all", max_athletes=None, reference_group="all"
    )
    total = len(load_data())
    assert result["profile_count"] == total
    assert len(result["table"]) == total
    assert f"Showing all {total} athletes" in result["figure"].axes[0].get_title()
    assert result["summary"] == (
        f"Figure 2-style z-score profiles: All athletes\nShowing all {total} athletes"
    )
    assert "Athlete_" not in repr(result["table"])
    assert all(row["anonymous_profile_label"] == f"Profile {index:02d}"
               for index, row in enumerate(result["table"], start=1))


def test_figure2_fifty_is_not_limited_to_higher_expertise_group():
    result = run_figure2(group="all", max_athletes=50)
    assert result["profile_count"] == 50
    assert result["profile_count"] > int((load_data()["elite_status"] == 1).sum())


def test_figure2_table_and_chart_use_one_sample():
    first = run_figure2(group="all", max_athletes=20)
    second = run_figure2(group="all", max_athletes=20)
    assert first["table"] == second["table"]
    first_lines = [line.get_ydata().tolist() for line in first["figure"].axes[0].lines[:20]]
    second_lines = [line.get_ydata().tolist() for line in second["figure"].axes[0].lines[:20]]
    assert first_lines == second_lines


def test_figure2_uses_reproducible_random_order_once(monkeypatch):
    import sports.figures
    from sports.analysis import FIGURE2_RANDOM_SEED

    source = load_data()
    captured = []

    def fake_figure(*, selected_dataframe, **kwargs):
        captured.append(selected_dataframe.index.tolist())
        return None

    monkeypatch.setattr(sports.figures, "create_figure2", fake_figure)
    first = run_figure2(group="all", max_athletes=20)
    second = run_figure2(group="all", max_athletes=20)
    expected = source.sample(n=20, random_state=FIGURE2_RANDOM_SEED).index.tolist()
    assert captured[0] == captured[1] == expected
    assert captured[0] != source.index[:20].tolist()
    assert [row["anonymous_profile_label"] for row in first["table"]] == [
        f"Profile {index:02d}" for index in range(1, 21)
    ]
    assert "Athlete_" not in repr(first["table"])
    assert first["table"] == second["table"]
