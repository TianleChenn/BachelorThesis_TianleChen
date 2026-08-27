import pytest

from privacy.prism_router import detect_entities, sensitivity_profile
from sports.service import _build_privacy_test


def _cohorts(prompt: str):
    return [entity for entity in detect_entities(prompt) if entity.category == "COHORT_FILTER"]


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("female athletes under 20", ["female", "under 20"]),
        ("under 20 athletes under 20", ["under 20"]),
        ("female female athletes", ["female"]),
        ("female and male athletes", ["female", "male"]),
        ("table tennis female athletes", ["table tennis", "female"]),
    ],
)
def test_distinct_cohort_values_are_counted_once_each(prompt, expected):
    entities = _cohorts(prompt)
    assert [entity.value for entity in entities] == expected
    assert sum(entity.weight for entity in entities) == pytest.approx(
        0.35 * len(expected)
    )


def test_repeated_controls_are_not_cohort_entities_and_aliases_share_one_action():
    prompt = (
        "Generate Table 1 logistic regression models using no controls, sex, age, "
        "and both sex and age."
    )
    entities = detect_entities(prompt)
    assert not any(
        entity.category == "COHORT_FILTER" and entity.value in {"age", "sex"}
        for entity in entities
    )
    actions = [
        entity for entity in entities if entity.category == "ANALYSIS_ACTION"
    ]
    assert [(entity.value, entity.weight) for entity in actions] == [("table1", 0.18)]


def test_same_dna_text_is_preserved_across_distinct_categories():
    entities = detect_entities("Analyze dna")
    keys = {(entity.category, entity.value) for entity in entities}
    assert ("RAW_FIELD", "dna") in keys
    assert ("GENETIC", "dna") in keys


def test_sensitivity_profile_deduplicates_entities():
    prompt = "Generate an analysis for female athletes under 20 under 20."
    _, entities, _, _ = sensitivity_profile(prompt)
    assert [entity.value for entity in entities].count("under 20") == 1
