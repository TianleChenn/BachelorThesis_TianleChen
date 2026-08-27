import math

from privacy.prism_router import sensitivity_profile


def test_prism_raw_risk_is_direct_weight_sum():
    risk, entities, _, flags = sensitivity_profile(
        "Explain blood micronutrients in the aggregate regression"
    )
    expected_raw_risk = sum(entity.weight for entity in entities)
    assert flags["raw_risk_score"] == round(expected_raw_risk, 4)
    assert risk == round(1.0 - math.exp(-expected_raw_risk), 4)


def test_old_custom_multipliers_are_not_used():
    _, entities, _, flags = sensitivity_profile("Explain blood micronutrients")
    assert flags["raw_risk_score"] == round(
        sum(entity.weight for entity in entities), 4
    )


def test_first_person_activates_context_indicator():
    _, entities, mask, flags = sensitivity_profile("Analyze my blood information")
    assert flags["context_indicator"] == 1
    assert flags["has_first_person"] is True
    assert len(mask) == len(entities)
    assert all(value == 1 for value in mask)


def test_non_person_general_prompt_has_zero_delta():
    _, entities, _, flags = sensitivity_profile("What analyses are available?")
    assert flags["context_indicator"] == 0
    assert not any(
        entity.category in {"ATHLETE_ID", "PERSON_CONTEXT"}
        for entity in entities
    )


def test_athlete_id_is_person_identifying_context():
    _, entities, mask, flags = sensitivity_profile(
        "Generate a protected profile for Athlete_003"
    )
    assert flags["context_indicator"] == 1
    assert flags["has_person_entity"] is True
    assert len(mask) == len(entities)
