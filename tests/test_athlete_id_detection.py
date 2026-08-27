import pytest

from llm.generated_code_verifier import inspect_generated_code
from privacy.athlete_id import ATHLETE_ID_PATTERN, normalize_athlete_id, redact_athlete_ids
from privacy.prism_router import detect_entities


@pytest.mark.parametrize(
    "text",
    [
        "Athlete23",
        "Athlete 23",
        "Athlete_23",
        "Athlete-23",
        "Athlete:23",
        "Athlete #23",
        "athlete__023",
        "ATHLETE - 0023",
        "athlete: 300",
        "athlete no. 23",
        "athlete number 23",
        "athlete id 23",
        "athlete id: 23",
        "athlete identifier 23",
        "Show vitamin D for athlete123456",
    ],
)
def test_all_athlete_number_formats_are_detected(text):
    entities = detect_entities(text)
    matches = [entity for entity in entities if entity.category == "ATHLETE_ID"]
    assert matches
    assert matches[0].value == ATHLETE_ID_PATTERN.search(text).group(0)


@pytest.mark.parametrize(
    "text",
    [
        "athletes",
        "athlete profile",
        "athlete data",
        "athlete group",
        "athlete abc",
        "23 athletes",
    ],
)
def test_non_identifier_athlete_text_is_not_detected(text):
    entities = detect_entities(text)
    assert not any(entity.category == "ATHLETE_ID" for entity in entities)


def test_athlete_id_normalization_preserves_number_digits():
    assert normalize_athlete_id("ATHLETE - 0023") == "Athlete_0023"
    assert normalize_athlete_id("athlete id: 23") == "Athlete_23"


def test_shared_pattern_redacts_supported_formats():
    text = "Compare Athlete #23 with athlete identifier 123456."
    assert redact_athlete_ids(text) == "Compare [REDACTED_ID] with [REDACTED_ID]."


def test_generated_code_rejects_expanded_athlete_id_format():
    code = "result = analysis.correlation(variables=['mental_health'], filters={'sex': 'Athlete #23'})"
    validation = inspect_generated_code(code,user_request="correlation",
        requested_analysis="correlation",requested_filters={})
    assert validation.structure_validation_passed is False
    assert "Athlete identifiers are forbidden" in validation.validation_error
