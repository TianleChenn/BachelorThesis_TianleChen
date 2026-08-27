import pandas as pd
import pytest

from llm.analysis_request_contracts import build_request_contract
from llm.code_generation_pools import ALLOWED_VALUE_POOL
from llm.generated_code_verifier import inspect_generated_code
from sports.filters import CANONICAL_FILTER_VALUES, apply_analysis_filters


def test_ice_hockey_is_the_shared_canonical_sport_literal():
    assert "ice hockey" in CANONICAL_FILTER_VALUES["sport"]
    assert "icehockey" not in CANONICAL_FILTER_VALUES["sport"]
    assert ALLOWED_VALUE_POOL["sports"] == list(CANONICAL_FILTER_VALUES["sport"])
    assert build_request_contract(
        "correlation", {"sport": "ice hockey"}
    ).arguments["filters"] == {"sport": "ice hockey"}


def test_compact_sport_spelling_is_rejected_at_contract_and_backend_boundaries():
    with pytest.raises(ValueError, match="Invalid canonical value"):
        build_request_contract("correlation", {"sport": "icehockey"})

    dataframe = pd.DataFrame({"sport": ["ice hockey"]})
    with pytest.raises(ValueError, match="Invalid canonical value"):
        apply_analysis_filters(dataframe, {"sport": "icehockey"})


def test_request_validator_requires_the_same_canonical_filter_literal():
    code = (
        "result = analysis.correlation("
        "variables=['muscular_strength', 'lower_body_dynamics', "
        "'muscle_power_genetics', 'blood_micronutrients', "
        "'basic_cognitive_function', 'mental_health', 'social_support', "
        "'training_conditions'], filters={'sport': 'ice hockey'}, method='pearson')"
    )
    result = inspect_generated_code(
        code,
        user_request="Correlate all domains for ice hockey athletes.",
        requested_analysis="correlation",
        requested_filters={"sport": "ice hockey"},
    )
    assert result.request_match_passed

    compact_code = code.replace("ice hockey", "icehockey")
    compact_result = inspect_generated_code(
        compact_code,
        user_request="Correlate all domains for ice hockey athletes.",
        requested_analysis="correlation",
        requested_filters={"sport": "ice hockey"},
    )
    assert not compact_result.request_match_passed
