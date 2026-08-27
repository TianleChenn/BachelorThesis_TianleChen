import json

import pytest

from llm.model_clients import ModelCallResult
from privacy.llm_minimal_4d_privacy_assessor import FEATURE_KEYS, MINIMAL_4D_SYSTEM_PROMPT
from privacy.llm_privacy_assessor import PRIVACY_ASSESSMENT_SYSTEM_PROMPT
from privacy.prompt_ablation import (
    DEFINED_4D_PRIVACY_PROMPT,
    FULL_PRIVACY_PROMPT,
    MINIMAL_PRIVACY_PROMPT,
    PROMPT_VERSIONS,
    build_privacy_ablation_messages,
    parse_privacy_ablation_response,
    prompt_sha256,
)


def test_privacy_p1_and_p3_reuse_production_prompts_exactly():
    assert MINIMAL_PRIVACY_PROMPT is MINIMAL_4D_SYSTEM_PROMPT
    assert FULL_PRIVACY_PROMPT is PRIVACY_ASSESSMENT_SYSTEM_PROMPT
    assert build_privacy_ablation_messages("minimal", "request")[0]["content"] == MINIMAL_4D_SYSTEM_PROMPT
    assert build_privacy_ablation_messages("full", "request")[0]["content"] == PRIVACY_ASSESSMENT_SYSTEM_PROMPT


def test_defined_privacy_prompt_adds_only_dimension_definitions_to_minimal():
    assert DEFINED_4D_PRIVACY_PROMPT.startswith(MINIMAL_4D_SYSTEM_PROMPT)
    for heading in ("Privacy Risk Score:", "Subject Scope:", "Data Sensitivity:", "Disclosure Level:"):
        assert heading in DEFINED_4D_PRIVACY_PROMPT
    for full_only in (
        "General Assessment Principle",
        "analysis_type",
        "eight derived domains",
        "fixed category weights",
        "sensitive_categories",
    ):
        assert full_only not in DEFINED_4D_PRIVACY_PROMPT


def test_all_privacy_variants_normalize_to_the_same_five_field_schema():
    minimal = json.dumps({**dict(zip(FEATURE_KEYS, (0.1, 0.2, 0.3, 0.4))), "blocked_request": False})
    full = json.dumps({
        **dict(zip(FEATURE_KEYS, (0.1, 0.2, 0.3, 0.4))),
        "analysis_type": "correlation_analysis",
        "blocked_request": False,
        "sensitive_categories": [],
        "explanation": "Aggregate request.",
        "confidence": 0.9,
    })
    parsed = [
        parse_privacy_ablation_response(version, full if version == "full" else minimal)
        for version in PROMPT_VERSIONS
    ]
    assert [item.__dict__ for item in parsed] == [parsed[0].__dict__] * 3
    assert set(parsed[0].__dict__) == set(FEATURE_KEYS) | {"blocked_request"}


def test_privacy_prompt_hashes_are_version_specific_and_stable():
    hashes = [prompt_sha256(version) for version in PROMPT_VERSIONS]
    assert len(set(hashes)) == 3
    assert all(len(value) == 64 for value in hashes)


def test_privacy_variants_use_identical_model_parameters_and_common_router(monkeypatch):
    import privacy.prompt_ablation as ablation

    calls = []
    response = json.dumps({**dict(zip(FEATURE_KEYS, (0.1, 0.2, 0.3, 0.4))), "blocked_request": False})

    def caller(messages, **kwargs):
        calls.append((messages, kwargs))
        return ModelCallResult(response, "gpt-4.1", "gpt-4.1", "test", True, False, False, None)

    monkeypatch.setattr(
        ablation,
        "route_ablation_assessment",
        lambda assessment: ("cloud", {"cloud": 0.8, "collaboration": 0.1, "local_edge": 0.1}),
    )
    for version in ("minimal", "defined"):
        result = ablation.evaluate_privacy_prompt(version, "aggregate request", caller=caller)
        assert result["predicted_route"] == "cloud"
    assert [kwargs for _, kwargs in calls] == [
        {"temperature": 0.0, "max_tokens": 500},
        {"temperature": 0.0, "max_tokens": 500},
    ]


def test_privacy_summary_requires_every_group_to_complete(capsys):
    from scripts.evaluate_privacy_prompt_ablation import _require_complete

    rows = []
    for benchmark, expected in (("controlled", 32), ("independent", 60)):
        for version in PROMPT_VERSIONS:
            for index in range(expected):
                rows.append({
                    "benchmark": benchmark,
                    "prompt_version": version,
                    "sample_id": f"{benchmark}_{index}",
                    "assessment_success": not (
                        benchmark == "controlled" and version == "defined" and index == 31
                    ),
                    "predicted_route": "cloud" if not (
                        benchmark == "controlled" and version == "defined" and index == 31
                    ) else None,
                })

    with pytest.raises(RuntimeError, match="Do not use partial results in the thesis"):
        _require_complete(rows)

    output = capsys.readouterr().out
    assert "controlled/defined: 31/32" in output
    assert "python scripts/evaluate_privacy_prompt_ablation.py --resume" in output
