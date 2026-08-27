from __future__ import annotations

import os
import json

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def close_all_matplotlib_figures():
    """
    Prevent figures created by one test from remaining open during
    later tests.
    """
    yield
    plt.close("all")


@pytest.fixture(autouse=True)
def mock_privacy_risk_llm(monkeypatch):
    """Never let the test suite contact the external privacy-assessment model."""
    from llm.model_clients import ModelCallResult
    from privacy import llm_privacy_assessor
    from privacy.prism_router import sensitivity_profile

    monkeypatch.setenv("PRIVACY_RISK_CACHE_ENABLED", "false")

    def fake_call(messages, **kwargs):
        prompt = str(messages[-1].get("content") or "")
        risk, _entities, _mask, flags = sensitivity_profile(prompt)
        categories = ["RAW_DATA"] if flags.get("has_raw_field") else []
        individual = bool(flags.get("individual_analysis_present"))
        aggregate = bool(flags.get("aggregate_analysis_present"))
        anonymous_individual = individual and "anonymous" in prompt.casefold()
        risk = .97 if flags.get("has_hard_block") else .58 if anonymous_individual else .80 if individual else .25 if aggregate and flags.get("has_sensitive_domain") else .08 if aggregate else risk
        analysis_type = "individual_profile" if individual else "descriptive_statistics" if aggregate else "general_explanation"
        payload = {
            "privacy_risk_score": risk,
            "subject_scope": .68 if anonymous_individual else .96 if individual else .15 if aggregate and flags.get("has_sensitive_domain") else .04,
            "data_sensitivity": .50 if flags.get("has_sensitive_domain") else .46 if anonymous_individual else .12,
            "disclosure_level": .98 if flags.get("has_hard_block") else .55 if anonymous_individual else .58 if individual else .25 if aggregate and flags.get("has_sensitive_domain") else .10 if aggregate else .08,
            "analysis_type": analysis_type,
            "blocked_request": bool(flags.get("has_hard_block")),
            "sensitive_categories": categories,
            "explanation": "Mocked semantic assessment for deterministic tests.",
            "confidence": 1.0,
        }
        return ModelCallResult(json.dumps(payload), "gpt-4.1", "mock-gpt-4.1", "mock", True, False, False, None)

    monkeypatch.setattr(llm_privacy_assessor, "call_privacy_risk_model", fake_call)
