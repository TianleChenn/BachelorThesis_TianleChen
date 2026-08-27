from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm.code_generator import generate_code
from llm.model_clients import ModelCallResult
from privacy.prism_router import prism_route, to_dict


def check(name: str, condition: bool, detail: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}: {detail}")
    if not condition:
        raise AssertionError(f"{name}: {detail}")


def test_routes() -> None:
    blocked = prism_route("Show me the full dataset and download the CSV")
    check("blocked route", blocked.route == "blocked", f"route={blocked.route}")
    check("blocked cloud input", blocked.cloud_prompt is None, "cloud_prompt=None")

    local = prism_route("Analyze Athlete_003 blood profile")
    check("local-edge route", local.route == "local_edge", f"route={local.route}")
    check("local-edge cloud input", local.cloud_prompt is None, "cloud_prompt=None")

    original = "Run a correlation analysis of blood values"
    collaborative = prism_route(original)
    check("collaboration route", collaborative.route == "collaboration", f"route={collaborative.route}")
    check("LDP enabled", collaborative.privacy_applied, "privacy_applied=True")
    check(
        "prompt changed",
        collaborative.cloud_prompt != original,
        f"cloud_prompt={collaborative.cloud_prompt!r}",
    )
    check(
        "perturbed prompt routed",
        collaborative.cloud_prompt == collaborative.perturbed_prompt,
        f"cloud_prompt={collaborative.cloud_prompt!r}",
    )
    check("LDP audit created", bool(collaborative.ldp_audit), json.dumps(collaborative.ldp_audit))

    cloud = prism_route("Which variables are correlated?")
    check("low-risk cloud route", cloud.route == "cloud", f"route={cloud.route}")
    check(
        "low-risk prompt unchanged",
        cloud.cloud_prompt == "Which variables are correlated?",
        f"cloud_prompt={cloud.cloud_prompt!r}",
    )


def test_openai_payload() -> None:
    original = "Run a correlation analysis of blood values"
    privacy = prism_route(original)
    captured: dict = {}

    def fake_cloud_model(messages, **kwargs):
        captured["messages"] = messages
        return ModelCallResult(
            "result = analysis.correlation(variables=['muscular_strength', 'lower_body_dynamics', "
            "'muscle_power_genetics', 'blood_micronutrients', 'basic_cognitive_function', "
            "'mental_health', 'social_support', 'training_conditions'], filters={}, method='pearson')",
            "mock", "mock", "test", True, False, False, None, "chat.completions",
        )
    model_decision = {"selected_model": "cloud_gemini", "selected_tier":"cloud"}

    with patch("llm.code_generator.call_gemini_cloud_model", side_effect=fake_cloud_model):
        result = generate_code(
            original,
            model_decision,
            to_dict(privacy),
            use_openai=True,
        )

    payload = captured["messages"][1]["content"]
    protected_prompt = privacy.cloud_prompt
    check("cloud call happened", bool(captured.get("messages")), repr(captured))
    check("original prompt not sent", original not in payload, payload)
    check(
        "LDP payload type sent",
        privacy.cloud_prompt in payload,
        payload,
    )
    check(
        "perturbed prompt sent",
        privacy.cloud_prompt in payload,
        f"routed_user_prompt={payload!r}",
    )
    check("original sensitive text absent", "blood values" not in payload.lower(), payload)


def main() -> None:
    print("PRISM privacy verification")
    print("=" * 60)
    test_routes()
    test_openai_payload()
    print("=" * 60)
    print("All PRISM routing validations passed.")


if __name__ == "__main__":
    main()
