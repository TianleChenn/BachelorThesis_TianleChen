"""Minimal four-value LLM privacy assessor used exclusively by Method B."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from llm.model_clients import call_privacy_risk_model
from llm.model_config import get_strong_model_name

SCHEMA_VERSION = "method-b-minimal-4d-blocked-v2"
CACHE_PATH = Path(__file__).resolve().parents[1] / "artifacts/llm_method_b_4d_privacy_cache.json"
MINIMAL_4D_SYSTEM_PROMPT = """Return JSON only with exactly these five fields. The four numeric values must be between 0 and 1, and blocked_request must be a JSON boolean:

{
  "privacy_risk_score": 0.0,
  "subject_scope": 0.0,
  "data_sensitivity": 0.0,
  "disclosure_level": 0.0,
  "blocked_request": false
}

Evaluate the privacy characteristics of the supplied user query.
Set blocked_request to true only for an explicit request for raw or exact athlete values, complete records or datasets, reconstruction of original records, or export/download of raw records. Do not block a request only because it is sensitive or individual-level.
Do not answer the user query.
Return JSON only. Do not return explanations or any additional keys."""
FEATURE_KEYS = (
    "privacy_risk_score", "subject_scope", "data_sensitivity", "disclosure_level"
)


@dataclass
class Minimal4DPrivacyAssessment:
    privacy_risk_score: float | None
    subject_scope: float | None
    data_sensitivity: float | None
    disclosure_level: float | None
    requested_model: str
    actual_model: str | None
    provider: str
    success: bool
    error: str | None
    cache_used: bool = False
    blocked_request: bool = False

    def to_dict(self) -> dict:
        return {**{key: getattr(self, key) for key in FEATURE_KEYS},
                "blocked_request": self.blocked_request}


def build_minimal_4d_messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": MINIMAL_4D_SYSTEM_PROMPT},
            {"role": "user", "content": str(prompt).strip()}]


def parse_minimal_4d_response(content: str, *, requested_model: str = "",
                              actual_model: str | None = None,
                              provider: str = "OpenAI") -> Minimal4DPrivacyAssessment:
    if not isinstance(content, str) or content.strip().startswith("```"):
        raise ValueError("Method B assessment must be plain JSON")
    data = json.loads(content)
    response_keys = set(FEATURE_KEYS) | {"blocked_request"}
    if not isinstance(data, dict) or set(data) != response_keys:
        raise ValueError("Method B assessment must contain exactly four 4D features and blocked_request")
    values = []
    for key in FEATURE_KEYS:
        value = data[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a JSON number")
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{key} must be finite and in [0, 1]")
        values.append(value)
    if type(data["blocked_request"]) is not bool:
        raise ValueError("blocked_request must be a JSON boolean")
    return Minimal4DPrivacyAssessment(*values, requested_model, actual_model, provider,
                                      True, None, blocked_request=data["blocked_request"])


def _cache_key(prompt: str, model: str) -> str:
    normalized = " ".join(prompt.casefold().split())
    return hashlib.sha256(f"{normalized}\0{model}\0{SCHEMA_VERSION}".encode()).hexdigest()


def _load_cache() -> dict:
    try:
        value = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def assess_privacy_minimal_4d(prompt: str, *, use_cache: bool = True) -> Minimal4DPrivacyAssessment:
    prompt = str(prompt or "").strip()
    if not prompt:
        raise ValueError("Method B privacy prompt must not be empty")
    model = get_strong_model_name()
    enabled = use_cache and os.getenv("PRIVACY_RISK_CACHE_ENABLED", "true").lower() not in {"0", "false", "no"}
    key = _cache_key(prompt, model)
    cache = _load_cache() if enabled else {}
    if key in cache:
        record = dict(cache[key]["assessment"])
        record["cache_used"] = True
        return Minimal4DPrivacyAssessment(**record)
    try:
        call = call_privacy_risk_model(build_minimal_4d_messages(prompt), temperature=0.0, max_tokens=160)
        if not call.success or not call.content:
            raise RuntimeError(call.error or "Method B privacy assessor unavailable")
        result = parse_minimal_4d_response(call.content, requested_model=call.requested_model,
                                           actual_model=call.actual_model, provider=call.provider)
        if enabled:
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            cache[key] = {"assessment": asdict(result), "schema_version": SCHEMA_VERSION,
                          "timestamp": datetime.now(timezone.utc).isoformat()}
            temporary = CACHE_PATH.with_suffix(".tmp.json")
            temporary.write_text(json.dumps(cache, indent=2), encoding="utf-8")
            temporary.replace(CACHE_PATH)
        return result
    except Exception as exc:
        return Minimal4DPrivacyAssessment(None, None, None, None, model, None, "OpenAI", False,
                                          f"{type(exc).__name__}: {exc}", blocked_request=False)
