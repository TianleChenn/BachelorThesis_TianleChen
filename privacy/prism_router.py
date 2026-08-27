"""PRISM-inspired privacy router.

This module adapts the PRISM idea to the sports-analysis platform:

1. Edge-side sports-sensitive entity detection
2. Sensitivity profiling with domain-specific weights
3. Soft gating over cloud / collaboration / local_edge
4. Strict blocking for raw data and row-level requests
5. Routed low-risk or locally perturbed prompts for cloud planning
6. Sanitized LDP-style audit without exposing original sensitive values

Important design rule:
The cloud LLM never receives raw athlete rows, exact backend values, or the
original high-risk user prompt. Low-risk prompts may be routed directly, while
collaboration prompts are perturbed locally before upload.
"""

from __future__ import annotations

import logging
import math
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

from .athlete_id import ATHLETE_ID_PATTERN
from llm.env import load_local_env
from .llm_privacy_assessor import ASSESSMENT_RULES_VERSION, assess_privacy_with_llm

from .llm_soft_gating_model import (
    DEFAULT_MODEL_PATH,
    FEATURE_NAMES,
    INPUT_DIM,
    LLMPrivacySoftGater,
    build_llm_gating_features,
    load_llm_privacy_soft_gater,
    predict_llm_privacy_route,
)


SAFE_DERIVED_ONLY_PATTERNS = [
    r"without\s+(?:revealing|showing|displaying|exposing|returning|providing)?\s*(?:any\s+)?(?:raw|exact|original)\s+(?:data|values?|measurements?|answers?|responses?|rows?|records?)",
    r"(?:do not|don't|never)\s+(?:show|reveal|return|provide|display|expose).*\b(?:raw|exact|original)\b",
    r"\bderived(?:\s+(?:scores?|profile))?\s+only\b",
    r"\bstandardized\s+(?:scores?|z-scores?|profile)\s+only\b",
    r"\bprotected\s+profile\s+only\b",
]

RAW_VALUE_INTENT_PATTERNS = [
    r"\b(?:show|give|return|reveal|provide|display|list|print|export|download)\b.*\b(?:exact|raw|original)\b.*\b(?:data|values?|measurements?|answers?|responses?|rows?|records?)\b",
    r"\b(?:exact|raw|original)\b.*\b(?:data|values?|measurements?|questionnaire (?:answers?|responses?)|rows?|responses?)\b",
    r"\b(?:show|give|return|reveal|provide|display|tell me|what is|list)\b.*\b(?:values?|measurements?|questionnaire answers?|questionnaire responses?)\b",
    r"\b(?:show|give|return|reveal|provide|list|link)\b.*\bquestionnaire\b.*\b(?:scores?|answers?|responses?)\b",
    r"\ball\s+(?:raw\s+)?questionnaire answers?\b",
]
FULL_ROW_INTENT_PATTERNS = [
    r"\b(?:complete|full)\b.*\b(?:athlete|database)?\s*(?:row|record)\b",
    r"\b(?:show|print|return|provide|list)\b.*\b(?:all|every)\b.*\b(?:athlete\s+)?(?:rows?|records?)\b",
]
COMPLETE_DATASET_INTENT_PATTERNS = [
    r"\b(?:complete|full)\b.*\b(?:private\s+)?(?:dataset|database)\b",
    r"\b(?:show|print|return|provide)\b.*\b(?:entire|complete|full)\b.*\b(?:dataset|database)\b",
]
RECONSTRUCTION_INTENT_PATTERNS = [
    r"\breconstruct\b.*\b(?:raw\s+)?(?:data|values?|record)\b",
    r"\binfer\b.*\boriginal\s+value\b",
    rf"except\s+{ATHLETE_ID_PATTERN.pattern}", rf"exclude\s+{ATHLETE_ID_PATTERN.pattern}",
    rf"without\s+{ATHLETE_ID_PATTERN.pattern}", rf"all athletes but\s+{ATHLETE_ID_PATTERN.pattern}",
    r"subtract the result", r"difference between all and all except",
]
EXPORT_RAW_RECORDS_INTENT_PATTERNS = [
    r"\b(?:export|download|spreadsheet|csv)\b.*\b(?:dataset|database|rows?|records?|raw)\b",
    r"\b(?:export|download)\b.*\bcsv\b",
]

RAW_FIELD_TERMS = {
    "vitamin b12": "blood raw field",
    "b12": "blood raw field",
    "vitamin d": "blood raw field",
    "25-oh-vitamin d": "blood raw field",
    "folic acid": "blood raw field",
    "ferritin": "blood raw field",
    "blood value": "blood raw field",
    "blood values": "blood raw field",
    "polygenic score": "genetic raw field",
    "polygenic": "genetic raw field",
    "dna": "genetic raw field",
    "genotype": "genetic raw field",
    "snp": "genetic raw field",
    "genetic value": "genetic raw field",
    "phq": "mental health raw field",
    "phq-4": "mental health raw field",
    "phq4": "mental health raw field",
    "pss": "mental health raw field",
    "pss-4": "mental health raw field",
    "pss4": "mental health raw field",
    "questionnaire answer": "questionnaire raw field",
    "questionnaire answers": "questionnaire raw field",
    "pass-q": "social support raw field",
    "passq": "social support raw field",
    "mspss": "social support raw field",
    "training satisfaction": "training raw field",
    "coach satisfaction": "training raw field",
    "grip strength": "performance raw field",
    "body weight": "performance raw field",
    "sprint time": "performance raw field",
    "10m sprint": "performance raw field",
    "tapping frequency": "performance raw field",
    "jump height": "performance raw field",
    "reactive strength index": "performance raw field",
    "sergeant jump": "performance raw field",
    "competition level": "expertise raw field",
    "competition success": "expertise raw field",
    "years highest level": "expertise raw field",
    "years senior top rankings": "expertise raw field",
}

SENSITIVE_DOMAIN_TERMS = {
    "blood": "MEDICAL",
    "micronutrient": "MEDICAL",
    "micronutrients": "MEDICAL",
    "medical": "MEDICAL",
    "health": "MEDICAL",
    "genetic": "GENETIC",
    "genetics": "GENETIC",
    "dna": "GENETIC",
    "polygenic": "GENETIC",
    "mental health": "MENTAL_HEALTH",
    "stress": "MENTAL_HEALTH",
    "anxiety": "MENTAL_HEALTH",
    "depression": "MENTAL_HEALTH",
    "questionnaire": "MENTAL_HEALTH",
    "psychological": "MENTAL_HEALTH",
    "psychology": "MENTAL_HEALTH",
    "nutrition": "MEDICAL",
    "nutrition-related": "MEDICAL",
    "medical-related": "MEDICAL",
    "medical-related domains": "MEDICAL",
    "inherited trait": "GENETIC",
    "inherited traits": "GENETIC",
    "inherited-factor": "GENETIC",
    "sensitive domain": "BIOMETRIC",
    "sensitive domains": "BIOMETRIC",
    "sensitive profile": "BIOMETRIC",
    "sensitive": "BIOMETRIC",
    "high-privacy": "BIOMETRIC",
}

ANALYSIS_ACTION_TERMS = {
    "table 1": "table1",
    "logistic": "table1",
    "table 2": "table2",
    "linear": "table2",
    "correlation": "correlation",
    "variance": "variance",
    "network": "network",
    "figure 1": "figure1",
    "fig. 1": "figure1",
    "figure 2": "figure2",
    "fig. 2": "figure2",
    "profile": "profile",
    "z-score": "profile",
    "metadata": "metadata",
    "routing evaluation": "routing_evaluation",
    "local report": "local_processing",
    "local interpretation": "local_processing",
    "local environment": "local_processing",
    "local privacy protection": "local_processing",
    "entirely on the local edge": "local_processing",
    "locally": "local_processing",
}

COHORT_TERMS = [
    "all athletes",
    "elite",
    "semi-elite",
    "semi_elite",
    "female",
    "male",
    "junior national team",
    "senior national team",
    "under 20",
    "20 and above",
    "volleyball",
    "ice hockey",
    "basketball",
    "gymnastics",
    "table tennis",
    "modern pentathlon",
]

FIRST_PERSON_PATTERNS = [r"\bI\b", r"\bmy\b", r"\bme\b", r"\bmine\b"]

PSYCHOLOGICAL_PATTERNS = [
    r"mental[-\s]?health", r"psychological", r"psychology", r"questionnaire",
    r"anxiety", r"depression", r"stress", r"readiness",
]
NUTRITION_PATTERNS = [
    r"nutrition", r"nutritional", r"micronutrient", r"blood", r"vitamin",
    r"ferritin", r"folic acid",
]
INHERITED_TRAIT_PATTERNS = [
    r"genetic", r"genetics", r"inherited[-\s]?(?:trait|factor)", r"dna",
    r"genotype", r"polygenic",
]
DERIVED_REQUEST_PATTERNS = [
    r"derived", r"standardized", r"z-score", r"profile", r"score", r"summary",
    r"summarize", r"interpret", r"pattern", r"indicator", r"visuali[sz]e",
]
INDIVIDUAL_ANALYSIS_PATTERNS = [
    r"individual",
    r"single athlete",
    r"one athlete",
    r"one(?: [a-z-]+){1,3} athlete",
    r"personal profile",
    r"anonymous(?: standardized)? profile\b",
    r"anonymous athlete",
    r"randomly selected (?:elite|semi-elite) athlete",
    r"specific athlete",
    r"identifiable athlete",
    r"identifiable(?: [a-z-]+){1,3} athlete",
    r"athlete identifier",
    r"personal athlete",
    r"current_subject",
    r"randomly selected(?: [a-z-]+){0,3} athlete",
]
AGGREGATE_ANALYSIS_PATTERNS = [
    r"aggregate", r"population", r"cohort", r"group", r"sample", r"team",
    r"average", r"statistics", r"regression", r"correlation", r"variance",
    r"table\s*1", r"table\s*2", r"figure\s*1", r"figure\s*2", r"network",
]
SENSITIVITY_WEIGHTS = {
    "RAW_DATA_REQUEST": 1.00,
    "ATHLETE_ID": 0.95,
    "RAW_FIELD": 0.95,
    "MEDICAL": 0.88,
    "GENETIC": 0.90,
    "MENTAL_HEALTH": 0.90,
    "BIOMETRIC": 0.76,
    "PERSON_CONTEXT": 0.72,
    "COHORT_FILTER": 0.35,
    "ANALYSIS_ACTION": 0.18,
    "LOCAL_PROCESSING": 0.62,
    "GENERAL": 0.05,
}

CATEGORY_DOMAIN = list(SENSITIVITY_WEIGHTS.keys())

ENTITY_VALUE_POOLS = {
    # Synthetic decoys only: never populate these pools from backend data.
    "ATHLETE_ID": ["Athlete_901", "Athlete_902", "Athlete_903"],
    "RAW_FIELD": ["protected_measurement", "restricted_field", "local_only_metric"],
    "MEDICAL": ["recovery status", "wellness indicator", "nutrition status"],
    "GENETIC": ["training background", "performance trait", "inherited factor"],
    "MENTAL_HEALTH": ["well-being", "recovery perception", "psychological readiness"],
    "BIOMETRIC": ["standardized performance", "movement score", "fitness indicator"],
    "PERSON_CONTEXT": ["the user", "the participant", "the requester"],
    "COHORT_FILTER": ["the selected cohort", "an aggregate group", "the study sample"],
    "ANALYSIS_ACTION": ["aggregate analysis", "protected analysis", "statistical summary"],
    "LOCAL_PROCESSING": ["local protected analysis", "edge-only analysis"],
    "GENERAL": ["general request", "safe task", "metadata request"],
    "RAW_DATA_REQUEST": ["aggregate summary", "protected output", "non-row-level result"],
}


@dataclass
class Entity:
    value: str
    category: str
    weight: float
    protect: bool
    reason: str


@dataclass
class RoutingDecision:
    # Bounded engineering score used by the existing routing controller.
    risk_score: float | None
    # Exact PRISM sum R(P) = sum(w_ci * I(e_i)); may exceed one.
    raw_risk_score: float | None
    # PRISM contextual indicator Delta.
    context_indicator: int
    route: str
    probabilities: dict | None
    entities: list[dict]
    protection_mask: list[int]
    blocked: bool
    reason: str
    perturbed_prompt: str | None = None
    semantic_sketch: str | None = None
    privacy_method: str | None = None
    cloud_prompt: str | None = None
    ldp_audit: list[dict] | None = None
    privacy_applied: bool = False
    routing_entropy: float | None = None
    dominant_probability: float | None = None
    privacy_confidence: float | None = None
    cloud_payload_type: str = "none"
    gating_source: str = "deterministic_fallback"
    gating_feature_names: list[str] | None = None
    gating_features: list[float] | None = None
    gating_input_dim: int = INPUT_DIM
    privacy_risk_level: str | None = None
    privacy_assessment_method: str | None = None
    privacy_assessment_model: str | None = None
    privacy_assessment_actual_model: str | None = None
    privacy_assessment_provider: str | None = None
    privacy_assessment_explanation: str | None = None
    privacy_assessment_confidence: float | None = None
    privacy_assessment_fallback_used: bool = False
    privacy_assessment_error: str | None = None
    privacy_indicators: dict | None = None
    sensitive_categories: list[str] | None = None
    llm_privacy_assessment: dict | None = None
    privacy_assessment_rules_version: str | None = None
    privacy_assessment_cache_used: bool = False
    privacy_assessment_success: bool = True
    privacy_assessment_fallback_reason: str | None = None
    privacy_assessment_validation_error: str | None = None
    gating_skipped: bool = False
    gating_model_name: str | None = None
    gating_model_path: str | None = None


def to_dict(obj) -> dict:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    return dict(obj)


def _contains_any_pattern(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _has_first_person(prompt: str) -> bool:
    return _contains_any_pattern(prompt, FIRST_PERSON_PATTERNS)


def classify_raw_data_intent(prompt: str) -> dict[str, bool]:
    """Classify explicit protected-data retrieval intent without using identity as policy.

    Safe-only phrases are removed before raw-value matching, so their words do
    not accidentally become affirmative retrieval signals. Dataset, full-row,
    reconstruction, and export attacks are classified independently.
    """
    normalized = str(prompt or "").strip()
    safe_only_evidence = _contains_any_pattern(normalized, SAFE_DERIVED_ONLY_PATTERNS)
    raw_intent_text = normalized
    for pattern in SAFE_DERIVED_ONLY_PATTERNS:
        raw_intent_text = re.sub(pattern, " [SAFE_DERIVED_ONLY] ", raw_intent_text, flags=re.IGNORECASE)
    flags = {
        "raw_value_request": _contains_any_pattern(raw_intent_text, RAW_VALUE_INTENT_PATTERNS),
        "full_row_request": _contains_any_pattern(normalized, FULL_ROW_INTENT_PATTERNS),
        "complete_dataset_request": _contains_any_pattern(normalized, COMPLETE_DATASET_INTENT_PATTERNS),
        "reconstruction_request": _contains_any_pattern(normalized, RECONSTRUCTION_INTENT_PATTERNS),
        "export_raw_records_request": _contains_any_pattern(normalized, EXPORT_RAW_RECORDS_INTENT_PATTERNS),
        "safe_derived_only_evidence": safe_only_evidence,
    }
    flags["hard_block"] = any(flags[name] for name in (
        "raw_value_request", "full_row_request", "complete_dataset_request",
        "reconstruction_request", "export_raw_records_request",
    ))
    return flags


def _normalize_prompt(prompt: str) -> str:
    return str(prompt or "").strip()


def _find_athlete_ids(prompt: str) -> list[str]:
    return list(dict.fromkeys(match.group(0) for match in ATHLETE_ID_PATTERN.finditer(prompt)))


def _detect_analysis_actions(prompt: str) -> list[str]:
    query = prompt.lower()
    matches: list[tuple[int, str]] = []
    for term, action in ANALYSIS_ACTION_TERMS.items():
        match = re.search(rf"(?<!\w){re.escape(term)}(?!\w)", query)
        if match:
            matches.append((match.start(), action))
    return list(dict.fromkeys(action for _, action in sorted(matches)))


def _detect_analysis_action(prompt: str) -> str | None:
    q = prompt.lower()
    complex_stat_terms = [
        "regression",
        "correlation",
        "variance",
        "coefficient",
        "standard error",
        "standard errors",
        "robust standard error",
        "robust standard errors",
        "residual",
        "residuals",
        "diagnostic",
        "diagnostics",
        "heteroscedasticity",
        "homoscedasticity",
        "durbin-watson",
        "breusch-pagan",
        "jarque-bera",
        "cook's distance",
        "vif",
        "confidence interval",
    ]
    if (
        any(term in q for term in ["detailed", "explain", "explanation", "interpret", "with explanation"])
        and any(term in q for term in complex_stat_terms)
    ):
        return "complex_analysis"
    actions=_detect_analysis_actions(prompt)
    return actions[0] if actions else None


def _normalise_entity_value(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def _detect_group_hints(prompt: str) -> list[str]:
    text = str(prompt).lower()
    candidates: list[tuple[int, int, str]] = []
    for term in COHORT_TERMS:
        term_text = _normalise_entity_value(term)
        if not term_text:
            continue
        pattern = rf"(?<!\w){re.escape(term_text)}(?!\w)"
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            candidates.append((match.start(), match.end(), term_text))

    selected: list[tuple[int, str]] = []
    occupied: list[tuple[int, int]] = []
    for start, end, term in sorted(
        candidates,
        key=lambda item: (item[0], -(item[1] - item[0])),
    ):
        if any(
            start < used_end and end > used_start
            for used_start, used_end in occupied
        ):
            continue
        occupied.append((start, end))
        selected.append((start, term))
    return list(dict.fromkeys(term for _, term in sorted(selected)))


def _detect_group_hint(prompt: str) -> str | None:
    hints=_detect_group_hints(prompt)
    return hints[0] if hints else None


def _deduplicate_entities(entities: list[Entity]) -> list[Entity]:
    unique: list[Entity] = []
    seen: set[tuple[str, str]] = set()
    for entity in entities:
        key = (
            str(entity.category).strip().upper(),
            _normalise_entity_value(entity.value),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(entity)
    return unique


def _detect_raw_field_terms(prompt: str) -> list[str]:
    q = prompt.lower()
    return sorted({term for term in RAW_FIELD_TERMS if term in q})


def _detect_sensitive_domain_matches(prompt: str) -> list[tuple[str, str]]:
    """Return the exact sensitive text and its privacy category.

    Example:
        "blood" -> ("blood", "MEDICAL")
        "mental health" -> ("mental health", "MENTAL_HEALTH")

    Longer expressions are processed first to avoid detecting both
    "mental health" and the nested word "health".
    """
    candidates: list[tuple[int, int, str, str]] = []
    occupied_spans: list[tuple[int, int]] = []

    ordered_terms = sorted(
        SENSITIVE_DOMAIN_TERMS.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )

    for term, domain in ordered_terms:
        pattern = re.compile(
            rf"(?<!\w){re.escape(term)}(?!\w)",
            flags=re.IGNORECASE,
        )

        for match in pattern.finditer(prompt):
            start, end = match.span()

            overlaps = any(
                start < used_end and end > used_start
                for used_start, used_end in occupied_spans
            )

            if overlaps:
                continue

            candidates.append(
                (
                    start,
                    end,
                    match.group(0),
                    domain,
                )
            )
            occupied_spans.append((start, end))

    candidates.sort(key=lambda item: item[0])

    return [
        (matched_text, domain)
        for _, _, matched_text, domain in candidates
    ]


def _is_raw_request(prompt: str) -> bool:
    return bool(classify_raw_data_intent(prompt)["hard_block"])


def detect_entities(prompt: str) -> list[Entity]:
    prompt = _normalize_prompt(prompt)
    entities: list[Entity] = []

    for athlete_id in _find_athlete_ids(prompt):
        entities.append(
            Entity(
                value=athlete_id,
                category="ATHLETE_ID",
                weight=SENSITIVITY_WEIGHTS["ATHLETE_ID"],
                protect=True,
                reason="Specific athlete identifier detected.",
            )
        )

    if _is_raw_request(prompt):
        entities.append(
            Entity(
                value="[RAW_DATA_REQUEST]",
                category="RAW_DATA_REQUEST",
                weight=SENSITIVITY_WEIGHTS["RAW_DATA_REQUEST"],
                protect=True,
                reason="Prompt asks for raw, exact, full-row, or dataset-level information.",
            )
        )

    for term in _detect_raw_field_terms(prompt):
        entities.append(
            Entity(
                value=term,
                category="RAW_FIELD",
                weight=SENSITIVITY_WEIGHTS["RAW_FIELD"],
                protect=True,
                reason=f"Prompt references raw field: {term}.",
            )
        )

    for matched_text, domain in _detect_sensitive_domain_matches(prompt):
        entities.append(
            Entity(
                # Store the real text found in the prompt.
                # Two-Layer LDP will replace this exact text.
                value=matched_text,
                category=domain,
                weight=SENSITIVITY_WEIGHTS[domain],
                protect=True,
                reason=f"Prompt references sensitive domain: {domain}.",
            )
        )

    for action in _detect_analysis_actions(prompt):
        category = "LOCAL_PROCESSING" if action == "local_processing" else "ANALYSIS_ACTION"
        entities.append(
            Entity(
                value=action,
                category=category,
                weight=SENSITIVITY_WEIGHTS[category],
                protect=False,
                reason="Prompt contains a supported analysis action.",
            )
        )

    for group_hint in _detect_group_hints(prompt):
        entities.append(
            Entity(
                value=group_hint,
                category="COHORT_FILTER",
                weight=SENSITIVITY_WEIGHTS["COHORT_FILTER"],
                protect=False,
                reason="Prompt contains a cohort or subgroup filter.",
            )
        )

    for first_person_pattern in FIRST_PERSON_PATTERNS:
        for match in re.finditer(first_person_pattern, prompt, flags=re.IGNORECASE):
            entities.append(
                Entity(
                    value=match.group(0),
                    category="PERSON_CONTEXT",
                    weight=SENSITIVITY_WEIGHTS["PERSON_CONTEXT"],
                    protect=True,
                    reason="First-person context detected.",
                )
            )

    return _deduplicate_entities(entities)


def sensitivity_profile(prompt: str) -> tuple[float, list[Entity], list[int], dict]:
    """Apply PRISM entity-level sensitivity profiling.

    ``raw_risk_score`` is the exact PRISM weight sum. ``risk_score`` is a
    separate monotonic mapping to [0, 1) for the existing routing controller.
    """
    prompt = _normalize_prompt(prompt)
    raw_data_intent = classify_raw_data_intent(prompt)
    entities = detect_entities(prompt)
    athlete_ids = [e for e in entities if e.category == "ATHLETE_ID"]
    raw_fields = [e for e in entities if e.category == "RAW_FIELD"]
    raw_request = any(e.category == "RAW_DATA_REQUEST" for e in entities)
    hard_block = raw_data_intent["hard_block"]
    sensitive_domains = [
        e for e in entities if e.category in {"MEDICAL", "GENETIC", "MENTAL_HEALTH"}
    ]

    has_first_person = _has_first_person(prompt)
    has_person_entity = any(
        entity.category in {"ATHLETE_ID", "PERSON_CONTEXT"}
        for entity in entities
    )
    context_indicator = int(has_first_person or has_person_entity)
    psychological_present = _contains_any_pattern(prompt, PSYCHOLOGICAL_PATTERNS)
    nutrition_present = _contains_any_pattern(prompt, NUTRITION_PATTERNS)
    inherited_trait_present = _contains_any_pattern(prompt, INHERITED_TRAIT_PATTERNS)
    raw_value_request = raw_data_intent["raw_value_request"]
    individual_athlete_present = bool(athlete_ids) or _contains_any_pattern(prompt, INDIVIDUAL_ANALYSIS_PATTERNS)
    individual_analysis_present = individual_athlete_present
    aggregate_analysis_present = (
        _contains_any_pattern(prompt, AGGREGATE_ANALYSIS_PATTERNS)
        and not individual_analysis_present
    )
    derived_only_request = _contains_any_pattern(prompt, DERIVED_REQUEST_PATTERNS) and not raw_value_request
    semantic_domains = {
        name for name, present in (
            ("psychological", psychological_present),
            ("nutrition", nutrition_present),
            ("inherited_trait", inherited_trait_present),
        ) if present
    }

    # Sports-domain extension to PRISM contextual masking: intrinsically
    # sensitive sports entities remain protected even when Delta is zero.
    protected_categories = {
        "RAW_DATA_REQUEST",
        "ATHLETE_ID",
        "RAW_FIELD",
        "MEDICAL",
        "GENETIC",
        "MENTAL_HEALTH",
        "BIOMETRIC",
        "PERSON_CONTEXT",
    }
    for entity in entities:
        entity.protect = entity.category in protected_categories
    protection_mask = [1 if entity.protect else 0 for entity in entities]

    # Exact PRISM formula: every listed entity was detected, so I(e_i) = 1.
    raw_risk_score = sum(float(entity.weight) for entity in entities)
    bounded_risk_score = (
        1.0 - math.exp(-raw_risk_score) if raw_risk_score > 0 else 0.0
    )

    flags = {
        "has_athlete_id": bool(athlete_ids),
        "has_raw_field": bool(raw_fields),
        "has_raw_request": bool(raw_request),
        "has_hard_block": bool(hard_block),
        "full_row_request": raw_data_intent["full_row_request"],
        "complete_dataset_request": raw_data_intent["complete_dataset_request"],
        "reconstruction_request": raw_data_intent["reconstruction_request"],
        "export_raw_records_request": raw_data_intent["export_raw_records_request"],
        "safe_derived_only_evidence": raw_data_intent["safe_derived_only_evidence"],
        "has_sensitive_domain": bool(sensitive_domains),
        "has_first_person": bool(has_first_person),
        "has_person_entity": bool(has_person_entity),
        "context_indicator": context_indicator,
        "individual_athlete_present": individual_athlete_present,
        "derived_only_request": derived_only_request,
        "raw_value_request": raw_value_request,
        "multi_sensitive_domain_count": len(semantic_domains) / 3.0,
        "psychological_present": psychological_present,
        "nutrition_present": nutrition_present,
        "inherited_trait_present": inherited_trait_present,
        "individual_analysis_present": individual_analysis_present,
        "aggregate_analysis_present": aggregate_analysis_present,
        "raw_risk_score": round(raw_risk_score, 4),
        "bounded_risk_score": round(bounded_risk_score, 4),
        "analysis_action": _detect_analysis_action(prompt),
        "group_hint": _detect_group_hint(prompt),
    }
    return round(bounded_risk_score, 4), entities, protection_mask, flags


ACTIVE_PRISM_GATER_PATH = DEFAULT_MODEL_PATH
ACTIVE_GATING_MODEL_NAME = "LLM-based 4D Soft Gating"
_ACTIVE_PRISM_GATER_CACHE: LLMPrivacySoftGater | None = None
_ACTIVE_PRISM_GATER_CACHE_PATH: Path | None = None


def get_active_prism_gater_path() -> Path:
    """Use the current model by default; switch only through explicit configuration."""
    load_local_env()
    configured = os.getenv("LLM_PRIVACY_GATER_MODEL_PATH", "").strip()
    path = Path(configured) if configured else ACTIVE_PRISM_GATER_PATH
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    return path.resolve()


def load_active_prism_gater() -> LLMPrivacySoftGater:
    global _ACTIVE_PRISM_GATER_CACHE, _ACTIVE_PRISM_GATER_CACHE_PATH
    path = get_active_prism_gater_path()
    if _ACTIVE_PRISM_GATER_CACHE is not None and _ACTIVE_PRISM_GATER_CACHE_PATH == path:
        return _ACTIVE_PRISM_GATER_CACHE
    model = load_llm_privacy_soft_gater(path)
    _ACTIVE_PRISM_GATER_CACHE = model
    _ACTIVE_PRISM_GATER_CACHE_PATH = path
    return model


def trained_soft_gating_features(features: list[float]) -> dict:
    """Compatibility wrapper around the active LLM-based 4D gater."""
    if len(features) != INPUT_DIM or any(not math.isfinite(float(v)) for v in features):
        raise ValueError(f"Soft Gating requires {INPUT_DIM} finite features")
    _, probabilities = predict_llm_privacy_route(features, load_active_prism_gater())
    return probabilities


def _entropy(probabilities: dict) -> float:
    values = [max(float(v), 1e-12) for v in probabilities.values()]
    return round(-sum(v * math.log(v) for v in values), 4)


def _randomized_response(true_value: str, domain: list[str], epsilon: float, rng: random.Random) -> str:
    """Apply k-ary randomized response over a finite domain."""
    unique_domain = list(dict.fromkeys(str(item) for item in domain if str(item)))
    if true_value not in unique_domain:
        unique_domain.insert(0, true_value)
    if len(unique_domain) == 1:
        return unique_domain[0]
    keep_probability = math.exp(epsilon) / (math.exp(epsilon) + len(unique_domain) - 1)
    if rng.random() < keep_probability:
        return true_value
    return rng.choice([item for item in unique_domain if item != true_value])


def two_layer_ldp(
    prompt: str,
    entities: list[Entity],
    epsilon_total: float = 1.0,
    alpha: float = 0.6,
    seed: int | None = None,
) -> tuple[str, list[dict]]:
    """Sanitized two-layer LDP-style prompt obfuscation.

    The audit never contains original sensitive values.
    """
    if epsilon_total <= 0:
        raise ValueError("epsilon_total must be greater than zero")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    rng = random.SystemRandom() if seed is None else random.Random(seed)
    perturbed = prompt
    audit: list[dict] = []

    # Replace longer text first.
    # This prevents "polygenic" from replacing part of
    # "polygenic score" before the complete phrase is handled.
    protected_entities = [
        (index, entity)
        for index, entity in enumerate(entities)
        if entity.protect and entity.value and entity.value in prompt
    ]

    protected_entities.sort(
        key=lambda item: len(item[1].value),
        reverse=True,
    )

    for index, entity in protected_entities:

        weight = min(max(entity.weight, 0.0), 1.0)
        denom = weight + (1 - weight) * alpha
        eps1 = epsilon_total * (weight / denom) if denom else 0.0
        eps2 = epsilon_total - eps1

        noisy_category = _randomized_response(entity.category, CATEGORY_DOMAIN, eps1, rng)
        decoy_values = list(ENTITY_VALUE_POOLS.get(noisy_category, ["protected value"]))
        if noisy_category == entity.category:
            noisy_value = _randomized_response(entity.value, [entity.value, *decoy_values], eps2, rng)
        else:
            noisy_value = rng.choice(decoy_values)

        # Replace the exact sensitive text detected in the prompt.
        # subn also returns how many replacements were made.
        pattern = re.compile(
            rf"(?<!\w){re.escape(entity.value)}(?!\w)",
            flags=re.IGNORECASE,
        )

        perturbed, replacement_count = pattern.subn(
            lambda _: noisy_value,
            perturbed,
        )

        audit.append(
            {
                "entity_index": index,
                "entity_category": entity.category,
                "protected": True,
                "replacement_applied": replacement_count > 0,
                "replacement_count": replacement_count,
                "original_value_visible": False,
                "original_value_preview": "[REDACTED]",
                "noisy_category": noisy_category,
                "noisy_value_preview": noisy_value,
                "category_changed": noisy_category != entity.category,
                "value_changed": noisy_value.casefold() != entity.value.casefold(),
                "epsilon_total": round(epsilon_total, 4),
                "epsilon_1_category": round(eps1, 4),
                "epsilon_2_value": round(eps2, 4),
            }
        )

    return perturbed, audit


def _llm_gating_features(assessment) -> list[float]:
    return build_llm_gating_features(assessment)


def _safe_local_assessment_fallback(assessment=None, error: Exception | None = None) -> RoutingDecision:
    error_text = assessment.error if assessment is not None else f"{type(error).__name__}: {str(error)[:300]}"
    explanation = (assessment.explanation if assessment is not None else
        "The LLM privacy assessor was unavailable. No semantic risk scores were generated, and the request was kept on the local edge as a safe fallback.")
    requested_model = assessment.requested_model if assessment is not None else None
    public_json = assessment.to_dict() if assessment is not None else {
        "privacy_risk_score": None, "subject_scope": None,
        "data_sensitivity": None, "disclosure_level": None, "analysis_type": "unknown",
        "blocked_request": False, "sensitive_categories": [], "explanation": explanation,
        "confidence": None}
    return RoutingDecision(
        risk_score=None, raw_risk_score=None, context_indicator=0, route="local_edge",
        probabilities=None, entities=[], protection_mask=[], blocked=False, reason=explanation,
        privacy_method="llm_privacy_assessment", privacy_applied=True, cloud_payload_type="none",
        gating_source="safe_local_fallback", gating_feature_names=list(FEATURE_NAMES),
        gating_features=None,
        privacy_assessment_method="llm_semantic_assessment",
        privacy_assessment_model=requested_model, privacy_assessment_provider="OpenAI",
        privacy_assessment_explanation=explanation, privacy_assessment_fallback_used=True,
        privacy_assessment_error=error_text, llm_privacy_assessment=public_json,
        privacy_assessment_rules_version=ASSESSMENT_RULES_VERSION,
        privacy_assessment_success=False, privacy_assessment_fallback_reason="assessor_unavailable_or_invalid",
        privacy_assessment_validation_error=error_text, gating_skipped=True,
        gating_model_name=ACTIVE_GATING_MODEL_NAME,
        gating_model_path=str(get_active_prism_gater_path()))


def prism_route(prompt: str) -> RoutingDecision:
    """Route production requests from an LLM semantic privacy assessment."""
    try:
        assessment = assess_privacy_with_llm(prompt, use_cache=True)
        logger.info("prism_route_received_assessment success=%s fallback=%s",
            assessment.success, assessment.fallback_used)
        if not assessment.success:
            return _safe_local_assessment_fallback(assessment=assessment)
        if assessment.blocked_request:
            return RoutingDecision(
                risk_score=assessment.privacy_risk_score,
                raw_risk_score=assessment.privacy_risk_score,
                context_indicator=int(assessment.subject_scope >= .5),
                route="blocked", probabilities=None, entities=[], protection_mask=[],
                blocked=True,
                reason="Blocked by the semantic privacy assessment before Soft Gating.",
                privacy_method="llm_privacy_assessment", privacy_applied=True,
                gating_source="llm_blocked_request_override",
                gating_feature_names=list(FEATURE_NAMES), gating_features=None,
                gating_input_dim=INPUT_DIM, gating_skipped=True,
                gating_model_name=ACTIVE_GATING_MODEL_NAME,
                gating_model_path=str(get_active_prism_gater_path()),
                privacy_assessment_method="llm_semantic_assessment",
                privacy_assessment_model=assessment.requested_model,
                privacy_assessment_actual_model=assessment.actual_model,
                privacy_assessment_provider=assessment.provider,
                privacy_assessment_explanation=assessment.explanation,
                privacy_assessment_confidence=assessment.confidence,
                privacy_assessment_fallback_used=assessment.fallback_used,
                privacy_assessment_error=assessment.error,
                sensitive_categories=assessment.sensitive_categories,
                llm_privacy_assessment=assessment.to_dict(),
                privacy_assessment_rules_version=ASSESSMENT_RULES_VERSION,
                privacy_assessment_cache_used=assessment.cache_used,
                privacy_assessment_success=True,
            )
        features = _llm_gating_features(assessment)
    except Exception as exc:
        return _safe_local_assessment_fallback(error=exc)

    model_probabilities = trained_soft_gating_features(features)
    route = max(model_probabilities, key=model_probabilities.get)

    probs = {key: round(float(value), 4) for key, value in model_probabilities.items()}
    probs["local_edge"] = round(1.0 - probs.get("cloud", 0.0) - probs.get("collaboration", 0.0), 4)
    perturbed_prompt = cloud_prompt = None
    ldp_audit = []
    privacy_applied = route in {"collaboration", "local_edge", "blocked"}
    payload_type = "none"
    reason = "The LLM privacy assessment and Soft Gating selected this route."
    if route == "cloud":
        cloud_prompt = prompt
        payload_type = "original_prompt"
    elif route == "collaboration":
        # Local entity matching is retained solely to apply LDP; it does not score or route.
        ldp_entities = detect_entities(prompt)
        perturbed_prompt, ldp_audit = two_layer_ldp(prompt, ldp_entities)
        cloud_prompt = perturbed_prompt
        payload_type = "two_layer_ldp_perturbed_prompt"
    elif route == "blocked":
        reason = "Blocked by the semantic privacy assessment before model execution."

    indicators = {name: float(getattr(assessment, name)) for name in FEATURE_NAMES[1:]}
    entropy = _entropy(model_probabilities)
    return RoutingDecision(
        risk_score=assessment.privacy_risk_score, raw_risk_score=assessment.privacy_risk_score,
        context_indicator=int(assessment.subject_scope >= .5), route=route, probabilities=probs,
        entities=[{"category": category} for category in assessment.sensitive_categories],
        protection_mask=[], blocked=route == "blocked", reason=reason,
        perturbed_prompt=perturbed_prompt, privacy_method="llm_privacy_assessment",
        cloud_prompt=cloud_prompt, ldp_audit=ldp_audit, privacy_applied=privacy_applied,
        routing_entropy=entropy, dominant_probability=round(max(model_probabilities.values()), 4),
        privacy_confidence=assessment.confidence, cloud_payload_type=payload_type,
        gating_source="llm_risk_score_plus_soft_gating", gating_feature_names=list(FEATURE_NAMES),
        gating_features=features,
        privacy_assessment_method="llm_semantic_assessment",
        privacy_assessment_model=assessment.requested_model,
        privacy_assessment_actual_model=assessment.actual_model,
        privacy_assessment_provider=assessment.provider,
        privacy_assessment_explanation=assessment.explanation,
        privacy_assessment_confidence=assessment.confidence,
        privacy_assessment_fallback_used=assessment.fallback_used,
        privacy_assessment_error=assessment.error, privacy_indicators=indicators,
        sensitive_categories=assessment.sensitive_categories,
        llm_privacy_assessment=assessment.to_dict(),
        privacy_assessment_rules_version=ASSESSMENT_RULES_VERSION,
        privacy_assessment_cache_used=assessment.cache_used,
        privacy_assessment_success=assessment.success,
        gating_model_name=ACTIVE_GATING_MODEL_NAME,
        gating_model_path=str(get_active_prism_gater_path()),
    )


def evaluate_prism_router() -> dict:
    tests = [
        ("What analyses are available?", "cloud"),
        ("Generate Figure 1-style group statistics", "cloud"),
        ("Run Table 2-style linear regression on the elite group", "cloud"),
        ("Explain blood micronutrients in the aggregate regression", "collaboration"),
        ("Analyze the relationship between genetics and lower-body dynamics", "collaboration"),
        ("Analyze Athlete_003 using a protected profile", "local_edge"),
        ("Generate the standardized z-score profile for Athlete_010.", "local_edge"),
        ("Show me the full dataset", "blocked"),
        ("Download the CSV", "blocked"),
        ("List all athlete rows with blood values", "blocked"),
        ("What is Athlete_003 vitamin B12 value?", "blocked"),
        ("Show PHQ4 of Athlete_010.", "blocked"),
    ]
    rows = []
    correct = 0
    for prompt, expected in tests:
        decision = prism_route(prompt)
        ok = decision.route == expected
        correct += int(ok)
        rows.append(
            {
                "prompt": prompt,
                "expected_route": expected,
                "predicted_route": decision.route,
                "risk_score": decision.risk_score,
                "raw_risk_score": decision.raw_risk_score,
                "context_indicator": decision.context_indicator,
                "cloud_payload_type": decision.cloud_payload_type,
                "correct": ok,
            }
        )
    return {"accuracy": round(correct / len(tests), 4), "rows": rows, "n": len(tests)}
