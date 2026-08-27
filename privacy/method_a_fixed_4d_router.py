"""Comparison-only Method A: deterministic 4D features with the shared gater."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from privacy.llm_soft_gating_model import FEATURE_NAMES
from privacy.prism_router import sensitivity_profile, trained_soft_gating_features


@dataclass(frozen=True)
class MethodAFixed4DDecision:
    route: str
    features: list[float]
    probabilities: dict[str, float] | None
    risk_score: float
    hard_blocked: bool
    gating_source: str


def _categories(entities) -> set[str]:
    return {str(entity.category).upper() for entity in entities}


def _is_general_methodology(prompt: str, entities) -> bool:
    """Identify method explanations that request no athlete/cohort result."""
    text = str(prompt).casefold()
    categories = _categories(entities)
    explanation = bool(re.search(
        r"\b(?:what\s+is|what\s+are|explain|define|describe|how\s+does|how\s+do)\b",
        text,
    ))
    targeted = bool(categories & {
        "ATHLETE_ID", "COHORT_FILTER", "PERSON_CONTEXT", "RAW_DATA_REQUEST",
        "RAW_FIELD", "MEDICAL", "GENETIC", "MENTAL_HEALTH", "BIOMETRIC",
    }) or "athlete" in text
    return explanation and not targeted


def _subject_scope(prompt: str, entities, flags: dict) -> float:
    text = str(prompt).casefold()
    categories = _categories(entities)
    if _is_general_methodology(prompt, entities):
        return 0.0
    if flags.get("has_athlete_id") or re.search(r"\b(?:identifiable|specified)\b.*\bathlete\b", text):
        return 1.0
    if flags.get("individual_analysis_present"):
        return 0.75
    cohort_values = [str(entity.value).casefold() for entity in entities if entity.category == "COHORT_FILTER"]
    if cohort_values:
        if set(cohort_values) == {"all athletes"}:
            return 0.15
        restrictive = sum(value not in {"all athletes"} for value in cohort_values)
        if restrictive >= 2 and "compare" not in text:
            return 0.55
        return 0.35
    if flags.get("aggregate_analysis_present") or "athlete" in text:
        return 0.15
    if "PERSON_CONTEXT" in categories and flags.get("has_person_entity"):
        return 0.75
    return 0.0


def _data_sensitivity(prompt: str, entities, flags: dict) -> float:
    text = str(prompt).casefold()
    categories = _categories(entities)
    if _is_general_methodology(prompt, entities):
        return .05
    scores = [0.0]
    category_scores = {"GENETIC": 1.0, "MENTAL_HEALTH": .90, "MEDICAL": .85, "BIOMETRIC": .65}
    scores.extend(category_scores[category] for category in categories if category in category_scores)
    domain_patterns = (
        (1.00, r"\b(?:genetic|genetics|inherited|dna|genotype|polygenic)\b"),
        (.90, r"\b(?:mental health|psychological|anxiety|depression|stress|questionnaire)\b"),
        (.85, r"\b(?:blood|nutrition|micronutrient|medical|vitamin|ferritin|folic acid)\b"),
        (.65, r"\b(?:biometric|sensitive profile)\b"),
        (.55, r"\b(?:basic cognitive|cognitive function)\b"),
        (.45, r"\b(?:muscular strength|lower-body|lower body|grip strength|sprint|jump height)\b"),
        (.40, r"\b(?:social support|pass-q|passq|mspss)\b"),
        (.30, r"\b(?:training conditions|training satisfaction|coach satisfaction)\b"),
    )
    scores.extend(score for score, pattern in domain_patterns if re.search(pattern, text))
    if max(scores) == 0.0:
        if flags.get("aggregate_analysis_present") or flags.get("individual_analysis_present") or "athlete" in text:
            scores.append(.15)
        elif "ANALYSIS_ACTION" in categories:
            scores.append(.05)
    return max(scores)


def _disclosure_level(prompt: str, entities, flags: dict) -> float:
    text = str(prompt).casefold()
    if _is_general_methodology(prompt, entities):
        return .05
    if any(flags.get(name) for name in ("full_row_request", "complete_dataset_request",
            "export_raw_records_request", "reconstruction_request")):
        return 1.0
    if flags.get("raw_value_request") or flags.get("has_raw_request"):
        return .90
    if flags.get("individual_analysis_present"):
        if flags.get("has_athlete_id") or re.search(r"\b(?:specific|identifiable)\b.*\bathlete\b", text):
            return .70
        return .60
    if flags.get("aggregate_analysis_present"):
        has_filtered_cohort = any(entity.category == "COHORT_FILTER" and
            str(entity.value).casefold() != "all athletes" for entity in entities)
        return .35 if has_filtered_cohort else .20
    if flags.get("analysis_action") or any(entity.category == "ANALYSIS_ACTION" for entity in entities):
        return .05
    return 0.0


def build_method_a_fixed_4d_features(prompt: str) -> tuple[list[float], list, dict]:
    risk_score, entities, _, flags = sensitivity_profile(prompt)
    features = [float(risk_score), _subject_scope(prompt, entities, flags),
        _data_sensitivity(prompt, entities, flags), _disclosure_level(prompt, entities, flags)]
    if len(features) != len(FEATURE_NAMES) or any(not math.isfinite(value) or not 0 <= value <= 1 for value in features):
        raise ValueError("Method A fixed-rule features must be four finite values in [0, 1]")
    return features, entities, flags


def route_with_method_a_fixed_4d(prompt: str) -> MethodAFixed4DDecision:
    features, _, flags = build_method_a_fixed_4d_features(prompt)
    if flags.get("has_hard_block"):
        return MethodAFixed4DDecision("blocked", features, None, features[0], True,
            "method_a_fixed_rules_hard_block")
    probabilities = trained_soft_gating_features(features)
    route = max(probabilities, key=probabilities.get)
    return MethodAFixed4DDecision(route, features, probabilities, features[0], False,
        "method_a_fixed_rules_4d_shared_soft_gating")
