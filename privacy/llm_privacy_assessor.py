"""Continuous semantic athlete privacy-risk assessment by a dedicated LLM."""

from __future__ import annotations

import hashlib, json, logging, math, os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from llm.model_clients import call_privacy_risk_model
from llm.model_config import get_strong_model_name

ASSESSMENT_RULES_VERSION = "athlete-privacy-rubric-v8-continuous-no-level"
ASSESSMENT_SCHEMA_VERSION = "continuous-4d-no-level-json-v2"
CACHE_PATH = Path(__file__).resolve().parents[1] / "artifacts/llm_privacy_risk_cache.json"
logger = logging.getLogger(__name__)
ALLOWED_ANALYSIS_TYPES = {"general_explanation", "descriptive_statistics", "logistic_regression",
    "multiple_linear_regression", "correlation_analysis", "variance_analysis", "network_analysis",
    "individual_profile", "metadata_analysis", "raw_data_lookup", "dataset_export",
    "dataset_reconstruction", "unknown"}

# 总体评估原则：LLM 必须整体理解请求，不能使用固定关键词、固定权重或简单规则计算分数。
# 评估同时考虑请求对象、运动员数据类型、披露程度、派生结果或原始数据、分析目的及完整上下文。
# 同一领域在不同场景下可以具有不同风险；例如群体血液回归与可识别个人原始血液值不同。
# 四维分别表示整体风险、请求针对谁、信息敏感程度以及用户要求披露的信息量。
PRIVACY_ASSESSMENT_SYSTEM_PROMPT = """You are the privacy risk assessor for an athlete data analysis system.
Your only task is to assess the privacy characteristics of the user request. Do not answer the request,
generate analysis code, invent athlete data, use fixed category weights, or sum predefined values.
Return four independent continuous estimates between
0.00 and 1.00 based on the complete semantic meaning of the request.

The system supports descriptive statistics, Table 1 logistic regression, Table 2 multiple linear
regression, correlation, Figure 2 variance, Figure 1 network analysis, anonymous individual profiles,
and metadata analysis. The eight derived domains are muscular strength, lower-body dynamics,
muscle-power genetics, blood micronutrients, basic cognitive function, mental health, social support,
and training conditions.

privacy_risk_score is overall privacy risk: near 0 means almost no exposure and near 1 means severe or
unacceptable exposure. Consider target scope, sensitivity, disclosure, identification/re-identification,
raw access, export and reconstruction together, without fixed-weight arithmetic.
Return privacy_risk_score as one continuous number between 0.00 and 1.00.
Do not convert the score into a category. Do not use predefined thresholds.
Do not use fixed category weights. Do not quantize the score into fixed steps.
Do not round the score to predefined values such as 0.25, 0.50, 0.75 or 1.00.
The score must reflect the complete semantic meaning and context of the request.
subject_scope asks only who is targeted: near 0 is general or broad aggregate; near 1 is one identifiable
athlete. Estimate broad groups, filtered/small cohorts and anonymous people continuously. Do not raise it
merely for sensitive or raw data.
Dimension 3

Data Sensitivity

Definition

Estimate how sensitive the requested athlete data are.

The athlete dataset contains the following analytical domains:
- Muscular Strength
- Lower-body Dynamics
- Muscle-power Genetics
- Blood Micronutrients
- Basic Cognitive Function
- Mental Health
- Social Support
- Training Conditions

These domains may have different privacy implications depending on the context of the request.
Evaluate the sensitivity based on the type of athlete data requested; whether the request involves
derived results or original data; the intended analysis; and the surrounding context of the request.
Do not determine the sensitivity solely from the domain name.
A request involving blood micronutrients, for example, may have different privacy implications depending
on whether it asks for an aggregate statistical analysis or an individual's original measurement.
A value near 0 indicates that the requested data are minimally sensitive.
A value near 1 indicates that the requested data are highly privacy-sensitive.
disclosure_level asks only how much detail is requested: near 0 is explanation without athlete results;
near 1 is original measurements, questionnaires, rows, full datasets, export or reconstruction. Estimate
summaries, visualizations, statistical analyses, derived results and anonymous profiles continuously.
Do not raise it merely because content is sensitive or individual.

Return exactly one analysis_type from: general_explanation, descriptive_statistics, logistic_regression,
multiple_linear_regression, correlation_analysis, variance_analysis, network_analysis, individual_profile,
metadata_analysis, raw_data_lookup, dataset_export, dataset_reconstruction, unknown. It is categorical.
Set blocked_request=true for identifiable protected raw data, original rows/records, questionnaire answers,
complete private datasets, bulk export/download, reconstruction, or differencing attacks. Do not block
legitimate aggregate statistics merely because their domain is sensitive. The four dimensions are
independent; do not copy the same score into every field.

General Assessment Principle

Evaluate the user request holistically.
Do not assign scores to each dimension using fixed rules, predefined keyword weights, or simple keyword matching.
Instead, consider the interaction between who the request targets; what athlete information is requested;
how much information the user asks the system to reveal; whether the request concerns derived analysis
results or original data; the intended analysis; and the complete semantic meaning and surrounding context.
The same athlete domain may produce different privacy scores under different request contexts.
For example, an aggregate statistical analysis of blood micronutrients should not receive the same privacy
assessment as a request for one identifiable athlete's original blood measurement.
The four scores must therefore reflect semantic understanding rather than keyword detection.
Assess each dimension according to its own definition: privacy_risk_score describes the overall privacy risk;
subject_scope describes who is targeted; data_sensitivity describes the sensitivity of the requested information;
and disclosure_level describes how much information is requested. Do not copy the same value into all four fields.
Return continuous values between 0.00 and 1.00.

Return JSON only with exactly: privacy_risk_score,
subject_scope, data_sensitivity, disclosure_level, analysis_type, blocked_request,
sensitive_categories, explanation, confidence. Explanation is at most two concise sentences."""


@dataclass
class PrivacyAssessmentResult:
    privacy_risk_score: float | None
    subject_scope: float | None
    data_sensitivity: float | None
    disclosure_level: float | None
    analysis_type: str
    blocked_request: bool
    sensitive_categories: list[str]
    explanation: str
    confidence: float | None
    requested_model: str
    actual_model: str | None
    provider: str
    success: bool
    fallback_used: bool
    error: str | None
    cache_used: bool = False

    @property
    def risk_score(self):
        return self.privacy_risk_score

    def to_dict(self):
        return {name: getattr(self, name) for name in ("privacy_risk_score", "subject_scope",
            "data_sensitivity", "disclosure_level", "analysis_type", "blocked_request",
            "sensitive_categories", "explanation", "confidence")}


def build_privacy_assessment_messages(prompt):
    return [{"role": "system", "content": PRIVACY_ASSESSMENT_SYSTEM_PROMPT},
        {"role": "user", "content": str(prompt).strip()}]


def parse_privacy_assessment_response(content, *, requested_model="gpt-4.1", actual_model=None,
                                      provider="openai_privacy_assessor"):
    if not isinstance(content, str) or content.strip().startswith("```"):
        raise ValueError("Privacy assessment must be plain JSON")
    data = json.loads(content)
    required = {"privacy_risk_score", "subject_scope", "data_sensitivity",
        "disclosure_level", "analysis_type", "blocked_request", "sensitive_categories",
        "explanation", "confidence"}
    if not isinstance(data, dict) or not required <= set(data):
        raise ValueError("Privacy assessment JSON is missing required fields")
    for name in ("privacy_risk_score", "subject_scope", "data_sensitivity", "disclosure_level"):
        value = data[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be a finite JSON number in [0, 1]")
    score = float(data["privacy_risk_score"])
    if data["analysis_type"] not in ALLOWED_ANALYSIS_TYPES or type(data["blocked_request"]) is not bool:
        raise ValueError("Invalid analysis_type or blocked_request")
    if not isinstance(data["sensitive_categories"], list) or any(not isinstance(x, str) for x in data["sensitive_categories"]):
        raise ValueError("sensitive_categories must be a JSON string array")
    confidence = data["confidence"]
    if confidence is not None and (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence) or not 0 <= confidence <= 1):
        raise ValueError("confidence must be null or a finite number in [0, 1]")
    explanation = data["explanation"]
    if not isinstance(explanation, str) or not explanation.strip() or explanation.count(".") > 2:
        raise ValueError("explanation must contain at most two concise sentences")
    return PrivacyAssessmentResult(score, float(data["subject_scope"]),
        float(data["data_sensitivity"]), float(data["disclosure_level"]), data["analysis_type"],
        data["blocked_request"], list(dict.fromkeys(data["sensitive_categories"])), explanation.strip(),
        None if confidence is None else float(confidence), requested_model, actual_model, provider, True, False, None)


def _cache_key(prompt, model):
    normalized = " ".join(prompt.casefold().split())
    material = f"{normalized}\0{model}\0{ASSESSMENT_RULES_VERSION}\0{ASSESSMENT_SCHEMA_VERSION}"
    return hashlib.sha256(material.encode()).hexdigest()


def _load_cache():
    try: return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError): return {}


def assess_privacy_with_llm(prompt: str, *, use_cache: bool = True) -> PrivacyAssessmentResult:
    """The single public entry point for assessment, validation, cache and fallback."""
    prompt = str(prompt or "").strip()
    if not prompt: raise ValueError("Privacy assessment prompt must not be empty")
    model = get_strong_model_name()
    prompt_hash = hashlib.sha256(" ".join(prompt.casefold().split()).encode()).hexdigest()[:16]
    logger.info("privacy_assessment_started prompt_hash=%s model=%s", prompt_hash, model)
    enabled = use_cache and os.getenv("PRIVACY_RISK_CACHE_ENABLED", "true").strip().lower() not in {"0", "false", "no"}
    key = _cache_key(prompt, model); cache = _load_cache() if enabled else {}
    if key in cache:
        record = dict(cache[key]["assessment"]); record["cache_used"] = True
        return PrivacyAssessmentResult(**record)
    try:
        call = call_privacy_risk_model(build_privacy_assessment_messages(prompt), temperature=0.0, max_tokens=500)
        if not call.success or not call.content:
            raise RuntimeError(call.error or "Privacy Risk Assessor unavailable")
        result = parse_privacy_assessment_response(call.content, requested_model=call.requested_model,
            actual_model=call.actual_model, provider=call.provider)
        logger.info("privacy_assessment_succeeded prompt_hash=%s actual_model=%s", prompt_hash, result.actual_model)
        if enabled:
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            cache[key] = {"assessment": asdict(result), "timestamp": datetime.now(timezone.utc).isoformat(),
                "rules_version": ASSESSMENT_RULES_VERSION, "schema_version": ASSESSMENT_SCHEMA_VERSION}
            temporary = CACHE_PATH.with_suffix(".tmp.json")
            temporary.write_text(json.dumps(cache, indent=2), encoding="utf-8"); temporary.replace(CACHE_PATH)
        return result
    except Exception as exc:
        logger.exception("privacy_assessment_failed prompt_hash=%s", prompt_hash)
        return PrivacyAssessmentResult(
            privacy_risk_score=None, subject_scope=None,
            data_sensitivity=None, disclosure_level=None, analysis_type="unknown",
            blocked_request=False, sensitive_categories=[],
            explanation=("The LLM privacy assessor was unavailable. No semantic risk scores were generated, "
                "and the request was kept on the local edge as a safe fallback."),
            confidence=None, requested_model=model, actual_model=None, provider="OpenAI",
            success=False, fallback_used=True, error=f"{type(exc).__name__}: {exc}")
