"""Deterministic verification and restricted dispatch of generated analysis code."""

from __future__ import annotations

import ast
import io
import math
import re
import tokenize
from dataclasses import asdict, dataclass, field
from typing import Any

from llm.analysis_request_contracts import (
    ARGUMENT_ALIASES,
    METHOD_ALIASES,
    build_request_contract,
)
from llm.error_sanitizer import sanitize_exception
from privacy.athlete_id import ATHLETE_ID_PATTERN
from sports.config import PREDICTORS
from sports.restricted_analysis_api import METHOD_ARGUMENT_SCHEMAS, RestrictedAnalysisAPI


FORBIDDEN_RAW_FIELDS = {
    "vitamin_b12", "vitamin_d", "ferritin", "folic_acid", "grip_strength",
    "body_weight", "phq4", "pss4", "genotype", "dna", "athlete_id",
}

_TRUSTED_ARGUMENT_DEFAULTS = {
    "table1": {"filters": {}},
    "table2": {"filters": {}, "group": "all"},
    "figure1": {
        "target": "expertise_value",
        "group_field": "elite_status",
        "correlation_threshold": 0.15,
        "variance_iterations": 1000,
        "filters": {},
    },
    "figure2": {
        "filters": {},
        "max_athletes": 50,
        "reference_group": "selected_cohort",
    },
    "correlation": {"filters": {}, "method": "pearson", "visualization": True},
    "variance_analysis": {
        "group_field": "elite_status",
        "groups": ["elite", "semi_elite"],
        "iterations": 1000,
        "filters": {},
        "visualization": True,
    },
    "individual_profile": {
        "reference_group": "all",
        "output_mode": "standardized_profile",
    },
}


@dataclass
class GeneratedCodeVerificationResult:
    allowed: bool = False
    executed: bool = False
    fully_correct: bool = False
    generated_method: str | None = None
    generated_arguments: dict | None = None
    expected_method: str | None = None
    expected_arguments: dict | None = None
    cleaned_code: str | None = None
    structure_validation_passed: bool = False
    request_match_passed: bool = False
    local_execution_passed: bool = False
    result_validation_passed: bool = False
    failure_stage: str | None = None
    validation_error: str | None = None
    request_mismatches: list[str] = field(default_factory=list)
    executed_locally: bool = True
    raw_data_exposed: bool = False
    restricted_execution: bool = True
    result: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def to_dict(obj: GeneratedCodeVerificationResult | dict) -> dict:
    return obj.to_dict() if isinstance(obj, GeneratedCodeVerificationResult) else dict(obj)


_CANDIDATE_START = re.compile(r"\bresult\s*=\s*analysis\.[A-Za-z_]\w*\s*\(")


def _balanced_call_end(text: str, start: int) -> int | None:
    open_index = text.find("(", start)
    depth = 0
    quote = None
    escaped = False
    if open_index < 0:
        return None
    for index in range(open_index, len(text)):
        char = text[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def _contains_comment(text: str) -> bool:
    try:
        return any(
            token.type == tokenize.COMMENT
            for token in tokenize.generate_tokens(io.StringIO(text).readline)
        )
    except tokenize.TokenError:
        return False


def _contains_statement(fragment: str) -> bool:
    fragment = "\n".join(
        line for line in fragment.splitlines()
        if line.strip() and not line.strip().startswith("```")
    ).strip()
    if not fragment:
        return False
    try:
        return bool(ast.parse(fragment, mode="exec").body)
    except SyntaxError:
        return False


def extract_restricted_assignment(text: str) -> tuple[str, int]:
    cleaned = "\n".join(
        line for line in str(text or "").strip().splitlines()
        if not line.strip().startswith("```")
    ).strip()
    candidates = []
    for match in _CANDIDATE_START.finditer(cleaned):
        end = _balanced_call_end(cleaned, match.start())
        if end is not None:
            candidates.append((match.start(), end))
    if len(candidates) != 1:
        return cleaned, len(candidates)
    start, end = candidates[0]
    before, after = cleaned[:start], cleaned[end:]
    if (_contains_statement(before) or _contains_statement(after)
            or _contains_comment(before) or _contains_comment(after)):
        return cleaned, 1
    return cleaned[start:end].strip(), 1


def clean_generated_code(text: str) -> str:
    return extract_restricted_assignment(text)[0]


def _literal(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant) and (
        node.value is None or isinstance(node.value, (str, int, float, bool))
    ):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_literal(item) for item in node.elts]
    if isinstance(node, ast.Dict):
        result = {}
        for key, value in zip(node.keys, node.values):
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                raise ValueError("Dictionary keys must be literal strings.")
            if key.value in result:
                raise ValueError("Duplicate dictionary keys are forbidden.")
            result[key.value] = _literal(value)
        return result
    if (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and type(node.operand.value) in {int, float}):
        return -node.operand.value
    raise ValueError("Arguments must contain safe literal values only.")


def _walk_values(value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_values(item)
    else:
        yield value


def _parse_structure(code: str) -> tuple[str, dict, str]:
    cleaned, candidate_count = extract_restricted_assignment(code)
    if candidate_count != 1:
        raise ValueError("Exactly one restricted analysis assignment is required.")
    if _contains_comment(cleaned):
        raise ValueError("Comments are not allowed in generated code.")
    tree = ast.parse(cleaned, mode="exec")
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        raise ValueError("Exactly one assignment is required.")
    assignment = tree.body[0]
    if (len(assignment.targets) != 1 or not isinstance(assignment.targets[0], ast.Name)
            or assignment.targets[0].id != "result"):
        raise ValueError("Assignment target must be exactly result.")
    call = assignment.value
    if not isinstance(call, ast.Call) or call.args:
        raise ValueError("result must use one keyword-only analysis call.")
    if (not isinstance(call.func, ast.Attribute)
            or not isinstance(call.func.value, ast.Name)
            or call.func.value.id != "analysis"):
        raise ValueError("Call target must be analysis.<approved_method>.")
    method = METHOD_ALIASES.get(call.func.attr, call.func.attr)
    if method not in METHOD_ARGUMENT_SCHEMAS:
        raise ValueError(f"Method '{call.func.attr}' is not allowed.")
    if any(keyword.arg is None for keyword in call.keywords):
        raise ValueError("Starred keyword arguments are forbidden.")
    names = [keyword.arg for keyword in call.keywords]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate keyword arguments are forbidden.")
    arguments = {keyword.arg: _literal(keyword.value) for keyword in call.keywords}
    normalized = {}
    for name, value in arguments.items():
        canonical = ARGUMENT_ALIASES.get(name, name)
        if canonical in normalized:
            raise ValueError(f"Duplicate canonical keyword '{canonical}' is forbidden.")
        normalized[canonical] = value
    schema = METHOD_ARGUMENT_SCHEMAS[method]
    allowed = set(schema["required"]) | set(schema["optional"])
    unexpected = set(normalized) - allowed
    missing = set(schema["required"]) - set(normalized)
    if unexpected:
        raise ValueError(f"Keyword '{sorted(unexpected)[0]}' is not allowed for {method}.")
    if missing:
        raise ValueError(f"Required keyword '{sorted(missing)[0]}' is missing for {method}.")
    lowered = [str(value).casefold() for value in _walk_values(normalized)]
    if any(value in FORBIDDEN_RAW_FIELDS for value in lowered):
        raise ValueError("Raw athlete fields are forbidden.")
    if any(ATHLETE_ID_PATTERN.fullmatch(value) for value in lowered):
        raise ValueError("Athlete identifiers are forbidden in generated code.")
    return method, normalized, cleaned


def _same_number(left: Any, right: Any) -> bool:
    return (type(left) in {int, float} and type(right) in {int, float}
            and math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12))


def _normalize_trusted_defaults(method: str, arguments: dict) -> dict:
    trusted_defaults = _TRUSTED_ARGUMENT_DEFAULTS.get(method, {})
    normalized = {
        name: (dict(value) if isinstance(value, dict) else list(value) if isinstance(value, list) else value)
        for name, value in trusted_defaults.items()
    }
    normalized.update(arguments)
    # RestrictedAnalysisAPI methods with a trusted filters={} default treat an
    # explicit None exactly like an omitted filter argument. This equivalence is
    # deliberately limited to filters; explicit requirements such as controls
    # must still match the hidden request contract.
    if trusted_defaults.get("filters") == {} and normalized.get("filters") is None:
        normalized["filters"] = {}
    return normalized


def _canonicalize_controls(controls) -> list[tuple[str, ...]] | None:
    """Normalize ordering while retaining duplicates and exact membership."""
    if not isinstance(controls, list):
        return None
    normalized = []
    for control_set in controls:
        if not isinstance(control_set, list) or any(
            not isinstance(value, str) for value in control_set
        ):
            return None
        normalized.append(tuple(sorted(control_set)))
    return sorted(normalized)


def _compare_arguments(method: str, generated: dict, expected: dict) -> list[str]:
    mismatches = []
    if set(generated) != set(expected):
        missing = sorted(set(expected) - set(generated))
        extra = sorted(set(generated) - set(expected))
        if missing:
            mismatches.append(f"Missing required arguments: {missing}.")
        if extra:
            mismatches.append(f"Unexpected arguments: {extra}.")
    for name in sorted(set(expected) & set(generated)):
        actual, wanted = generated[name], expected[name]
        if name in {"variables", "predictors"}:
            if not isinstance(actual, list):
                mismatches.append(f"Expected {name} to contain all 8 public domains.")
            elif len(actual) != len(PREDICTORS):
                mismatches.append(
                    f"Expected 8 public domains but generated {len(actual)}."
                )
            elif len(set(actual)) != len(actual) or set(actual) != set(PREDICTORS):
                mismatches.append(f"Expected exactly the eight approved public domains in {name}.")
            elif method == "individual_profile" and actual != wanted:
                mismatches.append("Individual profile domains are not in canonical order.")
        elif name == "controls":
            canonical = _canonicalize_controls(wanted)
            actual_controls = _canonicalize_controls(actual)
            if actual_controls != canonical:
                mismatches.append("Expected all four predefined Table 1 control specifications.")
        elif _same_number(actual, wanted):
            continue
        elif actual != wanted:
            if name == "filters":
                mismatches.append(f"Expected filters={wanted!r} but generated filters={actual!r}.")
            elif name == "method":
                mismatches.append(
                    f"Expected correlation method={wanted!r} but generated {actual!r}."
                )
            else:
                mismatches.append(f"Expected {name}={wanted!r} but generated {actual!r}.")
    return mismatches


def inspect_generated_code(
    code: str,
    *,
    user_request: str,
    requested_analysis: str,
    requested_filters: dict | None = None,
) -> GeneratedCodeVerificationResult:
    result = GeneratedCodeVerificationResult(cleaned_code=clean_generated_code(code))
    try:
        method, arguments, cleaned = _parse_structure(code)
        result.generated_method = method
        arguments = _normalize_trusted_defaults(method, arguments)
        result.generated_arguments = arguments
        result.cleaned_code = cleaned
        result.structure_validation_passed = True
    except Exception as exc:
        result.failure_stage = "format_validation"
        result.validation_error = sanitize_exception(exc)
        return result
    try:
        contract = build_request_contract(requested_analysis, requested_filters, user_request)
    except Exception as exc:
        result.failure_stage = "request_validation"
        result.validation_error = sanitize_exception(exc)
        return result
    result.expected_method = contract.method
    expected_arguments = _normalize_trusted_defaults(contract.method, contract.arguments)
    result.expected_arguments = expected_arguments
    if method != contract.method:
        result.request_mismatches.append(
            f"Expected method={contract.method!r} but generated {method!r}."
        )
    else:
        result.request_mismatches.extend(
            _compare_arguments(method, arguments, expected_arguments)
        )
    if result.request_mismatches:
        result.failure_stage = "request_validation"
        result.validation_error = " ".join(result.request_mismatches)
        return result
    result.request_match_passed = True
    return result


def _contains_sensitive_result_value(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).casefold()
            if key_text == "athlete_id" or ATHLETE_ID_PATTERN.fullmatch(str(item)):
                return True
            if _contains_sensitive_result_value(item):
                return True
    elif isinstance(value, list):
        return any(_contains_sensitive_result_value(item) for item in value)
    elif isinstance(value, str):
        return ATHLETE_ID_PATTERN.fullmatch(value) is not None
    return False


def _validate_result(method: str, output: Any, expected: dict) -> list[str]:
    errors = []
    if not isinstance(output, dict):
        return ["RestrictedAnalysisAPI result must be a dictionary."]
    if _contains_sensitive_result_value(output):
        errors.append("The result exposed an athlete identifier.")
    filters = expected.get("filters")
    if filters is not None and output.get("filters") != filters:
        errors.append(f"Returned filters did not match {filters!r}.")
    analysis_names = {
        "correlation": "Correlation analysis",
        "table1": "Table 1-style logistic regression",
        "table2": "Table 2-style linear regression",
        "figure1": "Figure 1-style group statistics",
        "variance_analysis": "Dynamic variance analysis",
    }
    if method in analysis_names and output.get("analysis") != analysis_names[method]:
        errors.append(f"Unexpected analysis result type for {method}.")
    if method == "correlation":
        if output.get("method") != "pearson":
            errors.append("Correlation result method was not pearson.")
        if len(output.get("rows") or output.get("table") or []) != 28:
            errors.append("Correlation result did not contain 28 pairwise rows.")
    elif method in {"table1", "table2"}:
        if "table" not in output and "rows" not in output:
            errors.append("Regression result did not contain table/rows.")
        if "model_stats" not in output:
            errors.append("Regression result did not contain model_stats.")
    elif method == "figure1":
        if "table" not in output:
            errors.append("Figure 1 result did not contain the expected table.")
        if not output.get("privacy_note") and "figure" not in output:
            errors.append("Figure 1 result did not contain a figure.")
    elif method == "variance_analysis":
        rows = output.get("table") or []
        if len(rows) != len(PREDICTORS):
            errors.append("Variance result did not contain one row per public domain.")
        if any(row.get("iterations") != 1000 for row in rows if isinstance(row, dict)):
            errors.append("Variance result iteration metadata was not 1000.")
    elif method == "figure2":
        if "figure 2" not in str(output.get("analysis") or output.get("summary") or "").casefold():
            errors.append("Output analysis type was not Figure 2.")
        expected_max = expected["max_athletes"]
        actual_max = output.get("max_athletes", expected_max)
        if actual_max != expected_max:
            errors.append("Returned max_athletes did not match the request.")
        count = output.get("profile_count")
        if expected_max is not None and isinstance(count, int) and count > expected_max:
            errors.append("Figure 2 profile_count exceeded the requested maximum.")
    elif method == "individual_profile":
        if output.get("analysis") != "Anonymous Athlete Profile":
            errors.append("Output was not the anonymous standardized profile.")
        if output.get("domain_count") != 8:
            errors.append("Individual profile did not contain eight domains.")
        if output.get("identifier_exposed") is not False:
            errors.append("Individual profile identifier_exposed was not false.")
        if output.get("raw_values_included") is not False:
            errors.append("Individual profile raw_values_included was not false.")
    return errors


def verify_and_execute_generated_code(
    code: str | None,
    *,
    user_request: str,
    requested_analysis: str,
    requested_filters: dict | None = None,
    subject_reference: str | None = None,
    close_figures_after_execution: bool = False,
) -> GeneratedCodeVerificationResult:
    if not code:
        return GeneratedCodeVerificationResult(
            failure_stage="format_validation",
            validation_error="No generated restricted code.",
        )
    result = inspect_generated_code(
        code,
        user_request=user_request,
        requested_analysis=requested_analysis,
        requested_filters=requested_filters,
    )
    if not result.request_match_passed:
        return result
    initial_figure_numbers: set[int] = set()
    pyplot = None
    if close_figures_after_execution:
        from sports import matplotlib_backend as _matplotlib_backend
        import matplotlib.pyplot as pyplot

        initial_figure_numbers = set(pyplot.get_fignums())
    try:
        api = RestrictedAnalysisAPI(subject_reference=subject_reference)
        output = getattr(api, result.generated_method)(**result.generated_arguments)
        result.local_execution_passed = True
        result.executed = True
        result.result = output
    except Exception as exc:
        result.failure_stage = "local_execution"
        result.validation_error = sanitize_exception(exc)
        return result
    finally:
        if pyplot is not None:
            for figure_number in set(pyplot.get_fignums()) - initial_figure_numbers:
                pyplot.close(figure_number)
    errors = _validate_result(result.generated_method, output, result.expected_arguments)
    if errors:
        result.failure_stage = "result_validation"
        result.validation_error = " ".join(errors)
        return result
    result.result_validation_passed = True
    result.fully_correct = all((
        result.structure_validation_passed,
        result.request_match_passed,
        result.local_execution_passed,
        result.result_validation_passed,
    ))
    result.allowed = result.fully_correct
    return result
