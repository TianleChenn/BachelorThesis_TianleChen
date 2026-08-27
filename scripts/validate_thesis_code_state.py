"""Read-only validation of the final Bachelor Thesis code and saved results."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FAILURES: list[str] = []
STATUS: dict[str, bool] = {}


def require_files(group: str, paths: list[str]) -> None:
    missing = [path for path in paths if not (ROOT / path).is_file()]
    if missing:
        STATUS[group] = False
        FAILURES.extend(f"{group}: missing {path}" for path in missing)
    else:
        STATUS.setdefault(group, True)


def load_json(group: str, path: str) -> dict:
    try:
        data = json.loads((ROOT / path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        STATUS[group] = False
        FAILURES.append(f"{group}: cannot read {path}: {exc}")
        return {}
    if not isinstance(data, dict):
        STATUS[group] = False
        FAILURES.append(f"{group}: {path} must contain a JSON object")
        return {}
    return data


def nested(data: dict, *keys: str):
    value = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def require_value(group: str, description: str, actual, expected, *, tolerance: float = 0.0) -> None:
    if tolerance:
        valid = (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance)
        )
    else:
        valid = actual == expected
    if not valid:
        STATUS[group] = False
        FAILURES.append(f"{group}: {description} expected {expected!r}, found {actual!r}")


GROUP_FILES = {
    "Runtime implementation": [
        "main.py",
        "frontend.py",
        "privacy/prism_router.py",
        "privacy/cloud_local_router.py",
        "privacy/llm_privacy_assessor.py",
        "privacy/llm_soft_gating_model.py",
        "llm/athlete_cloud_local_router.py",
        "llm/code_generation_prompt.py",
        "llm/code_generator.py",
        "llm/generated_code_verifier.py",
        "sports/restricted_analysis_api.py",
    ],
    "Privacy evaluation": [
        "scripts/evaluate_privacy_cloud_models.py",
        "scripts/evaluate_privacy_methods_frontend60.py",
        "evaluation/run_privacy_benchmark.py",
        "evaluation/frontend_realistic_benchmark_60.json",
        "evaluation/privacy_controlled_benchmark.json",
        "artifacts/prism_soft_gater_4d_llm_hard.pt",
        "artifacts/privacy_cloud_model_evaluation.json",
        "artifacts/thesis_evaluation/privacy_cloud_model_evaluation.json",
        "artifacts/privacy_methods_frontend60_comparison.json",
        "artifacts/privacy_benchmark_metrics.json",
        "artifacts/controlled_per_level_accuracy.csv",
        "artifacts/controlled_privacy_feature_summary.json",
    ],
    "Cost-aware router evaluation": [
        "scripts/collect_athlete_cloud_local_preferences.py",
        "scripts/train_athlete_cloud_local_router.py",
        "scripts/evaluate_athlete_cloud_local_router.py",
        "evaluation/athlete_cloud_local_training_prompts_100.json",
        "evaluation/athlete_cloud_local_independent_40.json",
        "artifacts/athlete_cloud_local_router.joblib",
        "artifacts/athlete_cloud_local_router.json",
        "artifacts/athlete_cloud_local_router_evaluation.json",
    ],
    "Cloud model evaluation": [
        "scripts/evaluate_cloud_codegen_models.py",
        "artifacts/cloud_codegen_model_evaluation.json",
    ],
    "Local model evaluation": [
        "scripts/evaluate_local_codegen_models.py",
        "artifacts/local_codegen_model_evaluation.json",
    ],
    "Privacy prompt evaluation": [
        "privacy/prompt_ablation.py",
        "scripts/evaluate_privacy_prompt_ablation.py",
        "evaluation/results/prompt_ablation/privacy_prompt_ablation_summary.json",
    ],
    "Code generation prompt evaluation": [
        "llm/code_generation_prompt_design_v2.py",
        "scripts/evaluate_codegen_prompt_design_v2.py",
        "scripts/evaluate_local_codegen_prompt_design_v2.py",
        "evaluation/results/prompt_design_v2/codegen_prompt_design_v2_summary.json",
        "evaluation/results/local_prompt_design_v2/local_codegen_prompt_design_v2_summary.json",
    ],
    "Numerical perturbation evaluation": [
        "privacy/numerical_perturbation.py",
        "evaluation/perturbation_benchmark_common.py",
        "evaluation/run_perturbation_noise_benchmark.py",
        "evaluation/run_perturbation_sample_size_benchmark.py",
        "artifacts/perturbation_noise/perturbation_noise_summary.csv",
        "artifacts/perturbation_sample_size/perturbation_sample_size_summary.csv",
    ],
}


LEGACY_FILES = [
    "llm/athlete_strong_weak_router.py",
    "privacy/routellm_router.py",
    "scripts/collect_athlete_router_preferences_v2.py",
    "scripts/generate_athlete_router_training_prompts_v2.py",
    "scripts/train_athlete_strong_weak_router.py",
    "scripts/evaluate_athlete_strong_weak_router.py",
    "llm/code_generation_prompt_ablation.py",
    "scripts/evaluate_codegen_prompt_ablation.py",
]


LEGACY_PATTERNS = [
    "athlete_" + "strong_weak_router",
    "route" + "llm_router",
    "athlete_router_" + "preferences_v2",
    "athlete_router_training_" + "prompts_v2",
    "evaluate_athlete_" + "strong_weak_router",
    "train_athlete_" + "strong_weak_router",
    "collect_athlete_router_" + "preferences_v2",
    "Official Route" + "LLM MF",
    "Interface " + "Guided",
    "code_generation_prompt_" + "ablation",
    "evaluate_codegen_prompt_" + "ablation",
]


def validate_no_legacy_references() -> None:
    group = "No legacy Strong/Weak implementation references"
    STATUS[group] = True
    for path in LEGACY_FILES:
        if (ROOT / path).exists():
            STATUS[group] = False
            FAILURES.append(f"{group}: legacy file still exists: {path}")
    for path in ROOT.rglob("*"):
        if path == Path(__file__).resolve() or not path.is_file() or path.suffix not in {".py", ".ps1"}:
            continue
        if "__pycache__" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in LEGACY_PATTERNS:
            if pattern in text:
                STATUS[group] = False
                relative = path.relative_to(ROOT).as_posix()
                FAILURES.append(f"{group}: {pattern!r} remains in {relative}")


def main() -> int:
    for group, paths in GROUP_FILES.items():
        require_files(group, paths)

    router = load_json("Cost-aware router evaluation", "artifacts/athlete_cloud_local_router.json")
    require_value("Cost-aware router evaluation", "training_samples", router.get("training_samples"), 50)
    require_value("Cost-aware router evaluation", "cloud_samples", router.get("cloud_samples"), 24)
    require_value("Cost-aware router evaluation", "local_samples", router.get("local_samples"), 26)
    require_value("Cost-aware router evaluation", "threshold", router.get("threshold"), 0.603939, tolerance=1e-6)

    router_eval = load_json(
        "Cost-aware router evaluation", "artifacts/athlete_cloud_local_router_evaluation.json"
    )
    require_value(
        "Cost-aware router evaluation", "valid_ground_truth_samples",
        router_eval.get("valid_ground_truth_samples"), 25,
    )
    require_value("Cost-aware router evaluation", "invalid_samples", router_eval.get("invalid_samples"), 15)
    require_value("Cost-aware router evaluation", "routing_accuracy", router_eval.get("routing_accuracy"), 0.96)
    require_value("Cost-aware router evaluation", "cloud_usage_rate", router_eval.get("cloud_usage_rate"), 0.40)

    cloud = load_json("Cloud model evaluation", "artifacts/cloud_codegen_model_evaluation.json")
    require_value("Cloud model evaluation", "evaluation_samples_per_model", cloud.get("evaluation_samples_per_model"), 40)
    local = load_json("Local model evaluation", "artifacts/local_codegen_model_evaluation.json")
    require_value("Local model evaluation", "evaluation_samples_per_model", local.get("evaluation_samples_per_model"), 40)

    cloud_prompt = load_json(
        "Code generation prompt evaluation",
        "evaluation/results/prompt_design_v2/codegen_prompt_design_v2_summary.json",
    )
    local_prompt = load_json(
        "Code generation prompt evaluation",
        "evaluation/results/local_prompt_design_v2/local_codegen_prompt_design_v2_summary.json",
    )
    for data, prefix, expected in (
        (cloud_prompt, "cloud", {"basic_interface": 0.0, "defined": 0.25, "full": 0.675}),
        (local_prompt, "local", {"basic_interface": 0.0, "defined": 0.50, "full": 0.55}),
    ):
        for version, value in expected.items():
            require_value(
                "Code generation prompt evaluation",
                f"{prefix} {version} fully_correct",
                nested(data, "prompts", version, "fully_correct"),
                value,
            )

    privacy_prompt = load_json(
        "Privacy prompt evaluation",
        "evaluation/results/prompt_ablation/privacy_prompt_ablation_summary.json",
    )
    expected_privacy = {
        ("controlled", "minimal"): 0.46875,
        ("controlled", "defined"): 0.71875,
        ("controlled", "full"): 0.90625,
        ("independent", "minimal"): 0.25,
        ("independent", "defined"): 1 / 3,
        ("independent", "full"): 5 / 12,
    }
    for (benchmark, version), value in expected_privacy.items():
        require_value(
            "Privacy prompt evaluation",
            f"{benchmark} {version} exact_route_accuracy",
            nested(privacy_prompt, "benchmarks", benchmark, version, "exact_route_accuracy"),
            value,
            tolerance=1e-9,
        )

    validate_no_legacy_references()

    for group in [*GROUP_FILES, "No legacy Strong/Weak implementation references"]:
        print(f"[{'OK' if STATUS.get(group, False) else 'FAIL'}] {group}")
    if FAILURES:
        print("\nValidation failures:")
        for failure in FAILURES:
            print(f"- {failure}")
        print("\nTHESIS CODE STATE: FAILED")
        return 1
    print("\nTHESIS CODE STATE: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
