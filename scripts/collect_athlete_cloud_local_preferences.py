"""Collect objective Gemini/Local preferences for Cloud/Local router training."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Configure headless plotting before importing the restricted-analysis stack,
# which imports pyplot through its figure helpers.
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)

from llm.code_generation_prompt import CODE_GENERATION_PROMPT_VERSION
from llm.env import load_local_env
from scripts.cloud_local_evaluation_common import evaluate_cloud_and_local

DATASET = ROOT / "evaluation" / "athlete_cloud_local_training_prompts_100.json"
RAW_OUTPUT = ROOT / "artifacts" / "athlete_cloud_local_preferences_raw.json"
VALID_OUTPUT = ROOT / "artifacts" / "athlete_cloud_local_preferences_valid.json"


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    json.loads(temporary.read_text(encoding="utf-8"))
    temporary.replace(path)


def _flatten(sample: dict, evaluated: dict) -> dict:
    cloud, local = evaluated["cloud"], evaluated["local"]
    return {
        **sample,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "prompt_version": CODE_GENERATION_PROMPT_VERSION,
        **{f"cloud_{key}": value for key, value in cloud.items()},
        **{f"local_{key}": value for key, value in local.items() if key not in {"input_tokens", "output_tokens"}},
        "preference": evaluated["preference"],
    }


def _summary(samples: list[dict], rows: list[dict]) -> dict:
    counts = Counter(row.get("preference", "invalid") for row in rows)
    per_task = defaultdict(Counter)
    for row in rows:
        per_task[row["analysis_type"]][row.get("preference", "invalid")] += 1
    return {
        "candidate_requests": len(samples), "processed_requests": len(rows),
        "valid_preferences": counts["cloud"] + counts["local"],
        "cloud_required": counts["cloud"], "local_sufficient": counts["local"],
        "invalid": counts["invalid"],
        "distribution_by_analysis_type": {key: dict(value) for key, value in sorted(per_task.items())},
    }


def collect(*, fresh: bool = False, resume: bool = False, limit: int | None = None) -> dict:
    load_local_env()
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    samples = list(dataset.get("samples") or [])
    if len(samples) != 100 or dataset.get("prompt_version") != CODE_GENERATION_PROMPT_VERSION:
        raise RuntimeError("Training dataset must contain exactly 100 samples using the active prompt version.")
    existing = {}
    if resume and not fresh and RAW_OUTPUT.exists():
        existing = {row["id"]: row for row in json.loads(RAW_OUTPUT.read_text(encoding="utf-8")).get("results", [])}
    pending = [sample for sample in samples if sample["id"] not in existing]
    if limit is not None:
        pending = pending[:limit]
    for sample in pending:
        existing[sample["id"]] = _flatten(sample, evaluate_cloud_and_local(sample))
        ordered = [existing[s["id"]] for s in samples if s["id"] in existing]
        _write(RAW_OUTPUT, {"dataset": str(DATASET), "results": ordered, "summary": _summary(samples, ordered)})
    rows = [existing[s["id"]] for s in samples if s["id"] in existing]
    valid = [row for row in rows if row["preference"] in {"cloud", "local"}]
    summary = _summary(samples, rows)
    _write(VALID_OUTPUT, {"dataset": str(DATASET), "results": valid, "summary": summary})
    print(f"Candidate requests: {summary['candidate_requests']}")
    print(f"Valid preferences: {summary['valid_preferences']}")
    print(f"Cloud required: {summary['cloud_required']}")
    print(f"Local sufficient: {summary['local_sufficient']}")
    print(f"Invalid: {summary['invalid']}")
    for task, counts in summary["distribution_by_analysis_type"].items():
        print(f"{task}: Cloud={counts.get('cloud', 0)}, Local={counts.get('local', 0)}, Invalid={counts.get('invalid', 0)}")
    if len(rows) == len(samples) and (summary["cloud_required"] < 5 or summary["local_sufficient"] < 5):
        raise RuntimeError("Too few valid Cloud or Local samples; add more diverse candidate requests without relabeling.")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    collect(fresh=args.fresh, resume=args.resume, limit=args.limit)


if __name__ == "__main__":
    main()
