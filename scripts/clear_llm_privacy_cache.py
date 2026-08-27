"""Clear only the versioned LLM privacy-assessment cache."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
path=ROOT/"artifacts/llm_privacy_risk_cache.json"
if path.exists(): path.unlink();print(f"Removed {path}")
else: print("LLM privacy cache is already empty")
