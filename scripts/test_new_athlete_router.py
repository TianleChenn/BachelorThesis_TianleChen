"""One-query local smoke test for the trained New Athlete Router."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from llm.athlete_strong_weak_router import predict_athlete_router

if __name__ == "__main__":
    result=predict_athlete_router(
        "Run the multiple linear regression using the eight athlete domains.",
        requested_analysis="table2", privacy_route="cloud", requires_code=True,
    )
    print(result["router_display_name"])
    print(f"Strong Model Probability: {result['strong_probability']:.6f}")
    print(f"Threshold: {result['threshold']:.6f}")
    print(f"Selected Model: {result['selected_model_label']}")
