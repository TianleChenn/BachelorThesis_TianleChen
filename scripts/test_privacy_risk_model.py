"""Direct test of the same public Privacy Assessor entry point used by PRISM."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from llm.env import load_local_env
from llm.model_config import get_strong_model_name
from privacy.llm_privacy_assessor import assess_privacy_with_llm

def main():
    load_local_env();result=assess_privacy_with_llm("Explain what Table 2 means.",use_cache=False)
    print(f"Requested Model:\n{result.requested_model}\n")
    print("Provider:\nOpenAI\n")
    print(f"API Key Loaded:\n{bool(os.getenv('OPENAI_API_KEY'))}\n")
    print("===== VALIDATED ASSESSMENT JSON =====")
    print(json.dumps(result.to_dict(),indent=2));print()
    print(f"Assessment Success:\n{result.success}\n")
    print(f"Fallback Used:\n{result.fallback_used}\n")
    print(f"Exception Type:\n{result.error.split(':',1)[0] if result.error else 'None'}\n")
    print(f"Exception Message:\n{result.error or 'None'}\n")
    print(f"Actual Model:\n{result.actual_model or 'Unavailable'}")
    return 0 if result.success else 1
if __name__=="__main__": raise SystemExit(main())
