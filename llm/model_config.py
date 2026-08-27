"""Canonical model identifiers for privacy assessment and local code generation."""

import os

LOCAL_EDGE_GENERATOR_MODEL = "Ministral-3-8B-Local"

def get_strong_model_name() -> str:
    """Return the shared configured Strong/Privacy-Assessor model name."""
    from llm.env import load_local_env
    load_local_env()
    return os.getenv("LLM_STRONG_MODEL", "").strip() or "gpt-4.1"
