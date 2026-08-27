import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

from llm.local_model_provider import LocalModelProvider


def _fresh_provider():
    LocalModelProvider._instance = None
    return LocalModelProvider()


def test_provider_reads_environment_configuration():
    with patch.dict(os.environ, {
        "LLM_LOCAL_PROVIDER": "transformers",
        "LLM_LOCAL_MODEL_ID": "test/local-model",
        "LLM_LOCAL_DEVICE": "auto",
        "LLM_LOCAL_MAX_NEW_TOKENS": "64",
    }):
        provider = _fresh_provider()
        assert provider.provider == "transformers"
        assert provider.model_id == "test/local-model"
        assert provider.max_new_tokens == 64


def test_openai_compatible_provider_uses_local_environment_only():
    env = {
        "LLM_LOCAL_PROVIDER": "openai_compatible",
        "LLM_LOCAL_MODEL": "Ministral-3-8B-Local",
        "LLM_LOCAL_BASE_URL": "http://127.0.0.1:8080/v1",
        "LLM_LOCAL_API_KEY": "none",
    }
    with patch.dict(os.environ, env, clear=False):
        provider = _fresh_provider()
    assert provider.provider == "openai_compatible"
    assert provider.model_id == "Ministral-3-8B-Local"
    assert provider.base_url == "http://127.0.0.1:8080/v1"


def test_openai_compatible_generation_calls_configured_local_endpoint():
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content="result = analysis.correlation(variables=['a', 'b'], filters={})"
    ))])
    env = {
        "LLM_LOCAL_PROVIDER": "openai_compatible",
        "LLM_LOCAL_MODEL": "Ministral-3-8B-Local",
        "LLM_LOCAL_BASE_URL": "http://127.0.0.1:8080/v1",
        "LLM_LOCAL_API_KEY": "none",
    }
    with patch.dict(os.environ, env, clear=False), patch(
        "llm.local_model_provider.call_local_codegen_model",
        return_value=SimpleNamespace(success=True, content=response.choices[0].message.content, error=None),
    ) as local_call:
        result = _fresh_provider().generate_restricted_code(messages=[{"role": "user", "content": "request"}])
    assert result["success"] is True
    assert local_call.call_args.kwargs["temperature"] == 0


def test_openai_compatible_provider_rejects_nonlocal_endpoint():
    with patch.dict(os.environ, {
        "LLM_LOCAL_PROVIDER": "openai_compatible",
        "LLM_LOCAL_BASE_URL": "https://api.mistral.ai/v1",
    }, clear=False):
        status = _fresh_provider().get_status(load_model=False)
    assert status.available is False
    assert "localhost" in status.reason


def test_provider_is_process_singleton():
    first = _fresh_provider()
    second = LocalModelProvider()
    assert first is second


def test_generation_failure_is_structured_and_sanitized():
    with patch.dict(os.environ, {"LLM_LOCAL_PROVIDER":"transformers"}, clear=False):
        provider = _fresh_provider()
        with patch.object(provider, "_load", side_effect=RuntimeError("api_key=secret")):
            result = provider.generate_restricted_code(messages=[{"role":"user","content":"safe request"}])
    assert result["success"] is False
    assert result["content"] is None
    assert result["provider"] == "transformers"
    assert "secret" not in result["error"]
    assert "DeltaGenerator" not in type(result).__name__


def test_outer_code_fence_cleaning_is_limited():
    code = "result = analysis.correlation(variables=['a', 'b'], filters={})"
    assert LocalModelProvider._clean_output(f"```python\n{code}\n```") == code
    surrounding = f"explanation\n```python\n{code}\n```"
    assert LocalModelProvider._clean_output(surrounding) == surrounding
