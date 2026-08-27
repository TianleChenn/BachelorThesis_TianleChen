from __future__ import annotations

import os
import re
import time
from urllib.parse import urlparse
from dataclasses import dataclass

from .env import load_local_env


@dataclass
class ModelCallResult:
    content: str | None
    requested_model: str
    actual_model: str | None
    provider: str
    success: bool
    unavailable: bool
    fallback_used: bool
    error: str | None
    endpoint: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    latency_seconds: float | None = None
    finish_reason: str | None = None
    content_state: str | None = None


@dataclass(frozen=True)
class CloudCodegenEvaluationModel:
    key: str
    display_name: str
    provider: str
    model_envs: tuple[str, ...]
    default_model: str
    api_key_env: str
    base_url_env: str | None
    default_base_url: str | None


_CLOUD_CODEGEN_EVALUATION_MODELS = (
    CloudCodegenEvaluationModel(
        "gpt4_1", "GPT-4.1", "openai",
        ("EVAL_OPENAI_MODEL", "LLM_STRONG_MODEL", "OPENAI_STRONG_MODEL"), "gpt-4.1",
        "OPENAI_API_KEY", None, None,
    ),
    CloudCodegenEvaluationModel(
        "gemini", "Gemini 3.5 Flash", "gemini",
        ("EVAL_GEMINI_MODEL", "LLM_GEMINI_MODEL"), "gemini-3.5-flash",
        "LLM_GEMINI_API_KEY", "LLM_GEMINI_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
    ),
    CloudCodegenEvaluationModel(
        "claude", "Claude Sonnet 5", "anthropic",
        ("EVAL_CLAUDE_MODEL", "LLM_CLAUDE_MODEL"), "claude-sonnet-5",
        "LLM_CLAUDE_API_KEY", "LLM_CLAUDE_BASE_URL", "https://api.anthropic.com/v1/",
    ),
)


def _first_env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return default


def is_reasoning_model(model_name: str) -> bool:
    return str(model_name or "").lower().startswith(("o1", "o3", "o4"))


def _responses_input(messages) -> str:
    if isinstance(messages, str):
        return messages
    parts = []
    for message in messages or []:
        if isinstance(message, dict):
            role = str(message.get("role") or "user").upper()
            content = message.get("content")
            if isinstance(content, str):
                parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


def _sanitized_error(exc: Exception, model: str, endpoint: str) -> str:
    status = getattr(exc, "status_code", None)
    code = getattr(exc, "code", None)
    body = getattr(exc, "body", None)
    if code is None and isinstance(body, dict):
        error_body = body.get("error") if isinstance(body.get("error"), dict) else body
        code = error_body.get("code") if isinstance(error_body, dict) else None
    message = str(getattr(exc, "message", None) or exc)
    message = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED_API_KEY]", message)
    message = re.sub(r"(?i)(api[_ -]?key[=: ]+)[^\s,;]+", r"\1[REDACTED]", message)
    return (
        f"{type(exc).__name__}: model={model}; endpoint={endpoint}; "
        f"http_status={status if status is not None else 'unknown'}; "
        f"provider_code={code if code is not None else 'unknown'}; message={message[:800]}"
    )


def _usage_values(response) -> tuple[int | None, int | None, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None, None
    def read(*names):
        for name in names:
            value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        return None
    input_tokens = read("prompt_tokens", "input_tokens")
    output_tokens = read("completion_tokens", "output_tokens")
    total_tokens = read("total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return input_tokens, output_tokens, total_tokens


def _is_provider_unavailable(exc: Exception) -> bool:
    """Identify service availability failures without mislabeling 4xx requests."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status >= 500
    unavailable_names = {
        "APIConnectionError", "APITimeoutError", "ConnectError",
        "ConnectTimeout", "ConnectionError", "ReadTimeout", "TimeoutError",
    }
    return any(cls.__name__ in unavailable_names for cls in type(exc).__mro__)


def _field(value, name: str):
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


def _visible_text_content(raw_content) -> str | None:
    """Extract only documented visible text, never reasoning/thought content."""
    if raw_content is None or isinstance(raw_content, str):
        return raw_content
    parts = raw_content if isinstance(raw_content, (list, tuple)) else (raw_content,)
    visible_parts: list[str] = []
    for part in parts:
        part_type = str(_field(part, "type") or "").strip().casefold()
        if part_type in {"reasoning", "reasoning_content", "thinking", "thought"}:
            continue
        if part_type not in {"", "text", "output_text"}:
            continue
        text = _field(part, "text")
        if isinstance(text, str):
            visible_parts.append(text)
    return "".join(visible_parts) if visible_parts else None


def _content_state(raw_content) -> str:
    if raw_content is None:
        return "None"
    if isinstance(raw_content, str):
        if raw_content == "":
            return "empty string"
        if not raw_content.strip():
            return "whitespace-only string"
        return "non-empty string"
    visible = _visible_text_content(raw_content)
    return "text content parts" if visible and visible.strip() else "content parts without visible text"


def _empty_content_error(
    *,
    requested_model: str,
    actual_model: str,
    endpoint: str,
    finish_reason: str | None,
    input_tokens: int | None,
    output_tokens: int | None,
    total_tokens: int | None,
    content_state: str,
) -> str:
    return (
        "Provider returned no visible response content: "
        f"requested_model={requested_model}; actual_model={actual_model}; "
        f"endpoint={endpoint}; finish_reason={finish_reason}; "
        f"input_tokens={input_tokens}; output_tokens={output_tokens}; "
        f"total_tokens={total_tokens}; content_state={content_state}."
    )


def _call(messages, model, provider, api_key, base_url, temperature, max_tokens, *,
          enable_response_format=True):
    endpoint = "responses" if is_reasoning_model(model) and not base_url else "chat.completions"
    started = time.perf_counter()
    try:
        from openai import OpenAI
        if not api_key:
            raise RuntimeError("provider credentials unavailable")
        client = OpenAI(api_key=api_key, base_url=base_url or None)
        finish_reason = None
        if endpoint == "responses":
            response = client.responses.create(model=model, input=_responses_input(messages))
            raw_content = response.output_text
            content = _visible_text_content(raw_content)
            incomplete = getattr(response, "incomplete_details", None)
            finish_reason = getattr(incomplete, "reason", None) or getattr(response, "status", None)
        else:
            kwargs = {"model": model, "messages": messages}
            if temperature is not None:
                kwargs["temperature"] = temperature
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            requests_json = any(
                isinstance(item, dict) and "json" in str(item.get("content", "")).lower()
                for item in (messages if isinstance(messages, list) else [])
            )
            if enable_response_format and requests_json:
                kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            raw_content = getattr(choice.message, "content", None)
            content = _visible_text_content(raw_content)
            finish_reason = getattr(choice, "finish_reason", None)
        actual = getattr(response, "model", None) or model
        input_tokens, output_tokens, total_tokens = _usage_values(response)
        content_state = _content_state(raw_content)
        has_visible_content = bool(content and content.strip())
        error = None if has_visible_content else _empty_content_error(
            requested_model=model,
            actual_model=actual,
            endpoint=endpoint,
            finish_reason=finish_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            content_state=content_state,
        )
        return ModelCallResult(
            content, model, actual, provider, has_visible_content, False, False, error,
            endpoint, input_tokens, output_tokens, total_tokens,
            time.perf_counter() - started, finish_reason, content_state,
        )
    except Exception as exc:
        return ModelCallResult(
            None, model, None, provider, False, _is_provider_unavailable(exc), False,
            _sanitized_error(exc, model, endpoint), endpoint,
            latency_seconds=time.perf_counter() - started,
        )


GEMINI_TOKEN_EXHAUSTION_RETRY_MAX_TOKENS = 16384
_TOKEN_EXHAUSTION_FINISH_REASONS = {"LENGTH", "MAX_TOKENS", "MAXIMUM_TOKENS"}


def _finish_reason_indicates_token_exhaustion(finish_reason: object) -> bool:
    normalized = re.sub(r"[\s-]+", "_", str(finish_reason or "").strip()).upper()
    return normalized in _TOKEN_EXHAUSTION_FINISH_REASONS


def get_cloud_codegen_evaluation_models() -> tuple[CloudCodegenEvaluationModel, ...]:
    return _CLOUD_CODEGEN_EVALUATION_MODELS


def get_cloud_codegen_evaluation_runtime(model_key: str) -> dict:
    load_local_env()
    registry = {model.key: model for model in _CLOUD_CODEGEN_EVALUATION_MODELS}
    if model_key not in registry:
        raise ValueError(f"Unknown cloud code-generation evaluation model: {model_key}")
    config = registry[model_key]
    api_key = _first_env(config.api_key_env)
    return {
        "key": config.key,
        "display_name": config.display_name,
        "provider": config.provider,
        "model": _first_env(*config.model_envs, default=config.default_model),
        "base_url": _first_env(config.base_url_env, default=config.default_base_url)
        if config.base_url_env else config.default_base_url,
        "api_key_loaded": bool(api_key),
        "api_key_env": config.api_key_env,
        "api_key": api_key,
    }


def call_cloud_codegen_evaluation_model(
    model_key: str,
    messages,
    *,
    max_tokens: int = 2048,
) -> ModelCallResult:
    runtime = get_cloud_codegen_evaluation_runtime(model_key)
    return _call(
        messages,
        runtime["model"],
        runtime["provider"],
        runtime["api_key"],
        runtime["base_url"],
        None,
        max_tokens,
    )


def call_cloud_privacy_evaluation_model(
    model_key: str,
    messages,
    *,
    max_tokens: int = 1500,
) -> ModelCallResult:
    """Call a registry model for evaluation with production-like deterministic settings."""
    runtime = get_cloud_codegen_evaluation_runtime(model_key)
    return _call(
        messages,
        runtime["model"],
        runtime["provider"],
        runtime["api_key"],
        runtime["base_url"],
        None,
        max_tokens,
        enable_response_format=False,
    )


def call_gemini_cloud_model(messages, temperature=None, max_tokens=4096):
    """Call Gemini code generation with provider-default sampling settings."""
    load_local_env()
    model = os.getenv("LLM_GEMINI_MODEL", "gemini-3.5-flash").strip()
    provider = os.getenv("LLM_GEMINI_PROVIDER", "openai_compatible").strip()
    api_key = os.getenv("LLM_GEMINI_API_KEY")
    base_url = os.getenv(
        "LLM_GEMINI_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
    ).strip()
    result = _call(
        messages, model, provider, api_key, base_url, None, max_tokens,
    )
    retry_allowed = (
        not result.success
        and result.actual_model is not None
        and not (result.content and result.content.strip())
        and _finish_reason_indicates_token_exhaustion(result.finish_reason)
        and isinstance(max_tokens, int)
        and max_tokens < GEMINI_TOKEN_EXHAUSTION_RETRY_MAX_TOKENS
    )
    if not retry_allowed:
        return result
    return _call(
        messages,
        model,
        provider,
        api_key,
        base_url,
        None,
        GEMINI_TOKEN_EXHAUSTION_RETRY_MAX_TOKENS,
    )


def get_local_codegen_runtime() -> dict:
    """Return validated local code-generation configuration without exposing credentials."""
    load_local_env()
    base_url = os.getenv("LLM_LOCAL_BASE_URL", "http://127.0.0.1:8080/v1").strip()
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise RuntimeError("LLM_LOCAL_BASE_URL must use localhost, 127.0.0.1, or ::1.")
    return {
        "provider": os.getenv("LLM_LOCAL_PROVIDER", "openai_compatible").strip(),
        "model": os.getenv("LLM_LOCAL_MODEL", "Ministral-3-8B-Local").strip(),
        "base_url": base_url,
        "api_key": os.getenv("LLM_LOCAL_API_KEY", "none"),
    }


def call_local_codegen_model(messages, temperature=0.0, max_tokens=4096):
    """Call only the localhost OpenAI-compatible Local code generator."""
    try:
        runtime = get_local_codegen_runtime()
    except Exception as exc:
        return ModelCallResult(
            None, os.getenv("LLM_LOCAL_MODEL", "Ministral-3-8B-Local"), None,
            "openai_compatible", False, True, False, str(exc), "chat.completions",
        )
    if runtime["provider"] != "openai_compatible":
        return ModelCallResult(
            None, runtime["model"], None, runtime["provider"], False, True, False,
            "The active Local code generator requires LLM_LOCAL_PROVIDER=openai_compatible.",
            "chat.completions",
        )
    return _call(
        messages, runtime["model"], runtime["provider"], runtime["api_key"],
        runtime["base_url"], temperature, max_tokens,
    )


def call_privacy_risk_model(messages, *, temperature=0.0, max_tokens=500):
    """Call the independent OpenAI privacy-risk assessor."""
    from llm.model_config import get_strong_model_name
    load_local_env()
    model = get_strong_model_name()
    return _call(messages, model, "openai_privacy_assessor", os.getenv("OPENAI_API_KEY"),
                 None, 0.0, max_tokens)
