from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from typing import Any

from privacy.athlete_id import ATHLETE_ID_PATTERN
from llm.env import load_local_env
from llm.model_clients import call_local_codegen_model


@dataclass
class LocalModelStatus:
    available: bool
    provider: str
    model_id: str
    device: str | None
    reason: str | None = None

    @property
    def error(self) -> str | None:
        return self.reason


class LocalModelProvider:
    """Process-safe local provider supporting llama.cpp and Transformers."""

    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        load_local_env()
        self.provider = os.getenv("LLM_LOCAL_PROVIDER", "transformers").strip().lower()
        if self.provider == "openai_compatible":
            self.model_id = os.getenv("LLM_LOCAL_MODEL", "").strip() or "Ministral-3-8B-Local"
        else:
            self.model_id = (
                os.getenv("LLM_LOCAL_MODEL_ID", "").strip()
                or os.getenv("LLM_LOCAL_MODEL", "").strip()
                or "Ministral-3-8B-Local"
            )
        self.base_url = os.getenv("LLM_LOCAL_BASE_URL", "http://127.0.0.1:8080/v1").strip()
        self.api_key = os.getenv("LLM_LOCAL_API_KEY", "none")
        self.device_setting = os.getenv("LLM_LOCAL_DEVICE", "auto").lower()
        self.max_new_tokens = int(os.getenv("LLM_LOCAL_MAX_NEW_TOKENS", "256"))
        self._tokenizer = None
        self._model = None
        self._device = None
        self._load_error = None
        self._load_lock = threading.Lock()
        self._generation_lock = threading.Lock()
        self._initialized = True

    @staticmethod
    def _sanitize_error(exc: Exception) -> str:
        text = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", str(exc))
        text = re.sub(r"(?i)(api[_ -]?key\s*[=:]\s*)\S+", r"\1[REDACTED]", text)
        return f"{type(exc).__name__}: {text[:500]}"

    def _load(self):
        if self._model is not None and self._tokenizer is not None:
            return
        with self._load_lock:
            if self._model is not None and self._tokenizer is not None:
                return
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                use_cuda = self.device_setting == "cuda" or (
                    self.device_setting == "auto" and torch.cuda.is_available()
                )
                self._device = "cuda" if use_cuda else "cpu"
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self._model = AutoModelForCausalLM.from_pretrained(self.model_id)
                self._model.to(self._device)
                self._model.eval()
                self._load_error = None
            except Exception as exc:
                self._load_error = self._sanitize_error(exc)
                self._tokenizer = None
                self._model = None
                raise RuntimeError(self._load_error) from exc

    def get_status(self, load_model: bool = False) -> LocalModelStatus:
        if self.provider == "openai_compatible":
            try:
                from urllib.parse import urlparse
                parsed = urlparse(self.base_url)
                if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
                    return LocalModelStatus(False, self.provider, self.model_id, None, "Local model endpoint must use localhost.")
                if load_model:
                    from openai import OpenAI
                    OpenAI(api_key=self.api_key or "none", base_url=self.base_url).models.list()
                return LocalModelStatus(True, self.provider, self.model_id, "local", None)
            except Exception as exc:
                return LocalModelStatus(False, self.provider, self.model_id, "local", self._sanitize_error(exc))
        try:
            import torch
            import transformers  # noqa: F401
        except ImportError as exc:
            return LocalModelStatus(False, self.provider, self.model_id, None, self._sanitize_error(exc))
        device = "cuda" if self.device_setting == "cuda" or (
            self.device_setting == "auto" and torch.cuda.is_available()
        ) else "cpu"
        if self.provider != "transformers":
            return LocalModelStatus(False, self.provider, self.model_id, device, "Unsupported local provider.")
        if not self.model_id.strip():
            return LocalModelStatus(False, self.provider, self.model_id, device, "Local model ID is empty.")
        if load_model:
            try:
                self._load()
            except Exception:
                return LocalModelStatus(False, self.provider, self.model_id, self._device or device, self._load_error)
            if self._tokenizer is None or self._model is None or not callable(getattr(self._model, "generate", None)):
                return LocalModelStatus(False, self.provider, self.model_id, self._device or device, "Local generation pipeline is unavailable.")
        return LocalModelStatus(True, self.provider, self.model_id, self._device or device, None)

    @staticmethod
    def _clean_output(text: str) -> str:
        text = str(text or "").strip()
        match = re.fullmatch(r"```(?:python)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        return match.group(1).strip() if match else text

    @staticmethod
    def clean_restricted_model_output(content: str) -> str:
        text = str(content or "").strip()
        if "```python" in text:
            text = text.split("```python", 1)[1].split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0]
        lines = [line for line in text.strip().splitlines() if not line.strip().startswith(("Here is", "Explanation:", "The code"))]
        return "\n".join(lines).strip()

    def generate_restricted_code(self, *, messages: list[dict[str, str]]) -> dict[str, Any]:
        try:
            if self.provider == "openai_compatible":
                status = self.get_status(load_model=False)
                if not status.available:
                    raise RuntimeError(status.reason or "Local model configuration is invalid.")
                safe_messages = [
                    {"role": str(message.get("role") or "user"),
                     "content": ATHLETE_ID_PATTERN.sub("CURRENT_SUBJECT", str(message.get("content") or ""))}
                    for message in messages
                ]
                response = call_local_codegen_model(safe_messages, temperature=0, max_tokens=self.max_new_tokens)
                if not response.success:
                    raise RuntimeError(response.error or "Local model is unavailable.")
                raw_content = str(response.content or "").strip()
                content = self.clean_restricted_model_output(raw_content)
                if not content:
                    return {"success": False, "content": None, "provider": self.provider,
                            "model": self.model_id, "device": "local", "error": "Local model returned empty output.",
                            "error_code": "empty_generation"}
                return {"success": True, "content": content, "provider": self.provider,
                        "model": self.model_id, "device": "local", "error": None,
                        "raw_content": raw_content, "error_code": None}
            self._load()
            import torch

            safe_messages = [
                {
                    "role": str(message.get("role") or "user"),
                    "content": ATHLETE_ID_PATTERN.sub(
                        "CURRENT_SUBJECT", str(message.get("content") or "")
                    ),
                }
                for message in messages
            ]
            if getattr(self._tokenizer, "chat_template", None):
                rendered = self._tokenizer.apply_chat_template(safe_messages, tokenize=False, add_generation_prompt=True)
            else:
                rendered = "\n\n".join(
                    f"{message['role'].upper()}: {message['content']}"
                    for message in safe_messages
                )
            inputs = self._tokenizer(rendered, return_tensors="pt")
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            input_length = inputs["input_ids"].shape[-1]
            with self._generation_lock, torch.inference_mode():
                output = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                    do_sample=False,num_beams=1,repetition_penalty=1.05,
                    pad_token_id=self._tokenizer.eos_token_id)
            generated = output[0][input_length:]
            raw_content=self._tokenizer.decode(generated, skip_special_tokens=True).strip()
            content = self.clean_restricted_model_output(raw_content)
            if not content:
                return {"success": False, "content": None, "provider": self.provider,
                    "model": self.model_id, "device": self._device,
                    "error": "Local model returned empty output.", "error_code": "empty_generation"}
            return {"success": True, "content": content, "provider": self.provider,
                "model": self.model_id, "device": self._device, "error": None,"raw_content":raw_content,"error_code":None}
        except Exception as exc:
            return {"success": False, "content": None, "provider": self.provider,
                "model": self.model_id, "device": self._device,
                "error": self._sanitize_error(exc), "error_code": "model_unavailable"}


def get_local_model_provider() -> LocalModelProvider:
    return LocalModelProvider()


def get_local_model_status(load_model: bool = True) -> LocalModelStatus:
    return get_local_model_provider().get_status(load_model=load_model)
