"""
LLM client: unified wrapper for model API calls.

Supports multi-provider, multi-model evaluation.
Configuration via config.yaml + environment variables.
"""

from __future__ import annotations
import os
import json
import time
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    yaml = None


def _find_config_root() -> Path:
    p = Path(__file__).resolve().parent
    while p != p.parent:
        if (p / "config.yaml").exists():
            return p
        p = p.parent
    return Path.cwd()


ROOT = _find_config_root()


def load_config() -> dict:
    """Load optional config.yaml. Credentials are supplied by the environment."""
    config_path = ROOT / "config.yaml"
    if config_path.exists() and yaml:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


class LLMClient:
    """
    Unified LLM client wrapping OpenAI-compatible APIs.
    Handles parameter adaptation for different model families.
    """

    def __init__(self, model: Optional[str] = None, provider_name: Optional[str] = None):
        self._cfg = load_config()
        self._model = model or self._cfg.get("default_model") or os.environ.get("LLM_MODEL", "")
        self._provider_name = provider_name or self._provider_for_model(self._model)
        self._provider = self._cfg.get("providers", {}).get(self._provider_name, {})

        if not self._provider:
            self._provider = {
                "base_url": os.environ.get("LLM_BASE_URL", ""),
                "api_key_env": "OPENAI_API_KEY",
                "model": self._model,
            }

    def _provider_for_model(self, model: str) -> str:
        for name, provider in self._cfg.get("providers", {}).items():
            if model in provider.get("models", []):
                return name
        return self._cfg.get("default_provider", "")

    @property
    def model(self) -> str:
        return self._model

    @property
    def api_key(self) -> str:
        env_key = self._provider.get("api_key_env", "LLM_API_KEY")
        return os.environ.get(env_key) or os.environ.get("LLM_API_KEY", "")

    def chat(self, messages: list[dict], model: str = "", **kwargs) -> str:
        """
        Send a chat completion request.

        Args:
            messages: list of {"role": "user"/"assistant", "content": "..."}
            model: model name (defaults to self._model)
            **kwargs: additional parameters (temperature, max_tokens, etc.)
        """
        target_model = model or self._model
        if not target_model:
            raise RuntimeError("Model ID is required")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the llm extra: pip install '.[llm]'") from exc

        base_url = self._provider.get("base_url", "")
        # Construct full base_url for OpenAI client: base_url + /v1 (if not already included)
        endpoint_path = self._provider.get("endpoint_path", "/v1/chat/completions")
        # OpenAI client appends /chat/completions to base_url, so we need base_url + /v1
        if not base_url.endswith("/v1"):
            if endpoint_path.startswith("/v1"):
                base_url = base_url.rstrip("/") + "/v1"
            else:
                base_url = base_url.rstrip("/") + endpoint_path.rsplit("/chat/completions", 1)[0]
        
        if not self.api_key:
            raise RuntimeError("API key is missing for the selected provider")
        client_args = {"api_key": self.api_key,
                       "timeout": self._provider.get("timeout_seconds", 120)}
        if base_url:
            client_args["base_url"] = base_url
        client = OpenAI(**client_args)

        # Parameter adaptation for different model families
        params = {"model": target_model, "messages": messages}

        max_tokens = kwargs.get("max_tokens", self._provider.get("max_tokens", 2000))
        max_tokens_param = self._provider.get("max_tokens_param_name", "max_tokens")
        params[max_tokens_param] = max_tokens

        # Some provider models do not accept a temperature parameter;
        # omit it for any model listed in omit_temperature_models.
        omit_temp = self._provider.get("omit_temperature_models", [])
        if target_model not in omit_temp:
            params["temperature"] = kwargs.get("temperature", self._provider.get("temperature", 0))

        response = client.chat.completions.create(**params)
        return response.choices[0].message.content or ""

    def chat_json(self, messages: list[dict] | str, model: str = "", **kwargs) -> dict:
        """
        Send a chat completion request and parse JSON response.
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        response = self.chat(messages, model=model, **kwargs)
        return self._extract_json(response)

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Extract JSON from a text response."""
        import re
        # Try to find JSON block
        json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if json_match:
            text = json_match.group(1)

        # Try to find raw JSON object
        brace_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        # Try full parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Last resort: find the largest JSON-like block
        for i in range(len(text), 0, -1):
            try:
                return json.loads(text[:i])
            except json.JSONDecodeError:
                continue

        raise json.JSONDecodeError(f"Cannot extract JSON from: {text[:200]}", text, 0)
