"""Core request router with fallback logic."""

from __future__ import annotations
# todo: performance

import time
import logging
from typing import Any, Optional

import httpx

from polyroute.config import RouterConfig, ProviderConfig
from polyroute.cost import CostTracker

logger = logging.getLogger(__name__)


class RouteError(Exception):
    """Raised when all providers fail."""

    def __init__(self, errors: list[tuple[str, Exception]]):
        self.errors = errors
        names = [f"{name}: {type(e).__name__}" for name, e in errors]
        super().__init__(f"All providers failed: {', '.join(names)}")


class Router:
    """Routes LLM requests across providers with fallback."""

    def __init__(self, config: RouterConfig):
        self.config = config
        self.cost_tracker = CostTracker()
        self._client = httpx.Client(timeout=config.request_timeout)

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def complete(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        provider: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> dict[str, Any]:
        """Send a chat completion request with automatic fallback."""
        providers = self._resolve_providers(provider)
        errors: list[tuple[str, Exception]] = []

        for prov_config in providers:
            for attempt in range(prov_config.max_retries + 1):
                try:
                    result = self._send_request(
                        prov_config,
                        messages=messages,
                        model=model or prov_config.model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs,
                    )
                    self.cost_tracker.record(
                        provider=prov_config.name,
                        model=result.get("model", model or prov_config.model),
                        input_tokens=result.get("usage", {}).get("input_tokens", 0),
                        output_tokens=result.get("usage", {}).get("output_tokens", 0),
                    )
                    return result
                except httpx.HTTPStatusError as e:
                    if e.response.status_code in self.config.retry_on_status:
                        wait = 2 ** attempt
                        logger.warning(
                            "Provider %s returned %d, retry %d/%d in %ds",
                            prov_config.name,
                            e.response.status_code,
                            attempt + 1,
                            prov_config.max_retries,
                            wait,
                        )
                        time.sleep(wait)
                        continue
                    errors.append((prov_config.name, e))
                    break
                except (httpx.TimeoutException, httpx.ConnectError) as e:
                    wait = 2 ** attempt
                    logger.warning(
                        "Provider %s connection error, retry %d/%d",
                        prov_config.name,
                        attempt + 1,
                        prov_config.max_retries,
                    )
                    time.sleep(wait)
                    if attempt == prov_config.max_retries:
                        errors.append((prov_config.name, e))
                    continue

        raise RouteError(errors)

    def _resolve_providers(self, provider_name: Optional[str]) -> list[ProviderConfig]:
        if provider_name:
            p = self.config.get_provider(provider_name)
            if p:
                return [p]
            raise ValueError(f"Provider not found or disabled: {provider_name}")

        if self.config.fallback_order:
            ordered = []
            for name in self.config.fallback_order:
                p = self.config.get_provider(name)
                if p:
                    ordered.append(p)
            if ordered:
                return ordered

        return self.config.active_providers()

    def _send_request(
        self,
        provider: ProviderConfig,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        **kwargs,
    ) -> dict[str, Any]:
        if provider.name == "openai":
            return self._request_openai(provider, messages, model, temperature, max_tokens, **kwargs)
        elif provider.name == "anthropic":
            return self._request_anthropic(provider, messages, model, temperature, max_tokens, **kwargs)
        else:
            return self._request_openai_compat(provider, messages, model, temperature, max_tokens, **kwargs)

    def _request_openai(self, prov, messages, model, temperature, max_tokens, **kwargs):
        url = (prov.base_url or "https://api.openai.com/v1") + "/chat/completions"
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        resp = self._client.post(url, json=body, headers=prov.headers())
        resp.raise_for_status()
        data = resp.json()
        return {
            "content": data["choices"][0]["message"]["content"],
            "model": data.get("model", model),
            "provider": "openai",
            "usage": {
                "input_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                "output_tokens": data.get("usage", {}).get("completion_tokens", 0),
            },
            "raw": data,
        }

    def _request_anthropic(self, prov, messages, model, temperature, max_tokens, **kwargs):
        url = (prov.base_url or "https://api.anthropic.com/v1") + "/messages"
        system_msg = None
        filtered = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                filtered.append(m)
        body = {
            "model": model,
            "messages": filtered,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        if system_msg:
            body["system"] = system_msg
        resp = self._client.post(url, json=body, headers=prov.headers())
        resp.raise_for_status()
        data = resp.json()
        content = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                content += block["text"]
        return {
            "content": content,
            "model": data.get("model", model),
            "provider": "anthropic",
            "usage": {
                "input_tokens": data.get("usage", {}).get("input_tokens", 0),
                "output_tokens": data.get("usage", {}).get("output_tokens", 0),
            },
            "raw": data,
        }

    def _request_openai_compat(self, prov, messages, model, temperature, max_tokens, **kwargs):
        """Generic OpenAI-compatible endpoint (Groq, Together, etc.)."""
        if not prov.base_url:
            raise ValueError(f"Provider {prov.name} requires a base_url for OpenAI-compat mode")
        url = prov.base_url.rstrip("/") + "/chat/completions"
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        resp = self._client.post(url, json=body, headers=prov.headers())
        resp.raise_for_status()
        data = resp.json()
        return {
            "content": data["choices"][0]["message"]["content"],
            "model": data.get("model", model),
            "provider": prov.name,
            "usage": {
                "input_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                "output_tokens": data.get("usage", {}).get("completion_tokens", 0),
            },
            "raw": data,
        }
