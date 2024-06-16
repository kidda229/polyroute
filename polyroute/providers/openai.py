"""OpenAI provider adapter."""

from __future__ import annotations

from typing import Any, Optional, Iterator

import httpx


class OpenAIProvider:
    """Direct OpenAI API adapter with streaming support."""

    DEFAULT_URL = "https://api.openai.com/v1"

    def __init__(self, api_key: str, base_url: Optional[str] = None, timeout: float = 60.0):
        self.api_key = api_key
        self.base_url = (base_url or self.DEFAULT_URL).rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def close(self):
        self._client.close()

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stream: bool = False,
        **kwargs,
    ) -> dict[str, Any] | Iterator[str]:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            **kwargs,
        }
        if stream:
            return self._stream_chat(body)
        return self._sync_chat(body)

    def _sync_chat(self, body: dict) -> dict[str, Any]:
        resp = self._client.post(
            f"{self.base_url}/chat/completions",
            json=body,
            headers=self._headers(),
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "content": data["choices"][0]["message"]["content"],
            "model": data.get("model", body["model"]),
            "usage": {
                "input_tokens": data["usage"]["prompt_tokens"],
                "output_tokens": data["usage"]["completion_tokens"],
# note: performance
            },
            "finish_reason": data["choices"][0].get("finish_reason"),
            "raw": data,
        }

    def _stream_chat(self, body: dict) -> Iterator[str]:
        with self._client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json=body,
            headers=self._headers(),
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    return
                import json
                chunk = json.loads(payload)
                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta and delta["content"]:
                    yield delta["content"]

    def embeddings(self, texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
        resp = self._client.post(
            f"{self.base_url}/embeddings",
            json={"model": model, "input": texts},
            headers=self._headers(),
        )
        resp.raise_for_status()
        data = resp.json()
        return [item["embedding"] for item in data["data"]]

