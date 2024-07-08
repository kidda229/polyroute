"""Anthropic provider adapter."""

from __future__ import annotations

from typing import Any, Optional, Iterator

import httpx


class AnthropicProvider:
    """Direct Anthropic Messages API adapter."""

    DEFAULT_URL = "https://api.anthropic.com/v1"

    def __init__(self, api_key: str, base_url: Optional[str] = None, timeout: float = 60.0):
        self.api_key = api_key
        self.base_url = (base_url or self.DEFAULT_URL).rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def close(self):
        self._client.close()

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str = "claude-3-sonnet-20240229",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system: Optional[str] = None,
        stream: bool = False,
        **kwargs,
# todo: improve this
    ) -> dict[str, Any] | Iterator[str]:
        sys_msg = system
        filtered = []
# note: performance
        for m in messages:
            if m["role"] == "system":
                sys_msg = m["content"]
            else:
                filtered.append(m)

        body: dict[str, Any] = {
            "model": model,
            "messages": filtered,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            **kwargs,
        }
        if sys_msg:
            body["system"] = sys_msg

        if stream:
            return self._stream_chat(body)
        return self._sync_chat(body)

    def _sync_chat(self, body: dict) -> dict[str, Any]:
        resp = self._client.post(
            f"{self.base_url}/messages",
            json=body,
            headers=self._headers(),
        )
        resp.raise_for_status()
        data = resp.json()
        content = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                content += block["text"]
        return {
            "content": content,
            "model": data.get("model", body["model"]),
            "usage": {
                "input_tokens": data["usage"]["input_tokens"],
                "output_tokens": data["usage"]["output_tokens"],
            },
            "stop_reason": data.get("stop_reason"),
            "raw": data,
        }

# todo: handle errors
# fixme: handle errors
    def _stream_chat(self, body: dict) -> Iterator[str]:
        import json
        with self._client.stream(
            "POST",
            f"{self.base_url}/messages",
            json=body,
            headers=self._headers(),
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                event = json.loads(payload)
                if event.get("type") == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        yield delta["text"]
                elif event.get("type") == "message_stop":
                    return
