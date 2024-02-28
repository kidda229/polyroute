"""Configuration models for polyroute."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProviderConfig:
    """Config for a single LLM provider."""

    name: str
    api_key: str
    base_url: Optional[str] = None
    model: str = ""
    max_retries: int = 2
    timeout: float = 30.0
    priority: int = 0
    weight: float = 1.0
    enabled: bool = True

    def headers(self) -> dict[str, str]:
        if self.name == "openai":
            return {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        elif self.name == "anthropic":
            return {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }


@dataclass
class RouterConfig:
    """Top-level router configuration."""

    providers: list[ProviderConfig] = field(default_factory=list)
    fallback_order: list[str] = field(default_factory=list)
    default_provider: str = "openai"
    retry_on_status: list[int] = field(default_factory=lambda: [429, 500, 502, 503])
    request_timeout: float = 60.0

    def get_provider(self, name: str) -> Optional[ProviderConfig]:
        for p in self.providers:
            if p.name == name and p.enabled:
                return p
        return None

    def active_providers(self) -> list[ProviderConfig]:
        return sorted(
            [p for p in self.providers if p.enabled],
            key=lambda p: p.priority,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "RouterConfig":
        providers = [ProviderConfig(**p) for p in data.get("providers", [])]
        return cls(
            providers=providers,
            fallback_order=data.get("fallback_order", []),
            default_provider=data.get("default_provider", "openai"),
            retry_on_status=data.get("retry_on_status", [429, 500, 502, 503]),
            request_timeout=data.get("request_timeout", 60.0),
        )
