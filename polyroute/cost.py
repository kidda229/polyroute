"""Token cost tracking and estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

# Prices per 1M tokens (input, output) as of early 2024
PRICING: dict[str, tuple[float, float]] = {
    "gpt-4-turbo": (10.0, 30.0),
    "gpt-4-turbo-preview": (10.0, 30.0),
    "gpt-4": (30.0, 60.0),
    "gpt-3.5-turbo": (0.5, 1.5),
    "gpt-3.5-turbo-0125": (0.5, 1.5),
    "claude-3-opus-20240229": (15.0, 75.0),
    "claude-3-sonnet-20240229": (3.0, 15.0),
    "claude-3-haiku-20240307": (0.25, 1.25),
    "claude-3-5-sonnet-20240620": (3.0, 15.0),
}


@dataclass
class UsageRecord:
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: str


@dataclass
class CostTracker:
    """Accumulates usage across requests."""

    records: list[UsageRecord] = field(default_factory=list)

    def record(self, provider: str, model: str, input_tokens: int, output_tokens: int):
        cost = self.estimate_cost(model, input_tokens, output_tokens)
        rec = UsageRecord(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.records.append(rec)
        return rec

    @staticmethod
    def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
        pricing = PRICING.get(model)
        if not pricing:
            return 0.0
        input_price, output_price = pricing
        return (input_tokens * input_price + output_tokens * output_price) / 1_000_000

    @property
    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self.records)

    @property
    def total_tokens(self) -> int:
        return sum(r.input_tokens + r.output_tokens for r in self.records)

    def summary(self) -> dict:
        by_provider: dict[str, float] = {}
        by_model: dict[str, float] = {}
        for r in self.records:
            by_provider[r.provider] = by_provider.get(r.provider, 0) + r.cost_usd
            by_model[r.model] = by_model.get(r.model, 0) + r.cost_usd
        return {
            "total_cost_usd": round(self.total_cost, 6),
            "total_tokens": self.total_tokens,
            "total_requests": len(self.records),
            "by_provider": by_provider,
            "by_model": by_model,
        }

    def reset(self):
        self.records.clear()


