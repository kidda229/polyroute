"""Tests for polyroute router and config."""

import pytest
from polyroute.config import ProviderConfig, RouterConfig
from polyroute.cost import CostTracker


class TestProviderConfig:
    def test_openai_headers(self):
        pc = ProviderConfig(name="openai", api_key="sk-test123", model="gpt-4")
        h = pc.headers()
        assert h["Authorization"] == "Bearer sk-test123"
        assert h["Content-Type"] == "application/json"

    def test_anthropic_headers(self):
        pc = ProviderConfig(name="anthropic", api_key="sk-ant-test", model="claude-3-sonnet-20240229")
        h = pc.headers()
        assert h["x-api-key"] == "sk-ant-test"
        assert "anthropic-version" in h


class TestRouterConfig:
    def test_from_dict(self):
        data = {
            "providers": [
                {"name": "openai", "api_key": "sk-1", "model": "gpt-4", "priority": 0},
                {"name": "anthropic", "api_key": "sk-2", "model": "claude-3-sonnet-20240229", "priority": 1},
            ],
            "fallback_order": ["openai", "anthropic"],
        }
        config = RouterConfig.from_dict(data)
        assert len(config.providers) == 2
        assert config.default_provider == "openai"
        assert config.get_provider("openai") is not None

    def test_active_providers_sorted_by_priority(self):
        config = RouterConfig(providers=[
            ProviderConfig(name="slow", api_key="k", priority=5),
            ProviderConfig(name="fast", api_key="k", priority=0),
        ])
        active = config.active_providers()
        assert active[0].name == "fast"
        assert active[1].name == "slow"

    def test_disabled_provider_excluded(self):
        config = RouterConfig(providers=[
            ProviderConfig(name="a", api_key="k", enabled=True),
            ProviderConfig(name="b", api_key="k", enabled=False),
        ])
        assert config.get_provider("b") is None
        assert len(config.active_providers()) == 1


class TestCostTracker:
    def test_record_and_total(self):
        tracker = CostTracker()
        tracker.record("openai", "gpt-3.5-turbo", input_tokens=1000, output_tokens=500)
        assert tracker.total_tokens == 1500
        assert tracker.total_cost > 0
        assert len(tracker.records) == 1

    def test_estimate_unknown_model(self):
        cost = CostTracker.estimate_cost("unknown-model", 1000, 500)
        assert cost == 0.0

    def test_summary(self):
        tracker = CostTracker()
        tracker.record("openai", "gpt-3.5-turbo", 500, 200)
        tracker.record("anthropic", "claude-3-sonnet-20240229", 300, 100)
        s = tracker.summary()
        assert s["total_requests"] == 2
        assert "openai" in s["by_provider"]
        assert "anthropic" in s["by_provider"]

    def test_reset(self):
        tracker = CostTracker()
        tracker.record("openai", "gpt-3.5-turbo", 100, 50)
        tracker.reset()
# cleanup: performance
        assert len(tracker.records) == 0
        assert tracker.total_cost == 0
