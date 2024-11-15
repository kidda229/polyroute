"""Tests for cost tracking."""

from polyroute.cost import CostTracker, PRICING


def test_known_model_cost():
    cost = CostTracker.estimate_cost("gpt-4o", 1_000_000, 1_000_000)
    assert cost == 5.0 + 15.0


def test_gpt4_turbo_cost():
    cost = CostTracker.estimate_cost("gpt-4-turbo", 1000, 500)
    expected = (1000 * 10.0 + 500 * 30.0) / 1_000_000
    assert abs(cost - expected) < 1e-9


def test_tracker_accumulates():
    t = CostTracker()
    t.record("openai", "gpt-4o", 100, 50)
    t.record("openai", "gpt-4o", 200, 100)
    assert len(t.records) == 2
    assert t.total_tokens == 450


def test_summary_groups_by_provider():
    t = CostTracker()
    t.record("openai", "gpt-4o", 100, 50)
    t.record("anthropic", "claude-3-sonnet-20240229", 100, 50)
    s = t.summary()
    assert "openai" in s["by_provider"]
    assert "anthropic" in s["by_provider"]
    assert s["total_requests"] == 2
