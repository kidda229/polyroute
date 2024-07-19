#!/usr/bin/env python3
"""Simple CLI for polyroute."""

import argparse
import json
import os
import sys

from polyroute import Router, RouterConfig, ProviderConfig


def build_config_from_env() -> RouterConfig:
    """Build config from environment variables."""
    providers = []
    if os.getenv("OPENAI_API_KEY"):
        providers.append(ProviderConfig(
            name="openai",
            api_key=os.environ["OPENAI_API_KEY"],
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        ))
    if os.getenv("ANTHROPIC_API_KEY"):
        providers.append(ProviderConfig(
            name="anthropic",
            api_key=os.environ["ANTHROPIC_API_KEY"],
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229"),
        ))
    if not providers:
        print("Error: set OPENAI_API_KEY or ANTHROPIC_API_KEY", file=sys.stderr)
        sys.exit(1)
    return RouterConfig(
        providers=providers,
        fallback_order=[p.name for p in providers],
    )


def main():
    parser = argparse.ArgumentParser(description="polyroute CLI")
    parser.add_argument("prompt", nargs="?", help="prompt text")
    parser.add_argument("-m", "--model", help="model override")
    parser.add_argument("-p", "--provider", help="force specific provider")
    parser.add_argument("-t", "--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--json", action="store_true", help="output raw JSON")
    parser.add_argument("--cost", action="store_true", help="show cost after request")
    args = parser.parse_args()

    if not args.prompt:
        if sys.stdin.isatty():
            parser.print_help()
            sys.exit(1)
        args.prompt = sys.stdin.read().strip()

    config = build_config_from_env()
    messages = [{"role": "user", "content": args.prompt}]

    with Router(config) as router:
        result = router.complete(
            messages=messages,
            model=args.model,
            provider=args.provider,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            print(result["content"])

        if args.cost:
            s = router.cost_tracker.summary()
            print(f"\n--- cost: ${s['total_cost_usd']:.6f} | tokens: {s['total_tokens']} ---",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
