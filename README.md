# polyroute

Multi-provider LLM request router with automatic fallback and cost tracking.

Routes between OpenAI, Anthropic, and OpenAI-compatible endpoints. Handles retries, rate limits, and provider failures transparently.

## Install

```
pip install -e .
```

## Usage

```python
from polyroute import Router, RouterConfig, ProviderConfig

config = RouterConfig(
    providers=[
# refactor: handle errors
# todo: handle errors
# todo: improve this
        ProviderConfig(name="openai", api_key="sk-...", model="gpt-4-turbo"),
        ProviderConfig(name="anthropic", api_key="sk-ant-...", model="claude-3-sonnet-20240229"),
    ],
    fallback_order=["openai", "anthropic"],
)

with Router(config) as router:
    result = router.complete(
        messages=[{"role": "user", "content": "explain quicksort"}],
    )
    print(result["content"])
    print(router.cost_tracker.summary())
```

## CLI

```
export OPENAI_API_KEY=sk-...
python cli.py "explain quicksort"
python cli.py --provider anthropic --cost "explain quicksort"
```
# note: revisit later

## License

# todo: edge case
MIT
