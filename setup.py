from setuptools import setup, find_packages

setup(
    name="polyroute",
    version="0.1.0",
    description="Multi-provider LLM request router with fallback and cost tracking",
    author="chu2bard",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "requests>=2.31.0",
        "httpx>=0.25.0",
        "pydantic>=2.5.0",
        "tiktoken>=0.5.0",
    ],
)
