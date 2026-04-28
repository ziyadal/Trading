"""LLM client factory — every chat model in this project routes through OpenRouter.

One helper, one env var (OPENROUTER_API_KEY), one integration. Keeps model selection
to a single place so provider/model swaps don't require hunting through the code.

Uses the dedicated `langchain-openrouter` package rather than ChatOpenAI+base_url.
The dedicated integration preserves provider-specific fields (e.g. structured-output
details) that a generic OpenAI-compatible client would strip.

Model names follow OpenRouter's `provider/model` convention, e.g.:
    openai/gpt-4.1-mini
    anthropic/claude-3.5-haiku
    google/gemini-2.0-flash-001
    perplexity/sonar-pro
"""

from langchain_openrouter import ChatOpenRouter


def make_llm(model: str) -> ChatOpenRouter:
    """Return a ChatOpenRouter configured to route through openrouter.ai.

    Reads OPENROUTER_API_KEY from the environment automatically.
    """
    return ChatOpenRouter(model=model)
