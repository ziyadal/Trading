"""
tests/test_news_agent.py — unit tests for news_agent.py

All tests use a mock client — no network calls are made.
"""

from datetime import date
from unittest.mock import MagicMock

import pytest

from news_agent import USER_PROMPT, run_news_agent


# ---------------------------------------------------------------------------
# Mock client helpers
# ---------------------------------------------------------------------------

def _make_mock_client(response: str) -> MagicMock:
    """Return a mock ChatOpenAI client whose .stream() yields the response
    one character at a time (simulates token streaming)."""
    mock = MagicMock()
    chunks = [MagicMock(content=ch) for ch in response]
    mock.stream.return_value = iter(chunks)
    return mock


FAKE_REPORT = """\
## BTC News Research Report — 2026-04-04

### 1. Macro Environment
Fed holds rates steady.

### 2. Crypto Market Sentiment
Greed index at 72.

### 3. BTC-Specific News
ETF inflows of $300M recorded.

### 4. Key Risk Events (Next 24–48h)
CPI data release tomorrow.

### 5. 24h Outlook
**Directional Bias:** Bullish
BTC looks strong heading into the next session.
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_run_returns_string():
    """run_news_agent() returns a non-empty string."""
    client = _make_mock_client(FAKE_REPORT)
    result = run_news_agent(client=client)
    assert isinstance(result, str)
    assert len(result) > 0


def test_report_contains_section_headers():
    """Returned report contains all 5 expected section headings."""
    client = _make_mock_client(FAKE_REPORT)
    result = run_news_agent(client=client)
    for header in [
        "Macro Environment",
        "Crypto Market Sentiment",
        "BTC-Specific News",
        "Key Risk Events",
        "24h Outlook",
    ]:
        assert header in result, f"Missing section: {header!r}"


def test_api_error_raises_cleanly():
    """If the client raises an exception, run_news_agent propagates it."""
    mock = MagicMock()
    mock.stream.side_effect = RuntimeError("API unavailable")
    with pytest.raises(RuntimeError, match="API unavailable"):
        run_news_agent(client=mock)


def test_date_injected_in_prompt():
    """The user prompt sent to the client contains today's date."""
    client = _make_mock_client(FAKE_REPORT)
    run_news_agent(client=client)

    # Retrieve the messages passed to client.stream()
    call_args = client.stream.call_args
    messages = call_args[0][0]  # positional first arg
    human_message = next(m for m in messages if m.__class__.__name__ == "HumanMessage")
    assert date.today().isoformat() in human_message.content
