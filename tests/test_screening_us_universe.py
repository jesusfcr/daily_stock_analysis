# -*- coding: utf-8 -*-
"""US screening universe discovery must not depend on pandas' blocked User-Agent.

Wikipedia answers ``403 Forbidden`` to the default UA pandas sends from
``read_html(url)``. The failure used to be logged at DEBUG and silently fell
back to ``SCREENING_US_TICKERS``, so the screener could only ever see whatever
hand-written tickers were configured.
"""

import pytest

from src.services.screening import snapshot_us


_SP500_HTML = """
<html><body><table>
<tr><th>Symbol</th><th>Security</th></tr>
<tr><td>BRK.B</td><td>Berkshire Hathaway</td></tr>
<tr><td>AAPL</td><td>Apple Inc.</td></tr>
<tr><td>MSFT</td><td>Microsoft</td></tr>
</table></body></html>
"""


class _FakeResponse:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        return None


def test_sp500_universe_sends_browser_user_agent(monkeypatch):
    captured = {}

    def fake_get(url, headers=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers or {}
        return _FakeResponse(_SP500_HTML)

    monkeypatch.setattr(snapshot_us.requests, "get", fake_get)

    tickers = snapshot_us.fetch_us_universe("sp500")

    # Sorted, and the Wikipedia dot class-share suffix is normalised for Yahoo.
    assert tickers == ["AAPL", "BRK-B", "MSFT"]
    assert captured["url"] == snapshot_us._SP500_WIKI_URL
    assert "Mozilla" in captured["headers"]["User-Agent"]


def test_sp500_failure_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("SCREENING_US_TICKERS", "NVDA, AMD")

    def blocked(*_args, **_kwargs):
        raise RuntimeError("HTTP Error 403: Forbidden")

    monkeypatch.setattr(snapshot_us.requests, "get", blocked)

    assert snapshot_us.fetch_us_universe("auto") == ["NVDA", "AMD"]


def test_auto_prefers_sp500_over_env(monkeypatch):
    monkeypatch.setenv("SCREENING_US_TICKERS", "NVDA")

    monkeypatch.setattr(
        snapshot_us, "requests",
        type("R", (), {"get": staticmethod(lambda *a, **k: _FakeResponse(_SP500_HTML))}),
    )

    assert snapshot_us.fetch_us_universe("auto") == ["AAPL", "BRK-B", "MSFT"]


def test_failed_source_is_reported_at_warning(monkeypatch, caplog):
    """A blocked universe source must not disappear into DEBUG logs."""
    monkeypatch.delenv("SCREENING_US_TICKERS", raising=False)

    def blocked(*_args, **_kwargs):
        raise RuntimeError("HTTP Error 403: Forbidden")

    monkeypatch.setattr(snapshot_us.requests, "get", blocked)

    with caplog.at_level("WARNING", logger=snapshot_us.__name__):
        tickers = snapshot_us.fetch_us_universe("auto")

    # Falls through to the hardcoded default, but says so out loud.
    assert tickers == list(snapshot_us._DEFAULT_US_UNIVERSE)
    assert any("sp500" in record.getMessage() and "403" in record.getMessage()
               for record in caplog.records if record.levelname == "WARNING")


def test_unknown_source_is_rejected():
    with pytest.raises(ValueError):
        snapshot_us.fetch_us_universe("nasdaq")
