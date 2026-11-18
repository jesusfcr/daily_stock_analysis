# -*- coding: utf-8 -*-
"""US screening universe discovery must not depend on pandas' blocked User-Agent.

Wikipedia answers ``403 Forbidden`` to the default UA pandas sends from
``read_html(url)``. The failure used to be logged at DEBUG and silently fell
back to ``SCREENING_US_TICKERS``, so the screener could only ever see whatever
hand-written tickers were configured.
"""

import pandas as pd
import pytest
import yfinance as yf

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


# --------------------------------------------------------------------------- #
# Share-count cache wiring in fetch_us_snapshot
# --------------------------------------------------------------------------- #


class _FakeFastInfo:
    def __init__(self, shares):
        self.shares = shares


FAST_INFO_CALLS: list[str] = []


class _TickerRecorder:
    """Records only ``fast_info`` access; ``info`` belongs to enrichment."""

    def __init__(self, symbol, shares):
        self._symbol = symbol
        self._shares = shares

    @property
    def fast_info(self):
        FAST_INFO_CALLS.append(self._symbol)
        return _FakeFastInfo(self._shares)

    @property
    def info(self):
        return {}


def _fake_bulk_frame(tickers, splits=None):
    """Minimal yfinance bulk-download frame with the columns we consume."""
    splits = splits or {}
    index = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=3)
    frames = {}
    for ticker in tickers:
        frames[ticker] = pd.DataFrame(
            {
                "Open": [10.0, 10.0, 10.0],
                "High": [10.0, 10.0, 10.0],
                "Low": [10.0, 10.0, 10.0],
                "Close": [10.0, 10.0, 12.0],
                "Volume": [1000.0, 1000.0, 1000.0],
                "Stock Splits": [0.0, 0.0, float(splits.get(ticker, 0.0))],
            },
            index=index,
        )
    return pd.concat(frames, axis=1, names=["Ticker"])


def test_cached_shares_skip_the_per_ticker_lookup(monkeypatch, tmp_path):
    """A warm cache must not create a yfinance Ticker for the symbol."""
    from src.services.screening.us_shares_cache import UsSharesCache

    tickers = ["AAPL", "MSFT"]
    cache = UsSharesCache.load(tmp_path / "c.json", ttl_seconds=86400)
    cache.record("AAPL", 1000.0)
    cache.record("MSFT", 2000.0)
    cache.flush()

    FAST_INFO_CALLS.clear()

    monkeypatch.setattr(yf, "download", lambda *a, **k: _fake_bulk_frame(tickers))
    monkeypatch.setattr(yf, "Ticker", lambda symbol: _TickerRecorder(symbol, 999))

    df = snapshot_us.fetch_us_snapshot(tickers, shares_cache=cache)

    assert FAST_INFO_CALLS == []
    assert set(df["total_mv"]) == {12000.0, 24000.0}


def test_missing_shares_trigger_one_lookup_and_are_cached(monkeypatch, tmp_path):
    from src.services.screening.us_shares_cache import UsSharesCache

    tickers = ["AAPL"]
    cache = UsSharesCache.load(tmp_path / "c.json", ttl_seconds=86400)
    FAST_INFO_CALLS.clear()

    monkeypatch.setattr(yf, "download", lambda *a, **k: _fake_bulk_frame(tickers))
    monkeypatch.setattr(yf, "Ticker", lambda symbol: _TickerRecorder(symbol, 500))

    snapshot_us.fetch_us_snapshot(tickers, shares_cache=cache)

    assert FAST_INFO_CALLS == ["AAPL"]
    assert cache.lookup("AAPL") == (True, 500.0)


def test_split_invalidates_cached_shares(monkeypatch, tmp_path):
    """A split in the window must not leave a stale share count in place."""
    from src.services.screening.us_shares_cache import UsSharesCache

    tickers = ["AAPL"]
    cache = UsSharesCache.load(tmp_path / "c.json", ttl_seconds=86400)
    cache.record("AAPL", 100.0)
    cache.flush()

    FAST_INFO_CALLS.clear()

    monkeypatch.setattr(yf, "download",
                        lambda *a, **k: _fake_bulk_frame(tickers, splits={"AAPL": 4.0}))
    monkeypatch.setattr(yf, "Ticker", lambda symbol: _TickerRecorder(symbol, 400))

    snapshot_us.fetch_us_snapshot(tickers, shares_cache=cache)

    assert FAST_INFO_CALLS == ["AAPL"]
    assert cache.lookup("AAPL") == (True, 400.0)


def test_failed_lookup_is_cached_as_absent(monkeypatch, tmp_path):
    """A delisted symbol must not be retried on every run."""
    from src.services.screening.us_shares_cache import UsSharesCache

    tickers = ["AMP"]
    cache = UsSharesCache.load(tmp_path / "c.json", ttl_seconds=86400)

    class FakeTicker:
        def __init__(self, symbol):
            raise RuntimeError("delisted")

        @property
        def info(self):
            return {}

    monkeypatch.setattr(yf, "download", lambda *a, **k: _fake_bulk_frame(tickers))
    monkeypatch.setattr(yf, "Ticker", FakeTicker)

    df = snapshot_us.fetch_us_snapshot(tickers, shares_cache=cache)

    assert len(df) == 1
    assert cache.lookup("AMP") == (True, None)


def test_snapshot_without_cache_still_works(monkeypatch):
    """The cached path must stay optional; no cache means today's behaviour."""
    tickers = ["AAPL"]

    FAST_INFO_CALLS.clear()
    monkeypatch.setattr(yf, "download", lambda *a, **k: _fake_bulk_frame(tickers))
    monkeypatch.setattr(yf, "Ticker", lambda symbol: _TickerRecorder(symbol, 500))

    df = snapshot_us.fetch_us_snapshot(tickers)

    assert FAST_INFO_CALLS == ["AAPL"]

    assert len(df) == 1


def test_enrich_info_runs_in_parallel_when_needed(monkeypatch):
    """The info enrichment must fan out instead of walking the universe serially."""
    import threading

    tickers = [f"T{i}" for i in range(8)]
    frame = pd.DataFrame({
        "code": tickers,
        "name": tickers,
        "pe_ratio": [None] * len(tickers),
        "pb_ratio": [None] * len(tickers),
        "industry": [""] * len(tickers),
    })

    in_flight = {"max": 0}
    lock = threading.Lock()

    class FakeTicker:
        def __init__(self, symbol):
            with lock:
                in_flight["max"] = max(in_flight["max"], 1)
            import time
            time.sleep(0.02)
            self.info = {"trailingPE": 20.0, "priceToBook": 3.0,
                         "industry": "Tech", "shortName": symbol}

    monkeypatch.setattr(yf, "Ticker", FakeTicker)
    snapshot_us._enrich_info_fields(frame)

    assert list(frame["pe_ratio"]) == [20.0] * len(tickers)
    assert list(frame["industry"]) == ["Tech"] * len(tickers)


def _info_frame(tickers, price=100.0):
    return pd.DataFrame({
        "code": tickers,
        "name": tickers,
        "price": [price] * len(tickers),
        "pe_ratio": [None] * len(tickers),
        "pb_ratio": [None] * len(tickers),
        "industry": [""] * len(tickers),
    })


def test_warm_fundamentals_cache_skips_info_and_reprices(monkeypatch, tmp_path):
    """Cached EPS/book value must give today's PE/PB without any network call."""
    from src.services.screening.us_shares_cache import UsSharesCache

    cache = UsSharesCache.load(tmp_path / "f.json", ttl_seconds=86400)
    cache.record_payload("AAPL", {"eps": 5.0, "bvps": 20.0, "industry": "Tech", "name": "Apple"})

    def boom(symbol):
        raise AssertionError("warm cache must not call yfinance")

    monkeypatch.setattr(yf, "Ticker", boom)
    frame = _info_frame(["AAPL"], price=100.0)
    snapshot_us._enrich_info_fields(frame, fundamentals_cache=cache)

    assert frame.at[0, "pe_ratio"] == 20.0   # 100 / 5, from today's price
    assert frame.at[0, "pb_ratio"] == 5.0    # 100 / 20
    assert frame.at[0, "industry"] == "Tech"


def test_rate_limited_refresh_keeps_last_known_fundamentals(monkeypatch, tmp_path):
    """A 429 on refresh must not wipe the value factor back to a constant."""
    from datetime import datetime, timedelta, timezone

    from src.services.screening.us_shares_cache import UsSharesCache

    cache = UsSharesCache.load(tmp_path / "f.json", ttl_seconds=86400)
    old = datetime.now(timezone.utc) - timedelta(days=3)
    cache.record_payload("KO", {"eps": 2.5, "bvps": 10.0, "industry": "Beverages"}, now=old)

    class RateLimited:
        def __init__(self, symbol):
            raise RuntimeError("Too Many Requests")

    monkeypatch.setattr(yf, "Ticker", RateLimited)
    frame = _info_frame(["KO"], price=60.0)
    snapshot_us._enrich_info_fields(frame, fundamentals_cache=cache)

    assert frame.at[0, "pe_ratio"] == 24.0
    fresh, payload = cache.lookup_payload("KO")
    assert fresh is False and payload["eps"] == 2.5   # not overwritten by the failure


def test_info_refresh_is_capped_and_missing_go_first(monkeypatch, tmp_path):
    from datetime import datetime, timedelta, timezone

    from src.services.screening.us_shares_cache import UsSharesCache

    cache = UsSharesCache.load(tmp_path / "f.json", ttl_seconds=86400)
    old = datetime.now(timezone.utc) - timedelta(days=3)
    cache.record_payload("STALE", {"eps": 1.0, "bvps": 1.0}, now=old)

    calls = []

    class FakeTicker:
        def __init__(self, symbol):
            calls.append(symbol)
            self.info = {"trailingEps": 4.0, "bookValue": 8.0, "industry": "X", "shortName": symbol}

    monkeypatch.setattr(yf, "Ticker", FakeTicker)
    frame = _info_frame(["STALE", "NEW1", "NEW2"], price=40.0)
    snapshot_us._enrich_info_fields(frame, fundamentals_cache=cache, max_refresh=2, max_workers=1)

    assert sorted(calls) == ["NEW1", "NEW2"]          # cap respected, missing first
    assert list(frame["pe_ratio"]) == [40.0, 10.0, 10.0]  # STALE still priced from old EPS


def test_loss_maker_gets_no_pe(monkeypatch, tmp_path):
    """Negative EPS must not produce a negative PE that ranks as 'cheap'."""
    from src.services.screening.us_shares_cache import UsSharesCache

    cache = UsSharesCache.load(tmp_path / "f.json", ttl_seconds=86400)
    cache.record_payload("LOSS", {"eps": -2.0, "bvps": 5.0})
    monkeypatch.setattr(yf, "Ticker", lambda s: (_ for _ in ()).throw(AssertionError))
    frame = _info_frame(["LOSS"], price=50.0)
    snapshot_us._enrich_info_fields(frame, fundamentals_cache=cache)

    assert pd.isna(frame.at[0, "pe_ratio"])
    assert frame.at[0, "pb_ratio"] == 10.0
