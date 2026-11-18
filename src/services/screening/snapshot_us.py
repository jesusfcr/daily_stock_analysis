# -*- coding: utf-8 -*-
# Derived from AlphaSift revision 9f522747caafd3c0b1ddb7e14d5cf44c8580b6cf.
# Licensed under Apache-2.0 and modified for daily_stock_analysis.
"""US equity snapshot via yfinance.

US snapshot provider for the screening L1 pipeline. Fetches a configurable
equity universe and returns the standard snapshot DataFrame schema.

HK is not supported yet: there is no HK universe source or ticker
configuration path, so ``market="hk"`` is rejected at the pipeline level
rather than silently screening the US pool.
"""

import io
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests

from src.core.trading_calendar import MARKET_TIMEZONE
from src.services.screening.us_shares_cache import UsSharesCache

logger = logging.getLogger(__name__)

_SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Wikipedia answers 403 to the default User-Agent pandas sends from
# ``read_html(url)``, which silently collapsed the US universe to whatever
# SCREENING_US_TICKERS happened to contain. Fetch the page ourselves with a
# browser UA and hand the markup to pandas instead.
_HTTP_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
_HTTP_TIMEOUT_SEC = 30.0

_DEFAULT_US_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B",
    "AVGO", "JPM", "LLY", "V", "MA", "UNH", "XOM", "COST", "HD", "PG",
    "JNJ", "ABBV", "WMT", "NFLX", "BAC", "KO", "CRM", "CVX", "MRK",
    "PEP", "AMD", "TMO", "LIN", "ACN", "CSCO", "MCD", "ABT", "ADBE",
    "WFC", "GE", "DHR", "TXN", "PM", "ISRG", "MS", "NEE", "INTU",
    "DIS", "QCOM", "CAT", "NOW",
]


def fetch_us_universe(source: str = "auto") -> list[str]:
    """Return a list of US equity tickers.

    Sources:
        sp500   — scrape S&P 500 from Wikipedia
        env     — read SCREENING_US_TICKERS (comma-separated)
        default — hardcoded top-50 US large-caps
        auto    — try sp500 → env → default
    """
    src = source.lower()
    if src == "auto":
        for s in ("sp500", "env", "default"):
            try:
                tickers = fetch_us_universe(s)
                if tickers:
                    logger.info("US universe from %s: %d tickers", s, len(tickers))
                    return tickers
            except Exception as e:
                # A blocked or broken source must stay visible: falling back to
                # the built-in list silently narrows the whole US universe.
                logger.warning("US universe source %s failed: %s", s, e)
        return list(_DEFAULT_US_UNIVERSE)

    if src == "sp500":
        return _fetch_sp500_tickers()
    elif src == "env":
        raw = os.getenv("SCREENING_US_TICKERS", "").strip()
        if not raw:
            raise ValueError("SCREENING_US_TICKERS not set")
        return [t.strip() for t in raw.split(",") if t.strip()]
    elif src == "default":
        return list(_DEFAULT_US_UNIVERSE)
    else:
        raise ValueError(f"Unknown US universe source: {source}")


def _fetch_text(url: str, timeout: float = _HTTP_TIMEOUT_SEC) -> str:
    """Fetch a page as text with a User-Agent hosts like Wikipedia accept."""
    response = requests.get(url, headers={"User-Agent": _HTTP_USER_AGENT}, timeout=timeout)
    response.raise_for_status()
    return response.text


def _fetch_sp500_tickers() -> list[str]:
    tables = pd.read_html(io.StringIO(_fetch_text(_SP500_WIKI_URL)))
    for tbl in tables:
        if "Symbol" in tbl.columns:
            return sorted(tbl["Symbol"].dropna().str.strip().str.replace(".", "-", regex=False).tolist())
    raise RuntimeError("Could not find Symbol column in S&P 500 Wikipedia table")


def _tickers_with_recent_splits(data: pd.DataFrame, tickers: list[str]) -> list[str]:
    """Symbols whose cached share count a split in the window would invalidate."""
    if data is None or getattr(data, "empty", True):
        return []
    affected: list[str] = []
    for ticker in tickers:
        try:
            splits = _split_series(data, ticker)
            if splits is None:
                continue
            if float(pd.to_numeric(splits, errors="coerce").fillna(0).sum()) != 0.0:
                affected.append(ticker)
        except Exception:
            continue
    return affected


def _split_series(data: pd.DataFrame, ticker: str):
    """Return the split series for one symbol, whichever frame shape yfinance used."""
    if isinstance(data.columns, pd.MultiIndex):
        if ticker not in data.columns.get_level_values(0):
            return None
        frame = data[ticker]
        return frame["Stock Splits"] if "Stock Splits" in frame.columns else None
    return data["Stock Splits"] if "Stock Splits" in data.columns else None


def _resolve_shares(ticker: str, shares_cache: UsSharesCache | None) -> float:
    """Share count from the cache, falling back to one ``fast_info`` lookup.

    ``fast_info`` is one HTTP round-trip per symbol and the main memory driver
    of the snapshot phase, so it only runs for symbols the cache cannot answer.
    """
    if shares_cache is not None:
        known, cached = shares_cache.lookup(ticker)
        if known:
            return cached or 0.0

    import yfinance as yf

    shares = 0.0
    try:
        info = yf.Ticker(ticker).fast_info
        shares = float(getattr(info, "shares", None) or 0)
    except Exception as exc:
        logger.debug("fast_info failed for %s: %s", ticker, exc)
        shares = 0.0
    if shares_cache is not None:
        shares_cache.record(ticker, shares if shares > 0 else None)
    return shares


def fetch_us_snapshot(
    tickers: list[str] | None = None,
    *,
    universe_source: str = "auto",
    max_workers: int = 8,
    shares_cache: UsSharesCache | None = None,
) -> pd.DataFrame:
    """Fetch a US equity snapshot in the screening schema.

    Uses yfinance to fetch current data for each ticker. Returns a
    DataFrame matching the standard snapshot columns: code, name, price,
    change_pct, amount, total_mv, pe_ratio, pb_ratio, volume_ratio,
    turnover_rate, industry.
    """
    import yfinance as yf

    if tickers is None:
        tickers = fetch_us_universe(universe_source)

    logger.info("Fetching US snapshot for %d tickers", len(tickers))

    hist_end = pd.Timestamp.now().normalize()
    hist_start = hist_end - pd.Timedelta(days=30)
    data = yf.download(
        tickers,
        start=hist_start.strftime("%Y-%m-%d"),
        end=hist_end.strftime("%Y-%m-%d"),
        group_by="ticker",
        auto_adjust=True,
        actions=True,
        progress=False,
        threads=True,
    )

    # A split changes the share count, so any cached value for that symbol is
    # stale even if it is still within its TTL.
    split_tickers = _tickers_with_recent_splits(data, tickers)
    if shares_cache is not None and split_tickers:
        shares_cache.invalidate(split_tickers)

    rows = []

    def _process_ticker(ticker: str) -> dict | None:
        try:
            if len(tickers) == 1:
                hist = data.copy()
                if isinstance(hist.columns, pd.MultiIndex):
                    hist.columns = hist.columns.droplevel("Ticker")
            else:
                if ticker not in data.columns.get_level_values(0):
                    return None
                hist = data[ticker].copy()
            if hist.empty:
                return None

            hist = hist[hist["Close"].notna()]
            if len(hist) < 2:
                return None

            latest = hist.iloc[-1]
            prev = hist.iloc[-2]
            price = float(latest["Close"])
            prev_close = float(prev["Close"])
            volume = float(latest["Volume"])
            change_pct = ((price - prev_close) / prev_close * 100) if prev_close > 0 else 0.0

            vol_20d = float(hist["Volume"].tail(20).mean())
            volume_ratio = (volume / vol_20d) if vol_20d > 0 else 1.0

            shares = _resolve_shares(ticker, shares_cache)
            market_cap = round(price * shares, 0) if shares > 0 else 0
            turnover_rate = (volume / shares * 100) if shares > 0 else 0.0

            return {
                "code": ticker,
                "name": ticker,
                "price": price,
                "change_pct": round(change_pct, 2),
                "amount": round(volume * price, 0),
                "total_mv": market_cap,
                "circ_mv": market_cap,
                "pe_ratio": None,
                "pb_ratio": None,
                "volume_ratio": round(volume_ratio, 2),
                "turnover_rate": round(turnover_rate, 4),
                "industry": "",
            }
        except Exception as e:
            logger.debug("Failed to process %s: %s", ticker, e)
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process_ticker, t): t for t in tickers}
        for future in as_completed(futures):
            result = future.result()
            if result:
                rows.append(result)

    if not rows:
        raise RuntimeError("yfinance returned no valid data for any ticker")

    if shares_cache is not None:
        shares_cache.flush()

    df = pd.DataFrame(rows)

    numeric_cols = [
        "price", "change_pct", "amount", "total_mv", "circ_mv",
        "pe_ratio", "pb_ratio", "volume_ratio", "turnover_rate",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]

    _enrich_info_fields(df, fundamentals_cache=_fundamentals_cache_for(shares_cache))

    df.attrs["snapshot_source"] = "yfinance"
    logger.info("US snapshot: %d rows from yfinance", len(df))
    return df


_MAX_INFO_REFRESH_PER_RUN = 150


def _fundamentals_cache_for(shares_cache: UsSharesCache | None) -> UsSharesCache | None:
    """Fundamentals cache living next to the shares cache, sharing its switch."""
    if shares_cache is None:
        return None
    from src.services.screening.us_shares_cache import FUNDAMENTALS_CACHE_FILENAME

    return UsSharesCache.load(
        shares_cache.path.parent / FUNDAMENTALS_CACHE_FILENAME,
        ttl_seconds=shares_cache.ttl_seconds,
    )


def _ratio(price: object, per_share: object) -> float | None:
    try:
        price_f, per_share_f = float(price), float(per_share)
    except (TypeError, ValueError):
        return None
    if not (price_f > 0 and per_share_f > 0):
        return None
    return round(price_f / per_share_f, 4)


def _enrich_info_fields(
    df: pd.DataFrame,
    *,
    max_workers: int = 8,
    fundamentals_cache: UsSharesCache | None = None,
    max_refresh: int = _MAX_INFO_REFRESH_PER_RUN,
) -> None:
    """Best-effort enrichment of pe_ratio, pb_ratio, industry from yfinance info.

    ``Ticker.info`` is one HTTP round-trip per symbol and Yahoo rate-limits it
    (HTTP 429) after a few full-universe passes, which silently turned the
    ``value`` factor into a constant. With a cache, per-share EPS and book
    value are stored and PE/PB are recomputed from today's price, so only
    missing or expired symbols hit the network, at most ``max_refresh`` per
    run; the rest of the expired ones keep their last known values.
    """
    import yfinance as yf

    needs_pe = df["pe_ratio"].isna().sum() > len(df) * 0.5
    if not needs_pe or df.empty:
        return

    tickers = [str(code) for code in df["code"].tolist()]
    known: dict[str, dict] = {}
    to_fetch = list(tickers)
    if fundamentals_cache is not None:
        missing: list[str] = []
        stale: list[str] = []
        for ticker in tickers:
            fresh, payload = fundamentals_cache.lookup_payload(ticker)
            if payload:
                known[ticker] = payload
            if not fresh:
                (stale if payload else missing).append(ticker)
        to_fetch = (missing + stale)[:max(0, max_refresh)]

    def _fetch(ticker: str) -> tuple[str, dict]:
        try:
            info = yf.Ticker(ticker).info
        except Exception:
            return ticker, {}
        if not isinstance(info, dict) or not info:
            return ticker, {}
        return ticker, {
            "pe": info.get("trailingPE"),
            "pb": info.get("priceToBook"),
            "eps": info.get("trailingEps"),
            "bvps": info.get("bookValue"),
            "industry": info.get("industry", ""),
            "name": info.get("shortName", ticker),
        }

    if to_fetch:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for ticker, payload in pool.map(_fetch, to_fetch):
                if not payload:
                    continue  # failed lookup: keep any stale value, retry next run
                known[ticker] = payload
                if fundamentals_cache is not None:
                    fundamentals_cache.record_payload(ticker, payload)
        if fundamentals_cache is not None:
            fundamentals_cache.flush()

    has_price = "price" in df.columns
    for idx in df.index:
        ticker = str(df.at[idx, "code"])
        info = known.get(ticker)
        if not info:
            continue
        price = df.at[idx, "price"] if has_price else None
        if pd.isna(df.at[idx, "pe_ratio"]) or df.at[idx, "pe_ratio"] == 0:
            pe = _ratio(price, info.get("eps")) if info.get("eps") is not None else info.get("pe")
            df.at[idx, "pe_ratio"] = pe
        if pd.isna(df.at[idx, "pb_ratio"]) or df.at[idx, "pb_ratio"] == 0:
            pb = _ratio(price, info.get("bvps")) if info.get("bvps") is not None else info.get("pb")
            df.at[idx, "pb_ratio"] = pb
        if not df.at[idx, "industry"]:
            df.at[idx, "industry"] = info.get("industry") or ""
        if not df.at[idx, "name"] or df.at[idx, "name"] == ticker:
            df.at[idx, "name"] = info.get("name") or ticker


def fetch_daily_history_yfinance(
    ticker: str,
    *,
    lookback_days: int = 120,
) -> pd.DataFrame:
    """Fetch daily OHLCV history for a US ticker via yfinance.

    Returns a DataFrame with columns: date, open, high, low, close, volume
    matching the schema expected by the daily enrichment logic.
    """
    import yfinance as yf

    today = pd.Timestamp.now(tz=MARKET_TIMEZONE["us"]).normalize()
    # yfinance excludes end; include today's US session regardless of host TZ.
    end = today + pd.DateOffset(days=1)
    start = today - pd.Timedelta(days=max(lookback_days * 2, 180))
    hist = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    if hist is None or hist.empty:
        raise RuntimeError(f"yfinance daily history empty for {ticker}")

    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.droplevel("Ticker")

    hist = hist.tail(max(lookback_days, 30)).copy()
    hist = hist.rename(columns={
        "Open": "开盘", "High": "最高", "Low": "最低",
        "Close": "收盘", "Volume": "成交量",
    })
    hist.index.name = "日期"
    hist = hist.reset_index()
    return hist
