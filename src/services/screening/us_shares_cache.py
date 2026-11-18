# -*- coding: utf-8 -*-
# Not derived from AlphaSift: original daily_stock_analysis code.
"""Disk cache for the slow per-ticker US ``fast_info`` lookups.

Building a US screening snapshot needs one thing that the bulk
``yf.download`` call cannot provide: each ticker's share count, used to derive
``total_mv`` and ``turnover_rate``. yfinance only exposes it through
``yf.Ticker(symbol).fast_info``, i.e. one HTTP round-trip per symbol. At
S&P 500 size that is the dominant cost of the snapshot phase and the peak
memory driver, because it materialises a ``Ticker`` object per symbol.

Share counts change far less often than prices: splits, issuance and
buybacks only. Prices keep coming from the bulk download on every run, so
caching ``shares`` (rather than ``market_cap``) keeps ``total_mv`` correct as
the tape moves; only a split can invalidate an entry, and the snapshot code
forces a refresh for any ticker whose split column is non-zero.

This module is deliberately dependency-free and knows nothing about yfinance
or about screening configuration, so it can be tested and reused on its own.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

logger = logging.getLogger(__name__)

CACHE_FILENAME = "us_shares_cache.json"
# Same file format, separate file: per-share EPS / book value / industry from
# ``Ticker.info``. Kept apart so the two TTLs and failure modes stay independent.
FUNDAMENTALS_CACHE_FILENAME = "us_fundamentals_cache.json"
CACHE_VERSION = 1
DEFAULT_TTL_DAYS = 7.0


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


@dataclass
class UsSharesCache:
    """Per-ticker share-count cache backed by a single JSON file.

    A ``None`` share count is a recorded outcome, not a gap: symbols whose
    lookup raised (delisted, transient upstream error) are remembered so they
    are not retried on every run.
    """

    path: Path
    ttl_seconds: float = DEFAULT_TTL_DAYS * 86400
    _entries: Dict[str, Dict[str, Any]] = field(default_factory=dict, repr=False)
    _dirty: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    # ------------------------------------------------------------------ reads

    @classmethod
    def load(cls, path: Path | str, *, ttl_seconds: float = DEFAULT_TTL_DAYS * 86400) -> "UsSharesCache":
        """Read the cache, tolerating a missing, unreadable or stale-schema file."""
        cache = cls(path=Path(path), ttl_seconds=ttl_seconds)
        try:
            payload = json.loads(cache.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return cache
        except (OSError, ValueError) as exc:
            logger.warning("US shares cache unreadable at %s (%s); starting empty", cache.path, exc)
            return cache
        if not isinstance(payload, dict) or payload.get("version") != CACHE_VERSION:
            logger.info("US shares cache at %s has an old or unknown schema; starting empty", cache.path)
            return cache
        entries = payload.get("tickers")
        if isinstance(entries, dict):
            cache._entries = {
                str(key).strip().upper(): value
                for key, value in entries.items()
                if isinstance(value, dict)
            }
        return cache

    def lookup(self, ticker: str, *, now: Optional[datetime] = None) -> Tuple[bool, Optional[float]]:
        """Return ``(is_fresh, shares)`` for one ticker.

        ``is_fresh`` is False when the entry is missing, malformed or expired,
        which tells the caller to perform a real lookup.
        """
        key = str(ticker or "").strip().upper()
        if not key:
            return False, None
        entry = self._entries.get(key)
        if not isinstance(entry, dict):
            return False, None
        fetched_at = _parse_timestamp(entry.get("fetched_at"))
        if fetched_at is None:
            return False, None
        reference = now or _utc_now()
        if self.ttl_seconds > 0 and (reference - fetched_at).total_seconds() > self.ttl_seconds:
            return False, None
        shares = entry.get("shares")
        if shares is None:
            return True, None
        try:
            value = float(shares)
        except (TypeError, ValueError):
            return False, None
        return (True, value) if value > 0 else (True, None)

    def lookup_payload(
        self, ticker: str, *, now: Optional[datetime] = None
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Return ``(is_fresh, payload)`` for caches that store a field dict.

        Unlike :meth:`lookup`, an expired entry still returns its payload: for
        slow-moving fundamentals a week-old value beats none when the upstream
        is rate-limiting, and the caller decides whether to refresh it.
        """
        key = str(ticker or "").strip().upper()
        entry = self._entries.get(key) if key else None
        if not isinstance(entry, dict):
            return False, None
        payload = {k: v for k, v in entry.items() if k != "fetched_at"}
        fetched_at = _parse_timestamp(entry.get("fetched_at"))
        if fetched_at is None:
            return False, payload
        reference = now or _utc_now()
        fresh = self.ttl_seconds <= 0 or (reference - fetched_at).total_seconds() <= self.ttl_seconds
        return fresh, payload

    def record_payload(
        self, ticker: str, payload: Dict[str, Any], *, now: Optional[datetime] = None
    ) -> None:
        """Remember one successful field-dict lookup."""
        key = str(ticker or "").strip().upper()
        if not key or not isinstance(payload, dict):
            return
        entry = dict(payload)
        entry["fetched_at"] = (now or _utc_now()).isoformat().replace("+00:00", "Z")
        self._entries[key] = entry
        self._dirty = True

    def shares(self, ticker: str, *, now: Optional[datetime] = None) -> Optional[float]:
        """Cached share count, or ``None`` when unknown or expired."""
        return self.lookup(ticker, now=now)[1]

    def is_fresh(self, ticker: str, *, now: Optional[datetime] = None) -> bool:
        return self.lookup(ticker, now=now)[0]

    def __len__(self) -> int:
        return len(self._entries)

    # ----------------------------------------------------------------- writes

    def record(self, ticker: str, shares: Optional[float], *, now: Optional[datetime] = None) -> None:
        """Remember one lookup result; ``None`` records a known-failed lookup."""
        key = str(ticker or "").strip().upper()
        if not key:
            return
        value: Optional[float]
        try:
            value = float(shares) if shares is not None else None
        except (TypeError, ValueError):
            value = None
        if value is not None and value <= 0:
            value = None
        self._entries[key] = {
            "shares": value,
            "fetched_at": (now or _utc_now()).isoformat().replace("+00:00", "Z"),
        }
        self._dirty = True

    def invalidate(self, tickers: Iterable[str]) -> None:
        """Drop entries so the next lookup goes to the network."""
        for ticker in tickers:
            key = str(ticker or "").strip().upper()
            if key and self._entries.pop(key, None) is not None:
                self._dirty = True

    def flush(self) -> bool:
        """Persist atomically. Returns False when the write is not possible."""
        if not self._dirty:
            return True
        payload = {"version": CACHE_VERSION, "tickers": self._entries}
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp_path, self.path)
        except OSError as exc:
            logger.warning("US shares cache write failed at %s (%s); continuing without it", self.path, exc)
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            return False
        self._dirty = False
        return True


def build_us_shares_cache(
    data_dir: Path | str | None,
    *,
    ttl_days: float = DEFAULT_TTL_DAYS,
    enabled: bool = True,
) -> Optional[UsSharesCache]:
    """Create the cache used by the US snapshot, or ``None`` when disabled.

    Returning ``None`` keeps the caller on the uncached path, so a broken or
    disabled cache degrades to today's behaviour instead of failing the run.
    """
    if not enabled:
        return None
    try:
        ttl_days = float(ttl_days)
    except (TypeError, ValueError):
        ttl_days = DEFAULT_TTL_DAYS
    if ttl_days <= 0:
        return None
    base = Path(data_dir) if data_dir else Path("data")
    return UsSharesCache.load(base / CACHE_FILENAME, ttl_seconds=ttl_days * 86400)
