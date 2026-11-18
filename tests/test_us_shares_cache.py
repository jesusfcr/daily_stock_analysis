# -*- coding: utf-8 -*-
"""Contract tests for the US share-count cache (no network, no yfinance)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from src.services.screening.us_shares_cache import (
    CACHE_FILENAME,
    CACHE_VERSION,
    UsSharesCache,
    build_us_shares_cache,
)


def _now() -> datetime:
    return datetime(2026, 11, 18, 12, 0, tzinfo=timezone.utc)


def test_missing_file_starts_empty_and_is_not_fresh(tmp_path):
    cache = UsSharesCache.load(tmp_path / CACHE_FILENAME, ttl_seconds=86400)

    assert len(cache) == 0
    assert cache.lookup("AAPL", now=_now()) == (False, None)
    assert cache.is_fresh("AAPL", now=_now()) is False


def test_record_then_reload_round_trip(tmp_path):
    path = tmp_path / CACHE_FILENAME
    cache = UsSharesCache.load(path, ttl_seconds=86400)
    cache.record("AAPL", 14_840_000_000, now=_now())
    assert cache.flush() is True

    reloaded = UsSharesCache.load(path, ttl_seconds=86400)
    assert reloaded.lookup("AAPL", now=_now()) == (True, 14_840_000_000.0)
    assert reloaded.shares("aapl", now=_now()) == 14_840_000_000.0


def test_recorded_failure_is_fresh_with_no_value(tmp_path):
    """A symbol whose lookup raised must not be retried on every run."""
    cache = UsSharesCache.load(tmp_path / CACHE_FILENAME, ttl_seconds=86400)
    cache.record("AMP", None, now=_now())

    assert cache.lookup("AMP", now=_now()) == (True, None)


def test_entries_expire_after_ttl(tmp_path):
    cache = UsSharesCache.load(tmp_path / CACHE_FILENAME, ttl_seconds=3600)
    cache.record("AAPL", 100.0, now=_now())

    assert cache.is_fresh("AAPL", now=_now() + timedelta(minutes=30)) is True
    assert cache.lookup("AAPL", now=_now() + timedelta(hours=2)) == (False, None)


def test_ttl_zero_disables_expiry_but_not_lookup(tmp_path):
    cache = UsSharesCache.load(tmp_path / CACHE_FILENAME, ttl_seconds=0)
    cache.record("AAPL", 100.0, now=_now())

    assert cache.lookup("AAPL", now=_now() + timedelta(days=3650)) == (True, 100.0)


def test_invalidate_forces_refetch(tmp_path):
    cache = UsSharesCache.load(tmp_path / CACHE_FILENAME, ttl_seconds=86400)
    cache.record("AAPL", 100.0, now=_now())
    cache.invalidate(["AAPL"])

    assert cache.lookup("AAPL", now=_now()) == (False, None)


def test_corrupt_file_degrades_to_empty_cache(tmp_path):
    path = tmp_path / CACHE_FILENAME
    path.write_text("{not json", encoding="utf-8")

    cache = UsSharesCache.load(path, ttl_seconds=86400)

    assert len(cache) == 0
    assert cache.lookup("AAPL", now=_now()) == (False, None)


def test_unknown_schema_is_discarded(tmp_path):
    path = tmp_path / CACHE_FILENAME
    path.write_text(json.dumps({"version": 99, "tickers": {"AAPL": {"shares": 1}}}), encoding="utf-8")

    cache = UsSharesCache.load(path, ttl_seconds=86400)

    assert len(cache) == 0


def test_write_failure_is_reported_not_raised(tmp_path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x", encoding="utf-8")
    cache = UsSharesCache.load(blocker / CACHE_FILENAME, ttl_seconds=86400)
    cache.record("AAPL", 100.0, now=_now())

    assert cache.flush() is False


def test_flush_is_a_noop_when_nothing_changed(tmp_path):
    path = tmp_path / CACHE_FILENAME
    cache = UsSharesCache.load(path, ttl_seconds=86400)

    assert cache.flush() is True
    assert path.exists() is False


def test_cache_file_is_written_atomically(tmp_path):
    path = tmp_path / CACHE_FILENAME
    cache = UsSharesCache.load(path, ttl_seconds=86400)
    cache.record("AAPL", 100.0, now=_now())
    cache.flush()

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == CACHE_VERSION
    assert payload["tickers"]["AAPL"]["shares"] == 100.0
    assert not path.with_suffix(path.suffix + ".tmp").exists()


@pytest.mark.parametrize("enabled,ttl,expect_none", [
    (False, 7, True),
    (True, 0, True),
    (True, -1, True),
    (True, 7, False),
])
def test_build_respects_enabled_and_ttl(tmp_path, enabled, ttl, expect_none):
    built = build_us_shares_cache(tmp_path, ttl_days=ttl, enabled=enabled)

    assert (built is None) is expect_none


def test_build_places_the_file_under_data_dir(tmp_path):
    built = build_us_shares_cache(tmp_path / "data")

    assert built is not None
    assert built.path == tmp_path / "data" / CACHE_FILENAME


def test_payload_roundtrip_and_stale_payload_is_still_returned(tmp_path):
    from datetime import timedelta

    cache = UsSharesCache.load(tmp_path / "f.json", ttl_seconds=3600)
    old = datetime.now(timezone.utc) - timedelta(hours=2)
    cache.record_payload("ko", {"eps": 2.5, "industry": "Beverages"}, now=old)
    assert cache.flush()

    reloaded = UsSharesCache.load(tmp_path / "f.json", ttl_seconds=3600)
    fresh, payload = reloaded.lookup_payload("KO")
    assert fresh is False
    assert payload == {"eps": 2.5, "industry": "Beverages"}
    assert reloaded.lookup_payload("MISSING") == (False, None)
