# -*- coding: utf-8 -*-
"""Tests for the standalone screening outcome tracker."""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timezone

import pandas as pd
import pytest

from src.services import screening_outcome_tracker as tracker

SESSIONS = pd.bdate_range("2026-11-16", "2026-11-27")  # Mon 16 .. Fri 27


def _prices(closes: list[float], opens: list[float] | None = None) -> pd.DataFrame:
    # default: no overnight gap, each session opens at the previous close
    opens = opens if opens is not None else [closes[0], *closes[:-1]]
    return pd.DataFrame({"Open": opens, "Close": closes}, index=SESSIONS[: len(closes)])


def _make_db(path, runs):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE screening_runs (id INTEGER PRIMARY KEY, run_id TEXT, strategy TEXT, "
                 "market TEXT, created_at TEXT, llm_ranked INTEGER, result_json TEXT)")
    for run_id, strategy, created_at, candidates in runs:
        conn.execute(
            "INSERT INTO screening_runs (run_id, strategy, market, created_at, llm_ranked, result_json) "
            "VALUES (?, ?, 'us', ?, 1, ?)",
            (run_id, strategy, created_at, json.dumps({"strategy_version": "1", "candidates": candidates})),
        )
    conn.commit()
    conn.close()


# -- signal session: no look-ahead ------------------------------------------

def test_run_after_close_uses_that_session():
    # 2026-11-17 22:30 UTC = 17:30 ET, after the close
    sig = tracker.signal_session(datetime(2026, 11, 17, 22, 30), SESSIONS)
    assert sig == pd.Timestamp("2026-11-17")


def test_run_during_market_hours_uses_previous_session():
    # 2026-11-18 15:00 UTC = 10:00 ET, the 18th has not closed yet
    sig = tracker.signal_session(datetime(2026, 11, 18, 15, 0), SESSIONS)
    assert sig == pd.Timestamp("2026-11-17")


def test_entry_is_next_open_not_signal_close():
    closes = [100, 100, 110, 120]
    opens = [100, 100, 105, 115]
    ret = tracker.forward_return(pd.Series(opens, SESSIONS[:4]), pd.Series(closes, SESSIONS[:4]),
                                 pd.Timestamp("2026-11-17"), 1)
    assert ret == pytest.approx((110 / 105 - 1) * 100)  # open 18th -> close 18th


def test_unmatured_horizon_returns_none():
    s = pd.Series([100, 101, 102], SESSIONS[:3])
    assert tracker.forward_return(s, s, pd.Timestamp("2026-11-17"), 5) is None


# -- dedupe -------------------------------------------------------------------

def test_latest_run_per_strategy_and_session_wins():
    def obs(run_id, hour):
        return tracker.Observation(run_id, "s", "1", datetime(2026, 11, 17, hour), "AAA",
                                   1, 80.0, 70.0, 75.0, True)
    pairs = tracker.dedupe_latest([obs("early", 21), obs("late", 23)], SESSIONS)
    assert {o.run_id for o, _ in pairs} == {"late"}


# -- end to end ---------------------------------------------------------------

def test_report_measures_excess_and_score_ic(tmp_path):
    db = tmp_path / "dsa.db"
    # three picks; LLM order matches the outcome, screen order is reversed
    _make_db(db, [("r1", "us_momentum", "2026-11-17 22:30:00", [
        {"code": "WIN", "rank": 1, "llm_score": 90, "screen_score": 60, "score": 80},
        {"code": "MID", "rank": 2, "llm_score": 70, "screen_score": 70, "score": 70},
        {"code": "LOSE", "rank": 3, "llm_score": 50, "screen_score": 80, "score": 60},
    ])])
    prices = {
        "SPY": _prices([100, 100, 101]),       # +1% open->close on the 18th
        "WIN": _prices([10, 10, 10.5]),        # +5%
        "MID": _prices([10, 10, 10.2]),        # +2%
        "LOSE": _prices([10, 10, 9.7]),        # -3%
    }
    report = tracker.run(db, tmp_path / "out", price_loader=lambda tickers, start: prices,
                         now_utc=datetime(2026, 11, 20, 23, 0, tzinfo=timezone.utc))

    row = next(r for r in report["summary"] if r["horizon"] == 1)
    assert row["picks"] == 3
    assert row["mean_excess_pct"] == pytest.approx(((5 - 1) + (2 - 1) + (-3 - 1)) / 3, abs=1e-3)
    assert row["hit_rate_pct"] == pytest.approx(66.7, abs=0.1)
    assert row["ic_llm_score"] == pytest.approx(1.0)
    assert row["ic_screen_score"] == pytest.approx(-1.0)
    assert (tmp_path / "out" / "screening_outcomes.json").exists()
    assert (tmp_path / "out" / "screening_outcomes.csv").exists()


def test_since_filters_old_runs(tmp_path):
    db = tmp_path / "dsa.db"
    _make_db(db, [("old", "s", "2026-11-16 22:30:00", [{"code": "AAA", "rank": 1}]),
                  ("new", "s", "2026-11-18 22:30:00", [{"code": "AAA", "rank": 1}])])
    obs = tracker.load_observations(db, since=date(2026, 11, 18))
    assert [o.run_id for o in obs] == ["new"]


def test_partial_bar_is_not_scored(tmp_path):
    db = tmp_path / "dsa.db"
    _make_db(db, [("r1", "s", "2026-11-17 22:30:00", [{"code": "AAA", "rank": 1, "llm_score": 80}])])
    prices = {"SPY": _prices([100, 100, 101]), "AAA": _prices([10, 10, 11])}
    # 18th at 12:00 ET: the 18th bar exists but is still moving
    report = tracker.run(db, tmp_path / "out", price_loader=lambda t, s: prices,
                         now_utc=datetime(2026, 11, 18, 17, 0, tzinfo=timezone.utc))
    assert report["summary"] == []


def test_missing_benchmark_fails_loudly(tmp_path):
    db = tmp_path / "dsa.db"
    _make_db(db, [("r1", "s", "2026-11-17 22:30:00", [{"code": "AAA", "rank": 1}])])
    with pytest.raises(RuntimeError, match="benchmark"):
        tracker.run(db, tmp_path / "out", price_loader=lambda t, s: {"AAA": _prices([10, 10, 11])},
                    now_utc=datetime(2026, 11, 20, 23, 0, tzinfo=timezone.utc))


def test_empty_database_writes_empty_report(tmp_path):
    db = tmp_path / "dsa.db"
    _make_db(db, [])
    report = tracker.run(db, tmp_path / "out", price_loader=lambda t, s: {})
    assert report["observations"] == 0 and report["summary"] == []
    assert "no matured outcomes" in tracker.format_report(report)
