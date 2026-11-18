# -*- coding: utf-8 -*-
# Not derived from AlphaSift: original daily_stock_analysis code.
"""Forward-return tracker for stored screening runs.

The built-in backtest scores single-stock analysis advice
(``analysis_history``); nothing measures screening picks. This module answers
two questions from data the screener already stores in ``screening_runs``:

* does each strategy (and strategy version) beat SPY after N sessions?
* does the LLM ranking order picks better than the plain screen score?

It is deliberately standalone: it only *reads* ``screening_runs`` with plain
sqlite3, fetches prices itself, keeps no state and writes only its own report
files. Nothing else in the application imports it.

Conventions, chosen to avoid look-ahead:

* signal session = the last NYSE session whose close (16:00 ET) precedes the
  run's ``created_at``; a run made during market hours uses yesterday's data;
* entry = next session's open, exit = close ``h`` sessions after entry, so a
  1-session return is open-to-close of the day after the signal;
* several runs of one strategy on one signal session count once (the latest);
* SPY over the same window is the benchmark; prices are split/dividend
  adjusted.

Run: ``python -m src.services.screening_outcome_tracker --since 2026-11-18``
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional
from zoneinfo import ZoneInfo

import pandas as pd

logger = logging.getLogger(__name__)

NY = ZoneInfo("America/New_York")
BENCHMARK = "SPY"
HORIZONS = (1, 5, 20)
SCORES = ("llm_score", "screen_score", "final_score")
_MARKET_CLOSE = time(16, 0)
_MIN_IC_PAIRS = 3


@dataclass
class Observation:
    run_id: str
    strategy: str
    strategy_version: str
    created_at: datetime  # naive UTC, as stored
    code: str
    rank: Optional[int]
    llm_score: Optional[float]
    screen_score: Optional[float]
    final_score: Optional[float]
    llm_ranked: bool


# ---------------------------------------------------------------- loading

def _num(value) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None  # drop NaN


def load_observations(db_path: str | Path, *, market: str = "us",
                      since: Optional[date] = None) -> list[Observation]:
    """Read picks from ``screening_runs`` without touching the ORM."""
    uri = f"file:{Path(db_path)}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        rows = conn.execute(
            "SELECT run_id, strategy, created_at, llm_ranked, result_json "
            "FROM screening_runs WHERE market = ? ORDER BY created_at",
            (market,),
        ).fetchall()
    out: list[Observation] = []
    for run_id, strategy, created_at, llm_ranked, result_json in rows:
        created = datetime.fromisoformat(str(created_at))
        if since and created.date() < since:
            continue
        try:
            result = json.loads(result_json or "{}")
        except ValueError:
            continue
        version = str(result.get("strategy_version") or "")
        for cand in result.get("candidates") or []:
            code = str(cand.get("code") or "").strip().upper()
            if not code:
                continue
            rank = cand.get("rank")
            out.append(Observation(
                run_id=str(run_id), strategy=str(strategy), strategy_version=version,
                created_at=created, code=code,
                rank=int(rank) if isinstance(rank, (int, float)) else None,
                llm_score=_num(cand.get("llm_score")),
                screen_score=_num(cand.get("screen_score")),
                final_score=_num(cand.get("score")),
                llm_ranked=bool(llm_ranked),
            ))
    return out


# --------------------------------------------------------------- sessions

def signal_session(created_at_utc: datetime, sessions: pd.DatetimeIndex) -> Optional[pd.Timestamp]:
    """Last session whose close is at or before the run time."""
    ny = created_at_utc.replace(tzinfo=timezone.utc).astimezone(NY)
    day = pd.Timestamp(ny.date())
    eligible = sessions[sessions <= day] if ny.time() >= _MARKET_CLOSE else sessions[sessions < day]
    return eligible[-1] if len(eligible) else None


def dedupe_latest(observations: Iterable[Observation],
                  sessions: pd.DatetimeIndex) -> list[tuple[Observation, pd.Timestamp]]:
    """Keep only the latest run per (strategy, signal session)."""
    latest_run: dict[tuple[str, pd.Timestamp], tuple[datetime, str]] = {}
    tagged: list[tuple[Observation, pd.Timestamp]] = []
    for obs in observations:
        sig = signal_session(obs.created_at, sessions)
        if sig is None:
            continue
        tagged.append((obs, sig))
        key = (obs.strategy, sig)
        if key not in latest_run or obs.created_at > latest_run[key][0]:
            latest_run[key] = (obs.created_at, obs.run_id)
    return [(o, s) for o, s in tagged if latest_run[(o.strategy, s)][1] == o.run_id]


# ---------------------------------------------------------------- returns

def forward_return(opens: pd.Series, closes: pd.Series, signal: pd.Timestamp,
                   horizon: int) -> Optional[float]:
    """Next-open to close ``horizon`` sessions later, in percent."""
    index = closes.index
    after = index[index > signal]
    if len(after) < horizon:
        return None  # not matured yet
    entry_day, exit_day = after[0], after[horizon - 1]
    entry, exit_ = _num(opens.get(entry_day)), _num(closes.get(exit_day))
    if not entry or exit_ is None or entry <= 0:
        return None
    return (exit_ / entry - 1.0) * 100.0


def build_outcomes(pairs: list[tuple[Observation, pd.Timestamp]], prices: dict[str, pd.DataFrame],
                   horizons: Iterable[int] = HORIZONS) -> pd.DataFrame:
    """One row per (pick, horizon) with return and excess over the benchmark."""
    bench = prices.get(BENCHMARK)
    rows = []
    for obs, sig in pairs:
        frame = prices.get(obs.code)
        if frame is None or bench is None:
            continue
        for h in horizons:
            ret = forward_return(frame["Open"], frame["Close"], sig, h)
            bench_ret = forward_return(bench["Open"], bench["Close"], sig, h)
            if ret is None or bench_ret is None:
                continue
            row = asdict(obs)
            row.update(signal_session=sig.date().isoformat(), horizon=h,
                       return_pct=ret, excess_pct=ret - bench_ret)
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- metrics

def _mean_rank_ic(group: pd.DataFrame, score: str) -> tuple[Optional[float], int]:
    """Average per-run Spearman correlation between a score and excess return."""
    ics = []
    for _, run in group.groupby("run_id"):
        valid = run[[score, "excess_pct"]].dropna()
        if len(valid) < _MIN_IC_PAIRS or valid[score].nunique() < 2 or valid["excess_pct"].nunique() < 2:
            continue
        ics.append(valid[score].rank().corr(valid["excess_pct"].rank()))
    return (float(sum(ics) / len(ics)), len(ics)) if ics else (None, 0)


def summarize(outcomes: pd.DataFrame) -> list[dict]:
    """Per strategy/version/horizon: hit rate, mean excess and rank IC per score."""
    if outcomes.empty:
        return []
    summary = []
    keys = ["strategy", "strategy_version", "horizon"]
    for (strategy, version, horizon), group in outcomes.groupby(keys):
        row = {
            "strategy": strategy, "strategy_version": version, "horizon": int(horizon),
            "picks": int(len(group)), "runs": int(group["run_id"].nunique()),
            "mean_return_pct": round(float(group["return_pct"].mean()), 3),
            "mean_excess_pct": round(float(group["excess_pct"].mean()), 3),
            "median_excess_pct": round(float(group["excess_pct"].median()), 3),
            "hit_rate_pct": round(float((group["excess_pct"] > 0).mean() * 100), 1),
        }
        for score in SCORES:
            ic, n = _mean_rank_ic(group, score)
            row[f"ic_{score}"] = None if ic is None else round(ic, 3)
            row[f"ic_{score}_runs"] = n
        summary.append(row)
    return summary


# ----------------------------------------------------------------- prices

def fetch_prices(tickers: list[str], start: date) -> dict[str, pd.DataFrame]:
    """Adjusted daily Open/Close per ticker from one bulk yfinance call."""
    import yfinance as yf

    data = yf.download(sorted(set(tickers)), start=start.isoformat(), auto_adjust=True,
                       group_by="ticker", progress=False, threads=True)
    out: dict[str, pd.DataFrame] = {}
    for ticker in set(tickers):
        try:
            frame = data[ticker] if isinstance(data.columns, pd.MultiIndex) else data
        except KeyError:
            continue
        frame = frame[["Open", "Close"]].dropna()
        if not frame.empty:
            frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
            out[ticker] = frame
    return out


def _drop_unfinished_session(prices: dict[str, pd.DataFrame], now_utc: datetime) -> None:
    """Today's bar is partial until the close; never score against it."""
    ny_now = now_utc.astimezone(NY)
    if ny_now.time() >= time(16, 15):
        return
    today = pd.Timestamp(ny_now.date())
    for ticker, frame in prices.items():
        prices[ticker] = frame[frame.index < today]


# -------------------------------------------------------------------- run

def run(db_path: str | Path, out_dir: str | Path, *, since: Optional[date] = None,
        price_loader: Callable[[list[str], date], dict[str, pd.DataFrame]] = fetch_prices,
        now_utc: Optional[datetime] = None) -> dict:
    now_utc = now_utc or datetime.now(timezone.utc)
    observations = load_observations(db_path, since=since)
    report: dict = {"generated_at": now_utc.isoformat(),
                    "since": since.isoformat() if since else None,
                    "observations": len(observations), "summary": []}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if observations:
        start = min(o.created_at for o in observations).date() - timedelta(days=10)
        prices = price_loader([o.code for o in observations] + [BENCHMARK], start)
        _drop_unfinished_session(prices, now_utc)
        bench = prices.get(BENCHMARK)
        if bench is None or bench.empty:
            raise RuntimeError("benchmark prices unavailable; refusing to report without SPY")
        pairs = dedupe_latest(observations, bench.index)
        outcomes = build_outcomes(pairs, prices)
        report.update(deduped_picks=len(pairs), summary=summarize(outcomes),
                      missing_prices=sorted({o.code for o, _ in pairs if o.code not in prices}))
        if not outcomes.empty:
            outcomes.to_csv(out / "screening_outcomes.csv", index=False)
    (out / "screening_outcomes.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def format_report(report: dict) -> str:
    lines = [f"observations={report['observations']} deduped={report.get('deduped_picks', 0)} "
             f"since={report.get('since')}"]
    if not report["summary"]:
        lines.append("no matured outcomes yet (1-session returns need one session after the signal)")
        return "\n".join(lines)
    header = f"{'strategy':24} {'v':>2} {'h':>3} {'picks':>5} {'runs':>4} {'excess%':>8} {'hit%':>6} " \
             f"{'IC llm':>7} {'IC screen':>9} {'IC final':>8}"
    lines.append(header)

    def fmt(value):
        return "   -" if value is None else f"{value:+.2f}"

    for row in report["summary"]:
        lines.append(
            f"{row['strategy']:24} {row['strategy_version']:>2} {row['horizon']:>3} {row['picks']:>5} "
            f"{row['runs']:>4} {row['mean_excess_pct']:>+8.2f} {row['hit_rate_pct']:>6.1f} "
            f"{fmt(row['ic_llm_score']):>7} {fmt(row['ic_screen_score']):>9} {fmt(row['ic_final_score']):>8}"
        )
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--db", default="data/stock_analysis.db")
    parser.add_argument("--out", default="data/screening/outcomes")
    parser.add_argument("--since", type=date.fromisoformat, default=None,
                        help="ignore runs created before this UTC date (YYYY-MM-DD)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    report = run(args.db, args.out, since=args.since)
    print(format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
