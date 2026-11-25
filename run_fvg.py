"""Show the gap that matters for each recent BOS / CHoCH move.

    python run_fvg.py XAUUSD_M5.csv
"""
import argparse

import pandas as pd

from analysis.fvg import find_moves
from analysis.market_structure import analyze
from market_data.candle_manager import CandleSeries

MINUTES = {"M15": 15, "H1": 60, "H4": 240}


def build(m5: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Build a higher timeframe from M5, dropping a still-forming last candle."""
    out = m5.resample(f"{minutes}min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    data_end = m5.index[-1] + pd.Timedelta(minutes=5)
    return out[out.index + pd.Timedelta(minutes=minutes) <= data_end]


def report(tf: str, df: pd.DataFrame, show: int) -> None:
    moves = find_moves(df, analyze(df))
    now = len(df) - 1
    print(f"\n=== {tf}: {len(df)} candles, {len(moves)} moves, last {min(show, len(moves))} shown ===")
    for m in moves[-show:]:
        ev, gap = m.event, m.selected_at(now)
        head = f"  {ev.kind:5} {ev.direction:4} confirmed {ev.ts}  ->"
        if gap is None:
            why = "no gap in the move" if not m.gaps else f"all {len(m.gaps)} gap(s) dead"
            print(f"{head} no gap ({why})")
            continue
        place = m.gaps.index(gap) + 1
        state = "touched" if gap.touched_at(now) else "untouched"
        print(f"{head} gap {gap.bottom:.2f} - {gap.top:.2f}, middle candle {gap.ts}, "
              f"#{place} of {len(m.gaps)}, {state}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="M5 candle file")
    p.add_argument("--show", type=int, default=8, help="moves to list per timeframe")
    args = p.parse_args()
    m5 = CandleSeries.from_csv(args.csv, "M5").df
    report("M5", m5, args.show)
    for tf, minutes in MINUTES.items():
        report(tf, build(m5, minutes), args.show)
    print("\nTimes are as in your file (UTC for the downloaded data).")


if __name__ == "__main__":
    main()