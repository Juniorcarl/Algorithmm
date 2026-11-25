"""Show the order blocks and breaker blocks for each timeframe.

    python run_ob.py XAUUSD_M5.csv
"""
import argparse

import pandas as pd

from analysis.breaker_blocks import find_breakers
from analysis.fvg import find_moves
from analysis.market_structure import analyze
from analysis.orderblocks import find_order_blocks
from market_data.candle_manager import CandleSeries

MINUTES = {"M15": 15, "H1": 60, "H4": 240}


def build(m5: pd.DataFrame, minutes: int) -> pd.DataFrame:
    out = m5.resample(f"{minutes}min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    data_end = m5.index[-1] + pd.Timedelta(minutes=5)
    return out[out.index + pd.Timedelta(minutes=minutes) <= data_end]


def _bias(structure) -> str:
    trend = getattr(structure, "trend", None)
    if trend is None:
        return "bearish"
    v = getattr(trend, "value", trend)
    return str(v).lower()


def _ob_status(ob) -> str:
    if ob.invalidated_at is not None:
        return f"DEAD {ob.invalidated_at:%d %b %H:%M}"
    if ob.touched_at is not None:
        return f"active (touched {ob.touched_at:%d %b %H:%M})"
    return "active"


def report(tf: str, df: pd.DataFrame, show: int) -> None:
    structure = analyze(df)
    bias = _bias(structure)
    moves = find_moves(df, structure)
    obs = find_order_blocks(df, moves, bias=bias)
    brs = find_breakers(df, obs, structure)
    live = sum(1 for o in obs if o.active)
    print(f"\n=== {tf}: {len(df)} candles, trend={bias}, "
          f"{len(obs)} OBs ({live} active), "
          f"{len(brs)} breakers (last {min(show, len(obs))} shown) ===")
    for ob in obs[-show:]:
        print(f"  OB  {ob.direction:7} {ob.bottom:.2f} - {ob.top:.2f}  "
              f"anchor {ob.anchor_ts:%d %b %H:%M}  known {ob.known_at:%d %b %H:%M}  "
              f"{ob.kind:5}  {_ob_status(ob)}")
    for br in brs[-show:]:
        bits = [f"failed {br.failed_at:%d %b %H:%M}"]
        bits.append(f"armed {br.armed_at:%d %b %H:%M}"
                    if br.armed_at is not None else "not armed")
        if br.retested_at is not None:
            bits.append(f"retested {br.retested_at:%d %b %H:%M}")
        if br.dead_at is not None:
            bits.append(f"DEAD {br.dead_at:%d %b %H:%M}")
        print(f"  BRK {br.direction:7} {br.bottom:.2f} - {br.top:.2f}  "
              + "  ".join(bits))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="M5 candle file")
    p.add_argument("--show", type=int, default=8, help="rows to list per timeframe")
    args = p.parse_args()
    m5 = CandleSeries.from_csv(args.csv, "M5").df
    report("M5", m5, args.show)
    for tf, minutes in MINUTES.items():
        report(tf, build(m5, minutes), args.show)
    print("\nTimes are as in your file (UTC for the downloaded data).")


if __name__ == "__main__":
    main()