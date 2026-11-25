"""
Run market-structure detection on an exported candle CSV and print what it finds,
so you can compare it with your own manual chart analysis.

Usage:
    python run_structure.py XAUUSD_M5.csv --tf M5 --build M15 H1 H4 D1
    python run_structure.py XAUUSD_M5.csv --tf M5 --last 40      # last 40 events per timeframe

Export from MT5: Tools > History Center (or chart > right click > Save as) as CSV.
"""
import argparse

from analysis.market_structure import analyze, top_down_bias
from market_data.candle_manager import CandleSeries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--tf", required=True, help="timeframe of the CSV, e.g. M5")
    ap.add_argument("--build", nargs="*", default=[], help="higher timeframes to build, e.g. M15 H1 H4 D1")
    ap.add_argument("--last", type=int, default=15, help="events to print per timeframe")
    args = ap.parse_args()

    base = CandleSeries.from_csv(args.csv, args.tf)
    series = {args.tf: base, **{tf: base.resample(tf) for tf in args.build}}
    results = {}
    for tf, s in series.items():
        res = analyze(s)
        results[tf] = res
        print(f"\n=== {tf}: {len(s)} candles, trend {res.trend.value}, "
              f"structural high {res.high}, low {res.low}, events {len(res.events)} ===")
        for ev in res.events[-args.last:]:
            print("  " + ev.describe())
    print("\n" + top_down_bias(results).describe())


if __name__ == "__main__":
    main()
