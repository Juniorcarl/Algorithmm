"""
Download free XAUUSD 5-minute candles (last ~60 days) from Yahoo Finance and save
them as a CSV that run_structure.py can read.

Usage:
    pip install yfinance
    python download_data.py                      # XAUUSD=X -> XAUUSD_M5.csv
    python download_data.py GC=F gold_futures.csv

Notes:
* Yahoo only serves about 60 days of 5-minute data.
* Prices are close to, but not identical to, your broker's chart.
* Times are saved in UTC.
"""
import sys

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("yfinance is not installed. Run:  pip install yfinance")
    sys.exit(1)


def main() -> None:
    symbol = sys.argv[1] if len(sys.argv) > 1 else "XAUUSD=X"
    out = sys.argv[2] if len(sys.argv) > 2 else "XAUUSD_M5.csv"

    print(f"Downloading {symbol} 5m candles (last 60 days)...")
    df = yf.download(symbol, period="60d", interval="5m", auto_adjust=False, progress=False)
    if df is None or df.empty:
        print("No data returned. Check your internet connection or try:  python download_data.py GC=F")
        sys.exit(1)

    if isinstance(df.columns, pd.MultiIndex):          # newer yfinance returns two-level columns
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]

    if df.index.tz is not None:                         # store as plain UTC time
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df.index.name = "time"

    df = df.dropna()
    df = df[~df.index.duplicated(keep="first")].sort_index()
    # make sure high/low really bound open/close (some feeds have tiny inconsistencies)
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)

    df.to_csv(out)
    print(f"Saved {len(df)} candles to {out}  ({df.index[0]} -> {df.index[-1]} UTC)")
    print("Next:  python run_structure.py " + out + " --tf M5 --build M15 H1 H4")


if __name__ == "__main__":
    main()