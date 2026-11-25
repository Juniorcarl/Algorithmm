"""
candle_manager.py - Module 1a of KYRO | TRADER

Purpose
-------
Hold OHLC candle data in one validated, consistent shape and build higher
timeframes from lower ones. Every analysis module (market structure, FVG,
order blocks, liquidity ...) reads candles from a CandleSeries.

Connects to the rest of the system
----------------------------------
    MT5 export / future mt5_connector.py  ->  CandleSeries  ->  analysis/*

Design rules
------------
* Closed candles only. `closed_only()` removes a still-forming candle so no
  rule is ever evaluated on unfinished data (no look-ahead).
* Index is a strictly increasing DatetimeIndex; columns are
  open/high/low/close/volume.
* Weekly/monthly resampling and the daily candle boundary depend on the broker
  time zone. The chart time zone is still an OPEN ITEM in the rules register,
  so for W1/MN prefer exporting those candles straight from MT5.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Union

import pandas as pd

# pandas offset aliases used when building higher timeframes
TIMEFRAMES = {
    "M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min",
    "H1": "60min", "H4": "240min", "D1": "1D",
    "W1": "W-MON",   # weeks start Monday (see module note about time zones)
    "MN": "MS",      # months start on the 1st
}
TF_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}
OHLCV = ["open", "high", "low", "close", "volume"]


class CandleError(ValueError):
    """Raised when candle data is malformed."""


@dataclass(frozen=True)
class Candle:
    ts: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @property
    def bullish(self) -> bool:
        return self.close > self.open

    @property
    def bearish(self) -> bool:
        return self.close < self.open


class CandleSeries:
    """A validated, time-ordered set of candles for one symbol and timeframe."""

    def __init__(self, df: pd.DataFrame, timeframe: Optional[str] = None, validate: bool = True):
        df = df.copy()
        df.columns = [str(c).lower() for c in df.columns]
        if "volume" not in df.columns:
            df["volume"] = 0.0
        if timeframe is not None and timeframe not in TIMEFRAMES:
            raise CandleError(f"Unknown timeframe {timeframe!r}; use one of {list(TIMEFRAMES)}")
        self.timeframe = timeframe
        self._df = df[OHLCV] if all(c in df.columns for c in OHLCV) else df
        if validate:
            self._validate()

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_csv(cls, path: Union[str, Path], timeframe: Optional[str] = None) -> "CandleSeries":
        """Load a CSV. Accepts MT5 exports (<DATE>/<TIME>/<OPEN>..., tab or comma)
        and plain files with a time/datetime/timestamp column."""
        raw = pd.read_csv(path, sep=None, engine="python")
        raw.columns = [str(c).strip().strip("<>").lower() for c in raw.columns]
        if "date" in raw.columns and "time" in raw.columns:
            ts = pd.to_datetime(raw["date"].astype(str) + " " + raw["time"].astype(str))
        else:
            tcol = next((c for c in ("datetime", "timestamp", "time", "date") if c in raw.columns), None)
            if tcol is None:
                raise CandleError("CSV needs a date/time column")
            ts = pd.to_datetime(raw[tcol])
        vcol = next((c for c in ("tick_volume", "tickvol", "volume", "vol") if c in raw.columns), None)
        df = pd.DataFrame(
            {
                "open": raw["open"], "high": raw["high"], "low": raw["low"], "close": raw["close"],
                "volume": raw[vcol] if vcol else 0.0,
            }
        )
        df.index = pd.DatetimeIndex(ts, name="ts")
        return cls(df.sort_index(), timeframe)

    # --------------------------------------------------------------- validation
    def _validate(self) -> None:
        df = self._df
        if not isinstance(df.index, pd.DatetimeIndex):
            raise CandleError("Index must be a DatetimeIndex")
        missing = [c for c in OHLCV if c not in df.columns]
        if missing:
            raise CandleError(f"Missing columns: {missing}")
        if len(df) == 0:
            return
        if df[["open", "high", "low", "close"]].isna().any().any():
            raise CandleError("NaN values in OHLC")
        if not df.index.is_monotonic_increasing or df.index.has_duplicates:
            raise CandleError("Timestamps must be strictly increasing and unique")
        bad_high = df["high"] < df[["open", "close", "low"]].max(axis=1)
        bad_low = df["low"] > df[["open", "close", "high"]].min(axis=1)
        if bad_high.any() or bad_low.any():
            first = df.index[(bad_high | bad_low).to_numpy().argmax()]
            raise CandleError(f"Inconsistent OHLC (high/low do not bound open/close) at {first}")

    # ---------------------------------------------------------------- accessors
    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return CandleSeries(self._df.iloc[item], self.timeframe, validate=False)
        r = self._df.iloc[item]
        return Candle(self._df.index[item], r.open, r.high, r.low, r.close, r.volume)

    def candles(self) -> Iterator[Candle]:
        for ts, r in self._df.iterrows():
            yield Candle(ts, r.open, r.high, r.low, r.close, r.volume)

    # -------------------------------------------------------------- operations
    def resample(self, timeframe: str) -> "CandleSeries":
        """Build a higher timeframe from this (lower) one."""
        if timeframe not in TIMEFRAMES:
            raise CandleError(f"Unknown timeframe {timeframe!r}")
        if self.timeframe in TF_MINUTES and timeframe in TF_MINUTES:
            if TF_MINUTES[timeframe] < TF_MINUTES[self.timeframe]:
                raise CandleError(f"Cannot build {timeframe} from {self.timeframe} (lower timeframe)")
        agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        out = self._df.resample(TIMEFRAMES[timeframe], closed="left", label="left").agg(agg)
        out = out.dropna(subset=["open"])
        return CandleSeries(out, timeframe)

    def closed_only(self, now: pd.Timestamp) -> "CandleSeries":
        """Drop the candle that is still forming at `now` (anti look-ahead)."""
        if self.timeframe not in TF_MINUTES:
            raise CandleError("closed_only needs a timeframe from M1..D1")
        close_time = self._df.index + pd.Timedelta(minutes=TF_MINUTES[self.timeframe])
        return CandleSeries(self._df[close_time <= pd.Timestamp(now)], self.timeframe, validate=False)
