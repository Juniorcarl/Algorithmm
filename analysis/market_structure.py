"""
market_structure.py - Module 1b of KYRO | TRADER

Purpose
-------
Read a CandleSeries bar by bar and report: the current trend, the structural
high/low, and every BOS / CHoCH event, using YOUR confirmed rules.

Confirmed rules implemented (rules register v0.1)
-------------------------------------------------
* BOS (= BMS): `closes_required` (3) consecutive candle CLOSES beyond a
  structural high/low. Fewer than 3 closes is not a break.
* Failed break: when price goes past a structural high/low and a candle closes
  back inside before 3 closes, the level moves to the furthest wick of that
  attempt. Applies to the continuation level AND the CHoCH level (HL/LH).
  A new BOS / CHoCH then needs 3 closes beyond the new level.
* CHoCH: 3 consecutive closes beyond the HL (in an uptrend) or the LH (in a
  downtrend) - the trend stops forming.
* Bullish = HH + HL, bearish = LL + LH. The HL is the pullback low that exists
  before the break (mirror for bearish).
* Closed candles only, strictly causal: the result for candle i never depends
  on candles after i (see tests/test_market_structure.py).

ASSUMPTIONS I made (not from you - please review)
-------------------------------------------------
M1. To know "a pullback happened" I use a minimal pivot (`pivot_strength`=2):
    a bar whose high/low is beaten by none of the 2 bars on each side. This is
    used ONLY to detect that a pullback exists. Labelling HH/HL/LH/LL and BOS /
    CHoCH is governed purely by the 3-close rule.
M2. After a BOS up, the structural high keeps rising with price ("extending")
    until a pullback low is confirmed after it. During extension no new BOS
    can fire, otherwise a strong run would produce a BOS every 3 candles.
M3. After the pullback, the structural high stops following price. It only
    moves through the failed-break rule above.
M4. At the very start (no trend yet), the reference levels are the highest /
    lowest confirmed pivots seen so far; the first 3-close break sets the
    trend and is flagged `initial=True`.
M5. After a CHoCH down, the LH reference is the HH that preceded the break
    (mirror for CHoCH up uses the last LL).
M6. After a high has moved to a sweep wick, the HL of the next BOS is still the
    lowest low since the ORIGINAL swing high, so a deep pullback made before the
    sweep is not lost (mirror for lows / LH).
M7. The failed-break rule also moves the starting reference levels of M4.

Connects to the rest of the system
----------------------------------
    CandleSeries -> analyze() -> StructureResult -> top_down_bias()
    Later modules (order blocks, setup validator) read the events and the
    trend per bar from the result.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from market_data.candle_manager import CandleSeries


class Trend(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    UNDEFINED = "UNDEFINED"


@dataclass(frozen=True)
class SwingPoint:
    kind: str          # "HH" | "HL" | "LH" | "LL"
    price: float
    idx: int
    ts: pd.Timestamp


@dataclass(frozen=True)
class StructureEvent:
    kind: str                 # "BOS" | "CHOCH"
    direction: str            # "UP" | "DOWN"
    level: float              # the high/low that was broken
    level_idx: int            # candle index that made that level
    break_start_idx: int      # first of the closing candles beyond the level
    confirm_idx: int          # candle where the Nth close happened (usable from here on)
    ts: pd.Timestamp          # close time reference (candle open time of confirm candle)
    new_trend: Trend
    swing: Optional[SwingPoint] = None   # HL/LH (BOS) or the origin HH/LL (CHoCH)
    initial: bool = False

    def describe(self) -> str:
        tag = " (initial)" if self.initial else ""
        sw = f", {self.swing.kind} {self.swing.price:.2f}" if self.swing else ""
        return (f"{self.ts}  {self.kind} {self.direction}{tag}: broke {self.level:.2f} "
                f"(set at bar {self.level_idx}), confirmed bar {self.confirm_idx}{sw} -> {self.new_trend.value}")


@dataclass
class StructureResult:
    trend: Trend = Trend.UNDEFINED
    high: Optional[float] = None       # current structural high (HH, or LH in a downtrend)
    high_idx: Optional[int] = None
    low: Optional[float] = None        # current structural low (HL, or LL in a downtrend)
    low_idx: Optional[int] = None
    events: List[StructureEvent] = field(default_factory=list)
    swings: List[SwingPoint] = field(default_factory=list)
    trend_by_bar: List[Trend] = field(default_factory=list)
    timeframe: Optional[str] = None

    def trend_at(self, idx: int) -> Trend:
        return self.trend_by_bar[idx]


@dataclass
class _Level:
    """A structural level that moves to the wick of a failed break."""
    price: float
    idx: int                        # candle that currently sets the price
    anchor: int                     # candle of the original swing (M6)
    probe_price: Optional[float] = None   # furthest point of an attempt still open
    probe_idx: Optional[int] = None

    @classmethod
    def at(cls, price: float, idx: int) -> "_Level":
        return cls(price, idx, idx)

    def reset_probe(self) -> None:
        self.probe_price = self.probe_idx = None

    def update(self, extreme: float, close: float, i: int, upper: bool) -> None:
        """Call once per closed candle that did not confirm a BOS / CHoCH.
        upper=True for a structural high, False for a structural low."""
        sign = 1 if upper else -1
        furthest = self.price if self.probe_price is None else self.probe_price
        if sign * (extreme - furthest) > 0:
            self.probe_price, self.probe_idx = extreme, i
        closed_back_inside = sign * (close - self.price) <= 0
        if closed_back_inside and self.probe_price is not None:
            self.price, self.idx = self.probe_price, self.probe_idx
            self.reset_probe()


def analyze(series: Union[CandleSeries, pd.DataFrame], closes_required: int = 3,
            pivot_strength: int = 2) -> StructureResult:
    df = series.df if isinstance(series, CandleSeries) else series
    tf = series.timeframe if isinstance(series, CandleSeries) else None
    idx_ts = df.index
    hi = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    cl = df["close"].to_numpy(dtype=float)
    n, N, k = len(df), closes_required, pivot_strength

    res = StructureResult(timeframe=tf)
    trend = Trend.UNDEFINED
    H: Optional[_Level] = None           # structural high (reference high while UNDEFINED)
    L: Optional[_Level] = None           # structural low  (reference low while UNDEFINED)
    ext_high = ext_low = False           # structural high/low still extending (no pullback yet)

    def closes_above(level: _Level, i: int) -> bool:
        return i - N + 1 >= 0 and bool(np.all(cl[i - N + 1:i + 1] > level.price))

    def closes_below(level: _Level, i: int) -> bool:
        return i - N + 1 >= 0 and bool(np.all(cl[i - N + 1:i + 1] < level.price))

    for i in range(n):
        # 1) a structural high/low that is still "extending" follows price (M2)
        if trend is Trend.BULLISH and ext_high and hi[i] > H.price:
            H = _Level.at(hi[i], i)
        if trend is Trend.BEARISH and ext_low and lo[i] < L.price:
            L = _Level.at(lo[i], i)

        # 2) causal pivot confirmation: pivot at j is only known k bars later (M1)
        j = i - k
        if j >= k:
            piv_h = hi[j] > hi[j - k:j].max() and hi[j] > hi[j + 1:j + k + 1].max()
            piv_l = lo[j] < lo[j - k:j].min() and lo[j] < lo[j + 1:j + k + 1].min()
            if trend is Trend.UNDEFINED:
                if piv_h and (H is None or hi[j] > H.price):
                    H = _Level.at(hi[j], j)
                if piv_l and (L is None or lo[j] < L.price):
                    L = _Level.at(lo[j], j)
            elif trend is Trend.BULLISH and ext_high and piv_l and j > H.idx:
                ext_high = False         # pullback confirmed -> stop following price (M3)
            elif trend is Trend.BEARISH and ext_low and piv_h and j > L.idx:
                ext_low = False

        # 3) 3-close break detection
        ev = None   # (kind, direction, broken level, initial)
        if trend is Trend.UNDEFINED:
            if H and closes_above(H, i):
                ev = ("BOS", "UP", H, True)
            elif L and closes_below(L, i):
                ev = ("BOS", "DOWN", L, True)
        elif trend is Trend.BULLISH:
            if closes_below(L, i):
                ev = ("CHOCH", "DOWN", L, False)
            elif not ext_high and closes_above(H, i):
                ev = ("BOS", "UP", H, False)
        else:  # BEARISH
            if closes_above(H, i):
                ev = ("CHOCH", "UP", H, False)
            elif not ext_low and closes_below(L, i):
                ev = ("BOS", "DOWN", L, False)

        # 4) no break on this candle: failed attempts move the level to the wick
        if ev is None:
            if H is not None and not ext_high:
                H.update(hi[i], cl[i], i, upper=True)
            if L is not None and not ext_low:
                L.update(lo[i], cl[i], i, upper=False)

        # 5) apply the BOS / CHoCH
        else:
            kind, direction, broken, initial = ev
            s = i - N + 1
            if direction == "UP":
                if kind == "BOS":
                    a = broken.anchor + 1
                    p = a + int(np.argmin(lo[a:i + 1]))
                    L = _Level.at(lo[p], p)
                    swing = SwingPoint("HL", float(lo[p]), p, idx_ts[p])
                else:  # CHoCH up: the trough that started the rally is the reference low (M5)
                    L.reset_probe()
                    swing = SwingPoint("LL", float(L.price), L.idx, idx_ts[L.idx])
                q = s + int(np.argmax(hi[s:i + 1]))
                H = _Level.at(hi[q], q)
                ext_high, ext_low, trend = True, False, Trend.BULLISH
            else:
                if kind == "BOS":
                    a = broken.anchor + 1
                    p = a + int(np.argmax(hi[a:i + 1]))
                    H = _Level.at(hi[p], p)
                    swing = SwingPoint("LH", float(hi[p]), p, idx_ts[p])
                else:  # CHoCH down: the peak that started the drop is the reference high (M5)
                    H.reset_probe()
                    swing = SwingPoint("HH", float(H.price), H.idx, idx_ts[H.idx])
                q = s + int(np.argmin(lo[s:i + 1]))
                L = _Level.at(lo[q], q)
                ext_low, ext_high, trend = True, False, Trend.BEARISH

            res.events.append(StructureEvent(kind, direction, float(broken.price), int(broken.idx),
                                             int(s), i, idx_ts[i], trend, swing, initial))
            res.swings.append(swing)

        res.trend_by_bar.append(trend)

    res.trend = trend
    if H is not None:
        res.high, res.high_idx = float(H.price), H.idx
    if L is not None:
        res.low, res.low_idx = float(L.price), L.idx
    return res


# --------------------------------------------------------------------------- bias
TF_ORDER = ["MN", "W1", "D1", "H4", "H1", "M30", "M15", "M5", "M1"]


@dataclass
class TopDownBias:
    bias: Trend                          # trend of the highest timeframe with a defined trend
    anchor_tf: Optional[str]
    by_tf: Dict[str, Trend]
    aligned: bool                        # every defined lower timeframe agrees with the bias
    counter_tfs: List[str]               # lower timeframes currently against the bias

    def describe(self) -> str:
        parts = ", ".join(f"{tf}:{t.value}" for tf, t in self.by_tf.items())
        state = "aligned" if self.aligned else f"counter on {self.counter_tfs}"
        return f"Bias {self.bias.value} (anchor {self.anchor_tf}); {parts}; {state}"


def top_down_bias(results: Dict[str, StructureResult]) -> TopDownBias:
    """Higher timeframe has priority (rule). Lower timeframes that run against the
    bias are reported in `counter_tfs`; whether such a counter-move is a valid
    scalp INTO an HTF POI is decided later by the setup validator (A4)."""
    ordered = {tf: results[tf].trend for tf in TF_ORDER if tf in results}
    anchor = next((tf for tf, t in ordered.items() if t is not Trend.UNDEFINED), None)
    if anchor is None:
        return TopDownBias(Trend.UNDEFINED, None, ordered, False, [])
    bias = ordered[anchor]
    lower = list(ordered)[list(ordered).index(anchor) + 1:]
    counter = [tf for tf in lower if ordered[tf] not in (bias, Trend.UNDEFINED)]
    return TopDownBias(bias, anchor, ordered, not counter, counter)