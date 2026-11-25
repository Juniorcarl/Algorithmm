"""orderblocks.py - Module 3a of KYRO | TRADER

Find the Order Block (OB) left behind by each confirmed BOS / CHoCH and
follow it until it dies.

Rules (register v0.2 - Q1 A, Q2 A, Q3, Q4 C, A5-A8)
---------------------------------------------------
* Both BOS and CHoCH create an OB.
* Anchor: the leg origin used by the FVG module (not the broken level).
  Bullish move -> the last BEARISH candle at, or up to MAX_STEP_BACK candles
  before, that low. Bearish move -> the mirror.
* Zone: the anchor candle's full range, wick included.
* Known only on the confirming candle of the break (no look-ahead).
* Dead when a candle CLOSES beyond the zone: below bottom (bullish) or above
  top (bearish). A wick through - even the whole zone - leaves it valid.
  touched_at is informational only.
* Status is evaluated only after confirmation (A8).
* bias: if "bullish" / "bearish", only OBs in that direction are returned.
"""
import warnings
from dataclasses import dataclass
from numbers import Integral
from typing import List, Optional

import pandas as pd

MAX_STEP_BACK = 3
_WARNED = set()

_KNOWN_INDEX_FIELDS = (
    "origin_idx", "origin_index", "origin_i", "origin",
    "start_idx", "start_index", "start_i", "start_pos",
    "leg_start_index", "leg_start_i",
    "from_index", "from_i",
)
_KNOWN_TS_FIELDS = (
    "origin_ts", "origin_time",
    "start_ts", "start_time", "start_dt",
    "leg_start_ts", "leg_start_time",
    "from_ts", "from_time",
)


@dataclass
class OrderBlock:
    direction: str
    top: float
    bottom: float
    anchor_ts: pd.Timestamp
    known_at: pd.Timestamp
    kind: str
    invalidated_at: Optional[pd.Timestamp] = None
    touched_at: Optional[pd.Timestamp] = None

    @property
    def active(self) -> bool:
        return self.invalidated_at is None


def _move_items(move):
    try:
        return list(vars(move).items())
    except TypeError:
        pass
    fields = getattr(move, "_fields", None)
    if fields:
        return [(f, getattr(move, f)) for f in fields]
    return [
        (a, getattr(move, a, None))
        for a in dir(move)
        if not a.startswith("_")
        and not callable(getattr(move, a, None))
    ]


def _as_index(df: pd.DataFrame, v, conf_i: int) -> Optional[int]:
    if isinstance(v, bool):
        return None
    if isinstance(v, Integral):
        iv = int(v)
        return iv if 0 <= iv < conf_i else None
    if isinstance(v, pd.Timestamp):
        iv = int(df.index.get_indexer([v])[0])
        return iv if 0 <= iv < conf_i else None
    return None


def _origin_index(df: pd.DataFrame, move) -> Optional[int]:
    conf_ts = pd.Timestamp(move.event.ts)
    conf_i = int(df.index.get_indexer([conf_ts])[0])
    if conf_i < 0:
        return None

    for name in _KNOWN_INDEX_FIELDS:
        v = getattr(move, name, None)
        iv = _as_index(df, v, conf_i)
        if iv is not None:
            return iv

    for name in _KNOWN_TS_FIELDS:
        v = getattr(move, name, None)
        if v is None:
            continue
        try:
            ts = pd.Timestamp(v)
        except Exception:
            continue
        iv = _as_index(df, ts, conf_i)
        if iv is not None:
            return iv

    candidates = []
    for name, v in _move_items(move):
        if name == "event":
            continue
        iv = _as_index(df, v, conf_i)
        if iv is not None:
            candidates.append(iv)
            continue
        if isinstance(v, (tuple, list)) and len(v) == 2:
            for elem in v:
                iv = _as_index(df, elem, conf_i)
                if iv is not None:
                    candidates.append(iv)
    if candidates:
        return min(candidates)

    if "fallback" not in _WARNED:
        _WARNED.add("fallback")
        warnings.warn(
            "Move has fields %s. None of them look like a start-of-leg "
            "index or timestamp, so the OB anchor uses the prefix extreme "
            "and may sit one candle off. Paste the Move class from "
            "analysis/fvg.py so this can be pinned down."
            % sorted(n for n, _ in _move_items(move)),
            RuntimeWarning)

    direction = str(move.event.direction).lower()
    prefix = df.iloc[: conf_i + 1]
    if direction == "up":
        return int(prefix["low"].to_numpy().argmin())
    return int(prefix["high"].to_numpy().argmax())


def _anchor_index(df: pd.DataFrame, start_i: int, direction: str) -> Optional[int]:
    want_bearish = direction == "up"
    i = start_i
    for _ in range(MAX_STEP_BACK + 1):
        if i < 0:
            return None
        row = df.iloc[i]
        c, o = float(row["close"]), float(row["open"])
        if want_bearish and c < o:
            return i
        if (not want_bearish) and c > o:
            return i
        i -= 1
    return None


def _follow(df: pd.DataFrame, ob: OrderBlock, known_i: int) -> None:
    for i in range(known_i + 1, len(df)):
        row = df.iloc[i]
        ts = df.index[i]
        if ob.direction == "bullish":
            if float(row["close"]) < ob.bottom:
                ob.invalidated_at = ts
                return
            if ob.touched_at is None and float(row["low"]) <= ob.top:
                ob.touched_at = ts
        else:
            if float(row["close"]) > ob.top:
                ob.invalidated_at = ts
                return
            if ob.touched_at is None and float(row["high"]) >= ob.bottom:
                ob.touched_at = ts


def find_order_blocks(df: pd.DataFrame, moves,
                      bias: Optional[str] = None) -> List[OrderBlock]:
    if bias is not None:
        bias = str(bias).lower()
        if bias not in ("bullish", "bearish"):
            bias = None
    out: List[OrderBlock] = []
    seen = set()
    for move in moves:
        ev = move.event
        direction = str(ev.direction).lower()
        if direction not in ("up", "down"):
            continue
        ob_dir = "bullish" if direction == "up" else "bearish"
        if bias is not None and ob_dir != bias:
            continue
        known_i = int(df.index.get_indexer([pd.Timestamp(ev.ts)])[0])
        if known_i < 0:
            continue
        start_i = _origin_index(df, move)
        if start_i is None or start_i >= known_i:
            continue
        anchor_i = _anchor_index(df, start_i, direction)
        if anchor_i is None:
            continue
        if (ob_dir, anchor_i) in seen:
            continue
        seen.add((ob_dir, anchor_i))
        row = df.iloc[anchor_i]
        ob = OrderBlock(
            direction=ob_dir,
            top=float(row["high"]),
            bottom=float(row["low"]),
            anchor_ts=df.index[anchor_i],
            known_at=df.index[known_i],
            kind=str(ev.kind),
        )
        _follow(df, ob, known_i)
        out.append(ob)
    return out