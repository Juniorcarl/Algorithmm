"""breaker_blocks.py - Module 3b of KYRO | TRADER

A failed Order Block becomes a Breaker Block, same range (register v0.2).

Armed (tradeable) by whichever comes first (A7):
  path 1 - its own invalidation reaches 3 consecutive closes beyond the zone
           (Q4 C); armed on the 3rd close.
  path 2 - the next BOS / CHoCH in the new direction confirms.
Retest: first return into the zone AFTER arming (informational).
Dead:   a candle closes back beyond the far side of the zone.
"""
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from analysis.orderblocks import OrderBlock


@dataclass
class BreakerBlock:
    direction: str
    top: float
    bottom: float
    failed_at: pd.Timestamp
    kind: str = ""
    anchor_ts: Optional[pd.Timestamp] = None
    armed_at: Optional[pd.Timestamp] = None
    retested_at: Optional[pd.Timestamp] = None
    dead_at: Optional[pd.Timestamp] = None

    @property
    def armed(self) -> bool:
        return self.armed_at is not None

    @property
    def live(self) -> bool:
        return self.armed_at is not None and self.dead_at is None


def _event_list(events):
    """Accept a list, a StructureResult wrapper, or anything iterable."""
    if events is None:
        return []
    if isinstance(events, (list, tuple)):
        return list(events)
    for attr in ("events", "structure_events", "all_events", "items"):
        v = getattr(events, attr, None)
        if v is not None and not callable(v):
            try:
                return list(v)
            except TypeError:
                pass
    try:
        return list(events)
    except TypeError:
        raise TypeError(
            "find_breakers got %r, which is neither iterable nor a "
            "container with .events. Please paste analysis/"
            "market_structure.py's StructureResult class so it can be "
            "wired up." % type(events).__name__
        )


def _three_close_bos(df: pd.DataFrame, start_i: int,
                     breaker: BreakerBlock) -> Optional[pd.Timestamp]:
    count = 0
    for i in range(start_i, len(df)):
        close = float(df.iloc[i]["close"])
        if breaker.direction == "bearish":
            beyond = close < breaker.bottom
        else:
            beyond = close > breaker.top
        if not beyond:
            return None
        count += 1
        if count >= 3:
            return df.index[i]
    return None


def _next_event(events, direction: str,
                after: pd.Timestamp) -> Optional[pd.Timestamp]:
    for ev in events:
        if str(ev.direction).lower() != direction:
            continue
        ts = pd.Timestamp(ev.ts)
        if ts >= after:
            return ts
    return None


def _follow_breaker(df: pd.DataFrame, br: BreakerBlock, armed_i: int) -> None:
    for i in range(armed_i + 1, len(df)):
        row = df.iloc[i]
        ts = df.index[i]
        if br.direction == "bearish":
            if float(row["close"]) > br.top:
                br.dead_at = ts
                return
            if br.retested_at is None and float(row["high"]) >= br.bottom:
                br.retested_at = ts
        else:
            if float(row["close"]) < br.bottom:
                br.dead_at = ts
                return
            if br.retested_at is None and float(row["low"]) <= br.top:
                br.retested_at = ts


def find_breakers(df: pd.DataFrame, obs: List[OrderBlock],
                  events) -> List[BreakerBlock]:
    events = _event_list(events)
    out: List[BreakerBlock] = []
    for ob in obs:
        if ob.invalidated_at is None:
            continue
        inv_i = int(df.index.get_indexer([pd.Timestamp(ob.invalidated_at)])[0])
        if inv_i < 0:
            continue
        br = BreakerBlock(
            direction="bearish" if ob.direction == "bullish" else "bullish",
            top=ob.top,
            bottom=ob.bottom,
            failed_at=ob.invalidated_at,
            kind=ob.kind,
            anchor_ts=ob.anchor_ts,
        )
        candidates = []
        t1 = _three_close_bos(df, inv_i, br)
        if t1 is not None:
            candidates.append(t1)
        new_dir = "down" if br.direction == "bearish" else "up"
        t2 = _next_event(events, new_dir, br.failed_at)
        if t2 is not None:
            candidates.append(t2)
        out.append(br)
        if not candidates:
            continue
        br.armed_at = min(candidates)
        armed_i = int(df.index.get_indexer([br.armed_at])[0])
        if armed_i >= 0:
            _follow_breaker(df, br, armed_i)
    return out