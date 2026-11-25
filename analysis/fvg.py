"""
fvg.py - Module 2 of KYRO | TRADER

Confirmed rules
---------------
* Gap (wick to wick). Bullish: candle 3's low is above candle 1's high.
  Bearish: candle 3's high is below candle 1's low.
* Dead: a later candle CLOSES beyond candle 1's full range (bullish: below
  candle 1's low; bearish: above candle 1's high). A wick does not kill it.
* Only gaps of a move that caused a BOS or CHoCH count (CHoCH counts too).
* Of those, the one that matters is the first gap from the start of the move.
  If it is dead, the next live one is used.

Design choices (please review)
------------------------------
F1. A move runs from the swing that started it (HL/LH for a BOS, LL/HH for a
    CHoCH) to the candle that confirmed the break. All three gap candles must be
    inside that range.
F2. Candle colour is not checked and there is no minimum gap size.
F3. "Touched" = the first candle after candle 3 that reaches the near edge.

No look-ahead: a gap exists only once candle 3 has closed, and a move only once
its break is confirmed. Always query with an index (alive_at, selected_at).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from analysis.market_structure import StructureEvent, StructureResult


@dataclass(frozen=True)
class FVG:
    direction: str                 # "UP" (bullish) | "DOWN" (bearish)
    c1_idx: int                    # candle 1; candle 3 is c1_idx + 2
    ts: pd.Timestamp               # time of the MIDDLE candle (easy to find on a chart)
    bottom: float
    top: float
    kill_level: float              # a close beyond this kills the gap
    touched_idx: Optional[int]     # first touch of the near edge
    dead_idx: Optional[int]        # first close beyond kill_level

    @property
    def c3_idx(self) -> int:
        return self.c1_idx + 2

    @property
    def near_edge(self) -> float:
        """The edge price reaches first on the way back (your entry edge)."""
        return self.top if self.direction == "UP" else self.bottom

    def alive_at(self, idx: int) -> bool:
        return self.c3_idx <= idx and (self.dead_idx is None or self.dead_idx > idx)

    def touched_at(self, idx: int) -> bool:
        return self.touched_idx is not None and self.touched_idx <= idx


@dataclass(frozen=True)
class Move:
    event: StructureEvent          # the BOS / CHoCH this move caused
    origin_idx: int                # swing that started the move
    gaps: Tuple[FVG, ...]          # its gaps, ordered from the start of the move

    def selected_at(self, idx: int) -> Optional[FVG]:
        """The gap that matters as of candle `idx`, or None."""
        if idx < self.event.confirm_idx:
            return None
        return next((g for g in self.gaps if g.alive_at(idx)), None)


def _first(mask: np.ndarray, offset: int) -> Optional[int]:
    hits = np.flatnonzero(mask)
    return int(hits[0]) + offset if hits.size else None


def find_gaps(df: pd.DataFrame) -> List[FVG]:
    """Every wick-to-wick gap, in candle order (no selection applied)."""
    hi = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    cl = df["close"].to_numpy(dtype=float)
    gaps: List[FVG] = []
    for a in range(len(df) - 2):
        c = a + 2
        after = c + 1
        if lo[c] > hi[a]:
            gaps.append(FVG("UP", a, df.index[a + 1], float(hi[a]), float(lo[c]), float(lo[a]),
                            _first(lo[after:] <= lo[c], after), _first(cl[after:] < lo[a], after)))
        elif hi[c] < lo[a]:
            gaps.append(FVG("DOWN", a, df.index[a + 1], float(hi[c]), float(lo[a]), float(hi[a]),
                            _first(hi[after:] >= hi[c], after), _first(cl[after:] > hi[a], after)))
    return gaps


def find_moves(df: pd.DataFrame, structure: StructureResult) -> List[Move]:
    """One Move per BOS / CHoCH, holding the gaps that move created."""
    gaps = find_gaps(df)
    starts = np.array([g.c1_idx for g in gaps], dtype=int)
    moves: List[Move] = []
    for ev in structure.events:
        origin = ev.swing.idx
        first = int(np.searchsorted(starts, origin, side="left"))
        last = int(np.searchsorted(starts, ev.confirm_idx - 2, side="right"))
        own = tuple(g for g in gaps[first:last] if g.direction == ev.direction)
        moves.append(Move(ev, origin, own))
    return moves