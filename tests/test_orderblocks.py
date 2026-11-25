# tests/test_orderblocks.py
from dataclasses import dataclass

import pandas as pd
import pytest

from analysis.orderblocks import find_order_blocks


@dataclass
class Ev:
    kind: str
    direction: str
    ts: pd.Timestamp


class Move:
    def __init__(self, event, start_index):
        self.event = event
        self.start_index = start_index


def frame(rows, start="2026-01-01 00:00"):
    idx = pd.date_range(start, periods=len(rows), freq="5min")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


ROWS = [
    [101.0, 101.5, 100.0, 100.2],   # 0 origin, bearish -> the OB
    [100.2, 103.0, 100.5, 102.5],   # 1
    [102.5, 105.0, 102.0, 104.5],   # 2
    [104.5, 106.0, 104.0, 105.5],   # 3 confirming
    [105.5, 105.6, 101.0, 102.0],   # 4 touch, close inside
    [102.0, 102.2, 99.5, 99.9],     # 5 close below 100 -> dead
]


def test_bullish_ob_anchor_and_lifecycle():
    df = frame(ROWS)
    ev = Ev("BOS", "up", df.index[3])
    obs = find_order_blocks(df, [Move(ev, 0)])
    assert len(obs) == 1
    ob = obs[0]
    assert ob.direction == "bullish"
    assert ob.top == 101.5 and ob.bottom == 100.0
    assert ob.anchor_ts == df.index[0]
    assert ob.known_at == df.index[3]
    assert ob.kind == "BOS"
    assert ob.touched_at == df.index[4]
    assert ob.invalidated_at == df.index[5]
    assert not ob.active


def test_origin_candle_bullish_steps_back():
    rows = [
        [101.2, 101.5, 100.5, 100.8],   # 0 bearish -> the OB
        [100.8, 101.0, 99.0, 100.9],    # 1 bullish hammer = origin
        [100.9, 103.0, 100.6, 102.8],   # 2
        [102.8, 105.0, 102.5, 104.8],   # 3
        [104.8, 106.0, 104.2, 105.6],   # 4 confirming
    ]
    df = frame(rows)
    ev = Ev("BOS", "up", df.index[4])
    obs = find_order_blocks(df, [Move(ev, 1)])
    assert len(obs) == 1
    assert obs[0].anchor_ts == df.index[0]
    assert obs[0].bottom == 100.5 and obs[0].top == 101.5


def test_bearish_ob_mirror():
    rows = [
        [99.0, 100.0, 98.8, 99.8],      # 0 origin bullish -> the OB
        [99.8, 100.1, 97.0, 97.5],      # 1
        [97.5, 98.0, 95.0, 95.5],       # 2
        [95.5, 96.0, 94.0, 94.5],       # 3 confirming
        [94.5, 99.5, 94.2, 99.0],       # 4 touch, close inside
        [99.0, 100.6, 98.9, 100.5],     # 5 close above 100 -> dead
    ]
    df = frame(rows)
    ev = Ev("CHoCH", "down", df.index[3])
    obs = find_order_blocks(df, [Move(ev, 0)])
    assert len(obs) == 1
    assert obs[0].direction == "bearish"
    assert obs[0].bottom == 98.8 and obs[0].top == 100.0
    assert obs[0].touched_at == df.index[4]
    assert obs[0].invalidated_at == df.index[5]


def test_no_moves_no_obs():
    df = frame(ROWS)
    assert find_order_blocks(df, []) == []


def test_status_only_after_confirmation():
    df = frame(ROWS[:4])
    ev = Ev("BOS", "up", df.index[3])
    obs = find_order_blocks(df, [Move(ev, 0)])
    assert len(obs) == 1
    assert obs[0].active
    assert obs[0].touched_at is None


def test_smoke_synthetic_trend():
    from analysis.fvg import find_moves
    from analysis.market_structure import analyze

    rows, p = [], 100.0
    for i in range(80):
        step = 0.6 if (i // 4) % 2 == 0 else -0.3
        o, c = p, p + step
        rows.append([o, max(o, c) + 0.2, min(o, c) - 0.2, c])
        p = c
    df = frame(rows)
    events = analyze(df)
    if not events:
        pytest.skip("synthetic series produced no structure events")
    moves = find_moves(df, events)
    if not moves:
        pytest.skip("synthetic series produced no moves")
    obs = find_order_blocks(df, moves)
    assert obs, "events exist but no OBs were produced"
    for ob in obs:
        row = df.loc[ob.anchor_ts]
        assert ob.top == float(row["high"]) and ob.bottom == float(row["low"])
        assert ob.known_at >= ob.anchor_ts