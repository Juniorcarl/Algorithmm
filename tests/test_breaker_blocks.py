# tests/test_breaker_blocks.py
from dataclasses import dataclass

import pandas as pd

from analysis.breaker_blocks import find_breakers
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


def _obs(df, start=0, conf=3, kind="BOS", direction="up"):
    ev = Ev(kind, direction, df.index[conf])
    return find_order_blocks(df, [Move(ev, start)])


def test_breaker_armed_by_invalidation_three_closes():
    rows = [
        [101.0, 101.5, 100.0, 100.2],   # 0 OB
        [100.2, 103.0, 100.5, 102.5],   # 1
        [102.5, 105.0, 102.0, 104.5],   # 2
        [104.5, 106.0, 104.0, 105.5],   # 3 confirming
        [105.5, 105.6, 101.0, 102.0],   # 4 touch
        [102.0, 102.2, 99.5, 99.9],     # 5 first close beyond -> failed
        [99.9, 100.2, 99.2, 99.5],      # 6 wick back in, still closes beyond
        [99.5, 99.7, 98.8, 99.0],       # 7 3rd close -> armed
        [99.0, 100.4, 98.9, 99.8],      # 8 retest of the zone
        [99.8, 102.2, 99.7, 102.0],     # 9 close above top -> dead
    ]
    df = frame(rows)
    obs = _obs(df)
    brs = find_breakers(df, obs, [])
    assert len(brs) == 1
    br = brs[0]
    assert br.direction == "bearish"
    assert br.failed_at == df.index[5]
    assert br.armed_at == df.index[7]
    assert br.retested_at == df.index[8]
    assert br.dead_at == df.index[9]
    assert not br.live


def test_breaker_armed_by_later_bos():
    rows = [
        [101.0, 101.5, 100.0, 100.2],
        [100.2, 103.0, 100.5, 102.5],
        [102.5, 105.0, 102.0, 104.5],
        [104.5, 106.0, 104.0, 105.5],
        [105.5, 105.6, 101.0, 102.0],
        [102.0, 102.2, 99.5, 99.9],     # 5 failed
        [99.9, 100.6, 99.6, 100.4],     # 6 back inside
        [100.4, 100.5, 99.4, 99.6],     # 7
        [99.6, 99.8, 99.0, 99.2],       # 8
        [99.2, 99.4, 98.6, 98.9],       # 9 later BOS down confirms here
        [98.9, 100.3, 98.8, 99.7],      # 10 retest
    ]
    df = frame(rows)
    obs = _obs(df)
    events = [Ev("BOS", "down", df.index[9])]
    brs = find_breakers(df, obs, events)
    assert len(brs) == 1
    br = brs[0]
    assert br.failed_at == df.index[5]
    assert br.armed_at == df.index[9]
    assert br.retested_at == df.index[10]
    assert br.dead_at is None


def test_breaker_not_armed_without_bos():
    rows = [
        [101.0, 101.5, 100.0, 100.2],
        [100.2, 103.0, 100.5, 102.5],
        [102.5, 105.0, 102.0, 104.5],
        [104.5, 106.0, 104.0, 105.5],
        [105.5, 105.6, 101.0, 102.0],
        [102.0, 102.2, 99.5, 99.9],     # 5 failed
        [99.9, 100.6, 99.6, 100.4],     # 6 back inside
    ]
    df = frame(rows)
    obs = _obs(df)
    brs = find_breakers(df, obs, [])
    assert len(brs) == 1
    assert brs[0].armed_at is None
    assert not brs[0].armed
    assert not brs[0].live


def test_bullish_breaker_mirror():
    rows = [
        [99.0, 100.0, 98.8, 99.8],      # 0 OB (bearish OB)
        [99.8, 100.1, 97.0, 97.5],      # 1
        [97.5, 98.0, 95.0, 95.5],       # 2
        [95.5, 96.0, 94.0, 94.5],       # 3 confirming
        [94.5, 99.5, 94.2, 99.0],       # 4 touch
        [99.0, 100.6, 98.9, 100.5],     # 5 close above -> failed
        [100.5, 100.9, 99.9, 100.7],    # 6
        [100.7, 101.2, 100.4, 101.0],   # 7 3rd close -> armed
        [101.0, 101.1, 99.0, 99.6],     # 8 retest
        [99.6, 99.7, 98.2, 98.5],       # 9 close below bottom -> dead
    ]
    df = frame(rows)
    obs = _obs(df, conf=3, kind="BOS", direction="down")
    brs = find_breakers(df, obs, [])
    assert len(brs) == 1
    br = brs[0]
    assert br.direction == "bullish"
    assert br.failed_at == df.index[5]
    assert br.armed_at == df.index[7]
    assert br.retested_at == df.index[8]
    assert br.dead_at == df.index[9]