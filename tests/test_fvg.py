"""Tests for Module 2 (fvg)."""
import numpy as np
import pandas as pd
import pytest

from analysis.fvg import find_gaps, find_moves
from analysis.market_structure import StructureEvent, StructureResult, SwingPoint, Trend, analyze


def frame(rows):
    """rows = [(high, low, close), ...]"""
    idx = pd.date_range("2026-01-01", periods=len(rows), freq="5min")
    h, l, c = (np.array(x, dtype=float) for x in zip(*rows))
    return pd.DataFrame({"open": c, "high": h, "low": l, "close": c}, index=idx)


def mirror(rows):
    return [(200 - l, 200 - h, 200 - c) for h, l, c in rows]


def structure(df, confirm, origin=0, direction="UP"):
    swing = SwingPoint("HL", 0.0, origin, df.index[origin])
    ev = StructureEvent("BOS", direction, 0.0, 0, confirm, confirm, df.index[confirm],
                        Trend.BULLISH, swing)
    return StructureResult(events=[ev])


# two bullish gaps: A = 102-104 (candles 0-2), B = 112-114 (candles 3-5)
LEG = [(102, 100, 101), (108, 101, 107), (110, 104, 109),
       (112, 108, 111), (118, 111, 117), (120, 114, 119)]


def test_wick_to_wick_gap():
    g = find_gaps(frame(LEG[:3]))[0]
    assert (g.direction, g.bottom, g.top, g.kill_level, g.near_edge) == ("UP", 102, 104, 100, 104)


def test_overlapping_wicks_are_no_gap():
    assert find_gaps(frame([(102, 100, 101), (108, 101, 107), (110, 102, 109)])) == []


def test_wick_beyond_candle_one_does_not_kill():
    g = find_gaps(frame(LEG[:3] + [(106, 98, 100)]))[0]
    assert g.dead_idx is None and g.touched_idx == 3


def test_close_beyond_candle_one_kills():
    g = find_gaps(frame(LEG[:3] + [(106, 98, 99)]))[0]
    assert g.dead_idx == 3 and g.alive_at(2) and not g.alive_at(3)


def test_selects_first_gap_from_start():
    df = frame(LEG)
    m = find_moves(df, structure(df, 5))[0]
    first = m.selected_at(5)
    assert (first.bottom, first.top) == (102, 104)
    assert m.selected_at(4) is None          # break not confirmed yet


def test_falls_back_to_next_live_gap():
    # a dip closes under candle 1 of gap A, then a second gap forms
    rows = LEG[:3] + [(104, 96, 98), (101, 97, 100), (112, 99, 111), (116, 106, 115)]
    df = frame(rows)
    m = find_moves(df, structure(df, 6))[0]
    assert len(m.gaps) == 2 and m.gaps[0].dead_idx == 3
    assert (m.selected_at(6).bottom, m.selected_at(6).top) == (101, 106)


def test_gap_outside_the_move_is_ignored():
    df = frame(LEG)
    m = find_moves(df, structure(df, 5, origin=3))[0]
    assert m.selected_at(5).bottom == 112


def test_move_without_gap():
    df = frame([(102, 100, 101), (103, 101, 102), (104, 101, 103)])
    assert find_moves(df, structure(df, 2))[0].selected_at(2) is None


def test_mirror_bearish():
    df = frame(mirror(LEG))
    g = find_moves(df, structure(df, 5, direction="DOWN"))[0].selected_at(5)
    assert (g.direction, g.bottom, g.top, g.near_edge) == ("DOWN", 96, 98, 96)


@pytest.mark.parametrize("seed", range(4))
def test_no_lookahead(seed):
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.normal(0, 1.5, 400))
    df = frame(list(zip(c + rng.random(400) / 2, c - rng.random(400) / 2, c)))
    full = find_moves(df, analyze(df))
    box = lambda g: None if g is None else (g.c1_idx, g.bottom, g.top)
    for cut in range(30, 400, 13):
        part = find_moves(df.iloc[:cut], analyze(df.iloc[:cut]))
        assert len(part) == sum(m.event.confirm_idx < cut for m in full)
        for a, b in zip(part, full):
            assert box(a.selected_at(cut - 1)) == box(b.selected_at(cut - 1))