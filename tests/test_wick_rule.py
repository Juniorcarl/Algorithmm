"""Tests for the failed-break rule: a level moves to the wick of a failed break."""
import numpy as np
import pandas as pd
import pytest

from analysis.market_structure import Trend, analyze


def frame(rows):
    """rows = [(high, low, close), ...]"""
    idx = pd.date_range("2026-01-01", periods=len(rows), freq="5min")
    h, l, c = (np.array(x, dtype=float) for x in zip(*rows))
    return pd.DataFrame({"open": c, "high": h, "low": l, "close": c}, index=idx)


def mirror(rows):
    return [(200 - l, 200 - h, 200 - c) for h, l, c in rows]


# pivot high 110, pivot low, 3 closes above -> BULLISH; then a pullback so the
# new high (116) stops following price. HL = 101, HH = 116.
UPTREND = [
    (104, 100, 102), (106, 102, 104), (110, 104, 106), (107, 103, 104),
    (105, 101, 103), (112, 103, 111), (114, 110, 113), (116, 112, 115),
    (114, 111, 112), (113, 108, 110), (114, 110, 112), (115, 111, 113),
]


def test_setup_is_bullish():
    r = analyze(frame(UPTREND))
    assert (r.trend, r.high, r.low, len(r.events)) == (Trend.BULLISH, 116, 101, 1)


def test_wick_only_moves_continuation_high():
    r = analyze(frame(UPTREND + [(119, 112, 114)]))
    assert (r.high, r.high_idx, len(r.events)) == (119, 12, 1)


def test_two_closes_then_back_inside_moves_high_to_furthest_wick():
    r = analyze(frame(UPTREND + [(120, 113, 117), (119, 115, 118), (118, 113, 115)]))
    assert (r.high, r.high_idx, len(r.events)) == (120, 12, 1)


def test_level_waits_while_attempt_is_open():
    r = analyze(frame(UPTREND + [(120, 113, 117), (119, 115, 118)]))
    assert r.high == 116


def test_three_closes_is_bos_on_the_unmoved_level():
    r = analyze(frame(UPTREND + [(120, 113, 117), (121, 116, 119), (122, 117, 120)]))
    assert [(e.kind, e.level) for e in r.events[1:]] == [("BOS", 116)]


def test_after_move_bos_needs_three_closes_beyond_new_wick():
    base = UPTREND + [(119, 112, 114)]
    below_wick = base + [(119, 116, 118)] * 3
    assert len(analyze(frame(below_wick)).events) == 1
    above_wick = base + [(121, 117, 120)] * 3
    assert analyze(frame(above_wick)).events[-1].level == 119


def test_hl_is_measured_from_original_swing_high():
    rows = UPTREND + [(119, 112, 114)] + [(121, 117, 120)] * 3
    assert analyze(frame(rows)).events[-1].swing.price == 108


def test_choch_level_moves_and_then_needs_three_closes():
    swept = UPTREND + [(113, 98, 103)]
    r = analyze(frame(swept))
    assert (r.low, r.low_idx, r.trend) == (98, 12, Trend.BULLISH)
    still = analyze(frame(swept + [(102, 98.5, 99)] * 3))
    assert still.trend is Trend.BULLISH
    broke = analyze(frame(swept + [(99, 95, 96)] * 3))
    assert (broke.events[-1].kind, broke.events[-1].level) == ("CHOCH", 98)


def test_mirror_bearish():
    r = analyze(frame(mirror(UPTREND + [(119, 112, 114), (113, 98, 103)])))
    assert (r.trend, r.low, r.high) == (Trend.BEARISH, 81, 102)


@pytest.mark.parametrize("seed", range(5))
def test_no_lookahead(seed):
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.normal(0, 1, 300))
    df = frame(list(zip(c + rng.random(300), c - rng.random(300), c)))
    full = analyze(df)
    for cut in range(20, 300, 7):
        part = analyze(df.iloc[:cut])
        assert part.trend_by_bar == full.trend_by_bar[:cut]
        assert part.events == [e for e in full.events if e.confirm_idx < cut]