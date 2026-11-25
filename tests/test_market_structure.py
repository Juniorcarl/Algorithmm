import pandas as pd
import pytest

from analysis.market_structure import (
    StructureResult, Trend, analyze, top_down_bias,
)
from market_data.candle_manager import CandleError, CandleSeries


def make_series(closes, start="2026-01-01 00:00", freq="5min", w0=0.5, dw=0.005):
    """Candles from a close path: open = previous close, wicks shrink slowly so that
    no two highs/lows tie exactly (keeps pivots unambiguous)."""
    idx = pd.date_range(start, periods=len(closes), freq=freq)
    rows, prev = [], closes[0]
    for i, c in enumerate(closes):
        o = prev
        w = w0 - dw * i
        rows.append((o, max(o, c) + w, min(o, c) - w, c, 100))
        prev = c
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx)
    return CandleSeries(df, "M5")


# bullish start: pivot high ~104.5 (bar 4), pivot low (bar 7), 3 closes above 104.5 on bars 11-13
BASE = [100, 101, 102, 103, 104, 103, 102, 101, 102, 103, 104, 105, 106, 107]


def test_two_closes_are_only_a_wick_no_bos():
    res = analyze(make_series(BASE[:13]))          # only 2 closes above the high
    assert res.events == [] and res.trend is Trend.UNDEFINED


def test_streak_reset_means_no_bos():
    path = BASE[:12] + [104] + [105, 106]          # 105 -> back to 104 -> 105, 106: never 3 in a row
    res = analyze(make_series(path))
    assert res.events == []


def test_three_closes_make_initial_bullish_bos():
    res = analyze(make_series(BASE))
    assert len(res.events) == 1
    ev = res.events[0]
    assert (ev.kind, ev.direction, ev.initial) == ("BOS", "UP", True)
    assert ev.confirm_idx == 13 and ev.break_start_idx == 11
    assert ev.swing.kind == "HL" and ev.swing.idx == 7
    assert res.trend is Trend.BULLISH


def test_bos_continuation_needs_pullback_then_three_closes():
    path = BASE + [108, 107, 106, 105, 106, 107, 109, 110, 111]
    res = analyze(make_series(path))
    kinds = [(e.kind, e.direction) for e in res.events]
    assert kinds == [("BOS", "UP"), ("BOS", "UP")]
    second = res.events[1]
    assert second.confirm_idx == 22 and second.swing.kind == "HL" and second.swing.idx == 17


def test_straight_run_does_not_spam_bos():
    path = BASE + [108, 109, 110, 111, 112, 113, 114, 115, 116]   # no pullback at all
    res = analyze(make_series(path))
    assert len(res.events) == 1


def test_choch_when_hl_taken_out_by_three_closes():
    path = BASE + [108, 107, 106, 105, 104, 103, 102, 101, 100, 99, 98]
    res = analyze(make_series(path))
    assert [(e.kind, e.direction) for e in res.events] == [("BOS", "UP"), ("CHOCH", "DOWN")]
    assert res.events[1].confirm_idx == 24
    assert res.trend is Trend.BEARISH


def test_wick_below_hl_is_not_a_choch():
    # dips below the HL for 2 closes only, then recovers
    path = BASE + [108, 107, 106, 105, 104, 103, 102, 101, 100, 99, 104, 106]
    res = analyze(make_series(path))
    assert [(e.kind) for e in res.events] == ["BOS"]
    assert res.trend is Trend.BULLISH


def test_no_lookahead_prefix_consistency():
    path = BASE + [108, 107, 106, 105, 106, 107, 109, 110, 111, 110, 108, 106, 104, 102, 100, 98, 96]
    full_series = make_series(path)
    full = analyze(full_series)
    for k in range(5, len(path) + 1):
        part = analyze(full_series[:k])
        expected = [e for e in full.events if e.confirm_idx < k]
        assert [(e.kind, e.direction, e.confirm_idx) for e in part.events] == \
               [(e.kind, e.direction, e.confirm_idx) for e in expected]
        assert part.trend_by_bar == full.trend_by_bar[:k]


def test_bearish_mirror():
    path = [200 - c + 100 for c in BASE]           # mirror image of BASE
    res = analyze(make_series(path))
    assert len(res.events) == 1
    ev = res.events[0]
    assert (ev.kind, ev.direction) == ("BOS", "DOWN") and ev.swing.kind == "LH"
    assert res.trend is Trend.BEARISH


def test_top_down_bias_htf_priority_and_counter_tf():
    def r(t):
        return StructureResult(trend=t)
    out = top_down_bias({"W1": r(Trend.BULLISH), "D1": r(Trend.UNDEFINED),
                         "H1": r(Trend.BEARISH), "M5": r(Trend.BULLISH)})
    assert out.bias is Trend.BULLISH and out.anchor_tf == "W1"
    assert out.counter_tfs == ["H1"] and not out.aligned
    assert top_down_bias({"H1": r(Trend.UNDEFINED)}).bias is Trend.UNDEFINED


# ------------------------------------------------------------- candle_manager
def test_resample_m5_to_m15_ohlc():
    s = make_series([100, 101, 102, 103, 104, 105])
    m15 = s.resample("M15")
    assert len(m15) == 2
    first = m15.df.iloc[0]
    assert first.open == s.df.open.iloc[0] and first.close == s.df.close.iloc[2]
    assert first.high == s.df.high.iloc[:3].max() and first.low == s.df.low.iloc[:3].min()
    assert first.volume == 300


def test_cannot_build_lower_timeframe():
    s = CandleSeries(make_series([1, 2, 3]).df, "M15")
    with pytest.raises(CandleError):
        s.resample("M5")


def test_validation_rejects_bad_ohlc_and_duplicates():
    s = make_series([100, 101, 102])
    bad = s.df.copy()
    bad.iloc[1, bad.columns.get_loc("high")] = 0.0
    with pytest.raises(CandleError):
        CandleSeries(bad, "M5")
    dup = pd.concat([s.df, s.df.iloc[[0]]])
    with pytest.raises(CandleError):
        CandleSeries(dup, "M5")


def test_closed_only_drops_forming_candle():
    s = make_series([100, 101, 102, 103])           # opens 00:00, 00:05, 00:10, 00:15
    now = pd.Timestamp("2026-01-01 00:17")
    assert len(s.closed_only(now)) == 3             # the 00:15 candle closes at 00:20


def test_csv_loader_mt5_export(tmp_path):
    p = tmp_path / "xau.csv"
    p.write_text("<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\n"
                 "2026.01.01\t00:00:00\t2000\t2002\t1999\t2001\t10\n"
                 "2026.01.01\t00:05:00\t2001\t2003\t2000\t2002\t12\n")
    s = CandleSeries.from_csv(p, "M5")
    assert len(s) == 2 and s.df.close.iloc[1] == 2002 and s.df.volume.iloc[0] == 10
