# KYRO | TRADER - rules register v0.1 (approved)

Source of truth: the trader. Anything not listed here is NOT a rule.

## Market structure
- Swings follow the PDF logic: a high that breaks the previous high is a higher high; the low before it is the higher low (mirror for bearish).
- BOS (= BMS): 3+ consecutive candle closes beyond a high/low. Fewer than 3 closes = a wick.
- CHoCH: 3 closes beyond the HL (uptrend) or LH (downtrend).
- External structure = HTF swing range; internal = structure inside it on a lower timeframe.
- Consolidation: no trading.

## Bias and timeframes
- Full top-down analysis (monthly/weekly down to 5m); no concept tied to a fixed timeframe.
- HTF bias has priority. A counter-trend scalp is valid only when it supports the HTF direction (e.g. LTF entry from a valid OB leading price into an HTF OB).

## POIs
- Order block: only the last opposite candle before a move that caused a BOS; full candle incl. wick; invalid on any close beyond it.
- Breaker block: a failed OB, same range; traded on a retest in the new direction; a BOS in that direction is required first.
- OBs are the strongest POI. Fibonacci is NOT used (removed from spec).

## Confirmation and entry
- Liquidity sweep: wick takes a swing high/low or session high/low (equal highs/lows do not count); no close-back needed; the wick must be inside the POI.
- FVG: wick-to-wick gap between candle 1 and candle 3; dead on a close beyond candle 1's full range. For confirmation only the FVG counts.
- Entry: at the confirmation FVG edge, first touch. Stop loss: beyond the sweep wick. Take profit: next higher-timeframe POI. Minimum R:R 1:3.

## Scoring and filters
- Confluences (3-drive, inducement, QML, flip zones) add to a score; a signal is possible without them.
- All sessions traded; no news filter.

## Assumptions approved
- A1: <3 closes beyond a level = sweep; 3+ = BOS.
- A2: next HTF POI under 1:3 R:R = no signal, logged as rejected.
- A3: "FVG edge" = the edge price reaches first.
- A4: HTF disagreement without a valid counter-trend scalp = WAIT.

## Open
- CRT (parked), chart time zone, consolidation detection, minimum FVG size, confluence score weights/threshold.
