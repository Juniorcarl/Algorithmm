"""Draw the program's own candles, BOS/CHoCH, gaps, OBs and breakers.
"""
import argparse
import json
import os
import webbrowser
from pathlib import Path

import pandas as pd

from analysis.breaker_blocks import find_breakers
from analysis.fvg import find_moves
from analysis.market_structure import analyze
from analysis.orderblocks import find_order_blocks
from market_data.candle_manager import CandleSeries
from run_fvg import MINUTES, build


def _epoch(ts):
    if ts is None:
        return None
    return int(pd.Timestamp(ts).timestamp())


def collect(df: pd.DataFrame, last: int) -> dict:
    structure = analyze(df)
    trend = getattr(structure, "trend", None)
    bias = str(getattr(trend, "value", "bearish")).lower()
    
    moves = find_moves(df, structure)
    obs = find_order_blocks(df, moves, bias=bias)
    brs = find_breakers(df, obs, structure)
    n = len(df)
    off = max(0, n - last) if last > 0 else 0

    def sh(i):
        return None if i is None else int(i) - off

    def idx(ts):
        if ts is None:
            return None
        return int(df.index.get_indexer([pd.Timestamp(ts)])[0])

    candles = [[_epoch(t), float(o), float(h), float(l), float(c)]
               for t, o, h, l, c in zip(df.index[off:], df["open"].to_numpy()[off:],
                                        df["high"].to_numpy()[off:], df["low"].to_numpy()[off:],
                                        df["close"].to_numpy()[off:])]

    events, gaps = [], []
    for m in moves:
        ev = m.event
        if ev.confirm_idx < off:
            continue
        chosen = m.selected_at(ev.confirm_idx)
        k = len(events)
        events.append({
            "kind": ev.kind, "dir": ev.direction, "level": ev.level,
            "level_idx": max(0, sh(ev.level_idx)), "confirm": sh(ev.confirm_idx),
            "start": max(0, sh(ev.break_start_idx)), "initial": ev.initial,
            "trend": ev.new_trend.value, "origin": sh(m.origin_idx),
            "ngaps": len(m.gaps),
        })
        for g in m.gaps:
            if g.dead_idx is not None and g.dead_idx < off:
                continue
            gaps.append({
                "ev": k, "dir": g.direction, "c1": sh(g.c1_idx), "c3": sh(g.c3_idx),
                "top": g.top, "bottom": g.bottom, "dead": sh(g.dead_idx),
                "touched": sh(g.touched_idx), "sel": g is chosen,
            })

    obs_data = []
    for ob in obs:
        obs_data.append({
            "dir": ob.direction,
            "top": float(ob.top), "bottom": float(ob.bottom),
            "anchor": sh(idx(ob.anchor_ts)),
            "known": sh(idx(ob.known_at)),
            "dead": sh(idx(ob.invalidated_at)),
            "touched": sh(idx(ob.touched_at)),
            "kind": ob.kind,
            "anchor_ts": _epoch(ob.anchor_ts),
            "known_ts": _epoch(ob.known_at),
            "dead_ts": _epoch(ob.invalidated_at),
        })

    brs_data = []
    for br in brs:
        brs_data.append({
            "dir": br.direction,
            "top": float(br.top), "bottom": float(br.bottom),
            "failed": sh(idx(br.failed_at)),
            "armed": sh(idx(br.armed_at)),
            "retested": sh(idx(br.retested_at)),
            "dead": sh(idx(br.dead_at)),
            "failed_ts": _epoch(br.failed_at),
            "armed_ts": _epoch(br.armed_at),
            "dead_ts": _epoch(br.dead_at),
        })

    # --- FILTERS ---
    
    # 1. Keep only the first OB if multiple overlap in the same direction
    final_obs = []
    for ob in obs_data:
        overlap = False
        for existing in final_obs:
            if existing["dir"] == ob["dir"]:
                if ob["bottom"] < existing["top"] and ob["top"] > existing["bottom"]:
                    overlap = True
                    break
        if not overlap:
            final_obs.append(ob)
    obs_data = final_obs

    # 2. Remove any FVG that overlaps an OB
    final_gaps = []
    for g in gaps:
        overlap = False
        for ob in obs_data:
            if ob["dir"] == g["dir"]:
                if g["bottom"] < ob["top"] and g["top"] > ob["bottom"]:
                    overlap = True
                    break
        if not overlap:
            final_gaps.append(g)
    gaps = final_gaps

    return {"candles": candles, "events": events, "gaps": gaps,
            "obs": obs_data, "brs": brs_data}


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>KYRO | TRADER chart - __TITLE__</title>
<style>
 html,body{margin:0;height:100%;background:#0f1218;color:#cfd6e4;font:13px system-ui,Segoe UI,sans-serif;overflow:hidden}
 #bar{min-height:44px;display:flex;align-items:center;flex-wrap:wrap;gap:14px;padding:6px 12px;background:#171b24;border-bottom:1px solid #262c3a}
 #bar b{color:#fff} select,label{color:#cfd6e4} select{background:#0f1218;border:1px solid #333b4d;padding:3px 6px;border-radius:4px;max-width:420px}
 .k{display:inline-block;width:10px;height:10px;margin-right:4px;vertical-align:-1px}
 #tip{position:fixed;pointer-events:none;background:#171b24ee;border:1px solid #333b4d;border-radius:6px;padding:6px 8px;display:none;line-height:1.5;font-size:12px}
 canvas{display:block}
</style></head><body>
<div id="bar">
 <b>__TITLE__</b>
 <span>Jump to: <select id="jump"></select></span>
 <label><input type="checkbox" id="showOB" checked> order blocks</label>
 <label><input type="checkbox" id="showBRK"> breakers</label>
 <label><input type="checkbox" id="other"> other gaps</label>
 <span><i class="k" style="background:#4da3ff"></i>BOS
  <i class="k" style="background:#ffb020"></i>CHoCH
  <i class="k" style="background:#26a69a"></i>bull
  <i class="k" style="background:#ef5350"></i>bear
  <span style="color:#7d889f">(OBs: solid = active, dashed = tapped, faded = dead)</span></span>
 <span style="color:#7d889f">wheel = zoom, drag = move</span>
</div>
<canvas id="c"></canvas><div id="tip"></div>
<script>
const D = __DATA__;
const C = D.candles, E = D.events, G = D.gaps, OBS = D.obs, BRKS = D.brs;
const n = C.length;
const cv = document.getElementById('c'), ctx = cv.getContext('2d'), tip = document.getElementById('tip');
const PR = 70, PB = 26, PT = 10, TOP = 44;
let W, H, dpr = window.devicePixelRatio || 1, barW = 9, left = 0, mx = -1, my = -1, drag = null;
const showOther = document.getElementById('other');
const showOB = document.getElementById('showOB');
const showBRK = document.getElementById('showBRK');

function fmt(t, add) { return new Date((t + (add || 0)) * 1000).toISOString().slice(0, 16).replace('T', ' '); }
function resize() {
  W = innerWidth; H = innerHeight - TOP;
  cv.width = W * dpr; cv.height = H * dpr;
  cv.style.width = W + 'px'; cv.style.height = H + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); draw();
}
const pw = () => W - PR;
const X = i => (i - left) * barW + barW / 2;
function range() {
  const i0 = Math.max(0, Math.floor(left)), i1 = Math.min(n - 1, Math.ceil(left + pw() / barW));
  let lo = Infinity, hi = -Infinity;
  for (let i = i0; i <= i1; i++) { if (C[i][3] < lo) lo = C[i][3]; if (C[i][2] > hi) hi = C[i][2]; }
  if (!isFinite(lo)) { lo = 0; hi = 1; }
  const pad = (hi - lo) * 0.06 || 1; return [lo - pad, hi + pad, i0, i1];
}
function draw() {
  ctx.clearRect(0, 0, W, H);
  const [lo, hi, i0, i1] = range(), ph = H - PT - PB;
  const Y = p => PT + (hi - p) / (hi - lo) * ph;
  ctx.font = '11px system-ui'; ctx.textBaseline = 'middle';
  const step = niceStep((hi - lo) / 8);
  ctx.strokeStyle = '#1b2130'; ctx.fillStyle = '#7d889f'; ctx.lineWidth = 1; ctx.textAlign = 'left';
  for (let p = Math.ceil(lo / step) * step; p < hi; p += step) {
    ctx.beginPath(); ctx.moveTo(0, Y(p)); ctx.lineTo(pw(), Y(p)); ctx.stroke();
    ctx.fillText(p.toFixed(step < 1 ? 2 : 0), pw() + 6, Y(p));
  }
  ctx.save(); ctx.beginPath(); ctx.rect(0, 0, pw(), H - PB); ctx.clip();

  let drawnLabels = [];

  // ===== ORDER BLOCKS =====
  if (showOB.checked) {
    OBS.forEach(ob => {
      if (ob.anchor === null) return;
      
      // Stop the box at the touch or death, otherwise extend to right edge
      let endIdx = ob.dead;
      if (ob.touched !== null && (endIdx === null || ob.touched < endIdx)) endIdx = ob.touched;
      const x0 = X(ob.anchor) - barW / 2;
      const x1 = endIdx === null ? X(n - 1) + barW / 2 : X(endIdx) + barW / 2;
      
      if (x1 < 0 || x0 > pw()) return;
      const bull = ob.dir === 'bullish';
      const rgb = bull ? '38,166,154' : '239,83,80';
      const dead = ob.dead !== null;
      const touched = !dead && ob.touched !== null;
      
      let fillA = 0.20, strokeA = 1.0, strokeW = 1.5, dash = [];
      if (dead) { fillA = 0.04; strokeA = 0.30; strokeW = 1; dash = [3, 3]; }
      else if (touched) { fillA = 0.12; strokeA = 0.75; strokeW = 1; dash = [4, 3]; }
      
      const yTop = Y(ob.top), yBot = Y(ob.bottom), h = Math.max(1, yBot - yTop);
      ctx.fillStyle = `rgba(${rgb},${fillA})`;
      ctx.fillRect(x0, yTop, x1 - x0, h);
      ctx.strokeStyle = `rgba(${rgb},${strokeA})`;
      ctx.lineWidth = strokeW; ctx.setLineDash(dash);
      ctx.strokeRect(x0, yTop, x1 - x0, h);
      ctx.setLineDash([]);
      
      if (!dead && !touched) {
        let ly = yTop - 2;
        let canDraw = true;
        for (let py of drawnLabels) { if (Math.abs(py - ly) < 12) canDraw = false; }
        if (canDraw) {
          ctx.fillStyle = `rgba(${rgb},${strokeA})`;
          ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
          ctx.fillText(`OB ${bull ? 'bull' : 'bear'} ${ob.bottom.toFixed(2)}-${ob.top.toFixed(2)}`,
            Math.max(x0, 2) + 2, ly);
          drawnLabels.push(ly);
          ctx.textBaseline = 'middle';
        }
      }
    });
  }

  // ===== BREAKERS =====
  if (showBRK.checked) {
    BRKS.forEach(br => {
      if (br.failed === null) return;
      
      // Stop at retest or death
      let endIdx = br.dead;
      if (br.retested !== null && (endIdx === null || br.retested < endIdx)) endIdx = br.retested;
      const x0 = X(br.failed) - barW / 2;
      const x1 = endIdx === null ? X(n - 1) + barW / 2 : X(endIdx) + barW / 2;
      
      if (x1 < 0 || x0 > pw()) return;
      const bull = br.dir === 'bullish';
      const rgb = bull ? '38,166,154' : '239,83,80';
      const dead = br.dead !== null;
      const armed = br.armed !== null;
      
      let fillA = 0.10, strokeA = 1.0, strokeW = 2, dash = [];
      if (dead) { fillA = 0.03; strokeA = 0.30; strokeW = 1; dash = [3, 3]; }
      else if (!armed) { fillA = 0.05; strokeA = 0.55; strokeW = 1; dash = [6, 4]; }
      
      const yTop = Y(br.top), yBot = Y(br.bottom), h = Math.max(1, yBot - yTop);
      ctx.fillStyle = `rgba(${rgb},${fillA})`;
      ctx.fillRect(x0, yTop, x1 - x0, h);
      ctx.strokeStyle = `rgba(${rgb},${strokeA})`;
      ctx.lineWidth = strokeW; ctx.setLineDash(dash);
      ctx.strokeRect(x0, yTop, x1 - x0, h);
      ctx.setLineDash([]);
      
      if (armed && br.armed <= (endIdx || n)) {
        const ax = X(br.armed);
        ctx.fillStyle = `rgba(${rgb},0.9)`;
        ctx.beginPath();
        ctx.moveTo(ax, yTop + 6); ctx.lineTo(ax - 4, yTop + 1); ctx.lineTo(ax + 4, yTop + 1);
        ctx.closePath(); ctx.fill();
      }
      if (br.retested !== null) {
        const rx = X(br.retested);
        ctx.fillStyle = `rgba(${rgb},0.9)`;
        ctx.beginPath(); ctx.arc(rx, yTop + 3, 2.5, 0, 7); ctx.fill();
      }
      if (!dead && armed) {
        ctx.fillStyle = `rgba(${rgb},${strokeA})`;
        ctx.textAlign = 'left'; ctx.textBaseline = 'top';
        ctx.fillText(`BRK ${bull ? 'bull' : 'bear'} ${br.bottom.toFixed(2)}-${br.top.toFixed(2)}`,
          Math.max(x0, 2) + 2, yBot + 2);
        ctx.textBaseline = 'middle';
      }
    });
  }

  // ===== GAPS =====
  G.forEach(g => {
    if (!g.sel && !showOther.checked) return;
    
    // Stop at touch or death
    let endIdx = g.dead;
    if (g.touched !== null && (endIdx === null || g.touched < endIdx)) endIdx = g.touched;
    const x0 = X(g.c1) - barW / 2;
    const x1 = endIdx === null ? X(n - 1) + barW / 2 : X(endIdx) + barW / 2;
    
    if (x1 < 0 || x0 > pw()) return;
    const bull = g.dir === 'UP', rgb = bull ? '38,166,154' : '239,83,80';
    ctx.fillStyle = `rgba(${rgb},${g.sel ? 0.30 : 0.10})`;
    ctx.fillRect(x0, Y(g.top), x1 - x0, Math.max(1, Y(g.bottom) - Y(g.top)));
    ctx.strokeStyle = `rgba(${rgb},${g.sel ? 1 : 0.5})`;
    ctx.lineWidth = g.sel ? 1.5 : 1;
    ctx.setLineDash(g.sel ? [] : [4, 3]);
    ctx.strokeRect(x0, Y(g.top), x1 - x0, Math.max(1, Y(g.bottom) - Y(g.top)));
    ctx.setLineDash([]);
    
    if (g.sel && g.touched === null) {
      let ly = Y(g.top) - 2;
      let canDraw = true;
      for (let py of drawnLabels) { if (Math.abs(py - ly) < 12) canDraw = false; }
      if (canDraw) {
        ctx.fillStyle = `rgb(${rgb})`; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
        ctx.fillText(`FVG ${g.bottom.toFixed(2)}-${g.top.toFixed(2)}`,
          Math.max(x0, 2) + 2, ly);
        drawnLabels.push(ly);
        ctx.textBaseline = 'middle';
      }
    }
  });

  // ===== BOS / CHoCH =====
  E.forEach(e => {
    const x0 = X(e.level_idx), x1 = X(e.confirm), y = Y(e.level);
    if (x1 < 0 || x0 > pw()) return;
    const col = e.kind === 'BOS' ? '#4da3ff' : '#ffb020';
    ctx.strokeStyle = col; ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
    ctx.fillStyle = col;
    ctx.beginPath(); ctx.arc(x1, y, 3.5, 0, 7); ctx.fill();
    ctx.textAlign = 'center'; ctx.textBaseline = e.dir === 'UP' ? 'bottom' : 'top';
    ctx.fillText(`${e.kind === 'BOS' ? 'BOS' : 'CHoCH'} ${e.dir === 'UP' ? '\u25B2' : '\u25BC'} ${e.level.toFixed(2)}${e.initial ? ' (initial)' : ''}`,
      (x0 + x1) / 2, e.dir === 'UP' ? y - 3 : y + 3);
    ctx.textBaseline = 'middle';
  });

  // ===== CANDLES =====
  const w = Math.max(1, barW * 0.7);
  for (let i = i0; i <= i1; i++) {
    const [t, o, h, l, c] = C[i], x = X(i), up = c >= o, col = up ? '#26a69a' : '#ef5350';
    ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, Y(h)); ctx.lineTo(x, Y(l)); ctx.stroke();
    ctx.fillRect(x - w / 2, Math.min(Y(o), Y(c)), w, Math.max(1, Math.abs(Y(o) - Y(c))));
  }
  ctx.restore();

  // ===== time axis + crosshair =====
  ctx.fillStyle = '#7d889f'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  const every = Math.max(1, Math.ceil(110 / barW));
  for (let i = Math.ceil(i0 / every) * every; i <= i1; i += every) ctx.fillText(fmt(C[i][0]).slice(5), X(i), H - PB / 2);
  if (mx >= 0 && mx < pw() && my >= 0 && my < H - PB) {
    ctx.strokeStyle = '#4a5368'; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(mx, 0); ctx.lineTo(mx, H - PB); ctx.moveTo(0, my); ctx.lineTo(pw(), my); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#2a3144'; ctx.fillRect(pw(), my - 9, PR, 18);
    ctx.fillStyle = '#fff'; ctx.textAlign = 'left';
    ctx.fillText((hi - (my - PT) / ph * (hi - lo)).toFixed(2), pw() + 6, my);
  }
}
function niceStep(r) { const p = Math.pow(10, Math.floor(Math.log10(r))), f = r / p; return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * p; }

cv.addEventListener('mousemove', e => {
  mx = e.offsetX; my = e.offsetY;
  if (drag) { left = drag.left - (mx - drag.x) / barW; clampLeft(); }
  const i = Math.floor(left + mx / barW);
  if (mx < pw() && i >= 0 && i < n && !drag) {
    const [t, o, h, l, c] = C[i];
    tip.innerHTML = `<b>${fmt(t)} UTC</b> &nbsp;(${fmt(t, 7200)} Botswana)<br>O ${o.toFixed(2)} &nbsp;H ${h.toFixed(2)} &nbsp;L ${l.toFixed(2)} &nbsp;<b>C ${c.toFixed(2)}</b>`;
    tip.style.display = 'block';
    tip.style.left = Math.min(e.clientX + 14, innerWidth - 330) + 'px';
    tip.style.top = (e.clientY + 18) + 'px';
  } else tip.style.display = 'none';
  draw();
});
cv.addEventListener('mouseleave', () => { mx = my = -1; tip.style.display = 'none'; draw(); });
cv.addEventListener('mousedown', e => { drag = { x: e.offsetX, left }; });
addEventListener('mouseup', () => { drag = null; });
cv.addEventListener('wheel', e => {
  e.preventDefault();
  const idx = left + e.offsetX / barW;
  barW = Math.min(40, Math.max(2, barW * (e.deltaY < 0 ? 1.15 : 1 / 1.15)));
  left = idx - e.offsetX / barW; clampLeft(); draw();
}, { passive: false });
function clampLeft() { left = Math.max(-10, Math.min(n - 5, left)); }
addEventListener('resize', resize);

// ===== jump dropdown =====
const sel = document.getElementById('jump');
function optgroup(label) {
  const g = document.createElement('optgroup'); g.label = label; sel.appendChild(g); return g;
}
const gE = optgroup('BOS / CHoCH');
E.forEach((e, k) => {
  const o = document.createElement('option');
  o.value = 'E' + k;
  o.textContent = `${fmt(C[e.confirm][0])}  ${e.kind} ${e.dir}  ${e.level.toFixed(2)}`;
  gE.appendChild(o);
});
const gO = optgroup('Order Blocks');
OBS.forEach((ob, k) => {
  const o = document.createElement('option');
  o.value = 'O' + k;
  const state = ob.dead !== null ? 'dead' : (ob.touched !== null ? 'touched' : 'active');
  const when = ob.anchor_ts ? fmt(ob.anchor_ts) : '?';
  o.textContent = `${when}  OB ${ob.dir}  ${ob.bottom.toFixed(2)}-${ob.top.toFixed(2)}  [${state}]`;
  gO.appendChild(o);
});
const gB = optgroup('Breakers');
BRKS.forEach((br, k) => {
  const o = document.createElement('option');
  o.value = 'B' + k;
  const state = br.dead !== null ? 'dead' : (br.armed === null ? 'wait' : 'armed');
  const when = br.failed_ts ? fmt(br.failed_ts) : '?';
  o.textContent = `${when}  BRK ${br.dir}  ${br.bottom.toFixed(2)}-${br.top.toFixed(2)}  [${state}]`;
  gB.appendChild(o);
});
sel.addEventListener('change', () => {
  const v = sel.value, kind = v[0], k = +v.slice(1);
  let target = null;
  if (kind === 'E') target = E[k].confirm;
  else if (kind === 'O') target = OBS[k].anchor;
  else if (kind === 'B') target = BRKS[k].failed;
  if (target != null) { left = target - pw() / barW / 2; clampLeft(); draw(); }
});

showOther.addEventListener('change', draw);
showOB.addEventListener('change', draw);
showBRK.addEventListener('change', draw);
resize();
left = n - pw() / barW + 12; clampLeft();
draw();
</script></body></html>"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="M5 candle file")
    p.add_argument("--tf", default="M15", choices=["M5"] + list(MINUTES), help="timeframe to draw")
    p.add_argument("--last", type=int, default=800, help="candles to draw (0 = all)")
    p.add_argument("--out", default="chart.html")
    p.add_argument("--no-open", action="store_true", help="do not open the browser")
    args = p.parse_args()

    m5 = CandleSeries.from_csv(args.csv, "M5").df
    df = m5 if args.tf == "M5" else build(m5, MINUTES[args.tf])
    data = collect(df, args.last)
    html = PAGE.replace("__TITLE__", f"XAUUSD {args.tf}").replace("__DATA__", json.dumps(data))
    out = Path(os.path.abspath(args.out))
    out.write_text(html, encoding="utf-8")
    print(f"{args.tf}: {len(data['candles'])} candles, "
          f"{len(data['events'])} BOS/CHoCH, {len(data['gaps'])} gaps, "
          f"{len(data['obs'])} OBs, {len(data['brs'])} breakers -> {out}")
    if not args.no_open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()