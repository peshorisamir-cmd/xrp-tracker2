import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import {
  rawFeatures, standardize, updateScaler, trainBatch, predictProb,
  defaultWeights, sgdUpdate, assess, FEATURE_NAMES, mean, stdev,
} from './engine.js';

// =====================================================================
// XRP/USD 15-min DIRECTIONAL tracker, v7
//
// This is the directional rebuild. Instead of estimating a price level and
// reading direction off it (v6, which chased random-walk noise), v7 outputs
// a calibrated probability P(close_15m > target) from an online logistic
// model trained on your own settled candles, using cross-asset (BTC/ETH)
// and momentum features. Everything is benchmarked against random walk and
// persistence so you only act when there is a real, demonstrated edge.
//
// For Kalshi: the number you trade on is dirProb / direction. Enter the
// matching side (Yes if UP, No if DOWN) only when the edge panel is green
// AND dirProb clears your threshold AND it is inside the entry window.
// =====================================================================

const POLL_MS = 5000;
const TICK_MS = 1000;
const MODEL_KEY = 'xrp_dir_model_v7';
const PAPER_KEY = 'xrp_dir_paper_v7';
const CUSTOM_PRICE_KEY = 'xrp_custom_price_v1';
const MAX_HISTORY = 600;
const RECENT_CANDLES_SHOWN = 24;
const KALSHI_BASE = 'https://api.elections.kalshi.com/trade-api/v2';

const fontStack = {
  display: "'Bricolage Grotesque', 'DM Sans', system-ui, sans-serif",
  body: "'DM Sans', system-ui, sans-serif",
  mono: "'JetBrains Mono', ui-monospace, monospace",
};

const palette = {
  upColor: '#22d39a', downColor: '#ff5d6a', accent: '#d4a93f',
  surface: '#11161d', surfaceHi: '#161d27', border: '#1d2530',
  muted: '#6b7785', text: '#e8edf2', dim: '#9aa5b3', bg: '#0a0d12',
};

function getIntervalStart(now = new Date()) {
  const m = now.getUTCMinutes();
  const k = Math.floor(m / 15) * 15;
  return new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate(), now.getUTCHours(), k, 0, 0));
}
const fmtPrice = (p, d = 4) => p == null || !isFinite(p) ? '...' : `$${Number(p).toFixed(d)}`;
const fmtTime = (d) => !d ? '...' : d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
const fmtPct = (x) => x == null || !isFinite(x) ? '...' : `${(x * 100).toFixed(1)}%`;

// ---------- persistence ----------
function loadModel() {
  try {
    const raw = localStorage.getItem(MODEL_KEY);
    if (!raw) return { version: 7, history: [], scaler: null, weights: defaultWeights() };
    const p = JSON.parse(raw);
    if (p.version !== 7) return { version: 7, history: [], scaler: null, weights: defaultWeights() };
    return p;
  } catch { return { version: 7, history: [], scaler: null, weights: defaultWeights() }; }
}
function saveModel(m) { try { localStorage.setItem(MODEL_KEY, JSON.stringify(m)); } catch {} }

const DEFAULT_PAPER = {
  enabled: false, stake: 20, dirThreshold: 0.62, minElapsed: 3, maxElapsed: 11,
  requireEdge: true, kalshiEfficiency: 0.85, spreadCents: 0.02,
  dailyMaxLoss: 60, maxConsecLosses: 3, cooldownMinutes: 60,
};
function loadPaper() {
  try {
    const raw = localStorage.getItem(PAPER_KEY);
    if (!raw) return { ...DEFAULT_PAPER, bets: [], pendingBet: null };
    const p = JSON.parse(raw);
    return { ...DEFAULT_PAPER, ...p, bets: p.bets || [], pendingBet: p.pendingBet || null };
  } catch { return { ...DEFAULT_PAPER, bets: [], pendingBet: null }; }
}
function savePaper(s) { try { localStorage.setItem(PAPER_KEY, JSON.stringify(s)); } catch {} }
function loadCustom() {
  try {
    const raw = localStorage.getItem(CUSTOM_PRICE_KEY);
    if (!raw) return { enabled: false, value: null };
    const p = JSON.parse(raw);
    return { enabled: !!p.enabled, value: isFinite(p.value) ? +p.value : null };
  } catch { return { enabled: false, value: null }; }
}
function saveCustom(c) { try { localStorage.setItem(CUSTOM_PRICE_KEY, JSON.stringify(c)); } catch {} }

// ---------- data fetch ----------
// In production (deployed), exchange APIs often block cross-origin browser
// requests, so route through the serverless proxy. On localhost there is no
// /api function, so hit the exchanges directly. Detect dev by hostname.
const IS_LOCAL = typeof window !== 'undefined' &&
  /^(localhost|127\.0\.0\.1|\[::1\])$/.test(window.location.hostname);
function proxied(url) {
  return IS_LOCAL ? url : `/api/proxy?url=${encodeURIComponent(url)}`;
}
async function jget(url) {
  const r = await fetch(proxied(url));
  if (!r.ok) throw new Error(`${r.status}`);
  return r.json();
}

async function fetchTicker(product) {
  const t = await jget(`https://api.exchange.coinbase.com/products/${product}/ticker`);
  const bid = parseFloat(t.bid), ask = parseFloat(t.ask);
  return { mid: (bid + ask) / 2, last: parseFloat(t.price), volume24h: parseFloat(t.volume) };
}
async function fetchKraken() {
  const r = await jget('https://api.kraken.com/0/public/Ticker?pair=XRPUSD');
  if (r.error && r.error.length) throw new Error(r.error.join(','));
  const d = r.result[Object.keys(r.result)[0]];
  const bid = parseFloat(d.b[0]), ask = parseFloat(d.a[0]);
  return { mid: (bid + ask) / 2, volume24h: parseFloat(d.v[1]) };
}
async function fetchBitstamp() {
  const r = await jget('https://www.bitstamp.net/api/v2/ticker/xrpusd/');
  const bid = parseFloat(r.bid), ask = parseFloat(r.ask);
  return { mid: (bid + ask) / 2, volume24h: parseFloat(r.volume) };
}
async function fetchCandles(product, granularity) {
  return jget(`https://api.exchange.coinbase.com/products/${product}/candles?granularity=${granularity}`);
}
async function fetchKalshi() {
  try {
    const data = await jget(`${KALSHI_BASE}/markets?status=open&limit=200`);
    const xrp = (data.markets || []).filter(m => m.ticker && m.ticker.toUpperCase().includes('XRP'));
    if (!xrp.length) return { available: false, error: 'no open XRP markets' };
    const now = Date.now();
    const up = xrp.map(m => ({ ...m, closeMs: new Date(m.close_time || m.expiration_time).getTime() }))
      .filter(m => m.closeMs > now && isFinite(m.closeMs)).sort((a, b) => a.closeMs - b.closeMs);
    return up.length ? { available: true, market: up[0] } : { available: false, error: 'none upcoming' };
  } catch (e) { return { available: false, error: e.message }; }
}

function computeIndex(vals) {
  const v = vals.filter(e => e && isFinite(e.mid) && isFinite(e.volume24h) && e.volume24h > 0);
  if (!v.length) return null;
  const tv = v.reduce((s, e) => s + e.volume24h, 0);
  return v.reduce((s, e) => s + e.mid * e.volume24h, 0) / tv;
}

// Current 15m open for a candle array (newest first) given interval start sec.
function currentCandleOpen(candles, startSec, fallback) {
  if (!Array.isArray(candles)) return fallback;
  const inInt = candles.filter(c => c[0] >= startSec);
  if (!inInt.length) return fallback;
  const sorted = [...inInt].sort((a, b) => a[0] - b[0]);
  return sorted[0][3];
}

function estimateKalshiPrice(dirConf, eff, spread) {
  const prob = 0.5 + eff * (dirConf - 0.5);
  return Math.min(0.99, Math.max(0.01, prob + spread / 2));
}

export default function App() {
  const [state, setState] = useState({
    indexPrice: null, c15: [], c1: [], btc: null, eth: null,
    kalshi: null, exErrors: [], coinbaseStats: null, updatedAt: null, loading: true,
  });
  const [now, setNow] = useState(new Date());
  const [model, setModel] = useState(() => loadModel());
  const [paper, setPaper] = useState(() => loadPaper());
  const [custom, setCustom] = useState(() => loadCustom());
  const bufRef = useRef({ intervalStartMs: null, samples: [], pending: null });

  const load = useCallback(async () => {
    const r = await Promise.allSettled([
      fetchTicker('XRP-USD'), fetchKraken(), fetchBitstamp(),
      fetchCandles('XRP-USD', 900), fetchCandles('XRP-USD', 60),
      fetchTicker('BTC-USD'), fetchTicker('ETH-USD'),
      fetchCandles('BTC-USD', 900), fetchCandles('ETH-USD', 900),
      fetchKalshi(),
      jget('https://api.exchange.coinbase.com/products/XRP-USD/stats').catch(() => null),
    ]);
    const val = (i) => r[i].status === 'fulfilled' ? r[i].value : null;
    const exs = [val(0), val(1), val(2)];
    const exErrors = ['Coinbase', 'Kraken', 'Bitstamp'].filter((_, i) => r[i].status === 'rejected');
    setState({
      indexPrice: computeIndex(exs),
      c15: Array.isArray(val(3)) ? val(3) : [],
      c1: Array.isArray(val(4)) ? val(4) : [],
      btc: { ticker: val(5), c15: Array.isArray(val(7)) ? val(7) : [] },
      eth: { ticker: val(6), c15: Array.isArray(val(8)) ? val(8) : [] },
      kalshi: val(9) || { available: false, error: 'fetch failed' },
      coinbaseStats: val(10),
      exErrors, updatedAt: new Date(), loading: false,
    });
  }, []);

  useEffect(() => {
    load();
    const a = setInterval(load, POLL_MS);
    const b = setInterval(() => setNow(new Date()), TICK_MS);
    return () => { clearInterval(a); clearInterval(b); };
  }, [load]);

  // ---------- live prediction ----------
  const pred = useMemo(() => {
    if (state.indexPrice == null || !state.c15.length) return null;
    const price = state.indexPrice;
    const start = getIntervalStart(now);
    const end = new Date(start.getTime() + 15 * 60_000);
    const elapsedMin = (now - start) / 60_000;
    const remainMin = Math.max(0, 15 - elapsedMin);
    const elapsedFrac = Math.max(0.01, Math.min(1, elapsedMin / 15));
    const startSec = start.getTime() / 1000;

    const open = currentCandleOpen(state.c1, startSec, price);
    const usingCustom = !!(custom.enabled && isFinite(custom.value) && custom.value > 0);
    const target = usingCustom ? custom.value : open;

    const btcOpen = currentCandleOpen(state.btc?.c15, startSec, state.btc?.ticker?.mid);
    const ethOpen = currentCandleOpen(state.eth?.c15, startSec, state.eth?.ticker?.mid);

    const raw = rawFeatures({
      price, target, open, elapsedFrac,
      xrp15: state.c15,
      btcCur: state.btc?.ticker ? { open: btcOpen, price: state.btc.ticker.mid } : null,
      ethCur: state.eth?.ticker ? { open: ethOpen, price: state.eth.ticker.mid } : null,
    });
    const vec = standardize(raw, model.scaler);
    let pUp = predictProb(model.weights, vec);

    // If the model is effectively untrained, fall back to a transparent
    // heuristic so the user is not shown a fake 50.0%: use the sign of the
    // intra-candle move scaled by progress. Clearly flagged as heuristic.
    const trained = model.weights.trained || 0;
    let heuristic = false;
    if (trained < 8) {
      heuristic = true;
      const z = raw.intraZ * Math.sqrt(elapsedFrac) + 0.3 * raw.btcRet;
      pUp = 1 / (1 + Math.exp(-z));
    }

    // distToTarget already partly baked into features; for a custom strike
    // far from price, nudge probability by how far price already is past it.
    const direction = pUp >= 0.5 ? 'UP' : 'DOWN';
    const dirConfidence = Math.max(pUp, 1 - pUp);

    return {
      start, end, elapsedMin, remainMin, elapsedFrac, open, target, usingCustom,
      price, raw, vec, pUp, direction, dirConfidence, heuristic, trained,
      btcRet: raw.btcRet, ethRet: raw.ethRet,
    };
  }, [state, now, model.scaler, model.weights, custom]);

  // ---------- log live prediction + settle closed candles + train ----------
  useEffect(() => {
    if (!pred || state.indexPrice == null) return;
    const intervalStartMs = pred.start.getTime();
    const buf = bufRef.current;

    // settle a pending (previous) candle once it appears closed in c15
    if (buf.pending) {
      const pcStartSec = buf.pending.startMs / 1000;
      const closed = state.c15.find(c => c[0] === pcStartSec);
      if (closed && buf.pending.lastSample) {
        const ls = buf.pending.lastSample;
        const actualClose = closed[4];
        const candleOpen = closed[3];
        const target = ls.target;
        const y = actualClose > target ? 1 : 0;
        const prevCandle = state.c15.find(c => c[0] === pcStartSec - 900);
        const lastDir = prevCandle ? (prevCandle[4] > prevCandle[3] ? 1 : 0) : (candleOpen != null ? 0 : 0);

        const record = {
          startSec: pcStartSec, open: candleOpen, actualClose, target,
          p: ls.pUp,                  // live prob logged at prediction time (honest)
          y, lastDir,
          raw: ls.raw,                // raw features for refit
          dir: ls.direction, dirConf: ls.dirConfidence,
          recordedAt: Date.now(),
        };

        setModel(m => {
          const history = [...m.history, record].slice(-MAX_HISTORY);
          // update scaler with this candle's raw features
          const scaler = updateScaler(m.scaler, ls.raw);
          // refit weights from all logged candles (cheap), walk-forward:
          // train on standardized vectors using the UPDATED scaler.
          const samples = history
            .filter(h => h.raw)
            .map(h => ({ vec: standardize(h.raw, scaler), y: h.y }));
          const weights = samples.length >= 8 ? trainBatch(samples, 14) : (m.weights || defaultWeights());
          const nm = { ...m, history, scaler, weights };
          saveModel(nm);
          return nm;
        });
        buf.pending = null;
      }
    }

    if (buf.intervalStartMs !== null && buf.intervalStartMs !== intervalStartMs) {
      buf.pending = { startMs: buf.intervalStartMs, lastSample: buf.lastSample };
    }
    buf.intervalStartMs = intervalStartMs;
    // keep the most recent live sample for this candle
    buf.lastSample = { pUp: pred.pUp, direction: pred.direction, dirConfidence: pred.dirConfidence, target: pred.target, raw: pred.raw, elapsedMin: pred.elapsedMin };
  }, [pred, state.indexPrice, state.c15]);

  // ---------- model assessment (benchmarks) ----------
  const evalReport = useMemo(() => {
    const settled = model.history.filter(h => isFinite(h.p) && (h.y === 0 || h.y === 1));
    return assess(settled.slice(-200));
  }, [model.history]);

  // ---------- paper trading ----------
  useEffect(() => { savePaper(paper); }, [paper]);

  useEffect(() => {
    if (!paper.enabled || !pred || pred.heuristic) return;
    if (pred.elapsedMin < paper.minElapsed || pred.elapsedMin > paper.maxElapsed) return;
    if (paper.requireEdge && !evalReport.edge) return;
    if (pred.dirConfidence < paper.dirThreshold) return;

    const startSec = Math.floor(pred.start.getTime() / 1000);
    if (paper.pendingBet?.candleStart === startSec) return;
    if (paper.bets.some(b => b.candleStart === startSec)) return;

    // risk guards
    const settled = paper.bets.filter(b => b.settled);
    if (settled.length >= paper.maxConsecLosses) {
      const tail = settled.slice(-paper.maxConsecLosses);
      if (tail.every(b => b.outcome === 'loss')) {
        const until = (tail[tail.length - 1].settledAt || Date.now()) + paper.cooldownMinutes * 60000;
        if (Date.now() < until) return;
      }
    }
    const today = new Date().toDateString();
    const todayPnL = settled.filter(b => new Date(b.settledAt || 0).toDateString() === today).reduce((s, b) => s + (b.pnl || 0), 0);
    if (todayPnL <= -paper.dailyMaxLoss) return;

    const entryPrice = estimateKalshiPrice(pred.dirConfidence, paper.kalshiEfficiency, paper.spreadCents);
    const contracts = paper.stake / entryPrice;
    setPaper(s => ({ ...s, pendingBet: {
      candleStart: startSec, candleEnd: Math.floor(pred.end.getTime() / 1000),
      enteredAt: Date.now(), entryElapsedMin: pred.elapsedMin,
      target: pred.target, direction: pred.direction, dirConfidence: pred.dirConfidence,
      pUp: pred.pUp, stake: paper.stake, entryPrice, contracts, settled: false,
    }}));
  }, [now, pred, paper, evalReport.edge]);

  useEffect(() => {
    if (!paper.pendingBet || paper.pendingBet.settled) return;
    const match = model.history.find(h => h.startSec === paper.pendingBet.candleStart);
    if (!match) return;
    const { target, direction, contracts, stake } = paper.pendingBet;
    const actualDir = match.actualClose > target ? 'UP' : 'DOWN';
    const outcome = Math.abs(match.actualClose - target) < 1e-9 ? 'push' : (actualDir === direction ? 'win' : 'loss');
    const pnl = outcome === 'push' ? 0 : (outcome === 'win' ? contracts - stake : -stake);
    setPaper(s => ({
      ...s,
      bets: [...s.bets, { ...s.pendingBet, settled: true, settledAt: Date.now(), actualClose: match.actualClose, outcome, pnl }].slice(-500),
      pendingBet: null,
    }));
  }, [model.history, paper.pendingBet]);

  // ---------- derived UI values ----------
  const stats = useMemo(() => {
    const b = paper.bets.filter(x => x.settled && x.outcome !== 'push');
    const wins = b.filter(x => x.outcome === 'win');
    const pnl = b.reduce((s, x) => s + (x.pnl || 0), 0);
    return { n: b.length, winRate: b.length ? wins.length / b.length : null, pnl };
  }, [paper.bets]);

  if (state.loading) {
    return <div style={{ minHeight: '100vh', background: palette.bg, color: palette.dim, display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: fontStack.body }}>Loading XRP + BTC + ETH data...</div>;
  }

  const p = palette;
  const cs = state.coinbaseStats;
  const ch24 = state.indexPrice && cs?.open ? state.indexPrice - parseFloat(cs.open) : 0;
  const recent = state.c15.slice(0, RECENT_CANDLES_SHOWN);

  return (
    <div style={{ minHeight: '100vh', background: p.bg, color: p.text, fontFamily: fontStack.body, padding: '24px 16px' }}>
      <link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;700&family=DM+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet" />
      <div style={{ maxWidth: 980, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 20 }}>

        <header style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
          <div>
            <div style={{ fontFamily: fontStack.display, fontWeight: 700, fontSize: 28, letterSpacing: '-0.02em' }}>
              XRP <span style={{ color: p.muted, fontWeight: 500 }}>/ USD · 15m direction</span>
            </div>
            <div style={{ color: p.muted, fontSize: 12, marginTop: 6, fontFamily: fontStack.mono }}>
              v7 directional · {3 - state.exErrors.length}/3 venues · BTC/ETH cross-asset · refresh 5s
            </div>
          </div>
          <div style={{ textAlign: 'right' }}>
            <div style={{ color: p.muted, fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.1em' }}>Updated</div>
            <div style={{ fontFamily: fontStack.mono, fontSize: 14, color: p.dim }}>{fmtTime(state.updatedAt)}</div>
          </div>
        </header>

        {/* Price + the directional call */}
        <section style={{ background: p.surface, border: `1px solid ${p.border}`, borderRadius: 14, padding: 24, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
          <div>
            <div style={{ color: p.muted, fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.12em', marginBottom: 6 }}>XRP index</div>
            <div style={{ fontFamily: fontStack.mono, fontWeight: 600, fontSize: 48, lineHeight: 1 }}>{fmtPrice(state.indexPrice)}</div>
            <div style={{ fontFamily: fontStack.mono, fontSize: 13, color: ch24 >= 0 ? p.upColor : p.downColor, marginTop: 8 }}>
              {ch24 >= 0 ? '+' : ''}{ch24.toFixed(4)} 24h
            </div>
            {pred && (
              <div style={{ marginTop: 14, fontFamily: fontStack.mono, fontSize: 11, color: p.muted, lineHeight: 1.7 }}>
                <div>open {fmtPrice(pred.open)} · target {fmtPrice(pred.target)}{pred.usingCustom ? ' (Kalshi strike)' : ''}</div>
                <div>BTC {pred.btcRet >= 0 ? '+' : ''}{pred.btcRet.toFixed(2)}σ · ETH {pred.ethRet >= 0 ? '+' : ''}{pred.ethRet.toFixed(2)}σ</div>
                <div>elapsed {pred.elapsedMin.toFixed(1)}m · {pred.remainMin.toFixed(1)}m left</div>
              </div>
            )}
          </div>

          {pred && (
            <div style={{ borderLeft: `1px solid ${p.border}`, paddingLeft: 24, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
              <div style={{ color: p.muted, fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.12em', marginBottom: 8 }}>
                P(close {pred.usingCustom ? '> strike' : '> open'})
              </div>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: 12 }}>
                <div style={{ fontFamily: fontStack.display, fontWeight: 700, fontSize: 52, lineHeight: 1, color: pred.direction === 'UP' ? p.upColor : p.downColor }}>
                  {pred.direction === 'UP' ? '↑' : '↓'} {fmtPct(pred.dirConfidence)}
                </div>
              </div>
              <div style={{ fontFamily: fontStack.mono, fontSize: 12, color: p.dim, marginTop: 10 }}>
                {pred.heuristic
                  ? `Heuristic only · model needs ${8 - pred.trained} more settled candles`
                  : `Model-trained on ${pred.trained} updates`}
              </div>
              {!pred.heuristic && !evalReport.edge && evalReport.ready && (
                <div style={{ marginTop: 10, fontSize: 11, color: p.downColor, fontFamily: fontStack.mono }}>
                  No proven edge yet · see panel below
                </div>
              )}
            </div>
          )}
        </section>

        {/* EDGE / BENCHMARK PANEL — the honesty layer */}
        <section style={{ background: p.surface, border: `1px solid ${evalReport.edge ? p.upColor + '66' : p.border}`, borderRadius: 14, padding: 20 }}>
          <div style={{ fontFamily: fontStack.mono, fontSize: 11, color: p.muted, textTransform: 'uppercase', letterSpacing: '0.12em', marginBottom: 12 }}>
            Edge vs baselines {evalReport.ready ? `· ${evalReport.n} settled` : ''}
          </div>
          {!evalReport.ready ? (
            <div style={{ color: p.dim, fontSize: 14 }}>{evalReport.reason}</div>
          ) : (
            <>
              <div style={{
                fontSize: 15, fontWeight: 600, marginBottom: 14,
                color: evalReport.edge ? p.upColor : p.accent,
              }}>
                {evalReport.edge ? '✓ ' : '⚠ '}{evalReport.verdict}
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 12 }}>
                <Metric label="Model log-loss" value={evalReport.modelLL?.toFixed(3)} sub={`vs RW ${evalReport.rwLL?.toFixed(3)}`} good={evalReport.beatsRW} />
                <Metric label="Dir accuracy" value={fmtPct(evalReport.modelAcc)} sub={`coinflip 50%`} good={evalReport.modelAcc > 0.5} />
                <Metric label="Persistence" value={fmtPct(evalReport.persAcc)} sub="baseline to beat" good={evalReport.beatsPers} />
                <Metric label="Acc @ ≥60%" value={evalReport.acc60 != null ? fmtPct(evalReport.acc60) : '—'} sub="when confident" good={evalReport.acc60 != null && evalReport.acc60 > 0.5} />
                <Metric label="Acc @ ≥70%" value={evalReport.acc70 != null ? fmtPct(evalReport.acc70) : '—'} sub="high conviction" good={evalReport.acc70 != null && evalReport.acc70 > 0.5} />
                <Metric label="Brier" value={evalReport.modelBrier?.toFixed(3)} sub="lower better" good={evalReport.modelBrier < 0.25} />
              </div>
            </>
          )}
        </section>

        {/* Feature weights — what the model is actually keying on */}
        {(model.weights.trained || 0) >= 8 && (
          <section style={{ background: p.surface, border: `1px solid ${p.border}`, borderRadius: 14, padding: 20 }}>
            <div style={{ fontFamily: fontStack.mono, fontSize: 11, color: p.muted, textTransform: 'uppercase', letterSpacing: '0.12em', marginBottom: 12 }}>
              Learned feature weights
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {FEATURE_NAMES.map((name, i) => {
                const w = model.weights.w[i];
                const maxAbs = Math.max(...model.weights.w.map(Math.abs), 0.01);
                const pctW = (Math.abs(w) / maxAbs) * 100;
                return (
                  <div key={name} style={{ display: 'grid', gridTemplateColumns: '120px 1fr 60px', alignItems: 'center', gap: 10, fontFamily: fontStack.mono, fontSize: 11 }}>
                    <span style={{ color: p.dim }}>{name}</span>
                    <div style={{ background: p.bg, borderRadius: 3, height: 8, position: 'relative', overflow: 'hidden' }}>
                      <div style={{ position: 'absolute', left: w >= 0 ? '50%' : `${50 - pctW / 2}%`, width: `${pctW / 2}%`, height: '100%', background: w >= 0 ? p.upColor : p.downColor }} />
                      <div style={{ position: 'absolute', left: '50%', width: 1, height: '100%', background: p.border }} />
                    </div>
                    <span style={{ color: w >= 0 ? p.upColor : p.downColor, textAlign: 'right' }}>{w >= 0 ? '+' : ''}{w.toFixed(2)}</span>
                  </div>
                );
              })}
            </div>
            <div style={{ marginTop: 10, fontSize: 11, color: p.muted, fontFamily: fontStack.mono }}>
              Positive = pushes toward UP. Read these to sanity-check the model is keying on signal (btcRet, momentum) not noise.
            </div>
          </section>
        )}

        {/* Custom Kalshi strike override */}
        <CustomStrike custom={custom} setCustom={(c) => { setCustom(c); saveCustom(c); }} indexPrice={state.indexPrice} />

        {/* Paper trading */}
        <PaperPanel paper={paper} setPaper={setPaper} pred={pred} stats={stats} edge={evalReport.edge} />

        {/* Recent candles */}
        <section style={{ background: p.surface, border: `1px solid ${p.border}`, borderRadius: 14, padding: 20 }}>
          <div style={{ fontFamily: fontStack.mono, fontSize: 11, color: p.muted, textTransform: 'uppercase', letterSpacing: '0.12em', marginBottom: 12 }}>Recent 15m candles</div>
          <div style={{ display: 'flex', gap: 3, alignItems: 'flex-end', height: 80 }}>
            {recent.slice().reverse().map((c, i) => {
              const up = c[4] >= c[3];
              const closes = recent.map(x => x[4]);
              const lo = Math.min(...closes), hi = Math.max(...closes);
              const h = 20 + ((c[4] - lo) / (hi - lo || 1)) * 56;
              return <div key={i} title={`O ${c[3].toFixed(4)} C ${c[4].toFixed(4)}`} style={{ flex: 1, height: h, background: up ? p.upColor : p.downColor, opacity: 0.55 + 0.45 * (i / recent.length), borderRadius: 2 }} />;
            })}
          </div>
        </section>

        <div style={{ textAlign: 'center', color: p.muted, fontSize: 11, fontFamily: fontStack.mono, paddingBottom: 20 }}>
          Paper only. Realistic ceiling for 15m XRP direction is ~52-55% after costs. If the edge panel is not green, the honest read is: do not trade.
        </div>
      </div>
    </div>
  );
}

function Metric({ label, value, sub, good }) {
  const p = palette;
  return (
    <div style={{ background: p.bg, borderRadius: 8, padding: 12 }}>
      <div style={{ color: p.muted, fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.1em' }}>{label}</div>
      <div style={{ fontFamily: fontStack.mono, fontSize: 22, fontWeight: 600, color: good ? p.upColor : p.text, marginTop: 4 }}>{value ?? '—'}</div>
      <div style={{ color: p.muted, fontSize: 10, fontFamily: fontStack.mono, marginTop: 2 }}>{sub}</div>
    </div>
  );
}

function CustomStrike({ custom, setCustom, indexPrice }) {
  const p = palette;
  const [draft, setDraft] = useState(custom.value != null ? String(custom.value) : '');
  const val = parseFloat(draft);
  const valid = isFinite(val) && val > 0;
  const devBp = (custom.enabled && valid && indexPrice) ? ((indexPrice - val) / val) * 10000 : null;
  return (
    <section style={{ background: p.surface, border: `1px solid ${custom.enabled ? p.accent + '66' : p.border}`, borderRadius: 14, padding: 20 }}>
      <div style={{ fontFamily: fontStack.mono, fontSize: 11, color: p.muted, textTransform: 'uppercase', letterSpacing: '0.12em', marginBottom: 12 }}>
        Kalshi strike override
      </div>
      <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
        <input value={draft} onChange={e => setDraft(e.target.value)} placeholder="e.g. 2.1500"
          style={{ background: p.bg, border: `1px solid ${p.border}`, color: p.text, borderRadius: 6, padding: '8px 12px', fontFamily: fontStack.mono, fontSize: 14, width: 140 }} />
        <button onClick={() => setCustom({ enabled: !custom.enabled, value: valid ? val : null })}
          style={{ padding: '8px 16px', background: custom.enabled ? p.accent : 'transparent', color: custom.enabled ? p.bg : p.dim, border: `1px solid ${custom.enabled ? p.accent : p.border}`, borderRadius: 6, fontFamily: fontStack.mono, fontSize: 13, cursor: 'pointer', fontWeight: 600 }}>
          {custom.enabled ? 'Using strike' : 'Use open'}
        </button>
        {devBp != null && (
          <span style={{ fontFamily: fontStack.mono, fontSize: 12, color: Math.abs(devBp) < 2 ? p.muted : devBp > 0 ? p.upColor : p.downColor }}>
            price is {devBp >= 0 ? '+' : ''}{devBp.toFixed(1)}bp vs strike
          </span>
        )}
      </div>
      <div style={{ marginTop: 10, fontSize: 11, color: p.muted, fontFamily: fontStack.mono }}>
        Type the exact strike on your Kalshi market. The probability becomes P(close above that strike), which is what the contract pays on.
      </div>
    </section>
  );
}

function PaperPanel({ paper, setPaper, pred, stats, edge }) {
  const p = palette;
  const recent = paper.bets.filter(b => b.settled).slice(-12).reverse();
  const inputStyle = { background: p.bg, border: `1px solid ${p.border}`, color: p.text, borderRadius: 4, padding: '4px 8px', width: 70, fontFamily: fontStack.mono, fontSize: 12 };
  return (
    <section style={{ background: p.surface, border: `1px solid ${p.border}`, borderRadius: 14, padding: 20 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
        <div style={{ fontFamily: fontStack.mono, fontSize: 11, color: p.muted, textTransform: 'uppercase', letterSpacing: '0.12em' }}>Paper trading</div>
        <button onClick={() => setPaper(s => ({ ...s, enabled: !s.enabled }))}
          style={{ padding: '6px 14px', background: paper.enabled ? p.upColor : 'transparent', color: paper.enabled ? p.bg : p.dim, border: `1px solid ${paper.enabled ? p.upColor : p.border}`, borderRadius: 6, fontFamily: fontStack.mono, fontSize: 12, cursor: 'pointer', fontWeight: 600 }}>
          {paper.enabled ? 'ON' : 'OFF'}
        </button>
      </div>

      {paper.enabled && paper.requireEdge && !edge && (
        <div style={{ background: '#2a1a0a', border: `1px solid ${p.accent}55`, color: '#ffd58a', padding: 10, borderRadius: 8, fontSize: 12, marginBottom: 12, fontFamily: fontStack.mono }}>
          Entries blocked: no proven edge yet. The model is logging and learning. Toggle "require edge" off to test anyway (not advised).
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginBottom: 14 }}>
        <Metric label="Settled" value={stats.n} sub="trades" good={false} />
        <Metric label="Win rate" value={stats.winRate != null ? fmtPct(stats.winRate) : '—'} sub="of settled" good={stats.winRate != null && stats.winRate > 0.5} />
        <Metric label="P&L" value={stats.n ? `${stats.pnl >= 0 ? '+' : '-'}$${Math.abs(stats.pnl).toFixed(2)}` : '—'} sub="paper" good={stats.pnl > 0} />
      </div>

      {recent.length > 0 && (
        <div style={{ marginBottom: 14 }}>
          {recent.map((b, i) => (
            <div key={i} style={{ display: 'grid', gridTemplateColumns: '0.9fr 0.4fr 0.7fr 1.1fr 0.6fr 0.6fr', fontFamily: fontStack.mono, fontSize: 12, padding: '6px 0', borderBottom: `1px solid ${p.border}55`, color: p.dim }}>
              <span>{new Date(b.candleStart * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
              <span style={{ color: b.direction === 'UP' ? p.upColor : p.downColor, fontWeight: 600 }}>{b.direction === 'UP' ? '↑' : '↓'}</span>
              <span>{fmtPct(b.dirConfidence)}</span>
              <span>{b.target.toFixed(4)}→{b.actualClose?.toFixed(4)}</span>
              <span style={{ color: b.outcome === 'win' ? p.upColor : p.downColor, textTransform: 'uppercase' }}>{b.outcome}</span>
              <span style={{ color: b.pnl >= 0 ? p.upColor : p.downColor }}>{b.pnl >= 0 ? '+' : '-'}${Math.abs(b.pnl).toFixed(2)}</span>
            </div>
          ))}
        </div>
      )}

      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'center', fontFamily: fontStack.mono, fontSize: 11, color: p.muted, paddingTop: 12, borderTop: `1px solid ${p.border}` }}>
        <span>Stake $</span>
        <input type="number" value={paper.stake} onChange={e => setPaper(s => ({ ...s, stake: Math.max(1, +e.target.value || 20) }))} style={inputStyle} />
        <span>Dir thresh</span>
        <input type="number" step="0.02" value={paper.dirThreshold} onChange={e => setPaper(s => ({ ...s, dirThreshold: Math.max(0.5, Math.min(0.95, +e.target.value || 0.62)) }))} style={inputStyle} />
        <span>Enter min..max min</span>
        <input type="number" value={paper.minElapsed} onChange={e => setPaper(s => ({ ...s, minElapsed: Math.max(0, +e.target.value || 3) }))} style={inputStyle} />
        <input type="number" value={paper.maxElapsed} onChange={e => setPaper(s => ({ ...s, maxElapsed: Math.max(1, +e.target.value || 11) }))} style={inputStyle} />
        <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}>
          <input type="checkbox" checked={paper.requireEdge} onChange={e => setPaper(s => ({ ...s, requireEdge: e.target.checked }))} />
          require edge
        </label>
        <button onClick={() => { if (window.confirm('Reset paper history?')) setPaper(s => ({ ...s, bets: [], pendingBet: null })); }}
          style={{ marginLeft: 'auto', padding: '5px 12px', background: 'transparent', color: p.muted, border: `1px solid ${p.border}`, borderRadius: 4, fontFamily: fontStack.mono, fontSize: 11, cursor: 'pointer' }}>
          Reset
        </button>
      </div>
    </section>
  );
}
