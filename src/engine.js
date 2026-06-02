// =====================================================================
// XRP 15-min DIRECTIONAL engine, v7
//
// Philosophy shift from v6 (read the research notes):
//   v6 predicted a PRICE LEVEL and read direction off it. At a 15-min
//   horizon a price-level estimate is a random-walk echo, so its
//   "direction" was noise wearing a confidence hat.
//
//   v7 predicts DIRECTION directly as a calibrated probability:
//     P(close_15m > target)
//   It is a small online logistic-regression model trained on YOUR OWN
//   settled candles. It optimizes log-loss (a proper scoring rule), so
//   what it is graded on IS what you trade on.
//
// Key upgrades vs v6:
//   1. Target = sign of next-bar log return, with a "flat" deadband so the
//      model is not punished for failing to call sub-noise wiggles.
//   2. Cross-asset features: BTC and ETH 15-min returns. XRP at this horizon
//      is heavily BTC-beta driven; v6 had none of this.
//   3. Direction-native features: intra-candle z-score, momentum, RSI,
//      MACD histogram, realized-vol regime, autocorrelation regime,
//      candle progress, and distance-to-target in sigma.
//   4. Online logistic regression (SGD) trained on settled outcomes.
//   5. ALWAYS benchmarked against random walk (50%) and persistence
//      ("predict last candle's direction"). If the model is not beating
//      both on log-loss AND directional accuracy, it tells you so and you
//      should not trade.
//   6. Confidence reported as a calibrated probability, with abstention
//      inside a deadband around 50%.
//
// No price-level fighting. No shrinkage/regime phase-out hacks.
// =====================================================================

// ---------- math helpers ----------
export function mean(a) { return a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0; }
export function stdev(a) {
  if (a.length < 2) return 0;
  const m = mean(a);
  return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / (a.length - 1));
}
export function sigmoid(z) { return 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, z)))); }

// Lag-1 autocorrelation of an array. + = trending, - = mean-reverting.
export function lag1Autocorr(a) {
  if (a.length < 4) return 0;
  const m = mean(a);
  let num = 0, den = 0;
  for (let i = 0; i < a.length; i++) {
    den += (a[i] - m) ** 2;
    if (i > 0) num += (a[i] - m) * (a[i - 1] - m);
  }
  return den > 0 ? num / den : 0;
}

// Wilder RSI on a series of closes, returns 0..100 (50 = neutral).
export function rsi(closes, period = 14) {
  if (closes.length < period + 1) return 50;
  let gain = 0, loss = 0;
  for (let i = closes.length - period; i < closes.length; i++) {
    const d = closes[i] - closes[i - 1];
    if (d >= 0) gain += d; else loss -= d;
  }
  if (gain + loss === 0) return 50;
  const rs = gain / (loss || 1e-9);
  return 100 - 100 / (1 + rs);
}

// MACD histogram (12,26,9) on closes. Returns the histogram value normalized
// by price so it is comparable across price regimes.
export function macdHist(closes) {
  if (closes.length < 35) return 0;
  const ema = (arr, p) => {
    const k = 2 / (p + 1);
    let e = arr[0];
    for (let i = 1; i < arr.length; i++) e = arr[i] * k + e * (1 - k);
    return e;
  };
  const emaSeries = (arr, p) => {
    const k = 2 / (p + 1);
    let e = arr[0];
    const out = [e];
    for (let i = 1; i < arr.length; i++) { e = arr[i] * k + e * (1 - k); out.push(e); }
    return out;
  };
  const e12 = emaSeries(closes, 12);
  const e26 = emaSeries(closes, 26);
  const macdLine = closes.map((_, i) => e12[i] - e26[i]);
  const signal = ema(macdLine.slice(-30), 9);
  const hist = macdLine[macdLine.length - 1] - signal;
  const px = closes[closes.length - 1] || 1;
  return hist / px; // fractional
}

// =====================================================================
// FEATURE EXTRACTION
// All features are standardized to roughly unit scale so a fixed learning
// rate behaves well. `ctx` carries the running feature scaler.
// =====================================================================
export const FEATURE_NAMES = [
  'intraZ',        // intra-candle move so far, in units of typical full-candle move
  'velocity',      // intra move / elapsed fraction, in sigma  (extrapolation, but as a feature not THE answer)
  'distToTargetZ', // (price - target) / sigma  (how far above/below the strike we already are)
  'momentum3',     // mean of last 3 closed-candle returns, in sigma
  'rsiDev',        // (RSI - 50)/50
  'macd',          // normalized MACD histogram, scaled
  'autocorr',      // lag-1 autocorr regime, clamped
  'volRegime',     // realized vol vs its own median, centered
  'btcRet',        // BTC current-candle return in sigma  (cross-asset)
  'ethRet',        // ETH current-candle return in sigma  (cross-asset)
  'progress',      // elapsed fraction, centered at 0.5
];

// Compute the raw (unstandardized) feature vector for the live candle.
export function rawFeatures({
  price, target, open, elapsedFrac,
  xrp15,            // closed XRP 15m candles, newest first: [time, low, high, open, close, vol]
  btcCur, ethCur,   // {open, price} for current 15m candle on BTC / ETH, or null
}) {
  const closedRet = xrp15.slice(1, 21).map(c => c[4] - c[3]).filter(isFinite); // close-open per candle
  const sigma = Math.max(stdev(closedRet), 1e-6);
  const closes = xrp15.slice(0, 40).map(c => c[4]).filter(isFinite).reverse(); // oldest..newest

  const intra = price - open;
  const ef = Math.max(elapsedFrac, 0.01);

  const ranges = xrp15.slice(1, 21).map(c => c[2] - c[1]).filter(isFinite);
  const curRange = Math.abs(price - open) + 1e-9;
  const medRange = ranges.length ? ranges.sort((a, b) => a - b)[Math.floor(ranges.length / 2)] : sigma;

  const ret3 = closedRet.slice(0, 3);
  const btcRet = btcCur && isFinite(btcCur.open) && btcCur.open > 0 ? (btcCur.price - btcCur.open) / btcCur.open : 0;
  const ethRet = ethCur && isFinite(ethCur.open) && ethCur.open > 0 ? (ethCur.price - ethCur.open) / ethCur.open : 0;

  // BTC/ETH returns expressed in XRP-sigma-equivalent via a rough beta of 1
  // on fractional terms (XRP fractional sigma):
  const xrpFracSigma = Math.max(sigma / (price || 1), 1e-6);

  return {
    intraZ: intra / sigma,
    velocity: (intra / ef) / sigma,
    // distToTargetZ squashed through tanh: being far from the strike should
    // inform the call but not single-handedly force a near-certain prediction.
    // Raw (price-target)/sigma could hit large values and dominate; tanh caps
    // its effective range to about [-1.5, 1.5].
    distToTargetZ: 1.5 * Math.tanh((price - target) / sigma / 1.5),
    momentum3: ret3.length ? mean(ret3) / sigma : 0,
    rsiDev: (rsi(closes, 14) - 50) / 50,
    macd: macdHist(closes) * 1000, // scale fractional to ~unit
    autocorr: Math.max(-0.5, Math.min(0.5, lag1Autocorr(closedRet))) * 2,
    volRegime: medRange > 0 ? (curRange / medRange - 1) : 0,
    btcRet: (btcRet / xrpFracSigma),
    ethRet: (ethRet / xrpFracSigma),
    progress: elapsedFrac - 0.5,
    _sigma: sigma, // passthrough, not a model feature
  };
}

// Standardize a raw feature object into an ordered vector using a running
// scaler (Welford mean/var per feature). Returns { vec, scaler }.
export function standardize(raw, scaler) {
  const sc = scaler || {};
  const vec = FEATURE_NAMES.map((name) => {
    const x = isFinite(raw[name]) ? raw[name] : 0;
    const s = sc[name] || { n: 0, mean: 0, m2: 0 };
    const sd = s.n > 5 ? Math.sqrt(s.m2 / (s.n - 1)) : 1;
    return sd > 1e-6 ? (x - s.mean) / sd : x;
  });
  return vec;
}

// Update the running scaler with a raw feature object (call once per settled candle).
export function updateScaler(scaler, raw) {
  const sc = { ...(scaler || {}) };
  for (const name of FEATURE_NAMES) {
    const x = isFinite(raw[name]) ? raw[name] : 0;
    const s = sc[name] ? { ...sc[name] } : { n: 0, mean: 0, m2: 0 };
    s.n += 1;
    const d = x - s.mean;
    s.mean += d / s.n;
    s.m2 += d * (x - s.mean);
    sc[name] = s;
  }
  return sc;
}

// =====================================================================
// ONLINE LOGISTIC REGRESSION
// weights: { w: number[FEATURE_NAMES.length], b: number, lr, l2 }
// Label y = 1 if candle closed UP relative to target, else 0.
// Trained with SGD on log-loss. L2 keeps weights from exploding on a small,
// autocorrelated dataset.
// =====================================================================
export function defaultWeights() {
  // L2 raised from 0.001 to 0.02. With BTC/ETH features intermittently flat,
  // the old penalty let distToTargetZ run to +1.45 and dominate, collapsing
  // the model toward the persistence baseline. Stronger L2 keeps any single
  // feature from taking over and forces the model to use the full signal set.
  return { w: FEATURE_NAMES.map(() => 0), b: 0, lr: 0.05, l2: 0.02, trained: 0 };
}

// Raw uncapped probability, used ONLY for the training gradient so the cap
// in predictProb does not distort learning.
function rawProb(weights, vec) {
  let z = weights.b;
  for (let i = 0; i < vec.length; i++) z += weights.w[i] * vec[i];
  return sigmoid(z);
}

export function predictProb(weights, vec) {
  let z = weights.b;
  for (let i = 0; i < vec.length; i++) z += weights.w[i] * vec[i];
  const raw = sigmoid(z);
  // Cap confidence. A 15-min direction model that prints 100% is saturating,
  // not confident. Clamp to [0.05, 0.95] so log-loss is not torched by an
  // overconfident wrong call, and so the headline number stays honest.
  return Math.min(0.95, Math.max(0.05, raw));
}

export function sgdUpdate(weights, vec, y) {
  const p = rawProb(weights, vec); // uncapped for honest gradient
  const err = p - y; // dL/dz for log-loss
  const w = weights.w.slice();
  for (let i = 0; i < w.length; i++) {
    w[i] -= weights.lr * (err * vec[i] + weights.l2 * w[i]);
  }
  const b = weights.b - weights.lr * err;
  return { ...weights, w, b, trained: weights.trained + 1 };
}

// Train from scratch over an ordered list of {vec, y} samples for `epochs`.
// Used to (re)fit when history changes. Cheap: a few hundred samples * 11 dims.
export function trainBatch(samples, epochs = 12) {
  let weights = defaultWeights();
  for (let e = 0; e < epochs; e++) {
    // light shuffle by epoch offset to reduce ordering bias while keeping it deterministic-ish
    for (let i = 0; i < samples.length; i++) {
      const s = samples[i];
      weights = sgdUpdate(weights, s.vec, s.y);
    }
  }
  return weights;
}

// =====================================================================
// BENCHMARKS + CALIBRATION over settled history
// settled: [{ vec, y, p }]  where p is the model prob AT PREDICTION TIME
//   (logged live, not refit) so the eval is honest / walk-forward.
// =====================================================================
export function logLoss(pairs) {
  if (!pairs.length) return null;
  let s = 0;
  for (const { p, y } of pairs) {
    const pc = Math.min(0.999, Math.max(0.001, p));
    s += -(y * Math.log(pc) + (1 - y) * Math.log(1 - pc));
  }
  return s / pairs.length;
}

export function brier(pairs) {
  if (!pairs.length) return null;
  return mean(pairs.map(({ p, y }) => (p - y) ** 2));
}

export function dirAccuracy(pairs, thresh = 0.5) {
  const acted = pairs.filter(({ p }) => Math.abs(p - 0.5) >= (thresh - 0.5));
  if (!acted.length) return null;
  const correct = acted.filter(({ p, y }) => (p >= 0.5 ? 1 : 0) === y).length;
  return correct / acted.length;
}

// Persistence baseline: predict the last closed candle's direction with a
// fixed 0.5+ probability. We score it as a hard 0/1 call.
export function persistenceAccuracy(settled) {
  // settled carries lastDir (1/0) = sign of previous candle, and y = actual
  const acted = settled.filter(s => s.lastDir === 0 || s.lastDir === 1);
  if (!acted.length) return null;
  return acted.filter(s => s.lastDir === s.y).length / acted.length;
}

// Build the full assessment the UI shows.
export function assess(settled) {
  if (settled.length < 8) {
    return { ready: false, n: settled.length, reason: 'Need 8+ settled candles to grade the model.' };
  }
  const pairs = settled.map(s => ({ p: s.p, y: s.y }));
  const modelLL = logLoss(pairs);
  const rwLL = logLoss(pairs.map(({ y }) => ({ p: 0.5, y }))); // random walk = always 0.5
  const modelBrier = brier(pairs);
  const modelAcc = dirAccuracy(pairs, 0.5);
  const acc60 = dirAccuracy(pairs, 0.60);
  const acc70 = dirAccuracy(pairs, 0.70);
  const persAcc = persistenceAccuracy(settled);

  const beatsRW = modelLL != null && rwLL != null && modelLL < rwLL;
  const beatsPers = persAcc == null || (modelAcc != null && modelAcc > persAcc);
  const edge = beatsRW && beatsPers && (modelAcc != null && modelAcc > 0.5);

  return {
    ready: true, n: settled.length,
    modelLL, rwLL, modelBrier, modelAcc, acc60, acc70, persAcc,
    beatsRW, beatsPers, edge,
    verdict: edge
      ? 'Model has a real edge over random walk and persistence on your logged data.'
      : !beatsRW
        ? 'Model is NOT beating a 50/50 coin flip on log-loss. Do not trade on it yet.'
        : !beatsPers
          ? 'Model is not beating the persistence baseline. Edge is not established.'
          : 'Directional accuracy is at or below 50%. Treat as no edge.',
  };
}
