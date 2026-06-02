# XRP 15m Directional Tracker (v7)

Predicts **P(XRP closes up at the next 15-minute mark)** as a calibrated
probability, for paper-testing Kalshi up/down trades. Built with Vite + React.

This is the directional rebuild of the earlier price-level estimator. Instead
of guessing a price and reading direction off it (which chased random-walk
noise), it outputs a probability from an online logistic model trained on your
own settled candles, using BTC/ETH cross-asset and momentum features. It is
benchmarked against random walk and persistence every refresh, and tells you
when it has no edge.

Paper trading only. No real orders. All state lives in your browser
(localStorage).

## Run locally

Requires Node 18+.

```bash
npm install
npm run dev
```

Open the localhost URL it prints. Leave it running: for the first ~8 settled
15-min candles (~2 hours) it shows a clearly-labeled heuristic, not a real
prediction. After that the model starts grading itself in the "Edge vs
baselines" panel.

## Deploy to Vercel

1. Push this folder to a new GitHub repo (see below).
2. On vercel.com: **Add New → Project**, import the repo.
3. Vercel auto-detects Vite. Defaults are correct:
   - Framework: Vite
   - Build command: `npm run build`
   - Output directory: `dist`
4. **Deploy.** Every push to `main` auto-redeploys.

### Push to GitHub

```bash
git init
git add .
git commit -m "XRP 15m directional tracker v7"
git remote add origin https://github.com/YOUR_USERNAME/xrp-tracker.git
git branch -M main
git push -u origin main
```

## If exchanges show as unavailable on the live site

The app calls Coinbase / Kraken / Bitstamp / Kalshi directly from the browser.
Those sometimes block cross-origin requests from a deployed domain even when
localhost works fine. If that happens, a serverless proxy at `api/proxy.js`
fixes it; ask and it can be wired in. Don't add it preemptively.

## How to use it for Kalshi

1. Wait for the **Edge vs baselines** panel to go green. Until then, the honest
   read is: no proven edge, don't trade.
2. Type your Kalshi market's exact strike into the **Kalshi strike override** so
   the probability is computed against the real contract line.
3. Enter the matching side (Yes if UP, No if DOWN) only when the panel is green
   AND the probability clears your threshold AND you're inside the entry window.

Realistic ceiling for 15-min XRP direction is ~52–55% after costs. If the panel
never goes green on your data, that is the tool working correctly, not a bug.
