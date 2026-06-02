// Serverless proxy so the browser can reach exchange APIs that block
// cross-origin requests from a deployed domain. Only whitelisted hosts.
export default async function handler(req, res) {
  const { url } = req.query;
  const allowed = [
    'api.exchange.coinbase.com',
    'api.kraken.com',
    'www.bitstamp.net',
    'api.elections.kalshi.com',
  ];
  if (!url) { res.status(400).json({ error: 'missing url' }); return; }
  let target;
  try { target = new URL(url); } catch { res.status(400).json({ error: 'bad url' }); return; }
  if (!allowed.includes(target.hostname)) { res.status(403).json({ error: 'host not allowed' }); return; }
  try {
    const r = await fetch(url, { headers: { 'User-Agent': 'xrp-tracker' } });
    const text = await r.text();
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Content-Type', 'application/json');
    // pass through status so the client can see upstream errors
    res.status(r.status).send(text);
  } catch (e) {
    res.status(502).json({ error: e.message });
  }
}
