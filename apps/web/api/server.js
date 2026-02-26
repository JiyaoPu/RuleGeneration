const express = require("express");

const app = express();
const port = process.env.PORT || 8080;

// 你的 SWA 域名（后续可换成 env）
const SWA_ORIGIN = process.env.SWA_ORIGIN || "https://icy-mud-07f3ea903.2.azurestaticapps.net";

async function fetchJson(url) {
  const r = await fetch(url, { headers: { "Cache-Control": "no-cache" } });
  if (!r.ok) {
    const text = await r.text();
    const err = new Error(`Fetch failed: ${r.status} ${r.statusText}`);
    err.detail = text.slice(0, 300);
    throw err;
  }
  return await r.json();
}

// 健康检查
app.get("/health", (req, res) => {
  res.json({ ok: true, time: new Date().toISOString() });
});

// /api/latest
app.get("/api/latest", async (req, res) => {
  try {
    const url = `${SWA_ORIGIN}/data/latest/metrics.json?ts=${Date.now()}`;
    const data = await fetchJson(url);
    res.set("Cache-Control", "no-store");
    res.json(data);
  } catch (e) {
    res.status(500).json({ error: e.message, detail: e.detail || null });
  }
});

// /api/run（如果你也要）
app.get("/api/run", async (req, res) => {
  try {
    const url = `${SWA_ORIGIN}/data/run/metrics.json?ts=${Date.now()}`;
    const data = await fetchJson(url);
    res.set("Cache-Control", "no-store");
    res.json(data);
  } catch (e) {
    res.status(500).json({ error: e.message, detail: e.detail || null });
  }
});

app.listen(port, () => {
  console.log(`Backend listening on port ${port}`);
});