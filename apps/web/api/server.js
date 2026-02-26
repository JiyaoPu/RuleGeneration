const express = require("express");

const app = express();
const port = process.env.PORT || 8080;

// SWA 静态站点域名（后续可用环境变量切换）
const SWA_ORIGIN =
  process.env.SWA_ORIGIN || "https://icy-mud-07f3ea903.2.azurestaticapps.net";

// ✅ 路线B关键：允许前端跨域访问后端（CORS）
// 建议在 App Service 环境变量里设置：ALLOWED_ORIGIN=https://icy-mud-07f3ea903.2.azurestaticapps.net
const ALLOWED_ORIGIN =
  process.env.ALLOWED_ORIGIN || "https://icy-mud-07f3ea903.2.azurestaticapps.net";

// CORS middleware（最小且安全：只放行一个 origin）
app.use((req, res, next) => {
  res.setHeader("Access-Control-Allow-Origin", ALLOWED_ORIGIN);
  res.setHeader("Vary", "Origin");
  res.setHeader("Access-Control-Allow-Methods", "GET,OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  res.setHeader("Access-Control-Max-Age", "86400"); // 预检缓存 24h

  // 处理预检请求
  if (req.method === "OPTIONS") return res.sendStatus(204);
  next();
});

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

// /api/run
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
  console.log(`SWA_ORIGIN=${SWA_ORIGIN}`);
  console.log(`ALLOWED_ORIGIN=${ALLOWED_ORIGIN}`);
});