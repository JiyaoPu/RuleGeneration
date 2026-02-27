// server.js
const express = require("express");
const { BlobServiceClient } = require("@azure/storage-blob");
const { DefaultAzureCredential } = require("@azure/identity");

const app = express();
const port = process.env.PORT || 8080;

// ====== CORS (Route B) ======
const ALLOWED_ORIGIN =
  process.env.ALLOWED_ORIGIN ||
  "https://icy-mud-07f3ea903.2.azurestaticapps.net";

app.use((req, res, next) => {
  // Allow only your SWA origin
  res.setHeader("Access-Control-Allow-Origin", ALLOWED_ORIGIN);
  res.setHeader("Vary", "Origin");

  // Methods used now (and future-proof POST)
  res.setHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");

  // Allow requested headers (preflight)
  const requestedHeaders = req.headers["access-control-request-headers"];
  res.setHeader(
    "Access-Control-Allow-Headers",
    requestedHeaders || "Content-Type, Cache-Control, Accept"
  );

  // Cache preflight response
  res.setHeader("Access-Control-Max-Age", "86400");

  // Handle preflight quickly
  if (req.method === "OPTIONS") return res.sendStatus(204);

  next();
});

// ✅ Parse JSON bodies for POST
app.use(express.json({ limit: "2mb" }));

// ====== Blob config ======
const STORAGE_ACCOUNT = process.env.STORAGE_ACCOUNT || "rgnspace3954763138";
const METRICS_CONTAINER = process.env.METRICS_CONTAINER || "rgnresults";

// Use env vars so you can change without redeploy
const LATEST_BLOB = process.env.LATEST_BLOB || "latest/metrics.json";
const RUN_BLOB = process.env.RUN_BLOB || "run/metrics.json"; // adjust if needed

// Where to store settings (fixed path for now)
const RUN_SETTINGS_JSON = process.env.RUN_SETTINGS_JSON || "run/settings.json";
const RUN_SETTINGS_TXT = process.env.RUN_SETTINGS_TXT || "run/settings.txt";

// Create once (avoid creating credential/client per request)
const credential = new DefaultAzureCredential(); // uses Managed Identity in App Service
const blobServiceClient = new BlobServiceClient(
  `https://${STORAGE_ACCOUNT}.blob.core.windows.net`,
  credential
);

function getContainerClient() {
  return blobServiceClient.getContainerClient(METRICS_CONTAINER);
}

function getBlobClient(blobPath) {
  return getContainerClient().getBlobClient(blobPath);
}

async function ensureContainer() {
  const containerClient = getContainerClient();
  await containerClient.createIfNotExists();
  return containerClient;
}

async function downloadBlobText(blobPath) {
  const blobClient = getBlobClient(blobPath);
  const resp = await blobClient.download();
  return await streamToString(resp.readableStreamBody);
}

function streamToString(readable) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    readable.on("data", (d) => chunks.push(d));
    readable.on("end", () => resolve(Buffer.concat(chunks).toString("utf-8")));
    readable.on("error", reject);
  });
}

function settingsToTxt(settings) {
  // Keep same style as your previous settings.txt: "key: value"
  // Order follows insertion order of object keys (browser usually preserves).
  return (
    Object.entries(settings || {})
      .map(([k, v]) => `${k}: ${String(v)}`)
      .join("\n") + "\n"
  );
}

async function uploadBlobText(blobPath, text, contentType) {
  const containerClient = await ensureContainer();
  const blockBlob = containerClient.getBlockBlobClient(blobPath);
  await blockBlob.upload(text, Buffer.byteLength(text), {
    blobHTTPHeaders: { blobContentType: contentType },
  });
}

// Health
app.get("/health", (req, res) => {
  res.json({ ok: true, time: new Date().toISOString() });
});

// ✅ latest from Blob
app.get("/api/latest", async (req, res) => {
  try {
    const blobClient = getBlobClient(LATEST_BLOB);
    const props = await blobClient.getProperties(); // lastModified/etag verification

    const text = await downloadBlobText(LATEST_BLOB);

    res.set("Cache-Control", "no-store");
    res.set("X-Blob-Path", LATEST_BLOB);
    res.set("X-Blob-ETag", String(props.etag || ""));
    res.set("X-Blob-Last-Modified", props.lastModified?.toISOString?.() || "");

    // Return raw JSON text (avoid double parse issues)
    res.type("application/json").send(text);
  } catch (e) {
    res.status(500).json({
      error: e.message,
      storage: STORAGE_ACCOUNT,
      container: METRICS_CONTAINER,
      blob: LATEST_BLOB,
    });
  }
});

// ✅ run from Blob (optional)
app.get("/api/run", async (req, res) => {
  try {
    const blobClient = getBlobClient(RUN_BLOB);
    const props = await blobClient.getProperties();

    const text = await downloadBlobText(RUN_BLOB);

    res.set("Cache-Control", "no-store");
    res.set("X-Blob-Path", RUN_BLOB);
    res.set("X-Blob-ETag", String(props.etag || ""));
    res.set("X-Blob-Last-Modified", props.lastModified?.toISOString?.() || "");

    res.type("application/json").send(text);
  } catch (e) {
    res.status(500).json({
      error: e.message,
      storage: STORAGE_ACCOUNT,
      container: METRICS_CONTAINER,
      blob: RUN_BLOB,
    });
  }
});

// ✅ NEW: save settings into Blob (run/settings.json + run/settings.txt)
app.post("/api/settings", async (req, res) => {
  try {
    const settings = req.body;

    if (!settings || typeof settings !== "object" || Array.isArray(settings)) {
      return res.status(400).json({ error: "Body must be a JSON object (settings)." });
    }

    // 1) JSON
    await uploadBlobText(
      RUN_SETTINGS_JSON,
      JSON.stringify(settings, null, 2),
      "application/json; charset=utf-8"
    );

    // 2) TXT
    await uploadBlobText(
      RUN_SETTINGS_TXT,
      settingsToTxt(settings),
      "text/plain; charset=utf-8"
    );

    res.set("Cache-Control", "no-store");
    res.json({
      ok: true,
      container: METRICS_CONTAINER,
      saved: [RUN_SETTINGS_JSON, RUN_SETTINGS_TXT],
      time: new Date().toISOString(),
    });
  } catch (e) {
    res.status(500).json({
      error: e.message,
      storage: STORAGE_ACCOUNT,
      container: METRICS_CONTAINER,
      saved: [RUN_SETTINGS_JSON, RUN_SETTINGS_TXT],
    });
  }
});

app.listen(port, () => {
  console.log(`Backend listening on port ${port}`);
  console.log(`ALLOWED_ORIGIN=${ALLOWED_ORIGIN}`);
  console.log(`STORAGE_ACCOUNT=${STORAGE_ACCOUNT}`);
  console.log(`METRICS_CONTAINER=${METRICS_CONTAINER}`);
  console.log(`LATEST_BLOB=${LATEST_BLOB}`);
  console.log(`RUN_BLOB=${RUN_BLOB}`);
  console.log(`RUN_SETTINGS_JSON=${RUN_SETTINGS_JSON}`);
  console.log(`RUN_SETTINGS_TXT=${RUN_SETTINGS_TXT}`);
});