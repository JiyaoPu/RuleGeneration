const express = require("express");
const { BlobServiceClient } = require("@azure/storage-blob");
const { DefaultAzureCredential } = require("@azure/identity");

const app = express();
const port = process.env.PORT || 8080;

// ====== CORS (Route B) ======
const ALLOWED_ORIGIN =
  process.env.ALLOWED_ORIGIN || "https://icy-mud-07f3ea903.2.azurestaticapps.net";

app.use((req, res, next) => {
  res.setHeader("Access-Control-Allow-Origin", ALLOWED_ORIGIN);
  res.setHeader("Vary", "Origin");
  res.setHeader("Access-Control-Allow-Methods", "GET,OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  res.setHeader("Access-Control-Max-Age", "86400");
  if (req.method === "OPTIONS") return res.sendStatus(204);
  next();
});

// ====== Blob config ======
const STORAGE_ACCOUNT = process.env.STORAGE_ACCOUNT || "rgnspace3954763138";
const METRICS_CONTAINER = process.env.METRICS_CONTAINER || "rgnresults";

// Use env vars so you can change without redeploy
const LATEST_BLOB = process.env.LATEST_BLOB || "latest/metrics.json";
const RUN_BLOB = process.env.RUN_BLOB || "run/metrics.json"; // adjust if needed

function getBlobClient(blobPath) {
  const credential = new DefaultAzureCredential(); // uses Managed Identity in App Service
  const service = new BlobServiceClient(
    `https://${STORAGE_ACCOUNT}.blob.core.windows.net`,
    credential
  );
  return service.getContainerClient(METRICS_CONTAINER).getBlobClient(blobPath);
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

// Health
app.get("/health", (req, res) => {
  res.json({ ok: true, time: new Date().toISOString() });
});

// ✅ latest from Blob
app.get("/api/latest", async (req, res) => {
  try {
    const blobClient = getBlobClient(LATEST_BLOB);
    const props = await blobClient.getProperties(); // useful to verify lastModified/etag

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

app.listen(port, () => {
  console.log(`Backend listening on port ${port}`);
  console.log(`ALLOWED_ORIGIN=${ALLOWED_ORIGIN}`);
  console.log(`STORAGE_ACCOUNT=${STORAGE_ACCOUNT}`);
  console.log(`METRICS_CONTAINER=${METRICS_CONTAINER}`);
  console.log(`LATEST_BLOB=${LATEST_BLOB}`);
  console.log(`RUN_BLOB=${RUN_BLOB}`);
});