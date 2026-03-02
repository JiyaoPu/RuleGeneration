// server.js

const { spawn } = require("child_process");
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
  res.setHeader("Access-Control-Allow-Origin", ALLOWED_ORIGIN);
  res.setHeader("Vary", "Origin");

  res.setHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");

  const requestedHeaders = req.headers["access-control-request-headers"];
  res.setHeader(
    "Access-Control-Allow-Headers",
    requestedHeaders || "Content-Type, Cache-Control, Accept"
  );

  res.setHeader("Access-Control-Max-Age", "86400");
  if (req.method === "OPTIONS") return res.sendStatus(204);

  next();
});

// ✅ Parse JSON bodies for POST
app.use(express.json({ limit: "2mb" }));

// ====== Blob config ======
const STORAGE_ACCOUNT = process.env.STORAGE_ACCOUNT || "rgnspace3954763138";
const METRICS_CONTAINER = process.env.METRICS_CONTAINER || "rgnresults";

const LATEST_BLOB = process.env.LATEST_BLOB || "latest/metrics.json";
const RUN_BLOB = process.env.RUN_BLOB || "run/metrics.json";

// Where to store settings (fixed path for now)
const RUN_SETTINGS_JSON = process.env.RUN_SETTINGS_JSON || "run/settings.json";
const RUN_SETTINGS_TXT = process.env.RUN_SETTINGS_TXT || "run/settings.txt";

// IMPORTANT: 你 AML job 里 input settings_json 用的是 datastore path
// 你之前已经建了 datastore: rgnresults_ds -> container rgnresults
// 所以这里固定生成对应的 AML URI（无需 UI 选 Datastore）
const AML_SETTINGS_URI =
  process.env.AML_SETTINGS_URI ||
  "azureml://datastores/rgnresults_ds/paths/run/settings.json";

// Create once
const credential = new DefaultAzureCredential();
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

async function saveRunSettingsToBlob(settings) {
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
}

// Health
app.get("/health", (req, res) => {
  res.json({ ok: true, time: new Date().toISOString() });
});

// ✅ latest from Blob
app.get("/api/latest", async (req, res) => {
  try {
    const blobClient = getBlobClient(LATEST_BLOB);
    const props = await blobClient.getProperties();
    const text = await downloadBlobText(LATEST_BLOB);

    res.set("Cache-Control", "no-store");
    res.set("X-Blob-Path", LATEST_BLOB);
    res.set("X-Blob-ETag", String(props.etag || ""));
    res.set("X-Blob-Last-Modified", props.lastModified?.toISOString?.() || "");

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

// ✅ run from Blob (optional, still keep)
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

// ✅ save settings into Blob (run/settings.json + run/settings.txt)
app.post("/api/settings", async (req, res) => {
  try {
    const settings = req.body;
    if (!settings || typeof settings !== "object" || Array.isArray(settings)) {
      return res
        .status(400)
        .json({ error: "Body must be a JSON object (settings)." });
    }

    await saveRunSettingsToBlob(settings);

    res.set("Cache-Control", "no-store");
    res.json({
      ok: true,
      container: METRICS_CONTAINER,
      saved: [RUN_SETTINGS_JSON, RUN_SETTINGS_TXT],
      aml_settings_uri: AML_SETTINGS_URI,
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

// ✅ Start Azure ML job (Run button -> Job)
app.post("/api/run", async (req, res) => {
  try {
    const settings = req.body;
    if (!settings || typeof settings !== "object" || Array.isArray(settings)) {
      return res
        .status(400)
        .json({ error: "Body must be a JSON object (settings)." });
    }

    // (1) Save settings to blob FIRST
    await saveRunSettingsToBlob(settings);

    // (2) Submit AML job via python script
    // ✅ App Service 上常见只有 python3，没有 python；同时允许用环境变量覆盖
    const PYTHON_BIN = process.env.PYTHON_BIN || "python3";

    const py = spawn(PYTHON_BIN, ["scripts/submit_aml_job.py"], {
      env: {
        ...process.env,
        AML_SETTINGS_URI,
      },
      stdio: ["pipe", "pipe", "pipe"],
    });

    let out = "";
    let err = "";

    // ✅ 关键：捕获 spawn 错误（比如 ENOENT：找不到 python/python3）
    py.on("error", (e) => {
      console.error("spawn python error:", e);

      // 避免重复响应
      if (res.headersSent) return;

      return res.status(500).json({
        ok: false,
        error: "failed to start submit_aml_job.py",
        python_bin: PYTHON_BIN,
        code: e.code,
        message: e.message,
        hint: "App Service 环境里可能没有 python(或python3)。可在 Configuration 里设置 PYTHON_BIN=/usr/bin/python3 或改为用 Azure ML REST API 触发 Job。",
      });
    });

    py.stdout.on("data", (d) => (out += d.toString("utf-8")));
    py.stderr.on("data", (d) => (err += d.toString("utf-8")));

    py.on("close", (code) => {
      if (res.headersSent) return;

      if (code !== 0) {
        return res.status(500).json({
          ok: false,
          error: "submit AML job failed",
          python_bin: PYTHON_BIN,
          exitCode: code,
          stderr: err.slice(-4000),
          stdout: out.slice(-4000),
        });
      }

      let payload = null;
      try {
        payload = JSON.parse(out.trim());
      } catch (e) {
        return res.status(500).json({
          ok: false,
          error: "submit AML job returned non-JSON",
          python_bin: PYTHON_BIN,
          stdout: out.slice(-4000),
          stderr: err.slice(-4000),
        });
      }

      res.set("Cache-Control", "no-store");
      return res.json({
        ok: true,
        saved_settings: [RUN_SETTINGS_JSON, RUN_SETTINGS_TXT],
        aml_settings_uri: AML_SETTINGS_URI,
        python_bin: PYTHON_BIN,
        ...payload,
      });
    });

    // stdin 可选：如果你的 python 脚本会读 stdin
    try {
      py.stdin.write(JSON.stringify(settings));
      py.stdin.end();
    } catch (e) {
      // 如果 spawn 失败，这里可能也会报错；但我们已经在 py.on('error') 里处理响应
      console.error("stdin write error:", e);
    }
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// ---- START SERVER (MUST BE LAST) ----
app.listen(port, () => {
  console.log(`Backend listening on port ${port}`);
  console.log(`ALLOWED_ORIGIN=${ALLOWED_ORIGIN}`);
  console.log(`STORAGE_ACCOUNT=${STORAGE_ACCOUNT}`);
  console.log(`METRICS_CONTAINER=${METRICS_CONTAINER}`);
  console.log(`LATEST_BLOB=${LATEST_BLOB}`);
  console.log(`RUN_BLOB=${RUN_BLOB}`);
  console.log(`RUN_SETTINGS_JSON=${RUN_SETTINGS_JSON}`);
  console.log(`RUN_SETTINGS_TXT=${RUN_SETTINGS_TXT}`);
  console.log(`AML_SETTINGS_URI=${AML_SETTINGS_URI}`);
});
