// server.js

const express = require("express");
const { BlobServiceClient } = require("@azure/storage-blob");
const { DefaultAzureCredential } = require("@azure/identity");

const app = express();
const port = process.env.PORT || 8080;

// ====== CORS ======
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

// JSON body
app.use(express.json({ limit: "2mb" }));

// ====== Blob config ======
const STORAGE_ACCOUNT = process.env.STORAGE_ACCOUNT || "rgnspace3954763138";
const METRICS_CONTAINER = process.env.METRICS_CONTAINER || "rgnresults";

const LATEST_BLOB = process.env.LATEST_BLOB || "latest/metrics.json";
const RUN_BLOB = process.env.RUN_BLOB || "run/metrics.json";

const RUN_SETTINGS_JSON = process.env.RUN_SETTINGS_JSON || "run/settings.json";
const RUN_SETTINGS_TXT = process.env.RUN_SETTINGS_TXT || "run/settings.txt";

const AML_SETTINGS_URI =
  process.env.AML_SETTINGS_URI ||
  "azureml://datastores/rgnresults_ds/paths/run/settings.json";

// ====== Azure ML config ======
const AZ_SUBSCRIPTION_ID = process.env.AZ_SUBSCRIPTION_ID;
const AZ_RESOURCE_GROUP = process.env.AZ_RESOURCE_GROUP;
const AZ_ML_WORKSPACE = process.env.AZ_ML_WORKSPACE;

const AZ_ML_COMPUTE = process.env.AZ_ML_COMPUTE || "RGN-Compute-Cluster";

// 已注册 component（你现在应填：rgn_train_component / 2）
const AZ_ML_COMPONENT_NAME =
  process.env.AZ_ML_COMPONENT_NAME || "rgn_train_component";
const AZ_ML_COMPONENT_VERSION =
  process.env.AZ_ML_COMPONENT_VERSION || "2";

const AZ_ML_DATASTORE = process.env.AZ_ML_DATASTORE || "rgnresults_ds";

// REST API version
const AZ_ML_API_VERSION = process.env.AZ_ML_API_VERSION || "2025-12-01";

// ====== Clients ======
const credential = new DefaultAzureCredential();
const blobServiceClient = new BlobServiceClient(
  `https://${STORAGE_ACCOUNT}.blob.core.windows.net`,
  credential
);

// ====== Blob helpers ======
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
  await uploadBlobText(
    RUN_SETTINGS_JSON,
    JSON.stringify(settings, null, 2),
    "application/json; charset=utf-8"
  );

  await uploadBlobText(
    RUN_SETTINGS_TXT,
    settingsToTxt(settings),
    "text/plain; charset=utf-8"
  );
}

// ====== AML REST helpers ======
function requireAmlEnv() {
  const missing = [];
  if (!AZ_SUBSCRIPTION_ID) missing.push("AZ_SUBSCRIPTION_ID");
  if (!AZ_RESOURCE_GROUP) missing.push("AZ_RESOURCE_GROUP");
  if (!AZ_ML_WORKSPACE) missing.push("AZ_ML_WORKSPACE");

  if (missing.length) {
    const err = new Error(`Missing AML env vars: ${missing.join(", ")}`);
    err.code = "MISSING_AML_ENV";
    throw err;
  }
}

function amlResourceId(...parts) {
  return [
    "",
    "subscriptions",
    AZ_SUBSCRIPTION_ID,
    "resourceGroups",
    AZ_RESOURCE_GROUP,
    "providers",
    "Microsoft.MachineLearningServices",
    "workspaces",
    AZ_ML_WORKSPACE,
    ...parts,
  ].join("/");
}

async function getArmToken() {
  const tok = await credential.getToken("https://management.azure.com/.default");
  if (!tok?.token) throw new Error("Failed to acquire ARM token");
  return tok.token;
}

function makeJobName(prefix = "webrun") {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  const ts =
    d.getUTCFullYear() +
    pad(d.getUTCMonth() + 1) +
    pad(d.getUTCDate()) +
    pad(d.getUTCHours()) +
    pad(d.getUTCMinutes()) +
    pad(d.getUTCSeconds());
  return `${prefix}_${ts}`;
}

function buildComponentJobBody({ jobName, settingsUri }) {
  const computeId = amlResourceId("computes", AZ_ML_COMPUTE);
  const componentId = amlResourceId(
    "components",
    AZ_ML_COMPONENT_NAME,
    "versions",
    AZ_ML_COMPONENT_VERSION
  );

  const outputsUri = `azureml://datastores/${AZ_ML_DATASTORE}/paths/jobs/${jobName}/`;

  return {
    properties: {
      jobType: "Pipeline",
      displayName: jobName,
      experimentName: "web_run",

      jobs: {
        train_step: {
          componentId,
          computeId,
          inputs: {
            settings_json: {
              jobInputType: "uri_file",
              uri: settingsUri,
            },
          },
          outputs: {
            outputs_dir: {
              jobOutputType: "uri_folder",
              mode: "Upload",
              uri: outputsUri,
            },
          },
        },
      },

      settings: {
        continueOnStepFailure: false,
      },
    },
  };
}

async function submitAmlComponentJob({ jobName, settingsUri }) {
  requireAmlEnv();

  const url =
    "https://management.azure.com" +
    amlResourceId("jobs", jobName) +
    `?api-version=${encodeURIComponent(AZ_ML_API_VERSION)}`;

  const token = await getArmToken();
  const body = buildComponentJobBody({ jobName, settingsUri });

  const resp = await fetch(url, {
    method: "PUT",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  const text = await resp.text();
  let json = null;
  try {
    json = text ? JSON.parse(text) : null;
  } catch (_) {}

  if (!resp.ok) {
    const err = new Error(
      `AML component job submit failed: HTTP ${resp.status} ${resp.statusText}`
    );
    err.details = text?.slice(0, 4000);
    err.httpStatus = resp.status;
    throw err;
  }

  return json;
}

async function getAmlJob(jobName) {
  requireAmlEnv();

  const url =
    "https://management.azure.com" +
    amlResourceId("jobs", jobName) +
    `?api-version=${encodeURIComponent(AZ_ML_API_VERSION)}`;

  const token = await getArmToken();
  const resp = await fetch(url, {
    method: "GET",
    headers: { Authorization: `Bearer ${token}` },
  });

  const text = await resp.text();
  let json = null;
  try {
    json = text ? JSON.parse(text) : null;
  } catch (_) {}

  if (!resp.ok) {
    const err = new Error(
      `AML job get failed: HTTP ${resp.status} ${resp.statusText}`
    );
    err.details = text?.slice(0, 4000);
    err.httpStatus = resp.status;
    throw err;
  }

  return json;
}

function extractStudioUrl(jobJson) {
  const p = jobJson?.properties || {};
  return (
    p.studioPortalUrl ||
    p.studioUrl ||
    p.services?.Studio?.endpoint ||
    p.services?.Studio?.uri ||
    null
  );
}

// ====== Routes ======

// Health
app.get("/health", (req, res) => {
  res.json({
    ok: true,
    time: new Date().toISOString(),
    server_mode: "REST_COMPONENT_ONLY",
  });
});

// latest from Blob
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
      ok: false,
      error: e.message,
      storage: STORAGE_ACCOUNT,
      container: METRICS_CONTAINER,
      blob: LATEST_BLOB,
    });
  }
});

// run from Blob (optional)
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
      ok: false,
      error: e.message,
      storage: STORAGE_ACCOUNT,
      container: METRICS_CONTAINER,
      blob: RUN_BLOB,
    });
  }
});

// save settings into Blob
app.post("/api/settings", async (req, res) => {
  try {
    const settings = req.body;
    if (!settings || typeof settings !== "object" || Array.isArray(settings)) {
      return res.status(400).json({
        ok: false,
        error: "Body must be a JSON object (settings).",
      });
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
    console.error("POST /api/settings failed:", e);
    res.status(500).json({
      ok: false,
      error: e.message,
      storage: STORAGE_ACCOUNT,
      container: METRICS_CONTAINER,
      saved: [RUN_SETTINGS_JSON, RUN_SETTINGS_TXT],
    });
  }
});

// Start Azure ML component job
app.post("/api/run", async (req, res) => {
  const requestTime = new Date().toISOString();
  console.log("========== POST /api/run ==========");
  console.log("time:", requestTime);

  try {
    const settings = req.body;
    if (!settings || typeof settings !== "object" || Array.isArray(settings)) {
      return res.status(400).json({
        ok: false,
        error: "Body must be a JSON object (settings).",
      });
    }

    console.log("settings keys:", Object.keys(settings).length);

    // 1) Save settings to blob
    await saveRunSettingsToBlob(settings);
    console.log("settings saved to blob:", RUN_SETTINGS_JSON, RUN_SETTINGS_TXT);

    // 2) Submit AML component job
    const jobName = makeJobName("webrun");
    console.log("submitting AML component job:", jobName);

    const job = await submitAmlComponentJob({
      jobName,
      settingsUri: AML_SETTINGS_URI,
    });

    const studioUrl = extractStudioUrl(job);
    console.log("AML component job submitted:", job?.name || jobName);
    console.log("studio url:", studioUrl || "(none)");

    res.set("Cache-Control", "no-store");
    return res.json({
      ok: true,
      job_name: job?.name || jobName,
      studio_url: studioUrl,
      aml_settings_uri: AML_SETTINGS_URI,
      saved_settings: [RUN_SETTINGS_JSON, RUN_SETTINGS_TXT],
      aml: {
        compute: AZ_ML_COMPUTE,
        component: `${AZ_ML_COMPONENT_NAME}:${AZ_ML_COMPONENT_VERSION}`,
        api_version: AZ_ML_API_VERSION,
      },
    });
  } catch (e) {
    console.error("POST /api/run failed:", e);
    return res.status(500).json({
      ok: false,
      error: String(e.message || e),
      details: e.details || undefined,
      httpStatus: e.httpStatus || undefined,
      aml: {
        compute: AZ_ML_COMPUTE,
        component: `${AZ_ML_COMPONENT_NAME}:${AZ_ML_COMPONENT_VERSION}`,
        api_version: AZ_ML_API_VERSION,
      },
    });
  }
});

// Debug: get job detail/status by name
app.get("/api/job/:name", async (req, res) => {
  try {
    const jobName = req.params.name;
    const job = await getAmlJob(jobName);
    res.set("Cache-Control", "no-store");
    res.json({
      ok: true,
      job_name: job?.name || jobName,
      status: job?.properties?.status || null,
      studio_url: extractStudioUrl(job),
      raw: job,
    });
  } catch (e) {
    console.error("GET /api/job/:name failed:", e);
    res.status(500).json({
      ok: false,
      error: String(e.message || e),
      details: e.details || undefined,
      httpStatus: e.httpStatus || undefined,
    });
  }
});

// ====== Start server ======
app.listen(port, () => {
  console.log("SERVER_MODE=REST_COMPONENT_ONLY");
  console.log(`Backend listening on port ${port}`);
  console.log(`ALLOWED_ORIGIN=${ALLOWED_ORIGIN}`);

  console.log(`STORAGE_ACCOUNT=${STORAGE_ACCOUNT}`);
  console.log(`METRICS_CONTAINER=${METRICS_CONTAINER}`);
  console.log(`LATEST_BLOB=${LATEST_BLOB}`);
  console.log(`RUN_BLOB=${RUN_BLOB}`);
  console.log(`RUN_SETTINGS_JSON=${RUN_SETTINGS_JSON}`);
  console.log(`RUN_SETTINGS_TXT=${RUN_SETTINGS_TXT}`);
  console.log(`AML_SETTINGS_URI=${AML_SETTINGS_URI}`);

  console.log(
    `AZ_SUBSCRIPTION_ID=${AZ_SUBSCRIPTION_ID ? "***set***" : "***missing***"}`
  );
  console.log(`AZ_RESOURCE_GROUP=${AZ_RESOURCE_GROUP || "***missing***"}`);
  console.log(`AZ_ML_WORKSPACE=${AZ_ML_WORKSPACE || "***missing***"}`);
  console.log(`AZ_ML_COMPUTE=${AZ_ML_COMPUTE}`);
  console.log(
    `AZ_ML_COMPONENT=${AZ_ML_COMPONENT_NAME}:${AZ_ML_COMPONENT_VERSION}`
  );
  console.log(`AZ_ML_DATASTORE=${AZ_ML_DATASTORE}`);
  console.log(`AZ_ML_API_VERSION=${AZ_ML_API_VERSION}`);
});