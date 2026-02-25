const fs = require("fs");
const path = require("path");

module.exports = async function (context, req) {
  try {
    const settings = req.body || {};

    context.log("Training request received:");
    context.log(JSON.stringify(settings, null, 2));

    // ===== 正确的 SWA 路径 =====
    const runDir = path.join(
      __dirname,
      "..",
      "..",
      "data",
      "run"
    );

    // 确保目录存在
    if (!fs.existsSync(runDir)) {
      fs.mkdirSync(runDir, { recursive: true });
    }

    const settingsPath = path.join(
      runDir,
      "last_run_settings.json"
    );

    fs.writeFileSync(
      settingsPath,
      JSON.stringify(settings, null, 2)
    );

    context.res = {
      status: 200,
      headers: {
        "Content-Type": "application/json"
      },
      body: {
        ok: true,
        message: "Training request received",
        timestamp: new Date().toISOString(),
        received_keys: Object.keys(settings)
      }
    };

  } catch (error) {
    context.log("Error in /api/run:", error);

    context.res = {
      status: 500,
      headers: {
        "Content-Type": "application/json"
      },
      body: {
        error: "Failed to process run request"
      }
    };
  }
};