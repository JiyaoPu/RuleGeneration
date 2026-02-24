const fs = require("fs");
const path = require("path");

module.exports = async function (context, req) {
  try {
    const settings = req.body || {};

    context.log("Training request received:");
    context.log(JSON.stringify(settings, null, 2));

    // ======== 示例行为：保存 settings 到文件 ========
    // 你可以删掉这段，如果不想写本地文件

    const repoRoot = path.resolve(__dirname, "..", "..", "..");
    const dataDir = path.join(repoRoot, "apps", "web", "data");

    if (!fs.existsSync(dataDir)) {
      fs.mkdirSync(dataDir, { recursive: true });
    }

    const settingsPath = path.join(dataDir, "last_run_settings.json");
    fs.writeFileSync(settingsPath, JSON.stringify(settings, null, 2));

    // ======== 返回响应 ========

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
    context.log("Error in /run:", error);

    context.res = {
      status: 500,
      body: {
        error: "Failed to process run request"
      }
    };
  }
};