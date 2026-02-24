const fs = require("fs");
const path = require("path");

module.exports = async function (context, req) {
  try {
    // repo 根目录
    const repoRoot = path.resolve(__dirname, "..", "..", "..");

    // 读取 apps/web/data/metrics.json
    const metricsPath = path.join(repoRoot, "apps", "web", "data", "metrics.json");

    let data = {};

    if (fs.existsSync(metricsPath)) {
      const raw = fs.readFileSync(metricsPath, "utf-8");
      data = JSON.parse(raw);
    } else {
      // 如果不存在，返回默认 mock 数据
      data = {
        status: "no_data",
        message: "metrics.json not found",
        timestamp: new Date().toISOString()
      };
    }

    context.res = {
      status: 200,
      headers: {
        "Content-Type": "application/json",
        "Cache-Control": "no-store"
      },
      body: data
    };

  } catch (error) {
    context.log("Error in /latest:", error);

    context.res = {
      status: 500,
      body: {
        error: "Failed to load latest metrics"
      }
    };
  }
};