const fs = require("fs");
const path = require("path");

module.exports = async function (context, req) {
  try {

    // 打印调试信息
    context.log("===== DEBUG INFO =====");
    context.log("process.cwd():", process.cwd());
    context.log("__dirname:", __dirname);

    // 计算路径（当前 API 目录结构）
    const metricsPath = path.join(__dirname, "..", "..", "data", "latest", "metrics.json");

    context.log("metricsPath:", metricsPath);
    context.log("exists:", fs.existsSync(metricsPath));

    let data;

    if (fs.existsSync(metricsPath)) {
      const raw = fs.readFileSync(metricsPath, "utf-8");
      data = JSON.parse(raw);
    } else {
      data = {
        status: "no_data",
        message: "metrics.json not found",
        path_checked: metricsPath
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

    // 🔴 关键：把错误内容返回给前端
    context.log("Error in /latest:", error);

    context.res = {
      status: 500,
      headers: {
        "Content-Type": "application/json"
      },
      body: {
        error: error.message,
        stack: error.stack
      }
    };
  }
};