const fs = require("fs");
const path = require("path");

module.exports = async function (context, req) {
  try {
    const metricsPath = path.join(__dirname, "..", "data", "latest", "metrics.json");

    let data;
    if (fs.existsSync(metricsPath)) {
      data = JSON.parse(fs.readFileSync(metricsPath, "utf-8"));
    } else {
      data = { status: "no_data", message: "metrics.json not found" };
    }

    context.res = {
      status: 200,
      headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
      body: data,
    };
  } catch (e) {
    context.log("latest error:", e);
    context.res = { status: 500, body: { error: String(e) } };
  }
};