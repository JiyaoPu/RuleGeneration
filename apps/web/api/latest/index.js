const fs = require("fs");
const path = require("path");

module.exports = async function (context, req) {
  try {
    const metricsPath = path.join(
      __dirname,
      "..",
      "..",
      "data",
      "latest",
      "metrics.json"
    );

    if (!fs.existsSync(metricsPath)) {
      context.res = {
        status: 404,
        headers: { "Content-Type": "application/json" },
        body: {
          status: "no_data",
          message: "metrics.json not found"
        }
      };
      return;
    }

    const raw = fs.readFileSync(metricsPath, "utf8");
    const data = JSON.parse(raw);

    context.res = {
      status: 200,
      headers: {
        "Content-Type": "application/json",
        "Cache-Control": "no-store"
      },
      body: data
    };

  } catch (error) {
    context.log("Error in /api/latest:", error);

    context.res = {
      status: 500,
      headers: { "Content-Type": "application/json" },
      body: {
        error: "Failed to load latest metrics"
      }
    };
  }
};