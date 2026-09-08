const fs = require("node:fs");
const path = require("node:path");

const dashboardDir = path.resolve(__dirname, "../../artifacts/dashboard");
const assetsDir = path.resolve(dashboardDir, "assets");

if (!assetsDir.startsWith(`${dashboardDir}${path.sep}`)) {
  throw new Error(`Refusing to clean outside dashboard artifacts: ${assetsDir}`);
}

fs.rmSync(assetsDir, { recursive: true, force: true });
