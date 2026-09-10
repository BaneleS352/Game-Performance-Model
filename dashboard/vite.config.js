import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { spawn } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

const dashboardDataPath = resolve(__dirname, "../artifacts/dashboard/dashboard_data.js");
const repoRoot = resolve(__dirname, "..");
const venvPython = resolve(repoRoot, process.platform === "win32" ? ".venv/Scripts/python.exe" : ".venv/bin/python");
const pythonCommand = existsSync(venvPython) ? venvPython : process.env.PYTHON ?? "python";

export default defineConfig({
  base: "./",
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/runs": "http://127.0.0.1:8000",
    },
  },
  plugins: [
    react(),
    {
      name: "local-dashboard-data",
      configureServer(server) {
        server.middlewares.use("/dashboard_data.js", (_request, response) => {
          response.setHeader("Content-Type", "text/javascript");
          response.end(existsSync(dashboardDataPath) ? readFileSync(dashboardDataPath, "utf8") : "window.DASHBOARD_DATA = undefined;\n");
        });
      },
    },
    {
      name: "local-dashboard-api",
      configureServer(server) {
        const api = spawn(pythonCommand, ["-m", "game_performance.server"], {
          cwd: repoRoot,
          env: { ...process.env, PYTHONPATH: resolve(__dirname, "../src") },
          stdio: "inherit",
          windowsHide: true,
        });
        api.on("error", (error) => {
          console.error(`Failed to start Python API with "${pythonCommand}": ${error.message}`);
        });
        api.on("exit", (code) => {
          if (code) console.error(`Python API exited with code ${code}. Check that Python dependencies are installed.`);
        });
        server.httpServer?.once("close", () => api.kill());
      },
    },
  ],
  build: {
    outDir: "../artifacts/dashboard",
    emptyOutDir: false,
  },
});
