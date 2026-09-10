import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { spawn } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

const dashboardDataPath = resolve(__dirname, "../artifacts/dashboard/dashboard_data.js");

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
        const api = spawn(process.env.PYTHON ?? "python", ["-m", "game_performance.server"], {
          cwd: resolve(__dirname, ".."),
          env: { ...process.env, PYTHONPATH: resolve(__dirname, "../src") },
          stdio: "inherit",
          windowsHide: true,
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
