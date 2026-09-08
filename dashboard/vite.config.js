import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { readFileSync } from "node:fs";
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
          response.end(readFileSync(dashboardDataPath, "utf8"));
        });
      },
    },
  ],
  build: {
    outDir: "../artifacts/dashboard",
    emptyOutDir: false,
  },
});
