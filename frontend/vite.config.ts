import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // Emit the standalone, server-less build to `dist/` next to the source.
  // Serve this directory with any static server (or `python -m http.server`)
  // to run pyspider entirely in the browser via Pyodide.
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
  server: {
    port: 5173,
  },
});
