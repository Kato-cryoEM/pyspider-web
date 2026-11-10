import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // Relative base — index.html references assets as `./assets/...`, which
  // works both at the site root (local `python -m http.server`) and under
  // a GitHub Pages subpath like `/pyspider-web/`.
  base: "./",
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
  server: {
    port: 5173,
  },
});
