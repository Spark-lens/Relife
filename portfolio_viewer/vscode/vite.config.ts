import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));

export default defineConfig({
  publicDir: false,
  plugins: [react()],
  build: {
    outDir: fileURLToPath(new URL("./dist", import.meta.url)),
    emptyOutDir: true,
    lib: {
      entry: fileURLToPath(new URL("./webview.tsx", import.meta.url)),
      formats: ["es"],
      fileName: "webview",
      cssFileName: "webview",
    },
    rollupOptions: {
      input: fileURLToPath(new URL("./webview.tsx", import.meta.url)),
    },
  },
  resolve: {
    alias: { "@": root },
  },
});
