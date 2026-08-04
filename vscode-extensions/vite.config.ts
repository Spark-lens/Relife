import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  define: {"process.env.NODE_ENV": JSON.stringify("production")},
  build: {
    outDir: "dist",
    emptyOutDir: true,
    cssCodeSplit: false,
    lib: {entry: "src/webview.tsx", formats: ["es"], fileName: () => "webview.js"},
    rollupOptions: {output: {assetFileNames: "webview.[ext]"}},
  },
});
