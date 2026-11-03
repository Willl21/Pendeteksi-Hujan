import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  root: "web",
  publicDir: "../public",
  build: { outDir: "../dist", emptyOutDir: true },
  plugins: [react()],
});
