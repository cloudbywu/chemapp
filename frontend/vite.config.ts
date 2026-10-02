import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Plotly's has-hover dependency expects Browserify's `global` alias. Replace
  // that identifier at build time with the browser global, without enabling
  // Node integration, injecting a runtime shim, or weakening the desktop CSP.
  define: { global: 'globalThis' },
  server: {
    host: "127.0.0.1",
    port: 3000,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
})
