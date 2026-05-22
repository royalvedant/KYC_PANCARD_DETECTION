import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // In local dev, proxy API calls to the FastAPI backend
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/luma.mp4': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
