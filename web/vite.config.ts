import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// base './' : läuft auf jedem Pfad (nginx, GitHub Pages). Die App ist statisch, Daten liegen in public/data.
export default defineConfig({
  base: './',
  plugins: [react(), tailwindcss()],
  // API_TARGET: point the dev proxy at another backend (for example a fake API for screenshots)
  server: { proxy: { '/api': process.env.API_TARGET ?? 'http://localhost:8000' } },
})
