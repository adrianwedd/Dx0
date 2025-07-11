import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Use the FastAPI static directory as the build target so the server
// can directly serve the compiled assets.
const outDir = '../sdb/ui/static'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    outDir,
    emptyOutDir: true,
  },
})
