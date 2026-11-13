import { defineConfig } from 'vite';

export default defineConfig({
  server: { port: 5173, host: '127.0.0.1' },
  preview: { port: 4173, host: '127.0.0.1' },
  // KTX2Loader finds its Basis transcoder with `new URL('../libs/basis/',
  // import.meta.url)`. Pre-bundled, the module lives in `.vite/deps` and that URL
  // lands on the SPA fallback, so dev served HTML as the transcoder. Served from
  // node_modules, the relative URL is real. The build is unaffected either way.
  optimizeDeps: { exclude: ['three/addons/loaders/KTX2Loader.js'] },
  build: {
    target: 'esnext',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          three: ['three', 'three/webgpu', 'three/tsl'],
        },
      },
    },
  },
});
