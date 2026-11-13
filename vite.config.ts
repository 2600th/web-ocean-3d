import { defineConfig } from 'vite';

export default defineConfig({
  server: { port: 5173, host: '127.0.0.1' },
  preview: { port: 4173, host: '127.0.0.1' },
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
