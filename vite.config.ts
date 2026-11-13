import { fileURLToPath } from 'node:url';
import { defineConfig, searchForWorkspaceRoot } from 'vite';

/**
 * The directory KTX2Loader fetches its Basis transcoder from. In a git worktree
 * (every agent's checkout) `node_modules` resolves to the main checkout's, which
 * is outside the project root, so dev's default `server.fs.allow` answers the
 * transcoder with a 403 — the loader's other imports get through only because
 * they are in the module graph, and a fetched `new URL()` asset is not.
 */
const BASIS_DIR = fileURLToPath(
  new URL('.', import.meta.resolve('three/examples/jsm/libs/basis/basis_transcoder.js')),
);

export default defineConfig({
  server: {
    port: 5173,
    host: '127.0.0.1',
    fs: { allow: [searchForWorkspaceRoot(process.cwd()), BASIS_DIR] },
  },
  preview: { port: 4173, host: '127.0.0.1' },
  // KTX2Loader finds its Basis transcoder with `new URL('../libs/basis/',
  // import.meta.url)`. Pre-bundled, the module lives in `.vite/deps` and that URL
  // lands on the SPA fallback, so dev served HTML as the transcoder. Served from
  // node_modules, the relative URL is real (given `fs.allow` above). The build
  // is unaffected either way.
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
