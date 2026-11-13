import { fileURLToPath } from 'node:url';
import { defineConfig, searchForWorkspaceRoot } from 'vite';

/**
 * The directory KTX2Loader fetches its Basis transcoder from. In a git worktree
 * (every agent's checkout) `node_modules` resolves to the main checkout's, which
 * is outside the project root, so dev's default `server.fs.allow` answers the
 * transcoder with a 403 — the loader's other imports get through only because
 * they are in the module graph, and a fetched asset is not. (`AssetLoader` points
 * the dev transcoder path here; un-bundling the loader instead loaded three twice.)
 */
const BASIS_DIR = fileURLToPath(
  new URL('.', import.meta.resolve('three/examples/jsm/libs/basis/basis_transcoder.js')),
);

export default defineConfig({
  // Per checkout, not `node_modules/.vite`. Worktrees link the main checkout's
  // node_modules, so every dev server shared one pre-bundle cache and kept
  // invalidating each other's — the page then failed with 504 "Outdated
  // Optimize Dep" and never booted.
  cacheDir: '.vite-cache',
  server: {
    port: 5173,
    host: '127.0.0.1',
    fs: { allow: [searchForWorkspaceRoot(process.cwd()), BASIS_DIR] },
  },
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
