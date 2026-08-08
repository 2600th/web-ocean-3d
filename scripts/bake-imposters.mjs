#!/usr/bin/env node

/**
 * Bake one front-facing, albedo-only billboard per vegetation species.
 *
 * The models are rendered in a real browser because GLTFLoader, Meshopt and
 * the browser's image decoders are the same path used by the app. Vite is
 * started here rather than assumed to be running, so this is a reproducible
 * asset step on a clean checkout:
 *
 *   node scripts/bake-imposters.mjs
 */

import { chromium } from '@playwright/test';
import sharp from 'sharp';
import { createServer } from 'node:net';
import { spawn } from 'node:child_process';
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { tmpdir } from 'node:os';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const OUTPUT_DIR = join(ROOT, 'public', 'imposters');
const ATLAS_PATH = join(OUTPUT_DIR, 'vegetation.png');
const SIDECAR_PATH = join(OUTPUT_DIR, 'vegetation.json');
const MODEL_DIR = join(ROOT, 'public', 'models', 'dressing');

const ATLAS_SIZE = 2048;
const CELL_SIZE = 512;
const GRID_COLUMNS = ATLAS_SIZE / CELL_SIZE;
const GRID_ROWS = ATLAS_SIZE / CELL_SIZE;
const MARGIN_PX = 24;
const BLEED_PX = 16;
const MIN_SILHOUETTE_COVERAGE = 0.08;
const MAX_SILHOUETTE_COVERAGE = 0.60;
const CENTROID_MIN = 0.12;
const CENTROID_MAX = 0.88;
const CONTENT_MARGIN_PX = 8;
// TODO: Once the KTX2 pipeline lands, encode this atlas for the runtime's
// alpha-capable GPU format instead of shipping PNG as the final texture.
const SCHEMA = 'web-ocean-3d/imposters@1';

/**
 * These are the exact placement choices in Props.ts. The scale is deliberately
 * kept in the sidecar rather than baked into the pixels: Props puts the random
 * scale/stretch in every instance matrix, while the geometry itself is unit
 * scale. `referenceScale` is useful to a runtime inspector and documents the
 * typical world size without changing that contract.
 */
const SPECIES = [
  {
    slug: 'fern_02',
    source: 'fern_02.glb',
    origin: 'cluster',
    include: { kind: 'suffix', value: '_b', expression: "name.endsWith('_b')" },
    props: { minScale: 2.4, maxScale: 4.2, minStretch: 1, maxStretch: 1 },
  },
  {
    slug: 'shrub_sorrel_01',
    source: 'shrub_sorrel_01.glb',
    origin: 'cluster',
    include: { kind: 'suffix', value: '_d', expression: "name.endsWith('_d')" },
    props: { minScale: 7, maxScale: 14, minStretch: 1, maxStretch: 1 },
  },
  {
    slug: 'calathea_orbifolia_01',
    source: 'calathea_orbifolia_01.glb',
    origin: 'cluster',
    include: { kind: 'suffix', value: '_a', expression: "name.endsWith('_a')" },
    props: { minScale: 2.2, maxScale: 3.8, minStretch: 1, maxStretch: 1 },
  },
  {
    slug: 'anthurium_botany_01',
    source: 'anthurium_botany_01.glb',
    origin: 'cluster',
    include: { kind: 'suffix', value: '_a', expression: "name.endsWith('_a')" },
    props: { minScale: 1.8, maxScale: 3.2, minStretch: 1, maxStretch: 1 },
  },
  {
    slug: 'grass_medium_01',
    source: 'grass_medium_01.glb',
    origin: 'stack',
    include: {
      kind: 'regex',
      pattern: '_(mid_b|small_b|tall_a|tall_c)_',
      expression: '/_(mid_b|small_b|tall_a|tall_c)_/.test(name)',
    },
    props: { minScale: 1.6, maxScale: 2.8, minStretch: 1, maxStretch: 1 },
  },
  {
    slug: 'grass_medium_02',
    source: 'grass_medium_02.glb',
    origin: 'stack',
    include: {
      kind: 'regex',
      pattern: '_(a|c|e)$',
      expression: '/_(a|c|e)$/.test(name)',
    },
    props: { minScale: 2, maxScale: 3.2, minStretch: 1, maxStretch: 1 },
  },
  {
    slug: 'tree_orchid',
    source: 'tree_orchid.glb',
    origin: 'asset',
    include: null,
    props: { minScale: 1.5, maxScale: 2.4, minStretch: 0.9, maxStretch: 1.2 },
  },
  {
    slug: 'tree_poinciana',
    source: 'tree_poinciana.glb',
    origin: 'asset',
    include: null,
    props: { minScale: 1.2, maxScale: 1.9, minStretch: 0.9, maxStretch: 1.15 },
  },
  {
    slug: 'jacaranda_tree',
    source: 'jacaranda_tree.glb',
    origin: 'asset',
    include: null,
    props: { minScale: 0.55, maxScale: 0.95, minStretch: 0.9, maxStretch: 1.1 },
  },
  {
    slug: 'island_tree_02',
    source: 'island_tree_02.glb',
    origin: 'asset',
    include: null,
    props: { minScale: 2.2, maxScale: 3.6, minStretch: 0.9, maxStretch: 1.15 },
  },
  {
    slug: 'palm_coconut',
    source: 'palm_coconut.glb',
    origin: 'asset',
    include: null,
    props: { minScale: 1.5, maxScale: 2.6, minStretch: 0.9, maxStretch: 1.25 },
  },
  {
    slug: 'palm_tall',
    source: 'palm_tall.glb',
    origin: 'asset',
    include: null,
    props: { minScale: 1.8, maxScale: 3.2, minStretch: 0.9, maxStretch: 1.3 },
  },
  {
    slug: 'pachira_aquatica_01',
    source: 'pachira_aquatica_01.glb',
    origin: 'cluster',
    include: { kind: 'suffix', value: '_d', expression: "name.endsWith('_d')" },
    props: { minScale: 2.2, maxScale: 3.8, minStretch: 1, maxStretch: 1 },
  },
];

function round(value, digits = 6) {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function referenceScale(props) {
  return (props.minScale + props.maxScale) * 0.5;
}

function referenceStretch(props) {
  return (props.minStretch + props.maxStretch) * 0.5;
}

function findFreePort() {
  return new Promise((resolvePort, reject) => {
    const probe = createServer();
    probe.unref();
    probe.once('error', reject);
    probe.listen(0, '127.0.0.1', () => {
      const address = probe.address();
      const port = typeof address === 'object' && address ? address.port : null;
      probe.close(() => {
        if (port === null) reject(new Error('OS did not return a free port'));
        else resolvePort(port);
      });
    });
  });
}

async function waitForServer(url, timeoutMs = 60_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url, { redirect: 'follow' });
      if (response.ok) return;
    } catch {
      // Vite is still starting.
    }
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 250));
  }
  throw new Error(`Vite did not respond at ${url}`);
}

async function startVite() {
  const port = await findFreePort();
  const url = `http://127.0.0.1:${port}`;
  const viteBin = join(ROOT, 'node_modules', 'vite', 'bin', 'vite.js');
  if (!existsSync(viteBin)) throw new Error(`Vite entry point is missing: ${viteBin}`);

  const child = spawn(
    process.execPath,
    [viteBin, '--host', '127.0.0.1', '--port', String(port), '--strictPort'],
    { cwd: ROOT, stdio: ['ignore', 'pipe', 'pipe'] },
  );
  let log = '';
  const record = (chunk) => {
    log = (log + chunk.toString()).slice(-8_000);
  };
  child.stdout.on('data', record);
  child.stderr.on('data', record);
  child.on('exit', (code) => {
    if (code !== 0 && code !== null) process.stderr.write(`[bake] vite exited ${code}\n${log}\n`);
  });

  try {
    await waitForServer(`${url}/scripts/modelkit/viewer.html`);
  } catch (error) {
    child.kill();
    throw new Error(`${error.message}\n${log}`);
  }
  return { url, stop: () => child.kill() };
}

function viewerHtml(origin) {
  const importMap = JSON.stringify({
    imports: {
      three: `${origin}/node_modules/three/build/three.module.js`,
      'three/addons/': `${origin}/node_modules/three/examples/jsm/`,
    },
  });

  return `<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <style>html, body { margin: 0; width: 2048px; height: 2048px; overflow: hidden; background: transparent; }</style>
    <script type="importmap">${importMap}</script>
  </head>
  <body>
    <canvas id="atlas" width="2048" height="2048"></canvas>
    <script type="module">
      import * as THREE from 'three';
      import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
      import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
      import { KTX2Loader } from 'three/addons/loaders/KTX2Loader.js';

      const ATLAS_SIZE = ${ATLAS_SIZE};
      const CELL_SIZE = ${CELL_SIZE};
      const MARGIN_PX = ${MARGIN_PX};
      const canvas = document.getElementById('atlas');
      const renderer = new THREE.WebGLRenderer({
        canvas,
        alpha: true,
        antialias: true,
        premultipliedAlpha: true,
        preserveDrawingBuffer: true,
        powerPreference: 'high-performance',
      });
      renderer.setPixelRatio(1);
      renderer.setSize(ATLAS_SIZE, ATLAS_SIZE, false);
      renderer.setClearColor(0x000000, 0);
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      renderer.toneMapping = THREE.NoToneMapping;
      renderer.autoClear = false;

      const scene = new THREE.Scene();
      const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 100000);
      // The dressing set is KTX2 as well as Meshopt, so the bake needs both
      // decoders or it cannot read the very models it is baking. KTX2Loader
      // needs a renderer to ask which compressed formats the device supports,
      // which is why this is wired after the renderer exists.
      const ktx2 = new KTX2Loader()
        .setTranscoderPath('/basis/')
        .detectSupport(renderer);
      const loader = new GLTFLoader()
        .setMeshoptDecoder(MeshoptDecoder)
        .setKTX2Loader(ktx2);
      let current = null;

      function matches(name, include) {
        if (!include) return true;
        if (include.kind === 'suffix') return name.endsWith(include.value);
        if (include.kind === 'regex') return new RegExp(include.pattern).test(name);
        throw new Error('Unknown include rule: ' + include.kind);
      }

      function basicMaterial(source) {
        const material = new THREE.MeshBasicMaterial({
          color: source.color?.clone?.() ?? new THREE.Color(0xffffff),
          map: source.map ?? null,
          alphaMap: source.alphaMap ?? null,
          vertexColors: Boolean(source.vertexColors),
          transparent: Boolean(source.transparent),
          opacity: source.opacity ?? 1,
          alphaTest: source.alphaTest ?? 0,
          side: source.side ?? THREE.FrontSide,
          depthWrite: source.depthWrite,
          depthTest: source.depthTest,
          fog: false,
          toneMapped: false,
        });
        material.premultipliedAlpha = Boolean(source.premultipliedAlpha);
        material.name = source.name ? source.name + '-albedo' : 'albedo';
        return material;
      }

      function sourceMaterials(source) {
        return Array.isArray(source) ? source.map(basicMaterial) : basicMaterial(source);
      }

      function disposeMaterial(material, textures) {
        const list = Array.isArray(material) ? material : [material];
        for (const item of list) {
          if (!item) continue;
          for (const key of ['map', 'alphaMap', 'aoMap', 'emissiveMap', 'lightMap', 'metalnessMap', 'normalMap', 'roughnessMap']) {
            const texture = item[key];
            if (texture) textures.add(texture);
          }
          item.dispose();
        }
      }

      function disposeTree(root) {
        if (!root) return;
        const textures = new Set();
        root.traverse((node) => {
          if (!node.isMesh) return;
          node.geometry?.dispose();
          disposeMaterial(node.material, textures);
        });
        for (const texture of textures) texture.dispose();
      }

      function recentre(box) {
        return new THREE.Vector3(
          -(box.min.x + box.max.x) * 0.5,
          -box.min.y,
          -(box.min.z + box.max.z) * 0.5,
        );
      }

      // Keep this in lockstep with AssetLoader.dequantiseGeometry. Meshopt's
      // KHR_mesh_quantization leaves positions in normalized integer buffers;
      // applying a node transform before widening them clamps large models to
      // the integer buffer's representable range and produces box-shaped junk.
      function dequantiseGeometry(geometry) {
        for (const [name, attribute] of Object.entries(geometry.attributes)) {
          if (!attribute.normalized && attribute.array instanceof Float32Array) continue;

          const items = attribute.itemSize;
          const values = new Float32Array(attribute.count * items);
          for (let i = 0; i < attribute.count; i++) {
            for (let c = 0; c < items; c++) values[i * items + c] = attribute.getComponent(i, c);
          }
          geometry.setAttribute(name, new THREE.BufferAttribute(values, items));
        }
      }
      function flatten(gltf, spec) {
        gltf.scene.updateMatrixWorld(true);
        const selected = [];
        gltf.scene.traverse((node) => {
          if (node.isMesh && matches(node.name, spec.include)) selected.push(node);
        });
        if (selected.length === 0) {
          const names = [];
          gltf.scene.traverse((node) => { if (node.isMesh) names.push(node.name); });
          throw new Error(spec.slug + ': include rule selected no mesh. Available names: ' + names.join(', '));
        }

        const root = new THREE.Group();
        root.name = spec.slug + '-bake';
        const copies = [];
        for (const source of selected) {
          const geometry = source.geometry.clone();
          dequantiseGeometry(geometry);
          geometry.applyMatrix4(source.matrixWorld);
          const copy = new THREE.Mesh(geometry, sourceMaterials(source.material));
          copy.frustumCulled = false;
          root.add(copy);
          copies.push({ geometry, copy });
        }

        if (spec.origin === 'stack') {
          for (const item of copies) {
            item.geometry.computeBoundingBox();
            if (!item.geometry.boundingBox) continue;
            item.geometry.translate(...recentre(item.geometry.boundingBox).toArray());
          }
        } else if (spec.origin === 'cluster') {
          const box = new THREE.Box3().setFromObject(root);
          root.position.copy(recentre(box));
        } else if (spec.origin !== 'asset') {
          throw new Error(spec.slug + ': unknown origin ' + spec.origin);
        }

        root.updateMatrixWorld(true);
        const bounds = new THREE.Box3().setFromObject(root);
        const size = bounds.getSize(new THREE.Vector3());
        if (!(size.x > 0 && size.y > 0)) throw new Error(spec.slug + ': degenerate bounds ' + size.toArray());
        return { root, original: gltf.scene, selected: selected.map((node) => node.name), bounds, size };
      }

      function frame(flattened, cell) {
        const box = flattened.bounds;
        const size = flattened.size;
        const centre = box.getCenter(new THREE.Vector3());
        const coverage = (CELL_SIZE - MARGIN_PX * 2) / CELL_SIZE;
        const viewSize = Math.max(size.x, size.y) / coverage;
        camera.left = -viewSize * 0.5;
        camera.right = viewSize * 0.5;
        camera.top = viewSize * 0.5;
        camera.bottom = -viewSize * 0.5;
        camera.near = 0.01;
        camera.far = Math.max(1000, size.z + 100);
        camera.position.set(centre.x, centre.y, box.max.z + Math.max(10, size.z + 10));
        camera.lookAt(centre.x, centre.y, centre.z);
        camera.updateProjectionMatrix();

        const y = ATLAS_SIZE - cell.y - CELL_SIZE;
        renderer.setViewport(cell.x, y, CELL_SIZE, CELL_SIZE);
        renderer.setScissor(cell.x, y, CELL_SIZE, CELL_SIZE);
        renderer.setScissorTest(true);
        renderer.clear(true, true, true);
        scene.add(flattened.root);
        renderer.render(scene, camera);
        renderer.getContext().finish?.();
        scene.remove(flattened.root);

        return {
          bounds: {
            minX: box.min.x,
            maxX: box.max.x,
            minY: box.min.y,
            maxY: box.max.y,
            minZ: box.min.z,
            maxZ: box.max.z,
          },
          size: { x: size.x, y: size.y, z: size.z },
          pivot: { x: -centre.x, y: -box.min.y },
          selected: flattened.selected,
          viewSize,
          camera: { x: centre.x, y: centre.y, z: camera.position.z },
        };
      }

      window.bakeSpecies = async (spec) => {
        if (current) {
          disposeTree(current.flattened.root);
          disposeTree(current.flattened.original);
          current = null;
        }
        const gltf = await loader.loadAsync(spec.url);
        const flattened = flatten(gltf, spec);
        current = { flattened };
        const result = frame(flattened, spec.cell);
        // Make the GPU finish before the model's textures are released. This is
        // construction-time work, so the small synchronous wait is preferable to
        // retaining every GLB's decoded image until the atlas is complete.
        renderer.getContext().finish?.();
        disposeTree(flattened.root);
        disposeTree(flattened.original);
        current = null;
        return result;
      };

      renderer.setViewport(0, 0, ATLAS_SIZE, ATLAS_SIZE);
      renderer.setScissor(0, 0, ATLAS_SIZE, ATLAS_SIZE);
      renderer.setScissorTest(false);
      renderer.clear(true, true, true);
      window.__imposterBakeReady = true;
    </script>
  </body>
</html>`;
}

function cellFor(index) {
  return {
    index,
    x: (index % GRID_COLUMNS) * CELL_SIZE,
    y: Math.floor(index / GRID_COLUMNS) * CELL_SIZE,
    width: CELL_SIZE,
    height: CELL_SIZE,
  };
}

function uvFor(cell) {
  // Pixel rects use the image's top-left origin. UVs use Three.js/WebGL's
  // bottom-left origin, which is the convention the runtime shader consumes.
  return {
    u0: round(cell.x / ATLAS_SIZE),
    u1: round((cell.x + cell.width) / ATLAS_SIZE),
    v0: round(1 - (cell.y + cell.height) / ATLAS_SIZE),
    v1: round(1 - cell.y / ATLAS_SIZE),
    origin: 'bottom-left',
  };
}

function metadataFor(spec, cell, rendered) {
  const propsReferenceScale = referenceScale(spec.props);
  const propsReferenceStretch = referenceStretch(spec.props);
  const worldWidth = rendered.size.x;
  const worldHeight = rendered.size.y;
  return {
    sourceGlb: spec.source,
    sourceGlbSlug: spec.slug,
    cell,
    uv: uvFor(cell),
    worldWidth: round(worldWidth),
    worldHeight: round(worldHeight),
    pivotHeight: round(rendered.pivot.y),
    pivot: {
      // The card is framed around the bounds centre, while Props places the
      // source at its origin. These offsets let the runtime preserve that anchor.
      offsetX: round(rendered.pivot.x),
      heightAboveBottom: round(rendered.pivot.y),
    },
    world: {
      width: round(worldWidth),
      height: round(worldHeight),
      depth: round(rendered.size.z),
      unitScale: 1,
      reference: {
        scale: round(propsReferenceScale),
        stretch: round(propsReferenceStretch),
        width: round(worldWidth * propsReferenceScale),
        height: round(worldHeight * propsReferenceScale * propsReferenceStretch),
      },
    },
    props: {
      origin: spec.origin,
      include: spec.include?.expression ?? null,
      selectedNodeNames: rendered.selected,
      minScale: spec.props.minScale,
      maxScale: spec.props.maxScale,
      minStretch: spec.props.minStretch,
      maxStretch: spec.props.maxStretch,
      scaleAppliedInBake: 1,
      scaleAppliedByRuntimePlacement: true,
    },
    frame: {
      marginPx: MARGIN_PX,
      viewSize: round(rendered.viewSize),
      camera: {
        projection: 'orthographic',
        view: 'front',
        horizontal: true,
        target: {
          x: round(rendered.camera.x),
          y: round(rendered.camera.y),
        },
      },
    },
  };
}

/**
 * Copies edge colours into transparent texels without changing alpha. A
 * mipmap samples neighbouring RGB even when their alpha is zero; leaving those
 * texels black is the dark halo seen around cutout foliage. The bounded BFS is
 * per cell, so it cannot contaminate an adjacent species or an empty cell.
 */
async function bleedTransparentEdges(inputPath, outputPath, cells) {
  const decoded = await sharp(inputPath).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  const { data, info } = decoded;
  if (info.width !== ATLAS_SIZE || info.height !== ATLAS_SIZE || info.channels !== 4) {
    throw new Error(`Unexpected atlas pixels: ${info.width}x${info.height}x${info.channels}`);
  }

  for (const cell of cells) {
    const seeded = new Uint8Array(CELL_SIZE * CELL_SIZE);
    for (let y = 0; y < CELL_SIZE; y++) {
      for (let x = 0; x < CELL_SIZE; x++) {
        const atlasOffset = ((cell.y + y) * ATLAS_SIZE + cell.x + x) * 4;
        if (data[atlasOffset + 3] !== 0) seeded[y * CELL_SIZE + x] = 1;
      }
    }

    for (let pass = 0; pass < BLEED_PX; pass++) {
      const newlySeeded = [];
      for (let y = 0; y < CELL_SIZE; y++) {
        for (let x = 0; x < CELL_SIZE; x++) {
          const local = y * CELL_SIZE + x;
          if (seeded[local]) continue;
          const neighbours = [];
          if (x > 0) neighbours.push(local - 1);
          if (x + 1 < CELL_SIZE) neighbours.push(local + 1);
          if (y > 0) neighbours.push(local - CELL_SIZE);
          if (y + 1 < CELL_SIZE) neighbours.push(local + CELL_SIZE);
          const source = neighbours.find((candidate) => seeded[candidate]);
          if (source === undefined) continue;

          const atlasOffset = ((cell.y + y) * ATLAS_SIZE + cell.x + x) * 4;
          const sourceX = source % CELL_SIZE;
          const sourceY = Math.floor(source / CELL_SIZE);
          const sourceOffset = ((cell.y + sourceY) * ATLAS_SIZE + cell.x + sourceX) * 4;
          data[atlasOffset] = data[sourceOffset];
          data[atlasOffset + 1] = data[sourceOffset + 1];
          data[atlasOffset + 2] = data[sourceOffset + 2];
          newlySeeded.push(local);
        }
      }
      for (const local of newlySeeded) seeded[local] = 1;
      if (newlySeeded.length === 0) break;
    }
  }

  await sharp(data, { raw: { width: ATLAS_SIZE, height: ATLAS_SIZE, channels: 4 } })
    .png({ compressionLevel: 9 })
    .toFile(outputPath);
}

async function validateAtlasSilhouettes(atlasPath, entries) {
  const decoded = await sharp(atlasPath).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  const { data, info } = decoded;
  if (info.width !== ATLAS_SIZE || info.height !== ATLAS_SIZE || info.channels !== 4) {
    throw new Error(`Unexpected atlas pixels during silhouette validation: ${info.width}x${info.height}x${info.channels}`);
  }

  for (const { spec, cell } of entries) {
    let alphaPixels = 0;
    let sumAlpha = 0;
    let weightedX = 0;
    let weightedY = 0;
    let minX = cell.width;
    let minY = cell.height;
    let maxX = -1;
    let maxY = -1;

    for (let y = 0; y < cell.height; y++) {
      for (let x = 0; x < cell.width; x++) {
        const alpha = data[((cell.y + y) * info.width + cell.x + x) * 4 + 3];
        if (alpha === 0) continue;
        alphaPixels++;
        sumAlpha += alpha;
        weightedX += x * alpha;
        weightedY += y * alpha;
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
      }
    }

    const coverage = alphaPixels / (cell.width * cell.height);
    const centroidX = weightedX / (sumAlpha * cell.width);
    const centroidY = weightedY / (sumAlpha * cell.height);
    const problems = [];
    if (coverage < MIN_SILHOUETTE_COVERAGE || coverage > MAX_SILHOUETTE_COVERAGE) {
      problems.push(`coverage ${(coverage * 100).toFixed(1)}% outside ${(MIN_SILHOUETTE_COVERAGE * 100).toFixed(0)}-${(MAX_SILHOUETTE_COVERAGE * 100).toFixed(0)}%`);
    }
    if (!Number.isFinite(centroidX) || !Number.isFinite(centroidY) ||
        centroidX <= CENTROID_MIN || centroidX >= CENTROID_MAX ||
        centroidY <= CENTROID_MIN || centroidY >= CENTROID_MAX) {
      problems.push(`centroid (${centroidX.toFixed(3)},${centroidY.toFixed(3)}) outside ${CENTROID_MIN}-${CENTROID_MAX}`);
    }
    if (minX < CONTENT_MARGIN_PX || minY < CONTENT_MARGIN_PX ||
        maxX >= cell.width - CONTENT_MARGIN_PX || maxY >= cell.height - CONTENT_MARGIN_PX) {
      problems.push(`content bounds ${minX},${minY}-${maxX},${maxY} escape the cell margin`);
    }
    if (problems.length > 0) throw new Error(`${spec.slug}: ${problems.join('; ')}`);

    process.stdout.write(
      `[bake] silhouette ${spec.slug}: ${(coverage * 100).toFixed(1)}% alpha, ` +
      `centroid (${centroidX.toFixed(3)},${centroidY.toFixed(3)}), ` +
      `bbox ${minX},${minY}-${maxX},${maxY}\n`,
    );
  }
}
function buildSidecar(renderedEntries) {
  const emptyCells = [];
  for (let index = SPECIES.length; index < GRID_COLUMNS * GRID_ROWS; index++) emptyCells.push(cellFor(index));

  const species = {};
  for (const entry of renderedEntries) {
    species[entry.spec.slug] = metadataFor(entry.spec, entry.cell, entry.rendered);
  }

  return {
    schema: SCHEMA,
    generatedAt: new Date().toISOString(),
    command: 'node scripts/bake-imposters.mjs',
    source: {
      spec: 'docs/superpowers/specs/2026-08-08-low-end-optimisation-design.md',
      modelDirectory: 'public/models/dressing',
      placementContract: 'src/scene/Props.ts',
    },
    atlas: {
      file: 'imposters/vegetation.png',
      width: ATLAS_SIZE,
      height: ATLAS_SIZE,
      cellSize: CELL_SIZE,
      columns: GRID_COLUMNS,
      rows: GRID_ROWS,
      cellOrigin: 'top-left',
      speciesCount: SPECIES.length,
      unusedCellCount: emptyCells.length,
      emptyCells,
    },
    bake: {
      camera: 'orthographic',
      view: 'front',
      horizontal: true,
      background: 'transparent',
      premultipliedAlpha: true,
      shading: 'albedo-only',
      toneMapping: 'none',
      marginPx: MARGIN_PX,
      bleedPx: BLEED_PX,
      scaleSemantics: 'world dimensions are at unit mesh scale; Props placement applies instance scale/stretch',
      normalMap: null,
      normalMapReason: 'Skipped; the runtime can light the card with its dome normal.',
      silhouetteValidation: {
        coverageMin: MIN_SILHOUETTE_COVERAGE,
        coverageMax: MAX_SILHOUETTE_COVERAGE,
        centroidMin: CENTROID_MIN,
        centroidMax: CENTROID_MAX,
        contentMarginPx: CONTENT_MARGIN_PX,
      },
    },
    species,
  };
}

async function main() {
  mkdirSync(OUTPUT_DIR, { recursive: true });
  for (const spec of SPECIES) {
    const modelPath = join(MODEL_DIR, spec.source);
    if (!existsSync(modelPath)) throw new Error(`Missing source GLB: ${modelPath}`);
  }

  const server = await startVite();
  const browserOptions = {
    headless: true,
    args: ['--enable-webgl', '--use-angle=swiftshader'],
  };
  let browser;
  let browserChannel = 'bundled-chromium';
  try {
    try {
      browser = await chromium.launch({ ...browserOptions, channel: 'chrome' });
      browserChannel = 'chrome';
    } catch {
      browser = await chromium.launch(browserOptions);
    }

    const page = await browser.newPage({ viewport: { width: ATLAS_SIZE, height: ATLAS_SIZE }, deviceScaleFactor: 1 });
    const pageErrors = [];
    page.on('pageerror', (error) => pageErrors.push(error.message));
    await page.goto(`${server.url}/scripts/modelkit/viewer.html`, { waitUntil: 'domcontentloaded' });
    await page.setContent(viewerHtml(server.url), { waitUntil: 'load' });
    await page.waitForFunction(() => window.__imposterBakeReady === true);

    const renderedEntries = [];
    for (let index = 0; index < SPECIES.length; index++) {
      const spec = SPECIES[index];
      const cell = cellFor(index);
      const url = `${server.url}/models/dressing/${spec.source}`;
      process.stdout.write(`[bake] ${String(index + 1).padStart(2, '0')}/${SPECIES.length} ${spec.slug} ...\n`);
      const rendered = await page.evaluate(
        async ({ spec: input, cell: inputCell, url: inputUrl }) => window.bakeSpecies({ ...input, cell: inputCell, url: inputUrl }),
        { spec, cell, url },
      );
      renderedEntries.push({ spec, cell, rendered });
      process.stdout.write(
        `[bake] ${spec.slug}: ${rendered.selected.length} node(s), ` +
        `${rendered.size.x.toFixed(3)}m x ${rendered.size.y.toFixed(3)}m, ` +
        `pivot ${rendered.pivot.y.toFixed(3)}m\n`,
      );
    }

    const tempDir = mkdtempSync(join(tmpdir(), 'web-ocean-imposters-'));
    const rawAtlasPath = join(tempDir, 'vegetation-raw.png');
    try {
      await page.locator('#atlas').screenshot({ path: rawAtlasPath, omitBackground: true });
      const cells = renderedEntries.map((entry) => entry.cell);
      await bleedTransparentEdges(rawAtlasPath, ATLAS_PATH, cells);
      await validateAtlasSilhouettes(ATLAS_PATH, renderedEntries);
    } finally {
      rmSync(tempDir, { recursive: true, force: true });
    }

    const sidecar = buildSidecar(renderedEntries);
    sidecar.renderer = {
      browser: browserChannel,
      renderer: 'three.WebGLRenderer',
      rendererOptions: { antialias: true, alpha: true, premultipliedAlpha: true },
    };
    writeFileSync(SIDECAR_PATH, `${JSON.stringify(sidecar, null, 2)}\n`, 'utf8');

    if (pageErrors.length > 0) throw new Error(`Viewer page errors: ${pageErrors.join(' | ')}`);
    process.stdout.write(`[bake] wrote ${ATLAS_PATH}\n[bake] wrote ${SIDECAR_PATH}\n`);
  } finally {
    await browser?.close();
    server.stop();
  }
}

main().catch((error) => {
  process.stderr.write(`[bake] failed: ${error.stack ?? error.message}\n`);
  process.exitCode = 1;
});
