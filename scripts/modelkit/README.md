# modelkit

Tools for looking at 3D assets and deciding what to do with them, separate from
the two scripts that actually build the shipped set:

| Script | Question it answers |
| --- | --- |
| `inspect.mjs` | What is in this file? Triangles per primitive, materials, alpha modes, texture slots, per-primitive bounds. |
| `shells.mjs` | Can this model be placed at any angle, or only seen from one side? |
| `plates.mjs` | Does this scan have a ground slab welded under it? |
| `sketchfab.mjs` | What is available under a licence this project can ship, and can I have it? |
| `viewer.html` | What does it actually look like? |

None of them modify anything. They read models and print or draw. The build
pipeline is `scripts/fetch-assets.mjs` (get the authoritative source) and
`scripts/optimize-assets.mjs` (decimate, LOD, encode, ship).

The one exception is `split-hero.mjs`, a build step: it cuts the hero ship's
rudder, whipstaff and four square yards out of their joined meshes into pivot
nodes and writes `public/models/hero/dutch_ship_medium.glb` back (KTX2 kept,
Meshopt re-encoded). Run it after `optimize-assets.mjs --force` rebuilds the
hero; it refuses a file that is already split.

---

## Why these exist

Every one of them was written to answer a question that had already produced a
wrong decision by being guessed at:

- **`shells.mjs`** exists because "some rock and cliff models are one sided" is a
  claim you cannot settle by looking. It counts edges used by exactly one
  triangle, and separately measures whether the model is a plate or a facade.
  Poly Haven's coastal set turned out to be geometrically closed *and*
  effectively one-sided — `coast_line_01` is a 55 × 2.6 × 43 m wave-cut slab and
  `coastal_cliff_02` is a 41 m wall 8 m deep. Both are correct assets for one
  placement and wrong for any other.
- **`plates.mjs`** exists because every Poly Haven tree ships with the patch of
  ground it was scanned on welded under the trunk, and at scene scale that reads
  as a white dinner plate under every tree. It finds the slab as a connected
  component rather than by a height rule, which is what makes the test survive a
  tree whose low branches rest on the ground.
- **`inspect.mjs`** exists because `alphaMode: BLEND` over a JPEG diffuse — no
  alpha channel to blend — is invisible in a viewer and costs the whole kind its
  place in the opaque pass.
- **`viewer.html`** exists because two of the models that looked best in a search
  listing turned out to be untextured white meshes, and one turned out to be a
  set of twenty-four forms laid out in a display row rather than a single plant.

---

## Usage

Run from the repository root; they resolve `@gltf-transform` out of the project's
own `node_modules`.

```bash
# Everything about one file, or every .glb in a directory
node scripts/modelkit/inspect.mjs public/models/dressing/island_tree_01.glb
node scripts/modelkit/inspect.mjs public/models/dressing

# One-sidedness: open shells and plate/facade shapes
node scripts/modelkit/shells.mjs public/models/dressing

# Scan ground slabs, as connected components
node scripts/modelkit/plates.mjs public/models/dressing/island_tree_02.glb
```

### Sketchfab

Needs a token in `SKETCHFAB_API_TOKEN` or a git-ignored `sketchfab-token` file at
the repository root. Search is public; downloading is not.

```bash
# Search, CC0 and CC-BY only — the licences this repo can ship
node scripts/modelkit/sketchfab.mjs search "coral reef"
node scripts/modelkit/sketchfab.mjs search "palm tree" --animated

# Everything about one model before committing to it
node scripts/modelkit/sketchfab.mjs info 26e787f2ff2e4c0fb004c3b0210805a3

# Pull candidates somewhere the viewer can reach them
node scripts/modelkit/sketchfab.mjs get 26e787f2ff2e4c0fb004c3b0210805a3 palm_coconut
```

`get` writes to `public/models/_preview/<name>.glb`, which is git-ignored and
served by the dev server — so the viewer can load it immediately. Delete the
directory when you are done choosing.

**It refuses anything that is not CC0 or CC-BY.** That is deliberate: this
repository ships its assets, and Sketchfab's "Standard" licence does not permit
that. `info` prints the licence first for the same reason.

### Viewer

With `npm run dev` running:

```
http://127.0.0.1:5173/scripts/modelkit/viewer.html
```

Then from the browser console:

```js
await loadModel('/models/dressing/island_tree_01.glb');   // any served path
await loadModel('/models/_preview/palm_coconut.glb');
await loadModel('/models/dressing/soft_coral_set.glb', { onlyNode: 'coral11_M_coral_0' });
setView(0.6, 0.15);      // yaw, elevation in radians
setGrid(false);
__info                   // size, yMin, triangle count of what is loaded
```

`onlyNode` is what makes a kit file usable: `soft_coral_set` is twenty-four
distinct corals in one glTF and the only way to choose between them is to look at
them one at a time.

---

## Reading the output

`shells.mjs` prints two independent measures, because "one sided" turns out to
mean two different things:

```
coast_line_01     43144t  open=0.3%   plate=0.05 wall=0.79   PLATE
coastal_cliff_02  56597t  open=0.8%   plate=0.25 wall=0.21   FACADE
rock_slab_a       40000t  open=0.0%   plate=0.55 wall=0.61   all round
island_tree_01    34135t  open=181%   plate=1.06 wall=0.98   open (foliage)
```

- **`open`** is boundary edges as a percentage of triangles. Under ~3% is a
  closed solid. Foliage runs past 100% and that is correct — a canopy is
  thousands of separate leaf cards, and the measure means nothing for it.
- **`plate`** is height over footprint. Under ~0.12 is a slab that only reads
  lying flat.
- **`wall`** is depth over length. Under ~0.3 is a facade that only reads from
  the front.

A model that fails either shape test can still be the right asset — it just has
exactly one correct placement, and the scatter that places it has to know that.
