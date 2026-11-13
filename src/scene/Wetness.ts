import * as THREE from 'three/webgpu';
import { float, materialColor, materialRoughness, mix, normalWorldGeometry, uniform } from 'three/tsl';

/**
 * Rain wetting for loaded glTF surfaces.
 *
 * Two things happen to a surface under rain, and both are cheap to express:
 *
 * **It darkens.** A water film is optically thicker than air, so light that
 * scatters out of the substrate is more likely to be reflected back into it at
 * the film's top surface and absorbed on a second pass. The albedo a viewer
 * measures therefore falls, typically by a third to a half on porous materials
 * like wood and canvas — which is most of this ship.
 *
 * **It gets glossier.** The film fills the microfacet valleys, so the effective
 * roughness collapses toward that of a flat water surface. This is what actually
 * reads as "wet" — the darkening alone looks like a lighting change.
 *
 * **And it is not uniform.** Rain falls downward, so the deck and the upper faces
 * of the rail soak while the underside of a beam stays dry. Wetting the whole
 * object equally is the giveaway that made the first version read as a lighting
 * change rather than as weather: nothing in the frame told you where the water
 * was coming from.
 *
 * That last part was previously written off as too expensive, on the grounds that
 * it needs the world normal and therefore a node graph per material, with a
 * rebuild of materials the asset loader shares between clones. That turned out to
 * be wrong on both counts. The loader already converts every glTF material to
 * `MeshPhysicalNodeMaterial`, so there is a node graph either way; and three
 * publishes `materialColor` and `materialRoughness`, which are precisely the
 * `color * map` and `roughness * roughnessMap.g` compositions its own shading
 * would have built. Wrapping those keeps the glTF's textures composing exactly as
 * before and needs no knowledge of which maps a given material happens to carry.
 *
 * Sharing is a feature here rather than a problem: the mask is evaluated per
 * fragment from the geometric normal, so one material serves every clone and each
 * still wets according to how its own surfaces are turned.
 *
 * The graph is attached in `adopt`, during scene load, and driven afterwards by a
 * single uniform — so no frame in gameplay pays a shader compile, and there is no
 * per-material CPU work per frame at all.
 */

/** Roughness a fully wetted surface tends toward. Flat water is ~0.05. */
const WET_ROUGHNESS = 0.09;

/** Fraction of its dry albedo a fully wetted surface keeps. */
const WET_ALBEDO = 0.58;

/**
 * How wet a downward-facing surface gets relative to an upward-facing one.
 *
 * Not zero. The underside of a rail over the sea is in spray and in the humid
 * air under a squall, and water creeps around an edge by capillarity rather than
 * stopping dead at the silhouette. Zero produces a hard terminator exactly at the
 * horizontal, which reads as a shading bug; this keeps the contrast that carries
 * the effect while leaving the transition believable.
 */
const SHELTERED_WETNESS = 0.15;

/**
 * Time constants, seconds.
 *
 * Deliberately asymmetric, because the physics is. A surface wets as fast as
 * rain lands on it — a few seconds in any real downpour — and then dries by
 * evaporation, which at sea, in wind, still takes the better part of a minute.
 * Equal constants make rain look like a switch.
 */
const WET_TAU = 3.5;
const DRY_TAU = 26;

interface Tracked {
  material: THREE.MeshStandardMaterial;
  /** Set when the wetting graph was attached, so `dispose` can detach it. */
  wired: boolean;
  roughness: number;
  color: THREE.Color;
  /** The node graphs the material had before wetting wrapped them, restored on dispose. */
  colorNode: unknown;
  roughnessNode: unknown;
}

export class SurfaceWetness {
  private readonly tracked: Tracked[] = [];
  private wetness = 0;
  /**
   * The one thing that changes per frame. Every adopted material reads it, so
   * updating wetness is a single uniform write regardless of how many materials
   * the scene loaded.
   */
  private readonly uWetness = uniform(0);
  /** Materials are shared between clones; adopting one twice would double-apply. */
  private readonly seen = new Set<THREE.Material>();

  /**
   * Adopts every standard material under `root`.
   *
   * Safe to call for several roots — the ship and the props share materials
   * through the asset loader's cache, and adopting a material twice would take
   * the already-darkened colour as its dry reference and walk it to black over
   * repeated calls.
   */
  adopt(root: THREE.Object3D): void {
    root.traverse((node) => {
      const mesh = node as THREE.Mesh;
      if (!mesh.isMesh) return;
      const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
      for (const material of materials) {
        if (this.seen.has(material)) continue;
        this.seen.add(material);
        const standard = material as THREE.MeshStandardMaterial;
        // Duck-typed, not flag-tested. The asset loader converts every glTF
        // material to `MeshPhysicalNodeMaterial`, which sets
        // `isMeshPhysicalNodeMaterial` and *not* `isMeshStandardMaterial` — so a
        // check against the classic flag silently adopts nothing, and the effect
        // is wired, tested for existence, and completely inert. What this needs
        // is a numeric `roughness` and a `Color`, so that is what it asks for.
        if (typeof standard.roughness !== 'number') continue;
        if (!(standard.color instanceof THREE.Color)) continue;
        const entry: Tracked = {
          material: standard,
          wired: false,
          roughness: standard.roughness,
          color: standard.color.clone(),
          colorNode: (standard as unknown as { colorNode?: unknown }).colorNode ?? null,
          roughnessNode: (standard as unknown as { roughnessNode?: unknown }).roughnessNode ?? null,
        };
        this.wire(entry);
        this.tracked.push(entry);
      }
    });
    // Re-apply so a root adopted after rain has started matches the rest.
    this.write();
  }

  /**
   * Attaches the wetting graph, or leaves the material alone if it has no node
   * slots to attach to.
   *
   * The fallback matters more than it looks. Everything the asset loader produces
   * is a node material, but `adopt` is duck-typed on purpose and a caller is free
   * to hand it a plain `MeshStandardMaterial`; silently wiring nothing would give
   * that caller a surface that never wets, with no indication why. So the scalar
   * path stays as the fallback and `wired` records which one an entry is on.
   */
  private wire(entry: Tracked): void {
    const material = entry.material as THREE.MeshStandardMaterial & {
      colorNode?: unknown;
      roughnessNode?: unknown;
    };
    if (!('colorNode' in material) || !('roughnessNode' in material)) return;

    // How exposed this fragment is to falling rain, from the geometric normal:
    // 1 straight up, `SHELTERED_WETNESS` straight down, and a smooth transition
    // through the vertical rather than a step at it.
    //
    // The *geometric* normal, not the shading one. A planked deck's normal map
    // turns individual boards by a few degrees and none of that changes where the
    // rain lands; feeding the mapped normal in would stipple the wetness with the
    // texture's detail, which is the same class of mistake as lighting a surface
    // by its bump map alone.
    const exposure = mix(
      float(SHELTERED_WETNESS),
      float(1),
      normalWorldGeometry.y.mul(0.5).add(0.5).clamp(0, 1),
    );
    const wet = this.uWetness.mul(exposure).clamp(0, 1);

    // `materialColor` and `materialRoughness` are three's own compositions of the
    // material's scalars with its maps, so wrapping them preserves the glTF's
    // textures exactly and works whether or not a given material has any.
    //
    // A material that already carries its own colour or roughness graph — the
    // island's procedural rock and set dressing — has that graph wrapped instead,
    // so wetting composes with it rather than replacing it.
    const dryColor = (entry.colorNode ?? materialColor) as typeof materialColor;
    const dryRoughness = (entry.roughnessNode ?? materialRoughness) as typeof materialRoughness;
    material.colorNode = dryColor.mul(mix(float(1), float(WET_ALBEDO), wet));
    material.roughnessNode = mix(dryRoughness, float(WET_ROUGHNESS), wet);
    material.needsUpdate = true;
    entry.wired = true;
  }

  /**
   * @param rain Rain rate, 0..1, from the weather system.
   */
  update(dt: number, rain: number): void {
    const target = Math.max(0, Math.min(1, rain));
    if (!(dt > 0)) return;
    const tau = target > this.wetness ? WET_TAU : DRY_TAU;
    // Exponential approach, framerate-independent.
    this.wetness += (target - this.wetness) * (1 - Math.exp(-dt / tau));
    this.write();
  }

  /**
   * Water arriving all at once rather than as rain: spray over the bow, green
   * water across the deck. Wets immediately by `amount` and then dries on the
   * same slow constant rain does, so a ship that has been taking it green stays
   * dark for the better part of a minute after the sea goes down.
   */
  douse(amount: number): void {
    if (!(amount > 0)) return;
    this.wetness = Math.min(1, this.wetness + amount * (1 - this.wetness));
    this.write();
  }

  /** Immediate set, for deterministic capture. Skips the time constants. */
  setWetness(value: number): void {
    this.wetness = Math.max(0, Math.min(1, value));
    this.write();
  }

  get value(): number {
    return this.wetness;
  }

  private write(): void {
    const w = this.wetness;
    this.uWetness.value = w;
    // Only entries that could not take the graph need the scalar treatment; a
    // wired material would otherwise be darkened twice, once here and once in
    // its own shader.
    for (const entry of this.tracked) {
      if (entry.wired) continue;
      if (w < 1e-4) {
        entry.material.roughness = entry.roughness;
        entry.material.color.copy(entry.color);
        continue;
      }
      entry.material.roughness = entry.roughness + (WET_ROUGHNESS - entry.roughness) * w;
      entry.material.color.copy(entry.color).multiplyScalar(1 + (WET_ALBEDO - 1) * w);
    }
  }

  /** Restores every adopted material to its dry state and forgets it. */
  dispose(): void {
    this.setWetness(0);
    for (const entry of this.tracked) {
      if (!entry.wired) continue;
      const material = entry.material as unknown as {
        colorNode: unknown;
        roughnessNode: unknown;
        needsUpdate: boolean;
      };
      material.colorNode = entry.colorNode;
      material.roughnessNode = entry.roughnessNode;
      material.needsUpdate = true;
    }
    this.tracked.length = 0;
    this.seen.clear();
  }
}
