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
 * ### Per object, not per scene
 *
 * Rain falls on everything, but spray does not: a sea coming green over the bow
 * soaks the ship and nothing else. This used to be one scalar for every adopted
 * material, so a wave across the deck darkened the rocks, the jetty and the
 * crates on an island a kilometre and a half away. Each adopted root now names a
 * *channel* — the ship, the island — and each channel has its own level: rain
 * drives all of them alike, `douse` drives one.
 *
 * The level reaches the shader through an object-scope uniform rather than one
 * uniform per channel, and that matters because materials can be shared: the
 * value is written per draw from the object being drawn, so one material on two
 * objects in two channels still wets each by its own amount. three does the same
 * for `modelWorldMatrix`; the object bind group is per render object. The lookup
 * is a cached parent walk, so a draw costs a map read.
 *
 * The graph is attached in `adopt`, during scene load, and driven afterwards by
 * plain numbers — so no frame in gameplay pays a shader compile.
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

/** Channel for anything adopted without naming one. */
const DEFAULT_CHANNEL = 'world';

interface Tracked {
  material: THREE.MeshStandardMaterial;
  /** Set when the wetting graph was attached, so `dispose` can detach it. */
  wired: boolean;
  /** Channel whose level a scalar-path (unwired) material follows. */
  channel: Channel;
  roughness: number;
  color: THREE.Color;
  /** The node graphs the material had before wetting wrapped them, restored on dispose. */
  colorNode: unknown;
  roughnessNode: unknown;
}

interface Channel {
  name: string;
  level: number;
}

export class SurfaceWetness {
  private readonly tracked: Tracked[] = [];
  private readonly channels = new Map<string, Channel>();
  /** Roots registered by `adopt`, and the channel each belongs to. */
  private readonly roots = new WeakMap<THREE.Object3D, Channel>();
  /** Per-object resolution cache; objects never change root in this scene. */
  private readonly resolved = new WeakMap<THREE.Object3D, Channel>();
  /**
   * The level for the object being drawn. Written per draw, so materials shared
   * across channels still wet by their own object's amount.
   */
  private readonly uWetness = uniform(0).onObjectUpdate(({ object }) =>
    object ? this.channelOf(object).level : 0,
  );
  /** Materials are shared between clones; adopting one twice would double-apply. */
  private readonly seen = new Set<THREE.Material>();

  constructor() {
    this.channel(DEFAULT_CHANNEL);
  }

  /**
   * Adopts every standard material under `root`, wetting it by `channel`.
   *
   * Safe to call for several roots — adopting a material twice would take the
   * already-darkened colour as its dry reference and walk it to black over
   * repeated calls, so a material is wired once and every object drawing it
   * reads its own channel at draw time.
   */
  adopt(root: THREE.Object3D, channel = DEFAULT_CHANNEL): void {
    const target = this.channel(channel);
    this.roots.set(root, target);
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
          channel: target,
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
   * @param rain Rain rate, 0..1, from the weather system. Drives every channel.
   */
  update(dt: number, rain: number): void {
    const target = Math.max(0, Math.min(1, rain));
    if (!(dt > 0)) return;
    for (const channel of this.channels.values()) {
      const tau = target > channel.level ? WET_TAU : DRY_TAU;
      // Exponential approach, framerate-independent.
      channel.level += (target - channel.level) * (1 - Math.exp(-dt / tau));
    }
    this.write();
  }

  /**
   * Water arriving all at once rather than as rain: spray over the bow, green
   * water across the deck. Wets one channel immediately by `amount` and then
   * dries on the same slow constant rain does, so a ship that has been taking it
   * green stays dark for the better part of a minute after the sea goes down —
   * and the island it is sailing past stays dry.
   */
  douse(amount: number, channel = DEFAULT_CHANNEL): void {
    if (!(amount > 0)) return;
    const c = this.channel(channel);
    c.level = Math.min(1, c.level + amount * (1 - c.level));
    this.write();
  }

  /** Immediate set of every channel, for deterministic capture. Skips the time constants. */
  setWetness(value: number): void {
    const v = Math.max(0, Math.min(1, value));
    for (const channel of this.channels.values()) channel.level = v;
    this.write();
  }

  /** The wettest channel, 0..1. */
  get value(): number {
    let v = 0;
    for (const channel of this.channels.values()) v = Math.max(v, channel.level);
    return v;
  }

  /** One channel's level, 0..1; 0 for a channel nothing has used. */
  level(channel: string): number {
    return this.channels.get(channel)?.level ?? 0;
  }

  private channel(name: string): Channel {
    let c = this.channels.get(name);
    if (c === undefined) {
      c = { name, level: this.channels.get(DEFAULT_CHANNEL)?.level ?? 0 };
      this.channels.set(name, c);
    }
    return c;
  }

  /** The channel of the nearest adopted ancestor, cached per object. */
  private channelOf(object: THREE.Object3D): Channel {
    const cached = this.resolved.get(object);
    if (cached !== undefined) return cached;
    let found: Channel | undefined;
    for (let node: THREE.Object3D | null = object; node !== null; node = node.parent) {
      found = this.roots.get(node);
      if (found !== undefined) break;
    }
    const channel = found ?? this.channel(DEFAULT_CHANNEL);
    // Only cache a hit: an object drawn before its root was adopted (or before
    // it was parented) must be able to find its channel later.
    if (found !== undefined) this.resolved.set(object, channel);
    return channel;
  }

  private write(): void {
    // Only entries that could not take the graph need the scalar treatment; a
    // wired material reads its level per draw and would otherwise be darkened
    // twice, once here and once in its own shader.
    for (const entry of this.tracked) {
      if (entry.wired) continue;
      const w = entry.channel.level;
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
