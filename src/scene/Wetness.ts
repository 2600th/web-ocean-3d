import * as THREE from 'three/webgpu';
import { float, materialColor, materialRoughness, max, mix, normalWorldGeometry, positionWorld, smoothstep, vec4 } from 'three/tsl';
import { uniform } from '../core/uniform';

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

/**
 * The hull's own water: a band at the waterline and spray where the sea hits.
 *
 * Rain wets from above; the sea wets from the side. Along the waterline the
 * planking is soaked to wherever the last wave reached, and the mark creeps back
 * down as it dries, so a hull working in a seaway carries a dark, glossy band
 * that rises and falls with the swell around her rather than a painted stripe.
 * And where a sea slaps the bow or a quarter, that stretch of the side and the
 * deck above it goes dark at once and dries over the next several seconds.
 *
 * Both are tracked on the CPU at four stations along each side — the sea height
 * from `OceanSampler` at eight points a frame — and reach the hull's shader as
 * four vec4 uniforms and the ship's inverse world matrix. The shader
 * interpolates along the hull and across the beam, so the cost is a matrix
 * multiply and a handful of mixes per hull fragment, on the ship's materials
 * only.
 */
const SHIP_CHANNEL = 'ship';
/** Seconds for the high-water mark to creep back down to the sea. */
const WATERLINE_DRY_TAU = 5;
/** Height above the wet line still damp, and the feather above that, metres. */
const WATERLINE_RUNUP = 0.12;
const WATERLINE_FEATHER = 0.35;
/** Seconds for a patch of spray to dry. Faster than a soaking: it is a film, not a soak. */
const SPRAY_DRY_TAU = 9;
/** Spray reaches the deck and the rail; past this far above the sea it thins out, metres. */
const SPRAY_REACH = [2.5, 7] as const;
/** Stations along the hull, as fractions of its length from the transom. */
const STATIONS = 4;

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

  // --- the hull's waterline and spray (see `SHIP_CHANNEL`) -------------------
  private hull: {
    object: THREE.Object3D;
    length: number;
    beam: number;
    height: (x: number, z: number) => number;
  } | null = null;
  /** World to the ship's frame, taken at render time so the band never lags the hull. */
  private readonly uShipInverse = uniform(new THREE.Matrix4()).onRenderUpdate((_, node) => {
    const m = node.value as THREE.Matrix4;
    if (this.hull) m.copy(this.hull.object.matrixWorld).invert();
    return m;
  });
  /** Hull length and beam, metres. */
  private readonly uHull = uniform(new THREE.Vector2(1, 1));
  /** 0 until a hull is tracked, so an untracked ship stays as rain alone leaves it. */
  private readonly uBandOn = uniform(0);
  /** High-water mark at each station, ship-frame metres; port and starboard. */
  private readonly uLinePort = uniform(new THREE.Vector4(-100, -100, -100, -100));
  private readonly uLineStbd = uniform(new THREE.Vector4(-100, -100, -100, -100));
  /** Spray wetness at each station, 0..1. */
  private readonly uSprayPort = uniform(new THREE.Vector4());
  private readonly uSprayStbd = uniform(new THREE.Vector4());
  private readonly line = new Float32Array(STATIONS * 2);
  private readonly sprayCells = new Float32Array(STATIONS * 2);
  /** Set on a reset: the next update takes the sea as the mark, with no history. */
  private lineFresh = true;

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
    const rainWet = this.uWetness.mul(exposure).clamp(0, 1);
    // The ship's materials also take the sea's water: the waterline and spray.
    const wet = entry.channel.name === SHIP_CHANNEL ? max(rainWet, this.hullWaterNode()) : rainWet;

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
   * The hull's waterline band and spray mask, 0..1, in the ship's frame. See
   * `SHIP_CHANNEL`.
   */
  private hullWaterNode() {
    const p = this.uShipInverse.mul(vec4(positionWorld, 1)).xyz;
    // Station coordinate 0..3 from transom to stem; piecewise linear between them.
    const t = p.x.div(this.uHull.x).add(0.5).mul(STATIONS - 1).clamp(0, STATIONS - 1);
    const along = (v: typeof this.uLinePort) =>
      mix(mix(mix(v.x, v.y, t.clamp(0, 1)), v.z, t.sub(1).clamp(0, 1)), v.w, t.sub(2).clamp(0, 1));
    const side = smoothstep(this.uHull.y.mul(-0.3), this.uHull.y.mul(0.3), p.z);
    const above = p.y.sub(mix(along(this.uLinePort), along(this.uLineStbd), side));
    const band = float(1).sub(smoothstep(WATERLINE_RUNUP, WATERLINE_RUNUP + WATERLINE_FEATHER, above));
    const spray = mix(along(this.uSprayPort), along(this.uSprayStbd), side).mul(
      float(1).sub(smoothstep(SPRAY_REACH[0], SPRAY_REACH[1], above)),
    );
    return max(band, spray).mul(this.uBandOn).clamp(0, 1);
  }

  /**
   * Starts tracking the hull's waterline: `height` is the sea surface at world
   * (x, z), `length` and `beam` the hull's in metres, `object` the hull whose
   * frame is +X forward, +Z starboard, y = 0 at the design waterline.
   */
  trackWaterline(object: THREE.Object3D, length: number, beam: number, height: (x: number, z: number) => number): void {
    this.hull = { object, length, beam, height };
    this.uHull.value.set(length, beam);
    this.uBandOn.value = 1;
    this.lineFresh = true;
  }

  /**
   * Sea thrown against the hull at world `at`, `strength` 0..1: wets that
   * stretch of side and the deck above it, which then dries over seconds.
   */
  spray(at: THREE.Vector3, strength: number): void {
    const hull = this.hull;
    if (!hull || !(strength > 0)) return;
    hull.object.updateWorldMatrix(true, false);
    const local = _local.copy(at).applyMatrix4(_inverse.copy(hull.object.matrixWorld).invert());
    const t = THREE.MathUtils.clamp((local.x / hull.length + 0.5) * (STATIONS - 1), 0, STATIONS - 1);
    const i0 = Math.min(STATIONS - 2, Math.floor(t));
    const f = t - i0;
    // Mostly the side it struck; a little over the deck to the other.
    const s = THREE.MathUtils.clamp(local.z / (hull.beam * 0.5), -1, 1);
    const amount = Math.min(1, strength * 0.9);
    for (let side = 0; side < 2; side++) {
      const share = side === 1 ? 0.5 + 0.5 * s : 0.5 - 0.5 * s;
      for (const [i, w] of [[i0, 1 - f], [i0 + 1, f]] as const) {
        const k = side * STATIONS + i;
        const add = amount * share * w;
        this.sprayCells[k] = Math.min(1, this.sprayCells[k] + add * (1 - this.sprayCells[k]));
      }
    }
    this.writeHull();
  }

  /** Sea height around the hull into the high-water marks, and the spray drying. */
  private updateHull(dt: number): void {
    const hull = this.hull;
    if (!hull) return;
    const object = hull.object;
    object.updateWorldMatrix(true, false);
    _inverse.copy(object.matrixWorld).invert();
    const down = this.lineFresh ? 1 : 1 - Math.exp(-dt / WATERLINE_DRY_TAU);
    for (let side = 0; side < 2; side++) {
      const z = (side === 0 ? -0.5 : 0.5) * hull.beam;
      for (let i = 0; i < STATIONS; i++) {
        // Stations a little inside the ends, where the hull still has sides.
        const x = (i / (STATIONS - 1) - 0.5) * hull.length * 0.9;
        const w = _local.set(x, 0, z).applyMatrix4(object.matrixWorld);
        w.y = hull.height(w.x, w.z);
        const sea = w.applyMatrix4(_inverse).y;
        const k = side * STATIONS + i;
        // Rises with the sea at once, falls back as it dries.
        const prev = this.line[k];
        this.line[k] = sea >= prev ? sea : prev + (sea - prev) * down;
      }
    }
    this.lineFresh = false;
    const dry = Math.exp(-dt / SPRAY_DRY_TAU);
    for (let k = 0; k < this.sprayCells.length; k++) this.sprayCells[k] *= dry;
    this.writeHull();
  }

  private writeHull(): void {
    const l = this.line;
    const c = this.sprayCells;
    this.uLinePort.value.set(l[0], l[1], l[2], l[3]);
    this.uLineStbd.value.set(l[4], l[5], l[6], l[7]);
    this.uSprayPort.value.set(c[0], c[1], c[2], c[3]);
    this.uSprayStbd.value.set(c[4], c[5], c[6], c[7]);
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
    this.updateHull(dt);
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
    // The hull's spray dries and its waterline forgets the last sea.
    this.sprayCells.fill(0);
    this.lineFresh = true;
    this.writeHull();
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
    this.hull = null;
    this.uBandOn.value = 0;
  }
}

const _local = /*@__PURE__*/ new THREE.Vector3();
const _inverse = /*@__PURE__*/ new THREE.Matrix4();
