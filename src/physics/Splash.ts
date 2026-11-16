import * as THREE from 'three/webgpu';
import {
  Fn,
  attribute,
  cameraFar,
  cameraNear,
  cameraPosition,
  cameraViewMatrix,
  dFdx,
  dFdy,
  exp,
  float,
  mix,
  modelWorldMatrixInverse,
  perspectiveDepthToViewZ,
  positionGeometry,
  positionWorld,
  smoothstep,
  texture,
  uniform,
  uniformArray,
  varyingProperty,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { smoothstepDown } from '../core/tslMath';
import {
  CLASS_CLUSTER,
  CLASS_DROP,
  CLASS_MIST,
  SPRAY_ALPHA_FLOOR,
  particleVertex,
  sharedSceneDepth,
  sheetRadiance,
  sprayAtlas,
  sprayMaterial,
  sprayUniforms,
} from './sprayLook';
import { MAX_RINGS, type Wake } from './Wake';

/**
 * Something hitting the water: an anchor let go, a cask or a spar coming down
 * off a swell, cargo hauled out. One look for all of them, the same water as
 * the bow sheet and the spray, scaled by what hit and how hard.
 *
 * ### What a splash is (and what it was)
 *
 * It was a burst of glowing additive discs — "cotton puffs" — for the anchor,
 * and a fan of spray particles for a floater. Footage of objects entering water
 * (and the fluid-mechanics literature on it: Worthington's *A Study of
 * Splashes*, 1908; Aristoff & Bush, "Water entry of small hydrophobic spheres",
 * JFM 619, 2009; Truscott, Epps & Belden, "Water entry of projectiles", Annu.
 * Rev. Fluid Mech. 46, 2014) shows four parts, and this draws each:
 *
 *  1. **The crown.** The object pushes water aside and up as a thin sheet round
 *     the cavity. Fluid leaves the cavity's rim fast at first and slower as the
 *     cavity widens; each parcel then flies ballistically. So the crown at time
 *     `t` is the locus of everything ejected so far — the earliest, fastest
 *     water at its rim, the latest at its base — and it rises, leans out,
 *     thins, and its rim corrugates into fingers that pinch off as drops
 *     (Rayleigh–Plateau). A mesh, one surface, torn by the spray atlas's
 *     ligament field toward the rim and as it ages, exactly as the bow sheet is.
 *  2. **The jet.** A heavy, fast object drags an air cavity down behind it; the
 *     cavity collapses and fires a Worthington jet straight up, its tip
 *     breaking into drops. Only above a few hundred joules of impact.
 *  3. **Drops and mist.** Velocity-stretched drops and clusters off the crown's
 *     rim and the jet's tip on ballistic arcs (`sprayLook`), and a faint mist for
 *     large impacts only. Where drops land they leave flecks of foam.
 *  4. **The surface.** A disc of churned foam at the entry, a ring wave that
 *     spreads from it carrying a band of foam (`Wake.setRing`: foam in the hull
 *     channel, ripples on the elevation channel, so the sea heaves and its
 *     normals ring), and a second patch of aerated water where the jet falls.
 *
 * ### Scale
 *
 * `fire(…, radius, speed, mass)`: the impact's kinetic energy sets how big and
 * how complete the splash is (a 60 J plop off a cask is a ten-centimetre crown
 * and a ring; a 9 kJ anchor stands a metre of crown, two of jet, and mist), the
 * speed sets the launch, and the radius the cavity.
 *
 * ### Cost
 *
 * Nothing per frame on the CPU but a few uniforms, no allocation after
 * construction; every position is closed form in the vertex stage from a
 * four-slot event table, like `Spray`. All three meshes are hidden when no
 * splash is alive, so an idle splash system draws nothing.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */

/** Splashes alive at once: a ring of slots, the oldest overwritten. One ring wave each. */
const SLOTS = MAX_RINGS;
/** Particles per splash. */
const PER_SLOT = 160;
const GRAVITY = 9.81;

const CROWN_AROUND = 64;
const CROWN_ALONG = 10;
const JET_AROUND = 12;
const JET_ALONG = 16;

/** Pending foam flecks (drops landing, the jet's fall). */
const MAX_PENDING = 32;

/** Crown launch elevation at first ejection and its fall-off, radians. */
const PHI0 = 1.25;
const PHI_LEAN = 0.3;

/**
 * Class and source per particle within a splash: crown-rim drops, clusters and
 * mist, and drops from the jet's tip (`true`).
 */
const PARTICLE_LAYOUT: readonly [number, boolean][] = [
  [CLASS_DROP, false],
  [CLASS_DROP, false],
  [CLASS_CLUSTER, false],
  [CLASS_DROP, false],
  [CLASS_DROP, true],
  [CLASS_CLUSTER, false],
  [CLASS_MIST, false],
  [CLASS_DROP, true],
];

const vCrown = /*@__PURE__*/ varyingProperty('vec4', 'splashCrown');
const vCrownMisc = /*@__PURE__*/ varyingProperty('vec4', 'splashCrownMisc');

export interface SplashOptions {
  /** Radius of what hit, metres: the cavity. */
  radius: number;
  /** Speed at which it hit the surface, m/s. */
  speed: number;
  /** Its mass, kg. */
  mass: number;
}

export class Splash {
  /** Root of everything this draws. Added to the scene by the caller. */
  readonly object = new THREE.Group();

  private readonly uClock = uniform(0);
  /** Per slot: (x, water y, z, t0). t0 far in the past switches a slot off. */
  private readonly uA = uniformArray(Array.from({ length: SLOTS }, () => new THREE.Vector4(0, 0, 0, -1000)));
  /** Per slot: (cavity radius, crown launch speed, ejection time, seed). */
  private readonly uB = uniformArray(Array.from({ length: SLOTS }, () => new THREE.Vector4(0.1, 1, 0.1, 0)));
  /** Per slot: (jet speed, jet start, jet duration, mist amount). Jet speed 0: no jet. */
  private readonly uC = uniformArray(Array.from({ length: SLOTS }, () => new THREE.Vector4()));
  /** Per slot: (particle life, finger count, crown life, size 0..1). */
  private readonly uD = uniformArray(Array.from({ length: SLOTS }, () => new THREE.Vector4(1, 16, 1, 0)));
  /** Per slot: (water y at the impact point now, -, -, -). */
  private readonly uE = uniformArray(Array.from({ length: SLOTS }, () => new THREE.Vector4()));

  private readonly crown: THREE.Mesh;
  private readonly jet: THREE.Mesh;
  private readonly particles: THREE.Mesh;
  private readonly materials: THREE.Material[] = [];
  private readonly geometries: THREE.BufferGeometry[] = [];

  private wake: Wake | null = null;
  private surface: ((x: number, z: number) => number) | null = null;
  private clock = 0;
  private cursor = 0;
  private serial = 0;
  /** Per slot, CPU side: start, end of life, ring speed, ring life, foam, amplitude, wavelength, width, x, z. */
  private readonly slotStart = new Float64Array(SLOTS).fill(-1000);
  private readonly slotEnd = new Float64Array(SLOTS).fill(-1000);
  private readonly ringSpeed = new Float32Array(SLOTS);
  private readonly ringLife = new Float32Array(SLOTS);
  private readonly ringFoam = new Float32Array(SLOTS);
  private readonly ringAmp = new Float32Array(SLOTS);
  private readonly ringWave = new Float32Array(SLOTS);
  private readonly ringWidth = new Float32Array(SLOTS);
  private readonly ringR0 = new Float32Array(SLOTS);
  private readonly slotX = new Float32Array(SLOTS);
  private readonly slotZ = new Float32Array(SLOTS);
  /** Pending foam: x, z, radius, amount, time. */
  private readonly pending = new Float64Array(MAX_PENDING * 5);
  private pendingCount = 0;

  constructor() {
    this.object.name = 'splash';

    // --- the crown ---------------------------------------------------------
    const crownGeometry = gridGeometry(CROWN_AROUND, CROWN_ALONG);
    const crownMaterial = sheetMaterial(() => this.crownVertex(), () => this.crownFragment());
    this.crown = new THREE.Mesh(crownGeometry, crownMaterial);
    this.crown.name = 'splash-crown';

    // --- the jet -----------------------------------------------------------
    const jetGeometry = gridGeometry(JET_AROUND, JET_ALONG);
    const jetMaterial = sheetMaterial(() => this.jetVertex(), () => this.jetFragment());
    this.jet = new THREE.Mesh(jetGeometry, jetMaterial);
    this.jet.name = 'splash-jet';

    // --- drops, clusters, mist --------------------------------------------
    const count = SLOTS * PER_SLOT;
    const quad = new THREE.InstancedBufferGeometry();
    quad.setAttribute(
      'position',
      new THREE.Float32BufferAttribute([-0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0], 3),
    );
    quad.setIndex([0, 1, 2, 0, 2, 3]);
    const slot = new Float32Array(count);
    const seed = new Float32Array(count * 4);
    const cls = new Float32Array(count);
    const fromJet = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      slot[i] = Math.floor(i / PER_SLOT);
      const k = i % PER_SLOT;
      for (let c = 0; c < 4; c++) seed[i * 4 + c] = hash(k, 31 + c);
      const [c, j] = PARTICLE_LAYOUT[k % PARTICLE_LAYOUT.length];
      cls[i] = c;
      fromJet[i] = j ? 1 : 0;
    }
    quad.setAttribute('splashSlot', new THREE.InstancedBufferAttribute(slot, 1));
    quad.setAttribute('splashSeed', new THREE.InstancedBufferAttribute(seed, 4));
    quad.setAttribute('splashClass', new THREE.InstancedBufferAttribute(cls, 1));
    quad.setAttribute('splashJet', new THREE.InstancedBufferAttribute(fromJet, 1));
    quad.instanceCount = count;
    quad.boundingSphere = new THREE.Sphere(new THREE.Vector3(), Number.POSITIVE_INFINITY);
    const particleMaterial = sprayMaterial(() => this.particleVertex(), false);
    this.particles = new THREE.Mesh(quad, particleMaterial);
    this.particles.name = 'splash-drops';
    this.particles.renderOrder = 6;

    for (const mesh of [this.crown, this.jet, this.particles]) {
      mesh.frustumCulled = false;
      mesh.castShadow = false;
      mesh.receiveShadow = false;
      mesh.matrixAutoUpdate = false;
      mesh.visible = false;
      this.object.add(mesh);
    }
    this.crown.renderOrder = 5;
    this.jet.renderOrder = 5;
    this.materials.push(crownMaterial, jetMaterial, particleMaterial);
    this.geometries.push(crownGeometry, jetGeometry, quad);
  }

  /** Where the foam and the ring waves go. */
  setWake(wake: Wake | null): void {
    this.wake = wake;
  }

  /**
   * The sea's height at (x, z), so a splash rides the swell it landed in: the
   * crown and the jet stand on the water as it is now, not as it was at impact
   * (a metre of swell in a second would otherwise bury or strand them).
   */
  setSurface(height: ((x: number, z: number) => number) | null): void {
    this.surface = height;
  }

  /**
   * Something hit the water at (x, z), the surface there at `y`.
   */
  fire(x: number, y: number, z: number, options: SplashOptions): void {
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return;
    const speed = Math.min(30, Math.max(0.3, options.speed));
    const radius = Math.min(1.2, Math.max(0.04, options.radius));
    const energy = Math.max(1, 0.5 * Math.max(0.1, options.mass) * speed * speed);
    const log = Math.log10(energy);
    // Size 0 at ~10 J (a dropped bottle), 1 at ~30 kJ (a cannonball).
    const size = clamp01((log - 1) / 3.5);
    // A jet once the cavity is worth collapsing: ~400 J up, full by ~6 kJ.
    const heavy = smooth01((log - 2.6) / 1.2);
    // Mist only for the big ones.
    const mist = smooth01((log - 3.3) / 1.0);

    const slot = this.cursor % SLOTS;
    this.cursor++;
    this.serial++;
    const s = hash(this.serial, 7);

    const crownSpeed = Math.min(4.8, (0.5 * speed + 0.8) * (0.8 + 0.4 * size));
    const eject = 0.05 + 0.1 * Math.sqrt(radius) * (0.6 + 0.8 * size);
    const jetSpeed = heavy > 0.02 ? Math.min(9, Math.max(1.5, 0.8 * speed)) * (0.55 + 0.45 * heavy) : 0;
    const jetStart = 1.3 * eject + 0.04 + 0.08 * radius;
    const jetSpan = 0.08 + 0.12 * radius;
    // Fingers round the rim: an even count, more for a wider, harder crown.
    const fingers = 2 * Math.round(5 + 9 * radius + 3 * size);
    const crownLife = eject + (2 * crownSpeed * Math.sin(PHI0)) / GRAVITY + 0.15;
    const jetLife = jetSpeed > 0 ? jetStart + jetSpan + (2 * jetSpeed) / GRAVITY + 0.2 : 0;
    const particleLife = (2 * crownSpeed * 1.5) / GRAVITY + 0.4 + 1.4 * mist;
    const life = Math.max(crownLife, jetLife, particleLife);

    (this.uA.array as THREE.Vector4[])[slot].set(x, y, z, this.clock);
    (this.uE.array as THREE.Vector4[])[slot].set(y, 0, 0, 0);
    (this.uB.array as THREE.Vector4[])[slot].set(radius, crownSpeed, eject, s);
    (this.uC.array as THREE.Vector4[])[slot].set(jetSpeed, jetStart, jetSpan, mist);
    (this.uD.array as THREE.Vector4[])[slot].set(Math.max(particleLife, jetLife), fingers, crownLife, size);

    this.slotStart[slot] = this.clock;
    this.slotX[slot] = x;
    this.slotZ[slot] = z;
    // The ring: a short gravity wave a metre or two long spreading at about its
    // group speed, carrying a band of foam that thins as it spreads.
    const wavelength = 1.4 + 1.4 * size;
    this.ringWave[slot] = wavelength;
    this.ringSpeed[slot] = 0.5 * Math.sqrt((GRAVITY * wavelength) / (2 * Math.PI)) + 0.3;
    this.ringLife[slot] = 2.5 + 2 * size;
    this.ringFoam[slot] = 0.5 + 0.9 * size;
    this.ringAmp[slot] = (0.01 + 0.07 * size) * Math.min(1, radius / 0.3 + 0.3);
    this.ringWidth[slot] = 0.35 + 0.5 * size;
    this.ringR0[slot] = radius * 1.5;
    this.slotEnd[slot] = this.clock + Math.max(life, this.ringLife[slot]);

    // Churned water at the entry now; where drops come down, flecks as they
    // land; and where the jet falls back, a second patch of aerated water.
    this.wake?.splash(x, z, radius * 1.6 + crownSpeed * 0.1, 0.22 + 0.28 * size);
    const flecks = 3 + Math.round(5 * size);
    for (let k = 0; k < flecks; k++) {
      const angle = (k / flecks + hash(this.serial, 40 + k) * 0.6) * Math.PI * 2;
      const u = crownSpeed * (1.1 + 0.4 * hash(this.serial, 60 + k));
      const flight = (2 * u * Math.sin(PHI0 - 0.1)) / GRAVITY;
      const reach = radius * 1.3 + u * Math.cos(PHI0 - 0.1) * flight;
      this.queue(x + Math.cos(angle) * reach, z + Math.sin(angle) * reach, 0.5, 0.12 + 0.2 * size, this.clock + flight);
    }
    if (jetSpeed > 0) {
      this.queue(x, z, radius * 1.8 + 0.3, 0.18 + 0.22 * heavy, this.clock + jetStart + jetSpan + (2 * jetSpeed) / GRAVITY);
    }
    this.setVisible(true);
  }

  /** Advances the clock to `elapsed` (simulation seconds); deposits foam and drives the rings. */
  update(elapsed: number): void {
    this.clock = elapsed;
    this.uClock.value = elapsed;

    // Foam that comes due: drops landing, a jet falling back.
    for (let i = 0; i < this.pendingCount; ) {
      const b = i * 5;
      if (this.pending[b + 4] <= elapsed) {
        this.wake?.splash(this.pending[b], this.pending[b + 1], this.pending[b + 2], this.pending[b + 3]);
        const last = (this.pendingCount - 1) * 5;
        for (let c = 0; c < 5; c++) this.pending[b + c] = this.pending[last + c];
        this.pendingCount--;
      } else {
        i++;
      }
    }

    let alive = false;
    for (let slot = 0; slot < SLOTS; slot++) {
      const age = elapsed - this.slotStart[slot];
      if (elapsed < this.slotEnd[slot] && age >= 0) {
        alive = true;
        const y = this.surface ? this.surface(this.slotX[slot], this.slotZ[slot]) : NaN;
        if (Number.isFinite(y)) (this.uE.array as THREE.Vector4[])[slot].x = y;
      }
      if (!this.wake) continue;
      const ringAge = age / this.ringLife[slot];
      if (!(age >= 0) || ringAge >= 1) {
        this.wake.setRing(slot, 0, 0, 0, 1, 0, 0, 1);
        continue;
      }
      const radius = this.ringR0[slot] + this.ringSpeed[slot] * age;
      // Foam thins as the band spreads over a longer circumference, and the
      // ripple decays as it spreads (∝ r^-1/2) and as it is damped.
      const spread = Math.sqrt(this.ringR0[slot] / Math.max(this.ringR0[slot], radius));
      const fade = (1 - ringAge) * (1 - ringAge);
      this.wake.setRing(
        slot,
        this.slotX[slot],
        this.slotZ[slot],
        radius,
        this.ringWidth[slot] * (1 + 0.3 * ringAge),
        this.ringFoam[slot] * fade * spread,
        this.ringAmp[slot] * fade * spread * Math.min(1, age * 6),
        this.ringWave[slot],
      );
    }
    this.setVisible(alive);
  }

  /** Clears every splash and pending fleck: the state at `elapsed` with nothing in the air. */
  resetClock(elapsed: number): void {
    this.clock = elapsed;
    this.uClock.value = elapsed;
    this.cursor = 0;
    this.serial = 0;
    this.pendingCount = 0;
    for (let slot = 0; slot < SLOTS; slot++) {
      (this.uA.array as THREE.Vector4[])[slot].set(0, 0, 0, -1000);
      (this.uE.array as THREE.Vector4[])[slot].set(0, 0, 0, 0);
      this.slotStart[slot] = -1000;
      this.slotEnd[slot] = -1000;
      this.wake?.setRing(slot, 0, 0, 0, 1, 0, 0, 1);
    }
    this.setVisible(false);
  }

  dispose(): void {
    for (const m of this.materials) m.dispose();
    for (const g of this.geometries) g.dispose();
    this.object.removeFromParent();
  }

  private setVisible(on: boolean): void {
    this.crown.visible = on;
    this.jet.visible = on;
    this.particles.visible = on;
  }

  private queue(x: number, z: number, radius: number, amount: number, time: number): void {
    if (this.pendingCount >= MAX_PENDING) return;
    const b = this.pendingCount * 5;
    this.pending[b] = x;
    this.pending[b + 1] = z;
    this.pending[b + 2] = radius;
    this.pending[b + 3] = amount;
    this.pending[b + 4] = time;
    this.pendingCount++;
  }

  // ------------------------------------------------------------------ shaders

  /** Slot record for this vertex. */
  private slot(): { A: any; B: any; C: any; D: any; T: any; base: any } {
    const index: any = (attribute('splashSlot', 'float') as any).toInt();
    const A: any = (this.uA.element(index) as any).toVar('splashA');
    const B: any = (this.uB.element(index) as any).toVar('splashB');
    const C: any = (this.uC.element(index) as any).toVar('splashC');
    const D: any = (this.uD.element(index) as any).toVar('splashD');
    const T: any = this.uClock.sub(A.w).toVar('splashT');
    // The impact point on the water as it stands now.
    const base: any = vec3(A.x, (this.uE.element(index) as any).x, A.z).toVar('splashBase');
    return { A, B, C, D, T, base };
  }

  /**
   * The crown as the locus of fluid ejected so far: the parcel at `tau` left
   * the cavity's rim at `te = (1 - tau) min(T, Te)` with a speed that decays
   * over the ejection and an elevation that leans out, and has flown since.
   */
  private crownVertex(): any {
    const { B, D, T, base: origin } = this.slot();
    const radius: any = B.x;
    const launch: any = B.y;
    const eject: any = B.z.max(1e-3);
    const fingers: any = D.y;
    const life: any = D.z;
    const theta: any = (positionGeometry as any).x.mul(Math.PI * 2).toVar('crownTheta');
    const tau: any = (positionGeometry as any).y.toVar('crownTau');

    const te: any = tau.oneMinus().mul(T.clamp(0, eject)).toVar('crownTe');
    const q: any = te.div(eject).toVar('crownQ');
    const tf: any = T.sub(te).max(0).toVar('crownTf');
    // The rim's corrugation: alternate stations of the earliest water a little
    // faster, which lifts them into the fingers the rim tears into.
    const corrugation: any = theta.mul(fingers).add(B.w.mul(Math.PI * 2)).sin();
    const speed: any = launch.mul(q.mul(-0.6).add(1)).mul(corrugation.mul(tau).mul(0.12).add(1));
    const phi: any = q.mul(-PHI_LEAN).add(PHI0);
    const base: any = radius.mul(q.mul(0.5).add(1));
    const r: any = base.add(speed.mul(phi.cos()).mul(tf));
    const h: any = speed.mul(phi.sin()).mul(tf).sub(tf.mul(tf).mul(0.5 * GRAVITY)).toVar('crownH');
    const alive: any = T.greaterThanEqual(0).and(T.lessThanEqual(life)).select(float(1), float(0));
    const local: any = vec3(theta.cos().mul(r), h, theta.sin().mul(r)).mul(alive);
    const world: any = origin.add(local).toVar('crownWorld');

    const viewZ: any = cameraViewMatrix.mul(vec4(world, 1)).z;
    vCrown.assign(vec4(theta, tau, T.div(life.max(1e-3)), h));
    vCrownMisc.assign(vec4(viewZ, fingers, radius, B.w));
    return modelWorldMatrixInverse.mul(vec4(world, 1)).xyz;
  }

  /**
   * The Worthington jet: fluid fired up the cavity's axis from its collapse,
   * fastest first, so the tip (`tau` 1) is the earliest water. Thin, with a
   * bulb at the tip that pinches off.
   */
  private jetVertex(): any {
    const { B, C, T, base: origin } = this.slot();
    const jetSpeed: any = C.x;
    const start: any = C.y;
    const span: any = C.z.max(1e-3);
    const theta: any = (positionGeometry as any).x.mul(Math.PI * 2).toVar('jetTheta');
    const tau: any = (positionGeometry as any).y.toVar('jetTau');

    const fired: any = T.sub(start).clamp(0, span);
    const te: any = start.add(tau.oneMinus().mul(fired)).toVar('jetTe');
    const tf: any = T.sub(te).max(0).toVar('jetTf');
    const w: any = jetSpeed.mul(te.sub(start).div(span).mul(-0.6).add(1));
    const h: any = w.mul(tf).sub(tf.mul(tf).mul(0.5 * GRAVITY)).toVar('jetH');
    const life: any = start.add(span).add(jetSpeed.mul(2 / GRAVITY)).add(0.2);
    const rho: any = B.x
      .mul(0.26)
      .mul(tau.mul(-0.6).add(1))
      .mul(smoothstep(0.86, 1, tau).mul(0.9).add(1))
      .mul(smoothstep(0, 0.05, tf).mul(0.7).add(0.3));
    const alive: any = jetSpeed
      .greaterThan(0)
      .and(T.greaterThanEqual(start))
      .and(T.lessThanEqual(life))
      .select(float(1), float(0));
    const local: any = vec3(theta.cos().mul(rho), h, theta.sin().mul(rho)).mul(alive);
    const world: any = origin.add(local).toVar('jetWorld');

    const viewZ: any = cameraViewMatrix.mul(vec4(world, 1)).z;
    vCrown.assign(vec4(theta, tau, T.sub(start).div(life.sub(start).max(1e-3)), h));
    vCrownMisc.assign(vec4(viewZ, 6, B.x, B.w));
    return modelWorldMatrixInverse.mul(vec4(world, 1)).xyz;
  }

  private crownFragment(): any {
    return sheetFragment({ rimFingers: true });
  }

  private jetFragment(): any {
    return sheetFragment({ rimFingers: false });
  }

  /** Drops and clusters off the crown's rim and the jet's tip, and mist. */
  private particleVertex(): any {
    const { A, B, C, D, T, base: origin } = this.slot();
    const h: any = attribute('splashSeed', 'vec4');
    const cls: any = attribute('splashClass', 'float');
    const fromJet: any = attribute('splashJet', 'float');

    const isDrop: any = smoothstepDown(cls, 0.4, 0.5).toVar('splashDrop');
    const isMist: any = smoothstep(1.5, 1.6, cls).mul(smoothstepDown(cls, 2.4, 2.5)).toVar('splashMist');
    const isCluster: any = smoothstep(2.5, 2.6, cls).toVar('splashCluster');

    const radius: any = B.x;
    const launch: any = B.y;
    const eject: any = B.z;
    const jetSpeed: any = C.x;
    const mistAmount: any = C.w;
    const size: any = D.w;

    const theta: any = h.y.mul(Math.PI * 2);
    const dir: any = vec3(theta.cos(), 0, theta.sin());

    // Off the crown: a parcel ejected in the first part of the ejection, a
    // little faster and steeper or shallower than the sheet, so it flies clear
    // of the rim as the fingers pinch off.
    const teRim: any = h.x.mul(eject).mul(0.8);
    const qRim: any = h.x.mul(0.8);
    const uRim: any = launch.mul(qRim.mul(-0.6).add(1)).mul(h.z.mul(0.45).add(1.05)).mul(isMist.mul(-0.35).add(1));
    const phiRim: any = qRim.mul(-PHI_LEAN).add(PHI0).add(h.w.sub(0.5).mul(0.5));
    const p0Rim: any = A.xyz.add(dir.mul(radius.mul(qRim.mul(0.5).add(1))));
    const v0Rim: any = dir.mul(uRim.mul(phiRim.cos())).add(vec3(0, uRim.mul(phiRim.sin()), 0));

    // Off the jet's tip, just after it is fired.
    const teJet: any = C.y.add(h.x.mul(C.z).mul(0.6));
    const wJet: any = jetSpeed.mul(h.x.mul(-0.36).add(1)).mul(h.z.mul(0.15).add(1));
    const p0Jet: any = origin.add(dir.mul(radius.mul(0.15)));
    const v0Jet: any = dir.mul(h.w.mul(0.6).add(0.25)).add(vec3(0, wJet, 0));

    const start: any = mix(teRim, teJet, fromJet);
    const p0: any = mix(p0Rim, p0Jet, fromJet);
    const v0: any = mix(v0Rim, v0Jet, fromJet).toVar('splashV0');
    const age: any = T.sub(start).toVar('splashAge');

    // Same linear drag toward the wind as `Spray`: drops nearly ballistic, mist
    // hanging and drifting.
    const k: any = isDrop.mul(0.3).add(isCluster.mul(0.6)).add(isMist.mul(4)).toVar('splashK');
    const wind: any = sprayUniforms.wind;
    const vInf: any = vec3(wind.x, float(-GRAVITY).div(k), wind.z);
    const decay: any = exp(k.negate().mul(age.max(0)));
    const centre: any = p0.add(vInf.mul(age)).add(v0.sub(vInf).mul(decay.oneMinus().div(k))).toVar('splashCentre');
    const velocity: any = vInf.add(v0.sub(vInf).mul(decay)).toVar('splashVelocity');

    // Until it is back in the water, roughly; mist for a second or two.
    const lifeSpan: any = isMist
      .mul(h.w.mul(0.8).add(1.2))
      .add(isMist.oneMinus().mul(v0.y.max(0.3).mul(2.2 / GRAVITY).add(0.25)))
      .toVar('splashLife');
    const life: any = age.div(lifeSpan);
    const enabled: any = mix(float(1), jetSpeed.greaterThan(0).select(float(1), float(0)), fromJet)
      .mul(mix(float(1), mistAmount.greaterThan(0.02).select(float(1), float(0)), isMist));
    const alive: any = life
      .greaterThanEqual(0)
      .and(life.lessThanEqual(1))
      .and(T.lessThanEqual(D.x.add(0.5)))
      .select(enabled, float(0))
      .toVar('splashAlive');

    const scale: any = size.mul(0.8).add(0.6);
    const width: any = isDrop
      .mul(h.y.mul(0.025).add(0.012))
      .mul(scale)
      .add(isCluster.mul(h.z.mul(0.1).add(0.06)).mul(radius.add(0.6)))
      .add(isMist.mul(h.y.mul(0.8).add(0.5)).mul(size.add(0.5)).mul(life.mul(1.4).add(0.6)));
    const density: any = isDrop.mul(0.9).add(isCluster.mul(0.8)).add(isMist.mul(mistAmount).mul(0.16));
    const fadeIn: any = smoothstep(0, 0.05, life);
    const fadeOut: any = smoothstepDown(life, isMist.mul(-0.4).add(0.8), 1);
    // Back into the sea a little below where it left it.
    const submerged: any = smoothstepDown(origin.y.sub(centre.y), -0.02, 0.15);
    const opacity: any = density.mul(fadeIn).mul(fadeOut).mul(submerged);

    return particleVertex({
      centre,
      velocity,
      life,
      alive,
      isDrop,
      isSheet: float(0),
      isMist,
      isCluster,
      cls,
      seed: h.z,
      width,
      streak: isDrop.mul(0.05).add(isCluster.mul(0.03)).add(isMist.mul(0.04)),
      opacity,
      soft: isDrop.mul(0.05).add(isCluster.mul(0.15)).add(isMist.mul(0.8)),
      range: 240,
    });
  }
}

/**
 * The look of the crown and the jet: one sheet of thrown water, torn by the
 * spray atlas's sheet and ligament fields as it rises and ages — the rim first,
 * into fingers — and lit by `sheetRadiance`, like the bow sheet.
 */
function sheetFragment(options: { rimFingers: boolean }): any {
  const tex = sprayAtlas();
  const theta: any = vCrown.x;
  const tau: any = vCrown.y.clamp(0, 1).toVar('splashFTau');
  const age: any = vCrown.z.clamp(0, 1).toVar('splashFAge');
  const height: any = vCrown.w;
  const fingers: any = vCrownMisc.y;
  const seed: any = vCrownMisc.w;

  // Around the crown in finger-widths, up it in a few tiles drawn out along
  // the flow, and advected up as the water climbs.
  const around: any = theta.mul(fingers).mul(1 / (Math.PI * 2)).mul(0.25);
  const flow: any = vec2(around, tau.mul(1.3).sub(age.mul(0.9))).add(vec2(seed.mul(3.7), seed.mul(1.9)));
  const t1: any = texture(tex, flow.mul(vec2(1, 0.6))).toVar('splashT1');
  const t2: any = texture(tex, flow.mul(vec2(2.3, 1.4)).add(vec2(0.37, 0.71))).toVar('splashT2');
  const toFingers: any = options.rimFingers ? smoothstep(0.45, 0.9, tau) : smoothstep(0.75, 1, tau);
  const field: any = mix(t1.x, t1.z, toFingers)
    .mul(0.65)
    .add(t2.y.mul(0.35))
    .sub(0.5)
    .mul(1.6)
    .add(0.5)
    .toVar('splashField');
  // Torn more as it ages, and from the rim down. The jet is a coherent column
  // that only breaks up at its tip, so it tears far less.
  const tearing: any = options.rimFingers
    ? smoothstep(0.2, 1.05, age.mul(0.8).add(tau.mul(0.45)))
    : smoothstep(0.55, 1.1, age.mul(0.5).add(tau.mul(0.6)));
  const cut: any = mix(float(options.rimFingers ? 0.15 : 0.08), float(0.9), tearing).toVar('splashCut');
  const torn: any = smoothstep(cut.sub(0.05), cut.add(0.04), field);
  // The sheet thins as it stretches: most opaque young and near its base.
  const membrane: any = mix(float(0.9), float(options.rimFingers ? 0.55 : 0.8), age).mul(tau.mul(-0.2).add(1));
  // Only above the water it came out of.
  const above: any = smoothstep(-0.02, 0.06, height);
  const ending: any = smoothstepDown(age, 0.7, 1);

  const sceneZ: any = perspectiveDepthToViewZ(sharedSceneDepth().x, cameraNear, cameraFar);
  const soft: any = vCrownMisc.x.sub(sceneZ).div(0.15).clamp(0, 1);
  const nearFade: any = smoothstep(0.3, 1.5, vCrownMisc.x.negate());
  const toEye: any = cameraPosition.sub(positionWorld).normalize();
  const facetNormal: any = dFdx(positionWorld).cross(dFdy(positionWorld)).normalize();
  const facing: any = smoothstep(0.05, 0.4, facetNormal.dot(toEye).abs());

  // Clearer low on a young crown, white where it is torn, high, or old.
  const aerated: any = smoothstep(0.25, 0.9, tau)
    .max(age.mul(0.9))
    .max(options.rimFingers ? 0.3 : 0.55);
  const radiance: any = sheetRadiance(aerated, field.mul(0.4).add(0.75));
  const alpha: any = torn.mul(membrane).mul(above).mul(ending).mul(soft).mul(nearFade).mul(facing).clamp(0, 1);
  return vec4(radiance, alpha);
}

function sheetMaterial(vertex: () => any, fragment: () => any): THREE.MeshBasicNodeMaterial {
  const material = new THREE.MeshBasicNodeMaterial();
  material.side = THREE.DoubleSide;
  material.transparent = true;
  material.depthWrite = false;
  material.fog = false;
  material.alphaTest = SPRAY_ALPHA_FLOOR;
  material.positionNode = Fn(() => vertex())() as any;
  material.colorNode = Fn(() => fragment())() as any;
  return material;
}

/** A unit grid (u around, v along) instanced once per slot. */
function gridGeometry(around: number, along: number): THREE.InstancedBufferGeometry {
  const positions: number[] = [];
  const index: number[] = [];
  for (let j = 0; j <= along; j++) {
    for (let i = 0; i <= around; i++) positions.push(i / around, j / along, 0);
  }
  const row = around + 1;
  for (let j = 0; j < along; j++) {
    for (let i = 0; i < around; i++) {
      const a = j * row + i;
      index.push(a, a + 1, a + row, a + 1, a + row + 1, a + row);
    }
  }
  const geometry = new THREE.InstancedBufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setIndex(index);
  const slots = new Float32Array(SLOTS);
  for (let s = 0; s < SLOTS; s++) slots[s] = s;
  geometry.setAttribute('splashSlot', new THREE.InstancedBufferAttribute(slots, 1));
  geometry.instanceCount = SLOTS;
  geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), Number.POSITIVE_INFINITY);
  return geometry;
}

function hash(i: number, salt: number): number {
  let h = (Math.imul(i + 1, 0x9e3779b1) ^ Math.imul(salt + 7, 0x85ebca77)) >>> 0;
  h ^= h >>> 15;
  h = Math.imul(h, 0x2c1b3c6d) >>> 0;
  h ^= h >>> 12;
  h = Math.imul(h, 0x297a2d39) >>> 0;
  h ^= h >>> 15;
  return (h >>> 0) / 4294967296;
}

function clamp01(x: number): number {
  return Math.min(1, Math.max(0, x));
}

function smooth01(x: number): number {
  const t = clamp01(x);
  return t * t * (3 - 2 * t);
}
