import * as THREE from 'three/webgpu';
import {
  Fn,
  cameraPosition,
  exp,
  float,
  max,
  mix,
  output,
  positionWorld,
  smoothstep,
  vec3,
  vec4,
} from 'three/tsl';
import { uniform } from '../core/uniform';

/* eslint-disable @typescript-eslint/no-explicit-any */
type Node = any;

/**
 * One aerial perspective for the whole frame.
 *
 * Before this the scene had three unrelated treatments of distance haze and one
 * conspicuous absence. The ocean mixed toward an authored `fogColor` on a scalar
 * exponential; `VolumetricFog` is a *sea* fog, marched, and switched off in the
 * clear presets; and the terrain, the props, the ship and the canopy had
 * **nothing at all**. That last one is why the island had full foreground
 * contrast at 1.4 km and read as a cut-out laid over a hazy sea, and why the sea
 * and the land met at the shoreline as two different atmospheres.
 *
 * ### What it computes
 *
 * Extinction and inscatter as separate terms, which is the physically correct
 * split and the one `Xl2XRW` makes explicitly:
 *
 *     out = object * T + inscatter * (1 - T),   T = exp(-tau)
 *
 * `tau` is **per channel**. That is the whole point and it is what a scalar
 * density cannot do: air scatters blue about five times as strongly as red, so
 * distance moves a colour *toward the sky* rather than toward a grey constant.
 * The ratio comes from `Atmosphere`'s own Rayleigh and Mie coefficients rather
 * than from a hand-picked triple, so the haze cannot drift away from the dome
 * that produced it — which is what a horizon step is.
 *
 * The *magnitude* does not come from those coefficients, and that distinction is
 * worth stating plainly. Preetham's betaR is a per-metre coefficient against an
 * 8.4 km zenith column and a fitted appearance model on top; taken literally over
 * a 1.5 km horizontal path it gives about one percent extinction, which is not
 * what anyone means by aerial perspective. So the coefficients set the *hue* and
 * the preset's own `fog.density` sets the *strength* — a number already
 * calibrated per preset against the water, and now doing the same job everywhere.
 *
 * The vertical profile is the standard exponential atmosphere, integrated in
 * closed form along the ray:
 *
 *     ∫₀ᵈ exp(-(y₀ + t·rd.y)/H) dt = d · exp(-y₀/H) · (1 - exp(-k)) / k,  k = d·rd.y/H
 *
 * which is the same shape `MdGfzh` and `Nt3XDM` both use. It is singular at
 * `k = 0` — a horizontal ray — so near zero it falls back to the series, which
 * is exact to the term that matters and has no division at all.
 *
 * ### How it reaches the frame
 *
 * As `scene.fogNode`. Three calls it after lighting and after the material's own
 * output is resolved, for every material that has not opted out — so terrain,
 * props, ship, canopy, hull and water all receive the identical function with no
 * per-material plumbing. Materials that must not receive it (the sky dome, the
 * cloud layer, the underwater dressing) already set `fog = false` for their own
 * reasons.
 *
 * Underwater is gated rather than opted out, because the seafloor is one mesh
 * that crosses the waterline: air haze applies where both the camera and the
 * shaded point are above the surface, and the water's own transmission owns
 * everything else.
 */

/**
 * Scale height of the haze layer, metres.
 *
 * Not the 8.4 km of the Rayleigh column. What actually hazes a view across a
 * tropical sea is the marine boundary layer — sea salt aerosol and water vapour
 * — which is a kilometre deep at most and is why a summit stands clear of the
 * murk its own beach sits in.
 */
const HAZE_SCALE_HEIGHT = 900;

/** Below this `k` the closed-form integral is replaced by its series. */
const SERIES_EDGE = 0.06;

/**
 * How far the inscatter colour is allowed to leave the preset's authored one.
 *
 * The presets carry a `fog.color` each, chosen against a look, and throwing it
 * away would be a regression dressed as a physics improvement. It anchors the
 * hue; the sky supplies the *gradient* — cooler overhead, paler at the horizon —
 * which is the part an authored constant cannot have and the part the eye reads
 * as depth.
 */
const SKY_INSCATTER_MIX = 0.62;

/**
 * Exponent on the zenith blend.
 *
 * Below 1 so the horizon colour holds through the first few degrees of
 * elevation, which is where nearly all of this scene's long rays live.
 */
const ZENITH_BLEND_POWER = 0.55;

/** Angular tightness of the forward Mie glow around the sun's bearing. */
const SUN_GLOW_POWER = 9;

/**
 * Scale height of the sea-surface haze layer, metres: the spray and sea-salt
 * aerosol that sits in the first tens of metres over open water. A level ray
 * from a deck runs its whole length inside it, so it thickens the haze over
 * the far sea and the island's beach without touching anything high in the
 * frame. (W6, 2026-11-18.)
 */
const SURFACE_SCALE_HEIGHT = 45;

/**
 * Elevation of the dome samples that make the horizon ring (`setHorizon`):
 * the first half degree over the horizon, which is the sky the far sea meets.
 */
const RING_ELEVATION = 0.008;

/**
 * The ring's weight in clear air. A little under 1: the half degree it is
 * sampled at sits under the clouds' far band, which the frame shows above the
 * far sea, and at full weight the haze came out 10-15 sRGB brighter than it.
 */
const RING_GAIN = 0.92;

/** Bearings the ring is sampled at; its Fourier fit is taken to third order. */
const RING_SAMPLES = 16;

/**
 * View elevation over which the haze leaves the ring for the authored blend.
 * Every downward ray is the ring's: a long near-level path's inscatter *is*
 * the horizon sky in that bearing, and a short steep one has no haze to colour.
 */
const RING_LIFT = 0.18;

export class AerialPerspective {
  /** Assign to `scene.fogNode`. Built once; every input is a uniform. */
  readonly node: Node;

  private readonly uBeta: any = uniform(new THREE.Vector3(0.35, 0.82, 1.83));
  private readonly uDensity = uniform(0.00009);
  private readonly uFogColor: any = uniform(new THREE.Color(0x7ea8cc));
  private readonly uHorizon: any = uniform(new THREE.Color(0.7, 0.78, 0.9));
  private readonly uZenith: any = uniform(new THREE.Color(0.32, 0.48, 0.82));
  private readonly uSunDir = uniform(new THREE.Vector3(0, 1, 0));
  private readonly uSunColor: any = uniform(new THREE.Color(1, 1, 1));
  private readonly uSunGlow = uniform(0.5);
  /**
   * 1 when the real camera is in air, 0 when it is under water.
   *
   * Written from the scene camera rather than read from `cameraPosition`, so the
   * planar reflection pass — which renders from a camera mirrored below the sea
   * — does not lose its haze. See `airFraction`.
   */
  private readonly uEyeInAir = uniform(1);
  /**
   * The dome's radiance along the horizon as a Fourier series in bearing:
   * constant, then cos and sin of one, two and three times the bearing.
   */
  private readonly uRing: any[] = Array.from({ length: 7 }, () => uniform(new THREE.Color(0, 0, 0)));
  /** 0 until `setHorizon` has run, so an unwired caller keeps the old haze. */
  private readonly uRingWeight = uniform(0);
  /** Density of the sea-surface layer at the sea, per metre. */
  private readonly uSurface = uniform(0);

  private readonly beta = new THREE.Vector3();
  private readonly ringDir = new THREE.Vector3();
  private readonly ringSample = new THREE.Color();

  constructor() {
    this.node = Fn(() => {
      const wp = positionWorld.toVar('apWorld');
      const v = wp.sub(cameraPosition).toVar('apRay');
      const dist = v.length().toVar('apDist');
      const rd = v.div(max(dist, 1e-4)).toVar('apDir');

      const air = this.airFraction(wp).toVar('apAir');
      const tau = this.opticalDepth(dist, rd)
        .add(this.surfaceDepth(dist, rd))
        .mul(air)
        .toVar('apTau');
      const transmittance = exp(tau.negate()).toVar('apT');

      const haze = this.inscatter(rd).toVar('apInscatter');
      return vec4(mix(haze, output.rgb, transmittance), output.a);
    })();
  }

  /**
   * Applies the same haze to a colour the fog hook cannot reach.
   *
   * The cloud layer draws on a camera-locked dome with `fog = false` and its own
   * blending, so it never passes through `setupFog` — but a deck that converges
   * to the horizon and does *not* dissolve into the same haze the sea and the
   * island dissolve into is exactly the hard-edged ceiling the gap analysis
   * describes. Same function, called by hand.
   */
  /**
   * Fraction of what lies at `distance` along `rayDir` that survives the haze.
   *
   * For a caller drawn *over the sky* rather than in front of the scene: the
   * cloud layer. Such a caller should fade into the dome behind it, which is
   * already the right radiance in every direction, rather than toward the
   * haze's own `inscatter` — that is built from the CPU-side horizon and
   * zenith *tints*, which are hues at full scale and at night are hundreds of
   * times brighter than the dome itself.
   */
  transmittance(distance: Node, rayDir: Node): Node {
    const depth = this.opticalDepth(distance, rayDir) as any;
    return exp(depth.dot(vec3(0.2126, 0.7152, 0.0722)).negate());
  }

  apply(color: Node, distance: Node, rayDir: Node): Node {
    const depth = this.opticalDepth(distance, rayDir).add(this.surfaceDepth(distance, rayDir));
    const transmittance = exp(depth.negate());
    return mix(this.inscatter(rayDir), color, transmittance);
  }

  /**
   * Publishes the state the haze has to agree with.
   *
   * Called every frame from the same place the dome and the water are updated,
   * so all three read one atmosphere. Writes uniforms only — no rebuild.
   */
  setSky(
    betaR: THREE.Vector3,
    betaM: THREE.Vector3,
    fogColor: THREE.Color,
    density: number,
    horizon: THREE.Color,
    zenith: THREE.Color,
    sunDir: THREE.Vector3,
    sunColor: THREE.Color,
  ): void {
    // Normalised so the **luminance-weighted** mean is 1, not the arithmetic
    // mean, and the difference is measurable rather than pedantic.
    //
    // The point of normalising at all is that the per-channel split should
    // redistribute the preset's density between the channels without changing
    // how much haze a viewer sees — a preset tuned for a look keeps that look
    // and gains a hue gradient it did not have. But "how much haze a viewer
    // sees" is a *luminance* quantity, and luminance is 72% green. Dividing by
    // the arithmetic mean put green at 0.82, so the far field was extinguished
    // 18% less than by the scalar fog this replaced — and `gallery-jitter`
    // caught it: high-frequency energy in the band under the horizon rose 10.7%
    // with the band's mean luminance essentially unchanged, which is the
    // signature of *less crushing* rather than of more noise. The distant sea
    // was showing more of its own sampling error because less haze was sitting
    // on top of it.
    //
    // Weighting by Rec. 709 puts green back at ~1.03 — the rate the presets were
    // calibrated against — while red at 0.44 and blue at 2.30 keep the hue shift
    // that is the whole reason for doing this per channel.
    this.beta.copy(betaR).add(betaM);
    const weighted = 0.2126 * this.beta.x + 0.7152 * this.beta.y + 0.0722 * this.beta.z;
    if (weighted > 0) this.beta.multiplyScalar(1 / weighted);
    else this.beta.set(1, 1, 1);
    this.uBeta.value.copy(this.beta);

    this.uDensity.value = density;
    this.uFogColor.value.copy(fogColor);
    this.uHorizon.value.copy(horizon);
    this.uZenith.value.copy(zenith);
    this.uSunDir.value.copy(sunDir);
    this.uSunColor.value.copy(sunColor);
  }

  /**
   * Samples the dome along the horizon, so the haze over the far sea is the
   * sky's own colour in every bearing.
   *
   * This is the horizon line (W6). The inscatter was built from the CPU-side
   * horizon and zenith *tints* and the preset's fog colour, which are hues at a
   * fill scale, not the dome's radiance: with the water drawn black the haze at
   * the horizon came out at sRGB 115 under a dome of 204 to 224 in
   * `clear-day-wide`, and 86 under 148 in `storm`, so nothing drawn beneath it
   * could meet the sky. `sky` is the `Atmosphere`, whose `skyRadiance` mirrors the
   * dome's shader (overcast, twilight, night and the highlight knee included);
   * sixteen bearings are fitted to third order, enough for the twilight glow
   * round the sun's bearing. Sixteen CPU evaluations a frame; uniform writes.
   */
  setHorizon(sky: {
    skyRadiance(dir: THREE.Vector3, out: THREE.Color): THREE.Color;
    readonly overcast: number;
  }): void {
    const ring = this.uRing.map((u) => (u.value as THREE.Color).setRGB(0, 0, 0));
    const h = Math.sqrt(1 - RING_ELEVATION * RING_ELEVATION);
    for (let i = 0; i < RING_SAMPLES; i++) {
      const a = (i / RING_SAMPLES) * Math.PI * 2;
      const c = sky.skyRadiance(this.ringDir.set(Math.cos(a) * h, RING_ELEVATION, Math.sin(a) * h), this.ringSample);
      ring[0].r += c.r / RING_SAMPLES;
      ring[0].g += c.g / RING_SAMPLES;
      ring[0].b += c.b / RING_SAMPLES;
      for (let m = 1; m <= 3; m++) {
        const wc = (2 / RING_SAMPLES) * Math.cos(m * a);
        const ws = (2 / RING_SAMPLES) * Math.sin(m * a);
        ring[2 * m - 1].r += c.r * wc;
        ring[2 * m - 1].g += c.g * wc;
        ring[2 * m - 1].b += c.b * wc;
        ring[2 * m].r += c.r * ws;
        ring[2 * m].g += c.g * ws;
        ring[2 * m].b += c.b * ws;
      }
    }
    // Under a deck the sky over the far sea is the cloud base, not the dome
    // the ring samples (the dome there is brighter than the deck drawn over
    // it, and the ring turned the storm's island a pale beige against dark
    // cloud). The authored, overcast-pulled haze already describes the deck.
    this.uRingWeight.value = RING_GAIN * (1 - 0.85 * Math.min(1, Math.max(0, sky.overcast)));
  }

  /**
   * Density of the sea-surface spray layer at the sea, per metre. It is the
   * same air in every preset, so it is absolute rather than a multiple of the
   * preset's haze; it rises with the wind as the crests tear, and in a gale
   * it is the spray haze the far sea dissolves into.
   */
  setSurfaceHaze(perMetre: number): void {
    this.uSurface.value = Math.max(0, perMetre);
  }

  /** How strong the forward glow around the sun's bearing is, 0..1. */
  setSunGlow(value: number): void {
    this.uSunGlow.value = Math.max(0, value);
  }

  /**
   * Where the *real* eye is relative to the surface, in metres. Call each frame
   * with the scene camera's height; the reflection pass must not supply its own.
   */
  setEyeHeight(y: number): void {
    // The same band `airFraction` used to apply to `cameraPosition.y`.
    const t = Math.min(1, Math.max(0, (y + 1.5) / 2.1));
    this.uEyeInAir.value = t * t * (3 - 2 * t);
  }

  // -------------------------------------------------------------------------

  /**
   * 1 in air, 0 under water, with a band across the surface.
   *
   * Both ends are tested. A submerged camera looking at the island above the
   * waterline must not paint air haze over water, and a camera in the air
   * looking into the lagoon must not paint it over the seabed — the water's own
   * extinction already owns that path and would be charged for twice.
   */
  private airFraction(worldPosition: Node): Node {
    // The eye's side of the surface is a **uniform**, not `cameraPosition.y`,
    // and that is not a micro-optimisation — reading the node is wrong.
    //
    // The planar reflector renders the scene from a virtual camera mirrored
    // below the mean sea plane, so `cameraPosition.y` is negative for the whole
    // reflection pass. Gating on it closed the air fraction there and the
    // island's *reflection* came out with none of the haze the island itself
    // has, which is a visible inconsistency in exactly the frames the haze was
    // added for. A mirrored camera is a rendering construction, not a submerged
    // eye.
    //
    // The uniform is written from the real camera once a frame, so both passes
    // agree. It straddles zero rather than sitting above it because the
    // waterline shot puts the eye eight centimetres *under* the surface with
    // half the frame still in air.
    return smoothstep(-1.5, 1.5, worldPosition.y).mul(this.uEyeInAir);
  }

  private opticalDepth(distance: Node, rayDir: Node): Node {
    return this.uBeta.mul(this.column(distance, rayDir, HAZE_SCALE_HEIGHT).mul(this.uDensity));
  }

  /**
   * The sea-surface layer's optical depth. Spray and sea salt scatter nearly
   * neutrally, so it is grey rather than per channel.
   */
  private surfaceDepth(distance: Node, rayDir: Node): Node {
    const t = this.column(distance, rayDir, SURFACE_SCALE_HEIGHT).mul(this.uSurface);
    return vec3(t, t, t);
  }

  /** Air mass along the ray through an exponential layer of scale height `H`. */
  private column(distance: Node, rayDir: Node, H: number): Node {
    const y0 = max(cameraPosition.y, 0);
    // Clamped for the same reason the cloud march clamps its span: nothing in
    // this world puts the camera a scale height above what it is looking at, and
    // an unbounded `exp(-k)` on a ray that did would return a column of hundreds.
    const k = rayDir.y.mul(distance).div(H).clamp(-4, 4);

    // The integral's shape factor, `(1 - exp(-k)) / k`. It is 1 at k = 0 and the
    // division is singular there, so near zero the series takes over.
    //
    // Blended rather than branched, and the denominator is floored *away from
    // zero on the correct side*. Both matter: a `select` still evaluates the
    // untaken side, and `mix(a, NaN, 0)` is NaN rather than `a` — so a divide
    // that produced an infinity at k = 0 would put a hole in the frame even
    // though its weight is zero.
    const sign = k.lessThan(0).select(float(-1), float(1));
    const den = k.abs().max(SERIES_EDGE).mul(sign);
    const series = float(1).sub(k.mul(0.5)).add(k.mul(k).mul(1 / 6));
    const exact = k.negate().exp().oneMinus().div(den);
    const shape = mix(series, exact, k.abs().smoothstep(SERIES_EDGE * 0.5, SERIES_EDGE));

    return distance.mul(exp(y0.div(-H))).mul(shape.max(0));
  }

  private inscatter(rayDir: Node): Node {
    const sky = mix(
      this.uHorizon,
      this.uZenith,
      rayDir.y.clamp(0, 1).pow(ZENITH_BLEND_POWER),
    );
    const base = mix(this.uFogColor, sky, SKY_INSCATTER_MIX).toVar('apBase');
    // Forward Mie: the haze is brightest and warmest looking down the sun's
    // bearing, which is most of what makes a hazy afternoon read as one.
    const glow = rayDir.dot(this.uSunDir).clamp(0, 1).pow(SUN_GLOW_POWER).mul(this.uSunGlow);
    const authored = base.add(vec3(this.uSunColor).mul(glow).mul(base));
    // The dome's own horizon in this bearing (`setHorizon`), for every ray at
    // or below the horizon, fading out over the first ten degrees above it.
    const flat = rayDir.xz.div(max(rayDir.xz.length(), 1e-4));
    const c1 = flat.x;
    const s1 = flat.y;
    const c2 = c1.mul(c1).sub(s1.mul(s1));
    const s2 = c1.mul(s1).mul(2);
    const c3 = c2.mul(c1).sub(s2.mul(s1));
    const s3 = s2.mul(c1).add(c2.mul(s1));
    const ring = vec3(this.uRing[0])
      .add(vec3(this.uRing[1]).mul(c1))
      .add(vec3(this.uRing[2]).mul(s1))
      .add(vec3(this.uRing[3]).mul(c2))
      .add(vec3(this.uRing[4]).mul(s2))
      .add(vec3(this.uRing[5]).mul(c3))
      .add(vec3(this.uRing[6]).mul(s3))
      .max(0);
    const level = float(1).sub(rayDir.y.smoothstep(0, RING_LIFT)).mul(this.uRingWeight);
    return mix(authored, ring, level);
  }
}
