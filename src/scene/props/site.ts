import { ISLAND, seafloorHeight } from '../Seafloor';

/**
 * Where the lighthouse stands: the seaward brow of the headland's southern
 * shoulder, on the bearing where `Seafloor` begins to raise the headland ridge.
 *
 * Chosen against two cameras. From `island-approach` (780 m out on bearing
 * 0.6) this spot is 38 degrees left of the lens axis against a half-field of
 * 43, so it is in frame, on the skyline of the shoulder; from the play area at
 * the origin it is 16 degrees off the bearing to the island's centre. The
 * ground here is ~17 m and falls seaward at about one in four, so the tower
 * reads as standing *on* the headland rather than in the trees behind the
 * beach — and it is 200 m from the fort, far enough that the two read as
 * separate landmarks rather than one cluttered one.
 *
 * In its own module so `Canopy` can keep its trees off the site without
 * importing `Props`.
 */
export const LIGHTHOUSE_BEARING = 1.15;
/** Metres in from the waterline on that bearing. */
export const LIGHTHOUSE_INSET = 62;
/** Radius around the tower kept clear of trees and scatter: tower, cottage and a yard. */
export const LIGHTHOUSE_KEEP_OUT = 17;

let site: { x: number; z: number } | null = null;

/** The tower's centre in world XZ. Bisects for the waterline once, then caches. */
export function lighthouseSite(): { x: number; z: number } {
  if (site) return site;
  const dx = Math.cos(LIGHTHOUSE_BEARING);
  const dz = Math.sin(LIGHTHOUSE_BEARING);
  let inner = ISLAND.radius * 0.4;
  let outer = ISLAND.radius * 1.7;
  for (let i = 0; i < 24; i++) {
    const mid = (inner + outer) / 2;
    if (seafloorHeight(ISLAND.x + dx * mid, ISLAND.z + dz * mid) > 0) inner = mid;
    else outer = mid;
  }
  const r = (inner + outer) / 2 - LIGHTHOUSE_INSET;
  site = { x: ISLAND.x + dx * r, z: ISLAND.z + dz * r };
  return site;
}

/** True when (x, z) is on the lighthouse's ground. */
export function inLighthouseClearing(x: number, z: number, margin = 0): boolean {
  const s = lighthouseSite();
  const r = LIGHTHOUSE_KEEP_OUT + margin;
  return (x - s.x) * (x - s.x) + (z - s.z) * (z - s.z) < r * r;
}
