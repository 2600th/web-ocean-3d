import { test, expect } from '@playwright/test';
import { CREW_GROUPS, CREW_ORDER, PART } from '../src/scene/crew/stations';
import { QUALITY_TIERS } from '../src/core/QualityManager';

/**
 * The crew's roster, checked in Node: who a tier puts aboard, and that every
 * look is something the mesh can draw. The pictures are in
 * `.claude/progress/18-crew-toy/`; these guard the bookkeeping a picture would
 * only show as a missing or a doubled figure.
 */

function aboard(n: number): string[] {
  const taken = CREW_GROUPS.map(() => 0);
  const roles: string[] = [];
  for (const g of CREW_ORDER.slice(0, n)) {
    roles.push(CREW_GROUPS[g].stations[taken[g]].look.role);
    taken[g]++;
  }
  return roles;
}

test.describe('crew roster', () => {
  test('the order never asks a group for more figures than it has', () => {
    const asked = CREW_GROUPS.map(() => 0);
    for (const g of CREW_ORDER) asked[g]++;
    CREW_GROUPS.forEach((group, i) => expect(asked[i], group.id).toBe(group.stations.length));
  });

  test('Low has no crew, Medium five, High and up nine', () => {
    expect(QUALITY_TIERS.low.crew).toBe(0);
    expect(QUALITY_TIERS.medium.crew).toBe(5);
    for (const tier of ['high', 'ultra'] as const) expect(QUALITY_TIERS[tier].crew).toBeGreaterThanOrEqual(9);
    expect(CREW_ORDER.length).toBe(9);
  });

  test("Medium's five are five different roles, captain and helmsman among them", () => {
    const roles = aboard(5);
    expect(roles).toContain('captain');
    expect(roles).toContain('helmsman');
    expect(roles).toContain('bosun');
    expect(roles).toContain('lookout');
    expect(new Set(roles).size).toBe(5);
  });

  test('every look wears a head, a body, arms and legs, and only parts the mesh has', () => {
    const heads: number[] = [PART.headBeard, PART.headCrop, PART.headMop, PART.headLong];
    const bodies: number[] = [PART.bodyBurly, PART.bodyCoat, PART.bodyTunic];
    const arms: number[] = [PART.armsBurly, PART.armsCoat, PART.armsTunic];
    const legs: number[] = [PART.legsBurly, PART.legsBoots, PART.legsTunic];
    const max = Math.max(...Object.values(PART));
    // Two 12-bit halves in the shader (`Crew.makeMaterial`).
    expect(max).toBeLessThan(24);
    for (const group of CREW_GROUPS) {
      for (const st of group.stations) {
        const p = st.look.parts;
        expect(p.filter((x) => heads.includes(x)).length, st.look.role).toBe(1);
        expect(p.filter((x) => bodies.includes(x)).length, st.look.role).toBe(1);
        expect(p.filter((x) => arms.includes(x)).length, st.look.role).toBe(1);
        expect(p.filter((x) => legs.includes(x)).length, st.look.role).toBe(1);
        for (const x of p) expect(x).toBeLessThanOrEqual(max);
      }
    }
  });

  test('the helmsman is placed from the whipstaff', () => {
    const helm = CREW_GROUPS.flatMap((g) => g.stations).filter((s) => s.atWheel);
    expect(helm.map((s) => s.look.role)).toEqual(['helmsman']);
  });
});
