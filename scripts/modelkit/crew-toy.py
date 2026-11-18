"""
crew-toy.py — builds the ship's crew (`public/models/crew/crew_toy.raw.glb`).

Run headless with Blender 5.2, then `node scripts/modelkit/crew-toy.mjs` to
compress the result into the shipped `public/models/crew/crew_toy.glb`:

    blender -b --factory-startup -P scripts/modelkit/crew-toy.py -- <kaykit-gltf-dir> <out.raw.glb>

`<kaykit-gltf-dir>` is `addons/kaykit_character_pack_adventures/Characters/gltf`
from https://github.com/KayKit-Game-Assets/KayKit-Character-Pack-Adventures-1.0
(Kay Lousberg, CC0 1.0): `Barbarian.glb`, `Knight.glb`, `Mage.glb`, `Rogue.glb`
and their `*_texture.png` palettes.

What it makes: ONE skinned mesh on the pack's shared rig, holding every
interchangeable part (four heads, three bodies, three pairs of arms and legs)
and the pirate gear modelled here (tricorn, knitted Monmouth cap, kerchief,
eye patch, sash). The runtime picks a costume per figure by hiding parts.
Each corner carries a vertex colour: R = tint slot / 255 (0 keep the atlas
colour, 1 shirt, 2 coat, 3 breeches, 4 headwear, 5 sash, 6 skin, 7 hair),
G = part bit / 255 (see PARTS). Tinted faces are moved onto a neutral grey
gradient cell so the runtime colour multiplies a clean ramp.

The four palettes are packed into one 1024 x 512 atlas (cells 64 px), with a
fourth tile of our own cells (neutral ramp, black, gold, leather).
Only the deform bones and six clips are kept.
"""
import bpy, bmesh, sys, os, math
import numpy as np
from mathutils import Vector, Matrix

argv = sys.argv[sys.argv.index('--') + 1:]
SRC, OUT = argv[0], argv[1]

KEEP_CLIPS = ['Idle', '2H_Melee_Idle', 'Walking_A', 'Cheer', 'PickUp', 'Interact']

# Part bits, read by src/scene/crew/Crew.ts (`PART`). Keep in step.
PARTS = {
    'Barbarian_Head': 0, 'Knight_Head': 1, 'Mage_Head': 2, 'Rogue_Head': 3,
    'Barbarian_Body': 4, 'Mage_Body': 5, 'Rogue_Body': 6,
    'Barbarian_ArmLeft': 7, 'Barbarian_ArmRight': 7,
    'Mage_ArmLeft': 8, 'Mage_ArmRight': 8,
    'Rogue_ArmLeft': 9, 'Rogue_ArmRight': 9,
    'Barbarian_LegLeft': 10, 'Barbarian_LegRight': 10,
    'Mage_LegLeft': 11, 'Mage_LegRight': 11,
    'Rogue_LegLeft': 12, 'Rogue_LegRight': 12,
}
P_TRICORN, P_CAP, P_KERCHIEF, P_PATCH, P_SASH, P_BEARD = 13, 14, 15, 16, 17, 18

SHIRT, COAT, BREECHES, HAT, SASH, SKIN, HAIR = 1, 2, 3, 4, 5, 6, 7

# (character, palette column, palette row from the top) -> tint slot.
SLOTS = {
    ('Barbarian', 0, 0): SKIN, ('Barbarian', 1, 0): HAIR,
    ('Barbarian', 0, 1): SHIRT, ('Barbarian', 1, 1): SHIRT, ('Barbarian', 3, 2): BREECHES,
    ('Knight', 0, 0): SKIN, ('Knight', 1, 0): HAIR,
    ('Mage', 0, 0): SKIN, ('Mage', 7, 2): SKIN, ('Mage', 1, 0): HAIR,
    ('Mage', 0, 1): COAT, ('Mage', 1, 1): COAT,
    ('Rogue', 0, 0): SKIN, ('Rogue', 1, 0): HAIR,
    ('Rogue', 0, 1): SHIRT, ('Rogue', 1, 1): SASH, ('Rogue', 3, 2): BREECHES,
}
# Per part overrides of the above.
PART_SLOTS = {
    ('Mage_Body', 7, 1): SHIRT,       # the waistcoat under the coat
    ('Mage_LegLeft', 7, 1): BREECHES, ('Mage_LegRight', 7, 1): BREECHES,
}

# Atlas tiles (column, row from the top) in a 2 x 2 grid of 512 x 256.
TILE = {'Rogue': (0, 0), 'Barbarian': (1, 0), 'Mage': (0, 1), 'Knight': (0, 0), 'Own': (1, 1)}
# Our own cells in the 'Own' tile, (column, row from the top).
C_NEUTRAL, C_BLACK, C_GOLD, C_LEATHER = (0, 0), (1, 0), (2, 0), (3, 0)


def tile_uv(tile, u, v):
    c, r = TILE[tile]
    return (c * 0.5 + u * 0.5, (1 - r) * 0.5 + v * 0.5)


def cell_uv(cell, fu, fv):
    """A point in one of our cells; fu, fv in 0..1 (fv = 1 at the top of the cell)."""
    col, row = cell
    return tile_uv('Own', (col + fu) / 8, (3 - row + fv) / 4)


# ---------------------------------------------------------------- import
bpy.ops.wm.read_factory_settings(use_empty=True)
chars = ['Rogue', 'Barbarian', 'Knight', 'Mage']
owner = {}
rigs = []
for n in chars:
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=os.path.join(SRC, n + '.glb'))
    for o in bpy.data.objects:
        if o not in before:
            owner[o.name] = n
            if o.type == 'ARMATURE':
                rigs.append(o)
rig = rigs[0]  # Rogue's; the four rigs are the same

# ---------------------------------------------------------------- atlas
def load_palette(name):
    img = bpy.data.images.load(os.path.join(SRC, name.lower() + '_texture.png'))
    w, h = img.size
    a = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)  # bottom-up
    # 1024 x 1024 -> 512 x 256: average 4 rows x 2 columns
    return a.reshape(256, 4, 512, 2, 4).mean(axis=(1, 3))

atlas = np.zeros((512, 1024, 4), dtype=np.float32)
atlas[..., 3] = 1
for n in ['Rogue', 'Barbarian', 'Mage']:
    c, r = TILE[n]
    atlas[(1 - r) * 256:(2 - r) * 256, c * 512:(c + 1) * 512] = load_palette(n)


def paint_cell(cell, fn):
    col, row = cell
    c, r = TILE['Own']
    y0 = (1 - r) * 256 + (3 - row) * 64
    x0 = c * 512 + col * 64
    for y in range(64):
        t = 1 - (y + 0.5) / 64  # 0 at the top of the cell, 1 at the bottom
        for x in range(64):
            s = (x + 0.5) / 64
            atlas[y0 + y, x0 + x, :3] = fn(t, s)


def ramp(top, bottom):
    top, bottom = np.array(top), np.array(bottom)
    return lambda t, s: (top + (bottom - top) * t) * (1.06 if 0.84 < s < 0.94 else 1.0)

paint_cell(C_NEUTRAL, ramp((1.0, 1.0, 1.0), (0.62, 0.62, 0.64)))
paint_cell(C_BLACK, ramp((0.10, 0.10, 0.11), (0.02, 0.02, 0.025)))
paint_cell(C_GOLD, ramp((1.0, 0.86, 0.42), (0.72, 0.45, 0.12)))
paint_cell(C_LEATHER, ramp((0.45, 0.29, 0.18), (0.22, 0.13, 0.08)))
# Fill the rest of our tile with the neutral ramp, so a bleeding sample stays grey.
atlas = np.clip(atlas, 0, 1)

aimg = bpy.data.images.new('crew_toy_atlas', 1024, 512, alpha=False)
aimg.pixels[:] = atlas.ravel()
aimg.file_format = 'PNG'
atlas_path = os.path.splitext(OUT)[0] + '_atlas.png'
aimg.filepath_raw = atlas_path
aimg.save()

mat = bpy.data.materials.new('crew_toy')
mat.use_nodes = True
nt = mat.node_tree
bsdf = nt.nodes['Principled BSDF']
tex = nt.nodes.new('ShaderNodeTexImage')
tex.image = aimg
nt.links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
bsdf.inputs['Roughness'].default_value = 0.4

# ---------------------------------------------------------------- parts
def set_corner_colour(me, values):
    """values: list per loop of (slot, part)."""
    attr = me.color_attributes.new('Col', 'FLOAT_COLOR', 'CORNER')
    buf = np.zeros(len(me.loops) * 4, dtype=np.float32)
    for i, (slot, part) in enumerate(values):
        buf[i * 4:i * 4 + 4] = (slot / 255, part / 255, 0, 1)
    attr.data.foreach_set('color', buf)
    me.color_attributes.active_color = attr
    me.color_attributes.render_color_index = me.color_attributes.find('Col')


def measure_eye():
    mh = bpy.data.objects['Mage_Head'].data
    uvl = mh.uv_layers.active.data
    pts = []
    for p in mh.polygons:
        u = sum(uvl[i].uv[0] for i in p.loop_indices) / p.loop_total
        v = sum(uvl[i].uv[1] for i in p.loop_indices) / p.loop_total
        if int(u * 8) == 2 and 3 - int(v * 4) == 0 and p.center.x > 0.03 and p.center.y < -0.3:
            pts.append(p.center.copy())
    zs = sorted(c.z for c in pts)
    zmid = zs[len(zs) // 2]
    low = [c for c in pts if c.z <= zmid]
    return sum(low, Vector()) / len(low)

EYE_MEASURED = measure_eye()

# Every hat must fit every head: hair above the brow is brought inside one
# envelope (an ellipsoid a little smaller than the hats' inside), so a mop or
# a quiff never pushes through a cap. Beards and hair below it are untouched.
HAIR_C = Vector((0.0, -0.02, 1.7))
HAIR_R = Vector((0.55, 0.56, 0.52))


def squash_hair(me, verts):
    for i in verts:
        v = me.vertices[i]
        # from 1.8 up (lower and the scalp would show through), blended in
        # over 0.1 m so the hairline does not crease
        floor = 1.8
        if v.co.z < floor:
            continue
        d = v.co - HAIR_C
        e = math.sqrt((d.x / HAIR_R.x) ** 2 + (d.y / HAIR_R.y) ** 2 + (d.z / HAIR_R.z) ** 2)
        if e > 0.97:
            k = min(1.0, (v.co.z - floor) / 0.1)
            target = HAIR_C + d * (0.97 / e)
            v.co = v.co.lerp(target, k)


kept = []
for o in list(bpy.data.objects):
    if o.type != 'MESH':
        continue
    if o.name not in PARTS:
        continue
    ch = owner[o.name]
    me = o.data
    uvl = me.uv_layers.active.data
    values = [None] * len(me.loops)
    hair = set()
    for p in me.polygons:
        u = sum(uvl[i].uv[0] for i in p.loop_indices) / p.loop_total
        v = sum(uvl[i].uv[1] for i in p.loop_indices) / p.loop_total
        col, rowb = min(int(u * 8), 7), min(int(v * 4), 3)
        row = 3 - rowb
        slot = PART_SLOTS.get((o.name, col, row), SLOTS.get((ch, col, row), 0))
        if slot == HAIR and PARTS[o.name] <= 3:
            hair.update(p.vertices)
        for li in p.loop_indices:
            uu, vv = uvl[li].uv
            if slot:
                fu = min(max(uu * 8 - col, 0.02), 0.98)
                fv = min(max(vv * 4 - rowb, 0.02), 0.98)
                uvl[li].uv = cell_uv(C_NEUTRAL, fu, fv)
            else:
                uvl[li].uv = tile_uv(ch, uu, vv)
            values[li] = (slot, PARTS[o.name])
    set_corner_colour(me, values)
    squash_hair(me, hair)
    me.materials.clear()
    me.materials.append(mat)
    for m in o.modifiers:
        if m.type == 'ARMATURE':
            m.object = rig
    mw = o.matrix_world.copy()
    o.parent = rig
    o.parent_bone = ''
    o.parent_type = 'OBJECT'
    o.matrix_world = mw
    kept.append(o)

# Where the left eye is: the Mage head's black faces on its left (+x), the
# lower cluster (the brows are black too). Measured before the UVs move.
EYE_L = EYE_MEASURED
print('EYE_L', tuple(round(x, 3) for x in EYE_L))
HEAD_C = Vector((0.0, -0.02, 1.68))


# ---------------------------------------------------------------- gear
def lathe(bm, profile, n=40, shape=None, cell=C_NEUTRAL, cells=None, close_top=False, close_bottom=False):
    """Revolve `profile` [(r, z), ...] about Z. shape(theta, i, r, z) -> (r, z).
    UV: cell per profile segment (cells) or one cell, ramped by height."""
    rings = []
    zs = [z for _, z in profile]
    zmin, zmax = min(zs), max(zs)
    for k in range(n):
        th = 2 * math.pi * k / n
        ring = []
        for i, (r, z) in enumerate(profile):
            rr, zz = shape(th, i, r, z) if shape else (r, z)
            ring.append(bm.verts.new((HEAD_C.x + rr * math.cos(th), HEAD_C.y + rr * math.sin(th), zz)))
        rings.append(ring)
    faces = []
    for k in range(n):
        a, b = rings[k], rings[(k + 1) % n]
        for i in range(len(profile) - 1):
            f = bm.faces.new((a[i], b[i], b[i + 1], a[i + 1]))
            c = cells[i] if cells else cell
            faces.append((f, c))
    if close_top:
        top = [rings[k][-1] for k in range(n)]
        f = bm.faces.new(top)
        faces.append((f, cells[-1] if cells else cell))
    if close_bottom:
        bot = [rings[k][0] for k in reversed(range(n))]
        f = bm.faces.new(bot)
        faces.append((f, cells[0] if cells else cell))
    return faces, (zmin, zmax)


def finish(name, bm, faces, slot_of_cell, part, bone, zr, smooth=True):
    me = bpy.data.meshes.new(name)
    uvl = bm.loops.layers.uv.new('UVMap')
    zmin, zmax = zr
    loopvals = {}
    for f, cell in faces:
        f.smooth = smooth
        for l in f.loops:
            t = (l.vert.co.z - zmin) / max(1e-6, zmax - zmin)
            l[uvl].uv = cell_uv(cell, 0.4, 0.15 + 0.7 * t)
    bm.normal_update()
    # Every shape here is built with its faces wound outward; an open shell
    # would only confuse recalc_face_normals.
    bm.to_mesh(me)
    bm.free()
    cellmap = {}
    # recover the cell of each face from its uv to set the slot
    uvd = me.uv_layers.active.data
    values = []
    for p in me.polygons:
        for li in p.loop_indices:
            u, v = uvd[li].uv
            lu = (u - 0.5) * 2
            col = int(lu * 8)
            values.append((slot_of_cell.get(col, 0), part))
    set_corner_colour(me, values)
    me.materials.append(mat)
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    vg = o.vertex_groups.new(name=bone)
    vg.add(list(range(len(me.vertices))), 1.0, 'REPLACE')
    m = o.modifiers.new('Armature', 'ARMATURE')
    m.object = rig
    o.parent = rig
    kept.append(o)
    return o

# Slot for each of our cells, by column.
HAT_SLOTS = {C_NEUTRAL[0]: HAT}


def tricorn():
    bm = bmesh.new()
    z0 = 1.98
    # crown: a low rounded drum
    crown = [(0.57, z0 - 0.08), (0.6, z0 + 0.1), (0.58, z0 + 0.3), (0.48, z0 + 0.44), (0.25, z0 + 0.5), (0.0, z0 + 0.51)]
    faces, _ = lathe(bm, crown, n=40)
    # brim: a closed section, cocked up on three sides, one point to the front (-Y)
    front = -math.pi / 2

    def cock(th, i, r, z):
        c3 = math.cos(3 * (th - front))
        flap = 0.5 - 0.5 * c3  # 0 at the three points, 1 mid-flap
        if i in (1, 2):  # outer edge
            return r + 0.09 * c3, z + 0.27 * flap
        return r, z
    brim = [(0.56, z0), (0.8, z0 - 0.02), (0.8, z0 + 0.04), (0.56, z0 + 0.05), (0.56, z0)]
    f2, _ = lathe(bm, brim, n=60, shape=cock, cells=[C_NEUTRAL, C_GOLD, C_NEUTRAL, C_NEUTRAL])
    return finish('Gear_Tricorn', bm, faces + f2, HAT_SLOTS, P_TRICORN, 'head', (z0 - 0.1, z0 + 0.5))


def monmouth_cap():
    bm = bmesh.new()
    z0 = 1.9
    prof = [(0.6, z0 - 0.04), (0.63, z0 + 0.04), (0.62, z0 + 0.12), (0.585, z0 + 0.14),
            (0.6, z0 + 0.3), (0.52, z0 + 0.48), (0.36, z0 + 0.6), (0.16, z0 + 0.66), (0.0, z0 + 0.67)]

    def lean(th, i, r, z):
        # the crown flops back a little
        y = math.sin(th)
        return r * (1 + 0.05 * y * max(0, (z - z0) / 0.6)), z - 0.04 * y * max(0, (z - z0) / 0.6)
    faces, zr = lathe(bm, prof, n=36, shape=lean)
    return finish('Gear_Cap', bm, faces, HAT_SLOTS, P_CAP, 'head', zr)


def kerchief():
    bm = bmesh.new()
    # a dome cut by a plane: 1.95 over the brow, 1.66 at the nape
    prof = [(0.585, 0.0), (0.6, 0.12), (0.57, 0.3), (0.47, 0.45), (0.3, 0.55), (0.0, 0.59)]

    def cut(th, i, r, z):
        y = math.sin(th)  # -1 front, +1 back
        # low over the nape, roomier at the back for a mop or a queue
        base = 1.92 - 0.24 * (y + 1)
        r = r * (1 + 0.12 * max(0.0, y))
        return r, base + z * (1 - 0.1 * y) * (1 + 0.25 * max(0.0, y)) if i else base
    faces, _ = lathe(bm, prof, n=36, shape=cut)
    # the knot and two tails at the nape
    knot = bmesh.ops.create_uvsphere(bm, u_segments=10, v_segments=6, radius=0.12)
    for v in knot['verts']:
        v.co = Vector((v.co.x * 1.1, v.co.y * 0.8 + 0.7, v.co.z + 1.6))
    for sx in (-1, 1):
        tail = bmesh.ops.create_cube(bm, size=1)
        for v in tail['verts']:
            v.co = Vector((v.co.x * 0.12 + sx * 0.08, v.co.y * 0.04 + 0.72, v.co.z * 0.3 + 1.42))
            v.co.x += sx * (1.6 - v.co.z) * 0.25
    extra = [f for f in bm.faces if f not in {fc for fc, _ in faces}]
    faces += [(f, C_NEUTRAL) for f in extra]
    return finish('Gear_Kerchief', bm, faces, HAT_SLOTS, P_KERCHIEF, 'head', (1.5, 2.45))


def eye_patch():
    bm = bmesh.new()
    patch = bmesh.ops.create_uvsphere(bm, u_segments=14, v_segments=8, radius=0.13)
    for v in patch['verts']:
        v.co = Vector((v.co.x + EYE_L.x, v.co.y * 0.3 + EYE_L.y - 0.035, v.co.z * 0.9 + EYE_L.z))
    faces = [(f, C_BLACK) for f in bm.faces]
    # the strap: a thin band around the head, low over the left eye, high over the right temple
    n = 48
    tilt = math.atan2(0.34, 1.0)
    rot = Matrix.Rotation(-tilt, 4, 'Y')
    centre = Vector((0.0, -0.02, EYE_L.z + 0.17))
    rings = []
    for k in range(n):
        th = 2 * math.pi * k / n
        ring = []
        for dr, dz in ((0.0, -0.025), (0.035, -0.025), (0.035, 0.025), (0.0, 0.025)):
            rx, ry = 0.575 + dr, 0.56 + dr
            p = Vector((rx * math.cos(th), ry * math.sin(th), dz))
            ring.append(bm.verts.new(rot @ p + centre))
        rings.append(ring)
    for k in range(n):
        a, b = rings[k], rings[(k + 1) % n]
        for i in range(4):
            f = bm.faces.new((a[i], b[i], b[(i + 1) % 4], a[(i + 1) % 4]))
            faces.append((f, C_BLACK))
    return finish('Gear_EyePatch', bm, faces, {}, P_PATCH, 'head', (EYE_L.z - 0.2, EYE_L.z + 0.6))


def sash():
    body = bpy.data.objects['Rogue_Body']
    band = [v.co for v in body.data.vertices if 0.58 < v.co.z < 0.74]
    rx = max(abs(c.x) for c in band) + 0.03
    ry = max(abs(c.y) for c in band) + 0.03
    cy = sum(c.y for c in band) / len(band)
    bm = bmesh.new()
    n = 36
    prof = [(1.0, 0.57), (1.04, 0.6), (1.06, 0.66), (1.04, 0.73), (1.0, 0.76)]
    rings = []
    for k in range(n):
        th = 2 * math.pi * k / n
        rings.append([bm.verts.new((rx * r * math.cos(th), cy + ry * r * math.sin(th), z)) for r, z in prof])
    faces = []
    for k in range(n):
        a, b = rings[k], rings[(k + 1) % n]
        for i in range(len(prof) - 1):
            faces.append((bm.faces.new((a[i], b[i], b[i + 1], a[i + 1])), C_NEUTRAL))
    # inner wall, so the band is not see-through from above
    # knot and two hanging ends on the left hip
    knot = bmesh.ops.create_uvsphere(bm, u_segments=10, v_segments=6, radius=0.09)
    for v in knot['verts']:
        v.co = Vector((v.co.x + rx * 0.72, v.co.y * 0.7 + cy - ry * 0.72, v.co.z + 0.66))
    for dx in (0.0, 0.1):
        end = bmesh.ops.create_cube(bm, size=1)
        for v in end['verts']:
            v.co = Vector((v.co.x * 0.1 + rx * 0.68 + dx, v.co.y * 0.04 + cy - ry * 0.78, v.co.z * 0.26 + 0.5 - dx * 0.4))
    known = {f for f, _ in faces}
    faces += [(f, C_NEUTRAL) for f in bm.faces if f not in known]
    return finish('Gear_Sash', bm, faces, {C_NEUTRAL[0]: SASH}, P_SASH, 'hips', (0.3, 0.8))


tricorn()
monmouth_cap()
kerchief()
eye_patch()
sash()

# ---------------------------------------------------------------- clean up
for o in list(bpy.data.objects):
    if o.type == 'MESH' and o not in kept:
        bpy.data.objects.remove(o, do_unlink=True)
    elif o.type == 'ARMATURE' and o is not rig:
        bpy.data.objects.remove(o, do_unlink=True)
    elif o.type not in ('MESH', 'ARMATURE'):
        bpy.data.objects.remove(o, do_unlink=True)
for b in rig.data.bones:
    if 'IK' in b.name or b.name.startswith('control-'):
        b.use_deform = False

for a in list(bpy.data.actions):
    if a.name not in KEEP_CLIPS:
        bpy.data.actions.remove(a)
rig.animation_data.action = bpy.data.actions['Idle']
try:
    rig.animation_data.action_slot = rig.animation_data.action_suitable_slots[0]
except Exception:
    pass

# One mesh, one material.
bpy.ops.object.select_all(action='DESELECT')
for o in kept:
    o.select_set(True)
body = bpy.data.objects['Rogue_Body']
bpy.context.view_layer.objects.active = body
bpy.ops.object.join()
body.name = 'Crew'
body.data.name = 'Crew'
while len(body.data.materials) > 1:
    body.data.materials.pop(index=1)
body.data.materials[0] = mat
for p in body.data.polygons:
    p.material_index = 0
print('TRIS', sum(len(p.vertices) - 2 for p in body.data.polygons), 'VERTS', len(body.data.vertices))
print('ACTIONS', [a.name for a in bpy.data.actions])

kw = dict(filepath=OUT, export_format='GLB', use_selection=False, export_animation_mode='ACTIONS',
          export_def_bones=True, export_skins=True, export_morph=False, export_yup=True,
          export_texcoords=True, export_normals=True, export_tangents=False, export_cameras=False,
          export_lights=False, export_extras=False, export_force_sampling=True)
try:
    bpy.ops.export_scene.gltf(**kw, export_vertex_color='ACTIVE')
except TypeError:
    bpy.ops.export_scene.gltf(**kw, export_colors=True)
print('WROTE', OUT)
