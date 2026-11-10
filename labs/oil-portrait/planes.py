"""v4 块面写生: paint the way a portrait painter works from life, area by area, in flat planes.

Learned from a real painter's 57 s time-lapse (user's reference, 2026-11-10): on a toned canvas with a loose sketch, the
face goes in first as a mosaic of flat planes (one flat brush stroke = one plane, crisp edges, dark planes then light),
then the hair as a few dark masses, then the features, then the clothes in a few wide strokes, then the background,
which cuts back into the silhouette; the last stretch is a close-up refinement of the features. Untouched areas stay bare
canvas with sketch lines until their turn, which is what makes the process read as a person painting rather than an
image coming into focus.

plane_strokes: quantise the target's colours inside an area, take each connected plane, and fill it with parallel flat
strokes along its long axis, each in the target's colour under it. The finer passes and the features reuse the v3 layer
planner (paint.plan_layer), restricted to an area mask.
"""
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

import paint
from paint import (H, W, MEDIA, orientation, REGIONS, AREAS, FEATURES, EYE_FEATURES, LINE_FEATURES, LAYERS_V2, rgb2lab, lab2rgb, kmeans,
                   ellipse_mask, focus_map, skin_mask, painter_order, render, shade, canvas_tooth, load_subject,
                   sketch_strokes, line_strokes, eye_touches, signature_strokes, tag_feature, plan_layer)

GROUND = np.array([.885, .872, .842])    # a canvas toned light warm grey before the session, as in the reference video



AREA_MAP = None     # optional area map registered to the photo (flat colours, made by the image model): see AREA_COLOURS
AREA_COLOURS = {'skin': (240, 160, 120), 'hair': (90, 50, 20), 'clothes': (60, 90, 210), 'bg': (210, 210, 210)}


def area_masks(ref, rng):
    """Background, skin, hair and clothes: from an area map when there is one, else from the target's colours and the
    face/figure ellipses (enough for a plain background and dark hair; a warm patterned dress or auburn hair fools it)."""
    if AREA_MAP:
        return area_masks_from_map(AREA_MAP)
    lab = rgb2lab(ndimage.gaussian_filter(ref, (3, 3, 0)))
    face = ellipse_mask(REGIONS['face'])
    fx, fy, frx, fry = REGIONS['face']
    head = ellipse_mask((fx, fy - .4 * fry, frx * 1.6, fry * 1.5))
    # background: the colours along the top edge and the upper side edges, grown inward from those edges
    seed = np.zeros((H, W), bool)
    seed[:10] = True
    seed[:int(H * .45), :10] = True
    seed[:int(H * .45), -10:] = True
    samples = lab[seed]
    centres, _ = kmeans(samples[rng.choice(len(samples), min(4000, len(samples)), replace=False)], 5, rng)
    dist = np.min([np.linalg.norm(lab - c, axis=-1) for c in centres], 0)
    near = (dist < 11) & ~face
    comp, _ = ndimage.label(near)
    bg = np.isin(comp, np.unique(comp[seed & near]))
    bg = ndimage.binary_opening(bg, iterations=3)
    person = ~bg
    person = ndimage.binary_fill_holes(ndimage.binary_closing(person, iterations=4))
    bg = ~person
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    skin = person & (lab[..., 0] > 28) & (lab[..., 1] > 3) & (lab[..., 1] < 42) & (lab[..., 2] > 6) & (chroma > 11)
    skin = ndimage.binary_opening(skin, iterations=2)
    comp, n = ndimage.label(skin)
    sizes = ndimage.sum(skin, comp, range(1, n + 1))
    skin = np.isin(comp, 1 + np.flatnonzero(sizes > 2500)) | (skin & face)
    skin = ndimage.binary_closing(skin, iterations=3) & person
    hair = person & ~skin & head & (lab[..., 0] < 62) & ~face
    hair = ndimage.binary_opening(ndimage.binary_closing(hair, iterations=3), iterations=2)
    skin &= ~(head & ~ellipse_mask(REGIONS['face'], 1.05) & (lab[..., 0] < 50))   # warm hair highlights are not skin
    skin |= face & person & ~hair                         # eyes, brows and lips belong to the face, not the clothes
    clothes = person & ~skin & ~hair
    feats = np.zeros((H, W), bool)
    for e in FEATURES:
        feats |= ellipse_mask(e, 1.3)
    AREAS.update(bg=bg, skin=skin, hair=hair, clothes=clothes, person=person, features=feats,
                 bg_edge=bg & ndimage.binary_dilation(person, iterations=22),
                 face=(face | (skin & ellipse_mask(REGIONS['face'], 1.25))) & person)
    return AREAS


def area_masks_from_map(path):
    m = load_subject(path)[0] * 255
    names = list(AREA_COLOURS)
    ref = np.array([AREA_COLOURS[n] for n in names], float)
    lbl = np.argmin(((m[..., None, :] - ref[None, None]) ** 2).sum(-1), -1)
    size = 7                                              # majority filter: clean edges, no stray pixels
    votes = np.stack([ndimage.uniform_filter((lbl == k).astype(np.float32), size) for k in range(len(names))])
    lbl = votes.argmax(0)
    masks = {n: lbl == k for k, n in enumerate(names)}
    person = ~masks['bg']
    face = ellipse_mask(REGIONS['face'])
    feats = np.zeros((H, W), bool)
    for e in FEATURES:
        feats |= ellipse_mask(e, 1.3)
    masks['skin'] |= face & person & ~masks['hair']
    masks['clothes'] &= ~masks['skin']
    AREAS.update(masks, person=person, features=feats, bg_edge=masks['bg'] & ndimage.binary_dilation(person, iterations=22),
                 face=(face | (masks['skin'] & ellipse_mask(REGIONS['face'], 1.25))) & person)
    return AREAS


def plane_strokes(ref, canvas, mask, R, k, rng, T=0., start_tip=(0, 0), soft=0., medium='flat', spacing=1.45):
    """Flat planes: the target's colours inside mask quantised to k mixes, each connected plane filled with parallel
    straight strokes along its long axis, each stroke in the target's colour under it. T skips strokes where the canvas
    is already within T (ΔE) of the target (a finer pass only corrects)."""
    lab = rgb2lab(ndimage.gaussian_filter(ref, (R * .3, R * .3, 0)))
    can = rgb2lab(canvas)
    if mask.sum() < 50:
        return []
    tx, ty = orientation(lab[..., 0], R, rng)             # the target's own brush direction (its painted streaks)
    c2, s2 = tx * tx - ty * ty, 2 * tx * ty               # doubled angle: a direction, not an arrow
    wv = np.array([1.35, 1., 1.])                         # value separates planes before hue does
    pts = lab[mask] * wv
    centres, _ = kmeans(pts[rng.choice(len(pts), min(len(pts), 30000), replace=False)], k, rng)
    lbl = np.full((H, W), -1)
    flat = np.flatnonzero(mask.ravel())
    for i in range(0, len(flat), 200000):
        chunk = flat[i:i + 200000]
        lbl.ravel()[chunk] = np.argmin(((lab.reshape(-1, 3)[chunk, None] * wv - centres[None]) ** 2).sum(-1), 1)
    size = max(3, int(R * .8) | 1)                         # majority filter: planes, not speckle
    votes = np.stack([ndimage.uniform_filter((lbl == c).astype(np.float32), size) for c in range(len(centres))])
    lbl = np.where(mask, votes.argmax(0), -1)
    strokes = []
    min_area = (1.2 * R) ** 2
    inside = ndimage.binary_dilation(mask, iterations=2)
    for c in range(len(centres)):
        comp, n = ndimage.label(lbl == c)
        for i, sl in enumerate(ndimage.find_objects(comp)):
            if sl is None:
                continue
            sub = comp[sl] == i + 1
            if sub.sum() < min_area:
                continue
            ys, xs = np.nonzero(sub)
            ys, xs = ys + sl[0].start, xs + sl[1].start
            P = np.c_[xs, ys].astype(float)
            cen = P.mean(0)
            evals = np.linalg.eigvalsh(np.cov((P - cen).T) + np.eye(2) * 1e-6)
            Rp = float(np.clip(2 * np.sqrt(evals[0]), 3., R))   # a brush no wider than the plane (an earring is not a dress)
            mc, ms = c2[ys, xs].mean(), s2[ys, xs].mean()
            if np.hypot(mc, ms) > .25:                     # strokes run the way the target's paint runs
                phi = np.arctan2(ms, mc) / 2
                u = np.array([np.cos(phi), np.sin(phi)])
            else:                                          # no clear direction: along the plane's long axis
                evals, evecs = np.linalg.eigh(np.cov((P - cen).T) + np.eye(2) * 1e-6)
                u = evecs[:, -1]
            tilt = rng.normal(0, .12)                      # a hand never lays planes exactly on the axis
            u = np.array([u[0] * np.cos(tilt) - u[1] * np.sin(tilt), u[0] * np.sin(tilt) + u[1] * np.cos(tilt)])
            v = np.array([-u[1], u[0]])
            pu, pv = (P - cen) @ u, (P - cen) @ v
            lo, hi = pv.min(), pv.max()
            offs = np.arange(lo + .6 * Rp, hi - .3 * Rp + 1e-6, spacing * Rp) if hi - lo > 1.2 * Rp else np.array([(lo + hi) / 2])
            for o in offs:
                band = np.abs(pv - o) < .55 * Rp
                if band.sum() < Rp:
                    continue
                bu = np.sort(pu[band])
                cuts = np.flatnonzero(np.diff(bu) > Rp)
                for run in np.split(bu, cuts + 1):
                    u0, u1 = run[0] - .25 * Rp, run[-1] + .25 * Rp
                    if u1 - u0 < .9 * Rp:
                        mid = (u0 + u1) / 2
                        u0, u1 = mid - .5 * Rp, mid + .5 * Rp
                    # an end that would run out of the area (onto the hair, the figure...) stops at its edge and lifts off:
                    # square ends stacked along a silhouette read as a staircase
                    press = [.85, .7]
                    for e, (uu, edge) in enumerate(((u0, run[0]), (u1, run[-1]))):
                        q = np.rint(cen + u * uu + v * o).astype(int)
                        if not (0 <= q[0] < W and 0 <= q[1] < H) or not inside[q[1], q[0]]:
                            if e == 0:
                                u0 = edge
                            else:
                                u1 = edge
                            press[e] = .3
                    sel = band & (pu >= run[0]) & (pu <= run[-1])
                    yy, xx = ys[sel], xs[sel]
                    if T and np.linalg.norm(lab[yy, xx] - can[yy, xx], axis=1).mean() < T:
                        continue
                    colour = lab2rgb(np.median(lab[yy, xx], 0))
                    m = max(2, int((u1 - u0) / (2 * Rp)) + 1)
                    tt = np.linspace(u0, u1, m)
                    jit = rng.normal(0, .06 * Rp, m)
                    line = [cen + u * t + v * (o + j) for t, j in zip(tt, jit)]
                    strokes.append(dict(points=[(float(a), float(b)) for a, b in line], R=float(Rp * rng.uniform(.9, 1.08)),
                                        rgb=[float(x) for x in colour], mix=int(c),
                                        mix_rgb=[float(x) for x in lab2rgb(centres[c] / wv)], medium=medium,
                                        seed=int(rng.integers(1 << 30)), load=float(rng.uniform(.95, 1.15)), press=tuple(press),
                                        press_fixed=True, soft=soft, pvar=.03, wobble=0))
    return painter_order(strokes, centres[:, 0] / wv[0], start_tip, rng)


def v2_spec(name, **over):
    spec = dict(next(s for s in LAYERS_V2 if s['name'] == name))
    spec.update(over)
    return spec


# area by area, as the painter in the reference video: (name, label, kind, args, video budget in seconds)
STEPS_V4 = [
    ('face', '脸部块面', 'planes', dict(area='skin', R=20, k=8), 11.),
    ('face2', '脸部块面', 'planes', dict(area='skin', R=8, k=14, T=5), 6.),
    ('hair', '头发', 'planes', dict(area='hair', R=13, k=6), 5.),
    ('strands', '头发', 'layer', v2_spec('texture', only=None, exclude=[], area='hair', cover=1.2), 3.),
    ('detail', '五官', 'layer', v2_spec('detail', area='features', skin=False), 3.),
    ('clothes', '衣服', 'planes', dict(area='clothes', R=24, k=7), 7.),
    ('clothes2', '衣服', 'layer', v2_spec('refine', area='clothes', focal=False, skin=False, medium='flat', R=9, T=7, gamma=0.), 4.),
    ('clothes3', '衣服', 'layer', v2_spec('detail', area='clothes', focal=False, skin=False, T=9, gamma=0.), 4.),
    ('background', '背景', 'planes', dict(area='bg', R=30, k=6), 6.),
    ('background2', '背景', 'planes', dict(area='bg', R=14, k=9, T=6), 3.),
    # cutting in: round, curved background strokes along the silhouette, as a painter cleans the edge of the figure
    ('cutin', '背景', 'layer', v2_spec('refine', area='bg_edge', focal=False, skin=False, gamma=0., T=5, R=7, medium='oil',
                                      maxlen=8), 3.),
    ('face3', '深入', 'planes', dict(area='face', R=5, k=16, T=6), 3.),
    ('polish', '深入', 'layer', v2_spec('polish', area='features', skin=False, feather_after=False), 3.),
    ('scumble', '干擦', 'layer', v2_spec('scumble'), 2.),
    ('accent', '深入', 'layer', v2_spec('accent', area='features'), 2.),
]


PACE_V4 = dict(feature_cap=5., eye_life={'polish': 7., 'lines': 6.})


def plan_v4(subject_path, sketch_path, out, focus_ellipses, signature=None, eyes=None, photo_path=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(paint.SEED)
    ref, crop, shape = load_subject(subject_path)
    shown = load_subject(photo_path)[0] if photo_path else ref
    Image.fromarray((shown * 255).astype(np.uint8)).save(out / 'reference.png')
    Image.fromarray((ref * 255).astype(np.uint8)).save(out / 'target.png')
    tooth = canvas_tooth(np.random.default_rng(paint.SEED + 1))
    np.save(out / 'tooth.npy', tooth.astype(np.float32))
    canvas = np.tile(GROUND, (H, W, 1))
    height, thick = np.zeros((H, W)), np.zeros((H, W))
    focus = focus_map(focus_ellipses, .35)
    FEATURES[:] = [tuple(e[:4]) for e in focus_ellipses if e[4] >= .95]
    LINE_FEATURES[:] = [tuple(e[:4]) for e in focus_ellipses if e[4] >= .999]
    EYE_FEATURES.clear()
    for ex, ey, *_ in (eyes or []):
        around = [k for k, f in enumerate(FEATURES) if ((ex - f[0]) / f[2]) ** 2 + ((ey - f[1]) / f[3]) ** 2 <= 1]
        if around:
            EYE_FEATURES.add(min(around, key=lambda k: FEATURES[k][2] * FEATURES[k][3]))
    area_masks(ref, rng)
    vis = np.zeros((H, W, 3))
    for name, colour in (('bg', (.85, .85, .85)), ('clothes', (.3, .4, .8)), ('hair', (.35, .2, .1)), ('skin', (.95, .7, .55))):
        vis[AREAS[name]] = colour
    Image.fromarray((vis * 255).astype(np.uint8)).save(out / 'areas.png')
    passes = []
    started = time.time()

    def run(name, label, strokes, budget):
        for st in strokes:
            render(st, canvas, height, thick, tooth)
        passes.append(dict(name=name, label=label, strokes=strokes, budget=budget))
        Image.fromarray((canvas * 255).astype(np.uint8)).save(out / f'pass-{len(passes):02d}-{name}.png')
        print(f'{name:10s} {len(strokes):5d} strokes  {time.time() - started:6.1f}s', flush=True)
        return strokes[-1]['points'][-1] if strokes else None

    sketch = sketch_strokes(sketch_path, shape, crop, rng) if sketch_path else []
    if sketch:
        run('sketch', '起稿', sketch, 5.)
    tip = (W * .5, H * .3)
    for name, label, kind, args, budget in STEPS_V4:
        if name == 'scumble' and 'texture' not in REGIONS:
            continue
        if kind == 'planes':
            a = dict(args)
            strokes = plane_strokes(ref, canvas, AREAS[a.pop('area')], rng=rng, start_tip=tip, **a)
        else:
            strokes = plan_layer(args, ref, canvas, focus, rng, tip)
        tip = run(name, label, strokes, budget) or tip
        if name == 'accent':
            if sketch:
                raw = line_strokes(sketch, ref, rng)
                cuts = painter_order([st for st in raw if st.get('cut')], np.zeros(1), tip, rng)
                lines = painter_order([st for st in raw if not st.get('cut')], np.zeros(1), cuts[-1]['points'][-1] if cuts else tip, rng)
                for st in cuts + lines:
                    st['mix'] = 0
                    tag_feature(st)
                tip = run('lines', '勾勒', cuts + lines, 0.) or tip
            if eyes:
                touches = eye_touches(ref, eyes, rng)
                for st in touches:
                    st['focal'] = True
                    st['eye'] = True
                tip = run('dianjing', '点睛', touches, 0.) or tip
    if signature:
        text, box = signature
        x0, y0, w, h = box
        under = canvas[y0:y0 + h, x0:x0 + w].mean((0, 1))
        colour = (.80, .52, .30) if under.mean() < .45 else (.42, .16, .12)
        run('sign', '落款', signature_strokes(text, box, colour, rng), 3.)
    Image.fromarray((canvas * 255).astype(np.uint8)).save(out / 'final-flat.png')
    Image.fromarray((shade(canvas, height, thick, tooth) * 255).astype(np.uint8)).save(out / 'final.png')
    (out / 'plan.json').write_text(json.dumps(dict(W=W, H=H, passes=passes, focus=focus_ellipses, face=list(REGIONS['face']),
                                                   ground=GROUND.tolist(), style='v4', pace=PACE_V4), ensure_ascii=False))
    return passes
