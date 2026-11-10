"""素描: draw a portrait in graphite the way a person does, toward a pencil drawing registered to the photo.

Order (each layer laid stroke by stroke and judged against the paper as it is at that moment):
  起形 light construction lines -> 铺大关系 the big darks area by area (the hair in long strands along its flow, the
  face's shadow side in diagonal hatching, the knit along its ribs, a little background) -> 揉擦 blending stump and
  tissue -> 深入 finer cross-hatching and strands toward the full drawing -> 提亮 kneaded eraser and eraser pen pull
  the lights back (the hair's shine was hatched over in the block-in, as a person does) -> 刻画五官 the features in
  fine lines at life pace -> 点睛 -> 签名.

Each layer builds toward a tone map: the drawing (the target) lightened and blurred for the early layers, the drawing
itself for the last, so the hatching is visible on the way and the sheet ends as the drawing. A line is laid only
where the paper is still lighter than its layer wants (an eraser stroke only where it is darker).

usage: sketch.py TARGET.png SKETCH.png AREAS.png OUT_DIR FOCUS_JSON EYES_JSON [SIGNATURE_JSON] --photo PHOTO --face JSON
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

import paint
import media
import planes
from paint import W, H, AREAS, FEATURES, EYE_FEATURES, REGIONS, load_subject, ellipse_mask

LUM = np.array([.299, .587, .114])
FLICK = np.array([math.cos(math.radians(-58)), math.sin(math.radians(-58))])      # a right hander's hatching slant
CROSS = np.array([math.cos(math.radians(-14)), math.sin(math.radians(-14))])
PACE = dict(feature_cap=6., eye_life={'polish': 8., 'dianjing': 3.})
SURF = None


def lum(rgb):
    return rgb @ LUM


def stroke(points, R, medium, rng, **kw):
    d = dict(points=[(float(x), float(y)) for x, y in points], R=float(R), medium=medium, seed=int(rng.integers(1 << 30)),
             lead=.4, rgb=[.4] * 3, mix=0, mix_rgb=[.4] * 3, press=(.45, .25), pvar=.1)
    d.update(kw)
    return d


def lay(st):
    p = SURF.render(st)
    if p is not None:
        SURF.commit(p)
    return st


def nearest_order(items, tip, key=lambda it: it[0]):
    """Greedy nearest-neighbour tour (what a hand does next: the closest unfinished bit)."""
    left = list(items)
    out = []
    tip = np.asarray(tip, float)
    while left:
        P = np.array([key(it) for it in left], float)
        k = int(np.argmin(((P - tip) ** 2).sum(1)))
        it = left.pop(k)
        out.append(it)
        tip = np.asarray(key(it), float)
    return out


def streamline(p, flow, ok, step, length, d, occupied=None):
    pts = [np.asarray(p, float)]
    walked = 0.
    while walked < length:
        x, y = pts[-1]
        ix, iy = int(x), int(y)
        if not (0 <= ix < W and 0 <= iy < H) or not ok[iy, ix]:
            break
        if occupied is not None and walked > 0 and occupied[iy, ix]:
            break
        t = np.array([flow[0][iy, ix], flow[1][iy, ix]])
        if t @ d < 0:
            t = -t
        d = .65 * d + .35 * t
        d /= np.linalg.norm(d) + 1e-9
        pts.append(pts[-1] + d * step)
        walked += step
    return pts


def split(run, max_len):
    """Cut a long line into the pieces a hand draws it in."""
    run = np.asarray(run, float)
    L = np.linalg.norm(np.diff(run, axis=0), axis=1).sum() if len(run) > 1 else 0.
    n = max(1, int(math.ceil(L / max_len)))
    cut = np.linspace(0, len(run) - 1, n + 1).round().astype(int)
    return [run[a:b + 1] for a, b in zip(cut[:-1], cut[1:]) if b - a >= 2]


def thin(run):
    run = np.asarray(run)
    return run[np.r_[0, np.arange(2, len(run) - 1, 2), len(run) - 1]] if len(run) > 5 else run


def gap(level, eraser=False):
    """How far the sheet is from a layer's tone (positive: work left to do)."""
    v, g = lum(SURF.canvas), lum(media.LEVELS[level])
    return (g - v) if eraser else (v - g)


def run_gap(run, level, eraser, half):
    run = np.asarray(run, float)
    t = np.gradient(run, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    nrm = np.stack([-t[:, 1], t[:, 0]], 1)
    g = []
    for k in (-half, 0., half):
        q = np.clip((run + nrm * k).astype(int), 0, [W - 1, H - 1])
        v, lv = lum(SURF.canvas[q[:, 1], q[:, 0]]), lum(media.LEVELS[level][q[:, 1], q[:, 0]])
        g.append((lv - v) if eraser else (v - lv))
    return float(np.mean(g))


def runs(pts, ok, min_len):
    out, cur = [], []
    for p, k in zip(pts, ok):
        if k:
            cur.append(p)
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    keep = []
    for r in out:
        r = np.asarray(r)
        if len(r) > 1 and np.linalg.norm(np.diff(r, axis=0), axis=1).sum() >= min_len:
            keep.append(r)
    return keep


def hatch(level, mask, rng, tip, *, R, spacing, length, pressure=.7, angle, patch=(46, 96), thr=.015, opacity=.85,
          medium='graphite', **kw):
    """Straight hatching in patches: parallel lines at a slant (a little turned patch to patch), laid where the sheet
    is still lighter than this layer's tone."""
    eraser = medium == 'eraser'
    active = mask & (ndimage.gaussian_filter(gap(level, eraser), 2.) > thr)
    if active.sum() < 40:
        return []
    reach = ndimage.binary_dilation(active, iterations=max(1, int(R))) & mask
    ys, xs = np.nonzero(active)
    step = patch[1] * .62
    centres = []
    for gy in np.arange(ys.min(), ys.max() + step, step):
        for gx in np.arange(xs.min(), xs.max() + step, step):
            cx, cy = gx + rng.uniform(-.4, .4) * step, gy + rng.uniform(-.4, .4) * step
            r = rng.uniform(*patch) / 2
            y0, y1, x0, x1 = int(max(0, cy - r)), int(min(H, cy + r)), int(max(0, cx - r)), int(min(W, cx + r))
            if y1 > y0 and x1 > x0 and active[y0:y1, x0:x1].mean() > .06:
                centres.append((np.array([cx, cy]), r))
    out = []
    for c0, r in nearest_order(centres, tip):
        a = math.atan2(angle[1], angle[0]) + math.radians(rng.normal(0, 5))
        t = np.array([math.cos(a), math.sin(a)])
        nrm = np.array([-t[1], t[0]])
        o = -r + rng.uniform(0, spacing)
        back = False                               # back and forth, the hand never leaving the paper for long
        while o < r:
            half = math.sqrt(max(0., r * r - o * o)) * rng.uniform(.85, 1.1)
            mid = c0 + nrm * o
            o += spacing * rng.uniform(.88, 1.12)
            if half < 4:
                continue
            pts = np.array([mid + t * u for u in np.linspace(-half, half, max(2, int(2 * half / 3)))])
            if back:
                pts = pts[::-1]
            back = not back
            pi = np.clip(pts.astype(int), 0, [W - 1, H - 1])
            inside = (pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H) & reach[pi[:, 1], pi[:, 0]]
            for r_ in runs(pts, inside, length[0] * .45):
                for piece in split(r_, length[1]):
                    if run_gap(piece, level, eraser, spacing * .5) < thr:
                        continue
                    out.append(lay(stroke(thin(piece), R * rng.uniform(.9, 1.1), medium, rng, level=level,
                                          pressure=pressure, opacity=opacity, **kw)))
        tip = out[-1]['points'][-1] if out else tip
    return out


def strands(level, mask, flow, rng, tip, *, R, spacing, length, pressure=.75, thr=.015, opacity=.85,
            medium='graphite', down=True, section=160., **kw):
    """Long lines along a flow field, evenly spaced (a new line starts only where none runs near): hair drawn strand
    by strand, section by section."""
    eraser = medium == 'eraser'
    active = mask & (ndimage.gaussian_filter(gap(level, eraser), 2.) > thr)
    if active.sum() < 40:
        return []
    ok = ndimage.binary_dilation(active, iterations=max(1, int(R))) & mask
    occupied = np.zeros((H, W), bool)
    r = max(1, int(spacing * .5))
    ys, xs = np.nonzero(active)
    seeds = np.column_stack([xs, ys])[rng.permutation(len(xs))[:max(1, int(len(xs) / (spacing * spacing * .6)))]]
    lines = []
    for sx, sy in seeds:
        if occupied[sy, sx]:
            continue
        t = np.array([flow[0][sy, sx], flow[1][sy, sx]])
        if down and t[1] < 0:
            t = -t
        back = streamline((sx, sy), flow, ok, 3., length[1] * .5, -t, occupied)
        fwd = streamline((sx, sy), flow, ok, 3., length[1] * .5, t, occupied)
        line = np.asarray(back[::-1] + fwd[1:])
        if len(line) < 2 or np.linalg.norm(np.diff(line, axis=0), axis=1).sum() < length[0]:
            continue
        if down and line[-1][1] < line[0][1]:
            line = line[::-1]
        for x, y in line[::2]:
            ix, iy = int(x), int(y)
            occupied[max(0, iy - r):iy + r + 1, max(0, ix - r):ix + r + 1] = True
        lines.append(line)
    blocks = {}                                    # section by section: the block of the head a line starts in
    for ln in lines:
        blocks.setdefault((int(ln[0][0] // section), int(ln[0][1] // section)), []).append(ln)
    out = []
    for _, key in nearest_order([(np.array(k, float) * section + section / 2, k) for k in blocks], tip):
        for k, ln in enumerate(sorted(blocks[key], key=lambda ln: ln[0][0])):
            if k % 2:                              # down one strand, up the next
                ln = ln[::-1]
            for piece in split(ln, length[1]):
                if run_gap(piece, level, eraser, spacing * .5) < thr:
                    continue
                out.append(lay(stroke(thin(piece), R * rng.uniform(.9, 1.1), medium, rng, level=level,
                                      pressure=pressure, opacity=opacity, press=(.35, .15), **kw)))
    return out


def scrub(mask, rng, tip, *, R, length, flow, opacity, step=None, clip=None, has=None, medium='stump', level=None,
          thr=.015):
    """Blending stump / tissue / kneaded eraser: overlapping short strokes along the form over a region."""
    active = mask if has is None else mask & has
    if active.sum() < 40:
        return []
    step = step or R * 1.25
    ys, xs = np.nonzero(active)
    pts = []
    for gy in np.arange(ys.min(), ys.max() + step, step):
        for gx in np.arange(xs.min(), xs.max() + step, step):
            cx, cy = gx + rng.uniform(-.45, .45) * step, gy + rng.uniform(-.45, .45) * step
            ix, iy = int(np.clip(cx, 0, W - 1)), int(np.clip(cy, 0, H - 1))
            if active[iy, ix]:
                pts.append(np.array([cx, cy]))
    out = []
    for (p0,) in nearest_order([(p,) for p in pts], tip):
        ix, iy = int(np.clip(p0[0], 0, W - 1)), int(np.clip(p0[1], 0, H - 1))
        t = np.array([flow[0][iy, ix], flow[1][iy, ix]])
        half = rng.uniform(*length) / 2
        p = np.asarray(streamline(p0, flow, mask, 3., half, -t)[::-1] + streamline(p0, flow, mask, 3., half, t)[1:])
        if len(p) < 3:
            continue
        if level and run_gap(p, level, medium == 'eraser', R * .5) < thr:
            continue
        out.append(lay(stroke(thin(p), R * rng.uniform(.85, 1.15), medium, rng, opacity=opacity, clip=clip,
                              press=(.6, .4), level=level)))
    return out


def fill_holes(level, mask, flow, rng, tip, *, R=1.3, rounds=2, clip=None, min_px=5, thr=.03):
    """Look over the sheet and touch up what the last layer skipped: specks and slivers still lighter than the
    drawing, each crossed with a few short lines along the form."""
    out = []
    for _ in range(rounds):
        holes = mask & (ndimage.gaussian_filter(gap(level), .7) > thr)
        comp, n = ndimage.label(holes, np.ones((3, 3)))
        if not n:
            break
        sizes = ndimage.sum_labels(np.ones((H, W)), comp, index=np.arange(1, n + 1))
        holes = np.isin(comp, np.flatnonzero(sizes >= min_px) + 1)
        if holes.sum() < 20:
            break
        region = ndimage.binary_dilation(holes, iterations=2) & mask
        add = strands(level, region, flow, rng, tip, R=R, spacing=max(1.6, R * 1.3), length=(6, 80), pressure=1., thr=.006,
                      opacity=.95, down=False, section=60., clip=clip)
        out += add
        tip = add[-1]['points'][-1] if add else tip
    return out


def fine_lines(target_v, region, rng, tip):
    """The drawing's own crisp dark lines inside the features (lid creases, lashes, the lip line, nostrils): the
    line-first engine's skeleton strokes over the target's thin dark marks."""
    from line_topology import extract_strokes
    v = ndimage.gaussian_filter(target_v, .6)
    bh = ndimage.grey_closing(v, size=(7, 7)) - v
    ink = (bh > .06) & region
    ink = ndimage.binary_opening(ink, np.ones((2, 2))) | ((bh > .12) & region)
    raw, *_ = extract_strokes(ink, size=W / 720)
    half = ndimage.distance_transform_edt(ink)
    out = []
    for st in raw:
        p = np.asarray(st['points'], float)
        if len(p) < 3:
            continue
        pi = np.clip(p.astype(int), 0, [W - 1, H - 1])
        R = float(np.clip(np.median(half[pi[:, 1], pi[:, 0]]) * 1.1, .8, 2.4))
        out.append(stroke(thin(p), R, 'graphite', rng, level='T', pressure=1., opacity=.95, press=(.3, .2), wobble=0))
    return [s for _, s in nearest_order([(np.mean(s['points'], 0), s) for s in out], tip)]


def plan_sketch(target_path, sketch_path, areas_path, out, focus, eyes, signature, photo_path):
    global SURF
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(paint.SEED)
    ref, crop, shape = load_subject(target_path)
    Image.fromarray((load_subject(photo_path)[0] * 255).astype(np.uint8)).save(out / 'reference.png')
    Image.fromarray((ref * 255).astype(np.uint8)).save(out / 'target.png')
    vt = lum(ref)
    paper_rgb = np.percentile(ref.reshape(-1, 3), 97, axis=0)
    P = float(lum(paper_rgb))
    tooth, ground = media.paper(np.random.default_rng(paint.SEED + 1), paper_rgb)
    media.set_paper(tooth, ground)
    np.save(out / 'tooth.npy', tooth)
    np.save(out / 'ground.npy', ground.astype(np.float32))
    T = np.minimum(ref, ground)                      # the drawing; no lighter than the paper it is drawn on
    blur = lambda s: ndimage.gaussian_filter(T, (s, s, 0))
    toward = lambda f, s: ground - f * (ground - blur(s))
    media.LEVELS.update(
        T=T,
        line=np.maximum(T, ground - .36),                         # construction lines: light, never darker than the end
        hair=ground - 1.05 * (ground - blur(16)),                                     # the hair's mass: its shine hatched over for now
        block=np.maximum(T, toward(.7, 4)),
        refine=np.maximum(T, toward(.92, 1.5)))
    for k, v in media.LEVELS.items():
        np.save(out / f'level-{k}.npy', v.astype(np.float16))
    FEATURES[:] = [tuple(e[:4]) for e in focus if e[4] >= .95]
    EYE_FEATURES.clear()
    for ex, ey, *_ in eyes:
        around = [k for k, f in enumerate(FEATURES) if ((ex - f[0]) / f[2]) ** 2 + ((ey - f[1]) / f[3]) ** 2 <= 1]
        if around:
            EYE_FEATURES.add(min(around, key=lambda k: FEATURES[k][2] * FEATURES[k][3]))
    planes.area_masks_from_map(areas_path, ref, rng)
    skin, hair, clothes, bg = AREAS['skin'], AREAS['hair'], AREAS['clothes'], AREAS['bg']
    for name in ('skin', 'hair', 'clothes', 'bg'):
        d = ndimage.distance_transform_edt(~AREAS[name])
        media.CLIPS[name] = np.clip(1 - (d - 2.) / 3., 0, 1).astype(np.float32)
        Image.fromarray((media.CLIPS[name] * 255).astype(np.uint8)).save(out / f'clip-{name}.png')
    feats = np.zeros((H, W), bool)          # around each feature (hand-marked ellipses)
    for e in FEATURES:
        feats |= ellipse_mask(e, 1.3)
    # what the close-up adds: the features' darkest marks (pupils, lash lines, the lip line, nostrils). Everything else
    # on the face is finished first, or the features look like pale masks (left light) or dark ovals (drawn early)
    local_dark = vt < ndimage.gaussian_filter(vt, 10) - .07
    core = ndimage.binary_dilation(local_dark & feats, iterations=1) & feats
    SURF = surf = media.PaperSurface(ground)
    flow_hair = paint.orientation(vt, 11., rng, default=90.)
    flow_form = paint.orientation(ndimage.gaussian_filter(vt, 3), 10., rng, default=-58.)
    passes = []
    started = time.time()
    state = dict(tip=(W * .45, H * .2))

    def run(name, label, strokes, budget, laid=True):
        if not laid:
            for st in strokes:
                lay(st)
        if not strokes:
            return
        if passes and passes[-1]['name'] == name:
            passes[-1]['strokes'] += strokes
        else:
            passes.append(dict(name=name, label=label, strokes=strokes, budget=budget))
        state['tip'] = strokes[-1]['points'][-1]
        Image.fromarray((np.clip(surf.canvas, 0, 1) * 255).astype(np.uint8)).save(out / f'pass-{len(passes):02d}-{name}.png')
        print(f'{name:10s} {len(strokes):5d} strokes  {time.time() - started:6.1f}s', flush=True)

    tip = lambda: state['tip']
    # 起形: the construction lines, light and searching
    lines = paint.sketch_strokes(sketch_path, shape, crop, rng)
    for st in lines:
        st.update(medium='graphite', level='line', pressure=.55, opacity=.8, R=1.05)
    run('sketch', '起形', lines, 8., laid=False)
    # 铺大关系: the big darks, area by area
    run('hair', '铺大关系', strands('hair', hair, flow_hair, rng, tip(), R=2.3, spacing=5., length=(40, 170),
                                pressure=.85, clip='hair'), 14.)
    run('hair', '铺大关系', strands('hair', hair, flow_hair, rng, tip(), R=2.1, spacing=3.8, length=(30, 150),
                                pressure=.9, clip='hair', thr=.02), 0.)
    run('face', '铺大关系', hatch('block', skin, rng, tip(), R=1.8, spacing=4.4, length=(24, 64), pressure=.72,
                              angle=FLICK, clip='skin'), 11.)
    run('face', '铺大关系', hatch('block', skin, rng, tip(), R=1.6, spacing=4.6, length=(20, 54), pressure=.66,
                              angle=CROSS, clip='skin', thr=.025), 0.)
    run('clothes', '铺大关系', strands('block', clothes, flow_form, rng, tip(), R=2.1, spacing=5., length=(24, 90),
                                   pressure=.7, clip='clothes', down=False), 7.)
    run('background', '铺大关系', hatch('block', bg, rng, tip(), R=2.4, spacing=6., length=(40, 110), pressure=.55,
                                    angle=FLICK, clip='bg', thr=.02, patch=(60, 130)), 5.)
    # 揉擦: stump on the skin, tissue along the hair and clothes
    on = ndimage.binary_dilation(lum(surf.canvas) < P - .02, iterations=6)
    run('blend', '揉擦', scrub(skin, rng, tip(), R=8., length=(26, 56), flow=flow_form, opacity=.6, clip='skin',
                             has=on, step=11.), 10.)
    run('blend', '揉擦', scrub(hair, rng, tip(), R=12., length=(80, 160), flow=flow_hair, opacity=.45, clip='hair',
                             step=17.), 0.)
    run('blend', '揉擦', scrub(clothes, rng, tip(), R=11., length=(40, 90), flow=flow_form, opacity=.45,
                             clip='clothes', step=15.), 0.)
    # 深入: finer cross-hatching and strands toward the full drawing
    run('refine', '深入', hatch('refine', skin, rng, tip(), R=1.3, spacing=3., length=(14, 44), pressure=.7,
                              angle=np.array([math.cos(-.6), math.sin(-.6)]), clip='skin', patch=(36, 72)), 14.)
    run('refine', '深入', strands('refine', hair, flow_hair, rng, tip(), R=1.5, spacing=3.4, length=(40, 220),
                                pressure=.85, clip='hair'), 0.)
    run('refine', '深入', strands('refine', clothes, flow_form, rng, tip(), R=1.5, spacing=3.6, length=(24, 80),
                                pressure=.75, clip='clothes', down=False), 0.)
    run('refine', '深入', hatch('refine', bg, rng, tip(), R=1.6, spacing=4.5, length=(30, 90), pressure=.6,
                              angle=FLICK, clip='bg', patch=(50, 110)), 0.)
    run('refine', '深入', scrub(skin & ~feats, rng, tip(), R=6., length=(20, 44), flow=flow_form, opacity=.35,
                              clip='skin', step=8.), 0.)
    # the face's last layer, the features' mid-tones included (only their darkest marks wait for the close-up)
    run('refine', '深入', hatch('T', skin & ~core, rng, tip(), R=1.6, spacing=2.2, length=(14, 44), pressure=.95,
                              angle=FLICK, clip='skin', opacity=.9, patch=(36, 72)), 0.)
    run('refine', '深入', hatch('T', skin & ~core, rng, tip(), R=1.6, spacing=2.4, length=(14, 44), pressure=.95,
                              angle=CROSS, clip='skin', opacity=.9, patch=(36, 72)), 0.)
    # 刻画五官: the drawing's own crisp lines, then fine tone, feature by feature at life pace
    polish = [lay(st) for st in fine_lines(vt, feats, rng, tip())]
    polish += hatch('T', core, rng, polish[-1]['points'][-1] if polish else tip(), R=1., spacing=1.6, length=(8, 26),
                    pressure=.95, angle=np.array([math.cos(-.75), math.sin(-.75)]), patch=(24, 44), opacity=.9)
    polish += hatch('T', core, rng, polish[-1]['points'][-1] if polish else tip(), R=1., spacing=1.8, length=(8, 26),
                    pressure=.95, angle=CROSS, patch=(24, 44), opacity=.9)
    by_feature = {}
    for st in polish:
        by_feature.setdefault(paint.tag_feature(st), []).append(st)
    ordered = []
    for k in sorted(by_feature, key=lambda k: (k in EYE_FEATURES, k)):
        for st in by_feature[k]:
            st['focal'] = k >= 0
        ordered += by_feature[k]
    ordered += fill_holes('T', core, flow_form, rng, ordered[-1]['points'][-1] if ordered else tip(), R=1., clip='skin')
    for st in ordered:
        if 'focal' not in st:
            st['focal'] = paint.tag_feature(st) >= 0
    run('polish', '刻画五官', ordered, 6.)
    # 点睛: each pupil's darkest touch
    touches = []
    for ex, ey, rx, ry in eyes:
        box = ellipse_mask((ex, ey, rx, ry))
        v = np.where(box, ndimage.gaussian_filter(vt, 1.5), 9)
        y, x = np.unravel_index(v.argmin(), v.shape)
        touches.append(stroke([(x - 2, y - 1), (x + 1, y + 1.5), (x + 2.5, y - .5)], 3., 'graphite', rng, level='T',
                              pressure=1., opacity=1., focal=True, eye=True, wobble=0))
    run('dianjing', '点睛', touches, 0., laid=False)
    # 提亮: the eraser pen along the hair's shine, the kneaded eraser on the skin's lights
    run('lift', '提亮', strands('T', hair, flow_hair, rng, tip(), R=1.8, spacing=3.6, length=(24, 160), pressure=1.,
                              medium='eraser', thr=.02, opacity=.9, clip='hair'), 8.)
    run('lift', '提亮', scrub(skin, rng, tip(), R=7., length=(10, 26), flow=flow_form, opacity=.6, clip='skin',
                            step=8., medium='eraser', level='T', thr=.02), 0.)
    # 收尾: the last layer of the hair, clothes and background, close and dense, the drawing itself
    run('finish', '收尾', strands('T', hair, flow_hair, rng, tip(), R=1.9, spacing=2.1, length=(30, 220), pressure=.95,
                                clip='hair', opacity=.9), 12.)
    run('finish', '收尾', strands('T', clothes, flow_form, rng, tip(), R=1.7, spacing=2.6, length=(24, 90),
                                pressure=.95, clip='clothes', down=False, opacity=.9), 0.)
    run('finish', '收尾', hatch('T', bg, rng, tip(), R=1.8, spacing=3., length=(30, 100), pressure=.9, angle=FLICK,
                              clip='bg', opacity=.9, patch=(50, 110)), 0.)
    for name, area, flow in (('hair', hair, flow_hair), ('skin', skin & ~core, flow_form), ('clothes', clothes, flow_form),
                             ('bg', bg, flow_form)):
        run('finish', '收尾', fill_holes('T', area, flow, rng, tip(), clip=name, R=1.8 if name == 'hair' else 1.3,
                                         min_px=14 if name == 'hair' else 6, thr=.05 if name == 'hair' else .03), 0.)
    if signature:
        text, box = signature
        sig = paint.signature_strokes(text, box, (.3, .3, .3), rng)
        for st in sig:
            st.update(medium='graphite', lead=.32, pressure=.85, opacity=.9, R=1.3)
        run('sign', '签名', sig, 3., laid=False)
    Image.fromarray((np.clip(surf.canvas, 0, 1) * 255).astype(np.uint8)).save(out / 'final.png')
    err = np.abs(ndimage.gaussian_filter(lum(surf.canvas), 2) - ndimage.gaussian_filter(lum(T), 2))
    print('mean |value error| (2px blur):', ' '.join(f'{n} {err[AREAS[n]].mean():.3f}' for n in ('skin', 'hair', 'clothes', 'bg')))
    print('strokes', sum(len(p['strokes']) for p in passes))
    (out / 'plan.json').write_text(json.dumps(dict(W=W, H=H, kind='sketch', title='素描肖像', passes=passes, focus=focus,
                                                   face=list(REGIONS['face']), pace=PACE), ensure_ascii=False))
    return passes


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('target')
    ap.add_argument('sketch')
    ap.add_argument('areas')
    ap.add_argument('out')
    ap.add_argument('focus')
    ap.add_argument('eyes')
    ap.add_argument('signature', nargs='?', default='-')
    ap.add_argument('--photo', required=True)
    ap.add_argument('--face', required=True)
    ap.add_argument('--crop-top', type=float, default=0.)
    a = ap.parse_args()
    paint.CROP_TOP = a.crop_top
    REGIONS['face'] = tuple(json.loads(a.face))
    plan_sketch(a.target, a.sketch, a.areas, a.out, json.loads(a.focus), json.loads(a.eyes),
                None if a.signature == '-' else json.loads(a.signature), a.photo)
