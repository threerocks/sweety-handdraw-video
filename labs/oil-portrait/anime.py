"""动漫插画: paint an anime illustration the way an illustrator does on a tablet, toward an illustration registered to
the photo.

Order: 草稿 a quick rough sketch in light blue on its own layer -> 线稿 clean line art traced over it, stroke by stroke
(the eyes at life pace) -> the rough layer is hidden -> 平涂 flat base colours under the line art, region by region,
each held inside its region as with a clipping mask -> 阴影 hard-edged cel shadows -> 高光 lights -> 渐变 soft airbrush
gradients and the last details -> 点睛 the eyes' highlights on top -> 签名.

The illustration is taken apart into what an illustrator paints in each step: its line art (thin dark marks, found
with a black-hat filter, traced into strokes by the line-first engine), the colours under the lines (the lines filled
in from their sides), and in each area (skin, hair, clothes, background, from an area map) its materials (a coarse
colour clustering: skin, eye whites, iris, lips...) and each material's shades (a finer clustering by lightness).

usage: anime.py TARGET.png ROUGH_SKETCH.png AREAS.png OUT_DIR FOCUS_JSON EYES_JSON [SIGNATURE_JSON] --photo PHOTO --face JSON
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
from paint import W, H, AREAS, FEATURES, EYE_FEATURES, REGIONS, load_subject, ellipse_mask, rgb2lab, kmeans

LUM = np.array([.299, .587, .114])
PACE = dict(feature_cap=6., eye_life={'lines': 7., 'dianjing': 3.})
ROUGH_RGB = [.42, .62, .92]
SURF = None
# per area: materials (coarse clusters, chroma first) and shades per material
CLUSTERS = {'skin': (4, 3), 'hair': (2, 4), 'clothes': (3, 3), 'bg': (3, 3)}


def lum(rgb):
    return rgb @ LUM


def stroke(points, R, medium, rng, **kw):
    rgb = kw.pop('rgb', [.5, .5, .5])
    d = dict(points=[(float(x), float(y)) for x, y in points], R=float(R), medium=medium, seed=int(rng.integers(1 << 30)),
             rgb=[float(c) for c in rgb], mix=0, mix_rgb=[float(c) for c in rgb], press=(.7, .6), pvar=.03, wobble=0)
    d.update(kw)
    return d


def lay(st):
    p = SURF.render(st)
    if p is not None:
        SURF.commit(p)
    return st


def nearest_order(items, tip, key=lambda it: it[0]):
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


# ------------------------------------------------------------------------------------------- taking apart
def line_art(A):
    """Coverage and colour of the line art: thin marks darker than both their sides."""
    v = ndimage.gaussian_filter(lum(A), .5)
    bh = ndimage.grey_closing(v, size=(9, 9)) - v
    la = np.clip((bh - .03) / .1, 0, 1) * (v < .62)
    comp, n = ndimage.label(la > .3, np.ones((3, 3)))
    if n:
        sizes = ndimage.sum_labels(np.ones((H, W)), comp, index=np.arange(1, n + 1))
        keep = np.isin(comp, np.flatnonzero(sizes >= 14) + 1)
        la = la * ndimage.binary_dilation(keep, iterations=2)
    return la


def with_clean_lines(A, la_own, lineart_path, eyes):
    """The line art an illustrator would have drawn: a clean line drawing registered to the illustration (made by
    the image model from it) for the outlines (face, hair silhouette, eyelids, mouth, clothes, leaves), and the
    illustration's own thin marks inside the hair and the irises (the drawing fills irises solid and draws brows
    through the bangs) and wherever the drawing has no line near. Returns coverage and colour of the line layer."""
    S = load_subject(lineart_path)[0]
    sa = np.clip((.96 - lum(S)) / .66, 0, 1)
    slc = np.clip((S - (1 - sa[..., None])) / np.maximum(sa, .3)[..., None], 0, 1)
    own = AREAS['hair'] & ndimage.binary_erosion(AREAS['hair'], iterations=5)
    for ex, ey, rx, ry in eyes:
        own |= ellipse_mask((ex, ey, rx * 1.15, ry * 1.15))
    sa = np.where(own, 0., sa)
    near = ndimage.distance_transform_edt(~(sa > .35)) <= 3
    la_t = np.where(near, 0., la_own)
    la = np.maximum(sa, la_t)
    return la, sa, slc


def under_lines(A, la):
    """The colours under the line art: each line pixel takes the colour of the nearest pixel beside the line."""
    m = ndimage.binary_dilation(la > .08, iterations=1)
    _, (iy, ix) = ndimage.distance_transform_edt(m, return_indices=True)
    c0 = A[iy, ix]
    return ndimage.median_filter(c0, size=(3, 3, 1))


def majority(lbl, mask, n, size=5):
    votes = np.stack([ndimage.uniform_filter(((lbl == k) & mask).astype(np.float32), size) for k in range(n)])
    return np.where(mask, votes.argmax(0), -1)


def merge_small(lbl, mask, min_px):
    small = np.zeros((H, W), bool)
    for k in np.unique(lbl[mask]):
        comp, n = ndimage.label(lbl == k)
        if n:
            sizes = ndimage.sum_labels(np.ones((H, W)), comp, index=np.arange(1, n + 1))
            small |= np.isin(comp, np.flatnonzero(sizes < min_px) + 1)
    keep = mask & ~small
    if small.any() and keep.any():
        _, (iy, ix) = ndimage.distance_transform_edt(~keep, return_indices=True)
        lbl = np.where(small & mask, lbl[iy, ix], lbl)
    return lbl


def take_apart(C0, rng):
    """Per area: materials (flat base colours) and their shades. Returns per-pixel flat (F1) and shaded (F2) colour
    maps and the list of regions to paint: (pass, area, name, mask, colour)."""
    lab = rgb2lab(ndimage.gaussian_filter(C0, (1.2, 1.2, 0)))
    F1, F2 = C0.copy(), C0.copy()
    flats, shades = [], []
    for area, (k1, k2) in CLUSTERS.items():
        mask = AREAS[area]
        if mask.sum() < 200:
            continue
        pts = lab[mask] * [.35, 1, 1]                     # materials: hue and chroma first
        sub = pts[rng.permutation(len(pts))[:60000]]
        cen, _ = kmeans(sub, k1, rng)
        mat = np.full((H, W), -1)
        mat[mask] = np.argmin(((pts[:, None] - cen[None]) ** 2).sum(-1), 1)
        mat = merge_small(majority(mat, mask, k1), mask, 90)
        for m in range(k1):
            mm = (mat == m) & mask
            if mm.sum() < 60:
                continue
            p2 = lab[mm]
            cen2, lb2 = kmeans(p2[rng.permutation(len(p2))[:40000]], k2, rng)
            sh = np.full((H, W), -1)
            sh[mm] = np.argmin(((p2[:, None] - cen2[None]) ** 2).sum(-1), 1)
            sh = merge_small(majority(sh, mm, k2), mm, 50)
            counts = [int(((sh == j) & mm).sum()) for j in range(k2)]
            base = int(np.argmax(counts))
            base_rgb = np.median(C0[(sh == base) & mm], 0)
            F1[mm] = base_rgb
            flats.append((area, f'{area}{m}', mm, base_rgb))
            base_L = rgb2lab(base_rgb)[0]
            for j in range(k2):
                sj = (sh == j) & mm
                if not sj.any():
                    continue
                rgb = np.median(C0[sj], 0)
                F2[sj] = rgb
                if j != base and sj.sum() >= 30:
                    kind = 'shadow' if rgb2lab(rgb)[0] < base_L else 'light'
                    shades.append((kind, area, f'{area}{m}s{j}', sj, rgb))
    return F1, F2, flats, shades


def soft_clip(mask, grow=1.):
    d_out = ndimage.distance_transform_edt(~mask)
    return np.clip(1 - (d_out - grow) / 1.2, 0, 1).astype(np.float32)


# ----------------------------------------------------------------------------------------------- strokes
def fill(mask, level, clip, rng, tip, *, R, rgb, medium='cel', angle=None, opacity=1., max_len=280., **kw):
    """Fill a region with a brush: back-and-forth strokes along its long axis, held inside the region."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 8:
        return []
    if angle is None:
        cov = np.cov(np.vstack([xs, ys])) if len(xs) > 2 else np.eye(2)
        w, v = np.linalg.eigh(cov)
        t = v[:, -1]
    else:
        t = np.array([math.cos(angle), math.sin(angle)])
    nrm = np.array([-t[1], t[0]])
    reach = ndimage.binary_dilation(mask, iterations=max(1, int(R * .35)))
    proj_n = xs * nrm[0] + ys * nrm[1]
    proj_t = xs * t[0] + ys * t[1]
    spacing = R * 1.45
    out = []
    back = False
    # start from the side nearer the hand
    offs = np.arange(proj_n.min() + R * .5, proj_n.max() + spacing * .5, spacing)
    if abs(np.dot(np.asarray(tip), nrm) - offs[-1]) < abs(np.dot(np.asarray(tip), nrm) - offs[0]):
        offs = offs[::-1]
    for o in offs:
        u = np.arange(proj_t.min() - R, proj_t.max() + R, 3.)
        pts = o * nrm[None] + u[:, None] * t[None]
        pi = np.clip(pts.astype(int), 0, [W - 1, H - 1])
        ok = (pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H) & reach[pi[:, 1], pi[:, 0]]
        runs, cur = [], []
        for p, k in zip(pts, ok):
            if k:
                cur.append(p)
            elif cur:
                runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)
        if back:
            runs = [r[::-1] for r in runs[::-1]]
        back = not back
        for r in runs:
            r = np.asarray(r)
            if len(r) < 2:
                r = np.vstack([r, r + t * 1.5])
            n = max(1, int(math.ceil(len(r) * 3 / max_len)))
            for piece in np.array_split(r, n):
                if len(piece) < 2:
                    continue
                if len(piece) > 6:
                    piece = piece[np.r_[0, np.arange(3, len(piece) - 1, 3), len(piece) - 1]]
                out.append(lay(stroke(piece, R * rng.uniform(.95, 1.05), medium, rng, rgb=rgb, level=level, clip=clip,
                                      opacity=opacity, **kw)))
    return out


def touch_up(level, mask, clip, rng, tip, *, R, rgb, thr=.06, medium='cel', opacity=1.):
    """Anything in the region still off its colour (a corner the brush missed): small fills over each bit."""
    out = []
    for _ in range(2):
        diff = np.abs(SURF.canvas - media.LEVELS[level]).max(-1)
        holes = mask & (diff > thr)
        comp, n = ndimage.label(holes, np.ones((3, 3)))
        if not n:
            break
        sizes = ndimage.sum_labels(np.ones((H, W)), comp, index=np.arange(1, n + 1))
        big = np.flatnonzero(sizes >= 6) + 1
        if not len(big):
            break
        objs = ndimage.find_objects(comp)
        items = []
        for k in big:
            sl = objs[k - 1]
            items.append((np.array([(sl[1].start + sl[1].stop) / 2, (sl[0].start + sl[0].stop) / 2]), k, sl))
        for c, k, sl in nearest_order(items, tip):
            m = np.zeros((H, W), bool)
            m[sl] = comp[sl] == k
            add = fill(m, level, clip, rng, tip, R=R, rgb=rgb, medium=medium, opacity=opacity)
            out += add
            tip = add[-1]['points'][-1] if add else tip
    return out


def airbrush(level, mask, clip, rng, tip, *, R, thr=.04, opacity=.45, step=None):
    """Soft gradients: wide low-flow airbrush strokes wherever the region is still off the illustration."""
    step = step or R * .9
    diff = ndimage.gaussian_filter(np.abs(SURF.canvas - media.LEVELS[level]).max(-1), R * .3)
    active = mask & (diff > thr)
    if active.sum() < 30:
        return []
    ys, xs = np.nonzero(active)
    pts = []
    for gy in np.arange(ys.min(), ys.max() + step, step):
        for gx in np.arange(xs.min(), xs.max() + step, step):
            ix, iy = int(min(W - 1, gx)), int(min(H - 1, gy))
            if active[iy, ix]:
                pts.append(np.array([gx, gy], float))
    out = []
    for (p,) in nearest_order([(p,) for p in pts], tip):
        a = rng.uniform(0, math.pi)
        d = np.array([math.cos(a), math.sin(a)]) * R * rng.uniform(.6, 1.2)
        rgb = media.LEVELS[level][min(H - 1, int(p[1])), min(W - 1, int(p[0]))]
        out.append(lay(stroke([p - d, p, p + d], R, 'air', rng, rgb=rgb, level=level, clip=clip, opacity=opacity)))
    return out


def line_strokes(la, rng):
    """The line art as strokes: skeletons of the marks, each as wide as the mark, in a person's drawing order."""
    from line_topology import extract_strokes
    from line_drawing import order, classify
    ink = la > .3
    raw, *_ = extract_strokes(ink, size=W / 720)
    pixels = np.dstack([np.repeat(((1 - la) * 255)[..., None], 3, 2), np.full((H, W), 255.)]).astype(np.uint8)
    classify(raw, np.zeros((H, W), int), np.full((H, W), 255, np.uint8), pixels, ['head'], ink=ink)
    ordered, _ = order(raw, 1, (W * .55, H * .3))
    half = ndimage.distance_transform_edt(ink)
    out = []
    for st in ordered:
        p = np.asarray(st['points'], float)
        if len(p) < 2:
            continue
        pi = np.clip(p.astype(int), 0, [W - 1, H - 1])
        R = float(np.clip(np.percentile(half[pi[:, 1], pi[:, 0]], 75) + 1.6, 1.6, 6.))
        if len(p) > 6:
            p = p[np.r_[0, np.arange(2, len(p) - 1, 2), len(p) - 1]]
        out.append(stroke(p, R, 'pen', rng, rgb=list(media.LINE_RGB), level='linec', alpha='line', layer='line',
                          press=(.45, .35)))
    return out


# ------------------------------------------------------------------------------------------------- plan
def plan_anime(target_path, rough_path, areas_path, out, focus, eyes, signature, photo_path, lineart_path=None):
    global SURF
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(paint.SEED)
    A, crop, shape = load_subject(target_path)
    Image.fromarray((load_subject(photo_path)[0] * 255).astype(np.uint8)).save(out / 'reference.png')
    Image.fromarray((A * 255).astype(np.uint8)).save(out / 'target.png')
    np.save(out / 'tooth.npy', np.full((H, W), .5, np.float32))
    FEATURES[:] = [tuple(e[:4]) for e in focus if e[4] >= .95]
    EYE_FEATURES.clear()
    for ex, ey, *_ in eyes:
        around = [k for k, f in enumerate(FEATURES) if ((ex - f[0]) / f[2]) ** 2 + ((ey - f[1]) / f[3]) ** 2 <= 1]
        if around:
            EYE_FEATURES.add(min(around, key=lambda k: FEATURES[k][2] * FEATURES[k][3]))
    planes.area_masks_from_map(areas_path, A, rng)
    la_own = line_art(A)
    C0 = under_lines(A, la_own)                   # the colours under the illustration's own lines
    linec = np.clip((A - C0 * (1 - la_own[..., None])) / np.maximum(la_own, .3)[..., None], 0, 1)
    linec = np.where(la_own[..., None] > .05, linec, media.LINE_RGB)
    if lineart_path:
        la, sa, slc = with_clean_lines(A, la_own, lineart_path, eyes)
        linec = np.where((sa >= la - 1e-6)[..., None] & (sa > .02)[..., None], slc, linec)
    else:
        la = la_own
    F1, F2, flats, shades = take_apart(C0, rng)
    media.LEVELS.update(A=A, C0=C0, F1=F1, F2=F2, linec=linec)
    media.ALPHAS.update(line=la.astype(np.float32))
    for k, v in media.LEVELS.items():
        np.save(out / f'level-{k}.npy', v.astype(np.float16))
    np.save(out / 'alpha-line.npy', la.astype(np.float16))
    Image.fromarray((F1 * 255).astype(np.uint8)).save(out / 'flats.png')
    Image.fromarray((F2 * 255).astype(np.uint8)).save(out / 'shades.png')
    Image.fromarray(((1 - la) * 255).astype(np.uint8)).save(out / 'lineart.png')
    for area in CLUSTERS:
        media.CLIPS[area] = soft_clip(AREAS[area], 1.5)
    for name, mask in [(n, m) for _, n, m, _ in flats] + [(n, m) for _, _, n, m, _ in shades]:
        media.CLIPS[name] = soft_clip(mask, 1.)
    for name, m in media.CLIPS.items():
        Image.fromarray((m * 255).astype(np.uint8)).save(out / f'clip-{name}.png')
    SURF = surf = media.LayerSurface((1., 1., 1.), ('rough', 'line', 'hi'))
    passes = []
    started = time.time()
    state = dict(tip=(W * .5, H * .3))

    def run(name, label, strokes, budget, laid=True, **extra):
        if not laid:
            for st in strokes:
                lay(st)
        if not strokes:
            return
        if passes and passes[-1]['name'] == name:
            passes[-1]['strokes'] += strokes
        else:
            passes.append(dict(name=name, label=label, strokes=strokes, budget=budget, **extra))
        state['tip'] = strokes[-1]['points'][-1]
        Image.fromarray((np.clip(surf.image(), 0, 1) * 255).astype(np.uint8)).save(out / f'pass-{len(passes):02d}-{name}.png')
        print(f'{name:10s} {len(strokes):5d} strokes  {time.time() - started:6.1f}s', flush=True)

    tip = lambda: state['tip']
    # 草稿: the rough, quick and light, on its own layer
    rough = paint.sketch_strokes(rough_path, shape, crop, rng)
    for st in rough:
        st.update(medium='pen', layer='rough', rgb=ROUGH_RGB, mix_rgb=ROUGH_RGB, opacity=.55, R=1.5, press=(.5, .4))
    run('rough', '草稿', rough, 8., laid=False)
    # 线稿: the clean line art over it; the eyes and the features' lines at life pace
    lines = line_strokes(la, rng)
    for st in lines:
        k = paint.tag_feature(st)
        st['focal'] = k >= 0
    run('lines', '线稿', lines, 22., laid=False)
    surf.shown['rough'] = 0.
    # 平涂: flat colours under the line art (the rough is hidden first)
    order = ['skin', 'hair', 'clothes', 'bg']
    flat_strokes = []
    for area in order:
        regs = [f for f in flats if f[0] == area]
        regs.sort(key=lambda f: -int(f[2].sum()))
        for _, name, mask, rgb in regs:
            R = float(np.clip(math.sqrt(mask.sum()) / 9, 4, 30))
            angle = math.pi / 2 if area == 'hair' else None
            add = fill(mask, 'F1', name, rng, tip(), R=R, rgb=rgb, angle=angle)
            add += touch_up('F1', mask, name, rng, add[-1]['points'][-1] if add else tip(), R=max(3., R * .35), rgb=rgb)
            flat_strokes += add
            if add:
                state['tip'] = add[-1]['points'][-1]
    run('flat', '平涂', flat_strokes, 20., hide=['rough'])
    # 阴影 / 高光: cel shapes of each material's shades
    for kind, pname, label, budget in (('shadow', 'shadow', '阴影', 16.), ('light', 'light', '高光', 10.)):
        add_all = []
        items = [s for s in shades if s[0] == kind]
        items.sort(key=lambda s: order.index(s[1]))
        for _, area, name, mask, rgb in items:
            comp, n = ndimage.label(mask)
            objs = ndimage.find_objects(comp)
            parts = []
            for k in range(1, n + 1):
                sl = objs[k - 1]
                m = np.zeros((H, W), bool)
                m[sl] = comp[sl] == k
                if m.sum() >= 12:
                    parts.append((np.array([(sl[1].start + sl[1].stop) / 2, (sl[0].start + sl[0].stop) / 2]), m))
            for c, m in nearest_order(parts, tip()):
                R = float(np.clip(math.sqrt(m.sum()) / 6, 2.5, 16))
                add = fill(m, 'F2', name, rng, tip(), R=R, rgb=rgb)
                add_all += add
                if add:
                    state['tip'] = add[-1]['points'][-1]
        run(pname, label, add_all, budget)
    # 渐变: airbrush gradients toward the illustration's own colours, then a last look over every area
    soft = []
    for area in order:
        soft += airbrush('C0', AREAS[area], area, rng, tip(), R=22., opacity=.5)
        if soft:
            state['tip'] = soft[-1]['points'][-1]
        soft += airbrush('C0', AREAS[area], area, rng, tip(), R=9., opacity=.6, thr=.04)
        if soft:
            state['tip'] = soft[-1]['points'][-1]
        soft += airbrush('C0', AREAS[area], area, rng, tip(), R=5., opacity=.7, thr=.035)
        if soft:
            state['tip'] = soft[-1]['points'][-1]
    run('gradient', '渐变', soft, 14.)
    detail = []
    for area in order:
        detail += touch_up('C0', AREAS[area], area, rng, tip(), R=2.5, rgb=[.5, .5, .5], thr=.06)
        if detail:
            state['tip'] = detail[-1]['points'][-1]
    for st in detail:
        st['rgb'] = [float(c) for c in media.LEVELS['C0'][min(H - 1, max(0, int(st['points'][0][1]))), min(W - 1, max(0, int(st['points'][0][0])))]]
        k = paint.tag_feature(st)
        st['focal'] = False
    run('gradient', '渐变', detail, 0.)
    # 点睛: the eyes' highlights, on a layer above the lines
    touches = []
    for ex, ey, rx, ry in eyes:
        box = ellipse_mask((ex, ey, rx * 1.3, ry * 1.3))
        hi = box & (lum(A) > .86) & (lum(A) - ndimage.gaussian_filter(lum(A), 5) > .12)
        comp, n = ndimage.label(hi)
        for k in range(1, n + 1):
            m = comp == k
            if m.sum() < 4:
                continue
            cy, cx = ndimage.center_of_mass(m)
            r = math.sqrt(m.sum() / math.pi)
            touches.append(lay(stroke([(cx - r * .3, cy), (cx + r * .3, cy + .2)], r + .8, 'cel', rng, rgb=[1, 1, 1],
                                      level='A', layer='hi', focal=True, eye=True, opacity=1.)))
    run('dianjing', '点睛', touches, 0.)
    if signature:
        text, box = signature
        sig = paint.signature_strokes(text, box, (.35, .16, .13), rng)
        for st in sig:
            st.update(medium='pen', layer='line', rgb=[.35, .16, .13], R=1.6, opacity=1.)
        run('sign', '签名', sig, 3., laid=False)
    final = np.clip(surf.image(), 0, 1)
    Image.fromarray((final * 255).astype(np.uint8)).save(out / 'final.png')
    err = np.abs(ndimage.gaussian_filter(final, (2, 2, 0)) - ndimage.gaussian_filter(A, (2, 2, 0))).mean(-1)
    print('mean |colour error| (2px blur):', ' '.join(f'{n} {err[AREAS[n]].mean():.3f}' for n in order))
    print('strokes', sum(len(p['strokes']) for p in passes))
    (out / 'plan.json').write_text(json.dumps(dict(W=W, H=H, kind='anime', title='动漫插画', passes=passes, focus=focus,
                                                   face=list(REGIONS['face']), pace=PACE, ground=[1., 1., 1.],
                                                   layers=['rough', 'line', 'hi']), ensure_ascii=False))
    return passes


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('target')
    ap.add_argument('rough')
    ap.add_argument('areas')
    ap.add_argument('out')
    ap.add_argument('focus')
    ap.add_argument('eyes')
    ap.add_argument('signature', nargs='?', default='-')
    ap.add_argument('--photo', required=True)
    ap.add_argument('--face', required=True)
    ap.add_argument('--crop-top', type=float, default=0.)
    ap.add_argument('--lineart', help='a clean line drawing registered to the illustration (outlines for the 线稿)')
    a = ap.parse_args()
    paint.CROP_TOP = a.crop_top
    REGIONS['face'] = tuple(json.loads(a.face))
    plan_anime(a.target, a.rough, a.areas, a.out, json.loads(a.focus), json.loads(a.eyes),
               None if a.signature == '-' else json.loads(a.signature), a.photo, a.lineart)
