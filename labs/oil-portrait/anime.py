"""动漫插画: paint an anime illustration the way an illustrator does on a tablet, toward an illustration registered to
the photo.

Order: 草稿 construction (head oval, guides, the big hair shape) and a quick rough sketch in light blue on its own
layer -> 线稿 clean line art over it, each line one whole tapered stroke, in a person's order (face, features, hair
outline, strands, clothes; the eyes, iris and pupil included, at life pace), already in the illustration's own line
weight and colour -> the rough layer is hidden -> 平涂 flat base colours under the line art, each region coloured in
with the pen: a careful stroke along its edge, then broad back-and-forth strokes across it (the eyes get their
whites, iris and pupil here too) -> 阴影 / 高光 cel shapes clipped to their base, laid the same way; thin streaks
(locks of hair) are single strokes along the hair -> 渐变 wide airbrush sweeps -> 眼睛 what is added to the eyes
late, in a close-up: iris shading, the lid's shadow, catchlights on top -> 调色 a short last adjustment -> 签名.

Every pixel of colour appears under the nib as it passes: a stroke either is the shape (a line, a streak, a dab),
follows the shape's edge, or colours in a shape whose edge is already there. No fills, nothing appears on its own.

The illustration is taken apart into what an illustrator paints in each step: its line art (thin dark marks, found
with a black-hat filter, plus a clean line drawing of the outlines), the colours under the lines (the lines filled
in from their sides), and in each area (skin, hair, eyes, clothes, background) its materials (a coarse colour
clustering: skin, lips, eye whites, iris...) and each material's hard-edged shades (a finer clustering by lightness;
a shade with a soft edge is a gradient and is left to the airbrush).

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
# nib_px / travel_px: the farthest the nib moves between two film frames, drawing and lifted (film.floor)
PACE = dict(feature_cap=5., eye_life={'lines': 6., 'polish': 4., 'dianjing': 2.5}, nib_px=85., travel_px=84.)
ROUGH_RGB = [.42, .62, .92]
SURF = None
# per area: materials (coarse clusters, chroma first) and shades per material
CLUSTERS = {'skin': (4, 3), 'hair': (1, 5),'eyes': (3, 3), 'clothes': (3, 3), 'bg': (3, 3)}
ORDER = ['eyes', 'skin', 'hair', 'clothes', 'bg']     # the eyes first: once colouring starts they are never bare
HOLD = 1 / 30.        # s of film a colour stroke (or a group of them laid in one motion) is on screen at least
FLAT_MIN = 300        # px: a smaller patch of flat colour joins the shape beside it
SHADE_MIN = 120       # px: a smaller cel shape is left to the last adjustment
MOST = {'shadow': 40, 'light': 42}   # the cel shapes painted in a pass: the largest ones
THIN = 5.5           # px: a shape no deeper than this is one stroke along its middle
SOFT_EDGE = .03       # a shade whose edge in the illustration is softer than this is a gradient, not a cel shape
SLIDER = ((300., 1680.), (740., 1680.))   # the 调色 slider of the colour panel, in canvas coordinates (below the canvas)


def lum(rgb):
    return rgb @ LUM


def stroke(points, R, medium, rng, **kw):
    rgb = kw.pop('rgb', [.5, .5, .5])
    d = dict(points=[(float(x), float(y)) for x, y in points], R=float(R), medium=medium, seed=int(rng.integers(1 << 30)),
             rgb=[float(c) for c in rgb], mix=0, mix_rgb=[float(c) for c in rgb], press=(.7, .6), pvar=.03, wobble=0)
    if medium in ('cel', 'air'):
        d['hold'] = HOLD                  # colour is laid with the hand on it: at least a frame on screen
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


def thin(p, step):
    """Keep about one point every step px of a dense path (both ends stay)."""
    p = np.asarray(p, float)
    if len(p) < 3:
        return p
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    n = max(2, int(s[-1] / step) + 1)
    idx = np.unique(np.searchsorted(s, np.linspace(0, s[-1], n)).clip(0, len(p) - 1))
    return p[idx]


def length(p):
    p = np.asarray(p, float)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


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


def with_clean_lines(A, la_own, lineart_path, areas, eyes):
    """The line art an illustrator would have drawn: a clean line drawing registered to the illustration (made by
    the image model from it) for the figure (face, hair outline and strands, eyelids, mouth, clothes), and the
    illustration's own thin marks in the irises (the drawing fills irises solid) and wherever the drawing has no line
    near. Inside the hair the illustration's own dark marks are not lines (gaps between highlights, dark locks): they
    stay in the colours. The background has no line art: it is painted, and soft. Returns the line layer's coverage
    and the own marks that are part of it."""
    S = load_subject(lineart_path)[0]
    sa = np.clip((.82 - ndimage.gaussian_filter(lum(S), .4)) / .5, 0, 1)
    iris = np.zeros((H, W), bool)
    for ex, ey, rx, ry in eyes:
        iris |= ellipse_mask((ex, ey, rx * 1.15, ry * 1.15))
    figure = ndimage.binary_dilation(areas['person'], iterations=4)
    sa = np.where(iris | ~figure, 0., sa)
    near = ndimage.distance_transform_edt(~(sa > .35)) <= 3
    inside_hair = ndimage.binary_erosion(areas['hair'], iterations=5)
    own = np.where(inside_hair | ~figure, 0., la_own)
    return np.maximum(sa, np.where(near, 0., own)), own


def line_look(A, C0, la_geom, own):
    """What the line layer looks like once drawn: where the illustration has a line of its own, that line (its
    weight, its colour); where only the clean drawing has one (a contour that in the illustration is just two colours
    meeting, a strand inside the hair), a thin line already tinted toward the colour under it, as an illustrator
    colours line art. So the line art is right while it is being drawn and the last adjustment has little to mend.
    Returns coverage and colour."""
    thin_ = np.clip(ndimage.gaussian_filter(la_geom, .5) * 1.6 - .6, 0, 1) * .75    # the drawing's line, slimmer and lighter
    far = ndimage.distance_transform_edt(~(la_geom > .35)) > 3     # the illustration's marks the drawing has no line for
    cover = np.maximum(thin_, np.where(far, own, 0.))
    own_rgb = ndimage.minimum_filter(A, size=(5, 5, 1))            # a line's colour is the darkest of the pixels around
    tint = ndimage.gaussian_filter(C0, (1.5, 1.5, 0)) * .42
    w = np.clip(ndimage.maximum_filter(own, 7) * 1.5, 0, 1)[..., None]   # beside one of its own lines: that line's colour
    return cover.astype(np.float32), own_rgb * w + tint * (1 - w)


def iris_rings(F1, areas, eyes):
    """The outlines of iris and pupil, from the eye's flat colours (everything but the white; the darkest inside it),
    smoothed into the round shapes a pen would draw. They belong to the line art: an eye is never left an empty
    socket."""
    v = lum(F1)
    out = np.zeros((H, W), bool)
    inner = ndimage.binary_erosion(areas['eyes'], iterations=2)
    for ex, ey, rx, ry in eyes:
        near = ellipse_mask((ex, ey, rx * 1.25, ry * 1.25)) & areas['eyes']
        if near.sum() < 30:
            continue
        iris = near & (v < .5 * (v[near].max() + np.median(v[near])))
        pupil = ellipse_mask((ex, ey, rx * .5, ry * .5)) & iris & (v <= v[iris].min() + .04) if iris.any() else iris
        for m, sig in ((iris, 4.), (pupil, 2.5)):
            m = ndimage.binary_fill_holes(ndimage.gaussian_filter(m.astype(np.float32), sig) > .5)
            out |= m & ~ndimage.binary_erosion(m, iterations=2) & inner
    return out


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


def eye_area(C0, eyes):
    """What is painted as 'the eyes': inside the eye features, the not skin-coloured shape around each iris (white,
    iris, pupil, the lash line), so the skin's flood leaves eye-shaped holes as it would inside the eyes' line art.
    The eyes are painted last, in a close-up."""
    lab = rgb2lab(ndimage.gaussian_filter(C0, (1., 1., 0)))
    base = np.median(lab[AREAS['skin']], 0)
    m = np.zeros((H, W), bool)
    for k in EYE_FEATURES:
        m |= ellipse_mask(FEATURES[k], 1.12)
    m &= AREAS['skin'] & (np.linalg.norm(lab - base, axis=-1) > 15)
    m = ndimage.binary_opening(ndimage.binary_closing(m, iterations=2), iterations=1)
    comp, n = ndimage.label(m)
    iris = np.zeros((H, W), bool)
    for ex, ey, rx, ry in eyes:
        iris |= ellipse_mask((ex, ey, rx * .6, ry * .6))
    keep = np.unique(comp[iris & m])
    return np.isin(comp, keep[keep > 0])


def take_apart(C0, areas, rng):
    """Per area: materials (flat base colours) and their hard-edged shades, each split into connected shapes.
    Returns the flat (F1) and shaded (F2) colour maps, the shape lists, and the two label maps the shapes live in."""
    lab = rgb2lab(ndimage.gaussian_filter(C0, (1.2, 1.2, 0)))
    sm = ndimage.gaussian_filter(C0, (.8, .8, 0))
    G = np.max(np.hypot(*np.gradient(sm, axis=(0, 1))), -1)       # how hard an edge is, per pixel
    G = ndimage.maximum_filter(G, 3)
    F1, F2 = C0.copy(), C0.copy()
    flat_lab, shade_lab = np.zeros((H, W), np.int32), np.zeros((H, W), np.int32)
    flats, shades = [], []
    soft_n = 0

    def shapes_of(mask, min_px):
        comp, n = ndimage.label(mask)
        objs = ndimage.find_objects(comp)
        sizes = ndimage.sum_labels(np.ones((H, W)), comp, index=np.arange(1, n + 1)) if n else []
        for k in range(n):
            if sizes[k] >= min_px:
                sl = objs[k]
                yield comp[sl] == k + 1, sl, int(sizes[k])

    for area, (k1, k2) in CLUSTERS.items():
        mask = areas[area]
        if mask.sum() < 60:
            continue
        pts = lab[mask] * ([.8, 1, 1] if area == 'eyes' else [.35, 1, 1])   # materials: hue and chroma first
        sub = pts[rng.permutation(len(pts))[:60000]]
        cen, _ = kmeans(sub, k1, rng)
        mat = np.full((H, W), -1)
        mat[mask] = np.argmin(((pts[:, None] - cen[None]) ** 2).sum(-1), 1)
        mat = merge_small(majority(mat, mask, k1, 3 if area == 'eyes' else 5), mask, 30 if area == 'eyes' else 90)
        if area != 'eyes':
            # a patch of another colour with a soft edge all round (blush, a glow) is not a material with its own flat:
            # it joins the material around it and is airbrushed later
            main = int(np.argmax([(mat == m).sum() for m in range(k1)]))
            limit = SOFT_EDGE * (1.7 if area == 'skin' else 1.)
            for m in range(k1):
                if m == main:
                    continue
                comp, n = ndimage.label(mat == m)
                for k, sl in enumerate(ndimage.find_objects(comp)):
                    psl = (slice(max(0, sl[0].start - 1), min(H, sl[0].stop + 1)), slice(max(0, sl[1].start - 1), min(W, sl[1].stop + 1)))
                    sub_m = comp[psl] == k + 1
                    ring = ndimage.binary_dilation(sub_m) & ~sub_m & mask[psl] & (mat[psl] != m)
                    if ring.sum() >= 6 and np.median(G[psl][ring]) < limit:
                        mat[psl][sub_m] = np.bincount(mat[psl][ring]).argmax()
        for m in range(k1):
            mm = (mat == m) & mask
            if mm.sum() < 30:
                continue
            p2 = lab[mm]
            cen2, lb2 = kmeans(p2[rng.permutation(len(p2))[:40000]], k2, rng)
            sh = np.full((H, W), -1)
            sh[mm] = np.argmin(((p2[:, None] - cen2[None]) ** 2).sum(-1), 1)
            sh = merge_small(majority(sh, mm, k2, 3 if area == 'eyes' else 5), mm, 16 if area == 'eyes' else 50)
            counts = [int(((sh == j) & mm).sum()) for j in range(k2)]
            base = int(np.argmax(counts))
            base_rgb = np.median(C0[(sh == base) & mm], 0)
            F1[mm] = base_rgb
            F2[mm] = base_rgb
            for sub_m, sl, n in shapes_of(mm, 20 if area == 'eyes' else FLAT_MIN):
                k = len(flats) + 1
                flat_lab[sl][sub_m] = k
                flats.append(dict(id=k, area=area, box=[sl[0].start, sl[0].stop, sl[1].start, sl[1].stop], n=n,
                                  rgb=base_rgb, lab='flat'))
            base_L = rgb2lab(base_rgb)[0]
            for j in range(k2):
                sj = (sh == j) & mm
                if j == base or not sj.any():
                    continue
                rgb = np.median(C0[sj], 0)
                kind = 'shadow' if rgb2lab(rgb)[0] < base_L else 'light'
                for sub_m, sl, n in shapes_of(sj, 14 if area == 'eyes' else SHADE_MIN):
                    ys, xs = np.nonzero(sub_m)
                    bid = int(flat_lab[sl][ys[0], xs[0]])
                    # its free edge (against its own base, not the base's outline): hard in the illustration, or soft?
                    ya, yb = max(0, sl[0].start - 1), min(H, sl[0].stop + 1)
                    xa, xb = max(0, sl[1].start - 1), min(W, sl[1].stop + 1)
                    full = np.zeros((yb - ya, xb - xa), bool)
                    full[sl[0].start - ya:sl[0].stop - ya, sl[1].start - xa:sl[1].stop - xa] = sub_m
                    ring = ndimage.binary_dilation(full) & ~full & (flat_lab[ya:yb, xa:xb] == bid) & (sh[ya:yb, xa:xb] == base)
                    hard = float(np.median(G[ya:yb, xa:xb][ring])) if ring.sum() >= 6 else 1.
                    if hard < SOFT_EDGE * (1.7 if area == 'skin' else 1.) and area != 'eyes':   # blush is airbrushed
                        soft_n += 1
                        continue
                    k = len(shades) + 1
                    shade_lab[sl][sub_m] = k
                    F2[sl][sub_m] = rgb
                    shades.append(dict(id=k, kind=kind, area=area, box=[sl[0].start, sl[0].stop, sl[1].start, sl[1].stop],
                                       n=n, rgb=rgb, base=bid, lab='shade', pt=(int(xs[0] + sl[1].start), int(ys[0] + sl[0].start))))
    stray = flat_lab == 0                     # specks too small to be a shape belong to the shape beside them
    if stray.any():
        _, (iy, ix) = ndimage.distance_transform_edt(stray, return_indices=True)
        flat_lab[stray] = flat_lab[iy, ix][stray]
        F1[stray] = F1[iy, ix][stray]
        F2[stray] = F2[iy, ix][stray]
        for f, sl in zip(flats, ndimage.find_objects(flat_lab)):
            f['box'] = [sl[0].start, sl[0].stop, sl[1].start, sl[1].stop]
    for s in shades:                          # a shade on a patch that joined its neighbour is clipped to that neighbour
        s['base'] = int(flat_lab[s['pt'][1], s['pt'][0]])
    print(f'flat shapes {len(flats)}, cel shapes {len(shades)} (+{soft_n} soft ones left to the airbrush)')
    return F1, F2, flats, shades, flat_lab, shade_lab


def soft_clip(mask, grow=1.):
    d_out = ndimage.distance_transform_edt(~mask)
    return np.clip(1 - (d_out - grow) / 1.2, 0, 1).astype(np.float32)


# ----------------------------------------------------------------------------------------------- shapes
def crop(lab, sh, pad=3):
    y0, y1, x0, x1 = sh['box']
    ya, yb, xa, xb = max(0, y0 - pad), min(H, y1 + pad), max(0, x0 - pad), min(W, x1 + pad)
    return lab[ya:yb, xa:xb] == sh['id'], ya, xa


def traced(mask):
    """Whole strokes along the middle of the marks in a (cropped) mask: the line-first engine's skeleton tracing."""
    from line_topology import extract_strokes
    if mask.sum() < 4:
        return []
    try:
        raw, *_ = extract_strokes(np.pad(mask, 2), size=W / 720)
    except Exception:
        return []
    return [np.asarray(st['points'], float) - 2 for st in raw if len(st['points']) >= 2]


def edge_runs(m, depth, free):
    """Paths along the inside of a shape's edge (depth px in), only where the edge is free (free: the pixels just
    outside it that the edge borders on), as runs long enough to be a stroke."""
    if m.all():
        return []
    d, (iy, ix) = ndimage.distance_transform_edt(m, return_indices=True)
    ring = (d >= depth) & (d < depth + 1.3)
    out = []
    for p in traced(ring):
        pi = np.clip(np.rint(p).astype(int), 0, [m.shape[1] - 1, m.shape[0] - 1])
        f = free[iy[pi[:, 1], pi[:, 0]], ix[pi[:, 1], pi[:, 0]]].astype(float)
        f = ndimage.uniform_filter1d(f, 7, mode='nearest') > .5
        start = None
        for i, v in enumerate(np.r_[f, False]):
            if v and start is None:
                start = i
            elif not v and start is not None:
                if length(p[start:i]) >= max(10., 3 * depth):
                    out.append(p[start:i])
                start = None
    return out


def joined(paths, tip):
    """Several paths as one motion: nearest first, each entered at its nearer end."""
    left, out, tip = [np.asarray(p, float) for p in paths], [], np.asarray(tip, float)
    while left:
        k = int(np.argmin([min(np.linalg.norm(p[0] - tip), np.linalg.norm(p[-1] - tip)) for p in left]))
        p = left.pop(k)
        if np.linalg.norm(p[-1] - tip) < np.linalg.norm(p[0] - tip):
            p = p[::-1]
        out.append(p)
        tip = p[-1]
    return np.vstack(out)


def cover_paths(m, R, tip, angle=None, bow=0., max_rows=None, top_first=False, gap=1.25):
    """Colouring in a shape with a brush of radius R: rows a brush width apart (along the shape's long axis, or at
    angle; bowed a little for a round form), joined into as few continuous back-and-forth motions as the shape
    allows: where a row is cut in two (hair either side of a face) each part carries on as its own motion, and a
    motion may be ended after max_rows rows (a lock of hair). Rows run to the shape's very ends, so the brush laps
    over its edge and the clip trims it. A tip or corner the rows miss is a short detour of the motion passing
    nearest; only a part too big for that (over 1.5% of the shape) gets a motion of its own. Returns polylines, in
    crop coordinates."""
    ys, xs = np.nonzero(m)
    if angle is None:
        cov = np.cov(np.vstack([xs, ys])) if len(xs) > 2 else np.eye(2)
        t = np.linalg.eigh(cov + np.eye(2) * 1e-6)[1][:, -1]
    else:
        t = np.array([math.cos(angle), math.sin(angle)])
    if top_first and t[1] < 0:
        t = -t
    nrm = np.array([-t[1], t[0]])
    pn, pt = xs * nrm[0] + ys * nrm[1], xs * t[0] + ys * t[1]
    width = pn.max() - pn.min()
    offs = np.arange(pn.min() + min(R * .6, width / 2), pn.max() - R * .2 + 1e-6, R * gap)
    if not len(offs):
        offs = np.array([(pn.min() + pn.max()) / 2])
    if abs(np.dot(tip, nrm) - offs[-1]) < abs(np.dot(tip, nrm) - offs[0]):
        offs = offs[::-1]
    chains, open_ = [], []
    for o in offs:
        band = np.abs(pn - o) < .5
        if band.sum() < 2:
            band = np.abs(pn - o) < 1.
        u = np.sort(pt[band])
        runs = [(r[0], r[-1]) for r in np.split(u, np.flatnonzero(np.diff(u) > max(3., R)) + 1) if len(r)] if len(u) else []
        used, still = set(), []
        for ch in open_:
            lo0, hi0 = ch[-1][1:]
            best, most = None, 0.
            for j, (lo, hi) in enumerate(runs):
                ov = min(hi, hi0) - max(lo, lo0)
                if j not in used and ov > most:
                    best, most = j, ov
            if best is not None and (max_rows is None or len(ch) < max_rows):
                ch.append((o,) + runs[best])
                used.add(best)
                still.append(ch)
        for j, r in enumerate(runs):
            if j not in used:
                chains.append([(o,) + r])
                still.append(chains[-1])
        open_ = still
    paths = []
    for ch in chains:
        pts = []
        lo, hi = ch[0][1:]
        back = False if top_first else abs(np.dot(tip, t) - hi) < abs(np.dot(tip, t) - lo)
        for o, lo, hi in ch:
            a, b = (hi, lo) if back else (lo, hi)
            n = max(2, int(abs(b - a) / 30) + 2)
            arc = bow * abs(b - a) * np.sin(np.pi * np.linspace(0, 1, n))
            pts += [(o + k) * nrm + uu * t for uu, k in zip(np.linspace(a, b, n), arc)]
            back = not back
        paths.append(np.array(pts))
    # the motions worth making: long enough to be a sweep; the rest are corners for a detour
    total = lambda p: float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    main = [p for p in paths if total(p) >= 3 * R] or [max(paths, key=total)]
    from PIL import ImageDraw
    img = Image.new('L', (m.shape[1], m.shape[0]), 0)
    dr = ImageDraw.Draw(img)
    for p in main:
        dr.line([tuple(q) for q in p] if len(p) > 1 else [tuple(p[0]), tuple(p[0] + .5)], fill=255, width=1)
    bare = m & (ndimage.distance_transform_edt(np.asarray(img) == 0) > R * .95)
    comp, n = ndimage.label(bare)
    if n:
        sizes = ndimage.sum_labels(np.ones(m.shape), comp, index=np.arange(1, n + 1))
        floor_px = max(12., .002 * m.sum())
        for k in np.argsort(-sizes)[:14]:
            if sizes[k] < floor_px:
                break
            cy, cx = np.nonzero(comp == k + 1)
            P = np.c_[cx, cy].astype(float)
            if sizes[k] > .015 * m.sum() and sizes[k] > 6 * R * R:      # a part of its own: its own (smaller) motion
                part = comp == k + 1
                deep = float(ndimage.distance_transform_edt(part).max())
                main += cover_paths(part, float(np.clip(deep * .9 + 1., 2.5, R)), main[-1][-1], angle, bow, max_rows, top_first)[:2]
                continue
            ax = np.linalg.eigh(np.cov(P.T) + np.eye(2) * 1e-6)[1][:, -1] if len(P) > 2 else np.array([1., 0.])
            pr = (P - P.mean(0)) @ ax
            ends = [P[int(np.argmin(pr))], P[int(np.argmax(pr))]]
            # the vertex of a motion that passes nearest: detour from there to the corner and on
            best = min(((float(np.linalg.norm(p - P.mean(0), axis=1).min()), i) for i, p in enumerate(main)))[1]
            p = main[best]
            j = int(np.argmin(np.linalg.norm(p - P.mean(0), axis=1)))
            if np.linalg.norm(ends[1] - p[j]) < np.linalg.norm(ends[0] - p[j]):
                ends = ends[::-1]
            main[best] = np.vstack([p[:j + 1], ends[0], ends[1], p[j:]])
    return main


def paint_shape(sh, labmap, rng, tip, *, free=None, edge=True, angle=None, bow=0., max_R=64., wide=.65, max_rows=None,
                top_first=False, extra=None):
    """One shape, coloured in with the pen in one or two motions, every pixel under the nib: a thin one (or a long
    narrow lock) as one stroke along its middle, a dab if tiny; otherwise, for a shape big enough to need it, a
    careful stroke along its edge just inside it (free: only where the edge is not the edge of the base it is clipped
    to), then the back-and-forth of cover_paths across the inside."""
    extra = extra or {}
    m, ya, xa = crop(labmap, sh)
    if m.sum() < 4:
        return []
    off = np.array([xa, ya], float)
    clip = [sh['lab'], sh['id']]
    rgb = sh['rgb']
    d = ndimage.distance_transform_edt(np.pad(m, 1))[1:-1, 1:-1] if m.all() else ndimage.distance_transform_edt(m)
    maxd = float(d.max())
    out = []
    tip_l = np.asarray(tip, float) - off

    def add(points, R, **kw):
        out.append(lay(stroke(np.asarray(points, float) + off, R, 'cel', rng, rgb=rgb, clip=clip, **{**extra, **kw})))

    streak = maxd <= 2 * THIN and sh['n'] > 14 * maxd * maxd      # long and narrow (a lock of hair): one pressed stroke
    if maxd <= THIN or streak:
        paths = [p for p in traced(m) if length(p) >= 3]
        if not paths:
            ys, xs = np.nonzero(m)
            r = math.sqrt(sh['n'] / math.pi)
            c = np.array([xs.mean(), ys.mean()])
            add([c - [r * .3, 0], c + [r * .3, 0]], r + 1.2, press=(.8, .7), role='dab')
            return out
        p = joined(paths, tip_l)
        pi = np.clip(np.rint(p).astype(int), 0, [m.shape[1] - 1, m.shape[0] - 1])
        R = float(np.percentile(d[pi[:, 1], pi[:, 0]], 90)) + 1.6
        add(thin(p, max(4., 2 * R)), R, press=(.45, .35), pvar=.08, role='streak')
        return out
    Rf = float(np.clip(maxd * wide, 3., max_R))                       # the brush that fills
    if edge and sh['n'] >= 1500:
        Re = float(np.clip(Rf * .4, 2.6, 12.))                        # the smaller one that takes the edge
        em = free[ya:ya + m.shape[0], xa:xa + m.shape[1]] if free is not None else np.ones(m.shape, bool)
        runs = edge_runs(m, Re * .8, em)
        if runs:
            p = joined(runs, tip_l)
            add(thin(p, max(4., 1.6 * Re)), Re * 1.15, press=(.7, .6), pvar=.06, wobble=1, role='edge')
            tip_l = p[-1]
    for p in cover_paths(m, Rf, tip_l, angle, bow, max_rows, top_first):
        if len(p) < 2:
            p = np.vstack([p, p + [.6, 0]])
        add(p, Rf, press=(.9, .85), wobble=1, pvar=.04, role='fill')
        tip_l = p[-1]
    return out


def sweeps(level, mask, clip, rng, tip, *, R, thr, opacity, angle, most=4, **kw):
    """Soft gradients: wherever the area is still off the illustration, the airbrush goes over it in one long
    back-and-forth motion per patch (the largest few patches)."""
    diff = ndimage.gaussian_filter(np.abs(SURF.canvas - media.LEVELS[level]).max(-1), R * .3)
    active = mask & (diff > thr)
    if active.sum() < 40:
        return []
    grow = max(1, int(R * .5))                 # one sweep over a patchy stretch, not a dab on every patch
    active = ndimage.binary_erosion(ndimage.binary_dilation(active, iterations=2 * grow), iterations=grow)
    active &= ndimage.binary_dilation(mask, iterations=2)
    comp, n = ndimage.label(active)
    sizes = ndimage.sum_labels(np.ones((H, W)), comp, index=np.arange(1, n + 1))
    objs = ndimage.find_objects(comp)
    items = [(np.array([(objs[k][1].start + objs[k][1].stop) / 2, (objs[k][0].start + objs[k][0].stop) / 2]), k)
             for k in np.argsort(-sizes)[:most] if sizes[k] >= 2 * R * R]
    out = []
    tip = np.asarray(tip, float)
    for c, k in nearest_order(items, tip):
        sl = objs[k]
        off = np.array([sl[1].start, sl[0].start], float)
        for p in cover_paths(comp[sl] == k + 1, R, tip - off, angle, gap=1.8)[:2]:
            if len(p) < 2:
                continue
            p = p + off
            mid = p[len(p) // 2]
            rgb = media.LEVELS[level][int(np.clip(mid[1], 0, H - 1)), int(np.clip(mid[0], 0, W - 1))]
            out.append(lay(stroke(p, R, 'air', rng, rgb=rgb, level=level, clip=clip, opacity=opacity, wobble=1, **kw)))
            tip = p[-1]
    return out


# ------------------------------------------------------------------------------------------------- lines
def construction(areas, eyes, rng):
    """The first marks of a rough: the head as an oval gone round twice, its centre line and eye line, ticks for
    nose and mouth, the big shape of the hair and the shoulders as a few loose curves."""
    out = []

    def loose(points, opacity=.38, R=1.9):
        out.append(stroke(points, R, 'pen', rng, rgb=ROUGH_RGB, layer='rough', opacity=opacity, press=(.4, .3), pvar=.25,
                          wobble=1, seconds=.2 + length(points) / 1100))      # construction is drawn fast, from the arm

    face = areas['skin'] & ellipse_mask(REGIONS['face'])
    ys, xs = np.nonzero(face)
    if len(xs) > 50:
        cen = np.array([xs.mean(), ys.mean()])
        ev, evec = np.linalg.eigh(np.cov(np.vstack([xs, ys])))
        major, minor = evec[:, 1] * (1 if evec[1, 1] > 0 else -1), evec[:, 0]
        ra, rb = 2.15 * math.sqrt(ev[1]), 2.15 * math.sqrt(ev[0])
        th = np.linspace(-.6, 4 * math.pi - .9, 44)
        grow = np.linspace(.97, 1.04, len(th))
        loose([cen + major * ra * math.cos(a) * g + minor * rb * math.sin(a) * g for a, g in zip(th, grow)])
        loose([cen - major * ra * 1.12, cen, cen + major * ra * 1.15], .3)                  # centre line
        if len(eyes) >= 2:
            e0, e1 = (np.array([e[0] * W, e[1] * H]) for e in eyes[:2])
            mid, dv = (e0 + e1) / 2, (e1 - e0)
            loose([mid - dv * 1.25, mid, mid + dv * 1.25], .3)                               # eye line
            dv = dv / (np.linalg.norm(dv) + 1e-9)
            for k, f in enumerate(FEATURES):
                if k not in EYE_FEATURES:
                    c = np.array([f[0] * W, f[1] * H])
                    loose([c - dv * f[2] * W * .9, c + dv * f[2] * W * .9], .3)              # nose, mouth ticks
    for name, sig in (('person', 20.), ('hair', 16.)):
        m = ndimage.gaussian_filter(areas[name].astype(np.float32), sig) > .5
        ring = m & ~ndimage.binary_erosion(m)
        paths = sorted(traced(ring), key=lambda p: -length(p))[:4]
        for p in paths:
            if length(p) < 160:
                continue
            if p[-1][1] < p[0][1]:
                p = p[::-1]
            for piece in np.array_split(p, max(1, int(length(p) / 520))):
                if len(piece) > 8:
                    loose(thin(piece, 46) + rng.normal(0, 2.2, (len(thin(piece, 46)), 2)), .34)
    return out


def ink_lines(la, areas, rng):
    """The line art as whole strokes in a person's order: the face's outline, the eyes (lash line first), nose and
    mouth, the hair's outline, its strands from left to right, the clothes, the background. Each stroke is as wide as
    its line, pulled downward or to the right, thin where it enters and leaves free of other lines."""
    from line_topology import extract_strokes
    ink = la > .3
    raw, *_ = extract_strokes(ink, size=W / 720)
    half = ndimage.distance_transform_edt(ink)
    skin = ndimage.binary_dilation(areas['skin'] | areas['eyes'], iterations=4)
    hair_edge = areas['hair'] & ~ndimage.binary_erosion(areas['hair'], iterations=8)
    groups = {}
    for st in raw:
        p = np.asarray(st['points'], float)
        if len(p) < 2:
            continue
        L = length(p)
        mid = p[len(p) // 2]
        k = paint.feature_index(*mid)
        if L < (3.5 if k >= 0 else 30.):
            continue
        pi = np.clip(np.rint(p).astype(int), 0, [W - 1, H - 1])
        frac = lambda m: float(m[pi[:, 1], pi[:, 0]].mean())
        if k >= 0:
            g = ('feat', k)
        elif frac(skin) > .5:
            g = ('face',)
        elif frac(areas['hair']) > .5:
            g = ('hairline',) if frac(hair_edge) > .55 else ('strand',)
        elif frac(areas['clothes']) > .5:
            g = ('clothes',)
        else:
            g = ('bg',)
        dx, dy = p[-1] - p[0]
        if (abs(dy) >= abs(dx) and dy < 0) or (abs(dy) < abs(dx) and dx < 0):
            p = p[::-1]
            pi = pi[::-1]
        R = float(np.clip(np.percentile(half[pi[:, 1], pi[:, 0]], 80) + 1.5, 1.7, 6.))
        ends = []
        for e, sgn in ((0, -1), (-1, 1)):                # an end running into another line keeps its full weight
            tan = p[e] - p[min(len(p) - 1, 4) if e == 0 else max(0, len(p) - 5)]
            tan = tan / (np.linalg.norm(tan) + 1e-9)
            q = np.rint(p[e] + tan * (R + 3)).astype(int)
            joined = 0 <= q[0] < W and 0 <= q[1] < H and ink[max(0, q[1] - 1):q[1] + 2, max(0, q[0] - 1):q[0] + 2].any()
            ends.append(1. if joined else .3)
        groups.setdefault(g, []).append(dict(p=p, L=L, R=R, press=tuple(ends)))

    def longest_then_near(items, tip):
        if not items:
            return []
        first = max(items, key=lambda it: it['L'])
        rest = [it for it in items if it is not first]
        return [first] + [it for (it,) in nearest_order([(it,) for it in rest], first['p'][-1], key=lambda t: t[0]['p'][0])]

    seq = []
    seq += [(it, False, False) for it in longest_then_near(groups.get(('face',), []), (W * .5, H * .3))]
    eye_ks = sorted(EYE_FEATURES, key=lambda k: FEATURES[k][0])
    for k in eye_ks + [k for k in range(len(FEATURES)) if k not in EYE_FEATURES]:
        seq += [(it, True, k in EYE_FEATURES) for it in longest_then_near(groups.get(('feat', k), []), None)]
    seq += [(it, False, False) for it in longest_then_near(groups.get(('hairline',), []), None)]
    seq += [(it, False, False) for it in sorted(groups.get(('strand',), []), key=lambda it: it['p'][0][0] + .15 * it['p'][0][1])]
    tip = seq[-1][0]['p'][-1] if seq else (0, 0)
    for g in (('clothes',), ('bg',)):
        its = [it for (it,) in nearest_order([(it,) for it in groups.get(g, [])], tip, key=lambda t: t[0]['p'][0])]
        seq += [(it, False, False) for it in its]
        tip = its[-1]['p'][-1] if its else tip
    out = []
    for it, focal, eye in seq:
        out.append(stroke(thin(it['p'], 5.), it['R'] + 1.2, 'pen', rng, rgb=list(media.LINE_RGB), level='linec', alpha='line', layer='line',
                          press=it['press'], focal=focal, eye=eye, solo=it['L'] > 70))
    print('line art:', ' '.join(f"{'-'.join(map(str, g))} {len(v)}" for g, v in groups.items()), '| strokes', len(out),
          '| median length', round(float(np.median([it['L'] for it, *_ in seq])), 1))
    return out


# ------------------------------------------------------------------------------------------------- plan
def plan_anime(target_path, rough_path, areas_path, out, focus, eyes, signature, photo_path, lineart_path=None):
    global SURF
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(paint.SEED)
    A, crop_, shape = load_subject(target_path)
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
    areas = {n: AREAS[n].copy() for n in ('skin', 'hair', 'clothes', 'bg', 'person')}
    la_own = line_art(A)
    la, la_own = with_clean_lines(A, la_own, lineart_path, areas, eyes) if lineart_path else (la_own, la_own)
    C0 = under_lines(A, la_own)                   # the colours under the illustration's own lines
    areas['eyes'] = eye_area(C0, eyes)
    areas['skin'] &= ~areas['eyes']
    F1, F2, flats, shades, flat_lab, shade_lab = take_apart(C0, areas, rng)
    rings = iris_rings(F1, areas, eyes)
    la = np.maximum(la, rings.astype(float))                       # the lines the pen follows
    look, linec = line_look(A, C0, la, la_own)                     # what it lays: the illustration's own line look
    media.LEVELS.update(A=A, C0=C0, linec=linec)
    media.ALPHAS.update(line=look)
    media.LABELS.update(flat=flat_lab, shade=shade_lab)
    for k, v in media.LEVELS.items():
        np.save(out / f'level-{k}.npy', v.astype(np.float16))
    np.save(out / 'alpha-line.npy', look.astype(np.float16))
    for k, v in media.LABELS.items():
        np.save(out / f'label-{k}.npy', v)
    Image.fromarray((F1 * 255).astype(np.uint8)).save(out / 'flats.png')
    Image.fromarray((F2 * 255).astype(np.uint8)).save(out / 'shades.png')
    Image.fromarray(((1 - la) * 255).astype(np.uint8)).save(out / 'lineart.png')
    for f in out.glob('clip-*.png'):
        f.unlink()
    for area in ORDER:
        media.CLIPS[area] = soft_clip(areas[area], 1.5)
        Image.fromarray((media.CLIPS[area] * 255).astype(np.uint8)).save(out / f'clip-{area}.png')
    SURF = surf = media.LayerSurface((1., 1., 1.), ('rough', 'line', 'hi', 'adjust', 'sign'))
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
        img = np.clip(surf.image(), 0, 1)
        Image.fromarray((img * 255).astype(np.uint8)).save(out / f'pass-{len(passes):02d}-{name}.png')
        err = np.abs(ndimage.gaussian_filter(img, (2, 2, 0)) - ndimage.gaussian_filter(A, (2, 2, 0))).mean()
        roles = {}
        for st in strokes:
            roles[st.get('role', '-')] = roles.get(st.get('role', '-'), 0) + 1
        print(f'{name:10s} {len(strokes):5d} strokes {roles}  err {err:.4f}  {time.time() - started:6.1f}s', flush=True)

    tip = lambda: np.asarray(state['tip'], float)
    # 草稿: construction, then the rough itself, quick and light, on its own layer
    rough = construction(areas, eyes, rng)
    sk = paint.sketch_strokes(rough_path, shape, crop_, rng)
    sk = [st for st in sk if length(st['points']) >= 28]          # a rough is a few dozen quick lines, not every lash
    for st in sk:
        st.update(medium='pen', layer='rough', rgb=ROUGH_RGB, mix_rgb=ROUGH_RGB, opacity=.6, R=1.5, press=(.5, .4),
                  seconds=.06 + length(st['points']) / 700)
    run('rough', '草稿', rough + sk, 6., laid=False, nib_px=240., travel_px=240.)     # the rough is dashed off
    # 线稿: the clean line art over it; the eyes' lines at life pace
    run('lines', '线稿', ink_lines(la, areas, rng), 22., laid=False, nib_px=115., travel_px=150.)
    surf.shown['rough'] = 0.
    # 平涂: flat colours under the line art (the rough is hidden first)
    cen = lambda f: ((f['box'][2] + f['box'][3]) / 2, (f['box'][0] + f['box'][1]) / 2)
    # how the fill strokes run in each area: hair along its fall, the face in shallow arcs across the cheeks
    FILL = {'skin': dict(angle=.12, bow=.05), 'hair': dict(angle=math.pi / 2, max_rows=3, top_first=True, edge=False), 'eyes': {}, 'clothes': {}, 'bg': dict(angle=math.pi / 2, edge=False)}

    def lay_flats(area, **extra):
        regs = sorted((f for f in flats if f['area'] == area), key=lambda f: -f['n'])
        big = regs[:1]                         # the area's main shape, then its neighbours, nearest first
        rest = [f for (f,) in nearest_order([(f,) for f in regs[1:]], cen(big[0]) if big else tip(), key=lambda t: cen(t[0]))]
        out_ = []
        for f in big + rest:
            add = paint_shape(f, flat_lab, rng, tip(), extra=extra, **FILL[area])
            out_ += add
            if add:
                state['tip'] = add[-1]['points'][-1]
        return out_

    flat_strokes = []
    for area in ORDER:                        # the eyes get their whites, iris and pupil here with everything else
        flat_strokes += lay_flats(area)
    run('flat', '平涂', flat_strokes, 22., hide=['rough'])
    # 阴影 / 高光: cel shapes of each material's shades, clipped to their base (the eyes wait for their close-up)
    keep = {k: {s['id'] for s in sorted((s for s in shades if s['kind'] == k and s['area'] != 'eyes'),
                                        key=lambda s: -s['n'])[:MOST[k]]} for k in MOST}
    for kind, label, budget in (('shadow', '阴影', 8.), ('light', '高光', 14.)):
        add_all = []
        for area in ORDER:
            if area == 'eyes':
                continue
            items = [s for s in shades if s['kind'] == kind and s['area'] == area and s['id'] in keep[kind]]
            for (s,) in nearest_order([(s,) for s in items], tip(), key=lambda t: cen(t[0])):
                add = paint_shape(s, shade_lab, rng, tip(), free=flat_lab == s['base'], max_R=30., wide=1., **FILL[area])
                add_all += add
                if add:
                    state['tip'] = add[-1]['points'][-1]
        run(kind, label, add_all, budget)
    # 渐变: airbrush gradients toward the illustration's own colours, wide sweeps first
    soft = []
    angles = {'skin': .15, 'hair': math.pi / 2, 'clothes': 1.1, 'bg': .6}
    rounds = {'bg': ((80., .03, .7),), 'hair': ((70., .03, .7),), 'clothes': ((44., .03, .7),), 'skin': ((26., .025, .65),)}
    for area in ('bg', 'hair', 'clothes', 'skin'):
        for R, thr, op in rounds[area]:
            add = sweeps('C0', areas[area], area, rng, tip(), R=R, thr=thr, opacity=op, angle=angles[area])
            soft += add
            print(f'  sweeps {area} R{R:.0f}: {len(add)}')
            if add:
                state['tip'] = add[-1]['points'][-1]
    run('gradient', '渐变', soft, 7.)
    # 眼睛: their shades and gradients in a close-up, then the highlights on a layer above the lines
    eye_strokes = []
    mid_x = W * float(np.mean([e[0] for e in eyes])) if eyes else W / 2          # one eye, then the other
    xs_grid = np.arange(W)[None, :]
    for side in (False, True):
        mine = lambda s: ((s['box'][2] + s['box'][3]) / 2 > mid_x) == side
        items =sorted((s for s in shades if s['area'] == 'eyes' and mine(s)), key=lambda s: (s['kind'] != 'shadow', -s['n']))
        for s in items:
            add = paint_shape(s, shade_lab, rng, tip(), free=flat_lab == s['base'], extra=dict(focal=True, eye=True))
            eye_strokes += add
            if add:
                state['tip'] = add[-1]['points'][-1]
        m = ndimage.binary_dilation(areas['eyes'], iterations=3) & ((xs_grid > mid_x) == side)
        add = sweeps('C0', m, None, rng, tip(), R=5., thr=.045, opacity=.65, angle=.1, focal=True, eye=True)
        eye_strokes += add
        if add:
            state['tip'] = add[-1]['points'][-1]
    print('  eyes: strokes', len(eye_strokes))
    run('polish', '眼睛', eye_strokes, 2.)
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
    run('dianjing', '眼睛', touches, 0.)
    img = np.clip(surf.image(), 0, 1)
    err = np.abs(ndimage.gaussian_filter(img, (2, 2, 0)) - ndimage.gaussian_filter(A, (2, 2, 0))).mean(-1)
    print('before the adjustment, mean |colour error| (2px blur):',
          ' '.join(f'{n} {err[areas[n]].mean():.3f}' for n in ORDER), f'| lines {err[la > .3].mean():.3f}')
    Image.fromarray((img * 255).astype(np.uint8)).save(out / 'before-adjust.png')
    # 调色: the last adjustment eases in (the hand drags a slider on the colour panel)
    run('adjust', '调色', [stroke(SLIDER, 4., 'adjust', rng, rgb=[1, 1, 1], level='A', layer='adjust', flood=2.5,
                                 solo=True, seconds=1.3, hold=1.1)], 1.5, laid=False)
    if signature:
        text, box = signature
        sig = paint.signature_strokes(text, box, (.35, .16, .13), rng)
        for st in sig:
            st.update(medium='pen', layer='sign', rgb=[.35, .16, .13], R=1.6, opacity=1.)
        run('sign', '签名', sig, 3., laid=False)
    final = np.clip(surf.image(), 0, 1)
    Image.fromarray((final * 255).astype(np.uint8)).save(out / 'final.png')
    err = np.abs(ndimage.gaussian_filter(final, (2, 2, 0)) - ndimage.gaussian_filter(A, (2, 2, 0))).mean(-1)
    print('final mean |colour error| (2px blur):', ' '.join(f'{n} {err[areas[n]].mean():.4f}' for n in ORDER))
    print('strokes', sum(len(p['strokes']) for p in passes))
    for p in passes:
        for st in p['strokes']:
            st['rgb'] = [float(c) for c in st['rgb']]
            st['mix_rgb'] = [float(c) for c in st['mix_rgb']]
    (out / 'plan.json').write_text(json.dumps(dict(W=W, H=H, kind='anime', title='动漫插画', passes=passes, focus=focus,
                                                   face=list(REGIONS['face']), pace=PACE, ground=[1., 1., 1.],
                                                   layers=['rough', 'line', 'hi', 'adjust', 'sign']), ensure_ascii=False))
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
