"""素描: draw a portrait in graphite the way a person does, toward a pencil drawing registered to the photo.

A stroke is a pencil stroke: one grade of lead, a pressure that changes along its length (never across its width),
graphite that adds up where strokes cross (media.graphite). Nothing is copied from the target pixel by pixel: the
target only tells the planner how dark each part should end up, and the planner answers with hatching.

Order (an academy portrait drawing):
  起形     long straight construction lines (the envelope of the head, the tilt of the face, eye / nose / mouth
           marks), the contour cut into straight chords, a kneaded eraser pats it all faint, then the true contour
  铺大关系  the big tones over the whole sheet: the hair's mass with the side of a soft lead along its flow, the
           face's shadows in one direction of light hatching, the features' darks stated at once, clothes, background
  揉擦     stump on the skin, tissue along the hair
  深入     area by area: hair in sharp strands, the face in cross-hatching at new angles, the features with it
  刻画五官  close-up: fine hatching and the crisp dark lines, the eyes at life pace
  点睛     the pupils' darkest touch
  提亮     eraser pen along the hair's shine, kneaded eraser on the skin's lights, catchlights
  收尾     the last close layer everywhere, the deepest accents
  签名

Every layer is planned from the sheet as it is at that moment (closed loop): a patch of hatching reads how dark the
paper already is around it and how dark this layer wants it, and presses accordingly; where nothing is needed the
pencil lifts. Patches are worked one next to the other, lines back and forth, so no area is left with holes.

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
from scipy.spatial import ConvexHull

import paint
import media
import planes
from media import GRADES
from paint import W, H, AREAS, FEATURES, EYE_FEATURES, REGIONS, load_subject, ellipse_mask

LUM = np.array([.299, .587, .114])
PACE = dict(feature_cap=2.2, eye_life={'polish': 3.6, 'dianjing': 2.5}, pause=.4)
SURF = None
GROUND = None
CT = None             # the target as graphite coverage (0 paper .. 1 densest)


def unit(deg):
    return np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))])


FLICK = unit(-52)     # a right hander's natural hatching slant
CROSS = unit(-14)
STEEP = unit(-80)
BACK = unit(38)


def lum(rgb):
    return rgb @ LUM


def cov(box=None):
    """Graphite coverage of the sheet as it is now."""
    if box is None:
        return media.coverage(SURF.canvas, GROUND)
    y0, y1, x0, x1 = box
    return media.coverage(SURF.canvas[y0:y1, x0:x1], GROUND[y0:y1, x0:x1])


def stroke(points, R, medium, rng, **kw):
    d = dict(points=[(round(float(x), 2), round(float(y), 2)) for x, y in points], R=round(float(R), 3), medium=medium,
             seed=int(rng.integers(1 << 30)), rgb=[.4] * 3, mix=0, mix_rgb=[.4] * 3, press=(.5, .3), pvar=.08, wobble=0)
    d.update({k: v for k, v in kw.items() if v is not None})
    if ADV[0] and medium == 'graphite':
        d['adv'] = round(float(ADV[0]), 2)
    if 'pp' in d:
        d['pp'] = [round(float(p), 3) for p in d['pp']]
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


def at(field, pts):
    q = np.clip(np.rint(pts).astype(int), 0, [W - 1, H - 1])
    return field[q[:, 1], q[:, 0]]


def arc_len(pts):
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) if len(pts) > 1 else 0.


def index_runs(ok, min_n):
    """Stretches of consecutive True at least min_n long, as (start, stop)."""
    out, a = [], None
    for i, k in enumerate(ok):
        if k and a is None:
            a = i
        elif not k and a is not None:
            if i - a >= min_n:
                out.append((a, i))
            a = None
    if a is not None and len(ok) - a >= min_n:
        out.append((a, len(ok)))
    return out


CAP = None            # coverage no stroke of the current layer may exceed where it passes (the pencil eases off over a light)


def set_cap(m):
    global CAP
    CAP = m


def capped(p, pts, c, soft):
    if CAP is None:
        return p
    return np.minimum(p, smooth1(np.maximum(0, at(CAP, pts) - c) / (1 - c + .03) / soft, 1.))


def press_for(need, c, f, soft, gain):
    """Pressure that makes hatching with line share f close a tone gap `need` over paper already covered c."""
    return gain * need / (f * (1 - c) + .03) / soft


def resample(v, n):
    return np.interp(np.linspace(0, len(v) - 1, n), np.arange(len(v)), v)


def pencil(pts, p, R, rng, grade, pmin, pmax, **kw):
    """One stroke from dense sample points and the pressure wanted at each."""
    n = len(pts)
    L = arc_len(pts)
    k = int(np.clip(L / 9, 2, 40))
    keep = np.unique(np.linspace(0, n - 1, k + 1).round().astype(int))
    pp = np.clip(resample(p, int(np.clip(L / 5, 2, 24))), pmin * .6, pmax)
    if kw.get('side'):          # the side of the lead is laid down and lifted off gradually: no blunt ends
        u = np.linspace(0, 1, len(pp))
        pp = pp * np.clip(np.minimum(u, 1 - u) / .22, .12, 1.) ** .8
    else:                       # a hand never presses two strokes alike
        pp = np.clip(pp * rng.uniform(.86, 1.14), pmin * .6, pmax)
    return stroke(pts[keep], R * rng.uniform(.92, 1.08), 'graphite', rng, grade=grade, pp=pp, **kw)


ADV = [None]          # sideways advance of the hatching being laid (its line spacing), written into each stroke


def smooth1(v, s):
    return ndimage.gaussian_filter1d(v, s, mode='nearest') if s > .3 and len(v) > 2 else v


def strand_flow(v, grad=1., rho=9.):
    # the direction hair runs in: the fine lines of its strands, pooled over a lock (a coarse reading follows the
    # edges of the shine instead, across the hair)
    gx, gy = ndimage.gaussian_filter(v, grad, order=(0, 1)), ndimage.gaussian_filter(v, grad, order=(1, 0))
    jxx, jxy, jyy = (ndimage.gaussian_filter(a, rho) for a in (gx * gx, gx * gy, gy * gy))
    theta = .5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2
    return np.cos(theta), np.sin(theta)


# ----------------------------------------------------------------------------------------------- hatching
def bands(goal, mask, active, rng, tip, *, grade, R, spacing, angle, length, sigma, smooth, gain, pmin, pmax, min_len,
          clip, bow, side, focal, jitter, f, soft, bridge=0.):
    """One direction of hatching laid the way a hand does it: a band of parallel strokes, each about `length` long,
    advancing sideways across the area, back and forth; then the next band beside it, its ragged ends meshing with
    the last one's. The sheet is read again before every band."""
    a0 = math.atan2(angle[1], angle[0]) + math.radians(rng.normal(0, jitter * .4))
    t = np.array([math.cos(a0), math.sin(a0)])
    nrm = np.array([-t[1], t[0]])
    ys, xs = np.nonzero(active)
    A, Bc = xs * t[0] + ys * t[1], xs * nrm[0] + ys * nrm[1]
    starts = list(np.arange(A.min() - length * .15, A.max(), length * .8))
    tip = np.asarray(tip, float)
    if abs(tip @ t - starts[-1]) < abs(tip @ t - starts[0]):
        starts = starts[::-1]
    out = []
    flip = False
    for a_lo in starts:
        sel = (A >= a_lo - 2) & (A <= a_lo + length + 2)
        if sel.sum() < 12:
            continue
        b_lo, b_hi = Bc[sel].min(), Bc[sel].max()
        C = ndimage.gaussian_filter(cov(), sigma)
        N = ndimage.gaussian_filter(goal, sigma) - C
        offs = []
        o = b_lo + rng.uniform(0, spacing)
        while o < b_hi:
            offs.append(o)
            o += spacing * rng.uniform(.84, 1.16)
        if flip:
            offs = offs[::-1]
        flip = not flip
        curve = bow * rng.uniform(.5, 1.5)
        drift = 0.
        back = False
        for o in offs:
            drift = .8 * drift + rng.normal(0, .05) * length              # the band's edges wander, never ruled
            half = length / 2 * rng.uniform(.84, 1.14)
            ca = a_lo + length / 2 + drift
            da = math.radians(rng.normal(0, jitter * .3))
            tt = np.array([math.cos(a0 + da), math.sin(a0 + da)])
            nn = np.array([-tt[1], tt[0]])
            u = np.linspace(-half, half, max(3, int(half)))
            pts = t * ca + nrm * o + tt * u[:, None] + nn * (curve * half * (1 - (u / half) ** 2))[:, None]
            inb = (pts[:, 0] >= 0) & (pts[:, 0] < W - 1) & (pts[:, 1] >= 0) & (pts[:, 1] < H - 1)
            if not inb.any():
                continue
            inside = inb & at(mask, pts)
            p = capped(smooth1(press_for(at(N, pts), at(C, pts), f, soft, gain), smooth / 2.), pts, at(C, pts), soft)
            on = p > pmin
            if bridge:                 # the pencil eases up over a short light stretch instead of lifting off
                on = ndimage.binary_closing(np.r_[False, on, False], np.ones(int(bridge / 2) | 1))[1:-1] | on
            for i0, i1 in index_runs(inside & on, max(2, int(min_len / 2))):
                seg, ps = pts[i0:i1], p[i0:i1]
                if ps.max() < 1.5 * pmin:              # a touch this faint over so short a stretch is not a stroke
                    continue
                if back:
                    seg, ps = seg[::-1], ps[::-1]
                out.append(lay(pencil(seg, ps, R, rng, grade, pmin, pmax, clip=clip, side=side or None, focal=focal,
                                      press=(.7, .5))))
            back = not back
    return out


def hatch(goal, mask, rng, tip, *, grade='HB', R=1., spacing=4., angle=FLICK, flow=None, patch=(56, 96), sigma=None,
          smooth=9., gain=.85, pmin=.12, pmax=1., min_len=7., clip=None, bow=.05, side=False, thr=.025, focal=None,
          jitter=5., step=.7, bridge=0.):
    """Hatching in patches, each a set of parallel, slightly bowed lines laid back and forth, patch next to patch.
    The pressure along every line follows how much tone this layer still wants there (smoothed over `smooth` px); the
    pencil lifts where none is wanted. With flow, each patch takes the direction of the form at its centre."""
    sigma = sigma or spacing * .9
    soft = GRADES[grade][0]
    ADV[0] = spacing
    f = min(1., 2 * R * (.75 if side else 1.) / spacing)
    need = ndimage.gaussian_filter(goal - cov(), sigma)
    active = mask & (need > thr)
    if active.sum() < 30:
        return []
    ys, xs = np.nonzero(active)
    if flow is None:
        return bands(goal, mask, active, rng, tip, grade=grade, R=R, spacing=spacing, angle=angle, length=float(np.mean(patch)),
                     sigma=sigma, smooth=smooth, gain=gain, pmin=pmin, pmax=pmax, min_len=min_len, clip=clip, bow=bow,
                     side=side, focal=focal, jitter=jitter, f=f, soft=soft, bridge=bridge)
    st_ = float(np.mean(patch)) * step
    centres = []
    for j, gy in enumerate(np.arange(ys.min() + st_ * .3, ys.max() + st_ * .6, st_)):
        row = []
        for gx in np.arange(xs.min() + st_ * .3, xs.max() + st_ * .6, st_):
            cx, cy = gx + rng.uniform(-.22, .22) * st_, gy + rng.uniform(-.22, .22) * st_
            r = rng.uniform(*patch) / 2
            y0, y1, x0, x1 = int(max(0, cy - r)), int(min(H, cy + r)), int(max(0, cx - r)), int(min(W, cx + r))
            if y1 > y0 and x1 > x0 and active[y0:y1, x0:x1].mean() > .04:
                row.append((np.array([cx, cy]), r))
        centres += row[::-1] if j % 2 else row
    if not centres:
        return []
    tip = np.asarray(tip, float)
    if np.linalg.norm(centres[-1][0] - tip) < np.linalg.norm(centres[0][0] - tip):
        centres = centres[::-1]
    out = []
    pad = int(3 * sigma) + 3
    for c0, r in centres:
        y0, y1 = int(max(0, c0[1] - r - pad)), int(min(H, c0[1] + r + pad))
        x0, x1 = int(max(0, c0[0] - r - pad)), int(min(W, c0[0] + r + pad))
        if y1 - y0 < 4 or x1 - x0 < 4:
            continue
        Cb = ndimage.gaussian_filter(cov((y0, y1, x0, x1)), sigma)
        Nb = ndimage.gaussian_filter(goal[y0:y1, x0:x1], sigma) - Cb
        if flow is not None:
            iy, ix = int(np.clip(c0[1], 0, H - 1)), int(np.clip(c0[0], 0, W - 1))
            a = math.atan2(flow[1][iy, ix], flow[0][iy, ix])
        else:
            a = math.atan2(angle[1], angle[0])
        a += math.radians(rng.normal(0, jitter))
        t = np.array([math.cos(a), math.sin(a)])
        nrm = np.array([-t[1], t[0]])
        curve = bow * rng.uniform(.5, 1.5) * rng.choice([1, 1, 1, -1])
        o = -r + rng.uniform(0, spacing)
        back = False
        while o < r:
            half = math.sqrt(max(0., r * r - o * o)) * rng.uniform(.82, 1.1)
            mid = c0 + nrm * o
            o += spacing * rng.uniform(.86, 1.14)
            if half < min_len * .5:
                continue
            u = np.linspace(-half, half, max(3, int(half)))                # a sample every 2 px
            pts = mid + t * u[:, None] + nrm * (curve * half * (1 - (u / half) ** 2))[:, None]
            inb = (pts[:, 0] >= 0) & (pts[:, 0] < W - 1) & (pts[:, 1] >= 0) & (pts[:, 1] < H - 1)
            inside = inb & at(mask, pts)
            q = np.clip(np.rint(pts).astype(int) - [x0, y0], 0, [x1 - x0 - 1, y1 - y0 - 1])
            p = capped(smooth1(press_for(Nb[q[:, 1], q[:, 0]], Cb[q[:, 1], q[:, 0]], f, soft, gain), smooth / 2.), pts,
                       Cb[q[:, 1], q[:, 0]], soft)
            for i0, i1 in index_runs(inside & (p > pmin), max(2, int(min_len / 2))):
                seg, ps = pts[i0:i1], p[i0:i1]
                if back:
                    seg, ps = seg[::-1], ps[::-1]
                out.append(lay(pencil(seg, ps, R, rng, grade, pmin, pmax, clip=clip, side=side or None, focal=focal)))
            back = not back
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


def flow_lines(mask, flow, rng, spacing, length, down=True, step=3.):
    """Evenly spaced lines along a flow field covering mask (a new line starts only where none runs near)."""
    occupied = np.zeros((H, W), bool)
    r = max(1, int(round(spacing * .4)))
    ys, xs = np.nonzero(mask)
    if len(xs) < 30:
        return []
    seeds = np.column_stack([xs, ys])[rng.permutation(len(xs))[:max(1, int(len(xs) / (spacing * spacing * .5)))]]
    lines = []
    for sx, sy in seeds:
        if occupied[sy, sx]:
            continue
        t = np.array([flow[0][sy, sx], flow[1][sy, sx]])
        if down and t[1] < 0:
            t = -t
        back = streamline((sx, sy), flow, mask, step, length[1] * .5, -t, occupied)
        fwd = streamline((sx, sy), flow, mask, step, length[1] * .5, t, occupied)
        line = np.asarray(back[::-1] + fwd[1:])
        if len(line) < 3 or arc_len(line) < length[0]:
            continue
        if down and line[-1][1] < line[0][1]:
            line = line[::-1]
        for x, y in line:
            ix, iy = int(x), int(y)
            occupied[max(0, iy - r):iy + r + 1, max(0, ix - r):ix + r + 1] = True
        lines.append(line)
    return lines


def sections(lines, tip, section):
    """Lines grouped by the block of the sheet they start in, blocks visited one next to the other, the lines in a
    block side by side, drawn down one and up the next."""
    blocks = {}
    for ln in lines:
        mid = ln[len(ln) // 2]
        blocks.setdefault((int(mid[0] // section), int(mid[1] // section)), []).append(ln)
    out = []
    cols = sorted({k[0] for k in blocks})
    if cols and abs(tip[0] - cols[-1] * section) < abs(tip[0] - cols[0] * section):
        cols = cols[::-1]
    keys = []
    for j, c in enumerate(cols):                          # column by column across the sheet, down one and up the next
        col = sorted((k for k in blocks if k[0] == c), key=lambda k: k[1])
        keys += col[::-1] if j % 2 else col
    cur = np.asarray(tip, float)
    for key in keys:
        left = list(blocks[key])
        while left:
            ends = np.array([[ln[0], ln[-1]] for ln in left])                 # (n, 2, 2)
            d = np.linalg.norm(ends - cur, axis=2)
            k, e = np.unravel_index(d.argmin(), d.shape)
            ln = left.pop(int(k))
            ln = ln[::-1] if e else ln
            out.append(ln)
            cur = ln[-1]
    return out


def strands(goal, mask, flow, rng, tip, *, grade='4B', R=1.2, spacing=3.2, length=(30, 170), sigma=None, smooth=6.,
            gain=.85, pmin=.12, pmax=1., clip=None, side=False, thr=.025, down=True, section=170., eraser=False,
            opacity=.8, min_len=10., step=3., focal=None, grow=None, bridge=0.):
    """Long lines along a flow field (hair drawn strand by strand, section by section; ribs of a knit). The pressure
    along each follows the tone still wanted under it. eraser=True: an eraser pen lifting what is too dark."""
    sigma = sigma or max(1., spacing * .8)
    ADV[0] = spacing if R < 3 else None
    soft = 1. if eraser else GRADES[grade][0]
    f = min(1., 2 * R * (.75 if side else 1.) / spacing)
    C = ndimage.gaussian_filter(cov(), sigma)
    G = ndimage.gaussian_filter(goal, sigma)
    N = (C - G) if eraser else (G - C)
    active = mask & (N > thr)
    if active.sum() < 30:
        return []
    ok = ndimage.binary_dilation(active, iterations=max(2, int(spacing)) if grow is None else grow) & mask
    out = []
    for ln in sections(flow_lines(ok, flow, rng, spacing, length, down, step), tip, section):
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(ln, axis=0), axis=1))]
        m = max(3, int(d[-1] / min(2., step)))
        pts = np.column_stack([np.interp(np.linspace(0, d[-1], m), d, ln[:, 0]), np.interp(np.linspace(0, d[-1], m), d, ln[:, 1])])
        n, c = at(N, pts), at(C, pts)
        p = (gain * n / (f * c + .03)) if eraser else press_for(n, c, f, soft, gain)
        p = smooth1(p, smooth / 2.)
        if not eraser:
            p = capped(p, pts, c, soft)
        on = p > pmin
        if bridge:                    # a strand is not broken off for a short light stretch: the pencil just eases up
            on = ndimage.binary_closing(np.r_[False, on, False], np.ones(int(bridge / min(2., step)) | 1))[1:-1] | on
        for i0, i1 in index_runs(on, max(2, int(min_len / min(2., step)))):
            seg, ps = pts[i0:i1], p[i0:i1]
            if eraser:
                k = np.unique(np.linspace(0, len(seg) - 1, int(np.clip(arc_len(seg) / 9, 2, 40)) + 1).round().astype(int))
                out.append(lay(stroke(seg[k], R * rng.uniform(.9, 1.1), 'eraser', rng, opacity=opacity, clip=clip,
                                      pp=np.clip(resample(ps, int(np.clip(len(seg) / 3, 2, 24))), .1, 1.), press=(.4, .3),
                                      focal=focal)))
            else:
                out.append(lay(pencil(seg, ps, R, rng, grade, pmin, pmax, clip=clip, side=side or None, press=(.4, .2),
                                      focal=focal)))
    return out


def touch_up(goal, region, rng, tip, *, grade='2B', thr=.14, min_px=5):
    ADV[0] = None
    # a last look for paper left bare by mistake (slivers between locks of hair, specks at an edge the hatching
    # stopped short of): each gets one short stroke along its length
    gap = ndimage.gaussian_filter(goal, .6) - ndimage.gaussian_filter(cov(), .6)
    comp, n = ndimage.label(region & (gap > thr), np.ones((3, 3)))
    items = []
    for i, sl in enumerate(ndimage.find_objects(comp)):
        ys, xs = np.nonzero(comp[sl] == i + 1)
        if len(xs) < min_px:
            continue
        P = np.column_stack([xs + sl[1].start, ys + sl[0].start]).astype(float)
        cen = P.mean(0)
        ev, evec = np.linalg.eigh(np.cov((P - cen).T) + np.eye(2) * .05)
        u, w = evec[:, -1], evec[:, 0]
        a, b = (P - cen) @ u, (P - cen) @ w
        g = float(gap[sl][comp[sl] == i + 1].mean())
        for k, o in enumerate(np.arange(b.min() + .5, b.max() + .01, 1.5) if b.max() - b.min() > 1.6 else [0.]):
            on = np.abs(b - o) < 1.
            if on.sum() < 2:
                continue
            ends = (a[on].min() - .5, a[on].max() + .5) if k % 2 == 0 else (a[on].max() + .5, a[on].min() - .5)
            pts = cen + w * o + u * np.linspace(*ends, max(3, int(abs(ends[1] - ends[0])) + 2))[:, None]
            items.append((cen + w * o, pts, .8, g))
    out = []
    soft = GRADES[grade][0]
    for _, pts, R, g in nearest_order(items, tip):
        c = at(cov(), pts)
        p = np.clip(g / (1 - c + .03) / soft, .2, 1.)
        out.append(lay(pencil(pts, p, R, rng, grade, .15, 1., press=(.6, .5))))
    return out


def scrub(mask, rng, tip, *, R, length, flow, opacity, step=None, clip=None, medium='stump', excess=None, spread=None):
    """Blending stump / tissue / kneaded eraser: overlapping short strokes along the form, row after row. With
    excess (a map of how much too dark the sheet is), an eraser presses in proportion and skips where there is none."""
    if mask.sum() < 40:
        return []
    step = step or R * 1.25
    ys, xs = np.nonzero(mask)
    pts = []
    for j, gy in enumerate(np.arange(ys.min() + step * .3, ys.max() + step, step)):
        row = []
        for gx in np.arange(xs.min() + step * .3, xs.max() + step, step):
            cx, cy = gx + rng.uniform(-.4, .4) * step, gy + rng.uniform(-.4, .4) * step
            ix, iy = int(np.clip(cx, 0, W - 1)), int(np.clip(cy, 0, H - 1))
            if mask[iy, ix]:
                row.append(np.array([cx, cy]))
        pts += row[::-1] if j % 2 else row
    if pts and np.linalg.norm(pts[-1] - np.asarray(tip, float)) < np.linalg.norm(pts[0] - np.asarray(tip, float)):
        pts = pts[::-1]
    out = []
    for k, p0 in enumerate(pts):
        ix, iy = int(np.clip(p0[0], 0, W - 1)), int(np.clip(p0[1], 0, H - 1))
        t = np.array([flow[0][iy, ix], flow[1][iy, ix]])
        half = rng.uniform(*length) / 2
        p = np.asarray(streamline(p0, flow, mask, 3., half, -t)[::-1] + streamline(p0, flow, mask, 3., half, t)[1:])
        if len(p) < 3:
            continue
        if k % 2:
            p = p[::-1]
        pp = None
        if excess is not None:
            e = at(excess, p)
            if e.max() < .03:
                continue
            pp = np.clip(resample(e / (e.max() + 1e-6), 6), .15, 1.)
            op = float(np.clip(opacity * e.max() / .25, .12, opacity))
        else:
            op = opacity
        out.append(lay(stroke(p[::2] if len(p) > 6 else p, R * rng.uniform(.85, 1.15), medium, rng, opacity=op, clip=clip,
                              press=(.6, .4), pp=pp, spread=spread, tremor=.05)))
    return out


def dark_lines(target_v, region, rng):
    """The drawing's own crisp dark lines (lash lines, lid creases, the lip line, nostrils, brow hairs) as centre
    lines with their half width: the line-first engine's skeleton strokes over the target's thin dark marks."""
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
        if len(p) < 3 or arc_len(p) < 5:
            continue
        out.append((p, float(np.clip(np.median(at(half, p)) * 1.05, .75, 1.9))))
    return out


def edge_lines(region, rng, s=1.):
    # where a dark shape meets a light one sharply (the rim of the iris, the lid against the white of the eye, the
    # edge of the lips): the lines a draughtsman states with the point before shading up to them
    g = ndimage.gaussian_filter(CT, s)
    mag = np.hypot(ndimage.sobel(g, 0), ndimage.sobel(g, 1)) / 8
    ridge = (mag > .045) & (mag >= ndimage.maximum_filter(mag, 3) * .92) & region
    from line_topology import extract_strokes
    raw, *_ = extract_strokes(ndimage.binary_dilation(ridge), size=W / 720)
    out = []
    for st in raw:
        p = np.asarray(st['points'], float)
        if len(p) >= 3 and arc_len(p) >= 7:
            out.append((p, .7))
    return out


def draw_lines(lines, goal, rng, tip, *, grade, gain=.9, pmin=.1, focal=True, smooth=3.):
    ADV[0] = None
    """Draw centre lines as single continuous pencil strokes, pressed as dark as the layer wants them."""
    soft = GRADES[grade][0]
    out = []
    order = nearest_order([(p.mean(0), p, R) for p, R in lines], tip)
    for _, p, R in order:
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        m = max(3, int(d[-1] / 1.5))
        pts = np.column_stack([np.interp(np.linspace(0, d[-1], m), d, p[:, 0]), np.interp(np.linspace(0, d[-1], m), d, p[:, 1])])
        c = at(cov(), pts)
        pr = smooth1(gain * (at(goal, pts) - c) / (1 - c + .03) / soft, smooth)
        if pr.max() < pmin:
            continue
        out.append(lay(pencil(pts, np.maximum(pr, pmin * .7), R, rng, grade, pmin, 1., press=(.35, .25), focal=focal or None)))
    return out


# ---------------------------------------------------------------------------------------------- construction
def straight(a, b, rng, *, over=.07, grade='HB', pressure=.5, R=1., restate=.3):
    """A long construction line: drawn from the shoulder, running past both ends, sometimes restated."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = np.linalg.norm(d)
    if L < 8:
        return []
    n = np.array([-d[1], d[0]]) / L
    out = []
    for k in range(1 + (rng.random() < restate)):
        a2 = a - d * over * rng.uniform(.4, 1.6) + n * rng.normal(0, 1.2 * k)
        b2 = b + d * over * rng.uniform(.4, 1.6) + n * rng.normal(0, 1.8 * k)
        mid = (a2 + b2) / 2 + n * rng.normal(0, .004 * L)
        out.append(stroke([a2, mid, b2], R, 'graphite', rng, grade=grade, pressure=pressure * (1 - .25 * k),
                          pp=[.55, 1., .95, .6], press=(.4, .3), tremor=.14))
    return out


def polygon(mask, n):
    """The convex envelope of a shape cut down to n straight edges."""
    edge = mask & ~ndimage.binary_erosion(mask)
    ys, xs = np.nonzero(edge)
    P = np.column_stack([xs, ys]).astype(float)
    P = P[::max(1, len(P) // 4000)]
    hull = P[ConvexHull(P).vertices]
    hull = list(hull)
    while len(hull) > n:
        m = len(hull)
        area = [abs(np.cross(hull[i] - hull[i - 1], hull[(i + 1) % m] - hull[i - 1])) for i in range(m)]
        hull.pop(int(np.argmin(area)))
    return np.array(hull)


def rdp(p, tol):
    """Douglas-Peucker: a line cut into straight chords."""
    p = np.asarray(p, float)
    if len(p) < 3:
        return p
    a, b = p[0], p[-1]
    d = b - a
    L = np.linalg.norm(d) + 1e-9
    dist = np.abs(np.cross(d, p - a)) / L
    k = int(dist.argmax())
    if dist[k] <= tol:
        return np.array([a, b])
    return np.vstack([rdp(p[:k + 1], tol)[:-1], rdp(p[k:], tol)])


def construction(sketch_lines, rng):
    """起形 as it is taught: envelope, axis and proportion marks, chords; then the lines to pat faint."""
    out, guides = [], []
    border = lambda q: q[0] < 5 or q[0] > W - 6 or q[1] < 5 or q[1] > H - 6
    head = AREAS['hair'] | AREAS['face']
    poly = polygon(head, 8)
    start = int(np.argmin(poly[:, 1]))
    poly = np.roll(poly, -start, axis=0)
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        if not (border(a) and border(b)):
            out += straight(a, b, rng)
            guides.append((a, b))
    # the tilt of the face: the line through the eyes, the middle line across it, and the marks along it
    eyes = [np.array([FEATURES[k][0] * W, FEATURES[k][1] * H]) for k in sorted(EYE_FEATURES)]
    others = [f for k, f in enumerate(FEATURES) if k not in EYE_FEATURES]
    fx, fy, frx, fry = REGIONS['face']
    if len(eyes) == 2:
        m = (eyes[0] + eyes[1]) / 2
        ex = (eyes[1] - eyes[0]) / (np.linalg.norm(eyes[1] - eyes[0]) + 1e-9)
        mouth = max(others, key=lambda f: f[2]) if others else None
        q = np.array([mouth[0] * W, mouth[1] * H]) if mouth else m + np.array([-ex[1], ex[0]]) * fry * H * .9
        down = (q - m) / (np.linalg.norm(q - m) + 1e-9)
        span = np.linalg.norm(q - m)
        top, chin = m - down * span * 1.25, q + down * span * .62
        out += straight(top, chin, rng, over=.03)
        guides.append((top, chin))
        wide = frx * W * 1.02
        for dist, half in ((0., wide), (-span * .33, wide * .8), (span * .52, wide * .45), (span, wide * .5), (span * 1.55, wide * .3)):
            c = m + down * dist
            out += straight(c - ex * half, c + ex * half, rng, over=.04, restate=.15)
            guides.append((c - ex * half, c + ex * half))
        for e in eyes:                                    # where each eye sits: two short ticks
            w = FEATURES[sorted(EYE_FEATURES)[0]][2] * W
            for s in (-1, 1):
                c = e + ex * s * w
                out += straight(c - down * 9, c + down * 9, rng, over=.0, restate=0.)
    body = AREAS['clothes'] | (AREAS['skin'] & ~AREAS['face'])
    if body.sum() > 2000:
        poly = polygon(body, 6)
        for a, b in zip(poly, np.roll(poly, -1, axis=0)):
            if not (border(a) and border(b)) and np.linalg.norm(b - a) > 60:
                out += straight(a, b, rng)
                guides.append((a, b))
    # the contour as straight chords (切直线): only the long lines, the big shapes
    chords = []
    for st in sketch_lines:
        p = np.asarray(st['points'], float)
        if arc_len(p) < 230:
            continue
        c = rdp(p, 16.)
        for a, b in zip(c[:-1], c[1:]):
            chords += straight(a, b, rng, over=.05, pressure=.55, restate=.1)
    return out, chords, guides


def pat(guides, rng):
    """A kneaded eraser rolled over the construction lines: they stay as a faint ghost under the drawing."""
    out = []
    for a, b in guides:
        a, b = np.asarray(a, float), np.asarray(b, float)
        L = np.linalg.norm(b - a)
        n = max(1, int(L / 190))
        for k in range(n):
            p, q = a + (b - a) * k / n, a + (b - a) * (k + 1) / n
            inside = at(AREAS['face'] | AREAS['skin'], np.array([(p + q) / 2]))[0]
            out.append(stroke([p, (p + q) / 2 + rng.normal(0, 3, 2), q], 13., 'eraser', rng, opacity=.92 if inside else .6,
                              press=(.7, .6), tremor=.05))
    return out


# ---------------------------------------------------------------------------------------------------- plan
def plan_sketch(target_path, sketch_path, areas_path, out, focus, eyes, signature, photo_path):
    global SURF, GROUND, CT
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(paint.SEED)
    ref, crop, shape = load_subject(target_path)
    Image.fromarray((load_subject(photo_path)[0] * 255).astype(np.uint8)).save(out / 'reference.png')
    Image.fromarray((ref * 255).astype(np.uint8)).save(out / 'target.png')
    vt = lum(ref)
    paper_rgb = np.percentile(ref.reshape(-1, 3), 97, axis=0)
    tooth, ground = media.paper(np.random.default_rng(paint.SEED + 1), paper_rgb)
    media.set_paper(tooth, ground)
    np.save(out / 'tooth.npy', tooth)
    np.save(out / 'ground.npy', ground.astype(np.float32))
    GROUND = ground
    CT = media.coverage(np.minimum(ref, ground), ground)
    B = lambda s: ndimage.gaussian_filter(CT, s)
    # an early layer never goes darker than the finished drawing anywhere within its own reach: the small lights
    # (the whites of the eyes, the ridge of the nose, the wet of the lip) are left as paper, not rubbed back later
    U = lambda s, k=1.: k * ndimage.gaussian_filter(ndimage.minimum_filter(CT, size=int(2 * s) | 1), s)
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
        # the hair runs a few pixels over its neighbours (a dark edge laid over a light one): no pale seam between areas
        media.CLIPS[name] = np.clip(1 - (d - (4. if name == 'hair' else 1.5)) / 3., 0, 1).astype(np.float32)
        Image.fromarray((media.CLIPS[name] * 255).astype(np.uint8)).save(out / f'clip-{name}.png')
    feats = np.zeros((H, W), bool)
    for e in FEATURES:
        feats |= ellipse_mask(e, 1.35)
    feats &= skin | ndimage.binary_dilation(skin, iterations=3)
    SURF = surf = media.PaperSurface(ground)
    flow_hair = strand_flow(vt)
    flow_form = paint.orientation(ndimage.gaussian_filter(vt, 3), 10., rng, default=-52.)
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

    # ---- 起形
    lines = paint.sketch_strokes(sketch_path, shape, crop, rng)
    envelope, chords, guides = construction(lines, rng)
    run('sketch', '起形', envelope, 11.5, laid=False)
    run('sketch', '起形', chords, 0., laid=False)
    run('sketch', '起形', pat(guides, rng), 0., laid=False)
    contour = []
    for st in lines:
        p = np.asarray(st['points'], float)
        contour.append(stroke(p, 1.05, 'graphite', rng, grade='HB', pressure=.5 * rng.uniform(.85, 1.15),
                              pp=[.7, 1., .9, 1., .75], press=(.4, .25), tremor=.12))
    run('sketch', '起形', contour, 0., laid=False)

    # ---- 铺大关系: the whole sheet, lightly, with few broad strokes
    # the hair is one even grey mass first (side of a soft lead, stroke lapping stroke, then the pinholes closed):
    # strands drawn over it later sit on grey, not on white paper
    mass = np.minimum(.6, .7 * B(13))
    side = dict(grade='4B', clip='hair', side=True, section=260., pmin=.05, pmax=.75)
    run('hair', '铺大关系', strands(mass, hair, flow_hair, rng, tip(), R=10., spacing=11., length=(110, 360), smooth=70.,
                                gain=.85, bridge=60., min_len=50., **side), 7.5)
    run('hair', '铺大关系', strands(mass, hair, flow_hair, rng, tip(), R=8., spacing=8.5, length=(60, 280), smooth=50.,
                                gain=.9, thr=.05, sigma=3., bridge=40., min_len=40., **side), 0.)
    run('hair', '铺大关系', strands(mass, hair, flow_hair, rng, tip(), R=5.5, spacing=5.5, length=(16, 140), smooth=20.,
                                gain=.55, thr=.2, sigma=1.5, grow=2, bridge=20., min_len=16., **dict(side, pmax=.45)), 0.)
    block = U(5, .72)
    set_cap(B(2.2))
    calm = skin & ~ndimage.binary_dilation(feats, iterations=3)       # the big stump stays off the features
    run('face', '铺大关系', hatch(block, skin, rng, tip(), grade='HB', R=1.2, spacing=3.8, angle=FLICK, smooth=14.,
                              clip='skin', patch=(96, 130), min_len=10., bridge=14.), 6.5)
    marks = dark_lines(vt, feats, rng)
    run('face', '铺大关系', draw_lines(marks, .6 * B(.8), rng, tip(), grade='2B', focal=False), 0.)
    run('clothes', '铺大关系', hatch(block, clothes, rng, tip(), grade='HB', R=1.2, spacing=5., flow=flow_form,
                                 smooth=16., clip='clothes', patch=(80, 130), jitter=3.), 3.)
    run('clothes', '铺大关系', hatch(U(7, .8), bg, rng, tip(), grade='HB', R=1.2, spacing=6.5, angle=FLICK,
                                    smooth=26., clip='bg', patch=(110, 170), thr=.03), 2.)

    set_cap(None)
    # ---- 揉擦: slow strokes of the stump along the form, over the shadows and half-tones (the lights stay clean)
    has = ndimage.gaussian_filter(cov(), 3) > .04
    tones = B(4.) > .13
    run('blend', '揉擦', scrub(calm & has & tones, rng, tip(), R=15., length=(70, 130), flow=flow_form, opacity=.75,
                             clip='skin', step=38., spread=.5), 5.5)
    run('blend', '揉擦', scrub(feats & has, rng, tip(), R=3.4, length=(14, 30), flow=flow_form, opacity=.5, clip='skin',
                             step=19., spread=.8), 0.)

    # ---- 深入: area by area
    # the hair's first sharp layer runs over its shine (strand after strand, the lights are not drawn around): the
    # eraser pulls them back out in 提亮, the way it is done on paper
    over = np.maximum(.88 * B(5.), .85 * B(26.))
    run('deep-hair', '深入', strands(over, hair, flow_hair, rng, tip(), grade='4B', R=2.2, spacing=4.5,
                                   length=(70, 300), smooth=9., clip='hair', section=200., pmin=.08, bridge=40.,
                                   min_len=24.), 8.5)
    # skin: light, close, soft. Hard lead, a little pressure, lines nearly touching, layer over layer at a new
    # angle, the stump once in between: tone first, hatching second
    soft = dict(grade='HB', clip='skin', pmin=.05, gain=.9)
    set_cap(B(1.4))
    run('deep-face', '深入', hatch(U(2.4, .9), skin, rng, tip(), R=1., spacing=2.3, angle=CROSS, smooth=8.,
                                 patch=(96, 124), min_len=10., bridge=18., **soft), 12.)
    run('deep-face', '深入', draw_lines(marks, .8 * B(.7), rng, tip(), grade='2B', focal=False), 0.)
    run('deep-face', '深入', scrub(calm & has & (B(3.) > .16), rng, tip(), R=14., length=(60, 110), flow=flow_form,
                                 opacity=.6, clip='skin', step=50., spread=.4), 0.)
    set_cap(B(.9))
    run('deep-face', '深入', hatch(U(1.6, .93), skin, rng, tip(), R=.85, spacing=1.85, angle=STEEP, smooth=5.,
                                 patch=(90, 116), min_len=9., bridge=12., **soft), 0.)
    # the skin is brought to its finished tone before the close-up, so the features are drawn into a finished face;
    # around the features this layer stops a little short and never covers their small lights: the sharp drawing
    # of the close-up needs paper left to draw on
    fw = ndimage.gaussian_filter(feats.astype(float), 5.)
    near = B(.8) * (1 - fw) + U(1.6, .82) * fw
    set_cap(np.minimum(B(.6) + .02, near + .04))
    wide = ndimage.binary_dilation(skin, iterations=2)
    run('deep-face', '深入', hatch(near, wide, rng, tip(), R=.75, spacing=1.55, angle=FLICK, smooth=3., sigma=1.2,
                                 patch=(84, 110), thr=.025, min_len=7., bridge=8., **soft), 0.)
    # the shadow side keeps an open, darker hatch on top: what a soft pencil adds where the form turns away
    run('deep-face', '深入', hatch(near, wide & (B(2.) > .34), rng, tip(), grade='2B', R=.85, spacing=2.3, angle=CROSS,
                                 smooth=3., sigma=1.2, clip='skin', patch=(70, 96), thr=.035, gain=.9, min_len=8.,
                                 bridge=6.), 0.)
    run('deep-rest', '深入', strands(U(2., .95), clothes, flow_form, rng, tip(), grade='2B', R=1.2, spacing=3.6,
                                   length=(30, 120), smooth=7., clip='clothes', down=False, section=140.), 3.5)
    run('deep-rest', '深入', hatch(U(1.5, 1.), clothes, rng, tip(), grade='2B', R=1., spacing=3., angle=BACK, smooth=6.,
                                 clip='clothes', patch=(80, 110), min_len=9., bridge=8.), 0.)
    run('deep-rest', '深入', hatch(U(3., 1.), bg, rng, tip(), grade='HB', R=1.1, spacing=4.6, angle=STEEP, smooth=14.,
                                 clip='bg', patch=(100, 150)), 0.)
    set_cap(None)
    run('deep-hair2', '深入', strands(np.maximum(B(1.4), .9 * over), hair, flow_hair, rng, tip(), grade='6B', R=1.9,
                                    spacing=3.7, length=(60, 300), smooth=6., clip='hair', section=200., pmin=.1, gain=1.,
                                    bridge=30., min_len=24., sigma=2.), 9.)
    # paper still showing between the locks along the fringe: strands laid exactly there
    fringe = hair & ndimage.binary_dilation(skin, iterations=18)
    run('deep-hair2', '深入', strands(B(.8), fringe, flow_hair, rng, tip(), grade='4B', R=1.15, spacing=2.4, length=(6, 120),
                                    sigma=1., smooth=3., clip='hair', thr=.13, grow=1, min_len=8., gain=.9, bridge=6.,
                                    section=120.), 0.)

    # ---- 刻画五官: a sharp pencil, lines along the form, each placed exactly; the eyes last and at life pace
    final = B(.35)
    flow_fine = paint.orientation(vt, 2.2, rng, default=-52.)
    set_cap(B(.3) + .03)
    inner = feats & ndimage.binary_erosion(skin, iterations=3)
    polish = strands(B(.6), inner, flow_fine, rng, tip(), R=.8, spacing=1.5, length=(3, 40), sigma=.7, smooth=1.2,
                     eraser=True, thr=.045, opacity=.6, gain=1., down=False, section=34., min_len=3., step=1.,
                     focal=True, grow=1)
    polish += draw_lines(marks, B(.4), rng, tip(), grade='4B', gain=.95, smooth=2.)
    polish += draw_lines(edge_lines(feats, rng, .8), ndimage.maximum_filter(B(.4), 3), rng, tip(), grade='2B', gain=.85,
                         smooth=2.)
    for grade, sp, thr_ in (('2B', 1.3, .025), ('4B', 1.2, .03), ('6B', 1.15, .045)):
        polish += strands(final, feats, flow_fine, rng, tip(), grade=grade, R=.56, spacing=sp, length=(4, 50),
                          sigma=.55, smooth=1., gain=.97, thr=thr_, down=False, section=34., min_len=4., step=1.,
                          focal=True, grow=1, bridge=2.)
    # the darks (iris, pupil, nostril, the corners of the mouth) are packed solid with short strokes across the
    # lines just laid, so no paper glints between them
    polish += hatch(final, feats & (B(1.5) > .5), rng, tip(), grade='6B', R=.7, spacing=1.3, angle=FLICK, smooth=1.5,
                    sigma=.8, gain=1., patch=(18, 30), min_len=3., thr=.04, focal=True, bow=.02, bridge=2.)
    # ... and rubbed once with the point of a small stump: the iris is a soft dark, not a set of rings
    darks = ndimage.binary_erosion(feats & (B(1.2) > .6), iterations=1)
    n0 = len(polish)
    polish += scrub(darks, rng, tip(), R=1.9, length=(5, 11), flow=flow_fine, opacity=.5, step=3.6, spread=.7)
    # then the sharp lines once more, on top: the lash line, the lid, the rim of the iris, the parting of the lips
    polish += draw_lines(marks, B(.4), rng, tip(), grade='6B', gain=1., smooth=1.5)
    polish += draw_lines(edge_lines(feats, rng, .8), ndimage.maximum_filter(B(.4), 3), rng, tip(), grade='4B', gain=.9,
                         smooth=1.5)
    for st in polish[n0:]:
        st['late'] = 1 + (st['medium'] == 'graphite')
    by_feature = {}
    for st in polish:
        by_feature.setdefault(paint.tag_feature(st), []).append(st)
    ordered = []
    stage = lambda st: st.get('late', 0) + 1 if st.get('late') else (0 if st['medium'] == 'eraser' else 1)
    for s_ in (0, 1, 2, 3):     # the eraser's point, the pencil, the small stump, the sharp lines again
        for k in sorted(by_feature, key=lambda k: (k in EYE_FEATURES, k)):
            for st in by_feature[k]:
                st['focal'] = k >= 0
            ordered += [st for st in by_feature[k] if stage(st) == s_]
    run('polish', '刻画五官', ordered, 2.)

    set_cap(None)
    # ---- 点睛: each pupil's darkest touch, then the catchlights lifted clean with the eraser's point
    touches = []
    for ex, ey, rx, ry in eyes:
        box = ellipse_mask((ex, ey, rx, ry))
        dark = box & (ndimage.gaussian_filter(vt, 2.5) < np.percentile(ndimage.gaussian_filter(vt, 2.5)[box], 30))
        y, x = ndimage.center_of_mass(dark)                # the pupil sits in the middle of the iris
        touches.append(stroke([(x - 2.5, y - 1), (x, y + 2), (x + 2.5, y), (x, y - 2), (x - 1, y)], 2.4, 'graphite', rng,
                              grade='6B', pressure=.9, focal=True, eye=True, tremor=.03))
        iris = box & (ndimage.gaussian_filter(vt, 4.) < np.percentile(vt[box], 60))
        spots, n = ndimage.label(iris & (ndimage.gaussian_filter(vt, .5) > ndimage.gaussian_filter(vt, 5.) + .1))
        sizes = ndimage.sum_labels(np.ones((H, W)), spots, index=np.arange(1, n + 1)) if n else []
        for k in np.argsort(sizes)[::-1][:2]:
            if sizes[k] < 2:
                continue
            cy, cx = ndimage.center_of_mass(spots == k + 1)
            r = float(np.clip(math.sqrt(sizes[k] / math.pi), 1.3, 2.2))
            touches.append(stroke([(cx - .5, cy - .3), (cx + .1, cy + .2), (cx + .5, cy)], r, 'eraser', rng, opacity=1.,
                                  focal=True, eye=True, press=(.9, .85), tremor=.0))
    touches.sort(key=lambda st: st['medium'] == 'eraser')        # both pupils, then the catchlights
    run('dianjing', '点睛', touches, 0., laid=False)

    # ---- 提亮: the hair's shine comes back out of the mass: the kneaded eraser first, broad along the flow, then
    # a few long strands with the eraser pen
    excess_h = ndimage.gaussian_filter(cov() - B(3.), 5.)
    run('lift', '提亮', scrub(hair & (excess_h > .05), rng, tip(), R=9., length=(60, 150), flow=flow_hair, opacity=.5,
                            clip='hair', step=17., medium='eraser', excess=excess_h), 4.5)
    run('lift', '提亮', strands(B(1.2), hair, flow_hair, rng, tip(), R=1.5, spacing=5., length=(50, 260), smooth=8.,
                              eraser=True, thr=.06, opacity=.8, clip='hair', gain=1., sigma=2., min_len=36.,
                              bridge=30.), 0.)
    excess = ndimage.gaussian_filter(cov() - B(1.5), 2.)
    run('lift', '提亮', scrub(skin & (excess > .015), rng, tip(), R=6., length=(10, 24), flow=flow_form, opacity=.45,
                            clip='skin', step=8., medium='eraser', excess=excess), 0.)

    # ---- 收尾: one more close look at every area; the deepest darks go in last
    run('finish', '收尾', strands(B(1.), hair, flow_hair, rng, tip(), grade='6B', R=1.4, spacing=3., length=(40, 260),
                                smooth=4., clip='hair', gain=.95, thr=.1, section=200., bridge=24., min_len=30., sigma=1.6), 6.5)
    set_cap(B(.6) + .02)
    run('finish', '收尾', hatch(B(1.2), clothes, rng, tip(), grade='2B', R=.95, spacing=2.8, angle=FLICK, smooth=5.,
                              clip='clothes', patch=(70, 100), thr=.08, min_len=14., bridge=8.), 0.)
    run('finish', '收尾', hatch(B(2.), bg, rng, tip(), grade='2B', R=1., spacing=3.6, angle=FLICK, smooth=9.,
                              clip='bg', patch=(90, 130), thr=.08, min_len=16., bridge=8.), 0.)
    set_cap(None)
    run('finish', '收尾', touch_up(B(.6), ndimage.binary_dilation(skin, iterations=12), rng, tip(), thr=.16, min_px=12), 0.)
    if signature:
        text, box = signature
        sig = paint.signature_strokes(text, box, (.3, .3, .3), rng)
        for st in sig:
            st.update(medium='graphite', grade='2B', pressure=.8, R=1.25, tremor=.05)
        run('sign', '签名', sig, 2.5, laid=False)
    Image.fromarray((np.clip(surf.canvas, 0, 1) * 255).astype(np.uint8)).save(out / 'final.png')
    T = np.minimum(ref, ground)
    err = np.abs(ndimage.gaussian_filter(lum(surf.canvas), 2) - ndimage.gaussian_filter(lum(T), 2))
    print('mean |value error| (2px blur):', ' '.join(f'{n} {err[AREAS[n]].mean():.3f}' for n in ('skin', 'hair', 'clothes', 'bg')))
    sgn = ndimage.gaussian_filter(lum(surf.canvas), 2) - ndimage.gaussian_filter(lum(T), 2)
    print('mean signed value error (+ too light):', ' '.join(f'{n} {sgn[AREAS[n]].mean():+.3f}' for n in ('skin', 'hair', 'clothes', 'bg')),
          f'features {sgn[feats].mean():+.3f} |{np.abs(sgn[feats]).mean():.3f}|')
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
