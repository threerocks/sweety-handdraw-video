"""v5 塑形写生: the v4 order (lines, face, hair, features, clothes, background, finishing), painted the way a hand does.

What v4 got wrong (user, 2026-11-11: 笔画没有轻重、深浅、不规则感; 过程局部不完整、断断续续):
- its planes were ruled with dead-straight, equal strokes, one colour mix at a time, so a face grew as scattered patches
  with bare holes between them;
- strokes were cut off by pixel area masks, so every silhouette was a staircase of square stroke ends;
- every stroke of a pass had the same weight.

v5:
- a form grows: the planes of an area are painted as a growing front (the darkest plane first, then whichever unpainted
  neighbour is darkest and nearest), each plane finished (its edge cut in, its inside filled, its last bare specks
  closed) before the brush moves to the next. At any pause the painted part is one coherent shape.
- an edge is drawn, not cut out: where a plane meets its area's outline, a smaller brush first runs along the outline
  (cut-in strokes); the fill strokes then end inside that band. Area masks are smoothed first.
- a stroke has a hand in it: it follows the target's own brush direction (curved streamlines), its width breathes, it
  lands and lifts, strokes differ in pressure, load, length and a little in direction; some carry two tones; light
  dragging strokes pull one plane into the next where the form turns softly.
"""
import json
import os
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

import paint
import planes
from paint import (H, W, MEDIA, orientation, REGIONS, AREAS, FEATURES, EYE_FEATURES, LINE_FEATURES, rgb2lab, lab2rgb, kmeans,
                   ellipse_mask, focus_map, painter_order, render, shade, canvas_tooth, load_subject, sketch_strokes,
                   line_strokes, eye_touches, signature_strokes, tag_feature, plan_layer, catmull)
from planes import GROUND, area_masks, outline_band, bare_holes, v2_spec, PRESETS

CLIP_GROW, CLIP_FEATHER = 2.5, 1.5


def smooth_areas(ref=None):
    """Area outlines a brush could follow: no pixel teeth, no one-pixel spurs. With the target: where hair and skin
    differ clearly in value, the border between them is the target's own edge (the area map is drawn separately and
    sits a few pixels off, which leaves a band of 'skin' in hair colours beside the cheek)."""
    if ref is not None and AREAS['hair'].any() and AREAS['skin'].any():
        L = rgb2lab(ndimage.gaussian_filter(ref, (1.5, 1.5, 0)))[..., 0]
        lh, ls = np.median(L[AREAS['hair']]), np.median(L[AREAS['skin']])
        if abs(lh - ls) > 14:
            hairlike = np.abs(L - lh) < np.abs(L - ls) * .8
            skinlike = np.abs(L - ls) < np.abs(L - lh) * .8
            feats = AREAS.get('features', np.zeros((H, W), bool))
            band = ndimage.binary_dilation(AREAS['hair'], iterations=12) & ndimage.binary_dilation(AREAS['skin'], iterations=12) & ~feats
            to_hair = band & AREAS['skin'] & hairlike
            to_skin = band & AREAS['hair'] & skinlike
            # only what stays attached to its new area (a dark nostril near the fringe is not hair)
            lab_h, _ = ndimage.label(AREAS['hair'] | to_hair)
            keep_h = np.isin(lab_h, np.unique(lab_h[AREAS['hair'] & ~to_skin]))
            to_hair &= keep_h
            AREAS['hair'] = (AREAS['hair'] | to_hair) & ~to_skin
            AREAS['skin'] = (AREAS['skin'] | to_skin) & ~to_hair
    sm = {n: ndimage.gaussian_filter(AREAS[n].astype(np.float32), 1.8) for n in ('hair', 'skin', 'clothes', 'bg')}
    stack = np.stack([sm[n] for n in ('hair', 'skin', 'clothes', 'bg')])
    best = stack.argmax(0)
    for k, n in enumerate(('hair', 'skin', 'clothes', 'bg')):
        AREAS[n] = best == k
    person = ~AREAS['bg']
    face = ellipse_mask(REGIONS['face'])
    AREAS.update(person=person, outline=outline_band(person),
                 face=(face | (AREAS['skin'] & ellipse_mask(REGIONS['face'], 1.25))) & person & ~AREAS['hair'])


def trace_contours(mask, min_len=24):
    """The outline of a mask as ordered, smoothed polylines with inward normals. Pixels on the picture's border are
    left out (nobody cuts in along the canvas edge)."""
    edge = mask & ~ndimage.binary_erosion(mask, np.ones((3, 3), bool), border_value=1)
    edge[0], edge[-1], edge[:, 0], edge[:, -1] = False, False, False, False
    inner = ndimage.gaussian_filter(mask.astype(np.float32), 3.)
    gy, gx = np.gradient(inner)
    todo = {(int(x), int(y)) for y, x in zip(*np.nonzero(edge))}
    nb = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    chains = []
    while todo:
        start = todo.pop()
        chain = [start]
        for _ in range(2):                                 # walk one way, then the other from the start
            cur, prev = chain[-1], None
            while True:
                opts = [(cur[0] + dx, cur[1] + dy) for dx, dy in nb if (cur[0] + dx, cur[1] + dy) in todo]
                if not opts:
                    break
                if prev is not None:
                    d = (cur[0] - prev[0], cur[1] - prev[1])
                    opts.sort(key=lambda o: -((o[0] - cur[0]) * d[0] + (o[1] - cur[1]) * d[1]) / math.hypot(o[0] - cur[0], o[1] - cur[1]))
                nxt = opts[0]
                todo.discard(nxt)
                chain.append(nxt)
                prev, cur = cur, nxt
            chain.reverse()
        if len(chain) >= min_len:
            p = ndimage.uniform_filter1d(np.asarray(chain, float), 9, axis=0, mode='nearest')
            xi, yi = np.clip(np.rint(p[:, 0]).astype(int), 0, W - 1), np.clip(np.rint(p[:, 1]).astype(int), 0, H - 1)
            n = np.stack([gx[yi, xi], gy[yi, xi]], 1)
            n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
            n = ndimage.uniform_filter1d(n, 9, axis=0, mode='nearest')
            n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
            chains.append((p, n))
    return chains


def quantise(lab, mask, R, k, rng, fill=False, split=7.):
    """The target's colours inside mask as k mixes; each connected patch of one mix is a plane. Returns the plane id
    map (-1 outside), and per plane its mix; slivers too small for a stroke join their nearest neighbour."""
    wv = np.array([1.35, 1., 1.])
    pts = lab[mask] * wv
    centres, _ = kmeans(pts[rng.choice(len(pts), min(len(pts), 30000), replace=False)], k, rng)
    lbl = np.full((H, W), -1)
    flat = np.flatnonzero(mask.ravel())
    for i in range(0, len(flat), 200000):
        chunk = flat[i:i + 200000]
        lbl.ravel()[chunk] = np.argmin(((lab.reshape(-1, 3)[chunk, None] * wv - centres[None]) ** 2).sum(-1), 1)
    size = max(3, int(R * .8) | 1)
    votes = np.stack([ndimage.uniform_filter((lbl == c).astype(np.float32), size) for c in range(len(centres))])
    lbl = np.where(mask, votes.argmax(0), -1)
    min_area = (.8 if fill else 1.4) * R * R
    for _ in range(2):
        small = np.zeros((H, W), bool)
        for c in range(len(centres)):
            comp, n = ndimage.label(lbl == c)
            if n:
                sizes = ndimage.sum_labels(np.ones((H, W)), comp, index=np.arange(1, n + 1))
                small |= np.isin(comp, np.flatnonzero(sizes < min_area) + 1)
        keep = mask & ~small
        if not small.any() or not keep.any():
            break
        _, (iy, ix) = ndimage.distance_transform_edt(~keep, return_indices=True)
        lbl = np.where(small, lbl[iy, ix], lbl)
    pid = np.full((H, W), -1, np.int32)
    mixes = []
    for c in range(len(centres)):
        comp, n = ndimage.label(lbl == c)
        pid[comp > 0] = comp[comp > 0] - 1 + len(mixes)
        mixes += [c] * n
    # a long plane (the shadow rim round a face, a hair mass) is worked a stretch at a time: cut it into pieces about
    # five brush widths long, so the brush finishes what is beside it before it follows the plane to the far side
    if split:
        nxt = len(mixes)
        for i, sl in enumerate(ndimage.find_objects(pid + 1)):
            if sl is None:
                continue
            ext = math.hypot(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start)
            if ext <= split * R:
                continue
            ys, xs = np.nonzero(pid[sl] == i)
            P = np.c_[xs, ys].astype(float)
            m = int(math.ceil(ext / (split * .7 * R)))
            cen, _ = kmeans(P[rng.choice(len(P), min(len(P), 4000), replace=False)], m, rng)
            part = np.argmin(((P[:, None] - cen[None]) ** 2).sum(-1), 1) if len(P) * m < 4e7 else np.zeros(len(P), int)
            for c in range(1, m):
                sel = part == c
                if sel.any():
                    pid[sl][ys[sel], xs[sel]] = nxt
                    mixes.append(mixes[i])
                    nxt += 1
    return pid, np.array(mixes, int), centres / wv


def stroke_of(pts, R, colour_lab, mix, mix_lab, rng, medium, soft=0., light=False, cut=False):
    """One brush stroke with its own touch: pressure, load, breathing width, sometimes two tones on the brush."""
    touch = float(np.clip(rng.lognormal(0, .25), .65, 1.45))
    lab = np.array(colour_lab, float) + rng.normal(0, 1., 3) * np.array([.9, .7, .7]) * (1. if R > 12 else .4)
    st = dict(points=[(float(a), float(b)) for a, b in pts], R=float(R * (.86 + .16 * min(touch, 1.25))),
              rgb=[float(x) for x in lab2rgb(lab)], mix=int(mix), mix_rgb=[float(x) for x in lab2rgb(mix_lab)], medium=medium,
              seed=int(rng.integers(1 << 30)), load=float(rng.uniform(.75, 1.3)),
              press=(float(rng.uniform(.5, .9)), float(rng.uniform(.25, .7))), press_fixed=True,
              soft=float(np.clip(soft + rng.uniform(-.08, .12), 0, 1)), pvar=.11, wobble=.9 if R > 8 else .6,
              body=float((.4 + .35 * touch) * min(1., R / 14.) ** .5))
    if cut:                                               # along an outline: a steady, well loaded stroke
        st.update(press=(.92, .85), wobble=.5, pvar=.04, load=1.6, body=.3)
    elif light:                                           # a light drag over wet paint
        st.update(medium='drag', load0=float(rng.uniform(.5, .64)), press=(float(rng.uniform(.4, .6)), float(rng.uniform(.15, .3))),
                  body=.5, soft=max(st['soft'], .45))
    elif rng.random() < .3:                               # a brush that picked up two neighbouring tones
        st['tone2'] = [float(x) for x in rng.choice([-1, 1]) * rng.uniform(.025, .05) * np.array([1., .95, .88])]
    return st


def fill_strokes(pm, box, R, tx, ty, lab, rng, cov, area, reach=(2.5, 6.), rho=1.25, medium='bristle', soft=0., mix=0,
                 mix_lab=None, brk=11., tip=None):
    """Cover one plane (bool mask pm, bbox box) with strokes along the flow, laid side by side across it. A stroke may
    run on past the plane while the colour under it stays its own (a brush does not stop at a contour line of the
    model), inside the area; cov (the whole area's coverage so far) keeps later planes from repainting it."""
    y0, y1, x0, x1 = box
    sub = pm[y0:y1, x0:x1]
    ys, xs = np.nonzero(sub)
    if not len(ys):
        return []
    fx, fy = tx[y0:y1, x0:x1][sub], ty[y0:y1, x0:x1][sub]
    phi = math.atan2((2 * fx * fy).mean(), (fx * fx - fy * fy).mean()) / 2
    u = np.array([math.cos(phi), math.sin(phi)])
    v = np.array([-u[1], u[0]])
    step = max(2, int(R / 2.5))
    grid = (ys % step == 0) & (xs % step == 0)
    if not grid.any():
        grid = np.zeros(len(ys), bool)
        grid[len(ys) // 2] = True
    P = np.c_[xs[grid] + x0, ys[grid] + y0].astype(float)
    row = np.floor((P @ v) / (rho * R))
    along = P @ u
    order = np.lexsort((np.where(row % 2 == 0, along, -along), row))    # side by side, back and forth
    hstep = max(2., R * .5)
    out = []
    rr = int(math.ceil(rho * R))
    oy, ox = np.mgrid[-rr:rr + 1, -rr:rr + 1]
    disk = ox ** 2 + oy ** 2 <= (rho * R) ** 2

    def stamp(p):
        cx, cy = int(round(p[0])), int(round(p[1]))
        ya, yb, xa, xb = cy - rr, cy + rr + 1, cx - rr, cx + rr + 1
        sy0, sx0 = max(0, -ya), max(0, -xa)
        ya, xa = max(0, ya), max(0, xa)
        yb, xb = min(H, yb), min(W, xb)
        if yb > ya and xb > xa:
            cov[ya:yb, xa:xb] |= disk[sy0:sy0 + yb - ya, sx0:sx0 + xb - xa]

    def walk(p, d, maxlen, col0, force):
        pts, length, turned, over, out_ = [], 0., 0., 0., 0.
        while length < maxlen:
            xi, yi = min(W - 1, max(0, int(round(p[0])))), min(H - 1, max(0, int(round(p[1]))))
            f = np.array([tx[yi, xi], ty[yi, xi]])
            if f @ d < 0:
                f = -f
            nd = .55 * f + .45 * d
            nd /= np.linalg.norm(nd) + 1e-9
            turned += math.acos(max(-1., min(1., nd @ d)))
            if turned > math.radians(75):
                break
            q = p + hstep * nd
            if q[0] < -R or q[1] < -R or q[0] >= W + R or q[1] >= H + R:
                break
            inside = 0 <= q[0] < W and 0 <= q[1] < H
            if inside:                                     # (past the canvas edge the paint simply runs off)
                qy, qx = int(q[1]), int(q[0])
                if not area[qy, qx]:
                    break
                if length >= force:
                    if np.linalg.norm(lab[qy, qx] - col0) > brk:
                        break
                    if not pm[qy, qx]:                     # a little way into the neighbours, not a finger across them
                        out_ += hstep
                        over += hstep * bool(cov[qy, qx])
                        if over > .7 * R or out_ > 2.5 * R:
                            break
            p, d = q, nd
            pts.append(p.copy())
            length += hstep
        return pts

    Pi = P.astype(int)
    tip = np.asarray(tip, float) if tip is not None else P[order[0]]
    while True:                                            # the next stroke starts beside the last one, wherever that is
        free = ~cov[Pi[:, 1], Pi[:, 0]]
        if not free.any():
            break
        idx = np.flatnonzero(free)
        if len(out) % 5 == 0:                              # how much paint already surrounds each spot: pockets and
            ya, yb, xa, xb = max(0, y0 - 2 * rr), min(H, y1 + 2 * rr), max(0, x0 - 2 * rr), min(W, x1 + 2 * rr)   # notches first
            dens = ndimage.uniform_filter(cov[ya:yb, xa:xb].astype(np.float32), int(3.5 * R) | 1)[Pi[:, 1] - ya, Pi[:, 0] - xa]
        j = idx[np.argmin(np.hypot(P[idx, 0] - tip[0], P[idx, 1] - tip[1]) - 7. * R * dens[idx])]
        p = P[j].copy()
        yi, xi = int(p[1]), int(p[0])
        d = np.array([tx[yi, xi], ty[yi, xi]])
        if d @ u < 0:
            d = -d
        if len(out) % 2:
            d = -d
        lmax = rng.uniform(*reach) * R
        col0 = lab[yi, xi]
        fwd = walk(p, d, lmax * rng.uniform(.4, .6), col0, 1.2 * R)
        back = walk(p, -d, lmax - len(fwd) * hstep, col0, 1.2 * R)
        pts = back[::-1] + [p] + fwd
        if len(pts) < 3:                                   # a dab: still a short pull, not a dot
            pts = [p - d * R * .9, p, p + d * R * .9]
        for q in pts:
            stamp(q)
        g = np.array(pts)
        gi = np.clip(np.rint(g).astype(int), 0, [W - 1, H - 1])
        col = np.median(lab[gi[:, 1], gi[:, 0]], 0)
        keep = np.r_[0, np.arange(2, len(g) - 1, 2), len(g) - 1] if len(g) > 4 else np.arange(len(g))
        if len(g) > 4:                                     # a pull of the arm is smooth: no tremor from the flow field
            g = np.vstack([g[:1], ndimage.uniform_filter1d(g, 5, axis=0, mode='nearest')[1:-1], g[-1:]])
        out.append(stroke_of(g[keep], R, col, mix, mix_lab if mix_lab is not None else col, rng, medium, soft))
        cov[yi, xi] = True
        tip = g[0] if np.hypot(*(g[0] - tip)) > np.hypot(*(g[-1] - tip)) else g[-1]   # it is pulled from the nearer end
    return out


def form_strokes(ref, canvas, height, thick, tooth, area, mask, R, k, rng, T=0., start_tip=(0, 0), soft=0., medium='bristle',
                 reach=(2.5, 6.), cutin=True, drags=True, contours=None, deep=None, max_len=None, spacing=None, axis=False,
                 brk=11., dark=.15):
    """Paint an area form by form, rendering as it goes (each plane closes its own bare specks before the next). Returns
    the strokes in painting order. T > 0: a correcting pass (only where the canvas is still off by more than T)."""
    lab = rgb2lab(ndimage.gaussian_filter(ref, (R * .3, R * .3, 0)))
    if mask.sum() < 50:
        return []
    lab_fine = rgb2lab(ndimage.gaussian_filter(ref, (1.5, 1.5, 0)))
    tx, ty = orientation(lab[..., 0], R, rng)
    pid, mixes, centres = quantise(lab, mask, R, k, rng)
    n = len(mixes)
    if not n:
        return []
    objs = ndimage.find_objects(pid + 1)
    info = []
    for i, sl in enumerate(objs):
        if sl is None:
            info.append(None)
            continue
        box = (max(0, sl[0].start - 2), min(H, sl[0].stop + 2), max(0, sl[1].start - 2), min(W, sl[1].stop + 2))
        sub = pid[box[0]:box[1], box[2]:box[3]] == i
        ys, xs = np.nonzero(sub)
        grown = ndimage.binary_dilation(sub, iterations=3)
        near = np.unique(pid[box[0]:box[1], box[2]:box[3]][grown & ~sub])
        info.append(dict(box=box, cen=np.array([xs.mean() + box[2], ys.mean() + box[0]]), L=centres[mixes[i]][0],
                         near=[int(a) for a in near if a >= 0], px=len(ys),
                         width=float(ndimage.distance_transform_edt(sub).max())))
    # which stretch of the area's outline each plane owns
    runs = {}
    rcut = float(np.clip(R * .42, 4., 11.))
    if cutin and contours and not T:
        for p, nrm in contours:
            q = p + nrm * rcut * 1.2
            qi = np.clip(np.rint(q).astype(int), 0, [W - 1, H - 1])
            own = pid[qi[:, 1], qi[:, 0]]
            own = ndimage.median_filter(own, 9, mode='nearest')
            cuts = np.flatnonzero(np.diff(own) != 0) + 1
            segs = np.split(np.arange(len(p)), cuts)
            merged = []                                    # a short stretch belongs to the stroke before it
            for seg in segs:
                if merged and (len(seg) < 4 * rcut or own[seg[0]] < 0 or own[merged[-1][0]] < 0 or len(merged[-1]) < 4 * rcut):
                    if own[merged[-1][0]] < 0 or (len(seg) > len(merged[-1]) and own[seg[0]] >= 0):
                        own[merged[-1]] = own[seg[0]]
                    merged[-1] = np.r_[merged[-1], seg]
                else:
                    merged.append(seg)
            for seg in merged:
                if own[seg[0]] >= 0 and len(seg) >= 2 * rcut:
                    runs.setdefault(int(own[seg[0]]), []).append((p[seg] + nrm[seg] * rcut * .8, p[seg]))
    Lr = np.array([i['L'] if i else 0 for i in info])
    lo, hi = Lr.min(), Lr.max() + 1e-6
    todo = {i for i in range(n) if info[i]}
    done = set()
    tip = np.asarray(start_tip, float)
    strokes = []
    can_lab = rgb2lab(canvas) if T else None
    cov = np.zeros((H, W), bool)
    reachable = ndimage.binary_dilation(mask, iterations=max(1, int(R * .25)))

    def lay(st):
        st['clip'] = area
        render(st, canvas, height, thick, tooth)
        strokes.append(st)

    while todo:
        front = [i for i in todo if any(j in done for j in info[i]['near'])] if not T else list(todo)
        cand = front or list(todo)
        if T:
            i = min(cand, key=lambda i: np.hypot(*(info[i]['cen'] - tip)))
        else:                                              # darkest first, but never far from where the brush is
            i = min(cand, key=lambda i: (info[i]['L'] - lo) / (hi - lo) * (dark if front else 3.) + np.hypot(*(info[i]['cen'] - tip)) / 300.)
        todo.discard(i)
        done.add(i)
        box = info[i]['box']
        y0, y1, x0, x1 = box
        sub = pid[y0:y1, x0:x1] == i
        pm = np.zeros((H, W), bool)
        pm[y0:y1, x0:x1] = sub
        if T and np.linalg.norm(lab[pm] - can_lab[pm], axis=1).mean() < T * .75:
            continue
        if not T and info[i]['px'] < .6 * R * R:           # a scrap (skin between strands of a fringe): the finer pass's work
            continue
        mix, mix_lab = mixes[i], centres[mixes[i]]
        Rp = float(np.clip(2.2 * info[i]['width'], min(R, max(3., R * .6)), R))
        mine = []
        # 1. the plane's stretch of the outline: cut in along it with a smaller brush
        for path, path0 in sorted(runs.get(i, []), key=lambda p: np.hypot(*(p[0][0] - tip))):
            arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
            a = 0.
            while a < arc[-1] - 2:
                b = min(arc[-1], a + rng.uniform(8, 16) * rcut)
                if arc[-1] - b < 4 * rcut:
                    b = arc[-1]
                sel = np.flatnonzero((arc >= a - rcut * 1.5) & (arc <= b))
                if len(sel) >= 3:
                    seg = path[sel]
                    gi = np.clip(np.rint(seg).astype(int), 0, [W - 1, H - 1])
                    ci = np.clip(np.rint(seg + (seg - path0[sel]) * 1.2).astype(int), 0, [W - 1, H - 1])
                    col = .6 * mix_lab + .4 * np.median(lab_fine[ci[:, 1], ci[:, 0]], 0)
                    if not T or np.linalg.norm(lab[gi[:, 1], gi[:, 0]] - can_lab[gi[:, 1], gi[:, 0]], axis=1).mean() >= T:
                        keep = np.r_[0, np.arange(3, len(seg) - 1, 3), len(seg) - 1]
                        mine.append(dict(stroke_of(seg[keep], min(rcut, Rp), col, mix, mix_lab, rng, medium, soft, cut=True), cutin=1))
                a = b
        # 2. the inside, side by side along the form
        fill = fill_strokes(pm, box, Rp, tx, ty, lab, rng, cov, reachable, reach=reach, brk=brk, tip=tip, medium=medium, soft=soft, mix=mix, mix_lab=mix_lab,
                            rho=spacing or 1.25)
        if max_len:                                        # presets with short strokes (tiles, knife)
            short = []
            for st in fill:
                p = catmull(st['points'], step=2.)
                arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
                m = max(1, int(math.ceil(arc[-1] / (max_len * 2 * Rp))))
                for a, b in zip(np.linspace(0, arc[-1], m + 1)[:-1], np.linspace(0, arc[-1], m + 1)[1:]):
                    sel = np.flatnonzero((arc >= a) & (arc <= b))
                    if len(sel) >= 3:
                        short.append(dict(st, points=[(float(x), float(y)) for x, y in p[sel[::max(1, len(sel) // 4)]]],
                                          seed=int(rng.integers(1 << 30))))
            fill = short
        if axis:
            for st in fill:
                p = np.array(st['points'])
                d = p[-1] - p[0]
                mid = p.mean(0)
                half = np.array([abs(d[0]) / 2, 0.]) if abs(d[0]) >= abs(d[1]) else np.array([0., abs(d[1]) / 2])
                half = np.maximum(half, [Rp * .5, 0] if half[0] else [0, Rp * .5])
                st['points'] = [tuple(float(x) for x in mid - half), tuple(float(x) for x in mid), tuple(float(x) for x in mid + half)]
                st['wobble'] = 0
        if T:
            keepf = []
            for st in fill:
                gi = np.clip(np.rint(catmull(st['points'], step=3.)).astype(int), 0, [W - 1, H - 1])
                if np.linalg.norm(lab[gi[:, 1], gi[:, 0]] - can_lab[gi[:, 1], gi[:, 0]], axis=1).mean() >= T:
                    keepf.append(st)
            fill = keepf
        def close(where, tip):
            holes = bare_holes(where, thick, min_px=5)
            if holes.sum() <= 12:
                return tip
            hm = ndimage.binary_dilation(holes, iterations=2) & pm
            pool = fill_strokes(hm, box, float(max(4., Rp * .6)), tx, ty, lab, rng, np.zeros((H, W), bool), reachable,
                                reach=(1.5, 3.5), medium=medium, soft=soft, mix=mix, mix_lab=mix_lab, rho=1.)
            while pool:                                    # nearest first: the hand does not hop about for specks
                k_ = int(np.argmin([min(np.hypot(*(np.array(s_['points'][0]) - tip)), np.hypot(*(np.array(s_['points'][-1]) - tip)))
                                    for s_ in pool]))
                s_ = pool.pop(k_)
                if np.hypot(*(np.array(s_['points'][-1]) - tip)) < np.hypot(*(np.array(s_['points'][0]) - tip)):
                    s_['points'] = s_['points'][::-1]
                s_['press'] = (.9, .8)
                lay(s_)
                tip = np.array(s_['points'][-1])
            return tip

        # the edge is cut in where the brush is working: each stretch of outline just before the first fill stroke that
        # comes near it (not the whole outline first, as in a colouring book)
        pend = [(np.array(st['points']), st) for st in mine]
        seq = []
        for st in fill:
            if pend:
                p = np.array(st['points'])
                reach_ = 2.5 * st['R'] + 2 * rcut
                still = []
                for q, c in pend:
                    d = np.hypot(q[:, None, 0] - p[None, :, 0], q[:, None, 1] - p[None, :, 1]).min()
                    (seq if d < reach_ else still).append(c) if d < reach_ else still.append((q, c))
                pend = still
            seq.append(st)
        seq += [c for _, c in pend]
        y0b, y1b, x0b, x1b = max(0, y0 - 40), min(H, y1 + 40), max(0, x0 - 40), min(W, x1 + 40)
        since, recent = 0, []
        for st in seq:
            a, b = np.array(st['points'][0]), np.array(st['points'][-1])
            if np.hypot(*(b - tip)) < np.hypot(*(a - tip)):          # start at the nearer end: back and forth
                st['points'] = st['points'][::-1]
                st['press'] = tuple(st['press'][::-1]) if st.get('cutin') else st['press']
            lay(st)
            tip = np.array(st['points'][-1])
            since += 1
            recent.append(st)
            if not T and since >= 14:                      # slivers left between strokes are closed as the brush goes
                since = 0
                pts_ = np.vstack([np.array(s_['points']) for s_ in recent[-14:]])
                ya, yb = int(max(y0b, pts_[:, 1].min() - 2 * Rp)), int(min(y1b, pts_[:, 1].max() + 2 * Rp))
                xa, xb = int(max(x0b, pts_[:, 0].min() - 2 * Rp)), int(min(x1b, pts_[:, 0].max() + 2 * Rp))
                if yb - ya > 4 and xb - xa > 4:
                    done_ = thick[ya:yb, xa:xb] >= .6
                    it = max(1, int(Rp * .5))
                    swept = ndimage.binary_erosion(ndimage.binary_dilation(done_, iterations=it), iterations=it + 1)
                    gaps = np.zeros((H, W), bool)
                    gaps[ya:yb, xa:xb] = swept & ~done_
                    tip = close(gaps & pm, tip)
        # 3. what is still bare inside this plane is closed now, with a smaller brush, before moving on
        if not T:
            tip = close(pm & (deep if deep is not None else True), tip)
            # 4. now and then a light drag across the edge to a neighbour already painted, where the form turns softly
            if drags and Rp > 7:
                for j in info[i]['near']:
                    if j not in done or j == i or rng.random() > .55:
                        continue
                    dE = np.linalg.norm(centres[mixes[j]] - mix_lab)
                    if not 3.5 < dE < 15:
                        continue
                    bj = info[j]['box']
                    yy0, yy1, xx0, xx1 = min(y0, bj[0]), max(y1, bj[1]), min(x0, bj[2]), max(x1, bj[3])
                    a_ = pid[yy0:yy1, xx0:xx1] == i
                    edge = ndimage.binary_dilation(a_, iterations=2) & (pid[yy0:yy1, xx0:xx1] == j)
                    ey, ex = np.nonzero(edge)
                    if len(ey) < 8:
                        continue
                    g = ndimage.gaussian_filter(a_.astype(np.float32), max(2., Rp * .4))
                    gy_, gx_ = np.gradient(g)
                    for _ in range(int(rng.integers(1, 3 + (len(ey) > 60)))):
                        e = rng.integers(len(ey))
                        nv = np.array([gx_[ey[e], ex[e]], gy_[ey[e], ex[e]]])
                        if np.linalg.norm(nv) < 1e-6:
                            continue
                        nv /= np.linalg.norm(nv)
                        ang = rng.normal(0, .5)
                        nv = np.array([nv[0] * math.cos(ang) - nv[1] * math.sin(ang), nv[0] * math.sin(ang) + nv[1] * math.cos(ang)])
                        b0 = np.array([ex[e] + xx0, ey[e] + yy0], float)
                        r3 = Rp * rng.uniform(.5, .75)
                        pts = [b0 + nv * r3 * 1.6, b0, b0 - nv * r3 * rng.uniform(1.2, 2.2)]
                        if AREAS.get('features') is not None and AREAS['features'][int(min(H - 1, max(0, b0[1]))), int(min(W - 1, max(0, b0[0])))]:
                            continue
                        lay(stroke_of(pts, r3, mix_lab, mix, mix_lab, rng, medium, soft, light=True))
                        tip = np.array(pts[-1])
    return strokes


def join_lines(strokes, gap=14., angle=40.):
    """The underdrawing as whole lines: fragments whose ends meet and run on in the same direction are one pull of the
    pencil; then the long contours first, each followed by the lines nearest its end."""
    lines = [np.asarray(st['points'], float) for st in strokes]
    changed = True
    while changed:
        changed = False
        for i in range(len(lines)):
            if lines[i] is None or len(lines[i]) < 2:
                continue
            for j in range(len(lines)):
                if i == j or lines[j] is None or len(lines[j]) < 2:
                    continue
                best = None
                for ri in (False, True):
                    for rj in (False, True):
                        a = lines[i][::-1] if ri else lines[i]
                        b = lines[j][::-1] if rj else lines[j]
                        d = np.hypot(*(b[0] - a[-1]))
                        if d > gap:
                            continue
                        da = a[-1] - a[max(0, len(a) - 4)]
                        db = b[min(len(b) - 1, 3)] - b[0]
                        cosang = (da @ db) / (np.linalg.norm(da) * np.linalg.norm(db) + 1e-9)
                        if cosang > math.cos(math.radians(angle)) and (best is None or d < best[0]):
                            best = (d, a, b)
                if best:
                    lines[i] = np.vstack([best[1], best[2]])
                    lines[j] = None
                    changed = True
    lines = [l for l in lines if l is not None]
    length = [np.linalg.norm(np.diff(l, axis=0), axis=1).sum() for l in lines]
    out, left = [], list(range(len(lines)))
    tip = np.array([W * .3, 0.])
    big = sorted([i for i in left if length[i] > 260], key=lambda i: -length[i])
    order = []
    for group in (big, [i for i in left if i not in big]):
        pool = list(group)
        while pool:
            d = [min(np.hypot(*(lines[i][0] - tip)), np.hypot(*(lines[i][-1] - tip))) for i in pool]
            i = pool.pop(int(np.argmin(d)))
            if np.hypot(*(lines[i][-1] - tip)) < np.hypot(*(lines[i][0] - tip)):
                lines[i] = lines[i][::-1]
            order.append(i)
            tip = lines[i][-1]
    base = strokes[0]
    rng = np.random.default_rng(paint.SEED + 5)
    for i in order:
        if length[i] < 8:
            continue
        out.append(dict(base, points=[(float(x), float(y)) for x, y in lines[i]], R=float(1.5 + .35 * rng.random()),
                        seed=int(rng.integers(1 << 30)), opacity=float(.78 + .22 * rng.random()), wobble=0, pvar=.3,
                        press=(.4, .25)))
    return out


# area by area, as the painter in the reference video: (name, label, kind, args, video budget in seconds)
STEPS_V5 = [
    ('face', '脸部块面', 'forms', dict(area='skin', R=26, k=6, reach=(2.5, 5.5)), 11.),
    ('face2', '脸部块面', 'forms', dict(area='skin', R=10, k=16, T=4.5, reach=(2., 5.), drags=False, brk=6., medium='bristle_flat'), 5.),
    ('hair', '头发', 'forms', dict(area='hair', R=19, k=2, reach=(5., 12.), brk=14., dark=.12, spacing=1.1), 13.),
    ('strands', '头发', 'layer', v2_spec('texture', only=None, exclude=[], area='hair', cover=1.9, R=4.2), 10.),
    ('detail', '五官', 'layer', v2_spec('detail', area='features', skin=False), 3.),
    ('clothes', '衣服', 'forms', dict(area='clothes', R=22, k=7, reach=(4., 9.), dark=.3), 6.),
    ('clothes2', '衣服', 'forms', dict(area='clothes', R=8, k=12, T=6, reach=(3., 8.), drags=False), 3.),
    ('clothes3', '衣服', 'layer', v2_spec('detail', area='clothes', focal=False, skin=False, T=9, gamma=0.), 3.),
    ('background', '背景', 'forms', dict(area='bg', R=32, k=6, reach=(3., 7.), dark=.25), 9.),
    ('background2', '背景', 'forms', dict(area='bg', R=13, k=9, T=6, reach=(2.5, 6.), drags=False), 3.),
    ('edges', '收边', 'layer', v2_spec('texture', only=None, exclude=[], area='outline', own=1., mixes=24, T=6, gamma=0.,
                                      R=3.4, cover=1.1), 5.),
    ('sheen', '提亮', 'layer', v2_spec('texture', only=None, exclude=[], area='hair', lighten=3., T=6.5, R=3.2, mixes=8, own=.9,
                                      cover=2.2, press=(.3, .1)), 4.),
    ('face3', '深入', 'forms', dict(area='face', R=5.5, k=20, T=4.5, reach=(2., 5.), drags=False, cutin=False, brk=5.,
                                  medium='bristle_flat'), 4.),
    ('polish', '深入', 'layer', v2_spec('polish', area='features', skin=False, feather_after=False), 3.),
    ('scumble', '干擦', 'layer', v2_spec('scumble'), 2.),
    ('accent', '深入', 'layer', v2_spec('accent', area='features'), 2.),
]
PACE_V5 = dict(feature_cap=3.5, eye_life={'polish': 5., 'lines': 4.5})
# passes of many fine strokes laid side by side: filmed as continuous motions of the hand (it stays on the canvas and the
# strokes appear as it passes), not as hundreds of separate dabs sharing a frame
GLIDE = {'strands', 'sheen', 'edges', 'face3', 'clothes3'}
PRESET = 'planes'


def plan_v5(subject_path, sketch_path, out, focus_ellipses, signature=None, eyes=None, photo_path=None):
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
    smooth_areas(ref)
    vis = np.zeros((H, W, 3))
    for name, colour in (('bg', (.85, .85, .85)), ('clothes', (.3, .4, .8)), ('hair', (.35, .2, .1)), ('skin', (.95, .7, .55))):
        vis[AREAS[name]] = colour
    Image.fromarray((vis * 255).astype(np.uint8)).save(out / 'areas.png')
    contours = {}
    for name in {a['area'] for _, _, kind, a, _ in STEPS_V5 if kind == 'forms'}:
        d = ndimage.distance_transform_edt(~AREAS[name])
        paint.CLIPS[name] = np.clip(1 - (d - CLIP_GROW) / CLIP_FEATHER, 0, 1).astype(np.float32)
        Image.fromarray((paint.CLIPS[name] * 255).astype(np.uint8)).save(out / f'clip-{name}.png')
        contours[name] = trace_contours(AREAS[name])
    # hair strands may leave the (smoothed) hair area only onto what the target paints as hair: wisps of a fringe on the
    # forehead yes, the cheek beside the hair no (round stroke ends there scallop the face's edge)
    labt = rgb2lab(ndimage.gaussian_filter(ref, (1.2, 1.2, 0)))
    lh, ls = np.median(labt[AREAS['hair']][:, 0]) if AREAS['hair'].any() else 0., np.median(labt[AREAS['skin']][:, 0]) if AREAS['skin'].any() else 100.
    near = ndimage.binary_dilation(AREAS['hair'], iterations=12)
    hairlike = (np.abs(labt[..., 0] - lh) < np.abs(labt[..., 0] - ls) * .8) if abs(lh - ls) > 12 else np.zeros((H, W), bool)
    strand = AREAS['hair'] | (near & hairlike & AREAS['skin']) | (ndimage.binary_dilation(AREAS['hair'], iterations=8) & ~AREAS['skin'])
    strand = ndimage.gaussian_filter(strand.astype(np.float32), 1.2) > .5
    paint.CLIPS['strand'] = np.clip(1 - (ndimage.distance_transform_edt(~strand) - .5) / 1.5, 0, 1).astype(np.float32)
    Image.fromarray((paint.CLIPS['strand'] * 255).astype(np.uint8)).save(out / 'clip-strand.png')
    passes = []
    started = time.time()

    def record(name, label, strokes, budget):
        passes.append(dict(name=name, label=label, strokes=strokes, budget=budget, group=name in GLIDE))
        Image.fromarray((canvas * 255).astype(np.uint8)).save(out / f'pass-{len(passes):02d}-{name}.png')
        print(f'{name:10s} {len(strokes):5d} strokes  {time.time() - started:6.1f}s', flush=True)
        return strokes[-1]['points'][-1] if strokes else None

    def run(name, label, strokes, budget):
        for st in strokes:
            render(st, canvas, height, thick, tooth)
        return record(name, label, strokes, budget)

    sketch = sketch_strokes(sketch_path, shape, crop, rng) if sketch_path else []
    if sketch:
        lines = join_lines(sketch)
        print(f'underdrawing: {len(sketch)} fragments -> {len(lines)} lines', flush=True)
        run('sketch', '起稿', lines, 5.)
    tip = (W * .5, H * .3)
    pre = dict(PRESETS[PRESET])
    for name, label, kind, args, budget in STEPS_V5:
        if name == 'scumble' and 'texture' not in REGIONS:
            continue
        if kind == 'forms':
            a = dict(args)
            a['R'] = a['R'] * pre.get('R', 1.)
            for key in ('medium', 'soft', 'max_len', 'spacing', 'axis'):
                if key in pre:
                    a[key] = pre[key] if key != 'medium' or pre[key] != 'flat' else 'bristle'
            if 'spacing' in pre:                           # presets give spacing in v4 units (1.45 = the v4 default)
                a['spacing'] = 1.25 * pre['spacing'] / 1.45
            area = a.pop('area')
            deep = ndimage.binary_erosion(AREAS[area], iterations=1)
            # a correcting pass keeps off the rim: the edge was cut in in its own colour, and the blurred target there
            # is a mix of both sides (brown dabs along a cheek)
            mask = ndimage.binary_erosion(AREAS[area], iterations=int(max(3, a['R'] * .6))) if a.get('T') else AREAS[area]
            strokes = form_strokes(ref, canvas, height, thick, tooth, area, mask, rng=rng, start_tip=tip,
                                   contours=contours.get(area), deep=deep, **a)
            if not a.get('T'):                             # safety net: nothing bare is left inside the area
                holes = bare_holes(ndimage.binary_erosion(AREAS[area], iterations=2), thick, min_px=6)
                if holes.sum() > 30:
                    fill = planes.plane_strokes(ref, canvas, holes, R=max(4., a['R'] * .3), k=2, rng=rng,
                                                start_tip=strokes[-1]['points'][-1] if strokes else tip, medium='bristle',
                                                spacing=1.1, fill=True)
                    for st in fill:
                        st['clip'] = area
                        render(st, canvas, height, thick, tooth)
                    strokes += fill
                    print(f'{"":10s} +{len(fill)} strokes fill {int(holes.sum())} bare px', flush=True)
            tip = record(name, label, strokes, budget) or tip
            if os.environ.get("V5_STOP") == name:        # debugging: plan only up to this pass
                break
            continue
        first = []
        if name == 'edges':                                # before the edges are worked: no ground left anywhere
            holes = bare_holes(np.ones((H, W), bool), thick, min_px=6)
            if holes.sum() > 20:
                first = planes.plane_strokes(ref, canvas, holes, R=5., k=3, rng=rng, start_tip=tip, medium='bristle',
                                             spacing=1.1, fill=True)
                for st in first:
                    render(st, canvas, height, thick, tooth)
                print(f'{"":10s} +{len(first)} strokes close {int(holes.sum())} bare px', flush=True)
        strokes = first + plan_layer(args, ref, canvas, focus, rng, first[-1]['points'][-1] if first else tip)
        if args.get('area') == 'hair':
            for st in strokes:
                st['clip'] = 'strand'
        for st in strokes[len(first):]:
            render(st, canvas, height, thick, tooth)
        tip = record(name, label, strokes, budget) or tip
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
                                                   ground=GROUND.tolist(), style='v5', preset=PRESET, pace=PACE_V5), ensure_ascii=False))
    return passes
