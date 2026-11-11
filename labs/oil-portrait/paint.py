"""Experiment: paint a portrait from a photo the way a person paints in oils, stroke by stroke.

Process (each pass planned against the canvas the previous passes left, like a painter checking the model):
  tone the ground -> pencil underdrawing (our line-first engine's stroke extraction and person-like order)
  -> block-in with a big filbert -> form -> refine -> detail (only where it matters) -> accents (catchlights) -> signature.

A stroke is a curved, tapered filbert mark along the form (structure-tensor flow), with bristle lanes, a paint load that
runs dry into broken dry-brush, wet pickup of the paint underneath, canvas tooth, and impasto relief. Strokes within a
pass are grouped by paint mix (k-means), dark to light, nearest-neighbour inside a mix: the order a painter works in.

usage: paint.py plan SUBJECT.png SKETCH.png OUT_DIR FOCUS_JSON [SIGNATURE_JSON|-] [EYES_JSON]
       (then film.py OUT_DIR renders the film)
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

SKILL = Path(__file__).resolve().parents[2] / 'skills' / 'sweety-handdraw-video' / 'scripts'   # the line-first engine
sys.path.insert(0, str(SKILL))

W, H = 1120, 1422            # canvas pixels (painted at display size)
CANVAS_RGB = np.array([.955, .945, .915])
SEED = 7

# ----------------------------------------------------------------------------------------------- colour
_M = np.array([[.4124, .3576, .1805], [.2126, .7152, .0722], [.0193, .1192, .9505]])
_WHITE = np.array([.95047, 1., 1.08883])


def lin(c):
    return np.where(c <= .04045, c / 12.92, ((np.clip(c, 0, None) + .055) / 1.055) ** 2.4)


def unlin(c):
    return np.where(c <= .0031308, c * 12.92, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - .055)


def rgb2lab(rgb):
    xyz = lin(np.asarray(rgb, float)) @ _M.T / _WHITE
    f = np.where(xyz > .008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab2rgb(lab):
    lab = np.asarray(lab, float)
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    xyz = np.where(f > .2069, f ** 3, (f - 16 / 116) / 7.787) * _WHITE
    return np.clip(unlin(xyz @ np.linalg.inv(_M).T), 0, 1)


def kmeans(points, k, rng, iterations=20):
    k = max(1, min(k, len(points)))
    centres = [points[rng.integers(len(points))]]
    for _ in range(k - 1):   # k-means++
        d = np.min([((points - c) ** 2).sum(1) for c in centres], 0)
        centres.append(points[rng.choice(len(points), p=d / d.sum())] if d.sum() > 0 else points[rng.integers(len(points))])
    centres = np.array(centres, float)
    for _ in range(iterations):
        label = np.argmin(((points[:, None] - centres[None]) ** 2).sum(-1), 1)
        for j in range(k):
            if (label == j).any():
                centres[j] = points[label == j].mean(0)
    return centres, np.argmin(((points[:, None] - centres[None]) ** 2).sum(-1), 1)


# --------------------------------------------------------------------------------------------- textures
def smooth_noise(shape, sigma, rng):
    n = ndimage.gaussian_filter(rng.standard_normal(shape), sigma)
    return (n - n.mean()) / (n.std() + 1e-9)


def canvas_tooth(rng):
    """Linen weave: over/under threads every ~4.5 px with irregular thread thickness, plus fibre noise. 0..1."""
    y, x = np.mgrid[0:H, 0:W].astype(float)
    p = 4.6
    wobble_x = 1.2 * smooth_noise((H, W), 18, rng)
    wobble_y = 1.2 * smooth_noise((H, W), 18, rng)
    warp = np.cos(np.pi * (x + wobble_x) / p) ** 2
    weft = np.cos(np.pi * (y + wobble_y) / p) ** 2
    over = (np.floor((x + wobble_x) / p) + np.floor((y + wobble_y) / p)) % 2
    weave = np.where(over > 0, warp * (.55 + .45 * weft), weft * (.55 + .45 * warp))
    weave = weave * (1 + .25 * smooth_noise((H, W), 6, rng)) + .18 * smooth_noise((H, W), .7, rng)
    weave = ndimage.gaussian_filter(weave, .6)
    return (weave - weave.min()) / (weave.max() - weave.min())


# ------------------------------------------------------------------------------------------ orientation
def orientation(lum, scale, rng, default=-58.):
    """Unit tangent field (along edges, i.e. across the gradient) from the structure tensor; where the image has no
    clear direction the field relaxes into a right-hander's natural slant with slow wander."""
    g = ndimage.gaussian_filter(lum, max(.8, scale * .5))
    gx, gy = ndimage.sobel(g, 1), ndimage.sobel(g, 0)
    rho = max(2., scale * 1.4)
    jxx, jxy, jyy = (ndimage.gaussian_filter(a, rho) for a in (gx * gx, gx * gy, gy * gy))
    theta = .5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2          # tangent angle
    energy = jxx + jyy
    coherence = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (energy + 1e-9)
    strength = coherence * np.clip(energy / (np.percentile(energy, 60) + 1e-9), 0, 1)
    wander = np.radians(default) + .35 * smooth_noise(lum.shape, 60, rng)
    c2 = strength * np.cos(2 * theta) + (1 - strength) * .35 * np.cos(2 * wander)
    s2 = strength * np.sin(2 * theta) + (1 - strength) * .35 * np.sin(2 * wander)
    c2, s2 = ndimage.gaussian_filter(c2, scale * .5 + .5), ndimage.gaussian_filter(s2, scale * .5 + .5)
    angle = .5 * np.arctan2(s2, c2)
    return np.cos(angle), np.sin(angle)


# ------------------------------------------------------------------------------------------ media
MEDIA = {
    #            opacity  load_len  dry   pickup  colvar  impasto  cap   lanes
    'wash':     dict(opacity=.34, load_len=9., dry=1.2, pickup=0., colvar=.035, impasto=.15, cap=.9, lanes=1.0),
    'pencil':   dict(opacity=.62, load_len=1e9, dry=1.6, pickup=0., colvar=0., impasto=0., cap=.6, lanes=0.),
    'oil_lean': dict(opacity=.94, load_len=24., dry=.6, pickup=.30, colvar=.022, impasto=.3, cap=.85, lanes=1.0),
    'oil':      dict(opacity=.96, load_len=20., dry=.8, pickup=.24, colvar=.018, impasto=.75, cap=.85, lanes=.9),
    'oil_fine': dict(opacity=.97, load_len=22., dry=.6, pickup=.14, colvar=.012, impasto=.9, cap=.8, lanes=.7),
    'oil_detail': dict(opacity=.97, load_len=30., dry=.1, pickup=.2, colvar=.004, impasto=.3, cap=.8, lanes=.25),   # soft round
    'oil_skin': dict(opacity=.95, load_len=26., dry=.15, pickup=.38, colvar=.005, impasto=.25, cap=.9, lanes=.3),
    'flat':     dict(opacity=.97, load_len=34., dry=.45, pickup=.18, colvar=.012, impasto=.75, cap=.35, lanes=.75, square=True),  # v4 planes
    'knife':    dict(opacity=.99, load_len=70., dry=.3, pickup=.3, colvar=.03, impasto=1.5, cap=.2, lanes=.12, square=True,
                     ridge=1.4),   # palette knife: a flat sharp-edged plane, paint squeezed into ridges at its edges
    'digital':  dict(opacity=.93, load_len=1e9, dry=0., pickup=.1, colvar=.006, impasto=.05, cap=.5, lanes=.12, square=True),
    'scumble':  dict(opacity=.85, load_len=5., dry=1.5, pickup=.08, colvar=.02, impasto=.12, cap=.7, lanes=.9),   # dry brush
    'blend':    dict(opacity=.6, load_len=1e9, dry=0., pickup=1., colvar=0., impasto=.08, cap=.9, lanes=.25),
    # v5 (forms.py): a flat hog-bristle brush worked by hand: the load runs out along a long stroke, wet paint is dragged
    'bristle':  dict(opacity=.97, load_len=15., dry=.85, pickup=.26, colvar=.02, impasto=1.0, cap=.5, lanes=.85),
    'bristle_flat': dict(opacity=.97, load_len=22., dry=.5, pickup=.16, colvar=.012, impasto=.9, cap=.35, lanes=.6, square=True),
    'bristle_soft': dict(opacity=.96, load_len=17., dry=.6, pickup=.3, colvar=.016, impasto=.8, cap=.75, lanes=.7),
    'drag':     dict(opacity=.8, load_len=9., dry=1.3, pickup=.4, colvar=.02, impasto=.4, cap=.6, lanes=1.0, square=True),
    'sign':     dict(opacity=.95, load_len=1e9, dry=.3, pickup=.05, colvar=0., impasto=.8, cap=.7, lanes=.3),
}


def catmull(points, step=.6):
    p = np.asarray(points, float)
    if len(p) == 1:
        p = np.vstack([p, p + [.5, 0]])
    if len(p) == 2:
        p = np.vstack([p[0], (p[0] + p[1]) / 2, p[1]])
    ext = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    out = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1:i + 3]
        n = max(2, int(np.linalg.norm(p2 - p1) / step))
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        out.append(.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(p[-1:])
    return np.vstack(out)


def geometry(stroke):
    """Dense centreline, arc length, tangents, normals and half-width profile of a stroke (deterministic)."""
    rng = np.random.default_rng(stroke['seed'])
    R = stroke['R']
    pts = catmull(stroke['points'])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.r_[True, seg > 1e-6]
    pts = pts[keep]
    if len(pts) < 2:
        pts = np.vstack([pts, pts + [.6, 0]])
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    L = s[-1]
    tan = np.gradient(pts, axis=0)
    tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-9
    nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)
    if stroke.get('wobble', 1) and L > 3 * R:   # a hand never pulls a perfectly smooth line
        k = np.interp(s, np.linspace(0, L, 6), rng.standard_normal(6))
        pts = pts + nrm * (k * R * .05 * float(stroke.get('wobble', 1)))[:, None]
    press = stroke.get('press', (.55, .35))
    land = min(.9 * R, .3 * L + 1)
    lift = min(1.4 * R, .45 * L + 1)
    sm = lambda e: e * e * (3 - 2 * e)
    p = (press[0] + (1 - press[0]) * sm(np.clip(s / land, 0, 1))) * (press[1] + (1 - press[1]) * sm(np.clip((L - s) / lift, 0, 1)))
    pv = stroke.get('pvar', .06)
    p *= 1 + pv * np.interp(s, np.linspace(0, L, max(3, int(L / 60) + 2)), rng.standard_normal(max(3, int(L / 60) + 2)))
    w = np.maximum(.6, R * p)
    return pts, s, tan, nrm, w


def render(stroke, canvas, height, thick, tooth, apply=True):
    """Paint one stroke. Returns the patch (bbox, pixel offsets, arc position, alpha, colour, height) so an animation
    can reveal it by arc length; also commits it to canvas/height/thick in place."""
    m = MEDIA[stroke['medium']]
    rng = np.random.default_rng(stroke['seed'] + 1)
    R = stroke['R']
    pts, s, tan, nrm, w = geometry(stroke)
    L = s[-1]
    wmax = w.max()
    x0, y0 = np.floor(pts.min(0) - wmax - 3).astype(int)
    x1, y1 = np.ceil(pts.max(0) + wmax + 3).astype(int)
    Hc, Wc = canvas.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(Wc, x1), min(Hc, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    pix = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)
    d, idx = cKDTree(pts).query(pix, distance_upper_bound=wmax + 2.5, workers=-1)
    ok = np.isfinite(d)
    if not ok.any():
        return None
    flat = np.flatnonzero(ok)
    pix, idx = pix[ok], idx[ok]
    v = pix - pts[idx]
    lat = (v * nrm[idx]).sum(1)
    lon = (v * tan[idx]).sum(1)
    n = len(pts)
    ss = s[idx] + np.where(((idx == 0) & (lon < 0)) | ((idx == n - 1) & (lon > 0)), lon, 0.)
    wpx = w[idx]
    over = np.where(ss < 0, -ss, np.where(ss > L, ss - L, 0.))
    if m.get('square'):                              # a flat brush: square ends, the corners only slightly worn
        rr = np.maximum(np.abs(lat) / wpx, over / (wpx * m['cap'])) * .85 + np.sqrt((lat / wpx) ** 2 + (over / (wpx * m['cap'])) ** 2) * .15
    else:
        rr = np.sqrt((lat / wpx) ** 2 + (over / (wpx * m['cap'])) ** 2)
    soft = stroke.get('soft', 0.)                    # edge hierarchy: a stroke where the form turns has a soft edge
    edge = np.clip((1 - rr) * wpx / (1.1 + soft * .6 * wpx) + .5, 0, 1)
    inside = edge > 0
    flat, ss, lat, wpx, edge = flat[inside], ss[inside], lat[inside], wpx[inside], edge[inside]
    u = np.clip(lat / wpx, -1, 1)
    # bristles: coarse lanes + fine streaks, constant along the stroke
    nb = int(np.clip(R * .9, 5, 42))
    lane_v = .72 + .28 * rng.random(nb + 2)
    streak_v = rng.random(3 * nb + 2)
    b = (u + 1) / 2
    lanes = np.interp(b * (nb + 1), np.arange(nb + 2), lane_v)
    streak = np.interp(b * (3 * nb + 1), np.arange(3 * nb + 2), streak_v)
    bristle = 1 - m['lanes'] * (1 - .5 * soft) * (1 - (.62 * lanes + .38 * (.7 + .3 * streak)))
    load = stroke.get('load0', 1.) * np.exp(-np.clip(ss, 0, None) / (m['load_len'] * R * stroke.get('load', 1.)))
    lift = np.clip(wpx / R, 0, 1)                    # a lifting brush lays less paint: feathered, dry ends
    amount = bristle * load * (1 - .22 * np.abs(u) ** 4) * (.5 + .5 * lift)
    gy, gx = (flat // (x1 - x0)) + y0, (flat % (x1 - x0)) + x0
    t = tooth[gy, gx]
    value = float(np.dot(stroke['rgb'], [.299, .587, .114]))
    dry = m['dry'] * (.35 + .65 * np.clip((value - .12) / .3, 0, 1))   # dark paint is laid rich: no ground specks in shadow
    alpha = np.clip((amount - .36 + .25 * (value < .3) + .55 * dry * (t - .5) * (1.2 - load)) * 3.4, 0, 1)
    alpha *= edge * m['opacity'] * stroke.get('opacity', 1.)
    if stroke.get('clip'):                           # v4 block-in: a straight stroke along a curved silhouette stays in its area
        alpha *= CLIPS[stroke['clip']][gy, gx]
    # colour: the mix, smeared with paint picked up along the way, with unmixed streaks
    colour = np.asarray(stroke['rgb'], float)
    under_idx = np.clip(np.rint(pts).astype(int), 0, [Wc - 1, Hc - 1])
    under = canvas[under_idx[:, 1], under_idx[:, 0]]
    # only wet paint comes up on the brush: bare ground (v4 starts on a toned, dry canvas) gives nothing, or dark
    # strokes laid straight onto the ground turn grey; a wash or any earlier stroke counts as wet
    wet = np.clip(thick[under_idx[:, 1], under_idx[:, 0]] / .08, 0, 1) if thick is not None else np.ones(n)
    ema = np.empty_like(under)
    emw = np.empty(n)
    acc, accw = under[0].copy(), float(wet[0])
    ds = np.r_[0, np.diff(s)]
    k = np.clip(ds / (3. * R), 0, 1)
    for i in range(n):
        acc += (under[i] - acc) * k[i]
        accw += (wet[i] - accw) * k[i]
        ema[i] = acc
        emw[i] = accw
    pick = m['pickup'] * (1 - np.exp(-np.clip(ss, 0, None) / (2. * R))) * (.6 + .4 * (1 - load))
    pick = pick * emw[np.clip(idx[inside], 0, n - 1)]
    paint = (1 - pick)[:, None] * colour + pick[:, None] * ema[np.clip(idx[inside], 0, n - 1)]
    paint = paint + ((streak - .5) * 2 * m['colvar'])[:, None] * np.array([1., .96, .9])
    if stroke.get('tone2'):                          # v5: two neighbouring tones on one brush, side by side
        paint = paint + u[:, None] * np.asarray(stroke['tone2'], float)
    paint = np.clip(paint, 0, 1)
    # impasto: ridges at bristle lanes and edges, paint piles where the brush lifts while still loaded
    ridge = np.exp(-((np.abs(u) - .86) / .1) ** 2)
    pile = .45 * load * np.exp(-((ss - L) / (.45 * R)) ** 2)
    body = .3 + 1.0 * value                           # thick lights, thin darks
    hgt = alpha * m['impasto'] * body * (.55 + .9 * (streak - .5) + .4 * (lanes - .86) / .14 * .3 + m.get('ridge', .25) * ridge + pile) * min(1., R / 12.) ** .5 * stroke.get('body', 1.)
    patch = dict(box=(y0, y1, x0, x1), gy=gy.astype(np.int32), gx=gx.astype(np.int32), ss=ss.astype(np.float32), alpha=alpha.astype(np.float32),
                 paint=paint.astype(np.float32), hgt=hgt.astype(np.float32), L=float(L), wpx=wpx.astype(np.float32),
                 path=pts.astype(np.float32), s=s.astype(np.float32))
    if apply:
        commit(patch, canvas, height, thick)
    return patch


def commit(patch, canvas, height, thick, upto=None):
    """Lay the stroke's paint on the canvas; with upto, only as far as the brush has travelled (for a display frame)."""
    gy, gx = patch['gy'], patch['gx']
    a = patch['alpha']
    if upto is not None:
        a = a * np.clip((upto + .45 * patch['wpx'] - patch['ss']) / 2.5, 0, 1)
    canvas[gy, gx] = canvas[gy, gx] * (1 - a[:, None]) + patch['paint'] * a[:, None]
    if height is not None:
        scale = 1. if upto is None else np.clip((upto + .45 * patch['wpx'] - patch['ss']) / 2.5, 0, 1)
        height[gy, gx] = height[gy, gx] * (1 - .7 * a) + patch['hgt'] * scale
        thick[gy, gx] = thick[gy, gx] + a * (1 - thick[gy, gx]) * .8


# ----------------------------------------------------------------------------------------- planning
def disk_stamp(cov, pts, r, scale=4):
    """Mark coverage (at 1/scale resolution) along a polyline."""
    h, w = cov.shape
    rr = max(1., r / scale)
    pts = np.asarray(pts, float) / scale
    dense = catmull(pts * scale, step=max(1., r * .5)) / scale if len(pts) > 1 else pts
    ri = int(math.ceil(rr))
    oy, ox = np.mgrid[-ri:ri + 1, -ri:ri + 1]
    disk = (ox ** 2 + oy ** 2) <= rr * rr
    for x, y in dense:
        cx, cy = int(round(x)), int(round(y))
        ya, yb, xa, xb = cy - ri, cy + ri + 1, cx - ri, cx + ri + 1
        sy0, sx0 = max(0, -ya), max(0, -xa)
        ya, xa = max(0, ya), max(0, xa)
        yb, xb = min(h, yb), min(w, xb)
        if yb > ya and xb > xa:
            cov[ya:yb, xa:xb] |= disk[sy0:sy0 + yb - ya, sx0:sx0 + xb - xa]


def sample(field, x, y):
    return field[min(H - 1, max(0, int(round(y)))), min(W - 1, max(0, int(round(x))))]


def trace(x, y, R, colour_lab, ref_lab, canvas_lab, tx, ty, maxlen, minlen, fc, direction, colour_break, stop=True):
    pts = [(x, y)]
    ldx, ldy = direction
    turned = 0.
    for i in range(1, maxlen):
        rx = sample(ref_lab, x, y)
        if stop and i > minlen and np.linalg.norm(rx - sample(canvas_lab, x, y)) < np.linalg.norm(rx - colour_lab):
            break
        dx, dy = sample(tx, x, y), sample(ty, x, y)
        if dx * ldx + dy * ldy < 0:
            dx, dy = -dx, -dy
        if i > 1:
            dx, dy = fc * dx + (1 - fc) * ldx, fc * dy + (1 - fc) * ldy
        norm = math.hypot(dx, dy) or 1.
        dx, dy = dx / norm, dy / norm
        if i > 1:
            turned += math.acos(max(-1., min(1., dx * ldx + dy * ldy)))
            if turned > math.radians(110):
                break
        nx, ny = x + R * dx, y + R * dy
        if not (-R <= nx < W + R and -R <= ny < H + R):
            break
        if i >= minlen and np.linalg.norm(sample(ref_lab, nx, ny) - colour_lab) > colour_break:
            break
        x, y, ldx, ldy = nx, ny, dx, dy
        pts.append((x, y))
    return pts


def plan_layer(spec, ref_rgb, canvas, focus, rng, start_tip):
    R = spec['R']
    blur = ndimage.gaussian_filter(ref_rgb, (spec.get('blur', .5) * R, spec.get('blur', .5) * R, 0))
    ref_lab, canvas_lab = rgb2lab(blur), rgb2lab(canvas)
    tx, ty = orientation(ref_lab[..., 0], R, rng)
    lum_g = ndimage.gaussian_filter(ref_lab[..., 0], max(1., R * .5))
    gmag = np.hypot(ndimage.sobel(lum_g, 1), ndimage.sobel(lum_g, 0))
    g_ref = np.percentile(gmag, 90) + 1e-6
    diff = np.linalg.norm(ref_lab - canvas_lab, axis=-1)
    if spec.get('darken'):         # stubble and the like are darker than the skin around them: only darker paint goes on
        diff = np.where(ref_lab[..., 0] < canvas_lab[..., 0] - 2, diff, 0.)
    if spec.get('lighten'):        # v5 提亮: only paint lighter than what is there (hair sheen, the lights of the face)
        diff = np.where(ref_lab[..., 0] > canvas_lab[..., 0] + spec['lighten'], diff, 0.)
    thresh = spec['T'] / np.clip(focus, .05, 1) ** spec['gamma']
    skin = skin_mask(ref_rgb) if spec.get('skin') else None
    if skin is not None:           # the face converges further on the model than the rest
        thresh = np.where(skin, thresh * spec.get('skin_T', 1.), thresh)
    feat = None
    if spec.get('focal') and FEATURES:   # the features converge furthest
        feat = np.zeros((H, W), bool)
        for e in FEATURES:
            feat |= ellipse_mask(e)
        thresh = np.where(feat, thresh * spec.get('feat_T', 1.), thresh)
    if spec.get('only'):           # a pass for one region (e.g. hair and clothes): elsewhere nothing exceeds the threshold
        inside = focus_map([list(REGIONS[spec['only']]) + [1.]], 0) > .9
        for name in spec.get('exclude', []):
            inside &= ~(focus_map([list(REGIONS[name]) + [1.]], 0) > .9)
        if spec.get('skip_features'):
            for e in FEATURES:
                inside &= ~ellipse_mask(e, 1.1)
        thresh = np.where(inside, thresh, 1e9)
    if spec.get('area'):           # v4: a pass for one area mask (hair, face, the features)
        thresh = np.where(AREAS[spec['area']], thresh, 1e9)
    step = max(2, int(round(spec.get('grid', 1.) * R)))
    cands = []
    for gy in range(0, H, step):
        for gx in range(0, W, step):
            cell = diff[gy:gy + step, gx:gx + step]
            if cell.mean() > thresh[min(H - 1, gy + step // 2), min(W - 1, gx + step // 2)]:
                j = np.unravel_index(np.argmax(cell + rng.random(cell.shape) * .01), cell.shape)
                cands.append((cell.mean(), gx + j[1], gy + j[0]))
    if not cands:
        return []
    cands.sort(key=lambda c: -c[0])
    cols = np.array([ref_lab[y, x] for _, x, y in cands])
    mixes, label = kmeans(cols, spec['mixes'], rng)
    cov = np.zeros((H // 4 + 1, W // 4 + 1), bool)
    strokes = []
    def candidates_left():
        # cells the pass still has not reached (holes between strokes would show the ground)
        extra = []
        for gy in range(0, H, step):
            for gx in range(0, W, step):
                c = cov[gy // 4:(gy + step) // 4 + 1, gx // 4:(gx + step) // 4 + 1]
                cell = diff[gy:gy + step, gx:gx + step]
                if c.mean() < .55 and cell.mean() > thresh[min(H - 1, gy + step // 2), min(W - 1, gx + step // 2)]:
                    free = np.kron(~c, np.ones((4, 4), bool))[:cell.shape[0], :cell.shape[1]]
                    if free.shape != cell.shape or not free.any():
                        continue
                    j = np.unravel_index(np.argmax(np.where(free, cell, -1)), cell.shape)
                    extra.append((cell.mean(), gx + j[1], gy + j[0]))
        return extra

    queue = list(zip(cands, cols, label))
    sweeps = spec.get('sweeps', 0)
    while queue:
        (err, x, y), lab_own, mix = queue.pop(0)
        if not queue and sweeps > 0:
            sweeps -= 1
            more = candidates_left()
            if more:
                mc = np.array([ref_lab[yy, xx] for _, xx, yy in more])
                ml = np.argmin(((mc[:, None] - mixes[None]) ** 2).sum(-1), 1)
                queue.extend(zip(more, mc, ml))
        if cov[y // 4, x // 4]:
            continue
        on_skin = skin is not None and skin[min(H - 1, max(0, int(y))), min(W - 1, max(0, int(x)))]
        on_feat = feat is not None and feat[min(H - 1, max(0, int(y))), min(W - 1, max(0, int(x)))]
        own = max(spec.get('own', .3), .85) if (on_skin or on_feat) else spec.get('own', .3)   # skin and features mixed carefully
        gap = np.linalg.norm(lab_own - mixes[mix])
        own = own + (1 - own) * np.clip((gap - 6) / 14, 0, 1) * .8       # a pile far off is corrected on the brush
        lab_c = mixes[mix] + own * (lab_own - mixes[mix])
        sgn = 1 if rng.random() < .5 else -1
        d0 = (sample(tx, x, y) * sgn, sample(ty, x, y) * sgn)
        stop = spec.get('stop', True)
        fwd = trace(x, y, R, lab_c, ref_lab, canvas_lab, tx, ty, spec['maxlen'], spec['minlen'], spec['fc'], d0, spec.get('break', 20), stop)
        back = trace(x, y, R, lab_c, ref_lab, canvas_lab, tx, ty, max(1, spec['maxlen'] - len(fwd)), 1, spec['fc'], (-d0[0], -d0[1]),
                     spec.get('break', 20), stop)
        pts = back[::-1][:-1] + fwd
        if len(pts) < 2 and not spec.get('dabs'):
            pts = [(x - d0[0] * R * .5, y - d0[1] * R * .5), (x + d0[0] * R * .5, y + d0[1] * R * .5)]
        elif len(pts) < 2:
            pts = [(x - d0[0] * R * .35, y - d0[1] * R * .35), (x + d0[0] * R * .35, y + d0[1] * R * .35)]
        # a right hand pulls rightward/downward more often than it pushes
        a, b = np.array(pts[0]), np.array(pts[-1])
        if (b - a) @ np.array([.75, .66]) < 0 and rng.random() < .75:
            pts = pts[::-1]
        disk_stamp(cov, pts, R * spec.get('cover', .9))
        strokes.append(dict(points=[(float(px), float(py)) for px, py in pts], R=float(R * (.9 + .2 * rng.random())),
                            rgb=[float(c) for c in lab2rgb(lab_c)], mix=int(mix), mix_rgb=[float(c) for c in lab2rgb(mixes[mix])],
                            medium=spec['medium'], seed=int(rng.integers(1 << 30)), load=float(.85 + .3 * rng.random()),
                            press=(spec.get('press', (.55, .35))[0], float(rng.uniform(.15, .6))),
                            soft=float(spec.get('soft', 0.) * np.clip(1 - gmag[y, x] / g_ref, 0, 1))))
        if on_skin:
            strokes[-1]['medium'] = 'oil_skin'
            strokes[-1]['soft'] = max(strokes[-1]['soft'], .65)
        elif on_feat:
            strokes[-1]['medium'] = 'oil_detail'
            strokes[-1]['soft'] = max(strokes[-1]['soft'], .2)
    if spec.get('focal') and FEATURES:   # eyes, mouth...: painted last in the pass, feature by feature (shown at life pace)
        plain, focal = [], {}
        for st in strokes:
            k = tag_feature(st)
            (plain if k < 0 else focal.setdefault(k, [])).append(st)
        ordered = painter_order(plain, mixes[:, 0], start_tip, rng)
        for k in sorted(focal):
            tip = ordered[-1]['points'][-1] if ordered else start_tip
            part = painter_order(focal[k], mixes[:, 0], tip, rng)
            for st in part:
                st['focal'] = True
            ordered += part
        return ordered
    return painter_order(strokes, mixes[:, 0], start_tip, rng)


def painter_order(strokes, mix_l, start_tip, rng):
    """A painter works area by area (the hair, a stretch of background, the face), and inside an area lays the darks
    first, then the half tones, then the lights, one mix at a time, each mix as one nearest-neighbour tour."""
    if not strokes:
        return []
    mids = np.array([np.mean(st['points'], 0) for st in strokes])
    k = int(np.clip(len(strokes) / 28, 1, 10))
    _, area = kmeans(mids, k, rng)
    centres = np.array([mids[area == a].mean(0) if (area == a).any() else [1e9, 1e9] for a in range(k)])
    tip = np.asarray(start_tip, float)
    todo = [a for a in range(k) if (area == a).any()]
    ordered = []
    while todo:
        a = min(todo, key=lambda a: np.hypot(*(centres[a] - tip)))
        todo.remove(a)
        members = [st for st, ar in zip(strokes, area) if ar == a]
        for mix in sorted({st['mix'] for st in members}, key=lambda m: mix_l[m]):
            pool = [st for st in members if st['mix'] == mix]
            while pool:
                d0 = [np.hypot(*(np.array(st['points'][0]) - tip)) for st in pool]
                d1 = [np.hypot(*(np.array(st['points'][-1]) - tip)) * 1.2 + 15 for st in pool]   # back-and-forth is fine
                j = int(np.argmin(np.minimum(d0, d1)))
                st = pool.pop(j)
                if d1[j] < d0[j]:
                    st['points'] = st['points'][::-1]
                    if st.get('press_fixed'):     # v4 planes: the lifted end stays where the plane meets its neighbour
                        st['press'] = tuple(st['press'][::-1])
                ordered.append(st)
                tip = np.array(st['points'][-1])
    return ordered


def blend_strokes(ref_rgb, canvas, focus, rng, R=7., start_tip=(0, 0), region=None, toward=0., patch_min=2., smooth_pct=45,
                  systematic=False):
    """柔化: with a clean soft brush, sweep along the form over smooth skin where the paint is still patchier than the
    model, carrying only the paint already there (no new colour). Features and real edges are left alone."""
    ref_lab, can_lab = rgb2lab(ndimage.gaussian_filter(ref_rgb, (1.5, 1.5, 0))), rgb2lab(canvas)
    tx, ty = orientation(ref_lab[..., 0], R, rng)
    lum = ndimage.gaussian_filter(ref_lab[..., 0], 2.)
    gmag = np.hypot(ndimage.sobel(lum, 1), ndimage.sobel(lum, 0))
    def local_std(a):
        m = ndimage.uniform_filter(a, 2 * R)
        return np.sqrt(np.clip(ndimage.uniform_filter(a * a, 2 * R) - m * m, 0, None))
    patchy = local_std(can_lab[..., 0]) - 1.3 * local_std(ref_lab[..., 0])
    if region is None:
        region = focus > .55
    smooth_ok = gmag < np.percentile(gmag[region], smooth_pct) if region.any() else gmag < 0
    model = ndimage.gaussian_filter(ref_rgb, (R * .5, R * .5, 0))
    out = []
    step = int(R * (2. if systematic else 1.6))
    for gy in range(step // 2, H, step):
        for gx in range(step // 2, W, step):
            x, y = gx + rng.uniform(-R, R) * .4, gy + rng.uniform(-R, R) * .4
            xi, yi = int(min(W - 1, max(0, x))), int(min(H - 1, max(0, y)))
            if not region[yi, xi] or not smooth_ok[yi, xi] or (not systematic and patchy[yi, xi] < patch_min):
                continue
            base = ref_lab[yi, xi]
            pts, cx, cy = [(x, y)], x, y
            dx, dy = sample(tx, x, y), sample(ty, x, y)
            for _ in range(5):
                ndx, ndy = sample(tx, cx, cy), sample(ty, cx, cy)
                if ndx * dx + ndy * dy < 0:
                    ndx, ndy = -ndx, -ndy
                dx, dy = .6 * ndx + .4 * dx, .6 * ndy + .4 * dy
                nn = math.hypot(dx, dy) or 1.
                cx, cy = cx + R * dx / nn, cy + R * dy / nn
                if not (0 <= cx < W and 0 <= cy < H) or np.linalg.norm(sample(ref_lab, cx, cy) - base) > 7:
                    break
                pts.append((cx, cy))
            if len(pts) < 3:
                continue
            colour = canvas[max(0, yi - 2):yi + 3, max(0, xi - 2):xi + 3].reshape(-1, 3).mean(0)
            colour = (1 - toward) * colour + toward * model[yi, xi]      # feathering carries a little of the right colour
            out.append(dict(points=[(float(a), float(b)) for a, b in pts], R=float(R * (.9 + .25 * rng.random())),
                            rgb=[float(c) for c in colour], mix=0, mix_rgb=[float(c) for c in colour], medium='blend',
                            seed=int(rng.integers(1 << 30)), load=1., press=(.4, .2), soft=1.))
    return painter_order(out, np.zeros(1), start_tip, rng)


def wash_strokes(rng):
    """Tone the ground: broad loose strokes of thin warm paint, zig-zagging down the canvas with overlap."""
    out = []
    colour = np.array([.74, .56, .42])
    rows = 11
    for r in range(rows):
        yc = (r + .5) * H / rows + rng.normal(0, 12)
        lr = r % 2 == 0
        n = 7
        xs = np.linspace(-80, W + 80, n)
        tilt = rng.uniform(-.18, .18) * W
        ys = yc + np.linspace(-tilt / 2, tilt / 2, n) + 26 * np.sin(np.linspace(0, 3, n) + r * 1.7)
        if not lr:
            xs, ys = xs[::-1], ys[::-1]
        out.append(dict(points=[(float(a), float(b)) for a, b in zip(xs, ys)], R=float(105 + 20 * rng.random()),
                        rgb=list(colour * (.95 + .08 * rng.random())), mix=0, mix_rgb=list(colour), medium='wash',
                        seed=int(rng.integers(1 << 30)), load=float(.8 + .4 * rng.random()), press=(.75, .6)))
    return out


def sketch_strokes(sketch_path, subject_shape, crop, rng):
    """Underdrawing from a line sketch: our engine's stroke extraction (skeleton -> whole strokes) and person-like order."""
    from line_topology import extract_strokes
    from line_drawing import order, classify
    img = Image.open(sketch_path).convert('L').resize(subject_shape[::-1][:2], Image.LANCZOS)
    a = np.asarray(img, float) / 255
    top, height_px, sx = crop
    a = a[top:top + height_px]
    a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS), float) / 255
    ink = a < .55
    strokes, *_ = extract_strokes(ink, size=W / 720)
    pixels = np.dstack([np.repeat((a * 255)[..., None], 3, 2), np.full((H, W), 255.)]).astype(np.uint8)
    classify(strokes, np.zeros((H, W), int), np.full((H, W), 255, np.uint8), pixels, ['head'], ink=ink)
    ordered, _ = order(strokes, 1, (W * .3, 0))
    out = []
    colour = [.33, .23, .18]
    for st in ordered:
        p = np.asarray(st['points'], float)
        if len(p) > 3:
            keep = np.r_[0, np.arange(3, len(p) - 1, 3), len(p) - 1]
            p = p[keep]
        if st['length'] < 6:
            continue
        out.append(dict(points=[(float(x), float(y)) for x, y in p], R=float(1.55 + .25 * rng.random()), rgb=colour, mix=0,
                        mix_rgb=colour, medium='pencil', seed=int(rng.integers(1 << 30)), load=1., press=(.35, .2), wobble=0,
                        pvar=.22, opacity=float(.75 + .25 * rng.random())))
    return out


def line_strokes(sketch, ref_rgb, rng):
    """勾勒: the edges a painter draws as one confident thin line in the focal features (eyelids, the lip line, glasses
    frames): the underdrawing's lines there, snapped onto the darkest line of the model, in the model's own colour."""
    lum = ndimage.gaussian_filter(ref_rgb @ np.array([.299, .587, .114]), .8) * 100
    region = np.zeros((H, W), bool)
    for e in LINE_FEATURES:
        region |= ellipse_mask(e, 1.3)
    offs = np.arange(-6, 7, dtype=float)
    out = []
    for st in sketch:
        dense = catmull(st['points'], step=1.)
        if len(dense) < 8:
            continue
        tan = np.gradient(dense, axis=0)
        tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-9
        nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)
        inside = region[np.clip(dense[:, 1].astype(int), 0, H - 1), np.clip(dense[:, 0].astype(int), 0, W - 1)]
        if not inside.any():
            continue
        cand = dense[:, None, :] + offs[None, :, None] * nrm[:, None, :]             # (n, 13, 2) across the line
        vals = ndimage.map_coordinates(lum, [cand[..., 1].ravel(), cand[..., 0].ravel()], order=1, mode='nearest').reshape(cand.shape[:2])
        j = np.argmin(vals, 1)
        shift = ndimage.median_filter(offs[j], 7, mode='nearest')
        snapped = dense + shift[:, None] * nrm
        centre = ndimage.map_coordinates(lum, [snapped[:, 1], snapped[:, 0]], order=1, mode='nearest')
        sides = sum(ndimage.map_coordinates(lum, [(snapped + k * 5 * nrm)[:, 1], (snapped + k * 5 * nrm)[:, 0]], order=1,
                                            mode='nearest') for k in (-1, 1)) / 2
        ok = inside & (ndimage.uniform_filter1d((sides - centre).astype(float), 5) > 5)
        ok = ndimage.binary_closing(ok, np.ones(9, bool)) & inside       # one continuous line across small gaps
        labels, n = ndimage.label(ok)
        for k in range(1, n + 1):
            idx = np.flatnonzero(labels == k)
            if len(idx) < 10:
                continue
            pts = ndimage.uniform_filter1d(snapped[idx], 5, axis=0, mode='nearest')
            nn = nrm[idx]
            cells = np.clip(np.rint(pts).astype(int), 0, [W - 1, H - 1])
            colour = np.median(ref_rgb[cells[:, 1], cells[:, 0]], 0)
            # the line's own width in the model: where it is darker than halfway between its centre and its sides
            c0, sd = centre[idx], sides[idx]
            dark = [ndimage.map_coordinates(lum, [(pts + o * nn)[:, 1], (pts + o * nn)[:, 0]], order=1, mode='nearest')
                    < (c0 + sd) / 2 for o in (-2.5, -1.5, -.5, .5, 1.5, 2.5)]
            half = float(np.clip(np.median(np.sum(dark, 0)) / 2 * .8, 1.0, 2.0))
            # a long line is pulled in several overlapping strokes of ~70 px, the brush reloaded for each
            arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
            pieces = max(1, int(np.ceil(arc[-1] / 70)))
            bounds = np.linspace(0, arc[-1], pieces + 1)
            for a0, a1 in zip(bounds[:-1], bounds[1:]):
                sel = np.flatnonzero((arc >= a0 - 4) & (arc <= a1 + 4))
                if len(sel) < 3:
                    continue
                seg, sn = pts[sel], nn[sel]
                keep = np.r_[0, np.arange(4, len(seg) - 1, 4), len(seg) - 1]
                # cut in: a narrow soft stroke of the neighbouring colour along each side cleans up what earlier dabs left
                for side in (-1, 1):
                    off = seg + side * (half + 1.6) * sn
                    oc = np.clip(np.rint(off).astype(int), 0, [W - 1, H - 1])
                    around = np.median(ref_rgb[oc[:, 1], oc[:, 0]], 0)
                    out.append(dict(points=[(float(a), float(b)) for a, b in off[keep]], R=1.5, rgb=[float(c) for c in around],
                                    mix=0, mix_rgb=[float(c) for c in around], medium='oil_detail', soft=.5,
                                    seed=int(rng.integers(1 << 30)), load=3., press=(.7, .5), wobble=0, focal=True, cut=True))
                out.append(dict(points=[(float(a), float(b)) for a, b in seg[keep]], R=half, rgb=[float(c) for c in colour],
                                mix=0, mix_rgb=[float(c) for c in colour], medium='oil_detail', seed=int(rng.integers(1 << 30)),
                                load=3., press=(.6, .45), wobble=0, focal=True))
    return out


def signature_strokes(text, box, colour, rng, font='/System/Library/Fonts/Supplemental/SnellRoundhand.ttc'):
    from PIL import ImageDraw, ImageFont
    from line_topology import extract_strokes
    from line_drawing import order
    x0, y0, w, h = box
    img = Image.new('L', (w, h), 0)
    f = ImageFont.truetype(font, int(h * .8), index=1)
    ImageDraw.Draw(img).text((4, h * .05), text, fill=255, font=f)
    ink = np.asarray(img) > 100
    strokes, *_ = extract_strokes(ink, size=1.)
    ordered = sorted(strokes, key=lambda st: np.asarray(st['points'])[:, 0].min())   # writing order, left to right
    out = []
    for st in ordered:
        p = np.asarray(st['points'], float)
        if len(p) > 3:
            p = p[np.r_[0, np.arange(2, len(p) - 1, 2), len(p) - 1]]
        out.append(dict(points=[(float(x + x0), float(y + y0)) for x, y in p], R=2.1, rgb=list(colour), mix=0, mix_rgb=list(colour),
                        medium='sign', seed=int(rng.integers(1 << 30)), load=1., press=(.5, .3), wobble=0))
    return out


LAYERS_V1 = [
    dict(name='block', label='铺大色', R=42, T=10, maxlen=12, minlen=3, fc=.75, mixes=8, gamma=0., medium='oil_lean', own=.3,
         blur=.8, sweeps=2, cover=.8, **{'break': 15}),
    dict(name='form', label='塑形', R=17, T=9, maxlen=10, minlen=3, fc=.7, mixes=12, gamma=.7, medium='oil', own=.3, sweeps=1,
         **{'break': 16}),
    dict(name='refine', label='深入刻画', R=8.5, T=6.5, maxlen=10, minlen=2, fc=.65, mixes=14, gamma=1.8, medium='oil', own=.5,
         **{'break': 15}),
    dict(name='detail', label='细节', R=4.2, T=6, maxlen=8, minlen=2, fc=.6, mixes=16, gamma=3.2, medium='oil_fine', own=.7,
         blur=.4, **{'break': 12}),
    dict(name='accent', label='点睛', R=2.0, T=10, maxlen=4, minlen=1, fc=.5, mixes=6, gamma=5., medium='oil_fine', own=.92,
         blur=.3, dabs=True, press=(.6, .4), **{'break': 10}),
]


# v2: paint toward a painted target (P2 route): softer edges where the form turns, a blending pass on the skin, lower
# thresholds and one more fine pass so the canvas converges on the target's design rather than stopping at a sketch.
LAYERS_V2 = [
    dict(name='block', label='铺大色', R=42, T=10, maxlen=12, minlen=3, fc=.75, mixes=8, gamma=0., medium='oil_lean', own=.6,
         blur=.7, sweeps=2, cover=.8, soft=.7, **{'break': 15}),
    dict(name='form', label='塑形', R=17, T=8, maxlen=10, minlen=3, fc=.7, mixes=14, gamma=.2, medium='oil', own=.4, sweeps=1,
         soft=.6, blend_after=True, skin=True, **{'break': 16}),
    dict(name='refine', label='深入刻画', R=8.5, T=6, maxlen=11, minlen=2, fc=.65, mixes=16, gamma=.8, medium='oil', own=.55,
         soft=.5, focal=True, skin=True, **{'break': 14}),
    # hair strands and knit: long thin strokes along the flow, in their own colours, only on the figure outside the face
    dict(name='texture', label='深入刻画', R=3.6, T=5.5, maxlen=32, minlen=5, fc=.8, mixes=18, gamma=0., medium='oil_fine', own=.85,
         blur=.35, soft=.15, stop=False, press=(.35, .15), cover=1.6, only='figure', exclude=['face'], **{'break': 11}),
    dict(name='detail', label='细节', R=4.2, T=5.5, maxlen=9, minlen=2, fc=.6, mixes=18, gamma=1.8, medium='oil_fine', own=.75,
         blur=.4, soft=.3, focal=True, skin=True, skin_T=.7, feat_T=.6, **{'break': 12}),
    dict(name='polish', label='细节', R=2.8, T=5, maxlen=7, minlen=2, fc=.6, mixes=18, gamma=3.2, medium='oil_fine', own=.85,
         blur=.4, soft=.2, focal=True, skin=True, skin_T=.7, feat_T=.6, feather_after=True, **{'break': 10}),
    # 干擦: where the skin keeps a fine texture (stubble, freckles; marked by hand with --texture), short dabs of a nearly
    # dry brush in the target's own fine colour catch only the tops of the canvas weave, over the softened skin
    dict(name='scumble', label='干擦', R=2.4, T=4, maxlen=3, minlen=1, fc=.5, mixes=6, gamma=0., medium='scumble', own=.95,
         blur=.2, dabs=True, press=(.5, .3), cover=1.0, only='texture', skip_features=True, darken=True, **{'break': 8}),
    # the last small accents; then the crisp lines (勾勒) and the pupils and catchlights (点睛) go on top of everything
    dict(name='accent', label='细节', R=2.0, T=9, maxlen=4, minlen=1, fc=.5, mixes=6, gamma=5., medium='oil_fine', own=.92,
         blur=.3, dabs=True, press=(.6, .4), focal=True, lines_after=True, touches_after=True, **{'break': 10}),
]
LAYERS = LAYERS_V1
# Ellipses [cx, cy, rx, ry] (canvas fractions) of the face and of the whole figure (head, hair, body): the hair/clothes
# texture pass works on the figure outside the face, the large softening brush on everything outside the figure.
REGIONS = {'face': (0.41, 0.37, 0.15, 0.16), 'figure': (0.42, 0.55, 0.40, 0.50)}
AREAS = {}         # v4 area masks (bg, skin, hair, clothes, face, features), set by planes.area_masks()
CLIPS = {}         # v4: area name -> soft 0..1 mask a block-in stroke is held inside (a painter cuts in along the silhouette)
FEATURES = []      # focal ellipses [cx, cy, rx, ry]: eyes, mouth, nose, lenses (focus weight >= .95), set by plan()
EYE_FEATURES = set()   # indices of the features that hold an eye
LINE_FEATURES = []     # crisp-edged features (focus weight 1.0: eyes, lips, glasses) that get the 勾勒 lines; a soft feature
                       # (weight .95, e.g. the nose) keeps soft edges


def tag_feature(st):
    k = feature_index(*np.mean(st['points'], 0))
    if k >= 0:
        st['feature'] = k
        st['eye'] = k in EYE_FEATURES
    return k


def ellipse_mask(e, grow=1.):
    y, x = np.mgrid[0:H, 0:W].astype(float)
    cx, cy, rx, ry = e[:4]
    return ((x - cx * W) / (rx * W * grow)) ** 2 + ((y - cy * H) / (ry * H * grow)) ** 2 <= 1


def feature_index(x, y):
    """Which focal feature a point lies in (-1: none)."""
    for k, (cx, cy, rx, ry) in enumerate(FEATURES):
        if ((x - cx * W) / (rx * W)) ** 2 + ((y - cy * H) / (ry * H)) ** 2 <= 1:
            return k
    return -1


def skin_mask(ref_rgb):
    """Skin of the face: inside the face ellipse, skin-coloured in the target, outside the focal features."""
    lab = rgb2lab(ndimage.gaussian_filter(ref_rgb, (3, 3, 0)))
    skin = ellipse_mask(REGIONS['face']) & (lab[..., 0] > 25) & (lab[..., 1] > 3) & (lab[..., 2] > 6) & (lab[..., 1] < 40)
    for e in FEATURES:
        skin &= ~ellipse_mask(e)
    return ndimage.binary_opening(skin, iterations=2)


def eye_touches(ref, eyes, rng):
    """点睛: the painter's last touches in each eye, read from the photo: the pupil laid dark, then one crisp catchlight."""
    out = []
    lum = ref @ [.299, .587, .114]
    fine = ndimage.gaussian_filter(lum, 1.)
    soft = ndimage.gaussian_filter(lum, 2.5)
    y, x = np.mgrid[0:H, 0:W]
    for cx, cy, rx, ry in eyes:
        inside = ((x - cx * W) / (rx * W)) ** 2 + ((y - cy * H) / (ry * H)) ** 2 <= .55
        py, px = np.unravel_index(np.argmin(np.where(inside, soft, 9)), soft.shape)
        near = inside & (np.hypot(x - px, y - py) <= rx * W * .32)
        hy, hx = np.unravel_index(np.argmax(np.where(near, fine, -9)), fine.shape)
        dark = ref[py - 1:py + 2, px - 1:px + 2].reshape(-1, 3).mean(0) * .82
        light = ref[hy - 1:hy + 2, hx - 1:hx + 2].reshape(-1, 3).max(0)
        light = light + (1 - light) * .35
        for (qx, qy), colour, R, ln in (((px, py), dark, 3.4, 2.5), ((hx, hy), light, 2.1, 1.2)):
            a = rng.uniform(0, np.pi)
            d = np.array([np.cos(a), np.sin(a)]) * ln
            out.append(dict(points=[(float(qx - d[0]), float(qy - d[1])), (float(qx + d[0]), float(qy + d[1]))], R=float(R),
                            rgb=[float(c) for c in colour], mix=900 + (R < 3), mix_rgb=[float(c) for c in colour],
                            medium='oil_detail', seed=int(rng.integers(1 << 30)), load=1.3, press=(.85, .7), wobble=0))
    return sorted(out, key=lambda st: st['mix'])     # one dark mix for both pupils, then one light for both catchlights


def focus_map(ellipses, floor=.12):
    y, x = np.mgrid[0:H, 0:W].astype(float)
    f = np.full((H, W), .0)
    for cx, cy, rx, ry, weight in ellipses:
        d = np.sqrt(((x - cx * W) / (rx * W)) ** 2 + ((y - cy * H) / (ry * H)) ** 2)
        f = np.maximum(f, weight * np.exp(-np.clip(d - 1, 0, None) ** 2 * 2.2) * np.where(d < 1, 1, 1))
    return np.clip(f, floor, 1)


CROP_TOP = .35     # where the canvas window sits in a taller picture (0 = top, 1 = bottom)


def load_subject(path, crop_top=None):
    img = Image.open(path).convert('RGB')
    w0, h0 = img.size
    crop_h = round(w0 * H / W)
    top = int((h0 - crop_h) * (CROP_TOP if crop_top is None else crop_top))
    sub = img.crop((0, top, w0, top + crop_h)).resize((W, H), Image.LANCZOS)
    return np.asarray(sub, float) / 255, (top, crop_h, w0), (h0, w0)


def plan(subject_path, sketch_path, out, focus_ellipses, signature=None, eyes=None, photo_path=None, layers=None):
    """subject_path is what the strokes aim at (the photo in v1, a painted target in v2); photo_path, if given, is the
    photo pinned beside the canvas in the film (defaults to the subject)."""
    layers = layers or LAYERS
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    ref, crop, shape = load_subject(subject_path)
    shown = load_subject(photo_path)[0] if photo_path else ref
    Image.fromarray((shown * 255).astype(np.uint8)).save(out / 'reference.png')
    Image.fromarray((ref * 255).astype(np.uint8)).save(out / 'target.png')
    tooth = canvas_tooth(np.random.default_rng(SEED + 1))
    np.save(out / 'tooth.npy', tooth.astype(np.float32))
    canvas = np.tile(CANVAS_RGB, (H, W, 1))
    height = np.zeros((H, W))
    thick = np.zeros((H, W))
    focus = focus_map(focus_ellipses, .35 if layers is LAYERS_V2 else .12)   # v2 also refines hair, clothes, background
    FEATURES[:] = [tuple(e[:4]) for e in focus_ellipses if e[4] >= .95] if layers is LAYERS_V2 else []
    LINE_FEATURES[:] = [tuple(e[:4]) for e in focus_ellipses if e[4] >= .999] if layers is LAYERS_V2 else []
    EYE_FEATURES.clear()
    for ex, ey, *_ in (eyes or []):     # the smallest feature around each eye is the eye (a lens around it is not)
        around = [k for k, f in enumerate(FEATURES) if ((ex - f[0]) / f[2]) ** 2 + ((ey - f[1]) / f[3]) ** 2 <= 1]
        if around:
            EYE_FEATURES.add(min(around, key=lambda k: FEATURES[k][2] * FEATURES[k][3]))
    sketch = []
    passes = []
    started = time.time()

    def run(name, label, strokes):
        for st in strokes:
            render(st, canvas, height, thick, tooth)
        passes.append(dict(name=name, label=label, strokes=strokes))
        Image.fromarray((canvas * 255).astype(np.uint8)).save(out / f'pass-{len(passes):02d}-{name}.png')
        print(f'{name:8s} {len(strokes):5d} strokes  {time.time() - started:6.1f}s', flush=True)

    run('wash', '铺底色', wash_strokes(rng))
    if sketch_path:
        sketch = sketch_strokes(sketch_path, shape, crop, rng)
        run('sketch', '起稿', sketch)
    tip = (W * .2, H * .1)
    for spec in layers:
        if spec.get('only') and spec['only'] not in REGIONS:    # e.g. no --texture region: no 干擦 pass
            continue
        strokes = plan_layer(spec, ref, canvas, focus, rng, tip)
        if spec['name'] == 'accent' and eyes and not spec.get('touches_after'):     # v1: touches inside the accent pass
            touches = eye_touches(ref, eyes, rng)
            for st in touches:
                st['focal'] = True
                st['eye'] = True
            strokes = strokes + touches
        run(spec['name'], spec['label'], strokes)
        if strokes:
            tip = strokes[-1]['points'][-1]
        if spec.get('feather_after'):
            strokes = blend_strokes(ref, canvas, focus, rng, R=5., start_tip=tip, region=skin_mask(ref), toward=.55,
                                    smooth_pct=75, systematic=True)
            run('feather', '柔化', strokes)
            if strokes:
                tip = strokes[-1]['points'][-1]
        if spec.get('lines_after') and sketch:
            raw = line_strokes(sketch, ref, rng)
            cuts = painter_order([st for st in raw if st.get('cut')], np.zeros(1), tip, rng)
            lines = painter_order([st for st in raw if not st.get('cut')], np.zeros(1), cuts[-1]['points'][-1] if cuts else tip, rng)
            strokes = cuts + lines
            for st in strokes:
                st['mix'] = 0
                tag_feature(st)
            run('lines', '勾勒', strokes)
            if strokes:
                tip = strokes[-1]['points'][-1]
        if spec.get('touches_after') and eyes:
            strokes = eye_touches(ref, eyes, rng)
            for st in strokes:
                st['focal'] = True
                st['eye'] = True
            run('dianjing', '点睛', strokes)
            if strokes:
                tip = strokes[-1]['points'][-1]
        if spec.get('blend_after'):
            background = ~(focus_map([list(REGIONS['figure']) + [1.]], 0) > .9)
            strokes = blend_strokes(ref, canvas, focus, rng, R=20., start_tip=tip, region=background)
            strokes += blend_strokes(ref, canvas, focus, rng, R=7., start_tip=strokes[-1]['points'][-1] if strokes else tip,
                                     region=skin_mask(ref), toward=.35, smooth_pct=75, systematic=True)
            run('blend', '柔化', strokes)
            if strokes:
                tip = strokes[-1]['points'][-1]
    if signature:
        text, box = signature
        x0, y0, w, h = box
        under = canvas[y0:y0 + h, x0:x0 + w].mean((0, 1))
        colour = (.80, .52, .30) if under.mean() < .45 else (.42, .16, .12)   # ochre on dark paint, red-brown on light
        run('sign', '落款', signature_strokes(text, box, colour, rng))
    Image.fromarray((canvas * 255).astype(np.uint8)).save(out / 'final-flat.png')
    Image.fromarray((shade(canvas, height, thick, tooth) * 255).astype(np.uint8)).save(out / 'final.png')
    (out / 'plan.json').write_text(json.dumps(dict(W=W, H=H, passes=passes, focus=focus_ellipses, face=list(REGIONS['face'])),
                                              ensure_ascii=False))
    return passes


def shade(canvas, height, thick, tooth, k=.6):
    """Raking light from the upper left over paint relief and the canvas weave the thin paint still shows."""
    relief = ndimage.gaussian_filter(height, .7) + .35 * tooth * (1 - np.clip(thick, 0, 1)) ** 1.5
    gy, gx = np.gradient(relief)
    light = 1 + k * (gx * .55 + gy * .75) * -1
    light = np.clip(light, .82, 1.12)
    return np.clip(canvas * light[..., None], 0, 1)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Plan and paint a portrait (writes passes, final.png, plan.json).')
    ap.add_argument('cmd', choices=['plan'])
    ap.add_argument('subject', help='what the strokes aim at: the photo (v1) or a painted target registered to it (v2)')
    ap.add_argument('sketch', help="registered line sketch for the underdrawing, or '-'")
    ap.add_argument('out')
    ap.add_argument('focus', help='JSON [[cx, cy, rx, ry, weight], ...] in canvas fractions')
    ap.add_argument('signature', nargs='?', default='-', help='JSON ["text", [x, y, w, h]] or -')
    ap.add_argument('eyes', nargs='?', default=None, help='JSON [[cx, cy, rx, ry], ...]')
    ap.add_argument('--photo', help='photo pinned beside the canvas (default: the subject)')
    ap.add_argument('--style', choices=['v1', 'v2', 'v4', 'v5'], default='v1')
    ap.add_argument('--crop-top', type=float, default=.35, help='canvas window position in a taller picture (0 top, 1 bottom)')
    ap.add_argument('--face', help='JSON [cx, cy, rx, ry] of the face (v2 texture pass leaves it alone)')
    ap.add_argument('--texture', help='JSON [cx, cy, rx, ry] of skin that keeps a fine texture, e.g. stubble (v2 干擦 pass)')
    ap.add_argument('--preset', default='planes', help='v4 style: planes | blocky | knife | sunlit | digital')
    ap.add_argument('--areas', help='v4: area map registered to the photo (flat skin/hair/clothes/background colours)')
    ap.add_argument('--figure', help='JSON [cx, cy, rx, ry] of the whole figure (v2 softens the background outside it)')
    a = ap.parse_args()
    CROP_TOP = a.crop_top
    if a.face:
        REGIONS['face'] = tuple(json.loads(a.face))
    if a.figure:
        REGIONS['figure'] = tuple(json.loads(a.figure))
    if a.texture:
        REGIONS['texture'] = tuple(json.loads(a.texture))
    if a.style == 'v5':
        import paint as engine
        import planes
        import forms
        engine.REGIONS.update(REGIONS)
        engine.CROP_TOP = CROP_TOP
        planes.AREA_MAP = a.areas
        forms.PRESET = a.preset
        forms.plan_v5(a.subject, None if a.sketch == '-' else a.sketch, a.out, json.loads(a.focus),
                      None if a.signature == '-' else json.loads(a.signature), json.loads(a.eyes) if a.eyes else None,
                      photo_path=a.photo)
    elif a.style == 'v4':
        import paint as engine          # planes.py imports this file as the module 'paint': hand it the settings
        import planes
        engine.REGIONS.update(REGIONS)
        engine.CROP_TOP = CROP_TOP
        planes.AREA_MAP = a.areas
        planes.PRESET = a.preset
        planes.plan_v4(a.subject, None if a.sketch == '-' else a.sketch, a.out, json.loads(a.focus),
                       None if a.signature == '-' else json.loads(a.signature), json.loads(a.eyes) if a.eyes else None,
                       photo_path=a.photo)
    else:
        plan(a.subject, None if a.sketch == '-' else a.sketch, a.out, json.loads(a.focus),
             None if a.signature == '-' else json.loads(a.signature), json.loads(a.eyes) if a.eyes else None,
             photo_path=a.photo, layers=LAYERS_V2 if a.style == 'v2' else LAYERS_V1)
