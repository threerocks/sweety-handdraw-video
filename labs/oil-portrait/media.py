"""Media for the sketch (素描) and anime (动漫插画) modes, and the surfaces a film paints on.

Sketch: graphite on paper (darkens only, light pressure catches just the paper's tooth), a blending stump that smears
what is already there, and an eraser that lifts graphite back to the paper. Anime: a digital pen on a line layer that
sits above the colours, a hard round brush and an airbrush on the colour layer, each held inside its region (the way
an illustrator paints under the line art with a clipping mask).

A stroke uses the oil engine's geometry (paint.geometry: centreline, taper, wobble) and its patch format (box, gy,
gx, ss, alpha, paint, wpx), so film.py reveals it by arc length exactly as it does an oil stroke.
"""
import math

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

import paint
from paint import W, H, geometry

PAPER = None          # per-pixel paper colour (H, W, 3), set by paper()
GRAIN = None          # paper tooth 0..1 (H, W)
CLIPS = paint.CLIPS   # region name -> soft 0..1 mask (shared with the oil engine and loaded by film.py)
LINE_RGB = np.array([.24, .13, .11])
LEVELS = {}           # name -> (H, W, 3) colour a layer builds toward, pixel by pixel (saved as level-<name>.npy)
ALPHAS = {}           # anime: name -> (H, W) coverage of the line art (saved as alpha-<name>.npy)
LABELS = {}           # anime: name -> (H, W) int32 map of shapes; a stroke with clip=[name, id] is held inside shape id
                      # (a clipping mask / a selection), saved as label-<name>.npy
PAPER_TINT = np.array([1.01, 1., .975])   # graphite reads a touch warm on cream paper


def paper(rng, rgb=(.952, .938, .906)):
    """Drawing paper: a fine irregular tooth with a faint fibre direction."""
    g = ndimage.gaussian_filter(rng.standard_normal((H, W)), .75)
    g = g / g.std() + .55 * (lambda n: n / n.std())(ndimage.gaussian_filter(rng.standard_normal((H, W)), 2.4))
    g = g + .3 * (lambda n: n / n.std())(ndimage.gaussian_filter(rng.standard_normal((H, W)), (.6, 3.2)))
    g = (g - g.mean()) / g.std()
    tooth = 1 / (1 + np.exp(-1.7 * g))
    ground = np.asarray(rgb) * (1 + .03 * (tooth - .5))[..., None]
    return tooth.astype(np.float32), ground


def set_paper(tooth, ground):
    global GRAIN, PAPER
    GRAIN, PAPER = tooth, ground


# ------------------------------------------------------------------------------------------------ strokes
def footprint(stroke, shape):
    """Pixels a stroke can touch: arc position, lateral offset and half-width there, relative radius rr (1 at the
    edge, round caps)."""
    pts, s, tan, nrm, w = geometry(stroke)
    L = s[-1]
    wmax = w.max()
    Hc, Wc = shape
    x0, y0 = np.floor(pts.min(0) - wmax - 3).astype(int)
    x1, y1 = np.ceil(pts.max(0) + wmax + 3).astype(int)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(Wc, x1), min(Hc, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    pix = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)
    d, idx = cKDTree(pts).query(pix, distance_upper_bound=wmax + 2.5, workers=-1)
    ok = np.isfinite(d)
    if not ok.any():
        return None
    pix, idx = pix[ok], idx[ok]
    v = pix - pts[idx]
    lat = (v * nrm[idx]).sum(1)
    lon = (v * tan[idx]).sum(1)
    n = len(pts)
    ss = s[idx] + np.where(((idx == 0) & (lon < 0)) | ((idx == n - 1) & (lon > 0)), lon, 0.)
    wpx = w[idx]
    over = np.where(ss < 0, -ss, np.where(ss > L, ss - L, 0.))
    rr = np.sqrt((lat / wpx) ** 2 + (over / wpx) ** 2)
    return dict(box=(y0, y1, x0, x1), gy=pix[:, 1].astype(np.int32), gx=pix[:, 0].astype(np.int32), ss=ss, lat=lat,
                wpx=wpx, rr=rr, L=float(L), path=pts, s=s, R=stroke['R'])


def hard_edge(f, aa=1.):
    """Anti-aliased coverage of a hard round tip."""
    return np.clip((1 - f['rr']) * f['wpx'] / aa + .5, 0, 1)


def patch_of(f, alpha, colour, layer=None):
    keep = alpha > .003
    return dict(box=f['box'], gy=f['gy'][keep], gx=f['gx'][keep], ss=f['ss'][keep].astype(np.float32),
                alpha=alpha[keep].astype(np.float32), paint=np.asarray(colour, np.float32)[keep] if np.ndim(colour) == 2
                else np.tile(np.asarray(colour, np.float32), (int(keep.sum()), 1)),
                hgt=np.zeros(int(keep.sum()), np.float32), L=f['L'], wpx=f['wpx'][keep].astype(np.float32),
                path=f['path'].astype(np.float32), s=f['s'].astype(np.float32), layer=layer)


def label_clip(name, k, box, grow=0):
    """Soft coverage of shape k of a label map inside box: anti-aliased, and grown half a pixel so two shapes painted
    side by side leave no light seam between them. With grow, the shape also reaches that many pixels into shapes
    with a higher number (numbers are the painting order: what is painted later goes over the overlap, so under the
    line art and at every border the colours meet with no paper between them)."""
    y0, y1, x0, x1 = box
    p = 2 + grow
    ya, yb, xa, xb = max(0, y0 - p), min(H, y1 + p), max(0, x0 - p), min(W, x1 + p)
    lab = LABELS[name][ya:yb, xa:xb]
    own = lab == k
    if grow:
        own = own | (ndimage.binary_dilation(own, iterations=grow) & (lab > k))
    m = np.clip(ndimage.gaussian_filter(own.astype(np.float32), .7) * 2., 0, 1)
    if grow:
        m[(lab < k) & (lab > 0)] = 0.      # nothing of it on what was painted before: no fringe of this colour left there
    return m, ya, xa


def clipped(stroke, f, a):
    c = stroke.get('clip')
    if not c:
        return a
    if isinstance(c, str):
        return a * CLIPS[c][f['gy'], f['gx']]
    m, ya, xa = label_clip(c[0], c[1], f['box'], c[2] if len(c) > 2 else 0)
    return a * m[f['gy'] - ya, f['gx'] - xa]


# graphite: the sheet holds a coverage c in 0..1 (0 bare paper, 1 the densest graphite), read back from its colour. A
# pencil stroke covers a share d of the paper still showing (so crossing strokes add up), up to what its grade can
# reach: a 2H never gets as dark as a 6B. d is the pencil's softness times the pressure, which changes along the
# stroke but not across it; a light touch leaves more of the paper's grain showing.
GRADES = {'2H': (.34, .46), 'HB': (.52, .66), '2B': (.72, .84), '4B': (.88, .94), '6B': (1., 1.)}   # softness, ceiling
DARK = np.array([.082, .082, .094])     # the densest graphite
_LUM = np.array([.299, .587, .114])


def coverage(rgb, ground):
    pv = ground @ _LUM
    return np.clip((pv - rgb @ _LUM) / (pv - DARK @ _LUM), 0, 1)


def tone(c, ground):
    return ground + (DARK * PAPER_TINT - ground) * c[..., None]


def pressure_along(stroke, f):
    """The hand's pressure at each pixel of the footprint: the planner's profile along the stroke (pp, evenly spaced
    along its length), a light landing and lift, and a slow tremor."""
    u = np.clip(f['ss'] / max(f['L'], 1e-6), 0, 1)
    pp = stroke.get('pp')
    p = np.interp(u, np.linspace(0, 1, len(pp)), pp) if pp else np.ones(len(u))
    rng = np.random.default_rng(stroke['seed'] + 5)
    n = max(3, int(f['L'] / 14) + 2)
    p = p * (1 + stroke.get('tremor', .1) * np.interp(u, np.linspace(0, 1, n), rng.standard_normal(n)))
    taper = np.clip(f['wpx'] / stroke['R'], 0, 1) ** 1.5
    return np.clip(p * stroke.get('pressure', 1.) * taper, 0, 1)


def graphite(stroke, canvas):
    """A pencil stroke: the point (a line of even weight across its width) or the side of the lead (side=True: a
    broad, soft-flanked, grainy band)."""
    f = footprint(stroke, canvas.shape[:2])
    if f is None:
        return None
    side = stroke.get('side', False)
    prof = hard_edge(f, .9) * ((1 - np.clip(f['rr'], 0, 1) ** 2) ** .7 if side else 1.)
    p = pressure_along(stroke, f)
    soft, ceil = GRADES[stroke.get('grade', 'HB')]
    g = GRAIN[f['gy'], f['gx']]
    ground = PAPER[f['gy'], f['gx']]
    c = coverage(canvas[f['gy'], f['gx']], ground)
    # the paper's grain shows under a light touch on bare paper; under heavy graphite its valleys are already full
    grain = np.clip(1 + (.3 if side else .45) * (g - .5) * 2 * (1 - .85 * p) * (1 - .8 * c), .1, 2.)
    d = np.clip(soft * p * grain, 0, 1) * clipped(stroke, f, prof)
    c2 = np.maximum(c, np.minimum(ceil, 1 - (1 - c) * (1 - d)))
    return patch_of(f, (c2 > c + 1e-4) * np.clip(prof * 4, 0, 1), tone(c2, ground))


def stump(stroke, canvas):
    """A blending stump or tissue: smears the graphite already there into a soft tone (lines melt into each other,
    the paper's grain fills in; a trace travels with the stump)."""
    f = footprint(stroke, canvas.shape[:2])
    if f is None:
        return None
    y0, y1, x0, x1 = f['box']
    reach = max(2., stroke['R'] * stroke.get('spread', .5))     # how far graphite is dragged along the stroke
    pad = int(reach * 2) + 3
    ya, yb, xa, xb = max(0, y0 - pad), min(canvas.shape[0], y1 + pad), max(0, x0 - pad), min(canvas.shape[1], x1 + pad)
    cross = stroke.get('cross', .8)               # a tissue also evens the tone a little across its path
    box = ndimage.gaussian_filter(canvas[ya:yb, xa:xb], (cross, cross, 0))
    # a stump drags graphite the way it moves: the tone is averaged along the stroke far more than across it, so
    # a rubbed passage keeps soft streaks in the stroke's direction instead of going out of focus
    tan = np.gradient(f['path'], axis=0)
    tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-9
    u = np.clip(f['ss'], 0, f['L'])
    tx, ty = np.interp(u, f['s'], tan[:, 0]), np.interp(u, f['s'], tan[:, 1])
    smeared = np.zeros((len(u), 3))
    wsum = 0.
    for o in np.linspace(-2, 2, 9):
        w = math.exp(-.5 * o * o)
        yy = np.clip(f['gy'] + ty * o * reach - ya, 0, yb - ya - 1)
        xx = np.clip(f['gx'] + tx * o * reach - xa, 0, xb - xa - 1)
        for c in range(3):
            smeared[:, c] += w * ndimage.map_coordinates(box[..., c], [yy, xx], order=1)
        wsum += w
    smeared = smeared / wsum * .997
    prof = np.clip(1 - f['rr'] ** 2, 0, 1) ** .8
    a = clipped(stroke, f, prof * stroke.get('opacity', .7))
    return patch_of(f, a, smeared)


def eraser(stroke, canvas):
    """An eraser pen (hard, narrow) or a kneaded eraser (soft): lifts a share of the graphite under it, by pressure."""
    f = footprint(stroke, canvas.shape[:2])
    if f is None:
        return None
    prof = hard_edge(f, 1.2) if stroke['R'] < 3.2 else np.clip(1 - f['rr'] ** 2.5, 0, 1)
    e = np.clip(pressure_along(stroke, f) * stroke.get('opacity', .6), 0, 1) * clipped(stroke, f, prof)
    ground = PAPER[f['gy'], f['gx']]
    c = coverage(canvas[f['gy'], f['gx']], ground)
    return patch_of(f, (e > .01) * np.clip(prof * 4, 0, 1), tone(c * (1 - e), ground))


def colour_of(stroke, f):
    return LEVELS[stroke['level']][f['gy'], f['gx']] if stroke.get('level') else stroke['rgb']


def pen(stroke, canvas):
    """Digital line art: a hard tip whose width follows the pen pressure, on the line layer (with a line map, the
    stroke lays that line's own coverage and colour: the line art it is tracing)."""
    f = footprint(stroke, canvas.shape[:2])
    if f is None:
        return None
    a = hard_edge(f, .85) * stroke.get('opacity', 1.)
    if stroke.get('alpha'):
        a = a * ALPHAS[stroke['alpha']][f['gy'], f['gx']]
    return patch_of(f, a, colour_of(stroke, f), layer=stroke.get('layer', 'line'))


def cel(stroke, canvas):
    """A hard round brush on the colour layer, held inside its region."""
    f = footprint(stroke, canvas.shape[:2])
    if f is None:
        return None
    a = clipped(stroke, f, hard_edge(f, 1.) * stroke.get('opacity', 1.))
    if stroke.get('alpha'):
        a = a * ALPHAS[stroke['alpha']][f['gy'], f['gx']]
    return patch_of(f, a, colour_of(stroke, f), layer=stroke.get('layer'))


def air(stroke, canvas):
    """A soft airbrush: a wide gaussian spray at low flow, held inside its region."""
    f = footprint(stroke, canvas.shape[:2])
    if f is None:
        return None
    a = clipped(stroke, f, np.exp(-3. * f['rr'] ** 2) * (f['rr'] < 1.15) * stroke.get('opacity', .3))
    return patch_of(f, a, colour_of(stroke, f), layer=stroke.get('layer'))


def adjust(stroke, canvas):
    """The last adjustment (the line art tinted toward the colours beside it, a soft glow, the tone curve): the whole
    picture eases to its finished state on a layer above everything, like dragging an adjustment slider."""
    gy, gx = np.mgrid[0:H, 0:W]
    gy, gx = gy.ravel().astype(np.int32), gx.ravel().astype(np.int32)
    n = len(gy)
    return dict(box=(0, H, 0, W), gy=gy, gx=gx, ss=np.zeros(n, np.float32), alpha=np.ones(n, np.float32),
                paint=LEVELS[stroke['level']].reshape(-1, 3).astype(np.float32), hgt=np.zeros(n, np.float32), L=2.5,
                wpx=np.zeros(n, np.float32), path=np.asarray(stroke['points'], np.float32),
                s=np.array([0., 2.5], np.float32), layer=stroke.get('layer', 'adjust'))


RENDER = {'graphite': graphite, 'stump': stump, 'eraser': eraser, 'pen': pen, 'cel': cel, 'air': air,
          'adjust': adjust}


# ----------------------------------------------------------------------------------------------- surfaces
def reveal(patch, upto):
    a = patch['alpha']
    if upto is not None:
        a = a * np.clip((upto + .45 * patch['wpx'] - patch['ss']) / 2.5, 0, 1)
    return a


class PaperSurface:
    """Sketch: one sheet of paper."""
    kind = 'sketch'

    def __init__(self, ground):
        self.canvas = np.array(ground, float)

    def render(self, st):
        return RENDER[st['medium']](st, self.canvas)

    def commit(self, p, upto=None):
        a = reveal(p, upto)[:, None]
        gy, gx = p['gy'], p['gx']
        self.canvas[gy, gx] = self.canvas[gy, gx] * (1 - a) + p['paint'] * a

    def save(self, p):
        return self.canvas[p['gy'], p['gx']].copy()

    def restore(self, p, saved):
        self.canvas[p['gy'], p['gx']] = saved

    def image(self):
        return self.canvas


class LayerSurface:
    """Anime: a colour layer with line layers above it (premultiplied colour + alpha). A rough sketch layer can be
    hidden once the clean line art is done, as an illustrator does."""
    kind = 'anime'

    def __init__(self, ground=(1., 1., 1.), layers=('rough', 'line')):
        self.canvas = np.tile(np.asarray(ground, float), (H, W, 1))
        self.order = list(layers)
        self.rgb = {n: np.zeros((H, W, 3)) for n in layers}
        self.a = {n: np.zeros((H, W)) for n in layers}
        self.shown = {n: 1. for n in layers}

    def render(self, st):
        return RENDER[st['medium']](st, self.canvas)

    def commit(self, p, upto=None):
        a = reveal(p, upto)
        gy, gx = p['gy'], p['gx']
        n = p.get('layer')
        if n:
            self.rgb[n][gy, gx] = self.rgb[n][gy, gx] * (1 - a[:, None]) + p['paint'] * a[:, None]
            self.a[n][gy, gx] = self.a[n][gy, gx] * (1 - a) + a
        else:
            self.canvas[gy, gx] = self.canvas[gy, gx] * (1 - a[:, None]) + p['paint'] * a[:, None]

    def save(self, p):
        n = p.get('layer')
        gy, gx = p['gy'], p['gx']
        return (self.rgb[n][gy, gx].copy(), self.a[n][gy, gx].copy()) if n else self.canvas[gy, gx].copy()

    def restore(self, p, saved):
        n = p.get('layer')
        gy, gx = p['gy'], p['gx']
        if n:
            self.rgb[n][gy, gx], self.a[n][gy, gx] = saved
        else:
            self.canvas[gy, gx] = saved

    def image(self):
        out = self.canvas.copy()
        for n in self.order:
            k = self.shown[n]
            if k > 0:
                out = out * (1 - k * self.a[n][..., None]) + k * self.rgb[n]
        return out
