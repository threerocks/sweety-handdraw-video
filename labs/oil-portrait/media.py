"""Media for the sketch (素描) and anime (动漫插画) modes, and the surfaces a film paints on.

Sketch: graphite on paper (darkens only, light pressure catches just the paper's tooth), a blending stump that smears
what is already there, and an eraser that lifts graphite back to the paper. Anime: a digital pen on a line layer that
sits above the colours, a hard round brush and an airbrush on the colour layer, each held inside its region (the way
an illustrator paints under the line art with a clipping mask).

A stroke uses the oil engine's geometry (paint.geometry: centreline, taper, wobble) and its patch format (box, gy,
gx, ss, alpha, paint, wpx), so film.py reveals it by arc length exactly as it does an oil stroke.
"""
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


def label_clip(name, k, box):
    """Soft coverage of shape k of a label map inside box: anti-aliased, and grown half a pixel so two shapes painted
    side by side leave no light seam between them."""
    y0, y1, x0, x1 = box
    ya, yb, xa, xb = max(0, y0 - 2), min(H, y1 + 2), max(0, x0 - 2), min(W, x1 + 2)
    m = ndimage.gaussian_filter((LABELS[name][ya:yb, xa:xb] == k).astype(np.float32), .7)
    return np.clip(m * 2., 0, 1), ya, xa


def clipped(stroke, f, a):
    c = stroke.get('clip')
    if not c:
        return a
    if isinstance(c, str):
        return a * CLIPS[c][f['gy'], f['gx']]
    m, ya, xa = label_clip(c[0], c[1], f['box'])
    return a * m[f['gy'] - ya, f['gx'] - xa]


def graphite(stroke, canvas):
    """A pencil line or the side of the lead: darkens toward the lead's own value; light pressure only marks the tops
    of the paper's tooth, heavy pressure fills the valleys too."""
    f = footprint(stroke, canvas.shape[:2])
    if f is None:
        return None
    side = stroke['R'] > 2.6                                   # shading with the side of the lead: soft flanks
    prof = hard_edge(f, .9) * ((1 - np.clip(f['rr'], 0, 1) ** 2) ** .6 if side else 1.)
    taper = np.clip(f['wpx'] / stroke['R'], 0, 1) ** .5
    pr = stroke.get('pressure', .6) * taper
    t = GRAIN[f['gy'], f['gx']]
    catch = np.clip((t - (1 - 1.15 * pr)) / .24, 0, 1)
    a = clipped(stroke, f, prof * catch * stroke.get('opacity', .85))
    under = canvas[f['gy'], f['gx']]
    if stroke.get('level'):            # the tone this layer of the drawing builds toward, pixel by pixel
        lead = LEVELS[stroke['level']][f['gy'], f['gx']]
    else:
        lead = stroke['lead'] * PAPER_TINT
    return patch_of(f, a, np.minimum(under, lead))


def stump(stroke, canvas):
    """A blending stump or tissue: smears the graphite already there into a soft tone (a little of it travels)."""
    f = footprint(stroke, canvas.shape[:2])
    if f is None:
        return None
    y0, y1, x0, x1 = f['box']
    sig = max(1.5, stroke['R'] * .55)
    pad = int(3 * sig)
    ya, yb, xa, xb = max(0, y0 - pad), min(canvas.shape[0], y1 + pad), max(0, x0 - pad), min(canvas.shape[1], x1 + pad)
    blur = ndimage.gaussian_filter(canvas[ya:yb, xa:xb], (sig, sig, 0))
    smeared = blur[f["gy"] - ya, f["gx"] - xa] * .997               # the stump carries a trace of graphite
    prof = np.clip(1 - f['rr'] ** 2, 0, 1) ** .8
    a = clipped(stroke, f, prof * stroke.get('opacity', .7))
    return patch_of(f, a, np.minimum(smeared, canvas[f['gy'], f['gx']] * 1.04))


def eraser(stroke, canvas):
    """A kneaded eraser dab or an eraser pen: lifts graphite toward the bare paper."""
    f = footprint(stroke, canvas.shape[:2])
    if f is None:
        return None
    prof = hard_edge(f, 1.4) if stroke['R'] < 3 else np.clip(1 - f['rr'] ** 3, 0, 1)
    a = clipped(stroke, f, prof * stroke.get('opacity', .6))
    under = canvas[f['gy'], f['gx']]
    if stroke.get('level'):            # lift only as far as the drawing's own light there
        return patch_of(f, a, np.maximum(under, LEVELS[stroke['level']][f['gy'], f['gx']]))
    return patch_of(f, a, PAPER[f['gy'], f['gx']])


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
