"""Film a painting plan: re-paint it stroke by stroke with a person's timing, a hand holding the brush (its bristles
carry the paint being used), a palette where every new mix is made, and the reference photo pinned under the canvas.

Timing: each stroke, lift and palette visit gets a person's real duration; each pass is then time-lapsed to its video
budget, starting slow and speeding up (as an edited speed-paint does). A palette visit is shown when the lapse is
gentle enough to read; in fast stretches the new mix appears on the palette as a cut.

usage: film.py PLAN_DIR [--out film.mp4] [--stills N] [--until SECONDS]
"""
import argparse
import glob
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import paint
import media
from paint import W, H, render, commit, geometry, shade

sys.path.insert(0, str(paint.SKILL))
from motion_profile import distance_fraction  # noqa: E402

HERE = Path(__file__).parent
KIND = ['oil']                         # oil | sketch (素描, paper) | anime (动漫插画, layers): set from plan.json
TITLES = {'oil': '油画肖像写生', 'sketch': '素描肖像', 'anime': '动漫插画'}
TOOLS = HERE / 'src' / 'tools-sketch.png'
FW, FH, FPS = 1280, 1920, 30
CX, CY = 80, 150                       # canvas origin in the film
TRAY = (80, 1604, 790, 284)            # palette tray x, y, w, h
PHOTO = (948, 1610, 214, 272)
HAND = HERE / 'src' / 'hand-brush.png'
HAND_SCALE = .7
NIB = (44., 42.)                       # where the bristles touch the canvas, in the hand asset
FONT_SANS = '/System/Library/Fonts/Hiragino Sans GB.ttc'
FONT_HAND = next(iter(sorted(glob.glob('/System/Library/AssetsV2/com_apple_MobileAsset_Font*/*/AssetData/Hannotate.ttc'))), FONT_SANS)
WATERMARK = '制作：@leo_xiaolei/刘老板'
BG = np.array([233, 228, 219], float)
INK = (43, 38, 34)

BUDGET = {'wash': 4.6, 'sketch': 8.5, 'block': 11.5, 'form': 10.0, 'refine': 6.5, 'detail': 5.5, 'accent': 7.0, 'sign': 3.0}
BUDGET_V2 = {'wash': 3.5, 'sketch': 7.0, 'block': 8.0, 'form': 7.5, 'blend': 3.0, 'refine': 5.5, 'texture': 6.5, 'detail': 4.5,
             'polish': 4.0, 'scumble': 3.0, 'feather': 3.0, 'lines': 0., 'accent': 5.5, 'dianjing': 0., 'sign': 3.0}
ZOOM = 2.0              # camera push-in on the face while the features are painted
HEADER, HEADER_FADE = 140, 24   # rows of title and pass labels kept fixed in a close-up, and their soft lower edge
LABELS_BOTTOM = 142     # last frame row of the pass labels
ZOOM_EASE = 1.0         # seconds to push in or pull out
ZOOM_GAP = 10.          # focal stretches closer than this stay in one close-up (the passes between are on the face too)
FOLLOW = .3             # in a close-up the camera drifts this share of the way toward the (smoothed) brush
JUMP = 200.             # px: a lift longer than this is never crossed faster than HAND_PX a frame (at the min-jerk peak);
HAND_PX = 240.          # shorter hops inside one area may share a frame, as a time-lapse shows them
STROKE_PX = 700.        # a brush stroke's peak travel per frame: only the most extreme sweeps are held back
MAX_TRIPS = 3           # visible palette visits per pass (the rest of the new mixes appear as cuts)
RAMP_STROKES = 8        # strokes over which a pass speeds up from the slow start
SLOW_PASSES = {'detail', 'polish', 'lines', 'accent', 'dianjing'}   # in these passes the focal features are painted at life pace
FOCAL_LAPSE = 1.        # the eyes' finishing work (polish, eyelid lines, 点睛) at a person's real speed (user, 2026-11-10:
FEATURE_LAPSE = 3.      # 视觉重点不加速); the other features (mouth, nose, lenses) at three times life pace
LIFE_PASSES = {'polish', 'lines', 'dianjing'}
FEATURE_CAP = 8.        # seconds a pass may spend on the non-eye features (their pace rises above 3x to fit)
EARLY_LAPSE = 5.        # the features' first detail round: still clearly stroke by stroke, not yet the finishing work
SLOW_LAPSE = 2.         # a pass starts at about twice life speed
TRIP_LAPSE = 9.         # palette visits are shown only when the lapse is gentler than this
SWIRL = .55             # seconds at life pace to work a new mix on the palette
PIGMENTS = [(.96, .95, .92), (.93, .81, .56), (.78, .58, .26), (.86, .32, .21), (.56, .13, .19), (.60, .30, .16),
            (.36, .23, .16), (.16, .19, .52), (.10, .38, .33), (.11, .10, .10)]


# ----------------------------------------------------------------------------------------------- timing
def stroke_seconds(st, L):
    m, R = st['medium'], st['R']
    if st.get('seconds'):             # anime: strokes that carry their own real duration (construction, the adjustment)
        return st['seconds']
    if m == 'wash':
        return .25 + L / 1400
    if m == 'pencil':
        return .05 + L / 450
    if m == 'sign':
        return .1 + L / 240
    if m == 'graphite':
        return .04 + L / 520
    if m == 'stump':
        return .08 + L / 380
    if m == 'eraser':
        return .06 + L / 400
    if m == 'pen':
        return .08 + L / 260
    if m == 'cel':
        return .06 + L / (500 + 25 * R)
    if m == 'air':
        return .06 + L / 700
    return .12 + L / (260 + 16 * R)


def travel_seconds(d):
    return .08 + .09 * math.log2(1 + d / 30)


def smooth(e):
    e = min(1., max(0., e))
    return e * e * (3 - 2 * e)


def lapses(evs, cruise):
    r = min(1., SLOW_LAPSE / cruise)
    n = sum(e['kind'] == 'stroke' for e in evs)
    ramp = max(1., min(RAMP_STROKES, .12 * n))
    out, strokes = [], 0
    for e in evs:
        lap = cruise * (r + (1 - r) * smooth(strokes / ramp))
        out.append(min(lap, e['slow']) if e.get('slow') else lap)    # focal features: life pace or close to it
        strokes += e['kind'] == 'stroke'
    return out


EVEN_RAMP = .12                      # share of a paced motion spent speeding up, and slowing down
EVEN_PEAK = 1 / (1 - EVEN_RAMP)      # its peak speed over its mean speed (a min-jerk motion: 1.875)


def even_fraction(tau, r=EVEN_RAMP):
    """Distance travelled at time tau of a motion that eases in, cruises at an even speed and eases out: a long
    colouring stroke is drawn at a steady pace, not in one swing."""
    tau = min(1., max(0., tau))
    if tau < r:
        return tau * tau / (2 * r) / (1 - r)
    if tau > 1 - r:
        return 1 - (1 - tau) ** 2 / (2 * r) / (1 - r)
    return (tau - r / 2) / (1 - r)


def floor(e):
    """Shortest video time an event may take: no long lift jumps across the canvas between two frames."""
    d = e.get('dist', 0.)
    nib = e.get('nib')
    if nib and e['kind'] in ('stroke', 'travel'):
        # anime: the nib never moves farther than this between two frames (px on screen at the motion's peak, drawing
        # and lifted), so every motion can be followed; a close-up shows the canvas twice as large
        z = ZOOM if e.get('slow') else 1.
        if e['kind'] == 'travel':
            return EVEN_PEAK * d * z / (nib[1] * FPS)
        return max(e['st'].get('hold', 0.), EVEN_PEAK * e['L'] * z / (nib[0] * FPS))
    if e['kind'] == 'stroke' and e['st'].get('hold'):     # a stroke that is on screen at least this long
        return e['st']['hold']
    if e['kind'] == 'travel' and d > JUMP:
        return 1.875 * d / (HAND_PX * FPS)
    if e['kind'] == 'tool':                        # sketch: down to the tools, a beat to swap, never a jump cut
        return 1.875 * d / (HAND_PX * FPS) + .2
    if e['kind'] == 'mix' and e.get('trip'):
        return 1.875 * d / (HAND_PX * FPS) * e['real'] / max(1e-6, e['go'])   # the outbound move gets go/real of it
    if e['kind'] == 'stroke' and KIND[0] == 'sketch':
        # a sketch is thousands of short marks: however fast the lapse, the pencil must be seen making them. In one
        # frame a hatching motion advances at most SKETCH_ADV px sideways, and a long stroke crosses at most SKETCH_PX
        if e['st']['medium'] == 'stump':             # a stump is rubbed slowly: its whole path, back and forth
            return e['L'] / STUMP_PX / FPS
        adv = sum(g[0].get('adv', SKETCH_ADV / SKETCH_MARKS) for g in e['group']) if 'group' in e else 0.
        return max(e.get('floorL', e['L']) / SKETCH_PX, adv / SKETCH_ADV) / FPS
    if e['kind'] == 'stroke' and e.get('floorL', e['L']) > 200:
        return 1.875 * e.get('floorL', e['L']) / (STROKE_PX * FPS)
    return 0.


SKETCH_PX = 340.        # sketch: the longest stroke of a motion travels at most this far in one frame
SKETCH_MARKS = 26.      # sketch: strokes of one hatching motion laid in one frame at most, when a stroke does not say
SKETCH_ADV = 75.        # how far its hatching advances (st['adv'], the line spacing): then this many px sideways a frame
STUMP_PX = 120.         # sketch: the blending stump travels at most this far in one frame
# sketch: the hand puts one tool down and picks up another at the tools laid out below the board (frame px), and is
# drawn holding it (a sidecar-calibrated asset each); a stroke's medium says which tool lays it
TOOL_OF = {'graphite': 'pencil', 'stump': 'stump', 'eraser': 'eraser'}
TOOL_SPOTS = {'pencil': (205., 1703.), 'stump': (414., 1721.), 'eraser': (613., 1729.)}
TOOL_HANDS = {'stump': HERE / 'src' / 'hand-tang-stump.png', 'eraser': HERE / 'src' / 'hand-tang-eraser.png'}


GROUP_HOP = 90.         # sketch/anime: a lift shorter than this between strokes is part of one hatching motion
GROUP_REAL = 1.6        # seconds of life-pace drawing one grouped motion holds at most


def group_events(evs):
    """Hatching, strands, fills: strokes laid one after another a few pixels apart are one motion of the hand (it
    stays down on the paper), filmed as one event that lays its strokes as the hand passes them. Fewer, longer
    events keep a pass of thousands of short strokes readable, and its time spent drawing rather than hopping."""
    out, cur = [], None
    i = 0
    while i < len(evs):
        e = evs[i]
        nxt = evs[i + 1] if i + 1 < len(evs) else None
        if e['kind'] == 'travel' and nxt is not None and nxt['kind'] == 'stroke' and not nxt['slow'] and not e['slow']:
            hop = float(np.linalg.norm(nxt['path'][0] - cur['path'][-1])) if cur is not None else 1e9
            if hop < GROUP_HOP and cur['real'] + nxt['real'] < GROUP_REAL and not nxt['st'].get('solo') \
                    and not cur['st'].get('solo'):
                seg = np.linspace(cur['path'][-1], nxt['path'][0], max(2, int(hop / 3) + 2))[1:]
                seg_s = cur['s'][-1] + np.cumsum(np.linalg.norm(np.diff(np.vstack([cur['path'][-1:], seg]), axis=0), axis=1))
                start = seg_s[-1]
                cur['path'] = np.vstack([cur['path'], seg, nxt['path'][1:]])
                cur['s'] = np.r_[cur['s'], seg_s, start + nxt['s'][1:]]
                cur['group'].append((nxt['st'], start, start + nxt['L']))
                cur['real'] += nxt['real'] + .02 + hop / 2500
                cur['L'] = float(cur['s'][-1])
                cur['floorL'] = max(cur['floorL'], nxt['L'])
                i += 2
                continue
            cur = dict(nxt, group=[(nxt['st'], 0., nxt['L'])], floorL=nxt['L'], path=nxt['path'].copy(), s=nxt['s'].copy())
            out += [e, cur]
            i += 2
            continue
        out.append(e)
        if e['kind'] != 'pause':
            cur = None
        i += 1
    return out


def video_time(evs, lap):
    return [max(e['real'] / l, floor(e)) for e, l in zip(evs, lap)]


def fit(evs, budget):
    real = sum(e['real'] for e in evs)
    if real <= budget:
        return [1.] * len(evs)
    lo, hi = 1., 400.
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        if sum(video_time(evs, lapses(evs, mid))) > budget:
            lo = mid
        else:
            hi = mid
    return lapses(evs, hi)


class Palette:
    """The mixing tray: pigments along the top, mixes smeared below, each new mix in the next free spot."""

    def __init__(self, rng):
        x, y, w, h = TRAY
        self.w, self.h = w, h
        self.rgb = np.ones((h, w, 3)) * np.array([.97, .965, .95])
        self.height = np.zeros((h, w))
        self.thick = np.ones((h, w))
        self.tooth = np.full((h, w), .5)
        self.rng = rng
        self.count = 0
        yy, xx = np.mgrid[0:h, 0:w]
        r = 30
        dx = np.maximum(np.maximum(r - xx, xx - (w - 1 - r)), 0)
        dy = np.maximum(np.maximum(r - yy, yy - (h - 1 - r)), 0)
        self.mask = np.clip(r - np.hypot(dx, dy) + .5, 0, 1)
        for k, c in enumerate(PIGMENTS):   # paint squeezed from tubes before the session
            cx = 52 + k * (w - 104) / (len(PIGMENTS) - 1)
            for j in range(2):
                a = rng.uniform(0, 2 * np.pi)
                pts = [(cx - 14 * np.cos(a), 36 - 6 * np.sin(a) + j * 5), (cx, 32 + j * 4), (cx + 14 * np.cos(a), 36 + 6 * np.sin(a) + j * 3)]
                render(dict(points=pts, R=17. - 4 * j, rgb=list(c), medium='oil_fine', seed=int(rng.integers(1 << 30)), load=1.6,
                            press=(.9, .8), wobble=0), self.rgb, self.height, self.thick, self.tooth)
        self.height *= 2.2

    def slot(self):
        k = self.count
        self.count += 1
        if k == 0:
            self.cursor = np.array([70., 130.])
            self.heading = 1.
        step = self.rng.uniform(34, 60)
        self.cursor[0] += self.heading * step
        self.cursor[1] += self.rng.normal(0, 26)
        if not 50 <= self.cursor[0] <= self.w - 50:
            self.heading *= -1
            self.cursor[0] = np.clip(self.cursor[0], 50, self.w - 50)
            self.cursor[1] += self.rng.uniform(-70, 70)
        self.cursor[1] = np.clip(self.cursor[1], 100, self.h - 40)
        cx, cy = self.cursor + self.rng.normal(0, 8, 2)
        a = self.rng.uniform(-.9, .9)
        r = self.rng.uniform(16, 32)
        bend = self.rng.normal(0, 7)
        return [(cx - r * np.cos(a), cy - r * np.sin(a)), (cx + bend * np.sin(a), cy - bend * np.cos(a)), (cx + r * np.cos(a), cy + r * np.sin(a))]

    def smear(self, rgb, pts):
        st = dict(points=pts, R=float(10 + 8 * self.rng.random()), rgb=list(rgb), medium='oil', seed=int(self.rng.integers(1 << 30)),
                  load=1.4, press=(.7, .5))
        return st, render(st, self.rgb, self.height, self.thick, self.tooth, apply=False)

    def image(self):
        img = shade(self.rgb, self.height, self.thick, self.tooth, k=.7)
        return img


# ------------------------------------------------------------------------------------------------ hand
def extend_arm(img, extra):
    """Continue the sleeve below the asset's bottom edge along the forearm's lean, widening slowly toward the upper arm.
    The cloth is the sleeve's own: the rows above the edge, mirrored back and forth, each stretched onto the outline the
    bottom rows continue (a repeated cross-section read as a stocking, and on sheer gauze as plaid)."""
    a = np.asarray(img).astype(float)
    h, w = a.shape[:2]
    alpha = a[..., 3] > 128
    rows = [h - 1 - k for k in (0, 60, 120)]
    cx = [np.flatnonzero(alpha[r]).mean() for r in rows]
    slope = (cx[0] - cx[2]) / 120.
    xs = np.flatnonzero(a[h - 8:h, :, 3].mean(0) > 128)
    centre, half = (xs.min() + xs.max()) / 2, (xs.max() - xs.min()) / 2
    grow = int(.2 * (xs.max() - xs.min())) + 2
    extent = []                                           # the sleeve's span in each row above the edge
    for r in range(h - 1, -1, -1):
        on = np.flatnonzero(a[r, :, 3] > 60)
        if len(on) < .6 * (xs.max() - xs.min()):
            break
        extent.append((on.min(), on.max()))
    n = max(1, min(len(extent), 360))
    out = np.zeros((h + extra, w + int(abs(slope) * extra) + grow + 40, 4))
    out[:h, :w] = a
    X = np.arange(out.shape[1], dtype=float)
    cols = np.arange(w)
    for k in range(extra):
        f = 1 + .2 * k / extra
        c = centre + slope * (k + 1)
        m = k % (2 * n)
        back = m if m < n else 2 * n - 1 - m              # ping-pong: up n rows, then down again
        lo, hi = extent[back] if extent else (xs.min(), xs.max())
        rel = (X - (c - half * f)) / (2 * half * f)
        src = lo + rel * (hi - lo)
        out[h + k] = np.stack([np.interp(src, cols, a[h - 1 - back, :, ch], left=0, right=0) for ch in range(4)], -1)
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


class Hand:
    def __init__(self, path=HAND):
        """path: a transparent hand asset; a sidecar <name>.json may give its own nib (asset pixels) and scale."""
        path = Path(path)
        side = path.with_suffix('.json')
        meta = json.loads(side.read_text()) if side.exists() else {}
        nib, scale = meta.get('nib', NIB), meta.get('scale', HAND_SCALE)
        src = Image.open(path).convert('RGBA')
        w, h = src.size
        src = src.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
        src = extend_arm(src, 900)
        pad = 160
        canvas = Image.new('RGBA', (src.width + 2 * pad, src.height + 2 * pad), (0, 0, 0, 0))
        canvas.paste(src, (pad, pad))
        self.pil = canvas
        img = np.asarray(canvas, float)
        self.nib = np.array(nib) * scale + pad
        rgb, a = img[..., :3], img[..., 3]
        # bristles: light warm-neutral pixels near the tip, before the ferrule; paint loads their outer two thirds
        yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
        dist = np.hypot(xx - self.nib[0], yy - self.nib[1])
        lum = rgb @ [.299, .587, .114]
        sat = rgb.max(-1) - rgb.min(-1)
        cream = (a > 40) & (lum > 150) & (sat < 70) & (rgb[..., 0] >= rgb[..., 2]) & (dist < 62 * scale / .7)
        length = dist[cream].max() if cream.any() else 1.
        self.tint = np.where(cream, np.clip(1.15 - dist / (length * .8), 0, 1) ** .8, 0.)
        from scipy import ndimage
        self.tint = ndimage.gaussian_filter(self.tint, 1.)
        self.box = (max(0, int(self.nib[1] - 80)), int(self.nib[1] + 80), max(0, int(self.nib[0] - 80)), int(self.nib[0] + 80))
        y0, y1, x0, x1 = self.box
        self.lum = lum[y0:y1, x0:x1]
        self.tint = self.tint[y0:y1, x0:x1]
        self.bristle_box = img[y0:y1, x0:x1].copy()
        self.shadow_img = Image.fromarray((ndimage.gaussian_filter(a / 255., 9) * 255).astype(np.uint8), 'L')

    def draw(self, frame, tip, lift, paint_rgb, angle, strength=.85):
        """Composite onto frame (uint8 HxWx3 array) with the nib at tip."""
        y0, y1, x0, x1 = self.box
        pil = self.pil.copy()
        if paint_rgb is not None:
            reg = self.bristle_box.copy()
            t = (self.tint * strength)[..., None]
            shading = np.clip(self.lum / 205., .55, 1.15)[..., None]
            wet = np.asarray(paint_rgb) * 255 * shading
            reg[..., :3] = reg[..., :3] * (1 - t) + wet * t
            pil.paste(Image.fromarray(reg.astype(np.uint8), 'RGBA'), (x0, y0))
        shadow = self.shadow_img
        if abs(angle) > .05:
            pil = pil.rotate(angle, Image.BICUBIC, center=tuple(self.nib))
            shadow = shadow.rotate(angle, Image.BICUBIC, center=tuple(self.nib))
        off = np.array([-7, -11]) * lift
        ox, oy = int(round(tip[0] - self.nib[0] + off[0])), int(round(tip[1] - self.nib[1] + off[1]))
        sx, sy = ox + int(16 + 34 * lift), oy + int(22 + 46 * lift)
        out = Image.fromarray(frame)
        dark = Image.new('RGB', shadow.size, (40, 30, 22))
        sa = shadow.point(lambda v: int(v * (.30 - .1 * lift)))
        out.paste(dark, (sx, sy), sa)
        out.paste(pil, (ox, oy), pil)
        return np.asarray(out)


# ---------------------------------------------------------------------------------------------- surfaces
class OilSurface:
    """Oil: canvas colour, paint relief and paint thickness, lit by a raking light for display."""
    kind = 'oil'

    def __init__(self, ground, tooth):
        self.canvas = np.tile(np.asarray(ground, float), (H, W, 1))
        self.height, self.thick, self.tooth = np.zeros((H, W)), np.zeros((H, W)), tooth

    def render(self, st):
        return render(st, self.canvas, self.height, self.thick, self.tooth, apply=False)

    def commit(self, p, upto=None):
        commit(p, self.canvas, self.height, self.thick, upto=upto)

    def save(self, p):
        gy, gx = p['gy'], p['gx']
        return self.canvas[gy, gx].copy(), self.height[gy, gx].copy(), self.thick[gy, gx].copy()

    def restore(self, p, saved):
        gy, gx = p['gy'], p['gx']
        self.canvas[gy, gx], self.height[gy, gx], self.thick[gy, gx] = saved

    def image(self):
        return shade(self.canvas, self.height, self.thick, self.tooth)


def make_surface(plan, d, tooth):
    kind = plan.get('kind', 'oil')
    if kind == 'sketch':
        ground = np.load(d / 'ground.npy').astype(float)
        media.set_paper(tooth.astype(np.float32), ground)
        for f in d.glob('level-*.npy'):
            media.LEVELS[f.stem[6:]] = np.load(f).astype(np.float32)
        return media.PaperSurface(ground)
    if kind == 'anime':
        for f in d.glob('level-*.npy'):
            media.LEVELS[f.stem[6:]] = np.load(f).astype(np.float32)
        for f in d.glob('alpha-*.npy'):
            media.ALPHAS[f.stem[6:]] = np.load(f).astype(np.float32)
        for f in d.glob('label-*.npy'):
            media.LABELS[f.stem[6:]] = np.load(f)
        return media.LayerSurface(plan.get('ground', (1., 1., 1.)), plan.get('layers', ('rough', 'line')))
    return OilSurface(plan.get('ground', paint.CANVAS_RGB), tooth)


class Swatches:
    """Anime: the drawing app's panel: the current colour large, the recent ones in a row, the layers (the one being
    drawn on marked, the rough greyed once hidden) and the slider of the last adjustment."""
    LAYERS = [('rough', '草稿'), ('line', '线稿'), (None, '上色'), ('hi', '高光')]
    SLIDER_X, SLIDER_Y = (300, 740), 226      # inside the panel; anime.SLIDER is the same span in canvas coordinates

    def __init__(self):
        self.recent = []
        self.cache = {}

    def image(self, rgb, layer='rough', hidden=(), knob=0.):
        key = None if rgb is None else tuple(np.round(np.asarray(rgb) * 255).astype(int))
        if key is not None and (not self.recent or self.recent[-1] != key):
            if key in self.recent:
                self.recent.remove(key)
            self.recent.append(key)
            self.recent = self.recent[-8:]
        ck = (key, tuple(self.recent), layer, tuple(hidden), round(knob, 3))
        if ck not in self.cache:
            x, y, w, h = TRAY
            img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
            dr = ImageDraw.Draw(img)
            if key is not None:
                dr.ellipse([40, 76, 176, 212], fill=key, outline=(205, 198, 188), width=3)
            for i, c in enumerate(reversed(self.recent[:-1] if key else self.recent)):
                cx, cy = 240 + i * 70, 62
                dr.rounded_rectangle([cx - 26, cy - 26, cx + 26, cy + 26], 12, fill=c, outline=(214, 208, 198), width=2)
            f = font(FONT_SANS, 28)
            for i, (name, label) in enumerate(self.LAYERS):
                x0 = 214 + i * 140
                on, off = name == layer, name in hidden
                dr.rounded_rectangle([x0, 112, x0 + 128, 172], 14, fill=(255, 255, 255, 255) if on else (243, 240, 235, 255),
                                     outline=(176, 72, 42, 255) if on else (214, 208, 198, 255), width=4 if on else 2)
                ink = (186, 180, 172, 255) if off else (INK + (255,) if on else (120, 112, 104, 255))
                dr.ellipse([x0 + 14, 134, x0 + 30, 150], outline=ink, width=2, fill=None if off else ink)
                dr.text((x0 + 42, 126), label, font=f, fill=ink)
            (x0, x1), ys = self.SLIDER_X, self.SLIDER_Y
            dr.text((214, ys - 16), '调色', font=f, fill=(120, 112, 104, 255))
            dr.line([x0, ys, x1, ys], fill=(214, 208, 198, 255), width=6)
            kx = x0 + (x1 - x0) * knob
            dr.line([x0, ys, kx, ys], fill=(176, 72, 42, 255), width=6)
            dr.ellipse([kx - 13, ys - 13, kx + 13, ys + 13], fill=(255, 255, 255, 255), outline=(176, 72, 42, 255), width=3)
            self.cache = {ck: img}
        return self.cache[ck]


# ---------------------------------------------------------------------------------------------- layout
def font(path, size, index=0):
    return ImageFont.truetype(path, size, index=index)


def base_frame(reference, watermark=True, kind=None):
    kind = kind or KIND[0]
    img = Image.new('RGB', (FW, FH), tuple(BG.astype(int)))
    yy, xx = np.mgrid[0:FH, 0:FW]
    vign = 1 - .07 * (((xx - FW / 2) / FW) ** 2 + ((yy - FH * .45) / FH) ** 2) * 2.4
    arr = np.asarray(img, float) * vign[..., None]
    img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    shadow = Image.new('L', (FW, FH), 0)
    d = ImageDraw.Draw(shadow)
    d.rectangle([CX + 4, CY + 10, CX + W + 4, CY + H + 14], fill=120)
    x, y, w, h = TRAY
    if kind != 'sketch':
        d.rounded_rectangle([x + 3, y + 8, x + w + 3, y + h + 10], 30, fill=90)
    px, py, pw, ph = PHOTO
    d.rectangle([px - 6, py - 2, px + pw + 14, py + ph + 14], fill=90)
    shadow = shadow.filter(ImageFilter.GaussianBlur(14))
    img.paste(Image.new('RGB', (FW, FH), (60, 48, 38)), (0, 0), shadow.point(lambda v: int(v * .55)))
    d = ImageDraw.Draw(img)
    d.rectangle([CX - 1, CY - 1, CX + W, CY + H], outline=(190, 182, 168))
    # pinned reference photo (polaroid border, a strip of tape)
    photo = Image.new('RGB', (pw + 16, ph + 16), (250, 248, 243))
    photo.paste(Image.fromarray(reference).resize((pw, ph), Image.LANCZOS), (8, 8))
    photo = photo.rotate(-2.2, Image.BICUBIC, expand=True, fillcolor=(0, 0, 0))
    mask = Image.new('L', (pw + 16, ph + 16), 255).rotate(-2.2, Image.BICUBIC, expand=True)
    img.paste(photo, (px - 8, py - 6), mask)
    tape = Image.new('RGBA', (110, 34), (226, 214, 180, 200))
    tape = tape.rotate(4, Image.BICUBIC, expand=True)
    img.paste(tape, (px + pw // 2 - 58, py - 22), tape)
    if kind == 'sketch' and TOOLS.exists():       # the pencils, stump and erasers laid out below the drawing board
        tools = Image.open(TOOLS).convert('RGBA')
        k = min((w + 40) / tools.width, (h + 40) / tools.height)
        tools = tools.resize((round(tools.width * k), round(tools.height * k)), Image.LANCZOS)
        img.paste(tools, (x - 20, y + (h - tools.height) // 2), tools)
    elif kind == 'anime':                          # the drawing app's colour panel (swatches are drawn per frame)
        d.rounded_rectangle([x, y, x + w, y + h], 26, fill=(250, 249, 247), outline=(214, 208, 198))
        d.text((x + 28, y + 20), '颜色', font=font(FONT_SANS, 26), fill=(120, 112, 104))
    d.text((CX, 38), TITLES.get(kind, TITLES['oil']), font=font(FONT_HAND, 52), fill=INK)
    if not watermark:
        return np.asarray(img).copy()
    wm_layer = watermark_layer()
    img.paste(wm_layer, (0, 0), wm_layer)
    return np.asarray(img).copy()


def watermark_layer():
    wm = font(FONT_SANS, 36)
    wm_layer = Image.new('RGBA', (FW, FH), (0, 0, 0, 0))
    ImageDraw.Draw(wm_layer).text((FW - 80, 52), WATERMARK, font=wm, fill=INK + (int(255 * .3),), anchor='ra',
                                  stroke_width=3, stroke_fill=tuple(BG.astype(int)) + (int(255 * .3),))
    return wm_layer


def camera_track(events, nframes, centre, tips=None):
    """Per-frame zoom and centre: push in on the face while focal features are painted (stretches closer than ZOOM_GAP
    merge into one close-up), ease in and out over ZOOM_EASE, and pull back for everything else."""
    spans = []
    for e in events:
        if e.get('slow') and e['kind'] in ('stroke', 'travel'):
            if spans and e['v0'] - spans[-1][1] < ZOOM_GAP:
                spans[-1][1] = e['v1']
            else:
                spans.append([e['v0'], e['v1']])
    t = np.arange(nframes) / FPS
    f = np.zeros(nframes)
    for a, b in spans:
        if b - a < 1.5:
            continue
        up = np.clip((t - (a - ZOOM_EASE)) / ZOOM_EASE, 0, 1)
        down = np.clip(((b + .4 + ZOOM_EASE) - t) / ZOOM_EASE, 0, 1)
        f = np.maximum(f, np.minimum(up, down))
    f = f * f * (3 - 2 * f)
    zoom = 1 + (ZOOM - 1) * f
    cx = FW / 2 + (centre[0] - FW / 2) * f
    cy = FH / 2 + (centre[1] - FH / 2) * f
    if tips is not None:     # drift a little toward where the brush works, smoothed over ~1.5 s
        from scipy import ndimage
        tx = ndimage.gaussian_filter1d(np.clip(tips[:, 0], 0, FW), FPS * .7, mode='nearest')
        ty = ndimage.gaussian_filter1d(np.clip(tips[:, 1], 0, FH), FPS * .7, mode='nearest')
        cx = cx + FOLLOW * f * (tx - centre[0])
        cy = cy + FOLLOW * f * (ty - centre[1])
    return zoom, cx, cy, spans


def apply_camera(frame, z, cx, cy):
    if z <= 1.001:
        return frame
    w, h = FW / z, FH / z
    x0 = min(max(cx - w / 2, 0), FW - w)
    y0 = min(max(cy - h / 2, LABELS_BOTTOM - HEADER / z), FH - h)   # the labels' own rows stay under the fixed header
    return np.asarray(Image.fromarray(frame).resize((FW, FH), Image.BICUBIC, box=(x0, y0, x0 + w, y0 + h)))


def pass_row(labels, current):
    img = Image.new('RGBA', (W + 80, 44), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    f = font(FONT_HAND, 31)
    x = 0
    for k, label in enumerate(labels):
        colour = INK + (255,) if k == current else ((120, 112, 104, 255) if k < current else (178, 170, 160, 255))
        if k == current:
            d.ellipse([x, 17, x + 11, 28], fill=(176, 72, 42, 255))
            x += 17
        d.text((x, 4), label, font=f, fill=colour)
        x += d.textlength(label, font=f) + 12
        if k < len(labels) - 1:
            d.text((x, 4), '·', font=f, fill=(178, 170, 160, 255))
            x += 18
    return img


# ------------------------------------------------------------------------------------------------ build
PACE_SCALE = [1.]
KEY_PASSES = {'sketch', 'face', 'hair', 'clothes', 'background'}   # what a short cut keeps most of (the reference video's beats)


def build(plan, palette):
    passes = plan['passes']
    pace = plan.get('pace', {})                      # v4: shorter feature caps; only the last stretch of eye work at life pace
    k = PACE_SCALE[0]                                # --length: every pass, cap and life-pace stretch shrinks alike
    feature_cap = pace.get('feature_cap', FEATURE_CAP) * k
    eye_life = {n: v * k for n, v in pace.get('eye_life', {}).items()}
    tip = np.array([FW + 200., FH + 300.])
    timeline = []
    tool = 'pencil'
    for pi, ps in enumerate(passes):
        evs = [dict(kind='pause', real=(1.2 if pi else .6) * min(1., k * 1.5) * pace.get('pause', 1.), pi=pi)]
        last_key = None
        focal_trip = False
        for st in ps['strokes']:
            pts, s, *_ = geometry(st)
            if st.get('flood'):               # anime: the hand slides along a slider while the adjustment eases in
                pts, s = np.asarray(st['points'], float)[[0, -1]], np.array([0., st['flood']])
            fp = pts + [CX, CY]
            key = (pi, st['mix'])
            slow = 0.
            if st['mix'] >= 900 or (st.get('focal') and ps['name'] in SLOW_PASSES):
                slow = FOCAL_LAPSE if (st.get('eye') and ps['name'] in LIFE_PASSES) or st['mix'] >= 900 else FEATURE_LAPSE
                if ps['name'] == 'detail':
                    slow = EARLY_LAPSE
                elif st.get('cut'):           # cutting in around a line is cleanup, not the line itself
                    slow = FEATURE_LAPSE
            if key != last_key and KIND[0] == 'oil':
                # one visible palette visit when the focal work of a pass begins; later mixes in it are picked up off-screen
                first = last_key is None or (slow and not focal_trip)
                focal_trip = focal_trip or bool(slow and first)
                evs.append(dict(kind='mix', pi=pi, rgb=st['mix_rgb'], first=first, real=0., slow=slow))
                last_key = key
            if KIND[0] == 'sketch' and TOOL_OF.get(st['medium'], tool) != tool:
                tool = TOOL_OF[st['medium']]
                evs.append(dict(kind='tool', pi=pi, tool=tool, b=np.array(TOOL_SPOTS[tool]), real=0., slow=0.))
            evs.append(dict(kind='travel', pi=pi, b=fp[0].copy(), real=0., slow=slow))
            evs.append(dict(kind='stroke', pi=pi, st=st, path=fp, s=s, L=float(s[-1]), real=stroke_seconds(st, s[-1]) + .08 * bool(slow),
                            rgb=st["rgb"], slow=slow, focal=bool(slow)))
        if ps['name'] in eye_life:      # the eye strokes before the last eye_life seconds join the other features
            acc = 0.
            for i in range(len(evs) - 1, -1, -1):
                e = evs[i]
                if e['kind'] == 'stroke' and e['slow'] == FOCAL_LAPSE and e['st']['mix'] < 900:
                    acc += e['real'] + .25
                    if acc > eye_life[ps['name']]:
                        e['slow'] = FEATURE_LAPSE
                        if i and evs[i - 1]['kind'] == 'travel':
                            evs[i - 1]['slow'] = FEATURE_LAPSE
        for base in (FEATURE_LAPSE, EARLY_LAPSE):
            others = [e for e in evs if e.get('slow') == base]
            need = sum(stroke_seconds(e['st'], e['L']) + .25 for e in others if e['kind'] == 'stroke') / base
            if need > feature_cap:
                for e in others:
                    e['slow'] = base * need / feature_cap
        nib_px = ps.get('nib_px', pace.get('nib_px'))       # anime plans cap the nib's travel per frame (see floor)
        if nib_px:
            for e in evs:
                e['nib'] = (nib_px, ps.get('travel_px', pace.get('travel_px', nib_px)))
        if KIND[0] != 'oil' or ps.get('group'):       # oil: only passes the plan marks (v5 strands, edges...)
            evs = group_events(evs)
        timeline.append(evs)

    def settle(evs, tip, decide):
        # positions and real durations of lifts and palette visits; decide which mixes get a visible visit
        cur = tip.copy()
        lap = decide
        shown = 0
        for k, e in enumerate(evs):
            e['a'] = cur.copy()
            if e['kind'] == 'pause':
                e['pos'] = cur.copy()
            elif e['kind'] == 'mix':
                if 'smear_pts' not in e:
                    e['smear_pts'] = palette.slot()
                sp = np.array(e['smear_pts']) + [TRAY[0], TRAY[1]]
                show = e['first'] or (not e['slow'] and lap is not None and lap[k] <= TRIP_LAPSE and shown < max(1, round(MAX_TRIPS * min(1., PACE_SCALE[0] * 1.5))))
                e['trip'] = bool(show) if decide is not None else e.get('trip', e['first'])
                shown += e['trip']
                if e['trip']:
                    d1 = np.linalg.norm(sp[0] - cur)
                    e['dist'] = d1
                    e['go'] = travel_seconds(d1)
                    e['real'] = e['go'] + SWIRL
                    e['film_pts'] = sp
                    cur = sp[-1].copy()
                else:
                    e['real'] = 0.
            elif e['kind'] == 'tool':
                e['dist'] = np.linalg.norm(e['b'] - cur)
                e['real'] = travel_seconds(e['dist']) + .3
                cur = e['b'].copy()
            elif e['kind'] == 'travel':
                d = np.linalg.norm(e['b'] - cur)
                e['dist'] = d
                e['real'] = travel_seconds(d)
                cur = e['b'].copy()
            elif e['kind'] == 'stroke':
                cur = e['path'][-1].copy()
        return cur

    out = []
    for pi, evs in enumerate(timeline):
        name = passes[pi]['name']
        settle(evs, tip, None)
        focal = lambda: sum(e['real'] / e['slow'] for e in evs if e.get('slow'))   # life-pace time comes on top of the budget
        budget = passes[pi].get('budget', BUDGET.get(name, 5.))       # v4 plans carry each pass's video seconds
        budget *= PACE_SCALE[0] ** (.5 if name in KEY_PASSES else 1.)   # a short cut keeps the big moves, trims the rest
        lap = fit(evs, budget + focal())
        settle(evs, tip, lap)
        lap = fit(evs, budget + focal())
        tip = settle(evs, tip, None)
        for e, l, v in zip(evs, lap, video_time(evs, lap)):
            e['video'] = v
            e['lapse'] = l
        out.extend(evs)
    return out


def bezier(a, b, u, lift=.12):
    d = b - a
    n = np.linalg.norm(d)
    if n < 1e-6:
        return a.copy()
    perp = np.array([d[1], -d[0]]) / n
    if perp[1] > 0:
        perp = -perp
    c = (a + b) / 2 + perp * lift * n
    return (1 - u) ** 2 * a + 2 * (1 - u) * u * c + u ** 2 * b


def along(path, s, target):
    i = int(np.clip(np.searchsorted(s, target), 1, len(s) - 1))
    f = (target - s[i - 1]) / max(1e-9, s[i] - s[i - 1])
    return path[i - 1] + (path[i] - path[i - 1]) * np.clip(f, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('plan_dir')
    ap.add_argument('--out', default=None)
    ap.add_argument('--stills', type=float, default=2.)
    ap.add_argument('--until', type=float, default=None)
    ap.add_argument('--intro', type=float, default=1.2)
    ap.add_argument('--outro', type=float, default=3.5)
    ap.add_argument('--length', type=float, default=None, help='fit the film to about this many seconds (a short cut)')
    ap.add_argument('--hand', default=None, help='transparent hand asset (optional sidecar .json: nib, scale); default: '
                    'the brush hand for oil, the pencil hand for a sketch, the stylus hand for anime')
    args = ap.parse_args()
    d = Path(args.plan_dir)
    plan = json.loads((d / 'plan.json').read_text())
    KIND[0] = plan.get('kind', 'oil')
    tooth = np.load(d / 'tooth.npy').astype(float)
    for f in d.glob('clip-*.png'):                                  # v4: the areas the block-in strokes are held inside
        paint.CLIPS[f.stem[5:]] = np.asarray(Image.open(f), np.float32) / 255
    reference = np.asarray(Image.open(d / 'reference.png').convert('RGB'))
    rng = np.random.default_rng(3)
    palette = Palette(rng)
    default_hand = {'sketch': HERE / 'src' / 'hand-tang-pencil.png', 'anime': HERE / 'src' / 'hand-tang-stylus.png'}
    hand = Hand(args.hand or default_hand.get(KIND[0], HAND))
    hands = {'pencil': hand}
    if KIND[0] == 'sketch' and not args.hand:
        hands.update({k: Hand(p) for k, p in TOOL_HANDS.items() if p.exists()})
    held = 'pencil'
    labels = list(dict.fromkeys(p['label'] for p in plan['passes']))     # passes sharing a name share one label
    label_of = [labels.index(p['label']) for p in plan['passes']] + [len(labels)]
    rows = [pass_row(labels, k) for k in range(len(labels) + 1)]
    if any(p['name'] in ('texture', 'polish') for p in plan['passes']):
        BUDGET.clear()
        BUDGET.update(BUDGET_V2)
    base = base_frame(reference, watermark=False)
    wm_layer = watermark_layer()
    wm_alpha = np.asarray(wm_layer)[..., 3:4] / 255.
    wm_rgb = np.asarray(wm_layer)[..., :3].astype(float)
    if args.length:                  # a short cut: scale the pacing until the film fits
        goal = args.length - (.5 + args.intro + 1.1 + args.outro)
        for _ in range(5):
            got = sum(e['video'] for e in build(plan, Palette(np.random.default_rng(3))))
            PACE_SCALE[0] *= goal / got
        print(f'pace scale {PACE_SCALE[0]:.2f}')
    events = build(plan, palette)
    # intro: the hand comes in from the lower right; outro: it leaves and the painting holds
    first_tip = next(e['b'] for e in events if e['kind'] == 'travel')
    off = np.array([FW + 260., FH + 420.])
    events.insert(0, dict(kind='enter', pi=0, a=off, b=first_tip, video=args.intro, real=args.intro, lapse=1.))
    events = [dict(kind='hold', pi=0, video=.5, pos=off)] + events
    last_tip = next(e['path'][-1] for e in reversed(events) if e['kind'] == 'stroke')
    events.append(dict(kind='leave', pi=len(plan['passes']) - 1, a=last_tip, b=off, video=1.1, lapse=1.))
    events.append(dict(kind='hold', pi=len(plan['passes']), video=args.outro, pos=off))
    t = 0.
    for e in events:
        e['v0'] = t
        t += e['video']
        e['v1'] = t
    total = t if args.until is None else min(t, args.until)
    nframes = int(math.ceil(total * FPS))
    summary = {}
    for e in events:
        if 'st' in e:
            p = plan['passes'][e['pi']]['name']
            summary.setdefault(p, dict(strokes=0, video=0., lapse=[]))
            summary[p]['strokes'] += 1
        if e.get('pi') is not None and e['kind'] not in ('hold', 'enter', 'leave') and e['pi'] < len(plan['passes']):
            p = plan['passes'][e['pi']]['name']
            summary.setdefault(p, dict(strokes=0, video=0., lapse=[]))
            summary[p]['video'] += e['video']
            summary[p]['lapse'].append(e.get('lapse', 1))
    for p, v in summary.items():
        print(f"{p:8s} {v['strokes']:4d} strokes  {v['video']:5.2f}s video  lapse {min(v['lapse']):.1f}-{max(v['lapse']):.1f}  "
              f"trips {sum(1 for e in events if e['kind'] == 'mix' and e.get('trip') and plan['passes'][e['pi']]['name'] == p)}")
    print(f'total {t:.2f}s, {nframes} frames')
    face = plan.get('face') or plan['focus'][0][:4]
    zoom, cam_x, cam_y, spans = camera_track(events, nframes, (CX + face[0] * W, CY + face[1] * H))
    header_mask = np.r_[np.ones(HEADER), np.linspace(1, 0, HEADER_FADE)][:, None] * np.ones((1, FW))
    print('close-ups', [(round(a, 1), round(b, 1)) for a, b in spans if b - a >= 1.5])

    out_path = Path(args.out) if args.out else d / 'film.mp4'
    stills = d / 'stills'
    stills.mkdir(exist_ok=True)
    ff = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{FW}x{FH}', '-r', str(FPS),
                           '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '17', '-pix_fmt', 'yuv420p',
                           '-movflags', '+faststart', str(out_path)], stdin=subprocess.PIPE)
    surf = make_surface(plan, d, tooth)              # oil: a toned canvas (v4: no wash pass); sketch: paper; anime: layers
    swatches = Swatches() if KIND[0] == 'anime' else None
    hide_at = {}                                     # anime: a layer hidden when a pass starts (the rough sketch)
    for e in events:
        for n in plan['passes'][min(e['pi'], len(plan['passes']) - 1)].get('hide', []) if e['kind'] not in ('hold', 'enter', 'leave') else []:
            hide_at.setdefault(n, e['v0'])
    patches = {}
    ei = 0
    paint_rgb = None
    tip_prev = None
    angle = 0.
    started = time.time()
    tray_x, tray_y, tw, th = TRAY

    def ensure(e):
        if id(e) not in patches:
            if e['kind'] == 'stroke':
                patches[id(e)] = surf.render(e['st'])
            else:
                patches[id(e)] = palette.smear(e['rgb'], e['smear_pts'])[1]
        return patches[id(e)]

    def advance(e, s_now):
        """A grouped motion: lay every stroke the hand has passed; return the one under it and how far along."""
        g, j = e['group'], e.get('done', 0)
        while j < len(g) and g[j][2] <= s_now:
            p = patches.pop((id(e), j), None) or surf.render(g[j][0])
            if p is not None:
                surf.commit(p)
            j += 1
        e['done'] = j
        if j < len(g) and g[j][1] <= s_now:
            if (id(e), j) not in patches:
                patches[(id(e), j)] = surf.render(g[j][0])
            return patches[(id(e), j)], s_now - g[j][1]
        return None, 0.

    def finish(e):
        if e['kind'] == 'stroke' and 'group' in e:
            advance(e, float('inf'))
        elif e['kind'] == 'stroke':
            p = ensure(e)
            if p is not None:
                surf.commit(p)
        elif e['kind'] == 'mix':
            p = ensure(e)
            if p is not None:
                commit(p, palette.rgb, palette.height, palette.thick)
        patches.pop(id(e), None)

    last_stroke_rgb = None
    last_colour, layer_now, knob = [None], ['rough'], [0.]
    tips = []
    for f in range(nframes):
        tf = f / FPS
        while ei < len(events) and events[ei]['v1'] <= tf:
            e = events[ei]
            finish(e)
            if e['kind'] == 'stroke':
                last_stroke_rgb = e['rgb']
            if e['kind'] == 'mix':
                paint_rgb = e['rgb']
            if e['kind'] == 'tool':
                held = e['tool']
            ei += 1
        e = events[ei] if ei < len(events) else events[-1]
        tau = 0. if e['video'] <= 0 else np.clip((tf - e['v0']) / e['video'], 0, 1)
        lift, partial, ppartial = 1., None, None
        k = e['kind']
        if k == 'stroke':
            s_now = e['L'] * (even_fraction(tau) if e.get('nib') else distance_fraction(tau, 'minjerk'))
            tip = along(e['path'], e['s'], s_now)
            lift = 0.
            partial = advance(e, s_now) if 'group' in e else (ensure(e), s_now)
            paint_rgb = e['rgb']
        elif k in ('travel', 'enter', 'leave', 'tool'):
            u = even_fraction(tau) if e.get('nib') else distance_fraction(tau, 'minjerk')
            tip = bezier(np.asarray(e['a'], float), np.asarray(e['b'], float), u)
            lift = math.sin(math.pi * min(1., tau * 1.15)) ** .6 if k == 'travel' else 1.
        elif k == 'mix':
            if e.get('trip'):
                go = e['go'] / e['real']
                if tau < go:
                    u = distance_fraction(tau / go, 'minjerk')
                    tip = bezier(np.asarray(e['a'], float), e['film_pts'][0], u)
                    lift = math.sin(math.pi * min(1., tau / go * 1.15)) ** .6
                else:
                    w = (tau - go) / (1 - go)
                    p = ensure(e)
                    pts = np.array(e['smear_pts'], float)
                    from paint import catmull
                    dense = catmull(pts) + [tray_x, tray_y]
                    ss = np.r_[0, np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
                    # scrub: along the smear, back, and along again
                    phase = (w * 3) % 2
                    pos = phase if phase <= 1 else 2 - phase
                    tip = along(dense, ss, ss[-1] * pos)
                    lift = 0.
                    if p is not None:
                        ppartial = (p, p['L'] * min(1., w * 1.6))
                    if w > .35:
                        paint_rgb = e['rgb']
            else:
                tip = np.asarray(e['a'], float)
        elif k == 'pause':
            away = np.array([150., 230.]) if e['pi'] else np.array([10., -16.])
            tip = np.asarray(e['pos'], float) + away * math.sin(math.pi * tau) ** 1.5
        else:
            tip = np.asarray(e.get('pos', [FW + 260, FH + 420]), float)
        # canvas (with the stroke in progress laid only as far as the brush has gone)
        saved = None
        if partial and partial[0] is not None:
            p, s_now = partial
            saved = surf.save(p)
            surf.commit(p, upto=s_now)
        for n, t0 in hide_at.items():
            surf.shown[n] = float(np.clip(1 - (tf - t0) / .6, 0, 1))
        frame = base.copy()
        frame[CY:CY + H, CX:CX + W] = (np.clip(surf.image(), 0, 1) * 255).astype(np.uint8)
        if saved is not None:
            surf.restore(p, saved)
        # palette (oil), the colour panel (anime); a sketch's tools are part of the base frame
        if KIND[0] == 'oil':
            psaved = None
            if ppartial:
                p, s_now = ppartial
                pgy, pgx = p['gy'], p['gx']
                psaved = (palette.rgb[pgy, pgx].copy(), palette.height[pgy, pgx].copy(), palette.thick[pgy, pgx].copy())
                commit(p, palette.rgb, palette.height, palette.thick, upto=s_now)
            tray = palette.image() * 255
            if psaved is not None:
                palette.rgb[pgy, pgx], palette.height[pgy, pgx], palette.thick[pgy, pgx] = psaved
            region = frame[tray_y:tray_y + th, tray_x:tray_x + tw].astype(float)
            m = palette.mask[..., None]
            frame[tray_y:tray_y + th, tray_x:tray_x + tw] = (region * (1 - m) + tray * m).astype(np.uint8)
        pil = Image.fromarray(frame)
        if swatches is not None:
            st_now = e['st'] if e['kind'] == 'stroke' else {}
            cur = st_now['rgb'] if st_now.get('medium') in ('cel', 'air') else last_colour[0]
            if st_now.get('medium') in ('cel', 'air'):
                last_colour[0] = cur
            if st_now and st_now['medium'] != 'adjust':
                layer_now[0] = st_now.get('layer')
            if st_now.get('medium') == 'adjust':
                knob[0] = float(distance_fraction(tau, 'minjerk'))
            sw = swatches.image(cur, layer_now[0], [n for n in hide_at if tf >= hide_at[n]], knob[0])
            pil.paste(sw, (tray_x, tray_y), sw)
        # pass label
        row = rows[label_of[min(e['pi'], len(label_of) - 1)]]
        pil.paste(row, (CX, 98), row)
        frame = np.asarray(pil).copy()
        # hand: a right forearm leans more on the left of the canvas; a little wrist roll with motion
        target = 2 - 10 * ((tip[0] - CX) / W - .5) + 6 * ((tip[1] - CY) / H - .5)
        if tip_prev is not None:
            target += np.clip((tip[0] - tip_prev[0]) * .04, -3, 3)
        angle += (target - angle) * .35
        tip_prev = tip
        strength = .6 if (paint_rgb is not None and e.get('st', {}).get('medium') == 'pencil') else .85
        frame = hands.get(held, hand).draw(frame, tip, lift, paint_rgb if KIND[0] == 'oil' else None, angle, strength)
        tips.append([float(tip[0]), float(tip[1]), float(lift), e['kind'], int(e['pi']), float(zoom[f])])
        header = frame[:HEADER + HEADER_FADE].astype(float) if zoom[f] > 1.001 else None
        frame = apply_camera(frame, zoom[f], cam_x[f], cam_y[f])
        if header is not None:     # in a close-up the title and pass labels stay put at their normal size, on paper
            k = ((zoom[f] - 1) / (ZOOM - 1) * header_mask)[..., None]
            top = frame[:HEADER + HEADER_FADE].astype(float)
            frame = frame.copy()
            frame[:HEADER + HEADER_FADE] = (top * (1 - k) + header * k).astype(np.uint8)
        frame = (frame * (1 - wm_alpha) + wm_rgb * wm_alpha).astype(np.uint8)
        ff.stdin.write(frame.tobytes())
        if args.stills and f % int(args.stills * FPS) == 0:
            Image.fromarray(frame).save(stills / f'{f:05d}.jpg', quality=88)
        if f % 150 == 0:
            print(f'frame {f}/{nframes}  {time.time() - started:.0f}s', flush=True)
    ff.stdin.close()
    ff.wait()
    Image.fromarray(frame).save(d / 'last-frame.png')
    (d / 'tips.json').write_text(json.dumps(tips))
    print('wrote', out_path, f'{time.time() - started:.0f}s')


if __name__ == '__main__':
    main()
