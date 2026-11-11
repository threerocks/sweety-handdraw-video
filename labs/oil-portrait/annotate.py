"""Automatic annotations for a portrait: the focus ellipses, eyes, face, figure and signature box the engines need.

Before this, a person (or the model driving the tools) looked at the picture and typed these coordinates in. That is
the step a weaker model gets wrong most easily, so it is computed here from face landmarks (landmarks.py, MediaPipe)
and the area map, and checked for sanity. All results are canvas fractions (the canvas shows the top part of a 2:3
picture, see paint.load_subject), except the signature box, which is in canvas pixels.

    python annotate.py PICTURE AREAS.png [--crop-top 0] [--anime] > annotations.json
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

HERE = Path(__file__).resolve().parent
FACE_PY = Path.home() / '.local/share/sweety-handdraw-video/venv-face/bin/python'
W, H = 1120, 1422                     # the engines' canvas
AREA_COLOURS = {'skin': (240, 160, 120), 'hair': (90, 50, 20), 'clothes': (60, 90, 210), 'bg': (210, 210, 210)}
SIGNATURE = 'Leo 2026'
SIG_BOX = (220, 66)                   # signature box size, canvas px
SIG_CORNERS = ((850, 46), (50, 46), (870, 1318), (50, 1318))     # top right first: the user's films sign there


class AnnotationError(RuntimeError):
    pass


def landmarks(picture):
    if not FACE_PY.exists():
        raise AnnotationError(f'face landmark environment missing: run  python make.py setup  (expects {FACE_PY})')
    r = subprocess.run([str(FACE_PY), str(HERE / 'landmarks.py'), str(picture)], capture_output=True, text=True)
    if r.returncode:
        raise AnnotationError(f'landmarks.py failed on {picture}:\n{r.stderr[-2000:]}')
    lm = json.loads(r.stdout)
    if not lm.get('ok'):
        raise AnnotationError(f'no face found in {picture}')
    return lm


def window(size, crop_top):
    w0, h0 = size
    crop_h = round(w0 * H / W)
    top = int((h0 - crop_h) * crop_top)
    return w0, top, crop_h


def area_masks(path, size):
    a = np.asarray(Image.open(path).convert('RGB').resize(size, Image.NEAREST), float)
    names = list(AREA_COLOURS)
    d = np.stack([((a - np.array(AREA_COLOURS[n])) ** 2).sum(-1) for n in names])
    lab = d.argmin(0)
    return {n: lab == i for i, n in enumerate(names)}


def ellipse_of(points, pad=1.):
    p = np.asarray(points, float)
    lo, hi = p.min(0), p.max(0)
    return (lo + hi) / 2, (hi - lo) / 2 * pad


def annotate(picture, areas=None, crop_top=0., anime=False, lm=None, target=None):
    """anime: the picture is an anime illustration (features are stylised, eyes larger); its own landmarks are used.
    target: the painting the engine aims at (registered to picture); whether teeth show is read from it."""
    lm = lm or landmarks(picture)
    w0, top, crop_h = window(lm['size'], crop_top)

    def fx(p):                              # pixel point -> canvas fraction
        return [round(float(p[0]) / w0, 4), round((float(p[1]) - top) / crop_h, 4)]

    def fr(r):                              # pixel half extents -> fractions of canvas width and height
        return [round(float(r[0]) / w0, 4), round(float(r[1]) / crop_h, 4)]

    notes = []
    m = area_masks(areas, lm['size']) if areas is not None else None
    # face: the landmark oval reaches up to the hairline and, in a turned head, round the hidden far cheek; the face
    # that is painted is the visible skin inside it
    c, r = ellipse_of(lm['oval'])
    if m is not None:
        poly = Image.new('L', tuple(lm['size']), 0)
        ImageDraw.Draw(poly).polygon([tuple(p) for p in lm['oval']], fill=1)
        vis = np.asarray(poly, bool) & m['skin']
        vis = ndimage.binary_opening(vis, iterations=3)
        if vis.sum() > .25 * np.asarray(poly, bool).sum():
            ys, xs = np.nonzero(vis)                 # moments, not bounds: a few stray pixels must not move it
            c = np.array([xs.mean(), ys.mean()])
            r = np.array([2.05 * xs.std(), 2.05 * ys.std()])
        else:
            notes.append('area map skin hardly overlaps the face: face from landmarks only')
    face = fx(c + [0, .05 * r[1]]) + fr(r * [1., 1.])
    # eyes: contour bounds (the iris centre holds the pupil); anime eyes are drawn larger than the mesh contour
    eyes, eye_focus = [], []
    for e in lm['eyes']:
        ec, er = ellipse_of(e['contour'])
        er = np.maximum(er, [e['iris_r'] * 1.6, e['iris_r'] * .9])
        centre = np.asarray(e['iris']) if anime else (ec + np.asarray(e['iris'])) / 2
        eyes.append(fx(centre) + fr(er * [1.0, 1.15]))
        eye_focus.append(fx(centre) + fr(er * [1.35, 1.6]) + [1.0])
    # mouth: teeth showing -> no outline pass round the lips (outlined teeth look like a cartoon), weight .95
    mc, mr = ellipse_of(lm['lips_outer'])
    teeth = teeth_share(target or picture, lm['lips_inner'])
    mouth = fx(mc) + fr(mr * [1.12, 1.2]) + [.95 if teeth > .05 else 1.0]
    if teeth > .05:
        notes.append(f'teeth showing ({teeth:.0%} of the mouth opening): no outline pass on the lips')
    # nose: from the bridge between the eyes to the base, as wide as the wings
    n = lm['nose']
    top_y = (np.asarray(n['bridge'])[1] + np.asarray(n['tip'])[1]) / 2
    nc = np.array([(n['wing_r'][0] + n['wing_l'][0]) / 2, (top_y + n['base'][1]) / 2])
    nr = np.array([abs(n['wing_l'][0] - n['wing_r'][0]) / 2 * .75, (n['base'][1] - top_y) / 2 + 6])
    nose = fx(nc) + fr(nr) + [.95]
    focus = [face[:2] + [round(face[2] * 1.05, 4), face[3]] + [.85]]
    # figure: the whole person in the area map (or the face grown down to the bottom when there is none)
    if m is not None:
        person = ~m['bg']
        ys, xs = np.nonzero(person[top:top + crop_h])
        if len(xs) < 1000:
            raise AnnotationError('area map has almost no person in it')
        figure = [round((xs.min() + xs.max()) / 2 / w0, 4), round((ys.min() + ys.max()) / 2 / crop_h, 4),
                  round((xs.max() - xs.min()) / 2 / w0 * .95, 4), round((ys.max() - ys.min()) / 2 / crop_h * .95, 4)]
    else:
        figure = [face[0], (face[1] + 1) / 2, min(.5, face[2] * 2.4), (1 - face[1]) / 2 + face[3]]
    focus.append(figure + [.5])
    focus += eye_focus + [mouth, nose]
    # glasses: accessory colour in a ring round an eye -> the lens is a crisp feature of its own
    if m is not None:
        for e, ef in zip(lm['eyes'], eye_focus):
            ec, er = ellipse_of(e['contour'])
            yy, xx = np.mgrid[0:lm['size'][1], 0:lm['size'][0]]
            d = ((xx - ec[0]) / (er[0] * 2.2)) ** 2 + ((yy - ec[1]) / (er[0] * 1.8)) ** 2
            ring = (d < 1) & (d > .35)
            if ring.sum() and m['clothes'][ring].mean() > .08:
                focus.append(fx(ec) + fr([er[0] * 1.9, er[0] * 1.5]) + [1.0])
                notes.append('glasses: a lens ellipse added round an eye')
    sig = signature_box(m, lm, top, crop_h, w0)
    out = dict(focus=focus, eyes=eyes, face=face, figure=figure, signature=[SIGNATURE, sig], crop_top=crop_top,
               anime=anime, notes=notes)
    problems = sanity(out)
    if problems:
        raise AnnotationError('annotations failed their checks: ' + '; '.join(problems))
    return out


def teeth_share(picture, inner):
    """Share of the mouth opening that is light and unsaturated (teeth)."""
    img = Image.open(picture).convert('RGB')
    poly = Image.new('L', img.size, 0)
    ImageDraw.Draw(poly).polygon([tuple(p) for p in inner], fill=1)
    inside = np.asarray(poly, bool)
    if inside.sum() < 20:
        return 0.
    a = np.asarray(img, float)[inside] / 255
    v, s = a.max(1), (a.max(1) - a.min(1)) / (a.max(1) + 1e-6)
    return float(((v > .55) & (s < .35)).mean())


def signature_box(m, lm, top, crop_h, w0):
    """The first corner (top right preferred) whose box sits on background, clear of the face."""
    sx = W / w0
    oval = np.asarray(lm['oval'])
    for x, y in SIG_CORNERS:
        bx0, by0 = x / sx, y / sx + top
        bx1, by1 = (x + SIG_BOX[0]) / sx, (y + SIG_BOX[1]) / sx + top
        if not (oval[:, 0].max() < bx0 or oval[:, 0].min() > bx1 or oval[:, 1].max() < by0 or oval[:, 1].min() > by1):
            continue
        if m is None:
            return [x, y, *SIG_BOX]
        box = m['bg'][int(by0):int(by1), int(bx0):int(bx1)]
        if box.size and box.mean() > .9:
            return [x, y, *SIG_BOX]
    return [SIG_CORNERS[0][0], SIG_CORNERS[0][1], *SIG_BOX]


def sanity(a):
    """Things that are always true of a portrait; a failure means the landmarks or the area map went wrong."""
    p = []
    face, (l, r) = a['face'], a['eyes']
    if not (0 < face[0] < 1 and 0 < face[1] < 1):
        p.append(f'face centre {face[:2]} off the canvas')
    if not (.06 < face[2] < .45 and .05 < face[3] < .45):
        p.append(f'face size {face[2:]} implausible')
    if l[0] >= r[0]:
        p.append('eyes not left/right')
    if abs(l[1] - r[1]) > .06:
        p.append('eyes at very different heights')
    d = r[0] - l[0]
    if not (.25 * face[2] * 2 < d < .75 * face[2] * 2):
        p.append(f'eye distance {d:.3f} implausible for a face {face[2] * 2:.3f} wide')
    mouth, nose = a['focus'][4], a['focus'][5]
    if not (nose[1] > (l[1] + r[1]) / 2 and mouth[1] > nose[1]):
        p.append('features out of order (eyes, nose, mouth from the top)')
    for e in a['eyes'] + [mouth[:4], nose[:4]]:
        if (e[0] - face[0]) ** 2 / (face[2] * 1.2) ** 2 + (e[1] - face[1]) ** 2 / (face[3] * 1.2) ** 2 > 1:
            p.append(f'feature {e[:2]} outside the face')
    return p


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('picture')
    ap.add_argument('areas', nargs='?')
    ap.add_argument('--crop-top', type=float, default=0.)
    ap.add_argument('--anime', action='store_true')
    ap.add_argument('--target', help='the painting the engine aims at (teeth are judged on it)')
    a = ap.parse_args()
    try:
        print(json.dumps(annotate(a.picture, a.areas, a.crop_top, a.anime, target=a.target)))
    except AnnotationError as e:
        print('ANNOTATION FAILED:', e, file=sys.stderr)
        sys.exit(2)
