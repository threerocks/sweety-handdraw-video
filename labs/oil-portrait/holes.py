"""Missed-spot detector for a planned run: light pinholes and slits that read as "forgot to fill this".

A pixel is a suspect when the sheet is much lighter there than its own neighbourhood (the 15 px median) while the
target is not (so a drawn highlight does not count). Suspects are grouped; a group is a missed spot when it is small
(under 60 px) or thin (under 3 px wide). Counted per area (hair, clothes, the skin's shadows) on every pass image
from the first lay-in on.

usage: holes.py RUN_DIR AREAS.png FACE_JSON [--crop-top 0] [--thr .1]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

import paint
import planes
from paint import AREAS, REGIONS, load_subject

LUM = np.array([.299, .587, .114])


def drawn_light(vt):
    """Where the target itself is lighter than its neighbourhood: a drawn highlight, not a missed spot."""
    q = (np.clip(vt, 0, 1) * 255).astype(np.uint8)
    return ndimage.gaussian_filter(vt, 1.) - ndimage.median_filter(q, 15) / 255. > .04


def missed(v, light, region, thr=.1):
    """Mask of missed spots in region, their count and pixel count (light: drawn_light of the target)."""
    q = (np.clip(v, 0, 1) * 255).astype(np.uint8)
    return spots((v - ndimage.median_filter(q, 15) / 255. > thr) & ~light & region)


def spots(sus):
    comp, n = ndimage.label(sus, np.ones((3, 3)))
    if not n:
        return np.zeros_like(sus), 0, 0
    idx = np.arange(1, n + 1)
    area = ndimage.sum_labels(np.ones(sus.shape), comp, idx)
    width = 2 * ndimage.maximum(ndimage.distance_transform_edt(sus), comp, idx)
    bad = (area < 60) | (width < 3)
    mask = np.isin(comp, idx[bad])
    return mask, int(bad.sum()), int(mask.sum())


def regions(areas_path, target, face):
    REGIONS['face'] = tuple(face)
    planes.area_masks_from_map(areas_path, target, np.random.default_rng(paint.SEED))
    er = lambda m: ndimage.binary_erosion(m, iterations=2)
    vt = target @ LUM
    shadow = er(AREAS['skin']) & (ndimage.gaussian_filter(vt, 3.) < np.percentile(vt[AREAS['skin']], 45))
    return dict(hair=er(AREAS['hair']), clothes=er(AREAS['clothes']), skin_shadow=shadow)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('areas')
    ap.add_argument('face')
    ap.add_argument('--crop-top', type=float, default=0.)
    ap.add_argument('--thr', type=float, default=.1)
    a = ap.parse_args()
    paint.CROP_TOP = a.crop_top
    d = Path(a.run)
    target = np.asarray(Image.open(d / 'target.png').convert('RGB'), float) / 255
    reg = regions(a.areas, target, json.loads(a.face))
    light = drawn_light(target @ LUM)
    out = {}
    print(f"{'pass':22s}" + ''.join(f'{k:>20s}' for k in reg) + '   (spots / px)')
    for p in sorted(d.glob('pass-*.png')) + [d / 'final.png']:
        v = np.asarray(Image.open(p).convert('RGB'), float) / 255 @ LUM
        q = (np.clip(v, 0, 1) * 255).astype(np.uint8)
        sus_all = (v - ndimage.median_filter(q, 15) / 255. > a.thr) & ~light
        row = {k: spots(sus_all & m)[1:] for k, m in reg.items()}
        out[p.stem] = row
        print(f'{p.stem:22s}' + ''.join(f'{c:>12d} /{px:>6d}' for c, px in row.values()))
    (d / 'holes.json').write_text(json.dumps(out))
