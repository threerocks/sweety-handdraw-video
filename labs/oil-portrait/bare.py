"""Unpainted paper left in a drawing: specks and slits where the paper still shows after an area was coloured.

usage: bare.py RUN_DIR

For every pass image from the first colouring pass on (pass-NN-*.png, before-adjust.png, final.png) prints
- white: pixels still near paper white (every channel above .93; paper is 1.0) where the target is clearly not light
  (its darkest channel under .86);
- light: pixels where paper shows through a thin or missed cover: lighter (by more than .08 in some channel, darker
  in none) than
  the flat colour and the cel shade laid there or right beside (flats.png, shades.png, level-C0.npy, if the plan has them)
  and the finished picture. Nothing a painter does on purpose is lighter than all of those;
- slits: connected marks of either kind at least 3 px long.
The gate for a finished flat pass: all three about zero.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

PAPER, LIGHT, THROUGH = .93, .86, .08


def bare_masks(img, target, *laid):
    white = (img.min(-1) > PAPER) & (target.min(-1) < LIGHT)
    ref = target
    for x in laid:
        if x is not None:
            ref = np.maximum(ref, x)
    ref = ndimage.maximum_filter(ref, size=(5, 5, 1))     # a blended border is never lighter than its lighter side
    over = img - ref
    light = (over.max(-1) > THROUGH) & (over.min(-1) > -.01) & ~white
    return white, light


def count(img, target, *laid):
    white, light = bare_masks(img, target, *laid)
    comp, n = ndimage.label(white | light, np.ones((3, 3)))
    slits = sum(max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) >= 3 for sl in ndimage.find_objects(comp))
    return int(white.sum()), int(light.sum()), int(slits), white | light


if __name__ == '__main__':
    d = Path(sys.argv[1])
    load = lambda p: np.asarray(Image.open(p).convert('RGB'), float) / 255
    target = load(d / 'target.png')
    laid = [load(d / n) for n in ('flats.png', 'shades.png') if (d / n).exists()]
    if (d / 'level-C0.npy').exists():          # anime: the colours under the line art, which the airbrush works toward
        laid.append(np.load(d / 'level-C0.npy').astype(float))
    names = sorted(p.name for p in d.glob('pass-*.png'))
    start = next(i for i, n in enumerate(names) if 'flat' in n)
    print(f'{"image":24s}   white   light  slits>=3px')
    for n in names[start:] + ['before-adjust.png', 'final.png']:
        if (d / n).exists():
            w, l, s, _ = count(load(d / n), target, *laid)
            print(f'{n:24s} {w:7d} {l:7d} {s:7d}')
