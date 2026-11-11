"""Mean colour error (ΔE76) of a painting against its target, by region, plus chroma and value-range ratios
(painting/target) — the 'greyer, flatter than the target' check.

    metric.py TARGET PAINTING...                 regions: the v1 subject's face / hair+clothes / background ellipses
    metric.py --areas RUN/areas.png TARGET PAINTING...   regions: the run's own area map (skin, hair, clothes, background)

TARGET must be the canvas-sized target a plan run writes (RUN/target.png), not the taller picture it was cut from.
"""
import sys
import numpy as np
from PIL import Image
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from paint import rgb2lab, focus_map, W, H
from scipy import ndimage

AREA_VIS = {'skin': (.95, .7, .55), 'hair': (.35, .2, .1), 'clothes': (.3, .4, .8), 'background': (.85, .85, .85)}


def load(p):
    return np.asarray(Image.open(p).convert('RGB').resize((W, H)), float) / 255


def regions_of(areas_path=None):
    if areas_path:
        vis = load(areas_path)
        names = list(AREA_VIS)
        lbl = np.argmin([((vis - np.array(AREA_VIS[n])) ** 2).sum(-1) for n in names], 0)
        return {n: lbl == k for k, n in enumerate(names)}
    face = focus_map([[0.41, 0.37, 0.16, 0.17, 1.]], 0) > .9
    figure = focus_map([[0.42, 0.55, 0.40, 0.50, 1.]], 0) > .9
    return {'face': face, 'hair+clothes': figure & ~face, 'background': ~figure}


def report(paint_path, target_path, areas_path=None):
    a, b = rgb2lab(ndimage.gaussian_filter(load(paint_path), (2, 2, 0))), rgb2lab(ndimage.gaussian_filter(load(target_path), (2, 2, 0)))
    out = []
    for name, m in regions_of(areas_path).items():
        if not m.any():
            continue
        de = np.linalg.norm(a - b, axis=-1)[m].mean()
        chroma = np.hypot(a[..., 1], a[..., 2])[m].mean() / np.hypot(b[..., 1], b[..., 2])[m].mean()
        vr = np.subtract(*np.percentile(a[..., 0][m], [95, 5])) / np.subtract(*np.percentile(b[..., 0][m], [95, 5]))
        out.append(f'{name} dE {de:5.1f} chroma {chroma:.2f} value-range {vr:.2f}')
    return ' | '.join(out)


if __name__ == '__main__':
    args = sys.argv[1:]
    areas = None
    if args[0] == '--areas':
        areas, args = args[1], args[2:]
    for p in args[1:]:
        print(f'{p:28s}', report(p, args[0], areas))
