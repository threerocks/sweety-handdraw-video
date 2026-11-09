"""Mean colour error (ΔE76) of a painting against its target, by region (face / hair+clothes / background), plus
chroma and value-range ratios (painting/target) — the 'greyer, flatter than the target' check."""
import sys
import numpy as np
from PIL import Image
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from paint import rgb2lab, focus_map, W, H
from scipy import ndimage

def load(p):
    return np.asarray(Image.open(p).convert('RGB').resize((W, H)), float) / 255

def report(paint_path, target_path):
    a, b = rgb2lab(ndimage.gaussian_filter(load(paint_path), (2, 2, 0))), rgb2lab(ndimage.gaussian_filter(load(target_path), (2, 2, 0)))
    face = focus_map([[0.41, 0.37, 0.16, 0.17, 1.]], 0) > .9
    figure = focus_map([[0.42, 0.55, 0.40, 0.50, 1.]], 0) > .9
    regions = {'face': face, 'hair+clothes': figure & ~face, 'background': ~figure}
    out = []
    for name, m in regions.items():
        de = np.linalg.norm(a - b, axis=-1)[m].mean()
        chroma = np.hypot(a[..., 1], a[..., 2])[m].mean() / np.hypot(b[..., 1], b[..., 2])[m].mean()
        vr = np.subtract(*np.percentile(a[..., 0][m], [95, 5])) / np.subtract(*np.percentile(b[..., 0][m], [95, 5]))
        out.append(f'{name} dE {de:5.1f} chroma {chroma:.2f} value-range {vr:.2f}')
    return ' | '.join(out)

if __name__ == '__main__':
    for p in sys.argv[2:]:
        print(f'{p:28s}', report(p, sys.argv[1]))
