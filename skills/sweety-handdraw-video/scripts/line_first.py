#!/usr/bin/env python3
"""Line-first pictures: register a coloured plate on the line plate it was painted from.

    line_first.py JOB inputs/art/i04-line.png inputs/art/i04-colour.png --out inputs/art/i04.png

The picture is made in two image calls: a pure black line plate first (checked and simplified while it is still cheap
to change), then a colour edit that takes the plate as its input. The edit keeps the lines (v3 experiment 2: 99.5-100%
of them within 3 px, 0-1 px shift), so the plate can be the drawing's authority instead of lines guessed from paint:
- the colour plate is moved onto the line plate (best overlap of their lines);
- the plate's lines are laid over the paint (multiply, as ink over colour), so the source shows exactly the lines the
  pen will draw;
- the plate's ink becomes the action's outline_mask (white = line), same size as the source.
Fails, with a prompt_fix for the colour edit, when the edit lost or moved lines or drew new ones away from the plate.
Writes OUT, the mask, OUT.line-first.json (measures and hashes) and OUT.line-first.jpg (plate | colour | overlap |
source) for the art-direction review; the source is recorded as composed_with_verified_layers.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from stroke_paths import thin_ink

PLATE_INK = 128      # luminance: a line plate's ink
LINE = 110           # luminance: a line in the coloured plate (dark and nearly neutral)
NEUTRAL = 70         # RGB spread below which a dark pixel is ink, not a dark colour
NEAR = 3             # px: a plate line counts as kept when the colour plate has a line this close
FAR = 6              # px: colour-plate line work farther than this from every plate line is new
MIN_KEPT = .97       # share of the plate's lines the colour edit must keep
MAX_ADDED = .05      # new thin line work, as a share of the plate's line length
MAX_SHIFT = .02      # of the long side: a larger offset means the edit redrew the composition
CONTRAST = 20        # luminance: a drawn line is this much darker than the paint around it (grain in a dark fill is not)
COLOUR_FIX = ('Colour this exact line drawing (image 1). Keep every black line exactly where it is, with the same thickness: '
              'do not add, remove, move or redraw any line, and do not change the composition or proportions.')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def luminance(rgb):
    return rgb[..., :3].astype(float) @ [.299, .587, .114]


def best_shift(fixed, moving, reach):
    """Integer (dy, dx) moving `moving` onto `fixed` with the most line overlap: coarse on widened lines (a 2 px step
    cannot miss them), then exact on the lines themselves. A colour edit barely moves anything, so among shifts that
    overlap within 1% as well, the smallest wins: a row of identical windows otherwise matches one window over."""
    score = lambda target, dy, dx: int(np.logical_and(target, np.roll(moving, (dy, dx), (0, 1))).sum())
    wide = ndimage.binary_dilation(fixed, iterations=2)
    step = 2 if reach > 4 else 1
    nearest = lambda scored: min((item for item in scored if item[0] >= .99 * max(value for value, _ in scored)), key=lambda item: abs(item[1][0]) + abs(item[1][1]))[1]
    best = nearest([(score(wide, dy, dx), (dy, dx)) for dy in range(-reach, reach + 1, step) for dx in range(-reach, reach + 1, step)])
    return nearest([(score(fixed, dy, dx), (dy, dx)) for dy in range(best[0] - 2, best[0] + 3) for dx in range(best[1] - 2, best[1] + 3)])


def move(image, dy, dx):
    """Shift an image by whole pixels, repeating its edge into the uncovered band (paper stays paper)."""
    pad = ((max(dy, 0), max(-dy, 0)), (max(dx, 0), max(-dx, 0))) + ((0, 0),) * (image.ndim - 2)
    padded = np.pad(image, pad, mode='edge')
    top, left = max(-dy, 0), max(-dx, 0)
    return padded[top:top + image.shape[0], left:left + image.shape[1]]


def register(plate_rgb, colour_rgba):
    """(moved colour RGBA, measures) of a colour plate on its line plate (both the same size)."""
    plate = (luminance(plate_rgb) < PLATE_INK)
    rgb, alpha = colour_rgba[..., :3], colour_rgba[..., 3]
    dark = (luminance(rgb) < LINE) & (np.ptp(rgb.astype(int), axis=2) < NEUTRAL) & (alpha > 128)
    reach = max(4, round(MAX_SHIFT * max(plate.shape) * 1.5))
    # Register on edges only: the inside of a dark fill (a graphite window body, black hair) would pull every plate line
    # into itself and win at any shift that lands the lines inside it.
    # A colour edit may invert a line inside a dark fill (light code lines on a graphite terminal): that line is a line.
    inverse = inverse_lines(colour_rgba)
    edge = dark & (ndimage.distance_transform_edt(dark) <= 4) & ~ndimage.binary_dilation(inverse, iterations=4)
    dy, dx = best_shift(plate, edge | inverse, reach)
    moved = move(colour_rgba, dy, dx)
    return moved, plate, {'shift_px': [int(dy), int(dx)], **line_measures(plate, moved)}


def line_measures(plate, moved):
    """Kept plate lines, new line work and dark fills of a registered colour plate."""
    rgb, alpha = moved[..., :3].astype(int), moved[..., 3] > 128
    dark = (luminance(moved) < LINE) & (np.ptp(rgb, axis=2) < NEUTRAL) & alpha
    inverse = inverse_lines(moved)
    # ... and so is a line the edit drew in a colour or a lighter grey (a blue checklist rule, an amber heartbeat, grey list
    # lines), or the edge of a coloured area: any mark at the line. A lost line leaves bare paper, or one undivided fill.
    edge = lambda mask: mask & (ndimage.distance_transform_edt(mask) <= 5)   # a line, or the rim of an area, of that kind
    tinted = (np.ptp(rgb, axis=2) >= NEUTRAL) & alpha
    grey = (luminance(moved) < 205) & ~tinted & ~dark & alpha
    to_colour = ndimage.distance_transform_edt(~(dark | inverse | edge(tinted) | edge(grey)))
    to_plate = ndimage.distance_transform_edt(~plate)
    # New line work: thin dark marks (a fill as dark as ink, such as black hair, survives an opening; a line does not)
    # farther than FAR from every plate line.
    thin = dark & ~ndimage.binary_opening(dark, iterations=4)
    # ... and darker than the paint around it: a navy sky painted at the edge of NEUTRAL flickers between ink and
    # colour pixel by pixel, and that grain is not a line.
    lum = np.where(alpha, luminance(moved), 255.)
    darker = ndimage.uniform_filter(lum, 15) - lum >= CONTRAST
    added = thin_ink(thin & darker & (to_plate > FAR))
    plate_length = max(1, int(thin_ink(plate).sum()))
    return {'inverse_line_px': int(inverse.sum()), 'kept_lines': round(float((to_colour[plate] <= NEAR).mean()), 4) if plate.any() else 0.,
            'added_line_share': round(int(added.sum()) / plate_length, 4), 'plate_line_px': plate_length,
            'dark_fill_share': round(float((dark & (to_plate > FAR)).sum()) / max(1, int(plate.sum())), 4)}


def inverse_lines(rgba):
    """Thin light lines lying inside dark fills (light code lines on a graphite window): lines the edit drew inverted."""
    lum, alpha = luminance(rgba), rgba[..., 3] > 128
    dark = (lum < LINE) & alpha
    light = (lum > 150) & alpha
    broad = ndimage.binary_opening(light, iterations=7)   # light areas wider than a drawn line (~14 px): paper, screens
    inside = ndimage.binary_closing(dark, iterations=8)
    return light & ~broad & inside


def compose(plate_rgb, moved):
    """Ink over colour: the line plate multiplies the paint, so lines darken and its white leaves the paint untouched.
    Where the colour edit drew a line inverted (light inside a dark fill), the light line stays: the pen draws it in the
    colour the picture shows."""
    rgb, alpha = moved[..., :3].astype(float), moved[..., 3].astype(float)
    rgb[alpha == 0] = 255.   # a transparent pixel's colour is undefined; paper under a line is white
    ink = plate_rgb[..., :3].astype(float) / 255.
    keep = ndimage.binary_dilation(inverse_lines(moved), iterations=2)
    ink[keep] = 1.
    darkness = np.clip((235 - luminance(plate_rgb)) / 175., 0, 1)
    out_rgb = np.rint(rgb * ink).clip(0, 255)
    out_alpha = np.maximum(alpha, np.rint(darkness * 255))
    return np.dstack([out_rgb, out_alpha]).astype(np.uint8)


def sheet(plate_rgb, colour_rgba, plate, measures, source, path, width=480):
    """plate | colour as returned | overlap (black kept, red lost, blue new) | registered source, on paper."""
    on_paper = lambda rgba: (rgba[..., :3] * (rgba[..., 3:] / 255.) + 251 * (1 - rgba[..., 3:] / 255.)).astype(np.uint8)
    dy, dx = measures['shift_px']
    overlap = np.full(plate.shape + (3,), 255, np.uint8)
    colour_dark = move((luminance(colour_rgba) < LINE) & (np.ptp(colour_rgba[..., :3].astype(int), axis=2) < NEUTRAL), dy, dx)
    overlap[colour_dark & ~plate] = (34, 64, 160)
    overlap[plate & ~ndimage.binary_dilation(colour_dark, iterations=NEAR)] = (215, 64, 43)
    overlap[plate & ndimage.binary_dilation(colour_dark, iterations=NEAR)] = (30, 30, 30)
    panels = [plate_rgb[..., :3], on_paper(colour_rgba), overlap, on_paper(source)]
    tiles = []
    for panel in panels:
        image = Image.fromarray(np.ascontiguousarray(panel))
        image.thumbnail((width, width * 2))
        tiles.append(image)
    canvas = Image.new('RGB', (sum(tile.width for tile in tiles) + 12 * 3, max(tile.height for tile in tiles)), 'white')
    x = 0
    for tile in tiles:
        canvas.paste(tile, (x, 0))
        x += tile.width + 12
    canvas.save(path, quality=88)


def plate_preview(plate_path, directory):
    """(source, mask) standing in for the coloured picture of a line plate, to time its drawing before the colour edit:
    every enclosed area gets one of three mid tones (the colour edit fills them; the open background stays paper)."""
    image = Image.open(plate_path).convert('RGBA')
    rgba = np.array(image).astype(float)
    rgb = (rgba[..., :3] * (rgba[..., 3:] / 255.) + 255 * (1 - rgba[..., 3:] / 255.)).astype(np.uint8)
    ink = luminance(rgb) < PLATE_INK
    # Areas are found on lines widened by 2 px (a small break in the ink still closes an area), then grown back to the ink.
    seeds, _ = ndimage.label(~ndimage.binary_dilation(ink, iterations=2))
    _, nearest = ndimage.distance_transform_edt(seeds == 0, return_indices=True)
    areas = np.where(ink, 0, seeds[nearest[0], nearest[1]])
    border = np.unique(np.r_[seeds[0], seeds[-1], seeds[:, 0], seeds[:, -1]])
    tones = np.array([[226, 178, 92], [120, 160, 200], [214, 120, 96]], np.uint8)
    out = rgb.copy()
    enclosed = (areas > 0) & ~np.isin(areas, border)
    out[enclosed] = tones[areas[enclosed] % 3]
    out = (out.astype(float) * (rgb.astype(float) / 255.)).astype(np.uint8)
    source, mask = Path(directory) / 'plate-preview.png', Path(directory) / 'plate-preview-outline.png'
    Image.fromarray(out).save(source)
    Image.fromarray(np.where(ink, 255, 0).astype(np.uint8)).save(mask)
    return source, mask


def drop_specks(plate_rgb, moved, area):
    """Remove isolated ink marks smaller than `area` px (stubble dots, dashes, pegboard holes) from the plate, and the same
    marks from the registered colour plate (filled from the nearest paint around them), so neither the pen nor the
    marker has them. Returns (plate, colour, removed pixel count, removed mark count)."""
    ink = luminance(plate_rgb) < PLATE_INK
    labels, count = ndimage.label(ink, structure=np.ones((3, 3)))
    sizes = np.bincount(labels.ravel())
    small = np.zeros(len(sizes), bool)
    small[1:] = sizes[1:] < area
    marks = small[labels]
    if not marks.any():
        return plate_rgb, moved, 0, 0
    kept_ink = ink & ~marks
    # The colour edit may draw such marks again where the plate has none (pegboard holes, stubble): small dark specks
    # away from every kept line go too.
    dark = (luminance(moved) < LINE) & (moved[..., 3] > 128)
    pieces, _ = ndimage.label(dark, structure=np.ones((3, 3)))
    piece_sizes = np.bincount(pieces.ravel())
    small_dark = (piece_sizes < area)[pieces] & dark & (ndimage.distance_transform_edt(~kept_ink) > FAR)
    zone = ndimage.binary_dilation(marks | small_dark, iterations=3)
    plate_rgb = plate_rgb.copy()
    plate_rgb[zone & (luminance(plate_rgb) < 235) & ~kept_ink] = 255
    _, nearest = ndimage.distance_transform_edt(zone, return_indices=True)
    moved = moved.copy()
    moved[zone] = moved[nearest[0], nearest[1]][zone]
    return plate_rgb, moved, int(marks.sum()), int(small.sum())


def run(job, plate_path, colour_path, out_path, mask_path=None, specks=0):
    job = Path(job).resolve()
    resolve = lambda path: Path(path) if Path(path).is_absolute() else job / path
    plate_path, colour_path, out_path = resolve(plate_path), resolve(colour_path), resolve(out_path)
    mask_path = resolve(mask_path) if mask_path else out_path.with_name(out_path.stem + '-outline.png')
    plate_image = Image.open(plate_path)
    plate_rgb = np.array(plate_image.convert('RGBA'))
    # A transparent line plate is ink on nothing: put it on white.
    plate_rgb = (plate_rgb[..., :3] * (plate_rgb[..., 3:] / 255.) + 255 * (1 - plate_rgb[..., 3:] / 255.)).astype(np.uint8)
    colour = Image.open(colour_path).convert('RGBA')
    resized = colour.size != plate_image.size
    if resized:
        colour = colour.resize(plate_image.size, Image.Resampling.LANCZOS)
    colour_rgba = np.array(colour)
    moved, plate, measures = register(plate_rgb, colour_rgba)
    if specks:
        plate_rgb, moved, removed_px, removed_marks = drop_specks(plate_rgb, moved, specks)
        plate = luminance(plate_rgb) < PLATE_INK
        measures.update(line_measures(plate, moved), specks_dropped={'below_px': specks, 'marks': removed_marks, 'pixels': removed_px})
    problems = []
    if not plate.any():
        problems.append('the line plate has no black lines')
    if measures['kept_lines'] < MIN_KEPT:
        problems.append(f'the colour edit kept {measures["kept_lines"]:.1%} of the plate lines (needs {MIN_KEPT:.0%})')
    if max(map(abs, measures['shift_px'])) > MAX_SHIFT * max(plate.shape):
        problems.append(f'the colour edit moved the picture by {measures["shift_px"]} px: it redrew the composition')
    if measures['added_line_share'] > MAX_ADDED:
        problems.append(f'the colour edit drew new lines ({measures["added_line_share"]:.1%} of the plate\'s line length) away from the plate')
    source = compose(plate_rgb, moved)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(source).save(out_path)
    Image.fromarray(np.where(plate, 255, 0).astype(np.uint8)).save(mask_path)
    sheet_path = out_path.with_suffix('.line-first.jpg')
    sheet(plate_rgb, colour_rgba, plate, measures, source, sheet_path)
    report = {'status': 'fail' if problems else 'pass', 'problems': problems, 'prompt_fix': COLOUR_FIX if problems else '',
              'plate': str(plate_path.relative_to(job)) if plate_path.is_relative_to(job) else str(plate_path),
              'colour': str(colour_path.relative_to(job)) if colour_path.is_relative_to(job) else str(colour_path),
              'source': str(out_path.relative_to(job)) if out_path.is_relative_to(job) else str(out_path),
              'outline_mask': str(mask_path.relative_to(job)) if mask_path.is_relative_to(job) else str(mask_path),
              'colour_resized': resized, **measures,
              'sha256': {'plate': digest(plate_path), 'colour': digest(colour_path), 'source': digest(out_path), 'outline_mask': digest(mask_path)},
              'sheet': str(sheet_path.name)}
    out_path.with_suffix('.line-first.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('job', type=Path)
    parser.add_argument('plate', type=Path, help='pure black line plate (generated first)')
    parser.add_argument('colour', type=Path, help='colour edit made with the line plate as its input image')
    parser.add_argument('--out', type=Path, required=True, help='registered source for the illustration action')
    parser.add_argument('--mask', type=Path, help='outline_mask to write (default: OUT stem + "-outline.png")')
    parser.add_argument('--drop-specks', type=int, default=0, metavar='PX',
                        help='remove isolated marks smaller than PX pixels (stubble dots, dashes, pegboard holes) from both plates')
    arguments = parser.parse_args()
    report = run(arguments.job, arguments.plate, arguments.colour, arguments.out, arguments.mask, arguments.drop_specks)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    sys.exit(main())
