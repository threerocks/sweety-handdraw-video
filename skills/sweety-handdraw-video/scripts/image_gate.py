#!/usr/bin/env python3
"""Check a freshly generated image before anything else is built on it, and say how to fix the prompt.

    image_gate.py JOB inputs/art/i04-seabed.png [--visual v04-seabed] [--width 800]

Runs in a few seconds, right after generation (before region marks, drawing review or build):
- transparency: half-transparent fills or a soft glow inside the subject (a fill painted see-through draws as a washed-out
  patch, and a glow halo cannot be drawn with a pen); fully clear gaps such as the space between legs are fine;
- line art: the same gate the build applies (texture, specks, too many crossings);
- drawing time: the strokes and minimum seconds at the planned size, against the seconds the storyboard gives this
  visual (storyboard_lint.py --budget).
--plate checks a line_first line plate before its colour edit, while it is still cheap to change: its enclosed areas
are given stand-in tones so the colouring time counts too (an estimate: the real palette may split areas further).
On failure it prints prompt_fix: a sentence to add to the image prompt (e.g. how many outlines to aim for) before
regenerating. It never edits the image or the records.
"""
import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

HOLE_SHARE = .004      # half-transparent fill larger than this share of the subject is a hole
GLOW_SHARE = .06       # semi-transparent pixels away from the edge larger than this share of the subject is a glow
STROKE_MARGIN = .85    # aim the prompt at this share of the strokes that fit the window


def transparency(image):
    """(problems, measures) for see-through holes and soft glow inside the subject of an RGBA image."""
    alpha = np.asarray(image.convert('RGBA'))[..., 3].astype(float)
    if alpha.min() >= 250:
        return [], {'transparent_background': False}
    solid = alpha > 200
    filled = ndimage.binary_fill_holes(ndimage.binary_closing(solid, iterations=3))
    area = max(1, int(filled.sum()))
    # A fully clear gap enclosed by the subject (between legs, inside a frame or a ring) is background showing through and
    # draws as paper, as it should. The defect is a fill painted half-transparent (light rays, haze, a see-through sea).
    holes = filled & (alpha >= 8) & (alpha < 200)
    holes = ndimage.binary_opening(holes, iterations=2)
    edge = filled & ~ndimage.binary_erosion(filled, iterations=4)
    glow = (alpha > 20) & (alpha < 200) & ~edge & ~holes
    measures = {'transparent_background': True, 'hole_share': round(float(holes.sum()) / area, 4), 'glow_share': round(float(glow.sum()) / area, 4)}
    problems = []
    if measures['hole_share'] > HOLE_SHARE:
        problems.append(('transparency_hole', f'{measures["hole_share"]:.1%} of the subject is painted half-transparent: it would draw as a washed-out patch'))
    if measures['glow_share'] > GLOW_SHARE:
        problems.append(('glow', f'{measures["glow_share"]:.1%} of the subject is a soft semi-transparent glow, which a pen cannot draw'))
    return problems, measures


def find_action(manifest, source, visual):
    for page in manifest.get('pages', []):
        for holder in page.get('visual_groups', []) + page.get('beats', []):
            for action in holder.get('actions') or []:
                if action.get('type') == 'illustration' and action.get('source') == source and (visual is None or action.get('visual') == visual):
                    return action
    return None


def budget_seconds(root, manifest, source, visual):
    """Seconds the storyboard plans for this visual (None when there is no storyboard or locked narration yet)."""
    try:
        import storyboard_lint
        board = storyboard_lint.load(root, manifest)
        report = storyboard_lint.plan(board, storyboard_lint.alignment_beats(root, manifest))
    except (ValueError, FileNotFoundError, KeyError, TypeError):
        return None, None
    for shot in board.get('shots', []):
        for planned in shot.get('visuals', []):
            if (visual and planned.get('id') == visual) or (not visual and planned.get('source') == source):
                row = next((item for item in report['visuals'] if item['visual'] == planned['id']), None)
                return (row['seconds'], planned['id']) if row else (None, planned['id'])
    return None, visual


def check(root, image_path, visual=None, width=None, plate=False):
    import line_drawing
    import line_first
    import stroke_assets
    root = Path(root).resolve()
    manifest = json.loads((root / 'job.json').read_text()) if (root / 'job.json').is_file() else {}
    source = str(Path(image_path).resolve().relative_to(root)) if Path(image_path).resolve().is_relative_to(root) else str(image_path)
    image = Image.open(image_path)
    problems, measures = ([], {'line_plate': True}) if plate else transparency(image)
    action = find_action(manifest, source, visual)
    seconds, visual = budget_seconds(root, manifest, source, visual)
    width = width or (action['width'] if action else 760)
    height = round(width * image.height / image.width) if not action else round(width * action['height'] / action['width'])
    drawing = {**manifest.get('drawing', {}), **((action or {}).get('drawing', {}))}
    plan_width, plan_height, _ = line_drawing.plan_box(width, height, drawing)
    parts = (action or {}).get('parts') or [{'id': 'whole', 'polygon': [[0, 0], [1, 0], [1, 1], [0, 1]]}]
    result = {'image': source, 'visual': visual, 'size': [width, height], 'plan_size': [plan_width, plan_height], **measures}
    with tempfile.TemporaryDirectory() as directory:
        stroke_assets.configure_assets(directory, '/dev/null', 0, int(manifest.get('canvas', {}).get('fps', 30)))
        try:
            # A picture with an explicit outline (line_first or a hand-made mask) is drawn from that mask, so it is measured with it.
            explicit = root / action['outline_mask'] if action and action.get('outline_mask') else None
            built, mask = line_first.plate_preview(image_path, directory) if plate else (Path(image_path), explicit)
            metadata = stroke_assets.build_illustration('gate', built, plan_width, plan_height, (action or {}).get('crop'), parts, mask,
                                                        drawing=drawing, lineart_exception=(action or {}).get('lineart_exception'))
            drawability = metadata['drawability']
            result.update(strokes=drawability['stroke_count'], minimum_seconds=drawability['minimum_seconds'])
        except ValueError as error:
            problems.append(('lineart', str(error).splitlines()[0]))
            drawability = None
    if drawability and seconds is not None:
        result['window_seconds'] = round(seconds, 2)
        if drawability['minimum_seconds'] > seconds:
            fit = max(6, int(math.floor(drawability['stroke_count'] * seconds / drawability['minimum_seconds'] * STROKE_MARGIN)))
            result['target_strokes'] = fit
            problems.append(('too_complex', f'needs {drawability["minimum_seconds"]:.1f}s ({drawability["stroke_count"]} strokes) but the storyboard '
                                            f'gives {seconds:.1f}s'))
    fixes = []
    for kind, _ in problems:
        if kind == 'transparency_hole':
            fixes.append('Every filled area is solid and fully opaque, with no white patches, light rays or see-through holes.')
        elif kind == 'glow':
            fixes.append('Flat colour only: no glow, halo, blur, light bloom or soft shadow around shapes.')
        elif kind == 'too_complex':
            fixes.append(f'Use at most {result["target_strokes"]} separate outlines in the whole image: drop secondary props and background, '
                         'merge small details into their parent shape, keep only what the narration needs.')
            if plate:
                # Cheaper than a new picture: an edit of the plate that names what to drop keeps everything else in place.
                result['edit_fix'] = ('Edit this line drawing (image 1): erase <the details the narration does not need, named one by one>. '
                                      'Keep every other line exactly where it is, with the same thickness; add nothing.')
        elif kind == 'lineart':
            fixes.append('Thick, even, continuous outlines that close every shape; no hatching, texture strokes, specks or broken lines.')
    result.update(status='fail' if problems else 'pass', problems=[{'kind': kind, 'detail': detail} for kind, detail in problems],
                  prompt_fix=' '.join(fixes))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('job', type=Path)
    parser.add_argument('image', type=Path)
    parser.add_argument('--visual')
    parser.add_argument('--width', type=int, help='planned on-screen width (default: the job action, else 760)')
    parser.add_argument('--plate', action='store_true', help='the image is a line_first line plate, checked before its colour edit')
    arguments = parser.parse_args()
    report = check(arguments.job, arguments.image, arguments.visual, arguments.width, arguments.plate)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    sys.exit(main())
