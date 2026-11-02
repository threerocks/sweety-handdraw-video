#!/usr/bin/env python3
"""Page layout helper: platform safe zones, layout templates that fill the frame, and a drawing-time probe.

    layout_plan.py zones --canvas 1080x1920
    layout_plan.py fit --canvas 1080x1920 --template hero_with_labels --aspect 1.5 --labels 2
    layout_plan.py fit --canvas 1080x1920 --template pair_rows --aspect 1.4 --aspect 1.2
    layout_plan.py probe JOB inputs/art/scene.png --width 900 [--parts-from VISUAL_ID]

Boxes are screen pixels for job.json actions (x, y, width, height). The templates are starting points that
put the main picture large and inside the platform-safe area; the director still decides what each slot shows."""
import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

from caption_assets import PLATFORM_UI, UI_GAP

# Fractions of the canvas height, sharing the platform UI band with caption_assets: vertical feeds (Douyin, 视频号,
# 小红书) cover the top ~7% with status and tabs and the bottom ~20% with account name, description and music; the
# caption sits right above that band (two lines from ~70%), and the right edge carries the like/comment rail.
_CAPTION_BOTTOM = 1 - PLATFORM_UI['bottom'] - UI_GAP
VERTICAL = {'top_ui': (0., PLATFORM_UI['top']), 'header': (PLATFORM_UI['top'], .17), 'main': (.17, .645), 'support': (.645, .70),
            'caption': (.70, _CAPTION_BOTTOM), 'bottom_ui': (1 - PLATFORM_UI['bottom'], 1.)}
LANDSCAPE = {'top_ui': (0., .04), 'header': (.04, .15), 'main': (.15, .78), 'support': (.78, .84), 'caption': (.84, .96), 'bottom_ui': (.96, 1.)}
RIGHT_RAIL = PLATFORM_UI['right']   # vertical: keep key content off the right-hand like/comment/share buttons
SIDE = .06         # side gutter
TEMPLATES = ('hero', 'hero_with_labels', 'pair_rows', 'pair_cols', 'trio_rows', 'grid4', 'list', 'compare')


def zones(width, height):
    table = VERTICAL if height > width else LANDSCAPE
    result = {name: {'x': round(width * SIDE), 'y': round(height * top), 'width': round(width * (1 - 2 * SIDE)),
                     'height': round(height * (bottom - top))} for name, (top, bottom) in table.items()}
    if height > width:
        result['safe_right'] = round(width * (1 - RIGHT_RAIL))
    # The watermark (when enabled) takes the quietest of four corners per page; keep one of them free of ink on every page.
    mark_w, mark_h, edge = round(width * .36), round(height * .04), round(height * .03)
    result['watermark_corners'] = {name: {'x': x, 'y': y, 'width': mark_w, 'height': mark_h} for name, (x, y) in {
        'top_right': (width - mark_w - edge, edge), 'top_left': (edge, edge),
        'bottom_right': (width - mark_w - edge, round(height * .80) - mark_h), 'bottom_left': (edge, round(height * .80) - mark_h)}.items()}
    return result


def contain(box, aspect):
    """Largest box of the given width/height aspect centered inside box."""
    width, height = box['width'], box['height']
    if width / height > aspect:
        width = height * aspect
    else:
        height = width / aspect
    return {'x': round(box['x'] + (box['width'] - width) / 2), 'y': round(box['y'] + (box['height'] - height) / 2),
            'width': round(width), 'height': round(height)}


def split(box, count, vertical=True, gap=.04):
    """count equal cells of box stacked (vertical) or side by side, separated by gap x the box side."""
    if vertical:
        space = box['height'] * gap
        each = (box['height'] - space * (count - 1)) / count
        return [{'x': box['x'], 'y': round(box['y'] + index * (each + space)), 'width': box['width'], 'height': round(each)} for index in range(count)]
    space = box['width'] * gap
    each = (box['width'] - space * (count - 1)) / count
    return [{'x': round(box['x'] + index * (each + space)), 'y': box['y'], 'width': round(each), 'height': box['height']} for index in range(count)]


def label_slots(box, count, size):
    rows = split(box, count, True, .15) if count else []
    return [{'x': row['x'], 'y': row['y'], 'width': row['width'], 'size': min(size, round(row['height'] * .8))} for row in rows]


def fit(width, height, template, aspects, labels=0):
    """Slots for a page: pictures (contain-fitted to their aspect ratios), header line, label lines, all screen pixels."""
    if template not in TEMPLATES:
        raise ValueError(f'template must be one of {TEMPLATES}')
    area = zones(width, height)
    vertical = height > width
    main = dict(area['main'])
    if vertical:
        main['width'] = min(main['width'], area['safe_right'] - main['x'])
    header = area['header']
    title_size = round(min(header['height'] * .55, width * (.075 if vertical else .045)))
    label_size = round(width * (.06 if vertical else .032))
    need = {'hero': 1, 'hero_with_labels': 1, 'pair_rows': 2, 'pair_cols': 2, 'trio_rows': 3, 'grid4': 4, 'list': len(aspects) or 3, 'compare': 2}[template]
    aspects = list(aspects) + [1.] * max(0, need - len(aspects))
    pictures, texts = [], []
    if template in ('hero', 'hero_with_labels'):
        body = dict(main)
        if template == 'hero_with_labels' and labels:
            band = round(main['height'] * min(.32, .11 * labels))
            body['height'] -= band
            texts = label_slots({'x': main['x'], 'y': body['y'] + body['height'], 'width': main['width'], 'height': band + (area['support']['height'] if vertical else 0)}, labels, label_size)
        pictures = [contain(body, aspects[0])]
    elif template in ('pair_rows', 'trio_rows'):
        pictures = [contain(cell, aspect) for cell, aspect in zip(split(main, need, True), aspects)]
    elif template in ('pair_cols', 'compare'):
        cells = split(main, 2, False, .1 if template == 'compare' else .05)
        if template == 'compare':
            for cell in cells:
                cell['height'] -= round(label_size * 1.8)
            texts = [{'x': cell['x'], 'y': cell['y'] + cell['height'] + round(label_size * .3), 'width': cell['width'], 'size': label_size} for cell in cells]
        pictures = [contain(cell, aspect) for cell, aspect in zip(cells, aspects)]
    elif template == 'grid4':
        rows = split(main, 2, True)
        pictures = [contain(cell, aspect) for cell, aspect in zip([cell for row in rows for cell in split(row, 2, False)], aspects)]
    elif template == 'list':
        rows = split(main, need, True, .08)
        for row, aspect in zip(rows, aspects):
            icon = {'x': row['x'], 'y': row['y'], 'width': round(row['height'] * 1.1), 'height': row['height']}
            pictures.append(contain(icon, aspect))
            texts.append({'x': icon['x'] + icon['width'] + round(label_size * .6), 'y': row['y'] + round((row['height'] - label_size) / 2),
                          'width': row['width'] - icon['width'] - round(label_size * .6), 'size': label_size})
    return {'canvas': [width, height], 'template': template, 'header': {'x': header['x'], 'y': header['y'], 'width': header['width'], 'size': title_size},
            'pictures': pictures, 'labels': texts, 'zones': area,
            'note': 'Text slot width is room for the line, not a promise it fits: about width/size CJK characters per line.'}


def probe(job, source, width, visual=None):
    """Minimum drawing seconds of one picture at a given on-screen width, with the parts of a job action that uses it."""
    import stroke_assets
    job = Path(job).resolve()
    manifest = json.loads((job / 'job.json').read_text())
    actions = [action for page in manifest['pages'] for holder in page.get('visual_groups', []) + page['beats']
               for action in holder.get('actions') or [] if action.get('type') == 'illustration']
    matches = [action for action in actions if action.get('source') == source and (visual is None or action.get('visual') == visual)]
    if not matches:
        raise ValueError('No illustration action in this job uses that source (give --parts-from the visual id)')
    action = matches[0]
    height = round(width * action['height'] / action['width'])
    resolve = lambda key: (job / action[key]) if action.get(key) else None
    drawing = {**manifest.get('drawing', {}), **action.get('drawing', {})}
    import line_drawing
    plan_width, plan_height, _ = line_drawing.plan_box(width, height, drawing)   # what the build plans; display size is re-rendered
    with tempfile.TemporaryDirectory() as directory:
        stroke_assets.configure_assets(directory, '/dev/null', 0, int(manifest.get('canvas', {}).get('fps', 30)))
        metadata = stroke_assets.build_illustration('probe', job / source, plan_width, plan_height, action.get('crop'), action.get('parts'),
                                                    resolve('outline_mask'), region_map_path=resolve('region_map'), part_regions=action.get('part_regions'),
                                                    lineart_exception=action.get('lineart_exception'), drawing=drawing)
    drawability = metadata['drawability']
    return {'source': source, 'size': [width, height], 'plan_size': [plan_width, plan_height], 'minimum_seconds': drawability['minimum_seconds'],
            'paced_seconds': round(metadata['seconds'], 2), 'strokes': drawability['stroke_count'], 'fill_regions': metadata['painting'].get('fill_regions'),
            'lineart_gate': drawability['lineart_gate']['passed'] if drawability.get('lineart_gate') else None}


def canvas_size(value):
    width, height = (int(part) for part in value.lower().split('x'))
    return width, height


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    zone = commands.add_parser('zones')
    zone.add_argument('--canvas', default='1280x1920')
    fitting = commands.add_parser('fit')
    fitting.add_argument('--canvas', default='1280x1920')
    fitting.add_argument('--template', required=True, choices=TEMPLATES)
    fitting.add_argument('--aspect', type=float, action='append', default=[], help='width/height of each picture, in slot order')
    fitting.add_argument('--labels', type=int, default=0)
    probing = commands.add_parser('probe')
    probing.add_argument('job')
    probing.add_argument('source', help='image path as written in job.json')
    probing.add_argument('--width', type=int, action='append', required=True)
    probing.add_argument('--parts-from', help='visual id of the action whose parts to use')
    arguments = parser.parse_args()
    if arguments.command == 'zones':
        result = zones(*canvas_size(arguments.canvas))
    elif arguments.command == 'fit':
        result = fit(*canvas_size(arguments.canvas), arguments.template, arguments.aspect, arguments.labels)
    else:
        result = [probe(arguments.job, arguments.source, width, arguments.parts_from) for width in arguments.width]
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    sys.exit(main())
