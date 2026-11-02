#!/usr/bin/env python3
"""Cover = a finished picture on the film's paper + the exact title as editable text layers (data in the job).

    cover.py JOB [--spec design/cover.json]

design/cover.json: {"base": "inputs/art/d01-list.png", "width": 720, "top": 790, "paper": "#F2EFE8" (default: job paper),
  "layers": {"title_1": {"text": "存款不到七十万，", "font": "hand", "size": 132, "color": "#22312B", "y": 170}, ...},
  "brief": "...", "prompt": "..." (optional)}
A layer's font is "hand" (the handwriting chain), "sans" (the caption chain) or {"path", "index"}; it is centred unless it
gives "x". Writes inputs/cover.png and design/cover-layers.json (the editable layers, with the fonts actually used and
each layer's box). The art-direction record gets a composed_with_verified_layers entry whose brief and prompt are bound
before composing; look at the cover, then: art_pipeline.py JOB record cover "已看成图：……".
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
KEY = (232, 16)   # white paper of a picture fades into the cover's paper from this channel minimum over this range


def resolve_font(spec, text):
    sys.path.insert(0, str(SCRIPTS))
    import fonts
    if isinstance(spec, dict):
        return fonts.select('handwriting', text, spec)
    if isinstance(spec, list):
        return spec[0], int(spec[1])
    return fonts.select('handwriting' if spec in (None, 'hand', 'handwriting') else 'caption', text)


def register(root, spec):
    """Brief and prompt for the composed cover, bound in the art-direction record before the cover is composed."""
    record_path = root / 'design/art-direction.json'
    if not record_path.is_file():
        return 'no art-direction record yet (art_pipeline.py prompts creates it)'
    record = json.loads(record_path.read_text())
    entry = next((item for item in record.get('images', []) if item.get('source') == '../inputs/cover.png'), None)
    if entry and entry.get('status') == 'pass' and (root / 'design/briefs/cover.md').is_file():
        return 'kept the reviewed cover entry: record it again after looking at the new cover'
    titles = ' / '.join(layer['text'] for layer in spec['layers'].values())
    brief = spec.get('brief') or (f'# 美术简报：cover（封面，合成）\n\n- 交付：封面合成，不另外生图。主视觉用 {spec["base"]} 整幅，白纸抠成纸面色。\n'
                                  f'- 文字：可编辑文字图层（design/cover-layers.json）：{titles}。标题与首句一致。\n')
    prompt = spec.get('prompt') or (f'Composition (no image generation): {spec["base"]} scaled to {spec.get("width", 720)} px wide at '
                                    f'y={spec.get("top", 790)} on the paper; title layers: {titles}.')
    for path, text in (('design/briefs/cover.md', brief), ('design/prompts/cover.txt', prompt)):
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text)
    if entry is None:
        record.setdefault('images', []).append({'source': '../inputs/cover.png', 'method': 'composed_with_verified_layers',
                                                'brief_file': 'briefs/cover.md', 'prompt_file': 'prompts/cover.txt', 'status': 'pending'})
        record_path.write_text(json.dumps(record, ensure_ascii=False, indent=1))
    bound = subprocess.run([sys.executable, str(SCRIPTS / 'art_direction.py'), 'bind-evidence', str(record_path)], capture_output=True, text=True)
    if bound.returncode:
        raise RuntimeError(bound.stderr.strip() or bound.stdout.strip())
    return 'registered and bound (status pending until recorded)'


def compose(root, spec):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    job = json.loads((root / 'job.json').read_text()) if (root / 'job.json').is_file() else {}
    canvas_width, canvas_height = spec.get('canvas') or (job.get('canvas', {}).get('width', 1280), job.get('canvas', {}).get('height', 1920))
    paper = spec.get('paper') or job.get('paper') or '#F2EFE8'
    rgb = tuple(int(paper.lstrip('#')[k:k + 2], 16) for k in (0, 2, 4))
    with Image.open(root / spec['base']) as opened:
        pixels = np.array(opened.convert('RGBA'))
    # The picture's white paper becomes the cover's paper (as the engine keys it on the page).
    whiteness = np.clip((pixels[..., :3].min(axis=2).astype(float) - KEY[0]) / KEY[1], 0, 1)
    pixels[..., 3] = (pixels[..., 3] * (1 - whiteness)).astype(np.uint8)
    base = Image.fromarray(pixels)
    width = int(spec.get('width', 720))
    scaled = base.resize((width, round(base.height * width / base.width)), Image.LANCZOS)
    canvas = Image.new('RGBA', (canvas_width, canvas_height), rgb + (255,))
    offset = (int(spec['left']) if 'left' in spec else (canvas_width - width) // 2, int(spec.get('top', 790)))
    canvas.alpha_composite(scaled, offset)
    canvas = canvas.convert('RGB')
    draw = ImageDraw.Draw(canvas)
    layers = {}
    for name, layer in spec['layers'].items():
        path, index = resolve_font(layer.get('font', 'hand'), layer['text'])
        font = ImageFont.truetype(path, int(layer['size']), index=index)
        box = draw.textbbox((0, 0), layer['text'], font=font)
        x = int(layer['x']) - box[0] if 'x' in layer else (canvas_width - (box[2] - box[0])) // 2 - box[0]
        draw.text((x, layer['y'] - box[1]), layer['text'], font=font, fill=layer.get('color', '#22312B'))
        layers[name] = {**layer, 'font': [path, index], 'bbox': [x + box[0], layer['y'], x + box[2], layer['y'] + box[3] - box[1]]}
    (root / 'inputs').mkdir(exist_ok=True)
    canvas.save(root / 'inputs/cover.png')
    (root / 'design/cover-layers.json').write_text(json.dumps({'base': spec['base'], 'paper': paper, 'canvas': [canvas_width, canvas_height],
                                                               'base_scaled_to': [width, scaled.height], 'base_offset': list(offset),
                                                               'layers': layers}, ensure_ascii=False, indent=1))
    return layers


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('job', type=Path)
    parser.add_argument('--spec', default='design/cover.json')
    parser.add_argument('--no-record', action='store_true', help='compose only (a trial); do not touch the art-direction record')
    args = parser.parse_args(argv)
    root = args.job.resolve()
    spec = json.loads((root / args.spec).read_text())
    status = 'not registered (--no-record)' if args.no_record else register(root, spec)
    layers = compose(root, spec)
    print(json.dumps({'cover': 'inputs/cover.png', 'layers': {name: layer['bbox'] for name, layer in layers.items()}, 'record': status,
                      'next': 'look at it, then art_pipeline.py JOB record cover "已看成图：……"'}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
