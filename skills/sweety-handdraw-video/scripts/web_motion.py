#!/usr/bin/env python3
"""Web motion layers: render an HTML/JS motion recipe to a transparent frame sequence in headless Chromium.

Recipes live in assets/web-motion/recipes.js and use bundled open-source libraries (anime.js, rough.js,
rough-notation, canvas-confetti, d3-geo and d3-array, topojson-client and world-atlas land outlines; matter-js, Zdog and MathJax
load only for the recipes that need them; licences in assets/web-motion/vendor and assets/web-motion/THIRD_PARTY_NOTICES.md). The page runs on a virtual clock
(assets/web-motion/shim.js), so every frame is exact and repeatable however slow the capture is; renders are cached
by content hash. Requires the optional `playwright` package and its Chromium (scripts/setup.sh --web-motion).

    web_motion.py preview kinetic_title '{"text": "一条消息的旅程", "keywords": ["旅程"]}' --size 900x260 --seconds 1.6 --out /tmp/title
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
ASSETS = PACKAGE / 'assets' / 'web-motion'
LIBRARIES = ('vendor/anime.js', 'vendor/rough.js', 'vendor/rough-notation.js', 'vendor/confetti.js', 'vendor/d3-array.js', 'vendor/d3-geo.js', 'vendor/topojson.js',
             'vendor/land-110m.js', 'recipes.js')
# Heavier libraries load only for the recipe that uses them (MathJax alone is 2 MB).
RECIPE_LIBRARIES = {'physics': ('vendor/matter.js',), 'spin3d': ('vendor/zdog.js',), 'formula': ('vendor/mathjax-tex-svg.js',)}
RECIPES = ('kinetic_title', 'rough_annotate', 'sketch_shapes', 'burst', 'number_ticker', 'callout', 'progress_ring', 'morph', 'rough_chart',
           'map_route', 'physics', 'spin3d', 'formula', 'roadmap', 'table', 'flow', 'swimlane', 'checklist', 'cards', 'pulse', 'custom')
# MathJax: typeset on request only, no menu, no file loads (the bundle already holds every TeX extension), self-contained SVG paths.
MATHJAX_CONFIG = ('window.MathJax={loader:{load:[]},startup:{typeset:false},options:{enableMenu:false},'
                  'svg:{fontCache:"none"},tex:{packages:{"[+]":["color","cancel","boldsymbol"]}}};')


def libraries(recipe):
    return LIBRARIES[:-1] + RECIPE_LIBRARIES.get(recipe, ()) + LIBRARIES[-1:]
VERSION = 'web-motion-2'


def available():
    """(True, None) when Playwright and its Chromium can run, else (False, reason with the fix)."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False, 'Python package playwright is missing: bash scripts/setup.sh --web-motion'
    try:
        with sync_playwright() as playwright:
            playwright.chromium.launch().close()
    except Exception as error:  # a missing browser build raises a long Playwright message
        return False, f'Chromium for Playwright is not installed ({str(error).splitlines()[0]}): bash scripts/setup.sh --web-motion'
    return True, None


def fingerprint(recipe, params, width, height, scale, fps, frames, seed):
    digest = hashlib.sha256(VERSION.encode())
    for name in ('shim.js',) + libraries(recipe):
        digest.update((ASSETS / name).read_bytes())
    digest.update(json.dumps([recipe, params, width, height, scale, fps, frames, seed], sort_keys=True, ensure_ascii=False).encode())
    return digest.hexdigest()[:20]


def page_html(recipe, params, width, height, scale, seconds, seed):
    scripts = (f'<script>{MATHJAX_CONFIG}</script>' if recipe == 'formula' else '') + ''.join(
        f'<script src="{(ASSETS / name).as_uri()}"></script>' for name in libraries(recipe))
    context = json.dumps({'width': width, 'height': height, 'scale': scale, 'duration': seconds * 1000})
    return (f'<!doctype html><html><head><meta charset="utf-8"><style>html,body{{margin:0;background:transparent;overflow:hidden}}'
            f'#root{{position:absolute;left:0;top:0;width:{width}px;height:{height}px}}</style>{scripts}</head><body><div id="root"></div>'
            f'<script>window.__seedRandom({int(seed)});window.__error=null;try{{if(!window.Recipes[{json.dumps(recipe)}])throw new Error("unknown recipe");'
            f'window.Recipes[{json.dumps(recipe)}](document.getElementById("root"),{json.dumps(params, ensure_ascii=False)},{context});}}'
            f'catch(e){{window.__error=String(e&&e.stack||e)}}</script></body></html>')


def prepare(recipe, params):
    """Fill text layout the browser cannot do well: word segmentation (jieba via caption_assets) and keyword ranges."""
    params = dict(params)
    if recipe == 'kinetic_title' and params.get('text') and 'words' not in params:
        from caption_assets import break_costs, keyword_spans, number_spans
        text, keywords = params['text'], params.get('keywords') or []
        costs = break_costs(text, keywords)
        locked = {index for start, end in keyword_spans(text, keywords) + number_spans(text) for index in range(start + 1, end)}
        cuts = [0] + [index for index in range(1, len(text)) if index not in locked
                      and (costs[index] <= 12 or text[index - 1].isspace() != text[index].isspace())] + [len(text)]
        params['words'] = [text[a:b] for a, b in zip(cuts, cuts[1:]) if b > a]
        params['accents'] = [list(span) for span in keyword_spans(text, keywords)]
    return params


def render(recipe, params, width, height, seconds, cache, fps=30, scale=1., seed=1):
    """Frames [0, ceil(seconds*fps)] of a recipe as RGBA PNGs (width*scale x height*scale) under cache/<hash>/.
    Returns {'directory', 'frames', 'fps', 'size'}; a cached render is reused."""
    if recipe not in RECIPES:
        raise ValueError(f'Unknown motion recipe {recipe!r}; use one of {", ".join(RECIPES)}')
    width, height = int(round(width)), int(round(height))
    if width < 4 or height < 4 or not 0 < seconds <= 30:
        raise ValueError('A motion layer needs a box of at least 4 px and 0..30 seconds')
    frames = int(math.ceil(seconds * fps)) + 1
    params = prepare(recipe, params)
    key = fingerprint(recipe, params, width, height, scale, fps, frames, seed)
    directory = Path(cache) / key
    record = directory / 'frames.json'
    if record.is_file():
        result = json.loads(record.read_text())
        result['clipped'] = edge_clipped(directory, result['frames'])
        return result
    ok, reason = available()
    if not ok:
        raise RuntimeError(reason)
    from playwright.sync_api import sync_playwright
    directory.mkdir(parents=True, exist_ok=True)
    page_path = directory / 'page.html'
    page_path.write_text(page_html(recipe, params, width, height, scale, seconds, seed))
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={'width': width, 'height': height}, device_scale_factor=scale)
        page.add_init_script(path=str(ASSETS / 'shim.js'))
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(page_path.as_uri())
        page.wait_for_load_state('load')
        failure = page.evaluate('window.__error')
        if failure or errors:
            browser.close()
            raise ValueError(f'Motion recipe {recipe} failed: {failure or errors[0]}')
        step = 1000. / fps
        for index in range(frames):
            if index:
                page.evaluate(f'window.__advance({step})')
            page.screenshot(path=str(directory / f'{index:05d}.png'), omit_background=True,
                            clip={'x': 0, 'y': 0, 'width': width, 'height': height})
        browser.close()
        if errors:
            raise ValueError(f'Motion recipe {recipe} failed while playing: {errors[0]}')
    result = {'directory': str(directory), 'frames': frames, 'fps': fps, 'size': [int(round(width * scale)), int(round(height * scale))],
              'recipe': recipe, 'seconds': seconds}
    record.write_text(json.dumps(result))
    return {**result, 'clipped': edge_clipped(directory, frames)}


def edge_clipped(directory, frames, margin=2, threshold=40):
    """True when the settled (last) frame has visible content on the box border: something runs off the overlay."""
    from PIL import Image
    import numpy as np
    alpha = np.asarray(Image.open(Path(directory) / f'{frames - 1:05d}.png').convert('RGBA'))[..., 3]
    border = np.concatenate([alpha[:margin].ravel(), alpha[-margin:].ravel(), alpha[:, :margin].ravel(), alpha[:, -margin:].ravel()])
    return bool((border > threshold).any())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('check')
    preview = commands.add_parser('preview')
    preview.add_argument('recipe', choices=RECIPES)
    preview.add_argument('params', help='JSON parameters')
    preview.add_argument('--size', default='900x300')
    preview.add_argument('--seconds', type=float, default=1.5)
    preview.add_argument('--scale', type=float, default=1.)
    preview.add_argument('--out', type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == 'check':
        ok, reason = available()
        print(json.dumps({'available': ok, 'fix': reason}, ensure_ascii=False))
        return
    width, height = (int(value) for value in arguments.size.lower().split('x'))
    result = render(arguments.recipe, json.loads(arguments.params), width, height, arguments.seconds, arguments.out, scale=arguments.scale)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
