"""Snapshot the hand-drawn catalog as a text list, validate style decisions. No preview wall."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import secrets
import shutil
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlsplit
from PIL import Image
from hosted_assets import fetch, resolve_asset

PACKAGE = Path(__file__).resolve().parents[1]
# Bundled styles with clean ink outlines and flat color are drawn stroke by stroke; styles without outlines
# (watercolor, gouache, paper cut, pixel, ink wash, crayon, painterly, 3D) are revealed by the brush instead.
DRAWN_STYLES = {'1', '4', '5', '17', '19', '20'}
MONOCHROME_STYLES = {'1'}  # pure black ink on paper: the picture is never coloured
MODE_LABELS = {
    'catalog_auto': '不主动选择，由模型自动匹配（优先逐笔绘制、上色的画风）',
    'inferred': '不选择列表中的，模型推理产出',
    'custom': '自定义描述风格',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_catalog(job, source, require_previews=False):
    """Build style/catalog.json from STYLES.md. Previews are best-effort and unused by native menus."""
    job, source = Path(job).resolve(), Path(source).resolve()
    directory = job / 'style'
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'previews').mkdir(exist_ok=True)
    (directory / 'recipes').mkdir(exist_ok=True)
    text = (source / 'STYLES.md').read_text()
    matches = list(re.finditer(r'^## (\d+(?:\.\d+)?)[.]? (.+)$', text, re.M))
    manifest_path = source / 'assets/image-manifest.json'
    remote_images = json.loads(manifest_path.read_text())['images'] if manifest_path.exists() else {}
    fallback = PACKAGE / 'assets/style-previews'
    hosted_previews = {}
    if (fallback / 'manifest.json').is_file():
        hosted_previews = {item['id']: item for item in json.loads((fallback / 'manifest.json').read_text())['images']}
    catalog = []
    for index, match in enumerate(matches):
        identity, name = match.groups()
        recipe = text[match.start():matches[index + 1].start() if index + 1 < len(matches) else len(text)]
        recipe_path = directory / 'recipes' / (identity + '.md')
        recipe_path.write_text(recipe)
        candidates = list((source / 'examples').glob(identity.zfill(2) + '-*.png')) if '.' not in identity else []
        candidates += list((source / ('assets/style-' + identity)).glob('anchor-*.png'))
        preview_path = directory / 'previews' / (identity + '.jpg')
        preview = None
        try:
            if candidates:
                image = Image.open(candidates[0]).convert('RGB')
                image.thumbnail((720, 720))
                image.save(preview_path, quality=87)
                preview = 'previews/' + preview_path.name
            elif identity in hosted_previews and source == (PACKAGE / 'dependencies/hand-drawn-styles').resolve():
                shutil.copyfile(resolve_asset(fallback / hosted_previews[identity]['thumbnail']), preview_path)
                preview = 'previews/' + preview_path.name
            else:
                keys = [key for key in remote_images if key.startswith('examples/' + identity.zfill(2) + '-')]
                keys += [key for key in remote_images if key.startswith('assets/style-' + identity + '/anchor-')]
                if keys:
                    record = remote_images[keys[0]]
                    urls = [record['url']] + ([record['fallback_url']] if record.get('fallback_url') else [])
                    content = fetch(keys[0], {**record, 'urls': urls}, source / keys[0])
                    image = Image.open(io.BytesIO(content)).convert('RGB')
                    image.thumbnail((720, 720))
                    image.save(preview_path, quality=87)
                    preview = 'previews/' + preview_path.name
                elif require_previews:
                    raise ValueError('No real preview for style ' + identity)
        except (OSError, ValueError, KeyError):
            if require_previews:
                raise
            preview = None
        bundled = source == (PACKAGE / 'dependencies/hand-drawn-styles').resolve()
        entry = {'id': identity, 'name': name, 'preview': preview,
                 'recipe': 'recipes/' + recipe_path.name, 'recipe_sha256': digest(recipe_path)}
        if bundled:
            entry.update(drawing_mode='draw' if identity in DRAWN_STYLES else 'reveal',
                         monochrome=identity in MONOCHROME_STYLES)
        catalog.append(entry)
    if not catalog:
        raise ValueError('No numbered styles in the selected STYLES.md')
    catalog.sort(key=lambda entry: tuple(int(part) for part in entry['id'].split('.')))
    snapshot = {'source': str(source), 'styles_sha256': digest(source / 'STYLES.md'), 'styles': catalog}
    (directory / 'catalog.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2))
    return snapshot


def text_style_menu(catalog, as_records=False):
    """Default option is model inference; numbered styles are soft hints, not constraints."""
    styles = catalog['styles'] if isinstance(catalog, dict) else catalog
    if as_records:
        rows = [{'id': 'inferred', 'label': '模型推理（默认）', 'hint': '按主题与内容自动选；列表不约束模型'}]
        for entry in styles:
            mode = entry.get('drawing_mode', '')
            tag = ' · 逐笔' if mode == 'draw' else (' · 揭示' if mode == 'reveal' else '')
            rows.append({'id': entry['id'], 'label': entry['name'], 'drawing_mode': mode, 'tag': tag})
        rows.append({'id': 'custom', 'label': '自定义描述', 'hint': '用户一句话'})
        return rows
    lines = ['0) 模型推理（默认）— 按主题与内容自动选；下列风格仅为可选项，不约束模型']
    for index, entry in enumerate(styles, 1):
        mode = entry.get('drawing_mode', '')
        tag = ' [逐笔]' if mode == 'draw' else (' [揭示]' if mode == 'reveal' else '')
        lines.append(f"{index}) {entry['id']}  {entry['name']}{tag}")
    lines.append(f"{len(styles) + 1}) 自定义描述")
    return lines


def validate_selection(selection, catalog):
    mode = selection.get('mode')
    if mode not in {'catalog', *MODE_LABELS}:
        raise ValueError('Select a numbered style or one of the three style routes')
    if mode == 'catalog' and selection.get('catalog_id') not in {entry['id'] for entry in catalog['styles']}:
        raise ValueError('Unknown style number')
    if mode != 'catalog' and selection.get('catalog_id'):
        raise ValueError('Only a directly selected style can carry catalog_id in selection.json')
    if mode == 'custom' and not selection.get('user_description', '').strip():
        raise ValueError('Describe the custom style before confirming')
    if mode != 'custom' and selection.get('user_description'):
        raise ValueError('A user description belongs to the custom route')
    return True


def verify_style(job):
    directory = Path(job) / 'style'
    selection_path, resolved_path = directory / 'selection.json', directory / 'resolved.json'
    if not selection_path.is_file() or not resolved_path.is_file():
        raise ValueError('Style choice and model resolution are required before image generation/build')
    catalog = json.loads((directory / 'catalog.json').read_text())
    selection = json.loads(selection_path.read_text())
    resolved = json.loads(resolved_path.read_text())
    validate_selection(selection, catalog)
    if selection.get('confirmation') not in {'user_dialog', 'explicit_user_request', 'user_delegated', 'timeout_default'} or not selection.get('evidence'):
        raise ValueError('Style choice needs real user selection or explicit delegation evidence')
    if resolved.get('selection_sha256') != digest(selection_path) or resolved.get('mode') != selection['mode']:
        raise ValueError('Style resolution is stale or silently changed the selected route')
    if resolved.get('status') != 'resolved' or not resolved.get('rationale') or not resolved.get('style_name'):
        raise ValueError('Resolve the selected style route before production')
    specification = resolved.get('style_spec', {})
    if any(not specification.get(key) for key in ('line', 'palette', 'shading', 'texture', 'background')):
        raise ValueError('Style specification needs line, palette, shading, texture and background')
    expected, monochrome = None, False
    if selection['mode'] in ('catalog', 'catalog_auto'):
        identity = resolved.get('catalog_id')
        entry = next((entry for entry in catalog['styles'] if entry['id'] == identity), None)
        if not entry or resolved.get('recipe_sha256') != entry['recipe_sha256'] or digest(directory / entry['recipe']) != entry['recipe_sha256']:
            raise ValueError('Catalog styles must use their complete, unchanged recipe snapshot')
        if selection['mode'] == 'catalog' and identity != selection['catalog_id']:
            raise ValueError('The model must not replace the selected numbered style')
        if selection['mode'] == 'catalog_auto' and not resolved.get('content_fit'):
            raise ValueError('Automatic catalog matching needs a content-fit explanation')
        expected = entry.get('drawing_mode')
        monochrome = entry.get('monochrome', False)
    elif resolved.get('catalog_id') or resolved.get('recipe_sha256') or resolved.get('catalog_recipe'):
        raise ValueError('Inferred/custom styles must be independent of hand-drawn recipes')
    if selection['mode'] == 'inferred' and not resolved.get('content_fit'):
        raise ValueError('An inferred style needs a content-based rationale')
    if selection['mode'] == 'custom' and resolved.get('style_basis') != selection['user_description']:
        raise ValueError('Custom style must be based on the exact user description')
    drawing_mode = resolved.get('drawing_mode', expected)
    if drawing_mode not in ('draw', 'reveal') or (expected and drawing_mode != expected):
        raise ValueError('style/resolved.json needs drawing_mode: "draw" only for clean ink outlines with flat color, otherwise "reveal"; '
                         'catalog styles keep the catalog value' + (f' ("{expected}")' if expected else ''))
    # Reveal brushes the finished image in region by region: it gives up drawing, so only the user can trade drawing
    # for a look. Clicking a card the picker labels "揭示" is that choice; any other route needs the user's own answer.
    informed_pick = selection['mode'] == 'catalog' and selection['confirmation'] == 'user_dialog'
    if drawing_mode == 'reveal' and not informed_pick and not str(resolved.get('reveal_accepted', '')).strip():
        raise ValueError('Reveal replaces stroke-by-stroke drawing, so only the user may choose it. Resolve a drawable version of '
                         'this look (clean continuous ink outlines, flat colour) with drawing_mode "draw"; or tell the user that '
                         'this look can only be brushed in, not drawn, and record their answer in resolved.json reveal_accepted')
    model_chosen = selection['mode'] == 'catalog_auto' or selection['confirmation'] in ('timeout_default', 'user_delegated')
    if model_chosen and monochrome and not resolved.get('monochrome_reason'):
        raise ValueError('A style the model picks should be drawn and coloured: prefer ' + ', '.join(sorted(DRAWN_STYLES - MONOCHROME_STYLES, key=float))
                         + '; black-and-white style 1 needs monochrome_reason: why this content must stay uncoloured')
    import style_fit
    fit = style_fit.check(job, resolved, selection)
    return {'mode': selection['mode'], 'name': resolved['style_name'], 'resolved_sha256': digest(resolved_path), 'drawing_mode': drawing_mode,
            'content_fit_check': fit}


def serve_picker(job, source, port=0):
    job = Path(job).resolve()
    if (job / 'style/selection.json').is_file():
        print('STYLE_SELECTION_ALREADY_SAVED=' + str(job / 'style/selection.json'), flush=True)
        return
    catalog = prepare_catalog(job, source)
    token = secrets.token_urlsafe(24)
    template = (PACKAGE / 'assets/style-picker.html').read_text()
    html = template.replace('__CATALOG__', json.dumps(catalog, ensure_ascii=False).replace('<', '\\u003c')).replace('__TOKEN__', token).encode()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *arguments):
            pass
        def reply(self, status, payload, content_type='application/json'):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(payload)
        def do_GET(self):
            path = urlsplit(self.path).path
            if path == '/':
                return self.reply(200, html, 'text/html; charset=utf-8')
            match = re.fullmatch(r'/previews/(\d+(?:\.\d+)?)\.jpg', path)
            if match:
                preview = job / 'style/previews' / (match.group(1) + '.jpg')
                if preview.is_file():
                    return self.reply(200, preview.read_bytes(), 'image/jpeg')
            self.reply(404, b'{}')
        def do_POST(self):
            expected_origin = 'http://127.0.0.1:' + str(self.server.server_port)
            if self.path != '/selection' or self.headers.get('Origin') != expected_origin or self.headers.get('X-Picker-Token') != token:
                return self.reply(403, b'{}')
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length < 12000:
                    raise ValueError('Invalid request size')
                payload = json.loads(self.rfile.read(length))
                selection = {key: payload[key] for key in ('mode', 'catalog_id', 'user_description') if key in payload}
                validate_selection(selection, catalog)
                selection.update(confirmation='user_dialog', evidence='Confirmed in the image style picker')
                target = job / 'style/selection.json'
                temporary = target.with_suffix('.tmp')
                temporary.write_text(json.dumps(selection, ensure_ascii=False, indent=2))
                temporary.replace(target)
                self.reply(200, json.dumps({'saved': True, 'path': str(target)}).encode())
                print('STYLE_SELECTION_SAVED ' + json.dumps(selection, ensure_ascii=False), flush=True)
            except (ValueError, TypeError, KeyError) as error:
                self.reply(400, json.dumps({'error': str(error)}).encode())
    server = HTTPServer(('127.0.0.1', port), Handler)
    server.timeout = .5
    print('STYLE_PICKER_URL=http://127.0.0.1:' + str(server.server_port), flush=True)
    deadline = time.monotonic() + 60
    while not (job / 'style/selection.json').is_file() and time.monotonic() < deadline:
        server.handle_request()
    if not (job / 'style/selection.json').is_file():
        selection = {'mode':'inferred', 'confirmation':'timeout_default',
                     'evidence':'Optional style choice waited 60 seconds; model infers style from content (list is non-binding)'}
        (job / 'style/selection.json').write_text(json.dumps(selection, ensure_ascii=False, indent=2))
    server.server_close()
    print('STYLE_SELECTION_SAVED=' + str(job / 'style/selection.json'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'serve', 'verify'])
    parser.add_argument('job', type=Path)
    parser.add_argument('--catalog-source', type=Path, default=PACKAGE / 'dependencies/hand-drawn-styles')
    parser.add_argument('--port', type=int, default=0)
    arguments = parser.parse_args()
    if arguments.command == 'serve':
        serve_picker(arguments.job, arguments.catalog_source, arguments.port)
    elif arguments.command == 'prepare':
        print(json.dumps(prepare_catalog(arguments.job, arguments.catalog_source), ensure_ascii=False))
    else:
        print(json.dumps(verify_style(arguments.job), ensure_ascii=False))
