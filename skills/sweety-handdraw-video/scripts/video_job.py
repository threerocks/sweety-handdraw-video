"""Compile content-designed pages against measured narration, not fixed scene lengths."""
from pathlib import Path
import copy
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import wave
import numpy as np
from PIL import Image
import stroke_assets
import line_drawing
from hand_assets import prepare_hand
from art_direction import verify_art_direction
from caption_assets import build_caption, caption_position, caption_safe_violations, resolve_cues
from watermark_assets import prepare_watermark
from narration_master import load_master
from audio_contract import resolve_audio_layers, validate_ending
from fonts import check_coverage, select as select_font
from storyboard_lint import anchor_beat, check_build, group_beats
from timeline_contract import validate_page_motion

PACKAGE = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 48000
ENDING_SETTLE = 2.5       # seconds the finished last picture holds after the end hold (3.0)
COVER_VOICE_OVERLAP = .5  # seconds of the cover fade the voice already speaks over (3.0; 0 = the 2.x timing)


def load_manifest(job_directory):
    job_directory = Path(job_directory).resolve()
    manifest = json.loads((job_directory / 'job.json').read_text())
    if manifest.get('version') != 1 or not manifest.get('pages'):
        raise ValueError('job.json requires version=1 and at least one page')
    return job_directory, manifest


def resolve_input(job_directory, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (job_directory / path).resolve()


def find_font(manifest):
    """Handwriting font: job.json font.path/index, else the first covering font of the platform chain (see fonts.py)."""
    return select_font('handwriting', '', manifest.get('font'))


def fix_commands(report):
    """One actionable line per missing item; image generation is a host capability scripts cannot detect."""
    mac = sys.platform == 'darwin'
    fixes = []
    if not all(report['tools'].values()):
        fixes.append('FFmpeg: ' + ('brew install ffmpeg' if mac else 'sudo apt install ffmpeg  (dnf install ffmpeg-free / pacman -S ffmpeg)') + '; or bash scripts/setup.sh')
    if 'font_error' in report:
        fixes.append('Fonts: ' + ('install Hannotate SC in Font Book (or set font.path in job.json)' if mac else
                     'sudo apt install fonts-noto-cjk fontconfig  (dnf install google-noto-sans-cjk-fonts / pacman -S noto-fonts-cjk)')
                     + '; run python3 scripts/fonts.py JOB to see every font tried')
    if report['missing_inputs']:
        fixes.append('Inputs: create or fix these job paths: ' + ', '.join(str(value) for value in report['missing_inputs']))
    if 'art_direction_error' in report:
        fixes.append('Art direction: record and bind the brief/prompt/review for every image (references/style-and-drawing.md)')
    if not report['narration_files_available']:
        fixes.append('Narration: start the local VoiceStudio service (http://127.0.0.1:3900) then python3 scripts/voice_studio.py JOB; '
                     'or set voxcpm.model_path and run python3 scripts/voxcpm_voice.py JOB with the MLX Python; '
                     'or, only with the user\'s own key, volcengine_tts.py / gemini_tts.py; '
                     'or supply narration.source + alignment (references/continuous-narration.md)')
    fixes.append('Images: Codex has built-in image generation; Claude Code needs an image-generation tool/MCP or user-provided images. '
                 'Scripts cannot detect this; without it, stop instead of producing a text-only film.')
    return fixes


def inspect_environment(job_directory):
    root, manifest = load_manifest(job_directory)
    report = {'python': sys.version.split()[0], 'tools': {name: shutil.which(name) for name in ('ffmpeg', 'ffprobe')}, 'missing_inputs': []}
    try:
        report['fonts'] = check_coverage(copy.deepcopy(manifest))
        report['font'] = (report['fonts']['handwriting']['path'], report['fonts']['handwriting']['index'])
    except (ValueError, FileNotFoundError) as error:
        report['font_error'] = str(error)
    inputs = [manifest.get('cover', {}).get('source')]
    audio = [manifest['narration'].get(key) for key in ('source', 'alignment')] if manifest.get('narration') else []
    for page in manifest['pages']:
        for beat in page['beats']:
            if not manifest.get('narration'):
                audio.append(beat.get('audio'))
            inputs.extend(action.get('source') for action in beat.get('actions',[]) if action['type'] == 'illustration')
    for page in manifest['pages']:
        for group in page.get('visual_groups', []):
            inputs.extend(action.get('source') for action in group['actions'] if action['type']=='illustration')
    inputs.extend(page.get('background') for page in manifest['pages'] if page.get('background'))
    if manifest.get('hand', {}).get('source'):
        inputs.append(manifest['hand']['source'])
    for value in inputs + audio:
        if not value or not resolve_input(root, value).is_file():
            report['missing_inputs'].append(value)
    visuals_missing = any(not value or not resolve_input(root, value).is_file() for value in inputs)
    try:
        report['art_direction']=verify_art_direction(root,manifest)
    except (ValueError,FileNotFoundError,KeyError) as error:
        report['art_direction_error']=str(error)
    if not manifest.get('runtime_fixture'):
        from voice_backends import voice_backends_report
        backends = voice_backends_report(manifest.get('voice_studio', {}).get('url', 'http://127.0.0.1:3900'))
        model = manifest.get('voxcpm', {}).get('model_path')
        if model and (Path(model).expanduser()/'config.json').is_file():
            backends['local_voxcpm2'].update(model=str(Path(model).expanduser()), available=bool(backends['local_voxcpm2']['python']))
        report['voice_backends'] = backends
    report['narration_files_available'] = all(beat.get('audio') and resolve_input(root,beat['audio']).is_file()
        for page in manifest['pages'] for beat in page['beats'])
    if manifest.get('narration'):
        try:
            load_master(root, manifest, root/'work/audio/doctor-master.wav')
            report['narration_files_available'] = True
        except (ValueError, FileNotFoundError, KeyError) as error:
            report['narration_error'] = str(error)
            report['narration_files_available'] = False
    elif not manifest.get('runtime_fixture'):
        report['narration_error'] = 'Production requires a complete narration master and alignment'
        report['narration_files_available'] = False
    renderable = all(report['tools'].values()) and not visuals_missing and 'font_error' not in report and 'art_direction_error' not in report
    report['ready'] = renderable and not report['missing_inputs'] and report['narration_files_available']
    report['capabilities'] = {
        'render': renderable,
        'generate_narration': bool(report.get('voice_backends') and (report['voice_backends']['local_voxcpm2']['available'] or report['voice_backends']['voice_studio']['reachable']
                                                                     or any(report['voice_backends']['cloud_optional'].values()))),
        'narration_ready': report['narration_files_available'],
        'generate_images': 'host-dependent: not detectable by scripts'}
    report['fixes'] = fix_commands(report)
    return report


def save_wave(path, samples):
    with wave.open(str(path), 'wb') as stream:
        stream.setparams((1, 2, SAMPLE_RATE, 0, 'NONE', 'not compressed'))
        stream.writeframes(samples.astype('<i2').tobytes())


def read_narration(source, destination, pace, trim):
    if not .5 <= pace <= 2:
        raise ValueError('audio.pace must be within 0.5..2.0')
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(source), '-af', ('anull' if pace == 1 else f'atempo={pace}'),
                    '-ar', str(SAMPLE_RATE), '-ac', '1', '-c:a', 'pcm_s16le', str(destination)], check=True)
    with wave.open(str(destination), 'rb') as stream:
        samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype='<i2')
    block = round(.02 * SAMPLE_RATE)
    blocks = np.pad(samples.astype(float) / 32768, (0, (-len(samples)) % block)).reshape(-1, block)
    rms = np.sqrt(np.mean(blocks**2, axis=1))
    active = np.flatnonzero(rms > max(.001, rms.max() * .02))
    if not len(active):
        raise ValueError(f'Narration has no audible samples: {source}')
    first = max(0, round((active[0] * .02 - .08) * SAMPLE_RATE)) if trim else 0
    last = min(len(samples), round(((active[-1] + 1) * .02 + .10) * SAMPLE_RATE)) if trim else len(samples)
    return samples[first:last], {'source_seconds': len(samples) / SAMPLE_RATE, 'trim_start_seconds': first / SAMPLE_RATE,
                                 'trim_end_seconds': last / SAMPLE_RATE, 'used_seconds': (last - first) / SAMPLE_RATE}


def subtitle_time(seconds):
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3600000)
    minutes, milliseconds = divmod(milliseconds, 60000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}'


def anchor_point(spec):
    """Rough center of an action in screen pixels (pen travel between elements)."""
    if spec.get('points'):
        return np.asarray(spec['points'], float).mean(0)
    return np.array([float(spec['x']) + float(spec.get('width', spec.get('size', 60))) / 2,
                     float(spec['y']) + float(spec.get('height', spec.get('size', 60))) / 2])


def expand_actions(specs):
    """Composite actions become the drawn primitives a presenter would draw in order. bar_chart: the axes as one pen
    line, then each bar colored up from the base with its value written above it, category labels last."""
    result = []
    for spec in specs:
        if spec['type'] != 'bar_chart':
            result.append(spec)
            continue
        values = [float(value) for value in spec['values']]
        labels = [str(label) for label in spec.get('labels', [])]
        if not values or (labels and len(labels) != len(values)) or min(values) < 0:
            raise ValueError('bar_chart needs nonnegative values and one label per value')
        x, y, width, height = (float(spec[key]) for key in ('x', 'y', 'width', 'height'))
        label_size = int(spec.get('label_size', max(24, round(height * .08))))
        value_size = int(spec.get('value_size', label_size))
        top = max(values + [float(spec.get('max', 0))]) or 1.
        plot_top, plot_bottom = y + value_size * 1.4, y + height - (label_size * 1.5 if labels else 0)
        slot = width / len(values)
        bar_width = slot * float(spec.get('bar_ratio', .6))
        weight = float(spec.get('weight', 4))
        common = {key: spec[key] for key in ('visual', 'z') if key in spec}
        first = {'from_beat': spec['from_beat']} if spec.get('from_beat') is not None else {}
        result.append({'type': 'line', 'points': [[x, plot_top - value_size * .3], [x, plot_bottom], [x + width, plot_bottom]],
                       'line_width': spec.get('axis_width', 4), 'color': spec.get('axis_color', '#302C26'), 'weight': weight * .15, **common, **first})
        color = spec.get('color', '#E9784E')
        colors = spec.get('colors') or [color] * len(values)
        fmt = spec.get('value_format', '{:g}')
        for index, value in enumerate(values):
            bar_height = max(4., (plot_bottom - plot_top) * value / top)
            left = x + slot * index + (slot - bar_width) / 2
            result.append({'type': 'marker', 'style': 'bar', 'x': round(left), 'y': round(plot_bottom - bar_height - 2), 'width': max(8, round(bar_width)),
                           'height': max(6, round(bar_height)), 'color': colors[index], 'z': 20, 'weight': weight * .5 / len(values), **common})
            if spec.get('show_values', True):
                text = fmt.format(value) + str(spec.get('unit', ''))
                result.append({'type': 'text', 'text': text, 'size': value_size, 'x': round(left + bar_width / 2 - value_size * .3 * len(text)),
                               'y': round(plot_bottom - bar_height - value_size * 1.35), 'color': spec.get('value_color', '#302C26'),
                               'weight': weight * .2 / len(values), **common})
        for index, label in enumerate(labels):
            center = x + slot * (index + .5)
            result.append({'type': 'text', 'text': label, 'size': label_size, 'x': round(center - label_size * .5 * len(label)),
                           'y': round(plot_bottom + label_size * .25), 'color': spec.get('label_color', '#302C26'),
                           'weight': weight * .15 / len(labels), **common})
    return result


# when a form adds a row, an item, a node, a tick (durations are not moments); a custom script names its own as `moments`
MOMENT_KEYS = {'ms', 'mark_ms', 'row_ms', 'moments'}


def scheduled_moments(params):
    """Milliseconds (from the layer's start) at which a motion layer adds something inside itself."""
    found = []
    def walk(value, key=None):
        if isinstance(value, dict):
            for name, item in value.items():
                walk(item, name)
        elif isinstance(value, list):
            for item in value:
                walk(item, key)
        elif key in MOMENT_KEYS and isinstance(value, (int, float)) and math.isfinite(value):
            found.append(float(value))
    walk(params)
    return found


GESTURE_RECIPES = ('rough_annotate', 'sketch_shapes', 'rough_chart', 'callout', 'number_ticker', 'progress_ring', 'kinetic_title', 'morph',
                   'map_route', 'burst', 'formula', 'physics', 'spin3d')


def gesture_path(spec, event):
    """Screen points the pen traces, without ink, when a board-space overlay appears: around a circled thing, under a
    chart or a title, a small loop at a number. The overlay is the pen's doing instead of appearing on its own."""
    x, y, width, height = event['x'], event['y'], event['width'], event['height']
    params, recipe = spec.get('params', {}), spec['recipe']
    cx, cy = x + width / 2, y + height / 2
    loop = lambda rx, ry, turns=1.: [[cx + rx * math.cos(-math.pi / 2 + 2 * math.pi * turns * k / 24), cy + ry * math.sin(-math.pi / 2 + 2 * math.pi * turns * k / 24)]
                                     for k in range(25)]
    if recipe == 'rough_annotate':
        kind = params.get('type', 'circle')
        if kind in ('underline', 'strike-through', 'crossed-off', 'highlight'):
            level = y + height * (.88 if kind == 'underline' else .5)
            return [[x + .08 * width, level], [x + .92 * width, level]]
        if kind in ('box', 'bracket'):
            return [[x + .1 * width, y + .1 * height], [x + .9 * width, y + .1 * height], [x + .9 * width, y + .9 * height],
                    [x + .1 * width, y + .9 * height], [x + .1 * width, y + .12 * height]]
        return loop(.42 * width, .42 * height)
    if recipe == 'sketch_shapes' and params.get('shapes'):
        shape = params['shapes'][0]
        if shape.get('points'):
            return [[x + point[0], y + point[1]] for point in shape['points']]
        left, top, right, bottom = shape.get('box', [0, 0, width, height])
        return [[x + left, y + top], [x + right, y + top], [x + right, y + bottom], [x + left, y + bottom]]
    if recipe in ('rough_chart', 'kinetic_title'):
        level = y + height * (.9 if recipe == 'rough_chart' else .78)
        return [[x + .1 * width, level], [x + .9 * width, level]]
    if recipe == 'formula':  # writing along the line: one small up-and-down per glyph, left to right
        count = max(2, formula_glyphs(params.get('tex', '')))
        return [[x + width * (.1 + .8 * k / (3 * count)), cy + .07 * height * math.sin(math.pi * k / 1.5)] for k in range(3 * count + 1)]
    if recipe == 'spin3d':  # an orbit around the object, as if setting it turning
        return loop(.36 * width, .12 * height)
    if recipe == 'physics' and params.get('scene', 'pour') == 'pour':  # where the props pour in, then down into the container
        return [[cx - .12 * width, y + .1 * height], [cx + .12 * width, y + .1 * height], [cx, y + .45 * height]]
    return loop(.18 * min(width, height), .18 * min(width, height))


def formula_glyphs(tex):
    """About how many marks MathJax draws for a TeX string (a command is one mark; braces, scripts and spaces none)."""
    return len(re.sub(r'[{}^_\s&]', '', re.sub(r'\\[A-Za-z]+', 'X', str(tex))))


def gesture_seconds(spec, seconds):
    """How long the pen traces an overlay: the time a formula takes to write (see recipes.js formula), else a short pass."""
    if spec['recipe'] == 'formula':
        params = spec.get('params', {})
        count = max(1, formula_glyphs(params.get('tex', '')))
        slot = params.get('glyph_ms') or float(np.clip((seconds * 1000 - params.get('delay', 0) - 500) / count, 90, 240))
        return min(seconds, (params.get('delay', 0) + slot * (count + .5)) / 1000)
    return float(np.clip(.6 * seconds, .5, 1.3))


ENGINE_SHA256 = None


def page_scale(page_spec):
    """Board resolution of a page: the largest camera zoom on it, so every close-up is a sharp crop (1..3)."""
    zooms = [float(key.get('zoom', 1)) for key in page_spec.get('camera') or []]
    scale = float((page_spec.get('board') or {}).get('scale', max([1.] + zooms)))
    if not 1 <= scale <= 3:
        raise ValueError('Page board scale must be within 1..3')
    return round(scale, 3)


def moment(spec, page, beats, default):
    """Film time named by "at": "beat:<id>" (+ offset) on this page, seconds from the page start, or default."""
    at = spec.get('at')
    if at is None:
        return default + float(spec.get('offset', 0.))
    if isinstance(at, str):
        beat_id = at[5:] if at.startswith('beat:') else at
        if beat_id not in beats:
            raise ValueError(f'"at" names a sentence that is not on this page: {at}')
        return beats[beat_id]['start'] + float(spec.get('offset', .15))
    return page['start'] + float(at)


SCREEN_FURNITURE = ('kinetic_title', 'roadmap')  # screen layers ink never sits on: titles and the progress strip


def ink_on_screen_layer(page_spec):
    """True when drawn ink (handwriting, a picture) lands on a screen-space motion layer of the page."""
    layers = [item for item in page_spec.get('overlays', []) if item.get('space', 'board') == 'screen'
              and item.get('recipe') not in SCREEN_FURNITURE and all(key in item for key in ('x', 'y', 'width', 'height'))]
    if not layers:
        return False
    actions = [action for group in page_spec.get('visual_groups', []) for action in group.get('actions', [])]
    actions += [action for beat in page_spec.get('beats', []) for action in beat.get('actions', [])]
    for action in actions:
        if action.get('type') == 'text' and 'x' in action and 'y' in action:
            size = float(action.get('size', 48))
            box = {'x': float(action['x']), 'y': float(action['y']), 'width': size * max(1, len(str(action.get('text', '')))), 'height': size}
        elif action.get('type') == 'illustration' and all(key in action for key in ('x', 'y', 'width', 'height')):
            box = action
        else:
            continue
        if any(box_overlap(box, layer) * min(box['width'] * box['height'], layer['width'] * layer['height'])
               >= .3 * box['width'] * box['height'] for layer in layers):
            return True
    return False


def asset_ink_box(work, name):
    """(left, top, right, bottom) of an asset's ink in its own pixels, or None."""
    path = Path(work) / 'assets' / f'{name}.png'
    if not path.is_file():
        return None
    pixels = np.asarray(Image.open(path).convert('RGBA'))
    ys, xs = np.nonzero((pixels[..., 3] > 32) & (pixels[..., :3].min(axis=2) < 232))
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1) if len(xs) else None


def guard_screen_titles(keys, page, page_spec, assets, work, width, height, gap=8):
    """Screen-space titles stay put while the auto camera pushes in, so a push could carry the picture into the title
    (the deposit film's last page: hair into 只讲, the window frame through 两件事). Back each push off until the page's
    ink clears every screen title; pages whose ink meets a title even unzoomed are left to director_lint."""
    from director_lint import title_box
    titles = [title_box(item) for item in page_spec.get('overlays', []) if item.get('recipe') == 'kinetic_title'
              and item.get('space', 'board') == 'screen' and all(key in item for key in ('x', 'y', 'width', 'height'))]
    if not titles:
        return keys
    boxes = []
    for event in page['events']:
        if event['caption'] or event['asset'] not in assets:
            continue
        ink = asset_ink_box(work, event['asset'])
        if ink:
            scale = float(event.get('scale', 1.))
            boxes.append([event['x'] + ink[0] / scale, event['y'] + ink[1] / scale, event['x'] + ink[2] / scale, event['y'] + ink[3] / scale])

    def clashes(center, zoom):
        cx = min(max(center[0], width / (2 * zoom)), width - width / (2 * zoom))
        cy = min(max(center[1], height / (2 * zoom)), height - height / (2 * zoom))
        for x0, y0, x1, y1 in boxes:
            left, top = (x0 - cx) * zoom + width / 2, (y0 - cy) * zoom + height / 2
            right, bottom = (x1 - cx) * zoom + width / 2, (y1 - cy) * zoom + height / 2
            if any(left < tx + tw + gap and tx - gap < right and top < ty + th + gap and ty - gap < bottom for tx, ty, tw, th in titles):
                return True
        return False

    if not boxes or clashes([width / 2, height / 2], 1.):
        return keys
    for key in keys:
        zoom = float(key['zoom'])
        if zoom > 1 and clashes(key['center'], zoom):
            while zoom > 1 and clashes(key['center'], zoom):
                zoom = max(1., zoom - .01)
            key['zoom'] = round(zoom, 3)
            page['auto'] = sorted(set(page.get('auto', [])) | {'camera_backed_off_for_title'})
    return keys


TAKE_TURNS_OVERLAP = .12  # share of the smaller title box two layers must share before one gives way (a 5 px graze is not a clash)


def box_overlap(first, second):
    """Intersection area of two x/y/width/height boxes over the smaller box's area (0 when they only touch)."""
    width = min(first['x'] + first['width'], second['x'] + second['width']) - max(first['x'], second['x'])
    height = min(first['y'] + first['height'], second['y'] + second['height']) - max(first['y'], second['y'])
    if width <= 0 or height <= 0:
        return 0.
    return width * height / max(1., min(first['width'] * first['height'], second['width'] * second['height']))


def take_turns(titles, page):
    """Titles that share a spot on screen take turns: one still showing when the next arrives finishes leaving first
    (it fades out in the gap, or as late as its own entrance allows), so two titles never read as one line."""
    titles = sorted(titles, key=lambda event: event['start'])
    for number, earlier in enumerate(titles):
        later = next((event for event in titles[number + 1:] if box_overlap(earlier, event) >= TAKE_TURNS_OVERLAP), None)
        leaving = earlier.get('exit') or {'style': 'fade', 'start': page['end'], 'end': page['end']}
        if later is None or leaving['end'] <= later['start'] + 1e-6:
            continue
        seconds = max(leaving['end'] - leaving['start'], .3)
        start = max(earlier['end'], later['start'] - seconds)
        earlier['exit'] = {'style': leaving['style'], 'start': round(start, 3), 'end': round(start + seconds, 3)}


def parallax_depth(spec):
    """An element's parallax depth: -1 (far, moves with the camera) .. 1 (near, moves twice as far as the page)."""
    depth = spec.get('depth')
    if isinstance(depth, bool) or not isinstance(depth, (int, float)) or not -1 <= depth <= 1:
        raise ValueError('Parallax "depth" is a number within -1 (far) .. 1 (near); 0 or absent is the page itself')
    return float(depth)


def resolve_motions(spec, event, page, beats, parts=()):
    """Motion presets of an element in film time (motion.py): `motion` entries and legacy `emphasis` ones.
    Emphasis and puppet moves wait for the last stroke; idle loops start when the element is finished (or later at
    "at") and repeat until the page turns; exits play at "at"."""
    import motion
    effects = []
    for effect in list(spec.get('emphasis') or []) + list(spec.get('motion') or []):
        kind = effect.get('type')
        if kind not in motion.TYPES:
            raise ValueError(f'Motion type must be one of {sorted(motion.TYPES)}')
        start = moment(effect, page, beats, event['end'] + .05)
        seconds = float(effect.get('seconds', motion.TYPES[kind]))
        if not (math.isfinite(start) and .2 <= seconds <= 8):
            raise ValueError('Motion seconds must be within 0.2..8')
        if kind not in motion.EXIT and start < event['end'] - 1e-6:
            # The element is still being drawn when its sentence comes: play the move right after the last stroke.
            start = event['end'] + .1
        entry = {'type': kind, 'start': round(start, 3), 'end': round(start + seconds, 3), 'strength': float(effect.get('strength', 1.))}
        if kind in motion.PUPPET:
            if effect.get('part') not in parts:
                raise ValueError(f'{kind} needs "part": one of this picture\'s parts {list(parts)}')
            entry.update(part=parts.index(effect['part']), side=effect.get('side', 'right'))
        effects.append(entry)
    return effects


def drawn_seconds(metadata, seconds):
    """How long an asset actually draws: pictures and stroke-order text stop at video pace, others use their window."""
    if metadata.get('type') == 'illustration':
        return float(metadata['seconds'])
    if metadata.get('handwriting', {}).get('mode') == 'stroke_order':
        return float(metadata['handwriting']['seconds'])
    if metadata.get('type') == 'line':
        # A confident pen line: about 1100 px/s plus a short lift per barb, never stretched over a long window.
        ink = [segment for segment in metadata['segments'] if segment['kind'] == 'ink']
        return min(seconds, max(.3, sum(segment['lengths'][-1] for segment in ink) / 1100 + .08 * len(ink)))
    if metadata.get('type') == 'counter':
        return min(seconds, 1.4)
    return seconds


class FilmBuilder:
    def __init__(self, job_directory):
        self.root, self.manifest = load_manifest(job_directory)
        self.art_direction = verify_art_direction(self.root, self.manifest)
        self.work = self.root / 'work'
        self.output = self.root / 'outputs'
        for directory in (self.work / 'assets', self.work / 'audio', self.work / 'review', self.output):
            directory.mkdir(parents=True, exist_ok=True)
        for stale in (self.work / 'assets').glob('element-*'):
            stale.unlink()  # assets are renumbered every build; a leftover layer file would attach to whatever now has its name
        canvas = self.manifest.get('canvas', {})
        if not canvas.get('width') or not canvas.get('height'):
            from aspect_ratio import resolve_canvas, DEFAULT_RATIO
            resolved = resolve_canvas(canvas.get('aspect_ratio', DEFAULT_RATIO), canvas.get('width'), canvas.get('height'), canvas.get('fps', 30))
            self.width, self.height, self.fps = resolved['width'], resolved['height'], resolved['fps']
            self.aspect_ratio = resolved['aspect_ratio']
        else:
            self.width = int(canvas['width'])
            self.height = int(canvas['height'])
            self.fps = int(canvas.get('fps', 30))
            self.aspect_ratio = canvas.get('aspect_ratio')
        if self.width < 320 or self.height < 240 or self.width % 2 or self.height % 2 or not 1 <= self.fps <= 60:
            raise ValueError('Canvas requires even width/height >=320x240 and fps within 1..60')
        self.fonts = check_coverage(self.manifest)  # every rendered character must have a glyph; pins role fonts
        self.font_path, self.font_index = find_font(self.manifest)
        stroke_assets.configure_assets(self.work / 'assets', self.font_path, self.font_index, self.fps)
        self.assets = {}
        self.next_asset = 0
        self.seen_beat_ids = set()
        self.caption_warnings = []
        import direction
        self.direction = direction.options(self.manifest)

    @staticmethod
    def action_weight(action):
        weight = float(action.get('weight', 3 if action['type'] == 'illustration' else 1))
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError('Action weight must be finite and positive')
        return weight

    @staticmethod
    def minimum_seconds(action):
        # A chart bar, a pen line and a counting number need a moment to read as motion, not a flash.
        if action['type'] == 'marker' and action.get('style') == 'bar':
            return .5
        if action['type'] in ('line', 'arrow'):
            return .3
        if action['type'] == 'counter':
            return .8
        # Readers need about 0.3 s per hanzi to see the stroke order; Latin and punctuation write faster.
        if action['type'] != 'text':
            return 0.
        return sum(.3 if '\u3400' <= character <= '\u9fff' else .12 for character in str(action['text']))

    @staticmethod
    def travel_gap(first, second, minimum):
        # Lift-and-move time grows with distance so the pen never teleports between elements.
        anchors = [anchor_point(spec) for spec in (first, second)]
        return float(np.clip(np.linalg.norm(anchors[1] - anchors[0]) / 1600, max(minimum, .12), .45))

    # Fields that only place or time an action on the page; they do not change the asset that is built.
    PLACEMENT_KEYS = ('x', 'y', 'visual', 'weight', 'from_beat', 'z', 'motion', 'emphasis', 'pen', 'depth')

    def asset_key(self, action, seconds, scale):
        """Content key of one built element: the action without its placement, its window, board scale, the bytes of
        every input file it reads, the font and the engine code. Same key, same pixels and time maps."""
        global ENGINE_SHA256
        if ENGINE_SHA256 is None:
            digest = hashlib.sha256()
            for path in sorted(Path(__file__).resolve().parent.glob('*.py')):
                if not path.name.startswith('test_'):
                    digest.update(path.name.encode() + path.read_bytes())
            ENGINE_SHA256 = digest.hexdigest()
        files = {}
        for key in ('source', 'outline_mask', 'region_map'):
            if action.get(key):
                path = resolve_input(self.root, action[key])
                files[key] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        spec = {key: value for key, value in action.items() if key not in self.PLACEMENT_KEYS}
        if action['type'] in ('line', 'arrow'):
            spec['offset'] = [action.get('x'), action.get('y')]  # a line's points are absolute: they are its shape
        payload = {'spec': spec, 'seconds': None if seconds is None else round(float(seconds), 4), 'scale': round(float(scale), 4),
                   'files': files, 'drawing': self.manifest.get('drawing', {}), 'font': [str(self.font_path), self.font_index],
                   'fps': self.fps, 'engine': ENGINE_SHA256}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:24]

    def create_asset(self, action, seconds=None, scale=1.):
        """Build one element, or reuse an identical build from work/asset-cache: changing a camera move, an overlay or a
        transition rebuilds nothing. Returns the page event."""
        key = self.asset_key(action, seconds, scale)
        cache = self.work / 'asset-cache' / key
        name = f'element-{self.next_asset + 1:04d}'
        directory = self.work / 'assets'
        if (cache / 'asset.json').is_file():
            self.next_asset += 1
            for path in cache.iterdir():
                if path.name != 'asset.json':
                    shutil.copyfile(path, directory / path.name.replace('ASSET', name))
            metadata = json.loads((cache / 'asset.json').read_text())
            metadata['name'] = name
            (directory / f'{name}.json').write_text(json.dumps(metadata))
            self.cache_hits = getattr(self, 'cache_hits', 0) + 1
            return self.place_asset(action, metadata, name, scale)
        event = self.build_asset(action, seconds, scale)
        cache.mkdir(parents=True, exist_ok=True)
        for path in list(directory.glob(f'{name}.*')) + list(directory.glob(f'{name}-*.png')):
            if path.suffix != '.json':
                shutil.copyfile(path, cache / path.name.replace(name, 'ASSET'))
        (cache / 'asset.json').write_text(json.dumps(self.assets[name]))
        return event

    def build_asset(self, action, seconds=None, scale=1.):
        """Build one element at its page's board resolution (scale > 1 when the camera zooms into the page);
        positions stay in screen units."""
        self.next_asset += 1
        name = f'element-{self.next_asset:04d}'
        kind = action['type']
        x, y = action.get('x'), action.get('y')
        if kind == 'text':
            text = str(action['text'])
            if not text.strip() or '\n' in text:
                raise ValueError('Each text action must contain one nonempty line')
            metadata = stroke_assets.build_text(name, text, round(int(action.get('size', 60)) * scale), action.get('color', '#302C26'), seconds=seconds,
                                                stroke_order=bool(action.get('stroke_order', True)))
        elif kind == 'marker':
            line_width = action.get('line_width')
            metadata = stroke_assets.build_brush(name, round(int(action['width']) * scale), round(int(action['height']) * scale),
                action.get('color', '#EAC958'), action.get('style', 'highlight'), line_width * scale if line_width else None, seconds=seconds)
        elif kind == 'illustration':
            source = resolve_input(self.root, action['source'])
            optional = lambda key: resolve_input(self.root, action[key]) if action.get(key) else None
            drawing = {**self.manifest.get('drawing', {}), **action.get('drawing', {})}
            if drawing.get('mode') == 'reveal' and self.art_direction.get('style', {}).get('drawing_mode') == 'draw':
                raise ValueError(f'{action["source"]}: this film is drawn; one picture cannot switch to reveal. Simplify or regenerate it so it can be drawn')
            # Plan strokes and timing at the layout size (capped at the plan side, so a big picture is not more lines to
            # draw); render the pixels at the full size on the page's board resolution.
            plan_width, plan_height, enlarge = line_drawing.plan_box(int(action['width']), int(action['height']), drawing)
            metadata = stroke_assets.build_illustration(name, source, plan_width, plan_height,
                action.get('crop'), action.get('parts'), optional('outline_mask'), seconds=seconds, region_map_path=optional('region_map'),
                part_regions=action.get('part_regions'), lineart_exception=action.get('lineart_exception'), drawing=drawing)
            if scale * enlarge > 1.001:
                metadata = stroke_assets.render_at(name, source, scale * enlarge, action.get('crop'),
                                                   None if drawing.get('mode') == 'reveal' else optional('outline_mask'))
        elif kind in ('line', 'arrow'):
            points = np.asarray(action['points'], float)
            if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
                raise ValueError('A line needs "points": [[x, y], [x, y], ...] in screen pixels')
            metadata = stroke_assets.build_line(name, points * scale, float(action.get('line_width', 5)) * scale, action.get('color', '#302C26'),
                                                action.get('head', 'end' if kind == 'arrow' else 'none'), float(action.get('curve', 0)),
                                                float(action.get('pressure', stroke_assets.PRESSURE)))
        elif kind == 'counter':
            metadata = stroke_assets.build_counter(name, action.get('from', 0), action['to'], self.font_path, round(int(action.get('size', 72)) * scale),
                                                   action.get('color', '#302C26'), action.get('decimals', 0), action.get('prefix', ''), action.get('suffix', ''),
                                                   self.font_index)
        else:
            raise ValueError(f'Unsupported action type: {kind}')
        return self.place_asset(action, metadata, name, scale)

    def place_asset(self, action, metadata, name, scale):
        """The page event of a built element: its board position, weight and layer."""
        kind = action['type']
        self.assets[name] = metadata
        x, y = action.get('x'), action.get('y')
        if kind in ('line', 'arrow'):
            x, y = metadata['offset'][0] / scale, metadata['offset'][1] / scale
        x, y = float(x), float(y)
        if kind not in ('line', 'arrow'):
            x, y = int(x), int(y)
        if not (0 <= x and 0 <= y and x + metadata['width'] / scale <= self.width + .5 and y + metadata['height'] / scale <= self.height + .5):
            raise ValueError(f'Element is outside canvas: {action}')
        event = {'asset': name, 'x': x, 'y': y, 'weight': self.action_weight(action),
                 'z': int(action.get('z', 10 if kind == 'marker' and action.get('style', 'highlight') == 'highlight' else 30)),
                 'role': kind}
        if scale != 1.:
            event['scale'] = scale
        if kind == 'counter':
            event['pen'] = False
        return event

    def overlay_events(self, page_spec, page, scale):
        """Web motion layers of a page (web_motion.py): rendered once to cached transparent frames, timed to its
        sentences. space "board" moves with the page camera, "screen" stays put (titles, lower thirds)."""
        import web_motion
        events = []
        beats = {beat['id']: beat for beat in page['beats']}
        for index, spec in enumerate(page_spec.get('overlays') or []):
            space = spec.get('space', 'board')
            if space not in ('board', 'screen'):
                raise ValueError('Overlay space is board or screen')
            x, y, width, height = (float(spec[key]) for key in ('x', 'y', 'width', 'height'))
            if x < 0 or y < 0 or x + width > self.width + .5 or y + height > self.height + .5:
                raise ValueError(f'Overlay {spec.get("recipe")} lies outside the canvas')
            seconds = float(spec.get('seconds', 1.5))
            start = moment(spec, page, beats, page['start'] + .5)
            board = scale if space == 'board' else 1.
            render = web_motion.render(spec['recipe'], spec.get('params', {}), width, height, seconds, self.work / 'web-motion',
                                       self.fps, board, int(spec.get('seed', 1 + index)))
            if render.get('clipped') and not spec.get('bleed') and not (spec['recipe'] == 'map_route' and spec.get('params', {}).get('projection') == 'flat'):  # a flat map's coastlines run off its frame by design
                raise ValueError(f'Overlay {spec["recipe"]} on page {page["index"] + 1} is cut off at the edge of its {width:g}x{height:g} box: '
                                 'enlarge width/height or shrink the content (titles shrink to fit automatically); set "bleed": true only '
                                 'for content meant to run off the edge')
            event = {'recipe': spec['recipe'], 'space': space, 'x': x, 'y': y, 'width': width, 'height': height, 'scale': board,
                     'start': round(start, 3), 'end': round(start + seconds, 3), 'frames': render['directory'], 'count': render['frames'],
                     'moments': sorted({round(start, 3)} | {round(start + value / 1000, 3) for value in scheduled_moments(spec.get('params', {}))
                                                            if 0 <= value <= seconds * 1000}),
                     'z': int(spec.get('z', 60)), **({'visual': spec['visual']} if spec.get('visual') else {})}
            if spec.get('depth'):
                if space != 'board':
                    raise ValueError('Only board overlays move with the camera, so only they take a parallax "depth"')
                event['depth'] = parallax_depth(spec)
            params = spec.get('params', {})
            if spec['recipe'] == 'burst':
                event['sound'] = 'chime'
            elif spec['recipe'] == 'kinetic_title' and params.get('style') == 'slam':
                event['sound'] = 'thump'
            if spec.get('exit'):
                exit_spec = spec['exit']
                exit_start = moment(exit_spec, page, beats, page['end'] - .5)
                event['exit'] = {'style': exit_spec.get('style', 'fade'), 'start': round(exit_start, 3),
                                 'end': round(exit_start + float(exit_spec.get('seconds', .35)), 3)}
            if event['end'] > page['end'] + 1e-6 or event['start'] < page['start'] - 1e-6:
                raise ValueError(f'Overlay {spec["recipe"]} must play within its page ({page["start"]:.2f}-{page["end"]:.2f}s)')
            if space == 'board' and spec['recipe'] in GESTURE_RECIPES and spec.get('pen', True):
                event['gesture'] = [[round(value, 1) for value in point] for point in gesture_path(spec, event)]
                event['gesture_seconds'] = gesture_seconds(spec, seconds)
            events.append(event)
        take_turns([event for event in events if event['recipe'] == 'kinetic_title'], page)
        return events

    def caption_events(self, cues, keywords=None):
        """Typeset captions placed above the phone's bottom UI (caption_assets.caption_position); a hand-set position
        that a platform would cover is reported, not silently kept."""
        options = self.manifest.get('captions', {})
        events = []
        for cue in cues:
            self.next_asset += 1
            name = f'element-{self.next_asset:04d}'
            metadata = build_caption(self.work / 'assets', name, cue['text'], options, self.width, self.height, keywords=keywords)
            self.assets[name] = metadata
            x, y = caption_position(metadata, options, self.width, self.height)
            for problem in caption_safe_violations((x, y, metadata['width'], metadata['height']), self.width, self.height):
                warning = f'captions: {problem}'
                if warning not in self.caption_warnings:
                    self.caption_warnings.append(warning)
            events.append({'asset': name, 'x': x, 'y': y, 'role': 'caption', 'caption': True, 'z': 91,
                           'start': cue['start'], 'end': cue['end'], 'hide': cue['end']})
        return events

    def build(self):
        options = self.manifest.get('audio', {})
        for name in ('sentence_pause_seconds','page_pause_seconds','end_hold_seconds'):
            value=float(options.get(name,0))
            if not math.isfinite(value) or value<0:
                raise ValueError('Audio pauses must be finite and nonnegative: '+name)
        cover_seconds = float(self.manifest.get('cover', {}).get('seconds', 1.0))
        if cover_seconds != 1.0:
            raise ValueError('Cover must last exactly 1 second: 0.5 seconds still, then 0.5 seconds fade')
        # The voice starts while the cover fades (0.5 s), so the first words come before the 1 s mark; the cover stays
        # 0.5 s still and fades over the first strokes instead of over blank paper.
        fixture = bool(self.manifest.get('runtime_fixture'))
        overlap = float(self.manifest.get('cover', {}).get('voice_overlap', 0. if fixture else COVER_VOICE_OVERLAP))
        if not 0 <= overlap <= .5:
            raise ValueError('cover.voice_overlap must be within 0..0.5 seconds (the cover fade)')
        voice_start = cover_seconds - overlap
        pieces = [np.zeros(round(voice_start * SAMPLE_RATE), dtype=np.int16)]
        # After the last word the finished picture stays a moment (user, 2026-11-02: "加停顿吧"): the film settles
        # instead of cutting at the final syllable. Voice-tail limits still apply to the narration master itself.
        settle = float(self.manifest.get('ending', {}).get('settle_seconds', 0. if fixture else ENDING_SETTLE))
        if not math.isfinite(settle) or not 0 <= settle <= 3.5:
            raise ValueError('ending.settle_seconds must be within 0..3.5 seconds')
        master_samples, master_beats, master_report = None, {}, None
        if self.manifest.get('narration') or not self.manifest.get('runtime_fixture'):
            master_samples, master_beats, master_report = load_master(self.root, self.manifest, self.work/'audio/master.wav')
        cursor = len(pieces[0]) / SAMPLE_RATE
        pages, cues, audio_report = [], [], []
        for page_index, page_spec in enumerate(self.manifest['pages']):
            if not page_spec.get('beats'):
                raise ValueError('Each page requires narration beats')
            page = {'index': page_index, 'start': cursor, 'beats': [], 'events': [],
                    'camera': page_spec.get('camera'), 'transition': page_spec.get('transition')}
            import direction
            auto_camera = self.direction['camera'] == 'auto' and not page_spec.get('camera') and not self.manifest.get('runtime_fixture')
            if auto_camera and ink_on_screen_layer(page_spec):
                # The pen writes on a screen-space layer (a coded certificate, a card): a moving board would slide the
                # writing off the card it belongs to (aunt-doctor film), so this page keeps a still camera.
                auto_camera = False
                page['auto'] = page.get('auto', []) + ['camera_held_for_screen_layer']
            scale = max(page_scale(page_spec), direction.AUTO_ZOOM) if auto_camera else page_scale(page_spec)
            if page_index and not page['transition'] and self.direction['transitions'] == 'auto' and not self.manifest.get('runtime_fixture'):
                page['transition'] = direction.auto_transition(page_index, page_spec, (pages[-1].get('transition') or {}).get('type'))
                page['auto'] = page.get('auto', []) + ['transition']
            if page_spec.get('handheld'):
                page['handheld'] = float(page_spec['handheld'])
            if scale != 1.:
                page['scale'] = scale
            if page_spec.get('background'):
                background_path = resolve_input(self.root, page_spec['background'])
                if not background_path.is_file():
                    raise ValueError(f'Page {page_index} background image is missing: {page_spec["background"]}')
                page['background'] = str(background_path)
            for beat_index, beat_spec in enumerate(page_spec['beats']):
                beat_id = beat_spec.get('id', f'p{page_index:02d}-b{beat_index:02d}')
                if not re.fullmatch(r'[A-Za-z0-9_-]+', beat_id) or beat_id in self.seen_beat_ids:
                    raise ValueError('Beat IDs must be unique safe filenames')
                self.seen_beat_ids.add(beat_id)
                if master_report:
                    aligned = master_beats[beat_id]
                    cursor = voice_start + aligned['start']
                    duration = aligned['end'] - aligned['start']
                    measurements = {'source_seconds': master_report['duration'], 'master_start': aligned['start'],
                                    'master_end': aligned['end'], 'used_seconds': duration}
                else:
                    samples, measurements = read_narration(resolve_input(self.root, beat_spec['audio']),
                        self.work / 'audio' / f'{beat_id}.wav', float(options.get('pace', 1.0)), bool(options.get('trim_silence', True)))
                    duration = len(samples) / SAMPLE_RATE
                beat = {'id': beat_id, 'start': cursor, 'end': cursor + duration, 'duration': duration, 'text': beat_spec['text']}
                if not str(beat['text']).strip():
                    raise ValueError('Narration text cannot be empty')
                if not beat_spec.get('actions') and not page_spec.get('visual_groups') and not page_spec.get('overlays'):
                    # A page may be explained entirely by its motion layers (a prism, a commit graph): the pen rests.
                    raise ValueError('Every beat needs actions, a page visual group or motion layers (overlays)')
                cue_source = aligned if master_report else beat_spec
                if cue_source.get('caption_cues'):
                    beat['caption_cues'] = cue_source['caption_cues']
                beat_cues = resolve_cues(beat)
                page['events'].extend(self.caption_events(beat_cues, beat_spec.get('caption_keywords')))
                page['beats'].append(beat)
                cues.extend(beat_cues)
                audio_report.append({'id': beat_id, 'source': master_report['source'] if master_report else str(resolve_input(self.root, beat_spec['audio'])), **measurements})
                if not master_report:
                    pieces.append(samples)
                    pause_samples = np.zeros(round(float(options.get('sentence_pause_seconds', .18)) * SAMPLE_RATE), dtype=np.int16)
                    pieces.append(pause_samples)
                    cursor += duration + len(pause_samples) / SAMPLE_RATE
            ids = [beat['id'] for beat in page['beats']]
            blocks = [(beat['start'] + .54, beat['end'] - max(.5, beat['duration'] * .13), .07, expand_actions(spec['actions']), beat['id'])
                      for spec, beat in zip(page_spec['beats'], page['beats']) if spec.get('actions')]
            for group in page_spec.get('visual_groups', []):
                covered = group_beats(ids, group)
                lead, hold = float(group.get('lead', .54)), float(group.get('hold', .6))
                if not math.isfinite(lead + hold) or lead < 0 or hold < .45:
                    raise ValueError('Visual group needs a nonnegative lead and at least 0.45s reading hold')
                if not group.get('actions'):
                    raise ValueError('Visual group has no actions')
                # A group keeps drawing across sentences; an action with from_beat starts when its sentence starts.
                start, end, current = page['beats'][ids.index(covered[0])]['start'] + lead, page['beats'][ids.index(covered[-1])]['end'] - hold, []
                for spec in expand_actions(group['actions']):
                    anchor = page['beats'][ids.index(anchor_beat(covered, ids, spec))]['start']
                    if current and spec.get('from_beat') is not None:
                        if anchor - .08 <= start:
                            raise ValueError('from_beat must name a sentence after the earlier actions of its visual group have started')
                        blocks.append((start, anchor - .08, .08, current, f'visual group {covered[0]}..{covered[-1]}'))
                        start, current = anchor, []
                    start = max(start, anchor)
                    current.append(spec)
                blocks.append((start, end, .08, current, f'visual group {covered[0]}..{covered[-1]}'))
            blocks.sort(key=lambda block: block[0])
            for (_, previous_end, _, _, previous), (start, _, _, _, label) in zip(blocks, blocks[1:]):
                if start < previous_end:
                    raise ValueError(f'{label} starts drawing while {previous} is still drawing (one pen); '
                                     'put both in one visual group and use "from_beat" to start an action with its sentence')
            for start, end, gap, specs, label in blocks:
                gaps = [self.travel_gap(first, second, gap) for first, second in zip(specs, specs[1:])] + [0.]
                floors = [self.minimum_seconds(spec) for spec in specs]
                available = end - start - sum(gaps)
                if available < .15 * len(specs):
                    raise ValueError(f'{label}: too many drawing actions for the measured narration; simplify or extend the window across beats with a visual group')
                if available - sum(floors) < .15 * len(specs):
                    # Readable writing time is preferred, not guaranteed: shrink it and say so.
                    print(f'WARNING {label}: text needs {sum(floors):.2f}s to show stroke order but only {available:.2f}s exist; '
                          'shorten the text or widen the window', file=sys.stderr)
                    floors = [floor * (available - .15 * len(specs)) / sum(floors) for floor in floors]
                weights = [self.action_weight(spec) for spec in specs]
                for index, (spec, floor, travel) in enumerate(zip(specs, floors, gaps)):
                    # Each action may take its weighted share of the time still left. Assets are built after scheduling,
                    # draw at video pace and hand unused time back: later actions may use it, the rest holds the picture.
                    left = end - start - sum(gaps[index:]) - sum(floors[index:])
                    seconds = floor + max(0., left) * weights[index] / sum(weights[index:])
                    action = self.create_asset(spec, seconds, scale)
                    seconds = min(seconds, drawn_seconds(self.assets[action['asset']], seconds))
                    action.update(start=start, end=start + seconds, caption=False, visual=spec.get('visual'))
                    if spec.get('depth'):
                        action['depth'] = parallax_depth(spec)
                    if spec['type'] == 'illustration':
                        action['image'] = [spec['source'], spec.get('crop')]
                    parts = [part['id'] for part in self.assets[action['asset']].get('parts', [])]
                    effects = resolve_motions(spec, action, page, {beat['id']: beat for beat in page['beats']}, parts)
                    if not effects and self.direction['motion'] == 'auto' and not self.manifest.get('runtime_fixture'):
                        effects = [dict(effect, auto=True) for effect in direction.auto_motions(action, self.direction['boil'])]
                    if effects:
                        action['motions'] = effects
                    page['events'].append(action)
                    start += seconds + travel
            page['speech_end'] = page['beats'][-1]['end']
            pause = float(options.get('page_pause_seconds', .24)) if page_index < len(self.manifest['pages']) - 1 else float(options.get('end_hold_seconds', .45)) + settle
            if pause < 0:
                raise ValueError('Audio pauses must be nonnegative')
            if master_report:
                if page_index + 1 < len(self.manifest['pages']):
                    next_id = self.manifest['pages'][page_index + 1]['beats'][0]['id']
                    cursor = voice_start + master_beats[next_id]['start']
                else:
                    cursor = voice_start + len(master_samples) / SAMPLE_RATE + pause
            else:
                pause_samples = np.zeros(round(pause * SAMPLE_RATE), dtype=np.int16)
                pieces.append(pause_samples)
                cursor += len(pause_samples) / SAMPLE_RATE
            page['end'] = cursor
            if auto_camera:
                page['last_drawing_end'] = max((event['end'] for event in page['events'] if not event['caption']), default=page['start'])
                keys = direction.auto_camera(page, self.assets, self.width, self.height, page_index)
                if keys:
                    keys = guard_screen_titles(keys, page, page_spec, self.assets, self.work, self.width, self.height)
                    for key in keys:
                        key['center'] = [min(max(key['center'][0], 0.), self.width), min(max(key['center'][1], 0.), self.height)]
                    page['camera'] = keys
                    page['auto'] = page.get('auto', []) + ['camera']
            # Numeric camera times are checked against the page length, known only now.
            validate_page_motion(page, {'width': self.width, 'height': self.height}, first=page_index == 0)
            for event in page['events']:
                if not event['caption']:
                    event['hide'] = cursor
                    import motion
                    # Automatic settles that would run past the page simply do not play.
                    event['motions'] = [effect for effect in event.get('motions', ())
                                        if not (effect.get('auto') and effect['end'] > cursor and effect['type'] not in motion.IDLE)]
                    if not event['motions']:
                        event.pop('motions')
                    for effect in event.get('motions', ()):
                        if effect['type'] in motion.IDLE:
                            continue  # loops simply stop when the page turns
                        if effect['end'] > cursor + 1e-6:
                            raise ValueError(f'Motion {effect["type"]} on {event["asset"]} runs past the end of its page')
            page['last_drawing_end'] = max((event['end'] for event in page['events'] if not event['caption']), default=page['start'])
            overlays = self.overlay_events(page_spec, page, scale)
            if overlays:
                page['overlays'] = overlays
                gestures, busy = [], [(event['start'], event['end']) for event in page['events'] if not event['caption'] and event.get('pen', True)]
                for overlay in overlays:
                    path = overlay.pop('gesture', None)
                    if not path:
                        continue
                    start = overlay['start']
                    end = min(overlay['end'], start + overlay.pop('gesture_seconds'), page['end'] - .2)
                    free = all(end + .25 <= first or start - .25 >= last for first, last in busy + [(g['start'], g['end']) for g in gestures])
                    if end - start >= .3 and free:
                        gestures.append({'start': round(start, 3), 'end': round(end, 3), 'points': path, 'recipe': overlay['recipe']})
                if gestures:
                    page['gestures'] = gestures
            pages.append(page)
        ending = self.manifest.get('ending', {})
        fade = float(ending.get('fade_seconds', 0))
        narration_end = voice_start + len(master_samples) / SAMPLE_RATE if master_report else pages[-1]['speech_end']
        hold = pages[-1]['end'] - narration_end
        if not math.isfinite(fade) or not 0 <= fade <= hold:
            raise ValueError('Ending fade must fit entirely after narration')
        caption_hold = float(ending.get('caption_hold_seconds', 0))
        if not math.isfinite(caption_hold) or not 0 <= caption_hold <= hold - fade + .001:
            raise ValueError('Final subtitle hold must end before the closing fade')
        if caption_hold or master_report:
            captions = [event for event in pages[-1]['events'] if event['caption']]
            final_caption_end = narration_end + caption_hold
            captions[-1]['end'] = final_caption_end
            captions[-1]['hide'] = final_caption_end
            cues[-1]['end'] = final_caption_end
        if master_report:
            pieces = [pieces[0], master_samples, np.zeros(round((float(options.get('end_hold_seconds', .45)) + settle) * SAMPLE_RATE), dtype=np.int16)]
        ending_review = validate_ending(self.manifest, master_samples, SAMPLE_RATE) if master_report and not self.manifest.get('runtime_fixture') else None
        narration = np.concatenate(pieces)
        save_wave(self.work / 'narration.wav', narration)
        # Check cue timing on the clean narration now, not after a render: a late cue costs only a re-alignment here.
        from align import caption_sync
        sync = caption_sync(narration, cues)
        if sync['status'] == 'FAIL' and not self.manifest.get('runtime_fixture'):
            raise ValueError('Captions start away from the speech; fix the alignment before rendering: ' + json.dumps(sync['outside_tolerance'], ensure_ascii=False))
        hand = self.manifest.get('hand', {})
        timeline = {'title': self.manifest['title'], 'source_notes': self.manifest.get('source_notes', []),
                    'canvas': {'width': self.width, 'height': self.height, 'fps': self.fps, **({'aspect_ratio': self.aspect_ratio} if getattr(self, 'aspect_ratio', None) else {})},
                    'paper': self.manifest.get('paper', '#FBF9F1'), 'cover_seconds': cover_seconds, 'voice_start_seconds': voice_start, 'ending': {'fade_seconds': fade, 'caption_hold_seconds': caption_hold, 'settle_seconds': settle},
                    'narration_master': master_report, 'caption_sync': sync,
                    'captions': {'presentation': 'burned', 'owner': 'caption_events'},
                    'audio_policy': resolve_audio_layers(self.root, self.manifest), 'ending_review': ending_review,
                    'duration': len(narration) / SAMPLE_RATE, 'pages': pages, 'cues': cues, 'audio_report': audio_report,
                    'assets': self.assets, 'font': {'path': self.font_path, 'index': self.font_index},
                    'hand': prepare_hand(hand, self.root, self.work, self.height, self.font_path, self.font_index),
                    'watermark': prepare_watermark(self.manifest.get('watermark'), self.work, self.width, self.height),
                    'art_direction': self.art_direction,
                    'manifest_sha256': hashlib.sha256((self.root / 'job.json').read_bytes()).hexdigest()}
        from subtitle_contract import caption_clearance, validate_subtitles
        validate_subtitles(timeline)
        caption_clearance(timeline, self.work / 'assets')
        # The pen never writes off screen: camera moves wait for the element being drawn, or a framing move is inserted.
        import framing
        timeline['framing'] = framing.keep_drawing_in_view(timeline)
        for page in timeline['pages']:
            validate_page_motion(page, {'width': self.width, 'height': self.height}, first=page is timeline['pages'][0])
        timeline['storyboard'] = check_build(self.root, self.manifest, timeline)
        timeline['storyboard']['warnings'] += self.caption_warnings
        for item in timeline['framing']['adjustments']:
            what = ', '.join(([f"moves waited {item['moves_waited_seconds']}s"] if 'moves_waited_seconds' in item else [])
                             + ([f"framing move ({item['framing']}) at {item['framed_at']}s"] if 'framed_at' in item else []))
            timeline['storyboard']['warnings'].append(f"page {item['page']}: {item['role']} {item['asset']} was drawn {item['overflow_px']}px outside the view; {what}")
        for item in timeline['framing']['unfixable']:
            timeline['storyboard']['warnings'].append(f"page {item['page']}: {item['role']} {item['asset']} is drawn {item['overflow_px']}px out of clear view at "
                                                      f"{item['at']}s: {item['reason']}")
        # Pen and marker sounds follow the finished drawing schedule (audio.foley, on by default).
        from foley import attach as attach_foley
        attach_foley(self.root, timeline, self.manifest)
        # Paper, eraser, whoosh and chime sounds for transitions and accents (audio.sfx, on by default, quiet).
        from sfx import attach as attach_sfx
        attach_sfx(self.root, timeline, self.manifest)
        cover = Image.open(resolve_input(self.root, self.manifest['cover']['source'])).convert('RGB')
        if abs(cover.width / cover.height - self.width / self.height) > .025:
            raise ValueError('Cover aspect ratio differs from canvas; generate or intentionally crop an appropriate cover')
        cover.resize((self.width, self.height), Image.Resampling.LANCZOS).save(self.output / 'cover.png')
        (self.work / 'timeline.json').write_text(json.dumps(timeline, ensure_ascii=False, indent=2))
        # Viewing metrics come from rendered frames, so they need the saved timeline and cover.
        import visual_metrics
        from video_renderer import Renderer
        design_path = self.root / self.manifest.get('design_file', 'design.json')
        design = json.loads(design_path.read_text()) if design_path.is_file() else {}
        timeline['visual_metrics'] = visual_metrics.measure(Renderer(self.root), design)
        timeline['storyboard']['warnings'] += timeline['visual_metrics']['warnings']
        (self.work / 'timeline.json').write_text(json.dumps(timeline, ensure_ascii=False, indent=2))
        if timeline['visual_metrics']['errors'] and not self.manifest.get('runtime_fixture'):
            raise ValueError('; '.join(timeline['visual_metrics']['errors']))
        editing = self.output / 'editing'
        editing.mkdir(exist_ok=True)
        legacy = self.output / 'captions.srt'
        if legacy.exists():
            legacy.replace(editing / 'previous-captions.srt')
        (editing / 'transcript.srt').write_text('\n'.join(f"{index + 1}\n{subtitle_time(cue['start'])} --> {subtitle_time(cue['end'])}\n{cue['text']}\n" for index, cue in enumerate(cues)))
        (self.output / 'script.txt').write_text(self.manifest['title'] + '\n\n' + '\n\n'.join(cue['text'] for cue in cues))
        metrics = timeline['visual_metrics']
        print(json.dumps({'duration_seconds': timeline['duration'], 'pages': len(pages), 'beats': len(cues), 'drawn_assets': len(self.assets),
                          'storyboard': timeline['storyboard']['stats'],
                          'viewing': {key: metrics[key] for key in ('first_meaningful_visual_seconds', 'longest_unchanged_seconds')},
                          'warnings': timeline['storyboard']['warnings']}, ensure_ascii=False))
        return timeline
