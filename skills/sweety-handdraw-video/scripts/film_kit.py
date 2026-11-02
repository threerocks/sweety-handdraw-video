#!/usr/bin/env python3
"""Author a film as data: compile design/film.json into job.json pages and storyboard.json.

    film_kit.py compile JOB [--film design/film.json]   job.json pages (+ drawing/audio/captions...), storyboard.json
    film_kit.py plan JOB                                design/drawing-plan.json from the film's pictures
    film_kit.py measure JOB NAME[@PLAN_SIDE] ...        minimum drawing seconds at the page size -> work/minimums.json
    film_kit.py at JOB BEAT [PHRASE]                    page seconds when a phrase is spoken (to write cues)

Cues time everything from the locked narration's word timings: "@p03b4" (that sentence), "@p03b4:推特在国内" (when the
phrase is spoken), "@p03b4:推特在国内+0.8" (seconds after it); {"=": "max(3000, ms('p05b2', '下一段', 1))"} for arithmetic.
A beat may carry "say" (what the voice reads, 七十万 for 70万): phrases are looked up in it. Schema and a minimal example:
references/production-kit.md.
"""
import argparse
import ast
import copy
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

WHOLE = [0, 0, 1, 1]
CJK = ('㐀', '鿿')
NUMERALS = '零一二三四五六七八九'
DEFAULTS = {
    'palette': {'ink': '#2B2F36', 'accent': '#C4553E', 'accent2': '#3F5F8A', 'grey': '#8C949C'},
    # 1280x1920 (2:3): header above the picture, the form band between the picture and the captions (1666), the
    # progress strip under the feed's top inset. Other canvases set their own areas.
    'areas': {'header': [77, 150, 1126, 170], 'strip': [77, 64, 1126, 56], 'band': [77, 1250, 1126, 380],
              'picture': [77, 340, 1126, 880], 'map': [77, 720, 1126, 460]},
    'timing': {'lead': .25, 'hook_lead': .05, 'text_lead': .3, 'hold_min': .6, 'minimum_factor': 1.08, 'hook_minimum_factor': 1.02,
               'title_lead': -.1, 'form_lead': -.15, 'until_lead': -.2, 'form_tail': 1.6, 'form_min': 1.5, 'form_room': .1,
               'mark_after': .8, 'write_cjk': .3, 'write_other': .12, 'write_pause': 1.},
    # Headers arrive with the page (at 0), whole glyphs one by one: a page never opens on an empty header.
    'titles': {'style': 'glyph_in', 'seconds': .8, 'size': 84, 'accent': '$accent2', 'font': 'hand'},
    'lines': {'style': 'blur_up', 'seconds': .9, 'size': 60, 'accent': '$accent', 'font': 'hand'},
    'drawing': {'mode': 'draw', 'speed': 1.4, 'plan_side': 540},
    'picture_drawing': {'sketch': False},
    'transitions': {'chapter': {'type': 'page_turn', 'seconds': .9}, 'quiet': {'type': 'slide', 'direction': 'left', 'seconds': .7},
                    'end': {'type': 'fade', 'seconds': .8}},
    'roadmap': {'groups': [], 'marks': '①②③④⑤⑥⑦⑧⑨⑩', 'strip_size': 34, 'map_size': 36, 'strip_seconds': 1.6, 'map_at': .3,
                'map_seconds': 2.4, 'strip_enter': 'settled', 'headers': [],
                'chapter_text': {'y': 470, 'size': 120, 'color': '$accent'},
                'chapter_title': {'at': 0, 'size': 64, 'accent': '$accent'}},
    'plan': {'look': {'head': ['头发外形', '脸', '耳朵和脖子'], 'torso': ['上衣']}, 'neck': '脖子很短，和头部同一模块画',
             'drawability_reason': ('线稿先行：线以线稿为准（line_first.py 配准，outline_mask），部位交界闭合、平涂加一档硬边阴影；'
                                    '按模块由简到繁描线，再上色；按口播窗口实测 minimum_seconds 核对。')},
    'storyboard': {'max_hold_seconds': 9},
}
TEXT_RECIPES = ('kinetic_title', 'callout', 'number_ticker', 'burst')
LIMB_GROUPS = {'left_arm': ['左臂', '左手'], 'right_arm': ['右臂', '右手'], 'lower_body': ['裤子', '鞋']}
ELEMENTS = ('title', 'line', 'form', 'strip', 'custom', 'annotate', 'callout', 'overlay')
COVER_SECONDS, VOICE_START = 1.0, .5   # video_job: the cover lasts 1 s, the voice starts under its fade at 0.5 s
TITLE_KEYS = {'title', 'line', 'at', 'visual', 'style', 'seconds', 'keywords', 'size', 'accent', 'color', 'font', 'area', 'x', 'y', 'width',
              'height', 'until', 'exit_at', 'exit', 'space', 'on', 'params', 'explains', 'lead'}
# Code animations: plain SVG driven by element.animate (the capture's virtual clock). p = the layer's params.
SVG_KIT = """
const NS = 'http://www.w3.org/2000/svg', W = ctx.width, H = ctx.height;
const svg = document.createElementNS(NS, 'svg'); svg.setAttribute('width', W); svg.setAttribute('height', H); root.appendChild(svg);
const add = (tag, attrs, parent) => { const n = document.createElementNS(NS, tag); for (const k in attrs) n.setAttribute(k, attrs[k]); (parent || svg).appendChild(n); return n; };
const word = (text, x, y, size, color, parent, anchor) => { const t = add('text', {x, y, 'font-size': size, fill: color, 'text-anchor': anchor || 'middle',
  'font-family': '"Hannotate SC", "HanziPen SC", "PingFang SC", sans-serif', 'dominant-baseline': 'middle'}, parent); t.textContent = text; return t; };
const show = (node, at, ms) => node.animate([{opacity: 0}, {opacity: 1}], {delay: at, duration: ms || 300, fill: 'both'});
"""
CUE = re.compile(r'^@(?P<beat>[A-Za-z0-9_\-]+?)(?::(?P<phrase>.*?))?(?P<after>[+-](?:\d+\.?\d*|\.\d+))?$')


def clean(value):
    return ''.join(ch for ch in str(value) if ch.isalnum())


def spoken_length(token):
    """Script characters a recognised word stands for: Whisper writes 五十四 as 54, 六十 as 60, 三百六十五 as 365
    (four digits and more are read digit by digit, like a year)."""
    if not token.isdigit():
        return len(token)
    n = int(token)
    if n <= 10:
        return 1
    if n < 100:
        return 2 if n < 20 or n % 10 == 0 else 3
    if n < 1000:
        tens, ones = divmod(n % 100, 10)
        return 2 if n % 100 == 0 else 4 if tens == 0 or ones == 0 else 5
    return len(token)


def joined_numbers(words):
    """Recognised words with split numbers joined again (Whisper may write 365 as "36" + "5")."""
    result = []
    for word in words:
        if result and clean(word['text']).isdigit() and clean(result[-1]['text']).isdigit():
            result[-1] = {**result[-1], 'text': clean(result[-1]['text']) + clean(word['text'])}
        else:
            result.append(word)
    return result


def chinese_number(n):
    if n < 10:
        return NUMERALS[n]
    tens, ones = divmod(n, 10)
    return (NUMERALS[tens] if tens > 1 else '') + '十' + (NUMERALS[ones] if ones else '')


def merged(base, extra, deep=('roadmap', 'plan')):
    """Defaults overlaid by the film's settings, key by key (one level deeper for the roadmap and plan settings)."""
    result = copy.deepcopy(base)
    for key, value in (extra or {}).items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            value = merged(result[key], value, ()) if key in deep else {**result[key], **copy.deepcopy(value)}
        result[key] = copy.deepcopy(value)
    return result


def latest(value):
    """The last moment (ms) a form's parameters schedule something."""
    if isinstance(value, dict):
        return max([latest(item) for key, item in value.items() if key.endswith('ms') or isinstance(item, (dict, list))] + [0])
    if isinstance(value, list):
        return max([latest(item) for item in value] + [0])
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def as_box(value):
    if isinstance(value, dict):
        return [value['x'], value['y'], value['width'], value['height']]
    return list(value)


def spoken_text(manifest, beat, text=None):
    """What the voice reads: the display text with the say map applied (narration.say, then the beat's own say)."""
    try:
        from narration_master import spoken_text as engine
    except ImportError:  # outside the Skill's venv: the same rule, without the whole-word guard for Latin terms
        mapping = {**((manifest.get('narration') or {}).get('say') or {}), **(beat.get('say') or {})}
        text = beat['text'] if text is None else text
        if not mapping:
            return text
        return re.sub('|'.join(re.escape(key) for key in sorted(mapping, key=len, reverse=True)), lambda m: mapping[m.group(0)], text)
    return engine(manifest, beat, text)


def intrinsic(recipe, params):
    """Seconds a form animates by itself when its params carry no cue times (a chart's axis and staggered events, a
    route's travel): the layer must last that long, or its frames stop half drawn (the history film's timeline froze
    after three of five events)."""
    stagger = float(params.get('stagger', 380)) / 1000
    if recipe == 'rough_chart':
        items = params.get('events') or params.get('values') or [None] * int(params.get('marked', 0) or 3)
        return .6 + stagger * len(items) + 1.2
    if recipe == 'map_route':
        return float(params.get('travel_ms', 2400)) / 1000 + 1.5
    if recipe in ('table', 'flow', 'cards', 'checklist', 'swimlane'):
        count = len(params.get('rows') or params.get('nodes') or params.get('items') or params.get('blocks') or [])
        return .8 + .5 * count + 1.
    if recipe in ('physics', 'spin3d'):
        return 4.
    return 0.


class Narration:
    """Beat texts (script), master times (alignment) and recognised words (master chunks) of the locked narration.
    A say map (job narration.say, a beat's say) turns display terms into what the voice reads (Grok -> 格罗克):
    phrases are looked up in the spoken text, which is what the word timings follow."""

    def __init__(self, root, job, say=None, beat_say=None):
        self.root = Path(root)
        alignment_path = self.root / (job.get('narration', {}).get('alignment') or 'inputs/narration/alignment.json')
        alignment = json.loads(alignment_path.read_text())
        self.times = {beat['id']: beat for beat in alignment['beats']}
        script_path = self.root / 'inputs/script.json'
        pages = json.loads(script_path.read_text()) if script_path.is_file() else job['pages']
        self.pages = [{'id': page.get('id'), 'title': page.get('title', ''), 'beats': [beat['id'] for beat in page['beats']]} for page in pages]
        self.beats = {}
        for page in pages:
            for beat in page['beats']:
                own = {**(beat.get('say') or {}), **((beat_say or {}).get(beat['id']) or {})}
                self.beats[beat['id']] = {'id': beat['id'], 'text': beat['text'], **({'say': own} if own else {})}
        self.manifest = {'narration': {'say': say if say is not None else (job.get('narration') or {}).get('say') or {}}}
        self.words, self.warned = {}, set()
        for path in sorted((alignment_path.parent / 'master-chunks').glob('*.json')):
            record = json.loads(path.read_text())
            attempts = record.get('attempts') or []
            kept = [attempt for attempt in attempts if str(attempt.get('status', '')).lower() == 'pass'] or attempts
            words = joined_numbers(kept[-1].get('words', [])) if kept else []
            if not words:
                continue
            if record.get('units'):
                offset = 0
                for unit in record['units']:
                    known = unit.get('beat') in self.beats
                    if known:
                        self.words.setdefault(unit['beat'], (offset, words))
                    offset += len(clean(self.spoken(unit['beat']) if known else unit.get('text', '')))
            else:
                for beat_id in self.beats:
                    spoken = self.spoken(beat_id)
                    if spoken and spoken in record.get('text', ''):
                        self.words.setdefault(beat_id, (clean(record['text']).find(clean(spoken)), words))

    def spoken(self, beat_id, text=None):
        return spoken_text(self.manifest, self.beats[beat_id], text)

    def phrase_offset(self, beat_id, phrase):
        """Seconds from the beat's first word to the word where the phrase begins (None without word timings)."""
        if beat_id not in self.words:
            if beat_id not in self.warned:
                self.warned.add(beat_id)
                print(f'warning: {beat_id} has no word timings; phrase cues fall back to the sentence start', file=sys.stderr)
            return None
        offset, words = self.words[beat_id]
        position = clean(self.spoken(beat_id)).find(clean(self.spoken(beat_id, phrase)))
        if position < 0:
            raise ValueError(f'{phrase!r} is not in the narration of {beat_id}: {self.spoken(beat_id)}')
        target, count, first = offset + position, 0, None
        for word in words:
            if count >= offset and first is None:
                first = word['start']
            count += spoken_length(clean(word['text']))
            if count > target:
                return word['start'] - (first or 0.)
        return 0.


class Film:
    def __init__(self, root, film_path=None):
        self.root = Path(root).resolve()
        self.film_path = self.root / (film_path or 'design/film.json')
        self.data = json.loads(self.film_path.read_text())
        self.job = json.loads((self.root / 'job.json').read_text())
        settings = merged(DEFAULTS, {key: value for key, value in self.data.items() if key in DEFAULTS})
        self.palette, self.areas, self.timing = settings['palette'], settings['areas'], settings['timing']
        self.titles, self.lines, self.roadmap = settings['titles'], settings['lines'], settings['roadmap']
        self.drawing, self.picture_drawing = settings['drawing'], settings['picture_drawing']
        self.transitions, self.plan_settings, self.board = settings['transitions'], settings['plan'], settings['storyboard']
        canvas = self.job.get('canvas', {})
        self.width, self.height = int(canvas.get('width', 1280)), int(canvas.get('height', 1920))
        self.cameras = {'still': [{'at': 0, 'center': [self.width // 2, self.height // 2], 'zoom': 1.0, 'move': 'cut'}],
                        **self.data.get('cameras', {})}
        self.pictures = self.data.get('pictures', {})
        self.characters = self.data.get('characters', [])
        self._sizes, self._explain = {}, {}
        self.minimums = {}
        path = self.root / 'work/minimums.json'
        if path.is_file():
            self.minimums = {name: value[0] for name, value in json.loads(path.read_text()).items()}

    # ------------------------------------------------------------------ narration and screens
    def load_narration(self):
        self.narration = Narration(self.root, self.job, self.data.get('say'), self.data.get('beat_say'))
        self.screens = []
        script_pages = {page['id']: page for page in self.narration.pages}
        page_of_beat = {beat: page for page in self.narration.pages for beat in page['beats']}
        for spec in self.data.get('screens') or [{'id': page['id']} for page in self.narration.pages]:
            beats = spec.get('beats') or script_pages[spec.get('page', spec['id'])]['beats']
            title = spec.get('title') or page_of_beat[beats[0]]['title']
            self.screens.append({'id': spec['id'], 'title': title, 'beats': list(beats), 'spec': spec})
        covered = [beat for screen in self.screens for beat in screen['beats']]
        expected = [beat for page in self.narration.pages for beat in page['beats']]
        if covered != expected:
            raise ValueError('screens must cover every narration beat once, in order; check: '
                             + ', '.join(sorted(set(covered) ^ set(expected)) or ['order']))
        times = self.narration.times
        self.page_of = {beat: screen['id'] for screen in self.screens for beat in screen['beats']}
        self.first = {screen['id']: times[screen['beats'][0]]['start'] for screen in self.screens}
        self.length = {screen['id']: times[screen['beats'][-1]]['end'] - times[screen['beats'][0]]['start'] for screen in self.screens}

    def at(self, beat, phrase=None, after=0.):
        """Page-relative seconds when a phrase of a beat is spoken (the beat's start when phrase is None)."""
        if beat not in self.page_of:
            raise ValueError(f'unknown beat {beat!r}')
        base = self.narration.times[beat]['start'] - self.first[self.page_of[beat]]
        offset = self.narration.phrase_offset(beat, phrase) if phrase else None
        return round(max(0., base + (offset or 0.) + after), 3)

    def cue(self, value):
        """(beat, phrase, after) of a cue ("@beat[:phrase][+s]" or {"cue", "phrase", "after"}), else None."""
        if isinstance(value, dict) and 'cue' in value:
            return value['cue'], value.get('phrase'), value.get('after')
        if isinstance(value, str) and value.startswith('@'):
            match = CUE.match(value)
            if match and match['beat'] in self.page_of:
                return match['beat'], match['phrase'] or None, float(match['after']) if match['after'] else None
            if match and re.fullmatch(r'p\d+\w*', match['beat'] or ''):
                raise ValueError(f'cue {value!r} names no beat of this film')
        return None

    def seconds(self, value, lead, beat_string=False):
        """Page seconds of an element's "at": a number as is, a cue resolved (phrase cues land `lead` seconds early)."""
        if value is None or isinstance(value, (int, float)):
            return value
        if isinstance(value, str) and value.startswith('beat:'):
            return value
        if isinstance(value, dict) and '=' in value:
            return self.evaluate(value['='], 0.)
        found = self.cue(value)
        if not found:
            raise ValueError(f'not a time: {value!r}')
        beat, phrase, after = found
        if beat_string and phrase is None and after is None:
            return f'beat:{beat}'
        if phrase is None and after is None:
            return self.at(beat)
        return self.at(beat, phrase, lead if after is None else after)

    def ms(self, beat, phrase=None, after=0., start=0.):
        return max(0, round((self.at(beat, phrase, after) - start) * 1000))

    def evaluate(self, text, start, screen=None):
        """Arithmetic over cue times: numbers, + - * / // %, max min round int abs float range, ms(...), at(...)."""
        functions = {'max': max, 'min': min, 'round': round, 'int': int, 'abs': abs, 'float': float,
                     'range': lambda *args: list(range(*[int(arg) for arg in args])),
                     'ms': lambda beat, phrase=None, after=0.: self.ms(beat, phrase, after, start),
                     'at': lambda beat, phrase=None, after=0.: self.at(beat, phrase, after)}
        names = {'start': start, 'length': self.length.get(screen, 0.) if screen else 0.}
        operators = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
                     ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b}

        def walk(node):
            if isinstance(node, ast.Expression):
                return walk(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str, type(None))):
                return node.value
            if isinstance(node, ast.BinOp) and type(node.op) in operators:
                return operators[type(node.op)](walk(node.left), walk(node.right))
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
                return -walk(node.operand) if isinstance(node.op, ast.USub) else walk(node.operand)
            if isinstance(node, (ast.List, ast.Tuple)):
                return [walk(item) for item in node.elts]
            if isinstance(node, ast.Name) and node.id in names:
                return names[node.id]
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in functions:
                return functions[node.func.id](*[walk(arg) for arg in node.args], **{item.arg: walk(item.value) for item in node.keywords})
            raise ValueError(f'unsupported expression in {text!r}: {ast.dump(node)[:80]}')
        return walk(ast.parse(text, mode='eval'))

    def token(self, value):
        """"$ink" -> palette colour, "$roadmap" -> the roadmap groups; anything else unchanged."""
        if isinstance(value, str) and value.startswith('$'):
            if value[1:] in self.palette:
                return self.palette[value[1:]]
            if value == '$roadmap':
                return copy.deepcopy(self.roadmap['groups'])
        return value

    def resolve(self, value, start=0., screen=None):
        """Cues inside a form's params become ms from the form's start; "$name" tokens their values."""
        if isinstance(value, dict):
            if set(value) == {'='}:
                return self.evaluate(value['='], start, screen)
            if 'cue' in value:
                beat, phrase, after = self.cue(value)
                return self.ms(beat, phrase, after or 0., start)
            return {key: self.resolve(item, start, screen) for key, item in value.items()}
        if isinstance(value, list):
            return [self.resolve(item, start, screen) for item in value]
        found = self.cue(value) if isinstance(value, str) else None
        if found:
            beat, phrase, after = found
            return self.ms(beat, phrase, after or 0., start)
        return self.token(value)

    # ------------------------------------------------------------------ pictures
    def picture(self, name):
        return self.pictures.get(name, {})

    def source(self, name):
        return self.picture(name).get('source', f'inputs/art/{name}.png')

    def size(self, name):
        if name not in self._sizes:
            from PIL import Image
            with Image.open(self.root / self.source(name)) as image:
                self._sizes[name] = image.size
        return self._sizes[name]

    def crop(self, name):
        crop = self.picture(name).get('crop')
        if not crop:
            return None
        if all(isinstance(value, (int, float)) and 0 <= value <= 1 for value in crop) and any(isinstance(v, float) for v in crop):
            w, h = self.size(name)
            return [round(crop[0] * w), round(crop[1] * h), round(crop[2] * w), round(crop[3] * h)]
        return list(crop)

    def box(self, name):
        """Screen box of a picture: fitted into its area (its own, else the area for its aspect, else "picture")."""
        w, h = self.size(name)
        crop = self.crop(name)
        if crop:
            w, h = crop[2] - crop[0], crop[3] - crop[1]
        aspect = 'portrait' if h > w * 1.2 else 'landscape' if w > h * 1.2 else 'square'
        area = self.picture(name).get('area')
        area = self.area(area) if area is not None else self.area(f'picture_{aspect}' if f'picture_{aspect}' in self.areas else 'picture')
        x0, y0, mw, mh = area
        scale = min(mw / w, mh / h)
        width, height = round(w * scale), round(h * scale)
        return x0 + (mw - width) // 2, y0 + (mh - height) // 2, width, height

    def area(self, value):
        if isinstance(value, str):
            if value not in self.areas:
                raise ValueError(f'unknown area {value!r}; define it under "areas"')
            return as_box(self.areas[value])
        return as_box(value)

    def drawing_of(self, name):
        picture = self.picture(name)
        found = {**self.drawing, **self.picture_drawing}
        if picture.get('construct'):
            found.pop('sketch', None)
        if 'plan_side' in picture:
            found['plan_side'] = picture['plan_side']
        for key, value in (picture.get('drawing') or {}).items():
            found.pop(key, None) if value is None else found.__setitem__(key, value)
        if 'colouring' in picture:
            found['colouring'] = picture['colouring']
        return found

    def parts(self, name):
        """(id, subject, role, stroke groups, box, extra) of a picture's modules in drawing order; people expanded."""
        look = {**DEFAULTS['plan']['look'], **self.plan_settings['look']}
        result = []
        for part in self.picture(name).get('parts', []):
            if 'person' in part:
                prefix, groups = part['person'], {**look, **LIMB_GROUPS, **part.get('groups', {})}
                for role in ('head', 'torso', 'left_arm', 'right_arm', 'lower_body'):
                    if part.get(role) is not None:
                        result.append((f'{prefix}_{role}', prefix, role, groups[role], part[role], {}))
                continue
            extra = {key: value for key, value in part.items() if key not in ('id', 'subject', 'role', 'groups', 'box')}
            result.append((part['id'], part.get('subject', part['id']), part.get('role', part['id']), part['groups'], part.get('box', WHOLE), extra))
        return result

    def local(self, box, name):
        """A box written on the full image (part_grid sheet) in the coordinates of the action's crop."""
        crop = self.crop(name)
        if not crop:
            return box
        w, h = self.size(name)
        x0, y0, x1, y1 = crop
        u = lambda v: min(1., max(0., (v * w - x0) / (x1 - x0)))
        t = lambda v: min(1., max(0., (v * h - y0) / (y1 - y0)))
        return [round(u(box[0]), 4), round(t(box[1]), 4), round(u(box[2]), 4), round(t(box[3]), 4)]

    def action_parts(self, name):
        parts = self.parts(name)
        if not parts:
            return [{'id': 'whole', 'box': WHOLE}]
        result = []
        for index, (part_id, _, _, _, box, extra) in enumerate(parts):
            extra = dict(extra)
            if 'exclude_boxes' in extra:
                extra['exclude_boxes'] = [self.local(item, name) for item in extra['exclude_boxes']]
            # the last module takes every pixel left; a head is a key part (real ink kept first)
            result.append({'id': part_id, 'box': WHOLE if index == len(parts) - 1 else self.local(box, name), **extra,
                           **({'key': True} if part_id.endswith('_head') else {})})
        return result

    def point(self, name, u, v):
        """Page pixels of a point in the full source picture's normalized coordinates (crop and layout applied)."""
        w, h = self.size(name)
        x0, y0, x1, y1 = self.crop(name) or (0, 0, w, h)
        x, y, width, height = self.box(name)
        return x + (u * w - x0) / (x1 - x0) * width, y + (v * h - y0) / (y1 - y0) * height

    def picture_characters(self, name):
        picture = self.picture(name)
        if 'characters' in picture:
            return list(picture['characters'])
        humans = {subject.get('human') or (subject.get('id') if subject.get('kind') == 'human' else None) for subject in picture.get('subjects', [])}
        return [character['id'] for character in self.characters if character['id'] in humans]

    def plan_entry(self, name):
        picture = self.picture(name)
        parts = [{'id': part_id, 'subject': subject, 'role': role, 'stroke_groups': groups} for part_id, subject, role, groups, _, _ in self.parts(name)]
        order = [part['id'] for part in parts]
        subjects = []
        for subject in picture.get('subjects', []):
            if 'human' in subject:
                subjects.append({'id': subject['human'], 'kind': 'human', 'identity_cues': subject['cues'],
                                 'omitted_roles': {'neck': self.plan_settings['neck'], **subject.get('omitted', {})},
                                 **({'order_exception': subject['order_exception']} if subject.get('order_exception') else {}),
                                 **({'colours': subject['colours']} if subject.get('colours') else {})})
            elif any(kind in subject for kind in ('object', 'animal', 'environment')):
                kind = next(kind for kind in ('object', 'animal', 'environment') if kind in subject)
                subjects.append({'id': subject[kind], 'kind': kind, 'identity_cues': subject['cues']})
            else:
                subjects.append(subject)
        entry = {'id': name, 'source': self.source(name), 'composition': picture['composition'],
                 'drawability_reason': picture.get('drawability_reason', self.plan_settings['drawability_reason']),
                 'outline_strategy': picture.get('outline_strategy', 'line_first'), 'subjects': subjects, 'parts': parts,
                 'outline_order': order, 'fill_order': order, 'spatial_order': 'top_to_bottom_left_to_right',
                 'narrative_requirements': [{'id': f'{name}-n{k + 1}', 'narration_quote': quote, 'observable': observable}
                                            for k, (quote, observable) in enumerate(picture.get('narrative', []))]}
        if picture.get('order_exception'):
            entry['order_exception'] = picture['order_exception']
        references = picture.get('references') or [character['reference'] for character in self.characters
                                                    if character['id'] in self.picture_characters(name)]
        if references:
            entry['references'] = references
        return entry

    def write_plan(self):
        plan = {'illustrations': [self.plan_entry(name) for name, picture in self.pictures.items() if picture.get('parts')]}
        (self.root / 'design').mkdir(exist_ok=True)
        (self.root / 'design/drawing-plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=1))
        return plan

    def minimum(self, name, plan_side=None):
        import line_drawing
        import stroke_assets
        x, y, width, height = self.box(name)
        drawing = self.drawing_of(name)
        if plan_side:
            drawing['plan_side'] = int(plan_side)
        pw, ph, _ = line_drawing.plan_box(width, height, drawing)
        crop = self.crop(name)
        outline = self.picture(name).get('outline_mask', f'inputs/art/{name}-outline.png')
        with tempfile.TemporaryDirectory() as directory:
            stroke_assets.configure_assets(directory, '/dev/null', 0, 30)
            meta = stroke_assets.build_illustration(name, self.root / self.source(name), pw, ph, crop, self.action_parts(name),
                                                    self.root / outline if outline else None, seconds=1.0, drawing=dict(drawing))
        return meta['drawability']['minimum_seconds'], meta['drawability']['stroke_count']

    # ------------------------------------------------------------------ actions and overlays
    def number(self, screen_id):
        return screen_id[1:] if re.match(r'^p\d', screen_id) else screen_id

    def picture_action(self, item, screen_id):
        name = item['picture']
        picture = self.picture(name)
        x, y, width, height = self.box(name)
        suffix = name.split('-', 1)[1] if '-' in name else name
        action = {'type': 'illustration', 'source': self.source(name)}
        if picture.get('line_first', True):
            action['outline_mask'] = picture.get('outline_mask', f'inputs/art/{name}-outline.png')
            action['line_first'] = f'inputs/art/{name}.line-first.json'
        elif picture.get('outline_mask'):
            action['outline_mask'] = picture['outline_mask']
        action.update({'drawing_plan_id': name, 'x': x, 'y': y, 'width': width, 'height': height,
                       'visual': item.get('visual') or picture.get('visual') or f'v{self.number(screen_id)}-{suffix}',
                       'weight': float(item.get('weight', 3.)), 'parts': self.action_parts(name)})
        if self.crop(name):
            action['crop'] = self.crop(name)
        own = {key: value for key, value in self.drawing_of(name).items() if self.drawing.get(key) != value}
        if own:
            action['drawing'] = own
        for key in ('motion', 'depth', 'z', 'pen'):
            if key in item:
                action[key] = item[key]
        if item.get('from_beat'):
            action['from_beat'] = item['from_beat']
        return action

    def text_action(self, item, screen_id):
        size = item.get('size', 84)
        content = item['text']
        x = (self.width - len(content) * size) / 2 if item.get('x', 'center') == 'center' else item['x']
        action = {'type': 'text', 'text': content, 'x': round(x), 'y': round(item['y']), 'size': size,
                  'color': self.token(item.get('color', '$ink')), 'visual': item.get('visual') or f'v{self.number(screen_id)}-text',
                  'weight': item.get('weight', .6)}
        action.update({key: value for key, value in item.items() if key not in ('text', 'x', 'y', 'size', 'color', 'visual', 'weight', 'from_beat')})
        if item.get('from_beat'):
            action['from_beat'] = item['from_beat']
        return action

    def place(self, overlay, element):
        """Area name/box, then single coordinates; "on": {"picture", "box": [u0, v0, u1, v1]} places it on a picture."""
        if element.get('on'):
            on = element['on']
            (ax, ay), (bx, by) = self.point(on['picture'], on['box'][0], on['box'][1]), self.point(on['picture'], on['box'][2], on['box'][3])
            overlay.update(x=round(ax), y=round(ay), width=round(bx - ax), height=round(by - ay))
        for key in ('x', 'y', 'width', 'height'):
            if key in element:
                overlay[key] = element[key]

    def exit_of(self, overlay, element):
        if element.get('until') is not None:
            beat, phrase, after = self.cue(element['until'])
            overlay['exit'] = {'style': 'fade', 'at': round(max(0., self.at(beat, phrase, after or 0.) + self.timing['until_lead']), 3), 'seconds': .3}
        elif element.get('exit_at') is not None:
            value = element['exit_at']
            at = value if isinstance(value, (int, float)) or str(value).startswith('beat:') else round(self.seconds(value, 0.), 3)
            overlay['exit'] = {'style': 'fade', 'at': at, 'seconds': .35}
        elif element.get('exit'):
            overlay['exit'] = {**element['exit'], 'at': self.seconds(element['exit'].get('at'), 0., True)}
        return overlay

    def title(self, element, screen_id, line=False):
        """A kinetic title: a header (at the page's arrival unless cued) or a short timed line in the band."""
        style = self.lines if line else self.titles
        content = element['line' if line else 'title']
        # The first screen's header waits for the cover to finish fading (the voice starts 0.5 s under the fade): a header
        # arriving during the fade reads as a second, ghosted title over the cover's own.
        default = (COVER_SECONDS - VOICE_START) if self.screens and screen_id == self.screens[0]['id'] else 0
        at = element.get('at', None if line else default)
        if at is None:
            raise ValueError(f'{screen_id}: a line needs "at" (a cue)')
        at = self.seconds(at, self.timing['title_lead'], beat_string=True)
        size = element.get('size', style['size'])
        params = {'text': content, 'keywords': list(element.get('keywords', [])), 'style': element.get('style', style['style']), 'size': size,
                  'font': element.get('font', style.get('font', 'hand')), 'color': self.token(element.get('color', '$ink')),
                  'accent': self.token(element.get('accent', style['accent']))}
        params.update({key: self.token(value) for key, value in element.items() if key not in TITLE_KEYS})
        params.update(self.resolve(element.get('params', {})))
        overlay = {'recipe': 'kinetic_title', 'space': element.get('space', 'screen'), **dict(zip(('x', 'y', 'width', 'height'),
                   self.area(element.get('area', 'header')))), 'at': at, 'seconds': element.get('seconds', style['seconds']),
                   'visual': element.get('visual') or f'v{self.number(screen_id)}-{"line" if line else "title"}', 'params': params}
        if line:
            # A line sits at the top of the band unless the film gives it an area (a box of its own) or a y.
            if 'area' in element:
                overlay.update(height=round(size * 1.6))
            else:
                overlay.update(y=self.area('band')[1], height=round(size * 1.6))
        self.place(overlay, element)
        return self.exit_of(overlay, element)

    def form(self, element, screen_id, recipe=None, params=None):
        """A motion form started at a cue; times inside its params come from the words too."""
        recipe = recipe or element['form']
        lead = element.get('lead', self.timing['form_lead'])
        start = self.seconds(element.get('at', f'@{self.screen_beats[screen_id][0]}'), lead)
        if isinstance(start, str):
            raise ValueError(f'{screen_id}: a form starts at a cue or a number, not {start!r}')
        params = copy.deepcopy(element.get('params', {}) if params is None else params)
        if recipe == 'checklist':
            params['items'] = [self.checklist_item(item, start) for item in params.get('items', [])]
        params = self.resolve(params, start, screen_id)
        if 'seconds' in element:
            seconds = element['seconds']
        else:
            room = self.length[screen_id] - start - self.timing['form_room']
            seconds = round(min(room, max(self.timing['form_min'], latest(params) / 1000 + self.timing['form_tail'], intrinsic(recipe, params))), 2)
        overlay = {'recipe': recipe, 'space': element.get('space', 'screen'),
                   **dict(zip(('x', 'y', 'width', 'height'), self.area(element.get('area', 'band')))), 'at': start, 'seconds': seconds,
                   'visual': element.get('visual') or f'v{self.number(screen_id)}-{recipe}',
                   'params': {'ink': self.palette['ink'], 'accent': self.palette['accent'], 'font': 'hand', **params}}
        for key in ('depth', 'pen'):
            if key in element:
                overlay[key] = element[key]
        self.place(overlay, element)
        return self.exit_of(overlay, element)

    def checklist_item(self, item, start):
        """{"text", "at": cue, "mark"} -> ms (when written) and mark_ms (mark_after later, or "mark_at"); struck lines in the accent."""
        if 'at' not in item:
            return item
        beat, phrase, after = self.cue(item['at'])
        result = {'text': item['text'], 'ms': self.ms(beat, phrase, after or 0., start)}
        if item.get('mark'):
            result['mark'] = item['mark']
            if item.get('mark_at') is not None:
                result['mark_ms'] = self.resolve(item['mark_at'], start)
            else:
                result['mark_ms'] = self.ms(beat, phrase, (after or 0.) + self.timing['mark_after'], start)
            if item['mark'] == 'strike' and 'color' not in item:
                result['color'] = self.palette['accent']
        result.update({key: self.token(value) for key, value in item.items() if key not in ('text', 'at', 'mark', 'mark_at')})
        return result

    def custom(self, element, screen_id):
        """A code animation (SVG helpers + the film's script) of what a picture cannot show moving."""
        source = element['custom']
        scripts = self.data.get('scripts', {})
        script = scripts[source] if source in scripts else source
        if isinstance(script, str) and script.endswith('.js') and '\n' not in script and (self.root / script).is_file():
            script = (self.root / script).read_text()  # a scripts entry or the custom value may name a file in the job
        if element.get('svg_kit', True):
            script = SVG_KIT + script
        start = self.seconds(element.get('at', f'@{self.screen_beats[screen_id][0]}'), element.get('lead', self.timing['form_lead']))
        moments = self.resolve(element.get('moments', []), start, screen_id)
        moments = [value for item in moments for value in (item if isinstance(item, list) else [item])]
        params = {'script': script, 'element': element['element'], 'explains': element['explains'],
                  'moments': sorted(round(value) for value in moments), **element.get('params', {})}
        return self.form({**element, 'params': params}, screen_id, 'custom')

    def annotate(self, element, screen_id):
        """A hand-drawn circle or box around part of a finished picture, when the narration names it."""
        name, area, pad = element['annotate'], element['box'], element.get('pad', 14)
        (ax, ay), (bx, by) = self.point(name, area[0], area[1]), self.point(name, area[2], area[3])
        overlay = {'recipe': 'rough_annotate', 'space': 'board', 'x': round(ax - pad), 'y': round(ay - pad), 'width': round(bx - ax + 2 * pad),
                   'height': round(by - ay + 2 * pad), 'at': self.seconds(element['at'], -.15, True), 'seconds': element.get('seconds', 1.0),
                   'visual': element.get('visual') or f'v{self.number(screen_id)}-mark',
                   'params': {'type': element.get('kind', 'circle'), 'color': self.token(element.get('color', '$accent')), 'stroke_width': 6,
                              'padding': 6, 'iterations': 2, 'duration_ms': 800}}
        return self.exit_of(overlay, element)

    def callout(self, element, screen_id):
        overlay = {'recipe': 'callout', 'space': element.get('space', 'board'), 'x': element['x'], 'y': element['y'],
                   'width': element.get('width', 520), 'height': element.get('height', 150), 'seconds': element.get('seconds', .9),
                   'visual': element.get('visual') or f'v{self.number(screen_id)}-note', 'at': self.seconds(element['at'], self.timing['title_lead'], True),
                   'params': {'text': element['callout'], 'size': element.get('size', 46), 'tail': element.get('tail', 'bottom-left'),
                              'fill': element.get('fill', '#FFFFFF')}}
        return self.exit_of(overlay, element)

    def group_of(self, number):
        counts = [group['count'] for group in self.roadmap['groups']]
        return next(index for index in range(len(counts)) if number <= sum(counts[:index + 1]))

    def strip(self, element, screen_id):
        """Where we are in a long list: {"group": g} lights the whole group, {"item": n} the n-th item (1-based)."""
        spec = element['strip'] if isinstance(element['strip'], dict) else {'group': element['strip']}
        groups, marks = self.roadmap['groups'], self.roadmap['marks']
        if 'item' in spec:
            number = spec['item']
            group = self.group_of(number)
            params = {'mode': 'strip', 'groups': copy.deepcopy(groups), 'done': number - 1, 'current': number - 1, 'focus': group}
        else:
            group = spec['group']
            before = sum(item['count'] for item in groups[:group])
            params = {'mode': 'strip', 'groups': copy.deepcopy(groups), 'done': before, 'active': [before, before + groups[group]['count']]}
        params.update({'size': self.roadmap['strip_size'], 'caption': spec.get('caption', f"{marks[group]} {groups[group]['label']}"),
                       'ink': self.palette['ink'], 'accent': self.palette['accent'], 'font': 'hand'})
        if self.strips_seen and self.roadmap.get('strip_enter'):
            params['enter'] = self.roadmap['strip_enter']   # from the second strip page on: earlier dots already drawn
        self.strips_seen += 1
        return {'recipe': 'roadmap', 'space': 'screen', **dict(zip(('x', 'y', 'width', 'height'), self.area('strip'))), 'at': 0.,
                'seconds': self.roadmap['strip_seconds'], 'visual': element.get('visual') or f'v{self.number(screen_id)}-where', 'params': params}

    def element(self, element, screen_id):
        kind = next((key for key in ELEMENTS if key in element), None)
        if kind is None:
            raise ValueError(f'{screen_id}: an overlay needs one of {ELEMENTS}: {json.dumps(element, ensure_ascii=False)[:120]}')
        if kind == 'overlay':
            overlay = copy.deepcopy(element['overlay'])
            if 'at' in overlay:
                overlay['at'] = self.seconds(overlay['at'], 0., True)
            overlay['params'] = self.resolve(overlay.get('params', {}), overlay['at'] if isinstance(overlay.get('at'), (int, float)) else 0., screen_id)
            result = self.exit_of(overlay, element)
        elif kind in ('title', 'line'):
            result = self.title(element, screen_id, line=kind == 'line')
        else:
            result = getattr(self, kind)(element, screen_id)
        if isinstance(element.get('explains'), list):
            self._explain[id(result)] = tuple(element['explains'])
        return result

    def chapter(self, screen):
        """A chapter page: the pen writes the group's name, the header names the chapter, the full map lights the group."""
        spec, screen_id = screen['spec'], screen['id']
        chapter = spec['chapter'] if isinstance(spec['chapter'], dict) else {'group': spec['chapter']}
        group, groups, nn = chapter['group'], self.roadmap['groups'], self.number(screen_id)
        name = chapter.get('name', groups[group]['label'])
        header = chapter.get('header') or (self.roadmap['headers'][group] if group < len(self.roadmap['headers']) else name)
        text_style, title_style = self.roadmap['chapter_text'], self.roadmap['chapter_title']
        write = self.text_action({'text': name, 'y': text_style['y'], 'size': text_style['size'], 'color': text_style['color'],
                                  'visual': f'v{nn}-name', 'weight': 1.}, screen_id)
        at = f'@{screen["beats"][0]}' if title_style.get('at') == 'beat' else title_style.get('at', 0)
        title = self.title({'title': header, 'at': at, 'visual': f'v{nn}-name', **{key: value for key, value in title_style.items() if key != 'at'}}, screen_id)
        roadmap = {'recipe': 'roadmap', 'space': 'screen', **dict(zip(('x', 'y', 'width', 'height'), self.area('map'))), 'at': self.roadmap['map_at'],
                   'seconds': self.roadmap['map_seconds'], 'visual': f'v{nn}-map',
                   'params': {'groups': copy.deepcopy(groups), 'done': sum(item['count'] for item in groups[:group]), 'focus': group,
                              'ink': self.palette['ink'], 'accent': self.palette['accent'], 'font': 'hand', 'size': self.roadmap['map_size']}}
        purpose = (f'第{chinese_number(group + 1)}组开场：笔写出组名“{name}”，全片地图点亮这一组，前面的组已完成——一眼看出这是章节，不是又一条')
        return [write], [title, roadmap], ('overview_recap', purpose)

    # ------------------------------------------------------------------ pages
    def visual_groups(self, screen, actions):
        """One visual group per page. A picture's drawing window ends where its measured minimum (plus a little room) is
        met, so the rest of the page is time to look at it; handwriting from a later beat follows the picture."""
        if not actions:
            return []
        spec, screen_id, timing = screen['spec'], screen['id'], self.timing
        hook = screen is self.screens[0]
        page_seconds = self.length[screen_id]
        texts = [action for action in actions if action['type'] == 'text']
        writing = sum(sum(timing['write_cjk'] if CJK[0] <= ch <= CJK[1] else timing['write_other'] for ch in action['text']) + timing['write_pause']
                      for action in texts)
        head = {'start_beat': screen['beats'][0], 'end_beat': screen['beats'][-1]}
        if actions[0]['type'] != 'illustration':
            lead = spec.get('lead', timing['text_lead'])
            return [{**head, 'lead': lead, 'hold': round(max(timing['hold_min'], page_seconds - lead - writing), 2), 'actions': actions}]
        lead = spec.get('lead', timing['hook_lead'] if hook else timing['lead'])
        factor = spec.get('minimum_factor', timing['hook_minimum_factor'] if hook else timing['minimum_factor'])
        cursor = 0.
        for action in actions:
            if action['type'] != 'illustration':
                continue
            name = action['drawing_plan_id']
            if name not in self.minimums:
                raise ValueError(f'{name}: no measured minimum drawing time; run film_kit.py measure JOB {name} (or art_pipeline.py sync-minimums after a build)')
            begin = max(cursor, self.at(action['from_beat'])) if action.get('from_beat') else cursor
            cursor = begin + self.minimums[name] * factor
        late = max([self.at(action['from_beat']) for action in actions if action.get('from_beat')] + [0.])
        end = max(cursor, late + writing) if writing else cursor
        hold = max(spec.get('hold_min', timing['hold_min']), page_seconds - lead - end)
        return [{**head, 'lead': lead, 'hold': round(hold, 2), 'actions': actions}]

    def build_screen(self, screen):
        spec, screen_id = screen['spec'], screen['id']
        actions, overlays, shot = [], [], spec.get('shot')
        if spec.get('chapter') is not None:
            actions, overlays, default_shot = self.chapter(screen)
            shot = shot or default_shot
        items = ([{'picture': spec['picture']}] if isinstance(spec.get('picture'), str) else [spec['picture']] if spec.get('picture') else [])
        for item in items + list(spec.get('actions', [])):
            actions.append(self.picture_action(item, screen_id) if 'picture' in item else self.text_action(item, screen_id))
        overlays += [self.element(element, screen_id) for element in spec.get('overlays', [])]
        page = {'id': screen_id, 'title': screen['title'],
                'beats': [dict(self.narration.beats[beat]) for beat in screen['beats']],
                'visual_groups': self.visual_groups(screen, actions), 'overlays': overlays}
        transition = spec.get('transition', 'chapter' if spec.get('chapter') is not None else None)
        if transition and screen is not self.screens[0]:
            page['transition'] = copy.deepcopy(self.transitions[transition] if isinstance(transition, str) else transition)
        camera = spec.get('camera', 'still' if spec.get('chapter') is not None else None)
        if camera:
            page['camera'] = copy.deepcopy(self.cameras[camera] if isinstance(camera, str) else camera)
        if spec.get('chapter') is not None or spec.get('chapter_page'):
            page['chapter'] = True
        if not shot:
            raise ValueError(f'{screen_id}: give the shot ["type", "purpose"] (storyboard_lint SHOT_TYPES)')
        return page, actions, overlays, (shot['type'], shot['purpose']) if isinstance(shot, dict) else tuple(shot)

    # ------------------------------------------------------------------ storyboard
    def explains(self, overlay):
        """(element, what it explains) of a form, read from its own content (or the element's "explains")."""
        if id(overlay) in self._explain:
            return self._explain[id(overlay)]
        recipe, params = overlay['recipe'], overlay['params']
        custom = self.board.get('explains', {})
        key = 'roadmap_strip' if recipe == 'roadmap' and params.get('mode') == 'strip' else recipe
        if key in custom:
            return tuple(custom[key])
        texts = lambda items, field='text': '、'.join(str(item.get(field, '')) for item in items if item.get(field))[:60]
        if key == 'roadmap_strip':
            return '顶部进度条', '这是第几条、属于哪一组、还剩几条'
        if recipe == 'roadmap':
            groups = params.get('groups') or self.roadmap['groups']
            total = sum(group.get('count', 1) for group in groups)
            return '全片地图', f'{chinese_number(len(groups))}组{chinese_number(total) if total < 100 else total}条的结构，这一组在哪、讲完了多少'
        if recipe == 'table':
            return '表格：' + ' / '.join(column for column in params['columns'] if column), '逐行对比：' + '、'.join(str(row[0]) for row in params['rows'])
        if recipe == 'checklist':
            return '清单', texts(params['items'])
        if recipe == 'flow':
            return '流程图', '→'.join(node['text'] for node in params['nodes'])[:60]
        if recipe == 'cards':
            groups = params.get('groups')
            return f'{params.get("count", "")} 张卡片', ('分成' + '、'.join(group['label'] for group in groups)) if groups else '同样的很多个'
        if recipe == 'swimlane':
            return '泳道：' + params.get('title', ''), '同一条时间线上谁在做、谁在等：' + texts(params.get('lanes', []), 'label')
        if recipe == 'pulse':
            return '心跳线', texts(params['beats'], 'label')
        if recipe == 'rough_chart':
            kind = params.get('kind', 'bar')
            if kind == 'dots':
                return '点阵', f"{params.get('total')} 个里有 {params.get('marked')} 个"
            if kind == 'timeline':
                return '时间轴', '→'.join(f"{event.get('when', '')}{event.get('label', '')}" for event in params.get('events', []))[:60]
            if kind == 'tally':
                return '正字计数', f"{params.get('count')}"
            return ({'pie': '饼图', 'line': '折线图'}.get(kind, '柱状图'),
                    '、'.join(f'{label} {value}' for label, value in zip(params.get('labels', []), params.get('values', []))))
        if recipe == 'rough_annotate':
            return '圈注', '口播点到的那一部分'
        if recipe == 'custom':
            return params.get('element', '动画'), params.get('explains', '口播讲的机制')
        if recipe == 'map_route':
            return '路线地图', '、'.join(stop.get('label', '') for stop in params.get('stops', [])) or '从哪里到哪里'
        if recipe == 'physics':
            return '真实物理的道具', texts(params.get('items', []), 'label') or '东西一样样落下'
        if recipe == 'number_ticker':
            return '滚动数字', f"{params.get('to')}{params.get('suffix', '')}"
        return recipe, recipe

    def composition(self, spec, visuals, overlays, chapter):
        """The picture says what and who; the forms around it say how. Pages without a picture focus on their main form."""
        context, seen = [], set()
        for overlay in overlays:
            if visuals[overlay['visual']]['kind'] != 'diagram':
                continue
            element, why = self.explains(overlay)
            if (overlay['visual'], element) not in seen:
                seen.add((overlay['visual'], element))
                context.append({'visual': overlay['visual'], 'element': element, 'explains': why})
        context += spec.get('context', [])
        pictures = [visual['id'] for visual in visuals.values() if visual['kind'] == 'illustration']
        if pictures:
            return {'layout': 'hero_with_context', 'focus': pictures[0], 'context': context}
        if not context:
            raise ValueError(f'{spec["id"]}: a page without a picture needs a diagram (a form) or "context" to focus on')
        focus = next((item['visual'] for item in context if visuals.get(item['visual'], {}).get('role') != 'navigation'), context[0]['visual'])
        element, why = spec.get('focus_note') or (('点亮的这一组与已完成的组', '前面讲完了什么，这一章要讲什么') if chapter
                                                  else ('关键数字与流程', '这一页讲清的事'))
        context.append({'visual': focus, 'element': element, 'explains': why})
        reason = spec.get('negative_space_reason') or (self.board.get('chapter_negative_space', '章节页刻意不放插画：只有笔写的组名和全片地图（动效层），和条目页一眼区分开')
                                                       if chapter else '这一页刻意不放插画：动效层讲清这一段')
        return {'layout': 'modules', 'focus': focus, 'context': context, 'negative_space_reason': reason}

    def shot(self, screen, actions, overlays, shot_type, purpose):
        visuals = {}
        for action in actions:
            if action['type'] == 'illustration':
                name = action['drawing_plan_id']
                picture = self.picture(name)
                entry = {'id': action['visual'], 'kind': 'illustration', 'subject': picture.get('subject', picture.get('composition', name))[:60],
                         'weight': action['weight'], 'source': action['source']}
                if self.picture_characters(name):
                    entry['characters'] = self.picture_characters(name)
                if picture.get('background'):
                    entry['background'] = picture['background']
                visuals[action['visual']] = entry
            else:
                visuals.setdefault(action['visual'], {'id': action['visual'], 'kind': 'text_group', 'subject': action['text']})
        for overlay in overlays:
            kind = 'text_group' if overlay['recipe'] in TEXT_RECIPES else 'diagram'
            params = overlay.get('params', {})
            subject = str(params.get('text') or params.get('title') or params.get('element') or overlay['recipe'])[:40]
            visuals.setdefault(overlay['visual'], {'id': overlay['visual'], 'kind': kind, 'subject': subject})
            if kind == 'diagram' and visuals[overlay['visual']]['kind'] == 'text_group':
                visuals[overlay['visual']].update(kind='diagram', subject=subject)   # pen writing on a coded diagram
            if overlay['recipe'] == 'roadmap' and params.get('mode') == 'strip':
                visuals[overlay['visual']]['role'] = 'navigation'   # where we are in the list, not what this page explains
        spec = {**screen['spec'], 'id': screen['id']}
        result = {'id': f's{self.number(screen["id"])}', 'beats': screen['beats'], 'type': shot_type, 'purpose': purpose,
                  'visuals': list(visuals.values()), 'composition': self.composition(spec, visuals, overlays, spec.get('chapter') is not None)}
        for key in ('density_exception', 'callback', 'board'):
            if spec.get(key):
                result[key] = spec[key]
        return result

    def compile(self):
        self.load_narration()
        self.screen_beats = {screen['id']: screen['beats'] for screen in self.screens}
        self.strips_seen = 0
        pages, shots = [], []
        for screen in self.screens:
            page, actions, overlays, (shot_type, purpose) = self.build_screen(screen)
            pages.append(page)
            shots.append(self.shot(screen, actions, overlays, shot_type, purpose))
        job = self.job
        job['pages'] = pages
        if self.data.get('say') is not None:
            job.setdefault('narration', {})['say'] = self.data['say']
        job['drawing'] = self.drawing
        for key, value in self.data.get('job', {}).items():
            job[key] = self.resolve(value)
        job.setdefault('storyboard', 'storyboard.json')
        if self.pictures:
            job.setdefault('drawing_plan', 'design/drawing-plan.json')
            job.setdefault('drawing_review', 'design/drawing-review.json')
        design = json.loads((self.root / job.get('design_file', 'design.json')).read_text())
        characters = [{'id': character['id'], 'reference': character['reference'],
                       'sha256': hashlib.sha256((self.root / character['reference']).read_bytes()).hexdigest()} for character in self.characters]
        storyboard = {'version': 1, 'presentation': design['presentation'], 'max_hold_seconds': self.board['max_hold_seconds'],
                      'characters': characters, 'shots': shots}
        storyboard.update({key: value for key, value in self.board.items() if key not in ('max_hold_seconds', 'explains', 'chapter_negative_space')})
        return job, storyboard

    def write(self):
        job, storyboard = self.compile()
        (self.root / 'job.json').write_text(json.dumps(job, ensure_ascii=False, indent=1))
        (self.root / job['storyboard']).write_text(json.dumps(storyboard, ensure_ascii=False, indent=1))
        return job, storyboard


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('compile', 'plan', 'measure', 'at'):
        command = commands.add_parser(name)
        command.add_argument('job', type=Path)
        command.add_argument('--film', help='film data file inside the job (default design/film.json)')
        if name == 'measure':
            command.add_argument('names', nargs='+', help='NAME or NAME@PLAN_SIDE (a trial size, not saved)')
        if name == 'at':
            command.add_argument('beat')
            command.add_argument('phrase', nargs='?')
    args = parser.parse_args(argv)
    film = Film(args.job, args.film)
    if args.command == 'compile':
        job, storyboard = film.write()
        print(json.dumps({'pages': len(job['pages']), 'overlays': sum(len(page['overlays']) for page in job['pages']),
                          'shots': len(storyboard['shots']), 'next': 'storyboard_lint.py JOB; director_lint.py JOB'}, ensure_ascii=False))
    elif args.command == 'plan':
        print(len(film.write_plan()['illustrations']), 'illustration plans -> design/drawing-plan.json')
    elif args.command == 'measure':
        path = film.root / 'work/minimums.json'
        found = json.loads(path.read_text()) if path.is_file() else {}
        for item in args.names:
            name, _, side = item.partition('@')
            try:
                value = [float(v) for v in film.minimum(name, side or None)]
            except ValueError as error:
                print(item, 'FAIL', str(error)[:300], flush=True)
                continue
            print(item, value, flush=True)
            if not side:
                found[name] = value
                path.parent.mkdir(exist_ok=True)
                path.write_text(json.dumps(found, indent=1))
    else:
        film.load_narration()
        print(json.dumps({'screen': film.page_of[args.beat], 'seconds': film.at(args.beat, args.phrase)}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
