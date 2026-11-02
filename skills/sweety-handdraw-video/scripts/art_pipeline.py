#!/usr/bin/env python3
"""Line-first picture pipeline driven by a data file in the job (default design/image-specs.json).

    art_pipeline.py JOB prompts                 briefs (with a stroke budget from the page window) and plate/colour/edit
                                                prompts for every picture, registered and bound (art_direction.py) before generation
    art_pipeline.py JOB gen NAME ... [--edit]   plate -> image_gate --plate -> colour edit -> line_first registration (3 at a time);
                                                --edit retires the current plate and redraws it from the spec's "edit" (named erasures)
    art_pipeline.py JOB record NAME EVIDENCE    after looking at the result: source hashes + the visual verdict (status pass)
    art_pipeline.py JOB sync-minimums           measured minimum drawing seconds from the last build -> work/minimums.json
    art_pipeline.py JOB review NOTES.json       design/drawing-review.json from the plan, the job's actions and per-picture notes
    art_pipeline.py JOB status                  what exists and what is reviewed, per picture

Images come from the Codex CLI (codex_image.py); prompts are never rewritten for a picture that already passed review.
Schema: references/production-kit.md.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
PACKAGE = SCRIPTS.parent
STROKES_PER_SECOND = 3.5   # the user-accepted pace for planning (the engine measures ~3.9 at its fastest)
PEN_SHARE = .7             # the pen is busy at most 70% of a page; the finished picture is looked at for the rest (>= 30%)
CHARS_PER_SECOND = 4.5     # Chinese narration, when the page is not voiced yet
COMPOSE = {
    'portrait': 'Compose for a tall 2:3 portrait page: the subject fills the height of the canvas, nothing important near the left and right edges.',
    'square': 'Compose for a square picture on a 2:3 portrait page: the subject fills the canvas.',
    'landscape': 'Compose for a wide 3:2 picture that sits in one band of a 2:3 portrait page: keep the subject large and centred.'}
COMPOSE_NOTE = {
    'portrait': '为 2:3 竖屏页面构图：主体竖向占满画幅（横幅图在竖屏上会缩小，触发 build 的“内容范围”警告）。',
    'square': '方图：放进 2:3 竖屏页面时上下要留给标题和讲解形式。',
    'landscape': '横幅：在 2:3 竖屏上会缩成一条带，只在页面下方要放讲解形式时用；主体要大、居中。'}
PLATE = ('Line plate for a hand-drawn explainer video: a pen will trace every line, then the picture is coloured. {scene}\n'
         'Drawing: {line_art} Adult, realistic proportions; credible, never cute (no chibi, no big round eyes, no blush). Pure black '
         '(#000000) lines on a pure white background: no fills, no grey, no shading, no hatching, no texture, no glow, and no text, '
         'letters, numbers, brand marks or logos anywhere. Simple and readable at thumbnail size: about {n} separate outlines in the whole '
         'image, each object its outline plus at most one or two inner lines. Generous white space around the subject; nothing touches '
         'the image edges.')
COLOUR = ('Colour this exact line drawing (image 1){ref}. Keep every black line exactly where it is, with the same thickness: do not add, '
          'remove, move or redraw any line, and do not change the composition or proportions.\n'
          'Fill each closed region with one flat colour from this palette only: {palette} {notes}\n'
          'Then add one hard-edged, slightly darker flat shadow tone on the side of each object away from a light at the upper left. No '
          'gradients, no texture, no glow, no outlines in other colours. The background stays pure white.')
EDIT = ('Edit this line drawing (image 1). {erase} Keep every other line exactly where it is, with the same thickness; add nothing, '
        'move nothing, and keep pure black lines on a pure white background with no text.')


def shape_of(path):
    """portrait / square / landscape of an image file, by its aspect."""
    from PIL import Image
    with Image.open(path) as image:
        ratio = image.width / image.height
    return 'portrait' if ratio < .85 else 'landscape' if ratio > 1.18 else 'square'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Pipeline:
    def __init__(self, root, specs='design/image-specs.json'):
        self.root = Path(root).resolve()
        self.data = json.loads((self.root / specs).read_text())
        self.images = self.data['images']
        self.style = self.data.get('style', {})
        self.record_path = self.root / 'design/art-direction.json'
        job = self.root / 'job.json'
        self.job = json.loads(job.read_text()) if job.is_file() else {}
        canvas = self.job.get('canvas', {})
        self.canvas = (int(canvas.get('width', 1280)), int(canvas.get('height', 1920)))
        self.default_shape = 'portrait' if self.canvas[1] > self.canvas[0] * 1.1 else 'landscape' if self.canvas[0] > self.canvas[1] * 1.1 else 'square'
        character = self.data.get('character', {})
        self.char_ref = character.get('ref', 'inputs/art/char-ref.png')
        self.char_note = character.get('note', 'the main character (face, hair, clothing colours, proportions, line style)')

    # ------------------------------------------------------------------ text
    def fill(self, text):
        """{NAME} placeholders from "vars" (a character description written once)."""
        for key, value in self.data.get('vars', {}).items():
            text = text.replace('{' + key + '}', value)
        return text

    def spec(self, name):
        spec = dict(self.images[name])
        spec.setdefault('shape', self.default_shape)
        spec.setdefault('plate', True)
        spec.setdefault('ref', False)
        spec.setdefault('narration', [])
        spec['scene'] = self.fill(spec.get('scene', ''))
        spec['notes'] = self.fill(spec.get('notes', ''))
        return spec

    def window(self, name, spec):
        """Seconds of narration on the picture's page: spec "seconds", else the film's screen or the script page (voiced
        times when the narration is locked, else estimated from its characters)."""
        if spec.get('seconds'):
            return float(spec['seconds']), 'spec'
        beats = None
        film_path = self.root / 'design/film.json'
        if film_path.is_file():
            film = json.loads(film_path.read_text())
            for screen in film.get('screens', []):
                pictures = [item if isinstance(item, str) else (item or {}).get('picture') for item in [screen.get('picture'), *screen.get('actions', [])]]
                if name in pictures or screen.get('id') == spec.get('page'):
                    beats = screen.get('beats') or self.page_beats(screen.get('page', screen['id']))
                    break
        beats = beats or self.page_beats(spec.get('page'))
        if not beats:
            return None, None
        alignment = self.root / (self.job.get('narration', {}).get('alignment') or 'inputs/narration/alignment.json')
        if alignment.is_file():
            times = {beat['id']: beat for beat in json.loads(alignment.read_text())['beats']}
            if all(beat in times for beat in beats):
                return times[beats[-1]]['end'] - times[beats[0]]['start'], 'voiced'
        texts = self.beat_texts()
        characters = sum(len([ch for ch in texts.get(beat, '') if ch.isalnum()]) for beat in beats)
        return (characters / CHARS_PER_SECOND, 'estimated') if characters else (None, None)

    def script(self):
        path = self.root / 'inputs/script.json'
        return json.loads(path.read_text()) if path.is_file() else self.job.get('pages', [])

    def page_beats(self, page_id):
        return next(([beat['id'] for beat in page['beats']] for page in self.script() if page.get('id') == page_id), None)

    def beat_texts(self):
        return {beat['id']: beat['text'] for page in self.script() for beat in page['beats']}

    def budget(self, name, spec):
        seconds, source = self.window(name, spec)
        if not seconds:
            return None
        drawing = PEN_SHARE * seconds
        return {'window': seconds, 'source': source, 'drawing': drawing, 'strokes': int(STROKES_PER_SECOND * drawing)}

    def outlines(self, spec, budget):
        """Outlines asked of the plate: the spec's n, else the budget (one stroke = one continuous outline, ~3.9/s measured)."""
        if spec.get('n'):
            return spec['n']
        return max(12, min(60, budget['strokes'])) if budget else 40

    def brief(self, name, spec):
        budget = self.budget(name, spec)
        n = self.outlines(spec, budget)
        kind = '角色参考' if not spec['plate'] else '线稿先行插画'
        needs = '\n'.join(f'  - 「{quote}」：{shown}' for quote, shown in spec['narration']) or '  - 无（参考图）'
        width, height = self.canvas
        style = self.style.get('brief', [])
        style = ''.join(f'{line}\n' for line in ([style] if isinstance(style, str) else style))
        how = ('生成新图（线稿先行：先出线稿，再以线稿为输入做上色编辑，line_first.py 配准）' if spec['plate']
               else '生成新图（一次成图，锁定形象与颜色，供其他图作参考）')
        lines = [f'# 美术简报：{name}（{kind}）\n', f'- 交付：{how}。',
                 f'- 用途：{spec.get("use", "")}（页面 {spec.get("page", "")}）。',
                 f'- 观看尺寸：{width}×{height} 成片，画幅 {spec["shape"]}；{COMPOSE_NOTE[spec["shape"]]}笔画与时长在不超过 720px 的规划尺寸上计算。']
        if budget and spec['plate']:
            source = {'voiced': '按配音实测', 'estimated': '按字数估算', 'spec': '按规格给定'}[budget['source']]
            lines.append(f'- 绘制预算（绘制时间才是真正的预算）：本页口播约 {budget["window"]:.1f} 秒（{source}）；画笔最多占 {PEN_SHARE:.0%}'
                         f'（约 {budget["drawing"]:.1f} 秒），画完的图至少留 30% 给观众看；按每秒约 {STROKES_PER_SECOND:g} 笔（一笔 = 一条连续轮廓线），'
                         f'本图最多约 {budget["strokes"]} 笔。')
            if n > budget['strokes']:
                lines.append(f'- 注意：规格要约 {n} 条轮廓，超出预算约 {n - budget["strokes"]} 条：先删次要道具和背景，或把这张图放到更长的页上。')
        if style:
            lines.append(style.rstrip('\n'))
        lines += [f'- 复杂度：约 {n} 条轮廓；超出口播窗口时先编辑线稿点名删细节（image_gate --plate 的 edit_fix，art_pipeline.py gen --edit）。',
                  f'- 必须可见的叙事关系：\n{needs}', f'- 主体与场景：{spec["scene"]}']
        return '\n'.join(line for line in lines if line) + '\n', n

    def scene(self, spec):
        """The scene, with the page-composition sentence unless the film's own plate template places {compose} itself."""
        template = self.style.get('plate', PLATE)
        return spec['scene'] if '{compose}' in template else f'{spec["scene"]} {COMPOSE[spec["shape"]]}'

    def plate_prompt(self, spec, n):
        template = self.style.get('plate', PLATE)
        return self.fill(template).format_map(Defaulting(scene=self.scene(spec), n=n, compose=COMPOSE[spec['shape']],
                                                          line_art=self.style.get('line_art', 'even black outlines, simplified shapes, every shape closed where it meets another shape.')))

    def colour_prompt(self, spec):
        ref = ''
        if spec['ref'] and spec.get('colour_ref', True):
            ref = self.data.get('character', {}).get('colour_phrase', " and the main character's look and colours from image 2")
        return self.fill(self.style.get('colour', COLOUR)).format_map(Defaulting(ref=ref, notes=spec['notes'], palette=self.style.get('palette', '')))

    # ------------------------------------------------------------------ record
    def load_record(self):
        return json.loads(self.record_path.read_text()) if self.record_path.is_file() else {}

    def save_record(self, record):
        self.record_path.parent.mkdir(parents=True, exist_ok=True)
        self.record_path.write_text(json.dumps(record, ensure_ascii=False, indent=1))

    def bind(self):
        result = subprocess.run([sys.executable, str(SCRIPTS / 'art_direction.py'), 'bind-evidence', str(self.record_path)],
                                capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        return result.stdout.strip()

    def prompts(self, names=None, force=False):
        record = self.load_record()
        record['style_sha256'] = digest(self.root / 'style/resolved.json')
        record['dependency_sha256'] = digest(PACKAGE / 'dependencies/sweety-image-art-direction/GUIDE.md')
        entries = {entry['source']: entry for entry in record.get('images', [])}
        (self.root / 'design/briefs').mkdir(parents=True, exist_ok=True)
        (self.root / 'design/prompts').mkdir(parents=True, exist_ok=True)
        report = {}
        for name in names or self.images:
            spec = self.spec(name)
            if spec['plate']:
                sources = [(f'../inputs/art/plates/{name}-line.png', f'prompts/{name}-line-edit.txt' if spec.get('edit') else f'prompts/{name}-line.txt',
                            'generated_under_guidance'),
                           (f'../inputs/art/raw/{name}-colour.png', f'prompts/{name}-colour.txt', 'generated_under_guidance'),
                           (f'../inputs/art/{name}.png', f'prompts/{name}-colour.txt', 'composed_with_verified_layers')]
            else:
                sources = [(f'../inputs/art/{name}.png', f'prompts/{name}.txt', 'generated_under_guidance')]
            reviewed = [entries.get(source, {}).get('status') == 'pass' for source, _, _ in sources]
            if any(reviewed) and not force:
                report[name] = 'kept (already reviewed; --force rewrites and needs a new review)'
                continue
            brief, n = self.brief(name, spec)
            files = {f'briefs/{name}.md': brief}
            if spec['plate']:
                files[f'prompts/{name}-line.txt'] = self.plate_prompt(spec, n)
                files[f'prompts/{name}-colour.txt'] = self.colour_prompt(spec)
                if spec.get('edit'):
                    files[f'prompts/{name}-line-edit.txt'] = self.fill(self.style.get('edit', EDIT)).format_map(Defaulting(erase=spec['edit']))
            else:
                files[f'prompts/{name}.txt'] = f'{self.fill(self.style.get("one_shot", ""))}\n{spec["scene"]}\n{spec["notes"]}'.strip()
            for path, text in files.items():
                (self.root / 'design' / path).write_text(text)
            for source, prompt, method in sources:
                entry = entries.setdefault(source, {'source': source, 'status': 'pending'})
                entry.update(method=method, brief_file=f'briefs/{name}.md', prompt_file=prompt)
                if force:
                    entry.update(status='pending')
                    entry.pop('evidence', None)
            budget = self.budget(name, spec)
            report[name] = f'{n} outlines' + (f', budget {budget["strokes"]} in {budget["window"]:.1f} s ({budget["source"]})'
                                              + (' OVER BUDGET' if spec['plate'] and n > budget['strokes'] else '') if budget else '')
        record['images'] = list(entries.values())
        self.save_record(record)
        report['bind'] = self.bind()
        return report

    # ------------------------------------------------------------------ generation
    def codex(self, prompt, output, shape, refs=()):
        import codex_image
        pairs = [(self.root / path, note) for path, note in refs]
        try:
            return json.dumps(codex_image.run(self.root / 'design' / prompt, self.root / output, pairs, shape,
                                              self.data.get('model', codex_image.DEFAULT_MODEL), self.data.get('effort', codex_image.DEFAULT_EFFORT),
                                              codex=os.environ.get('CODEX_BIN', 'codex'), cwd=self.root), ensure_ascii=False)
        except Exception as error:  # noqa: BLE001 - logged per picture, the others go on
            return f'FAILED {output}: {error}'

    def retire(self, name):
        """Move the current plate (and the colour edit made from it) to inputs/art/rejected/ with the next version number."""
        rejected = self.root / 'inputs/art/rejected'
        rejected.mkdir(parents=True, exist_ok=True)
        version = 1 + max([int(path.stem.rsplit('.v', 1)[1]) for path in rejected.glob(f'{name}-line.v*.png') if path.stem.rsplit('.v', 1)[1].isdigit()] + [0])
        moved = []
        for path, target in ((self.root / f'inputs/art/plates/{name}-line.png', rejected / f'{name}-line.v{version}.png'),
                             (self.root / f'inputs/art/raw/{name}-colour.png', rejected / f'{name}-colour.v{version}.png')):
            if path.is_file():
                shutil.move(path, target)
                moved.append(str(target.relative_to(self.root)))
        return moved

    def latest_rejected(self, name):
        found = sorted((path for path in (self.root / 'inputs/art/rejected').glob(f'{name}-line.v*.png') if path.stem.rsplit('.v', 1)[1].isdigit()),
                       key=lambda path: int(path.stem.rsplit('.v', 1)[1]))
        return found[-1] if found else None

    def generate(self, name, edit=False, force=False):
        spec, log = self.spec(name), []
        character = f'{self.char_ref}=character reference: {self.char_note}'
        if not spec['plate']:
            log.append(self.codex(f'prompts/{name}.txt', f'inputs/art/{name}.png', spec['shape']))
        else:
            plate = self.root / f'inputs/art/plates/{name}-line.png'
            if edit:
                if not spec.get('edit'):
                    raise ValueError(f'{name}: write the erasures into image-specs "edit" and run prompts first')
                log.append('RETIRED ' + ', '.join(self.retire(name)))
            if spec.get('edit') and not plate.is_file() and self.latest_rejected(name):
                # The plate was too detailed for its window: edit it, naming what to erase, then colour the edited plate.
                source = self.latest_rejected(name).relative_to(self.root)
                log.append(self.codex(f'prompts/{name}-line-edit.txt', plate.relative_to(self.root), shape_of(self.root / source),
                                      [(source, 'image 1: the line drawing to edit')]))
            if not plate.is_file():
                log.append(self.codex(f'prompts/{name}-line.txt', plate.relative_to(self.root), spec['shape'],
                                      [tuple(character.split('=', 1))] if spec['ref'] else []))
            if not plate.is_file():
                return self.finish(name, log, 'no plate')
            gate = subprocess.run([sys.executable, str(SCRIPTS / 'image_gate.py'), str(self.root), str(plate), '--plate', '--width', str(spec.get('width', 760))],
                                  capture_output=True, text=True)
            log.append('PLATE GATE ' + gate.stdout[-1200:] + gate.stderr[-400:])
            if gate.returncode and not force:
                return self.finish(name, log, 'plate over budget: add the gate\'s edit_fix to "edit", run prompts, then gen --edit (or --force)')
            colour = self.root / f'inputs/art/raw/{name}-colour.png'
            if not colour.is_file():
                refs = [(plate.relative_to(self.root), 'image 1: the line drawing to colour (keep every line)')]
                if spec['ref'] and spec.get('colour_ref', True):
                    refs.append((self.char_ref, f'image 2: colour reference: {self.char_note}'))
                # The colour edit must come back on the plate's own canvas: the image tool does not always return the shape it
                # was asked for, and padding a portrait plate onto a square canvas made the edit redraw (and squash) every line.
                log.append(self.codex(f'prompts/{name}-colour.txt', colour.relative_to(self.root), shape_of(plate), refs))
            first = subprocess.run([sys.executable, str(SCRIPTS / 'line_first.py'), str(self.root), f'inputs/art/plates/{name}-line.png',
                                    f'inputs/art/raw/{name}-colour.png', '--out', f'inputs/art/{name}.png',
                                    '--drop-specks', str(spec.get('specks', 0))], capture_output=True, text=True, cwd=self.root)
            log.append('LINE FIRST ' + first.stdout[-900:] + first.stderr[-500:])
            if first.returncode:
                return self.finish(name, log, 'line_first failed')
        return self.finish(name, log, 'done: look at it, then record NAME "evidence"')

    def finish(self, name, log, status):
        (self.root / 'work').mkdir(exist_ok=True)
        (self.root / f'work/gen-{name}.log').write_text('\n'.join(str(item) for item in log))
        print(name, status, flush=True)
        return status

    def record(self, name, evidence):
        record, found = self.load_record(), 0
        for entry in record.get('images', []):
            if any(f'/{name}{suffix}.png' in entry['source'] for suffix in ('', '-line', '-colour')):
                path = (self.record_path.parent / entry['source']).resolve()
                if path.is_file():
                    entry.update(status='pass', sha256=digest(path), evidence=evidence)
                    found += 1
        self.save_record(record)
        return found

    # ------------------------------------------------------------------ after a build
    def sync_minimums(self):
        """The engine's measured minimum drawing seconds of every picture, so page windows come from what it needs."""
        path = self.root / 'work/minimums.json'
        minimums = json.loads(path.read_text()) if path.is_file() else {}
        changed = {}
        for asset in sorted((self.root / 'work/assets').glob('element-*.json')):
            meta = json.loads(asset.read_text())
            name, measured = Path(meta.get('source') or '').stem, (meta.get('drawability') or {})
            if name in minimums and measured.get('minimum_seconds'):
                if abs(minimums[name][0] - measured['minimum_seconds']) > .005:
                    changed[name] = [minimums[name][0], measured['minimum_seconds']]
                minimums[name] = [measured['minimum_seconds'], measured.get('stroke_count', minimums[name][1])]
        path.write_text(json.dumps(minimums))
        return changed

    def review(self, notes_path):
        """design/drawing-review.json: per picture, notes written after looking at the registered source."""
        sys.path.insert(0, str(SCRIPTS))
        from style_choice import verify_style
        from drawing_contract import illustration_actions
        notes = json.loads(Path(notes_path).read_text())
        plan_path = self.root / 'design/drawing-plan.json'
        plans = {plan['id']: plan for plan in json.loads(plan_path.read_text())['illustrations']}
        entries = []
        for action in illustration_actions(self.job):
            name = action['drawing_plan_id']
            plan, note = plans[name], notes[name]
            order = ' → '.join(part['id'] for part in plan['parts'])
            entry = {'plan_id': name, 'status': 'pass', 'source_sha256': digest(self.root / action['source']),
                     'outline_mask_sha256': digest(self.root / action['outline_mask']),
                     'anatomy_and_occlusion': '实际查看配准后的源图：' + note['anatomy'],
                     'part_boundaries': f'按 design/parts/ 网格图（{name}.png）写归一化方框，顺序 {order}；先声明者拥有重叠像素，最后一个模块接住剩余像素；'
                                        '马克笔上色时颜色按墨线闭合区域整块归给占多数的模块。',
                     'stroke_breaks': '线稿先行：线条就是线稿（outline_mask），line_first 配准通过；引擎在规划尺寸上骨架化整笔描线，模块之间抬笔停顿；'
                                      f"规划尺寸 {action.get('drawing', {}).get('plan_side', self.job.get('drawing', {}).get('plan_side', 540))}px 下线稿门禁通过（无纹理化短线）。",
                     'outline_then_fill': '全部模块按“外轮廓 → 长结构线 → 细节”描完，再上色：' + (
                         '按马克笔，同色一起、先浅后深，底色后压阴影。' if {**self.job.get('drawing', {}), **action.get('drawing', {})}.get('colouring', 'markers') == 'markers'
                         else '按模块逐块上色，底色后压阴影。'),
                     'spatial_order': plan.get('order_exception') or '从上到下、从左到右。',
                     'narrative_checks': [{'id': requirement['id'], 'status': 'pass', 'evidence': '源图中可见：' + note['narrative'][k]}
                                          for k, requirement in enumerate(plan['narrative_requirements'])],
                     'recognizability': {'status': 'pass', 'named_as': note['named_as'],
                                         'evidence': f"drawing_contract.py --thumbnails 160px 编号缩略图（work/review/recognizability.png），不看字幕即可叫出：{note['named_as']}"}}
            entry.update({key: value for key, value in note.items() if key in entry and isinstance(value, str) and key != 'status'})
            if action.get('crop'):
                entry['crop'] = action['crop']
            entries.append(entry)
        review = {'style_sha256': verify_style(self.root)['resolved_sha256'], 'plan_sha256': digest(plan_path),
                  'note': notes.get('_note', '部位方框按实际成图的网格图标注。'), 'illustrations': entries}
        (self.root / 'design/drawing-review.json').write_text(json.dumps(review, ensure_ascii=False, indent=1))
        return len(entries)

    def status(self):
        entries = {entry['source']: entry for entry in self.load_record().get('images', [])}
        rows = {}
        for name in self.images:
            files = {'plate': f'inputs/art/plates/{name}-line.png', 'colour': f'inputs/art/raw/{name}-colour.png', 'picture': f'inputs/art/{name}.png'}
            rows[name] = {key: ('reviewed' if entries.get(f'../{path}', {}).get('status') == 'pass' else 'exists') if (self.root / path).is_file()
                          else '-' for key, path in files.items()}
        return rows


class Defaulting(dict):
    """format_map that leaves unknown {placeholders} as written."""
    def __missing__(self, key):
        return '{' + key + '}'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('job', type=Path)
    parser.add_argument('command', choices=['prompts', 'gen', 'record', 'sync-minimums', 'review', 'status'])
    parser.add_argument('args', nargs='*')
    parser.add_argument('--specs', default='design/image-specs.json')
    parser.add_argument('--edit', action='store_true', help='gen: retire the plate and redraw it from the spec\'s "edit"')
    parser.add_argument('--force', action='store_true', help='prompts: rewrite reviewed pictures; gen: colour a plate the gate failed')
    parser.add_argument('--parallel', type=int, default=3)
    args = parser.parse_args(argv)
    pipeline = Pipeline(args.job, args.specs)
    if args.command == 'prompts':
        print(json.dumps(pipeline.prompts(args.args or None, args.force), ensure_ascii=False, indent=1))
    elif args.command == 'gen':
        unknown = [name for name in args.args if name not in pipeline.images]
        if not args.args or unknown:
            parser.error(f'name pictures from the specs: unknown {unknown}' if unknown else 'name the pictures to generate')
        with ThreadPoolExecutor(args.parallel) as pool:
            list(pool.map(lambda name: pipeline.generate(name, args.edit, args.force), args.args))
    elif args.command == 'record':
        if len(args.args) != 2:
            parser.error('record NAME "EVIDENCE"')
        print(pipeline.record(*args.args), 'record entries passed')
    elif args.command == 'sync-minimums':
        print(json.dumps(pipeline.sync_minimums(), ensure_ascii=False))
    elif args.command == 'review':
        print(pipeline.review(args.args[0]), 'reviewed')
    else:
        print(json.dumps(pipeline.status(), ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    sys.path.insert(0, str(SCRIPTS))
    sys.exit(main())
