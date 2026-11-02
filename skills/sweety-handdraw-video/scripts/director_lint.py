#!/usr/bin/env python3
"""Director checks for attention and emotional continuity, run after the narration script and design.json exist (and
again after the narration is locked, when beat times can be measured).

    director_lint.py JOB

Why: past films lost viewers in the first seconds, told stories that did not land, and broke their own mood with a
stray sound. These are the 2025-2026 practices of retaining explainer accounts turned into checks (sources in
references/opening-design.md and story-design.md); each check names the fix. Errors stop production; warnings are for
the director to answer in design.json. Nothing here measures whether a film is good: only real viewing and platform
data do that.
"""
import argparse
import json
import re
import sys
from pathlib import Path

FIRST_LINE_CHARS = 25        # a hook line a viewer hears in about 3 seconds
COVER_TITLE_CHARS = 14       # a title read in the cover's first second
GREETING = re.compile(r'^(大家好|哈喽|嗨|今天(我们|给大家|来)|我们来(聊|讲)|欢迎(来到|收看)|在.{0,8}(长河|时代)中)')
ENDING_COLLAPSE = re.compile(r'今天就(到这|先到这)|以上就是|我们下期再见|下期见|感谢(观看|收看)|好了，就')
AI_TONE = [(re.compile(r'不是[^。！？]{1,16}[，,]?而是'), '“不是……而是……”'), (re.compile(r'值得一提的是'), '“值得一提的是”'),
           (re.compile(r'总而言之|综上所述'), '“总而言之/综上所述”'), (re.compile(r'在这个.{0,6}的时代'), '“在这个……的时代”'),
           (re.compile(r'赋能|抓手|底层逻辑|闭环'), '空泛术语'), (re.compile(r'让我们一起'), '“让我们一起”')]
ORDINAL = re.compile(r'第[一二三四五六七八九十百\d]+[条点个步招]')
REHOOK_SECONDS = 75          # longest stretch without a re-hook in a list longer than this
# Motion forms that explain (not titles or circles): a viewer sees the structure, the comparison, the process.
FORMS = {'table', 'flow', 'swimlane', 'checklist', 'cards', 'pulse', 'rough_chart', 'number_ticker', 'morph', 'physics', 'map_route',
         'spin3d', 'formula', 'progress_ring'}
MIN_FORMS = 4                # distinct explanation forms in a film over two minutes
PLAIN_SHARE = .3             # pages with a picture and words but no explanation form, at most this share
# What the narration says, and the forms that show it (heuristic: warnings for the director to answer).
# Time words alone (以后, 半年前) are not a comparison: a then/now contrast needs both sides on the page.
CUES = [(re.compile(r'对比|相比|比起|便宜|更贵|而不是|vs|比[^，。！？]{1,8}[更还]|(?:以前|过去|当年)[^。！？]{0,30}(?:现在|如今)', re.I),
         {'table', 'cards', 'rough_chart', 'swimlane'}, 'a comparison: a table or side-by-side shows it'),
        (re.compile(r'先.{0,12}再|流程|跑通|部署|循环|链路|一步步'), {'flow', 'swimlane'}, 'a process: a flow diagram shows it'),
        (re.compile(r'[^、，。：]{1,14}、[^、，。：]{1,14}、'), {'checklist', 'table', 'flow', 'cards'}, 'a list: a checklist or table shows it'),
        (re.compile(r'等你|等它|定时|小时后|异步|心跳|唤醒'), {'swimlane', 'pulse', 'rough_chart'}, 'waiting or a schedule: a swimlane or timeline shows it'),
        # “千万不要” is an adverb, not a number
        (re.compile(r'\d+|[一二两三四五六七八九十百千万]+(?:倍|万|千|成)(?![不别要])'), {'rough_chart', 'number_ticker', 'cards', 'table'}, 'a number: a chart or counter shows it')]


def text_of(manifest):
    return [(beat['id'], beat['text']) for page in manifest['pages'] for beat in page['beats']]


def check(root):
    root = Path(root)
    manifest = json.loads((root / 'job.json').read_text())
    design = json.loads((root / manifest.get('design_file', 'design.json')).read_text())
    beats = text_of(manifest)
    errors, warnings = [], []
    first_id, first = beats[0]
    first_sentence = re.split(r'[。！？?!]', first)[0]
    clean = lambda value: re.sub(r'[\s，。、！？,.!?：:“”"\'·]', '', value)
    if len(clean(first_sentence)) > FIRST_LINE_CHARS:
        errors.append(f'{first_id}: the first sentence has {len(clean(first_sentence))} characters; a hook line is heard in about 3 s (<= {FIRST_LINE_CHARS})')
    if GREETING.search(first):
        errors.append(f'{first_id}: opens with a greeting or scene-setting ({GREETING.search(first).group(0)}); open on the conflict, question or result')
    hook = design.get('opening_hook', {})
    if hook.get('spoken_text') and clean(hook['spoken_text']) not in clean(first):
        errors.append('opening_hook.spoken_text is not the first narration line any more; update the design or the script')
    title = ''.join(design.get('cover', {}).get('title', []) if isinstance(design.get('cover', {}).get('title'), list) else [design.get('cover', {}).get('title', '')])
    if title:
        if len(clean(title)) > COVER_TITLE_CHARS:
            warnings.append(f'cover title has {len(clean(title))} characters: a feed viewer reads about {COVER_TITLE_CHARS} in the first second')
        shared = set(clean(title)) & set(clean(first_sentence))
        if len(shared) < .5 * len(set(clean(title))):
            warnings.append('cover title and first sentence say different things: the eye and the ear should get one hook (Kallaway 2025)')
    from align import english_words
    from narration_master import spoken_text
    english = {}
    for page in manifest['pages']:
        for word in (word for beat in page['beats'] for word in english_words(spoken_text(manifest, beat))):
            english[word] = english.get(word, 0) + 1
    if english:
        listed = ', '.join(f'{word} ×{count}' for word, count in sorted(english.items(), key=lambda item: -item[1]))
        warnings.append(f'the narration says {sum(english.values())} English words ({listed}): a Chinese voice often says them so an English ear '
                        'does not hear them (the v3 film: bug, git, Amp, Thorsten Ball). Use the Chinese term where there is a natural one and '
                        'transliterated names: narration.say {"Grok": "格罗克"} keeps the subtitle as written and only changes what is read. '
                        'Probe the rest before voicing (voxcpm_voice.py JOB --probe-terms); they are also checked word by word when voiced')
    for beat_id, text in beats:
        for pattern, label in AI_TONE:
            if pattern.search(text):
                warnings.append(f'{beat_id}: {label} reads as machine-written; say it plainly')
    last_page = manifest['pages'][-1]
    for beat in last_page['beats']:
        if ENDING_COLLAPSE.search(beat['text']):
            errors.append(f'{beat["id"]}: “{ENDING_COLLAPSE.search(beat["text"]).group(0)}” announces the end and viewers leave before it; end on the payoff')
    ordinals = [beat_id for beat_id, text in beats if ORDINAL.search(text)]
    plan = design.get('retention_plan') or {}
    if len(ordinals) >= 7:
        if len(plan.get('rehooks', [])) < 2:
            errors.append(f'a {len(ordinals)}-item list needs retention_plan.rehooks (>= 2): group it, say where we are, and re-open the question')
        if not plan.get('progress'):
            warnings.append('a long list should show progress (for example “十五条过半”)')
    if not plan.get('emotion_arc'):
        warnings.append('retention_plan.emotion_arc is missing: write how the feeling moves page by page, so sound and pictures can keep it')
    if not plan.get('ending'):
        warnings.append('retention_plan.ending is missing: callback to the hook, a line worth sharing, a frame worth saving')
    rehooks = [item.get('beat') for item in plan.get('rehooks', [])]
    alignment_path = root / manifest.get('narration', {}).get('alignment', 'inputs/narration/alignment.json')
    if alignment_path.is_file() and rehooks:
        times = {beat['id']: beat['start'] for beat in json.loads(alignment_path.read_text())['beats']}
        marks = [0.] + sorted(times[beat] for beat in rehooks if beat in times) + [max(times.values())]
        longest = max(b - a for a, b in zip(marks, marks[1:]))
        if longest > REHOOK_SECONDS and len(ordinals) >= 7:
            warnings.append(f'{longest:.0f} s pass without a re-hook; add one within about {REHOOK_SECONDS} s')
    sound = design.get('sound_design') or {}
    audio = manifest.get('audio', {})
    if not sound:
        warnings.append('design.sound_design is missing: decide the bed, the few sound effects and the pen sound before mixing')
    sfx = audio.get('sfx', True)
    if (sfx is True or (isinstance(sfx, dict) and sfx.get('accents', True))) and not sound.get('accents_reason'):
        warnings.append('audio.sfx accents (chimes, thumps) are on: they break a calm explanation; turn them off or give sound_design.accents_reason')
    from sfx import QUIET_MOVES, TRANSITION_SOUNDS
    quiet_moves = isinstance(sfx, dict) and sfx.get('quiet_moves', False)
    sounded = [page.get('transition', {}).get('type') for page in manifest['pages'][1:] if isinstance(page.get('transition'), dict)]
    loud = [kind for kind in sounded if TRANSITION_SOUNDS.get(kind) and (quiet_moves or kind not in QUIET_MOVES)]
    if len(loud) > 6:
        warnings.append(f'{len(loud)} page transitions carry a sound; keep 3-5 sounding transitions (chapter turns), the rest silent')
    music = audio.get('music')
    if isinstance(music, dict) and music.get('lyrics'):
        errors.append('music with lyrics competes with the narration for words; use an instrumental bed')
    hierarchy(manifest, ordinals, rehooks, errors, warnings)
    forms = form_fit(manifest, errors, warnings, alignment_path)
    page_checks(root, manifest, errors, warnings, alignment_path)
    return {'errors': errors, 'warnings': warnings, 'list_items': len(ordinals), 'rehooks': rehooks, 'forms': forms}


def overlays(page, recipe):
    return [item for item in page.get('overlays', []) if item.get('recipe') == recipe]


def hierarchy(manifest, ordinals, rehooks, errors, warnings):
    """A long list must show its structure, or a chapter title reads as one more item and the viewer is lost (the first
    v3 film: “大标题的时候还疑惑怎么讲了标题不讲内容”). Map first, chapters as chapters, and on every item page where it is."""
    if len(ordinals) < 7:
        return
    pages = manifest['pages']
    page_of = {beat['id']: index for index, page in enumerate(pages) for beat in page['beats']}
    label = lambda index: pages[index].get('id', f'page {index + 1}')
    full = lambda index: any(item.get('params', {}).get('mode', 'full') != 'strip' for item in overlays(pages[index], 'roadmap'))
    strip = lambda index: any(item.get('params', {}).get('mode') == 'strip' for item in overlays(pages[index], 'roadmap'))
    first = page_of[ordinals[0]]
    if not any(full(index) for index in range(first + 1)):
        errors.append(f'a {len(ordinals)}-item list needs its map before the first item: a roadmap overlay (groups, numbered items) '
                      'on the hook, so every later title has a place to belong')
    for beat in rehooks:
        if beat in page_of and not full(page_of[beat]):
            errors.append(f'{beat}: a chapter turn shows the full roadmap with its group lit, on a page of its own; '
                          'otherwise the chapter title reads as one more item')
    items = sorted({page_of[beat] for beat in ordinals})
    missing = [label(index) for index in items if not strip(index)]
    if missing:
        errors.append('item pages need the progress strip (roadmap mode "strip": done, this one, still to come): ' + ', '.join(missing))
    unnumbered = [label(index) for index in items if not any(item.get('params', {}).get('badge') is not None for item in overlays(pages[index], 'kinetic_title'))]
    if unnumbered:
        warnings.append('item titles should carry their number (kinetic_title "badge"), a level below chapter titles: ' + ', '.join(unnumbered))


def form_fit(manifest, errors, warnings, alignment_path):
    """Each page shows its content in the form the content has. Counts the explanation forms and flags pages whose words
    call for a form they do not use; a picture alone, however good, cannot show a comparison or a process."""
    pages = manifest['pages']
    # A custom recipe (model-written SVG) counts when it names the element it draws and the mechanism it explains: a light
    # animation of how something works (light that moves with the car, a stopping-distance ruler) is a form, decoration is not.
    explains = lambda item: item.get('recipe') == 'custom' and (item.get('params') or {}).get('element') and (item.get('params') or {}).get('explains')
    # Each custom animation is its own form (a prism is not a commit graph): it counts by the element it draws.
    used = [({item.get('recipe') for item in page.get('overlays', [])} & FORMS)
            | {'custom:' + str((item.get('params') or {}).get('element', ''))[:24] for item in page.get('overlays', []) if explains(item)}
            for page in pages]
    seconds = 0.
    if alignment_path.is_file():
        beats = json.loads(alignment_path.read_text())['beats']
        seconds = max(beat['end'] for beat in beats) if beats else 0.
    distinct = set().union(*used) if used else set()
    if seconds > 120 and len(distinct) < MIN_FORMS:
        warnings.append(f'{len(distinct)} explanation forms over {seconds:.0f} s ({", ".join(sorted(distinct)) or "none"}): '
                        f'use at least {MIN_FORMS} (table, flow, swimlane, checklist, cards, chart...) chosen by what each page says')
    pictured = [index for index, page in enumerate(pages) if any(action.get('type') == 'illustration' for group in page.get('visual_groups', [])
                                                                 for action in group.get('actions', []))]
    plain = [pages[index].get('id', f'page {index + 1}') for index in pictured if not used[index]]
    if pictured and len(plain) > PLAIN_SHARE * len(pages):
        warnings.append(f'{len(plain)} of {len(pages)} pages are a picture and words only ({", ".join(plain)}): '
                        'add the form that explains each one, or a light animation of its mechanism')
    for index, page in enumerate(pages):
        text = ORDINAL.sub('', ''.join(beat['text'] for beat in page['beats']))
        shown = shown_numbers(page)
        for cue, fitting, what in CUES:
            match = cue.search(text)
            if not match or used[index] & fitting:
                continue
            said = spoken_numbers(' '.join(found.group(0) for found in cue.finditer(text))) if what.startswith('a number') else set()
            if said and said <= shown:
                continue  # the number is already on screen, written in the page's title or handwriting
            warnings.append(f'{page.get("id", f"page {index + 1}")}: the narration has {what} ({match.group(0)})')
    repetition(pages, used, warnings)
    return sorted(distinct)


NUMBER = re.compile(r'\d+(?:\.\d+)?[十百千万亿]*|[零一二两三四五六七八九十百千万亿点]+')


def spoken_numbers(text):
    """Values of the numbers a text says (七十万 and 70万 both give 700000); single small words like 一 are skipped."""
    from align import chinese_number
    values = set()
    for token in NUMBER.findall(re.sub(r'(?<=\d),(?=\d{3})', '', text)):
        if not any(ch.isdigit() for ch in token) and (len(token) < 2 or token in ('一一', '一点')):
            continue
        value = chinese_number(token)
        if re.fullmatch(r'\d+(?:\.\d+)?', value):
            values.add(value)
    return values


def shown_numbers(page):
    """Numbers already written on the page: titles, lines, labels and handwritten text."""
    texts = [str(item.get('params', {}).get(key, '')) for item in page.get('overlays', []) for key in ('text', 'title', 'caption', 'to')]
    texts += [str(action.get('text', '')) for group in page.get('visual_groups', []) for action in group.get('actions', [])]
    texts += [str(action.get('text', '')) for beat in page.get('beats', []) for action in beat.get('actions', [])]
    return set().union(*(spoken_numbers(text) for text in texts)) if texts else set()


REPEAT_RUN = 3     # the same explanation form on more pages in a row than this reads as a template
DOMINANT_SHARE = .6


def repetition(pages, used, warnings):
    """One form everywhere turns explanation into a template (the deposit film: six chapters, six picture-plus-checklist
    pages). Flags a run of the same form on more than REPEAT_RUN form pages in a row, or one form on most form pages."""
    formed = [(pages[index].get('id', f'page {index + 1}'), forms) for index, forms in enumerate(used) if forms]
    run, best = [], []
    for page_id, forms in formed:
        run = [(page_id, forms)] if not run or not (run[-1][1] & forms) else run + [(page_id, run[-1][1] & forms)]
        if len(run) > len(best):
            best = list(run)
    if len(best) > REPEAT_RUN:
        form = ', '.join(sorted(best[-1][1]))
        warnings.append(f'{len(best)} form pages in a row use {form} ({best[0][0]}..{best[-1][0]}): vary the form by what each page says '
                        '(a table for a comparison, a flow for a process, a chart for a trend)')
    if len(formed) >= 5:
        counts = {}
        for _, forms in formed:
            for form in forms:
                counts[form] = counts.get(form, 0) + 1
        form, count = max(counts.items(), key=lambda item: item[1])
        if count / len(formed) > DOMINANT_SHARE and len(best) <= REPEAT_RUN:
            warnings.append(f'{form} carries {count} of {len(formed)} form pages: the film reads as one template; vary the forms')


TITLE_INK_WARN, TITLE_INK_FAIL = .003, .02   # share of a title's text box covered by picture ink
BLANK_ARRIVAL = 1.0                          # seconds a turned page may show nothing but paper and the strip
FLAT_FILL = .45                              # one flat colour over this share of a large picture reads as an empty page


def picture_ink(root, action):
    """(mask, x, y) of the ink a picture lays on its page: the source fitted into its box from the top-left."""
    import numpy as np
    from PIL import Image
    image = Image.open(root / action['source']).convert('RGBA')
    if action.get('crop'):
        image = image.crop(tuple(action['crop']))
    scale = min(float(action['width']) / image.width, float(action['height']) / image.height)
    image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))))
    pixels = np.asarray(image)
    return (pixels[..., 3] > 32) & (pixels[..., :3].min(axis=2) < 232), float(action['x']), float(action['y']), pixels


def title_box(overlay):
    """Estimated box of a title's glyphs inside its area (centred, wrapped to the area width)."""
    params = overlay.get('params') or {}
    text, size = str(params.get('text', '')), float(params.get('size', 64))
    wide = lambda ch: '\u2e80' <= ch <= '\u9fff' or '\uff00' <= ch <= '\uffef' or '\u3000' <= ch <= '\u303f'
    natural = sum(size if wide(ch) else .3 * size if ch == ' ' else .58 * size for ch in text) + (1.4 * size if params.get('badge') is not None else 0)
    width = min(float(overlay['width']), natural)
    lines = max(1, -(-natural // max(1., float(overlay['width']))))
    height = min(float(overlay['height']), lines * size * 1.3)
    return float(overlay['x']) + (float(overlay['width']) - width) / 2, float(overlay['y']) + (float(overlay['height']) - height) / 2, width, height


def page_checks(root, manifest, errors, warnings, alignment_path):
    """What a viewer sees on each page: titles clear of the picture, no blank arrival after a turn, no page that is only
    words or one flat colour."""
    import numpy as np
    beats = {beat['id']: beat for beat in json.loads(alignment_path.read_text())['beats']} if alignment_path.is_file() else {}
    for index, page in enumerate(manifest['pages']):
        label = page.get('id', f'page {index + 1}')
        actions = [action for group in page.get('visual_groups', []) for action in group.get('actions', [])]
        pictures = [action for action in actions if action.get('type') == 'illustration' and all(k in action for k in ('x', 'y', 'width', 'height'))
                    and (root / action.get('source', '')).is_file()]
        titles = [item for item in page.get('overlays', []) if item.get('recipe') == 'kinetic_title' and all(k in item for k in ('x', 'y', 'width', 'height'))]
        inks = [picture_ink(root, action) for action in pictures]
        for title in titles:
            x, y, width, height = title_box(title)
            covered = 0
            for mask, left, top, _ in inks:
                x0, y0 = max(0, round(x - left - 6)), max(0, round(y - top - 6))
                x1, y1 = min(mask.shape[1], round(x + width - left + 6)), min(mask.shape[0], round(y + height - top + 6))
                if x1 > x0 and y1 > y0:
                    covered += int(mask[y0:y1, x0:x1].sum())
            share = covered / max(1., (width + 12) * (height + 12))
            text = str((title.get('params') or {}).get('text', ''))[:16]
            if share > TITLE_INK_FAIL:
                errors.append(f'{label}: the picture runs into the title “{text}” ({share:.0%} of its text box is ink); move the title or the picture')
            elif share > TITLE_INK_WARN:
                warnings.append(f'{label}: the picture touches the title “{text}” ({share:.1%} of its text box); leave a clear gap')
        canvas = float((manifest.get('canvas') or {}).get('width', 1280)) * float((manifest.get('canvas') or {}).get('height', 1920))
        for (mask, _, _, pixels), action in zip(inks, pictures):
            if mask.sum() < 1 or mask.size / canvas < .3:
                continue
            colours = (pixels[..., :3][mask] // 24).astype(np.int32)
            keys, counts = np.unique(colours[:, 0] * 4096 + colours[:, 1] * 64 + colours[:, 2], return_counts=True)
            if counts.max() / mask.size > FLAT_FILL:
                warnings.append(f'{label}: one flat colour covers {counts.max() / mask.size:.0%} of {Path(action["source"]).name}; '
                                'a large flat area reads as an empty page: give it texture, objects or a smaller fill')
        explained = any(item.get('recipe') in FORMS or (item.get('recipe') == 'custom' and (item.get('params') or {}).get('explains'))
                        for item in page.get('overlays', []))
        ids = [beat['id'] for beat in page['beats']]
        seconds = sum(beats[i]['end'] - beats[i]['start'] for i in ids if i in beats)
        if not pictures and not explained and not any(action.get('type') == 'illustration' for action in actions):
            limit = 6. if page.get('chapter') else 4.
            if seconds > limit:
                warnings.append(f'{label}: {seconds:.1f} s of narration over words only; add a picture or a form'
                                + (' (a chapter board may hold words alone for up to 6 s)' if page.get('chapter') else ''))
        if index and ids and ids[0] in beats:
            origin = beats[ids[0]]['start']
            moments = []
            for item in page.get('overlays', []):
                if item.get('recipe') == 'roadmap' and (item.get('params') or {}).get('mode') == 'strip':
                    continue  # the progress strip is furniture, not content
                at = item.get('at', 0.)
                if isinstance(at, str):
                    beat_id = at[5:] if at.startswith('beat:') else at
                    if beat_id in beats:
                        moments.append(beats[beat_id]['start'] - origin + float(item.get('offset', .15)))
                elif isinstance(at, (int, float)):
                    moments.append(float(at))
            for group in page.get('visual_groups', []):
                if group.get('start_beat') in beats and group.get('actions'):
                    moments.append(beats[group['start_beat']]['start'] - origin + float(group.get('lead', .3)) + .5)
            if moments and min(moments) > BLANK_ARRIVAL:
                warnings.append(f'{label}: the page arrives blank for {min(moments):.1f} s after the turn; bring its title in with the page '
                                '(at 0) or start the first stroke sooner')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('job', type=Path)
    report = check(parser.parse_args().job)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 1 if report['errors'] else 0


if __name__ == '__main__':
    sys.exit(main())
