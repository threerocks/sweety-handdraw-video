#!/usr/bin/env python3
"""Storyboard gate: visual cadence, reuse, variety, visual count and drawing time come from the real narration."""
import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from line_drawing import VIDEO_LAPSE
from presentation import MODES, DEFAULT_MODE, get_mode, policy as presentation_policy, validate_presentation
from composition import measure_pages, validate_boards, validate_composition, validate_board_pages, validate_board_timing

COVER = 1.0  # video time = narration time + the fixed one-second cover
PICTURES = ('illustration', 'diagram')
# custom: a model-written animation of a mechanism (light moving with a car, a stopping-distance ruler) is a diagram too.
DIAGRAM_RECIPES = {'rough_chart', 'map_route', 'morph', 'sketch_shapes', 'progress_ring', 'rough_annotate', 'physics', 'spin3d', 'formula', 'custom',
                   'roadmap', 'table', 'flow', 'swimlane', 'checklist', 'cards', 'pulse'}
TEXT_RECIPES = {'kinetic_title', 'number_ticker', 'callout', 'formula'}
OVERLAY_SETTLE = .6  # s an overlay item takes to arrive after its moment (pop, slide, draw-on): then it can be read
KINDS = PICTURES + ('text_group',)
LEAD, HOLD, GAP = .54, .6, .08
# The minimum structure that makes a declared shot type true; the director still judges the content.
SHOT_TYPES = {
    'scene_with_background': ('an illustration with a declared background', lambda visuals: any(visual.get('kind') == 'illustration' and visual.get('background') for visual in visuals)),
    'comparison': ('two pictures (illustration/diagram) side by side', lambda visuals: sum(visual.get('kind') in PICTURES for visual in visuals) >= 2),
    'process_sequence': ('at least two visuals as steps', lambda visuals: len(visuals) >= 2),
    'closeup': ('an illustration', lambda visuals: any(visual.get('kind') == 'illustration' for visual in visuals)),
    'key_points': ('a text_group', lambda visuals: any(visual.get('kind') == 'text_group' for visual in visuals)),
    'overview_recap': ('at least two visuals', lambda visuals: len(visuals) >= 2),
    'character_action': ('an illustration featuring a declared character', lambda visuals: any(visual.get('kind') == 'illustration' and visual.get('characters') for visual in visuals)),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(root, manifest):
    reference = manifest.get('storyboard')
    return json.loads((Path(root) / reference).read_text()) if reference else None


def presentation_mode(board):
    """Director mode on the storyboard; default focus_diagram keeps old numeric gates working."""
    if board.get('presentation'):
        return validate_presentation(board['presentation'], 'storyboard.json')['mode']
    return board.get('presentation_mode') or DEFAULT_MODE


def max_hold(board):
    mode = presentation_mode(board)
    rules = presentation_policy(mode)
    low, high = rules['hold_range']
    default = rules['max_hold_seconds']
    value = float(board.get('max_hold_seconds', default))
    if not low <= value <= high:
        raise ValueError(f'storyboard max_hold_seconds must be within {low}..{high} for presentation mode {mode}')
    return value, mode, rules


def group_beats(beat_ids, group):
    """Beat ids covered by a page visual group; start/end may be page-local indices or beat ids."""
    def index(value):
        if isinstance(value, str) and value in beat_ids:
            return beat_ids.index(value)
        if isinstance(value, int) and 0 <= value < len(beat_ids):
            return value
        raise ValueError(f'Visual group beat {value!r} is not a beat id or index on this page')
    first, last = index(group['start_beat']), index(group.get('end_beat', group['start_beat']))
    if first > last:
        raise ValueError('Visual group beats must form an ordered range inside this page')
    return beat_ids[first:last + 1]


STROKES_PER_SECOND = 3.9  # measured v3 pace at the minimum (23 production pictures, 2026-10-29): construction, strokes, lifts,
                          # thinking pauses between modules, erasing, colouring by marker with shadows (draw-v3 alone: 6.1)
LEAD_IN = .3               # a layer timed a moment before its words (so it lands with them) belongs to their sentence
ANNOTATED_BEATS = 5        # a picture each sentence adds a planned title, keyword or mark to may carry this many beats (~20 s,
                           # one point of an explainer); the max_hold check still limits time with nothing new
MAX_DRAWING_SHARE = .7    # the pen is busy at most this share of the narration; the finished pictures hold the rest
# Strokes per second at a picture's minimum by process (colouring, construction) at drawing.speed 1: v3 production pace
# with construction and marker colouring 3.9 (above); the aunt-doctor film (2026-11-02, 8 pictures at speed 1.4) measured
# markers 7.7/s, parts 8.5/s and construction + markers 5.0/s, i.e. 5.5, 6.1 and 3.6 at speed 1. Construction with parts
# colouring is not measured: it adds construction's measured cost per stroke (0.07 s) to parts.
PROCESS_RATE = {('markers', True): STROKES_PER_SECOND, ('markers', False): 5.4, ('parts', True): 4.3, ('parts', False): 6.1}
FIGURE_STROKES = 45           # fewest strokes that keep one person recognizable (head, hair mass, hands; style-and-drawing.md)
# Strokes of a usual drawn picture after plate edits: with a person, and without (v3 films 2026-10-29..11-02: aunt-doctor
# median 75 with a person on every picture; headlight and agents medians 52 and 49, mostly objects; deposit 108).
TYPICAL_PICTURE_STROKES = {'person': 75, 'other': 50}
OTHER_PEN_SHARE = .1          # pen share beyond the pictures' minimums (same four films: +6 to +26 points, median +12)


def stroke_rate(drawing=None):
    """Strokes per second the pen finishes at a picture's minimum with these drawing settings ("auto" construction is
    laid in for any picture with a person's worth of strokes, so it counts as construction)."""
    drawing = drawing or {}
    colouring = drawing.get('colouring', 'markers')
    rate = PROCESS_RATE.get((colouring, drawing.get('sketch', 'auto') is not False), STROKES_PER_SECOND)
    return rate * float(drawing.get('speed', 1.))


def visual_drawings(manifest):
    """{visual id: drawing settings} of the illustrations a job.json already places (job drawing, then the action's)."""
    found = {}
    for page in (manifest or {}).get('pages', []):
        for holder in [*page.get('beats', []), *page.get('visual_groups', [])]:
            for action in holder.get('actions') or []:
                if action.get('type') == 'illustration' and action.get('visual'):
                    found[action['visual']] = {**manifest.get('drawing', {}), **action.get('drawing', {})}
    return found


def motion_only_visuals(board, manifest):
    """Visuals shown only by web-motion overlays (a flow, a ticker, a kinetic title): they play beside the pen and take
    none of its drawing time. Known from job.json overlays, or from a storyboard visual that names its recipe."""
    pen, motion = set(), {visual['id'] for shot in board.get('shots', []) for visual in shot.get('visuals', []) if visual.get('recipe')}
    for page in (manifest or {}).get('pages', []):
        motion |= {overlay.get('visual') for overlay in page.get('overlays') or []}
        for holder in [*page.get('beats', []), *page.get('visual_groups', [])]:
            pen |= {action.get('visual') for action in holder.get('actions') or []}
    return motion - pen - {None}


def tier(seconds, hold, rate=STROKES_PER_SECOND):
    """What fits the drawing seconds: by strokes (the pace follows the drawing settings), and past max_hold a picture
    needs parts that pace it (or a split), whatever its pace."""
    strokes = max(3, int(seconds * rate))
    if seconds <= hold and strokes < 12:
        return f'label or tiny icon: at most ~{strokes} strokes'
    if seconds <= hold and strokes < 24:
        return f'simple icon/silhouette: at most ~{strokes} strokes, flat fills'
    if seconds <= hold:
        return f'one subject: at most ~{strokes} strokes, flat fills, no texture lines'
    return (f'at most ~{strokes} strokes: a scene with background (declare background; each semantic part paces the drawing) '
            'or split into two visuals')


def spans(board, beats):
    """[(shot, start, end)] in video time; a shot lasts until the next shot's first beat."""
    shots = board.get('shots') or []
    if not shots or not all(shot.get('beats') for shot in shots):
        raise ValueError('Storyboard needs shots, each covering at least one beat')
    if [beat_id for shot in shots for beat_id in shot['beats']] != [beat['id'] for beat in beats]:
        raise ValueError('Storyboard shots must list every narration beat id exactly once, in narration order')
    times = {beat['id']: beat for beat in beats}
    ends = [times[shot['beats'][0]]['start'] for shot in shots[1:]] + [beats[-1]['end']]
    return [(shot, times[shot['beats'][0]]['start'], end) for shot, end in zip(shots, ends)]


def plan(board, beats, manifest=None):
    """Checks that need only the storyboard and beat times; returns the per-visual drawing budget. With job.json, the
    budget follows each picture's drawing settings (speed, colouring, construction)."""
    hold, mode, rules = max_hold(board)
    errors, warnings = validate_boards(board, mode), []
    drawings, default_drawing = visual_drawings(manifest), (manifest or {}).get('drawing', {})
    motion_only = motion_only_visuals(board, manifest)
    draw_limit = min(MAX_DRAWING_SHARE, rules['drawing_share'])
    forecast = 0.
    timed = spans(board, beats)
    times = {beat['id']: beat for beat in beats}
    characters = {character.get('id') for character in board.get('characters', [])}
    seen, rows = set(), []
    for shot, start, end in timed:
        label, visuals = shot.get('id', shot['beats'][0]), shot.get('visuals', [])
        for visual in visuals:
            if visual.get('kind') not in KINDS or not visual.get('id') or visual['id'] in seen or not str(visual.get('subject', '')).strip():
                errors.append(f'{label}: every visual needs a unique id, a subject and kind in {KINDS}')
            seen.add(visual.get('id'))
            if set(visual.get('characters', [])) - characters:
                errors.append(f'{label}/{visual.get("id")}: undeclared character; add it with a locked reference image')
        if shot.get('type') not in SHOT_TYPES:
            errors.append(f'{label}: type must be one of {sorted(SHOT_TYPES)}')
        elif not SHOT_TYPES[shot['type']][1](visuals):
            errors.append(f'{label}: a {shot["type"]} shot needs {SHOT_TYPES[shot["type"]][0]}')
        if not str(shot.get('purpose', '')).strip():
            errors.append(f'{label}: state what this shot shows beyond the narration (purpose)')
        errors.extend(validate_composition(shot, mode))
        # Estimate: one visual group per shot, drawn in order and split by weight like build(); web-motion overlays play
        # beside the pen and take none of its time.
        weights = [float(visual.get('weight', 3 if visual.get('kind') == 'illustration' else 1)) for visual in visuals]
        pen_weights = [0. if visual.get('id') in motion_only else weight for visual, weight in zip(visuals, weights)]
        cursor = start + LEAD
        window = times[shot['beats'][-1]]['end'] - HOLD - cursor - GAP * max(0, len(visuals) - 1)
        pen_window = times[shot['beats'][-1]]['end'] - HOLD - cursor - GAP * max(0, sum(weight > 0 for weight in pen_weights) - 1)
        for visual, weight, pen_weight in zip(visuals, weights, pen_weights):
            turn = window * weight / sum(weights)   # when it appears, for the cadence estimate
            if not pen_weight:
                rows.append({'shot': label, 'type': shot.get('type'), 'visual': visual.get('id'), 'kind': visual.get('kind'),
                             'subject': visual.get('subject'), 'start': round(cursor, 2), 'seconds': 0., 'scene': False,
                             'complexity': 'web motion beside the pen'})
                cursor += turn + GAP
                continue
            seconds = pen_window * pen_weight / sum(pen_weights)
            rate = stroke_rate(drawings.get(visual.get('id'), default_drawing))
            rows.append({'shot': label, 'type': shot.get('type'), 'visual': visual.get('id'), 'kind': visual.get('kind'), 'subject': visual.get('subject'),
                         'start': round(cursor, 2), 'seconds': round(seconds, 2), 'scene': bool(visual.get('background')),
                         'complexity': tier(seconds * rules['drawing_share'], hold, rate) if visual.get('kind') in PICTURES else 'short text lines'})
            if seconds < .5:
                errors.append(f'{label}/{visual.get("id")}: only {seconds:.1f}s to draw; give the shot more beats or fewer visuals')
            if visual.get('kind') == 'illustration':
                rows[-1]['strokes_per_second'] = round(rate, 2)
                # A drawn picture keeps the pen at least this long; a person needs a person's strokes whatever the prompt asks.
                forecast += min(seconds, TYPICAL_PICTURE_STROKES['person' if visual.get('characters') else 'other'] / rate)
                if visual.get('characters') and seconds * rate < FIGURE_STROKES:
                    warnings.append(f'{label}/{visual.get("id")}: {seconds:.1f}s fits about {int(seconds * rate)} strokes at this drawing pace '
                                    f'({rate:.1f}/s), fewer than the ~{FIGURE_STROKES} a recognizable person needs (generated pictures with a person '
                                    f'measure 70-190 before plate edits): span more beats with a visual group, merge it with a neighbouring '
                                    f'picture, or show this beat as a diagram or text form (a visual shown by web motion takes no pen time: '
                                    f'give it its "recipe" on the storyboard so the budget leaves it out)')
            cursor += turn + GAP
    narration = timed[-1][2] - timed[0][1]
    visuals = [visual for shot, _, _ in timed for visual in shot.get('visuals', [])]
    pictures = sum(visual.get('kind') in PICTURES for visual in visuals)
    # Mode-aware cadence: picture_book needs fewer full visuals; accumulating_board fewer redraws.
    required = max(1, math.ceil(narration / hold * rules['cadence_factor']))
    shares, text_only = {}, 0.
    for shot, start, end in timed:
        shares[shot.get('type')] = shares.get(shot.get('type'), 0) + (end - start) / narration
        if not any(visual.get('kind') in PICTURES for visual in shot.get('visuals', [])):
            text_only += (end - start) / narration
    sparse = sorted(((end - start) / max(1, len(shot.get('visuals', []))), shot.get('id')) for shot, start, end in timed)[::-1][:3]
    if len(visuals) < required:
        errors.append(f'{len(visuals)} planned visuals for {narration:.1f}s of narration; at least {required} are needed '
                      f'(a new visual every {hold:g}s). Add visuals to the sparsest shots: '
                      + ', '.join(f'{identity} ({seconds:.1f}s per visual)' for seconds, identity in sparse))
    if pictures < math.ceil(.8 * required) and not board.get('text_only_reason'):
        errors.append(f'{pictures} illustrations/diagrams planned; at least {math.ceil(.8 * required)} are needed. '
                      'If this host cannot generate images, stop and tell the user instead of shipping text-only visuals')
    if text_only > .2 and not board.get('text_only_reason'):
        errors.append(f'Text-only shots fill {text_only:.0%} of the runtime (limit 20%); add illustrations/diagrams or declare text_only_reason')
    if narration > 45 and len(shares) < 3:
        errors.append(f'{narration:.0f}s film uses {len(shares)} shot type(s); use at least 3 of {sorted(SHOT_TYPES)}')
    dominant = max(shares, key=shares.get)
    if narration > 45 and shares[dominant] > .6:
        errors.append(f'Shot type {dominant} fills {shares[dominant]:.0%} of the runtime (limit 60%); vary composition')
    # Planned cadence estimate: scene visuals are assumed to be paced by their parts (build checks the real parts).
    points = [row['start'] + step * hold for row in rows for step in range(1 + (int(row['seconds'] // hold) if row['scene'] else 0))]
    stretch = longest(points, timed[0][1], timed[-1][2])
    if stretch[0] > hold:
        warnings.append(f'Estimated {stretch[0]:.1f}s with nothing new between {stretch[1]:.1f}s and {stretch[2]:.1f}s; split that shot or add a visual')
    # Per-shot density from presentation mode (heroes = illustration/diagram count).
    max_pics = rules['max_picture_visuals_per_shot']
    for shot, start, end in timed:
        label = shot.get('id', shot['beats'][0])
        # A progress strip (role "navigation") is the page's place in a long list, which director_lint requires on every
        # item page; it is not something this shot explains, so it does not count against the mode's picture budget.
        pics = [v for v in shot.get('visuals', []) if v.get('kind') in PICTURES and v.get('role') != 'navigation']
        labels = [v for v in shot.get('visuals', []) if v.get('kind') == 'text_group']
        if len(pics) > max_pics and not shot.get('density_exception'):
            errors.append(f'{label}: {len(pics)} picture visuals but presentation mode {mode} allows at most {max_pics} '
                          f'({rules["summary"]}); split the shot, accumulate on one board, or set density_exception')
        if len(labels) > rules['max_labels_per_shot'] and not shot.get('density_exception'):
            errors.append(f'{label}: {len(labels)} text_group visuals exceed mode {mode} label budget '
                          f'({rules["max_labels_per_shot"]})')
        if mode == 'picture_book' and any(v.get('background') for v in pics) and len(pics) > 1:
            errors.append(f'{label}: picture_book pages stay image-led with one hero; drop extra pictures or change mode')
        if mode == 'accumulating_board' and not shot.get('board') and len(pics) >= 2:
            warnings.append(f'{label}: accumulating_board works best with a shared "board" id so beats add marks instead of redrawing')
    if mode in ('accumulating_board', 'dense_infographic') and not board.get('presentation', {}).get('reason') and not board.get('presentation'):
        warnings.append(f'presentation mode {mode} should be declared under storyboard.presentation with a reason')
    stats = {'narration_seconds': round(narration, 2), 'max_hold_seconds': hold, 'required_visuals': required,
             'planned_visuals': len(visuals), 'planned_pictures': pictures, 'text_only_share': round(text_only, 3),
             'shot_type_shares': {key: round(value, 3) for key, value in shares.items()},
             'presentation_mode': mode, 'presentation_density': rules['density'],
             'drawing_share_limit': rules['drawing_share'], 'pen_forecast_share': round(forecast / narration, 3)}
    if narration > 45 and forecast / narration + OTHER_PEN_SHARE > draw_limit:
        warnings.append(f'With pictures of a usual size (~{TYPICAL_PICTURE_STROKES["person"]} strokes with a person, '
                        f'~{TYPICAL_PICTURE_STROKES["other"]} without) the drawn illustrations alone keep the pen busy '
                        f'about {forecast / narration:.0%} of the narration, and handwriting, drawn labels and pen moves usually add '
                        f'~{OTHER_PEN_SHARE:.0%}; mode {mode} allows {draw_limit:.0%}. Decide before generating any image: '
                        + pen_remedies(mode, rules, [drawings.get(visual.get('id'), default_drawing) for visual in visuals if visual.get('kind') == 'illustration']))
    return {'errors': errors, 'warnings': warnings, 'stats': stats, 'visuals': rows}


def pen_remedies(mode, rules, drawings):
    """What frees pen time, with its measured size, for the drawing settings in use."""
    drawings = drawings or [{}]
    remedies = ['plan fewer drawn pictures (let one picture span several beats, show the others as explanation forms)']
    if any(drawing.get('colouring', 'markers') == 'markers' for drawing in drawings):
        saving = 1 - PROCESS_RATE[('markers', False)] / PROCESS_RATE[('parts', False)]
        remedies.append(f'colour by parts instead of markers ("drawing": {{"colouring": "parts"}}, about {saving:.0%} less time per picture)')
    if any(drawing.get('sketch', 'auto') is not False for drawing in drawings):
        saving = 1 - STROKES_PER_SECOND / PROCESS_RATE[('markers', False)]
        remedies.append(f'keep the construction sketch for the one picture that shows the process ("sketch": false on the others, '
                        f'about {saving:.0%} less time each)')
    if rules['drawing_share'] < MAX_DRAWING_SHARE:
        remedies.append(f'if each page is one picture plus labels or an explanation form, that is focus_diagram (pen limit '
                        f'{MAX_DRAWING_SHARE:.0%}), not {mode}: change presentation.mode in design.json and storyboard.json and give the reason')
    return '; '.join(remedies)


def longest(points, start, end):
    points = sorted([start, end] + [point for point in points if start < point < end])
    return max((right - left, left, right) for left, right in zip(points, points[1:]))


def page_actions(manifest):
    """(page_index, beat ids where the action may draw, action) for every action in job.json."""
    for page_index, page in enumerate(manifest['pages']):
        ids = [beat.get('id', f'p{page_index:02d}-b{index:02d}') for index, beat in enumerate(page['beats'])]
        for beat_id, beat in zip(ids, page['beats']):
            for action in beat.get('actions', []):
                yield page_index, [beat_id], action
        for group in page.get('visual_groups', []):
            covered = group_beats(ids, group)
            for action in group['actions']:
                yield page_index, covered[covered.index(anchor_beat(covered, ids, action)):], action
        # A web motion layer tagged with a planned visual is that visual (a chart, a map, a title...): it plays on its page.
        for overlay in page.get('overlays', []):
            if overlay.get('visual'):
                yield page_index, ids, {'type': 'overlay', 'recipe': overlay.get('recipe'), 'params': overlay.get('params') or {}, 'visual': overlay['visual']}


def anchor_beat(covered, beat_ids, action):
    """An action with from_beat starts with that sentence; it must lie inside its group."""
    if action.get('from_beat') is None:
        return covered[0]
    beat_id = group_beats(beat_ids, {'start_beat': action['from_beat']})[0]
    if beat_id not in covered:
        raise ValueError(f'from_beat {action["from_beat"]!r} lies outside its visual group')
    return beat_id


def usage(root, manifest, board):
    """Every planned visual is drawn in its own shot; every illustration is planned; characters cite locked references."""
    root, errors = Path(root), validate_board_pages(manifest, board)
    visuals = {visual['id']: (shot, visual) for shot in board['shots'] for visual in shot.get('visuals', []) if visual.get('id')}
    placed = {}
    for page_index, beat_ids, action in page_actions(manifest):
        identity = action.get('visual')
        if identity is None:
            if action['type'] == 'illustration':
                errors.append(f'page {page_index}: illustration {action["source"]} is not in the storyboard; plan it and tag the action with "visual"')
            continue
        if identity not in visuals:
            errors.append(f'page {page_index}: action visual {identity!r} is not planned in the storyboard')
            continue
        shot = visuals[identity][0]
        if not set(beat_ids) & set(shot['beats']):
            errors.append(f'{identity} is planned for shot {shot.get("id")} but drawn during beats {beat_ids}')
        placed.setdefault(identity, []).append(action)
    for identity, (shot, visual) in visuals.items():
        types = [action['type'] for action in placed.get(identity, [])]
        recipes = {action.get('recipe') for action in placed.get(identity, []) if action['type'] == 'overlay'}
        if visual.get('role') == 'navigation' and not (placed.get(identity) and all(
                action.get('recipe') == 'roadmap' and (action.get('params') or {}).get('mode') == 'strip' for action in placed[identity])):
            errors.append(f'{shot.get("id")}/{identity}: only a roadmap progress strip may be a "navigation" visual')
        # A counting number is a text label; a chart, or a pen line/arrow with something else, is a diagram; so are
        # motion layers that chart, map, morph, sketch, simulate, turn in 3D or write a formula, and titles/tickers/callouts/formulas are text.
        need = {'illustration': 'illustration' in types,
                'text_group': bool({'text', 'counter'} & set(types)) or bool(TEXT_RECIPES & recipes),
                'diagram': 'bar_chart' in types or (bool({'marker', 'line', 'arrow'} & set(types)) and len(types) >= 2)
                           or bool(DIAGRAM_RECIPES & recipes)}.get(visual.get('kind'), True)
        if not need:
            errors.append(f'{shot.get("id")}/{identity}: planned {visual.get("kind")} "{visual.get("subject")}" is missing from job.json '
                          '(diagram = marker + another action); planned visuals cannot be skipped')
        for action in placed.get(identity, []):
            if action['type'] == 'illustration' and not (root / action['source']).is_file():
                errors.append(f'{identity}: image is missing: {action["source"]}')
    references = {}
    for character in board.get('characters', []):
        path = root / character.get('reference', '')
        if not path.is_file() or character.get('sha256') != sha256(path):
            errors.append(f'character {character.get("id")}: reference image must exist and match its locked sha256')
        references[character.get('id')] = path.resolve()
    if any(visual.get('characters') for _, visual in visuals.values()):
        plans = {}
        if manifest.get('drawing_plan'):
            plans = {item['id']: item for item in json.loads((root / manifest['drawing_plan']).read_text())['illustrations']}
        for identity, (_, visual) in visuals.items():
            for action in placed.get(identity, []):
                cited = {(root / path).resolve() for path in plans.get(action.get('drawing_plan_id'), {}).get('references', [])}
                missing = [name for name in visual.get('characters', []) if references.get(name) not in cited]
                if action['type'] == 'illustration' and missing:
                    errors.append(f'{identity}: its drawing plan must cite the locked reference image of {missing} in "references"')
    return errors


def timing(board, timeline):
    """Actual cadence and main-illustration reuse from the built timeline."""
    hold, mode, rules = max_hold(board)
    errors = []
    visuals = {visual['id']: visual for shot in board['shots'] for visual in shot.get('visuals', [])}
    shot_of = {beat_id: shot for shot in board['shots'] for beat_id in shot['beats']}
    beats = [beat for page in timeline['pages'] for beat in page['beats']]
    points = []
    for page in timeline['pages']:
        # A planned motion layer appearing is something new on screen too, and so is each thing a form adds later.
        points += [moment for overlay in page.get('overlays', []) if overlay.get('visual') in visuals for moment in overlay.get('moments', [overlay['start']])]
        for event in page['events']:
            if event['caption']:
                continue
            points.append(event['start'])
            if visuals.get(event.get('visual'), {}).get('background'):
                parts = timeline['assets'][event['asset']].get('parts', [])
                points += [event['start'] + part['outline_start'] * (event['end'] - event['start']) for part in parts if 'outline_start' in part]
    start, end = beats[0]['start'], beats[-1]['end']
    ordered = sorted([start, end] + [point for point in points if start < point < end])
    for left, right in zip(ordered, ordered[1:]):
        if right - left > hold:
            inside = [beat['id'] for beat in beats if beat['start'] < right and beat['end'] > left]
            shots = sorted({shot_of[beat_id].get('id') for beat_id in inside if beat_id in shot_of})
            errors.append(f'Nothing new is drawn for {right - left:.1f}s ({left:.1f}-{right:.1f}s, shots {shots}); limit {hold:g}s. '
                          'Add or split a visual there, or make it a scene whose parts pace the drawing')
    stretch = longest(points, start, end)
    run, previous, flagged = 0, None, False
    for page in timeline['pages']:
        illustrations = sorted((event for event in page['events'] if event.get('image')), key=lambda event: event['start'])
        for beat in page['beats']:
            started = [event for event in illustrations if event['start'] < beat['end']]
            current = json.dumps(started[-1]['image']) if started else None
            run = run + 1 if current and current == previous else int(bool(current))
            previous, flagged = current, flagged and run > 1
            shot = shot_of.get(beat['id'], {})
            own = {visual['id'] for visual in shot.get('visuals', [])}
            # A beat that brings a planned visual of its shot (a handwritten title, keyword, circle or callout over the
            # picture) adds to it: the picture may carry the page while each sentence adds something, up to
            # ANNOTATED_BEATS; past that, or with a beat that adds nothing, it is a lingering picture.
            # A form adds inside itself too (a row, a tick, a strike, a node): any of its scheduled moments in this beat counts.
            adds = any(not event['caption'] and not event.get('image') and event.get('visual') in own and beat['start'] <= event['start'] < beat['end']
                       for event in page['events']) \
                or any(overlay.get('visual') in own and any(beat['start'] - LEAD_IN <= moment < beat['end'] for moment in overlay.get('moments', [overlay['start']]))
                       for overlay in page.get('overlays', []))
            adding_to_board = adds and (mode == 'accumulating_board' and shot.get('board') or run <= ANNOTATED_BEATS)
            if run >= 3 and not adding_to_board and not str(shot.get('callback', '')).strip() and not flagged:
                flagged = True
                source, crop = json.loads(current)
                errors.append(f'{source}{" crop " + str(crop) if crop else ""} stays the main visual for more than 2 consecutive beats (through {beat["id"]}, shot {shot.get("id")}); '
                              'start a new illustration or mark the shot "callback" with a reason')
    holds = []
    for page in timeline['pages']:
        primary = sorted((event for event in page['events'] if not event['caption']), key=lambda event: event['start'])
        for index, event in enumerate(primary):
            if event.get('image'):
                following = primary[index + 1]['start'] if index + 1 < len(primary) else page.get('end', event['end'])
                holds.append(max(0., following - event['end']))
    stats = {'longest_static_seconds': round(stretch[0], 2), 'longest_static_range': [round(stretch[1], 2), round(stretch[2], 2)]}
    if holds:
        # How long a finished picture stays readable before the next element: the first Codex film held
        # only 0.56s on average, validated films hold 1.1-1.7s. Warn when the pen leaves too little time.
        stats['finished_hold_median_seconds'] = round(statistics.median(holds), 2)
        stats['finished_hold_min_seconds'] = round(min(holds), 2)
    return errors, stats


def drawing_time(timeline):
    """Engine drawability vs the scheduled window: too short to draw is an error, too short for detail a warning."""
    errors, warnings = [], []
    for page in timeline['pages']:
        for event in page['events']:
            measured = timeline['assets'][event['asset']].get('drawability') if not event['caption'] else None
            if not measured:
                continue
            window, label = event['end'] - event['start'], event.get('visual') or event['asset']
            asset = timeline['assets'][event['asset']]
            if asset.get('time_compression', 1) > 1.001 or asset.get('lines', {}).get('speed_over_cap') or asset.get('painting', {}).get('fill_over_cap'):
                errors.append(f'{label}: the engine exceeded its drawing budget; simplify the image or extend its window across beats instead of compressing its strokes')
            if window < measured.get('minimum_seconds', 0) - .02:  # the reported minimum is rounded to 0.01 s
                strokes = f' ({measured["stroke_count"]} strokes)' if measured.get('stroke_count') else ''
                errors.append(f'{label}: {window:.1f}s drawing window but the image needs at least {measured["minimum_seconds"]:.1f}s{strokes}; '
                              'simplify the image or extend its window across beats with a page visual group')
            elif window < max(measured.get('minimum_seconds', 0), measured.get('required_seconds', 0) / VIDEO_LAPSE) - .05:
                warnings.append(f'{label}: {window:.1f}s window is shorter than video pace; the pen hurries above it')
    return errors, warnings


def narration_seconds(timeline):
    return max(1e-6, timeline['pages'][-1]['speech_end'] - timeline['pages'][0]['beats'][0]['start'])


def pen_busy_share(timeline):
    """Share of the narration during which the pen is on the page at all (overlaps merged): what the finished
    pictures cannot use for holding. It counts lines, colour, reveal, writing and pen-up moves alike."""
    spans = sorted((event['start'], event['end']) for page in timeline['pages'] for event in page['events'] if not event['caption'])
    covered, reach = 0., -math.inf
    for start, end in spans:
        covered += max(0., end - max(start, reach))
        reach = max(reach, end)
    return covered / narration_seconds(timeline)


def overlay_settled(overlay):
    """When a web-motion overlay's last item has arrived: its last moment plus the arrival itself (or its start)."""
    moments = overlay.get('moments') or [overlay['start']]
    return min(overlay['end'], max(moments) + OVERLAY_SETTLE)


def finished_picture_hold_share(timeline, board=None):
    """Readable idle time after a picture is complete; blank lead-in and the closing tail do not count.
    A picture is a drawn illustration, a diagram (drawn, or a web-motion explanation form such as a flow, a table or a
    map, complete once its last item has arrived) or, on a page with neither (a chapter card, a quote), the page's
    finished text: that is what the viewer reads there."""
    seconds = 0.
    narration_end = timeline['pages'][-1]['speech_end']
    diagrams = {visual.get('id') for shot in (board or {}).get('shots', []) for visual in shot.get('visuals', []) if visual.get('kind') == 'diagram'}
    for page in timeline['pages']:
        events = [event for event in page['events'] if not event['caption']]
        overlays = page.get('overlays') or []
        picture_groups = {}
        for event in events:
            if event.get('image') or event.get('visual') in diagrams:
                picture_groups.setdefault(event.get('visual') or event['asset'], []).append(event)
        completed_graphics = [(max(event['end'] for event in group), min(event.get('hide', narration_end) for event in group)) for group in picture_groups.values()]
        start = page['beats'][0]['start']
        end = min(narration_end, page.get('end', page['speech_end']))
        # An overlay holds its last frame until the page ends, or until its exit starts.
        shown = lambda overlay: (overlay.get('exit') or {}).get('start', end)
        forms = [overlay for overlay in overlays if overlay.get('recipe') in DIAGRAM_RECIPES or overlay.get('visual') in diagrams]
        completed_graphics += [(overlay_settled(overlay), shown(overlay)) for overlay in forms]
        if not completed_graphics:
            text = {}
            for event in events:
                text.setdefault(event.get('visual') or event['asset'], []).append(event)
            completed_graphics = [(max(event['end'] for event in group), min(event.get('hide', narration_end) for event in group)) for group in text.values()]
            completed_graphics += [(overlay_settled(overlay), shown(overlay)) for overlay in overlays if overlay.get('recipe') in TEXT_RECIPES]
        boundaries = sorted({start, end} | {max(start, min(end, event[key])) for event in events for key in ('start', 'end')}
                            | {max(start, min(end, moment)) for graphic in completed_graphics for moment in graphic})
        for left, right in zip(boundaries, boundaries[1:]):
            drawing = any(event['start'] < right and event['end'] > left for event in events)
            completed_picture = any(completed <= left and hidden >= right for completed, hidden in completed_graphics)
            if not drawing and completed_picture:
                seconds += right - left
    return seconds / narration_seconds(timeline)


def pen_time(timeline):
    """What the pen spends the narration on, as shares: line (ink and sketch strokes of drawn pictures), colour
    (their marker and shading), reveal (pictures brushed in from the finished image), handwriting, marker, pen_up."""
    shares = dict.fromkeys(('line', 'colour', 'reveal', 'handwriting', 'marker', 'pen_up'), 0.)
    for page in timeline['pages']:
        for event in page['events']:
            if event['caption']:
                continue
            asset = timeline['assets'][event['asset']]
            for segment in asset.get('segments', []):
                kind = segment['kind']
                if kind in ('travel', 'pause'):
                    key = 'pen_up'
                elif asset['type'] == 'illustration':
                    key = 'reveal' if asset.get('mode') == 'reveal' else 'line' if kind in ('ink', 'sketch') else 'colour'
                else:
                    key = 'handwriting' if asset['type'] == 'handwritten_text' else 'marker'
                shares[key] += (segment['end'] - segment['start']) * (event['end'] - event['start'])
    return {key: round(value / narration_seconds(timeline), 3) for key, value in shares.items()}


def check_build(root, manifest, timeline):
    """Called by build(): hard failure on errors, warnings kept in the timeline."""
    board = load(root, manifest)
    if board is None and not manifest.get('runtime_fixture'):
        raise ValueError('Production requires a storyboard: set job.json "storyboard" (see story-design.md) and run storyboard_lint.py')
    errors, warnings = drawing_time(timeline)
    report = {'errors': errors, 'warnings': warnings, 'stats': {}}
    if board is not None:
        beats = [beat for page in timeline['pages'] for beat in page['beats']]
        planned = plan(board, beats, manifest)
        late, cadence = timing(board, timeline)
        share, spent = pen_busy_share(timeline), pen_time(timeline)
        reading_share = finished_picture_hold_share(timeline, board)
        _, mode, rules = max_hold(board)
        draw_limit = min(MAX_DRAWING_SHARE, rules.get('drawing_share', MAX_DRAWING_SHARE))
        if share > draw_limit:
            late.append(f'The pen is busy {share:.0%} of the narration (limit {draw_limit:.0%} for mode {mode}; ' +
                        ', '.join(f'{key} {value:.0%}' for key, value in spent.items() if value) + '), so finished pictures barely hold: '
                        'give pictures less to draw, let one board accumulate across beats instead of redrawing, '
                        + pen_remedies(mode, rules, list(visual_drawings(manifest).values()) or [manifest.get('drawing', {})]))
        median_hold = cadence.get('finished_hold_median_seconds')
        min_hold = rules.get('min_finished_hold_median', 1.0)
        if median_hold is not None and median_hold < min_hold:
            warnings = warnings + [f'A finished picture holds only {median_hold:.2f}s before the next element '
                                   f'(median; mode {mode} expects ≥{min_hold:g}s): '
                                   'give pictures less to draw or let one board accumulate across beats instead of redrawing']
        shortest_hold = cadence.get('finished_hold_min_seconds')
        if shortest_hold is not None and shortest_hold < 1.:
            warnings.append(f'A finished picture holds only {shortest_hold:.2f}s before the next element; below the 1.00s reading threshold. Keep the page and simplify the next drawing')
        if reading_share < .30 - 1e-6:
            message = (f'Finished pictures get only {reading_share:.0%} of the narration as static reading time (minimum 30%); '
                       'blank lead-in and closing silence do not count. Simplify the board or finish its drawings earlier')
            if manifest.get('runtime_fixture'):
                warnings.append(message)
            else:
                late.append(message)
        composition = measure_pages(root, timeline, board, fixture=manifest.get('runtime_fixture', False))
        report = {'errors': planned['errors'] + usage(root, manifest, board) + late + errors + composition['errors'] + validate_board_timing(board, timeline),
                  'warnings': planned['warnings'] + warnings + composition['warnings'],
                  'stats': {**planned['stats'], **cadence, 'pen_busy_share': round(share, 3), 'pen_time_share': spent,
                            'finished_picture_hold_share': round(reading_share, 3)},
                  'composition': composition}
    if report['errors']:
        raise ValueError('Storyboard lint failed:\n- ' + '\n- '.join(report['errors']))
    return report


def alignment_beats(root, manifest):
    if not manifest.get('narration', {}).get('alignment'):
        raise ValueError('The storyboard budget needs narration.alignment: lock the complete narration first')
    alignment = json.loads((Path(root) / manifest['narration']['alignment']).read_text())
    return [{'id': beat['id'], 'start': COVER + float(beat['start']), 'end': COVER + float(beat['end'])} for beat in alignment['beats']]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', type=Path)
    parser.add_argument('--budget', action='store_true', help='Before image generation: plan checks and drawing seconds per planned visual')
    arguments = parser.parse_args()
    root = arguments.job.resolve()
    manifest = json.loads((root / 'job.json').read_text())
    board = load(root, manifest)
    if board is None:
        raise ValueError('job.json has no "storyboard"')
    report = plan(board, alignment_beats(root, manifest), manifest)
    if not arguments.budget:
        report['errors'] += usage(root, manifest, board)
        report['note'] = 'Actual cadence, reuse and drawing-time checks run in build'
    report['status'] = 'fail' if report['errors'] else 'pass'
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 1 if report['errors'] else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError, KeyError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
