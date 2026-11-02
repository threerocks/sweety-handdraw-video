"""Check explanatory composition and measure visible artwork at its actual page scale."""
from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageColor

LAYOUTS = ('hero_with_context', 'scene', 'comparison', 'sequence', 'modules', 'accumulating_board')
MIN_CONTENT_EXTENT = {'picture_book': .22, 'focus_diagram': .32, 'dense_infographic': .50,
                      'accumulating_board': .50, 'lecture_annotate': .30, 'step_sequence': .28, 'narrative_comic': .35}
TINY_PICTURE_EXTENT = .08


def validate_boards(board, mode):
    if mode != 'accumulating_board':
        return []
    errors, definitions = [], {}
    layouts = board.get('boards', [])
    if not isinstance(layouts, list):
        return ['boards must be a list of planned board layouts']
    for layout in layouts:
        if not isinstance(layout, dict):
            errors.append('Each board layout must be an object')
            continue
        identity = layout.get('id')
        if not identity or identity in definitions or not str(layout.get('purpose', '')).strip():
            errors.append('Each accumulating board needs a unique id and teaching purpose')
        definitions[identity] = layout
        regions = layout.get('regions', [])
        if not isinstance(regions, list) or not all(isinstance(region, dict) for region in regions):
            errors.append(f'board {identity}: regions must be a list of region objects')
            continue
        region_ids, assigned = set(), set()
        if len(regions) < 2:
            errors.append(f'board {identity}: plan at least two content regions before generating images')
        for region in regions:
            bounds = region.get('box', [])
            valid_box = (isinstance(bounds, list) and len(bounds) == 4
                         and all(isinstance(value, (int, float)) and math.isfinite(value) for value in bounds)
                         and 0 <= bounds[0] < bounds[2] <= 1 and 0 <= bounds[1] < bounds[3] <= 1)
            if not valid_box or not region.get('id') or region['id'] in region_ids or not str(region.get('purpose', '')).strip():
                errors.append(f'board {identity}: each region needs a unique id, purpose and normalized box [left,top,right,bottom]')
            region_ids.add(region.get('id'))
            visual_ids = region.get('visuals', [])
            if not isinstance(visual_ids, list) or not all(isinstance(visual_id, str) for visual_id in visual_ids):
                errors.append(f'board {identity}: region visuals must be a list of visual ids')
                continue
            if not visual_ids or any(visual_id in assigned for visual_id in visual_ids) or len(set(visual_ids)) != len(visual_ids):
                errors.append(f'board {identity}: assign each planned visual to one content region')
            assigned.update(visual_ids)
        valid_regions = [region for region in regions if isinstance(region.get('box'), list) and len(region['box']) == 4]
        if len(valid_regions) == len(regions) and all(all(isinstance(value, (int, float)) for value in region['box']) for region in regions):
            positions = [(region['box'][1], region['box'][0]) for region in regions]
            if positions != sorted(positions):
                errors.append(f'board {identity}: order content regions top to bottom, then left to right')
        expected = {visual['id'] for shot in board.get('shots', []) if shot.get('board') == identity for visual in shot.get('visuals', [])}
        if assigned != expected:
            errors.append(f'board {identity}: region visuals must exactly match the shots on this board')
    for shot in board.get('shots', []):
        if shot.get('board') not in definitions:
            errors.append(f'{shot.get("id", "shot")}: accumulating_board requires a declared board layout, not a new page per sentence')
    return errors


def validate_board_timing(board, timeline):
    if validate_boards(board, 'accumulating_board'):
        return []
    errors = []
    for layout in board.get('boards', []):
        region_index = {visual: index for index, region in enumerate(layout.get('regions', [])) for visual in region.get('visuals', [])}
        events = sorted((event for page in timeline['pages'] for event in page['events']
                         if event.get('visual') in region_index and not event.get('caption')), key=lambda event: event['start'])
        reading_order = [region_index[event['visual']] for event in events]
        if reading_order != sorted(reading_order):
            errors.append(f'board {layout["id"]}: drawing jumps between regions against the planned reading order')
    return errors


def validate_board_pages(manifest, board):
    if board.get('presentation', {}).get('mode') != 'accumulating_board' and board.get('presentation_mode') != 'accumulating_board':
        return []
    layout_errors = validate_boards(board, 'accumulating_board')
    if layout_errors:
        return layout_errors
    beat_pages = {beat.get('id', f'p{page_index:02d}-b{beat_index:02d}'): page_index
                  for page_index, page in enumerate(manifest.get('pages', [])) for beat_index, beat in enumerate(page['beats'])}
    pages_by_board, errors = {}, []
    for shot in board.get('shots', []):
        if not shot.get('board'):
            continue
        pages_by_board.setdefault(shot['board'], set()).update(beat_pages[identity] for identity in shot['beats'] if identity in beat_pages)
    for identity, pages in pages_by_board.items():
        if len(pages) > 1:
            errors.append(f'board {identity}: accumulating board spans pages {sorted(pages)}; keep its beats and drawings on one physical page until the explanation is complete')
    definitions = {layout['id']: layout for layout in board.get('boards', [])}
    region_of = {visual: region for layout in definitions.values() for region in layout.get('regions', []) for visual in region.get('visuals', [])}
    canvas = manifest.get('canvas', {})
    width, height = canvas.get('width', 1080), canvas.get('height', 1920)
    for page in manifest.get('pages', []):
        actions = [action for beat in page['beats'] for action in beat.get('actions', [])]
        actions += [action for group in page.get('visual_groups', []) for action in group['actions']]
        for action in actions:
            region = region_of.get(action.get('visual'))
            if not region:
                continue
            left, top, right, bottom = region['box']
            x, y = action.get('x', 0), action.get('y', 0)
            if not left * width - 1 <= x <= right * width or not top * height - 1 <= y <= bottom * height:
                errors.append(f'{action.get("visual")}: action anchor lies outside its planned board region {region["id"]}')
            if action.get('type') != 'text' and (x + action.get('width', 0) > right * width + 1 or y + action.get('height', 0) > bottom * height + 1):
                errors.append(f'{action.get("visual")}: artwork does not fit its planned board region {region["id"]}')
    return errors


def validate_composition(shot, mode):
    label = shot.get('id', 'shot')
    composition = shot.get('composition')
    if composition is None:
        return [f'{label}: {mode} needs composition (layout, focus visual, context with element and explains)'] if mode in ('dense_infographic', 'accumulating_board') else []
    if not isinstance(composition, dict):
        return [f'{label}: composition must be an object']
    errors = []
    visuals = {visual.get('id'): visual for visual in shot.get('visuals', [])}
    if composition.get('layout') not in LAYOUTS:
        errors.append(f'{label}: composition.layout must be one of {LAYOUTS}')
    focus = visuals.get(composition.get('focus'), {})
    if focus.get('kind') not in ('illustration', 'diagram'):
        errors.append(f'{label}: composition.focus must name a planned illustration/diagram')
    context = composition.get('context', [])
    if not isinstance(context, list):
        return errors + [f'{label}: composition.context must be a list']
    seen = set()
    for support in context:
        if not isinstance(support, dict):
            errors.append(f'{label}: each context element needs visual, element and explains')
            continue
        visual_id = support.get('visual')
        element = str(support.get('element', '')).strip()
        explains = str(support.get('explains', '')).strip()
        if visual_id not in visuals or visuals[visual_id].get('kind') not in ('illustration', 'diagram') or not element or not explains:
            errors.append(f'{label}: context must cite a planned picture and explain a visible element')
        if (visual_id, element) in seen:
            errors.append(f'{label}: repeated composition context does not add explanatory content')
        seen.add((visual_id, element))
    if mode in ('dense_infographic', 'accumulating_board') and len(seen) < 2:
        errors.append(f'{label}: dense composition needs at least two distinct supporting elements with a teaching purpose; simplify their strokes, not their meaning')
    return errors


def check_presentation_match(design_presentation, board):
    from presentation import validate_presentation
    planned = validate_presentation(board.get('presentation'), 'storyboard.json')
    if planned['mode'] != design_presentation['mode']:
        raise ValueError('design.json and storyboard.json presentation.mode must match')


def extent_ratio(mask):
    bounds = mask.getbbox()
    if bounds is None:
        return 0.
    return (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]) / (mask.width * mask.height)


def measure_pages(root, timeline, board, fixture=False):
    """Extent is a layout diagnostic, not an aesthetic score; white padding and captions do not count."""
    from PIL import ImageChops
    width, height = timeline['canvas']['width'], timeline['canvas']['height']
    scale = min(1., 384 / max(width, height))
    sample_width = max(1, round(width * scale))
    paper = np.array(ImageColor.getrgb(timeline.get('paper', '#FBF9F1')), dtype=np.int16)
    mode = board.get('presentation', {}).get('mode') or board.get('presentation_mode', 'focus_diagram')
    errors, warnings, pages = [], [], []
    cached_masks = {}
    shots = board.get('shots', [])
    visual_kinds = {visual.get('id'): visual.get('kind') for shot in shots for visual in shot.get('visuals', [])}
    for index, page in enumerate(timeline['pages']):
        events = page.get('events', [])
        captions = [event['y'] for event in events if event.get('caption')]
        content_bottom = min([height] + captions)
        sample_height = max(1, round(content_bottom * scale))
        content_mask = Image.new('L', (sample_width, sample_height))
        picture_mask = Image.new('L', content_mask.size)
        pictures = 0
        for event in events:
            metadata = timeline['assets'][event['asset']]
            is_picture = (bool(event.get('image')) or metadata.get('type') == 'illustration'
                          or visual_kinds.get(event.get('visual')) == 'diagram')
            if event.get('caption') or (not is_picture and metadata.get('type') != 'handwritten_text'):
                continue
            name = event['asset']
            board = float(event.get('scale', 1.))  # assets of a zoomed page are built at its board resolution
            if name not in cached_masks:
                path = Path(root) / 'work/assets' / f'{name}.png'
                with Image.open(path) as image:
                    rgba = np.array(image.convert('RGBA'))
                contrast = np.max(np.abs(rgba[:, :, :3].astype(np.int16) - paper), axis=2)
                foreground = (rgba[:, :, 3] > 24) & (contrast > 24)
                mask = Image.fromarray((foreground * 255).astype(np.uint8))
                cached_masks[name] = mask.resize((max(1, round(mask.width * scale / board)), max(1, round(mask.height * scale / board))), Image.Resampling.NEAREST)
            placed = Image.new('L', content_mask.size)
            placed.paste(cached_masks[name], (round(event['x'] * scale), round(event['y'] * scale)))
            content_mask = ImageChops.lighter(content_mask, placed)
            if is_picture:
                picture_mask = ImageChops.lighter(picture_mask, placed)
                pictures += 1
        # Explanation forms drawn by web motion are page content too (a prism, a commit graph, a table): their last frame
        # counts, so a page explained by an animation is not reported as empty. Titles and the progress strip do not.
        for overlay in page.get('overlays', []):
            if overlay.get('recipe') in ('kinetic_title', 'roadmap', 'burst') or not overlay.get('frames'):
                continue
            frames = sorted(Path(overlay['frames']).glob('*.png'))
            if not frames:
                continue
            with Image.open(frames[-1]) as image:
                alpha = np.array(image.convert('RGBA'))[:, :, 3] > 24
            layer = Image.fromarray((alpha * 255).astype(np.uint8)).resize(
                (max(1, round(overlay['width'] * scale)), max(1, round(overlay['height'] * scale))), Image.Resampling.NEAREST)
            placed = Image.new('L', content_mask.size)
            placed.paste(layer, (round(overlay['x'] * scale), round(overlay['y'] * scale)))
            content_mask = ImageChops.lighter(content_mask, placed)
        extent, picture_extent = extent_ratio(content_mask), extent_ratio(picture_mask)
        beat_ids = {beat['id'] for beat in page.get('beats', [])}
        page_shots = [shot for shot in shots if beat_ids.intersection(shot.get('beats', []))]
        exceptions = [shot.get('composition', {}).get('negative_space_reason', '') for shot in page_shots]
        exceptions = [reason.strip() if isinstance(reason, str) else '' for reason in exceptions]
        intentional_space = bool(exceptions) and all(exceptions)
        if not fixture and picture_extent < TINY_PICTURE_EXTENT and not intentional_space:
            errors.append(f'page {index + 1}: visible picture extent is only {picture_extent:.0%} of the content area; enlarge the main subject or add explanatory context, then recheck drawing time. Intentional negative space needs composition.negative_space_reason on every shot on this page')
        if extent < MIN_CONTENT_EXTENT[mode]:
            warnings.append(f'page {index + 1}: content extent {extent:.0%} is below the {mode} review threshold {MIN_CONTENT_EXTENT[mode]:.0%}; inspect the actual page for small subjects and unused space. This is not a beauty score')
        pages.append({'page': index + 1, 'picture_uses': pictures, 'content_extent_ratio': round(extent, 3),
                      'picture_extent_ratio': round(picture_extent, 3),
                      'visible_ink_ratio': round(float(np.count_nonzero(np.array(content_mask))) / (sample_width * sample_height), 3),
                      'intentional_negative_space': intentional_space})
    return {'errors': errors, 'warnings': warnings, 'pages': pages, 'visual_acceptance_claimed': False}
