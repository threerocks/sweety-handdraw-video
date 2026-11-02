"""Measured viewing metrics of a built film, from rendered frames and the timeline (no video decode needed):
when the first meaningful picture appears, how much of each finished page carries content, whether artwork sits
under a phone's bottom UI, and the longest stretch where nothing on screen changes."""
import numpy as np
from PIL import Image
import layout_plan

FIRST_VISUAL_SHARE = .01   # of the screen inked before the opening counts as showing something (a title alone is ~0.5%)
OPENING_WARN, OPENING_FAIL = 3., 4.   # film seconds, the 1-second cover included
UNCHANGED_WARN = {'vertical': 8., 'landscape': 10.}
SAMPLE = 8                 # analysis downsampling


def ink_share(renderer, index, time, box=None):
    """Share of the screen (or of box) that differs from paper, as framed by the page camera, without pen or captions."""
    board = renderer.page_frame(index, time, include_hand=False, include_captions=False)
    frame = renderer.apply_camera(board, *renderer.camera(index, time), renderer.scale(index))
    if hasattr(renderer, 'draw_overlays'):
        renderer.draw_overlays(frame, index, time, 'screen')  # a chart, a map or a title on the screen layer is on screen too
    small = np.asarray(frame.convert('RGB').resize((renderer.width // SAMPLE, renderer.height // SAMPLE), Image.Resampling.BOX), np.int16)
    ink = np.abs(small - np.array(renderer.paper[:3], np.int16)).max(2) > 30
    if box is not None:
        top, bottom = (round(value / SAMPLE) for value in box)
        ink = ink[top:bottom]
    return float(ink.mean()) if ink.size else 0., ink


def page_at(renderer, time):
    import bisect
    return min(len(renderer.starts) - 1, max(0, bisect.bisect_right(renderer.starts, time) - 1))


def opening(renderer, until=8., step=.2):
    cover = renderer.timeline['cover_seconds']
    time = cover
    while time <= min(until, renderer.timeline['duration']):
        share, _ = ink_share(renderer, page_at(renderer, time), time)
        if share >= FIRST_VISUAL_SHARE:
            return round(time, 2), round(share, 3)
        time += step
    return None, 0.


def pages(renderer):
    """Per page at the end of its drawing: ink share, content bounding box share and ink under the bottom UI band."""
    zones = layout_plan.zones(renderer.width, renderer.height)
    vertical = renderer.height > renderer.width
    result = []
    for page in renderer.timeline['pages']:
        moment = min(page['end'] - .05, page.get('last_drawing_end', page['end']) + .2)
        share, ink = ink_share(renderer, page['index'], moment)
        rows, columns = np.nonzero(ink)
        bbox = ((rows.max() - rows.min() + 1) * (columns.max() - columns.min() + 1) / ink.size) if len(rows) else 0.
        # The phone's bottom UI covers the bottom inset the captions also avoid: 20% of a 9:16 frame, less of a 2:3 frame
        # (letterboxed on the phone), so the band follows caption_safe_zone, not the 9:16 layout table.
        from caption_assets import caption_safe_zone
        inset = caption_safe_zone(renderer.width, renderer.height)['bottom']
        under = float(ink[round((renderer.height - inset) / SAMPLE):].mean()) if vertical and inset else 0.
        result.append({'page': page['index'] + 1, 'ink_share': round(share, 3), 'content_box_share': round(float(bbox), 3),
                       'ink_under_bottom_ui': round(under, 4)})
    return result


def unchanged(timeline):
    """Longest interval between visual changes: drawing, camera moves, emphasis moves and page turns all count."""
    changes = []
    for page in timeline['pages']:
        changes.append((page['start'], page['start'] + .3))
        for event in page['events']:
            if not event['caption']:
                changes.append((event['start'], event['end']))
                changes += [(effect['start'], effect['end']) for effect in list(event.get('emphasis', ())) + list(event.get('motions', ()))
                            if effect.get('type') not in ('float', 'breathe', 'sway', 'boil')]  # idle loops are texture, not a new beat
        # A motion overlay (a handwritten title, a circle, an arrow) arriving or leaving is a change on screen too.
        for overlay in page.get('overlays') or []:
            changes.append((overlay['start'], overlay['end']))
            if overlay.get('exit'):
                changes.append((overlay['exit']['start'], overlay['exit']['end']))
        beats = {beat['id']: beat['start'] - page['start'] for beat in page['beats']}
        for key in page.get('camera') or []:
            at = key.get('at', 0)
            at = beats.get(at[5:], 0.) if isinstance(at, str) else float(at)
            if at > 0:
                changes.append((page['start'] + at, page['start'] + at + float(key.get('seconds', .8))))
    changes.sort()
    end = max(page['speech_end'] for page in timeline['pages'])
    longest, where, cursor = 0., None, timeline['cover_seconds']
    for start, stop in changes:
        if start > cursor and start - cursor > longest:
            longest, where = start - cursor, [round(cursor, 2), round(start, 2)]
        cursor = max(cursor, stop)
    if end - cursor > longest:
        longest, where = end - cursor, [round(cursor, 2), round(end, 2)]
    return round(longest, 2), where


def measure(renderer, design=None):
    """Metrics plus the warnings and errors they imply. A slow opening needs design.opening_hook.slow_open_reason."""
    timeline = renderer.timeline
    first, share = opening(renderer)
    page_report = pages(renderer)
    longest, where = unchanged(timeline)
    orientation = 'vertical' if renderer.height > renderer.width else 'landscape'
    warnings, errors = [], []
    reason = ((design or {}).get('opening_hook') or {}).get('slow_open_reason')
    if first is None or first > OPENING_FAIL:
        message = (f'Opening: nothing recognisable on screen until {first}s (cover included); draw the hook picture or title first and large, '
                   f'or record design.json opening_hook.slow_open_reason')
        (warnings if reason else errors).append(message)
    elif first > OPENING_WARN:
        warnings.append(f'Opening: first meaningful picture at {first}s; aim for <= {OPENING_WARN:g}s including the cover')
    if longest > UNCHANGED_WARN[orientation]:
        warnings.append(f'Nothing changes on screen for {longest}s ({where[0]}-{where[1]}s); add a camera move, an emphasis or the next element')
    for page in page_report:
        if page['ink_under_bottom_ui'] > .01:
            warnings.append(f"page {page['page']}: artwork under the phone's bottom UI band; keep pictures above the caption line")
    import direction
    warnings += direction.camera_warnings(timeline['pages']) + direction.transition_warnings(timeline['pages'])
    return {'first_meaningful_visual_seconds': first, 'first_visual_ink_share': share, 'longest_unchanged_seconds': longest,
            'longest_unchanged_range': where, 'orientation': orientation, 'pages': page_report,
            'thresholds': {'opening_warn': OPENING_WARN, 'opening_fail': OPENING_FAIL, 'unchanged_warn': UNCHANGED_WARN[orientation]},
            'warnings': warnings, 'errors': errors}
