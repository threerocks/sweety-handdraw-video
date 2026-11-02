"""Shared pre-render checks for timing, arm coverage, page camera/transitions and the fixed cover behavior."""
import math
import numpy as np
from PIL import Image


TRANSITIONS = ('slide', 'pan', 'cut', 'fade', 'wipe', 'zoom', 'erase', 'page_turn', 'slide_over', 'marker_wipe', 'ink_bleed', 'iris',
               'card_flip', 'zoom_through', 'whip', 'push', 'paper_tear', 'brush_reveal', 'cube', 'shards', 'tiles', 'flash')
TRANSITION_SECONDS = {'erase': (.6, 2.), 'page_turn': (.6, 1.8), 'marker_wipe': (.6, 1.6), 'ink_bleed': (.6, 1.6), 'iris': (.5, 1.4),
                      'card_flip': (.5, 1.4), 'zoom_through': (.6, 1.6), 'whip': (.25, .7), 'paper_tear': (.7, 1.8),
                      'brush_reveal': (.6, 1.6), 'cube': (.5, 1.4), 'shards': (.6, 1.6), 'tiles': (.6, 1.6),
                      'flash': (.2, .7)}  # paper and ink need longer than a camera move
TRANSITION_STYLES = {'erase': ('felt', 'rubber'), 'iris': ('round', 'blot')}
WIPE_DIRECTIONS = ('left', 'right', 'up', 'down')
MAX_ZOOM = 3.
CAMERA_FIELDS = {'at', 'center', 'zoom', 'ease', 'seconds', 'follow', 'rotate', 'shake', 'move', 'impact', 'hop', 'subject', 'auto'}


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_page_motion(page, canvas, first=False):
    """Camera keyframes and the transition into this page (pages[].camera / pages[].transition)."""
    keys, transition = page.get('camera'), page.get('transition')
    if keys is not None:
        if not isinstance(keys, list):
            raise ValueError('Page camera must be a list of keyframes')
        beats = {beat['id'] for beat in page.get('beats', [])}
        from motion import CAMERA_MOVES, camera_key
        for key in keys:
            if not isinstance(key, dict) or set(key) - CAMERA_FIELDS:
                raise ValueError('Camera keyframes accept only ' + '/'.join(sorted(CAMERA_FIELDS)))
            if 'move' in key and key['move'] not in CAMERA_MOVES:
                raise ValueError('Camera move is one of ' + ', '.join(CAMERA_MOVES))
            move, key = key.get('move'), camera_key(key)
            if not number(key.get('impact', 0)) or not 0 <= key.get('impact', 0) <= 40 or not number(key.get('hop', 0)) or not 0 <= key.get('hop', 0) <= .6:
                raise ValueError('Camera impact is within 0..40 px and hop within 0..0.6')
            at = key.get('at', 0)
            if isinstance(at, str):
                if not at.startswith('beat:') or at[5:] not in beats:
                    raise ValueError('Camera "at" must be "beat:<id>" of a beat on the same page: ' + at)
            elif not number(at) or not 0 <= at < page['end'] - page['start']:
                raise ValueError('Camera "at" must be seconds from the page start, inside the page')
            center = key.get('center', [canvas['width'] / 2, canvas['height'] / 2])
            if not (isinstance(center, list) and len(center) == 2 and all(map(number, center))
                    and 0 <= center[0] <= canvas['width'] and 0 <= center[1] <= canvas['height']):
                raise ValueError('Camera center must be [x, y] inside the canvas')
            if not number(key.get('zoom', 1)) or not 1 <= key.get('zoom', 1) <= MAX_ZOOM:
                raise ValueError(f'Camera zoom must stay within 1.0..{MAX_ZOOM:g}')
            from motion import EASINGS
            if key.get('ease', 'inout') not in EASINGS + ('inout',) or not isinstance(key.get('follow', False), bool):
                raise ValueError(f'Camera ease is one of {", ".join(EASINGS)} and follow is true/false')
            if not number(key.get('rotate', 0)) or not -20 <= key.get('rotate', 0) <= 20 or not number(key.get('shake', 0)) or not 0 <= key.get('shake', 0) <= 20:
                raise ValueError('Camera rotate is within -20..20 degrees and shake within 0..20 px')
            if not number(key.get('seconds', .8)) or not 0 <= key.get('seconds', .8) <= 3:
                raise ValueError('Camera move seconds must be within 0..3')
            if key.get('auto', 'framing') != 'framing':
                raise ValueError('Camera "auto" only marks a move the build inserted to keep drawing in view ("framing")')
            subject = key.get('subject')
            if move == 'dolly_zoom' and not subject:
                raise ValueError('A dolly_zoom camera key needs "subject": the visual id that keeps its size while the page zooms')
            if subject is not None:
                visuals = {event.get('visual') for event in page.get('events', ()) if not event.get('caption')}
                if not isinstance(subject, str) or (page.get('events') and subject not in visuals):
                    raise ValueError(f'Camera subject {subject!r} must be the visual id of an element drawn on this page')
                if at == 0:
                    raise ValueError('A dolly zoom (camera subject) starts after the page opens: give it "at" > 0')
    if transition is not None:
        if first:
            raise ValueError('pages[].transition describes the move into a page; the first page has none')
        if not isinstance(transition, dict) or transition.get('type') not in TRANSITIONS or set(transition) - {'type', 'seconds', 'direction', 'target', 'color', 'angle', 'style'}:
            raise ValueError('Transition must be {"type": ' + '|'.join(TRANSITIONS) + ', "seconds": ..., optional direction/target/color/angle/style}')
        if 'style' in transition and transition['style'] not in TRANSITION_STYLES.get(transition['type'], ()):
            raise ValueError(f'{transition["type"]} transition style is one of {", ".join(TRANSITION_STYLES.get(transition["type"], ())) or "(none)"}')
        target = transition.get('target')
        if target is not None and not (isinstance(target, list) and len(target) == 2 and all(map(number, target))
                                       and 0 <= target[0] <= canvas['width'] and 0 <= target[1] <= canvas['height']):
            raise ValueError('Transition target is a screen point [x, y] inside the canvas')
        low, high = TRANSITION_SECONDS.get(transition['type'], (.3, 1.))
        seconds = transition.get('seconds', max(low, .6))
        if transition['type'] != 'cut' and (not number(seconds) or not low <= seconds <= high):
            raise ValueError(f'{transition["type"]} transition seconds must be within {low:g}..{high:g}')
        if transition.get('direction', 'left') not in WIPE_DIRECTIONS:
            raise ValueError('Wipe direction is left, right, up or down')
    return True


def validate_timeline(timeline):
    if timeline['cover_seconds']!=1:
        raise ValueError('Cover must last exactly one second')
    if timeline['hand'].get('kind', 'hand') != 'pen' and timeline['hand'].get('arm_bottom_at_highest_tip',0)<timeline['canvas']['height']:
        raise ValueError('The forearm cut edge can become visible in-frame')
    for page in timeline['pages']:
        events=sorted((event for event in page['events'] if not event['caption']),key=lambda event:event['start'])
        if any(left['end']>right['start'] for left,right in zip(events,events[1:])):
            raise ValueError('Main drawing groups overlap; only one physical pen is allowed')
        if page['last_drawing_end']>page['speech_end']-.45:
            raise ValueError('Page drawing has no completion hold before narration ends')
        validate_page_motion(page, timeline['canvas'], page is timeline['pages'][0])
    if timeline.get('cues') or timeline.get('assets'):
        from subtitle_contract import validate_subtitles
        validate_subtitles(timeline)
    return True


def validate_cover(renderer):
    original=np.array(renderer.cover.convert('RGB'))
    for moment in (0.,.25,.49,.5):
        if not np.array_equal(np.array(renderer.frame(moment)),original):
            raise ValueError('Cover moves or fades during its first half second')
    for moment in (.6,.75,.9):
        under=renderer.under_cover(moment) if hasattr(renderer,'under_cover') else Image.new('RGBA',renderer.cover.size,renderer.paper)
        expected=np.array(Image.blend(renderer.cover,under,(moment-.5)/.5).convert('RGB'))
        if not np.array_equal(np.array(renderer.frame(moment)),expected):
            raise ValueError('Cover does not fade smoothly into the opening in its last half second')
    return True
