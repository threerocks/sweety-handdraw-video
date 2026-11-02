"""Build optional user-supplied text and keep its gentle drift inside the frame."""
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageColor
from caption_assets import caption_font


def prepare_watermark(options, work, width, height):
    if not options or options.get('enabled') is False or not str(options.get('text', '')).strip():
        return None
    text = str(options['text']).strip()
    opacity = float(options.get('opacity', .24))
    if not math.isfinite(opacity) or not 0 < opacity <= .35:
        raise ValueError('Quiet watermark opacity must be within 0..0.35')
    path, index = caption_font(options)
    size = int(options.get('font_size', round(height * 27 / 1080)))
    font = ImageFont.truetype(path, max(12, size), index=index)
    lines, line = [], ''
    for char in text:
        if char == '\n' or (line and font.getlength(line + char) > width * .58):
            lines.append(line)
            line = ''
        if char != '\n':
            line += char
    if line:
        lines.append(line)
    boxes = [font.getbbox(line) for line in lines]
    line_height = max(box[3]-box[1] for box in boxes)
    gap = round(size*.35)
    # A thin paper-coloured halo at the same opacity keeps the mark readable where it drifts over a dark picture (night
    # scenes): dark text alone vanished there. On paper the halo is invisible. "halo": false turns it off.
    halo = options.get('halo', '#F2EFE8')
    stroke = max(1, round(size * .08)) if halo else 0
    margin = 7 + stroke
    sprite = Image.new('RGBA', (max(box[2]-box[0] for box in boxes)+2*margin,
                              len(lines)*line_height+(len(lines)-1)*gap+2*margin))
    if sprite.height > height * .3:
        raise ValueError('Watermark text is too long for a quiet background mark')
    draw = ImageDraw.Draw(sprite)
    color = ImageColor.getrgb(options.get('color', '#374753'))+(round(255*opacity),)
    outline = ImageColor.getrgb(halo)+(round(255*opacity),) if halo else None
    for row, (line, box) in enumerate(zip(lines, boxes)):
        draw.text((margin-box[0], margin+row*(line_height+gap)-box[1]), line, font=font, fill=color, stroke_width=stroke, stroke_fill=outline)
    target = Path(work)/'watermark.png'
    sprite.save(target)
    position = options.get('position', 'auto')
    if position not in ('auto', 'legacy') + ANCHORS:
        raise ValueError('watermark.position must be auto, legacy or one of ' + '/'.join(ANCHORS))
    return {'source':str(target.resolve()), 'text':text, 'opacity':opacity, 'position':position,
            'width':sprite.width, 'height':sprite.height, 'font_size':size,
            'color':options.get('color', '#374753'), 'halo':halo or None, 'start_seconds':1.,
            'motion':'gentle_drift' if position == 'legacy' else 'quiet_corner', 'font':{'path':path,'index':index}}


def watermark_position(mark, width, height, seconds):
    scale = height/1080
    x = width-mark['width']-90*scale + 45*scale*math.sin(seconds/17)
    y = 210*scale + 35*scale*math.sin(seconds/23)
    margin = round(24*scale)
    return (round(max(margin,min(width-mark['width']-margin,x))),
            round(max(margin,min(height-mark['height']-margin,y))))


# ---------------------------------------------------------------- automatic placement (3.0)
# The mark used to sit at one fixed spot under the title and, in every v3 film, covered a face, a head or bubble text.
# Now each page picks the quietest of four corners from its own rendered frames (ink, colour, overlays and the pen all
# count as busy; the caption band and the feed's safe insets are never candidates). The choice holds for the page and
# cross-fades at page boundaries when it changes.
ANCHORS = ('top_right', 'top_left', 'bottom_right', 'bottom_left')
DRIFT = 8          # px at 1080 height: the mark breathes in place instead of wandering into the picture
SWAP = .3          # seconds to fade out before and in after a page boundary when the corner changes
BUSY_DIFF = 28     # channel difference from paper that counts as ink
WARN_BUSY, FAIL_BUSY = .02, .06   # share of the mark's box covered by ink in the best corner


def watermark_candidates(mark, timeline, width, height, rows=7):
    """{slot: (x, y)} top-left corners in preference order: the four corners first, then positions down the right and
    left edges (pages often keep a progress strip or a title in the top corners). Bottom slots sit above the highest
    caption; the feed's safe insets are respected."""
    from caption_assets import caption_safe_zone
    scale = height / 1080
    safe = caption_safe_zone(width, height)
    margin, drift = round(24 * scale), round(DRIFT * scale)
    left = max(margin, safe['left']) + drift
    right = width - mark['width'] - max(margin, safe['right']) - drift
    top = max(margin, safe['top']) + drift
    caption_tops = [event['y'] for page in timeline.get('pages', ()) for event in page.get('events', ()) if event.get('caption')]
    floor = min(caption_tops) if caption_tops else height - max(margin, safe['bottom'])
    bottom = max(top, floor - round(14 * scale) - mark['height'] - drift)
    slots = {'top_right': (right, top), 'top_left': (left, top), 'bottom_right': (right, bottom), 'bottom_left': (left, bottom)}
    for k in range(1, rows + 1):
        y = round(top + (bottom - top) * k / (rows + 1))
        slots[f'right_{k}'] = (right, y)
    for k in range(1, rows + 1):
        slots[f'left_{k}'] = (left, round(top + (bottom - top) * k / (rows + 1)))
    return slots


def busy_share(frame, paper, box):
    import numpy as np
    x, y, w, h = box
    pixels = np.asarray(frame.convert('RGB').crop((x, y, x + w, y + h)), dtype=np.int16)
    return float((np.abs(pixels - np.array(paper[:3], dtype=np.int16)).max(axis=2) > BUSY_DIFF).mean()) if pixels.size else 0.


def clean_view(renderer, index, time):
    """The page as the viewer sees it at a time, without the hand (it leaves) and without captions (their band is
    never a candidate)."""
    board = renderer.page_frame(index, time, include_hand=False, include_captions=False)
    center, zoom, rotate = renderer.camera_state(index, time)
    frame = renderer.apply_camera(board, center, zoom, renderer.scale(index), rotate)
    renderer.draw_overlays(frame, index, time, 'screen')
    return frame


def plan_watermark(renderer, samples=6):
    """Choose a corner per page from frames sampled across it and write work/watermark-plan.json."""
    import json
    timeline, mark = renderer.timeline, renderer.timeline.get('watermark')
    if not mark:
        return None
    width, height = renderer.width, renderer.height
    fixed = mark.get('position', 'auto')
    candidates = watermark_candidates(mark, timeline, width, height)
    pad = round(DRIFT * height / 1080)
    pages, previous = [], None
    for page in timeline['pages']:
        start, end = max(page['start'], mark['start_seconds']), page['end']
        times = [start + (end - start) * (k + 1) / samples for k in range(samples)]
        times[-1] = end - .05
        frames = [clean_view(renderer, page['index'], time) for time in times if time > start]
        scores = {}
        for anchor, (x, y) in candidates.items():
            box = (x - pad, y - pad, mark['width'] + 2 * pad, mark['height'] + 2 * pad)
            scores[anchor] = round(max([busy_share(frame, renderer.paper, box) for frame in frames] + [0.]), 4)
        if fixed in candidates:
            anchor = fixed
        else:
            best = min(scores.values())
            # Stay where the previous page put it unless another corner is clearly quieter; then prefer the usual order.
            near = [name for name in candidates if scores[name] <= best + .005]
            anchor = previous if previous in near else near[0]
        previous = anchor
        pages.append({'page': page['index'], 'start': page['start'], 'end': page['end'], 'anchor': anchor,
                      'position': list(candidates[anchor]), 'busy': scores[anchor], 'scores': scores})
    worst = max((entry['busy'] for entry in pages), default=0.)
    plan = {'version': 1, 'mode': fixed if fixed in candidates else 'auto', 'candidates': {k: list(v) for k, v in candidates.items()},
            'swap_seconds': SWAP, 'pages': pages, 'worst_busy': worst,
            'status': 'FAIL' if worst > FAIL_BUSY else 'WARN' if worst > WARN_BUSY else 'PASS'}
    (Path(renderer.work) / 'watermark-plan.json').write_text(json.dumps(plan, indent=1))
    return plan


def planned_watermark(plan, mark, width, height, seconds):
    """((x, y), alpha) of the planned mark at a film time: a small drift in place, cross-faded across page boundaries
    where the corner changes."""
    pages = plan['pages']
    index = max(0, min(len(pages) - 1, sum(1 for entry in pages if entry['start'] <= seconds) - 1))
    entry, alpha = pages[index], 1.
    swap = float(plan.get('swap_seconds', SWAP))
    following = pages[index + 1] if index + 1 < len(pages) else None
    if following and following['anchor'] != entry['anchor'] and following['start'] - seconds < swap:
        alpha = max(0., (following['start'] - seconds) / swap)
    if index and pages[index - 1]['anchor'] != entry['anchor'] and seconds - entry['start'] < swap:
        alpha = min(alpha, max(0., (seconds - entry['start']) / swap))
    scale = height / 1080
    x = entry['position'][0] + DRIFT * scale * math.sin(seconds / 7)
    y = entry['position'][1] + DRIFT * scale * math.sin(seconds / 9)
    return (round(x), round(y)), alpha
