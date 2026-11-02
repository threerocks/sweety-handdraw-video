"""Order, time and render whole line strokes like a right-handed person: per part the silhouette first,
then long interior structure, then details and short marks. Every mark is drawn by the nib: a picture that cannot
be drawn in its window is simplified upstream, never faded in."""
import json
import math
import re
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from motion_profile import PEAK, time_fraction
from line_topology import turning
from stroke_paths import calculate_lengths

LINE_SPEED = 420.     # px/s along curvature-weighted length: a person inking a confident contour at 1080p
SPEED_FACTOR = 4.     # calibrated cap: speed-drawing videos run 4-8x life and still read as human when the order is coherent
FRAME_PX = 80.        # most an inking nib may move in one frame and still read as continuous motion
TRAVEL_PX = 120.      # peak per-frame move of a lifted pen: a quick flick between strokes, never a teleport
DETAIL_BOOST = 1.6    # details and short marks may run this much faster than the main strokes under pressure
VIDEO_LAPSE = 3.      # pictures are drawn at about 3x real pace, then hold; never slowed down to fill a window
PART_PAUSE = .18     # thinking pause between semantic parts at human pace (scaled by lapse)
THINK = .1           # ... never shorter than this: before a new module the pen lifts and the hand visibly hesitates
REVERSE_COST = 250.  # px of extra travel a right-hander accepts rather than draw a line upward or leftward
HURRIED = REVERSE_COST / SPEED_FACTOR  # a hand at the speed cap cares less about direction
KEY_PART = re.compile(r'head|face|eye|mouth|nose|hand|finger|paw', re.I)
SILHOUETTE, KEY, STRUCTURE, DETAIL, TEXTURE = range(5)
# Stroke planning (draw-v3), in fractions of the picture diagonal unless noted.
ATTACH = .012        # a stroke starting this close to ink already drawn grows out of it
DETACHED = .12       # extra cost of starting a new line away from the drawn ink of the same object
TIER_COST = {SILHOUETTE: 0., KEY: .04, STRUCTURE: .07, DETAIL: .14, TEXTURE: .25}
OBJECT_MASS = .35    # a bigger object is drawn first unless a smaller one is much nearer
UPWARD = .35         # px of cost per px the pen must climb back up the page (a right-hander works downward)
LAPSED = 1.5         # time-lapse factor from which one-frame strokes and lifts are allowed
REFERENCE_SIDE = 400.  # px: picture side the human-pace constants were calibrated on
MAX_SIZE_SCALE = 1.8   # larger pictures draw at most this much faster per screen pixel


def hashed(value):
    return (math.sin(value * 12.9898 + 4.1414) * 43758.5453) % 1.


def weighted_lengths(points):
    """Arc length inflated on tight turns so the nib slows down on curves and corners (turn measured over
    +-6 px so skeleton pixel jitter does not count as curvature)."""
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if not len(steps):
        return np.zeros(1)
    turn = np.radians(turning(points, 6))[1:] / 6
    return np.r_[0., np.cumsum(steps * (1 + 1.5 * np.clip(turn / (np.pi / 8), 0, 1)))]


def line_cap(speed):
    """Mean nib speed cap: SPEED_FACTOR x human pace (each stroke is further held inside FRAME_PX per frame)."""
    return LINE_SPEED * SPEED_FACTOR * speed


def frame_floor(fps, lapse):
    """Shortest pen motion: two frames at human pace; one frame once time-lapsed, where a quick flick is what a
    sped-up drawing shows (the per-frame distance caps still keep it from teleporting)."""
    return (1. if lapse >= LAPSED else 2.) / fps


def lift_seconds(distance, fps=30, scale=1., lapse=1., size=1.):
    """Pen-up time: 80-150 ms at human pace, time-lapsed with the ink, never so short that the renderer's lifted
    move (75% of it in min-jerk motion, peak 1.875x mean) exceeds TRAVEL_PX (scaled with the picture) per frame. A
    time-lapsed hand also moves between strokes faster, by the square root of the lapse: at 4x a long hop takes half
    the frames it takes at life pace, still a visible move (a fixed cap made the hops, not the strokes, use most of a
    complex picture's time)."""
    return max(frame_floor(fps, lapse), (.08 + .07 * min(1., distance / (300. * size))) * scale / lapse,
               1.875 * distance / (.75 * TRAVEL_PX * size * fps * math.sqrt(max(1., lapse))))


PLAN_LONG_SIDE = 720   # strokes and timing are planned at most this large; a bigger picture is re-rendered sharp from the plan


def plan_box(width, height, drawing=None):
    """(plan width, plan height, render factor). A picture laid out larger than the plan side keeps the strokes and the
    drawing time of its planned size: more pixels do not mean more lines to draw, only sharper ones."""
    side = float((drawing or {}).get('plan_side', PLAN_LONG_SIDE))
    if not 360 <= side <= 1600:
        raise ValueError('drawing.plan_side must be within 360..1600')
    factor = min(1., side / max(width, height))
    return max(8, round(width * factor)), max(8, round(height * factor)), 1. / factor


def picture_size(width, height):
    """How much larger than the calibration size a picture is drawn. Timing follows the picture, not the screen:
    a picture twice as large is not twice the work, so speeds and per-frame caps scale with it (bounded)."""
    return float(np.clip(math.sqrt(width * height) / REFERENCE_SIDE, 1., MAX_SIZE_SCALE))


def slide_reach(strokes):
    """Gap a pen crosses without lifting: the next line starts on (or right beside) the ink it just laid."""
    if not strokes:
        return 3.
    every = np.vstack([stroke['points'] for stroke in strokes])
    return max(3., ATTACH * float(np.linalg.norm(every.max(0) - every.min(0))))


def stroke_profile(stroke, lapse):
    """Min-jerk at human pace; when time-lapsed, long strokes cruise between eased ends (lower peak speed)."""
    return 'plateau' if lapse > 1.5 and stroke['length'] >= 60 else 'minjerk'


def enclosed(ink, gap=2):
    """Everything inside the outermost lines: the ink and the areas it encloses (a break of up to 2 * gap px still
    closes). A generated picture's soft rim or sticker halo lies outside the line, so the outline stays the outline."""
    seeds, _ = ndimage.label(~ndimage.binary_dilation(ink, iterations=gap))
    border = np.unique(np.r_[seeds[0], seeds[-1], seeds[:, 0], seeds[:, -1]])
    outside = np.isin(seeds, border[border > 0])
    return ~(ndimage.binary_dilation(outside, iterations=gap) & ~ink)


def areas(ink, gap=2):
    """Labels of the areas the lines close (0 on the ink): found on lines widened by gap px, so a small break still
    closes an area, then grown back up to the ink."""
    seeds, _ = ndimage.label(~ndimage.binary_dilation(ink, iterations=gap))
    _, nearest = ndimage.distance_transform_edt(seeds == 0, return_indices=True)
    return np.where(ink, 0, seeds[nearest[0], nearest[1]]).astype(np.int32)


def classify(strokes, owners, alpha, pixels, part_ids, key_parts=(), ink=None):
    """Assign each whole stroke to the part owning most of it, and a priority tier. A module goes from simple to complex:
    its outline, then its long interior lines (a jaw, a hairline), then (in a key part such as a face or a hand) the
    features that stand on their own (an eye, a brow, a mouth), then the short details that grow off them (hair strands,
    folds)."""
    height, width = alpha.shape
    owned = owners >= 0
    _, nearest = ndimage.distance_transform_edt(~owned, return_indices=True)
    owner_map = owners[nearest[0], nearest[1]]
    # A module is drawn outline first: its silhouette is where its own area meets another part or the paper, not only the
    # outer edge of the whole picture (in a scene that edge is the desk and the window, never the face).
    solid = enclosed(ink) if ink is not None and ink.any() else ndimage.binary_fill_holes(alpha > 30)
    exterior = [ndimage.distance_transform_edt(ndimage.binary_fill_holes(solid & (owner_map == part))) for part in range(len(part_ids))]
    # Objects are the solid shapes of the picture: an eye inside a face belongs to the face it sits in.
    shapes, _ = ndimage.label(alpha > 30, structure=np.ones((3, 3)))
    rgb = pixels[:, :, :3].astype(float)
    luminance = np.where(alpha > 30, rgb @ [.299, .587, .114], 255.)
    ink_reference = np.percentile(luminance[alpha > 30], 2) if (alpha > 30).any() else 0.
    offsets = np.array([(dx, dy) for dy in range(-2, 3) for dx in range(-2, 3)])
    every = np.vstack([stroke['points'] for stroke in strokes]) if strokes else np.zeros((0, 2))
    owners_of = np.concatenate([np.full(len(stroke['points']), k) for k, stroke in enumerate(strokes)]) if strokes else np.zeros(0, int)
    tree = cKDTree(every) if len(every) else None
    reach = max(3., ATTACH * float(np.linalg.norm(every.max(0) - every.min(0)))) if len(every) else 3.
    for index, stroke in enumerate(strokes):
        cells = np.clip(np.rint(stroke['points']).astype(int), 0, [width - 1, height - 1])
        stroke['part'] = int(np.bincount(owner_map[cells[:, 1], cells[:, 0]]).argmax())
        found = shapes[cells[:, 1], cells[:, 0]]
        stroke['object'] = int(np.bincount(found[found > 0]).argmax()) if (found > 0).any() else -1 - index
        window = np.clip(cells[:, None, :] + offsets[None], 0, [width - 1, height - 1])
        values = luminance[window[..., 1], window[..., 0]]
        darkest = np.argmin(values, axis=1)
        chosen = window[np.arange(len(cells)), darkest]
        stroke['color'] = ndimage.uniform_filter1d(rgb[chosen[:, 1], chosen[:, 0]], 9, axis=0, mode='nearest')
        tone = np.clip((250 - values.min(1)) / max(20., 250 - ink_reference), 0, 1)
        stroke['tone'] = float(tone.mean())
        silhouette = (exterior[stroke['part']][cells[:, 1], cells[:, 0]] <= 4).mean() >= .5
        key = part_ids[stroke['part']] in key_parts or bool(KEY_PART.search(part_ids[stroke['part']]))
        # A feature stands on its own: closed, or neither end touching another line (an eye dot, a brow, a mouth).
        touching = [owners_of[k] for end in stroke['points'][[0, -1]] for k in tree.query_ball_point(end, reach)] if tree is not None else []
        alone = stroke.get('closed', False) or all(owner == index for owner in touching)
        if silhouette and stroke['length'] >= 20 and stroke['tone'] >= .3 and not stroke['micro']:
            stroke['tier'] = SILHOUETTE
        elif key and alone and stroke['tone'] >= .3:
            stroke['tier'] = KEY
        elif stroke['micro'] or stroke['tone'] < .3 or (stroke['length'] < 25 and stroke['tone'] < .5 and not key):
            stroke['tier'] = TEXTURE
        else:
            stroke['tier'] = STRUCTURE if stroke['length'] >= 40 else DETAIL
        stroke['weights'] = weighted_lengths(stroke['points'])
        stroke['index'] = index
    return strokes


def orient(stroke, tip, reverse=REVERSE_COST):
    """Right-hander prior as a point index order: loops start at the top going counter-clockwise,
    lines steeper than ~30 degrees run down, flatter ones run right; tiny marks start nearer the pen."""
    points = stroke['points']
    count = len(points)
    forward, backward = np.arange(count), np.arange(count)[::-1]
    if stroke['closed']:
        top = int(np.argmin(points[:, 1] * 1000 + points[:, 0]))
        ring = np.r_[np.arange(top, count), np.arange(0, top + 1)]
        loop = points[ring]
        area = np.sum(loop[:-1, 0] * loop[1:, 1] - loop[1:, 0] * loop[:-1, 1])
        return (ring[::-1] if area > 0 else ring), None
    delta = points[-1] - points[0]
    if np.linalg.norm(delta) < 8:
        near = np.linalg.norm(points[0] - tip) <= np.linalg.norm(points[-1] - tip)
        return (forward if near else backward), None
    if abs(delta[1]) >= .6 * abs(delta[0]):
        return (forward if delta[1] >= 0 else backward), reverse
    return (forward if delta[0] >= 0 else backward), reverse


def apply_order(stroke, index):
    if not np.array_equal(index, np.arange(len(stroke['points']))):
        for key in ('points', 'radius', 'color'):
            stroke[key] = stroke[key][index]
        stroke['weights'] = weighted_lengths(stroke['points'])


def groups(pool, reach):
    """The separate objects inside a part: strokes on one solid shape of the picture (classify sets 'object'),
    else strokes that touch (two samples within reach px)."""
    if all('object' in stroke for stroke in pool):
        found = {}
        for stroke in pool:
            found.setdefault(stroke['object'], []).append(stroke)
        return list(found.values())
    parent = list(range(len(pool)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    samples = [stroke['points'][::2] for stroke in pool]
    owners = np.concatenate([np.full(len(points), index) for index, points in enumerate(samples)])
    for first, second in cKDTree(np.vstack(samples)).query_pairs(reach):
        a, b = root(owners[first]), root(owners[second])
        if a != b:
            parent[a] = b
    found = {}
    for index in range(len(pool)):
        found.setdefault(root(index), []).append(pool[index])
    return list(found.values())


def oriented_start(stroke, tip, reverse):
    """(index order, start point) for the right-hander's preferred direction, unless the far end is much nearer."""
    index, penalty = orient(stroke, tip, reverse)
    points = stroke['points']
    if penalty is not None and np.linalg.norm(points[index[-1]] - tip) + penalty < np.linalg.norm(points[index[0]] - tip):
        index = index[::-1]
    return index, points[index[0]]


def sequence(group, tip, reverse, diagonal, reach):
    """Draw one object like a person with a plan, from simple to complex: its outline, then its long interior structure,
    then the features of a key part, then the small details; inside each stage the lines that grow out of ink already
    on the page and lie near the pen come first, rarely back up the page."""
    ordered, drawn = [], None
    for phase in ((SILHOUETTE,), (STRUCTURE,), (KEY,), (DETAIL, TEXTURE)):
        todo = [stroke for stroke in group if stroke['tier'] in phase]
        while todo:
            best = None
            for stroke in todo:
                index, start = oriented_start(stroke, tip, reverse)
                cost = np.linalg.norm(start - tip) + TIER_COST[stroke['tier']] * diagonal
                cost += UPWARD * max(0., tip[1] - float(stroke['points'][:, 1].min()))
                if drawn is not None:
                    ends = stroke['points'][[0, -1]]
                    cost += 0. if drawn.query(ends)[0].min() <= reach else DETACHED * diagonal
                elif stroke['tier'] == SILHOUETTE:
                    cost += .5 * float(stroke['points'][:, 1].min()) - .1 * stroke.get('length', 0.)
                if best is None or cost < best[0]:
                    best = (cost, stroke, index)
            _, stroke, index = best
            todo = [item for item in todo if item is not stroke]
            apply_order(stroke, index)
            ordered.append(stroke)
            tip = stroke['points'][-1]
            drawn = cKDTree(np.vstack([item['points'] for item in ordered]))
    return ordered, tip


def order(strokes, part_count, tip, reverse=REVERSE_COST):
    """Plan each semantic part as a person would: object by object (strokes that touch form an object), the
    nearest substantial object next, and inside an object the silhouette first, then lines that grow out of
    what is already drawn. Replaces draw-v2's per-tier top-to-bottom scan, which hopped across the picture."""
    ordered = []
    tip = np.asarray(tip, float)
    if not strokes:
        return ordered, tip
    every = np.vstack([stroke['points'] for stroke in strokes])
    diagonal = max(1., float(np.linalg.norm(every.max(0) - every.min(0))))
    reach = max(3., ATTACH * diagonal)
    for part in range(part_count):
        pool = [stroke for stroke in strokes if stroke['part'] == part]
        if not pool:
            continue
        objects = groups(pool, reach)
        length = lambda stroke: stroke.get('length', float(calculate_lengths(stroke['points'])[-1]))
        mass = sum(length(stroke) for stroke in pool) or 1.
        while objects:
            def cost(group):
                # Reading order of the object (top first, then left), a little pen travel, and bigger objects first.
                corner = np.vstack([stroke['points'] for stroke in group]).min(0)
                nearest = min(np.linalg.norm(stroke['points'][[0, -1]] - tip, axis=1).min() for stroke in group)
                return corner[1] + .35 * corner[0] + .3 * nearest - OBJECT_MASS * diagonal * sum(map(length, group)) / mass
            group = min(objects, key=cost)
            objects = [item for item in objects if item is not group]
            planned, tip = sequence(group, tip, reverse, diagonal, reach)
            ordered.extend(planned)
    return ordered, tip


def part_pause(lapse):
    """Seconds the pen hovers before the next module: a person's moment of looking at what to draw next."""
    return max(THINK, PART_PAUSE / max(lapse, 1e-6))


def duration(stroke, speed, fps, lapse=1., size=1.):
    """(seconds, profile): at least the frame floor, and the profile's peak speed stays inside FRAME_PX per frame."""
    profile = stroke_profile(stroke, lapse)
    return max(frame_floor(fps, lapse), stroke['weights'][-1] / min(speed, FRAME_PX * size * fps / PEAK[profile])), profile


def gap_seconds(distance, fps, scale, lapse, size, slide, speed):
    """Pen movement between two strokes: a slide along the ink when the next line starts where this one ended,
    otherwise a lift."""
    if distance <= slide:
        return max(distance / max(speed, 1e-6), .2 / fps)  # still a distinct moment between two strokes
    return lift_seconds(distance, fps, scale, lapse, size)


def line_seconds(ordered, speed, fps, detail_speed=None, lapse=1., part_count=None, size=1., slide=0.):
    total, previous, last_part = 0., None, None
    for stroke in ordered:
        rate = detail_speed if detail_speed and stroke['tier'] >= DETAIL else speed
        total += duration(stroke, rate, fps, lapse, size)[0]
        if previous is not None:
            total += gap_seconds(np.linalg.norm(stroke['points'][0] - previous), fps, 1.1, lapse, size, slide, rate)
        if last_part is not None and stroke['part'] != last_part:
            total += part_pause(lapse)
        previous, last_part = stroke['points'][-1], stroke['part']
    return total


def requirements(strokes, part_count, tip, speed, fps, size=1.):
    """Seconds for: every stroke at human pace; every stroke at the cap; silhouette + key features at the cap."""
    every = [dict(stroke) for stroke in strokes]
    essential = [dict(stroke) for stroke in strokes if stroke['tier'] in (SILHOUETTE, KEY)]
    cap, slide = line_cap(speed), slide_reach(strokes)
    at_cap = lambda kept, detail: line_seconds(order(kept, part_count, tip, HURRIED)[0], cap, fps, detail, SPEED_FACTOR, size=size, slide=slide)
    return {'required': line_seconds(order(every, part_count, tip)[0], LINE_SPEED * speed, fps, size=size, slide=slide),
            'all_at_cap': at_cap(every, cap * DETAIL_BOOST), 'minimum': at_cap(essential, None)}


def schedule(strokes, part_count, tip, budget, start, total, speed=1., fps=30, size=1.):
    """Fit every stroke into budget seconds (faster up to the cap, never fewer strokes). Returns ordered strokes with
    times (fractions of total), segments, per-part [start, end] fractions and counts."""
    base, cap, slide = LINE_SPEED * speed, line_cap(speed), slide_reach(strokes)
    budget += 1e-6  # a window at exactly the minimum sums the same terms; float noise must not read as haste past the cap
    # A hurried hand takes the nearer end more often: soften the direction prior only for haste beyond video pace
    # (a time-lapse of a calm hand keeps its order and directions); at the cap it is the order the minimum assumed.
    reverse = REVERSE_COST / float(np.clip(line_seconds(order(strokes, part_count, tip)[0], base, fps, size=size, slide=slide) / VIDEO_LAPSE / max(budget, 1e-6), 1, SPEED_FACTOR))
    ordered, _ = order(strokes, part_count, tip, reverse)
    seconds = lambda pace, detail=None: line_seconds(ordered, pace, fps, detail, max(1., pace / base), size=size, slide=slide)
    if seconds(cap, cap * DETAIL_BOOST) > budget:
        ordered, _ = order(strokes, part_count, tip, HURRIED)
    low, high = base, cap
    if seconds(base) <= budget:
        speed_used = max(base / 1.6, base * seconds(base) / max(budget, 1e-6))
    else:
        if seconds(cap, cap * DETAIL_BOOST) > budget:
            high = cap * 64
        for _ in range(48):
            middle = (low + high) / 2
            pressure = min(1., (middle - base) / max(1e-6, cap - base))
            if seconds(middle, middle * (1 + (DETAIL_BOOST - 1) * pressure)) > budget:
                low = middle
            else:
                high = middle
        speed_used = high
    pressure = min(1., max(0., (speed_used - base) / max(1e-6, cap - base)))
    detail_speed = speed_used * (1 + (DETAIL_BOOST - 1) * pressure)
    lapse = max(1., speed_used / base)
    segments, cursor, previous, parts = [], start, None, []
    for part in range(part_count):
        part_start = cursor
        mine = [stroke for stroke in ordered if stroke['part'] == part]
        for stroke in mine:
            rate = detail_speed if stroke['tier'] >= DETAIL else speed_used
            if previous is not None:
                distance = float(np.linalg.norm(stroke['points'][0] - previous))
                lift = gap_seconds(distance, fps, .9 + .2 * hashed(stroke['index']), lapse, size, slide, rate)
                segments.append({'kind': 'slide' if distance <= slide else 'travel', 'start': cursor / total, 'end': (cursor + lift) / total,
                                 'points': [previous.tolist(), stroke['points'][0].tolist()], 'lengths': [0., distance]})
                cursor += lift
            seconds, profile = duration(stroke, rate, fps, lapse, size)
            fraction = time_fraction(stroke['weights'] / max(1e-6, stroke['weights'][-1]), profile)
            stroke['times'] = (cursor + seconds * fraction) / total
            segments.append({'kind': 'ink', 'start': cursor / total, 'end': (cursor + seconds) / total, 'profile': profile,
                             'points': stroke['points'].tolist(), 'lengths': stroke['weights'].tolist()})
            cursor += seconds
            previous = stroke['points'][-1]
        parts.append([part_start / total, cursor / total])
        # Humans pause briefly when switching semantic parts (not between strokes of the same part).
        if mine and part + 1 < part_count and any(s['part'] == part + 1 for s in ordered):
            pause = part_pause(lapse)
            segments.append({'kind': 'pause', 'start': cursor / total, 'end': (cursor + pause) / total,
                             'points': [previous.tolist(), previous.tolist()] if previous is not None else None})
            cursor += pause
    end = cursor / total
    counts = {'drawn_stroke_count': len(ordered), 'line_speed_px_s': round(speed_used, 1), 'speed_over_cap': bool(speed_used > cap * 1.001),
              'slides': sum(segment['kind'] == 'slide' for segment in segments),
              'drawn_fast_count': sum(s['tier'] >= DETAIL and pressure > 0 for s in ordered),
              'part_pause_seconds': PART_PAUSE}
    return ordered, segments, parts, end, counts


def taper(stroke):
    distances = calculate_lengths(stroke['points'])
    reach = max(1., min(.15 * distances[-1], 18.))
    envelope = np.clip(np.minimum(distances, distances[-1] - distances) / reach, 0, 1)
    return .3 + .7 * envelope * envelope * (3 - 2 * envelope)


def stamp(stroke, shape, alpha, color, records, times, opacity):
    radius = stroke['radius'] * taper(stroke)
    for k, ((x, y), r) in enumerate(zip(stroke['points'], radius)):
        left, right = int(max(0, x - r - 2)), int(min(shape[1], x + r + 3))
        top, bottom = int(max(0, y - r - 2)), int(min(shape[0], y + r + 3))
        if right <= left or bottom <= top:
            continue
        ys, xs = np.ogrid[top:bottom, left:right]
        coverage = np.clip(r + .5 - np.hypot(xs - x, ys - y), 0, 1) * opacity[k]
        window = (slice(top, bottom), slice(left, right))
        stronger = coverage > alpha[window]
        color[window][stronger] = stroke['color'][k]
        np.maximum(alpha[window], coverage, out=alpha[window])
        for threshold, time, owner in records:
            earlier = (coverage > threshold) & (times[k] < time[window])
            time[window][earlier] = times[k]
            owner[window][earlier] = stroke['part']


def render(shape, strokes):
    """Draw the line layer: variable width from the source ink, pressure taper, slight opacity variation.
    Each pixel's reveal time is when the nib passes it."""
    alpha = np.zeros(shape, np.float32)
    color = np.zeros(shape + (3,), np.float32)
    # Solid hits (coverage > .3) decide time/owner; faint anti-alias edges fall back to their first touch.
    records = [(threshold, np.full(shape, np.inf, np.float32), np.full(shape, -1, np.int16)) for threshold in (.3, 0.)]
    for stroke in strokes:
        count = len(stroke['points'])
        opacity = (.9 + .1 * hashed(stroke['index'] + 7)) * (1 + .05 * np.sin(np.linspace(0, 3, count) + stroke['index']))
        stamp(stroke, shape, alpha, color, records, stroke['times'], np.clip(opacity, 0, 1))
    (_, hit, owner), (_, touch, touch_owner) = records
    only = ~np.isfinite(hit) & np.isfinite(touch)
    hit[only], owner[only] = touch[only], touch_owner[only]
    return color, alpha, hit, owner


# Ink of a real pen (drawing.texture.ink "pressure"): the line is the source's own, so texture only reshapes it gently.
# A long line whose end is free (it meets no other ink) lands and lifts thin; the ink is a little lighter where the pen
# ran dry. The taper is the line's own ink cut by a smooth, anti-aliased nib envelope, and the density a smooth
# multiplier: the outline stays clean (no rough edge), the variation lives inside the ink. Short marks (dashes, the
# strokes of a glyph) and ends the artist already tapered keep their shape.
TAPER_MIN = .4        # nib width at a free line end, as a share of the line's width
TAPER_WIDTHS = 4.     # the taper runs over this many line widths (at most a third of the stroke)
TAPER_STROKE = 24.    # plan px: a stroke shorter than this (or than 12 line widths) is a mark, not tapered
INK_DENSITY = .08     # ink is up to this much lighter where the pen ran a little dry
INK_PATCH_PX = 6.     # plan px over which the ink density changes
INK_SEED = 2026


def ink_density(plan_shape, shape, seed=INK_SEED):
    """Multiplier (1 - INK_DENSITY .. 1) of ink strength: one smooth pattern drawn at plan size, so every render size of
    a picture shows the same."""
    field = ndimage.gaussian_filter(np.random.default_rng(seed).random(tuple(plan_shape)).astype(np.float32), INK_PATCH_PX)
    low, high = np.percentile(field, [5, 95])
    field = np.clip((field - low) / max(1e-6, float(high - low)), 0, 1)
    if tuple(field.shape) != tuple(shape):
        from PIL import Image
        field = np.asarray(Image.fromarray(field).resize((shape[1], shape[0]), Image.Resampling.BILINEAR), np.float32)
    return 1 - INK_DENSITY * field


def dense_path(path, step=.5):
    path = np.asarray(path, float)
    arc = calculate_lengths(path)
    if arc[-1] <= step:
        return path[[0, -1]], np.array([0., arc[-1]])
    fine = np.linspace(0, arc[-1], int(math.ceil(arc[-1] / step)) + 1)
    return np.c_[np.interp(fine, arc, path[:, 0]), np.interp(fine, arc, path[:, 1])], fine


def straightened(points, arc, reach):
    """An open path whose last `reach` px at each end run straight on along the direction just behind them: a skeleton
    curls off at a line's end (it ends at the corner of the cap), a hand's stroke does not."""
    points = points.copy()
    if len(points) < 4 or np.linalg.norm(points[0] - points[-1]) <= 2 or arc[-1] < 4 * reach:
        return points
    for distances in (arc, arc[-1] - arc):
        inner = np.nonzero(distances >= reach)[0]
        behind = np.nonzero(distances >= 2 * reach)[0]
        if not len(inner) or not len(behind):
            continue
        k = inner[0] if distances is arc else inner[-1]
        j = behind[0] if distances is arc else behind[-1]
        direction = points[k] - points[j]
        direction /= max(1e-9, float(np.linalg.norm(direction)))
        tip = distances < reach
        points[tip] = points[k] + direction * (reach - distances[tip])[:, None]
    return points


def centred(alpha, half, points, limit):
    """A stroke's centreline and half-width measured on the ink itself: a nib path wanders about the line (it visits
    broad corners, a skeleton sits half a pixel off), so each sample moves to the middle of the line's anti-aliased
    cross-section, and the cross-section's ink gives the width."""
    cells = np.clip(np.rint(points).astype(int), 0, limit)
    reach = np.max([half[np.clip(cells[:, 1] + dy, 0, limit[1]), np.clip(cells[:, 0] + dx, 0, limit[0])]
                    for dy in (-1, 0, 1) for dx in (-1, 0, 1)], axis=0) + 1.5
    if len(points) > 4:
        points = ndimage.gaussian_filter1d(points, 3., axis=0, mode='nearest')
    tangent = np.gradient(points, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    normal = np.c_[-tangent[:, 1], tangent[:, 0]]
    offsets = np.arange(-float(reach.max()), float(reach.max()) + .25, .5)
    across = points[:, None, :] + offsets[None, :, None] * normal[:, None, :]
    profile = ndimage.map_coordinates(alpha, [across[..., 1].ravel(), across[..., 0].ravel()], order=1, mode='constant').reshape(across.shape[:2])
    profile *= np.abs(offsets)[None, :] <= reach[:, None]
    ink = profile.sum(1)
    shift = np.where(ink > .5, (profile * offsets).sum(1) / np.maximum(ink, 1e-9), 0.)
    points = points + ndimage.uniform_filter1d(shift, 9, mode='nearest')[:, None] * normal
    width = np.where(ink > .5, ink * .5 / 2, np.maximum(reach - 2., .5))
    return points, ndimage.uniform_filter1d(np.maximum(width, .5), 9, mode='nearest')


def ramp_zone(from_end, reach, length):
    """Samples within a free end's taper (it runs from the ink's tip, .7 of a half-width past the centreline end)."""
    return from_end + .7 * reach < length


def ink_texture(alpha, paths, scale=1., plan_shape=None, seed=INK_SEED):
    """Line alpha (0..1, any render size) with pressure: a taper at the free ends of long strokes and a little ink
    density variation. paths: stroke centrelines in this size's pixels, in drawing order; scale: pixels of
    this size per plan pixel; plan_shape: the plan size (the density pattern). Only ink the source already has is
    changed; it is never moved."""
    alpha = np.asarray(alpha, np.float32)
    result = alpha.copy()
    ink = alpha > .5
    usable = [np.asarray(path, float) for path in paths if len(path) > 1]
    if ink.any() and usable:
        half = ndimage.distance_transform_edt(ink)
        dense = [dense_path(path) for path in usable]
        owner = np.concatenate([np.full(len(points), index) for index, (points, _) in enumerate(dense)])
        tree = cKDTree(np.vstack([points for points, _ in dense]))
        limit = np.array(alpha.shape[::-1]) - 1
        cell = lambda points: np.clip(np.rint(points).astype(int), 0, limit)

        def free(index, points, arc, end, reach):
            """Does this end of the stroke stop on paper (no other ink at it, none straight on past it)?"""
            inward = min(len(points) - 1, int(6 * scale) + 1) if end == 0 else max(0, len(points) - 2 - int(6 * scale))
            outward = points[end] - points[inward]
            outward /= max(1e-9, float(np.linalg.norm(outward)))
            radius = reach + 2.5 * scale
            nearby = tree.query_ball_point(points[end], radius)
            own = np.abs(arc[np.clip(np.array(nearby, int) - (owner < index).sum(), 0, len(arc) - 1)] - arc[end])
            if any(owner[k] != index or distance > 3 * radius for k, distance in zip(nearby, own)):
                return None
            probes = [points[end] + outward * (reach + k * scale) for k in (1., 2.5)]
            if any(0 <= q[0] <= limit[0] and 0 <= q[1] <= limit[1] and alpha[int(round(q[1])), int(round(q[0]))] > .5 for q in probes):
                return None
            return outward

        samples, base, shaped, tips = [], [], [], []
        for index, (points, arc) in enumerate(dense):
            cells = cell(points)
            rough = np.maximum(half[cells[:, 1], cells[:, 0]] - .5, .5)
            body = float(np.median(rough))
            ends = []
            if np.linalg.norm(points[0] - points[-1]) > 2 * scale and arc[-1] >= max(TAPER_STROKE * scale, 24 * body):
                ends = [end for end in (0, len(points) - 1) if free(index, points, arc, end, float(rough[end])) is not None]
            if not ends:   # the stroke keeps its ink: its samples only claim the pixels that are its own
                samples.append(points)
                base.append(rough)
                shaped.append(rough)
                tips.append(np.zeros(len(points), np.int64))
                continue
            traced = points
            points = straightened(points, arc, 2 * float(half[cell(points[[0, -1]])[:, 1], cell(points[[0, -1]])[:, 0]].max()) + 1.5)
            points, width = centred(alpha, half, points, limit)
            body = float(np.median(width))
            taper = np.ones(len(points))
            end_ids = np.zeros(len(points), np.int64)
            extra_points, extra_base, extra_shaped, extra_tips = [], [], [], []
            for end in ends:
                tip, reach = points[end], float(width[end])
                from_end = arc if end == 0 else arc[-1] - arc
                behind = (from_end > 2 * body) & (from_end < 4 * body)
                if behind.any() and float(width[behind].mean()) < .75 * body:
                    continue      # the artist already lifted the pen here
                outward = free(index, traced, arc, end, reach)
                if outward is None:
                    continue
                # A free end: the stroke runs on a little past its centreline end to the ink's own tip, landing light.
                identity = 2 * index + (end > 0) + 1
                length = float(np.clip(TAPER_WIDTHS * 2 * body, 4 * scale, arc[-1] / 3))
                ramp = np.clip((from_end + .7 * reach) / length, 0, 1)
                taper *= TAPER_MIN + (1 - TAPER_MIN) * ramp * ramp * (3 - 2 * ramp)
                end_ids = np.where(ramp_zone(from_end, reach, length), identity, end_ids)
                beyond = np.arange(.5, .7 * reach + 1e-9, .5)
                if len(beyond):
                    ramp = np.clip((.7 * reach - beyond) / length, 0, 1)
                    extra_points.append(tip + outward * beyond[:, None])
                    extra_base.append(np.full(len(beyond), reach))
                    extra_shaped.append(reach * (TAPER_MIN + (1 - TAPER_MIN) * ramp * ramp * (3 - 2 * ramp)))
                    extra_tips.append(np.full(len(beyond), identity))
            samples += [points] + extra_points
            base += [width] + extra_base
            shaped += [width * taper] + extra_shaped
            tips += [end_ids] + extra_tips
        samples, base, shaped, tips = np.vstack(samples), np.concatenate(base), np.concatenate(shaped), np.concatenate(tips)
        if tips.max() > 0:
            # Only the ink around a tapered end can change: query those pixels.
            bound = float(base.max()) + 2.
            low = np.floor(samples[tips > 0].min(0) - bound).astype(int)
            high = np.ceil(samples[tips > 0].max(0) + bound).astype(int)
            window = alpha[max(0, low[1]):high[1] + 1, max(0, low[0]):high[0] + 1]
            ys, xs = np.nonzero(window > 0)
            ys, xs = ys + max(0, low[1]), xs + max(0, low[0])
            # The nearest nib sample: its distance and the widths there, smooth along the stroke and across it.
            distance, nearest = cKDTree(samples).query(np.c_[xs, ys], distance_upper_bound=bound)
            found = np.isfinite(distance)
            nearest = np.where(found, nearest, 0)
            tube = lambda widths: np.where(found, np.clip(widths[nearest] + .5 - np.where(found, distance, 0), 0, 1), 0)
            # At a free end the tapered nib is the whole shape (the corners of a blunt cap go too, so no island is left);
            # everywhere else the ink is the source's own. An end whose ink the measured line does not describe (a
            # glyph's curl crossing its bar, a blob) is left as drawn.
            source = alpha[ys, xs]
            end = np.where(found, tips[nearest], 0)
            explained = np.bincount(end, np.minimum(source, tube(base)), minlength=tips.max() + 1)
            ink_there = np.bincount(end, source, minlength=tips.max() + 1)
            clean = explained >= .9 * np.maximum(ink_there, 1e-9)
            clean[0] = False
            result[ys, xs] = np.where(clean[end], np.minimum(source, tube(shaped)), source)
    if plan_shape is not None:
        result *= ink_density(plan_shape, alpha.shape, seed)
    return (np.rint(result * 255) / 255).astype(np.float32)


def simplify(points, epsilon):
    """Ramer-Douglas-Peucker polyline simplification."""
    if len(points) < 3:
        return points
    direction = points[-1] - points[0]
    norm = np.linalg.norm(direction)
    offsets = points - points[0]
    distances = np.abs(direction[0] * offsets[:, 1] - direction[1] * offsets[:, 0]) / norm if norm > 1e-9 else np.linalg.norm(offsets, axis=1)
    k = int(np.argmax(distances))
    if distances[k] <= epsilon:
        return points[[0, -1]]
    return np.vstack([simplify(points[:k + 1], epsilon)[:-1], simplify(points[k:], epsilon)])


CONSTRUCTION_COLOR = (105., 145., 196.)   # a light blue pencil: the plan under the drawing, never mistaken for ink
OVERSHOOT = .04                            # construction lines run past their ends by this share, as a quick hand does


def wobble(points, seed, amplitude=1.4):
    """A slow hand tremor across a polyline (px), the same for the same seed."""
    count = len(points)
    phase = np.linspace(0, 1, count) * max(1., calculate_lengths(points)[-1] / 90.)
    drift = np.c_[np.sin(phase * 2.1 + hashed(seed) * 6.28) + .5 * np.sin(phase * 5.3 + hashed(seed + 3) * 6.28),
                  np.sin(phase * 1.7 + hashed(seed + 1) * 6.28) + .5 * np.sin(phase * 4.1 + hashed(seed + 5) * 6.28)]
    return points + drift * amplitude / 1.5


def through(points, samples=24):
    """Catmull-Rom curve through the given points (a gesture line), ends extended a little past the first and last."""
    points = np.asarray(points, float)
    if len(points) == 2:
        direction = points[1] - points[0]
        return np.vstack([points[0] - OVERSHOOT * direction, points[1] + OVERSHOOT * direction])
    padded = np.vstack([2 * points[0] - points[1], points, 2 * points[-1] - points[-2]])
    curve = []
    for k in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[k - 1:k + 3]
        for t in np.linspace(0, 1, samples, endpoint=False):
            curve.append(.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3))
    curve.append(points[-1])
    curve = np.array(curve)
    head, tail = curve[1] - curve[0], curve[-1] - curve[-2]
    reach = OVERSHOOT * calculate_lengths(curve)[-1]
    return np.vstack([curve[0] - head / max(1e-9, np.linalg.norm(head)) * reach, curve, curve[-1] + tail / max(1e-9, np.linalg.norm(tail)) * reach])


def construction_strokes(shapes, width, height, part_ids):
    """Model-planned construction (起形): a few simple shapes laid in lightly before any ink, the big relations first,
    as an artist blocks in a picture before refining it module by module. Shapes are in normalized picture coordinates:
      {"shape": "ellipse", "center": [x, y], "radius": [rx, ry], "angle": degrees}  drawn once round and a little past
      {"shape": "box", "box": [left, top, right, bottom]}                             one stroke round the four corners
      {"shape": "line", "points": [[x, y], ...]}                                      a gesture line through the points
      {"shape": "polygon", "points": [[x, y], ...]}                                   closed
    "part" (optional) names the module the shape plans."""
    from line_topology import resample
    scale = np.array([width - 1., height - 1.])
    strokes = []
    for index, shape in enumerate(shapes):
        kind = shape.get('shape')
        if kind == 'ellipse':
            (cx, cy), (rx, ry) = np.asarray(shape['center'], float) * scale, np.asarray(shape['radius'], float) * scale
            turn = np.radians(shape.get('angle', 0.))
            # From the top, counter-clockwise like a right hand, once round and a little past where it began.
            t = np.linspace(-np.pi / 2, -np.pi / 2 - 2 * np.pi * (1 + 2 * OVERSHOOT), 160)
            local = np.c_[rx * np.cos(t), ry * np.sin(t)]
            points = local @ np.array([[np.cos(turn), np.sin(turn)], [-np.sin(turn), np.cos(turn)]]) + [cx, cy]
        elif kind == 'box':
            left, top, right, bottom = shape['box']
            corners = np.array([[left, top], [right, top], [right, bottom], [left, bottom], [left, top]]) * scale
            points = np.vstack([np.linspace(a, b, 12) for a, b in zip(corners[:-1], corners[1:])])
        elif kind in ('line', 'polygon'):
            anchors = np.asarray(shape['points'], float)
            if len(anchors) < 2:
                raise ValueError('A construction line or polygon needs at least two points')
            anchors = np.vstack([anchors, anchors[:1]]) if kind == 'polygon' else anchors
            points = (np.vstack([np.linspace(a, b, 12) for a, b in zip(anchors[:-1], anchors[1:])]) * scale if kind == 'polygon'
                      else through(anchors * scale))
        else:
            raise ValueError('Construction shapes are ellipse, box, line or polygon')
        points = resample(wobble(resample(points), index * 13 + 1))
        part = part_ids.index(shape['part']) if shape.get('part') in part_ids else 0
        strokes.append({'points': points, 'part': part, 'radius': np.full(len(points), .9),
                        'color': np.tile(CONSTRUCTION_COLOR, (len(points), 1)), 'index': index, 'tier': SILHOUETTE,
                        'weights': weighted_lengths(points), 'construction': True})
    return strokes


HEAD_PART = re.compile(r'head|face|skull|头|脸', re.I)
AUTO_CONSTRUCTION_STROKES = 24   # a picture with fewer strokes (an icon, a label) is drawn straight, without blocking in
MAX_CONSTRUCTION = 16             # shapes laid in automatically; a planned construction may have up to MAX_PLANNED
MAX_PLANNED = 40


def auto_construction(owners, alpha, part_ids, minimum_share=.012):
    """The construction a person would lay in from the modules themselves when the model planned none, in the modules'
    drawing order: an ellipse for a round module (a head also gets its centre line and eye line), a gesture line along a
    long thin one (an arm, a pen, a tail), otherwise a box turned to the module's main axis. Normalized shapes, the
    same form a model plan uses (drawing.sketch)."""
    height, width = alpha.shape
    visible = alpha > 30
    total = max(1, int(visible.sum()))
    scale = np.array([width - 1., height - 1.])
    shapes = []
    for index, part_id in enumerate(part_ids):
        ys, xs = np.nonzero(visible & (owners == index))
        if len(xs) < max(30, minimum_share * total) or len(shapes) >= MAX_CONSTRUCTION:
            continue
        points = np.c_[xs, ys].astype(float)
        center = points.mean(0)
        values, vectors = np.linalg.eigh(np.cov(points.T))
        major, minor = vectors[:, 1], vectors[:, 0]
        along, across = (points - center) @ major, (points - center) @ minor
        (a0, a1), (c0, c1) = np.percentile(along, [2, 98]), np.percentile(across, [2, 98])
        length, breadth = max(a1 - a0, 1.), max(c1 - c0, 1.)
        middle = center + major * (a0 + a1) / 2 + minor * (c0 + c1) / 2
        fill = len(xs) / max(1., math.pi / 4 * length * breadth)
        normal = lambda point: (np.asarray(point, float) / scale).round(4).tolist()
        if HEAD_PART.search(part_id) or (length / breadth < 1.5 and fill > .72):
            shapes.append({'part': part_id, 'shape': 'ellipse', 'center': normal(middle),
                           'radius': (np.array([length / 2, breadth / 2]) / scale).round(4).tolist() if abs(major[0]) >= abs(major[1])
                           else (np.array([breadth / 2, length / 2]) / scale).round(4).tolist(), 'angle': 0.})
            if HEAD_PART.search(part_id):
                top, bottom = middle - [0, .5 * max(length, breadth) * .95], middle + [0, .5 * max(length, breadth) * .95]
                shapes.append({'part': part_id, 'shape': 'line', 'points': [normal(top), normal(middle), normal(bottom)]})
                eye = middle + [0, .08 * max(length, breadth)]
                shapes.append({'part': part_id, 'shape': 'line', 'points': [normal(eye - [.5 * min(length, breadth), 0]), normal(eye + [.5 * min(length, breadth), 0])]})
        elif length / breadth > 3.:
            # A gesture line through the module's own middle at three stations along it.
            stations = []
            for low, high in ((a0, a0 + length / 3), (a0 + length / 3, a1 - length / 3), (a1 - length / 3, a1)):
                chosen = (along >= low) & (along <= high)
                stations.append(points[chosen].mean(0) if chosen.any() else center)
            shapes.append({'part': part_id, 'shape': 'line', 'points': [normal(point) for point in stations]})
        else:
            corners = [middle + major * da * length / 2 + minor * dc * breadth / 2 for da, dc in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            shapes.append({'part': part_id, 'shape': 'polygon', 'points': [normal(corner) for corner in corners]})
    return shapes


def check_construction(shapes, part_ids):
    """A planned construction must be a short list of simple shapes inside the picture, naming real modules."""
    if not isinstance(shapes, list) or not shapes or len(shapes) > MAX_PLANNED:
        raise ValueError(f'drawing.sketch must be true, false, "auto" or a list of 1..{MAX_PLANNED} construction shapes')
    inside = lambda values: all(isinstance(value, (int, float)) and -.05 <= value <= 1.05 for value in np.ravel(values))
    for shape in shapes:
        kind = shape.get('shape')
        if shape.get('part') is not None and shape['part'] not in part_ids:
            raise ValueError(f'Construction shape names an unknown part: {shape["part"]}')
        if kind == 'ellipse':
            ok = inside(shape.get('center', [None])) and len(shape.get('radius', [])) == 2 and all(0 < value <= 1 for value in shape['radius'])
        elif kind == 'box':
            box = shape.get('box', [])
            ok = len(box) == 4 and inside(box) and box[0] < box[2] and box[1] < box[3]
        elif kind in ('line', 'polygon'):
            ok = len(shape.get('points', [])) >= 2 and inside(shape['points'])
        else:
            ok = False
        if not ok:
            raise ValueError(f'Construction shapes are ellipse(center, radius), box(box), line(points) or polygon(points) in 0..1 '
                             f'picture coordinates: {json.dumps(shape, ensure_ascii=False)[:120]}')


ERASER = .03          # eraser half-width, as a share of the picture diagonal
ERASE_SHARE, ERASE_MIN, ERASE_MAX = .09, .5, 2.   # rubbing the construction out before colour: share of the picture's time, bounds (s)
ERASE_RAMP = .12      # seconds a construction line takes to disappear under the rubbing eraser


def eraser_passes(sketches, tip, diagonal):
    """(passes, radius): the eraser rubs each construction line out with short back-and-forth scrubs across it while
    moving along it, the nearest line next (a line is taken from whichever end is closer)."""
    from line_topology import resample
    radius = max(6., ERASER * diagonal)
    remaining, passes, tip = list(range(len(sketches))), [], np.asarray(tip, float)
    while remaining:
        ends = [(min(np.linalg.norm(sketches[k]['points'][0] - tip), np.linalg.norm(sketches[k]['points'][-1] - tip)), k) for k in remaining]
        index = min(ends)[1]
        remaining.remove(index)
        points = sketches[index]['points']
        if np.linalg.norm(points[-1] - tip) < np.linalg.norm(points[0] - tip):
            points = points[::-1]
        samples = resample(points, .9 * radius) if len(points) > 2 else points
        if len(samples) < 2:
            samples = np.vstack([points[0], points[-1]])
        direction = np.gradient(samples, axis=0)
        normal = np.c_[-direction[:, 1], direction[:, 0]] / np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-9)
        side = np.where(np.arange(len(samples)) % 2, 1., -1.)[:, None]
        scrub = resample(samples + side * .5 * radius * normal, 2.)
        passes.append(scrub)
        tip = scrub[-1]
    return passes, radius


def erase_times(sketch_alpha, passes, times, radius):
    """When each construction pixel is rubbed out: the time of the nearest scrub sample (every pixel the eraser
    reaches; a pixel it misses goes with the nearest one it took)."""
    every = np.vstack(passes)
    when = np.concatenate(times)
    ys, xs = np.nonzero(sketch_alpha > 0)
    result = np.full(sketch_alpha.shape, np.inf, np.float32)
    if len(xs):
        _, nearest = cKDTree(every).query(np.c_[xs, ys])
        result[ys, xs] = when[nearest]
    return result


def sketch_strokes(owners, alpha, part_count):
    """Light construction pass: each part's silhouette simplified to a few straight-ish segments with a small wobble."""
    from line_topology import graph, resample, trace
    from stroke_paths import thin_ink
    strokes = []
    for part in range(part_count):
        labels, count = ndimage.label(ndimage.binary_opening((owners == part) & (alpha > 30), iterations=2))
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        if not count or sizes.max() < 400:
            continue
        shape = ndimage.binary_fill_holes(labels == sizes.argmax())
        paths = trace(graph(thin_ink(shape & ~ndimage.binary_erosion(shape))))
        if not paths:
            continue
        rough = simplify(max(paths, key=len), max(3., .05 * math.sqrt(sizes.max())))
        wobble = np.array([[hashed(part * 31 + k) - .5, hashed(part * 17 + k + 5) - .5] for k in range(len(rough))]) * 4
        points = resample(rough + wobble)
        strokes.append({'points': points, 'part': part, 'radius': np.full(len(points), .8),
                        'color': np.full((len(points), 3), 120.), 'index': part, 'tier': SILHOUETTE,
                        'weights': weighted_lengths(points)})
    return strokes
