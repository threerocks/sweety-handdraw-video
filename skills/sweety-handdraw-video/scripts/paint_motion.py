"""Color like a person with a marker: per part, per color region, first a pass along the inside of the region's edge,
then short back-and-forth strokes across the shape (a slanted hatch on round shapes), then touch-ups for corners the
marker missed, and finally a few broad soft shading strokes. A fresh stroke is streaky, felt-textured color; it turns
solid where the next stroke overlaps it or a moment later, so paint never stamps in as flat blocks."""
import math
import numpy as np
from PIL import Image
from scipy import ndimage
from line_drawing import frame_floor, lift_seconds, weighted_lengths

FILL_SPEED = 700.     # px/s of marker travel for a person coloring at 1080p
SHADE_SECONDS = .35   # per shaded part, at human pace
SHADE_RAMP = .3       # seconds over which a broad soft stroke blends flat color into the source
SCALES = (1., 1.5, 2., 3.)
CAREFUL = 2.          # up to this marker scale a region gets its edge pass and overlapping strokes; the broadest marker
                      # (a window too short for care) sweeps rows at the old spacing with no edge pass
FILL_FACTOR = 6.      # coloring cap: a broad marker has no detail to follow, so time-lapsed coloring reads as human at 6x
FILL_FRAME_PX = 120.  # a broad marker may sweep this far per frame: its own width keeps the swept color continuous
PART_PAUSE = .12      # short rest when the marker moves to the next semantic part
ROW_SPACING = 1.1     # stroke step in marker radii: neighbouring strokes overlap, so most paint gets a second coat
QUICK_SPACING = 1.25  # ... and the broadest marker's step
REACH = 9.            # longest stroke in marker radii; a wider shape is colored in lanes, as a wrist would
ELONGATED = 1.8       # a shape this much longer than wide is colored across, the marker travelling along it
HATCH = math.radians(-60)   # a right hand's natural slant on a round shape (lower left to upper right)
TILT = math.radians(18)     # strokes across a long shape lean this much toward the natural slant
EDGE_MIN = 2.         # a region thicker than this many radii first gets a pass along the inside of its edge
EDGE_INSET = .55      # that pass runs this many radii inside the edge
STREAK = 3.2          # px period of the felt streaks along a stroke
WET = (.42, .9)       # alpha range of a single streaky coat
COAT_DELAY = .3       # seconds after which a single coat reads solid (sooner when the next stroke overlaps it)
COAT_REACH = 6.       # ... but no later than the marker takes to travel this many radii
COAT_RAMP = .12       # seconds over which streaky color settles to solid
JOIN = 4.             # passes whose ends are this many radii apart are joined without lifting the marker
FRONT_FRAMES = 1.5    # frames over which a freshly touched pixel comes in: a moving marker smears, it does not stamp
# Colouring by marker (drawing.colouring "markers"): one marker colours every area of its colour before the next is
# picked up, light before dark; each colour's shadow side follows with a darker marker over the base coat.
SAME_MARKER = 32.     # RGB distance within which two areas take the same marker
SHADOW_STEP = 12.     # luminance (or 12% of the base, if more) by which a shadow is darker than its base coat
SHADOW_SHARE = .03    # a shadow smaller than this share of its area (and than SHADOW_PX) is left to the base coat
SHADOW_PX = 40.
SHADOW_BRUSH = .7     # the shadow marker is this much narrower than the base marker of the area
MARKER_SWAP = .25     # seconds at human pace to put one marker down and pick the next up
MARKER_SHADE_RAMP = .08  # seconds over which a shadow coat settles (a marker stroke, not a soft blend)
# The colouring front (where paint has reached so far) is drawn at display resolution and anti-aliased: every pixel keeps
# its signed distance to the felt's reach, before and when the marker touches it, so a close-up interpolates the edge
# between plan pixels instead of scaling up their staircase.
TOUCH = .1            # a marker touches a pixel where its coverage passes this (radius + ~1.6 px from the nib)
FAR = -3.             # px: signed distance of a pixel no marker has come near
SOLID = .5            # px inside the reach from which a pixel is wholly covered
FRONT_PX = 1.         # display px over which a colouring front goes from paper to paint
# Marker media (drawing.texture.marker "layered"): the finished coat shows the lanes a real nib leaves along the sweep the
# hand made (a time-lapse marker is broader than any real one, so the lanes are a real nib's width, not its): where two
# lanes overlap the second coat darkens the first, as layered alcohol markers do; a lane leaves a little more ink at its
# edge; the felt leaves faint streaks along the sweep. All tone changes multiply the finished colour, keep each area's
# mean colour, and are smooth at any size.
NIB = 9.              # plan px: half the width of a real marker's lane (times the picture's size)
LANE_STEP = 1.55      # lanes step this many half-widths: neighbours overlap by about a fifth of their width
LANE_JITTER = .12     # ... give or take this share of a step, wander this many half-widths along their length and lean
LANE_WANDER = .18     # by up to LANE_TILT radians: a hand never lays two strokes exactly parallel
LANE_TILT = .03
LANE_SOFT = 2.5       # plan px over which a dried lane edge fades
LAYER_TONE = .09      # darkening per extra coat (multiply)
POOL_TONE = .03       # extra darkening where a lane's edge dried (pooling)
POOL_WIDTH = .12      # ... a band this share of the nib wide, just inside the lane edge
STREAK_TONE = .018    # felt streaks along the sweep
STREAK_PX = 2.6       # plan px between felt streaks
TONE_RANGE = (.86, 1.08)   # a coat never darkens or lightens the source more than this


def inpaint_lines(pixels, line_zone):
    """Replace ink (and its anti-aliased halo) with the nearest paint so lines live only in the line layer."""
    if not line_zone.any() or line_zone.all():
        return pixels.copy()
    _, nearest = ndimage.distance_transform_edt(line_zone, return_indices=True)
    return pixels[nearest[0], nearest[1]]


def palette(colors, count=3, rounds=8, merge=28.):
    """Small k-means; clusters whose colors are perceptually close are merged (noise is not a region)."""
    luminance = colors @ [.299, .587, .114]
    order = np.argsort(luminance)
    centers = np.array([colors[order[int(q * (len(colors) - 1))]] for q in np.linspace(.1, .9, count)])
    for _ in range(rounds):
        labels = np.argmin(((colors[:, None] - centers[None]) ** 2).sum(2), 1)
        centers = np.array([colors[labels == k].mean(0) if (labels == k).any() else centers[k] for k in range(count)])
    target = list(range(count))
    for first in range(count):
        for second in range(first + 1, count):
            if np.linalg.norm(centers[first] - centers[second]) < merge:
                target = [target[first] if value == target[second] else value for value in target]
    return np.array(target)[labels]


def region_pixels(region, image):
    (left, top), (height, width) = region['offset'], region['mask'].shape
    return image[top:top + height, left:left + width][region['mask']]


SMALL_AREA = 60.      # px (at the calibration size): a closed area this large can be coloured on its own ...
AREA_DEPTH = 2.5      # ... when it holds a pixel this far from its edge (an area, not a seam between two lines)
SHARE_REACH = 12.     # px (at the calibration size): such an area this close to a kept area of its colour shares its region
WRONG_SHARE = .2      # an area is absorbed in another colour when this share of its pixels would take another marker's


def own_colour_areas(found, rgb, kept, size=1.):
    """Labels of the closed areas to colour as regions: `kept` (the areas large beside their part) and every smaller
    area, big enough to colour, that the nearest kept area would otherwise absorb in another marker's colour. Absorbed,
    such an area took that neighbour's flat colour and showed its own only when a darker 'shadow' marker went over it
    (sky-colors film, 2026-11-02: a small man under a sky blue on the left and orange on the right, colouring by marker
    with the whole picture as one part, was painted as the sky on either side of his centreline; the sun as the sky).
    An area so kept that lies within SHARE_REACH of a kept area of its own colour (a trouser leg across the belt line
    from the jacket, sky seen between the legs) is coloured with it as one region, as one marker does in one round: it
    is relabelled in `found` and not returned; so is a speck too small for a pass that has such an area within reach.
    found: area label of each pixel (0 = none, such as the ink), changed in place; rgb: the picture's colours there."""
    count = int(found.max()) + 1
    sizes = np.bincount(found.ravel(), minlength=count)
    if not kept or count <= 1:
        return list(kept)
    edge = np.zeros(found.shape, bool)
    edge[:-1] |= found[:-1] != found[1:]
    edge[1:] |= found[1:] != found[:-1]
    edge[:, :-1] |= found[:, :-1] != found[:, 1:]
    edge[:, 1:] |= found[:, 1:] != found[:, :-1]
    present = np.nonzero(sizes[1:])[0] + 1
    depth = np.zeros(count)
    depth[present] = ndimage.maximum(ndimage.distance_transform_edt(~edge), found, present)
    candidates = (sizes >= SMALL_AREA * size * size) & (depth >= AREA_DEPTH)
    candidates[0] = False
    candidates[kept] = False
    colours = np.zeros((count, 3))
    for channel in range(3):
        colours[present, channel] = ndimage.median(rgb[..., channel], found, present)
    large, kept = set(kept), list(kept)

    def absorbed_wrong(labels):
        """Of these areas, the ones whose pixels the nearest kept areas would give another marker's colour (WRONG_SHARE
        of them or more; pixel by pixel, so an area of two paints with no line between them, split by its neighbours
        each in its own colour, stays absorbed)."""
        _, nearest = ndimage.distance_transform_edt(~np.isin(found, kept), return_indices=True)
        absorber = found[nearest[0], nearest[1]]
        where = labels[found]
        if not where.any():
            return np.zeros(0, np.int64)
        wrong = np.linalg.norm(rgb[where] - colours[absorber[where]], axis=-1) >= SAME_MARKER
        share = np.bincount(found[where], weights=wrong, minlength=count) / np.maximum(sizes, 1)
        return np.nonzero(labels & (share >= WRONG_SHARE))[0]

    for _ in range(8):   # an area kept in one round can absorb a smaller one in the next
        candidates[kept] = False
        added = absorbed_wrong(candidates) if candidates.any() else []
        if not len(added):
            break
        kept += added.tolist()
    # Areas too small or thin for a pass of their own (a sleeve's edge beside the sky) go with a kept area of their colour
    # within reach rather than the nearest one, so they too take their own colour.
    rest = sizes > 0
    rest[0] = False
    rest[kept] = False
    specks = absorbed_wrong(rest) if rest.any() else []
    reach = SHARE_REACH * size
    pad = int(math.ceil(reach)) + 1
    for value in sorted(set(kept) - large, key=lambda value: -sizes[value]) + sorted(specks, key=lambda value: -sizes[value]):
        same = [other for other in kept if other != value and np.linalg.norm(colours[other] - colours[value]) < SAME_MARKER]
        mine = found == value
        ys, xs = np.nonzero(mine)
        window = (slice(max(0, ys.min() - pad), ys.max() + pad + 1), slice(max(0, xs.min() - pad), xs.max() + pad + 1))
        target = np.isin(found[window], same)
        if not target.any():
            continue
        distance, nearest = ndimage.distance_transform_edt(~target, return_indices=True)
        closest = np.unravel_index(np.argmin(np.where(mine[window], distance, np.inf)), distance.shape)
        if distance[closest] > reach:
            continue
        found[mine] = found[window][nearest[0][closest], nearest[1][closest]]
        if value in kept:
            kept.remove(value)
    return kept


def color_regions(rgb, visible, owners, part_count, labels=None, size=1.):
    """Regions to paint: the model's region map when given, else palette regions per part.
    Specks are absorbed by their nearest region instead of becoming separate passes."""
    regions = []
    for part in range(part_count):
        mask = visible & (owners == part)
        area = int(mask.sum())
        if area < 60:  # a few specks are not a region to colour; the line layer carries this part
            continue
        cells = np.argwhere(mask)
        (top, left), (bottom, right) = cells.min(0), cells.max(0) + 1
        crop = (slice(top, bottom), slice(left, right))
        local = mask[crop]
        found = np.zeros(local.shape, np.int32)
        if labels is not None:
            found[local] = labels[crop][local]
        elif area >= 120:
            clusters = np.zeros(local.shape, np.int32)
            clusters[local] = palette(rgb[crop][local].astype(float)) + 1
            for cluster in np.unique(clusters[local]):
                pieces, count = ndimage.label(ndimage.binary_opening(clusters == cluster, iterations=2))
                sizes = np.bincount(pieces.ravel())
                for piece in range(1, count + 1):
                    if sizes[piece] >= max(400 * size * size, .03 * area):  # the same piece of paint at any picture size
                        found[pieces == piece] = found.max() + 1
        sizes = np.bincount(found[local], minlength=found.max() + 1)
        large = [value for value in range(1, len(sizes)) if sizes[value] >= max(60, .01 * area)]
        if labels is not None:
            # A closed area small beside its whole part (colouring by marker: the whole picture) is still its own area
            # when absorbing it would paint it another colour (or shares a nearby area of its own colour).
            large = own_colour_areas(found, rgb[crop], large, size)
        found[~np.isin(found, large)] = 0
        if not large:
            found[local], large = 1, [1]
        _, nearest = ndimage.distance_transform_edt(found == 0, return_indices=True)
        found = np.where(local, found[nearest[0], nearest[1]], 0)
        for value in large:
            region = {'part': part, 'offset': np.array([left, top]), 'mask': found == value}
            region.update(area=int(region['mask'].sum()), flat=np.median(region_pixels(region, rgb), axis=0))
            regions.append(region)
    regions.sort(key=lambda region: (region['part'], region['offset'][1], region['offset'][0], -region['area']))
    return regions


def axes(points):
    center = points.mean(0)
    axis = np.linalg.eigh(np.cov(points.T))[1][:, -1] if len(points) > 3 else np.array([1., 0.])
    axis = axis if (axis[0] > 0 if abs(axis[0]) >= abs(axis[1]) else axis[1] > 0) else -axis
    across = np.array([-axis[1], axis[0]])
    return center, axis, (across if (across[1] > 0 if abs(across[1]) >= abs(across[0]) else across[0] > 0) else -across)


def rotate(vector, angle):
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array([cosine * vector[0] - sine * vector[1], sine * vector[0] + cosine * vector[1]])


def hatch_axes(points, tilt=None):
    """(center, stroke direction, sweep direction). A long shape is stroked across and swept along its length,
    leaning toward the hand's slant; a round one gets the slant itself. The sweep runs down (or right)."""
    center = points.mean(0)
    if len(points) > 3:
        values, vectors = np.linalg.eigh(np.cov(points.T))
        elongation = math.sqrt(max(values[1], 1e-6) / max(values[0], 1e-6))
    else:
        elongation, vectors = 1., np.eye(2)
    slant = np.array([math.cos(HATCH), math.sin(HATCH)])
    if elongation >= ELONGATED:
        axis = vectors[:, 0]
        axis = axis if axis @ slant >= 0 else -axis
        # Lean toward the slant, never past it.
        lean = math.copysign(min(TILT, math.acos(float(np.clip(axis @ slant, -1, 1)))), axis[0] * slant[1] - axis[1] * slant[0])
        axis = rotate(axis, lean)
    else:
        axis = slant
    if tilt:
        axis = rotate(axis, tilt)
    across = np.array([-axis[1], axis[0]])
    if (across[1] if abs(across[1]) >= abs(across[0]) else across[0]) < 0:
        across = -across
    return center, axis, across


def resample(polyline, step):
    polyline = np.asarray(polyline, float)
    if len(polyline) < 2:
        return polyline
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(polyline, axis=0), axis=1))]
    if arc[-1] <= step:
        return polyline[[0, -1]]
    fine = np.linspace(0, arc[-1], int(math.ceil(arc[-1] / step)) + 1)
    return np.c_[np.interp(fine, arc, polyline[:, 0]), np.interp(fine, arc, polyline[:, 1])]


def edge_paths(distance, offset, radius, step):
    """Closed paths along the inside of the region edge, EDGE_INSET radii in, starting at the top."""
    inset = EDGE_INSET * radius
    ys, xs = np.nonzero(np.abs(distance - inset) <= .8)
    if not len(xs):
        return []
    points = np.c_[xs, ys].astype(float)
    keys = np.floor(points / step).astype(np.int64)
    _, first = np.unique(keys[:, 0] * 1_000_003 + keys[:, 1], return_index=True)
    points = points[np.sort(first)]
    free = np.ones(len(points), bool)
    paths, current = [], int(np.lexsort((points[:, 0], points[:, 1]))[0])
    while free.any():
        free[current] = False
        path = [current]
        while free.any():
            gaps = np.linalg.norm(points - points[path[-1]], axis=1)
            gaps[~free] = np.inf
            nearest = int(np.argmin(gaps))
            if gaps[nearest] > 2.6 * step:
                break
            free[nearest] = False
            path.append(nearest)
        line = points[path]
        if len(line) > 2 and np.linalg.norm(line[-1] - line[0]) <= 2.6 * step:
            line = np.vstack([line, line[:1]])
        if len(line) >= 3:
            # A hand does not follow pixel steps: ease the path, keeping its ends.
            smooth = line.copy()
            smooth[1:-1] = (line[:-2] + 2 * line[1:-1] + line[2:]) / 4
            paths.append(resample(smooth, step) + offset)
        if free.any():
            remaining = np.nonzero(free)[0]
            current = int(remaining[np.argmin(np.linalg.norm(points[remaining] - line[-1], axis=1))])
    return paths


def hatch(points, center, axis, across, radius, step, spacing=None):
    """Back-and-forth strokes along `axis`, stepped along `across` with rounded turns. Strokes longer than REACH
    radii are cut into lanes (lane borders wander from row to row); a row that no longer overlaps the previous one in
    its lane starts a new cell. Returns polylines."""
    along_values, across_values = (points - center) @ axis, (points - center) @ across
    spacing = (spacing or ROW_SPACING) * radius
    low, high = along_values.min(), along_values.max()
    lanes = max(1, int(round((high - low) / (REACH * radius))))
    rows = []
    for index, offset in enumerate(np.arange(across_values.min() + .45 * radius, across_values.max() + .45 * radius, spacing)):
        band = np.sort(along_values[np.abs(across_values - offset) <= .6 * spacing])
        if not len(band):
            continue
        borders = low + (high - low) * np.arange(1, lanes) / lanes + .35 * radius * np.sin(1.7 * index + 2.3 * np.arange(1, lanes))
        pieces = []
        for run in np.split(band, np.where(np.diff(band) > 2 * radius)[0] + 1):
            a, b = run[0] - .3 * radius, run[-1] + .3 * radius
            cuts = [a] + [value for value in borders if a + radius < value < b - radius] + [b]
            pieces += [(offset, start, finish, int(np.searchsorted(borders, (start + finish) / 2))) for start, finish in zip(cuts, cuts[1:])]
        rows.append(pieces)
    unused = {(row, k) for row, runs in enumerate(rows) for k in range(len(runs))}
    cells, bulge = [], .35 * spacing
    position = lambda along, offset: center + along * axis + offset * across
    while unused:
        row, k = min(unused, key=lambda item: (rows[item[0]][item[1]][3], item))
        chain = [(row, k)]
        unused.discard((row, k))
        while row + 1 < len(rows):
            _, low_end, high_end, lane = rows[row][k]
            overlaps = [(min(high_end, b) - max(low_end, a), j) for j, (_, a, b, other) in enumerate(rows[row + 1])
                        if (row + 1, j) in unused and other == lane and min(high_end, b) > max(low_end, a)]
            if not overlaps:
                break
            row, k = row + 1, max(overlaps)[1]
            chain.append((row, k))
            unused.discard((row, k))
        polyline, previous = [], None
        for index, (row, k) in enumerate(chain):
            offset, a, b, _ = rows[row][k]
            if index % 2:
                a, b = b, a
            if previous is not None:
                end, end_offset, direction = previous
                turn = max(end, a) + bulge if direction > 0 else min(end, a) - bulge
                u = np.linspace(0, 1, 10)[1:-1, None]
                start, middle, finish = position(end, end_offset), position(turn, (end_offset + offset) / 2), position(a, offset)
                polyline.append((1 - u) ** 2 * start + 2 * u * (1 - u) * middle + u * u * finish)
            along = np.linspace(a, b, max(2, int(abs(b - a) / step) + 1))
            polyline.append(position(along[:, None], offset))
            previous = (b, offset, 1 if b >= a else -1)
        cells.append(np.vstack(polyline))
    return cells


def nearest_order(cells, tip):
    """Cells in the order a hand takes them: always the nearest end next, walking a cell backwards when that is closer."""
    remaining, ordered = list(cells), []
    while remaining:
        if tip is None:
            index, reverse = 0, False
        else:
            ends = [(min(np.linalg.norm(cell[0] - tip), np.linalg.norm(cell[-1] - tip)), index) for index, cell in enumerate(remaining)]
            index = min(ends)[1]
            reverse = np.linalg.norm(remaining[index][-1] - tip) < np.linalg.norm(remaining[index][0] - tip)
        cell = remaining.pop(index)
        cell = cell[::-1] if reverse else cell
        ordered.append(cell)
        tip = cell[-1]
    return ordered


def joined(cells, radius, step):
    """Consecutive passes a few marker widths apart become one scribble: a person does not lift the marker for the next
    lane or a touch-up nearby. The connecting move paints only what it crosses inside the region."""
    result = []
    for cell in cells:
        if result and np.linalg.norm(cell[0] - result[-1][-1]) <= JOIN * radius:
            result[-1] = np.vstack([result[-1], resample(np.vstack([result[-1][-1], cell[0]]), step)[1:-1], cell])
        else:
            result.append(cell)
    return result


def uncovered(mask, offset, cells, radius):
    """Pixels of the region no marker sample reaches (the renderer touches what lies within radius + ~1.6 px)."""
    free = np.ones(mask.shape, bool)
    if cells:
        samples = np.rint(np.vstack(cells) - offset).astype(int)
        inside = (samples[:, 0] >= 0) & (samples[:, 1] >= 0) & (samples[:, 0] < mask.shape[1]) & (samples[:, 1] < mask.shape[0])
        free[samples[inside, 1], samples[inside, 0]] = False
    if free.all():
        return mask.copy()
    return mask & (ndimage.distance_transform_edt(free) > radius + .5)


def touch_ups(mask, offset, cells, radius, step, axis):
    """Short dabs for corners and slivers the strokes missed: a person goes back for them at the end."""
    dabs = []
    for _ in range(3):
        missed = uncovered(mask, offset, cells + dabs, radius)
        if not missed.any():
            break
        pieces, _ = ndimage.label(missed, np.ones((3, 3)))
        for label, piece in enumerate(ndimage.find_objects(pieces), 1):
            ys, xs = np.nonzero(pieces[piece] == label)
            points = np.c_[xs + piece[1].start, ys + piece[0].start].astype(float) + offset
            center = points.mean(0)
            along = (points - center) @ axis
            if np.ptp(along) <= radius:
                dabs.append(np.repeat(center[None], 2, axis=0))
            else:
                dabs.append(resample(np.vstack([center + along.min() * axis, center + along.max() * axis]), step))
    return dabs


def strokes(region, radius, careful=True):
    """The marker path of one region: edge pass, hatch, touch-ups (each a list of polylines, in drawing order)."""
    mask, offset = region['mask'], region['offset']
    if 'distance' not in region:
        region['distance'] = ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1].astype(np.float32)
    distance, step = region['distance'], max(1.5, radius / 4)
    ys, xs = np.nonzero(mask)
    points = np.c_[xs, ys].astype(float) + offset
    if careful:
        center, axis, across = hatch_axes(points, region.get('tilt'))
    else:
        # No time for care: long rows along the shape (fewest turns), no edge pass, no touch-ups.
        center, axis, across = axes(points)
        if region.get('tilt'):
            axis = rotate(axis, region['tilt'])
            across = np.array([-axis[1], axis[0]])
    region.update(center=center, axis=axis, across=across, edge=None, careful=careful)
    cells = []
    if careful and region.get('edges', True) and distance.max() > EDGE_MIN * radius:
        cells = edge_paths(distance, offset, radius, step)
        if cells:
            region['edge'] = (EDGE_INSET + 1) * radius
            inner = distance > (EDGE_INSET + .75) * radius
            ys, xs = np.nonzero(inner)
            points = np.c_[xs, ys].astype(float) + offset
    if len(points):
        cells += nearest_order(hatch(points, center, axis, across, radius, step, None if careful else QUICK_SPACING),
                               cells[-1][-1] if cells else None)
    if careful:
        cells += nearest_order(touch_ups(mask, offset, cells, radius, step, axis), cells[-1][-1] if cells else None)
    return joined(cells, radius, step) or [np.repeat(center[None], 2, axis=0)]


def plan(regions, scale, size=1.):
    """Every region, however small, gets its own pass of the marker: nothing settles in without a brush.
    A larger picture takes a proportionally broader marker (size), so coloring it is not more work."""
    careful = scale <= CAREFUL
    part_area = {}
    for region in regions:
        part_area[region['part']] = part_area.get(region['part'], 0) + region['area']
    for region in regions:
        # A careful colorist sizes the marker to each shape; in a hurry one broad marker does the whole part.
        area = region['area'] if careful else part_area[region['part']]
        region['radius'] = float(np.clip(math.sqrt(area) / 7, 7, 24 * size) * scale * region.get('brush', 1.))
        region['cells'] = strokes(region, region['radius'], careful)
        for shadow in region.get('shadows', []):
            shadow['radius'] = max(4. * size, SHADOW_BRUSH * region['radius'])
            shadow['cells'] = strokes(shadow, shadow['radius'], careful)


def shaded_parts(regions, rgb, part_count):
    """Parts whose source differs visibly from flat color get a short shading pass."""
    result = []
    for part in range(part_count):
        mine = [region for region in regions if region['part'] == part]
        if mine and np.average([np.abs(region_pixels(region, rgb) - region['flat']).mean() for region in mine],
                               weights=[region['area'] for region in mine]) > 6:
            result.append(part)
    return result


def by_marker(regions, rgb, size=1.):
    """Annotate regions for colouring by marker: the marker each takes (areas of nearly the same colour share one) and
    the shadow zones a darker marker lays over its base coat later. Returns the markers, light to dark."""
    markers = []
    for region in sorted(regions, key=lambda item: -item['area']):
        match = next((k for k, colour in enumerate(markers) if np.linalg.norm(colour - region['flat']) < SAME_MARKER), None)
        if match is None:
            markers.append(np.asarray(region['flat'], float))
            match = len(markers) - 1
        region['marker'] = match
        pixels = region_pixels(region, rgb)
        base = float(region['flat'] @ [.299, .587, .114])
        local = np.zeros(region['mask'].shape, bool)
        local[region['mask']] = pixels @ [.299, .587, .114] < base - max(SHADOW_STEP, .12 * base)
        local = ndimage.binary_opening(local, iterations=1)
        pieces, count = ndimage.label(local)
        region['shadows'] = []
        for piece in range(1, count + 1):
            mask = pieces == piece
            if mask.sum() < max(SHADOW_PX * size * size, SHADOW_SHARE * region['area']):
                continue
            cells = np.argwhere(mask)
            (top, left), (bottom, right) = cells.min(0), cells.max(0) + 1
            region['shadows'].append({'part': region['part'], 'mask': mask[top:bottom, left:right], 'edges': False,
                                      'offset': region['offset'] + [left, top], 'area': int(mask.sum())})
    first_part = {k: min(region['part'] for region in regions if region['marker'] == k) for k in range(len(markers))}
    order = sorted(range(len(markers)), key=lambda k: (-round(float(markers[k] @ [.299, .587, .114]) / 12), first_part[k]))
    for region in regions:
        region['marker_order'] = order
    return order


def nearest_first(regions, tip):
    """Color a part the way its lines were drawn: start with the region nearest the marker (a larger region wins a
    near tie) and keep moving to the closest uncolored one, instead of scanning the part from the top."""
    remaining, ordered, tip = list(regions), [], np.asarray(tip, float)
    while remaining:
        region = min(remaining, key=lambda item: np.linalg.norm(item['cells'][0][0] - tip) - .5 * math.sqrt(item.get('area', 0)))
        remaining = [item for item in remaining if item is not region]
        ordered.append(region)
        tip = region['cells'][-1][-1]
    return ordered


def timeline(regions, shading, part_count, tip, pace, lifts=1., start=0., total=1., fps=30, record=False, size=1.):
    """Lay out every marker pass and shading band in order; returns (segments, parts, bands, end seconds, tip)."""
    tip = np.asarray(tip, float)
    segments, cursor, parts, bands = [], start, [], []
    shade_time = SHADE_SECONDS * min(1., FILL_SPEED * 1.5 / pace)
    lapse = max(1., pace / (FILL_SPEED * size))

    def travel(destination):
        nonlocal cursor
        lift = lift_seconds(np.linalg.norm(destination - tip), fps, lifts, lapse, size)
        segments.append({'kind': 'travel', 'start': cursor / total, 'end': (cursor + lift) / total,
                         'points': [tip.tolist(), destination.tolist()]})
        cursor += lift

    def paint(region, kind='color'):
        nonlocal cursor, tip
        passes = []
        for cell in region['cells']:
            travel(cell[0])
            weights = weighted_lengths(cell)
            seconds = max(frame_floor(fps, lapse), weights[-1] / pace)
            segments.append({'kind': kind, 'start': cursor / total, 'end': (cursor + seconds) / total,
                             'points': cell.tolist(), 'lengths': weights.tolist(), 'radius': region['radius'] + 2.5})
            passes.append((cell, (cursor + seconds * weights / max(1e-6, weights[-1])) / total))
            cursor += seconds
            tip = cell[-1]
        if record:
            region.update(passes=passes, end=cursor / total, window=3 * region['radius'] / pace / total,
                          coat=min(COAT_DELAY, COAT_REACH * region['radius'] / pace) / total)

    def swap():
        nonlocal cursor
        pause = MARKER_SWAP * min(1., FILL_SPEED / max(pace, 1e-6))
        segments.append({'kind': 'pause', 'start': cursor / total, 'end': (cursor + pause) / total, 'points': [tip.tolist(), tip.tolist()]})
        cursor += pause

    if regions and 'marker' in regions[0]:
        spans = {}
        for number, marker in enumerate(regions[0]['marker_order']):
            group = [region for region in regions if region['marker'] == marker]
            if number:
                swap()
            # One marker in hand: its areas in one efficient round, nearest next, wherever the module.
            ordered = nearest_first(group, tip)
            for region in ordered:
                begin = cursor
                paint(region)
                spans.setdefault(region['part'], []).append((begin, cursor))
            shadows = [(region, shadow) for region in ordered for shadow in region.get('shadows', [])]
            if shadows:
                swap()   # the darker marker of the same colour
            for region, shadow in shadows:
                begin = cursor
                paint(shadow, 'shade')
                spans.setdefault(region['part'], []).append((begin, cursor))
        parts = [[min(a for a, _ in spans[part]) / total, max(b for _, b in spans[part]) / total] if part in spans else [cursor / total] * 2
                 for part in range(part_count)]
        return segments, parts, [], cursor, tip

    for part in range(part_count):
        part_start = cursor
        for region in nearest_first([region for region in regions if region['part'] == part], tip):
            paint(region)
        if part in shading:
            mine = np.vstack([np.c_[np.nonzero(r['mask'])[::-1]].astype(float) + r['offset'] for r in regions if r['part'] == part])
            center, axis, across = axes(mine)
            along, side = (mine - center) @ axis, (mine - center) @ across
            count = int(np.clip(round(np.ptp(side) / 90), 1, 4))
            edges = np.linspace(side.min(), side.max(), count + 1)
            band = {'part': part, 'center': center, 'axis': axis, 'across': across, 'edges': edges,
                    'along': (along.min(), along.max()), 'times': []}
            for index in range(count):
                points = center + np.linspace(along.min(), along.max(), 24)[:, None] * axis + (edges[index] + edges[index + 1]) / 2 * across
                travel(points[0])
                seconds = shade_time / count
                segments.append({'kind': 'shade', 'start': cursor / total, 'end': (cursor + seconds) / total, 'points': points.tolist(),
                                 'lengths': weighted_lengths(points).tolist(), 'radius': float(np.ptp(edges) / count / 2 + 4)})
                band['times'].append((cursor / total, (cursor + seconds) / total))
                cursor += seconds
                tip = points[-1]
            bands.append(band)
        parts.append([part_start / total, cursor / total])
        if part_start < cursor and part + 1 < part_count and any(r['part'] == part + 1 for r in regions):
            pause = PART_PAUSE * min(1., FILL_SPEED / max(pace, 1e-6))
            segments.append({'kind': 'pause', 'start': cursor / total, 'end': (cursor + pause) / total,
                             'points': [tip.tolist(), tip.tolist()]})
            cursor += pause
    return segments, parts, bands, cursor, tip


def fill_cap(speed, fps):
    return min(FILL_SPEED * FILL_FACTOR, FILL_FRAME_PX * fps) * speed


def requirements(regions, rgb, part_count, tip, speed=1., fps=30, shade=True, size=1.):
    """Seconds for: base marker at human pace with shading; a 1.5x marker at the cap with shading;
    the broadest marker at the cap with short lifts and no shading."""
    shading = shaded_parts(regions, rgb, part_count) if shade else []
    result = {}
    cap = fill_cap(speed, fps)
    for key, scale, pace, shade, lifts in (('required', SCALES[0], FILL_SPEED * speed, shading, 1.), ('moderate', SCALES[1], cap, shading, 1.),
                                           ('minimum', SCALES[-1], cap, [], .5)):
        plan(regions, scale, size)
        result[key] = timeline(regions, shade, part_count, tip, pace, lifts, fps=fps, size=size)[3]
    return result


def schedule(regions, rgb, part_count, tip, budget, start, total, speed=1., fps=30, shade=True, size=1.):
    """Pick marker size, pace and lift length so the fill fits the budget: bigger marker first, then pace
    up to the cap; shading is dropped before the cap is exceeded. Returns segments, per-part [start, end]
    fractions, shading bands, end fraction, pen tip and counts."""
    shading = shaded_parts(regions, rgb, part_count) if shade else []
    base, cap = FILL_SPEED * speed, fill_cap(speed, fps)
    seconds = lambda pace, lifts=1., shade=None: timeline(regions, shading if shade is None else shade, part_count, tip, pace, lifts, fps=fps, size=size)[3]
    for scale in SCALES:
        plan(regions, scale, size)
        if seconds(cap) <= budget:
            break
    if seconds(cap) > budget:
        shading = []
    lifts = next((value for value in (1., .7, .5) if seconds(cap, value) <= budget), .5)
    low, high = base / 1.6, cap * 8
    if seconds(low, lifts) <= budget:
        high = low
    for _ in range(40):
        if high - low < 1:
            break
        middle = (low + high) / 2
        low, high = (middle, high) if seconds(middle, lifts) > budget else (low, middle)
    segments, parts, bands, end, tip = timeline(regions, shading, part_count, tip, high, lifts, start, total, fps, record=True, size=size)
    counts = {'fill_regions': len(regions), 'fill_passes': sum(len(r['cells']) for r in regions),
              'marker_radius_scale': scale, 'fill_speed_px_s': round(high, 1), 'fill_over_cap': bool(high > cap * 1.001),
              'shaded_parts': len(shading)}
    return segments, parts, bands, end / total, tip, counts


def bristles(values, seed):
    """Smooth 1-D value noise in [0, 1]: the streaks a marker's felt leaves along its stroke."""
    rng = np.random.default_rng(seed)
    low = math.floor(float(values.min())) - 1
    lattice = rng.random(int(math.ceil(float(values.max()) - low)) + 3)
    cell = np.floor(values - low).astype(int)
    fraction = values - low - cell
    fraction = fraction * fraction * (3 - 2 * fraction)
    return lattice[cell] * (1 - fraction) + lattice[cell + 1] * fraction


def texture(region, box_shape, seed):
    """Alpha of a single wet coat over the region crop: felt streaks running with the strokes (parallel to the edge in
    the edge band), varying slowly along them."""
    top, left = region['offset'][1], region['offset'][0]
    ys, xs = np.mgrid[top:top + box_shape[0], left:left + box_shape[1]].astype(np.float32)
    u = (xs - region['center'][0]) * region['across'][0] + (ys - region['center'][1]) * region['across'][1]
    v = (xs - region['center'][0]) * region['axis'][0] + (ys - region['center'][1]) * region['axis'][1]
    if region.get('edge'):
        u = np.where(region['distance'] <= region['edge'], region['distance'] * 1.7, u)
    # Ink pools unevenly: a slow blotch over the streaks keeps an edge band from reading as contour rings. (Drawn coarse and
    # enlarged: a blur this wide keeps nothing finer than a few px.)
    sigma = max(2., .6 * region['radius'])
    coarse = max(1, int(sigma / 3))
    small = tuple(max(1, -(-side // coarse)) for side in box_shape)
    blotch = ndimage.gaussian_filter(np.random.default_rng(seed + 2).random(small).astype(np.float32), sigma / coarse, mode='nearest')
    if coarse > 1:
        blotch = resized(blotch, (small[0] * coarse, small[1] * coarse))[:box_shape[0], :box_shape[1]]
    blotch = (blotch - blotch.min()) / max(1e-6, float(np.ptp(blotch)))
    streak = .5 * bristles(u / STREAK, seed) + .28 * bristles(u / (3.1 * STREAK) + v / (4 * region['radius']), seed + 1) + .22 * blotch
    return (WET[0] + (WET[1] - WET[0]) * streak).astype(np.float32)


def ramps(total, fps):
    """[front, coat] ramps of the paint layer in fractions of a drawing event lasting `total` seconds."""
    return np.array([FRONT_FRAMES / fps / total, COAT_RAMP / total], np.float32)


def lanes(region, size, seed):
    """Tone multiplier (mean 1 over the region) of layered marker lanes at the region's pixels (in mask order)."""
    (left, top), mask = region['offset'], region['mask']
    ys, xs = np.nonzero(mask)
    xs, ys = (xs + left).astype(np.float32), (ys + top).astype(np.float32)
    centre = region.get('center', np.array([xs.mean(), ys.mean()]))
    axis, across = region.get('axis', np.array([1., 0.])), region.get('across', np.array([0., 1.]))
    u = (xs - centre[0]) * across[0] + (ys - centre[1]) * across[1]
    v = (xs - centre[0]) * axis[0] + (ys - centre[1]) * axis[1]
    nib = NIB * size
    step = LANE_STEP * nib
    rng = np.random.default_rng(seed)
    low = float(u.min()) - step
    count = int(math.ceil((float(u.max()) - low) / step)) + 3
    jitter = (rng.random(count) * 2 - 1) * LANE_JITTER * step
    tilt = np.tan((rng.random(count) * 2 - 1) * LANE_TILT)
    wander = rng.random((count, int(float(np.ptp(v)) / (6 * nib)) + 3))
    along = (v - float(v.min())) / (6 * nib)
    cell = np.clip(np.floor(along).astype(int), 0, wander.shape[1] - 2)
    fraction = np.clip(along - cell, 0, 1)
    fraction = fraction * fraction * (3 - 2 * fraction)
    index = np.floor((u - low) / step).astype(int)
    coats = np.zeros(len(u), np.float32)
    pooled = np.zeros(len(u), np.float32)
    for offset in (-1, 0, 1, 2):
        lane = np.clip(index + offset, 0, count - 1)
        middle = low + lane * step + jitter[lane] + tilt[lane] * v \
            + LANE_WANDER * nib * (2 * (wander[lane, cell] * (1 - fraction) + wander[lane, cell + 1] * fraction) - 1)
        depth = nib - np.abs(u - middle)                       # px inside the lane's edge
        covered = np.clip(depth / LANE_SOFT + .5, 0, 1)
        coats += covered
        pooled += covered * np.exp(-np.square(np.maximum(depth, 0) / max(1., POOL_WIDTH * nib)))
    streak = bristles(u / STREAK_PX, seed + 1) - .5
    shift = -LAYER_TONE * coats - POOL_TONE * pooled - STREAK_TONE * 2 * streak
    return np.clip(1 + shift - float(shift.mean()), *TONE_RANGE).astype(np.float32)


def render(shape, regions, bands, owners, layered=False, size=1.):
    """Marker reveal fields at plan size, for front() to turn into the display maps at any size: first touch time, the
    signed distance (px) to the felt's reach before and when touched, the time a pixel is wholly covered, the streaky
    strength of a wet coat, the time the coat reads solid (second overlapping stroke, or a moment later), the shading
    time (flat -> source), flat colours, the region of each pixel and when each region is done. layered: also the
    finished-colour tone of layered marker coats ('tone', a multiplier around 1)."""
    first = np.full(shape, np.inf, np.float32)
    near = np.full(shape, FAR, np.float32)
    near_time = np.full(shape, np.inf, np.float32)
    inside = np.full(shape, FAR, np.float32)
    solid = np.full(shape, np.inf, np.float32)
    wetness = np.zeros(shape, np.float32)
    full = np.full(shape, np.inf, np.float32)
    flat = np.zeros(shape + (3,), np.float32)
    label = np.full(shape, -1, np.int32)
    tone = np.ones(shape, np.float32) if layered else None
    rng = np.random.default_rng(7)
    grain = (.8 + .2 * rng.random(shape)).astype(np.float32)
    ragged = ndimage.gaussian_filter(rng.random(shape).astype(np.float32), 1.2)
    ragged = (ragged - ragged.min()) / max(1e-6, float(np.ptp(ragged)))
    for index, region in enumerate(regions):
        (left, top), (height, width) = region['offset'], region['mask'].shape
        mask, radius = region['mask'], region['radius']
        edge = max(4., region.get('soft', 0.) * radius)
        box = (slice(top, top + height), slice(left, left + width))
        flat[box][mask] = region['flat']
        label[box][mask] = index
        wet = texture(region, (height, width), 11 + 2 * index) if region.get('streaks', True) else grain[box]
        first_local, grain_local = first[box], ragged[box]
        near_local, near_time_local, inside_local, solid_local = near[box], near_time[box], inside[box], solid[box]
        second_local = np.full((height, width), np.inf, np.float32)
        rough = .3 if region.get('careful', True) else 0.   # a quick broad pass keeps its full, even edge
        for cell, times in region['passes']:
            for (x, y), when in zip(cell - region['offset'], times):
                x0, x1 = int(max(0, x - radius - 3)), int(min(width, x + radius + 4))
                y0, y1 = int(max(0, y - radius - 3)), int(min(height, y + radius + 4))
                if x1 <= x0 or y1 <= y0:
                    continue
                ys, xs = np.ogrid[y0:y1, x0:x1]
                window = (slice(y0, y1), slice(x0, x1))
                inner = mask[window]
                # The felt edge is ragged: coverage only ever shrinks inside the nominal disc.
                raw = (radius + 2 - np.hypot(xs - x, ys - y)) / edge - rough * grain_local[window]
                reach = (raw - TOUCH) * edge     # signed px from the felt's reach (> 0 inside)
                firsts = first_local[window]
                seen = np.isfinite(firsts)
                touched = (raw > TOUCH) & inner & ~seen
                firsts[touched] = when
                # Before the touch, how close the felt has come (and when): the edge between this pixel and a painted one.
                ahead = inner & ~seen & ~touched & (reach > near_local[window])
                near_local[window][ahead] = reach[ahead]
                near_time_local[window][ahead] = when
                # During the touching visit, how deep inside the reach it got.
                current = inner & (seen | touched) & (when - firsts <= region['window']) & (reach > inside_local[window])
                inside_local[window][current] = reach[current]
                whole = inner & (reach >= SOLID) & (when < solid_local[window])
                solid_local[window][whole] = when
                again = (raw > .5) & inner & seen & (when - firsts > region['window']) & ~np.isfinite(second_local[window])
                second_local[window][again] = when
        missing = mask & ~np.isfinite(first_local)
        if missing.any():
            _, nearest = ndimage.distance_transform_edt(~(np.isfinite(first_local) & mask), return_indices=True)
            first_local[missing] = first_local[nearest[0], nearest[1]][missing]
            inside_local[missing], near_local[missing], near_time_local[missing] = -FAR, FAR, np.inf
            solid_local[missing] = first_local[missing]
        solid_local[mask] = np.minimum(np.maximum(solid_local[mask], first_local[mask]), region['end'])
        wetness[box][mask] = wet[mask]
        settled = np.minimum(second_local, first_local + region.get('coat', 0.))
        full[box][mask] = np.minimum(np.maximum(settled, first_local), region['end'])[mask]
        if layered and mask.any():
            tone[box][mask] = lanes(region, size, 31 + 7 * index)
    shade = full.copy()
    for region in regions:
        for shadow in region.get('shadows', []):
            if 'passes' not in shadow:
                continue
            # The shadow side turns from base coat to shadow where the darker marker passes.
            (left, top), (height, width) = shadow['offset'], shadow['mask'].shape
            mask, radius, when_local = shadow['mask'], shadow['radius'], np.full(shadow['mask'].shape, np.inf, np.float32)
            for cell, times in shadow['passes']:
                for (x, y), when in zip(cell - shadow['offset'], times):
                    x0, x1 = int(max(0, x - radius - 2)), int(min(width, x + radius + 3))
                    y0, y1 = int(max(0, y - radius - 2)), int(min(height, y + radius + 3))
                    if x1 <= x0 or y1 <= y0:
                        continue
                    ys, xs = np.ogrid[y0:y1, x0:x1]
                    window = (slice(y0, y1), slice(x0, x1))
                    hit = (np.hypot(xs - x, ys - y) <= radius + 1) & mask[window] & (when < when_local[window])
                    when_local[window][hit] = when
            missing = mask & ~np.isfinite(when_local)
            if missing.any() and np.isfinite(when_local).any():
                _, nearest = ndimage.distance_transform_edt(~np.isfinite(when_local), return_indices=True)
                when_local[missing] = when_local[nearest[0], nearest[1]][missing]
            box = (slice(top, top + height), slice(left, left + width))
            shade[box][mask] = np.maximum(when_local[mask], full[box][mask])
    band_times(shade, bands, owners)
    fields = {'first': first, 'near': near, 'near_time': near_time, 'inside': np.minimum(inside, -FAR), 'solid': solid,
              'wet': wetness, 'full': full, 'shade': shade, 'region': label, 'rim': rims(label), 'group': surfaces(regions),
              'end': np.array([region['end'] for region in regions], np.float32)}
    if layered:
        fields['tone'] = tone
    return fields, flat


def band_times(shade, bands, owners, scale=(1., 1.)):
    """Write into `shade` (any render size; owners at that size) when each broad soft shading stroke passes: bands are
    planned at plan size, so the band a pixel falls in and how far along it lies come from its plan position (scale: render
    px per plan px). Neighbouring bands blend over one render pixel at their shared edge, so the edge is a clean line, not
    a staircase."""
    for band in bands:
        ys, xs = np.nonzero(owners == band['part'])
        if not len(ys):
            continue
        relative = np.c_[(xs + .5) / scale[0] - .5, (ys + .5) / scale[1] - .5] - np.asarray(band['center'], float)
        side, along = relative @ np.asarray(band['across'], float), relative @ np.asarray(band['axis'], float)
        edges = np.asarray(band['edges'], float)
        starts, ends = np.array(band['times'], float).T
        count = len(starts)
        low, high = band['along']
        fraction = np.clip((along - low) / max(1., high - low), 0, 1) * .8
        when = lambda index: starts[index] + fraction * (ends[index] - starts[index])
        index = np.clip(np.searchsorted(edges, side) - 1, 0, count - 1)
        times = when(index)
        if count > 1:
            px = (scale[0] + scale[1]) / 2
            # Within half a render pixel of an inner band edge the pixel straddles both bands: its time blends.
            for neighbour, distance in ((index - 1, side - edges[index]), (index + 1, edges[np.minimum(index + 1, count)] - side)):
                share = np.clip(.5 - distance * px, 0, .5) * ((neighbour >= 0) & (neighbour < count))
                times = times + share * (when(np.clip(neighbour, 0, count - 1)) - times)
        shade[ys, xs] = times


def apply_tone(pixels, tone):
    """RGBA pixels with the layered-marker tone multiplied into their colour. Near-white paint (paper left white, a
    highlight) is not coloured with a marker, so it takes none of it. tone: plan-size or same-size multiplier map."""
    tone = np.asarray(tone, np.float32)
    if tone.shape != pixels.shape[:2]:
        tone = resized(tone, pixels.shape[:2])
    rgb = pixels[..., :3].astype(np.float32)
    strength = np.clip((250. - rgb @ np.array([.299, .587, .114], np.float32)) / 40., 0, 1)
    result = pixels.copy()
    result[..., :3] = np.clip(np.rint(rgb * (1 + strength * (tone - 1))[..., None]), 0, 255).astype(pixels.dtype)
    return result


def remove_tone(pixels, tone):
    """The colours apply_tone started from (to within rounding): a plan layer compared with an untoned render."""
    tone = np.asarray(tone, np.float32)
    rgb = pixels[..., :3].astype(np.float32)
    strength = np.clip((250. - rgb @ np.array([.299, .587, .114], np.float32)) / 40., 0, 1)
    result = pixels.copy()
    result[..., :3] = np.clip(np.rint(rgb / (1 + strength * (tone - 1))[..., None]), 0, 255).astype(pixels.dtype)
    return result


def resized(values, shape):
    """A smooth plan-size field (a tone, a density) at another size: bilinear, pixel centres aligned."""
    return np.asarray(Image.fromarray(np.asarray(values, np.float32)).resize((shape[1], shape[0]), Image.Resampling.BILINEAR), np.float32)


def surfaces(regions):
    """Per region, the group whose regions may share a colouring front where they meet without a visible edge: every
    region when colouring by marker (modules overlap in time by design), else the regions of one module."""
    return np.array([0 if 'marker' in region else region['part'] for region in regions], np.int32)


EDGE_RGB = 32.        # RGB distance below which two neighbouring areas show no edge: the front runs across their border
RIM_SIGMA = 1.        # px: a region's border is smoothed this much before the front follows it across an unseen edge


def rims(label):
    """Smoothed px distance of each pixel to its region's border with another painted region (inf deep inside): the
    border of a region is a staircase of plan pixels; a front that runs along it (where no visible edge separates two
    areas) follows this smooth version instead, so a close-up does not scale up its steps."""
    reach = int(math.ceil(2.5 * RIM_SIGMA))
    padded = np.pad(label, reach, mode='edge')
    h, w = label.shape
    same = np.zeros(label.shape, np.float32)
    total = 0.
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            weight = math.exp(-(dy * dy + dx * dx) / (2 * RIM_SIGMA ** 2))
            other = padded[reach + dy:reach + dy + h, reach + dx:reach + dx + w]
            same += weight * ((other == label) | (other < 0))
            total += weight
    share = same / total
    # A blurred step rises 1 / (sigma sqrt(2 pi)) per px at the border: its level converts back to a distance.
    rim = (share - .5) * RIM_SIGMA * math.sqrt(2 * math.pi)
    return np.where((label >= 0) & (share < 1 - 1e-4), rim, np.inf).astype(np.float32)


FIELDS = ('first', 'near', 'near_time', 'inside', 'solid', 'wet', 'full', 'shade', 'region', 'rim', 'group', 'end', 'frame', 'tone')
TIME_FIELDS = ('first', 'near_time', 'solid', 'full', 'shade', 'end', 'frame')   # fractions of the drawing event


def stored(fields):
    """Fields as asset map entries (paint_*), kept with a plan-size asset so a close-up re-render can redraw the front."""
    return {'paint_' + key: value for key, value in fields.items() if key in FIELDS}


def loaded(maps):
    """The fields stored in an asset's maps, or None for an asset built before the anti-aliased front."""
    if 'paint_first' not in maps:
        return None
    return {key: np.asarray(maps['paint_' + key]) for key in FIELDS if 'paint_' + key in maps}


def neighbours(fields, shape, home=None, colours=None, across_groups=False):
    """(home, picks, weights[, weights across groups]) of each display pixel (flattened): its own plan pixel, and its
    four bilinear plan neighbours with their weights, kept only where they are the same region or meet it with no
    visible edge (see front()) within the region's group (its module, when colouring by module), as paint times must
    be; across_groups: also the weights that ignore the group (flat colours may blend across modules)."""
    h, w = np.asarray(fields['first']).shape
    height, width = shape
    centre = lambda count, size: np.clip((np.arange(count) + .5) * size / count - .5, 0, size - 1)
    fy, fx = centre(height, h), centre(width, w)
    y0, x0 = np.floor(fy).astype(np.int64), np.floor(fx).astype(np.int64)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    wy, wx = (fy - y0).astype(np.float32)[:, None], (fx - x0).astype(np.float32)[None, :]
    if home is None:
        home = (np.rint(fy).astype(np.int64)[:, None], np.rint(fx).astype(np.int64)[None, :])
    home = np.broadcast_to(home[0] * w + home[1], shape).ravel()
    picks = [np.broadcast_to(rows[:, None] * w + columns[None, :], shape).ravel() for rows, columns in ((y0, x0), (y0, x1), (y1, x0), (y1, x1))]
    bilinear = [np.broadcast_to(weight, shape).ravel() for weight in ((1 - wy) * (1 - wx), (1 - wy) * wx, wy * (1 - wx), wy * wx)]
    label = np.asarray(fields['region']).ravel()
    mine = label[home]
    grouped, free = [], []
    group = np.r_[np.asarray(fields['group'], np.int32), -2] if 'group' in fields and len(fields['group']) else None
    flat_colours = np.asarray(colours, np.float32)[..., :3].reshape(-1, 3) if colours is not None else None
    for weight, pick in zip(bilinear, picks):
        other = label[pick]
        same = other == mine
        joined = same.copy()
        loose = same.copy()
        if flat_colours is not None and group is not None:
            # Only where the regions differ: does the picture show an edge between the two plan pixels?
            where = np.nonzero(~same & (other >= 0) & (mine >= 0))[0]
            if len(where):
                seamless = np.square(flat_colours[pick[where]] - flat_colours[home[where]]).sum(-1) < EDGE_RGB ** 2
                loose[where] = seamless
                joined[where] = seamless & (group[other[where]] == group[mine[where]])
        grouped.append(weight * joined)
        free.append(weight * loose)

    def normalised(weights):
        alone = sum(weights) <= 1e-6       # no neighbour shares the pixel's region: it follows its own plan pixel
        first_pick = np.where(alone, home, picks[0])
        weights = [np.where(alone, 1., weights[0])] + weights[1:]
        norm = sum(weights)
        return [first_pick] + picks[1:], [(weight / np.maximum(norm, 1e-12)).astype(np.float32) for weight in weights]

    grouped_picks, grouped_weights = normalised(grouped)
    if not across_groups:
        return home, grouped_picks, grouped_weights
    free_picks, free_weights = normalised(free)
    return home, grouped_picks, grouped_weights, free_picks, free_weights


def front(fields, shape=None, factor=1., home=None, colours=None, flat=None):
    """Display maps of the colouring at `shape` (default: plan size) from plan-size fields: 'color' (when paint first
    shows), 'color_cover' (its level until it settles: the streaky wet coat, times the anti-aliased share of the pixel
    the felt has covered), 'color_full' (when it reads solid) and 'shade' (flat colour -> finished colour).
    Each display pixel interpolates the signed distance to the felt's reach between its four plan neighbours in the same
    region, at each time one of them changes, so the edge between painted and not yet painted is placed between plan
    pixels, FRONT_PX wide. home: (rows, columns) of the plan pixel each display pixel belongs to (its region, its
    settle delay); default the nearest. colours: the plan-size picture (RGB); where two regions of one group meet with
    no visible edge (an outline only the line plate has), the front runs across their border too. flat: the plan-size
    flat marker colours (RGB); returned as 'flat' at this size, blended the same way, so such a border fades over a plan
    pixel instead of stepping."""
    first = np.asarray(fields['first'], np.float32)
    shape = first.shape if shape is None else tuple(shape)
    get = lambda key: np.asarray(fields[key]).ravel()
    blend_across = flat is not None and colours is not None and len(np.unique(fields.get('group', [0]))) > 1
    found = neighbours(fields, shape, home, colours, across_groups=blend_across)
    home, picks, weights = found[:3]
    flat_picks, flat_weights = found[3:] if blend_across else (picks, weights)
    label = get('region')
    mine = label[home]
    painted = (mine >= 0) & np.isfinite(first.ravel()[home])
    gain = np.float32(factor / FRONT_PX)
    rim = get('rim') if 'rim' in fields else np.full(first.size, np.inf, np.float32)
    tolerance = float(fields.get('frame', 0.)) / 2
    firsts = [first.ravel()[pick] for pick in picks]
    used = [weight > 0 for weight in weights]
    earliest = np.minimum.reduce([np.where(use, value, np.inf) for use, value in zip(used, firsts)])
    latest = np.maximum.reduce([np.where(use, value, -np.inf) for use, value in zip(used, firsts)])
    solids = [get('solid')[pick] for pick in picks]
    # Inside a stroke every neighbour is touched within half a frame and at once wholly: the front passes between frames
    # there, and the pixel simply comes in with the first of them. Only an edge that stays an edge needs the front.
    with np.errstate(invalid='ignore'):
        simple = painted & (latest - earliest <= tolerance)
        for use, pick, value, solid in zip(used, picks, firsts, solids):
            simple &= ~use | ((get('inside')[pick] >= 1.) & ~np.isfinite(rim[pick]) & (solid - value <= tolerance))
    color = np.where(simple, earliest, np.inf).astype(np.float32)
    whole = np.where(simple, np.maximum.reduce([np.where(use, solid, -np.inf) for use, solid in zip(used, solids)]), np.inf).astype(np.float32)
    plateau = np.where(simple, 1., 0.).astype(np.float32)
    active = np.nonzero(painted & ~simple)[0]
    if len(active):
        # Each neighbour's states (signed px to the paint's edge) before it is approached, near, touched and wholly
        # covered; a painted pixel by a region's border reads no deeper than the smoothed border, so where two areas meet
        # without a visible edge the front follows that border, not its staircase of plan pixels.
        # A display pixel between two regions sees each unpainted side at its own smoothed border too (never so close
        # that it would show paint before either side is touched).
        regions = [np.where(weight[active] > 0, label[pick[active]], -1) for weight, pick in zip(weights, picks)]
        straddle = np.maximum.reduce(regions) != np.minimum.reduce([np.where(region >= 0, region, np.iinfo(np.int32).max) for region in regions])
        floor = .5 / gain + .02
        states = []
        for weight, pick in zip(weights, picks):
            pick = pick[active]
            edge = rim[pick]
            inside = get('inside')[pick]
            outside = np.where(straddle, -np.maximum(edge, floor), -np.inf)
            states.append((weight[active], get('near_time')[pick], np.maximum(get('near')[pick], outside), first.ravel()[pick],
                           np.minimum(inside, edge), get('solid')[pick], np.minimum(np.maximum(inside, 1.), edge), np.maximum(FAR, outside)))
        level = lambda at: np.clip(sum(weight * np.where(solid <= at, deep, np.where(touch <= at, inside, np.where(near_at <= at, near, far)))
                                       for weight, near_at, near, touch, inside, solid, deep, far in states) * gain + .5, 0, 1)
        events = [np.where(state[0] > 0, times, np.inf) for state in states for times in (state[1], state[3], state[5])]
        sub_color = np.full(len(active), np.inf, np.float32)
        sub_whole = np.full(len(active), np.inf, np.float32)
        seen = []
        for at in events:
            amount = level(at)
            finite = np.isfinite(at)
            sub_color = np.where(finite & (amount >= 1 / 255) & (at < sub_color), at, sub_color)
            sub_whole = np.where(finite & (amount >= 254 / 255) & (at < sub_whole), at, sub_whole)
            seen.append((at, amount))
        sub_plateau = np.zeros(len(active), np.float32)
        for at, amount in seen:
            sub_plateau = np.where(np.isfinite(at) & (at < sub_whole) & (amount > sub_plateau), amount, sub_plateau)
        color[active], whole[active], plateau[active] = sub_color, sub_whole, sub_plateau
    # A pixel the front never quite reaches (a sliver thinner than its smoothed border) comes in with its own plan pixel.
    lost = painted & ~np.isfinite(color)
    color = np.where(lost, first.ravel()[home], color)
    # A coat settles its own delay after it arrives (the next overlapping stroke, or a moment later) and never before the
    # pixel is wholly covered; every pixel of a region is solid by the time the region is done.
    with np.errstate(invalid='ignore'):
        delay = blend_same(get('full') - first.ravel(), picks, weights)
        lag = blend_same(get('shade') - get('full'), picks, weights)
    settle = color + np.where(np.isfinite(delay), delay, 0.)
    # The partial (anti-aliased) level holds only on an edge that stays an edge: a pixel the felt covers before its coat
    # would settle anyway is the moving front of a stroke, and shows the whole wet coat.
    plateau = np.where(np.isfinite(color) & ((whole <= settle) | lost), 1., plateau)
    wet = sum(weight * np.where(painted, get('wet')[pick], 0.) for weight, pick in zip(weights, picks))
    cover = np.where(np.isfinite(color), np.clip(wet, 0, 1) * plateau, 0.)
    ends = np.asarray(fields['end'], np.float32)
    done = np.where(mine >= 0, ends[np.maximum(mine, 0)] if len(ends) else np.inf, np.inf)
    full = np.minimum(np.maximum(settle, whole), done)
    full = np.where(np.isfinite(color), np.maximum(full, color), np.inf)
    shade = np.where(np.isfinite(color), full + np.where(np.isfinite(lag), lag, 0.), get('shade')[home])
    result = {key: value.reshape(shape).astype(np.float32) for key, value in
              (('color', color), ('color_cover', cover), ('color_full', full), ('shade', shade))}
    if flat is not None:   # flat colours blend across modules too
        flat = np.asarray(flat, np.float32)[..., :3].reshape(-1, 3)
        result['flat'] = np.rint(sum(weight[:, None] * flat[pick] for weight, pick in zip(flat_weights, flat_picks))).reshape(shape + (3,)).astype(np.uint8)
    return result


def blend_same(values, picks, weights):
    """Weighted mean of a field (flattened) over the neighbours that have a finite value."""
    total, sum_weights = 0., 0.
    for pick, weight in zip(picks, weights):
        value = values[pick]
        usable = weight * np.isfinite(value)
        total = total + np.where(usable > 0, value, 0.) * usable
        sum_weights = sum_weights + usable
    return np.where(sum_weights > 0, total / np.maximum(sum_weights, 1e-12), np.inf).astype(np.float32)
