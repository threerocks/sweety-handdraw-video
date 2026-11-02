"""Whole-image line topology: clean the outline once, skeletonize it once, and join pieces through
junctions by tangent continuity so a stroke follows the line a person would draw in one motion."""
from collections import deque
import math
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from stroke_paths import thin_ink, calculate_lengths

STEPS = [(-1, 0), (0, 1), (1, 0), (0, -1), (-1, -1), (-1, 1), (1, 1), (1, -1)]
MICRO = 15.          # px; shorter pieces are absorbed by a neighbour or treated as texture
MAX_STROKE = 420.    # px; longer runs are split where a hand would re-grip


def clean_outline(outline, hole=12, speck=20):
    """Fill pinholes inside ink and drop isolated specks before skeletonizing."""
    gaps, _ = ndimage.label(~outline)
    small = np.bincount(gaps.ravel()) <= hole
    small[0] = False
    filled = outline | small[gaps]
    blobs, _ = ndimage.label(filled, structure=np.ones((3, 3)))
    keep = np.bincount(blobs.ravel()) >= speck
    keep[0] = False
    return keep[blobs]


def graph(skeleton):
    ys, xs = np.nonzero(skeleton)
    pixels = set(zip(ys.tolist(), xs.tolist()))
    return {(y, x): [(y + dy, x + dx) for dy, dx in STEPS if (y + dy, x + dx) in pixels
                     and not (dy and dx and ((y, x + dx) in pixels or (y + dy, x) in pixels))]
            for y, x in pixels}


def trace(adjacency):
    """Split the skeleton graph into chains between junctions/endpoints (x, y order)."""
    used, paths = set(), []

    def walk(start, step):
        chain, previous, current = [start, step], start, step
        used.add(frozenset((start, step)))
        while len(adjacency[current]) == 2 and current != start:
            following = [n for n in adjacency[current] if n != previous]
            if not following or frozenset((current, following[0])) in used:
                break
            previous, current = current, following[0]
            used.add(frozenset((previous, current)))
            chain.append(current)
        return np.array([(x, y) for y, x in chain], float)

    for node in sorted(adjacency, key=lambda p: (len(adjacency[p]) == 2, p)):
        if not adjacency[node]:
            paths.append(np.array([[node[1], node[0]], [node[1] + .5, node[0]]], float))
        for step in adjacency[node]:
            if frozenset((node, step)) not in used:
                paths.append(walk(node, step))
    return paths


def prune_spurs(skeleton, limit=6., rounds=2):
    skeleton = skeleton.copy()
    for _ in range(rounds):
        adjacency = graph(skeleton)
        for path in trace(adjacency):
            ends = [(int(path[0][1]), int(path[0][0])), (int(path[-1][1]), int(path[-1][0]))]
            degrees = [len(adjacency.get(end, ())) for end in ends]
            if calculate_lengths(path)[-1] < limit and min(degrees) <= 1 and max(degrees) >= 3:
                junction = ends[int(np.argmax(degrees))]
                for x, y in path.astype(int):
                    if (y, x) != junction:
                        skeleton[y, x] = False
    return skeleton


def end_direction(path, at_start, reach=6):
    k = min(reach, len(path) - 1)
    vector = path[k] - path[0] if at_start else path[-1 - k] - path[-1]
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-6 else vector


def supported(a, b, support_distance):
    samples = np.linspace(a, b, max(3, int(np.linalg.norm(b - a) * 2) + 1))
    values = support_distance[np.rint(samples[:, 1]).astype(int), np.rint(samples[:, 0]).astype(int)]
    return (values <= 1.5).mean() >= .8


def join(paths, support_distance, gap=10., turn=35.):
    """Pair path ends whose tangents continue (<= turn degrees); bridge small gaps only over ink."""
    joins = 0
    while len(paths) > 1:
        ends = [(index, start, path[0] if start else path[-1]) for index, path in enumerate(paths) for start in (True, False)]
        proposals = []
        for first, second in cKDTree([end[2] for end in ends]).query_pairs(gap):
            (i, start_i, point_i), (j, start_j, point_j) = ends[first], ends[second]
            if i == j:
                continue
            cosine = -np.dot(end_direction(paths[i], start_i), end_direction(paths[j], start_j))
            angle = math.degrees(math.acos(float(np.clip(cosine, -1, 1))))
            distance = float(np.linalg.norm(point_i - point_j))
            if angle <= turn and (distance <= 1.5 or supported(point_i, point_j, support_distance)):
                proposals.append((angle + 4 * distance, i, start_i, j, start_j, distance))
        used, joined = set(), []
        for _, i, start_i, j, start_j, distance in sorted(proposals):
            if i in used or j in used:
                continue
            used |= {i, j}
            head = paths[i][::-1] if start_i else paths[i]
            tail = paths[j] if start_j else paths[j][::-1]
            bridge = np.linspace(head[-1], tail[0], max(2, int(math.ceil(distance)) + 1))[1:-1]
            joined.append(np.vstack([head, bridge, tail[1:] if distance <= 1.5 else tail]))
        if not joined:
            break
        joins += len(joined)
        paths = [path for index, path in enumerate(paths) if index not in used] + joined
    return paths, joins


def resample(path, step=1.):
    lengths = calculate_lengths(path)
    if lengths[-1] < step:
        return path[[0, -1]].astype(float)
    samples = np.linspace(0, lengths[-1], max(2, int(lengths[-1] / step) + 1))
    return np.c_[np.interp(samples, lengths, path[:, 0]), np.interp(samples, lengths, path[:, 1])]


def turning(points, window=4):
    index = np.arange(len(points))
    before = points - points[np.clip(index - window, 0, len(points) - 1)]
    after = points[np.clip(index + window, 0, len(points) - 1)] - points
    norms = np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1)
    cosine = np.where(norms > 1e-9, (before * after).sum(1) / np.maximum(norms, 1e-9), 1.)
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def split_for_hand(points, corner=120., maximum=MAX_STROKE):
    """Lift only at hairpin corners and re-grip on very long runs; ordinary corners are drawn through
    (the nib slows there via the curvature weighting)."""
    if len(points) < 30:
        return [points]
    angles = turning(points)
    peaks = [k for k in range(15, len(points) - 15) if angles[k] >= corner and angles[k] == angles[k - 3:k + 4].max()]
    pieces, last = [], 0
    for k in peaks:
        if k - last >= 15:
            pieces.append(points[last:k + 1])
            last = k
    pieces.append(points[last:])
    result = []
    while pieces:
        piece = pieces.pop(0)
        if len(piece) <= maximum:
            result.append(piece)
            continue
        third = len(piece) // 3
        middle = third + int(np.argmax(turning(piece)[third:2 * third]))
        pieces[:0] = [piece[:middle + 1], piece[middle:]]
    return result


def extract_strokes(outline, support=None, size=1.):
    """Return whole strokes [{'points','radius','closed','micro'}] for an outline mask, and statistics.
    size: how much larger than the calibration size the picture is drawn (line_drawing.picture_size); a hand re-grips
    after a proportionally longer run on a larger picture."""
    clean = clean_outline(outline)
    if not clean.any():
        clean = outline.copy()
    skeleton = prune_spurs(thin_ink(clean))
    if not skeleton.any() and clean.any():
        # A mark only a few pixels across (an enlarged tiny ring) thins and prunes to nothing: trace its rim instead,
        # so the pen still draws it rather than the picture having no strokes at all.
        skeleton = clean & ~ndimage.binary_erosion(clean)
    support_mask = clean if support is None else (support | clean)
    support_distance = ndimage.distance_transform_edt(~support_mask)
    raw = trace(graph(skeleton))
    paths, joins = join(raw, support_distance)
    paths = [resample(path) for path in paths]
    lengths = [calculate_lengths(path)[-1] for path in paths]
    shortest = min(MICRO, .04 * max(outline.shape))
    long_paths = [path for path, length in zip(paths, lengths) if length >= shortest]
    tree = cKDTree(np.vstack(long_paths)) if long_paths else None
    strokes, absorbed = [], 0
    half_width = ndimage.distance_transform_edt(clean)
    for path, length in zip(paths, lengths):
        micro = length < shortest
        if micro and tree is not None and tree.query(path)[0].max() <= 3:
            absorbed += 1
            continue
        if len(path) > 9:
            path = np.r_[path[:1], ndimage.uniform_filter1d(path, 5, axis=0, mode='nearest')[1:-1], path[-1:]]
        closed = len(path) > 8 and np.linalg.norm(path[0] - path[-1]) <= 1.5
        for piece in ([path] if micro else split_for_hand(path, maximum=MAX_STROKE * size)):
            cells = np.clip(np.rint(piece).astype(int), 0, [clean.shape[1] - 1, clean.shape[0] - 1])
            radius = ndimage.uniform_filter1d(np.clip(half_width[cells[:, 1], cells[:, 0]] - .6, .6, 5.), 7, mode='nearest')
            strokes.append({'points': piece, 'radius': radius, 'closed': bool(closed and len(piece) == len(path)),
                            'micro': micro, 'length': float(calculate_lengths(piece)[-1])})
    statistics = {'outline_px': int(outline.sum()), 'holes_filled_px': int((clean & ~outline).sum()),
                  'specks_removed_px': int((outline & ~clean).sum()), 'skeleton_px': int(skeleton.sum()),
                  'raw_pieces': len(raw), 'junction_joins': joins, 'micro_absorbed': absorbed}
    return strokes, clean, skeleton, statistics


def cover_ink_edges(strokes, visible_ink, maximum=8., support=None):
    """Visit broad source corners before timing; never stretch a timestamp away from the nib.

    Short excursions stay inside the source line zone and return to the same contour. Their
    added length is included in ordinary speed/budget checks. Curved edges follow their connected ink zone instead of crossing paper.
    """
    ys, xs = np.nonzero(visible_ink)
    candidates = np.c_[xs, ys]
    visits = 0
    support = visible_ink if support is None else support
    if not len(candidates) or not strokes:
        return visits
    while True:
        counts = np.array([len(stroke['points']) for stroke in strokes])
        points = np.vstack([stroke['points'] for stroke in strokes])
        distances, nearest = cKDTree(points).query(candidates)
        worst = int(distances.argmax())
        if distances[worst] < maximum:
            return visits
        stroke_index = int(np.searchsorted(np.cumsum(counts), nearest[worst], side='right'))
        point_index = int(nearest[worst] - counts[:stroke_index].sum())
        stroke = strokes[stroke_index]
        origin, target = stroke['points'][point_index], candidates[worst]
        outward = np.linspace(origin, target, max(3, int(math.ceil(distances[worst])) + 1))
        cells = np.rint(outward).astype(int)
        if not support[cells[:, 1], cells[:, 0]].all():
            # A curved contour can make the nearest straight shortcut cross paper. Walk
            # inside the existing ink zone back to a contour sample instead.
            anchors = {tuple(cell[::-1]): index for index, cell in enumerate(np.rint(points).astype(int))}
            start = tuple(target[::-1])
            parents, queue, end = {start: None}, deque([start]), None
            while queue:
                cell = queue.popleft()
                if cell in anchors:
                    end = cell
                    break
                for dy, dx in STEPS:
                    next_cell = cell[0] + dy, cell[1] + dx
                    y, x = next_cell
                    if 0 <= y < support.shape[0] and 0 <= x < support.shape[1] and support[y, x] and next_cell not in parents:
                        parents[next_cell] = cell
                        queue.append(next_cell)
            if end is None:
                # An isolated source mark needs its own stroke and pen lift, never a
                # connector to another component or a remote pixel's timestamp.
                component = np.zeros_like(visible_ink)
                for y, x in parents:
                    component[y, x] = visible_ink[y, x]
                details, _, _, _ = extract_strokes(component)
                if not details and component.any():
                    # A compact blot whose skeleton thins away still needs the nib: one short dab across it.
                    ys, xs = np.nonzero(component)
                    cells = np.c_[xs, ys].astype(float)
                    center = cells.mean(0)
                    axis = np.linalg.eigh(np.cov(cells.T))[1][:, -1] if len(cells) > 2 else np.array([1., 0.])
                    reach = max(1., float(np.abs((cells - center) @ axis).max()))
                    dab = resample(np.array([center - reach * axis, center + reach * axis]))
                    details = [{'points': dab, 'radius': np.full(len(dab), 1.5), 'closed': False, 'micro': True,
                                'length': float(calculate_lengths(dab)[-1])}]
                if not details:
                    raise ValueError('Visible ink has no drawable contour; repair the outline')
                strokes.extend(details)
                visits += len(details)
                continue
            anchor = anchors[end]
            stroke_index = int(np.searchsorted(np.cumsum(counts), anchor, side='right'))
            point_index = int(anchor - counts[:stroke_index].sum())
            stroke = strokes[stroke_index]
            path = [stroke['points'][point_index]]
            while end is not None:
                path.append(np.array(end[::-1], float))
                end = parents[end]
            outward = np.array(path)
            outward = outward[np.r_[True, np.linalg.norm(np.diff(outward, axis=0), axis=1) > 1e-8]]
        excursion = np.vstack([outward[1:], outward[-2::-1]])
        stroke['points'] = np.vstack([stroke['points'][:point_index + 1], excursion, stroke['points'][point_index + 1:]])
        stroke['radius'] = np.r_[stroke['radius'][:point_index + 1], np.full(len(excursion), .6), stroke['radius'][point_index + 1:]]
        stroke['length'] = float(calculate_lengths(stroke['points'])[-1])
        visits += 1
