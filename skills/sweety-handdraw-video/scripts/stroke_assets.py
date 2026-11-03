from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.spatial import cKDTree
import hashlib
import handwriting
import illustration_layers
import line_drawing
import line_topology
import lineart_gate
import paint_motion
from motion_profile import time_fraction
import stroke_paths as stroke_utils

ASSET_DIRECTORY = None
FONT_PATH = None
FONT_INDEX = 0
FPS = 30


def configure_assets(directory, font_path, font_index=0, fps=30):
    global ASSET_DIRECTORY, FONT_PATH, FONT_INDEX, FPS
    ASSET_DIRECTORY = Path(directory)
    ASSET_DIRECTORY.mkdir(parents=True, exist_ok=True)
    FONT_PATH = str(font_path)
    FONT_INDEX = int(font_index)
    FPS = int(fps)


def lengths(points):
    return stroke_utils.calculate_lengths(points)


def ordered_paths(mask):
    skeleton = stroke_utils.thin_ink(mask)
    paths = [np.array(path) for path in stroke_utils.trace_strokes(skeleton)]
    covered = np.zeros(skeleton.shape, dtype=bool)
    for path in paths:
        coordinates = np.rint(path).astype(int)
        covered[coordinates[:, 1], coordinates[:, 0]] = True
    # Follow only adjacent leftover pixels. A branch starts a new physical stroke.
    pending = set(zip(*np.where(skeleton & ~covered)))
    while pending:
        point = min(pending)
        chain = []
        while point in pending:
            pending.remove(point)
            y, x = point
            chain.append([x, y])
            neighbors = sorted((y+dy, x+dx) for dy in (-1,0,1) for dx in (-1,0,1)
                               if (dy or dx) and (y+dy, x+dx) in pending)
            if not neighbors:
                break
            point = neighbors[0]
        if len(chain) == 1:
            chain.append([chain[0][0]+.01, chain[0][1]])
        paths.append(np.array(chain, dtype=float))
    if not paths:
        ys, xs = np.where(mask)
        if not len(xs):
            return []
        paths = [np.array([[xs[0],ys[0]],[xs[0]+.01,ys[0]]],dtype=float)]
    covered[:] = False
    for path in paths:
        coordinates = np.rint(path).astype(int)
        covered[coordinates[:,1],coordinates[:,0]] = True
    # Keep broad stroke tips within nib distance without connecting across blank paper. The nib reach follows the
    # glyph's own stroke width: a large title's thick strokes are covered by the skeleton, not by dozens of stubs.
    widths = ndimage.distance_transform_edt(mask)[skeleton]
    reach = max(7., 1.5 * float(np.median(widths))) if len(widths) else 7.
    while True:
        distance, nearest = ndimage.distance_transform_edt(~covered, return_indices=True)
        distance[~mask] = 0
        y, x = np.unravel_index(np.argmax(distance), mask.shape)
        if distance[y,x] <= reach:
            break
        anchor = np.array([nearest[1,y,x],nearest[0,y,x]],dtype=float)
        extension = np.linspace(anchor,[x,y],max(2,math.ceil(distance[y,x])+1))
        coordinates = np.rint(extension).astype(int)
        if not mask[coordinates[:,1],coordinates[:,0]].all():
            extension = np.array([[x,y],[x+.01,y]],dtype=float)
            coordinates = np.rint(extension).astype(int)
        paths.append(extension)
        covered[coordinates[:,1],coordinates[:,0]] = True
    oriented = [path if tuple(path[0][::-1]) <= tuple(path[-1][::-1]) else path[::-1] for path in paths]
    return sorted(oriented,key=lambda path:(round(path[:,1].min()/16),path[:,0].min()))


def time_paths(paths, shape, start=0., duration=1.):
    reveal = np.full(shape, np.inf, dtype=np.float32)
    segments = []
    weights = [max(1., lengths(path)[-1]) for path in paths]
    travel_weights = [max(3., np.linalg.norm(paths[i + 1][0] - paths[i][-1]) * .22) for i in range(len(paths) - 1)]
    total = sum(weights) + sum(travel_weights)
    cursor = start
    for index, (path, weight) in enumerate(zip(paths, weights)):
        path_length = lengths(path)
        interval = duration * weight / max(1, total)
        for point, distance in zip(path, path_length):
            x, y = np.rint(point).astype(int)
            if 0 <= y < shape[0] and 0 <= x < shape[1]:
                reveal[y, x] = min(reveal[y, x], cursor + interval * distance / max(.001, path_length[-1]))
        segments.append({'kind':'ink','start':cursor,'end':cursor + interval,'points':path.tolist(),'lengths':path_length.tolist()})
        cursor += interval
        if index + 1 < len(paths):
            transit = duration * travel_weights[index] / max(1, total)
            segments.append({'kind':'travel','start':cursor,'end':cursor + transit,'points':[path[-1].tolist(),paths[index + 1][0].tolist()]})
            cursor += transit
    if np.isfinite(reveal).any():
        _, nearest = ndimage.distance_transform_edt(~np.isfinite(reveal), return_indices=True)
        reveal = reveal[nearest[0], nearest[1]]
    return reveal, segments


def save_asset(name, pixels, reveal, segments, extra=None):
    Image.fromarray(pixels).save(ASSET_DIRECTORY / f'{name}.png')
    np.savez_compressed(ASSET_DIRECTORY / f'{name}.npz', reveal=reveal.astype(np.float32))
    metadata = {'name':name,'width':pixels.shape[1],'height':pixels.shape[0],'segments':segments,**(extra or {})}
    (ASSET_DIRECTORY / f'{name}.json').write_text(json.dumps(metadata,ensure_ascii=False))
    return metadata


def measure_text(text, font, margin=5):
    """Reserve actual glyph bearings and descenders, not the nominal font size."""
    placements, cursor = [], 0.
    for character in text:
        placements.append((character, round(cursor), font.getbbox(character)))
        cursor += font.getlength(character)
    left = min([0] + [x + bounds[0] for _, x, bounds in placements])
    top = min([0] + [bounds[1] for _, _, bounds in placements])
    right = max([math.ceil(cursor)] + [x + bounds[2] for _, x, bounds in placements])
    bottom = max([0] + [bounds[3] for _, _, bounds in placements])
    return {'width': right-left+2*margin, 'height': bottom-top+2*margin,
            'origin': [margin-left, margin-top], 'placements': placements}


def build_text(name, text, size, color='#302C26', seconds=None, stroke_order=True):
    font = ImageFont.truetype(FONT_PATH, size, index=FONT_INDEX)
    line = handwriting.build_line(text, font, size, color, seconds, FPS) if stroke_order else None
    if line is not None:
        pixels, reveal, owners, segments, characters, origin, report = line
        metadata = save_asset(name,pixels,reveal,segments,{'type':'handwritten_text','text':text,'font_size':size,
                              'text_origin':origin,'safe_margin':5,'characters':characters,'handwriting':report})
        np.savez_compressed(ASSET_DIRECTORY / f'{name}.npz', reveal=reveal, owners=owners)
        return metadata
    # Whole-line font fallback keeps one consistent look when any ideograph lacks stroke data.
    missing = [character for character in text if handwriting.is_hanzi(character) and handwriting.character_data(character) is None]
    report = {'mode':'font','reason':'missing stroke data: '+''.join(dict.fromkeys(missing)) if missing else 'stroke order disabled'}
    layout = measure_text(text, font)
    width, height = layout['width'], layout['height']
    layer = Image.new('RGBA',(width,height))
    glyphs = []
    for character, offset, bounds in layout['placements']:
        glyph = Image.new('L',(max(1,bounds[2]-bounds[0]),max(1,bounds[3]-bounds[1])))
        glyph_draw = ImageDraw.Draw(glyph)
        glyph_draw.text((-bounds[0],-bounds[1]),character,font=font,fill=255)
        mask = np.array(glyph) > 30
        paths = [handwriting.orient(path) for path in ordered_paths(mask)]
        left = layout['origin'][0]+offset+bounds[0]
        top = layout['origin'][1]+bounds[1]
        glyphs.append((character, left, top, np.array(glyph), paths))
        ink = Image.new('RGBA',glyph.size,color)
        ink.putalpha(glyph)
        layer.alpha_composite(ink,(left,top))
    pixels = np.array(layer)
    reveal = np.full((height, width), np.inf, dtype=np.float32)
    owners = np.full((height, width), -1, dtype=np.int16)
    segments, character_intervals = [], []
    weights = [max(4., sum(lengths(path)[-1] for path in paths)) for _, _, _, _, paths in glyphs]
    cursor = 0.
    for index, ((character, left, top, glyph, paths), weight) in enumerate(zip(glyphs, weights)):
        interval = weight / sum(weights)
        if paths:
            local, strokes = time_paths(paths, glyph.shape, cursor, interval * .94)
            glyph_height, glyph_width = glyph.shape
            mask = glyph > 0
            patch = reveal[top:top+glyph_height, left:left+glyph_width]
            patch[mask] = local[mask]
            owners[top:top+glyph_height, left:left+glyph_width][mask] = index
            for stroke in strokes:
                stroke['points'] = (np.array(stroke['points']) + [left, top]).tolist()
                stroke['character_index'] = index
            segments.extend(strokes)
        character_intervals.append({'index':index,'character':character,'start':cursor,'end':cursor+interval})
        cursor += interval
    metadata = save_asset(name,pixels,reveal,segments,{'type':'handwritten_text','text':text,'font_size':size,
                          'text_origin':layout['origin'],'safe_margin':5,'characters':character_intervals,'handwriting':report})
    np.savez_compressed(ASSET_DIRECTORY / f'{name}.npz', reveal=reveal, owners=owners)
    return metadata


def build_brush(name, width, height, color, style='highlight', line_width=None, seconds=None):
    if style == 'highlight':
        radius = min(20, height / 2.4)
        xs = np.linspace(radius,width - radius,max(5,math.ceil(width / 3)))
        amplitude = max(0,height / 2 - radius)
        ys = height / 2 + amplitude * np.sin((xs - radius) / 17 * math.pi)
        points = np.c_[xs,ys]
    elif style == 'underline':
        radius = (line_width or 5) / 2
        xs = np.linspace(4,width - 4,max(3,math.ceil(width / 2)))
        points = np.c_[xs,height / 2 + 2 * np.sin(xs / max(1,width) * math.pi * 1.4)]
    elif style == 'circle':
        radius = (line_width or 5) / 2
        theta = np.linspace(-math.pi / 3,5 * math.pi / 3,max(50,round((width + height) / 2)))
        points = np.c_[width / 2 + (width / 2 - radius - 3) * np.cos(theta),height / 2 + (height / 2 - radius - 3) * np.sin(theta)]
    elif style == 'box':
        radius = (line_width or 4) / 2
        corners = np.array([[5,5],[width-6,4],[width-5,height-6],[4,height-5],[5,5]],dtype=float)
        points = np.vstack([np.linspace(a,b,max(2,round(np.linalg.norm(b-a)/2))) for a,b in zip(corners[:-1],corners[1:])])
    elif style == 'bar':
        # A chart bar colored like a marker: horizontal zigzag rows rising from the base, covering the whole box.
        radius = float(np.clip(min(width, height) / 3, 4, 14))
        spacing = 1.3 * radius
        rows = np.arange(height - .6 * radius, .6 * radius - spacing, -spacing)
        rows = np.r_[rows[rows > .6 * radius], .6 * radius] if len(rows) else np.array([height / 2])
        points = []
        for index, y in enumerate(rows):
            xs = np.linspace(.4 * radius, width - .4 * radius, max(2, int(width / 2)))
            points.append(np.c_[xs if index % 2 == 0 else xs[::-1], np.full(len(xs), y)])
        points = np.vstack(points)
        radius = spacing * .8
    else:
        raise ValueError(style)
    rgba = np.zeros((height,width,4),dtype=np.uint8)
    rgba[:,:,:3] = tuple(int(color[index:index+2],16) for index in (1,3,5))
    reveal = np.full((height,width),np.inf,dtype=np.float32)
    distances = lengths(points)
    yy,xx = np.ogrid[:height,:width]
    for point,distance in zip(points,distances):
        footprint = (xx-point[0])**2 + (yy-point[1])**2 <= radius**2
        reveal[footprint] = np.minimum(reveal[footprint],distance/max(.01,distances[-1]))
    if style == 'bar':
        # Every pixel of the bar is colored when the marker passes its nearest point; the brush reach is measured.
        ys, xs = np.mgrid[:height, :width]
        reach, nearest = cKDTree(points).query(np.c_[xs.ravel(), ys.ravel()])
        reveal = (distances[nearest] / max(.01, distances[-1])).reshape(height, width).astype(np.float32)
        radius = float(reach.max()) + 1.
    rgba[:,:,3] = np.isfinite(reveal).astype(np.uint8)*255
    segments=[{'kind':'brush','start':0.,'end':1.,'points':points.tolist(),'lengths':distances.tolist(),'radius':radius}]
    return save_asset(name,rgba,reveal,segments,{'type':'marker_'+style})


def line_points(points, curve=0.):
    """Polyline through the given points; with curve != 0 a two-point line bows by curve x its length."""
    points = np.asarray(points, float)
    if len(points) == 2 and curve:
        start, end = points
        normal = np.array([-(end - start)[1], (end - start)[0]])
        control = (start + end) / 2 + curve * normal
        u = np.linspace(0, 1, max(8, int(np.linalg.norm(end - start) / 4)))[:, None]
        return (1 - u) ** 2 * start + 2 * u * (1 - u) * control + u * u * end
    return np.vstack([np.linspace(a, b, max(2, int(np.linalg.norm(b - a) / 2) + 1))[:-1] for a, b in zip(points[:-1], points[1:])] + [points[-1:]])


PRESSURE = .6   # default pen pressure variation of lines and arrows (0 = an even technical-pen line)


def pressure_widths(path, width, pressure=PRESSURE, flick=False):
    """Nib width along a stroke as a hand presses it: the nib lands and swells over the first few px, thins a little
    where the hand moves fastest (a minimum-jerk speed peak mid-stroke) and lifts off in a longer taper at the end.
    A flick (an arrowhead barb) is short and quick: light at both ends, with no thinning. The body of a stroke keeps
    the given width; pressure 0 gives an even line."""
    distances = lengths(path)
    total = max(1., float(distances[-1]))
    if pressure <= 0:
        return np.full(len(path), float(width))
    smooth = lambda value: value * value * (3 - 2 * value)
    u = distances / total
    land = smooth(np.clip(distances / max(2., min(10., .12 * total)), 0, 1))
    lift = smooth(np.clip((total - distances) / max(3., min(26., .28 * total)), 0, 1))
    if flick:
        weight = (.5 + .5 * land) * (.6 + .4 * lift)
    else:
        speed = 16 * (u * (1 - u)) ** 2               # minimum-jerk speed profile, 1 at mid-stroke
        weight = (.3 + .7 * land) * (.12 + .88 * lift) * (1 - .25 * speed) * (1 + .04 * np.sin(distances / 23. + len(path)))
    weight = weight / max(1e-6, float(np.median(weight)))
    return float(width) * ((1 - pressure) + pressure * weight)


def ink_mask(strokes, widths, size):
    """Anti-aliased coverage (0..255) of round-nib strokes whose width varies point by point."""
    alpha = np.zeros((int(size[1]), int(size[0])), np.float32)
    for stroke, width in zip(strokes, widths):
        steps = np.maximum(1, np.ceil(np.linalg.norm(np.diff(stroke, axis=0), axis=1) / .6)).astype(int)
        dense = np.vstack([np.linspace(a, b, n, endpoint=False) for a, b, n in zip(stroke[:-1], stroke[1:], steps)] + [stroke[-1:]])
        radii = np.interp(lengths(dense), lengths(stroke), np.asarray(width, float) / 2)
        for (x, y), r in zip(dense, radii):
            left, right = int(max(0, x - r - 2)), int(min(alpha.shape[1], x + r + 3))
            top, bottom = int(max(0, y - r - 2)), int(min(alpha.shape[0], y + r + 3))
            if right <= left or bottom <= top:
                continue
            ys, xs = np.ogrid[top:bottom, left:right]
            np.maximum(alpha[top:bottom, left:right], np.clip(r + .5 - np.hypot(xs - x, ys - y), 0, 1), out=alpha[top:bottom, left:right])
    return (alpha * 255).round().astype(np.uint8)


def build_line(name, points, width=5, color='#302C26', head='none', curve=0., pressure=PRESSURE):
    """A pen line or arrow drawn stroke by stroke: the shaft from its first point, then each arrowhead barb.
    points are in the caller's coordinates; the asset is the bounding box, and the offset to place it is returned.
    pressure (0..1) is how much the nib width follows the hand (pressure_widths); 0 draws an even line."""
    if head not in ('none', 'end', 'both'):
        raise ValueError('Line head must be none, end or both')
    path = line_points(points, curve)
    if len(path) < 2 or np.linalg.norm(path[-1] - path[0]) + calculate_length(path) < 4:
        raise ValueError('A line needs at least two distinct points')
    strokes = [path]
    barb = max(14., 3.6 * width)

    def barbs(tip, before):
        direction = tip - before
        direction /= max(1e-6, np.linalg.norm(direction))
        result = []
        for angle in (math.radians(28), -math.radians(28)):
            turn = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
            result.append(np.array([tip - barb * (turn @ direction), tip]))
        return result
    if head in ('end', 'both'):
        strokes += barbs(path[-1], path[max(0, len(path) - 6)])
    if head == 'both':
        strokes += barbs(path[0], path[min(len(path) - 1, 5)])
    every = np.vstack(strokes)
    margin = width + 4
    origin = np.floor(every.min(0)) - margin
    size = (np.ceil(every.max(0)) - origin + margin).astype(int)
    local = [stroke - origin for stroke in strokes]
    pressure = float(np.clip(pressure, 0, 1))
    alpha = ink_mask(local, [pressure_widths(stroke, width, pressure, flick=index > 0) for index, stroke in enumerate(local)], size)
    paths = [line_topology.resample(stroke) for stroke in local]
    reveal, segments = time_paths(paths, alpha.shape)
    reveal[alpha == 0] = np.inf
    for segment in segments:
        if segment['kind'] == 'ink':
            segment['radius'] = width / 2 + 2.
    rgba = np.zeros(alpha.shape + (4,), np.uint8)
    rgba[:, :, :3] = tuple(int(color[index:index + 2], 16) for index in (1, 3, 5))
    rgba[:, :, 3] = alpha
    metadata = save_asset(name, rgba, reveal, segments, {'type': 'line', 'head': head})
    metadata['offset'] = origin.tolist()
    return metadata


def calculate_length(path):
    return float(lengths(path)[-1])


def build_counter(name, start, end, font_path, size, color='#302C26', decimals=0, prefix='', suffix='', font_index=0):
    """A number that counts up (or down) once drawn into place: typeset frames are produced by the renderer;
    the asset is the final value in a box wide enough for every intermediate value."""
    font = ImageFont.truetype(str(font_path), int(size), index=int(font_index))
    text = lambda value: f'{prefix}{value:,.{int(decimals)}f}{suffix}'
    samples = [text(value) for value in np.linspace(float(start), float(end), 12)]
    width = int(max(font.getlength(sample) for sample in samples)) + 12
    box = font.getbbox(text(float(end)))
    height = int(box[3] - min(0, box[1]) + 12)
    image = Image.new('RGBA', (width, height))
    ImageDraw.Draw(image).text((6, 6 - min(0, box[1])), text(float(end)), font=font, fill=color)
    np.savez_compressed(ASSET_DIRECTORY / f'{name}.npz', reveal=np.zeros((height, width), np.float32))
    image.save(ASSET_DIRECTORY / f'{name}.png')
    metadata = {'name': name, 'type': 'counter', 'width': width, 'height': height, 'segments': [],
                'counter': {'from': float(start), 'to': float(end), 'decimals': int(decimals), 'prefix': prefix, 'suffix': suffix,
                            'font': {'path': str(font_path), 'index': int(font_index)}, 'size': int(size), 'color': color, 'origin': [6, 6 - min(0, box[1])]}}
    (ASSET_DIRECTORY / f'{name}.json').write_text(json.dumps(metadata, ensure_ascii=False))
    return metadata


def paper_color(rgb):
    """Median colour of the image border when it is light, near-neutral paper (cream, warm white); else white."""
    ring = max(2, round(.02 * min(rgb.shape[:2])))
    border = np.concatenate([rgb[:ring].reshape(-1, 3), rgb[-ring:].reshape(-1, 3), rgb[:, :ring].reshape(-1, 3), rgb[:, -ring:].reshape(-1, 3)])
    paper = np.median(border, axis=0)
    return paper if paper.min() >= 200 and np.ptp(paper) <= 30 else np.full(3, 255.)


LINE_BAND = 5.2    # px: dark pixels this close to the edge of a dark area may be its outline
FILL_DEPTH = 6.    # px: dark pixels this deep inside a dark area are paint, not line
DARKER_LINE = 18.  # luminance: an outline around a dark fill must be at least this much darker than the fill
BLACK_FILL = 60.   # luminance: a fill this dark is a black shape; its thin rim is drawn as its outline
INK_TONE_MARGIN = 45.  # luminance above the picture's typical ink that a line pixel may still have
WEAK_CONTRAST = 110.  # RGB distance below which ink and the paint beneath it cannot be unmixed reliably


def ink_lines(pixels, dark, size=1.):
    """Pixels a pen inks: thin dark marks, and the rim of a dark area only where a line darker than the area
    runs along it. The edge band of a navy hoodie, a night sky around the moon or a teal screen is paint;
    drawing it as line made the line pass grainy and showed the fill twice."""
    # Line widths grow with the picture (size = line_drawing.picture_size): the thresholds follow them.
    depth = ndimage.distance_transform_edt(dark)
    band = dark & (depth < LINE_BAND * size)
    deep = dark & (depth >= FILL_DEPTH * size)
    if not deep.any():
        return band
    luminance = pixels[:, :, :3].astype(float) @ [.299, .587, .114]
    # Reference tone of the fill each band pixel borders: the nearest deep interior, median-smoothed against grain.
    distance, nearest = ndimage.distance_transform_edt(~deep, return_indices=True)
    fill_tone = ndimage.median_filter(luminance[nearest[0], nearest[1]], size=5)
    bordering = distance <= (FILL_DEPTH + 1.5) * size
    darker = luminance < fill_tone - DARKER_LINE
    rim = (depth <= 2.2 * size) & (fill_tone < BLACK_FILL)
    lines = band & (~bordering | darker | rim)
    # A small colored area (a teal screen between a robot's eyes) has no deep interior to compare with:
    # line pixels must also be about as dark as the picture's own ink, measured on its unmistakable thin lines.
    core = band & ~bordering & (depth >= 1.5 * size)
    if core.sum() >= 40:
        lines &= luminance <= np.percentile(luminance[core], 30) + INK_TONE_MARGIN
    # Crumbs of a fill edge that passed the tests by grain are not marks: drop pieces under 20 px cut from a larger area.
    pieces, count = ndimage.label(lines, structure=np.ones((3, 3)))
    if count:
        areas, _ = ndimage.label(band, structure=np.ones((3, 3)))
        piece_size, area_size = np.bincount(pieces.ravel()), np.bincount(areas.ravel())
        crumbs = (piece_size[pieces] < 20) & (area_size[areas] >= 60) & lines
        lines &= ~crumbs
    return lines


def prepare_artwork(source_path, max_width, max_height, crop=None, outline_mask_path=None):
    """Crop, remove the paper background, fit the box, and find the drawable outline (shared with region marks)."""
    artwork=Image.open(source_path).convert('RGBA')
    source_size = artwork.size
    # Generated art often sits on cream or warm-white paper: key against the paper measured on the whole source,
    # not pure white, or the paper itself becomes a region the marker has to colour.
    paper = paper_color(np.array(artwork)[:, :, :3].astype(float)) if artwork.getextrema()[3] == (255, 255) else None
    if crop is not None:
        left, top, right, bottom = crop
        if not (0 <= left < right <= artwork.width and 0 <= top < bottom <= artwork.height):
            raise ValueError('Artwork crop must stay inside the source image')
        artwork = artwork.crop(tuple(crop))
    if paper is not None:
        rgb = np.array(artwork)[:, :, :3]
        alpha = np.rint(np.clip(((paper - 2) - rgb).max(axis=2) / 18, 0, 1) * 255).astype(np.uint8)
        artwork = Image.fromarray(np.dstack([rgb, alpha]))
    scale = min(max_width/artwork.width, max_height/artwork.height)
    artwork = artwork.resize((max(1,round(artwork.width*scale)),max(1,round(artwork.height*scale))),Image.Resampling.LANCZOS)
    pixels=np.array(artwork)
    maximum=pixels[:,:,:3].max(axis=2)
    alpha=pixels[:,:,3]
    # Faint isolated paper grain left by the white key is not drawable content.
    labels, count = ndimage.label(alpha > 0, structure=np.ones((3, 3)))
    noise = (np.bincount(labels.ravel()) < 20) & (np.r_[255, ndimage.maximum(alpha, labels, np.arange(1, count + 1))] < 64)
    alpha[noise[labels]] = 0
    red,green,blue=[pixels[:,:,channel].astype(float) for channel in range(3)]
    cool_ink=(blue>=red+4)&(blue>=green-4)&(red<218)
    dark=((maximum<125)|cool_ink)&(alpha>30)
    size=line_drawing.picture_size(pixels.shape[1],pixels.shape[0])
    band=dark&(ndimage.distance_transform_edt(dark)<LINE_BAND*size)
    outline=ink_lines(pixels,dark,size)
    support=(maximum<165)&(alpha>30)
    if outline_mask_path is not None:
        explicit = Image.open(outline_mask_path).convert('L')
        if explicit.size != source_size:
            raise ValueError('Outline mask must match source image dimensions before crop')
        if crop is not None:
            explicit = explicit.crop(tuple(crop))
        # Shrink by coverage, not by sampling: a nearest-neighbour shrink breaks a thin line (a hair, a glasses rim) into
        # dots the pen would have to dab one by one. A plan pixel a quarter covered by the mask's line is line. Enlarging
        # (a close-up render) interpolates and cuts at half, so a line edge stays smooth instead of a staircase.
        shrinking = explicit.width > artwork.width
        explicit = explicit.resize(artwork.size, Image.Resampling.BOX if shrinking else Image.Resampling.BILINEAR)
        outline = (np.array(explicit) > (63 if shrinking else 127)) & (alpha > 0)
        band = outline
    if not (alpha>0).any():
        raise ValueError('Illustration is empty after background removal')
    return {'artwork':artwork,'pixels':pixels,'alpha':alpha,'outline':outline,'band':band,'support':support}


def box_polygon(box):
    """A part given as a normalized box [left, top, right, bottom]: enough with 2.0's snapping, which hands every inked area
    to the part that owns most of it, so a box need not trace the shape."""
    if not (isinstance(box, (list, tuple)) and len(box) == 4 and all(isinstance(value, (int, float)) for value in box)
            and 0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
        raise ValueError('A part needs "polygon" or "box": [left, top, right, bottom] normalized to 0..1')
    left, top, right, bottom = box
    return [[left, top], [right, top], [right, bottom], [left, bottom]]


def polygon_owners(parts, artwork, alpha):
    owners = np.full(alpha.shape, -1, dtype=np.int16)
    names = []
    hsv = None
    for index, part in enumerate(parts):
        if not part.get('id') or part['id'] in names:
            raise ValueError('Every semantic part needs a unique id')
        names.append(part['id'])
        region = Image.new('L', artwork.size)
        polygon = np.asarray(part['polygon'] if 'polygon' in part else box_polygon(part.get('box')), dtype=float)
        if len(polygon)<3 or not ((polygon>=0)&(polygon<=1)).all():
            raise ValueError('Part polygons must have >=3 normalized vertices')
        ImageDraw.Draw(region).polygon([tuple(point) for point in polygon * [artwork.width-1,artwork.height-1]],fill=255)
        mask = (np.array(region)>0)&(alpha>0)&(owners<0)
        # Preserve held objects and other occluding shapes for their own semantic part.
        for excluded_polygon in part.get('exclude_polygons', []) + [box_polygon(box) for box in part.get('exclude_boxes', [])]:
            excluded_polygon = np.asarray(excluded_polygon, dtype=float)
            if len(excluded_polygon)<3 or not ((excluded_polygon>=0)&(excluded_polygon<=1)).all():
                raise ValueError('Excluded polygons must have >=3 normalized vertices')
            excluded_region = Image.new('L', artwork.size)
            ImageDraw.Draw(excluded_region).polygon([tuple(point) for point in excluded_polygon * [artwork.width-1,artwork.height-1]],fill=255)
            mask &= np.array(excluded_region)==0
        # A background prop visible between curls belongs to the prop, not the head polygon.
        for exclusion in part.get('exclude_hsv', []):
            if len(exclusion)!=4 or not (0<=exclusion[0]<=exclusion[1]<=360 and 0<=exclusion[2]<=1 and 0<=exclusion[3]<=1):
                raise ValueError('exclude_hsv requires [hue_min, hue_max, saturation_min, value_min]')
            hsv=np.asarray(artwork.convert('RGB').convert('HSV'),dtype=float) if hsv is None else hsv
            hue=hsv[:,:,0]*360/255
            mask &= ~((hue>=exclusion[0])&(hue<=exclusion[1])&(hsv[:,:,1]>=exclusion[2]*255)&(hsv[:,:,2]>=exclusion[3]*255))
        if not mask.any():
            raise ValueError(f'Empty semantic part: {part["id"]}')
        owners[mask] = index
    # A model's polygon cannot trace the edge and anti-alias halo of a line it cuts through: faint pixels, and any
    # within a few px of a part, join the nearest part. Solid content farther from every part still has to be declared.
    stray = (alpha>0)&(owners<0)
    if stray.any() and (owners>=0).any():
        distance, nearest = ndimage.distance_transform_edt(owners<0, return_indices=True)
        edge = stray&((distance<=4)|(alpha<64))
        owners[edge] = owners[nearest[0], nearest[1]][edge]
    if ((alpha>0)&(owners<0)).any():
        raise ValueError('Semantic parts do not cover every visible pixel; inspect and repair masks')
    return names, owners


def region_owners(region_map_path, part_regions, alpha, source_path):
    """Set-of-Mark ownership: the model maps numbered regions (region_marks.py) to ordered semantic parts."""
    if not region_map_path:
        raise ValueError('part_regions requires the region_map written by region_marks.py')
    data = np.load(region_map_path)
    if 'source_sha256' in data.files and str(data['source_sha256']) != hashlib.sha256(Path(source_path).read_bytes()).hexdigest():
        raise ValueError('Region map belongs to another image; rerun region_marks.py and relabel')
    labels = data['labels']
    if labels.shape != alpha.shape:
        labels = np.array(Image.fromarray(labels.astype(np.int32)).resize((alpha.shape[1], alpha.shape[0]), Image.Resampling.NEAREST))
    present = set(np.unique(labels[alpha > 0]).tolist()) - {0}
    names, owners, seen = [], np.full(alpha.shape, -1, np.int16), {}
    for index, part in enumerate(part_regions):
        if not part.get('id') or part['id'] in names:
            raise ValueError('Every semantic part needs a unique id')
        names.append(part['id'])
        numbers = [int(number) for number in part.get('regions', [])]
        if not numbers:
            raise ValueError(f'Empty semantic part: {part["id"]}')
        for number in numbers:
            if number in seen:
                raise ValueError(f'Region {number} is assigned to both {seen[number]} and {part["id"]}')
            if number not in present:
                raise ValueError(f'Region {number} of {part["id"]} is not a visible region of this region map')
            seen[number] = part['id']
        owners[np.isin(labels, numbers)] = index
    missing = sorted(present - set(seen))
    if missing:
        raise ValueError('Visible regions are not assigned to any part: ' + ', '.join(map(str, missing)) + '; add them to part_regions')
    stray = (alpha > 0) & (owners < 0)
    if stray.any():
        _, nearest = ndimage.distance_transform_edt(owners < 0, return_indices=True)
        owners[stray] = owners[nearest[0], nearest[1]][stray]
    return names, owners, labels


def source_ink_layer(pixels, paint, clean, zone):
    """Unmix source ink from the paint beneath, preserving the original antialiased edge."""
    _, core = ndimage.distance_transform_edt(~clean, return_indices=True)
    rgb = pixels[:, :, :3].astype(np.float32)
    ink_rgb, under, painted = rgb[core[0], core[1]], paint[:, :, :3].astype(np.float32), paint[:, :, 3] > 0
    toward = ink_rgb - under
    contrast = (toward ** 2).sum(2)
    share = np.clip(((rgb - under) * toward).sum(2) / np.maximum(contrast, 1.), 0, 1)
    # Ink over a paint of similar color (black line on navy) divides grain by a small contrast: the noise read as
    # specks of ink scattered over the fill. There the pixel is ink or paint, with a short soft step between.
    weak = contrast < WEAK_CONTRAST ** 2
    snapped = np.clip((share - .3) / .45, 0, 1)
    share = np.where(weak, snapped * snapped * (3 - 2 * snapped), share)
    alpha = drop_loose_specks(np.where(painted, share, pixels[:, :, 3] / 255.) * zone)
    return np.where(painted[..., None], ink_rgb, rgb), (np.rint(alpha * 255) / 255).astype(np.float32)


def drop_loose_specks(alpha, size=6, reach=2):
    """A few pixels of ink standing alone beside a fill edge are unmixing residue, not a mark anyone drew;
    they stay in the paint layer. Specks touching a real line (within reach px) are its ragged edge and stay."""
    labels, count = ndimage.label(alpha > .3, structure=np.ones((3, 3)))
    if not count:
        return alpha
    sizes = np.bincount(labels.ravel())
    small = sizes <= size
    small[0] = False
    lines = (labels > 0) & ~small[labels]
    near = ndimage.binary_dilation(lines, iterations=reach)
    loose = small[labels] & ~near
    if loose.any():
        alpha = alpha.copy()
        alpha[ndimage.binary_dilation(loose) & ~near] = 0
    return alpha


def source_ink(pixels, paint, clean, zone, strokes):
    """Bind each source pixel to the closest actual nib sample and that stroke's part."""
    rgb, alpha = source_ink_layer(pixels, paint, clean, zone)
    ys, xs = np.nonzero(alpha > 0)
    if not strokes:  # a tiny mark with nothing to trace: its ink shows at once, owned by the first part
        ink_time, ink_owner = np.full(alpha.shape, np.inf, np.float32), np.full(alpha.shape, -1, np.int16)
        ink_time[ys, xs], ink_owner[ys, xs] = 0., 0
        return rgb, alpha, ink_time, ink_owner
    points = np.vstack([stroke['points'] for stroke in strokes])
    times = np.concatenate([stroke['times'] for stroke in strokes])
    parts = np.concatenate([np.full(len(stroke['points']), stroke['part'], np.int16) for stroke in strokes])
    _, closest = cKDTree(points).query(np.c_[xs, ys])
    ink_time = np.full(alpha.shape, np.inf, np.float32)
    ink_owner = np.full(alpha.shape, -1, np.int16)
    ink_time[ys, xs], ink_owner[ys, xs] = times[closest], parts[closest]
    return rgb, alpha, ink_time, ink_owner


def color_pieces(outline, pixels, alpha, colors=16):
    """Pieces of one paint inside the areas the ink encloses: (piece label map, count, enclosed mask, solid flags). A piece
    is solid when it is an area rather than a seam (holds a pixel 2.5 px from its edge, or is a whole enclosed shape)."""
    enclosed = (alpha > 0) & ~ndimage.binary_dilation(outline)
    if not enclosed.any():
        return np.zeros(alpha.shape, np.int64), 0, enclosed, np.zeros(1, bool)
    palette_image = Image.fromarray(np.ascontiguousarray(pixels[:, :, :3])).quantize(colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    quantized = np.array(palette_image)
    # Faint shading splits one flat colour into near-identical palette entries: those are one paint, not separate pieces.
    swatches = np.array(palette_image.getpalette()[:3 * colors], float).reshape(-1, 3)
    same = np.arange(len(swatches))
    for first in range(len(swatches)):
        for second in range(first + 1, len(swatches)):
            if np.linalg.norm(swatches[first] - swatches[second]) < 22:
                same[same == same[second]] = same[first]
    quantized = same[quantized]
    pieces, count = np.zeros(alpha.shape, np.int64), 0
    for value in np.unique(quantized[enclosed]):
        labels, found = ndimage.label(enclosed & (quantized == value))
        pieces[labels > 0] = labels[labels > 0] + count
        count += found
    sizes = np.bincount(pieces[enclosed], minlength=count + 1)
    # A piece is an area, not a seam: it must hold a pixel 2.5 px from its edge (a thin shading line follows its area).
    edge = np.zeros(alpha.shape, bool)
    edge[:-1] |= pieces[:-1] != pieces[1:]
    edge[1:] |= pieces[1:] != pieces[:-1]
    edge[:, :-1] |= pieces[:, :-1] != pieces[:, 1:]
    edge[:, 1:] |= pieces[:, 1:] != pieces[:, :-1]
    depth = np.zeros(count + 1)
    depth[1:] = ndimage.maximum(ndimage.distance_transform_edt(~edge), pieces, np.arange(1, count + 1))
    # ... unless the ink encloses that piece alone: a small closed shape (a button, a prop between curls) is deliberate.
    cells, _ = ndimage.label(enclosed)
    cell_of = np.zeros(count + 1, np.int64)
    cell_of[pieces[enclosed]] = cells[enclosed]
    whole = sizes == np.bincount(cells[enclosed], minlength=cells.max() + 1)[cell_of]
    solid = ((sizes >= max(24, 2e-4 * enclosed.sum())) & (depth >= 2.5)) | (whole & (sizes > 0))
    solid[0] = False
    return pieces, count, enclosed, solid


def snap_owners(owners, outline, pixels, alpha, part_count, colors=16):
    """A model's part polygon is coarse (often a box) and cuts straight through a shape: coloring part by part then
    paints a rectangle of a continuous area at a time. So each piece of one color inside an area the ink encloses goes
    wholly to the part that owns most of it, and every other pixel (anti-aliasing, the paint under a line) follows the
    nearest piece. This is the paint order only: strokes are still classified by the polygon owners."""
    pieces, count, enclosed, solid = color_pieces(outline, pixels, alpha, colors)
    if not count:
        return owners
    inside = (pieces > 0) & (owners >= 0)
    tally = np.bincount(pieces[inside] * part_count + owners[inside], minlength=(count + 1) * part_count).reshape(count + 1, part_count)
    snapped = owners.copy()
    keep = solid[pieces]
    snapped[keep] = tally.argmax(1)[pieces[keep]]
    rest = (alpha > 0) & ~keep
    if rest.any() and keep.any():
        _, nearest = ndimage.distance_transform_edt(~keep, return_indices=True)
        snapped[rest] = snapped[nearest[0], nearest[1]][rest]
    return snapped


def build_reveal(name, source_path, artwork, pixels, alpha, names, owners, seconds, speed, ownership):
    """Reveal mode for styles without ink outlines (watercolor, gouache, paper cut, pixel...): part by part the hand
    sweeps a broad soft brush and the finished picture shows only where it has passed. No line layer, no shading pass."""
    shape = alpha.shape
    _, nearest = ndimage.distance_transform_edt(owners < 0, return_indices=True)
    owners_everywhere = owners[nearest[0], nearest[1]]
    visible, rgb = alpha > 0, pixels[:, :, :3].astype(float)
    # One region per part: the brush follows the part, not the palette of a painted surface.
    regions = paint_motion.color_regions(rgb, visible, owners_everywhere, len(names), np.ones(shape, np.int32))
    for region in regions:
        region.update(tilt=.35, soft=.6, brush=2.5, edges=False, streaks=False)
    ys, xs = np.nonzero(visible)
    tip = np.array([xs.mean(), ys.min()], dtype=float)
    size = line_drawing.picture_size(*shape[::-1])
    speed = speed * size
    fill = paint_motion.requirements(regions, rgb, len(names), tip, speed, FPS, shade=False, size=size)
    paced = max(fill['minimum'], fill['required'] / line_drawing.VIDEO_LAPSE)
    total = min(float(seconds), paced) if seconds else float(np.clip(paced, 1, 30))
    segments, fill_parts, _, _, _, counts = paint_motion.schedule(regions, rgb, len(names), tip, total, 0., total, speed, FPS, shade=False, size=size)
    fields, _ = paint_motion.render(shape, regions, [], owners_everywhere)
    fields['frame'] = np.float32(1. / (FPS * total))
    full = fields['full']
    ending = float(max([segment['end'] for segment in segments] + [full[np.isfinite(full)].max(initial=0.)]))
    if ending > 1:
        fields = scale_times(fields, ending)
        for segment in segments:
            segment['start'], segment['end'] = segment['start'] / ending, segment['end'] / ending
        fill_parts = [[value / ending for value in part] for part in fill_parts]
    never = np.full(shape, np.inf, np.float32)
    fields['shade'] = never   # no shading pass: the brush lays the finished colour
    front = paint_motion.front(fields)
    maps = {'ink': never, 'ink_owner': np.full(shape, -1, np.int16), 'color': front['color'],
            'color_cover': front['color_cover'], 'color_full': front['color_full'], 'shade': never, 'shade_ramp': np.float32(1e-3),
            'paint_ramps': paint_motion.ramps(total, FPS) / max(1., ending), **paint_motion.stored(fields)}
    empty = np.zeros(shape + (4,), np.uint8)
    final = illustration_layers.prepare(maps, empty, pixels, pixels)['final']
    owners_final = np.where(final[:, :, 3] > 0, owners_everywhere, -1).astype(np.int16)
    for suffix, layer in (('line', empty), ('paint', pixels), ('shade', pixels)):
        Image.fromarray(layer).save(ASSET_DIRECTORY/f'{name}-{suffix}.png')
    Image.fromarray(final).save(ASSET_DIRECTORY/f'{name}.png')
    np.savez_compressed(ASSET_DIRECTORY/f'{name}.npz', owners=owners_final, **maps)
    part_reports = [{'id': part_id, 'index': index, 'outline_start': 0., 'outline_end': 0.,
                     'fill_start': fill_parts[index][0], 'fill_end': fill_parts[index][1]} for index, part_id in enumerate(names)]
    segments.sort(key=lambda segment: segment['start'])
    drawability = {'line_length_px': 0, 'stroke_count': 0, 'drawn_stroke_count': 0, 'fill_area_px': int(visible.sum()),
                   'required_seconds': round(float(fill['required']), 2), 'minimum_seconds': round(float(fill['minimum']), 2), 'lineart_gate': None}
    metadata = {'name': name, 'type': 'illustration', 'engine': 'draw-v3', 'mode': 'reveal', 'source': str(source_path), 'width': artwork.width,
                'height': artwork.height, 'seconds': total, 'segments': segments, 'parts': part_reports, 'layers': ['paint'],
                'lines': {}, 'painting': counts, 'drawability': drawability, 'ownership': ownership, 'lineart_exception': None,
                'time_compression': round(max(1., ending), 3),
                'idle_seconds': round(max(0., total - max([segment['end'] for segment in segments] + [0.]) * total), 2)}
    (ASSET_DIRECTORY/f'{name}.json').write_text(json.dumps(metadata))
    return metadata


def scale_times(fields, ending):
    """Paint fields of an asset compressed into its window: every time field divided by `ending`."""
    return {key: (np.asarray(value) / ending).astype(np.float32) if key in paint_motion.TIME_FIELDS else value
            for key, value in fields.items()}


TEXTURE = {'ink': ('pressure', 'none'), 'marker': ('flat', 'layered')}   # the choices of each key
TEXTURE_ON = {'ink': 'pressure', 'marker': 'flat'}    # default (true): pressure ink; the finished colour is the source's
TEXTURE_OFF = {'ink': 'none', 'marker': 'flat'}       # false: the even technical pen of 2.x
# A layered marker multiplies nib-wide lanes into every finished colour (overlaps ~9% darker). On a film's flat fills they
# read as diagonal stripes over the whole picture, so it is opt-in only: {"marker": "layered"}.


def texture_option(drawing):
    """drawing.texture: real-media texture of a drawn picture. {"ink": "pressure" | "none", "marker": "flat" | "layered"};
    true (default) is pressure ink and flat marker colour (exactly the source's), false (or "none") the even technical
    pen of 2.x. "layered" leaves visible marker lanes in the finished colour and is never on by default."""
    value = (drawing or {}).get('texture', True)
    if value is True or value is None:
        return dict(TEXTURE_ON)
    if value is False or value in ('none', 'flat'):
        return dict(TEXTURE_OFF)
    if not isinstance(value, dict) or set(value) - set(TEXTURE):
        raise ValueError('drawing.texture must be true, false or {"ink": "pressure"|"none", "marker": "layered"|"flat"}')
    result = {key: value.get(key, TEXTURE_ON[key]) for key in TEXTURE}
    for key, choices in TEXTURE.items():
        if result[key] not in choices:
            raise ValueError(f'drawing.texture.{key} must be ' + ' or '.join(f'"{choice}"' for choice in choices))
    return result


def to_render(points, scale):
    """Plan coordinates (a pixel's centre is its index) at a render `scale` (x, y) times larger, pixel centres aligned:
    plan pixel p covers render pixels p*s .. p*s + s - 1, whose centre is p*s + (s - 1) / 2. Scaling by s alone put every
    nib path of a close-up (s - 1) / 2 px up and left of its picture (1.1 px at 3.2x), and colour on the far side of a
    stroke then read as being outside the brush (3.0 promo, the park picture at plan 400)."""
    scale = np.asarray(scale, float)
    return np.asarray(points, float) * scale + (scale - 1) / 2


def ink_paths(segments, scale=(1., 1.)):
    """Centrelines of the inked strokes of an asset (in drawing order), scaled to a render size."""
    return [to_render(segment['points'], scale) for segment in segments if segment['kind'] == 'ink' and len(segment['points']) > 1]


CONSTRUCTION_SHARE = .18    # planned construction takes at most this share of a picture's drawing time
CONSTRUCTION_SECONDS = 2.5  # ... and at most this long


def build_illustration(name, source_path, max_width, max_height, crop=None, parts=None, outline_mask_path=None, seconds=None,
                       region_map_path=None, part_regions=None, lineart_exception=None, drawing=None):
    """Draw, don't reveal: whole strokes inked part by part, then marker color and soft shading, as layers."""
    drawing = dict(drawing or {})
    speed = float(drawing.get('speed', 1.))
    if not (math.isfinite(speed) and .25 <= speed <= 4):
        raise ValueError('drawing.speed must be within 0.25..4')
    mode = drawing.get('mode', 'draw')
    if mode not in ('draw', 'reveal'):
        raise ValueError('drawing.mode must be "draw" or "reveal"')
    texture = texture_option(drawing)
    art = prepare_artwork(source_path, max_width, max_height, crop, None if mode == 'reveal' else outline_mask_path)
    artwork, pixels, alpha, outline = art['artwork'], art['pixels'], art['alpha'], art['outline']
    shape, labels = alpha.shape, None
    if part_regions:
        names, owners, labels = region_owners(region_map_path, part_regions, alpha, source_path)
        declared = part_regions
    elif parts:
        names, owners = polygon_owners(parts, artwork, alpha)
        declared = parts
    else:
        raise ValueError('Illustrations require model-designed semantic parts: part_regions (region_marks.py) or ordered normalized polygons')
    if mode == 'reveal':
        return build_reveal(name, source_path, artwork, pixels, alpha, names, owners, seconds, speed, 'regions' if part_regions else 'polygons')

    for index, part_id in enumerate(names):
        if not (outline & (owners == index)).any():
            # A part drawn only as a dark fill with no darker line around it: its own rim is the outline a person draws.
            rim = art['band'] & (owners == index) & (ndimage.distance_transform_edt(art['band'] & (owners == index)) <= 2.2)
            outline = outline | rim
        if not (outline & (owners == index)).any():
            raise ValueError('Part has no drawable outline; prepare a contour layer/mask before generation: '+part_id)
    size = line_drawing.picture_size(shape[1], shape[0])
    strokes, clean, skeleton, topology = line_topology.extract_strokes(outline, art['support'], size)
    gate = lineart_gate.evaluate(outline, strokes, clean, skeleton)
    if not gate['passed'] and not lineart_exception:
        raise ValueError(lineart_gate.failure_message(gate))
    # Paint follows the drawing's own areas; a model's coarse part polygons only classify the strokes.
    paint_owners = owners if part_regions else snap_owners(owners, outline, pixels, alpha, len(names))
    _, nearest = ndimage.distance_transform_edt(paint_owners < 0, return_indices=True)
    owners_everywhere = paint_owners[nearest[0], nearest[1]]
    # The soft grey edge of a line (anti-aliasing, compression ringing) belongs to the line, not to marker paint:
    # neutral pixels next to a line are refilled from the nearest real paint (paper on line-only art).
    neutral = np.ptp(pixels[:, :, :3].astype(int), axis=2) <= 24
    line_zone = ndimage.binary_dilation(clean, iterations=2) | (ndimage.binary_dilation(clean, iterations=5) & neutral)
    shade_pixels = paint_motion.inpaint_lines(pixels, line_zone)
    shade_pixels[:, :, 3] = np.where(shade_pixels[:, :, 3] >= 96, shade_pixels[:, :, 3], 0)
    _, source_line_alpha = source_ink_layer(pixels, shade_pixels, clean, line_zone)
    topology['edge_visits'] = line_topology.cover_ink_edges(strokes, source_line_alpha > 0, support=line_zone)
    line_drawing.classify(strokes, owners, alpha, pixels, names, {part['id'] for part in declared if part.get('key')}, clean)
    painted = shade_pixels[:, :, 3] > 0
    shade_rgb = shade_pixels[:, :, :3].astype(float)
    colouring = drawing.get('colouring', 'markers')
    if colouring not in ('parts', 'markers'):
        raise ValueError('drawing.colouring must be "parts" or "markers"')
    if colouring == 'markers':
        # Colour like a person with a set of markers: the areas the lines close, one marker per colour, light to dark,
        # each colour's shadow side laid over its base coat with a darker marker.
        # A marker colours an area the lines close as one piece, whichever modules it touches: split by module, one area
        # became several separate passes and the pen spent its time travelling between them.
        regions = paint_motion.color_regions(shade_rgb, painted, np.zeros_like(owners_everywhere), 1, line_drawing.areas(clean), size)
        for region in regions:
            (left, top), (height, width) = region['offset'], region['mask'].shape
            votes = owners_everywhere[top:top + height, left:left + width][region['mask']]
            region['part'] = int(np.bincount(votes[votes >= 0], minlength=len(names)).argmax()) if (votes >= 0).any() else 0
        paint_motion.by_marker(regions, shade_rgb, size)
    else:
        regions = paint_motion.color_regions(shade_rgb, painted, owners_everywhere, len(names), labels, size)
    ys, xs = np.nonzero(alpha > 0)
    tip = np.array([xs.mean(), ys.min()], dtype=float)
    # Timing follows the picture, not the screen: a larger picture is drawn with proportionally longer, faster moves.
    size = line_drawing.picture_size(shape[1], shape[0])
    speed = speed * size
    lines = line_drawing.requirements(strokes, len(names), tip, speed, FPS, size)
    fill = paint_motion.requirements(regions, shade_rgb, len(names), tip, speed, FPS, size=size) if regions else {'required': 0., 'moderate': 0., 'minimum': 0.}
    if colouring == 'markers':
        ramp = paint_motion.MARKER_SHADE_RAMP if any(region['shadows'] for region in regions) else 0.
    else:
        shaded = bool(regions) and bool(paint_motion.shaded_parts(regions, shade_rgb, len(names)))
        ramp = paint_motion.SHADE_RAMP if shaded else 0.
    paced = max(lines['all_at_cap'] + fill['minimum'] + ramp, (lines['required'] + fill['required'] + ramp) / line_drawing.VIDEO_LAPSE)
    paced /= 1. - 1. / illustration_layers.SHARP
    # drawing.sketch: true lays in each part's simplified silhouette; a list is the model's planned construction (起形),
    # simple shapes that block in the whole picture before any ink, given more of the time so each shape reads.
    # Construction (v3): the model's plan when given, else laid in from the modules themselves; a picture of only a few
    # strokes (an icon, a label) is drawn straight. false turns it off; true is the older faded silhouette sketch.
    sketch_option = drawing.get('sketch', 'auto')
    if isinstance(sketch_option, list):
        line_drawing.check_construction(sketch_option, names)
        planned = sketch_option
    elif sketch_option == 'auto' and len(strokes) >= line_drawing.AUTO_CONSTRUCTION_STROKES:
        planned = line_drawing.auto_construction(owners, alpha, names) or None
    elif sketch_option in (True, False, 'auto'):
        planned = None
    else:
        raise ValueError('drawing.sketch must be true, false, "auto" or a list of construction shapes')
    drawing['sketch'] = planned or (sketch_option is True)
    share, ceiling_seconds = (CONSTRUCTION_SHARE, CONSTRUCTION_SECONDS) if planned else (.12, 1.5)
    paced += min(ceiling_seconds, share / (1 - share) * paced) if drawing.get('sketch') else 0.
    # A planned construction is rubbed out with an eraser after the inking, before any colour (markers smear graphite).
    paced += float(np.clip(line_drawing.ERASE_SHARE * paced, line_drawing.ERASE_MIN, line_drawing.ERASE_MAX)) if planned else 0.
    total = min(float(seconds), paced) if seconds else float(np.clip(paced, 1, 30))
    total = max(total, 1. / FPS)  # a picture with nothing to trace (a tiny mark) still occupies one frame, never zero time
    sketches = (line_drawing.construction_strokes(planned, shape[1], shape[0], names) if planned
                else line_drawing.sketch_strokes(owners, alpha, len(names)) if drawing.get('sketch') else [])
    top_tip = np.asarray(tip, float)
    ceiling = float(seconds) if seconds else 30.
    # Lay the picture out again when the schedule overruns a window that still has room: the fill minimum
    # was first measured from the picture top, but the marker actually starts where the lines end, so give
    # the picture its measured minimum instead of compressing it. A window truly too small still
    # compresses the whole asset uniformly below.
    for attempt in range(3):
        tip = top_tip.copy()
        segments, cursor = [], 0.
        if sketches:
            if planned:
                # A blocked-in shape is drawn deliberately (about video pace), never flicked in a frame.
                each = [max(4. / FPS, min(.6, stroke['weights'][-1] / (line_drawing.LINE_SPEED * speed * line_drawing.VIDEO_LAPSE))) for stroke in sketches]
            else:
                each = [max(2. / FPS, min(.4, stroke['weights'][-1] / (line_drawing.line_cap(speed) * 1.6))) for stroke in sketches]
            hops = [float(np.linalg.norm(stroke['points'][0] - previous['points'][-1])) for previous, stroke in zip(sketches, sketches[1:])]
            # The moves between planned shapes count against the same budget (and cross a large picture in proportion).
            hop_size = size if planned else 1.
            moving = sum(line_drawing.lift_seconds(hop, FPS, 1., size=hop_size) for hop in hops) if planned else .1 * len(each)
            factor = min(1., min(ceiling_seconds, share * total) / (sum(each) + moving))
            for index, (stroke, length) in enumerate(zip(sketches, each)):
                length = max((3. if planned else 2.) / FPS, length * factor)
                if index:
                    lift = line_drawing.lift_seconds(hops[index - 1], FPS, factor, size=hop_size)
                    segments.append({'kind': 'travel', 'start': cursor / total, 'end': (cursor + lift) / total,
                                     'points': [sketches[index - 1]['points'][-1].tolist(), stroke['points'][0].tolist()]})
                    cursor += lift
                stroke['times'] = (cursor + length * time_fraction(stroke['weights'] / max(1e-6, stroke['weights'][-1]), 'minjerk')) / total
                segments.append({'kind': 'sketch', 'start': cursor / total, 'end': (cursor + length) / total, 'profile': 'minjerk',
                                 'points': stroke['points'].tolist(), 'lengths': stroke['weights'].tolist()})
                cursor += length
            tip = sketches[-1]['points'][-1]
        # The source ink settles locally after the nib passes. Reserve that edge ramp
        # instead of accelerating the whole picture by 0.4% at the last frame.
        erase_seconds = float(np.clip(line_drawing.ERASE_SHARE * total, line_drawing.ERASE_MIN, line_drawing.ERASE_MAX)) if planned else 0.
        sketch_end = cursor
        available = max(.001, total - cursor - erase_seconds - ramp - total / illustration_layers.SHARP)
        # Water-fill the time: silhouette + key lines, a basic fill, every line at the cap, a moderate fill, then human pace.
        line_budget, fill_budget, remaining = 0., 0., available
        steps = [(lines['minimum'], 0.), (0., fill['minimum']), (lines['all_at_cap'] - lines['minimum'], 0.),
                 (0., fill['moderate'] - fill['minimum']), (lines['required'] - lines['all_at_cap'], fill['required'] - fill['moderate'])]
        for line_step, fill_step in steps:
            line_step, fill_step = max(0., line_step), max(0., fill_step)
            share = min(1., remaining / max(1e-9, line_step + fill_step))
            line_budget, fill_budget, remaining = line_budget + line_step * share, fill_budget + fill_step * share, remaining - (line_step + fill_step) * share
        if remaining > 1e-6:
            stretch = min(1.6, available / max(1e-6, line_budget + fill_budget))
            line_budget, fill_budget = line_budget * stretch, fill_budget * stretch
        ordered, line_segments, line_parts, line_end, line_counts = line_drawing.schedule(
            strokes, len(names), tip, line_budget, cursor, total, speed, FPS, size)
        segments += line_segments
        line_rgb, line_alpha, ink, ink_owner = source_ink(pixels, shade_pixels, clean, line_zone, strokes)
        tip = ordered[-1]['points'][-1] if ordered else tip
        erase_end = line_end * total
        if planned:
            passes, eraser = line_drawing.eraser_passes(sketches, tip, float(np.hypot(*shape)))
            hops = [float(np.linalg.norm(passes[0][0] - tip))] + [float(np.linalg.norm(b[0] - a[-1])) for a, b in zip(passes, passes[1:])]
            moves = [line_drawing.lift_seconds(hop, FPS, .6, 2., size) for hop in hops]
            # Most of the time goes to rubbing: the hops between lines share at most half of it (never under a frame each).
            squeeze = min(1., .5 * erase_seconds / max(1e-6, sum(moves)))
            moves = [max(1. / FPS, move * squeeze) for move in moves]
            lengths = [line_drawing.calculate_lengths(scrub) for scrub in passes]
            rubbing = max(.2, erase_seconds - sum(moves))
            pass_times, cursor_erase, at = [], erase_end, tip
            for scrub, move, length in zip(passes, moves, lengths):
                segments.append({'kind': 'travel', 'start': cursor_erase / total, 'end': (cursor_erase + move) / total,
                                 'points': [np.asarray(at).tolist(), scrub[0].tolist()]})
                cursor_erase += move
                seconds_here = max(2. / FPS, rubbing * length[-1] / max(1e-6, sum(item[-1] for item in lengths)))
                segments.append({'kind': 'erase', 'start': cursor_erase / total, 'end': (cursor_erase + seconds_here) / total,
                                 'points': scrub.tolist(), 'lengths': length.tolist(), 'radius': eraser})
                pass_times.append((cursor_erase + seconds_here * length / max(1e-6, length[-1])) / total)
                cursor_erase += seconds_here
                at = scrub[-1]
            erase_end, tip = cursor_erase, at
        # The marker minimum must start where the lines end, not at the picture top: the storyboard gate
        # reads minimum_seconds, and the fill below begins its travel at this tip.
        fill_minimum = paint_motion.requirements(regions, shade_rgb, len(names), tip, speed, FPS, shade=False, size=size)['minimum'] if regions else 0.
        minimum_total = (lines['all_at_cap'] + fill_minimum + ramp) / (1. - 1. / illustration_layers.SHARP) + (sketch_end + erase_seconds if planned else 0.)
        if attempt < 2 and total < minimum_total - 1e-6 and total < ceiling - 1e-6:
            total = min(ceiling, minimum_total)
            continue
        fill_parts, bands, fill_counts = [[erase_end / total, erase_end / total] for _ in names], [], {}
        if regions:
            fill_segments, fill_parts, bands, _, tip, fill_counts = paint_motion.schedule(
                regions, shade_rgb, len(names), tip, fill_budget, erase_end, total, speed, FPS, size=size)
            segments += fill_segments
        fields, flat = paint_motion.render(shape, regions, bands, owners_everywhere, layered=texture['marker'] == 'layered', size=size)
        fields['frame'] = np.float32(1. / (FPS * total))
        full, shade = fields['full'], fields['shade']
        maps = {'ink': ink, 'ink_owner': ink_owner, 'shade_ramp': np.float32(max(ramp, 1e-3) / total), 'paint_ramps': paint_motion.ramps(total, FPS)}
        # Two-frame floors and lift minimums can overrun an impossible budget: compress the whole asset uniformly.
        finite = lambda values: values[np.isfinite(values)]
        ending = float(max([segment['end'] for segment in segments] + [line_end, finite(ink + 1 / illustration_layers.SHARP).max(initial=0.),
                            finite(full).max(initial=0.),
                            finite(shade).max(initial=0.) + float(maps['shade_ramp'])] + ([line_end + .5 / total] if sketches else [])
                           + ([(erase_end + line_drawing.ERASE_RAMP) / total] if planned else [])))
        if ending <= 1. or total >= ceiling - 1e-9:
            break
        total = min(ceiling, total * ending)
    paint_layer = np.dstack([np.rint(flat), shade_pixels[:, :, 3]]).astype(np.uint8)
    line_layer = np.dstack([np.rint(line_rgb), np.rint(line_alpha * 255)]).astype(np.uint8)
    sketch_layer = None
    if sketches:
        sketch_rgb, sketch_alpha, sketch_time, _ = line_drawing.render(shape, sketches)
        sketch_layer = np.dstack([np.rint(sketch_rgb), np.rint(sketch_alpha * 120)]).astype(np.uint8)
        maps.update(sketch=sketch_time, sketch_fade=np.array([line_end, .5 / total], np.float32))
        if planned:
            # Each construction pixel goes when the rubbing eraser passes it.
            maps.update(sketch_erase=line_drawing.erase_times(sketch_alpha, passes, pass_times, eraser),
                        erase_ramp=np.float32(line_drawing.ERASE_RAMP / total),
                        sketch_fade=np.array([erase_end / total, line_drawing.ERASE_RAMP / total], np.float32))
    if ending > 1:
        for key in ('ink', 'shade_ramp', 'paint_ramps', 'sketch', 'sketch_fade', 'sketch_erase', 'erase_ramp'):
            if key in maps:
                maps[key] = (maps[key] / ending).astype(np.float32)
        fields = scale_times(fields, ending)
        for segment in segments:
            segment['start'], segment['end'] = segment['start'] / ending, segment['end'] / ending
        line_parts = [[value / ending for value in part] for part in line_parts]
        fill_parts = [[value / ending for value in part] for part in fill_parts]
    # The colouring front at this size, anti-aliased; the fields stay with the asset for a close-up re-render.
    maps.update(paint_motion.front(fields), **paint_motion.stored(fields))
    if colouring == 'markers' and regions:
        # Colour goes by marker across modules: report each module's colouring as its own pixels received it.
        first, full = maps['color'], maps['color_full']
        for index in range(len(names)):
            mine = (owners_everywhere == index) & np.isfinite(first)
            if mine.any():
                fill_parts[index] = [float(first[mine].min()), float(full[mine & np.isfinite(full)].max(initial=first[mine].min()))]
    # Real media: a pen that lands and lifts (pressure taper, a little width and ink density), markers whose coats layer.
    shade_layer = paint_motion.apply_tone(shade_pixels, fields['tone']) if 'tone' in fields else shade_pixels
    if texture['ink'] == 'pressure':
        line_alpha = line_drawing.ink_texture(line_alpha, ink_paths(sorted(segments, key=lambda segment: segment['start'])), 1., shape)
        line_layer = np.dstack([np.rint(line_rgb), np.rint(line_alpha * 255)]).astype(np.uint8)
    final = illustration_layers.prepare(maps, line_layer, paint_layer, shade_layer, sketch_layer)['final']
    owners_final = np.where(final[:, :, 3] > 0, owners_everywhere, -1).astype(np.int16)
    for suffix, layer in (('line', line_layer), ('paint', paint_layer), ('shade', shade_layer), ('sketch', sketch_layer)):
        if layer is not None:
            Image.fromarray(layer).save(ASSET_DIRECTORY/f'{name}-{suffix}.png')
    Image.fromarray(final).save(ASSET_DIRECTORY/f'{name}.png')
    np.savez_compressed(ASSET_DIRECTORY/f'{name}.npz', owners=owners_final, **maps)
    part_reports = [{'id': part_id, 'index': index, 'outline_start': line_parts[index][0], 'outline_end': line_parts[index][1],
                     'fill_start': fill_parts[index][0], 'fill_end': fill_parts[index][1]} for index, part_id in enumerate(names)]
    segments.sort(key=lambda segment: segment['start'])
    drawability = {'line_length_px': round(sum(stroke['length'] for stroke in strokes)), 'stroke_count': len(strokes),
                   'drawn_stroke_count': line_counts['drawn_stroke_count'],
                   'fill_area_px': int(painted.sum()), 'required_seconds': round(lines['required'] + fill['required'] + ramp, 2),
                   'minimum_seconds': round(lines['all_at_cap'] + fill_minimum + ramp + (sketch_end + erase_seconds if planned else 0.), 2), 'lineart_gate': gate}
    metadata = {'name': name, 'type': 'illustration', 'engine': 'draw-v3', 'source': str(source_path), 'width': artwork.width,
                'height': artwork.height, 'seconds': total, 'segments': segments, 'parts': part_reports,
                'layers': ['paint', 'shade'] + (['sketch'] if sketches else []) + ['line'],
                'lines': {**topology, **line_counts}, 'painting': fill_counts, 'drawability': drawability,
                'ownership': 'regions' if part_regions else 'polygons', 'lineart_exception': lineart_exception, 'time_compression': round(max(1., ending), 3),
                'picture_scale': round(size, 3), 'construction': planned, 'colouring': colouring, 'texture': texture,
                'shading_bands': [{'part': int(band['part']), 'center': np.asarray(band['center'], float).tolist(),
                                   'axis': np.asarray(band['axis'], float).tolist(), 'across': np.asarray(band['across'], float).tolist(),
                                   'edges': np.asarray(band['edges'], float).tolist(), 'along': [float(value) for value in band['along']],
                                   'times': [[float(start) / max(1., ending), float(end) / max(1., ending)] for start, end in band['times']]}
                                  for band in bands],
                'idle_seconds': round(max(0., total - max([segment['end'] for segment in segments] + [0.]) * total), 2)}
    (ASSET_DIRECTORY/f'{name}.json').write_text(json.dumps(metadata))
    return metadata


def upsample(values, shape):
    """Nearest-neighbour resample of a per-pixel map to a larger grid (each fine pixel takes its coarse pixel)."""
    values = np.asarray(values)
    rows = np.minimum((np.arange(shape[0]) * values.shape[0] / shape[0]).astype(int), values.shape[0] - 1)
    columns = np.minimum((np.arange(shape[1]) * values.shape[1] / shape[1]).astype(int), values.shape[1] - 1)
    return values[rows][:, columns]


GUIDE_STEP = 300.   # squared-RGB cost of taking a neighbouring coarse pixel instead of the nearest one


def guided_index(coarse, fine):
    """For every fine pixel, the coarse pixel among the 3 x 3 around its nearest one whose colour and coverage match it
    best. Carrying the plan's maps through this index puts their boundaries on the fine picture's own edges: plain
    nearest-neighbour carries a 3 px staircase into every colour edge of a 3x close-up render (the v3 film's rough
    edges). coarse, fine: RGBA arrays of the same picture at plan and render size. Returns (rows, columns)."""
    h, w = coarse.shape[:2]
    shape = fine.shape[:2]
    rows = np.minimum((np.arange(shape[0]) * h / shape[0]).astype(int), h - 1)[:, None] + np.zeros(shape, int)
    columns = np.minimum((np.arange(shape[1]) * w / shape[1]).astype(int), w - 1)[None, :] + np.zeros(shape, int)
    coarse, fine = coarse.astype(np.float32), fine.astype(np.float32)
    covered = fine[..., 3] > 0
    best_rows, best_columns, best = rows, columns, None
    for dy, dx in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
        r, c = np.clip(rows + dy, 0, h - 1), np.clip(columns + dx, 0, w - 1)
        candidate = coarse[r, c]
        cost = np.where(covered, ((candidate[..., :3] - fine[..., :3]) ** 2).sum(2), 0.) \
            + ((candidate[..., 3] > 0) != covered) * 1e5 + (dy * dy + dx * dx) * GUIDE_STEP
        if best is None:
            best = cost
            continue
        better = cost < best
        best = np.where(better, cost, best)
        best_rows, best_columns = np.where(better, r, best_rows), np.where(better, c, best_columns)
    return best_rows, best_columns


def render_at(name, source_path, factor, crop=None, outline_mask_path=None):
    """Re-render a built illustration's pixels at `factor` x its planned size, keeping the plan: strokes, timing and
    parts stay those of the layout size (so a camera close-up does not add detail strokes or drawing time), while
    the line, paint and shade layers come from the source at the larger size and every time map is carried over by
    nearest-neighbour upsampling. Used for pages whose camera pushes in."""
    if factor <= 1.001:
        return json.loads((ASSET_DIRECTORY / f'{name}.json').read_text())
    metadata = json.loads((ASSET_DIRECTORY / f'{name}.json').read_text())
    maps = dict(np.load(ASSET_DIRECTORY / f'{name}.npz'))
    width, height = round(metadata['width'] * factor), round(metadata['height'] * factor)
    art = prepare_artwork(source_path, width, height, crop, outline_mask_path)
    pixels, alpha = art['pixels'], art['alpha']
    shape = alpha.shape
    # The plan's paint fields (anti-aliased colouring front, marker tone) are redrawn at this size, then dropped.
    fields = paint_motion.loaded(maps)
    maps = {key: value for key, value in maps.items() if not key.startswith('paint_') or key == 'paint_ramps'}
    texture = metadata.get('texture') or {}
    sx, sy = width / metadata['width'], height / metadata['height']
    big = {key: (upsample(value, shape) if np.ndim(value) == 2 else value) for key, value in maps.items()}
    rows = columns = None
    if metadata.get('mode') == 'reveal':
        line_layer = np.zeros(shape + (4,), np.uint8)
        paint_layer = shade_pixels = pixels
        if fields is not None:
            front = paint_motion.front(fields, shape, factor, colours=np.array(Image.open(ASSET_DIRECTORY / f'{name}-shade.png').convert('RGBA')))
            big.update(color=front['color'], color_cover=front['color_cover'], color_full=front['color_full'])
    else:
        # Lines at this size are the picture's own (the line plate's mask, or the ink found at this size) where the plan
        # timed a line, so a line edge is smooth instead of a planned pixel scaled up; the zone adds the ink's soft edge.
        coarse_line = np.array(Image.open(ASSET_DIRECTORY / f'{name}-line.png').convert('RGBA'))[:, :, 3]
        # (Where the plan timed ink: the line layer itself may be thinned by pressure, the source's line is not.)
        planned = upsample((coarse_line > 0) | np.isfinite(maps['ink']), shape)
        clean = art['outline'] & ndimage.binary_dilation(planned, iterations=max(1, int(np.ceil(factor))))
        if not clean.any():
            clean = upsample(coarse_line > 150, shape)
        line_zone = ndimage.binary_dilation(clean, iterations=2)
        shade_pixels = paint_motion.inpaint_lines(pixels, line_zone)
        shade_pixels[:, :, 3] = np.where(shade_pixels[:, :, 3] >= 96, shade_pixels[:, :, 3], 0)
        # Every plan map (paint times, parts, flat colours) follows the fine picture's edges, not the plan's pixel grid.
        coarse_shade = np.array(Image.open(ASSET_DIRECTORY / f'{name}-shade.png').convert('RGBA'))
        if fields is not None and 'tone' in fields:   # compare colours as painted, before the layered-marker tone
            coarse_shade = paint_motion.remove_tone(coarse_shade, fields['tone'])
        rows, columns = guided_index(coarse_shade, shade_pixels)
        big = {key: (value[rows, columns] if np.ndim(value) == 2 else value) for key, value in maps.items()}
        coarse_flat = np.array(Image.open(ASSET_DIRECTORY / f'{name}-paint.png').convert('RGBA'))[:, :, :3]
        flat = coarse_flat[rows, columns]
        if fields is not None:
            front = paint_motion.front(fields, shape, factor, (rows, columns), coarse_shade, coarse_flat)
            flat = front.pop('flat')
            big.update(front)
        line_rgb, line_alpha = source_ink_layer(pixels, shade_pixels, clean, line_zone)
        if fields is not None and 'tone' in fields:   # layered marker coats, as the plan's shade layer has them
            shade_pixels = paint_motion.apply_tone(shade_pixels, fields['tone'])
        # Fine ink the coarse plan never timed (a hair-thin edge) takes the time of the nearest timed ink.
        ink = big['ink']
        timed = np.isfinite(ink)
        if (line_alpha > 0).any() and timed.any():
            _, nearest = ndimage.distance_transform_edt(~timed, return_indices=True)
            ink = np.where(timed, ink, ink[nearest[0], nearest[1]])
            big['ink'] = np.where(line_alpha > 0, ink, np.inf).astype(np.float32)
            big['ink_owner'] = np.where(line_alpha > 0, big['ink_owner'][nearest[0], nearest[1]], -1).astype(np.int16)
        if texture.get('ink') == 'pressure':
            line_alpha = line_drawing.ink_texture(line_alpha, ink_paths(metadata['segments'], (sx, sy)), (sx + sy) / 2,
                                                  (metadata['height'], metadata['width']))
        line_layer = np.dstack([np.rint(line_rgb), np.rint(line_alpha * 255)]).astype(np.uint8)
        painted = shade_pixels[:, :, 3] > 0
        # Paint a fine pixel when the marker reaches the nearest painted pixel of the same part (parts keep their order);
        # a speck of paint only the larger render shows, far from any planned paint, is left out (a few pixels).
        owners_big = maps['owners'][rows, columns]
        owned = owners_big >= 0
        if owned.any() and not owned.all():
            _, nearest = ndimage.distance_transform_edt(~owned, return_indices=True)
            owners_big = owners_big[nearest[0], nearest[1]]
        stray = np.zeros(shape, bool)
        for key in ('color', 'color_full', 'shade'):
            if key not in big:
                continue
            values = big[key].copy()
            for part in np.unique(owners_big[painted]):
                mine = owners_big == part
                known = np.isfinite(values) & mine
                missing = painted & mine & ~np.isfinite(values)
                if not missing.any():
                    continue
                if not known.any():
                    stray |= missing
                    continue
                distance, nearest = ndimage.distance_transform_edt(~known, return_indices=True)
                close = missing & (distance <= 3 * factor)
                values[close] = values[nearest[0][close], nearest[1][close]]
                stray |= missing & ~close
            big[key] = values.astype(np.float32)
        if stray.any():
            shade_pixels[stray, 3] = 0
            painted &= ~stray
        paint_layer = np.dstack([flat, shade_pixels[:, :, 3]]).astype(np.uint8)
        if fields is not None:   # the front's coat strength, already at this size
            big['color_cover'] = np.where(painted & (big['color_cover'] == 0), 1., big['color_cover']).astype(np.float32)
        if metadata.get('shading_bands'):   # broad soft shading strokes, redrawn at this size
            paint_motion.band_times(big['shade'], metadata['shading_bands'], owners_big, (sx, sy))
        elif 'color_cover' in maps:  # a continuous coat strength (felt streaks): interpolated, not blocky
            cover = np.clip(ndimage.zoom(np.asarray(maps['color_cover'], np.float32), (shape[0] / maps['color_cover'].shape[0],
                                         shape[1] / maps['color_cover'].shape[1]), order=1), 0, 1)[:shape[0], :shape[1]]
            big['color_cover'] = np.where(painted & (cover == 0), 1., cover).astype(np.float32)
    sketch = None
    if (ASSET_DIRECTORY / f'{name}-sketch.png').exists():
        sketch = np.array(Image.open(ASSET_DIRECTORY / f'{name}-sketch.png').convert('RGBA').resize((shape[1], shape[0]), Image.Resampling.BILINEAR))
    if 'color_cover' in maps and metadata.get('mode') == 'reveal' and fields is None:  # a continuous coat strength: interpolated, not blocky
        big['color_cover'] = np.clip(ndimage.zoom(np.asarray(maps['color_cover'], np.float32), (shape[0] / maps['color_cover'].shape[0],
                                     shape[1] / maps['color_cover'].shape[1]), order=1), 0, 1)[:shape[0], :shape[1]]
    owners = big.pop('owners')
    final = illustration_layers.prepare(big, line_layer, paint_layer, shade_pixels, sketch)['final']
    owners_big = owners
    owned = owners_big >= 0
    if owned.any() and not owned.all():  # a fine edge pixel outside the coarse picture belongs to its nearest part
        _, nearest = ndimage.distance_transform_edt(~owned, return_indices=True)
        owners_big = owners_big[nearest[0], nearest[1]]
    owners_final = np.where(final[:, :, 3] > 0, owners_big, -1).astype(np.int16)
    if fields is not None and metadata.get('colouring') == 'markers':
        # The front at this size can take a time from a neighbouring plan pixel of another module in the same area: report
        # each module's colouring as its own pixels received it here, as the plan did.
        color, settle = big['color'], big['color_full']
        for part in metadata['parts']:
            mine = (owners_final == part['index']) & np.isfinite(color)
            if mine.any():
                part['fill_start'] = float(color[mine].min())
                part['fill_end'] = float(max(settle[mine & np.isfinite(settle)].max(initial=0.), color[mine].max()))
    for suffix, layer in (('line', line_layer), ('paint', paint_layer), ('shade', shade_pixels), ('sketch', sketch)):
        if layer is not None:
            Image.fromarray(layer).save(ASSET_DIRECTORY / f'{name}-{suffix}.png')
    Image.fromarray(final).save(ASSET_DIRECTORY / f'{name}.png')
    np.savez_compressed(ASSET_DIRECTORY / f'{name}.npz', owners=owners_final, **big)
    for segment in metadata['segments']:
        points = to_render(segment['points'], (sx, sy))
        weights = np.asarray(segment['lengths'], float) * (sx + sy) / 2 if 'lengths' in segment else None
        if segment['kind'] in ('ink', 'sketch') and len(points) > 1 and weights is not None and len(weights) == len(points):
            # A sample every pixel along the inked path (the verifier checks the nib never jumps between samples). The
            # nib keeps the plan's timing: its curvature-weighted length is carried over to the new samples.
            arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
            fine = np.linspace(0, arc[-1], max(2, int(np.ceil(arc[-1])) + 1))
            weights = np.interp(fine, arc, weights)
            points = np.c_[np.interp(fine, arc, points[:, 0]), np.interp(fine, arc, points[:, 1])]
        segment['points'] = points.tolist()
        if weights is not None:
            segment['lengths'] = weights.tolist()
        if 'radius' in segment:
            segment['radius'] *= (sx + sy) / 2
    metadata.update(width=width, height=height, render_scale=round(factor, 3))
    (ASSET_DIRECTORY / f'{name}.json').write_text(json.dumps(metadata))
    return metadata
