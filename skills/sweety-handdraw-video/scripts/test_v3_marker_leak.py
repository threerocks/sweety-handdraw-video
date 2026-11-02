"""Colouring by marker: a small figure in front of a large flat background gets its own colour in its own pass (sky-colors
film, 2026-11-02: a small man under a sky blue on the left and orange on the right was painted as the sky on either side
of his centreline, and turned blue only when a darker 'shadow' marker went over him). Marker colouring treats the whole
picture as one part, so every closed area under 1% of the picture was absorbed by its nearest large area."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

import illustration_layers
import line_drawing
import paint_motion
import stroke_assets

W, H = 300, 450   # 1% of the picture is 1350 px: every closed area of the figure is smaller
SKY_LEFT, SKY_RIGHT, GROUND = (120, 180, 230), (240, 135, 50), (130, 180, 100)
SKIN, JACKET, TROUSERS, SUN, INK = (232, 190, 160), (60, 90, 150), (40, 55, 85), (245, 200, 51), (20, 20, 24)
PARTS = [{'id': 'sky', 'box': [0, 0, 1, .66], 'exclude_boxes': [[.43, .43, .57, .67]]},
         {'id': 'figure', 'box': [.43, .43, .57, .67]},
         {'id': 'ground', 'box': [0, 0, 1, 1]}]
# Interiors (x0, y0, x1, y1), a few px inside their outlines.
JACKET_BOX, LEG_BOXES, HEAD_BOX, SUN_BOX = (137, 227, 164, 254), ((140, 263, 146, 295), (154, 263, 160, 295)), (145, 205, 156, 216), (55, 55, 70, 70)
STRADDLE = (142, 152, 159, 168)        # a closed box across the sky's colour change, with no line between the two paints


def scene(root):
    """A sky blue on the left and orange on the right over a hill line; a small figure (skin head, blue jacket, two
    dark trouser legs split by a line) stands across the colour change. A yellow sun (lighter than the sky), a closed
    ring of the sky's own colour, and a small closed box across the colour change. Returns (source, outline mask)."""
    image, mask = Image.new('RGB', (W, H), SKY_LEFT), Image.new('L', (W, H), 0)
    draw = ImageDraw.Draw(image)
    draw.rectangle([150, 0, W, 300], fill=SKY_RIGHT)
    draw.rectangle([0, 300, W, H], fill=GROUND)
    for pen, ink in ((draw, INK), (ImageDraw.Draw(mask), 255)):
        fill = lambda colour: colour if pen is draw else None
        pen.line([150, 0, 150, 150], fill=ink, width=3)                       # the divider runs to the box ...
        pen.rectangle([140, 150, 161, 170], outline=ink, width=3)              # ... which closes blue and orange together
        pen.line([150, 170, 150, 199], fill=ink, width=3)                     # ... and on to the head
        pen.line([0, 300, W, 300], fill=ink, width=3)
        pen.ellipse([138, 198, 162, 222], fill=fill(SKIN), outline=ink, width=3)
        pen.rectangle([132, 222, 168, 258], fill=fill(JACKET), outline=ink, width=3)
        pen.rectangle([136, 258, 164, 299], fill=fill(TROUSERS), outline=ink, width=3)
        pen.line([150, 258, 150, 299], fill=ink, width=3)
        pen.ellipse([46, 46, 78, 78], fill=fill(SUN), outline=ink, width=3)
        pen.ellipse([220, 60, 244, 84], outline=ink, width=3)
    image.save(root / 'scene.png')
    mask.save(root / 'scene-outline.png')
    return root / 'scene.png', root / 'scene-outline.png'


def interior(box, shape=(H, W), factor=1):
    mask = np.zeros(shape, bool)
    x0, y0, x1, y1 = (round(value * factor) for value in box)
    mask[y0:y1, x0:x1] = True
    return mask


def distance(pixels, colour):
    return np.linalg.norm(pixels[..., :3].astype(float) - np.asarray(colour, float), axis=-1)


class MarkerRegionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source, self.outline = scene(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def regions(self):
        """Regions as colouring by marker builds them: the areas the lines close, the whole picture as one part."""
        rgb = np.array(Image.open(self.source)).astype(float)
        ink = np.array(Image.open(self.outline)) > 127
        labels = line_drawing.areas(ink)
        regions = paint_motion.color_regions(rgb, np.ones((H, W), bool), np.zeros((H, W), np.int16), 1, labels)
        flat, index = np.zeros((H, W, 3)), np.full((H, W), -1)
        for number, region in enumerate(regions):
            (left, top), (height, width) = region['offset'], region['mask'].shape
            flat[top:top + height, left:left + width][region['mask']] = region['flat']
            index[top:top + height, left:left + width][region['mask']] = number
        return regions, flat, index

    def test_a_small_closed_area_of_its_own_colour_is_its_own_region(self):
        regions, flat, _ = self.regions()
        for box, colour in [(JACKET_BOX, JACKET), (HEAD_BOX, SKIN), (SUN_BOX, SUN)] + [(leg, TROUSERS) for leg in LEG_BOXES]:
            self.assertLess(float(distance(flat[interior(box)], colour).max()), 12, colour)
        # The jacket is no longer the sky on either side of the centreline.
        self.assertGreater(float(distance(flat[interior(JACKET_BOX)], SKY_RIGHT).min()), 100)

    def test_areas_of_one_colour_share_a_region_and_add_no_pass(self):
        regions, flat, index = self.regions()
        # The two trouser legs (a line apart) are one marker's one round; the ring of sky stays with the sky.
        left_leg, right_leg = (set(np.unique(index[interior(leg)])) for leg in LEG_BOXES)
        self.assertEqual(left_leg, right_leg)
        ring, sky = index[72, 232], index[200, 260]
        self.assertEqual(ring, sky)
        # Two skies, the ground, a head, a jacket, the legs and the sun: nothing more.
        self.assertEqual(len(regions), 7)

    def test_an_area_of_two_paints_with_no_line_between_keeps_both(self):
        _, flat, _ = self.regions()
        x0, y0, x1, y1 = STRADDLE
        self.assertLess(float(distance(flat[y0:y1, x0:148], SKY_LEFT).max()), 12)
        self.assertLess(float(distance(flat[y0:y1, 153:x1], SKY_RIGHT).max()), 12)

    def test_without_closed_areas_parts_colouring_is_unchanged(self):
        # The palette regions of a part (no region map) never passed through the closed-area rule.
        rgb = np.array(Image.open(self.source)).astype(float)
        owners = np.zeros((H, W), np.int16)
        owners[:, 150:] = 1
        first = paint_motion.color_regions(rgb, np.ones((H, W), bool), owners, 2)
        original = paint_motion.own_colour_areas
        paint_motion.own_colour_areas = lambda found, rgb, kept, size=1.: list(kept)
        try:
            second = paint_motion.color_regions(rgb, np.ones((H, W), bool), owners, 2)
        finally:
            paint_motion.own_colour_areas = original
        self.assertEqual(len(first), len(second))
        for one, two in zip(first, second):
            np.testing.assert_array_equal(one['mask'], two['mask'])


class MarkerFilmTests(unittest.TestCase):
    """The built picture, frame by frame: the figure shows no sky colour while the skies are coloured, and comes in with
    its own marker after them (light to dark)."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source, self.outline = scene(self.root)
        stroke_assets.configure_assets(self.root, '/dev/null', 0, 30)

    def tearDown(self):
        self.temporary.cleanup()

    def build(self, name, texture=True, factor=None):
        metadata = stroke_assets.build_illustration(name, self.source, W, H, None, PARTS, self.outline, seconds=12,
                                                    drawing={'sketch': False, 'texture': texture})
        if factor:
            metadata = stroke_assets.render_at(name, self.source, factor, None, self.outline)
        return metadata, illustration_layers.load(self.root, name)

    def check_figure(self, layers, factor=1):
        shape = layers['alpha'].shape
        jacket, orange = interior(JACKET_BOX, shape, factor), interior((200, 100, 290, 290), shape, factor)
        orange &= ~interior((215, 55, 249, 89), shape, factor)          # the ring's own outline
        # The flat marker coat of the figure is its own colour, not the sky's.
        self.assertLess(float(np.median(distance(layers['flat'][jacket] * 255, JACKET))), 12)
        # Its own pass: after the orange sky's marker has finished.
        colour = layers['color']
        self.assertTrue(np.isfinite(colour[jacket]).all())
        self.assertGreaterEqual(float(colour[jacket].min()), float(colour[orange].max()))
        # Frame by frame, painted figure pixels are the jacket's colour, never a sky's.
        for progress in np.linspace(0, 1, 41):
            frame = illustration_layers.compose(layers, np.float32(progress))
            painted = jacket & (frame[..., 3] > 40)
            if not painted.any():
                continue
            pixels = frame[painted]
            own = distance(pixels, JACKET)
            for sky in (SKY_LEFT, SKY_RIGHT):
                self.assertTrue((own < distance(pixels, sky)).all(), f'sky colour on the figure at {progress:.3f}')
        np.testing.assert_array_equal(illustration_layers.compose(layers, np.float32(1.)), layers['final'])

    def test_the_figure_is_coloured_in_its_own_pass(self):
        for texture in (True, {'marker': 'flat'}):
            with self.subTest(texture=texture):
                metadata, layers = self.build('layered' if texture is True else 'flat', texture)
                self.assertEqual(metadata['colouring'], 'markers')
                self.check_figure(layers)

    def test_the_finished_picture_is_still_the_source(self):
        _, flat = self.build('flat', {'ink': 'none', 'marker': 'flat'})
        _, layered = self.build('layered')
        source = np.array(Image.open(self.source)).astype(int)
        away = ~ndimage.binary_dilation(np.array(Image.open(self.outline)) > 0, iterations=4)
        np.testing.assert_array_equal(flat['final'][away, :3].astype(int), source[away, :3])
        painted = away & (flat['alpha'] > .99)
        self.assertLess(float(illustration_layers.delta_e(layered['final'], flat['final'])[painted].mean()), 3.)

    def test_a_close_up_keeps_the_figure_in_its_own_colour(self):
        _, layers = self.build('close', factor=2)
        self.check_figure(layers, factor=2)


if __name__ == '__main__':
    unittest.main()
