"""v3.0.2: the three engine bugs the 3.0 promo film ran into.

- A close-up re-render mapped the nib path by `point * s`, (s - 1) / 2 px up and left of the picture it draws (pixel
  centres map to `point * s + (s - 1) / 2`), so colour on the far side of a stroke read as outside the brush.
- A near-square line plate (1199 x 1312) is padded onto a square canvas for the colour edit; line_first stretched the
  returned square back onto the plate and squashed every line (68.6% kept).
- `art_pipeline.py JOB gen --edit NAME`, the form the docs gave, was rejected: flags stopped the names."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw

import art_pipeline
import illustration_layers
import line_first
import stroke_assets
from test_v3_media import PARTS, picture
from video_verifier import match_reveal_to_tip


class CloseUpNibTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = picture(self.root / 'source.png')
        stroke_assets.configure_assets(self.root, '/dev/null')

    def tearDown(self):
        self.temporary.cleanup()

    def test_render_coordinates_align_pixel_centres(self):
        # Plan pixel 10 covers render pixels 30, 31, 32 at 3x: its centre is 31, not 30.
        np.testing.assert_allclose(stroke_assets.to_render([[10., 20.]], (3., 3.)), [[31., 61.]])
        np.testing.assert_allclose(stroke_assets.to_render([[10., 20.]], (1., 1.)), [[10., 20.]])

    def test_a_close_up_keeps_the_nib_on_the_picture(self):
        plan = stroke_assets.build_illustration('pic', self.source, 300, 220, parts=PARTS, seconds=6)
        planned = [np.asarray(segment['points'], float) for segment in plan['segments']]
        close = stroke_assets.render_at('pic', self.source, 3)
        sx, sy = close['width'] / plan['width'], close['height'] / plan['height']
        for before, segment in zip(planned, close['segments']):
            if segment['kind'] == 'color':
                np.testing.assert_allclose(np.asarray(segment['points']), before * [sx, sy] + [(sx - 1) / 2, (sy - 1) / 2], atol=1e-6)
        # Colour stays within the brush, with room to spare: the bound the verifier uses for a close-up.
        layers = illustration_layers.load(self.root, 'pic')
        painted = np.isfinite(layers['color'])
        distances, radii = match_reveal_to_tip(close, layers['color'], painted, {'color'})
        slack = .04 + (3. + 2 ** .5) * close['render_scale']
        self.assertTrue((distances <= radii + slack).all())


def line_plate(size):
    """A line plate of a few closed shapes and open strokes, black on white."""
    image = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(image)
    w, h = size
    draw.rectangle((w * .1, h * .1, w * .45, h * .4), outline='black', width=4)
    draw.ellipse((w * .55, h * .12, w * .9, h * .45), outline='black', width=4)
    draw.polygon([(w * .15, h * .9), (w * .5, h * .55), (w * .85, h * .9)], outline='black', width=4)
    draw.line((w * .1, h * .5, w * .9, h * .52), fill='black', width=4)
    return image


def coloured(plate):
    """The plate with flat fills behind its lines, as a colour edit returns it."""
    colour = plate.copy()
    w, h = plate.size
    for point, fill in (((w * .27, h * .25), (230, 120, 90)), ((w * .72, h * .28), (90, 160, 210)), ((w * .5, h * .8), (120, 190, 110))):
        ImageDraw.floodfill(colour, tuple(int(v) for v in point), fill)
    return colour


class NearSquarePlateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def run_pair(self, plate, colour):
        plate.save(self.root / 'plate.png')
        colour.save(self.root / 'colour.png')
        return line_first.run(self.root, 'plate.png', 'colour.png', 'out.png')

    def test_a_colour_edit_on_a_padded_square_canvas_is_cut_back_not_stretched(self):
        plate = line_plate((600, 656))                  # aspect .91: the image route calls it square
        canvas = Image.new('RGB', (656, 656), 'white')  # ... pads it to a square canvas
        canvas.paste(plate, (28, 0))
        returned = coloured(canvas).resize((627, 627), Image.Resampling.LANCZOS)   # and gets back a square of its own size
        report = self.run_pair(plate, returned)
        self.assertEqual(report['status'], 'pass', report['problems'])
        self.assertEqual(report['colour_fit'], 'centre')
        self.assertGreaterEqual(report['kept_lines'], line_first.MIN_KEPT)

    def test_an_edit_on_the_plates_own_aspect_is_still_stretched(self):
        plate = line_plate((600, 900))
        report = self.run_pair(plate, coloured(plate).resize((640, 960), Image.Resampling.LANCZOS))
        self.assertEqual(report['status'], 'pass', report['problems'])
        self.assertEqual(report['colour_fit'], 'stretched')

    def test_fits_cut_the_plate_box_from_the_middle(self):
        found = dict(line_first.fits(Image.new('RGBA', (400, 400)), (300, 400)))
        self.assertEqual(set(found), {'stretched', 'centre'})
        self.assertEqual(found['centre'].size, (300, 400))
        self.assertEqual([how for how, _ in line_first.fits(Image.new('RGBA', (600, 800)), (300, 400))], ['stretched'])


class GenFlagOrderTests(unittest.TestCase):
    def test_flags_go_before_or_after_the_names(self):
        calls = []

        class Pipeline:
            images = {'c02-park': {}, 'a': {}, 'b': {}}

            def __init__(self, job, specs):
                pass

            def generate(self, name, edit, force):
                calls.append((name, edit, force))

        with mock.patch.object(art_pipeline, 'Pipeline', Pipeline):
            for argv in (['gen', '--edit', 'c02-park'], ['gen', 'c02-park', '--edit'], ['gen', '--parallel', '1', 'a', 'b', '--force']):
                art_pipeline.main(['/job'] + argv)
        self.assertEqual(calls, [('c02-park', True, False), ('c02-park', True, False), ('a', False, True), ('b', False, True)])


if __name__ == '__main__':
    unittest.main()
