"""v3 drawing media: the colouring front is drawn at display resolution and anti-aliased (a close-up no longer scales up
the plan's pixel staircase while marker colour is laid down), and real-media texture (drawing.texture): pressure ink
(free ends land and lift thin, the body thins a little, the ink is a little lighter in places) and layered markers
(lanes of a real nib along the sweep, darker where two coats overlap, a little pooling at a lane's edge)."""
import json
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

import illustration_layers
import line_drawing
import paint_motion
import stroke_assets
from video_verifier import match_reveal_to_tip

PARTS = [{'id': 'left', 'box': [0, 0, .5, 1]}, {'id': 'right', 'box': [.5, 0, 1, 1]}]


def picture(path, scale=3):
    """Two flat colour areas closed by an outline, and an open curved line with two free ends."""
    image = Image.new('RGB', (300 * scale, 220 * scale), (252, 250, 244))
    draw = ImageDraw.Draw(image)
    s = lambda *values: [value * scale for value in values]
    draw.rounded_rectangle(s(12, 12, 288, 208), radius=18 * scale, fill=(222, 200, 140), outline=(24, 24, 28), width=3 * scale)
    draw.line(s(150, 12, 150, 208), fill=(24, 24, 28), width=3 * scale)
    draw.rectangle(s(152, 15, 285, 205), fill=(236, 150, 72))
    draw.line(s(40, 150, 70, 110, 110, 100), fill=(24, 24, 28), width=3 * scale, joint='curve')
    image.save(path)
    return path


def paint_alpha(layers, progress):
    front, coat = layers['paint_ramps']
    ramp = illustration_layers.ramp
    with np.errstate(invalid='ignore'):
        wet = layers['cover'] * np.clip((progress - layers['color']) / ramp(layers['color'], front), 0, 1)
        dry = np.clip((progress - layers['full']) / ramp(layers['full'], coat), 0, 1)
    return np.nan_to_num((wet + (1 - wet) * dry) * layers['alpha'])


def front_stats(alpha, zone, factor):
    """Share of the colouring front's crossings (paint shows / does not, between two pixels) that sit on an edge of the
    plan's pixel grid (a plan-pixel staircase puts them all there), and share of the front band with a partial
    (anti-aliased) alpha."""
    crossings, on_grid = 0, 0
    for a, z in ((alpha, zone), (alpha.T, zone.T)):
        inside = a >= .25
        rows, cols = np.nonzero((inside[:, 1:] != inside[:, :-1]) & z[:, 1:] & z[:, :-1])
        position = cols + 1.
        crossings += len(rows)
        on_grid += int((np.abs(np.rint(position / factor) * factor - position) < .5).sum())
    band = ndimage.binary_dilation(alpha >= .5) & ndimage.binary_dilation(alpha < .5) & zone
    partial = (alpha > .08) & (alpha < .92) & band
    return on_grid / max(1, crossings), partial.sum() / max(1, band.sum()), crossings


class MediaTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = picture(self.root / 'source.png')

    def tearDown(self):
        self.temporary.cleanup()

    def build(self, name, texture=True, factor=None, directory=None):
        stroke_assets.configure_assets(directory or self.root, '/dev/null')
        metadata = stroke_assets.build_illustration(name, self.source, 300, 220, parts=PARTS, seconds=6, drawing={'texture': texture})
        if factor:
            metadata = stroke_assets.render_at(name, self.source, factor)
        return metadata, illustration_layers.load(directory or self.root, name)

    def test_colouring_front_is_drawn_at_display_size_and_anti_aliased(self):
        # The plan's first touch and settle times scaled up by nearest neighbour (as close-ups were) against the front
        # drawn at 3x.
        self.build('plan', directory=self.root / 'plan')
        fields = np.load(self.root / 'plan' / 'plan.npz')
        metadata, close = self.build('close', factor=3)
        shape = close['alpha'].shape
        scaled = dict(close, **{key: stroke_assets.upsample(fields[field], shape)
                                for key, field in (('color', 'paint_first'), ('cover', 'paint_wet'), ('full', 'paint_full'))})
        zone = (close['alpha'] > .99) & ~ndimage.binary_dilation(close['line_alpha'] > .05, iterations=4)
        progress = float(np.quantile(close['full'][np.isfinite(close['full']) & zone], .5))
        grid_new, partial_new, count_new = front_stats(paint_alpha(close, progress), zone, 3.)
        grid_old, _, count_old = front_stats(paint_alpha(scaled, progress), zone, 3.)
        self.assertGreater(count_new, 20)
        self.assertGreater(grid_old, .9)                 # a staircase: every crossing on a plan pixel edge
        self.assertLess(grid_new, .6)
        self.assertGreater(partial_new, .2)               # (a wet coat is partial too, so the old maps are no reference here)
        # Still the same drawing: colour only where the brush is, and every colour settles by the end.
        painted = np.isfinite(close['color'])
        distances, radii = match_reveal_to_tip(metadata, close['color'], painted, {'color'})
        self.assertTrue((distances <= radii + .04 + (3 + 2 ** .5) * 3).all())
        np.testing.assert_array_equal(illustration_layers.compose(close, 1.), close['final'])

    def test_plan_size_front_is_anti_aliased_and_keeps_the_brush(self):
        metadata, layers = self.build('plan')
        painted = np.isfinite(layers['color'])
        distances, radii = match_reveal_to_tip(metadata, layers['color'], painted, {'color'})
        self.assertTrue((distances <= radii + .04).all())
        self.assertTrue((layers['full'][painted] >= layers['color'][painted]).all())
        self.assertGreater(float(((layers['cover'] > 0) & (layers['cover'] < .4) & painted).sum()), 0)   # partial edge pixels

    def test_texture_is_deterministic(self):
        self.build('one', directory=self.root / 'a')
        self.build('one', directory=self.root / 'b')
        for suffix in ('.png', '-line.png', '-shade.png', '-paint.png'):
            self.assertEqual((self.root / 'a' / f'one{suffix}').read_bytes(), (self.root / 'b' / f'one{suffix}').read_bytes())
        first, second = np.load(self.root / 'a' / 'one.npz'), np.load(self.root / 'b' / 'one.npz')
        for key in first.files:
            np.testing.assert_array_equal(first[key], second[key])

    def test_texture_off_is_the_even_pen_and_flat_marker(self):
        metadata, off = self.build('off', texture=False)
        self.assertEqual(metadata['texture'], {'ink': 'none', 'marker': 'flat'})
        self.assertNotIn('paint_tone', np.load(self.root / 'off.npz').files)
        _, explicit = self.build('explicit', texture={'ink': 'none', 'marker': 'flat'})
        for key in ('final', 'line_alpha', 'shade_rgb'):
            np.testing.assert_array_equal(off[key], explicit[key])
        # The shade layer is the source's own colour wherever no line is near.
        artwork = np.array(stroke_assets.prepare_artwork(self.source, 300, 220)['artwork']).astype(int)
        shade = np.array(Image.open(self.root / 'off-shade.png')).astype(int)
        away = (artwork[..., 3] == 255) & ~ndimage.binary_dilation(off['line_alpha'] > 0, iterations=8)
        np.testing.assert_array_equal(shade[away, :3], artwork[away, :3])
        _, ink_only = self.build('ink', texture={'ink': 'pressure', 'marker': 'flat'})
        np.testing.assert_array_equal(ink_only['shade_rgb'], off['shade_rgb'])

    def test_texture_stays_within_its_tolerance(self):
        # The finished picture is no longer the source exactly; these bounds are what the texture may do to it.
        _, off = self.build('off', texture=False)
        _, on = self.build('on')
        painted = (off['alpha'] > .99) & (off['line_alpha'] < .02)
        difference = illustration_layers.delta_e(on['final'], off['final'])
        self.assertLess(float(difference[painted].mean()), 3.)              # just noticeable on average, no more
        luminance = lambda pixels: pixels[..., :3].astype(float) @ [.299, .587, .114]
        shift = luminance(on['final'])[painted] / np.maximum(luminance(off['final'])[painted], 1) - 1
        low, high = paint_motion.TONE_RANGE
        self.assertGreaterEqual(shift.min(), low - 1 - .01)
        self.assertLessEqual(shift.max(), high - 1 + .01)
        for area in (np.s_[:, :140], np.s_[:, 170:]):                         # each area keeps its mean colour
            mine = painted.copy()
            mine[...] = False
            mine[area] = painted[area]
            self.assertLess(abs(float(luminance(on['final'])[mine].mean() / luminance(off['final'])[mine].mean()) - 1), .015)
        # Ink only thins: never added, and the line's core stays where the source has it (registration).
        self.assertFalse((on['line_alpha'] > off['line_alpha'] + 2 / 255).any())
        core = ndimage.distance_transform_edt(off['line_alpha'] >= .5) >= 1.5
        self.assertGreater(float(((on['line_alpha'] >= .5) & core).sum() / core.sum()), .99)

    def test_free_line_ends_land_and_lift_thin(self):
        _, off = self.build('off', texture=False)
        _, on = self.build('on')
        # The open curve (40,150) -> (110,100): width across it near each end and in its middle.
        def width(layers, x, y):
            patch = layers['line_alpha'][y - 6:y + 7, x - 6:x + 7]
            return float(patch.sum())
        middle_on, middle_off = width(on, 70, 112), width(off, 70, 112)
        for x, y in ((42, 147), (107, 101)):
            self.assertLess(width(on, x, y) / middle_on, .8 * width(off, x, y) / middle_off)
        self.assertGreater(middle_on, .8 * middle_off)
        # The outline stays clean: no new fragments, no specks.
        count = lambda layers: ndimage.label(layers['line_alpha'] > .2, structure=np.ones((3, 3)))[1]
        self.assertEqual(count(on), count(off))

    def test_shading_band_edges_blend_at_display_size(self):
        band = {'part': 0, 'center': [10., 10.], 'axis': [0., 1.], 'across': [1., 0.], 'edges': [-10., 0., 10.],
                'along': [-10., 10.], 'times': [[.2, .3], [.6, .7]]}
        owners = np.zeros((60, 60), np.int16)
        shade = np.full((60, 60), np.inf, np.float32)
        paint_motion.band_times(shade, [band], owners, (3., 3.))
        row = shade[30]
        self.assertTrue(((row > .2 + 1e-3) & (row < .5) & (np.abs(row - (.2 + .1 * .8 * .5)) > .02)).any() or
                        ((row > .32) & (row < .58)).any())           # a straddling pixel takes a blended time
        self.assertLessEqual(int(((row > .32) & (row < .58)).sum()), 2)  # ... only at the edge itself

    def test_texture_option_is_checked_and_keys_the_asset_cache(self):
        with self.assertRaisesRegex(ValueError, 'drawing.texture'):
            stroke_assets.texture_option({'texture': {'ink': 'bristly'}})
        with self.assertRaisesRegex(ValueError, 'drawing.texture'):
            stroke_assets.texture_option({'texture': {'paper': 'rough'}})
        self.assertEqual(stroke_assets.texture_option({}), {'ink': 'pressure', 'marker': 'layered'})
        self.assertEqual(stroke_assets.texture_option({'texture': False}), {'ink': 'none', 'marker': 'flat'})
        import video_job
        action = {'type': 'illustration', 'source': 'source.png', 'width': 300, 'height': 220, 'x': 0, 'y': 0}
        keys = []
        for drawing in ({}, {'texture': False}, {'texture': {'ink': 'none'}}):
            builder = types.SimpleNamespace(manifest={'drawing': drawing}, root=self.root, font_path='/dev/null', font_index=0, fps=30,
                                            PLACEMENT_KEYS=video_job.FilmBuilder.PLACEMENT_KEYS)
            keys.append(video_job.FilmBuilder.asset_key(builder, action, 6., 1.))
        self.assertEqual(len(set(keys)), 3)


if __name__ == '__main__':
    unittest.main()
