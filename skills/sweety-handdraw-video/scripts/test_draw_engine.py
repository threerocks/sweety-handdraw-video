"""draw-v2 engine: continuous strokes, line-art gate, Set-of-Mark ownership, layered no-pop output, human-likeness metrics."""
from pathlib import Path
import json
import tempfile
import unittest
import numpy as np
from PIL import Image, ImageDraw
import engine_metrics
import illustration_layers
import line_topology
import lineart_gate
import region_marks
import stroke_assets

DRAWABILITY = {'line_length_px', 'stroke_count', 'drawn_stroke_count', 'fill_area_px',
               'required_seconds', 'minimum_seconds', 'lineart_gate'}


def figure(path):
    image = Image.new('RGBA', (240, 300))
    draw = ImageDraw.Draw(image)
    draw.ellipse((70, 10, 170, 110), fill='#EAC4A5', outline='#252B39', width=4)
    draw.rectangle((50, 120, 190, 220), fill='#F1924A', outline='#252B39', width=4)
    draw.line((120, 120, 120, 220), fill='#252B39', width=3)
    draw.rectangle((65, 230, 105, 290), fill='#557B8D', outline='#252B39', width=4)
    draw.rectangle((135, 230, 175, 290), fill='#557B8D', outline='#252B39', width=4)
    image.save(path)
    return [{'id': name, 'polygon': [[0, top], [1, top], [1, bottom], [0, bottom]]}
            for name, top, bottom in [('head', 0, .385), ('torso', .385, .75), ('legs', .75, 1)]]


def texture(path):
    rng = np.random.default_rng(1)
    image = Image.new('RGBA', (300, 300))
    draw = ImageDraw.Draw(image)
    draw.ellipse((20, 20, 280, 280), fill='#C8A070', outline='#202020', width=3)
    for _ in range(500):
        x, y = rng.uniform(40, 260, 2)
        angle, length = rng.uniform(0, np.pi), rng.uniform(4, 14)
        draw.line((x, y, x + length * np.cos(angle), y + length * np.sin(angle)), fill='#202020', width=2)
    image.save(path)


class DrawEngineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        stroke_assets.configure_assets(self.directory, '/dev/null', 0, 30)

    def tearDown(self):
        self.temporary.cleanup()

    def test_strokes_continue_through_junctions_and_drop_specks(self):
        image = Image.new('L', (200, 200))
        draw = ImageDraw.Draw(image)
        draw.line((20, 100, 180, 100), fill=255, width=3)
        draw.line((100, 20, 100, 180), fill=255, width=3)
        draw.ellipse((40, 40, 160, 160), outline=255, width=3)
        draw.rectangle((5, 5, 6, 6), fill=255)
        strokes, _, _, statistics = line_topology.extract_strokes(np.array(image) > 0)
        straight = [stroke for stroke in strokes if np.ptp(stroke['points'][:, 0]) < 6 or np.ptp(stroke['points'][:, 1]) < 6]
        self.assertEqual(len(straight), 2)
        self.assertTrue(all(stroke['length'] > 150 for stroke in straight))
        self.assertLessEqual(len(strokes), 4)
        self.assertGreater(statistics['specks_removed_px'], 0)
        self.assertTrue(all(stroke['length'] >= 15 for stroke in strokes))

    def test_layered_asset_has_no_pop_and_reports_drawability(self):
        source = self.directory / 'figure.png'
        parts = figure(source)
        metadata = stroke_assets.build_illustration('figure', source, 240, 300, parts=parts, seconds=8)
        self.assertEqual(metadata['engine'], 'draw-v3')
        self.assertEqual(set(metadata['drawability']), DRAWABILITY)
        self.assertTrue(metadata['drawability']['lineart_gate']['passed'])
        self.assertLessEqual(metadata['drawability']['minimum_seconds'], metadata['drawability']['required_seconds'])
        layers = illustration_layers.load(self.directory, 'figure')
        final = np.array(Image.open(self.directory / 'figure.png'))
        np.testing.assert_array_equal(illustration_layers.compose(layers, 1.), final)
        self.assertEqual(int(illustration_layers.compose(layers, 0.)[:, :, 3].max()), 0)
        middle = illustration_layers.compose(layers, .5)[:, :, 3]
        self.assertTrue(0 < np.count_nonzero(middle) < np.count_nonzero(final[:, :, 3]))
        for part in metadata['parts']:
            self.assertTrue(all(0 <= part[key] <= 1 for key in ('outline_start', 'outline_end', 'fill_start', 'fill_end')))

    def test_finished_picture_keeps_the_source_ink(self):
        # The pen times the reveal; the pixels are the source's own. Re-stamped tapered strokes broke clean outlines,
        # and plain Zhang-Suen thinning ate free 45-degree line ends back to their junction, so they were never drawn.
        large = Image.new('RGB', (560, 640), (251, 249, 241))
        draw = ImageDraw.Draw(large)
        draw.rounded_rectangle((60, 60, 500, 300), radius=40, outline='#151515', width=13)
        draw.line((60, 180, 500, 180), fill='#151515', width=13)
        draw.polygon([(150, 300), (410, 300), (460, 600), (100, 600)], outline='#151515', width=13)
        draw.line((150, 520, 230, 440, 280, 490, 350, 400, 420, 520), fill='#151515', width=11, joint='curve')
        source = self.directory / 'printer-source.png'
        large.resize((280, 320), Image.Resampling.LANCZOS).save(source)
        halves = [{'id': 'body', 'polygon': [[0, 0], [1, 0], [1, .5], [0, .5]]}, {'id': 'paper', 'polygon': [[0, .5], [1, .5], [1, 1], [0, 1]]}]
        # The even technical pen of 2.x (drawing.texture false): the finished picture is the source.
        stroke_assets.build_illustration('printer', source, 280, 320, parts=halves, seconds=8, drawing={'texture': False})
        expected = np.array(stroke_assets.prepare_artwork(source, 280, 320)['artwork']).astype(int)
        final = np.array(Image.open(self.directory / 'printer.png')).astype(int)
        on_paper = lambda pixels: pixels[:, :, :3] * pixels[:, :, 3:] / 255 + 250 * (1 - pixels[:, :, 3:] / 255)
        wrong = np.abs(on_paper(final) - on_paper(expected)).max(axis=2) > 40
        self.assertLess(wrong.sum(), .005 * (expected[:, :, 3] > 0).sum())
        # Pressure ink (the default) only reshapes the source's own ink: free ends taper, the body thins a little, the ink
        # is a little lighter in places. It never adds ink, the line's core stays where the source has it (the line stays
        # registered), and the picture moves by a small mean colour difference (CIE76; measured ~5.5 on these bare lines).
        stroke_assets.build_illustration('pressed', source, 280, 320, parts=halves, seconds=8)
        pressed = np.array(Image.open(self.directory / 'pressed.png')).astype(int)
        ink = expected[:, :, 3] > 0
        self.assertFalse((pressed[:, :, 3] > final[:, :, 3] + 2).any())
        from scipy import ndimage
        core = ndimage.distance_transform_edt(expected[:, :, 3] > 127) >= 2
        self.assertGreater(np.count_nonzero((pressed[:, :, 3] >= 128) & core) / core.sum(), .99)
        self.assertLess(float(illustration_layers.delta_e(pressed, expected)[ink].mean()), 8.)

    def test_human_likeness_metrics_on_clean_drawing(self):
        source = self.directory / 'figure.png'
        stroke_assets.build_illustration('figure', source, 240, 300, parts=figure(source), seconds=10)
        outline = stroke_assets.prepare_artwork(source, 240, 300)['outline']
        # Sample the path at a fixed density (10 s at 15 fps): per-frame thresholds would otherwise vary with the pace.
        report = engine_metrics.measure(self.directory, 'figure', outline=outline, fps=15, seconds=10)
        self.assertGreater(report['median_length_px'], 60)
        self.assertLess(report['under_15px_length'], .02)
        self.assertEqual(report['cut_by_parts'], 0)
        # The right-hander direction prior (top->bottom, left->right) costs some travel versus pure nearest-neighbour.
        self.assertLess(report['travel_over_nearest'], 6)
        self.assertLess(report['tip']['outline']['upward_fraction'], .3)
        # Flat source colors: a marker shows the final color at once, but grainy first passes must still build up.
        self.assertLess(report['first_visible_equals_final'], .9)
        self.assertGreater(report['modified_more_than_once'], .1)
        self.assertGreater(report['line_edge_f1_240'], .85)
        self.assertLess(report['tip']['outline']['jump_over_60px'], .05)

    def test_texture_line_art_must_be_simplified_unless_deliberately_excepted(self):
        source = self.directory / 'texture.png'
        texture(source)
        whole = [{'id': 'disc', 'polygon': [[0, 0], [1, 0], [1, 1], [0, 1]]}]
        with self.assertRaisesRegex(ValueError, 'Regenerate or simplify'):
            stroke_assets.build_illustration('texture', source, 300, 300, parts=whole, seconds=6)
        metadata = stroke_assets.build_illustration('texture', source, 300, 300, parts=whole, seconds=6, lineart_exception='deliberate bark texture')
        self.assertFalse(metadata['drawability']['lineart_gate']['passed'])
        outline = stroke_assets.prepare_artwork(source, 300, 300)['outline']
        self.assertFalse(lineart_gate.evaluate(outline)['passed'])

    def test_cream_paper_is_background_not_a_region_to_colour(self):
        # Style recipes ask for cream / warm-white paper; only the drawing may be inked and coloured, never the paper.
        transparent, cream = self.directory / 'figure.png', self.directory / 'cream.png'
        parts = figure(transparent)
        paper = Image.new('RGBA', (280, 340), (252, 249, 240, 255))
        paper.alpha_composite(Image.open(transparent), (20, 20))
        paper.save(cream)
        art = stroke_assets.prepare_artwork(cream, 280, 340)
        self.assertEqual(art['alpha'][5, 5], 0)
        self.assertEqual(art['alpha'][170, 90], 255)
        expected = stroke_assets.build_illustration('plain', transparent, 240, 300, parts=parts, seconds=10)['drawability']['fill_area_px']
        crop = [20, 20, 260, 320]
        measured = stroke_assets.build_illustration('cream', cream, 240, 300, crop=crop, parts=parts, seconds=10)['drawability']['fill_area_px']
        self.assertLess(abs(measured - expected), .05 * expected)

    def test_reveal_mode_paints_the_source_part_by_part_without_lines(self):
        # Styles without ink outlines: the line-art gate does not apply and the brush shows the finished picture.
        source = self.directory / 'texture.png'
        texture(source)
        halves = [{'id': 'top', 'polygon': [[0, 0], [1, 0], [1, .5], [0, .5]]}, {'id': 'bottom', 'polygon': [[0, .5], [1, .5], [1, 1], [0, 1]]}]
        with self.assertRaisesRegex(ValueError, 'drawing.mode'):
            stroke_assets.build_illustration('texture', source, 300, 300, parts=halves, seconds=3, drawing={'mode': 'wipe'})
        metadata = stroke_assets.build_illustration('texture', source, 300, 300, parts=halves, seconds=3, drawing={'mode': 'reveal'})
        self.assertEqual(metadata['mode'], 'reveal')
        self.assertEqual(set(metadata['drawability']), DRAWABILITY)
        self.assertEqual(metadata['drawability']['stroke_count'], 0)
        self.assertLessEqual(metadata['drawability']['minimum_seconds'], 3)
        self.assertFalse(any(segment['kind'] == 'ink' for segment in metadata['segments']))
        top, bottom = metadata['parts']
        self.assertLessEqual(top['fill_end'], bottom['fill_start'] + 1e-6)
        layers = illustration_layers.load(self.directory, 'texture')
        final = np.array(Image.open(self.directory / 'texture.png'))
        np.testing.assert_array_equal(illustration_layers.compose(layers, 1.), final)
        np.testing.assert_array_equal(final[:, :, :3][final[:, :, 3] == 255], np.array(Image.open(source))[:, :, :3][final[:, :, 3] == 255])
        middle = illustration_layers.compose(layers, top['fill_end'])[:, :, 3]
        self.assertGreater(np.count_nonzero(middle[:140]), .8 * np.count_nonzero(final[:140, :, 3]))
        self.assertEqual(np.count_nonzero(middle[160:]), 0)

    def test_region_marks_assign_whole_regions_to_parts(self):
        source = self.directory / 'figure.png'
        figure(source)
        record = region_marks.build_region_marks(source, 240, 300, self.directory / 'regions' / 'figure')
        self.assertTrue(Path(record['overlay']).is_file())
        labels = np.load(record['region_map'])['labels']
        regions = {region['id']: region for region in record['regions']}
        self.assertEqual(set(np.unique(labels)) - {0}, set(regions))
        head = [identity for identity, region in regions.items() if region['anchor'][1] < 115]
        rest = [identity for identity in regions if identity not in head]
        metadata = stroke_assets.build_illustration('marked', source, 240, 300, seconds=6, region_map_path=record['region_map'],
                                                    part_regions=[{'id': 'head', 'regions': head}, {'id': 'body', 'regions': rest}])
        self.assertEqual([part['id'] for part in metadata['parts']], ['head', 'body'])
        self.assertEqual(metadata['ownership'], 'regions')
        with self.assertRaisesRegex(ValueError, 'not assigned to any part: ' + str(rest[0])):
            stroke_assets.build_illustration('partial', source, 240, 300, seconds=6, region_map_path=record['region_map'],
                                             part_regions=[{'id': 'head', 'regions': head}, {'id': 'body', 'regions': rest[1:]}])
        other = self.directory / 'other.png'
        texture(other)
        with self.assertRaisesRegex(ValueError, 'another image'):
            stroke_assets.build_illustration('stale', other, 240, 300, seconds=6, region_map_path=record['region_map'],
                                             part_regions=[{'id': 'all', 'regions': list(regions)}], lineart_exception='test')
        self.assertEqual(json.loads((self.directory / 'regions' / 'figure.json').read_text())['source_sha256'],
                         str(np.load(record['region_map'])['source_sha256']))

    def test_drawing_speed_is_bounded(self):
        source = self.directory / 'figure.png'
        parts = figure(source)
        with self.assertRaisesRegex(ValueError, 'drawing.speed'):
            stroke_assets.build_illustration('fast', source, 240, 300, parts=parts, seconds=6, drawing={'speed': 10})
        slow = stroke_assets.build_illustration('slow', source, 240, 300, parts=parts, seconds=6)
        fast = stroke_assets.build_illustration('fast', source, 240, 300, parts=parts, seconds=6, drawing={'speed': 2})
        self.assertLess(fast['drawability']['required_seconds'], slow['drawability']['required_seconds'])


if __name__ == '__main__':
    unittest.main()
