"""v3 line-first pictures: a colour plate registered on its line plate, the plate's lines laid over the paint, the plate
as the outline mask, and the edits that must be refused (lost, moved or new lines)."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import line_first
from drawing_contract import validate_line_first


def plate(size=(240, 320)):
    """A cup and a ball as a black line plate on white."""
    image = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(image)
    draw.rectangle([40, 120, 140, 260], outline='black', width=5)
    draw.arc([120, 160, 180, 220], -90, 90, fill='black', width=5)
    draw.ellipse([150, 40, 220, 110], outline='black', width=5)
    draw.line([60, 150, 120, 150], fill='black', width=5)
    return image


def colour(lines, shift=(0, 0), keep=True, extra=False):
    """The colour edit of a plate: flat fills under the same lines, optionally shifted, missing or with new lines."""
    image = lines.copy()
    ImageDraw.floodfill(image, (90, 200), (230, 170, 40))
    ImageDraw.floodfill(image, (185, 75), (60, 110, 200))
    if not keep:
        ImageDraw.Draw(image).rectangle([36, 116, 144, 264], outline=(230, 170, 40), width=9)
    if extra:
        draw = ImageDraw.Draw(image)
        for y in range(20, 300, 12):
            draw.line([10, y, 30, y + 6], fill='black', width=3)
    array = np.roll(np.array(image), shift, (0, 1))
    return Image.fromarray(array)


class LineFirstTests(unittest.TestCase):
    def run_case(self, colour_image, line_image=None):
        directory = Path(tempfile.mkdtemp())
        (line_image or plate()).save(directory / 'line.png')
        colour_image.save(directory / 'colour.png')
        return directory, line_first.run(directory, 'line.png', 'colour.png', 'source.png')

    def test_registers_a_shifted_colour_plate_and_keeps_the_plate_lines(self):
        directory, report = self.run_case(colour(plate(), shift=(3, -4)))
        self.assertEqual(report['status'], 'pass', report['problems'])
        self.assertEqual(report['shift_px'], [-3, 4])
        self.assertGreaterEqual(report['kept_lines'], .99)
        source = np.array(Image.open(directory / 'source.png'))
        mask = np.array(Image.open(directory / 'source-outline.png'))
        self.assertEqual(source.shape[:2], mask.shape)
        # Every plate line is ink in the source, and the paint between the lines is the colour plate's.
        lines = np.array(plate().convert('L')) < 128
        self.assertTrue((mask[lines] == 255).all())
        self.assertLess(source[..., :3][lines].astype(float).mean(), 40)
        self.assertEqual(tuple(source[200, 90, :3]), (230, 170, 40))
        written = json.loads((directory / 'source.line-first.json').read_text())
        self.assertEqual(written['sha256']['source'], line_first.digest(directory / 'source.png'))
        self.assertTrue((directory / 'source.line-first.jpg').is_file())

    def test_refuses_an_edit_that_lost_lines(self):
        _, report = self.run_case(colour(plate(), keep=False))
        self.assertEqual(report['status'], 'fail')
        self.assertIn('kept', report['problems'][0])
        self.assertIn('Keep every black line', report['prompt_fix'])

    def test_refuses_an_edit_that_drew_new_lines(self):
        _, report = self.run_case(colour(plate(), extra=True))
        self.assertEqual(report['status'], 'fail')
        self.assertTrue(any('new lines' in problem for problem in report['problems']))

    def test_a_dark_fill_is_paint_not_a_new_line(self):
        image = colour(plate())
        ImageDraw.floodfill(image, (185, 75), (25, 25, 30))   # black hair: as dark as ink, but a fill
        _, report = self.run_case(image)
        self.assertEqual(report['status'], 'pass', report['problems'])
        self.assertGreater(report['dark_fill_share'], 0)

    def test_grain_in_a_dark_fill_is_not_a_new_line_but_a_line_drawn_on_it_is(self):
        image = colour(plate())
        ImageDraw.floodfill(image, (185, 75), (25, 49, 92))   # a night sky whose spread sits at the edge of NEUTRAL
        array = np.array(image).astype(int)
        sky = (array == (25, 49, 92)).all(axis=2)
        grain = np.random.default_rng(1).integers(-6, 7, array.shape)
        array[sky] = (array + grain)[sky]
        grainy = Image.fromarray(array.clip(0, 255).astype(np.uint8))
        _, report = self.run_case(grainy)
        self.assertEqual(report['status'], 'pass', report['problems'])
        draw = ImageDraw.Draw(grainy)
        for y in (60, 75, 90):
            draw.line([162, y, 208, y], fill='black', width=3)
        _, report = self.run_case(grainy)
        self.assertTrue(any('new lines' in problem for problem in report['problems']))

    def test_transparent_paper_under_a_line_becomes_white_ink_ground(self):
        rgba = np.array(colour(plate()).convert('RGBA'))
        white = (rgba[..., :3] > 245).all(2)
        rgba[white] = (0, 0, 0, 0)
        directory, report = self.run_case(Image.fromarray(rgba))
        self.assertEqual(report['status'], 'pass', report['problems'])
        source = np.array(Image.open(directory / 'source.png'))
        self.assertEqual(source[5, 5, 3], 0)                          # paper stays transparent
        self.assertEqual(source[150, 90, 3], 255)                     # a line is opaque ink

    def test_the_drawing_contract_binds_the_registered_source_and_mask(self):
        directory, report = self.run_case(colour(plate()))
        action = {'source': 'source.png', 'outline_mask': 'source-outline.png', 'line_first': 'source.line-first.json'}
        validate_line_first(directory, action, directory / 'source.png')
        with self.assertRaisesRegex(ValueError, 'needs "line_first"'):
            validate_line_first(directory, {**action, 'line_first': None}, directory / 'source.png')
        # A hand-edited source is no longer the registered one.
        Image.open(directory / 'colour.png').save(directory / 'source.png')
        with self.assertRaisesRegex(ValueError, 'not the one line_first.py registered'):
            validate_line_first(directory, action, directory / 'source.png')
        _, failed = self.run_case(colour(plate(), keep=False))
        self.assertEqual(failed['status'], 'fail')
        directory = Path(tempfile.mkdtemp())
        (directory / 'source.line-first.json').write_text(json.dumps(failed))
        with self.assertRaisesRegex(ValueError, 'failed line-first registration'):
            validate_line_first(directory, action, directory / 'source.png')

    def test_a_line_plate_is_timed_with_its_colouring_before_the_colour_edit(self):
        import image_gate
        directory = Path(tempfile.mkdtemp())
        plate().save(directory / 'line.png')
        colour(plate()).save(directory / 'colour.png')
        line_first.run(directory, 'line.png', 'colour.png', 'source.png')
        estimate = image_gate.check(directory, directory / 'line.png', width=240, plate=True)
        lines_only = image_gate.check(directory, directory / 'line.png', width=240)
        self.assertTrue(estimate['line_plate'])
        self.assertEqual(estimate['strokes'], lines_only['strokes'])
        # The enclosed areas are coloured in the estimate: it takes longer than the bare lines and about as long as the picture.
        self.assertGreater(estimate['minimum_seconds'], lines_only['minimum_seconds'])
        source, mask = line_first.plate_preview(directory / 'line.png', directory)
        preview = np.array(Image.open(source))
        self.assertTrue((preview[5, 5, :3] > 245).all())            # open background stays paper
        self.assertFalse((preview[200, 90, :3] > 245).all())        # the cup is filled

    def test_light_lines_inside_a_dark_fill_are_kept_and_stay_light(self):
        # A terminal window: the colour edit paints the body graphite and draws the plate's code lines light.
        lines = Image.new('RGB', (300, 240), 'white')
        draw = ImageDraw.Draw(lines)
        draw.rectangle([40, 40, 260, 200], outline='black', width=5)
        for y in (90, 120, 150):
            draw.line([70, y, 220, y], fill='black', width=5)
        colour_image = lines.copy()
        ImageDraw.floodfill(colour_image, (150, 70), (43, 47, 54))
        cdraw = ImageDraw.Draw(colour_image)
        for y in (90, 120, 150):
            cdraw.line([70, y, 220, y], fill=(217, 214, 207), width=7)
        directory, report = self.run_case(colour_image, lines)
        self.assertEqual(report['status'], 'pass', report['problems'])
        self.assertGreater(report['inverse_line_px'], 0)
        source = np.array(Image.open(directory / 'source.png'))
        self.assertGreater(source[120, 150, :3].astype(float).mean(), 150)   # the code line stays light, not inked black
        self.assertLess(source[105, 150, :3].astype(float).mean(), 70)       # the body stays graphite


if __name__ == '__main__':
    unittest.main()
