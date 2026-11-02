"""v3 planned drawing: the model's construction (simple shapes blocked in before any ink) rubbed out by an eraser
before colour, module outlines judged by the ink rather than a sticker halo, each module drawn from simple to complex
(outline, structure, features, details), and colouring by marker (one colour at a time, light to dark, shadows over
the base coat)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import illustration_layers
import line_drawing
import line_topology
import stroke_assets


def face(halo=True):
    """A face module on paper: a round outline (with a light sticker halo outside it), two eye dots and a hair strand
    growing off the outline."""
    image = Image.new('RGB', (240, 240), 'white')
    draw = ImageDraw.Draw(image)
    if halo:
        draw.ellipse([22, 22, 218, 218], fill=(232, 232, 232))
    draw.ellipse([30, 30, 210, 210], fill=(245, 200, 170), outline='black', width=5)
    draw.ellipse([80, 100, 94, 114], fill='black')
    draw.ellipse([146, 100, 160, 114], fill='black')
    draw.line([120, 32, 112, 70], fill='black', width=5)
    return image


class ConstructionTests(unittest.TestCase):
    def test_planned_shapes_become_light_strokes_in_picture_pixels(self):
        shapes = [{'part': 'head', 'shape': 'ellipse', 'center': [.5, .3], 'radius': [.2, .15]},
                  {'shape': 'box', 'box': [.1, .6, .4, .9]},
                  {'shape': 'line', 'points': [[.5, .45], [.55, .7], [.6, .95]]},
                  {'shape': 'polygon', 'points': [[.6, .6], [.9, .6], [.75, .9]]}]
        strokes = line_drawing.construction_strokes(shapes, 400, 600, ['body', 'head'])
        self.assertEqual(len(strokes), 4)
        self.assertEqual(strokes[0]['part'], 1)
        self.assertTrue(all(stroke['construction'] for stroke in strokes))
        ellipse = strokes[0]['points']
        # Drawn once round and a little past where it began, from the top.
        self.assertGreater(line_drawing.calculate_lengths(ellipse)[-1], 2 * np.pi * np.sqrt((80 ** 2 + 90 ** 2) / 2) * .98)
        self.assertLess(abs(ellipse[0][1] - (.3 * 599 - .15 * 599)), 4)
        # A gesture line runs a little past both ends.
        line = strokes[2]['points']
        self.assertLess(line[0][1], .45 * 599)
        self.assertGreater(line[-1][1], .95 * 599)
        with self.assertRaisesRegex(ValueError, 'ellipse, box, line or polygon'):
            line_drawing.construction_strokes([{'shape': 'star'}], 100, 100, ['a'])

    def test_a_planned_sketch_is_drawn_before_any_ink_and_fades(self):
        directory = Path(tempfile.mkdtemp())
        face().save(directory / 'face.png')
        stroke_assets.configure_assets(directory, '/dev/null', 0, 30)
        sketch = [{'shape': 'ellipse', 'center': [.5, .5], 'radius': [.38, .38]}, {'shape': 'line', 'points': [[.25, .45], [.75, .45]]}]
        meta = stroke_assets.build_illustration('face', directory / 'face.png', 240, 240, None,
                                                [{'id': 'head', 'polygon': [[0, 0], [1, 0], [1, 1], [0, 1]]}], drawing={'sketch': sketch})
        kinds = [segment['kind'] for segment in meta['segments'] if segment['kind'] in ('sketch', 'ink')]
        self.assertEqual(kinds[:2], ['sketch', 'sketch'])
        self.assertNotIn('sketch', kinds[2:])
        self.assertIn('sketch', meta['layers'])
        first_sketch = next(segment for segment in meta['segments'] if segment['kind'] == 'sketch')
        self.assertGreaterEqual((first_sketch['end'] - first_sketch['start']) * meta['seconds'] * 30, 2.99)   # never flicked in


def scene(directory):
    """Two shapes in two colours with a shadow side each, on paper: a light yellow box and a dark blue disc."""
    image = Image.new('RGB', (300, 220), 'white')
    draw = ImageDraw.Draw(image)
    draw.rectangle([20, 40, 140, 190], fill=(245, 205, 80), outline='black', width=4)
    draw.rectangle([100, 44, 136, 186], fill=(200, 150, 40))
    draw.ellipse([170, 50, 280, 160], fill=(50, 70, 150), outline='black', width=4)
    draw.pieslice([174, 54, 276, 156], 20, 160, fill=(25, 35, 90))
    draw.ellipse([170, 50, 280, 160], outline='black', width=4)
    image.save(directory / 'scene.png')
    return directory / 'scene.png'


class EraseAndMarkerTests(unittest.TestCase):
    def build(self, drawing):
        directory = Path(tempfile.mkdtemp())
        source = scene(directory)
        stroke_assets.configure_assets(directory, '/dev/null', 0, 30)
        parts = [{'id': 'box', 'box': [0, 0, .5, 1]}, {'id': 'disc', 'box': [.5, 0, 1, 1]}]
        return directory, stroke_assets.build_illustration('scene', source, 300, 220, None, parts, drawing=drawing)

    def test_the_construction_is_rubbed_out_between_the_ink_and_the_colour(self):
        sketch = [{'shape': 'box', 'box': [.05, .15, .48, .88]}, {'shape': 'ellipse', 'center': [.75, .48], 'radius': [.19, .25]}]
        directory, meta = self.build({'sketch': sketch})
        kinds = [segment['kind'] for segment in meta['segments'] if segment['kind'] not in ('travel', 'pause', 'slide')]
        last_ink, first_erase = max(k for k, kind in enumerate(kinds) if kind == 'ink'), kinds.index('erase')
        first_colour = kinds.index('color')
        self.assertLess(last_ink, first_erase)
        self.assertLess(max(k for k, kind in enumerate(kinds) if kind == 'erase'), first_colour)
        maps = np.load(directory / 'scene.npz')
        erase, sketch_time = maps['sketch_erase'], maps['sketch']
        drawn = np.isfinite(sketch_time)
        self.assertTrue(np.isfinite(erase[drawn]).all())
        self.assertTrue((erase[drawn] > sketch_time[drawn]).all())
        # Every construction pixel is gone before any colour arrives; none is left in the final frame.
        colour_start = min(segment['start'] for segment in meta['segments'] if segment['kind'] == 'color')
        self.assertLessEqual(float(erase[drawn].max()), colour_start + 1e-6)
        layers = illustration_layers.load(directory, 'scene')
        middle = float(np.median(erase[drawn]))
        frame = layers['sketch_alpha'] * np.clip((middle - layers['sketch']) * illustration_layers.SHARP, 0, 1)
        self.assertGreater(frame.sum(), 0)
        self.assertEqual(float(illustration_layers.compose(layers, np.float32(1.))[..., 3].sum()), float(layers['final'][..., 3].sum()))

    def test_markers_go_light_to_dark_and_each_shadow_follows_its_base_coat(self):
        _, meta = self.build({'colouring': 'markers'})
        fills = [segment for segment in meta['segments'] if segment['kind'] in ('color', 'shade')]
        middle = lambda segment: np.mean(segment['points'], axis=0)
        sides = ['box' if middle(segment)[0] < 150 else 'disc' for segment in fills]
        kinds = [segment['kind'] for segment in fills]
        # The light yellow box with its shadow first, then the dark blue disc with its shadow.
        steps = [step for k, step in enumerate(zip(sides, kinds)) if not k or step != (sides[k - 1], kinds[k - 1])]
        self.assertEqual(steps, [('box', 'color'), ('box', 'shade'), ('disc', 'color'), ('disc', 'shade')])
        pauses = [segment for segment in meta['segments'] if segment['kind'] == 'pause' and segment['start'] > fills[0]['start']]
        self.assertGreaterEqual(len(pauses), 3)   # a marker swap before each shadow and before the second colour

    def test_the_renderer_shows_the_eraser_while_it_rubs(self):
        from video_renderer import Renderer, eraser_sprite
        image, contact, shadow = eraser_sprite(1920)
        self.assertEqual(image.size, shadow.size)
        self.assertTrue(0 < contact[0] < image.width and 0 < contact[1] < image.height)
        renderer = object.__new__(Renderer)
        segments = [{'kind': 'ink', 'start': 0., 'end': .4}, {'kind': 'travel', 'start': .4, 'end': .5},
                    {'kind': 'erase', 'start': .5, 'end': .7}, {'kind': 'travel', 'start': .7, 'end': .8}, {'kind': 'color', 'start': .8, 'end': 1.}]
        renderer.timeline = {'assets': {'a': {'segments': segments}}, 'pages': [{'events': [{'asset': 'a', 'caption': False, 'start': 0., 'end': 10.}]}]}
        self.assertEqual([renderer.pen_kind(time) for time in (2., 4.5, 6., 7.5, 9.)], ['ink', 'erase', 'erase', 'color', 'color'])


class ModuleOrderTests(unittest.TestCase):
    def strokes(self, image):
        directory = Path(tempfile.mkdtemp())
        image.save(directory / 'face.png')
        art = stroke_assets.prepare_artwork(directory / 'face.png', 240, 240)
        strokes, clean, _, _ = line_topology.extract_strokes(art['outline'], art['support'])
        owners = np.where(art['alpha'] > 0, 0, -1).astype(np.int16)
        line_drawing.classify(strokes, owners, art['alpha'], art['pixels'], ['head'], (), clean)
        return strokes

    def test_the_outline_is_judged_by_the_ink_not_a_halo_around_it(self):
        tiers = {line_drawing.SILHOUETTE: 0}
        for stroke in self.strokes(face(halo=True)):
            tiers[stroke['tier']] = tiers.get(stroke['tier'], 0) + 1
        self.assertGreaterEqual(tiers[line_drawing.SILHOUETTE], 1)

    def test_outline_then_features_then_details(self):
        strokes = self.strokes(face())
        ordered, _ = line_drawing.order(strokes, 1, np.array([120., 0.]))
        tiers = [stroke['tier'] for stroke in ordered]
        self.assertEqual(tiers[0], line_drawing.SILHOUETTE)
        eyes = [k for k, stroke in enumerate(ordered) if stroke['tier'] == line_drawing.KEY]
        strand = [k for k, stroke in enumerate(ordered) if 100 < stroke['points'][:, 0].mean() < 130 and stroke['points'][:, 1].mean() < 75
                  and stroke['tier'] != line_drawing.SILHOUETTE]
        self.assertEqual(len(eyes), 2)
        self.assertTrue(strand and min(strand) > max(eyes))
        self.assertEqual(tiers, sorted(tiers, key=lambda tier: {0: 0, 2: 1, 1: 2, 3: 3, 4: 3}[tier]))


if __name__ == '__main__':
    unittest.main()


class CloseUpRenderTests(unittest.TestCase):
    def test_enlarged_plan_maps_follow_the_fine_edge_not_a_staircase(self):
        # The v3 film's rough edges: a 3x close-up carried the 540 px plan's maps by nearest neighbour, a 3 px staircase
        # along every colour edge. A diagonal blue/grey edge, drawn smooth at both sizes.
        def picture(size):
            image = Image.new('RGBA', (size, size), (210, 205, 195, 255))
            ImageDraw.Draw(image).polygon([(0, 0), (size, 0), (0, size * .8)], fill=(48, 110, 176, 255))
            return np.array(image)
        coarse, fine = picture(60), picture(180)
        blue = lambda image: image[..., 0] < 120
        truth = blue(fine)
        labels = blue(coarse)
        rows, columns = stroke_assets.guided_index(coarse, fine)
        guided = labels[rows, columns]
        nearest = stroke_assets.upsample(labels, truth.shape)
        self.assertLess((guided != truth).sum(), .2 * (nearest != truth).sum())
        self.assertLessEqual(np.abs(rows - np.minimum(np.arange(180) // 3, 59)[:, None]).max(), 1)   # never farther than a neighbour
