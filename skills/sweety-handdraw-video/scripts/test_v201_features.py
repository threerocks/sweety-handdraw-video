"""2.0.1: parallax and dolly zoom, pressure-sensitive pen lines, the asset cache and plan-size drawing, the image gate,
part grids and box parts, pen gestures and sound effects, storyboard lint of motion layers, titles kept whole by the
auto camera, and the physics, 3D and formula recipes."""
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import line_drawing
import stroke_assets
from timeline_contract import validate_page_motion
from video_renderer import Renderer

PAPER = (251, 249, 241, 255)


def ink_box(frame):
    """Bounding box (left, top, right, bottom) of the dark pixels of a frame."""
    dark = np.asarray(frame.convert('L')) < 100
    ys, xs = np.nonzero(dark)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


class ParallaxTests(unittest.TestCase):
    def renderer(self, camera, events):
        renderer = object.__new__(Renderer)
        renderer.width, renderer.height, renderer.fps, renderer.paper = 400, 600, 30, PAPER
        square = Image.new('RGBA', (40, 40), (30, 30, 30, 255))
        renderer.artwork = {name: square for name in 'abc'}
        renderer.timeline = {'pages': [{'start': 0., 'end': 10., 'beats': [], 'camera': camera, 'events': events}],
                             'assets': {name: {'width': 40, 'height': 40, 'segments': []} for name in 'abc'}}
        return renderer

    @staticmethod
    def event(asset, x, y, **extra):
        return {'asset': asset, 'x': x, 'y': y, 'z': 30, 'start': 0., 'end': .1, 'hide': 10., 'caption': False, **extra}

    def test_near_layers_travel_further_than_the_page_and_far_layers_less(self):
        events = [self.event('a', 180, 280), self.event('b', 180, 100, depth=.8), self.event('c', 180, 460, depth=-.5)]
        renderer = self.renderer([{'at': 0, 'zoom': 1.5}, {'at': 1, 'zoom': 1.5, 'center': [260, 300], 'seconds': .5}], events)
        board = renderer.page_frame(0, 3., include_hand=False, include_captions=False)
        flat, near, far = (ink_box(board.crop((0, top, 400, top + 160)))[0] for top in (200, 40, 400))
        self.assertEqual(flat, 180)                         # the page itself does not move on the board
        self.assertAlmostEqual(near, 180 - .8 * 60, delta=1)  # the camera moved 60 px right: a near layer slides back
        self.assertAlmostEqual(far, 180 + .5 * 60, delta=1)   # a far layer follows the camera part of the way
        still = renderer.page_frame(0, .5, include_hand=False, include_captions=False)
        self.assertEqual(ink_box(still.crop((0, 40, 400, 200)))[0], 180)  # no camera travel, no parallax

    def test_a_far_layer_is_drawn_under_the_page_ink(self):
        events = [self.event('a', 180, 280), self.event('b', 180, 280, depth=-1)]
        renderer = self.renderer([{'at': 0, 'zoom': 1.5}, {'at': 1, 'zoom': 1.5, 'center': [260, 300], 'seconds': .5}], events)
        renderer.artwork = {'a': Image.new('RGBA', (40, 40), (30, 30, 30, 255)), 'b': Image.new('RGBA', (40, 40), (200, 30, 30, 255))}
        board = renderer.page_frame(0, 3., include_hand=False, include_captions=False)
        self.assertEqual(board.getpixel((200, 300))[:3], (30, 30, 30))   # the page square covers the far one where they overlap

    def test_dolly_zoom_keeps_the_subject_size_while_the_page_zooms(self):
        events = [self.event('a', 60, 60), self.event('b', 180, 280, visual='hero')]
        camera = [{'at': 0, 'zoom': 1.}, {'at': 1, 'move': 'dolly_zoom', 'zoom': 2., 'center': [200, 300], 'subject': 'hero'}]
        page = {'start': 0., 'end': 10., 'beats': [], 'camera': camera, 'events': events}
        self.assertTrue(validate_page_motion(page, {'width': 400, 'height': 600}))
        with self.assertRaises(ValueError):
            validate_page_motion({**page, 'camera': [camera[0], {**camera[1], 'subject': 'nobody'}]}, {'width': 400, 'height': 600})
        with self.assertRaises(ValueError):
            validate_page_motion({**page, 'camera': [camera[0], {'at': 1, 'move': 'dolly_zoom', 'zoom': 2.}]}, {'width': 400, 'height': 600})
        renderer = self.renderer(camera, events)
        _, zoom, _, subject, factor = renderer.camera_track(0, 1 + 1.2)
        self.assertEqual(subject, 'hero')
        self.assertAlmostEqual(zoom * factor, 1., places=6)    # screen size = board size x zoom stays the same
        board = renderer.page_frame(0, 5., include_hand=False, include_captions=False)
        left, top, right, bottom = ink_box(board.crop((100, 200, 300, 400)))
        self.assertAlmostEqual(right - left, 20, delta=1)       # half size on the board, so the same size on screen at 2x
        self.assertAlmostEqual((left + right) / 2 + 100, 200, delta=1)
        self.assertEqual(ink_box(board.crop((0, 0, 160, 160)))[2] - ink_box(board.crop((0, 0, 160, 160)))[0], 40)


class PressureLineTests(unittest.TestCase):
    def test_a_pressed_line_lands_swells_and_lifts_off(self):
        path = stroke_assets.line_points([[0, 0], [400, 0]])
        widths = stroke_assets.pressure_widths(path, 6.)
        self.assertLess(widths[0], 4.)
        self.assertLess(widths[-1], 3.2)
        self.assertAlmostEqual(float(np.median(widths)), 6., delta=.4)
        self.assertTrue(np.allclose(stroke_assets.pressure_widths(path, 6., 0.), 6.))

    def test_lines_and_arrows_keep_their_box_and_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            stroke_assets.configure_assets(directory, '/dev/null', 0, 30)
            even = stroke_assets.build_line('even', [[20, 40], [300, 90]], 6, head='end', pressure=0.)
            pressed = stroke_assets.build_line('pressed', [[20, 40], [300, 90]], 6, head='end')
            self.assertEqual((even['width'], even['height'], even['offset']), (pressed['width'], pressed['height'], pressed['offset']))
            alpha = np.asarray(Image.open(Path(directory) / 'pressed.png'))[..., 3]
            even_alpha = np.asarray(Image.open(Path(directory) / 'even.png'))[..., 3]
            self.assertLess(alpha.sum(), even_alpha.sum())        # tapered ends
            self.assertGreater(alpha.sum(), .75 * even_alpha.sum())
            reveal = np.load(Path(directory) / 'pressed.npz')['reveal']
            self.assertTrue(np.isfinite(reveal[alpha > 0]).all())



class AssetCacheTests(unittest.TestCase):
    def test_moving_an_element_reuses_its_build_but_changing_it_does_not(self):
        from test_fonts import fake_font
        from video_job import FilmBuilder
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            font = fake_font(root / 'font.ttf', '中文测试。')
            Image.new('RGB', (640, 360), 'white').save(root / 'cover.png')
            manifest = {'version': 1, 'runtime_fixture': True, 'title': '测试', 'canvas': {'width': 640, 'height': 360, 'fps': 24},
                        'cover': {'source': 'cover.png', 'seconds': 1}, 'font': {'path': font}, 'captions': {'font': {'path': font}},
                        'pages': [{'beats': [{'id': 'a', 'text': '测试。', 'audio': 'a.wav', 'actions': []}]}]}
            (root / 'job.json').write_text(json.dumps(manifest, ensure_ascii=False))
            builder = FilmBuilder(root)
            line = {'type': 'line', 'points': [[20, 30], [300, 120]], 'line_width': 5}
            first = builder.create_asset({**line, 'z': 30}, .6)
            second = builder.create_asset({**line, 'z': 50, 'visual': 'v01'}, .6)   # placement only: same pixels
            self.assertEqual(getattr(builder, 'cache_hits', 0), 1)
            self.assertNotEqual(first['asset'], second['asset'])
            for event in (first, second):
                self.assertTrue((root / 'work/assets' / f'{event["asset"]}.png').is_file())
            self.assertEqual((root / 'work/assets' / f'{first["asset"]}.png').read_bytes(), (root / 'work/assets' / f'{second["asset"]}.png').read_bytes())
            builder.create_asset({**line, 'line_width': 8}, .6)   # a different line is built afresh
            self.assertEqual(builder.cache_hits, 1)


class PlanSizeTests(unittest.TestCase):
    def test_big_pictures_are_planned_at_a_capped_size_and_rendered_up(self):
        self.assertEqual(line_drawing.plan_box(500, 400), (500, 400, 1.))
        width, height, enlarge = line_drawing.plan_box(1000, 1440)
        self.assertEqual(max(width, height), line_drawing.PLAN_LONG_SIDE)
        self.assertAlmostEqual(enlarge, 1440 / 720)
        self.assertEqual(line_drawing.plan_box(1000, 1440, {'plan_side': 1000})[1], 1000)
        with self.assertRaises(ValueError):
            line_drawing.plan_box(1000, 1440, {'plan_side': 100})


def disc(size=200, hole=False, glow=False, gap=False):
    image = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    if glow:  # a soft halo around the subject
        draw.ellipse((2, 2, size - 2, size - 2), fill=(250, 240, 180, 90))
    draw.ellipse((20, 20, size - 20, size - 20), fill=(60, 120, 200, 255), outline=(20, 20, 20, 255), width=5)
    if hole:  # a fill painted half-transparent, like the light rays of the rejected seabed
        draw.ellipse((70, 70, 130, 130), fill=(240, 240, 230, 110))
    if gap:   # a fully clear gap enclosed by the subject: background showing through, not a defect
        draw.ellipse((80, 80, 120, 120), fill=(0, 0, 0, 0))
    return image


class ImageGateTests(unittest.TestCase):
    def test_see_through_holes_and_glows_are_caught_before_the_build(self):
        import image_gate
        problems, measures = image_gate.transparency(disc())
        self.assertEqual(problems, [])
        self.assertTrue(measures['transparent_background'])
        self.assertEqual([kind for kind, _ in image_gate.transparency(disc(hole=True))[0]], ['transparency_hole'])
        self.assertEqual(image_gate.transparency(disc(gap=True))[0], [])   # a see-through gap is not a hole
        self.assertEqual([kind for kind, _ in image_gate.transparency(disc(glow=True))[0]], ['glow'])
        self.assertEqual(image_gate.transparency(Image.new('RGB', (50, 50), 'white'))[1], {'transparent_background': False})

    def test_the_gate_reports_strokes_and_a_prompt_fix(self):
        import image_gate
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'hole.png'
            disc(hole=True).save(path)
            report = image_gate.check(directory, path, width=400)
            self.assertEqual(report['status'], 'fail')
            self.assertIn('opaque', report['prompt_fix'])
            self.assertGreater(report.get('strokes', 1), 0)


class PartTests(unittest.TestCase):
    def test_parts_may_be_boxes(self):
        self.assertEqual(stroke_assets.box_polygon([.1, .2, .5, .6]), [[.1, .2], [.5, .2], [.5, .6], [.1, .6]])
        for bad in ([.5, .2, .1, .6], [0, 0, 1], [0, 0, 1.2, 1]):
            with self.assertRaises(ValueError):
                stroke_assets.box_polygon(bad)
        artwork = Image.new('RGBA', (100, 100))
        alpha = np.zeros((100, 100), np.uint8)
        alpha[10:90, 10:90] = 255
        owners = stroke_assets.polygon_owners([{'id': 'top', 'box': [0, 0, 1, .5], 'exclude_boxes': [[0, 0, .3, .5]]},
                                               {'id': 'rest', 'box': [0, 0, 1, 1]}], artwork, alpha)[1]
        self.assertEqual(owners[20, 60], 0)
        self.assertEqual(owners[20, 20], 1)    # excluded from the first part, so the next part takes it
        self.assertEqual(owners[80, 60], 1)

    def test_part_grid_finds_objects_and_draws_a_labelled_sheet(self):
        import part_grid
        image = Image.new('RGBA', (300, 200), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.ellipse((10, 10, 110, 110), fill=(200, 80, 60, 255), outline=(20, 20, 20, 255), width=4)
        draw.rectangle((170, 60, 280, 180), fill=(60, 140, 90, 255), outline=(20, 20, 20, 255), width=4)
        found = part_grid.objects(image)
        self.assertEqual(len(found), 2)
        self.assertEqual(found[0]['object'], 1)
        left, top, right, bottom = found[1]['box']
        self.assertAlmostEqual(left, 170 / 300, delta=.02)
        self.assertAlmostEqual(bottom, 181 / 200, delta=.02)
        sheet = part_grid.draw(image, found)
        self.assertGreater(sheet.width, 300)


class GestureAndLintTests(unittest.TestCase):
    def test_overlays_get_a_pen_gesture_that_fits_them(self):
        from video_job import gesture_path, gesture_seconds
        event = {'x': 100, 'y': 200, 'width': 400, 'height': 200}
        writing = np.array(gesture_path({'recipe': 'formula', 'params': {'tex': 'E = mc^2'}}, event))
        self.assertEqual((writing[0, 0], writing[-1, 0]), (140, 460))         # left to right across the line
        self.assertLess(np.ptp(writing[:, 1]), .2 * 200)                         # small up-and-downs, glyph by glyph
        self.assertAlmostEqual(gesture_seconds({'recipe': 'formula', 'params': {'tex': 'E = mc^2'}}, 3.), 5.5 * .24, places=3)
        self.assertEqual(gesture_seconds({'recipe': 'kinetic_title'}, 3.), 1.3)
        orbit = np.array(gesture_path({'recipe': 'spin3d', 'params': {}}, event))
        self.assertGreater(np.ptp(orbit[:, 0]), 2 * np.ptp(orbit[:, 1]))   # a flat orbit around the object
        circle = np.array(gesture_path({'recipe': 'rough_annotate', 'params': {'type': 'circle'}}, event))
        self.assertTrue(((circle[:, 0] >= 100) & (circle[:, 0] <= 500)).all())

    def test_a_tagged_motion_layer_counts_as_its_planned_visual(self):
        import storyboard_lint
        manifest = {'pages': [{'beats': [{'id': 'b1', 'text': '。'}], 'overlays': [{'recipe': 'formula', 'visual': 'v01-eq'}, {'recipe': 'burst'}]}]}
        actions = list(storyboard_lint.page_actions(manifest))
        self.assertEqual(actions, [(0, ['b1'], {'type': 'overlay', 'recipe': 'formula', 'params': {}, 'visual': 'v01-eq'})])
        self.assertIn('formula', storyboard_lint.TEXT_RECIPES & storyboard_lint.DIAGRAM_RECIPES)
        self.assertTrue({'physics', 'spin3d'} <= storyboard_lint.DIAGRAM_RECIPES)


class SoundEffectTests(unittest.TestCase):
    def test_options(self):
        import sfx
        self.assertEqual(sfx.parse_options(None)['gain_db'], 0.)
        self.assertIsNone(sfx.parse_options(False))
        self.assertIsNone(sfx.parse_options({'enabled': False}))
        self.assertFalse(sfx.parse_options({'accents': False})['accents'])
        for bad in ({'gain_db': 20}, {'volume': 1}, 'loud', {'seed': 1.5}):
            with self.assertRaises(ValueError):
                sfx.parse_options(bad)

    def test_transitions_and_accents_sound_where_they_are_drawn(self):
        import sfx
        page = lambda start, end, **extra: {'start': start, 'end': end, 'beats': [], 'events': [], 'last_drawing_end': start, **extra}
        timeline = {'canvas': {'fps': 30}, 'duration': 6.,
                    'pages': [page(0., 3., overlays=[{'start': 1., 'sound': 'chime'}]),
                              page(3., 6., transition={'type': 'page_turn', 'seconds': 1.}, camera=[{'at': 0}, {'at': 1, 'move': 'slam', 'zoom': 1.5}])]}
        # v3 default: restraint. The chapter page turn sounds; accents are off until a director asks for them.
        _, quiet_report = sfx.synthesize(timeline, sfx.parse_options(True))
        self.assertEqual({entry['sound'] for entry in quiet_report['sounds']}, {'paper_flip'})
        audio, report = sfx.synthesize(timeline, sfx.parse_options({'accents': True}))
        sounds = {(entry['sound'], round(entry['at'], 2)) for entry in report['sounds']}
        start = sfx.transition_windows(timeline)[0][0]
        self.assertIn(('paper_flip', round(start, 2)), sounds)
        self.assertIn(('chime', 1.04), sounds)
        self.assertIn(('thump', 4.18), sounds)
        self.assertEqual(len(audio), 6 * sfx.SAMPLE_RATE)
        quiet = slice(int(.2 * sfx.SAMPLE_RATE), int(.9 * sfx.SAMPLE_RATE))
        self.assertLess(np.abs(audio[quiet]).max(), 1e-4)
        self.assertLess(20 * math.log10(np.abs(audio).max()), sfx.PEAK_DBFS + 1)


class TitleFramingTests(unittest.TestCase):
    def test_a_push_in_never_slices_a_title_already_on_the_page(self):
        import direction
        width, height = 1280, 1920
        title = (60, 260, 1220, 420)
        segment = {'start': 5., 'boxes': [(300, 420, 980, 1100)]}
        center, zoom = direction.frame_for(segment['boxes'], width, height, True)
        self.assertTrue(direction.cut_through(direction.view_rect(center, zoom, width, height), title))   # the naive push slices it
        center, zoom = direction.keep_titles_whole(center, zoom, segment, [(1., title)], width, height, True)
        self.assertFalse(direction.cut_through(direction.view_rect(center, zoom, width, height), title))
        later = direction.keep_titles_whole(*direction.frame_for(segment['boxes'], width, height, True), segment, [(6., title)], width, height, True)
        self.assertEqual(later, direction.frame_for(segment['boxes'], width, height, True))   # not yet written: nothing to keep


@unittest.skipUnless(__import__('web_motion').available()[0], 'Playwright + Chromium not installed (setup.sh --web-motion)')
class NewRecipeTests(unittest.TestCase):
    def last_frame(self, result):
        return np.asarray(Image.open(Path(result['directory']) / f'{result["frames"] - 1:05d}.png'))[..., 3]

    def test_physics_3d_and_formula_render_inside_their_box(self):
        import web_motion
        cache = Path(tempfile.mkdtemp())
        cases = [('physics', {'scene': 'pour', 'items': [{'shape': 'circle', 'size': 30, 'count': 5}], 'drop_ms': 300}, 260, 300, 1.6),
                 ('physics', {'scene': 'topple', 'items': [{'shape': 'box', 'size': 40, 'count': 4}]}, 320, 200, .8),
                 ('spin3d', {'shape': 'gem'}, 200, 200, .8), ('spin3d', {'shape': 'coin'}, 200, 200, .8),
                 ('formula', {'tex': '\\frac{a}{b} = \\sqrt{c}', 'highlight': {'from': 0, 'to': 1, 'style': 'box'}}, 320, 200, 1.2)]
        for recipe, params, width, height, seconds in cases:
            result = web_motion.render(recipe, params, width, height, seconds, cache, fps=10)
            self.assertFalse(result['clipped'], recipe)
            self.assertGreater((self.last_frame(result) > 0).mean(), .01, recipe)
        pour = web_motion.render(*cases[0][:2], 260, 300, 1.6, cache, fps=10)
        self.assertEqual(self.last_frame(pour)[:40].max(), 0)    # everything poured has landed by the end

    def test_recipes_keep_their_layout_on_a_zoomed_board(self):
        import web_motion
        cache = Path(tempfile.mkdtemp())
        for recipe, params in (('spin3d', {'shape': 'gem'}), ('physics', {'scene': 'pour', 'items': [{'size': 30, 'count': 4}], 'drop_ms': 300})):
            boxes = []
            for scale in (1., 1.35):
                result = web_motion.render(recipe, params, 260, 260, 1.4, cache, fps=10, scale=scale)
                self.assertFalse(result['clipped'], (recipe, scale))
                box = Image.open(Path(result['directory']) / f'{result["frames"] - 1:05d}.png').getbbox()
                boxes.append([value / scale for value in box])
            self.assertLess(max(abs(a - b) for a, b in zip(*boxes)), 4, recipe)   # same place and size, only sharper

    def test_libraries_load_only_for_their_recipe_and_tex_errors_are_reported(self):
        import web_motion
        self.assertNotIn('vendor/mathjax-tex-svg.js', web_motion.libraries('kinetic_title'))
        self.assertIn('vendor/mathjax-tex-svg.js', web_motion.libraries('formula'))
        with self.assertRaisesRegex(ValueError, 'TeX error'):
            web_motion.render('formula', {'tex': '\\frac{a}{'}, 200, 100, .3, Path(tempfile.mkdtemp()), fps=10)


if __name__ == '__main__':
    unittest.main()
