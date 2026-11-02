"""2.0: coloring that follows the drawing (snapped ownership, edge pass, wet coats), camera moves and density rules,
the transition library, automatic direction and the web motion recipes."""
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import direction
import illustration_layers
import motion
import paint_motion
import stroke_assets
import transitions
from timeline_contract import TRANSITIONS, validate_page_motion
from video_renderer import TRANSITION_EFFECTS, Renderer

CANVAS = {'width': 1080, 'height': 1920}


def square_region(size=120, part=0):
    mask = np.zeros((size, size), bool)
    mask[10:size - 10, 10:size - 10] = True
    return {'part': part, 'offset': np.array([0, 0]), 'mask': mask, 'area': int(mask.sum()), 'flat': np.array([40., 80., 160.])}


class ColoringTests(unittest.TestCase):
    def test_a_coarse_box_does_not_split_an_enclosed_area(self):
        # One blue shape inside an ink ring, cut in half by two box-shaped parts: it goes wholly to the larger share.
        size = 100
        outline = np.zeros((size, size), bool)
        outline[10, 10:90] = outline[89, 10:90] = outline[10:90, 10] = outline[10:90, 89] = True
        alpha = np.zeros((size, size), np.uint8)
        alpha[10:90, 10:90] = 255
        pixels = np.zeros((size, size, 4), np.uint8)
        pixels[..., :3] = (40, 80, 160)
        pixels[..., 3] = alpha
        owners = np.full((size, size), -1, np.int16)
        owners[10:90, 10:55] = 0
        owners[10:90, 55:90] = 1
        snapped = stroke_assets.snap_owners(owners, outline, pixels, alpha, 2)
        inside = alpha > 0
        self.assertEqual(set(np.unique(snapped[inside & ~outline])), {0})

    def test_a_thin_seam_follows_its_area_but_a_small_closed_shape_keeps_its_part(self):
        size = 120
        outline = np.zeros((size, size), bool)
        outline[5, 5:115] = outline[114, 5:115] = outline[5:115, 5] = outline[5:115, 114] = True
        outline[50, 50:58] = outline[57, 50:58] = outline[50:58, 50] = outline[50:58, 57] = True   # a small closed button
        alpha = np.zeros((size, size), np.uint8)
        alpha[5:115, 5:115] = 255
        pixels = np.zeros((size, size, 4), np.uint8)
        pixels[..., :3] = (40, 80, 160)
        pixels[30:110, 90:92, :3] = (20, 50, 120)        # a darker seam, too light to be ink
        pixels[51:57, 51:57, :3] = (230, 120, 60)       # the button's own colour
        pixels[..., 3] = alpha
        owners = np.zeros((size, size), np.int16)
        owners[:, 88:] = 1                               # the seam lies in another part's box
        owners[51:57, 51:57] = 2
        snapped = stroke_assets.snap_owners(owners, outline, pixels, alpha, 3)
        self.assertEqual(snapped[60, 90], 0)
        self.assertEqual(snapped[53, 53], 2)

    def test_careful_marker_covers_every_pixel_with_an_edge_pass_first(self):
        region = square_region()
        paint_motion.plan([region], 1.)
        self.assertGreater(len(region['cells']), 0)
        self.assertIsNotNone(region['edge'])
        self.assertFalse(paint_motion.uncovered(region['mask'], region['offset'], region['cells'], region['radius']).any())
        # The first pass runs along the inside of the edge: its points keep about EDGE_INSET radii from the border.
        first = region['cells'][0]
        inset = region['distance'][np.clip(np.rint(first[:, 1]).astype(int), 0, 119), np.clip(np.rint(first[:, 0]).astype(int), 0, 119)]
        self.assertLess(abs(float(np.median(inset)) - paint_motion.EDGE_INSET * region['radius']), 3)

    def test_the_broadest_marker_skips_care_to_fit_short_windows(self):
        careful, quick = square_region(), square_region()
        paint_motion.plan([careful], 1.)
        paint_motion.plan([quick], paint_motion.SCALES[-1])
        self.assertIsNone(quick['edge'])
        self.assertFalse(quick['careful'])

    def test_a_fresh_coat_is_streaky_and_settles_solid_by_the_end(self):
        directory = Path(tempfile.mkdtemp())
        image = Image.new('RGBA', (240, 240))
        draw = ImageDraw.Draw(image)
        draw.rectangle((30, 30, 210, 210), fill='#3C64B4', outline='#141414', width=5)
        image.save(directory / 'box.png')
        stroke_assets.configure_assets(directory, '/System/Library/Fonts/Hiragino Sans GB.ttc', 0, 30)
        whole = [[0, 0], [1, 0], [1, 1], [0, 1]]
        stroke_assets.build_illustration('box', directory / 'box.png', 240, 240, parts=[{'id': 'box', 'polygon': whole}], seconds=8)
        layers = illustration_layers.load(directory, 'box')
        final = np.array(Image.open(directory / 'box.png'))
        np.testing.assert_array_equal(illustration_layers.compose(layers, 1.), final)
        painted = np.isfinite(layers['color']) & (layers['alpha'] > 0)
        middle = float(np.median(layers['color'][painted]))
        alpha = illustration_layers.compose(layers, np.float32(middle))[..., 3].astype(float)
        fresh = painted & (layers['color'] < middle) & (layers['full'] > middle)
        self.assertTrue(fresh.any())
        self.assertLess(float(np.median(alpha[fresh])), 245)   # wet, streaky: not yet solid
        self.assertGreater(float(np.median(alpha[fresh])), 60)


class NarrationTimingTests(unittest.TestCase):
    def test_a_syllable_the_recognizer_dropped_still_gets_its_time(self):
        from align import piece_ranges, SAMPLE_RATE
        rate = SAMPLE_RATE
        tone = lambda seconds: (.3 * np.sin(np.arange(int(seconds * rate)) * 2 * np.pi * 220 / rate)).astype(np.float32)
        samples = np.concatenate([np.zeros(int(.05 * rate), np.float32), tone(.3), np.zeros(int(.45 * rate), np.float32), tone(1.2)])
        # Word times only for the sentence after the pause: "叮" was spoken first but not recognized.
        words = [{'text': '隔着', 'start': .8, 'end': 1.3}, {'text': '大海', 'start': 1.3, 'end': 2.0}]
        ranges, _ = piece_ranges(samples, ['叮。', '隔着大海。'], words, rate)
        self.assertLess(ranges[0][0], .15)
        self.assertGreater(ranges[0][1] - ranges[0][0], .25)
        self.assertGreaterEqual(ranges[1][0], ranges[0][1] - 1e-6)


class CameraTests(unittest.TestCase):
    def renderer(self, keys):
        renderer = object.__new__(Renderer)
        renderer.width, renderer.height, renderer.fps = 1080, 1920, 30
        renderer.timeline = {'pages': [{'start': 0., 'end': 10., 'beats': [], 'camera': keys}]}
        return renderer

    def test_named_moves_fill_in_timing(self):
        self.assertEqual(motion.camera_key({'move': 'slam', 'zoom': 2})['seconds'], .18)
        self.assertEqual(motion.camera_key({'move': 'slam', 'seconds': .3})['seconds'], .3)
        page = {'start': 0, 'end': 10, 'beats': [], 'camera': [{'at': 0, 'zoom': 1}, {'at': 1, 'move': 'slam', 'zoom': 2}]}
        self.assertTrue(validate_page_motion(page, CANVAS))
        with self.assertRaises(ValueError):
            validate_page_motion({**page, 'camera': [{'at': 1, 'move': 'teleport'}]}, CANVAS)
        with self.assertRaises(ValueError):
            validate_page_motion({**page, 'camera': [{'at': 1, 'impact': 90}]}, CANVAS)

    def test_zoom_moves_in_log_space(self):
        renderer = self.renderer([{'at': 0, 'zoom': 1}, {'at': 1, 'zoom': 2.56, 'seconds': 1, 'ease': 'linear'}])
        _, zoom, _ = renderer.camera_state(0, 1.5)
        self.assertAlmostEqual(zoom, 1.6, places=3)   # halfway in time is the geometric mean

    def test_a_hop_dips_the_zoom_mid_move_and_lands_on_target(self):
        keys = [{'at': 0, 'zoom': 2, 'center': [300, 600]}, {'at': 1, 'zoom': 2, 'center': [800, 1400], 'move': 'hop', 'ease': 'linear'}]
        renderer = self.renderer(keys)
        self.assertLess(renderer.camera_state(0, 1.6)[1], 1.9)
        self.assertAlmostEqual(renderer.camera_state(0, 3)[1], 2.)

    def test_an_impact_shakes_then_settles(self):
        renderer = self.renderer([{'at': 0, 'zoom': 1.5}, {'at': 1, 'zoom': 2, 'move': 'slam'}])
        arrive = 1 + .18
        rest = renderer.camera_state(0, arrive + 2)[0]
        moved = max(np.linalg.norm(renderer.camera_state(0, arrive + t)[0] - rest) for t in (.02, .04, .06, .08))
        self.assertGreater(moved, 1.)
        self.assertLess(np.linalg.norm(renderer.camera_state(0, arrive + 1.2)[0] - rest), .05)


class DirectionTests(unittest.TestCase):
    def test_density_rules_flag_restless_camera_work(self):
        keys = [{'at': 0, 'zoom': 1.}] + [{'at': 1 + i, 'zoom': 1. + .2 * (i + 1), 'move': 'push_in'} for i in range(4)]
        warnings = direction.camera_warnings([{'start': 0., 'end': 30., 'camera': keys, 'beats': []}])
        self.assertTrue(any('push-ins in a row' in warning for warning in warnings))
        quick = [{'at': 0, 'zoom': 1.}] + [{'at': 1 + i, 'zoom': 1.2 + .1 * (i % 2), 'move': 'whip'} for i in range(8)]
        warnings = direction.camera_warnings([{'start': 0., 'end': 60., 'camera': quick, 'beats': []}])
        self.assertTrue(any('whips/slams/snaps' in warning for warning in warnings))

    def test_transitions_vary_and_chapters_get_a_signature(self):
        previous, kinds = None, []
        for index in range(1, 13):
            spec = direction.auto_transition(index, {'chapter': index == 6}, previous)
            kinds.append(spec['type'])
            self.assertNotEqual(spec['type'], previous)
            self.assertIn(spec['type'], TRANSITIONS)
            previous = spec['type']
        self.assertIn(kinds[5], ('page_turn', 'erase', 'paper_tear'))
        pages = [{'transition': {'type': 'cube'}} for _ in range(4)]
        self.assertTrue(direction.transition_warnings(pages))


class TransitionTests(unittest.TestCase):
    def test_every_effect_starts_on_the_old_page_and_ends_on_the_new(self):
        old = Image.new('RGBA', (90, 160), (230, 60, 50, 255))
        ImageDraw.Draw(old).rectangle((20, 40, 70, 120), fill=(20, 20, 20, 255))
        new = Image.new('RGBA', (90, 160), (40, 120, 200, 255))

        class Paper:
            paper = (251, 249, 241, 255)
        for kind in TRANSITION_EFFECTS:
            self.assertIn(kind, TRANSITIONS)
            for extra in ({}, {'direction': 'up'}, {'style': 'rubber'} if kind == 'erase' else {'style': 'blot'} if kind == 'iris' else {}):
                window = {'type': kind, **extra}
                frames = [getattr(transitions, kind)(Paper(), old, new, fraction, window) for fraction in (0., .5, 1.)]
                for frame in frames:
                    self.assertEqual(frame.size, old.size, kind)
                end = np.asarray(frames[-1].convert('RGB'), float)
                self.assertLess(np.abs(end - np.asarray(new.convert('RGB'), float)).mean(), 8, (kind, extra))


@unittest.skipUnless(__import__('web_motion').available()[0], 'Playwright + Chromium not installed (setup.sh --web-motion)')
class WebMotionTests(unittest.TestCase):
    def test_new_recipes_render(self):
        import web_motion
        cache = Path(tempfile.mkdtemp())
        cases = [('kinetic_title', {'text': '海底光缆', 'keywords': ['光缆'], 'style': style}) for style in ('slam', 'scramble', 'word_by_word')]
        cases += [('morph', {'shapes': [{'shape': 'dot', 'box': [10, 10, 30, 30]}, {'shape': 'star', 'box': [60, 20, 140, 100]}], 'hold_ms': 50}),
                  ('rough_chart', {'kind': 'bar', 'values': [1, 3]}), ('rough_chart', {'kind': 'tally', 'count': 7}),
                  ('map_route', {'route': [[121.5, 31.2], [-122.4, 37.8]], 'stops': [{'at': [121.5, 31.2], 'label': '上海'}]})]
        for recipe, params in cases:
            result = web_motion.render(recipe, params, 200, 160, .4, cache, fps=10)
            last = Image.open(Path(result['directory']) / f'{result["frames"] - 1:05d}.png')
            self.assertEqual(last.size, (200, 160))
            self.assertGreater(np.asarray(last)[..., 3].max(), 0, recipe)

    def test_titles_shrink_to_fit_and_cut_off_content_is_reported(self):
        import web_motion
        cache = Path(tempfile.mkdtemp())
        # The 2.0 showcase bug: a title too wide for one line wrapped and its second line was cut off by the box.
        title = web_motion.render('kinetic_title', {'text': '一条消息的环球旅行', 'keywords': ['环球旅行'], 'style': 'slam', 'size': 104},
                                  950, 170, .6, cache, fps=10)
        self.assertFalse(title['clipped'])
        alpha = np.asarray(Image.open(Path(title['directory']) / f'{title["frames"] - 1:05d}.png'))[..., 3]
        rows = np.nonzero(alpha.max(1) > 40)[0]
        self.assertGreater(rows[-1] - rows[0], 50)   # still a big title, not shrunk to nothing
        edge = web_motion.render('sketch_shapes', {'shapes': [{'kind': 'rect', 'box': [0, 0, 200, 100]}]}, 200, 100, .3, cache, fps=10)
        self.assertTrue(edge['clipped'])

    def test_a_long_line_settles_inside_its_own_time(self):
        import web_motion
        cache = Path(tempfile.mkdtemp())
        # The v3 film bug: 10 words at 70 ms stagger + 900 ms need 1.5 s; in a 0.9 s layer the last words froze mid-blur.
        for style in ('blur_up', 'mask_up', 'typewriter'):
            line = web_motion.render('kinetic_title', {'text': '换心态 · 给眼睛和手 · 换合作方式 · 只审核心', 'style': style, 'size': 46},
                                     1126, 74, .9, cache, fps=30)
            frames = [np.asarray(Image.open(Path(line['directory']) / f'{index:05d}.png'))[..., 3].astype(int) for index in (line['frames'] - 2, line['frames'] - 1)]
            self.assertEqual(np.abs(frames[1] - frames[0]).sum(), 0, style)   # nothing still moving when the layer is held
            right = frames[1][:, 900:]
            self.assertGreater((right > 200).sum(), 200, style)               # the last word is solid, not a blur


if __name__ == '__main__':
    unittest.main()
