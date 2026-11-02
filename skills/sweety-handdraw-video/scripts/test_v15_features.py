"""v1.5: clean ink extraction, size-independent drawing time, charts/arrows/counters, emphasis, board scale,
transitions, layout zones, viewing metrics and cue onset snapping."""
import math
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import stroke_assets


class InkExtractionTests(unittest.TestCase):
    def test_dark_fill_edge_is_paint_but_its_darker_outline_is_ink(self):
        # A navy hoodie with a near-black outline: only the outline is a line; the navy rim stays paint.
        pixels = np.full((120, 120, 4), 255, np.uint8)
        image = Image.fromarray(pixels)
        ImageDraw.Draw(image).ellipse((10, 10, 110, 110), fill=(45, 60, 90, 255), outline=(15, 15, 20, 255), width=3)
        pixels = np.array(image)
        maximum = pixels[:, :, :3].max(axis=2)
        dark = (maximum < 125) & (pixels[:, :, 3] > 30)
        lines = stroke_assets.ink_lines(pixels, dark)
        luminance = pixels[:, :, :3].astype(float) @ [.299, .587, .114]
        self.assertTrue(lines[luminance < 30].mean() > .9)
        navy_rim = (luminance > 50) & dark & ~lines
        self.assertGreater(navy_rim.sum(), 500)

    def test_thin_line_art_without_fill_is_unchanged(self):
        image = Image.new('RGBA', (100, 60), (255, 255, 255, 255))
        ImageDraw.Draw(image).line((8, 45, 50, 10, 90, 45), fill=(20, 20, 20, 255), width=3)
        pixels = np.array(image)
        dark = (pixels[:, :, :3].max(axis=2) < 125)
        self.assertTrue(np.array_equal(stroke_assets.ink_lines(pixels, dark), dark & (np.zeros(dark.shape) == 0)))

    def test_loose_specks_are_dropped_but_line_edges_stay(self):
        alpha = np.zeros((40, 40), np.float32)
        alpha[20, 5:35] = 1.
        alpha[19, 10] = .6        # ragged edge touching the line
        alpha[5, 5] = alpha[5, 6] = .8   # residue far from any line
        cleaned = stroke_assets.drop_loose_specks(alpha)
        self.assertEqual(cleaned[5, 5], 0)
        self.assertEqual(cleaned[19, 10], np.float32(.6))
        self.assertTrue((cleaned[20, 5:35] == 1).all())


class DrawingTimeTests(unittest.TestCase):
    def test_picture_size_scales_motion_and_is_bounded(self):
        import line_drawing
        self.assertEqual(line_drawing.picture_size(300, 300), 1.)
        self.assertAlmostEqual(line_drawing.picture_size(800, 800), 2.0 if line_drawing.MAX_SIZE_SCALE >= 2 else line_drawing.MAX_SIZE_SCALE)
        self.assertEqual(line_drawing.frame_floor(30, 1.), 2 / 30)
        self.assertEqual(line_drawing.frame_floor(30, 3.), 1 / 30)

    def test_a_larger_picture_is_not_proportionally_slower(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = Image.new('RGBA', (400, 400))
            draw = ImageDraw.Draw(image)
            for index in range(6):
                draw.ellipse((20 + index * 55, 40, 70 + index * 55, 90), fill='#e8a33d', outline='#202020', width=4)
                draw.rectangle((20 + index * 55, 200, 70 + index * 55, 330), fill='#6ba3c9', outline='#202020', width=4)
            image.save(root / 'row.png')
            stroke_assets.configure_assets(root, '/dev/null', 0, 30)
            whole = [{'id': 'all', 'polygon': [[0, 0], [1, 0], [1, 1], [0, 1]]}]
            # Above the 400 px calibration size, 1.8x the width costs far less than 1.8x the time (draw-v2: about 2x).
            small = stroke_assets.build_illustration('small', root / 'row.png', 400, 400, parts=whole)
            large = stroke_assets.build_illustration('large', root / 'row.png', 720, 720, parts=whole)
            ratio = large['drawability']['minimum_seconds'] / small['drawability']['minimum_seconds']
            self.assertLess(ratio, 1.45)
            self.assertEqual(large['engine'], 'draw-v3')
            self.assertGreater(large['picture_scale'], 1.)


class NewActionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        from fonts import select
        path, index = select('handwriting', '0123456789.万+', None)
        stroke_assets.configure_assets(self.root, path, index, 30)
        self.font = (path, index)

    def tearDown(self):
        self.directory.cleanup()

    def test_arrow_is_a_shaft_and_two_barbs_drawn_by_the_pen(self):
        metadata = stroke_assets.build_line('arrow', np.array([[100., 300.], [400., 120.]]), 6, '#d8452e', 'end', .2)
        ink = [segment for segment in metadata['segments'] if segment['kind'] == 'ink']
        self.assertEqual(len(ink), 3)
        self.assertTrue(all(segment['radius'] >= 3 for segment in ink))
        reveal = np.load(self.root / 'arrow.npz')['reveal']
        alpha = np.array(Image.open(self.root / 'arrow.png'))[:, :, 3]
        self.assertTrue(np.isfinite(reveal[alpha > 0]).all())
        self.assertEqual(len(metadata['offset']), 2)

    def test_bar_marker_fills_its_box_from_the_base_up(self):
        stroke_assets.build_brush('bar', 60, 200, '#e9784e', 'bar')
        alpha = np.array(Image.open(self.root / 'bar.png'))[:, :, 3]
        reveal = np.load(self.root / 'bar.npz')['reveal']
        self.assertTrue((alpha == 255).all())
        self.assertLess(reveal[-5:].mean(), reveal[:5].mean())

    def test_bar_chart_expands_to_axes_bars_values_and_labels(self):
        from video_job import expand_actions
        actions = expand_actions([{'type': 'bar_chart', 'x': 100, 'y': 300, 'width': 800, 'height': 600, 'values': [3, 7.5, 12],
                                   'labels': ['一月', '二月', '三月'], 'unit': '万', 'visual': 'v9', 'from_beat': 'b2'}])
        kinds = [action['type'] for action in actions]
        self.assertEqual(kinds.count('line'), 1)
        self.assertEqual(kinds.count('marker'), 3)
        self.assertEqual(kinds.count('text'), 6)
        self.assertEqual(actions[0].get('from_beat'), 'b2')
        bars = [action for action in actions if action['type'] == 'marker']
        self.assertEqual([bar['height'] for bar in bars], sorted(bar['height'] for bar in bars))
        self.assertTrue(all(action.get('visual') == 'v9' for action in actions))
        self.assertIn('12万', [action.get('text') for action in actions])

    def test_counter_counts_and_settles_on_its_final_value(self):
        metadata = stroke_assets.build_counter('count', 0, 1.4, self.font[0], 80, decimals=1, suffix='万+', font_index=self.font[1])
        self.assertEqual(metadata['type'], 'counter')
        from video_renderer import Renderer
        renderer = Renderer.__new__(Renderer)
        start = np.array(renderer.counter_frame(metadata, 0.))[:, :, 3]
        end = np.array(renderer.counter_frame(metadata, 1.))[:, :, 3]
        final = np.array(Image.open(self.root / 'count.png'))[:, :, 3]
        self.assertTrue(np.array_equal(end, final))
        self.assertFalse(np.array_equal(start, final))


class EmphasisAndBoardTests(unittest.TestCase):
    def test_emphasis_moves_return_to_rest(self):
        import motion
        layer = Image.new('RGBA', (80, 40), (200, 60, 40, 255))
        for kind, seconds in motion.EMPHASIS.items():
            effect = {'type': kind, 'start': 0., 'end': seconds}
            for time in (0., seconds):
                total = motion.combine([item for item in [motion.state(effect, time)] if item])
                self.assertTrue(motion.is_identity(total), (kind, time))
        total = motion.combine([motion.state({'type': 'pop', 'start': 0., 'end': .6}, .25)])
        popped, offset = motion.apply(layer, total)
        self.assertGreater(popped.width, layer.width)
        self.assertLess(offset[0], 0)

    def test_emphasis_waits_for_the_drawing_and_stays_on_its_page(self):
        from video_job import resolve_motions
        page = {'start': 10.}
        beats = {'b1': {'start': 11.}, 'b2': {'start': 14.}}
        event = {'end': 13.}
        effects = resolve_motions({'emphasis': [{'type': 'pop', 'at': 'beat:b1'}, {'type': 'glow', 'at': 'b2'}]}, event, page, beats)
        self.assertAlmostEqual(effects[0]['start'], 13.1)
        self.assertAlmostEqual(effects[1]['start'], 14.15)
        idle = resolve_motions({'motion': [{'type': 'float'}, {'type': 'part_nod', 'part': 'head', 'at': 'b2'}]}, event, page, beats, ['head', 'body'])
        self.assertEqual((idle[0]['start'], idle[1]['part']), (13.05, 0))
        with self.assertRaisesRegex(ValueError, 'not on this page'):
            resolve_motions({'emphasis': [{'type': 'pop', 'at': 'beat:x'}]}, event, page, beats)
        with self.assertRaisesRegex(ValueError, 'Motion type'):
            resolve_motions({'emphasis': [{'type': 'spin', 'at': 1}]}, event, page, beats)
        with self.assertRaisesRegex(ValueError, 'part'):
            resolve_motions({'motion': [{'type': 'part_wave', 'part': 'tail'}]}, event, page, beats, ['head'])

    def test_board_scale_follows_the_largest_zoom(self):
        from video_job import page_scale
        self.assertEqual(page_scale({}), 1.)
        self.assertEqual(page_scale({'camera': [{'zoom': 1.2}, {'zoom': 2.2}]}), 2.2)
        self.assertEqual(page_scale({'camera': [{'zoom': 2}], 'board': {'scale': 1.5}}), 1.5)
        with self.assertRaisesRegex(ValueError, 'scale'):
            page_scale({'board': {'scale': 4}})

    def test_pen_on_a_zoomed_board_is_in_screen_units(self):
        from video_renderer import Renderer
        renderer = Renderer.__new__(Renderer)
        metadata = {'segments': [{'kind': 'ink', 'start': 0., 'end': 1., 'points': [[0., 0.], [200., 100.]], 'lengths': [0., 223.6]}]}
        renderer.timeline = {'assets': {'a': metadata}}
        point, _ = renderer.event_state({'asset': 'a', 'x': 100, 'y': 50, 'scale': 2.}, 1.)
        self.assertEqual(point.tolist(), [200., 100.])  # the asset is twice the size on its board, so 200 px there is 100 on screen

    def test_transitions_start_on_the_old_page_and_end_on_the_new(self):
        import transitions

        class Board:
            paper = (250, 248, 240, 255)
        old = Image.new('RGBA', (180, 320), (30, 30, 30, 255))
        new = Image.new('RGBA', (180, 320), (250, 248, 240, 255))
        mean = lambda image: float(np.asarray(image.convert('L')).mean())
        early, late = transitions.erase(Board(), old, new, .05, {}), transitions.erase(Board(), old, new, .95, {})
        self.assertLess(mean(early), 120)
        self.assertGreater(mean(late), 240)
        half = transitions.wipe(Board(), old, new, .5, {'direction': 'left'})
        self.assertTrue(60 < mean(half) < 200)
        self.assertEqual(transitions.zoom(Board(), old, new, 1., {}).size, old.size)


class LayoutMetricsAndCueTests(unittest.TestCase):
    def test_vertical_templates_stay_inside_the_safe_area(self):
        import layout_plan
        for template in layout_plan.TEMPLATES:
            plan = layout_plan.fit(1080, 1920, template, [1.5, 1.2, 1., .8], 2)
            zones = plan['zones']
            for box in plan['pictures']:
                self.assertGreaterEqual(box['y'], zones['header']['y'] + zones['header']['height'] - 1, template)
                self.assertLessEqual(box['y'] + box['height'], zones['caption']['y'] + 1, template)
                self.assertLessEqual(box['x'] + box['width'], zones['safe_right'] + 1, template)
        hero = layout_plan.fit(1080, 1920, 'hero', [1.5])['pictures'][0]
        self.assertGreater(hero['width'], 800)

    def test_camera_moves_and_emphasis_count_as_changes(self):
        import visual_metrics
        page = {'start': 1., 'speech_end': 30., 'beats': [{'id': 'a', 'start': 1.}],
                'events': [{'caption': False, 'start': 1., 'end': 5., 'emphasis': [{'start': 12., 'end': 13.}]}],
                'camera': [{'at': 18, 'seconds': 1.}]}
        timeline = {'cover_seconds': 1., 'pages': [page]}
        longest, where = visual_metrics.unchanged(timeline)
        self.assertEqual((longest, where), (10., [20., 30.]))  # the camera move at 18 s into the page ends at 20 s
        page['overlays'] = [{'start': 22., 'end': 23., 'exit': {'start': 26., 'end': 26.5}}]   # a handwritten keyword, then its fade
        self.assertEqual(visual_metrics.unchanged(timeline), (7., [5., 12.]))

    def test_speech_rate_counts_syllables_not_letters(self):
        from align import spoken_syllables, syllables
        self.assertEqual(spoken_syllables('这也是我高呼All In AI的理由'), 13)  # was 16 characters: 8.3 "chars"/s failed a normal line
        self.assertEqual(spoken_syllables('The quick brown fox jumps over the lazy dog.'), 11)
        self.assertEqual(spoken_syllables('二零二七年以后，你要活成什么样子？'), 15)
        try:
            import pypinyin  # noqa: F401
        except ImportError:
            return
        self.assertIsInstance(syllables('孤舟'), list)  # the pinyin helper of the ASR homophone check is untouched

    def test_cue_starts_snap_to_speech_onsets_or_run_on(self):
        from align import snap_to_onsets
        ranges = snap_to_onsets([0., 2.4], [(0., 1.2), (1.4, 3.5), (3.6, 4.4)])
        self.assertEqual(ranges[1][0], 2.4)
        self.assertEqual(ranges[0][1], 2.4)
        self.assertEqual(ranges[1][1], ranges[2][0])  # no pause before the third cue: the second runs on to it


class AssetLibraryTests(unittest.TestCase):
    def make_job(self, root, with_image=True):
        import hashlib, json
        root.mkdir(parents=True)
        (root / 'design').mkdir()
        record = {'dependency_sha256': 'dep', 'images': []}
        if with_image:
            (root / 'inputs').mkdir()
            Image.new('RGBA', (40, 40), (200, 80, 40, 255)).save(root / 'inputs/prop.png')
            for name in ('brief.md', 'prompt.txt'):
                (root / 'design' / name).write_text('一台打开的笔记本电脑，侧面，粗线平涂')
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            record['images'].append({'source': '../inputs/prop.png', 'sha256': digest(root / 'inputs/prop.png'), 'method': 'generated_under_guidance',
                                     'brief_file': 'brief.md', 'brief_sha256': digest(root / 'design/brief.md'),
                                     'prompt_file': 'prompt.txt', 'prompt_sha256': digest(root / 'design/prompt.txt'),
                                     'status': 'pass', 'evidence': '看过：线条闭合、平涂、可分部位'})
        (root / 'design/art-direction.json').write_text(json.dumps(record, ensure_ascii=False))
        (root / 'job.json').write_text(json.dumps({'version': 1, 'art_direction': {'record': 'design/art-direction.json'}, 'pages': []}))

    def test_add_find_and_reuse_needs_a_fresh_review(self):
        import json, os
        from art_direction import check_record
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            os.environ['HANDDRAW_LIBRARY'] = str(root / 'library')
            try:
                import asset_library
                self.make_job(root / 'first')
                item = asset_library.add(root / 'first', 'inputs/prop.png', ['电脑', '办公'], '笔记本')
                self.assertEqual([entry['id'] for entry in asset_library.find(['电脑'])], [item['id']])
                self.assertEqual(asset_library.find(['厨房']), [])
                self.make_job(root / 'second', with_image=False)
                asset_library.use(root / 'second', item['id'], 'inputs/laptop.png')
                record_path = root / 'second/design/art-direction.json'
                required = [((root / 'second/inputs/laptop.png').resolve(), None)]
                with self.assertRaisesRegex(ValueError, 'no completed art-direction review'):
                    check_record(record_path, required, 'dep')
                record = json.loads(record_path.read_text())
                record['images'][-1].update(status='pass', evidence='在新片里看过：与本片画风一致')
                record_path.write_text(json.dumps(record, ensure_ascii=False))
                self.assertEqual(check_record(record_path, required, 'dep'), 1)
            finally:
                os.environ.pop('HANDDRAW_LIBRARY', None)


if __name__ == '__main__':
    unittest.main()
