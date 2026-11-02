"""3.0 final-version fixes: the watermark takes a quiet corner per page, the voice starts under the cover fade, the
film settles after the last word, the auto camera keeps pictures out of screen titles and still when ink sits on a
screen layer, titles take turns only when they really overlap, say/show narration, and the new director checks."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import director_lint
import narration_master
import watermark_assets
from smoke_test import write_test_tone
from video_job import FilmBuilder, box_overlap, guard_screen_titles, ink_on_screen_layer, take_turns


class WatermarkPlacementTests(unittest.TestCase):
    def fake(self, frames):
        class Renderer:
            pass
        renderer = Renderer()
        renderer.width, renderer.height, renderer.paper = 1280, 1920, (242, 239, 232, 255)
        renderer.work = Path(tempfile.mkdtemp())
        renderer.timeline = {'watermark': {'width': 420, 'height': 60, 'start_seconds': 1.},
                             'pages': [{'index': index, 'start': 1. + 5 * index, 'end': 6. + 5 * index, 'events': []} for index in range(len(frames))]}
        return renderer

    def test_each_page_takes_its_quietest_slot_and_the_mark_crossfades_between_pages(self):
        quiet = Image.new('RGBA', (1280, 1920), (242, 239, 232, 255))
        top_busy = quiet.copy()
        ImageDraw.Draw(top_busy).rectangle((0, 0, 1280, 400), fill='#334455')   # a dark picture across the whole top
        renderer = self.fake([quiet, top_busy])
        original = watermark_assets.clean_view
        watermark_assets.clean_view = lambda renderer_, index, time: [quiet, top_busy][index]
        try:
            plan = watermark_assets.plan_watermark(renderer)
        finally:
            watermark_assets.clean_view = original
        self.assertEqual(plan['pages'][0]['anchor'], 'top_right')
        self.assertNotIn(plan['pages'][1]['anchor'], ('top_right', 'top_left'))
        self.assertEqual(plan['status'], 'PASS')
        self.assertTrue((renderer.work / 'watermark-plan.json').is_file())
        mark = renderer.timeline['watermark']
        _, before = watermark_assets.planned_watermark(plan, mark, 1280, 1920, 5.9)      # fading out before the turn
        _, after = watermark_assets.planned_watermark(plan, mark, 1280, 1920, 6.1)       # fading in after it
        _, steady = watermark_assets.planned_watermark(plan, mark, 1280, 1920, 8.)
        self.assertLess(before, 1.)
        self.assertLess(after, 1.)
        self.assertEqual(steady, 1.)

    def test_an_all_busy_page_fails_and_positions_are_validated(self):
        busy = Image.new('RGBA', (1280, 1920), '#223344')
        renderer = self.fake([busy])
        original = watermark_assets.clean_view
        watermark_assets.clean_view = lambda *_: busy
        try:
            self.assertEqual(watermark_assets.plan_watermark(renderer)['status'], 'FAIL')
        finally:
            watermark_assets.clean_view = original
        with self.assertRaises(ValueError):
            watermark_assets.prepare_watermark({'text': '署名', 'position': 'middle'}, renderer.work, 1280, 1920)
        self.assertEqual(watermark_assets.prepare_watermark({'text': '署名'}, renderer.work, 1280, 1920)['position'], 'auto')


class OpeningAndEndingTimingTests(unittest.TestCase):
    def build(self, extra):
        root = Path(tempfile.mkdtemp())
        Image.new('RGB', (640, 360), 'white').save(root / 'cover.png')
        write_test_tone(root / 'a.wav', 2, 220)
        manifest = {'version': 1, 'runtime_fixture': True, 'title': '开口', 'canvas': {'width': 640, 'height': 360, 'fps': 24},
                    'cover': {'source': 'cover.png', 'seconds': 1}, 'audio': {'trim_silence': False},
                    'pages': [{'beats': [{'id': 'a', 'text': '一句话。', 'audio': 'a.wav', 'actions': [{'type': 'text', 'text': '标题', 'x': 30, 'y': 30, 'size': 36}]}]}]}
        for key, value in extra.items():
            manifest[key] = {**manifest.get(key, {}), **value}
        (root / 'job.json').write_text(json.dumps(manifest, ensure_ascii=False))
        return FilmBuilder(root).build()

    def test_the_voice_starts_under_the_cover_fade_and_the_film_settles(self):
        plain = self.build({})
        early = self.build({'cover': {'voice_overlap': .5}, 'ending': {'settle_seconds': 2.}})
        self.assertEqual(plain['voice_start_seconds'], 1.)
        self.assertEqual(early['voice_start_seconds'], .5)
        self.assertAlmostEqual(early['pages'][0]['beats'][0]['start'], plain['pages'][0]['beats'][0]['start'] - .5, places=3)
        self.assertAlmostEqual(early['duration'], plain['duration'] - .5 + 2., places=2)
        self.assertEqual(early['ending']['settle_seconds'], 2.)

    def test_out_of_range_values_are_refused(self):
        with self.assertRaises(ValueError):
            self.build({'cover': {'voice_overlap': .8}})
        with self.assertRaises(ValueError):
            self.build({'ending': {'settle_seconds': 9}})


class CameraAndTitleTests(unittest.TestCase):
    def test_a_graze_is_not_a_clash_but_a_real_overlap_is(self):
        first = {'x': 0, 'y': 0, 'width': 1000, 'height': 100, 'start': 0., 'end': 1.}
        graze = {'x': 0, 'y': 95, 'width': 1000, 'height': 100, 'start': 2., 'end': 3.}
        self.assertLess(box_overlap(first, graze), .12)
        take_turns([first, graze], {'end': 5.})
        self.assertNotIn('exit', first)
        on_top = {'x': 0, 'y': 30, 'width': 1000, 'height': 100, 'start': 2., 'end': 3.}
        take_turns([first, on_top], {'end': 5.})
        self.assertIn('exit', first)

    def test_ink_on_a_screen_layer_holds_the_camera(self):
        card = {'recipe': 'custom', 'space': 'screen', 'x': 100, 'y': 300, 'width': 800, 'height': 500}
        writing = {'type': 'text', 'text': '荣誉国民', 'x': 300, 'y': 420, 'size': 80}
        self.assertTrue(ink_on_screen_layer({'overlays': [card], 'visual_groups': [{'actions': [writing]}]}))
        self.assertFalse(ink_on_screen_layer({'overlays': [dict(card, space='board')], 'visual_groups': [{'actions': [writing]}]}))
        self.assertFalse(ink_on_screen_layer({'overlays': [dict(card, recipe='kinetic_title')], 'visual_groups': [{'actions': [writing]}]}))

    def test_a_push_in_backs_off_before_the_picture_reaches_a_screen_title(self):
        work = Path(tempfile.mkdtemp())
        (work / 'assets').mkdir()
        art = Image.new('RGBA', (500, 800), (0, 0, 0, 0))
        ImageDraw.Draw(art).rectangle((0, 0, 499, 799), fill='#335577')
        art.save(work / 'assets' / 'pic.png')
        page = {'events': [{'asset': 'pic', 'x': 390, 'y': 340, 'caption': False}]}
        spec = {'overlays': [{'recipe': 'kinetic_title', 'space': 'screen', 'x': 77, 'y': 150, 'width': 1126, 'height': 170,
                              'params': {'text': '其实只讲两件事', 'size': 76}}]}
        keys = [{'at': 0., 'center': [640., 1042.], 'zoom': 1.0, 'move': 'cut'}, {'at': .05, 'center': [640., 1042.], 'zoom': 1.08, 'move': 'drift'}]
        out = guard_screen_titles([dict(key) for key in keys], page, spec, {'pic': {}}, work, 1280, 1920)
        self.assertLess(out[1]['zoom'], 1.08)
        self.assertGreaterEqual(out[1]['zoom'], 1.)
        self.assertIn('camera_backed_off_for_title', page['auto'])


class SayShowTests(unittest.TestCase):
    manifest = {'narration': {'say': {'Grok': '格罗克', '70万': '七十万'}},
                'pages': [{'beats': [{'id': 'a', 'text': '存款不到70万，GROK 也懂。', 'say': {'懂': '懂得'}}]}]}

    def test_the_voice_reads_the_spoken_form_and_subtitles_keep_the_written_one(self):
        beat = self.manifest['pages'][0]['beats'][0]
        self.assertEqual(narration_master.spoken_text(self.manifest, beat), '存款不到七十万，格罗克 也懂得。')
        self.assertIn('[say]', narration_master.script_text(self.manifest))
        self.assertEqual(narration_master.script_text({'pages': [{'beats': [{'id': 'x', 'text': '原样。'}]}]}), '原样。')
        chunk = narration_master.narration_chunks(self.manifest)[0]
        self.assertEqual(chunk['text'], '存款不到七十万，格罗克 也懂得。')
        self.assertEqual(chunk['units'][0]['show'], '存款不到70万，GROK 也懂。')
        audio = np.zeros(48000 * 2, np.float32)
        audio[4800:40000] = .2 * np.sin(np.arange(35200) / 7)
        captions = narration_master.chunk_alignment([(chunk, 0., audio, None)], None,
                                                    lambda beat_id, piece: narration_master.spoken_text(self.manifest, beat, piece))
        self.assertEqual(''.join(text for text, _, _ in captions['a']), '存款不到70万，GROK 也懂。')
        self.assertIn('ample', ''.join(narration_master.spoken_text({'narration': {'say': {'Amp': '安普'}}}, {'text': 'ample'})))

    def test_digits_and_english_are_listed_for_a_probe(self):
        terms = narration_master.risky_terms({'pages': [{'beats': [{'id': 'x', 'text': '用 Cursor 写了 3 个程序。'}]}]})
        self.assertIn('cursor', terms)
        self.assertIn('3', terms)
        self.assertNotIn('grok', narration_master.risky_terms(self.manifest))


class DirectorChecksTests(unittest.TestCase):
    def test_time_words_are_not_comparisons_and_shown_numbers_need_no_counter(self):
        comparison = director_lint.CUES[0][0]
        self.assertIsNone(comparison.search('半年以后，她报名了'))
        self.assertIsNotNone(comparison.search('以前要三天，现在只要一小时'))
        self.assertEqual(director_lint.spoken_numbers('存款不到七十万'), {'700000'})
        page = {'overlays': [{'recipe': 'kinetic_title', 'params': {'text': '存款不到70万'}}], 'visual_groups': []}
        self.assertEqual(director_lint.shown_numbers(page), {'700000'})

    def test_a_run_of_one_form_is_flagged(self):
        pages = [{'id': f'p{k}'} for k in range(6)]
        warnings = []
        director_lint.repetition(pages, [{'checklist'}] * 5 + [{'table'}], warnings)
        self.assertTrue(any('in a row' in item for item in warnings))
        warnings = []
        director_lint.repetition(pages, [{'checklist'}, {'table'}, {'flow'}, {'checklist'}, {'map_route'}, {'table'}], warnings)
        self.assertEqual(warnings, [])

    def test_a_picture_running_into_a_title_is_an_error(self):
        root = Path(tempfile.mkdtemp())
        (root / 'art').mkdir()
        art = Image.new('RGB', (500, 800), 'white')
        ImageDraw.Draw(art).rectangle((0, 0, 499, 200), fill='#335577')
        art.save(root / 'art' / 'p.png')
        picture = {'type': 'illustration', 'source': 'art/p.png', 'x': 390, 'y': 200, 'width': 500, 'height': 800}
        title = {'recipe': 'kinetic_title', 'space': 'screen', 'x': 77, 'y': 150, 'width': 1126, 'height': 170, 'params': {'text': '其实只讲两件事', 'size': 76}}
        manifest = {'pages': [{'id': 'p1', 'beats': [{'id': 'a', 'text': '一句。'}], 'visual_groups': [{'actions': [picture]}], 'overlays': [title]}]}
        errors, warnings = [], []
        director_lint.page_checks(root, manifest, errors, warnings, root / 'none.json')
        self.assertTrue(any('runs into the title' in item for item in errors))


if __name__ == '__main__':
    unittest.main()


class SecondReviewFixesTests(unittest.TestCase):
    def test_a_cue_never_ends_on_the_next_condition(self):
        import caption_assets
        self.assertEqual(caption_assets.cue_pieces('记住一句话：没推送，用 amend 或 reset；推送了，就用 revert。', 14),
                         ['记住一句话：', '没推送，用 amend 或 reset；', '推送了，就用 revert。'])

    def test_self_animating_forms_last_long_enough(self):
        import film_kit
        self.assertGreaterEqual(film_kit.intrinsic('rough_chart', {'events': [1, 2, 3, 4, 5]}), 3.5)
        self.assertGreaterEqual(film_kit.intrinsic('map_route', {'travel_ms': 2600}), 4.)
        self.assertEqual(film_kit.intrinsic('kinetic_title', {}), 0.)

    def test_a_motion_only_page_arrives_already_started(self):
        from video_renderer import Renderer
        renderer = Renderer.__new__(Renderer)
        renderer.timeline = {'pages': [{'events': []}, {'events': [{'caption': True}]}, {'events': [{'caption': False}]}]}
        window = {'start': 8.9, 'end': 9.6}
        self.assertEqual(renderer.arrival(1, window, 9.0), 9.6)   # only captions: arrives as it looks when the move ends
        self.assertEqual(renderer.arrival(2, window, 9.0), 9.0)   # a drawn page arrives live
