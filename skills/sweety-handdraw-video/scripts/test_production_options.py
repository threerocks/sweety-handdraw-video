"""Native options: config_type gate, aspect presets, model_infer vs active_select."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from production_options import ChoiceSession, validate_choice, apply_choices, default_bundle
from aspect_ratio import resolve_canvas, PRESETS, DEFAULT_RATIO
from hand_assets import prepare_hand
from watermark_assets import prepare_watermark, watermark_position
from timeline_contract import validate_timeline
from caption_assets import build_caption
from video_job import find_font
from presentation import validate_presentation, MODES, page_content_policy


class AspectRatioTests(unittest.TestCase):
    def test_default_is_vertical_two_three(self):
        canvas = resolve_canvas()
        self.assertEqual(canvas['aspect_ratio'], '2:3')
        self.assertEqual((canvas['width'], canvas['height']), (1280, 1920))

    def test_named_presets_and_custom(self):
        for name, size in PRESETS.items():
            self.assertEqual((resolve_canvas(name)['width'], resolve_canvas(name)['height']), size)
        custom = resolve_canvas('custom', 1000, 1000)
        self.assertEqual(custom['aspect_ratio'], 'custom')
        self.assertEqual((custom['width'], custom['height']), (1000, 1000))
        with self.assertRaises(ValueError):
            resolve_canvas('7:5')
        with self.assertRaises(ValueError):
            resolve_canvas('custom', 100, 100)


class PresentationTests(unittest.TestCase):
    def test_modes_cover_sparse_and_dense(self):
        self.assertEqual(MODES['picture_book']['density'], 'sparse')
        self.assertEqual(MODES['dense_infographic']['density'], 'dense')
        self.assertEqual(MODES['accumulating_board']['density'], 'dense')
        policy = page_content_policy('picture_book')
        self.assertEqual(policy['max_heroes_per_page'], 1)
        self.assertGreaterEqual(policy['hold_after_draw_s'], 2.0)

    def test_reason_required(self):
        with self.assertRaisesRegex(ValueError, 'reason'):
            validate_presentation({'mode': 'picture_book'})
        out = validate_presentation({'mode': 'picture_book', 'reason': '儿童睡前故事'})
        self.assertEqual(out['mode'], 'picture_book')


class ProductionOptionsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.catalog = {'styles': [{'id': '1', 'name': 'Test'}]}
        self.hands = [{'id': 'navy'}, {'id': 'pen-only'}]
        self.voices = [{'id': 'iapetus', 'available': True}, {'id': 'external', 'available': False}]
        self.defaults = default_bundle('解释愤怒如何伤害自己', style={'mode': 'inferred'}, voice='iapetus')
        # default_bundle uses magnetic-narrator voice id in voice dict - override
        self.defaults['voice'] = {'voice': 'iapetus'}
        self.defaults['hand'] = {'preset': 'navy'}

    def tearDown(self):
        self.temporary.cleanup()

    def session(self, explicit=None, config_type=None):
        defaults = dict(self.defaults)
        if config_type:
            defaults['config_type'] = config_type
        return ChoiceSession(self.root, defaults, explicit or {}, self.catalog, self.hands, self.voices, 60, now=0)

    def test_model_infer_completes_immediately_without_stages(self):
        session = self.session(config_type='model_infer')
        status = session.advance(0)
        self.assertTrue(status['complete'])
        self.assertEqual(status['config_type'], 'model_infer')
        self.assertIsNone(status['stage'])
        persisted = json.loads((self.root / 'options/selection.json').read_text())
        self.assertTrue(persisted['complete'])
        self.assertEqual(persisted['choices']['aspect']['value']['aspect_ratio'], DEFAULT_RATIO)
        self.assertEqual(persisted['choices']['style']['value']['mode'], 'inferred')
        style = json.loads((self.root / 'style/selection.json').read_text())
        self.assertEqual(style['mode'], 'inferred')

    def test_active_select_times_out_per_stage_with_new_order(self):
        session = self.session(config_type='active_select')
        self.assertEqual(session.advance(0)['stage'], 'aspect')
        self.assertEqual(session.advance(60)['stage'], 'style')
        self.assertEqual(session.advance(120)['stage'], 'voice')
        self.assertEqual(session.advance(180)['stage'], 'captions')
        self.assertTrue(session.advance(240)['complete'])
        stages = json.loads((self.root / 'options/selection.json').read_text())['stages']
        self.assertEqual(stages, ['aspect', 'style', 'voice', 'captions'])

    def test_explicit_aspect_and_style_skip_wait(self):
        session = self.session(config_type='active_select', explicit={
            'aspect': {'aspect_ratio': '16:9'},
            'style': {'mode': 'catalog', 'catalog_id': '1'},
        })
        self.assertEqual(session.advance(0)['stage'], 'voice')
        self.assertEqual(session.choices['aspect']['value']['aspect_ratio'], '16:9')

    def test_missing_content_and_voice_limits(self):
        self.defaults['content'] = ' '
        with self.assertRaisesRegex(ValueError, 'MISSING_CONTENT'):
            self.session()
        with self.assertRaises(ValueError):
            validate_choice('voice', {'voice': 'external'}, self.catalog, self.hands, self.voices)
        with self.assertRaises(ValueError):
            validate_choice('advertising', {'text': '甲' * 17}, self.catalog, self.hands, self.voices)

    def test_applied_choices_set_canvas_and_layers(self):
        session = self.session(config_type='model_infer', explicit={
            'aspect': {'aspect_ratio': '9:16'},
            'hand': {'preset': 'pen-only'},
            'advertising': {'text': '完整品牌名称'},
            'watermark': {'text': '版权署名'},
            'captions': {'mode': 'plate'},
        })
        self.assertTrue(session.advance(0)['complete'])
        (self.root / 'job.json').write_text(json.dumps({'version': 1, 'hand': {'source': 'old-hand.png'}, 'pages': []}))
        job = apply_choices(self.root)
        self.assertEqual(job['canvas']['aspect_ratio'], '9:16')
        self.assertEqual((job['canvas']['width'], job['canvas']['height']), (1080, 1920))
        self.assertEqual(job['hand'], {'preset': 'pen-only', 'pen_text': '完整品牌名称'})
        self.assertEqual(job['watermark']['text'], '版权署名')
        self.assertEqual(job['captions']['color'], '#FFFFFF')

    def test_pen_only_and_watermark_still_work(self):
        font_path, font_index = find_font({})
        hand = prepare_hand({'preset': 'pen-only', 'pen_text': '手绘口播Skill'}, self.root, self.root, 1920, font_path, font_index)
        self.assertEqual(hand['kind'], 'pen')
        self.assertTrue(validate_timeline({'cover_seconds': 1, 'canvas': {'height': 1920}, 'hand': hand, 'pages': []}))
        with self.assertRaisesRegex(ValueError, '16'):
            prepare_hand({'preset': 'pen-only', 'pen_text': '甲' * 17}, self.root, self.root, 1920, font_path, font_index)
        self.assertIsNone(prepare_watermark({}, self.root, 1080, 1920))
        text = '『手绘口播.Skill』今晚 18:00 全网首发。\n版权所有：@leo_xiaolei'
        mark = prepare_watermark({'text': text}, self.root, 1080, 1920)
        self.assertEqual(mark['text'], text)
        alpha = np.array(Image.open(mark['source']))[:, :, 3]
        self.assertGreater(alpha.max(), 0)
        self.assertLessEqual(alpha.max(), 62)
        # a paper-coloured halo at the same opacity: readable over a dark night picture, invisible on paper
        rgba = np.array(Image.open(mark['source']))
        self.assertTrue(((rgba[..., 0] > 200) & (rgba[..., 3] > 0)).any())
        plain = np.array(Image.open(prepare_watermark({'text': text, 'halo': False}, self.root, 1080, 1920)['source']))
        self.assertFalse(((plain[..., 0] > 200) & (plain[..., 3] > 0)).any())
        positions = [watermark_position(mark, 1080, 1920, time) for time in range(0, 3600, 11)]
        self.assertGreater(len(set(positions)), 20)
        for x, y in positions:
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(x + mark['width'], 1080)
            self.assertLessEqual(y + mark['height'], 1920)

    def test_caption_plate_is_an_explicit_supported_choice(self):
        metadata = build_caption(self.root, 'caption', '完整字幕', {'background': {'color': '#243039', 'opacity': .76}, 'color': '#FFFFFF'}, 1080, 1920)
        alpha = np.array(Image.open(self.root / 'caption.png'))[:, :, 3]
        self.assertTrue((alpha == 194).any())
        self.assertEqual(metadata['background']['color'], '#243039')


if __name__ == '__main__':
    unittest.main()
