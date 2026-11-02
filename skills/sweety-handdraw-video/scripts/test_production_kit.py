"""Production kit: the Codex image route (argument order, own-thread pick, padded references), word-time cues (digits,
say maps), a tiny film compiled from data into a job and storyboard the lints accept, and the art pipeline's briefs."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

import art_pipeline
import codex_image
import film_kit
import storyboard_lint
from timeline_contract import validate_page_motion

FAKE_CODEX = '''#!{python}
import json, os, stat, sys, time
from pathlib import Path
from PIL import Image
home = Path(os.environ['CODEX_HOME']) / 'generated_images'
args = sys.argv[1:]
st, null = os.fstat(0), os.stat('/dev/null')
sizes = [list(Image.open(args[k + 1]).size) for k, arg in enumerate(args) if arg == '-i']
Path(os.environ['FAKE_CODEX_LOG']).write_text(json.dumps({{'argv': args, 'sizes': sizes,
    'stdin_is_null': stat.S_ISCHR(st.st_mode) and st.st_rdev == null.st_rdev}}))
if os.environ.get('FAKE_CODEX_FAIL'):
    sys.exit(3)
(home / 'thread-own').mkdir(parents=True, exist_ok=True)
(home / 'thread-own' / 'a.png').write_bytes(Path(os.environ['FAKE_OWN_PNG']).read_bytes())
time.sleep(.05)
(home / 'thread-other').mkdir(parents=True, exist_ok=True)
(home / 'thread-other' / 'newer.png').write_bytes(b'another run')   # a parallel run's newer image must not be taken
print(json.dumps({{'type': 'thread.started', 'thread_id': 'thread-own'}}))
print(json.dumps({{'type': 'turn.completed'}}))
'''


class CodexImageTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.fake = self.root / 'codex'
        self.fake.write_text(FAKE_CODEX.format(python=sys.executable))
        self.fake.chmod(0o755)
        Image.new('RGB', (64, 64), (10, 200, 30)).save(self.root / 'own.png')
        Image.new('RGB', (100, 100), (240, 230, 220)).save(self.root / 'ref.png')
        (self.root / 'prompt.txt').write_text('A plain line plate of a teapot.')
        self.environment = mock.patch.dict(os.environ, {'CODEX_HOME': str(self.root / 'home'), 'FAKE_CODEX_LOG': str(self.root / 'log.json'),
                                                        'FAKE_OWN_PNG': str(self.root / 'own.png')})
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.directory.cleanup()

    def test_prompt_before_refs_stdin_null_and_own_thread(self):
        result = codex_image.run(self.root / 'prompt.txt', self.root / 'out/picture.png', [(self.root / 'ref.png', 'style')], 'portrait',
                                 model='some-model', effort='low', codex=str(self.fake), cwd=self.root)
        log = json.loads((self.root / 'log.json').read_text())
        argv = log['argv']
        prompt_index = next(k for k, arg in enumerate(argv) if '<prompt>' in arg)
        self.assertLess(prompt_index, argv.index('-i'), 'the prompt must come before -i (it takes several values)')
        self.assertTrue(log['stdin_is_null'])
        self.assertEqual(argv[argv.index('-m') + 1], 'some-model')
        self.assertIn('1024x1536 portrait (2:3)', argv[prompt_index])
        self.assertEqual(log['sizes'], [[100, 150]], 'the square reference is padded onto the 2:3 canvas')
        self.assertEqual((self.root / 'out/picture.png').read_bytes(), (self.root / 'own.png').read_bytes())
        self.assertEqual(result['thread'], 'thread-own')

    def test_no_model_flag_when_empty_and_failure_after_retries(self):
        with mock.patch.dict(os.environ, {'FAKE_CODEX_FAIL': '1'}):
            with self.assertRaises(RuntimeError) as raised:
                codex_image.run(self.root / 'prompt.txt', self.root / 'x.png', (), 'square', model='', effort='', attempts=2,
                                codex=str(self.fake), cwd=self.root)
        self.assertIn('attempt 2', str(raised.exception))
        argv = json.loads((self.root / 'log.json').read_text())['argv']
        self.assertNotIn('-m', argv)
        self.assertNotIn('-i', argv)

    def test_padding_follows_the_requested_canvas(self):
        with tempfile.TemporaryDirectory() as scratch:
            for size, shape, expected in (((120, 100), 'portrait', (120, 180)), ((100, 120), 'landscape', (180, 120)),
                                          ((100, 150), 'square', (150, 150))):
                source = self.root / f'{size[0]}x{size[1]}.png'
                Image.new('RGB', size, (240, 230, 220)).save(source)
                padded = codex_image.pad_to_shape(source, shape, scratch)
                with Image.open(padded) as image:
                    self.assertEqual(image.size, expected)
                    self.assertEqual(image.getpixel((0, 0)), (240, 230, 220), 'padding takes the border colour')
            fitting = self.root / 'fits.png'
            Image.new('RGB', (200, 300), 'white').save(fitting)
            self.assertEqual(codex_image.pad_to_shape(fitting, 'portrait', scratch), fitting)


def tiny_job(root):
    """Three script pages, voiced: word timings with Whisper digits (24, 36+5) and a say map (Grok -> 格罗克)."""
    root = Path(root)
    (root / 'inputs/narration/master-chunks').mkdir(parents=True)
    script = [{'id': 'p01', 'title': '开场', 'beats': [{'id': 'p01b1', 'text': '第二十四条：用Grok搜索。'}, {'id': 'p01b2', 'text': '一年三百六十五篇。'}]},
              {'id': 'p02', 'title': '第二组', 'beats': [{'id': 'p02b1', 'text': '第二组开始。'}]},
              {'id': 'p03', 'title': '流程', 'beats': [{'id': 'p03b1', 'text': '先看清单。'}, {'id': 'p03b2', 'text': '再看流程。'}]}]
    (root / 'inputs/script.json').write_text(json.dumps(script, ensure_ascii=False))
    times = {'p01b1': (0., 2.2), 'p01b2': (2.4, 3.8), 'p02b1': (4.0, 8.0), 'p03b1': (8.2, 10.0), 'p03b2': (10.2, 12.0)}
    (root / 'inputs/narration/alignment.json').write_text(json.dumps({'beats': [
        {'id': beat, 'text': text, 'start': times[beat][0], 'end': times[beat][1]} for page in script for beat, text in
        ((item['id'], item['text']) for item in page['beats'])]}, ensure_ascii=False))
    words = lambda pairs: [{'text': text, 'start': start, 'end': start + .2} for text, start in pairs]
    chunks = [(['p01b1', 'p01b2'], words([('第', 0), ('24', .2), ('条', .5), ('用', .8), ('格罗克', 1.0), ('搜索', 1.5),
                                          ('一', 2.4), ('年', 2.6), ('36', 2.8), ('5', 3.0), ('篇', 3.3)])),
              (['p02b1'], words([('第', 0), ('二', .3), ('组', .6), ('开始', .9)])),
              (['p03b1', 'p03b2'], words([('先', 0), ('看', .3), ('清单', .6), ('再', 2.0), ('看', 2.3), ('流程', 2.6)]))]
    texts = {beat['id']: beat['text'] for page in script for beat in page['beats']}
    for index, (beats, recognised) in enumerate(chunks):
        (root / f'inputs/narration/master-chunks/{index:03d}.json').write_text(json.dumps({
            'text': ''.join(texts[beat] for beat in beats), 'units': [{'beat': beat, 'text': texts[beat]} for beat in beats],
            'attempts': [{'status': 'PASS', 'words': recognised}]}, ensure_ascii=False))
    (root / 'design').mkdir()
    (root / 'design.json').write_text(json.dumps({'presentation': {'mode': 'focus_diagram', 'reason': 'one form per page'}}))
    (root / 'job.json').write_text(json.dumps({'version': 1, 'title': '测试', 'canvas': {'width': 1280, 'height': 1920, 'fps': 30},
                                               'design_file': 'design.json',
                                               'narration': {'source': 'inputs/narration/master.wav', 'alignment': 'inputs/narration/alignment.json'},
                                               'pages': []}, ensure_ascii=False))
    film = {'say': {'Grok': '格罗克'},
            'roadmap': {'groups': [{'label': '第一组', 'count': 1}, {'label': '第二组', 'count': 2}], 'headers': ['第一组', '第二组 · 开始']},
            'scripts': {'pulse': "show(word('跳', W / 2, H / 2, 40, '#000'), p.beat_ms);"},
            'screens': [
                {'id': 'p01', 'overlays': [{'title': '用 Grok 搜索'}, {'strip': {'item': 1}},
                                           {'form': 'checklist', 'at': '@p01b1:Grok', 'visual': 'v01-list', 'params': {'items': [
                                               {'text': '第 24 条', 'at': '@p01b1:条', 'mark': 'check'}, {'text': '365 篇', 'at': '@p01b2:篇', 'mark': 'strike'}]}}],
                 'shot': ['key_points', '开场：两条清单']},
                {'id': 'p02', 'chapter': 1},
                {'id': 'p03', 'transition': 'quiet', 'overlays': [
                    {'strip': {'group': 1}}, {'title': '看流程', 'at': '@p03b1'},
                    {'custom': 'pulse', 'at': '@p03b2', 'visual': 'v03-anim', 'element': '跳动的字', 'explains': '流程的节拍',
                     'moments': ['@p03b2', {'=': "ms('p03b2') + 500"}], 'params': {'beat_ms': '@p03b2+0.2'}}],
                 'shot': ['process_sequence', '流程：节拍动画']}]}
    (root / 'design/film.json').write_text(json.dumps(film, ensure_ascii=False))
    return root


class FilmKitTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = tiny_job(self.directory.name)
        self.film = film_kit.Film(self.root)
        self.film.load_narration()

    def tearDown(self):
        self.directory.cleanup()

    def test_phrases_follow_digits_and_say(self):
        self.assertEqual(self.film.at('p01b1', '条'), .5, 'Whisper "24" stands for 二十四 (three characters)')
        self.assertEqual(self.film.at('p01b1', 'Grok'), 1.0, 'a phrase is read through the say map')
        self.assertEqual(self.film.at('p01b1', '格罗克'), 1.0)
        self.assertEqual(self.film.at('p01b2', '篇'), 3.3, '"36" + "5" is one number, 三百六十五')
        self.assertEqual(self.film.at('p01b2', '篇', -.15), 3.15)
        self.assertEqual(self.film.at('p03b2'), 2.0, 'page seconds: from the first beat of the screen')
        with self.assertRaises(ValueError):
            self.film.at('p01b1', '不存在')
        self.assertEqual(film_kit.spoken_length('310'), 4)
        self.assertEqual(film_kit.spoken_length('2014'), 4)

    def test_cues_and_expressions(self):
        self.assertEqual(self.film.cue('@p01b2:篇+0.8'), ('p01b2', '篇', .8))
        self.assertEqual(self.film.cue('@p01b1'), ('p01b1', None, None))
        self.assertIsNone(self.film.cue('@leo_xiaolei'))
        with self.assertRaises(ValueError):
            self.film.cue('@p09b9:x')
        self.assertEqual(self.film.evaluate("max(3000, ms('p01b2', '篇'))", 0.), 3300)
        self.assertEqual(self.film.evaluate('range(0, 1000, 400)', 0.), [0, 400, 800])
        with self.assertRaises(ValueError):
            self.film.evaluate("__import__('os')", 0.)

    def test_compiles_a_valid_job_and_storyboard(self):
        job, board = self.film.write()
        pages = {page['id']: page for page in job['pages']}
        self.assertEqual(job['narration']['say'], {'Grok': '格罗克'})
        header = pages['p01']['overlays'][0]
        self.assertEqual((header['at'], header['params']['style']), (.5, 'glyph_in'), 'the first header waits for the cover fade')
        self.assertEqual(pages['p03']['overlays'][1]['at'], 'beat:p03b1')
        first_strip, second_strip = pages['p01']['overlays'][1]['params'], pages['p03']['overlays'][0]['params']
        self.assertNotIn('enter', first_strip)
        self.assertEqual(second_strip['enter'], 'settled')
        checklist = pages['p01']['overlays'][2]
        self.assertEqual(checklist['at'], .85)
        self.assertEqual([(item['ms'], item['mark_ms']) for item in checklist['params']['items']], [(0, 450), (2450, 3250)])
        self.assertEqual(checklist['params']['items'][1]['color'], film_kit.DEFAULTS['palette']['accent'])
        custom = pages['p03']['overlays'][2]
        self.assertTrue(custom['params']['script'].startswith(film_kit.SVG_KIT))
        self.assertEqual((custom['params']['moments'], custom['params']['beat_ms']), ([0, 500], 200))
        chapter = pages['p02']
        self.assertTrue(chapter['chapter'])
        self.assertEqual(chapter['transition']['type'], 'page_turn')
        self.assertEqual(chapter['visual_groups'][0]['actions'][0]['text'], '第二组')
        # the engine's own checks
        beats = storyboard_lint.alignment_beats(self.root, job)
        report = storyboard_lint.plan(board, beats)
        self.assertEqual(report['errors'], [])
        self.assertEqual(storyboard_lint.usage(self.root, job, board), [])
        times = {beat['id']: beat for beat in json.loads((self.root / 'inputs/narration/alignment.json').read_text())['beats']}
        for index, page in enumerate(job['pages']):
            timed = {**page, 'start': times[page['beats'][0]['id']]['start'], 'end': times[page['beats'][-1]['id']]['end']}
            validate_page_motion(timed, {'width': 1280, 'height': 1920}, first=index == 0)
        self.assertEqual([shot['id'] for shot in board['shots']], ['s01', 's02', 's03'])
        self.assertEqual(board['shots'][2]['composition']['focus'], 'v03-anim', 'the progress strip is navigation, not the focus')


class ArtPipelineTest(unittest.TestCase):
    def test_briefs_carry_the_page_budget_and_reviewed_pictures_are_kept(self):
        with tempfile.TemporaryDirectory() as directory:
            root = tiny_job(directory)
            (root / 'style').mkdir()
            (root / 'style/resolved.json').write_text('{}')
            (root / 'design/image-specs.json').write_text(json.dumps({
                'vars': {'HERO': 'a calm adult reader'},
                'style': {'brief': ['- 画风：测试'], 'line_art': 'even black outlines.', 'palette': 'warm white, teal.'},
                'images': {'a01-desk': {'page': 'p03', 'use': '清单页', 'ref': False, 'scene': '{HERO} at a desk.', 'notes': 'Desk teal.',
                                        'narration': [['先看清单', '一个人在桌前看清单']]}}}, ensure_ascii=False))
            pipeline = art_pipeline.Pipeline(root)
            report = pipeline.prompts()
            brief = (root / 'design/briefs/a01-desk.md').read_text()
            strokes = int(art_pipeline.STROKES_PER_SECOND * art_pipeline.PEN_SHARE * 3.8)   # p03 is voiced for 3.8 s
            self.assertIn(f'本图最多约 {strokes} 笔', brief)
            plate = (root / 'design/prompts/a01-desk-line.txt').read_text()
            self.assertIn('a calm adult reader at a desk.', plate)
            self.assertIn('Compose for a tall 2:3 portrait page', plate, 'portrait is the default on a 2:3 canvas')
            self.assertIn(f'about {max(12, strokes)} separate outlines', plate)
            record = json.loads((root / 'design/art-direction.json').read_text())
            self.assertEqual({entry['status'] for entry in record['images']}, {'pending'})
            self.assertTrue(all(entry.get('brief_sha256') and entry.get('prompt_sha256') for entry in record['images']))
            self.assertIn('bound_images', report['bind'])
            # a reviewed picture keeps its bound brief and prompts
            for entry in record['images']:
                entry['status'] = 'pass'
            (root / 'design/art-direction.json').write_text(json.dumps(record))
            (root / 'design/image-specs.json').write_text((root / 'design/image-specs.json').read_text().replace('Desk teal.', 'Desk red.'))
            self.assertIn('kept', art_pipeline.Pipeline(root).prompts()['a01-desk'])
            self.assertIn('Desk teal.', (root / 'design/prompts/a01-desk-colour.txt').read_text())


if __name__ == '__main__':
    unittest.main()
