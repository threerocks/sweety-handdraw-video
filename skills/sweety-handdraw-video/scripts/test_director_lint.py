"""Director checks: a short hook line that is not a greeting and matches the cover, no ending announcement, a long list
that is grouped and re-hooked, plain wording, and a restrained sound plan."""
import json
import tempfile
import unittest
from pathlib import Path

import director_lint


def job(beats, design=None, audio=None):
    root = Path(tempfile.mkdtemp())
    pages = [{'beats': [{'id': f'b{k}', 'text': text}]} for k, text in enumerate(beats)]
    (root / 'job.json').write_text(json.dumps({'pages': pages, 'audio': audio or {'sfx': {'accents': False}}}, ensure_ascii=False))
    base = {'opening_hook': {'spoken_text': beats[0].split('。')[0]}, 'cover': {'title': ['AI 写的代码，', '他基本不审了']},
            'retention_plan': {'rehooks': [], 'emotion_arc': '好奇 → 豁然', 'ending': '回扣开头'}, 'sound_design': {'bed': 'focused'}}
    (root / 'design.json').write_text(json.dumps({**base, **(design or {})}, ensure_ascii=False))
    return root


class DirectorLintTests(unittest.TestCase):
    def test_a_clean_hook_passes(self):
        report = director_lint.check(job(['AI 写的代码，他基本不审了。', '说这话的是一位工程师。']))
        self.assertEqual(report['errors'], [])

    def test_greetings_long_first_lines_and_end_announcements_are_errors(self):
        self.assertTrue(director_lint.check(job(['大家好，今天我们来聊聊 AI。', '下面开始。']))['errors'])
        long = '在人工智能飞速发展并且不断改变我们每一个人工作方式的今天我们需要重新思考。'
        self.assertTrue(any('characters' in item for item in director_lint.check(job([long, '好。']))['errors']))
        self.assertTrue(any('announces the end' in item for item in director_lint.check(job(['AI 写的代码，他基本不审了。', '今天就到这里。']))['errors']))

    def test_a_long_list_needs_rehooks_and_ai_tone_is_flagged(self):
        items = ['AI 写的代码，他基本不审了。'] + [f'第{n}条：做一件事。' for n in '一二三四五六七八']
        report = director_lint.check(job(items))
        self.assertTrue(any('rehooks' in item for item in report['errors']))
        report = director_lint.check(job(['AI 写的代码，他基本不审了。', '这不是工具，而是同事。']))
        self.assertTrue(any('不是' in item for item in report['warnings']))

    def test_sound_plans_that_break_the_mood_are_flagged(self):
        report = director_lint.check(job(['AI 写的代码，他基本不审了。', '好。'], audio={'sfx': True}))
        self.assertTrue(any('accents' in item for item in report['warnings']))
        root = job(['AI 写的代码，他基本不审了。', '好。'], audio={'sfx': {'accents': False}, 'music': {'source': 'song.mp3', 'lyrics': True}})
        self.assertTrue(any('lyrics' in item for item in director_lint.check(root)['errors']))

    def test_a_long_list_shows_its_map_its_chapters_and_where_we_are(self):
        items = ['AI 写的代码，他基本不审了。'] + [f'第{n}条：做一件事。' for n in '一二三四五六七八']
        design = {'retention_plan': {'rehooks': [{'beat': 'b5'}, {'beat': 'b8'}], 'progress': '过半', 'emotion_arc': '好奇', 'ending': '回扣'}}
        root = job(items, design)
        report = director_lint.check(root)
        self.assertTrue(any('its map before the first item' in item for item in report['errors']))
        self.assertTrue(any('b5: a chapter turn' in item for item in report['errors']))
        self.assertTrue(any('progress strip' in item for item in report['errors']))
        manifest = json.loads((root / 'job.json').read_text())
        roadmap = lambda mode: {'recipe': 'roadmap', 'params': {'mode': mode, 'groups': [{'label': 'a', 'count': 8}]}}
        manifest['pages'][0]['overlays'] = [roadmap('full')]
        for page in manifest['pages'][1:]:
            page['overlays'] = [roadmap('strip'), {'recipe': 'kinetic_title', 'params': {'text': '做一件事', 'badge': 1}}]
        for index in (5, 8):
            manifest['pages'][index]['overlays'].append(roadmap('full'))
        (root / 'job.json').write_text(json.dumps(manifest, ensure_ascii=False))
        report = director_lint.check(root)
        self.assertFalse([item for item in report['errors'] if 'roadmap' in item or 'progress strip' in item], report['errors'])

    def test_words_that_call_for_a_form_are_flagged_when_the_page_lacks_it(self):
        root = job(['AI 写的代码，他基本不审了。', '你会雇一个便宜一半、却随机笨两成的人吗？'])
        self.assertTrue(any('a comparison' in item for item in director_lint.check(root)['warnings']))
        manifest = json.loads((root / 'job.json').read_text())
        manifest['pages'][1]['overlays'] = [{'recipe': 'table', 'params': {}}]
        (root / 'job.json').write_text(json.dumps(manifest, ensure_ascii=False))
        report = director_lint.check(root)
        self.assertFalse(any('b1' in item or 'page 2' in item for item in report['warnings'] if 'comparison' in item))
        self.assertEqual(report['forms'], ['table'])

    def test_a_custom_animation_that_explains_a_mechanism_is_a_form(self):
        root = job(['车灯只照五十米，你却能开完全程。', '你往前开一米，光就往前推一米。'])
        manifest = json.loads((root / 'job.json').read_text())
        manifest['pages'][1]['overlays'] = [{'recipe': 'custom', 'params': {'script': '...'}}]
        (root / 'job.json').write_text(json.dumps(manifest, ensure_ascii=False))
        self.assertEqual(director_lint.check(root)['forms'], [])                      # decoration is not a form
        manifest['pages'][1]['overlays'][0]['params'].update(element='跟着车走的光束', explains='光随车往前推')
        (root / 'job.json').write_text(json.dumps(manifest, ensure_ascii=False))
        self.assertEqual(director_lint.check(root)['forms'], ['custom:跟着车走的光束'])     # each custom animation is its own form

    def test_english_words_in_the_narration_are_listed_before_voicing(self):
        report = director_lint.check(job(['AI 写的代码，他基本不审了。', '处理 bug 报告，几个 Agent 同时跑。']))
        listed = [item for item in report['warnings'] if 'English words' in item]
        self.assertTrue(listed and 'bug ×1' in listed[0] and 'ai ×1' in listed[0])
        self.assertFalse([item for item in director_lint.check(job(['代码他基本不审了。', '处理问题报告。']))['warnings'] if 'English words' in item])


if __name__ == '__main__':
    unittest.main()
