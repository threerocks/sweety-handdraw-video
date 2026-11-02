"""v3 explanation forms: roadmap, table, flow, swimlane, checklist, cards, pulse, and the numbered point title."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import web_motion

INK, AMBER = '#2B2F36', '#C77F1B'
GROUPS = [{'label': '换心态', 'count': 3}, {'label': '给眼睛和手', 'count': 4}, {'label': '换合作方式', 'count': 4}, {'label': '只审核心', 'count': 4}]
CASES = [
    ('roadmap', {'groups': GROUPS, 'done': 3, 'focus': 1}, (1126, 420)),
    ('roadmap', {'mode': 'strip', 'groups': GROUPS, 'done': 4, 'current': 4, 'caption': '② 给它眼睛和手', 'size': 34}, (1126, 70)),
    # 29 items in six groups and a caption (the deposit film): the dots shrink to fit, nothing runs off the 56 px band
    ('roadmap', {'mode': 'strip', 'groups': [{'label': 'g', 'count': 5}] * 5 + [{'label': 'g', 'count': 4}], 'done': 10, 'active': [10, 15],
                 'caption': '③ 持续输出，攒作品', 'size': 34}, (1126, 56)),
    ('roadmap', {'groups': [{'label': '通道', 'count': 5, 'numbers': [1, 2, 3, 4, 5]}, {'label': 'AI', 'count': 5, 'numbers': [9, 11, 12, 16, 17]}],
                 'done': 5, 'active': [5, 10], 'focus': 1}, (1126, 420)),
    ('table', {'columns': ['', '最强模型', '便宜一半'], 'rows': [['价格', '1×', '0.5×'], ['值得雇吗', '✓', '✗']], 'highlight': [{'row': 1, 'col': 2, 'ms': 600}],
               'row_ms': [300, 700]}, (1126, 360)),
    ('flow', {'nodes': [{'id': 'a', 'text': '写代码', 'at': [.2, .5]}, {'id': 'b', 'text': '部署', 'at': [.8, .5], 'shape': 'circle', 'size': [.2, .4]}],
              'edges': [{'from': 'a', 'to': 'b', 'text': '跑通'}], 'tokens': [{'path': ['a', 'b'], 'ms': 600, 'edge_ms': 400}],
              'shake': [{'id': 'b', 'ms': 900, 'mark': True}]}, (1126, 300)),
    ('swimlane', {'title': '同步', 'lanes': [{'label': '人'}, {'label': 'Agent'}], 'span': 4, 'sweep_ms': 900,
                  'blocks': [{'lane': 0, 'from': 0, 'to': 1, 'text': '问'}, {'lane': 1, 'from': 1, 'to': 3, 'text': '做'},
                             {'lane': 0, 'from': 1, 'to': 3, 'kind': 'wait', 'text': '等'}]}, (1126, 300)),
    ('checklist', {'items': [{'text': '处理 bug 报告', 'mark': 'check', 'ms': 100}, {'text': '烂摊子', 'mark': 'strike', 'ms': 300}]}, (1126, 240)),
    ('cards', {'count': 6, 'columns': 3, 'flicker': {'indices': [1, 4], 'step_ms': 200, 'ms': 300, 'question': True},
               'groups': [{'indices': [0, 1], 'label': 'A', 'x': .25, 'ms': 900}, {'indices': [2, 3, 4, 5], 'label': 'B', 'x': .7, 'ms': 900}]}, (1126, 380)),
    ('pulse', {'beats': [{'ms': 300, 'label': '两小时后'}, {'ms': 900, 'label': '每天 9 点'}], 'sweep_ms': 1200}, (1126, 320)),
    ('kinetic_title', {'text': '让它看清整个开发流程', 'badge': 4, 'keywords': ['看清'], 'size': 76, 'style': 'mask_up'}, (1126, 170)),
]


@unittest.skipUnless(web_motion.available()[0], 'Playwright + Chromium not installed (setup.sh --web-motion)')
class FormTests(unittest.TestCase):
    def test_every_form_renders_inside_its_box_and_settles(self):
        cache = Path(tempfile.mkdtemp())
        for recipe, params, (width, height) in CASES:
            result = web_motion.render(recipe, {'ink': INK, 'accent': AMBER, **params}, width, height, 2.2, cache, fps=15)
            last, before = (np.asarray(Image.open(Path(result['directory']) / f'{index:05d}.png'))[..., 3].astype(int)
                            for index in (result['frames'] - 1, result['frames'] - 2))
            self.assertGreater((last > 0).sum(), 500, recipe)
            self.assertFalse(result['clipped'], f'{recipe} {params.get("mode", "")} runs off its box')
            self.assertEqual(np.abs(last - before).sum(), 0, f'{recipe} still moving when it is held')

    def test_a_typed_english_line_keeps_its_spaces(self):
        cache = Path(tempfile.mkdtemp())
        def width(text):
            result = web_motion.render('kinetic_title', {'text': text, 'style': 'typewriter', 'size': 60, 'color': INK}, 1126, 170, 1.2, cache, fps=15)
            ink = np.asarray(Image.open(Path(result['directory']) / f'{result["frames"] - 1:05d}.png'))[..., 3] > 0
            columns = np.nonzero(ink.any(axis=0))[0]
            return columns[-1] - columns[0]
        # per character, "It's like" typed as "It'slike": the spaces were dropped, so both lines came out the same width
        self.assertGreater(width('aa bb cc') - width('aabbcc'), 2 * 0.2 * 60)

    def test_cards_split_into_groups_and_a_flow_token_travels(self):
        cache = Path(tempfile.mkdtemp())
        cards = web_motion.render('cards', {'count': 4, 'columns': 4, 'groups': [{'indices': [0, 1], 'label': 'A', 'x': .2, 'ms': 800},
                                                                                {'indices': [2, 3], 'label': 'B', 'x': .75, 'ms': 800}]},
                                  1000, 300, 2.0, cache, fps=10)
        columns = lambda index: np.flatnonzero(np.asarray(Image.open(Path(cards['directory']) / f'{index:05d}.png'))[150:260, :, 3].max(0) > 0)
        early, late = columns(7), columns(cards['frames'] - 1)
        gap = lambda xs: np.diff(xs).max()
        self.assertGreater(gap(late), gap(early) + 60)   # a wide gap opens between the two groups
        flow = web_motion.render('flow', {'nodes': [{'id': 'a', 'text': 'A', 'at': [.15, .5]}, {'id': 'b', 'text': 'B', 'at': [.85, .5]}],
                                          'edges': [{'from': 'a', 'to': 'b'}], 'tokens': [{'path': ['a', 'b'], 'ms': 700, 'edge_ms': 600, 'color': '#FF0000'}]},
                                 1000, 200, 1.6, cache, fps=10)
        red = lambda index: np.flatnonzero((np.asarray(Image.open(Path(flow['directory']) / f'{index:05d}.png'))[..., :3] == [255, 0, 0]).all(2).any(0))
        self.assertLess(red(8).mean(), red(13).mean())   # the token moves from A toward B


class MomentTests(unittest.TestCase):
    def test_a_forms_moments_are_its_scheduled_additions_not_its_durations(self):
        from video_job import scheduled_moments
        params = {'row_ms': [300, 900], 'sweep_ms': 4000, 'items': [{'ms': 100, 'mark': 'check', 'mark_ms': 1200}],
                  'tokens': [{'ms': 2000, 'edge_ms': 500}], 'stagger': 80}
        self.assertEqual(sorted(scheduled_moments(params)), [100., 300., 900., 1200., 2000.])
        # A custom script's timing names are its own (react_ms, brake_ms): it lists the times it adds something as moments.
        self.assertEqual(scheduled_moments({'react_ms': 500, 'drive_ms': 3000, 'moments': [500, 2600]}), [500., 2600.])


if __name__ == '__main__':
    unittest.main()
