"""v3 text breaks: Chinese wraps between words in boxes, titles and subtitles (never "对面司/机", "Opus/4.5？",
"托尔斯滕·/鲍尔"); no line opens with closing punctuation; no lone word on a last line; the glyph_in title style; the
roadmap strip that stays in place across a page turn."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import caption_assets
import web_motion

NO_START = caption_assets.NO_START
NAME = '托尔斯滕·鲍尔'
SENTENCE = '这是托尔斯滕·鲍尔今年十月分享的做法，不是标准答案。'
# the failing cases of the 2026-11-02 frame audit, with the boxes the films used
CALLOUT = ('callout', {'text': '现在谁还在用 Opus 4.5？', 'size': 46, 'tail': 'bottom-right', 'fill': '#FFFFFF'}, (520, 150))
TABLE = ('table', {'font': 'hand', 'columns': ['国标测试点', '亮度要求', '为了'], 'widths': [0.44, 0.26, 0.3], 'size': 38, 'first_left': True,
                   'rows': [['前方 75 米右侧', '≥ 12 lx', '看得见路'], ['前方 50 米左侧（对面司机）', '≤ 15 lx', '不晃对面']], 'row_ms': [100, 200]}, (1126, 380))
FLOW = ('flow', {'font': 'hand', 'size': 42, 'node_size': [0.28, 0.4], 'nodes': [{'id': 'a', 'text': '市里组建援非医疗队', 'at': [0.16, 0.5], 'ms': 0},
                 {'id': 'b', 'text': '各医院抽调医生', 'at': [0.5, 0.5], 'ms': 0}, {'id': 'c', 'text': '阿姨：报名！', 'at': [0.84, 0.5], 'ms': 0}],
                 'edges': [{'from': 'a', 'to': 'b', 'ms': 0}]}, (1200, 400))
ENDING = {'text': '存款不到七十万的你，会先做哪一条？', 'keywords': ['哪一条'], 'size': 66, 'style': 'word_by_word', 'font': 'hand'}
OPENING = {'text': '存款不到七十万，请照做？', 'keywords': ['七十万'], 'size': 92, 'font': 'hand', 'color': '#22312B', 'accent': '#C4553E'}


def cuts(pieces):
    """Offsets where consecutive pieces meet."""
    at, found = 0, []
    for piece in pieces[:-1]:
        at += len(piece)
        found.append(at)
    return found


def inside(text, word):
    """Offsets strictly inside every occurrence of word."""
    found, start = set(), text.find(word)
    while start >= 0:
        found.update(range(start + 1, start + len(word)))
        start = text.find(word, start + 1)
    return found


class SubtitleBreakTests(unittest.TestCase):
    def test_a_name_joined_by_a_middle_dot_is_one_unit(self):
        spans = lambda text: [text[a:b] for a, b in caption_assets.name_spans(text)]
        self.assertEqual(spans(SENTENCE), [NAME])
        self.assertEqual(spans('托尔斯滕·鲍尔。'), [NAME])
        self.assertEqual(spans('Jean·Paul 说'), ['Jean·Paul'])
        self.assertEqual(spans('说明文件 · 技能 · 脚本'), [])   # a spaced dot separates a list, it joins nothing

    def test_cues_and_lines_never_split_a_name(self):
        # the film cut "这是托尔斯滕·" | "鲍尔今年十月…" (the old cost after · was a cheap 6)
        for limit in range(7, 16):
            for pieces in (caption_assets.cue_pieces(SENTENCE, limit), caption_assets.split_text(SENTENCE, limit)):
                self.assertEqual(''.join(pieces), SENTENCE)
                self.assertTrue(any(NAME in piece for piece in pieces), (limit, pieces))
        # a name longer than the whole line has to break: after its dot, not inside a part
        self.assertIn('托尔斯滕·', caption_assets.split_text(SENTENCE, 6))

    def test_without_jieba_the_name_still_stays_whole(self):
        saved = caption_assets.USE_JIEBA
        caption_assets.USE_JIEBA = False
        try:
            for limit in range(9, 16):
                self.assertTrue(any(NAME in piece for piece in caption_assets.split_text(SENTENCE, limit)), limit)
        finally:
            caption_assets.USE_JIEBA = saved

    def test_no_piece_opens_with_closing_punctuation_or_splits_a_word_or_number(self):
        texts = {'现在谁还在用 Opus 4.5？对吧？': ['Opus', '4.5？'], '前方 50 米左侧（对面司机），亮度不超过 15 lx。': ['（对面', '司机），', '15'],
                 '增长了 70%，约 3.5 万元。': ['70%，', '3.5'], 'GPT-5 和 Claude Code 的对比，结果出乎意料。': ['GPT-5', 'Claude', 'Code', '对比，'],
                 SENTENCE: [NAME, '做法，']}
        for text, words in texts.items():
            for limit in range(5, 15):
                for pieces in (caption_assets.split_text(text, limit), caption_assets.cue_pieces(text, limit)):
                    self.assertEqual(''.join(pieces), text)
                    for piece in pieces[1:]:
                        self.assertNotIn(piece.lstrip()[:1], NO_START, (text, limit, pieces))
                    locked = set().union(*(inside(text, word) for word in words if caption_assets.shown_width(word, False) <= limit))
                    self.assertFalse(locked & set(cuts(pieces)), (text, limit, pieces))

    def test_the_last_resort_wrap_keeps_closing_punctuation_on_its_line(self):
        fits = lambda line: len(line) <= 6
        lines = caption_assets.character_wrap('现在谁还在用 Opus 4.5？对吧，好的。', fits)
        self.assertEqual(''.join(lines), '现在谁还在用 Opus 4.5？对吧，好的。')
        self.assertTrue(all(line[:1] not in NO_START for line in lines), lines)
        self.assertEqual(caption_assets.character_wrap('现在谁还在用 Opus 4.5？对吧，好的。', fits, kinsoku=False)[-1], '，好的。')  # the old wrap


LINES_JS = r"""
() => {   // every block of text in the overlay and how it wraps: its lines, each with its own spaces
  const out = [];
  for (const el of document.querySelectorAll('#root *')) {
    if (!['block', 'flex'].includes(getComputedStyle(el).display)) continue;
    const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT), chars = [];
    for (let n = walker.nextNode(); n; n = walker.nextNode()) {
      let owner = n.parentElement;
      while (owner && owner !== el && !['block', 'flex', 'inline-flex'].includes(getComputedStyle(owner).display)) owner = owner.parentElement;
      if (owner !== el) continue;
      for (let i = 0; i < n.textContent.length; i++) {
        const r = document.createRange(); r.setStart(n, i); r.setEnd(n, i + 1);
        const rect = [...r.getClientRects()].pop();
        if (rect) chars.push([n.textContent[i], rect.top, rect.bottom, rect.right]);
      }
    }
    if (!chars.some((c) => c[0].trim())) continue;
    const lines = []; let top = null;
    for (const [c, t, b] of chars) {
      if (top === null || t > top + (b - t) * 0.5) { lines.push(''); top = t; }
      lines[lines.length - 1] += c;
    }
    out.push({ lines, right: Math.max(...chars.map((c) => c[3])) });
  }
  return out;
}"""
ROWS_JS = r"""
() => { const box = document.querySelector('#root > div'), rows = {};
  for (const item of box.children) (rows[Math.round(item.offsetTop)] ||= []).push(item.textContent);
  return Object.keys(rows).sort((a, b) => a - b).map((k) => rows[k].join('')); }"""


@unittest.skipUnless(web_motion.available()[0], 'Playwright + Chromium not installed (setup.sh --web-motion)')
class BoxAndTitleBreakTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from playwright.sync_api import sync_playwright
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch()
        cls.directory = Path(tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()

    def open(self, recipe, params, size, seconds=1.6):
        params = web_motion.prepare(recipe, params)
        path = self.directory / f'{recipe}.html'
        path.write_text(web_motion.page_html(recipe, params, size[0], size[1], 1., seconds, 1))
        page = self.browser.new_page(viewport={'width': size[0], 'height': size[1]})
        page.add_init_script(path=str(web_motion.ASSETS / 'shim.js'))
        page.goto(path.as_uri())
        page.wait_for_load_state('load')
        self.assertIsNone(page.evaluate('window.__error'))
        self.addCleanup(page.close)
        return page

    def blocks(self, case):
        recipe, params, size = case
        return self.open(recipe, params, size).evaluate(LINES_JS)

    def assertWrapsBetweenWords(self, blocks, words):
        for block in blocks:
            text = ''.join(block['lines'])
            for line in block['lines'][1:]:
                self.assertNotIn(line.strip()[:1], NO_START, block['lines'])
            for word in words:
                if word in text:
                    self.assertTrue(any(word in line for line in block['lines']), (word, block['lines']))

    def test_the_audited_boxes_wrap_between_words(self):
        callout, table, flow = (self.blocks(case) for case in (CALLOUT, TABLE, FLOW))
        self.assertWrapsBetweenWords(callout, ['Opus 4.5？', '现在'])
        self.assertWrapsBetweenWords(table, ['对面司机）', '50 米', '≤ 15 lx', '≥ 12 lx'])
        self.assertWrapsBetweenWords(flow, ['医疗队', '援非', '组建'])
        if sys.platform == 'darwin':   # the fonts the films were made with: the exact break positions
            lines = lambda blocks: [[line.strip() for line in block['lines']] for block in blocks if len(block['lines']) > 1]
            self.assertEqual(lines(callout), [['现在谁还在用', 'Opus 4.5？']])
            self.assertEqual(lines(table), [['前方 50 米左侧', '（对面司机）']])
            self.assertEqual(lines(flow), [['市里组建', '援非医疗队']])

    def test_word_units_keep_names_numbers_and_punctuation_together(self):
        page = self.open(*CALLOUT)
        texts = [SENTENCE, '现在谁还在用 Opus 4.5？', '增长了 70%，约 3.5 万元', 'GPT-5 和 Claude Code 的对比', '前方 50 米左侧（对面司机），≥ 12 lx',
                 '市里组建援非医疗队', '「好」，他说。马丁·路德·金']
        units = {text: page.evaluate('(t) => window.RecipeText.wordUnits(t)', text) for text in texts}
        for text, found in units.items():
            self.assertEqual(''.join(found), text)
            for unit in found[1:]:
                self.assertNotIn(unit[:1], NO_START, found)
            for unit in found[:-1]:
                self.assertNotIn(unit[-1:], caption_assets.NO_END, found)
        has = lambda text, unit: any(unit in found for found in units[text])
        self.assertTrue(has(SENTENCE, NAME))
        self.assertIn('Opus 4.5？', units['现在谁还在用 Opus 4.5？'])
        self.assertIn('70%，', units['增长了 70%，约 3.5 万元'])
        self.assertIn('3.5 万元', units['增长了 70%，约 3.5 万元'])
        self.assertIn('GPT-5', units['GPT-5 和 Claude Code 的对比'])
        self.assertTrue(has('前方 50 米左侧（对面司机），≥ 12 lx', '≥ 12 lx') and has('前方 50 米左侧（对面司机），≥ 12 lx', '50 米'))
        self.assertTrue(has('市里组建援非医疗队', '医疗队') and has('市里组建援非医疗队', '援非'))
        self.assertTrue(has('「好」，他说。马丁·路德·金', '马丁·路德·金'))

    def test_a_unit_wider_than_its_box_still_stays_inside(self):
        blocks = self.blocks(('callout', {'text': 'https://example.com/a/very/long/path/that/cannot/fit', 'size': 40}, (300, 300)))
        self.assertGreater(len(blocks[0]['lines']), 1)
        self.assertLessEqual(blocks[0]['right'], 300)

    def test_a_wrapped_title_is_balanced_and_never_breaks_inside_a_word(self):
        page = self.open('kinetic_title', ENDING, (1126, 170))
        seen = []
        for _ in range(6):   # word_by_word lands its key word at 1.45x: the lines must not jump while it shrinks
            seen.append(page.evaluate(ROWS_JS))
            page.evaluate('window.__advance(300)')
        self.assertTrue(all(rows == seen[0] for rows in seen), seen)
        self.assertEqual(len(seen[0]), 2)
        self.assertGreater(len(seen[0][-1]), 3, seen[0])   # the audit: a last line of only "哪一条？"
        if sys.platform == 'darwin':
            self.assertEqual(seen[0], ['存款不到七十万的你，', '会先做哪一条？'])
        # glyph styles used to wrap between any two glyphs ("开/发")
        rows = self.open('kinetic_title', {'text': '让它看清整个开发流程', 'size': 80, 'style': 'spring_pop'}, (600, 200)).evaluate(ROWS_JS)
        self.assertGreater(len(rows), 1)
        self.assertTrue(set(cuts(rows)) <= {2, 4, 6, 8}, rows)


@unittest.skipUnless(web_motion.available()[0], 'Playwright + Chromium not installed (setup.sh --web-motion)')
class GlyphInAndSettledStripTests(unittest.TestCase):
    def frames(self, recipe, params, size, seconds, fps=30):
        result = web_motion.render(recipe, params, size[0], size[1], seconds, self.cache, fps=fps)
        return [np.asarray(Image.open(Path(result['directory']) / f'{index:05d}.png').convert('RGBA')).astype(int) for index in range(result['frames'])]

    def setUp(self):
        self.cache = Path(tempfile.mkdtemp())

    @staticmethod
    def partial_frames(frames):
        """Frames in which a visible glyph shows under 80 % of its settled height (cut by a mask)."""
        final = frames[-1][..., 3]
        ink = list((final > 40).any(axis=0)) + [False]
        cells, start = [], None
        for x, on in enumerate(ink):
            if on and start is None:
                start = x
            elif not on and start is not None:
                cells.append((max(0, start - 3), x + 3))
                start = None
        def extent(alpha):
            rows = np.nonzero((alpha > 0.35 * alpha.max()).any(axis=1))[0]
            return rows[-1] - rows[0] + 1 if len(rows) else 0
        settled = [extent(final[:, a:b]) for a, b in cells]
        return [index for index, frame in enumerate(frames)
                if any(frame[:, a:b, 3].max() >= 25 and extent(frame[:, a:b, 3]) < 0.8 * full for (a, b), full in zip(cells, settled))]

    def test_glyph_in_never_shows_part_of_a_glyph(self):
        glyph_in = self.frames('kinetic_title', {**OPENING, 'style': 'glyph_in'}, (1126, 170), 1.4)
        mask_up = self.frames('kinetic_title', {**OPENING, 'style': 'mask_up'}, (1126, 170), 1.4)
        self.assertGreater(len(self.partial_frames(mask_up)), 3)   # the check sees a mask's half glyphs
        self.assertEqual(self.partial_frames(glyph_in), [])
        # glyph by glyph: the visible ink grows in steps, and the title rests exactly where mask_up's does
        ink = [(frame[..., 3] > 40).sum() for frame in glyph_in]
        final = ink[-1]
        self.assertLess(ink[3], 0.25 * final)
        self.assertTrue(any(0.3 * final < value < 0.7 * final for value in ink))
        self.assertTrue(np.array_equal(glyph_in[-1], glyph_in[-2]), 'still moving when it is held')
        self.assertTrue(np.array_equal(glyph_in[-1], mask_up[-1]))
        # the key word keeps its accent colour
        accent = np.array([0xC4, 0x55, 0x3E])
        near = lambda frame: ((np.abs(frame[..., :3] - accent).sum(axis=2) < 60) & (frame[..., 3] > 200)).sum()
        self.assertGreater(near(glyph_in[-1]), 500)
        self.assertEqual(near(glyph_in[2]), 0)

    def test_a_settled_roadmap_strip_is_in_place_on_the_first_frame(self):
        strip = {'mode': 'strip', 'groups': [{'label': 'g', 'count': 5}] * 5 + [{'label': 'g', 'count': 4}], 'done': 5, 'active': [5, 10],
                 'caption': '② 用起 AI', 'size': 34, 'ink': '#22312B', 'accent': '#C4553E'}
        default = self.frames('roadmap', strip, (1126, 56), 1.6, fps=15)
        settled = self.frames('roadmap', {**strip, 'enter': 'settled'}, (1126, 56), 1.6, fps=15)
        ink = lambda frame: (frame[..., 3] > 40).sum()
        self.assertLess(ink(default[0]), 0.05 * ink(default[-1]))   # the old page turn: an empty strip
        self.assertGreater(ink(settled[0]), 0.6 * ink(settled[-1]))
        # the done group does not move at all; the page's own group arrives in the accent colour
        self.assertTrue(np.array_equal(settled[0][:, :150], settled[-1][:, :150]))
        accent = np.array([0xC4, 0x55, 0x3E])
        lit = lambda frame: ((np.abs(frame[..., :3] - accent).sum(axis=2) < 60) & (frame[..., 3] > 200)).sum()
        self.assertEqual(lit(settled[0]), 0)
        self.assertGreater(lit(settled[-1]), 200)
        self.assertTrue(np.array_equal(settled[-1], settled[-2]), 'still moving when it is held')


if __name__ == '__main__':
    unittest.main()
