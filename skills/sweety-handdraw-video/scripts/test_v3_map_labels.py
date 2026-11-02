"""map_route labels and zoom, kinetic_title keyword colours. The history film's audit (2026-11-02): 大宛, 康居, 大月氏 and
大夏 piled into one black knot of labels, 匈奴 sat on 长安 and read "匈奴安", and a five-stop route across Central Asia was
about 100 px long on a half globe. Now each label takes a free spot (with a leader line when it moves away from its dot),
a short route zooms in to fill the box, and labels are at least 34 px in a 1126 px box."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import web_motion

INK = '#1F1A17'
CHANGAN, XIONGNU, DUNHUANG, KASHGAR = [108.94, 34.34], [102.6, 37.9], [94.66, 40.14], [75.99, 39.47]
DAYUAN, KANGJU, YUEZHI, DAXIA = [70.94, 40.53], [69.2, 41.3], [67.28, 37.22], [66.9, 36.76]
WEST = [CHANGAN, XIONGNU, DUNHUANG, KASHGAR, YUEZHI]


def stops(*pairs):
    return [{'at': at, 'label': label} for at, label in pairs]


# the film's three maps, in the boxes it used
CASES = {
    'four countries': ((1126, 720), {'route': [DAYUAN, KANGJU, YUEZHI, DAXIA], 'travel_ms': 2200,
                                     'stops': stops((DAYUAN, '大宛'), (KANGJU, '康居'), (YUEZHI, '大月氏'), (DAXIA, '大夏'))}),
    'xiongnu on changan': ((1126, 900), {'route': WEST, 'travel_ms': 2600, 'stops': stops((CHANGAN, '长安'), (XIONGNU, '匈奴'), (YUEZHI, '大月氏'))}),
    'five stops': ((1126, 760), {'route': WEST, 'travel_ms': 3000,
                                 'stops': stops((CHANGAN, '长安'), (XIONGNU, '河西'), (DUNHUANG, '敦煌'), (KASHGAR, '喀什'), (YUEZHI, '大月氏'))}),
}
BASE = {'ink': INK, 'font': 'hand', 'projection': 'globe'}
# a long route keeps the globe that turns under the marker: labels must ride with their dots
TURNING = {'pacific': ((1126, 760), {'route': [[121.5, 31.2], [139.7, 35.7], [-122.4, 37.8]], 'travel_ms': 2600,
                                     'stops': stops(([121.5, 31.2], '上海'), ([139.7, 35.7], '东京'), ([-122.4, 37.8], '旧金山'))})}


def apart(a, b, gap=0.):
    return a[2] + gap <= b[0] or b[2] + gap <= a[0] or a[3] + gap <= b[1] or b[3] + gap <= a[1]


def clear_of(box, x, y, r):
    return np.hypot(max(box[0] - x, 0, x - box[2]), max(box[1] - y, 0, y - box[3])) >= r


INK_IN_BOXES = r"""
() => { const canvas = document.querySelector('#root canvas'), g = canvas.getContext('2d');
  return document.getElementById('root').__mapLabels.map(({ box }) => {
    const [x0, y0] = [Math.ceil(box[0]), Math.ceil(box[1])], w = Math.floor(box[2]) - x0, h = Math.floor(box[3]) - y0;
    const data = g.getImageData(x0, y0, w, h).data;
    let dark = 0;
    for (let i = 0; i < data.length; i += 4) if (data[i + 3] > 200 && data[i] + data[i + 1] + data[i + 2] < 240) dark++;
    return dark; }); }"""


@unittest.skipUnless(web_motion.available()[0], 'Playwright + Chromium not installed (setup.sh --web-motion)')
class MapLabelTests(unittest.TestCase):
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

    def open(self, params, size, seconds=5.):
        path = self.directory / 'map.html'
        path.write_text(web_motion.page_html('map_route', params, size[0], size[1], 1., seconds, 1))
        page = self.browser.new_page(viewport={'width': size[0], 'height': size[1]})
        page.add_init_script(path=str(web_motion.ASSETS / 'shim.js'))
        page.goto(path.as_uri())
        page.wait_for_load_state('load')
        self.assertIsNone(page.evaluate('window.__error'))
        self.addCleanup(page.close)
        return page

    def assertLabelsClear(self, labels, geometry, size, where):
        width, height = size
        for i, item in enumerate(labels):
            box = item['box']
            self.assertTrue(0 < box[0] and box[2] < width and 0 < box[1] and box[3] < height, (where, item))
            for other in labels[i + 1:]:
                self.assertTrue(apart(box, other['box']), (where, item['label'], other['label']))
            for other in labels:   # every shown stop's dot, its own included
                self.assertTrue(clear_of(box, *other['dot'], geometry['dot'] - 1), (where, item['label'], 'covers the dot of', other['label']))
            self.assertFalse(clear_of(box, *item['dot'], geometry['dot'] + 4.5 * geometry['size']), (where, item['label'], 'far from its dot'))

    def test_clustered_labels_never_overlap_each_other_the_dots_or_the_marker(self):
        for name, (size, params) in {**CASES, **TURNING}.items():
            page = self.open({**BASE, **params}, size)
            geometry = page.evaluate("document.getElementById('root').__map")
            for step in range(52):   # the whole travel and a held second, every 100 ms
                page.evaluate('window.__advance(100)')
                labels = page.evaluate("document.getElementById('root').__mapLabels") or []
                marker = page.evaluate("document.getElementById('root').__mapMarker")
                self.assertLabelsClear(labels, geometry, size, (name, step))
                for item in labels:
                    self.assertTrue(clear_of(item['box'], marker[0], marker[1], marker[2] - 1), (name, step, item['label'], 'under the marker'))
            # settled: every stop labelled (on the turning globe the first ones have gone round the back), fully shown, its
            # ink drawn where it is reported
            if name in CASES:
                self.assertEqual(sorted(item['label'] for item in labels), sorted(stop['label'] for stop in params['stops']), name)
            self.assertTrue(labels, name)
            self.assertTrue(all(item['pop'] == 1 for item in labels), name)
            self.assertTrue(all(dark > 150 for dark in page.evaluate(INK_IN_BOXES)), name)

    def test_labels_are_readable_on_a_phone(self):
        for name, (size, params) in CASES.items():
            self.assertGreaterEqual(self.open({**BASE, **params}, size).evaluate("document.getElementById('root').__map.size"), 34, name)
        sized = self.open({**BASE, **CASES['four countries'][1], 'label_size': 24}, (1126, 720))
        self.assertEqual(sized.evaluate("document.getElementById('root').__map.size"), 24)

    def test_a_label_pushed_off_its_dot_gets_a_leader_line(self):
        # three stops a hair apart: at least one label has to stand away from its dot
        crowd = {**BASE, 'route': [[70, 40], [70.15, 40.1], [70.3, 40.]], 'travel_ms': 600, 'zoom': 2,
                 'stops': stops(([70, 40], '甲地'), ([70.15, 40.1], '乙地'), ([70.3, 40.], '丙地'))}
        page = self.open(crowd, (1126, 720))
        page.evaluate('window.__advance(3000)')
        labels = page.evaluate("document.getElementById('root').__mapLabels")
        self.assertEqual(len(labels), 3)
        self.assertLabelsClear(labels, page.evaluate("document.getElementById('root').__map"), (1126, 720), 'crowd')
        self.assertTrue(any(item['leader'] for item in labels))

    def test_a_long_route_keeps_the_turning_globe(self):
        page = self.open({**BASE, 'route': [[121.5, 31.2], [-122.4, 37.8]], 'stops': stops(([121.5, 31.2], '上海'))}, (1126, 760))
        self.assertEqual(page.evaluate("document.getElementById('root').__map.zoom"), 1)


@unittest.skipUnless(web_motion.available()[0], 'Playwright + Chromium not installed (setup.sh --web-motion)')
class RenderedMapAndTitleTests(unittest.TestCase):
    def setUp(self):
        self.cache = Path(tempfile.mkdtemp())

    def last(self, recipe, params, size, seconds, fps=10):
        result = web_motion.render(recipe, params, size[0], size[1], seconds, self.cache, fps=fps)
        self.assertFalse(result['clipped'], f'{recipe} runs off its box')
        return np.asarray(Image.open(Path(result['directory']) / f'{result["frames"] - 1:05d}.png').convert('RGBA')).astype(int)

    def test_a_short_route_zooms_in_to_fill_the_box(self):
        red = lambda frame: (frame[..., 0] > 230) & (frame[..., 1] < 40) & (frame[..., 2] < 40) & (frame[..., 3] > 200)
        def extent(frame):
            ys, xs = np.nonzero(red(frame))
            return max(xs.max() - xs.min(), ys.max() - ys.min())
        for name, (size, params) in CASES.items():
            route = {**BASE, **params, 'color': '#FF0000', 'locator': False}   # the locator outlines its patch in the route colour
            seconds = (1000 + params['travel_ms'] + 400) / 1000
            zoomed, whole = extent(self.last('map_route', route, size, seconds)), extent(self.last('map_route', {**route, 'zoom': False}, size, seconds))
            self.assertGreaterEqual(zoomed, 0.45 * min(size), name)
            self.assertLess(whole, 0.3 * min(size), name)   # the film: a short stub on a half globe

    def test_keyword_colors_render_in_every_style(self):
        red, blue, amber = np.array([0xE0, 0x4B, 0x3C]), np.array([0x3E, 0x7C, 0xC4]), np.array([0xC7, 0x7F, 0x1B])
        near = lambda frame, colour: ((np.abs(frame[..., :3] - colour).sum(axis=2) < 60) & (frame[..., 3] > 200)).sum()
        title = {'text': '红的、蓝的和绿的', 'keywords': ['红的', '蓝的', '绿的'], 'keyword_colors': {'红的': '#E04B3C', '蓝的': '#3E7CC4'},
                 'accent': '#C77F1B', 'color': INK, 'size': 72}
        for style in ('glyph_in', 'blur_up', 'spring_pop', 'mask_up', 'typewriter', 'slam', 'scramble', 'word_by_word'):
            frame = self.last('kinetic_title', {**title, 'style': style}, (1126, 170), 2.6)
            for colour, what in ((red, '红的'), (blue, '蓝的'), (amber, '绿的 keeps the accent')):
                self.assertGreater(near(frame, colour), 300, (style, what))
        # a word listed only in keyword_colors is still coloured
        frame = self.last('kinetic_title', {**title, 'keywords': [], 'style': 'glyph_in'}, (1126, 170), 2.0)
        self.assertGreater(near(frame, red), 300)
        self.assertGreater(near(frame, blue), 300)
        self.assertEqual(near(frame, amber), 0)


if __name__ == '__main__':
    unittest.main()
