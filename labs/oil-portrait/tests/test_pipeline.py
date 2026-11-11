"""Tests for the parts of the portrait pipeline that used to need a person's (or a model's) judgement.

    python3 labs/oil-portrait/tests/test_pipeline.py          # a few minutes; no image generation

They use the s7 pictures in the local cache (output/, not in git: real photos stay off the public repo) and skip
when those are missing. What they guard:
- the prompts make.py sends have no per-photo blanks to fill in;
- automatic annotations land on the eyes (within 25 px of the hand annotations of the best films) and pass sanity;
- a generated picture that is shifted is put back on the photo (face landmarks; line drawings by edges);
- a broken area map or a picture without the face is refused;
- the photo crop keeps the face;
- the acceptance checks measure what they say (hand motion, eyes, the best films pass).
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
PY = Path.home() / '.local/share/sweety-handdraw-video/venv/bin/python'
if Path(sys.executable).resolve() != PY.resolve() and PY.exists():
    os.execv(str(PY), [str(PY), __file__] + sys.argv[1:])
sys.path.insert(0, str(HERE))

import numpy as np                     # noqa: E402
from PIL import Image                  # noqa: E402

import annotate                        # noqa: E402
import check                           # noqa: E402
import make                            # noqa: E402

S7 = ROOT / 'output/oil/s7'
CACHE = None
for p in (ROOT / 'output/portrait/.assets').glob('*/target-oil.png'):
    CACHE = p.parent
    break
RESULTS = []


def test(fn):
    def run():
        try:
            fn()
            RESULTS.append((fn.__name__, 'PASS', ''))
        except SkipTest as e:
            RESULTS.append((fn.__name__, 'SKIP', str(e)))
        except AssertionError as e:
            RESULTS.append((fn.__name__, 'FAIL', str(e)))
        except make.Stop as e:
            RESULTS.append((fn.__name__, 'FAIL', 'Stop: ' + str(e)[:300]))
    run.__name__ = fn.__name__
    return run


class SkipTest(Exception):
    pass


def need(path):
    if path is None or not Path(path).exists():
        raise SkipTest(f'missing {path} (local pictures are not in git)')


@test
def prompts_have_no_blanks():
    for f in ['lineart-photo.txt', 'areas.txt', 'target-v4.txt', 'target-sketch.txt', 'target-anime.txt', 'lineart-anime.txt']:
        t = (HERE / 'prompts' / f).read_text()
        assert '按照片补充' not in t and '（' + '…' not in t, f'{f} still has a blank to fill in per photo'
    for k, v in make.KINDS.items():
        assert (HERE / 'prompts' / v['prompt']).exists(), k


@test
def annotations_land_on_eyes():
    need(S7 / 'subject.png')
    a = annotate.annotate(S7 / 'subject.png', S7 / 'areas.png', 0., target=S7 / 'v4/target.png')
    hand = [[0.527, 0.337], [0.737, 0.347]]        # the hand annotations of the best s7 films
    for (x, y, *_), (hx, hy) in zip(a['eyes'], hand):
        d = np.hypot((x - hx) * 1120, (y - hy) * 1422)
        assert d < 25, f'eye {d:.0f} px from the hand annotation'
    an = annotate.annotate(S7 / 'anime/target.png', S7 / 'anime/areas.png', 0., anime=True, target=S7 / 'anime/target.png')
    for (x, y, *_), (hx, hy) in zip(an['eyes'], [[0.51, 0.338], [0.726, 0.346]]):
        d = np.hypot((x - hx) * 1120, (y - hy) * 1422)
        assert d < 30, f'anime eye {d:.0f} px from the hand annotation'


@test
def annotation_sanity_refuses_nonsense():
    bad = dict(face=[.5, .4, .2, .2], eyes=[[.6, .35, .03, .02], [.4, .35, .03, .02]],
               focus=[[.5, .4, .2, .2, .85], [.5, .5, .4, .4, .5], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1],
                      [.5, .2, .05, .02, 1], [.5, .45, .03, .03, .95]])
    p = annotate.sanity(bad)
    assert any('left/right' in x for x in p) and any('order' in x for x in p), p


@test
def shifted_target_is_registered():
    need(S7 / 'v4/target.png')
    lm = make.landmarks(S7 / 'subject.png')
    with tempfile.TemporaryDirectory() as t:
        a, b = Path(t) / 'as-is.png', Path(t) / 'shifted.png'
        shutil.copy(S7 / 'v4/target.png', a)
        shutil.copy(S7 / 'v4/target.png', b)
        make.shift_image(b, 9, -6)
        for p in (a, b):
            ok, note = make.register_by_face(p, lm, 14)
            assert ok, note
        d = np.abs(make.key_points(make.landmarks(a)) - make.key_points(make.landmarks(b))).max()
        assert d < 4., f'(landmark jitter is 1-2 px) the shifted copy lands {d:.1f} px away from the unshifted one ({note})'


@test
def picture_without_face_is_refused():
    need(S7 / 'areas.png')
    lm = make.landmarks(S7 / 'subject.png')
    ok, note = make.register_by_face(S7 / 'areas.png', lm, 14)     # a flat area map has no face to find
    assert not ok, note


@test
def shifted_line_drawing_is_registered():
    need(S7 / 'sketch.png')
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / 'lines.png'
        shutil.copy(S7 / 'sketch.png', p)
        make.shift_image(p, 8, 6)
        q = Path(t) / 'as-is.png'
        shutil.copy(S7 / 'sketch.png', q)
        ok, note = make.register_by_edges(p, S7 / 'subject.png')
        assert ok, note
        ok, note0 = make.register_by_edges(q, S7 / 'subject.png')
        a = np.asarray(Image.open(p).convert('L'), float)
        b = np.asarray(Image.open(q).convert('L'), float)
        assert np.abs(a - b).mean() < 1., f'shifted copy not put back: {note} vs {note0}'
        need(ROOT / 'output/oil/s4/subject.png')
        ok, note = make.register_by_edges(q, ROOT / 'output/oil/s4/subject.png')   # a drawing of another photo
        assert not ok, 'a drawing of another photo was accepted: ' + note


@test
def area_map_checks():
    need(S7 / 'areas.png')
    lm = make.landmarks(S7 / 'subject.png')
    ok, note = make.check_areas(S7 / 'areas.png', lm)
    assert ok, note
    ok, note = make.check_areas(S7 / 'v4/target.png', lm)       # a painting is not a four-colour map
    assert not ok, 'a painting passed as an area map'


@test
def crop_keeps_the_face():
    need(S7 / 'subject.png')
    with tempfile.TemporaryDirectory() as t:
        wide = Path(t) / 'wide.png'
        src = Image.open(S7 / 'subject.png').convert('RGB')
        canvas = Image.new('RGB', (2200, 1536), (200, 200, 200))
        canvas.paste(src, (1100, 0))                       # the person on the right of a wide photo
        canvas.save(wide)
        out = Path(t) / 'subject.png'
        make.crop_subject(wide, out)
        assert Image.open(out).size == (1024, 1536)
        lm = make.landmarks(out)
        assert lm['ok'], 'face lost in the crop'
        cx = np.mean([p[0] for p in lm['oval']]) / 1024
        assert .25 < cx < .75, f'face at {cx:.2f} of the width'


@test
def hand_motion_metric():
    tips = [[i * 10., 0., 0., 'stroke', 0, 1.] for i in range(100)] + [[2000., 0., 0., 'stroke', 0, 1.]]
    p95, mx = check.hand_motion(tips)
    assert 9 < p95 < 11 and mx > 1000, (p95, mx)
    zoomed = [[i * 10., 0., 0., 'stroke', 0, 2.] for i in range(50)]
    assert 19 < check.hand_motion(zoomed)[0] < 21


@test
def best_films_pass_acceptance():
    films = {'oil': ('output/oil/s7/v5/run30', 'output/paint/portrait-s7-oil-v5-20261111'),
             'anime': ('output/oil/s7/anime-v2/run13', 'output/paint/portrait-s7-anime-v2-20261111')}
    for k, (run, d) in films.items():
        need(ROOT / run / 'tips.json')
        rep = check.run_checks(ROOT / run, ROOT / d)
        assert rep['verdict'] == 'PASS', (k, [c for c in rep['checks'] if not c['ok']])


@test
def cli_help_and_status():
    r = subprocess.run([sys.executable, str(HERE / 'make.py'), '--help'], capture_output=True, text=True)
    assert r.returncode == 0 and 'run' in r.stdout
    r = subprocess.run([sys.executable, str(HERE / 'make.py'), 'run', '/no/such/photo.jpg', '--kind', 'oil'],
                       capture_output=True, text=True)
    assert r.returncode != 0 and '找不到照片' in r.stdout, r.stdout[-300:]


@test
def shared_picture_is_drawn_once():
    """--kind all: three threads ask for the same line drawing; the image model must be called once (2026-11-12:
    racing threads drew it three times and overwrote each other, so the kinds planned on different pictures)."""
    import threading
    import time
    need(S7 / 'sketch.png')
    calls = []
    real = make.codex_image

    def fake(prompt, out, refs, log):
        calls.append(out)
        time.sleep(.5)
        shutil.copy(S7 / 'sketch.png', out)
    make.codex_image = fake
    try:
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / 'sketch.png'
            notes = []
            th = [threading.Thread(target=lambda: notes.append(make.generate('lines', 'p.txt', [], out,
                                                                                lambda p: (True, 'ok'), Path(t) / 'log')))
                  for _ in range(3)]
            [x.start() for x in th]
            [x.join() for x in th]
            assert len(calls) == 1, f'the image model was called {len(calls)} times for one picture'
            assert out.exists() and len(notes) == 3
    finally:
        make.codex_image = real


@test
def landmarks_are_cached():
    need(S7 / 'subject.png')
    a = make.landmarks(S7 / 'subject.png')
    b = make.landmarks(S7 / 'subject.png')
    assert a == b, 'the same picture gave different landmarks'


@test
def plans_are_deterministic():
    """The same inputs give the same plan, byte for byte (anime: the quickest engine to plan)."""
    import hashlib
    job = ROOT / 'output/portrait/s7-auto'
    need(job / 'annotations-anime.json')
    A = ROOT / 'output/portrait/.assets' / json.loads((job / 'manifest.json').read_text())['sha']
    a = json.loads((job / 'annotations-anime.json').read_text())
    j = json.dumps
    with tempfile.TemporaryDirectory() as t:
        hashes = []
        for k in range(2):
            run = Path(t) / f'r{k}'
            subprocess.run([sys.executable, 'anime.py', A / 'target-anime.png', A / 'sketch.png', A / 'anime-areas.png', run,
                            j(a['focus']), j(a['eyes']), j(a['signature']), '--photo', A / 'subject.png', '--face',
                            j(a['face']), '--crop-top', '0', '--lineart', A / 'anime-lineart.png'], cwd=HERE,
                           capture_output=True)
            hashes.append(hashlib.sha256((run / 'plan.json').read_bytes()).hexdigest())
        assert hashes[0] == hashes[1], 'two plans from the same inputs differ'


@test
def missed_paint_detector_separates():
    """The anime check: the v1 film the user saw white specks in fails clearly; the v2 film passes; a white shirt
    (painted white) and catchlights are not counted (s6, 2026-11-12)."""
    bad, good = ROOT / 'output/oil/s7/anime/run3', ROOT / 'output/oil/s7/anime-v2/run13'
    need(bad / 'plan.json')
    need(good / 'plan.json')
    lim = check.LIMITS['anime']['missed']
    b = check.missed_paint('anime', bad, None, None)[0]
    g = check.missed_paint('anime', good, None, None)[0]
    assert b['white_px'] > 10 * lim['white_px'], b
    assert g['white_px'] <= lim['white_px'] and g['slits'] <= lim['slits'], g
    shirt = ROOT / 'output/portrait/subject-55d5f4/anime/plan'
    if (shirt / 'plan.json').exists():
        s = check.missed_paint('anime', shirt, None, None)[0]
        assert s['white_px'] <= lim['white_px'], s


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if callable(fn) and getattr(fn, '__name__', '') == name and name not in ('test', 'need', 'SkipTest') \
                and not name.startswith('_') and fn.__module__ == '__main__' and name.islower() and name not in (
                'load',):
            if name in ('os', 'sys'):
                continue
            fn()
    w = max(len(n) for n, *_ in RESULTS)
    for n, r, msg in RESULTS:
        print(f'{n:{w}s}  {r}  {msg}')
    fails = sum(r == 'FAIL' for _, r, _ in RESULTS)
    print(f'\n{len(RESULTS) - fails}/{len(RESULTS)} passed' + (f', {fails} FAILED' if fails else ''))
    sys.exit(1 if fails else 0)
