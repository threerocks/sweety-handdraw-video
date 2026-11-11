"""Acceptance checks for a portrait film: measured, not judged by eye. make.py runs them after every film; a film that
fails a critical check is not delivered as done.

    python check.py RUN_DIR [--deliver DIR] [--areas AREAS.png] [--json OUT.json]

Each check has a value, a limit and a verdict. The limits come from the best films so far (s7, 2026-11-11: oil v5,
sketch v2, anime v2) with some margin, see LIMITS. What they catch:
- film_length       the film is a watchable length
- opening_cover     the film opens on the finished picture (~.5 s), then the empty canvas
- ending_hold       after the last stroke the finished picture stays on screen
- hand_motion       the hand never jumps: on-screen nib travel between frames while drawing (p95)
- missed_paint      no bare paper/ground showing where the area is painted (the user's 白点白线)
- likeness          the face of the finished picture is close to the target painting
- eyes_on_eyes      the eye annotations sit on dark pupils in the target (a wrong annotation paints eyes on skin)
- loudness          the music is at -16 LUFS
- deliverables      every file is there
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = Path(__file__).resolve().parent
PY = sys.executable
FPS = 30
LIMITS = {
    # kind: film seconds, hand p95 px/frame while drawing, missed-paint limits, likeness limit
    # best films (s7): oil hand p95 249, missed 12 px / 54 specks, likeness .018; sketch 287, 1 hair spot, .022;
    # anime 115 (the lines pass is looser than the colouring), 2 white px / 0 slits, .019. Oil and sketch time-lapse
    # long passes on purpose, so their hand limit guards against getting worse than that, not an ideal.
    'oil': dict(length=(90, 190), hand_p95=310, missed=dict(ground_px=60, light_specks=140), likeness=.06),
    'sketch': dict(length=(90, 190), hand_p95=360, missed=dict(hair_spots=40), likeness=.065),
    'anime': dict(length=(90, 190), hand_p95=145, missed=dict(white_px=40, slits=3), likeness=.055),
}


def load(p):
    return np.asarray(Image.open(p).convert('RGB'), float) / 255


def check(name, value, ok, limit, critical=True, note=''):
    return dict(name=name, value=value, limit=limit, ok=bool(ok), critical=critical, note=note)


def film_frames(path, times):
    out = []
    for t in times:
        r = subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{t:.3f}', '-i', str(path), '-frames:v', '1', '-f', 'rawvideo',
                            '-pix_fmt', 'rgb24', '-'], capture_output=True)
        out.append(np.frombuffer(r.stdout, np.uint8).reshape(1920, 1280, 3).astype(float) / 255 if r.stdout else None)
    return out


def duration(path):
    r = subprocess.run(['ffprobe', '-v', '0', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(path)],
                       capture_output=True, text=True)
    return float(r.stdout.strip() or 0)


def loudness(path):
    r = subprocess.run(['ffmpeg', '-hide_banner', '-nostats', '-i', str(path), '-vn', '-af', 'ebur128', '-f', 'null', '-'],
                       capture_output=True, text=True)
    m = re.findall(r'I:\s+(-?[\d.]+) LUFS', r.stderr)
    return float(m[-1]) if m else None


def hand_motion(tips):
    """p95 of the nib's on-screen travel between consecutive frames while it is drawing."""
    d = []
    for a, b in zip(tips, tips[1:]):
        if a[3] == 'stroke' and b[3] == 'stroke' and a[2] < .5 and b[2] < .5:
            z = b[5] if len(b) > 5 else 1.
            d.append(np.hypot(b[0] - a[0], b[1] - a[1]) * z)
    return (float(np.percentile(d, 95)), float(np.max(d))) if d else (0., 0.)


def missed_paint(kind, run, areas, face):
    """The kind's own detector on the finished picture (numbers as in the evidence of 2026-11-11)."""
    if kind == 'oil':
        r = subprocess.run([PY, '-c', f'import forms, sys; rows = forms.bare_report({str(run)!r}); '
                                      'import json; print("JSON", json.dumps(rows))'],
                           cwd=HERE, capture_output=True, text=True)
        rows = json.loads(r.stdout.split('JSON', 1)[1]) if 'JSON' in r.stdout else None
        if not rows:
            return None, r.stderr[-500:]
        last = next(x for x in rows if x[0].startswith('final (flat)')) if any(
            x[0].startswith('final (flat)') for x in rows) else rows[-1]
        return dict(ground_px=last[1], light_specks=last[4]), ''
    if kind == 'sketch':
        r = subprocess.run([PY, 'holes.py', str(run), str(areas), json.dumps(face)], cwd=HERE, capture_output=True, text=True)
        f = Path(run) / 'holes.json'
        if not f.exists():
            return None, r.stderr[-500:]
        h = json.loads(f.read_text())
        return dict(hair_spots=h['final']['hair'][0]), ''
    r = subprocess.run([PY, 'bare.py', str(run)], cwd=HERE, capture_output=True, text=True)
    rows = [l.split() for l in r.stdout.splitlines()[1:] if l.strip()]
    if not rows:
        return None, r.stderr[-500:]
    worst = max(rows, key=lambda x: int(x[1]))          # every image from the flats on
    return dict(white_px=int(worst[1]), slits=max(int(x[3]) for x in rows)), ''


def likeness(kind, run, face):
    """Mean difference to the target inside the face ellipse, 2 px blur (0..1). Anime is compared before the final
    colour adjustment, which by design makes it equal the target."""
    pic = run / ('before-adjust.png' if (run / 'before-adjust.png').exists() else 'final.png')
    a, b = load(pic), load(run / 'target.png')
    a, b = ndimage.gaussian_filter(a, (2, 2, 0)), ndimage.gaussian_filter(b, (2, 2, 0))
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    m = ((xx / w - face[0]) / face[2]) ** 2 + ((yy / h - face[1]) / face[3]) ** 2 < 1
    if kind == 'sketch':
        lum = np.array([.299, .587, .114])
        return float(np.abs(a @ lum - b @ lum)[m].mean())
    return float(np.abs(a - b)[m].mean())


def eyes_on_eyes(run, eyes):
    """Each eye centre is darker than the face around it in the target (a pupil, not cheek or hair)."""
    t = load(run / 'target.png') @ np.array([.299, .587, .114])
    h, w = t.shape
    worst = -1.
    for cx, cy, rx, ry in eyes:
        x, y = int(cx * w), int(cy * h)
        r = max(3, int(min(rx * w, ry * h) * .5))
        core = t[max(0, y - r):y + r, max(0, x - r):x + r]
        R = int(rx * w * 3)
        ring = t[max(0, y - R):y + R, max(0, x - R):x + R]
        worst = max(worst, float(np.percentile(core, 20) - np.median(ring)))
    return worst


def run_checks(run, deliver=None, areas=None):
    run = Path(run)
    plan = json.loads((run / 'plan.json').read_text())
    kind = plan.get('kind', 'oil')
    L = LIMITS[kind]
    face = plan.get('face') or plan['focus'][0][:4]
    eyes = plan.get('eyes_used') or [f[:4] for f in plan['focus'][2:4]]
    film = Path(deliver) / 'film.mp4' if deliver else run / 'film.mp4'
    out = []
    dur = duration(film)
    out.append(check('film_length', round(dur, 1), L['length'][0] <= dur <= L['length'][1], L['length']))
    cover = Path(deliver) / 'cover.png' if deliver else None
    f0, f1 = film_frames(film, [.05, .9])
    if f0 is not None and f1 is not None:
        canvas0, canvas1 = f0[120:1540, 80:1200], f1[120:1540, 80:1200]
        busy0, busy1 = float(canvas0.std(axis=(0, 1)).mean()), float(canvas1.std(axis=(0, 1)).mean())
        ok = busy0 > .08 and busy1 < busy0 * .6
        out.append(check('opening_cover', dict(picture_detail_at_0s=round(busy0, 3), at_0_9s=round(busy1, 3)), ok,
                         'detail at 0 s > .08, at .9 s below 60% of it'))
    tips = json.loads((run / 'tips.json').read_text()) if (run / 'tips.json').exists() else []
    if tips:
        tail = 0
        for t in reversed(tips):
            if t[3] != 'hold':
                break
            tail += 1
        out.append(check('ending_hold', round(tail / FPS, 1), tail / FPS >= 4.5, '>= 4.5 s'))
        p95, mx = hand_motion(tips)
        out.append(check('hand_motion', dict(p95=round(p95), max=round(mx)), p95 <= L['hand_p95'], f"p95 <= {L['hand_p95']} px/frame"))
    else:
        out.append(check('ending_hold', None, False, 'tips.json missing'))
    if areas or kind != 'sketch':
        miss, err = missed_paint(kind, run, areas, face)
        if miss is None:
            out.append(check('missed_paint', None, False, L['missed'], note='detector failed: ' + err))
        else:
            ok = all(miss[k] <= v for k, v in L['missed'].items())
            out.append(check('missed_paint', miss, ok, L['missed']))
    lk = likeness(kind, run, face)
    out.append(check('likeness', round(lk, 4), lk <= L['likeness'], f"<= {L['likeness']}"))
    eo = eyes_on_eyes(run, eyes)
    out.append(check('eyes_on_eyes', round(eo, 3), eo < -.03, 'pupil darker than its surroundings by > .03'))
    if deliver:
        d = Path(deliver)
        lu = loudness(d / 'film.mp4')
        out.append(check('loudness', lu, lu is not None and -17.5 <= lu <= -14.5, '-16 ± 1.5 LUFS', critical=False))
        need = ['film.mp4', 'film-silent.mp4', 'target.png', 'cover.png']
        missing = [n for n in need if not (d / n).exists()]
        pic = [p.name for p in d.glob('*.png') if p.stem in ('painting', 'drawing')]
        out.append(check('deliverables', missing or 'all present', not missing and bool(pic), need + ['painting|drawing.png']))
    crit = [c for c in out if c['critical']]
    report = dict(kind=kind, run=str(run), checks=out, passed=sum(c['ok'] for c in out), total=len(out),
                  verdict='PASS' if all(c['ok'] for c in crit) else 'FAIL')
    return report


def markdown(report):
    lines = [f"# 验收报告：{report['kind']}", '', f"结论：**{report['verdict']}**（{report['passed']}/{report['total']} 项通过）", '',
             '| 检查 | 数值 | 标准 | 结果 |', '|---|---|---|---|']
    for c in report['checks']:
        lines.append(f"| {c['name']} | {c['value']} | {c['limit']} | {'通过' if c['ok'] else '未通过'}{'' if c['critical'] else '（非关键）'} |")
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--deliver')
    ap.add_argument('--areas')
    ap.add_argument('--json')
    a = ap.parse_args()
    rep = run_checks(a.run, a.deliver, a.areas)
    if a.json:
        Path(a.json).write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(markdown(rep))
    sys.exit(0 if rep['verdict'] == 'PASS' else 1)
