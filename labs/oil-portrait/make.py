"""One command from a photo to a finished portrait film: 油画 (oil), 素描 (sketch) or 动漫插画 (anime).

    python labs/oil-portrait/make.py setup                       # once: checks and installs what is missing
    python labs/oil-portrait/make.py run PHOTO --kind sketch     # or oil, anime, all
    python labs/oil-portrait/make.py status NAME                 # what is done, the acceptance verdicts, the films

Everything a person used to decide by looking is decided here, so any model (or none) gets the same film:
- the photo is cropped to 2:3 around the face;
- the image model draws the line drawing, area map and target paintings from fixed prompts (prompts/); each result is
  checked (size, registration to the photo by face landmarks or edges, area colours) and redrawn up to 3 times;
- the eyes, mouth, nose, face, figure and signature place come from face landmarks (annotate.py), with sanity checks;
- plan, film, music, cover and delivery use the settings of the best films (oil v5, sketch v2, anime v2);
- check.py measures the film (length, cover, ending, hand motion, missed paint, likeness, eyes, loudness).

Work files: output/portrait/NAME/. Generated pictures are cached per photo (output/portrait/.assets/PHOTO_HASH/), so
running again, or another kind, reuses them. A stage whose output exists is skipped; --redo STAGE redoes it and what
follows. Films: output/paint/portrait-NAME-KIND-DATE/ (film.mp4 with music, film-silent.mp4, the picture, target,
cover, report.md).
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HOME = Path.home() / '.local/share/sweety-handdraw-video'
PY = HOME / 'venv/bin/python'                    # the engines (numpy 2, scipy, Pillow)
FACE_PY = HOME / 'venv-face/bin/python'          # landmarks (MediaPipe needs numpy < 2)
SCRIPTS = ROOT / 'skills/sweety-handdraw-video/scripts'
OUT = ROOT / 'output/portrait'
DELIVER = ROOT / 'output/paint'
KINDS = {
    # target prompt, music mood and seed, the picture's file name, Chinese name
    'oil': dict(prompt='target-v4.txt', music=('warm', 7), picture='painting', zh='油画'),
    'sketch': dict(prompt='target-sketch.txt', music=('calm', 21), picture='drawing', zh='素描'),
    'anime': dict(prompt='target-anime.txt', music=('curious', 5), picture='drawing', zh='动漫插画'),
}
STAGES = ['subject', 'assets', 'annotate', 'plan', 'film', 'deliver', 'check']
TRIES = 3                                        # image-model attempts per picture before giving up
LOCK = threading.Lock()


class Stop(RuntimeError):
    """A failure with a message for the user (Chinese) and what to do next."""


def say(msg):
    with LOCK:
        print(time.strftime('%H:%M:%S'), msg, flush=True)


def sh(cmd, log, cwd=HERE, env=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, 'a') as f:
        f.write(f'\n$ {" ".join(map(str, cmd))}\n')
        f.flush()
        r = subprocess.run([str(c) for c in cmd], cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env=env)
    if r.returncode:
        tail = log.read_text()[-1500:]
        raise Stop(f'命令失败（退出码 {r.returncode}）：{Path(str(cmd[1])).name}\n日志：{log}\n{tail}')


# ------------------------------------------------------------------------------------------------- setup
def codex_bin():
    found = shutil.which('codex')
    if found:
        return found
    cands = sorted(glob.glob(str(Path.home() / '.local/share/fnm/node-versions/*/installation/bin/codex')))
    return cands[-1] if cands else None


def setup(install=True):
    ok = True
    for name, test in [('ffmpeg', shutil.which('ffmpeg')), ('ffprobe', shutil.which('ffprobe'))]:
        say(f'{"OK " if test else "缺少"} {name}')
        ok &= bool(test)
    r = subprocess.run([str(PY), '-c', 'import numpy, scipy, PIL; print(numpy.__version__)'], capture_output=True, text=True) \
        if PY.exists() else None
    say(f'{"OK " if r and not r.returncode else "缺少"} 引擎环境 {PY}')
    ok &= bool(r and not r.returncode)
    if not FACE_PY.exists() and install:
        say('安装人脸关键点环境（一次性，约 700 MB）…')
        subprocess.run(['/usr/bin/python3', '-m', 'venv', str(FACE_PY.parents[1])], check=True)
        subprocess.run([str(FACE_PY), '-m', 'pip', 'install', '-q', 'mediapipe==0.10.21'], check=True)
    r = subprocess.run([str(FACE_PY), '-c', 'import mediapipe'], capture_output=True) if FACE_PY.exists() else None
    say(f'{"OK " if r and not r.returncode else "缺少"} 人脸关键点环境 {FACE_PY}')
    ok &= bool(r and not r.returncode)
    c = codex_bin()
    say(f'{"OK " if c else "缺少"} codex（生成目标画用）{c or ""}')
    ok &= bool(c)
    for f in ['src/hand-tang-red.png', 'src/hand-tang-pencil.png', 'src/hand-tang-stylus.png', 'src/tools-sketch.png']:
        e = (HERE / f).exists()
        say(f'{"OK " if e else "缺少"} {f}')
        ok &= e
    return ok


# ------------------------------------------------------------------------------------------------- pictures
def landmarks(path):
    r = subprocess.run([str(FACE_PY), str(HERE / 'landmarks.py'), str(path)], capture_output=True, text=True)
    if r.returncode:
        raise Stop(f'人脸关键点失败：{path}\n{r.stderr[-800:]}')
    return json.loads(r.stdout)


def crop_subject(photo, dst):
    """2:3 crop around the face, 1024x1536: the face a little above the upper third (the canvas shows the top 85%)."""
    from PIL import Image
    img = Image.open(photo).convert('RGB')
    w, h = img.size
    lm = landmarks(photo)
    if not lm.get('ok'):
        raise Stop(f'照片里找不到人脸：{photo}。请换一张正面或半侧面、脸部清楚的照片。')
    oval = lm['oval']
    fx = sum(p[0] for p in oval) / len(oval)
    fy = sum(p[1] for p in oval) / len(oval)
    face_h = max(p[1] for p in oval) - min(p[1] for p in oval)
    if w / h > 2 / 3:                                # too wide: keep the full height, cut the sides round the face
        cw, ch = round(h * 2 / 3), h
    else:                                            # too tall: keep the full width
        cw, ch = w, round(w * 3 / 2)
    # the face should fill a fair part of the picture: when it is small, crop tighter (down to a face of 1/4 height)
    scale = min(1., max(face_h * 4 / ch, .45)) if face_h * 4 < ch else 1.
    cw, ch = round(cw * scale), round(ch * scale)
    x0 = min(max(fx - cw / 2, 0), w - cw)
    y0 = min(max(fy - ch * .36, 0), h - ch)
    img.crop((int(x0), int(y0), int(x0) + cw, int(y0) + ch)).resize((1024, 1536), Image.LANCZOS).save(dst)
    if face_h / ch < .12:
        say(f'注意：脸只占画面高度的 {face_h / ch:.0%}，偏小，五官细节会弱一些')


def codex_image(prompt, out, refs, log):
    env = dict(os.environ)
    c = codex_bin()
    if not c:
        raise Stop('找不到 codex 命令（生成目标画要用）。先运行  python labs/oil-portrait/make.py setup')
    env['PATH'] = str(Path(c).parent) + os.pathsep + env.get('PATH', '')
    cmd = [PY, SCRIPTS / 'codex_image.py', prompt, out, '--shape', 'portrait', '--attempts', '2']
    for r in refs:
        cmd += ['--ref', str(r)]
    sh(cmd, log, cwd=ROOT, env=env)


def normalise(path):
    from PIL import Image
    img = Image.open(path).convert('RGB')
    w, h = img.size
    if abs(w / h - 2 / 3) > .02:
        raise Stop(f'生成的图不是 2:3（{w}x{h}）')
    if (w, h) != (1024, 1536):
        img.resize((1024, 1536), Image.LANCZOS).save(path)


def shift_image(path, dx, dy):
    """Move a generated picture by whole pixels (edges repeated) so it lies on the photo."""
    import numpy as np
    from PIL import Image
    a = np.asarray(Image.open(path).convert('RGB'))
    dx, dy = int(round(dx)), int(round(dy))
    if not dx and not dy:
        return
    ys = np.clip(np.arange(a.shape[0]) - dy, 0, a.shape[0] - 1)
    xs = np.clip(np.arange(a.shape[1]) - dx, 0, a.shape[1] - 1)
    Image.fromarray(a[ys][:, xs]).save(path)


def key_points(lm):
    import numpy as np
    mouth = np.mean(lm['lips_outer'], 0)
    return np.array([lm['eyes'][0]['iris'], lm['eyes'][1]['iris'], lm['nose']['tip'], mouth])


def register_by_face(path, photo_lm, tol):
    """The picture's face landmarks against the photo's: a common offset is corrected, a distorted face is refused."""
    import numpy as np
    lm = landmarks(path)
    if not lm.get('ok'):
        return False, '生成的图里找不到人脸'
    d = key_points(lm) - key_points(photo_lm)
    shift = np.median(d, 0)
    resid = float(np.abs(d - shift).max())
    if np.hypot(*shift) > 60:
        return False, f'整体偏移 {shift.round(1).tolist()} px，太大'
    if resid > tol:
        return False, f'五官和照片对不上（偏差 {resid:.0f} px > {tol}）'
    if np.hypot(*shift) > 1.5:
        shift_image(path, -shift[0], -shift[1])
    return True, f'对齐：移动 {(-shift).round(1).tolist()} px，五官偏差 {resid:.0f} px'


def edge_map(path, dark_lines=False):
    """Half-resolution, blurred (4 px) line/edge strength, standardised: lines of a drawing or edges of a picture."""
    import numpy as np
    from PIL import Image
    from scipy import ndimage
    a = np.asarray(Image.open(path).convert('L').resize((512, 768), Image.BILINEAR), float) / 255
    e = (1 - a) if dark_lines else ndimage.gaussian_gradient_magnitude(a, 1.)
    e = ndimage.gaussian_filter(e, 2.)
    return (e - e.mean()) / (e.std() + 1e-9)


def register_by_edges(path, reference, limit=48, min_ncc=.25):
    """A line drawing against the edges of the picture it was drawn from: the shift (up to limit px) where they
    correlate best is corrected; a drawing of something else correlates under min_ncc and is refused (measured on the
    s4/s6/s7 drawings: .43-.63 for the right picture, .01-.10 for a wrong one)."""
    import numpy as np
    from scipy import signal
    A, B = edge_map(path, True), edge_map(reference)
    c = signal.fftconvolve(B, A[::-1, ::-1], mode='same')
    H, W = c.shape
    lim = limit // 2
    win = c[H // 2 - lim:H // 2 + lim + 1, W // 2 - lim:W // 2 + lim + 1]
    iy, ix = np.unravel_index(np.argmax(win), win.shape)
    dy, dx = iy - lim, ix - lim
    moved = np.roll(np.roll(A, dy, 0), dx, 1)[lim:H - lim, lim:W - lim]
    ref = B[lim:H - lim, lim:W - lim]
    ncc = float((moved * ref).mean() / (moved.std() * ref.std() + 1e-9))
    if ncc < min_ncc:
        return False, f'线稿和参考对不上（相关 {ncc:.2f} < {min_ncc}），可能画的不是这张图'
    shift_image(path, dx * 2, dy * 2)
    return True, f'对齐：移动 [{dx * 2}, {dy * 2}] px（相关 {ncc:.2f}）'


def check_areas(path, photo_lm):
    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(HERE))
    import annotate
    a = np.asarray(Image.open(path).convert('RGB'), float)
    pal = np.array(list(annotate.AREA_COLOURS.values()), float)
    d = np.sqrt(((a[:, :, None, :] - pal[None, None]) ** 2).sum(-1)).min(-1)
    clean = float((d < 70).mean())
    if clean < .85:
        return False, f'分区图不是四种纯色（只有 {clean:.0%} 的像素是）'
    m = annotate.area_masks(path, (1024, 1536))
    pts = key_points(photo_lm).astype(int)
    on_skin = float(np.mean([m['skin'][y, x] for x, y in pts]))
    if on_skin < .75:
        return False, f'五官没有落在皮肤区（{on_skin:.0%}）'
    share = {k: float(v.mean()) for k, v in m.items()}
    if share['bg'] < .02 or share['skin'] < .02:
        return False, f'分区比例不对 {share}'
    return True, '四色 {:.0%}，五官在皮肤区'.format(clean)


def generate(name, prompt_file, refs, out, validate, log):
    """Draw one picture with the image model; validate it; up to TRIES attempts. Returns the validation note."""
    if out.exists() and (out.with_suffix('.ok.json')).exists():
        return json.loads(out.with_suffix('.ok.json').read_text())['note']
    last = ''
    for k in range(1, TRIES + 1):
        say(f'  生成 {name}（第 {k} 次）…')
        tmp = out.with_name(f'{out.stem}-try{k}.png')
        try:
            codex_image(prompt_file, tmp, refs, log)
            normalise(tmp)
            ok, note = validate(tmp)
        except Stop as e:
            ok, note = False, str(e).splitlines()[0]
        if ok:
            shutil.move(tmp, out)
            out.with_suffix('.ok.json').write_text(json.dumps(dict(note=note, attempt=k, prompt=str(prompt_file))))
            say(f'  {name}：{note}')
            return note
        say(f'  {name} 不合格：{note}')
        last = note
    raise Stop(f'{name} 生成 {TRIES} 次都不合格（最后一次：{last}）。可以再运行一次同样的命令重试；'
               f'若总是失败，换一张脸部更清楚、正面一些的照片。')


# ------------------------------------------------------------------------------------------------- the job
class Job:
    def __init__(self, photo=None, name=None):
        if photo:
            photo = Path(photo).expanduser().resolve()
            if not photo.exists():
                raise Stop(f'找不到照片：{photo}')
            self.sha = hashlib.sha256(photo.read_bytes()).hexdigest()[:12]
            self.name = name or (re.sub(r'[^A-Za-z0-9_-]+', '-', photo.stem).strip('-')[:24] or 'photo') + '-' + self.sha[:6]
            self.dir = OUT / self.name
            self.dir.mkdir(parents=True, exist_ok=True)
            m = self.manifest()
            m.update(photo=str(photo), sha=self.sha, name=self.name)
            self.save(m)
        else:
            self.name = name
            self.dir = OUT / name
            if not (self.dir / 'manifest.json').exists():
                raise Stop(f'没有这个任务：{name}（在 {OUT} 下找不到）')
            self.sha = self.manifest()['sha']
        self.assets = OUT / '.assets' / self.sha
        self.assets.mkdir(parents=True, exist_ok=True)
        self.logs = self.dir / 'logs'

    def manifest(self):
        f = self.dir / 'manifest.json'
        return json.loads(f.read_text()) if f.exists() else {}

    def save(self, m):
        with LOCK:
            (self.dir / 'manifest.json').write_text(json.dumps(m, ensure_ascii=False, indent=1))

    def note(self, key, value):
        with LOCK:
            m = self.manifest()
            m[key] = value
            (self.dir / 'manifest.json').write_text(json.dumps(m, ensure_ascii=False, indent=1))

    # stage 1
    def subject(self):
        s = self.assets / 'subject.png'
        if not s.exists():
            say('裁成 2:3 …')
            crop_subject(self.manifest()['photo'], s)
        self._photo_lm = landmarks(s)
        if not self._photo_lm.get('ok'):
            raise Stop('裁好的照片里找不到人脸')
        return s

    # stage 2
    def assets_for(self, kind):
        A, P = self.assets, HERE / 'prompts'
        subj = A / 'subject.png'
        lm = self._photo_lm
        jobs = {
            'lines': (P / 'lineart-photo.txt', [subj], A / 'sketch.png', lambda p: register_by_edges(p, subj)),
            'areas': (P / 'areas.txt', [subj], A / 'areas.png', lambda p: check_areas(p, lm)),
        }
        tol = {'oil': 14, 'sketch': 14, 'anime': 30}[kind]
        jobs['target-' + kind] = (P / KINDS[kind]['prompt'], [subj], A / f'target-{kind}.png',
                                  lambda p: register_by_face(p, lm, tol))
        with ThreadPoolExecutor(3) as ex:
            futs = {k: ex.submit(generate, k, *v, self.logs / f'asset-{k}.log') for k, v in jobs.items()}
            notes = {k: f.result() for k, f in futs.items()}
        if kind == 'anime':
            tgt = A / 'target-anime.png'
            alm = landmarks(tgt)
            more = {
                'anime-areas': (P / 'areas.txt', [tgt], A / 'anime-areas.png', lambda p: check_areas(p, alm)),
                'anime-lines': (P / 'lineart-anime.txt', [tgt, subj], A / 'anime-lineart.png',
                                lambda p: register_by_edges(p, tgt)),
            }
            with ThreadPoolExecutor(2) as ex:
                futs = {k: ex.submit(generate, k, *v, self.logs / f'asset-{k}.log') for k, v in more.items()}
                notes.update({k: f.result() for k, f in futs.items()})
        self.note(f'assets-{kind}', notes)

    # stage 3
    def annotate(self, kind):
        sys.path.insert(0, str(HERE))
        import annotate
        f = self.dir / f'annotations-{kind}.json'
        if f.exists():
            return json.loads(f.read_text())
        A = self.assets
        try:
            if kind == 'anime':
                a = annotate.annotate(A / 'target-anime.png', A / 'anime-areas.png', 0., anime=True,
                                      target=A / 'target-anime.png')
            else:
                a = annotate.annotate(A / 'subject.png', A / 'areas.png', 0., lm=self._photo_lm,
                                      target=A / f'target-{kind}.png')
        except annotate.AnnotationError as e:
            raise Stop(f'自动标注五官失败：{e}')
        f.write_text(json.dumps(a))
        return a

    # stage 4
    def plan(self, kind, ann):
        run = self.dir / kind / 'plan'
        if (run / 'plan.json').exists() and (run / 'final.png').exists():
            return run
        if run.exists():
            shutil.rmtree(run)
        A = self.assets
        j = json.dumps
        common = [j(ann['focus']), j(ann['eyes']), j(ann['signature'])]
        if kind == 'oil':
            cmd = [PY, 'paint.py', 'plan', A / 'target-oil.png', A / 'sketch.png', run, j(ann['focus']),
                   j(ann['signature']), j(ann['eyes']), '--photo', A / 'subject.png', '--style', 'v5', '--preset',
                   'planes', '--crop-top', '0', '--face', j(ann['face']), '--figure', j(ann['figure']),
                   '--areas', A / 'areas.png']
        elif kind == 'sketch':
            cmd = [PY, 'sketch.py', A / 'target-sketch.png', A / 'sketch.png', A / 'areas.png', run, *common,
                   '--photo', A / 'subject.png', '--face', j(ann['face']), '--crop-top', '0']
        else:
            cmd = [PY, 'anime.py', A / 'target-anime.png', A / 'sketch.png', A / 'anime-areas.png', run, *common,
                   '--photo', A / 'subject.png', '--face', j(ann['face']), '--crop-top', '0',
                   '--lineart', A / 'anime-lineart.png']
        say(f'[{KINDS[kind]["zh"]}] 规划笔画（约 5–20 分钟）…')
        sh(cmd, self.logs / f'plan-{kind}.log')
        if not (run / 'final.png').exists():
            raise Stop(f'规划没有产出 final.png，见 {self.logs / f"plan-{kind}.log"}')
        return run

    # stage 5
    def film(self, kind, run):
        if (run / 'film.mp4').exists() and (run / 'tips.json').exists() and (run / 'last-frame.png').exists():
            return
        say(f'[{KINDS[kind]["zh"]}] 渲染视频（约 10–15 分钟）…')
        sh([PY, 'film.py', run, '--stills', '0'], self.logs / f'film-{kind}.log')

    # stage 6
    def deliver(self, kind, run, music=None):
        from PIL import Image
        import numpy as np
        date = datetime.date.today().strftime('%Y%m%d')
        d = DELIVER / f'portrait-{self.name}-{kind}-{date}'
        d.mkdir(parents=True, exist_ok=True)
        dur = float(subprocess.run(['ffprobe', '-v', '0', '-show_entries', 'format=duration', '-of', 'csv=p=0',
                                    str(run / 'film.mp4')], capture_output=True, text=True).stdout)
        if not music:
            mood, seed = KINDS[kind]['music']
            music = self.dir / kind / f'music-{mood}.wav'
            if not music.exists():
                sh([PY, SCRIPTS / 'music_synth.py', mood, '--seconds', f'{dur + 6:.0f}', '--seed', str(seed),
                    '--output', music], self.logs / f'music-{kind}.log')
        shutil.copy(run / 'film.mp4', d / 'film-silent.mp4')
        fo = max(0., dur - 4)
        sh(['ffmpeg', '-v', 'error', '-y', '-i', run / 'film.mp4', '-stream_loop', '-1', '-i', music, '-filter_complex',
            f'[1:a]atrim=0:{dur:.3f},afade=t=in:d=1,afade=t=out:st={fo:.3f}:d=4,loudnorm=I=-16:TP=-1.5:LRA=11[a]',
            '-map', '0:v', '-map', '[a]', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart',
            d / 'film.mp4'], self.logs / f'deliver-{kind}.log')
        shutil.copy(run / 'target.png', d / 'target.png')
        shutil.copy(run / 'final.png', d / f'{KINDS[kind]["picture"]}.png')
        # cover: the finished picture as the film ends, without the watermark (its place taken from a clean frame)
        sys.path.insert(0, str(HERE))
        import film
        plan = json.loads((run / 'plan.json').read_text())
        film.KIND[0] = plan.get('kind', 'oil')
        ref = np.asarray(Image.open(run / 'reference.png').convert('RGB'))
        base = np.asarray(film.base_frame(ref, watermark=False, kind=film.KIND[0]))
        sh(['ffmpeg', '-v', 'error', '-y', '-sseof', f'{-(film.OUTRO - .04):.2f}', '-i', run / 'film.mp4', '-frames:v', '1',
            d / 'cover.png'], self.logs / f'deliver-{kind}.log')
        cover = np.asarray(Image.open(d / 'cover.png').convert('RGB')).copy()
        cover[30:100, 640:1240] = base[30:100, 640:1240]
        Image.fromarray(cover).save(d / 'cover.png')
        sh(['ffmpeg', '-v', 'error', '-y', '-i', run / 'film.mp4', '-vf', f'fps=12/{dur:.2f},scale=320:-1,tile=6x2',
            '-frames:v', '1', d / 'sheet.jpg'], self.logs / f'deliver-{kind}.log')
        return d

    # stage 7
    def check(self, kind, run, d):
        sys.path.insert(0, str(HERE))
        import check
        areas = self.assets / ('anime-areas.png' if kind == 'anime' else 'areas.png')
        rep = check.run_checks(run, d, areas)
        rep['assets'] = self.manifest().get(f'assets-{kind}')
        (d / 'report.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1))
        (d / 'report.md').write_text(check.markdown(rep))
        self.note(f'result-{kind}', dict(deliver=str(d), verdict=rep['verdict'], passed=rep['passed'], total=rep['total']))
        return rep

    def one(self, kind, redo=(), music=None):
        if 'assets' in redo:
            for p in self.assets.glob('*.ok.json'):
                p.unlink()
        if {'annotate', 'assets'} & set(redo):
            (self.dir / f'annotations-{kind}.json').unlink(missing_ok=True)
        if {'plan', 'annotate', 'assets'} & set(redo) and (self.dir / kind / 'plan').exists():
            shutil.rmtree(self.dir / kind / 'plan')
        if 'film' in redo:
            (self.dir / kind / 'plan' / 'film.mp4').unlink(missing_ok=True)
        say(f'[{KINDS[kind]["zh"]}] 准备目标画和分区图…')
        self.assets_for(kind)
        ann = self.annotate(kind)
        say(f'[{KINDS[kind]["zh"]}] 五官标注完成：' + ('；'.join(ann['notes']) or '正常'))
        run = self.plan(kind, ann)
        self.film(kind, run)
        d = self.deliver(kind, run, music)
        rep = self.check(kind, run, d)
        return d, rep


def summary(job, results):
    lines = ['', '=' * 60, f'任务 {job.name} 完成情况', '=' * 60]
    for kind, (d, rep) in results.items():
        dur = next((c['value'] for c in rep['checks'] if c['name'] == 'film_length'), '?')
        lines.append(f"{KINDS[kind]['zh']}：验收 {rep['verdict']}（{rep['passed']}/{rep['total']}），时长 {dur} 秒")
        lines.append(f"  视频：{d / 'film.mp4'}")
        lines.append(f"  观看：! open {d / 'film.mp4'}")
        lines.append(f"  报告：{d / 'report.md'}")
        for c in rep['checks']:
            if not c['ok']:
                lines.append(f"  未通过：{c['name']} = {c['value']}（标准 {c['limit']}）")
    bad = [k for k, (_, r) in results.items() if r['verdict'] != 'PASS']
    lines.append('')
    lines.append('全部通过验收。' if not bad else
                 f"{'、'.join(KINDS[k]['zh'] for k in bad)} 未通过验收：把上面「未通过」的项和报告路径如实告诉用户，不要说已完成。")
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('setup')
    r = sub.add_parser('run')
    r.add_argument('photo')
    r.add_argument('--kind', required=True, choices=['oil', 'sketch', 'anime', 'all'])
    r.add_argument('--name', help='任务名（默认：照片文件名 + 哈希）')
    r.add_argument('--music', help='背景音乐文件（默认按画法合成）')
    r.add_argument('--redo', action='append', default=[], choices=STAGES, help='重做这一步和之后的步骤')
    s = sub.add_parser('status')
    s.add_argument('name', nargs='?')
    a = ap.parse_args()
    try:
        if a.cmd == 'setup':
            sys.exit(0 if setup() else 1)
        if a.cmd == 'status':
            names = [a.name] if a.name else sorted(p.parent.name for p in OUT.glob('*/manifest.json'))
            for n in names:
                m = Job(name=n).manifest()
                print(n, m.get('photo'))
                for k in KINDS:
                    if f'result-{k}' in m:
                        res = m[f'result-{k}']
                        print(f"  {KINDS[k]['zh']}: {res['verdict']} {res['passed']}/{res['total']}  {res['deliver']}/film.mp4")
            return
        if not setup(install=False):
            raise Stop('环境不完整：先运行  python labs/oil-portrait/make.py setup')
        job = Job(a.photo, a.name)
        say(f'任务 {job.name}，工作目录 {job.dir}')
        job.subject()
        kinds = list(KINDS) if a.kind == 'all' else [a.kind]
        redo = set(a.redo)
        if 'subject' in redo:
            raise Stop('照片换了请用新的任务名；同一张照片不需要重做裁剪')
        results, errors = {}, {}
        with ThreadPoolExecutor(len(kinds)) as ex:
            futs = {k: ex.submit(job.one, k, redo, a.music) for k in kinds}
            for k, f in futs.items():
                try:
                    results[k] = f.result()
                except Stop as e:
                    errors[k] = str(e)
        if results:
            print(summary(job, results))
        for k, e in errors.items():
            print(f"\n{KINDS[k]['zh']} 失败：{e}")
        if errors or any(r['verdict'] != 'PASS' for _, r in results.values()):
            sys.exit(1)
    except Stop as e:
        print('\n失败：' + str(e))
        sys.exit(2)


if __name__ == '__main__':
    if Path(sys.executable).resolve() != PY.resolve() and PY.exists() and os.environ.get('MAKE_NO_REEXEC') != '1':
        os.execv(str(PY), [str(PY), __file__] + sys.argv[1:])        # always the engines' environment
    main()
