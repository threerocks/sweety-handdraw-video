"""Transition and accent sounds synthesized from the finished timeline: paper for page turns and tears, felt or rubber
for erasers, a whoosh for whips and dives, a brush swish for ink and paint wipes, a soft chime when sparkles burst and a
low thump when the camera slams in. Every sound is made here from seeded noise, filters and sine partials; nothing is
recorded or downloaded.

The layer is quiet by design (it sits under the voice, ducks further while someone speaks, and is low-passed so it
never hisses) and sample-synced to the same transition windows the renderer draws.

job.json "audio": {"sfx": true | false | {"gain_db": 0, "transitions": true, "accents": false, "quiet_moves": false, "seed": 0}}
(on by default). Restraint by default (v3): only the signature chapter turns sound; ordinary pans, pushes and wipes are
silent and accent chimes and thumps are off, because each extra sound is one more chance to break a viewer's mood
(a sound that does not belong costs attention; the tenth whoosh is noise). Turn them on with a reason."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from scipy import signal

SAMPLE_RATE = 48000
VERSION = 'sfx-1'
LAYER_GAIN = .5                 # mixer gain; the file is written 6 dB hotter so 16-bit keeps the detail
PEAK_DBFS = -26.                # loudest point of a sound at gain_db 0, before ducking
DUCK_DB = 6.
SMOOTH_HZ = 6000.
OPTION_KEYS = {'enabled', 'gain_db', 'transitions', 'accents', 'quiet_moves', 'seed'}

QUIET_MOVES = {'push', 'pan', 'slide', 'wipe', 'zoom', 'legacy'}   # ordinary camera-like moves: silent unless quiet_moves
# Transition type -> sound and its level relative to PEAK_DBFS.
TRANSITION_SOUNDS = {
    'page_turn': ('paper_flip', 0.), 'slide_over': ('paper_slide', -3.), 'card_flip': ('paper_flip', -2.), 'cube': ('paper_slide', -4.),
    'tiles': ('paper_taps', -4.), 'paper_tear': ('tear', 0.), 'shards': ('crumple', -2.), 'erase': ('rub', -2.),
    'whip': ('whoosh', 0.), 'zoom_through': ('whoosh', -2.), 'push': ('swish', -6.), 'pan': ('swish', -7.), 'slide': ('swish', -7.),
    'wipe': ('swish', -6.), 'zoom': ('swish', -6.), 'fade': None, 'cut': None, 'legacy': ('swish', -9.),
    'marker_wipe': ('brush', -3.), 'brush_reveal': ('brush', -2.), 'ink_bleed': ('swell', -3.), 'iris': ('swell', -5.), 'flash': ('shimmer', -6.),
}


def parse_options(value):
    """audio.sfx: omitted/true/{} -> on; false or {"enabled": false} -> off (None)."""
    if value is None or value is True:
        value = {}
    if value is False:
        return None
    if not isinstance(value, dict):
        raise ValueError('audio.sfx must be true, false or an object')
    unknown = set(value) - OPTION_KEYS
    if unknown:
        raise ValueError('Unknown audio.sfx option(s): ' + ', '.join(sorted(unknown)))
    if not value.get('enabled', True):
        return None
    options = {'gain_db': float(value.get('gain_db', 0.)), 'transitions': bool(value.get('transitions', True)),
               'accents': bool(value.get('accents', False)), 'quiet_moves': bool(value.get('quiet_moves', False)), 'seed': value.get('seed', 0)}
    if not math.isfinite(options['gain_db']) or not -18 <= options['gain_db'] <= 6:
        raise ValueError('audio.sfx.gain_db must be within -18..6 dB')
    if not isinstance(options['seed'], int) or isinstance(options['seed'], bool):
        raise ValueError('audio.sfx.seed must be an integer')
    return options


def band(noise, low, high, order=2):
    sos = signal.butter(order, [low, min(high, SAMPLE_RATE / 2 - 100)], 'bandpass', fs=SAMPLE_RATE, output='sos')
    return signal.sosfilt(sos, noise)


def envelope(count, attack, release, shape=1.):
    """Rise over `attack` share, fall over `release` share of count samples."""
    t = np.linspace(0, 1, count)
    rise = np.clip(t / max(attack, 1e-3), 0, 1) ** shape
    fall = np.clip((1 - t) / max(release, 1e-3), 0, 1) ** shape
    return rise * fall


def normalized(sound):
    peak = np.abs(sound).max()
    return sound / peak if peak > 0 else sound


def make(kind, seconds, rng):
    """One sound, peak 1, about `seconds` long (a few are fixed-length by nature)."""
    count = max(1, round(seconds * SAMPLE_RATE))
    noise = rng.standard_normal(count)
    t = np.arange(count) / SAMPLE_RATE
    if kind == 'paper_flip':
        # A sheet lifted and laid over: a rising papery rustle, a soft flap at the end.
        rustle = band(noise, 1400, 6500) * envelope(count, .55, .35, 1.6) * (.7 + .3 * band(rng.standard_normal(count), 20, 60) / 3)
        flap_len = min(count, round(.09 * SAMPLE_RATE))
        flap = np.zeros(count)
        flap[-flap_len:] = band(rng.standard_normal(flap_len), 120, 600) * np.exp(-np.linspace(0, 5, flap_len)) * 2.
        return normalized(rustle + flap)
    if kind == 'paper_slide':
        return normalized(band(noise, 900, 5000) * envelope(count, .3, .6, 1.4))
    if kind == 'paper_taps':
        taps = np.zeros(count)
        for start in np.linspace(.05, .75, 7) * count:
            length = round(.03 * SAMPLE_RATE)
            index = int(start)
            piece = band(rng.standard_normal(length), 800, 4000) * np.exp(-np.linspace(0, 6, length))
            taps[index:index + length] += piece[:max(0, count - index)]
        return normalized(taps)
    if kind == 'tear':
        # Fibres giving way: crackling grains that thicken, then a release.
        grains = (rng.random(count) < np.linspace(.002, .02, count)).astype(float)
        grains = np.convolve(grains, np.exp(-np.linspace(0, 8, 120)), 'same')
        crackle = band(noise * (0.4 + grains * 3), 1000, 7000) * envelope(count, .7, .2, 1.2)
        return normalized(crackle)
    if kind == 'crumple':
        clicks = (rng.random(count) < .006 * envelope(count, .2, .6)).astype(float)
        clicks = np.convolve(clicks, np.exp(-np.linspace(0, 10, 90)), 'same')
        return normalized(band(noise * clicks * 4 + .15 * noise * envelope(count, .1, .8), 900, 5500))
    if kind == 'rub':
        # Felt on a board: low broad rubbing that swells with each stroke of the eraser.
        strokes = .55 + .45 * np.abs(np.sin(2 * math.pi * 2.2 * t))
        return normalized(band(noise, 250, 2400) * strokes * envelope(count, .08, .15))
    if kind == 'rubber':
        strokes = .5 + .5 * np.abs(np.sin(2 * math.pi * 9 * t))
        return normalized(band(noise, 700, 4200) * strokes * envelope(count, .06, .12))
    if kind == 'whoosh':
        # Air past the lens: band-passed noise whose centre sweeps up and back.
        frames = 64
        out = np.zeros(count)
        edges = np.linspace(0, count, frames + 1).astype(int)
        for index in range(frames):
            share = index / (frames - 1)
            centre = 400 + 1800 * math.sin(math.pi * share)
            segment = noise[edges[index]:edges[index + 1]]
            if len(segment) > 8:
                out[edges[index]:edges[index + 1]] = band(segment, centre * .6, centre * 1.6, 1)
        return normalized(out * envelope(count, .45, .45, 2.))
    if kind == 'swish':
        return normalized(band(noise, 500, 3200) * envelope(count, .4, .5, 1.8))
    if kind == 'brush':
        bristles = .8 + .2 * band(rng.standard_normal(count), 30, 90) / 2
        return normalized(band(noise, 600, 3500) * bristles * envelope(count, .2, .4, 1.3))
    if kind == 'swell':
        return normalized(band(noise, 120, 900) * envelope(count, .6, .35, 1.5))
    if kind == 'shimmer':
        return normalized(band(noise, 3500, 9000) * envelope(count, .1, .8, 2.))
    if kind == 'chime':
        # A small bell: inharmonic partials with fast decay, a little detuned so it does not ring like a test tone.
        partials = [(1568., 1.), (2349., .5), (3136., .35), (4186., .2)]
        return normalized(sum(weight * np.sin(2 * math.pi * frequency * t + rng.random() * 6) * np.exp(-t * (5 + frequency / 900))
                              for frequency, weight in partials))
    if kind == 'thump':
        body = np.sin(2 * math.pi * (95 - 40 * t / max(t[-1], 1e-6)) * t) * np.exp(-t * 18)
        click = band(noise, 800, 3000) * np.exp(-t * 90) * .3
        return normalized(body + click)
    raise ValueError(f'Unknown sound {kind}')


def transition_windows(timeline):
    """(start, end, type, style) of every transition, exactly as the renderer draws them."""
    from video_renderer import Renderer
    renderer = object.__new__(Renderer)
    renderer.timeline, renderer.fps = timeline, timeline['canvas']['fps']
    windows = []
    for index in range(1, len(timeline['pages'])):
        window = renderer.transition(index)
        if window:
            windows.append((window['start'], window['end'], window['type'], window.get('style')))
    return windows


def accent_events(timeline):
    """(time, sound, level dB) for sparkle bursts, slammed titles and camera slams."""
    from motion import camera_key
    events = []
    for page in timeline['pages']:
        for overlay in page.get('overlays', []):
            if overlay.get('sound'):
                events.append((overlay['start'] + .04, overlay['sound'], -4. if overlay['sound'] == 'chime' else -6.))
        beats = {beat['id']: beat['start'] - page['start'] for beat in page.get('beats', [])}
        for key in page.get('camera') or []:
            key = camera_key(key)
            if not key.get('impact'):
                continue
            at = key.get('at', 0)
            at = beats.get(at[5:], 0.) if isinstance(at, str) else float(at)
            events.append((page['start'] + at + float(key.get('seconds', .8)), 'thump', -3.))
    return events


def synthesize(timeline, options, narration=None):
    frames = round(timeline['duration'] * SAMPLE_RATE)
    output = np.zeros(frames, dtype=np.float64)
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f'{VERSION}|{options["seed"]}'.encode()).digest()[:8], 'little'))
    peak = 10 ** ((PEAK_DBFS + options['gain_db']) / 20)
    placed = []

    def place(start, seconds, kind, level_db, label):
        sound = make(kind, seconds, rng) * peak * 10 ** (level_db / 20)
        offset = max(0, round(start * SAMPLE_RATE))
        count = max(0, min(len(sound), frames - offset))
        if count:
            output[offset:offset + count] += sound[:count]
            placed.append({'at': round(start, 3), 'sound': kind, 'for': label})

    if options['transitions']:
        for start, end, kind, style in transition_windows(timeline):
            entry = TRANSITION_SOUNDS.get(kind)
            if not entry or (kind in QUIET_MOVES and not options['quiet_moves']):
                continue
            sound, level = entry
            if kind == 'erase' and style == 'rubber':
                sound = 'rubber'
            place(start, max(.12, end - start), sound, level, kind)
    if options['accents']:
        for time, sound, level in accent_events(timeline):
            place(time, {'chime': 1.1, 'thump': .35}.get(sound, .5), sound, level, sound)
    smooth = signal.butter(2, SMOOTH_HZ, 'lowpass', fs=SAMPLE_RATE, output='sos')
    output = signal.sosfilt(smooth, output)
    if narration is not None and len(narration):
        from foley import narration_duck
        duck = narration_duck(narration, DUCK_DB)
        output *= np.interp(np.arange(frames) / SAMPLE_RATE, np.arange(len(duck)) * .01 + .01, duck)
    return output.astype(np.float32), {'sounds': placed}


def attach(root, timeline, manifest):
    """Build work/audio/sfx.wav and append it to timeline['audio_policy']['layers']. Returns the layer or None."""
    options = parse_options(manifest.get('audio', {}).get('sfx'))
    if options is None:
        return None
    work = Path(root) / 'work'
    from audio_gate import read_pcm, db, loudness_lufs
    narration_path = work / 'narration.wav'
    narration = read_pcm(narration_path) if narration_path.is_file() else None
    if narration is not None and narration.ndim != 1:
        narration = narration.mean(axis=1)
    mono, report = synthesize(timeline, options, narration)
    if not report['sounds']:
        return None
    from foley import write_pcm16
    audio = work / 'audio'
    audio.mkdir(parents=True, exist_ok=True)
    path = audio / 'sfx.wav'
    write_pcm16(path, np.stack([mono, mono], 1) / LAYER_GAIN)
    written = read_pcm(path) * LAYER_GAIN
    levels = {'peak_dbfs': db(np.abs(written).max()), 'loudness_lufs': loudness_lufs(written)}
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    (audio / 'sfx-report.json').write_text(json.dumps({'version': VERSION, 'options': options, **report, 'levels': levels,
                                                       'source_sha256': sha}, ensure_ascii=False, indent=2) + '\n')
    layer = {'source': str(path.resolve()), 'gain': LAYER_GAIN, 'start': 0., 'sha256': sha, 'kind': 'sfx',
             'reason': 'transition and accent sounds synced to the picture (procedural, sfx.py)',
             'sfx': {'version': VERSION, 'gain_db': options['gain_db'], 'sounds': len(report['sounds']), 'levels': levels}}
    policy = timeline.setdefault('audio_policy', {'background_music': False, 'layers': []})
    policy['layers'] = [entry for entry in policy.get('layers', []) if entry.get('kind') != 'sfx'] + [layer]
    return layer
