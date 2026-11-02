"""Pen and marker foley synthesized from the drawing timeline, sample-synced to the nib.

Every sound is made here from seeded noise and filters; no recording is used. Each non-caption event's segments
drive two voices: a nib voice for ink and sketch strokes (band-passed noise, brighter and louder as the nib speeds
up, roughened by a paper-grain texture read along the path so the grain rate follows the pen) with a tiny noise tick
at every pen-down, and a rub voice for color, shade and brush passes (lower, broader felt noise whose level dips
at every zigzag turn because the nib slows there). Travel and pause segments are silent, so the sound stops while
the pen is lifted. The layer follows the nib's x position with subtle equal-power panning, ducks a few dB under
speech using the narration envelope, and has a soft ceiling so it can never compete with the voice.

Levels are absolute, not normalized per film, so a sparse film is not made louder than a dense one."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import wave
import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d
from motion_profile import distance_fraction

SAMPLE_RATE = 48000
CONTROL_RATE = 1000             # envelopes, brightness and pan are planned per millisecond
VERSION = 'foley-1'
LAYER_GAIN = .25                # mixer gain in the layer; the file is written 12 dB hotter so 16-bit keeps the detail
REFERENCE_DBFS = -34.           # one channel, centered, a steady ink stroke at the reference speed, before ducking
CEILING_DBFS = -24.             # soft ceiling on the final layer (after LAYER_GAIN) at gain_db 0
KNEE = .5                       # the ceiling starts bending at half its level
DUCK_DB = 4.
PAN_WIDTH = .45                 # 0 = mono, 1 = hard left/right at the canvas edges
ATTACK_SECONDS, RELEASE_SECONDS = .004, .006
CONTACT_FLOOR = .12             # nib resting on the paper while nearly still still rubs a little
TEXTURE_STEP_PX = .4            # paper grain resolution along the path (at a 1080 px short side)
TEXTURE_SIZE = 1 << 17
TAIL_SECONDS = .06
OPTION_KEYS = {'enabled', 'gain_db', 'duck_db', 'tool', 'seed', 'stereo', 'scope'}
SCOPES = ('writing', 'all')     # writing: only handwritten words make a sound (default); all: drawing and coloring too
SMOOTH_HZ = 5200.               # gentle low-pass on the layer: the hiss above it read as harsh against the voice

# Drawing tool timbres: nib bands (bright, dark), rub bands, grain depth, pen-down tick and overall level.
TOOLS = {
    'marker': {'nib_bright': (1800, 6500), 'nib_dark': (700, 3000), 'rub_bright': (900, 3800), 'rub_dark': (260, 1800),
               'grain': .55, 'tick': .5, 'level': 1.},
    'pen': {'nib_bright': (2600, 8000), 'nib_dark': (1100, 4200), 'rub_bright': (1000, 4200), 'rub_dark': (300, 2000),
            'grain': .8, 'tick': .6, 'level': .8},
    'pencil': {'nib_bright': (2200, 7500), 'nib_dark': (900, 3600), 'rub_bright': (900, 4000), 'rub_dark': (300, 1900),
               'grain': 1., 'tick': .45, 'level': .85},
    'brush': {'nib_bright': (900, 4000), 'nib_dark': (300, 1800), 'rub_bright': (600, 3000), 'rub_dark': (200, 1400),
              'grain': .25, 'tick': .12, 'level': .7},
}
HAND_TOOLS = {'navy': 'marker', 'pen-only': 'marker', 'diamond-woman': 'marker', 'child-pencil': 'pencil',
              'waner-brush': 'brush', 'teacher-pen': 'pen', 'cardigan-fountain': 'pen'}
# Segment kind -> (voice, level, reference nib speed in px/s on a 1080 px short side, brightness offset, tick share)
KINDS = {'ink': ('nib', 1., 700., 0., 1.), 'sketch': ('nib', .45, 600., .2, .5),
         'color': ('rub', .8, 1400., 0., .35), 'shade': ('rub', .5, 900., -.2, .25), 'brush': ('rub', .75, 1200., .1, .35),
         'erase': ('rub', .55, 1600., .35, .3)}
SILENT_KINDS = {'travel', 'pause'}
VOICES = ('nib', 'text', 'rub')
TEXT_SHIFT, TEXT_GRAIN = .85, .8  # handwriting: the same nib a little lower and smoother, a soft pencil-on-paper whisper


def parse_options(value):
    """audio.foley: omitted/true/{} -> on with defaults; false or {"enabled": false} -> off. Returns None when off."""
    if value is None or value is True:
        value = {}
    if value is False:
        return None
    if not isinstance(value, dict):
        raise ValueError('audio.foley must be true, false or an object')
    unknown = set(value) - OPTION_KEYS
    if unknown:
        raise ValueError('Unknown audio.foley option(s): ' + ', '.join(sorted(unknown)))
    if not value.get('enabled', True):
        return None
    options = {'gain_db': float(value.get('gain_db', 0.)), 'duck_db': float(value.get('duck_db', DUCK_DB)),
               'tool': value.get('tool', 'auto'), 'seed': value.get('seed', 0), 'stereo': bool(value.get('stereo', True)),
               'scope': value.get('scope', 'writing')}
    if options['scope'] not in SCOPES:
        raise ValueError('audio.foley.scope must be writing or all')
    if not math.isfinite(options['gain_db']) or not -18 <= options['gain_db'] <= 6:
        raise ValueError('audio.foley.gain_db must be within -18..6 dB')
    if not math.isfinite(options['duck_db']) or not 0 <= options['duck_db'] <= 12:
        raise ValueError('audio.foley.duck_db must be within 0..12 dB')
    if options['tool'] != 'auto' and options['tool'] not in TOOLS:
        raise ValueError('audio.foley.tool must be auto or one of ' + ', '.join(TOOLS))
    if not isinstance(options['seed'], int) or isinstance(options['seed'], bool):
        raise ValueError('audio.foley.seed must be an integer')
    return options


def tool_for(timeline, options):
    if options['tool'] != 'auto':
        return options['tool']
    hand = timeline.get('hand') or {}
    return HAND_TOOLS.get(hand.get('preset'), 'pencil' if 'pencil' in str(hand.get('preset', '')) else 'marker')


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(str(part) for part in parts).encode()).digest()[:8], 'little')


def bandpass(band):
    return signal.butter(2, band, 'bandpass', fs=SAMPLE_RATE, output='sos')


def white_noise_rms(sos):
    """RMS a unit white noise keeps through this filter (from its power response), so bands compare by level."""
    _, response = signal.sosfreqz(sos, worN=8192)
    return float(np.sqrt(np.mean(np.abs(response) ** 2)))


def filters(tool):
    bank = {}
    for name in ('nib_bright', 'nib_dark', 'rub_bright', 'rub_dark'):
        sos = bandpass(TOOLS[tool][name])
        bank[name] = (sos, 1 / white_noise_rms(sos))
    for name in ('bright', 'dark'):
        sos = bandpass(tuple(TEXT_SHIFT * edge for edge in TOOLS[tool]['nib_' + name]))
        bank['text_' + name] = (sos, 1 / white_noise_rms(sos))
    tick = bandpass((1200, 5200) if tool != 'brush' else (500, 2500))
    bank['tick'] = (tick, 1 / white_noise_rms(tick))
    return bank


def paper_texture(seed):
    """Unit-variance grain along the path: smoothed noise over ~1 px so the modulation rate tracks nib speed."""
    rng = np.random.default_rng(seed_for('paper', seed))
    grain = uniform_filter1d(rng.standard_normal(TEXTURE_SIZE), 3, mode='wrap')
    return (grain - grain.mean()) / grain.std()


def sound_segments(metadata):
    return [segment for segment in metadata.get('segments', []) if segment['kind'] in KINDS and len(segment.get('points', [])) >= 1]


def stroke_track(segment, duration, scale):
    """Per-millisecond positions (asset px) and nib speed (px/s at a 1080 px short side) inside one segment."""
    start, end = segment['start'] * duration, segment['end'] * duration
    first = int(math.floor(start * CONTROL_RATE))
    last = max(first, int(math.ceil(end * CONTROL_RATE)))
    index = np.arange(first, last + 1)
    fraction = np.clip((index / CONTROL_RATE - start) / max(end - start, 1e-9), 0, 1)
    points = np.asarray(segment['points'], dtype=float)
    lengths = np.asarray(segment.get('lengths') or np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))], dtype=float)
    if len(points) < 2 or lengths[-1] <= 0:
        x, y = np.full(len(index), points[0][0]), np.full(len(index), points[0][1])
    else:
        along = distance_fraction(fraction, segment.get('profile', 'linear')) * lengths[-1]
        x, y = np.interp(along, lengths, points[:, 0]), np.interp(along, lengths, points[:, 1])
    step = np.hypot(np.diff(x, prepend=x[0]), np.diff(y, prepend=y[0])) / scale
    speed = uniform_filter1d(step * CONTROL_RATE, 5, mode='nearest')
    inside = (index / CONTROL_RATE >= start - .5 / CONTROL_RATE) & (index / CONTROL_RATE <= end + .5 / CONTROL_RATE)
    return index, x, y, speed, step, inside


def plan_event(event, metadata, canvas, handwriting):
    """Control-rate plan of one drawing event: per-voice amplitude and brightness, pan, path distance and ticks."""
    duration = event['end'] - event['start']
    count = int(math.ceil((duration + TAIL_SECONDS) * CONTROL_RATE)) + 2
    plan = {'amp': {voice: np.zeros(count) for voice in VOICES}, 'bright': {voice: np.zeros(count) for voice in VOICES},
            'pan': np.full(count, np.nan), 'distance': np.zeros(count), 'ticks': [], 'contact_seconds': 0., 'segments': {}}
    board = float(event.get('scale', 1.))  # assets of a zoomed page are measured in its board pixels
    scale = min(canvas['width'], canvas['height']) / 1080 * board
    travelled = 0.
    attack = max(1, round(ATTACK_SECONDS * CONTROL_RATE))
    release = np.exp(-np.arange(1, round(6 * RELEASE_SECONDS * CONTROL_RATE) + 1) / (RELEASE_SECONDS * CONTROL_RATE))
    for number, segment in enumerate(sound_segments(metadata)):
        voice, level, reference, brightness, tick = KINDS[segment['kind']]
        if handwriting and segment['kind'] == 'ink':  # handwriting: a touch crisper and lighter than drawing
            voice, level = 'text', .9 * level
        index, x, y, speed, step, inside = stroke_track(segment, duration, scale)
        index, x, y, speed, step = index[inside], x[inside], y[inside], speed[inside], step[inside]
        if not len(index) or index[0] >= count:
            continue
        keep = index < count
        index, x, y, speed, step = index[keep], x[keep], y[keep], speed[keep], step[keep]
        amp = level * np.maximum(np.clip(speed / reference, 0, 1.8) ** .55, CONTACT_FLOOR)
        ramp = np.minimum(1., (np.arange(len(amp)) + 1) / attack)
        amp = amp * (.5 - .5 * np.cos(np.pi * ramp))
        tail = index[-1] + 1 + np.arange(len(release))
        tail_keep = tail < count
        plan['amp'][voice][index] = np.maximum(plan['amp'][voice][index], amp)
        plan['amp'][voice][tail[tail_keep]] = np.maximum(plan['amp'][voice][tail[tail_keep]], amp[-1] * release[tail_keep])
        bright = np.clip(.15 + .7 * np.clip(speed / reference, 0, 1.5) / 1.5 + brightness, 0, 1)
        plan['bright'][voice][index] = bright
        plan['bright'][voice][tail[tail_keep]] = bright[-1]
        plan['pan'][index] = (event['x'] + x / board) / canvas['width']
        plan['distance'][index] = travelled + np.cumsum(step)
        travelled = plan['distance'][index[-1]] + 7919.  # the next stroke reads another patch of paper
        plan['ticks'].append((int(index[0]), voice, level * tick, number))
        plan['contact_seconds'] += len(index) / CONTROL_RATE
        plan['segments'][segment['kind']] = plan['segments'].get(segment['kind'], 0) + 1
    known = np.flatnonzero(np.isfinite(plan['pan']))
    if len(known):
        plan['pan'] = np.interp(np.arange(count), known, plan['pan'][known])
    else:
        plan['pan'][:] = .5
    return plan


def narration_duck(narration, depth_db):
    """Gain per 10 ms frame: depth_db lower while the narration speaks (40 ms attack, 300 ms release)."""
    from audio_gate import frame_levels
    levels = 20 * np.log10(frame_levels(narration) + 1e-9)
    if not len(levels) or depth_db <= 0:
        return np.ones(max(1, len(levels)))
    speech = np.clip((levels - (np.percentile(levels, 95) - 30)) / 12, 0, 1)
    smoothed = np.empty_like(speech)
    value, rise, fall = 0., 1 - math.exp(-.01 / .04), 1 - math.exp(-.01 / .3)
    for index, target in enumerate(speech):
        value += (target - value) * (rise if target > value else fall)
        smoothed[index] = value
    return 10 ** (-depth_db * smoothed / 20)


def render_event(plan, bank, tool, texture, rng):
    """Mono foley for one event at the sample rate, plus the per-sample pan position (0 left .. 1 right)."""
    count = len(plan['pan'])
    frames = int(round((count - 1) * SAMPLE_RATE / CONTROL_RATE))
    control = np.arange(count) / CONTROL_RATE
    times = np.arange(frames) / SAMPLE_RATE
    noise = rng.standard_normal(frames)
    distance = np.interp(times, control, plan['distance']) / TEXTURE_STEP_PX + rng.integers(0, TEXTURE_SIZE)
    whole = np.floor(distance).astype(np.int64)
    part = distance - whole
    grain = texture[whole % TEXTURE_SIZE] * (1 - part) + texture[(whole + 1) % TEXTURE_SIZE] * part
    mono = np.zeros(frames)
    for voice in VOICES:
        if not plan['amp'][voice].any():
            continue
        excitation = noise * np.maximum(.05, 1 + TOOLS[tool]['grain'] * (TEXT_GRAIN if voice == 'text' else 1.) * .6 * grain)
        amp = np.interp(times, control, plan['amp'][voice])
        bright = np.interp(times, control, plan['bright'][voice])
        (sos_bright, gain_bright), (sos_dark, gain_dark) = bank[voice + '_bright'], bank[voice + '_dark']
        tone = bright * gain_bright * signal.sosfilt(sos_bright, excitation) + (1 - bright) * gain_dark * signal.sosfilt(sos_dark, excitation)
        mono += amp * tone
    sos_tick, gain_tick = bank['tick']
    length = round(.008 * SAMPLE_RATE)
    shape = np.exp(-np.arange(length) / (.0012 * SAMPLE_RATE))
    for start, voice, level, _ in plan['ticks']:
        at = round(start * SAMPLE_RATE / CONTROL_RATE)
        if at >= frames or level <= 0:
            continue
        burst = gain_tick * signal.sosfilt(sos_tick, rng.standard_normal(length)) * shape
        tap = np.sin(2 * np.pi * rng.uniform(170, 320) * np.arange(length) / SAMPLE_RATE) * np.exp(-np.arange(length) / (.003 * SAMPLE_RATE))
        tick = (burst * (.5 if voice == 'rub' else 1.) + .35 * tap) * TOOLS[tool]['tick'] * level * rng.uniform(.6, 1.) * 1.4
        end = min(frames, at + length)
        mono[at:end] += tick[:end - at]
    return mono * TOOLS[tool]['level'], np.interp(times, control, plan['pan'])


def soft_ceiling(samples, ceiling):
    """Unity below KNEE x ceiling, then a tanh bend that never exceeds the ceiling."""
    knee = KNEE * ceiling
    magnitude = np.abs(samples)
    over = magnitude > knee
    bent = knee + (ceiling - knee) * np.tanh((magnitude[over] - knee) / (ceiling - knee))
    samples[over] = np.sign(samples[over]) * bent
    return samples


def write_pcm16(path, stereo):
    data = np.rint(np.clip(stereo, -1, 32767 / 32768) * 32768).astype('<i2')
    with wave.open(str(path), 'wb') as stream:
        stream.setparams((2, 2, SAMPLE_RATE, 0, 'NONE', 'not compressed'))
        stream.writeframes(data.tobytes())


def synthesize(timeline, options, narration=None):
    """Final-level foley (after LAYER_GAIN) as float32 frames x 2, and a report. narration: mono floats for ducking."""
    tool = tool_for(timeline, options)
    bank = filters(tool)
    texture = paper_texture(options['seed'])
    frames = round(timeline['duration'] * SAMPLE_RATE)
    output = np.zeros((frames, 2), dtype=np.float32)
    duck = narration_duck(narration, options['duck_db']) if narration is not None else None
    level = 10 ** ((REFERENCE_DBFS + options['gain_db']) / 20) * math.sqrt(2)  # centered pan keeps cos(pi/4) per channel
    width = PAN_WIDTH if options['stereo'] else 0.
    stats = {'events': 0, 'segments': {}, 'contact_seconds': 0.}
    for page in timeline['pages']:
        for position, event in enumerate(page['events']):
            metadata = timeline['assets'].get(event['asset'], {})
            if event.get('caption') or not sound_segments(metadata):
                continue
            if options['scope'] == 'writing' and metadata.get('type') != 'handwritten_text':
                continue  # drawing and coloring stay silent: a constant scratch under the voice was tiring
            plan = plan_event(event, metadata, timeline['canvas'], metadata.get('type') == 'handwritten_text')
            if not plan['ticks']:
                continue
            rng = np.random.default_rng(seed_for(VERSION, options['seed'], event['asset'], f"{event['start']:.6f}", position))
            mono, pan = render_event(plan, bank, tool, texture, rng)
            offset = round(event['start'] * SAMPLE_RATE)
            count = max(0, min(len(mono), frames - offset))
            if not count:
                continue
            mono, pan = mono[:count] * level, pan[:count]
            if duck is not None:
                mono *= np.interp((offset + np.arange(count)) / SAMPLE_RATE, np.arange(len(duck)) * .01 + .01, duck)
            angle = (np.clip((pan - .5) * 2 * width, -1, 1) + 1) * np.pi / 4
            output[offset:offset + count, 0] += (mono * np.cos(angle)).astype(np.float32)
            output[offset:offset + count, 1] += (mono * np.sin(angle)).astype(np.float32)
            stats['events'] += 1
            stats['contact_seconds'] += plan['contact_seconds']
            for kind, number in plan['segments'].items():
                stats['segments'][kind] = stats['segments'].get(kind, 0) + number
    smooth = signal.butter(2, SMOOTH_HZ, 'lowpass', fs=SAMPLE_RATE, output='sos')
    output[:] = signal.sosfilt(smooth, output, axis=0).astype(np.float32)
    soft_ceiling(output, 10 ** ((CEILING_DBFS + options['gain_db']) / 20))
    report = {'tool': tool, 'scope': options['scope'], **stats, 'contact_seconds': round(stats['contact_seconds'], 2)}
    if duck is not None and len(duck) > 1:
        from audio_gate import speech_frames
        speaking = speech_frames(narration)[:len(duck)]
        applied = 20 * np.log10(duck[:len(speaking)])
        report['duck_applied_db'] = {'speech_mean': round(float(applied[speaking].mean()), 2) if speaking.any() else None,
                                     'pause_mean': round(float(applied[~speaking].mean()), 2) if (~speaking).any() else None}
    return output, report


def build_layer(work_dir, timeline, options=None):
    """Write work/audio/foley.wav from the timeline and return a renderer mix layer, or None when foley is off or
    nothing is drawn. options: the job's audio.foley value (None/True -> defaults, False -> off, or a dict)."""
    parsed = parse_options(options)
    if parsed is None:
        return None
    work = Path(work_dir)
    from audio_gate import read_pcm, loudness_lufs, db, framed_rms, speech_frames
    narration_path = work / 'narration.wav'
    narration = read_pcm(narration_path) if narration_path.is_file() else None
    if narration is not None and narration.ndim != 1:
        narration = narration.mean(axis=1)
    final, report = synthesize(timeline, parsed, narration)
    if not report['events']:
        return None
    audio = work / 'audio'
    audio.mkdir(parents=True, exist_ok=True)
    path = audio / 'foley.wav'
    write_pcm16(path, final / LAYER_GAIN)
    written = read_pcm(path) * LAYER_GAIN  # measure what the mixer will actually read
    contact = np.abs(written).max(axis=1) > 10 ** (-80 / 20)
    levels = {'peak_dbfs': db(np.abs(written).max()), 'loudness_lufs': loudness_lufs(written),
              'active_rms_dbfs': db(np.sqrt(np.mean(written[contact].astype(np.float64) ** 2))) if contact.any() else None,
              'active_share': round(float(contact.mean()), 3)}
    if narration is not None:
        middle = written.mean(axis=1, dtype=np.float64)
        speaking = speech_frames(narration)
        from audio_gate import framed_activity
        active = framed_activity(middle, len(speaking))
        levels['duck_db'] = parsed['duck_db']
        levels['duck_applied_db'] = report.get('duck_applied_db')
        during, between = framed_rms(middle, speaking & active), framed_rms(middle, ~speaking & active)
        levels['speech_rms_dbfs'] = db(during) if during else None
        levels['pause_rms_dbfs'] = db(between) if between else None
    record = {'version': VERSION, 'options': parsed, **report, 'levels': levels, 'layer_gain': LAYER_GAIN,
              'reference_dbfs': REFERENCE_DBFS + parsed['gain_db'], 'ceiling_dbfs': CEILING_DBFS + parsed['gain_db'],
              'sample_rate': SAMPLE_RATE, 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (audio / 'foley-report.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    return {'source': str(path.resolve()), 'gain': LAYER_GAIN, 'start': 0., 'sha256': record['source_sha256'], 'kind': 'foley',
            'reason': 'pen/marker sound synced to the drawing (procedural, foley.py)',
            'foley': {'version': VERSION, 'tool': report['tool'], 'gain_db': parsed['gain_db'], 'duck_db': parsed['duck_db'],
                      'seed': parsed['seed'], 'events': report['events'], 'contact_seconds': report['contact_seconds'], 'levels': levels}}


def attach(root, timeline, manifest):
    """Build the foley layer for a finished timeline and append it to timeline['audio_policy']['layers'].
    Call once in build after the timeline dict exists and before timeline.json is written. Returns the layer or None."""
    layer = build_layer(Path(root) / 'work', timeline, manifest.get('audio', {}).get('foley'))
    if layer:
        policy = timeline.setdefault('audio_policy', {'background_music': False, 'layers': []})
        policy['layers'] = [entry for entry in policy.get('layers', []) if entry.get('kind') != 'foley'] + [layer]
    return layer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render the foley layer of a built job (work/timeline.json) to a wav.')
    parser.add_argument('job', type=Path)
    parser.add_argument('--output', type=Path, help='Write here instead of work/audio/foley.wav (final level, no LAYER_GAIN)')
    parser.add_argument('--options', default='{}', help='audio.foley options as JSON')
    arguments = parser.parse_args()
    timeline = json.loads((arguments.job / 'work/timeline.json').read_text())
    if arguments.output:
        from audio_gate import read_pcm
        parsed = parse_options(json.loads(arguments.options))
        narration = read_pcm(arguments.job / 'work/narration.wav')
        final, report = synthesize(timeline, parsed, narration)
        write_pcm16(arguments.output, final)
        print(json.dumps(report, ensure_ascii=False))
    else:
        print(json.dumps(build_layer(arguments.job / 'work', timeline, json.loads(arguments.options)), ensure_ascii=False, indent=2))
