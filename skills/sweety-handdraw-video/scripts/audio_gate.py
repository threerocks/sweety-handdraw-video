#!/usr/bin/env python3
"""Objective narration gate: stationary tones, pause depth, DC offset and clipping.

FAIL when a narrow line (30-4000 Hz) stands >= 8 dB above its +-25 Hz neighbourhood in >= 50% (and at
least 2) of the 2-second windows (hum, drone, sustained music note), when the quietest 5% of 20 ms frames are less
than 25 dB below the speech p95 level (pauses never become quiet: noise or music bed), or when DC
offset / clipping exceed the limits below. Cloning references and single generated chunks are short dense
speech that may hold less than 5% pause frames, so they use the quietest 2% instead (still > -24 dB on every known-bad
reference/film measured, < -29 dB on HTDemucs+DeepFilterNet3 cleaned ones). Signals never replace listening.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import wave
import numpy as np
from scipy import signal
from scipy.ndimage import median_filter

SAMPLE_RATE = 48000
TONE_BAND_HZ = (30, 4000)
TONE_EXCESS_DB = 8.
TONE_PERSISTENCE = .5
TONE_WINDOW_SECONDS = 2.
TONE_MIN_WINDOWS = 2  # one 2 s window of a held vowel shows harmonic lines; hum repeats in every window
MAX_PAUSE_DEPTH_DB = -25.
PAUSE_PERCENTILE = 5
SHORT_CLIP_PAUSE_PERCENTILE = 2
MAX_DC_OFFSET = .005
CLIP_LEVEL = .999
MAX_CLIPPED_FRACTION = 1e-4


def decode(path, start=None, duration=None):
    """First channel at unity gain (ffmpeg's -ac 1 adds +3 dB to duplicated stereo and fakes clipping)."""
    command = ['ffmpeg', '-v', 'error'] + (['-ss', str(start)] if start is not None else []) + ['-i', str(path)]
    command += (['-t', str(duration)] if duration is not None else []) + ['-map', 'a:0', '-af', 'pan=mono|c0=c0', '-ar', str(SAMPLE_RATE), '-f', 'f32le', '-']
    return np.frombuffer(subprocess.run(command, capture_output=True, check=True).stdout, '<f4').astype(np.float64)


def tonal_lines(x):
    size = int(TONE_WINDOW_SECONDS * SAMPLE_RATE)
    rows = []
    for start in range(0, len(x) - size + 1, size):
        window = x[start:start + size]
        if np.sqrt(np.mean(window ** 2)) < 1e-5:
            continue  # digital silence such as cover or tail padding
        frequencies, power = signal.welch(window, SAMPLE_RATE, nperseg=size // 2)
        level = 10 * np.log10(power + 1e-24)
        rows.append(level - median_filter(level, size=51, mode='nearest'))
    if not rows:
        return []
    band = (frequencies >= TONE_BAND_HZ[0]) & (frequencies <= TONE_BAND_HZ[1])
    hits = (np.array(rows)[:, band] > TONE_EXCESS_DB).sum(axis=0)
    persistence = hits / len(rows)
    lines = []
    # Adjacent FFT bins often tie on persistence. Rank the actual spectral peak next,
    # rather than letting NumPy's platform-dependent tie order name a neighbouring bin.
    excess = np.median(np.asarray(rows)[:, band], axis=0)
    for index in np.lexsort((frequencies[band], -excess, -persistence)):
        if persistence[index] < TONE_PERSISTENCE:
            break
        if hits[index] < TONE_MIN_WINDOWS:
            continue
        if all(abs(frequencies[band][index] - hz) > 4 for hz, _ in lines):
            lines.append((float(frequencies[band][index]), round(float(persistence[index]), 2)))
    return lines[:5]


VOICE_PAUSE_DROP_DB = 25.   # a line this much quieter in the pauses than in speech is the voice's own pitch, not hum
DEEP_PAUSE_FRAMES = 8        # 10 ms frames: a pause frame this far from speech holds no ringing of the voice
MIN_PAUSE_FRAMES = 10        # deep pause frames needed to judge by the pauses; fewer, and the pitch decides
PITCH_TOLERANCE = .1         # a frame's pitch peak within 10% of a line (or its 2nd, 3rd harmonic) sits on it
PITCH_SHARE = .25            # a line the voice's pitch peak sits on in this share of voiced frames is the voice


def pitch_peaks(x):
    """The strongest peak between 70 and 400 Hz of each voiced 40 ms frame (the pitch, or one of its harmonics)."""
    from align import activity
    active = activity(x, SAMPLE_RATE)
    hop, width = round(.01 * SAMPLE_RATE), round(.04 * SAMPLE_RATE)
    frequencies = np.fft.rfftfreq(8192, 1 / SAMPLE_RATE)
    band = (frequencies >= 70) & (frequencies <= 400)
    found = []
    for frame in np.flatnonzero(active)[::4]:
        window = x[frame * hop:frame * hop + width]
        if len(window) == width:
            found.append(frequencies[band][np.argmax(np.abs(np.fft.rfft(window * np.hanning(width), 8192))[band])])
    return np.array(found)


def voice_lines(x, lines):
    """Split stationary lines into hum and the voice's own pitch. A long list read evenly by a male voice holds its pitch
    near 120 Hz for seconds, which looks like a stationary line (the v3 film: seven of eight takes of one line failed as
    “hum at 115-127 Hz”). Hum stays in the pauses; a voice line is gone there (measured >= 80 ms from any speech). A
    clip with no such pauses is judged by pitch: a line on the speaker's pitch or a harmonic of it is the voice."""
    if not lines:
        return [], []
    from align import activity
    from scipy import ndimage
    active = activity(x, SAMPLE_RATE)
    if not active.any():
        return lines, []
    hop = round(.01 * SAMPLE_RATE)
    first, last = np.flatnonzero(active)[[0, -1]]
    deep = ~ndimage.binary_dilation(active, iterations=DEEP_PAUSE_FRAMES)
    deep[:first] = deep[last:] = False
    pauses, speech = np.flatnonzero(deep), np.flatnonzero(active)
    peaks = pitch_peaks(x) if len(pauses) < MIN_PAUSE_FRAMES else None
    hum, voice = [], []
    for hz, persistence in lines:
        if len(pauses) >= MIN_PAUSE_FRAMES:
            low, high = max(1., hz - 8), min(SAMPLE_RATE / 2 - 1, hz + 8)
            narrow = signal.sosfiltfilt(signal.butter(2, [low, high], btype='band', fs=SAMPLE_RATE, output='sos'), x)
            level = lambda frames: 20 * np.log10(np.sqrt(np.mean(np.concatenate([narrow[i * hop:(i + 1) * hop] for i in frames]) ** 2)) + 1e-12)
            is_voice = level(speech) - level(pauses) >= VOICE_PAUSE_DROP_DB
        else:
            on_line = [any(abs(peak - k * hz) <= PITCH_TOLERANCE * k * hz for k in (1, 2, 3)) for peak in peaks]
            is_voice = bool(on_line) and float(np.mean(on_line)) >= PITCH_SHARE
        (voice if is_voice else hum).append((hz, persistence))
    return hum, voice


def frame_levels(x, frame=.02, hop=.01):
    width, step = round(frame * SAMPLE_RATE), round(hop * SAMPLE_RATE)
    if len(x) < width:
        return np.zeros(0)
    count = 1 + (len(x) - width) // step
    return np.sqrt(np.mean(x[np.arange(width)[None] + step * np.arange(count)[:, None]] ** 2, axis=1))


def pause_depth(x, percentile=PAUSE_PERCENTILE):
    levels = frame_levels(x)
    levels = 20 * np.log10(levels[levels > 1e-6])
    return float(np.percentile(levels, percentile) - np.percentile(levels, 95)) if len(levels) else 0.


def measure(source, label=None, short_clip=False):
    x = np.asarray(source, dtype=np.float64) if isinstance(source, np.ndarray) else decode(source)
    if not len(x) or not np.isfinite(x).all() or not np.any(x):
        return {'source': label or str(source), 'status': 'FAIL', 'failures': ['empty, silent or non-finite audio']}
    clipped = float(np.mean(np.abs(x) >= CLIP_LEVEL))
    hum, voice = voice_lines(x, tonal_lines(x))
    report = {'source': label or (None if isinstance(source, np.ndarray) else str(source)), 'duration_seconds': round(len(x) / SAMPLE_RATE, 3),
              'tonal_lines_hz_persistence': hum, **({'voice_pitch_lines': voice} if voice else {}),
              'pause_percentile': SHORT_CLIP_PAUSE_PERCENTILE if short_clip else PAUSE_PERCENTILE,
              'pause_depth_db': round(pause_depth(x, SHORT_CLIP_PAUSE_PERCENTILE if short_clip else PAUSE_PERCENTILE), 1),
              'dc_offset': round(float(np.mean(x)), 5), 'peak_dbfs': round(float(20 * np.log10(np.abs(x).max())), 2),
              'clipped_fraction': clipped}
    failures = []
    if report['tonal_lines_hz_persistence']:
        failures.append('stationary tone/hum/music line at ' + ', '.join(f'{hz:.0f} Hz' for hz, _ in report['tonal_lines_hz_persistence']))
    if report['pause_depth_db'] > MAX_PAUSE_DEPTH_DB:
        failures.append(f"pauses not quiet ({report['pause_depth_db']} dB > {MAX_PAUSE_DEPTH_DB} dB): noise or music bed")
    if abs(report['dc_offset']) > MAX_DC_OFFSET:
        failures.append(f"DC offset {report['dc_offset']}")
    if clipped > MAX_CLIPPED_FRACTION:
        failures.append(f'clipping in {clipped:.2%} of samples')
    return {**report, 'status': 'FAIL' if failures else 'PASS', 'failures': failures}


def require_clean(source, context, label=None, short_clip=False):
    report = measure(source, label, short_clip)
    if report['status'] != 'PASS':
        raise ValueError(f"{context}: audio gate FAIL ({'; '.join(report['failures'])}). "
                         '参考音频请先用 scripts/clean_reference.py 分离人声/降噪并重新导入；生成的旁白请重新生成该段。')
    return report


# ITU-R BS.1770 K-weighting at 48 kHz (pre-filter shelf, then RLB high-pass).
K_SHELF = ([1.53512485958697, -2.69169618940638, 1.19839281085285], [1., -1.69065929318241, .73248077421585])
K_HIGHPASS = ([1., -2., 1.], [1., -1.99004745483398, .99007225036621])
# A layered mix (narration + foley/music/effects) is accepted when the narration is provably intact inside it:
# mix minus the known layers must equal the normalized narration (dual mono) to within these limits.
MAX_NARRATION_RESIDUAL_DB = -30.   # residual RMS relative to the narration RMS
MIN_NARRATION_CORRELATION = .999
MAX_NARRATION_GAIN_ERROR_DB = .5
MIN_VOICE_MARGIN_DB = 6.           # narration over all layers, both measured on speech frames


def finite(value):
    return value if np.isfinite(value) else None


def db(value):
    return round(float(20 * np.log10(max(float(value), 1e-12))), 2)


def loudness_lufs(samples, rate=SAMPLE_RATE):
    """Integrated loudness (BS.1770-4: K-weighting, 400 ms blocks, -70 LUFS absolute and -10 LU relative gates).
    Mono input counts once; stereo sums both channels, so dual-mono measures 3 LU louder than its mono source."""
    if rate != SAMPLE_RATE:
        raise ValueError('loudness_lufs uses the 48 kHz K-weighting coefficients')
    x = np.asarray(samples)
    x = x[:, None] if x.ndim == 1 else x
    block, hop = round(.4 * rate), round(.1 * rate)
    if len(x) < block:
        return float('-inf')
    energy = np.zeros(len(x) + 1)
    for channel in range(x.shape[1]):  # one channel at a time keeps long films within memory
        weighted = signal.lfilter(*K_HIGHPASS, signal.lfilter(*K_SHELF, x[:, channel].astype(np.float64)))
        energy[1:] += np.cumsum(weighted ** 2)
    starts = np.arange(0, len(x) - block + 1, hop)
    power = (energy[starts + block] - energy[starts]) / block
    level = -.691 + 10 * np.log10(power + 1e-30)
    power = power[level > -70]
    if not len(power):
        return float('-inf')
    relative = -.691 + 10 * np.log10(power.mean()) - 10
    power = power[-.691 + 10 * np.log10(power) > relative]
    return round(float(-.691 + 10 * np.log10(power.mean())), 2)


def speech_frames(narration, frame=.02, hop=.01):
    """Frames where the narration speaks, by the same relative rule align.activity uses for caption timing."""
    levels = 20 * np.log10(frame_levels(np.asarray(narration, dtype=np.float64), frame, hop) + 1e-9)
    if not len(levels):
        return np.zeros(0, dtype=bool)
    return levels > max(np.percentile(levels, 95) - 35, np.percentile(levels, 10) + 6)


def framed_rms(x, mask, frame=.02, hop=.01):
    levels = frame_levels(x, frame, hop)[:len(mask)]
    selected = levels[mask[:len(levels)]]
    return float(np.sqrt(np.mean(selected ** 2))) if len(selected) else 0.


def framed_activity(x, count, threshold=1e-5, hop=.01):
    """Per 20 ms analysis frame (10 ms hop): does the signal rise above threshold anywhere inside it."""
    step = round(hop * SAMPLE_RATE)
    loud = np.abs(x) > threshold
    blocks = np.zeros(count + 1, dtype=bool)
    usable = min(len(loud) // step, count + 1)
    blocks[:usable] = loud[:usable * step].reshape(usable, step).any(axis=1)
    return blocks[:count] | blocks[1:count + 1]


def decode_layer(path, rate=SAMPLE_RATE):
    """Stereo float samples exactly as the renderer's mixer reads a layer (aformat=channel_layouts=stereo)."""
    command = ['ffmpeg', '-v', 'error', '-i', str(path), '-af', 'aformat=channel_layouts=stereo', '-ar', str(rate), '-f', 'f32le', '-']
    return np.frombuffer(subprocess.run(command, capture_output=True, check=True).stdout, '<f4').reshape(-1, 2)


def layer_levels(path, gain):
    """Peak and integrated loudness of a layer file at its mixer gain (what audio-mix-report.json records)."""
    samples = decode_layer(path) * np.float32(gain)
    loudness = loudness_lufs(samples)
    return {'peak_dbfs': db(np.abs(samples).max()) if len(samples) and np.any(samples) else None,
            'loudness_lufs': loudness if np.isfinite(loudness) else None}


def place_layer(layer, frames, rate=SAMPLE_RATE, samples=None):
    """One layer at its mixer gain and start (adelay works in whole milliseconds), cut to the narration length."""
    samples = decode_layer(layer['source'], rate) if samples is None else samples
    placed = np.zeros((frames, 2), dtype=np.float32)
    offset = round(float(layer.get('start', 0)) * 1000) * rate // 1000
    count = max(0, min(len(samples), frames - offset))
    placed[offset:offset + count] = np.float32(layer['gain']) * samples[:count]
    return placed


def read_pcm(path):
    """16-bit PCM wav as float32 frames x channels (mono stays 1-D)."""
    with wave.open(str(path), 'rb') as stream:
        if stream.getsampwidth() != 2:
            raise ValueError(f'{path}: expected 16-bit PCM')
        channels = stream.getnchannels()
        samples = np.frombuffer(stream.readframes(stream.getnframes()), '<i2').astype(np.float32) / 32768
    return samples if channels == 1 else samples.reshape(-1, channels)


def measure_layered_mix(narration, mix, layers, rate=SAMPLE_RATE):
    """Gate for a mix with intended layers (foley, music, effects).

    The narration stem keeps the full narration gate (hum, pause depth, DC, clipping). The mix must equal the stem
    (dual mono, unity gain) plus the known layers, so the voice in the film is provably the gated stem, intact; the
    non-music part of the mix must hold no stationary tone (foley and effects never hum); the mix must not clip or
    carry DC; and the layers must stay MIN_VOICE_MARGIN_DB under the voice while it speaks. Music is exempt from the
    tone check only: it is tonal by design and was requested by the user."""
    narration = np.asarray(narration, dtype=np.float32)
    mix = np.asarray(mix, dtype=np.float32)
    stem = measure(narration.astype(np.float64), 'work/normalized-narration.wav')
    failures = [f'narration stem: {failure}' for failure in stem['failures']]
    if abs(len(narration) - len(mix)) > rate // 100:
        failures.append(f'mix length differs from the narration by {abs(len(narration) - len(mix)) / rate:.3f}s')
    frames = min(len(narration), len(mix))
    narration, mix = narration[:frames], mix[:frames]
    placed = [place_layer(layer, frames, rate) for layer in layers]
    known = np.sum(placed, axis=0, dtype=np.float32) if placed else np.zeros((frames, 2), np.float32)
    recovered = mix - known
    reference = float(np.sqrt(np.mean(narration.astype(np.float64) ** 2))) + 1e-12
    residual = float(np.sqrt(np.mean((recovered - narration[:, None]).astype(np.float64) ** 2)))
    side = float(np.sqrt(np.mean(((recovered[:, 0] - recovered[:, 1]) / 2).astype(np.float64) ** 2)))
    middle = recovered.mean(axis=1, dtype=np.float64)
    del recovered
    correlation = float(np.corrcoef(middle, narration)[0, 1]) if np.std(middle) > 0 else 0.
    gain = float(np.dot(middle, narration) / max(float(np.dot(narration, narration)), 1e-12))
    del middle
    speaking = speech_frames(narration)
    voice = framed_rms(narration, speaking)
    music = [index for index, layer in enumerate(layers) if layer.get('kind') == 'music']
    nonmusic = mix.mean(axis=1, dtype=np.float64) - sum((placed[index].mean(axis=1, dtype=np.float64) for index in music), np.zeros(frames))
    clipped = float(np.mean(np.abs(mix) >= CLIP_LEVEL))
    beds = framed_rms(known.mean(axis=1, dtype=np.float64), speaking)
    report = {'narration_residual_db': db(residual / reference), 'narration_side_db': db(side / reference),
              'narration_correlation': round(correlation, 6), 'narration_gain_db': db(gain) if gain > 0 else None,
              'tonal_lines_without_music_hz_persistence': tonal_lines(nonmusic),
              'peak_dbfs': db(np.abs(mix).max()), 'clipped_fraction': clipped,
              'dc_offset': [round(float(value), 5) for value in mix.mean(axis=0, dtype=np.float64)],
              'loudness_lufs': loudness_lufs(mix, rate), 'narration_loudness_lufs': loudness_lufs(narration, rate),
              'narration_speech_rms_dbfs': db(voice), 'layers_speech_rms_dbfs': db(beds),
              'voice_margin_db': round(db(voice) - db(beds), 2)}
    del nonmusic
    if report['narration_residual_db'] > MAX_NARRATION_RESIDUAL_DB:
        failures.append(f"narration is not intact in the mix (residual {report['narration_residual_db']} dB)")
    if correlation < MIN_NARRATION_CORRELATION:
        failures.append(f'mix minus layers no longer matches the narration (correlation {correlation:.4f})')
    if gain <= 0 or abs(report['narration_gain_db']) > MAX_NARRATION_GAIN_ERROR_DB:
        failures.append(f"narration gain changed in the mix ({report['narration_gain_db']} dB)")
    if report['tonal_lines_without_music_hz_persistence']:
        failures.append('stationary tone/hum outside the music at ' + ', '.join(
            f'{hz:.0f} Hz' for hz, _ in report['tonal_lines_without_music_hz_persistence']))
    if clipped > MAX_CLIPPED_FRACTION:
        failures.append(f'mix clips in {clipped:.2%} of samples')
    if max(abs(value) for value in report['dc_offset']) > MAX_DC_OFFSET:
        failures.append(f"mix DC offset {report['dc_offset']}")
    if report['voice_margin_db'] < MIN_VOICE_MARGIN_DB:
        failures.append(f"layers mask the voice: only {report['voice_margin_db']} dB under the narration while it speaks")
    entries = []
    for layer, alone in zip(layers, placed):
        middle = alone.mean(axis=1, dtype=np.float64)
        active = framed_activity(middle, len(speaking))
        during, between = framed_rms(middle, speaking & active), framed_rms(middle, ~speaking & active)
        entries.append({'kind': layer.get('kind'), 'source': Path(layer['source']).name, 'gain': layer['gain'],
                        'start': layer.get('start', 0), 'peak_dbfs': db(np.abs(alone).max()),
                        'loudness_lufs': finite(loudness_lufs(alone, rate)), 'active_share': round(float(active.mean()), 3) if len(active) else 0.,
                        'speech_rms_dbfs': db(during) if during else None, 'pause_rms_dbfs': db(between) if between else None,
                        'pause_minus_speech_db': round(db(between) - db(during), 2) if during and between else None})
    return {'source': f"work/mix.wav (narration + {len(layers)} layer{'s' if len(layers) != 1 else ''})", 'narration': stem,
            'mix': report, 'layers': entries, 'status': 'FAIL' if failures else 'PASS', 'failures': failures}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', nargs='+', type=Path)
    parser.add_argument('--short-clip', action='store_true', help='Cloning reference or single chunk: quietest 2%% pause statistic')
    arguments = parser.parse_args()
    results = [measure(path, short_clip=arguments.short_clip) for path in arguments.files]
    for result in results:
        print(json.dumps(result, ensure_ascii=False))
    sys.exit(0 if all(result['status'] == 'PASS' for result in results) else 1)
