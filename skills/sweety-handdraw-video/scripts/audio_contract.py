"""Make narration-only delivery and a complete, short ending explicit invariants."""
from pathlib import Path
import hashlib
import math
import numpy as np


def resolve_audio_layers(root, manifest):
    options = manifest.get('audio', {})
    music = options.get('music', None)
    from foley import parse_options
    parse_options(options.get('foley'))  # fail early on a bad audio.foley; the layer itself needs the finished timeline
    from sfx import parse_options as parse_sfx
    parse_sfx(options.get('sfx'))
    layers = []
    if music not in (None, False):
        from music_synth import parse_source
        # A user's own music file needs their instruction. A builtin bed is composed here in code (nothing sampled), so the
        # director may choose it for the film's emotional continuity, with the reason written down.
        directed = isinstance(music, dict) and parse_source(music.get('source')) and str(music.get('director_reason', '')).strip()
        if not isinstance(music, dict) or not (music.get('user_instruction') or directed):
            raise ValueError('Background music needs an explicit user instruction, or a builtin:<mood> bed with director_reason')
        layer = {**music, 'kind': 'music', 'start': 0.}
        from music_synth import parse_source, render_source, DEFAULT_GAIN
        if parse_source(music.get('source')):
            # builtin:<mood> is composed and played by music_synth.py at the narration's length, then prepared
            # (fades, sidechain ducking, dips) exactly like a user's file. Nothing is sampled.
            if music.get('prepared'):
                raise ValueError('Builtin music is always prepared by the build; remove prepared:true')
            generated, record = render_source(Path(root) / 'work', music)
            layer.update(source=str(generated), builtin=record)
            layer.setdefault('gain', DEFAULT_GAIN)
        if not music.get('prepared'):
            # The raw source becomes a deterministic bed during build: loop crossfade, fades, sidechain
            # under the narration and optional dips (music_bed.py). prepared:true keeps a host-made bed.
            from music_bed import prepare
            bed, preparation = prepare(root, Path(root) / 'work', layer)
            layer['source'] = str(bed)
            layer['preparation'] = {key: preparation[key] for key in
                                    ('crossfade_seconds', 'fade_in_seconds', 'fade_out_seconds', 'duck', 'dips')}
        layers.append(layer)
    for effect in options.get('sound_effects', []):
        if not effect.get('reason') or not effect.get('beat_id'):
            raise ValueError('A sound effect needs a narrative reason and the beat it supports')
        layers.append({**effect, 'kind': 'effect'})
    beat_ids = {beat['id'] for page in manifest.get('pages', []) for beat in page['beats']}
    resolved = []
    for layer in layers:
        source = (Path(root) / layer['source']).resolve()
        start, gain = float(layer.get('start', 0)), float(layer.get('gain', .12))
        if not source.is_file() or not math.isfinite(start + gain) or start < 0 or not 0 < gain <= .3:
            raise ValueError('Audio layers need an existing source, a nonnegative time and gain within (0, .3]')
        if layer['kind'] == 'effect' and layer['beat_id'] not in beat_ids:
            raise ValueError('Sound effect points to an unknown narration beat')
        from audio_gate import layer_levels
        resolved.append({**layer, 'source': str(source), 'start': start, 'gain': gain,
                         'sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'levels': layer_levels(source, gain)})
    builtin = next((layer['builtin']['mood'] for layer in resolved if layer.get('builtin')), None)
    return {'background_music': any(layer['kind'] == 'music' for layer in resolved),
            'layers': resolved,
            'delivery_notice': '默认未添加背景音乐；可自行合成，或在发布平台选择配乐。' if not music else
                               f'按用户明确要求加入了程序合成的背景音乐（builtin:{builtin}）。' if builtin else '按用户明确要求加入了背景音乐。'}


def measure_voice_tail(samples, sample_rate):
    values = np.asarray(samples, dtype=float) / 32768.
    block = max(1, round(.01 * sample_rate))
    blocks = np.pad(values, (0, (-len(values)) % block)).reshape(-1, block)
    rms = np.sqrt(np.mean(blocks ** 2, axis=1))
    active = np.flatnonzero(rms > .001)
    if not len(active):
        raise ValueError('Narration has no measurable voice')
    last_active = min(len(values), (int(active[-1]) + 1) * block) / sample_rate
    return {'last_active_seconds': last_active, 'source_tail_seconds': len(values) / sample_rate - last_active,
            'activity_threshold_dbfs': -60, 'analysis_window_seconds': .01}


def validate_ending(manifest, samples, sample_rate):
    hold = float(manifest.get('audio', {}).get('end_hold_seconds', .45))
    closing = manifest.get('ending', {}).get('closing_text')
    script = ''.join(beat['text'] for page in manifest['pages'] for beat in page['beats'])
    if closing and not script.endswith(closing):
        raise ValueError('The requested closing text must be included in the complete narration before synthesis')
    if not math.isfinite(hold) or not .2 <= hold <= .8:
        raise ValueError('Ending hold must be 0.2–0.8 seconds after the complete narration master')
    tail = measure_voice_tail(samples, sample_rate)
    if tail['source_tail_seconds'] < .12:
        raise ValueError('Narration ends too close to active sound; inspect the final word before locking the master')
    if tail['source_tail_seconds'] + hold > 1.2:
        raise ValueError('Ending silence exceeds 1.2 seconds; revise only the verified outer tail before locking')
    return {**tail, 'hold_seconds': hold, 'total_tail_seconds': tail['source_tail_seconds'] + hold,
            'closing_text': closing, 'scope': 'Protects samples and pacing; complete pronunciation still needs final-track review'}
