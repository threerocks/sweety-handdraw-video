#!/usr/bin/env python3
"""Generate narration per semantic chunk with an explicitly configured, already installed VoxCPM2 (MLX).

Modes follow the official VoxCPM2 guide (references/voxcpm2.md): combined = Hi-Fi cloning with the same
clean clip as reference and prompt plus its exact transcript; reference = timbre only, optional style
instruction; design = voice from a text description (varies per call, so lock a designed reference).
"""
import argparse
import hashlib
import json
from pathlib import Path
from voice_studio import voice_reference, load_voice_catalog
from narration_master import build_master, probe_terms

MODES = ('combined', 'reference', 'design')


def resolved_options(options, root):
    catalog = load_voice_catalog(root)
    identifier = options.get('voice', catalog['default'])
    profile = next((voice for voice in catalog['voices'] if voice['id'] == identifier), {})
    approved = profile.get('verified_generation_parameters', {})
    fields = ('mode', 'cfg_value', 'timesteps', 'seed', 'model_path', 'enhancement_model_path', 'style')
    defaults = {key: approved[key] for key in fields if key in approved}
    return {**defaults, 'voice': identifier, **options}


def enhancement_fingerprint(path):
    if not path:
        return None
    directory = Path(path).expanduser().resolve()
    weights = directory/'model.safetensors'
    if not directory.is_dir() or not weights.is_file():
        raise ValueError('Speech cleanup requires an explicitly configured, installed local model')
    return {'model_path':str(directory), 'weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest()}


def generation_kwargs(params, text, reference=None, transcript=None):
    """Official defaults: cfg 2.0, 10 steps; max length = 6 patches per text token + 10 (retry_badcase)."""
    kwargs = {'text': text, 'inference_timesteps': params['timesteps'], 'cfg_value': params['cfg_value'], 'max_tokens': len(text) * 6 + 10}
    if params['mode'] == 'combined':
        return {**kwargs, 'ref_audio': str(reference), 'prompt_audio': str(reference), 'prompt_text': transcript}
    if params['mode'] == 'reference':
        return {**kwargs, 'ref_audio': str(reference), **({'instruct': params['style']} if params.get('style') else {})}
    return {**kwargs, 'instruct': params['design']}


def generate(job, probe=False):
    import mlx.core as mx
    import numpy as np
    from mlx_audio.tts.utils import load
    root = Path(job).resolve()
    manifest = json.loads((root/'job.json').read_text())
    options = resolved_options(manifest.get('voxcpm', {}), root)
    model_path = Path(options.get('model_path', '')).expanduser()
    if not options.get('model_path') or not (model_path/'config.json').is_file():
        raise ValueError('Set voxcpm.model_path to an already installed local VoxCPM2 model')
    params = {'mode': options.get('mode', 'combined'), 'cfg_value': float(options.get('cfg_value', 2.)),
              'timesteps': int(options.get('timesteps', 10)), 'style': options.get('style'), 'design': options.get('design')}
    if params['mode'] not in MODES or (params['mode'] == 'design') != bool(params['design']):
        raise ValueError('VoxCPM2 mode must be combined, reference or design; voxcpm.design is required in design mode only')
    reference, transcript = voice_reference(options, root) if params['mode'] != 'design' else (None, None)
    enhancement = enhancement_fingerprint(options.get('enhancement_model_path'))
    model = load(str(model_path.resolve()))
    cleaner = None
    if enhancement:
        from mlx_audio.sts.models.deepfilternet import DeepFilterNetModel
        cleaner = DeepFilterNetModel.from_pretrained(enhancement['model_path'], subfolder=None)

    def synthesize(text, seed):
        mx.random.seed(seed)
        result = list(model.generate(**generation_kwargs(params, text, reference, transcript)))
        samples = np.concatenate([np.asarray(chunk.audio) for chunk in result]).astype(np.float32)
        if cleaner is not None:
            if result[0].sample_rate != 48000:
                raise ValueError('Configured speech cleanup requires a 48 kHz narration waveform')
            samples = np.asarray(cleaner.enhance_array(samples), dtype=np.float32)
        return samples, result[0].sample_rate

    identity = {'engine': 'voxcpm2-mlx', 'model_path': str(model_path.resolve()), 'voice': options.get('voice'), **params, 'enhancement': enhancement,
                'reference_sha256': hashlib.sha256(reference.read_bytes()).hexdigest() if reference else None, 'reference_text': transcript}
    if probe:
        report = probe_terms(root, manifest, synthesize, options.get('asr', True))
        print(json.dumps([{key: item[key] for key in ('term', 'status', 'advice') if key in item} for item in report], ensure_ascii=False, indent=1))
        return
    target, alignment = build_master(root, manifest, synthesize, identity, options)
    print('READY', target, alignment, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('job',type=Path)
    parser.add_argument('--probe-terms', action='store_true', help='voice each English word / number in its clause first (work/term-probe)')
    arguments = parser.parse_args()
    generate(arguments.job, arguments.probe_terms)
