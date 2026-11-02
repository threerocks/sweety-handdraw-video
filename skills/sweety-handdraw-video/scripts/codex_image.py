#!/usr/bin/env python3
"""Generate one image through the Codex CLI's built-in image tool (the route for hosts without an image tool, such as
Claude Code) and copy it into the job. The prompt file must already be bound by art_direction.py: this only transports it.

    codex_image.py PROMPT_FILE OUTPUT_PNG [--shape portrait|square|landscape] [--ref IMAGE[=NOTE] ...]
                   [--model NAME] [--effort low] [--attempts 2] [--codex codex]

Proven details kept from the films: stdin is /dev/null (codex otherwise waits on it); the prompt argument comes before
the -i references (-i takes several values and would swallow it); only images in this run's own thread directory under
$CODEX_HOME/generated_images count, so parallel runs never take each other's output. New: the canvas the tool returns
follows the first reference's aspect, so every reference is padded onto the requested canvas aspect first.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SHAPES = {'portrait': (2, 3, '1024x1536 portrait (2:3)'), 'square': (1, 1, '1024x1024 square'),
          'landscape': (3, 2, '1536x1024 landscape (3:2)')}
DEFAULT_MODEL = os.environ.get('CODEX_IMAGE_MODEL', 'gpt-6-astra')
DEFAULT_EFFORT = os.environ.get('CODEX_IMAGE_EFFORT', 'low')


def generated_root():
    return Path(os.environ.get('CODEX_HOME') or Path.home() / '.codex') / 'generated_images'


def instruction(prompt, shape, notes):
    reference = ''
    if notes:
        reference = ('Attached reference image(s), in order: ' + '; '.join(f'image {k + 1} = {note}' for k, note in enumerate(notes))
                     + '. Pass them to the image generation tool as reference images so the character design (face, hair, '
                     'glasses, clothing colours, proportions) and the line/colour style stay identical. ')
    return ('Task: call your built-in image generation tool exactly once to create one new image. '
            f'Canvas: {SHAPES[shape][2]}. {reference}'
            'Use the text between <prompt> tags as the image prompt, verbatim, without adding style words. '
            'Do not draw with code, do not edit or post-process, do not generate variants. '
            'When the tool returns, reply with the single word DONE.\n\n'
            f'<prompt>\n{prompt}\n</prompt>')


def pad_to_shape(source, shape, directory):
    """The reference on a canvas of the requested aspect (centred, border colour), or the original if it already fits."""
    from PIL import Image
    import numpy as np
    wide, tall, _ = SHAPES[shape]
    with Image.open(source) as opened:
        image = opened.convert('RGBA')
    flat = Image.new('RGB', image.size, (255, 255, 255))
    flat.paste(image, mask=image.getchannel('A'))
    width, height = flat.size
    target = wide / tall
    if abs(width / height - target) <= .01 * target:
        return Path(source)
    size = (width, round(width / target)) if width / height > target else (round(height * target), height)
    pixels = np.asarray(flat)
    border = np.concatenate([pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]])
    colour = tuple(int(value) for value in np.median(border, axis=0))
    canvas = Image.new('RGB', size, colour)
    canvas.paste(flat, ((size[0] - width) // 2, (size[1] - height) // 2))
    padded = Path(directory) / f'{len(list(Path(directory).iterdir())):02d}-{Path(source).stem}-{shape}.png'
    canvas.save(padded)
    return padded


def thread_of(stdout):
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            found = event.get('thread_id') or (event.get('thread') or {}).get('id')
            if found:
                return found
    return None


def command(prompt, shape, refs, model, effort, codex, cwd):
    """refs: [(padded path, note)]. The prompt (one argument) precedes every -i."""
    line = [codex, 'exec', '--json', '--skip-git-repo-check', '-s', 'read-only', '-C', str(cwd)]
    if model:
        line += ['-m', model]
    if effort:
        line += ['-c', f'model_reasoning_effort="{effort}"']
    line.append(instruction(prompt, shape, [note for _, note in refs]))
    for path, _ in refs:
        line += ['-i', str(path)]
    return line


def run(prompt_file, output, refs=(), shape='square', model=DEFAULT_MODEL, effort=DEFAULT_EFFORT, attempts=2, codex='codex',
        cwd=None, timeout=900):
    """Generate; refs are (path, note). Returns a result dict; raises RuntimeError when no attempt produced an image."""
    prompt = Path(prompt_file).read_text().strip()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    with tempfile.TemporaryDirectory(prefix='codex-refs-') as scratch:
        padded = [(pad_to_shape(path, shape, scratch), note) for path, note in refs]
        for attempt in range(attempts):
            started = time.time()
            process = subprocess.run(command(prompt, shape, padded, model, effort, codex, cwd or Path.cwd()), capture_output=True,
                                     text=True, timeout=timeout, stdin=subprocess.DEVNULL)
            thread = thread_of(process.stdout)
            folder = generated_root() / thread if thread else None
            # Only this thread's own output counts: parallel runs must never pick up each other's images.
            candidates = sorted(folder.glob('*.png'), key=lambda path: path.stat().st_mtime) if folder and folder.is_dir() else []
            if candidates:
                shutil.copyfile(candidates[-1], output)
                return {'output': str(output), 'source': str(candidates[-1]), 'thread': thread, 'shape': shape,
                        'padded_refs': [str(path) != str(original) for (path, _), (original, _) in zip(padded, refs)],
                        'seconds': round(time.time() - started, 1), 'attempt': attempt + 1}
            failures.append(f'attempt {attempt + 1}: no image (exit {process.returncode}, thread {thread})\n'
                            + process.stdout[-1500:] + process.stderr[-1500:])
    raise RuntimeError('\n'.join(failures))


def parse_ref(item):
    path, _, note = item.partition('=')
    return Path(path).resolve(), note or 'style and character reference'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('prompt_file')
    parser.add_argument('output')
    parser.add_argument('--ref', action='append', default=[], help='PATH=NOTE, in the order the prompt names them')
    parser.add_argument('--shape', choices=sorted(SHAPES), default='square')
    parser.add_argument('--model', default=DEFAULT_MODEL, help='codex model ("" = the codex default); env CODEX_IMAGE_MODEL')
    parser.add_argument('--effort', default=DEFAULT_EFFORT, help='model_reasoning_effort ("" = the codex default)')
    parser.add_argument('--attempts', type=int, default=2)
    parser.add_argument('--codex', default=os.environ.get('CODEX_BIN', 'codex'), help='codex executable')
    args = parser.parse_args(argv)
    try:
        result = run(args.prompt_file, args.output, [parse_ref(item) for item in args.ref], args.shape, args.model, args.effort,
                     args.attempts, args.codex)
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        print(error, file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
