"""Keep one complete narration waveform authoritative for every visual timing.

The master is generated per semantic chunk (one beat, or sentences of a long beat), each gated and
regenerated on failure, then joined with measured pauses; long one-shot generation drifts and buzzes.
"""
from pathlib import Path
import hashlib
import json
import math
import re
import subprocess
import wave
import numpy as np

SAMPLE_RATE = 48000
MAX_CHUNK_CHARS = 100
MIN_CHUNK_CHARS = 8
SENTENCE_END = '。！？!?…'
CLAUSE_END = '，,、；;：:'
# Spoken characters per second of speech span. Accepted VoxCPM2 films ran 6.3 (snake) and 6.7 (wolf)
# chars/s overall and 5.9-7.5 per sentence group, so 6 would reject both accepted voices; skipped or
# babbled text is caught by ASR coverage, the rate only flags gross stretching or truncation.
SPEECH_RATE_RANGE = (3., 8.)
GENERATION_ATTEMPTS = 3
EDGE_SECONDS = (.06, .1)
# Silence between chunks (speech end to speech start). Measured at punctuation with align.activity:
# accepted VoxCPM2 films (snake, wolf) sentence median .15/.23 s, clause .15/.21 s; 89 sentence ends of
# clean human long-form narration median .60 s. Sentence/beat stay near the model's own cadence, a page
# turn takes the human sentence pause; jobs may override per kind (join_pause_seconds).
JOIN_PAUSE_SECONDS = {'clause': .2, 'sentence': .3, 'beat': .4, 'page': .6}
END_TAIL_SECONDS = .3
# Where a chunk really ends. The activity threshold alone cut 0.1 s after "speech", often through a breath or the
# model's own after-sound, with no fade: listeners heard a tick or a clipped extra syllable at sentence ends.
TAIL_QUIET_DB = -40.      # the last syllable has ended once the level stays this far under the chunk peak...
TAIL_QUIET_SECONDS = .03  # ...for this long
TAIL_BURST_DB = -30.      # a sound this loud after that quiet is not speech we asked for (an artifact after the sentence)
TAIL_LOUD_DB = -25.       # a chunk that still ends this loud, with no quiet after its last word, was cut off
TAIL_KEEP_SECONDS = .12   # natural decay and breath kept after the last syllable
MIN_SYLLABLE_SECONDS = .12  # a stop consonant's closure is shorter: no quiet inside this ends the last syllable
ENGLISH_ATTEMPTS = 6      # seeds tried for a chunk whose English words an English ear must hear
CHECKS_VERSION = 4        # 2: the audio actually kept is checked again (complete ending, audible English); 3: short English words exact;
                          # 4: a recorded tail cut inside the last word fails even when recognition still "hears" the word
INSIDE_WORD_SECONDS = .1  # a cut this far before the recognized end of the last word has cut the word itself
FADE_OUT_SECONDS = .05    # shortest raised-cosine fade at a cut, so no edge lands on a sound
FADE_IN_SECONDS = .01


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def say_map(manifest, beat=None):
    """Display term -> spoken form: narration.say for the whole job, a beat's own say on top. Subtitles show the
    term as written (Grok, 70万); the voice reads the spoken form (格罗克, 七十万) and ASR checks that."""
    merged = {**((manifest.get('narration') or {}).get('say') or {}), **(((beat or {}).get('say')) or {})}
    for key, value in merged.items():
        if not isinstance(key, str) or not key.strip() or not isinstance(value, str) or not value.strip():
            raise ValueError('narration.say / beat.say map written terms to nonempty spoken text')
    return merged


def say_pattern(mapping):
    # Latin terms match whole words only (Amp is not the start of Ample); longest first (70万 before 70).
    def term(key):
        latin = key.isascii() and any(ch.isalpha() for ch in key)
        body = f'(?i:{re.escape(key)})' if latin else re.escape(key)  # GROK, grok and Grok are one term
        return ('(?<![A-Za-z])' if key[0].isascii() and key[0].isalpha() else '') + body + \
               ('(?![A-Za-z])' if key[-1].isascii() and key[-1].isalpha() else '')
    return re.compile('|'.join(term(key) for key in sorted(mapping, key=len, reverse=True)))


def spoken_text(manifest, beat, text=None):
    """What the voice reads for a beat (or a piece of its display text)."""
    text = beat['text'] if text is None else text
    mapping = say_map(manifest, beat)
    folded = {key.lower(): value for key, value in mapping.items()}
    return say_pattern(mapping).sub(lambda match: mapping.get(match.group(0), folded.get(match.group(0).lower(), match.group(0))), text) if mapping else text


def display_text(manifest):
    return ''.join(beat['text'] for page in manifest['pages'] for beat in page['beats'])


def script_text(manifest):
    """The script a master is generated from: the display text, plus the spoken text when a say map changes it."""
    shown = display_text(manifest)
    said = ''.join(spoken_text(manifest, beat) for page in manifest['pages'] for beat in page['beats'])
    return shown if said == shown else shown + '\n[say]\n' + said


def generation_target(root, manifest):
    """Master path inside the job plus the complete script; the closing must already be in it."""
    settings = manifest.get('narration', {})
    target = (Path(root) / settings.get('source', 'inputs/narration/master.wav')).resolve()
    if not target.is_relative_to(Path(root).resolve()):
        raise ValueError('Generated narration master must stay inside the job')
    text = script_text(manifest)
    closing = manifest.get('ending', {}).get('closing_text')
    if closing and not display_text(manifest).endswith(closing):
        raise ValueError('Add the requested closing to the complete script before generating narration')
    return target, text


def pieces(text, breaks):
    return re.findall('[^' + breaks + ']+[' + breaks + ']*|[' + breaks + ']+', text)


def pack(parts, limit):
    packed = []
    for part in parts:
        if packed and len(packed[-1]) + len(part) <= limit:
            packed[-1] += part
        else:
            packed.append(part)
    return packed


def split_long(text, limit=MAX_CHUNK_CHARS):
    """A beat over the limit splits at sentence ends; only a single over-long sentence splits at clauses."""
    if len(text) <= limit:
        return [text]
    parts = []
    for sentence in pieces(text, SENTENCE_END):
        parts += [sentence] if len(sentence) <= limit else pack(pieces(sentence, CLAUSE_END), limit)
    return pack(parts, limit)


def narration_chunks(manifest, limit=MAX_CHUNK_CHARS):
    """Synthesis units: one beat each; beats ending mid-sentence or tiny ones join the next beat."""
    from align import spoken
    chunks = []
    for page_index, page in enumerate(manifest['pages']):
        for beat in page['beats']:
            for part, shown in enumerate(split_long(beat['text'], limit)):
                text = spoken_text(manifest, beat, shown)
                unit = {'beat': beat['id'], 'page': page_index, 'text': text, **({'show': shown} if shown != text else {})}
                last = chunks[-1] if chunks else None
                if last and part == 0 and (last['text'][-1] not in SENTENCE_END or len(spoken(last['text'])) < MIN_CHUNK_CHARS):
                    last['text'] += text
                    last['units'].append(unit)
                    continue
                join = None if not last else 'clause' if last['text'][-1] not in SENTENCE_END else 'sentence' if last['units'][-1]['beat'] == beat['id'] \
                    else 'beat' if last['units'][-1]['page'] == page_index else 'page'
                chunks.append({'text': text, 'units': [unit], 'join': join})
    if len(chunks) > 1 and len(spoken(chunks[-1]['text'])) < MIN_CHUNK_CHARS:
        tail = chunks.pop()
        chunks[-1]['text'] += tail['text']
        chunks[-1]['units'] += tail['units']
    return chunks


def to_rate(samples, rate):
    from scipy.signal import resample_poly
    samples = np.asarray(samples, dtype=np.float64).reshape(-1)
    if rate == SAMPLE_RATE:
        return samples
    factor = math.gcd(int(rate), SAMPLE_RATE)
    return resample_poly(samples, SAMPLE_RATE // factor, int(rate) // factor)


def write_pcm(path, samples):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + '.writing.wav')
    with wave.open(str(temporary), 'wb') as stream:
        stream.setparams((1, 2, SAMPLE_RATE, 0, 'NONE', 'not compressed'))
        stream.writeframes(np.clip(np.rint(np.asarray(samples) * 32767), -32768, 32767).astype('<i2').tobytes())
    temporary.replace(path)


def read_pcm(path):
    with wave.open(str(path)) as stream:
        return np.frombuffer(stream.readframes(stream.getnframes()), dtype='<i2').astype(np.float64) / 32767


def frame_levels(samples, rate=SAMPLE_RATE):
    """Level of 20 ms frames every 10 ms, in dB under the chunk peak."""
    hop, width = round(.01 * rate), round(.02 * rate)
    if len(samples) < width:
        return np.full(1, -120.)
    peak = max(float(np.abs(samples).max()), 1e-9)
    starts = np.arange(0, len(samples) - width + 1, hop)
    energy = np.concatenate([[0.], np.cumsum(np.asarray(samples, float) ** 2)])
    rms = np.sqrt(np.maximum(energy[starts + width] - energy[starts], 0.) / width)
    return 20 * np.log10(np.maximum(rms, 1e-9) / peak)


def speech_tail(samples, words=None, rate=SAMPLE_RATE):
    """(cut seconds or None, report). The last syllable ends at the first quiet run after the last word starts; a burst
    after that quiet is dropped; None when the chunk still ends loud with no quiet after its last word (cut off)."""
    from align import speech_span
    levels = frame_levels(samples, rate)
    count = len(levels)
    duration = len(samples) / rate
    if words:
        # Not the last word's start: a syllable that opens on a stop (遍 b-, 他 t-, 个 g-) begins with a silent closure,
        # and that quiet read as the end of speech cut the syllable off (the v3 film lost 遍 of 多试几遍). The quiet that
        # ends speech comes after the word's first MIN_SYLLABLE_SECONDS, near where recognition says the word ends.
        last = words[-1]
        anchor = max(float(last['start']) + MIN_SYLLABLE_SECONDS, float(last.get('end', last['start'])) - .1)
    else:
        anchor = max(0., speech_span(samples, rate)[1] - .4)
    quiet = max(1, round(TAIL_QUIET_SECONDS / .01))
    first = min(count - 1, max(0, int(anchor / .01)))
    end_frame = None
    for frame in range(first, count - quiet + 1):
        if (levels[frame:frame + quiet] < TAIL_QUIET_DB).all():
            end_frame = frame
            break
    if end_frame is None:
        if levels[-3:].max() > TAIL_LOUD_DB:
            return None, {'status': 'cut_off', 'final_level_db': round(float(levels[-3:].max()), 1)}
        end_frame = first + int(np.argmin(levels[first:]))
    speech_end = end_frame * .01 + .01
    burst = next((frame for frame in range(end_frame + quiet, count) if levels[frame] > TAIL_BURST_DB), None)
    cut = speech_end + TAIL_KEEP_SECONDS
    if burst is not None:
        cut = min(cut, burst * .01 - .01)
    cut = min(max(cut, speech_end + .02), duration)
    report = {'status': 'artifact_dropped' if burst is not None and burst * .01 < duration else 'clean', 'speech_end': round(speech_end, 3),
              'cut': round(cut, 3), 'dropped_seconds': round(duration - cut, 3)}
    return cut, report


def finish_edges(samples, cut, speech_end, rate=SAMPLE_RATE):
    """Cut at `cut` with a raised-cosine fade from the end of speech (at least FADE_OUT_SECONDS) and a short fade-in."""
    samples = np.array(samples[:max(1, round(cut * rate))], float)
    fade = max(round(FADE_OUT_SECONDS * rate), len(samples) - round(speech_end * rate))
    fade = min(fade, len(samples))
    if fade > 1:
        samples[-fade:] *= .5 + .5 * np.cos(np.linspace(0, np.pi, fade))
    lead = min(round(FADE_IN_SECONDS * rate), len(samples))
    if lead > 1:
        samples[:lead] *= .5 - .5 * np.cos(np.linspace(0, np.pi, lead))
    return samples


def clips_last_word(tail, words):
    """Why a tail cut cuts into the last recognized word, or None. Recognition of the kept audio cannot catch this: it
    completes a familiar phrase from context (the v3 film's 多试几遍 cut 0.14 s into 遍 was still transcribed as 遍)."""
    if not words or not tail or tail.get('cut') is None:
        return None
    last = words[-1]
    if float(tail['cut']) < float(last.get('end', last['start'])) - INSIDE_WORD_SECONDS:
        return f"the tail cut at {tail['cut']:.2f}s falls inside the last word {last.get('text', '')!r} ({last['start']:.2f}-{last['end']:.2f}s)"
    return None


def clean_edges(samples, words=None, rate=SAMPLE_RATE):
    """(samples ending cleanly after the last syllable, report); samples None when the chunk was cut off mid-sound."""
    cut, report = speech_tail(samples, words, rate)
    if cut is None:
        return None, report
    return finish_edges(samples, cut, report['speech_end'], rate), report


def check_chunk(samples, text, path, asr, english=False):
    """Noise gate, speech rate and ASR coverage (asr: True = local ASR when installed, or a callable). The last
    characters must be heard too (end_slack 0): a lost final syllable is a CER of one in thirty and passed before.
    english: also check that an English ear hears the line's English words (True = local recognizer, or a callable)."""
    from audio_gate import measure
    from align import speech_span, spoken_syllables, covers, transcribe, english_heard
    start, end = speech_span(samples)
    rate = spoken_syllables(text) / max(.01, end - start)
    result = {'gate': measure(samples, path.name, short_clip=True), 'chars_per_second': round(rate, 2), 'rate_unit': 'syllables'}
    failures = list(result['gate']['failures'])
    if not SPEECH_RATE_RANGE[0] <= rate <= SPEECH_RATE_RANGE[1]:
        failures.append(f'speech rate {rate:.2f} syllables/s outside {SPEECH_RATE_RANGE}')
    words = (asr if callable(asr) else transcribe)(path) if asr else None
    if words:
        result['asr'] = covers(text, words['text'], end_slack=0)
        result['words'] = words['words']
        if result['asr']['status'] != 'pass':
            failures.append(f"ASR does not cover the text, ending included (CER {result['asr']['cer']}): {words['text'][-12:]!r}")
    else:
        result['asr'] = {'status': 'not_available' if asr else 'disabled'}
    if english and asr:
        heard = (english if callable(english) else english_heard)(path, text)
        result['english'] = heard or {'status': 'not_available'}
        if heard and heard['status'] != 'pass':
            failures.append('an English ear does not hear: ' + ', '.join(item['word'] for item in heard['words'] if not item['heard']))
    return {**result, 'failures': failures, 'status': 'FAIL' if failures else 'PASS'}


def verify_kept(path, text, asr, english=True):
    """Check the audio that is actually kept (after the tail cut), not the take before it: ending and English."""
    check = check_chunk(read_pcm(path), text, path, asr, english=english)
    return {'checks_version': CHECKS_VERSION, 'status': check['status'], 'failures': check['failures'],
            'asr': check.get('asr'), **({'english': check['english']} if 'english' in check else {})}


def risky_terms(manifest):
    """{term: the clause it is read in} for what the voice reads after the say map: English words (a Chinese voice
    often says them so an English ear does not hear them) and Arabic numerals (the local VoxCPM2 does not normalize
    digits; 70万 must be read from 七十万)."""
    from align import english_words
    terms = {}
    for page in manifest['pages']:
        for beat in page['beats']:
            said = spoken_text(manifest, beat)
            # The probe reads the whole sentence: a two-word clause is read slowly and trips the speech-rate gate.
            for sentence in re.split(r'(?<=[。！？!?])', said):
                for term in english_words(sentence) + re.findall(r'\d+(?:\.\d+)?%?', sentence):
                    terms.setdefault(term, sentence.strip())
    return terms


def probe_terms(root, manifest, synthesize, asr=True, seeds=2, seed=2026):
    """Voice every risky term in its own clause before the master: a failing chunk costs up to six seeds of several
    minutes each (x-start talk), a clause costs seconds. Writes work/term-probe/report.json with advice."""
    root = Path(root)
    folder = root / 'work' / 'term-probe'
    folder.mkdir(parents=True, exist_ok=True)
    report = []
    for number, (term, clause) in enumerate(risky_terms(manifest).items()):
        if re.fullmatch(r'\d+(?:\.\d+)?%?', term):
            report.append({'term': term, 'line': clause, 'status': 'needs_say',
                           'advice': f'digits are read unreliably: narration.say {{"{term}": "<汉字读法>"}} (the subtitle keeps {term})'})
            continue
        takes = []
        for attempt in range(seeds):
            audio, rate, *_ = synthesize(clause, seed + attempt)
            path = folder / f'{number:02d}-{attempt}.wav'
            write_pcm(path, to_rate(audio, rate))
            check = check_chunk(read_pcm(path), clause, path, asr, english=True)
            # Only what the term decides counts here: recognition and the English ear (rate and noise belong to the master).
            failures = [item for item in check['failures'] if item.startswith(('ASR does not cover', 'an English ear'))]
            takes.append({'seed': seed + attempt, 'status': 'FAIL' if failures else 'PASS', 'failures': failures,
                          'other_checks': [item for item in check['failures'] if item not in failures], 'file': path.name})
        passed = sum(take['status'] == 'PASS' for take in takes)
        entry_ = {'term': term, 'line': clause, 'passed': passed, 'of': seeds, 'takes': takes,
                  'status': 'pass' if passed == seeds else 'risky' if passed else 'fail'}
        if passed < seeds:
            entry_['advice'] = f'narration.say {{"{term}": "<中文读法或中文词>"}}: the subtitle keeps {term}, the voice reads the Chinese'
        report.append(entry_)
    (folder / 'report.json').write_text(json.dumps({'terms': report}, ensure_ascii=False, indent=1))
    return report


def generate_chunk(synthesize, chunk, path, seed, asr, english=True):
    """synthesize(text, seed) -> (samples, rate) or (samples, rate, words) when the engine returns timestamps.
    A line with English words gets more seeds: whether the voice says "git" or "Amp" so an English ear hears it varies
    from seed to seed (the v3 probe: the same text was heard in one seed and not in the next)."""
    from align import speech_span, english_words
    attempts = []
    tries = ENGLISH_ATTEMPTS if english and asr and english_words(chunk['text']) else GENERATION_ATTEMPTS
    for attempt in range(tries):
        audio, rate, *timestamps = synthesize(chunk['text'], seed + attempt)
        audio = to_rate(audio, rate)
        if not len(audio) or not np.isfinite(audio).all():
            attempts.append({'seed': seed + attempt, 'status': 'FAIL', 'failures': ['empty or non-finite audio']})
            continue
        start, end = speech_span(audio)
        first = max(0, round((start - EDGE_SECONDS[0]) * SAMPLE_RATE))
        # Keep a generous tail for now: where speech really ends is decided with the recognized words below.
        audio = audio[first:round((end + .6) * SAMPLE_RATE)]
        write_pcm(path, audio)
        check = check_chunk(read_pcm(path), chunk['text'], path, asr)
        if timestamps and timestamps[0]:
            shift = first / SAMPLE_RATE
            check['words'] = [{**word, 'start': word['start'] - shift, 'end': word['end'] - shift} for word in timestamps[0]]
            check['timestamps'] = 'engine'
        cleaned, tail = clean_edges(read_pcm(path), check.get('words'))
        check['tail'] = tail
        if cleaned is None:
            check['failures'] = check['failures'] + [f"the chunk ends mid-sound ({tail['final_level_db']} dB): the last word is cut off"]
            check['status'] = 'FAIL'
        elif clips_last_word(tail, check.get('words')):
            check['failures'] = check['failures'] + [clips_last_word(tail, check.get('words'))]
            check['status'] = 'FAIL'
        else:
            write_pcm(path, cleaned)
            if check['status'] == 'PASS':
                check['kept'] = verify_kept(path, chunk['text'], asr, english)
                if check['kept']['status'] != 'PASS':
                    check['failures'] = check['failures'] + ['after the tail cut: ' + failure for failure in check['kept']['failures']]
                    check['status'] = 'FAIL'
        attempts.append({'seed': seed + attempt, **check})
        if check['status'] == 'PASS':
            return attempts
    path.unlink(missing_ok=True)
    unheard = sorted({item['word'] for attempt in attempts for item in (attempt.get('kept') or {}).get('english', {}).get('words', []) if not item['heard']})
    advice = (f" English never heard in {tries} seeds: {', '.join(unheard)}. This voice cannot say them: write a Chinese term (bug -> 问题)"
              " or a transliteration (Thorsten Ball -> 托尔斯滕·鲍尔) in the script, or list the word with a reason in narration.english_exceptions.") if unheard else ''
    raise ValueError(f"Chunk failed {tries} attempts: {chunk['text'][:24]}... " + json.dumps([item['failures'] for item in attempts], ensure_ascii=False) + advice)


def chunk_alignment(layout, limit=None, speak=None):
    """Beat ranges from chunk placement; caption cues inside each chunk from timestamps or pauses.
    limit (CJK characters per cue, caption_assets.cue_char_limit) packs clauses for the canvas; a clause longer than
    the limit splits at word-aware positions only when the chunk has character timestamps to time the pieces."""
    from align import caption_texts, piece_ranges
    captions = {}
    for chunk, offset, audio, words in layout:
        def cut(text):
            return caption_texts(text, limit, split_long=bool(words)) if limit else caption_texts(text)
        texts = []
        for unit in chunk['units']:
            shown = unit.get('show', unit['text'])
            pieces_ = cut(shown)
            said = [speak(unit['beat'], piece) for piece in pieces_] if speak and 'show' in unit else pieces_
            if ''.join(said) != unit['text']:
                pieces_, said = [shown], [unit['text']]  # a say term straddles a cut: one cue for the unit, still timed by the voice
            texts += [(unit['beat'], piece, spoken_piece) for piece, spoken_piece in zip(pieces_, said)]
        # Pieces are timed on what the voice says, and shown as written.
        ranges, method = piece_ranges(audio, [spoken_piece for _, _, spoken_piece in texts], words)
        for (beat, text, _), (left, right) in zip(texts, ranges):
            captions.setdefault(beat, []).append((text, (offset + left, offset + right), method))
    return captions


def build_master(root, manifest, synthesize, identity, options=None):
    """Generate every chunk with gates and retries, join with measured pauses, write master + alignment."""
    from audio_gate import measure
    options = options or {}
    root = Path(root).resolve()
    target, text = generation_target(root, manifest)
    record_path = target.with_suffix('.voice.json')
    if target.exists() and not record_path.exists():
        raise ValueError('Existing untracked narration will not be overwritten: ' + str(target))
    seed, asr = int(options.get('seed', 2026)), options.get('asr', True)
    joins = {**JOIN_PAUSE_SECONDS, **options.get('join_pause_seconds', {})}
    chunks = narration_chunks(manifest, int(options.get('max_chunk_chars', MAX_CHUNK_CHARS)))
    folder = target.with_name(target.stem + '-chunks')
    layout, records, master, cursor = [], [], [], 0
    # Words the director accepts as the voice says them (narration.english_exceptions: {word: reason}).
    exceptions = {word.lower() for word in (manifest.get('narration', {}).get('english_exceptions') or {})}

    def english(path, line):
        from align import english_heard
        heard = english_heard(path, line)
        if heard and exceptions:
            for item in heard['words']:
                if item['word'] in exceptions and not item['heard']:
                    item.update(heard=True, exception=True)
            heard['status'] = 'pass' if all(item['heard'] for item in heard['words']) else 'fail'
        return heard

    for index, chunk in enumerate(chunks):
        path = folder / f'{index:03d}.wav'
        signature = hashlib.sha256(json.dumps({'identity': identity, 'text': chunk['text'], 'seed': seed}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        chunk_record_path = path.with_suffix('.json')
        record = json.loads(chunk_record_path.read_text()) if chunk_record_path.is_file() else {}
        if not (path.is_file() and record.get('signature') == signature):
            if path.exists() and not record:
                raise ValueError('Existing untracked chunk will not be overwritten: ' + str(path))
            attempts = generate_chunk(synthesize, chunk, path, seed, asr, english)
            record = {'signature': signature, 'index': index, 'text': chunk['text'], 'units': chunk['units'], 'join': chunk['join'],
                      'attempts': attempts, 'sha256': sha256(path), 'checks_version': CHECKS_VERSION}
            chunk_record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        elif asr and record.get('checks_version', 1) < CHECKS_VERSION:
            # Made before the current checks: check it now, and make it again if its ending or English fails, or if the
            # record shows its tail was cut inside the last word (the audio is already cut: no check of it can bring it back).
            kept = verify_kept(path, chunk['text'], asr, english)
            used = record['attempts'][-1]
            clipped = clips_last_word(used.get('tail'), used.get('words'))
            if kept['status'] != 'PASS' or clipped:
                path.unlink()
                attempts = generate_chunk(synthesize, chunk, path, seed, asr, english)
                record = {'signature': signature, 'index': index, 'text': chunk['text'], 'units': chunk['units'], 'join': chunk['join'],
                          'attempts': attempts, 'sha256': sha256(path), 'replaced': {'reason': kept['failures'] + ([clipped] if clipped else [])}}
            else:
                record['attempts'][-1]['kept'] = kept
            record['checks_version'] = CHECKS_VERSION
            chunk_record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        audio = read_pcm(path)
        # Chunks made before the tail rule are cleaned here too: the master never inherits a hard cut or an after-sound.
        audio, tail = clean_edges(audio, record['attempts'][-1].get('words'))
        if audio is None:
            raise ValueError(f"Chunk {index} ends mid-sound ({tail['final_level_db']} dB): regenerate it (delete {path.name} and its .json)")
        record['tail'] = tail
        if chunk['join']:
            silence = max(0., joins[chunk['join']] - EDGE_SECONDS[0] - EDGE_SECONDS[1])
            master.append(np.zeros(round(silence * SAMPLE_RATE)))
            cursor += len(master[-1])
        layout.append((chunk, cursor / SAMPLE_RATE, audio, record['attempts'][-1].get('words')))
        master.append(audio)
        cursor += len(audio)
        records.append({**{key: record[key] for key in ('index', 'text', 'join', 'sha256')}, 'attempts': len(record['attempts']), 'tail': record['tail']})
    master.append(np.zeros(round(END_TAIL_SECONDS * SAMPLE_RATE)))
    samples = np.concatenate(master)
    gate = measure(samples, str(target))
    if gate['status'] != 'PASS':
        raise ValueError('Joined narration master failed the audio gate: ' + '; '.join(gate['failures']))
    write_pcm(target, samples)
    from align import close_fluent_gaps, entry
    from caption_assets import cue_char_limit
    by_id = {beat['id']: beat for page in manifest['pages'] for beat in page['beats']}
    captions = chunk_alignment(layout, cue_char_limit(manifest), lambda beat_id, piece: spoken_text(manifest, by_id[beat_id], piece))
    beats = []
    # Each chunk found its pauses against its own loudness; on the joined master (the clock the caption sync gate
    # measures) a gap the voice speaks straight through is closed, keeping every start where the words are.
    order = [beat['id'] for page in manifest['pages'] for beat in page['beats']]
    flat = close_fluent_gaps(samples, [item[1] for beat_id in order for item in captions[beat_id]])
    cursor = 0
    for beat_id in order:
        captions[beat_id] = [(text, flat[cursor + index], method) for index, (text, _, method) in enumerate(captions[beat_id])]
        cursor += len(captions[beat_id])
    for page in manifest['pages']:
        for beat in page['beats']:
            items = captions[beat['id']]
            beats.append(entry(beat['id'], beat['text'], [item[0] for item in items], [item[1] for item in items], items[0][2]))
    alignment_path = (root / manifest.get('narration', {}).get('alignment', 'inputs/narration/alignment.json')).resolve()
    if alignment_path.exists() and json.loads(alignment_path.read_text()).get('method') != 'semantic_chunks':
        raise ValueError('Existing hand-made alignment will not be overwritten: ' + str(alignment_path))
    alignment_path.parent.mkdir(parents=True, exist_ok=True)
    alignment_path.write_text(json.dumps({'version': 1, 'source_sha256': sha256(target), 'script_sha256': hashlib.sha256(text.encode()).hexdigest(),
                                          'method': 'semantic_chunks', 'beats': beats}, ensure_ascii=False, indent=2))
    record_path.write_text(json.dumps({'method': 'semantic_chunks', 'identity': identity, 'seed': seed, 'text': text,
        'duration': len(samples) / SAMPLE_RATE, 'chunks': records, 'join_pause_seconds': joins, 'gate': gate,
        'alignment': str(alignment_path), 'subjective_listening': 'not_performed'}, ensure_ascii=False, indent=2))
    return target, alignment_path


def load_master(root, manifest, destination):
    settings = manifest.get('narration')
    if not settings or not settings.get('source') or not settings.get('alignment'):
        raise ValueError('Production requires narration.source and narration.alignment: finish the complete voice before visuals')
    options = manifest.get('audio', {})
    if float(options.get('pace', 1)) != 1 or options.get('trim_silence', False):
        raise ValueError('A locked narration master cannot be stretched or silence-trimmed')
    if any(float(options.get(key, 0)) != 0 for key in ('sentence_pause_seconds', 'page_pause_seconds')):
        raise ValueError('Page and sentence pauses cannot insert silence into a narration master')
    source = (Path(root) / settings['source']).resolve()
    alignment_path = (Path(root) / settings['alignment']).resolve()
    alignment = json.loads(alignment_path.read_text())
    if alignment.get('version', 1) != 1:
        raise ValueError('Unsupported narration alignment version')
    text = script_text(manifest)
    if alignment.get('source_sha256') != sha256(source):
        raise ValueError('Narration alignment belongs to a different audio master')
    if alignment.get('script_sha256') != hashlib.sha256(text.encode()).hexdigest():
        raise ValueError('Narration alignment belongs to a different script')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(source), '-ar', str(SAMPLE_RATE),
                    '-ac', '1', '-c:a', 'pcm_s16le', str(destination)], check=True)
    with wave.open(str(destination)) as stream:
        samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype='<i2').copy()
    if not len(samples) or not np.any(samples):
        raise ValueError('Narration master is empty or silent')
    duration = len(samples) / SAMPLE_RATE
    expected = [beat for page in manifest['pages'] for beat in page['beats']]
    entries = alignment.get('beats', [])
    if len(entries) != len(expected):
        raise ValueError('Narration alignment must cover every beat exactly once')
    previous_end = 0.
    beats = {}
    for expected_beat, entry in zip(expected, entries):
        identifier = expected_beat.get('id')
        if not identifier or entry.get('id') != identifier or entry.get('text') != expected_beat['text'] or identifier in beats:
            raise ValueError('Narration alignment IDs, order and text must match the script')
        start, end = float(entry['start']), float(entry['end'])
        if not math.isfinite(start + end) or not 0 <= previous_end <= start < end <= duration + 1 / SAMPLE_RATE:
            raise ValueError('Narration ranges must be finite, ordered, nonoverlapping and inside the master')
        beats[identifier] = {**entry, 'start': start, 'end': end}
        previous_end = end
    report = {'mode': 'continuous_master', 'source': str(source), 'source_sha256': sha256(source),
              'alignment': str(alignment_path), 'alignment_sha256': sha256(alignment_path),
              'duration': duration, 'sample_rate': SAMPLE_RATE, 'samples': len(samples),
              'pcm_sha256': hashlib.sha256(samples.astype('<i2').tobytes()).hexdigest(),
              'inserted_page_silence_seconds': 0, 'internal_samples_preserved': True}
    return samples, beats, report


def verify_compiled_master(root, timeline):
    """Detect lost breaths, inserted gaps, stale inputs or stretched master samples."""
    report = timeline.get('narration_master')
    if not report:
        return None
    if sha256(report['source']) != report['source_sha256'] or sha256(report['alignment']) != report['alignment_sha256']:
        raise ValueError('Narration master or alignment changed after build')
    with wave.open(str(Path(root) / 'work/narration.wav')) as stream:
        if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            raise ValueError('Compiled narration format changed')
        samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype='<i2')
    start = round(float(timeline.get('voice_start_seconds', timeline['cover_seconds'])) * SAMPLE_RATE)
    master = samples[start:start + report['samples']]
    if hashlib.sha256(master.tobytes()).hexdigest() != report['pcm_sha256']:
        raise ValueError('Compiled narration changed the locked master waveform')
    if np.any(samples[:start]) or np.any(samples[start + report['samples']:]):
        raise ValueError('Cover and ending padding must not contain extra narration')
    return {'continuous_master': True, 'pcm_preserved': True, 'inserted_page_silence_seconds': 0}


def rebuild_master(root):
    """Re-join an existing master from its generated chunks under the current edge rules (no voice model needed).
    Every chunk must still match its text, voice and seed; a chunk that would need generating stops the rebuild."""
    root = Path(root).resolve()
    manifest = json.loads((root / 'job.json').read_text())
    target, _ = generation_target(root, manifest)
    record = json.loads(target.with_suffix('.voice.json').read_text())

    def synthesize(text, seed):
        raise ValueError('A chunk no longer matches its text or voice; regenerate the narration with the voice tool: ' + text[:24])
    options = {'seed': record['seed'], 'join_pause_seconds': record.get('join_pause_seconds', {}), 'asr': True}
    alignment = (root / manifest.get('narration', {}).get('alignment', 'inputs/narration/alignment.json')).resolve()
    if alignment.is_file() and json.loads(alignment.read_text()).get('method') != 'semantic_chunks':
        alignment.replace(alignment.with_name(alignment.stem + '.before-rebuild.json'))
    return build_master(root, manifest, synthesize, record['identity'], options)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['rebuild'])
    parser.add_argument('job', type=Path)
    arguments = parser.parse_args()
    master, alignment = rebuild_master(arguments.job)
    print(json.dumps({'master': str(master), 'alignment': str(alignment)}))

