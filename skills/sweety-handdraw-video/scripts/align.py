#!/usr/bin/env python3
"""Measure beat ranges and caption cues from narration audio; timestamps are never hand-written.

Caption cues come from character timestamps when available (forced aligner, ASR, or cloud TTS
subtitles), otherwise from a deterministic pause split: N pieces use the N-1 most prominent pauses.
"""
import argparse
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import subprocess
import unicodedata
import numpy as np

SAMPLE_RATE = 48000
PUNCTUATION = set('，。！？；：、“”‘’（）《》〈〉【】—…·,.!?;:"\'()[]<>-~ \n\t')
SENTENCE_END = '。！？!?…'
CAPTION_BREAKS = '。！？!?…；;，,、：:'
MAX_CAPTION_CHARS = 24
MIN_PAUSE_SECONDS = .08
SYNC_TOLERANCE_SECONDS = .3
SYNC_GAP_SECONDS = .2
# Clause breaks in fluent narration can be 30-50 ms; shifting cues by 0.4 s still fails nearly every cue.
SYNC_PAUSE_SECONDS = .03
FLUENT_PAUSE_SECONDS = .08   # shorter dips inside speech are not heard as pauses (close_fluent_gaps)
TRANSCRIBE = r'''
import json, sys, mlx_whisper
language = sys.argv[3] if len(sys.argv) > 3 else 'zh'
result = mlx_whisper.transcribe(sys.argv[1], path_or_hf_repo=sys.argv[2], language=language, word_timestamps=True)
try:
    from opencc import OpenCC
    convert = OpenCC('t2s').convert if language == 'zh' else (lambda text: text)
except ImportError:
    convert = lambda text: text
words = [{'text': convert(word['word']).strip(), 'start': word['start'], 'end': word['end']}
         for segment in result['segments'] for word in segment.get('words', [])]
print(json.dumps({'text': convert(result['text']), 'words': words}, ensure_ascii=False))
'''
# English words inside Chinese narration, heard by an English ear: Chinese recognition finds where they are (it is
# lenient: it writes "Agent" for a Chinese-accented guess), each span is cut out and transcribed as English.
ENGLISH_HEAR = r'''
import json, re, subprocess, sys, tempfile, mlx_whisper
path, model = sys.argv[1], sys.argv[2]
found = mlx_whisper.transcribe(path, path_or_hf_repo=model, language='zh', word_timestamps=True, initial_prompt='以下是普通话的句子，夹有英文词。')
spans = []
for word in (w for s in found['segments'] for w in s.get('words', [])):
    if re.search('[A-Za-z]', word['word']):
        if spans and word['start'] - spans[-1][1] < .12:
            spans[-1] = (spans[-1][0], word['end'])
        else:
            spans.append((word['start'], word['end']))
heard = []
with tempfile.TemporaryDirectory() as folder:
    for index, (start, end) in enumerate(spans):
        clip = f'{folder}/{index}.wav'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', str(max(0., start - .1)), '-t', str(end - start + .2), '-i', path, '-ac', '1', '-ar', '16000', clip], check=True)
        text = mlx_whisper.transcribe(clip, path_or_hf_repo=model, language='en', temperature=0.)['text']
        heard.append({'start': round(start, 3), 'end': round(end, 3), 'text': text.strip()})
print(json.dumps({'spans': heard}, ensure_ascii=False))
'''
FORCE_ALIGN = r'''
import json, sys
from mlx_audio.stt.utils import load_model
result = load_model(sys.argv[2]).generate(audio=sys.argv[1], text=sys.argv[3], language=sys.argv[4] if len(sys.argv) > 4 else 'Chinese')
print(json.dumps({'words': [{'text': item.text, 'start': item.start_time, 'end': item.end_time} for item in result.items]}, ensure_ascii=False))
'''


CHINESE_DIGITS = dict(zip('零〇一二两三四五六七八九', (0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9)))
CHINESE_UNITS = {'十': 10, '百': 100, '千': 1000, '万': 10 ** 4, '亿': 10 ** 8}
NUMBER_DIGITS = '0-9零〇一二两三四五六七八九'
NUMBER_PATTERN = '[' + NUMBER_DIGITS + '十百千万亿]+(?:[.点][' + NUMBER_DIGITS + '十百千万亿]+)*'


def spoken(text):
    # A decimal point carries a value; sentence punctuation does not.
    return ''.join(character for index, character in enumerate(text)
                   if character not in PUNCTUATION or (character == '.' and 0 < index < len(text) - 1
                       and text[index - 1].isdigit() and text[index + 1].isdigit()))


def chinese_number(text):
    """Parse exact Chinese/Arabic magnitude notation without float rounding or range guessing."""
    normalized = ''.join(str(CHINESE_DIGITS.get(character, character)) for character in text).replace('点', '.')
    # Adjacent spoken nonzero digits can be an approximation (两三万, 四五十), not 23/45.
    if re.search('[一二两三四五六七八九]{2}', text) and len(text) < 4:
        return text

    def value(token):
        for unit in ('亿', '万'):
            if unit in token:
                if token.count(unit) != 1:
                    raise ValueError('Repeated magnitude')
                left, right = token.split(unit)
                return value(left or '1') * CHINESE_UNITS[unit] + (value(right) if right else 0)
        total, previous, cursor = Decimal(0), 10000, 0
        for match in re.finditer(r'([0-9]+(?:\.[0-9]+)?)?([十百千])', token):
            if match.start() != cursor or CHINESE_UNITS[match[2]] >= previous:
                raise ValueError('Invalid small-unit order')
            previous = CHINESE_UNITS[match[2]]
            total += Decimal(match[1] or '1') * previous
            cursor = match.end()
        return total + (Decimal(token[cursor:]) if token[cursor:] else 0)

    try:
        result = format(value(normalized), 'f')
        return result.rstrip('0').rstrip('.') if '.' in result else result
    except (InvalidOperation, ValueError):
        return text


def numbers(text):
    """Normalize comparison text only; retain different magnitudes, fractions and signs."""
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))', '', text)
    text = re.sub('百分之(' + NUMBER_PATTERN + ')', lambda match: chinese_number(match[1]) + '%', text)
    text = re.sub(NUMBER_PATTERN, lambda match: chinese_number(match[0]), text)
    return re.sub(r'-(?=\d)', '负', text)


def edit_rate(expected, actual):
    """Levenshtein distance over any two sequences, divided by the expected length."""
    row = np.arange(len(actual) + 1)
    for index, item in enumerate(expected, 1):
        previous, row = row, np.empty_like(row)
        row[0] = index
        for column, other in enumerate(actual, 1):
            row[column] = min(previous[column] + 1, row[column - 1] + 1, previous[column - 1] + (item != other))
    return float(row[-1] / max(1, len(expected)))


def cer(expected, actual):
    return edit_rate(spoken(numbers(expected)).lower(), spoken(numbers(actual)).lower())


def syllables(text):
    """Toneless pinyin per Han character in context; digits, Latin letters and signs stay one token each. Common accent
    pairs count as one sound (zh/z, ch/c, sh/s, n/l, en/eng, in/ing, an/ang): a voice that says 审 like 省 still said the
    syllable, and the check is for a syllable missing or misread, not for accent (the v3 film: “这些他审” heard as
    “这些他省” in every take of a voice whose 审 passed everywhere else)."""
    from pypinyin import lazy_pinyin
    return [accent_merged(item) for item in lazy_pinyin(text, errors=lambda other: list(other))]


def accent_merged(syllable):
    for retroflex, flat in (('zh', 'z'), ('ch', 'c'), ('sh', 's')):
        if syllable.startswith(retroflex):
            syllable = flat + syllable[2:]
            break
    if syllable.startswith('n') and syllable not in ('n', 'ng'):
        syllable = 'l' + syllable[1:]
    for back, front in (('ang', 'an'), ('eng', 'en'), ('ing', 'in')):
        if syllable.endswith(back):
            return syllable[:-len(back)] + front
    return syllable


def anchored(expected, actual, error, max_cer, end_slack=2):
    blocks = [block for block in SequenceMatcher(None, expected, actual, autojunk=False).get_matching_blocks() if block.size]
    return bool(blocks) and error <= max_cer and blocks[0].a <= 2 and blocks[-1].a + blocks[-1].size >= len(expected) - end_slack


def covers(expected, actual, max_cer=.15, homophones=True, end_slack=2):
    """ASR must cover the whole text: low CER and matched characters near both ends (numbers compared by value).

    Whisper writes rare characters as common homophones (蓑笠 -> 梭利, 钓 -> 吊, 大雪 -> 大学), which fails short
    classical lines that were read correctly. With homophones=True a failing character comparison is checked again on
    toneless pinyin (when pypinyin is installed): homophones pass; skipped, babbled or misread numbers still fail.
    Verbatim transcripts (voice references) pass homophones=False."""
    # Letter case is the recognizer's spelling, not the voice's: "amend" read correctly comes back as "Amend".
    raw_expected, raw_actual = expected.lower(), actual.lower()
    expected, actual = spoken(numbers(raw_expected)), spoken(numbers(raw_actual))
    error = edit_rate(expected, actual)
    result = {'cer': round(error, 3), 'status': 'pass' if anchored(expected, actual, error, max_cer, end_slack) else 'fail',
              'asr_text': raw_actual, 'comparison_text': actual}
    if result['status'] == 'fail' and homophones:
        # Number normalization reads a homophone as a numeral (牵牛 heard as 千牛 -> 1000牛), so the pinyin check also
        # compares the unnormalized text; a misread number differs in both forms.
        try:
            pairs = [(syllables(e), syllables(a)) for e, a in ((expected, actual), (spoken(raw_expected), spoken(raw_actual)))]
        except ImportError:
            result['pinyin'] = 'unavailable'
        else:
            scored = [(edit_rate(e, a), e, a) for e, a in pairs]
            pinyin_error, expected_syllables, actual_syllables = min(scored, key=lambda item: item[0])
            result['pinyin_cer'] = round(pinyin_error, 3)
            if anchored(expected_syllables, actual_syllables, pinyin_error, max_cer, end_slack):
                result.update(status='pass', matched_by='pinyin')
    return result


LATIN_WORD = re.compile(r'[A-Za-z][A-Za-z\-]*')
# An English ear writes some correctly spoken words as their sound-alikes (native "git" is transcribed "get").
SOUND_ALIKE = {'git': {'get'}, 'ai': {'ay', 'aye', 'a i'}}
HEARD_RATIO = .75


def english_words(text):
    """The English words a Chinese narration line asks the voice to say (lowercase, in order)."""
    return [word.lower() for word in LATIN_WORD.findall(text)]


def english_heard(path, text, python=None, model=None):
    """Which English words of a Chinese narration line an English ear hears in the audio. A Chinese-accented guess
    that Chinese recognition still writes as "bug" is often heard as "the" in English: the film's "bug", "git", "Amp" and
    "Thorsten Ball" were all passed by the Chinese check. Returns {'status', 'words': [{word, heard}], 'heard'} or
    None when no local recognizer is installed."""
    expected = english_words(text)
    if not expected:
        return {'status': 'pass', 'words': [], 'heard': ''}
    from voice_backends import tools, run_python
    found = tools()
    python, model = python or found['mlx_python'], model or found['asr_model']
    if not python or not model:
        return None
    spans = run_python(python, ENGLISH_HEAR, path, model)['spans']
    heard = ' '.join(span['text'] for span in spans)
    tokens = re.findall(r'[a-z]+', heard.lower())
    candidates = set(tokens) | {a + b for a, b in zip(tokens, tokens[1:])}
    def hit(word):
        word = word.replace('-', '')
        alike = SOUND_ALIKE.get(word, set()) | {word, word.rstrip('s') if len(word) > 3 else word}
        for heard_token in candidates:
            if heard_token in alike:
                return True
            if len(word) <= 3:
                # a short word must be heard as itself: "am" scores 0.8 against "amp" and passed a voice that said "I am"
                # ("I am pleased" for "Amp"); only a little trailing bleed is allowed ("Ampli")
                if heard_token.startswith(word) and len(heard_token) <= len(word) + 2:
                    return True
                continue
            if SequenceMatcher(None, word, heard_token).ratio() >= HEARD_RATIO or word in heard_token:
                return True
        return any(' '.join(tokens[k:k + 2]) in alike for k in range(len(tokens)))
    words = [{'word': word, 'heard': hit(word)} for word in expected]
    return {'status': 'pass' if all(item['heard'] for item in words) else 'fail', 'words': words, 'heard': heard}


def char_times(text, words):
    """Map every spoken character of text to (start, end) seconds through ASR/aligner/TTS word times."""
    recognized = []
    for word in words:
        characters = spoken(word['text'])
        step = (word['end'] - word['start']) / max(1, len(characters))
        recognized += [(character, word['start'] + index * step, word['start'] + (index + 1) * step) for index, character in enumerate(characters)]
    expected = spoken(text)
    times = [None] * len(expected)
    matcher = SequenceMatcher(None, expected, ''.join(item[0] for item in recognized), autojunk=False)
    for tag, a0, a1, b0, b1 in matcher.get_opcodes():
        if tag == 'equal' or (tag == 'replace' and a1 - a0 == b1 - b0):
            for offset in range(a1 - a0):
                times[a0 + offset] = recognized[b0 + offset][1:]
    known = [index for index, value in enumerate(times) if value]
    if not known:
        raise ValueError('Timestamps do not match the narration text')
    for index, value in enumerate(times):
        if value is None:  # interpolate unmatched characters between their matched neighbours
            left = max((k for k in known if k < index), default=None)
            right = min((k for k in known if k > index), default=None)
            start = times[left][1] if left is not None else times[right][0]
            end = times[right][0] if right is not None else times[left][1]
            times[index] = (start, max(start, end))
    return times, round(len(known) / len(expected), 3)


def activity(samples, rate=SAMPLE_RATE):
    hop, width = round(.01 * rate), round(.02 * rate)
    samples = np.asarray(samples, dtype=np.float64)
    if len(samples) < width:
        return np.zeros(0, dtype=bool)
    count = 1 + (len(samples) - width) // hop
    # Frame energy from a running sum: the same values as a frame matrix without its memory (an 11-minute master is GBs).
    energy = np.concatenate([[0.], np.cumsum(samples * samples)])
    starts = hop * np.arange(count)
    levels = 20 * np.log10(np.sqrt(np.maximum(energy[starts + width] - energy[starts], 0.) / width) + 1e-9)
    threshold = max(np.percentile(levels, 95) - 35, np.percentile(levels, 10) + 6)
    return levels > threshold


def speech_span(samples, rate=SAMPLE_RATE):
    active = np.flatnonzero(activity(samples, rate))
    if not len(active):
        raise ValueError('No speech found')
    return active[0] * .01, active[-1] * .01 + .02


def pauses(samples, rate=SAMPLE_RATE, minimum=MIN_PAUSE_SECONDS):
    """Interior quiet runs as (start, end) seconds, between the first and last speech frames."""
    active = activity(samples, rate)
    found, start = [], None
    first, last = np.flatnonzero(active)[[0, -1]]
    for index in range(first, last + 1):
        if not active[index] and start is None:
            start = index
        elif active[index] and start is not None:
            if (index - start) * .01 >= minimum:
                found.append((start * .01 + .02, index * .01))
            start = None
    return found


def onsets(samples, rate=SAMPLE_RATE, minimum=.05):
    first, _ = speech_span(samples, rate)
    return [first] + [end for _, end in pauses(samples, rate, minimum)]


def pause_split(samples, weights, rate=SAMPLE_RATE):
    """Ranges for consecutive text pieces: the N-1 longest pauses, unless that gives implausible rates."""
    start, end = speech_span(samples, rate)
    if len(weights) == 1:
        return [(start, end)], 'single'
    found = pauses(samples, rate)
    def ranges(chosen):
        edges = [start] + [value for pause in chosen for value in pause] + [end]
        return [(edges[2 * index], edges[2 * index + 1]) for index in range(len(weights))]
    def plausible(result):
        return all(weight < 3 or 1.5 <= weight / max(.01, right - left) <= 10 for weight, (left, right) in zip(weights, result))
    longest = sorted(sorted(found, key=lambda pause: pause[1] - pause[0])[-(len(weights) - 1):]) if len(found) >= len(weights) - 1 else []
    if longest and plausible(ranges(longest)):
        return ranges(longest), 'longest_pauses'
    chosen, total, cursor = [], sum(weights), start
    for share in np.cumsum(weights)[:-1] / total:
        expected = start + share * (end - start)
        window = max(.8, .25 * (end - start))
        nearby = [pause for pause in found if pause[0] > cursor and abs((pause[0] + pause[1]) / 2 - expected) <= window]
        pick = max(nearby, key=lambda pause: pause[1] - pause[0]) if nearby else (expected, expected)
        chosen.append(pick)
        cursor = pick[1]
    return ranges(chosen), 'pauses_near_text_proportion'


def caption_texts(text, limit=MAX_CAPTION_CHARS, split_long=False):
    """Cue texts joining back to text. Default: clauses packed up to limit characters, a long clause stays whole.
    split_long=True (needs character timestamps to time the pieces): limit is CJK characters per cue (Latin letters
    count ~0.6, e.g. caption_assets.cue_char_limit), long clauses split at word-aware positions, no packing across
    a sentence end."""
    if split_long:
        from caption_assets import cue_pieces
        return cue_pieces(text, limit)
    clauses = re.findall('[^' + CAPTION_BREAKS + ']+[' + CAPTION_BREAKS + ']*|[' + CAPTION_BREAKS + ']+', text)
    pieces = []
    for clause in clauses:
        if pieces and (not spoken(clause) or len(pieces[-1]) + len(clause) <= limit):
            pieces[-1] += clause
        else:
            pieces.append(clause)
    return pieces


def ranges_from_times(texts, times):
    result, cursor = [], 0
    for text in texts:
        count = len(spoken(text))
        result.append((times[cursor][0], times[cursor + count - 1][1]) if count else (times[max(0, cursor - 1)][1],) * 2)
        cursor += count
    return result


def snap_to_onsets(found, ranges, tolerance=SYNC_TOLERANCE_SECONDS / 2):
    """ASR character times can sit a second away from the voice. A piece that starts away from every speech onset
    (by more than half the sync tolerance, leaving room for the encoded track) moves to the nearest onset inside it;
    with no pause there the speech is fluent, so the previous piece runs on to it without a gap.
    found: onset seconds on the same clock as ranges."""
    found = np.asarray(found, float)
    result = [list(span) for span in ranges]
    for index in range(1, len(result)):
        start, end = result[index]
        if len(found) and np.min(np.abs(found - start)) <= tolerance:
            continue
        inside = found[(found > result[index - 1][0] + .25) & (found < end - .25)] if len(found) else found
        if len(inside):
            start = float(inside[np.argmin(np.abs(inside - start))])
            result[index][0] = start
        result[index - 1][1] = start
    return [tuple(span) for span in result]


def close_fluent_gaps(samples, ranges, rate=SAMPLE_RATE, tolerance=SYNC_TOLERANCE_SECONDS):
    """A short gap between pieces (an ASR word boundary at a comma) where the voice never pauses is closed: the earlier
    piece runs on to the next start, so a caption never shows a break the listener does not hear. Starts never move:
    an independent ASR pass placed them within a few hundredths of a second, while nearby pauses were a second away."""
    # A dip shorter than FLUENT_PAUSE_SECONDS is not heard as a pause, and loudness normalization can shrink it below
    # the onset detector's floor: a caption gap over it would hinge on which stem is measured.
    quiet = [pause for pause in pauses(samples, rate, SYNC_PAUSE_SECONDS) if pause[1] - pause[0] >= FLUENT_PAUSE_SECONDS]
    result = [list(span) for span in ranges]
    for previous, following in zip(result, result[1:]):
        gap = following[0] - previous[1]
        if 0 < gap <= tolerance and not any(left < following[0] + tolerance and right > previous[1] - tolerance for left, right in quiet):
            previous[1] = following[0]
    return [tuple(span) for span in result]


def implausible_rates(texts, ranges, ratio=2.5, minimum=6):
    """True when one piece of a chunk is read more than ratio times as fast as another (pieces of >= minimum characters)."""
    rates = [len(spoken(text)) / max(.05, right - left) for text, (left, right) in zip(texts, ranges) if len(spoken(text)) >= minimum]
    return len(rates) >= 2 and max(rates) > ratio * min(rates)


def piece_ranges(samples, texts, words=None, rate=SAMPLE_RATE):
    """Ranges (seconds within samples) for consecutive text pieces, from timestamps or pauses."""
    if words:
        times, matched = char_times(''.join(texts), words)
        found = onsets(samples, rate, SYNC_PAUSE_SECONDS)
        ranges = ranges_from_times(texts, times)
        # Speech before the first recognized word (a lone syllable the recognizer dropped, such as an onomatopoeia
        # "叮") belongs to the first piece; otherwise that piece would get no time at all.
        if len(found) and found[0] < ranges[0][0] - SYNC_TOLERANCE_SECONDS / 2:
            ranges[0] = (float(found[0]), ranges[0][1])
        snapped = snap_to_onsets(found, ranges)
        if implausible_rates(texts, snapped):
            # Recognizer times can bunch a clause into a fraction of a second (13 characters in 0.83 s after 9 in 2.7 s):
            # a reader cannot read that cue. Fall back to pauses near the text proportion, as for audio without times.
            fallback, method = pause_split(samples, [max(1, len(spoken(text))) for text in texts], rate)
            return fallback, method + ' (timestamps bunched)'
        return snapped, f'timestamps (matched {matched:.0%})'
    return pause_split(samples, [max(1, len(spoken(text))) for text in texts], rate)


def entry(beat_id, text, texts, ranges, method):
    """One alignment beat: absolute range plus caption cues relative to the beat start."""
    # Round the beat first, so no cue ends past the beat its reader measures (end - start of the rounded values).
    start, end = round(ranges[0][0], 3), round(ranges[-1][1], 3)
    cues = [{'text': piece, 'start': round(max(0., left - start), 3), 'end': round(min(end, right) - start, 3)} for piece, (left, right) in zip(texts, ranges)]
    cues[-1]['end'] = min(cues[-1]['end'], round(end - start, 3))
    for previous, following in zip(cues, cues[1:]):
        previous['end'] = min(previous['end'], following['start'])
    return {'id': beat_id, 'text': text, 'start': round(start, 3), 'end': round(end, 3), 'caption_cues': cues, 'cue_method': method}


def beat_entry(samples, beat_id, text, offset, words=None, rate=SAMPLE_RATE, limit=None):
    texts = caption_texts(text, limit, split_long=bool(words)) if limit else caption_texts(text)
    ranges, method = piece_ranges(samples, texts, words, rate)
    return entry(beat_id, text, texts, [(offset + left, offset + right) for left, right in ranges], method)


def read_wave(path):
    """Mono float samples of a wav at SAMPLE_RATE (decoded by FFmpeg)."""
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-ac', '1', '-ar', str(SAMPLE_RATE), '-f', 's16le', '-'],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, '<i2').astype(np.float32) / 32768


def caption_sync(samples, cues, rate=SAMPLE_RATE, tolerance=SYNC_TOLERANCE_SECONDS):
    """A cue after a caption gap (>= SYNC_GAP_SECONDS) must start within tolerance of a speech onset (pause
    end); a cue continuing fluent speech, where energy shows no word boundary, must start where speech is."""
    found, active = np.array(onsets(samples, rate, SYNC_PAUSE_SECONDS)), activity(samples, rate)
    offsets, outside, previous_end = [], [], None
    for cue in cues:
        offset = float(np.min(np.abs(found - cue['start'])))
        if previous_end is None or cue['start'] - previous_end >= SYNC_GAP_SECONDS:
            offsets.append(offset)
            ok = offset <= tolerance
        else:
            index = round(cue['start'] * 100)
            ok = bool(active[max(0, index - round(tolerance * 100)):index + round(tolerance * 100)].any())
        if not ok:
            outside.append({'text': cue['text'], 'start': cue['start'], 'nearest_onset_offset': round(offset, 3)})
        previous_end = cue['end']
    return {'status': 'FAIL' if outside else 'PASS', 'tolerance_seconds': tolerance, 'cues': len(cues), 'cues_after_gap': len(offsets),
            'max_offset_seconds': round(max(offsets, default=0.), 3), 'outside_tolerance': outside}


LANGUAGE_NAMES = {'zh': 'Chinese', 'en': 'English', 'ja': 'Japanese', 'ko': 'Korean', 'fr': 'French', 'de': 'German', 'es': 'Spanish'}


def transcribe(path, python=None, model=None, text=None, language='zh'):
    """Character timestamps from an installed forced aligner (with text) or local Whisper; None if missing.
    language: ISO code of the narration (job.json "language", default zh)."""
    from voice_backends import tools, run_python
    found = tools()
    python = python or found['mlx_python']
    if text and python and found['forced_aligner']:
        return run_python(python, FORCE_ALIGN, path, found['forced_aligner'], text, LANGUAGE_NAMES.get(language, language))
    model = model or found['asr_model']
    if not python or not model:
        return None
    return run_python(python, TRANSCRIBE, path, model, language)


def spoken_syllables(text):
    """Spoken syllables, the unit of the speech-rate gates: one per CJK character, vowel groups per Latin word
    (an all-caps acronym is spelled letter by letter), one per digit. Counting "All In AI" as seven characters made
    a normal line read as too fast."""
    count = 0
    for token in re.findall(r'[\u3400-\u9fff\uf900-\ufaff]|[A-Za-z]+|\d', text):
        if len(token) == 1 and '\u3400' <= token <= '\ufaff' or token.isdigit():
            count += 1
        elif token.isupper() and len(token) <= 5:
            count += len(token) + sum(letter == 'W' for letter in token)
        else:
            groups = re.findall(r'[aeiouy]+', token.lower())
            count += max(1, len(groups) - (token.lower().endswith('e') and len(groups) > 1))
    return count


def align_master(source, beats, words, output, script_sha256, limit=None, speak=None):
    """Alignment for a whole user-provided master from ASR/aligner character times; limit: CJK characters per cue.
    speak(beat, text) gives what the voice says for displayed text (narration.say); times follow the spoken form."""
    speak = speak or (lambda beat, text: text)
    times, matched = char_times(''.join(speak(beat, beat['text']) for beat in beats), words)
    try:
        found = onsets(read_wave(source), SAMPLE_RATE, SYNC_PAUSE_SECONDS)
    except (OSError, ValueError, subprocess.CalledProcessError):
        found = None  # no readable audio (a placeholder in tests): keep the ASR times as they are
    pieces, cursor = [], 0
    for beat in beats:
        count = len(spoken(speak(beat, beat['text'])))
        texts = caption_texts(beat['text'], limit, split_long=True) if limit else caption_texts(beat['text'])
        said = [speak(beat, text) for text in texts]
        if ''.join(said) != speak(beat, beat['text']):
            texts, said = [beat['text']], [speak(beat, beat['text'])]
        pieces.append((beat, texts, ranges_from_times(said, times[cursor:cursor + count])))
        cursor += count
    if found is not None:
        # Snap across the whole master, so a sentence's first cue (and so the sentence itself) also starts on the voice.
        flat = snap_to_onsets(found, [span for _, _, ranges in pieces for span in ranges])
        cursor, snapped = 0, []
        for beat, texts, ranges in pieces:
            snapped.append((beat, texts, flat[cursor:cursor + len(ranges)]))
            cursor += len(ranges)
        pieces = snapped
    entries = [entry(beat['id'], beat['text'], texts, ranges, 'whole_master_timestamps') for beat, texts, ranges in pieces]
    for previous, following in zip(entries, entries[1:]):
        if previous['end'] > following['start']:
            raise ValueError('Character timestamps overlap between beats: ' + following['id'])
    alignment = {'version': 1, 'source_sha256': hashlib.sha256(Path(source).read_bytes()).hexdigest(), 'script_sha256': script_sha256,
                 'method': 'whole_master_character_timestamps', 'matched_characters': matched, 'beats': entries}
    Path(output).write_text(json.dumps(alignment, ensure_ascii=False, indent=2))
    return alignment


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('job', type=Path, help='Job whose narration.source is a finished user-provided master')
    parser.add_argument('--words', type=Path, help='JSON with {"words":[{"text","start","end"}]} from any aligner; default: local ASR')
    arguments = parser.parse_args()
    root = arguments.job.resolve()
    manifest = json.loads((root/'job.json').read_text())
    from narration_master import script_text, spoken_text
    settings = manifest['narration']
    source = root/settings['source']
    words = json.loads(arguments.words.read_text()) if arguments.words else transcribe(source, language=manifest.get('language', 'zh'))
    if not words:
        raise SystemExit('No local ASR/aligner installed; provide --words from an aligner or cloud TTS timestamps')
    beats = [beat for page in manifest['pages'] for beat in page['beats']]
    from caption_assets import cue_char_limit
    result = align_master(source, beats, words['words'], root/settings['alignment'], hashlib.sha256(script_text(manifest).encode()).hexdigest(),
                          cue_char_limit(manifest), lambda beat, text: spoken_text(manifest, beat, text))
    print(json.dumps({'beats': len(result['beats']), 'matched_characters': result['matched_characters']}, ensure_ascii=False))
