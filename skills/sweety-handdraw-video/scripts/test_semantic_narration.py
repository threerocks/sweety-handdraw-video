"""Semantic chunking, gated per-chunk generation, measured alignment and caption sync (no model needed)."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from align import caption_texts, caption_sync, char_times, covers, pause_split, align_master, spoken
from narration_master import narration_chunks, build_master, split_long, MAX_CHUNK_CHARS, SAMPLE_RATE
from test_audio_gate import speech_like


def manifest(beats_per_page):
    return {'pages': [{'beats': [{'id': f'p{page}-{index}', 'text': text} for index, text in enumerate(beats)]}
                      for page, beats in enumerate(beats_per_page)],
            'narration': {'source': 'inputs/narration/master.wav', 'alignment': 'inputs/narration/alignment.json'}}


def fake_tts(text, seed, broken=()):
    """About 5 spoken characters per second, 0.25 s pause after every clause; seeds in `broken` hum."""
    clauses = [piece for piece in caption_texts(text, 1) if spoken(piece)]
    samples = speech_like([(len(spoken(piece)) / 5, .25) for piece in clauses], seed=seed)
    if seed in broken:
        samples += .02 * np.sin(2 * np.pi * 60 * np.arange(len(samples)) / SAMPLE_RATE)
    return samples, SAMPLE_RATE


class SemanticNarrationTests(unittest.TestCase):
    def test_chunks_keep_sentences_whole_and_respect_the_limit(self):
        job = manifest([['一条蛇在爬行时，不小心被锯子划伤，', '结果满嘴鲜血。'], ['剧痛之下，它彻底爆发。'], ['好。', '这就是全部内容。']])
        chunks = narration_chunks(job)
        self.assertEqual([chunk['text'] for chunk in chunks], ['一条蛇在爬行时，不小心被锯子划伤，结果满嘴鲜血。', '剧痛之下，它彻底爆发。', '好。这就是全部内容。'])
        self.assertEqual([chunk['join'] for chunk in chunks], [None, 'page', 'page'])
        self.assertEqual([unit['beat'] for unit in chunks[0]['units']], ['p0-0', 'p0-1'])
        long_beat = '这是一句完整的话。' * 30
        parts = split_long(long_beat)
        self.assertTrue(all(len(part) <= MAX_CHUNK_CHARS and part.endswith('。') for part in parts))
        self.assertEqual(''.join(parts), long_beat)

    def test_build_master_gates_retries_and_aligns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = manifest([['第一页先讲一个故事，', '故事里有一条蛇。'], ['它被锯子划伤以后，非常生气。', '于是咬向锯齿。']])
            calls = []
            def synthesize(text, seed):
                calls.append(seed)
                return fake_tts(text, seed, broken={2026})
            target, alignment_path = build_master(root, job, synthesize, {'engine': 'fixture'}, {'asr': False})
            record = json.loads(target.with_suffix('.voice.json').read_text())
            self.assertEqual(record['gate']['status'], 'PASS')
            first = json.loads((target.parent / 'master-chunks/000.json').read_text())
            self.assertEqual([attempt['status'] for attempt in first['attempts']], ['FAIL', 'PASS'])
            alignment = json.loads(alignment_path.read_text())
            self.assertEqual([beat['id'] for beat in alignment['beats']], ['p0-0', 'p0-1', 'p1-0', 'p1-1'])
            ends = [(beat['start'], beat['end']) for beat in alignment['beats']]
            self.assertTrue(all(left[1] <= right[0] for left, right in zip(ends, ends[1:])))
            from audio_gate import decode
            cues = [{'text': cue['text'], 'start': beat['start'] + cue['start'], 'end': beat['start'] + cue['end']}
                    for beat in alignment['beats'] for cue in beat['caption_cues']]
            self.assertEqual(caption_sync(decode(target), cues)['status'], 'PASS')
            self.assertEqual(caption_sync(decode(target), [{**cue, 'start': cue['start'] + .6, 'end': cue['end'] + .6} for cue in cues])['status'], 'FAIL')
            count = len(calls)
            build_master(root, job, synthesize, {'engine': 'fixture'}, {'asr': False})
            self.assertEqual(len(calls), count)
            (target.parent / 'master-chunks/000.json').unlink()
            with self.assertRaisesRegex(ValueError, 'untracked chunk'):
                build_master(root, job, synthesize, {'engine': 'changed'}, {'asr': False})

    def test_chunk_failing_every_attempt_stops_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'failed 3 attempts'):
                build_master(directory, manifest([['一句完整的旁白。']]), lambda text, seed: fake_tts(text, seed, broken={2026, 2027, 2028}),
                             {'engine': 'fixture'}, {'asr': False})

    def test_caption_sync_finds_a_short_clause_break_in_fluent_speech(self):
        from scipy import signal
        from test_audio_gate import BAND
        rng = np.random.default_rng(0)
        parts = []
        for speech, pause in [(.3, .4), (1.5, .04), (1.5, .5)]:
            time = np.arange(round(speech * SAMPLE_RATE)) / SAMPLE_RATE
            parts += [signal.sosfilt(BAND, rng.standard_normal(len(time))) * (.6 + .4 * np.sin(np.pi * 4 * time) ** 2), np.zeros(round(pause * SAMPLE_RATE))]
        samples = np.concatenate(parts)
        samples = .5 * samples / np.abs(samples).max() + 10 ** (-70 / 20) * rng.standard_normal(len(samples))
        cues = [{'text': '前半句，', 'start': .7, 'end': 2.}, {'text': '后半句。', 'start': 2.24, 'end': 3.74}]
        self.assertEqual(caption_sync(samples, cues)['status'], 'PASS')
        for shift in (.5, -.5):
            self.assertEqual(caption_sync(samples, [{**cue, 'start': cue['start'] + shift, 'end': cue['end'] + shift} for cue in cues])['status'], 'FAIL')

    def test_a_caption_gap_the_voice_speaks_through_is_closed_without_moving_starts(self):
        from align import close_fluent_gaps
        from scipy import signal
        from test_audio_gate import BAND
        rng = np.random.default_rng(1)
        parts = []
        for speech, pause in [(.3, .4), (1.5, .04), (1.5, .4), (1., .3)]:   # a 40 ms dip inside speech, then a real pause
            time = np.arange(round(speech * SAMPLE_RATE)) / SAMPLE_RATE
            parts += [signal.sosfilt(BAND, rng.standard_normal(len(time))) * (.6 + .4 * np.sin(np.pi * 4 * time) ** 2), np.zeros(round(pause * SAMPLE_RATE))]
        samples = np.concatenate(parts)
        samples = .5 * samples / np.abs(samples).max() + 10 ** (-70 / 20) * rng.standard_normal(len(samples))
        # ASR gaps of 0.2 s over the dip (2.2-2.24 s) and over the pause (3.74-4.14 s)
        ranges = [(.7, 2.04), (2.24, 3.94), (4.14, 5.14)]
        closed = close_fluent_gaps(samples, ranges)
        self.assertEqual([start for start, _ in closed], [start for start, _ in ranges])
        self.assertEqual(closed[0][1], 2.24)   # the dip is not a pause: the first caption runs on
        self.assertEqual(closed[1][1], 3.94)   # the real pause keeps its gap

    def test_engine_timestamps_and_pause_split_place_cues(self):
        samples = speech_like([(1.2, .4), (2., .4), (1., .3)])
        ranges, method = pause_split(samples, [6, 10, 5])
        self.assertEqual(method, 'longest_pauses')
        self.assertAlmostEqual(ranges[1][0], 1.6, delta=.05)
        self.assertAlmostEqual(ranges[2][0], 4., delta=.05)
        words = [{'text': '你好，', 'start': .1, 'end': .5}, {'text': '世界', 'start': .7, 'end': 1.1}]
        times, matched = char_times('你好，世界。', words)
        self.assertEqual(matched, 1.)
        self.assertEqual(times[2], (.7, .9))
        self.assertEqual(covers('你好世界。', '你好，世界')['status'], 'pass')
        self.assertEqual(covers('你好世界，今天天气很好。', '你好世界')['status'], 'fail')

    def test_asr_compares_numbers_by_value_not_spelling(self):
        # The script spells numbers out; ASR writes digits. Same value passes, a misread value still fails.
        script = '这类工具占了百分之四十五，速度快了三点五倍，二零二六年发布。'
        self.assertEqual(covers(script, '这类工具占了45%,速度快了3.5倍,2026年发布')['status'], 'pass')
        self.assertEqual(covers(script, '这类工具占了百分之45，速度快了３．５倍，二〇二六年发布')['cer'], 0.)
        self.assertEqual(covers(script, '这类工具占了54%,速度快了3.5倍,2026年发布')['cer'] > 0, True)
        self.assertEqual(covers('增长了百分之四十五。', '增长了54%')['status'], 'fail')

    def test_asr_homophones_of_rare_characters_pass_on_pinyin(self):
        # Whisper transcript of a correctly read line (Jiang Xue): every error is a homophone.
        line = '孤舟蓑笠翁，独钓寒江雪。只有一叶小船，一位披着蓑衣的老人，在大雪里独自垂钓。'
        heard = '孤州梭利翁独吊寒江雪只有一夜小船一位披着梭伊的老人在大雪里独自垂吊'
        try:
            import pypinyin  # noqa: F401
        except ImportError:
            result = covers(line, heard)
            self.assertEqual((result['status'], result['pinyin']), ('fail', 'unavailable'))
            return
        result = covers(line, heard)
        self.assertEqual((result['status'], result['matched_by'], result['pinyin_cer']), ('pass', 'pinyin', 0.))
        self.assertGreater(result['cer'], .15)
        self.assertEqual(covers(line, heard, homophones=False)['status'], 'fail')
        # Homophones do not excuse skipped words, babble or a different number.
        self.assertEqual(covers(line, '孤州梭利翁独吊寒江雪只有一夜小船')['status'], 'fail')
        self.assertEqual(covers(line, '孤州梭利翁独吊寒江雪只有一个大爷坐在船上睡觉打呼噜')['status'], 'fail')
        self.assertEqual(covers('增长了百分之四十五。', '增长了54%')['status'], 'fail')
        # A homophone the number normalizer reads as a numeral (牵牛 heard as 千牛) still passes.
        moon = '月亮从东边的山后慢慢升起来，停在南斗和牵牛两组星宿之间。'
        self.assertEqual(covers(moon, '月亮从东边的山后慢慢升起来停在南斗和千牛两组星宿之间')['status'], 'pass')
        self.assertEqual(covers('他带了四十五个人。', '他带了54个人')['status'], 'fail')

    def test_whole_master_alignment_from_character_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'master.wav'
            source.write_bytes(b'audio')
            beats = [{'id': 'a', 'text': '前一句。'}, {'id': 'b', 'text': '后一句。'}]
            words = [{'text': character, 'start': index * .25 + (.5 if index > 2 else 0), 'end': index * .25 + .2 + (.5 if index > 2 else 0)}
                     for index, character in enumerate('前一句后一句')]
            result = align_master(source, beats, words, Path(directory) / 'alignment.json', 'hash')
            self.assertEqual([(beat['start'], beat['end']) for beat in result['beats']], [(0., .7), (1.25, 1.95)])


if __name__ == '__main__':
    unittest.main()
