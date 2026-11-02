"""Narration checks added after the v3 film: a final syllable that opens on a stop keeps its vowel, the kept audio must
end with the line's last characters, and English words must be heard by an English ear (no model needed)."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import align
import narration_master
from narration_master import SAMPLE_RATE, speech_tail
from test_audio_gate import speech_like
from test_semantic_narration import fake_tts, manifest


def stop_ending():
    """Speech, then a last syllable that opens on a stop: 60 ms of closure silence, then its vowel (the 遍 of 多试几遍)."""
    body = speech_like([(1.2, .0)], seed=3)
    vowel = speech_like([(.22, .3)], seed=4)
    return np.concatenate([body, np.zeros(round(.06 * SAMPLE_RATE)), vowel])


class TailTests(unittest.TestCase):
    def test_a_stop_closure_in_the_last_syllable_is_not_the_end_of_speech(self):
        samples = stop_ending()
        # recognition places the last word from the closure (1.20 s) to the end of its vowel (1.48 s)
        words = [{'text': '几', 'start': .9, 'end': 1.2}, {'text': '遍', 'start': 1.2, 'end': 1.48}]
        cut, report = speech_tail(samples, words)
        self.assertGreaterEqual(cut, 1.48)                 # the vowel after the closure is kept
        self.assertNotEqual(report['status'], 'artifact_dropped')


class EndingTests(unittest.TestCase):
    def test_a_missing_last_character_fails_the_kept_audio_check(self):
        text = '一个模型十次错一次，你不知道错的是哪一次，结果每件事都得多试几遍。'
        self.assertEqual(align.covers(text, text[:-2])['status'], 'pass')                 # the lenient check let it through
        self.assertEqual(align.covers(text, text[:-2], end_slack=0)['status'], 'fail')    # the kept-audio check does not
        self.assertEqual(align.covers(text, text[:-1], end_slack=0)['status'], 'pass')    # only the full stop is missing

    def test_an_accent_pair_is_the_same_syllable_but_a_misread_number_is_not(self):
        try:
            import pypinyin  # noqa: F401
        except ImportError:
            self.skipTest('pypinyin not installed')
        line = '核心文件不到五个，这些他审；数据库迁移，也审。'
        self.assertEqual(align.covers(line, '核心文件不到5个,这些他省,数据库迁移也省', end_slack=0)['status'], 'pass')   # shen/sheng
        self.assertEqual(align.covers('有的四十六条消息。', '有的六十四条消息', end_slack=0)['status'], 'fail')


class EnglishTests(unittest.TestCase):
    def heard(self, text, transcript):
        with mock.patch('voice_backends.tools', return_value={'mlx_python': 'python', 'asr_model': 'model'}), \
             mock.patch('voice_backends.run_python', return_value={'spans': [{'text': transcript}]}):
            return align.english_heard('line.wav', text)

    def test_english_words_must_be_heard_as_english(self):
        self.assertEqual(self.heard('处理 bug 报告', 'the')['status'], 'fail')                 # the film's "bug"
        self.assertEqual(self.heard('在 Amp 里，线程和 git 提交', 'Ampli. and Git.')['status'], 'pass')
        self.assertEqual(self.heard('git 提交', 'get')['status'], 'pass')                      # native "git" is written "get"
        self.assertEqual(self.heard('在 Amp 里', 'I am pleased.')['status'], 'fail')           # "am" is not "amp"
        self.assertEqual(self.heard('几个 Agent 同时', 'the agents.')['status'], 'pass')        # longer words stay fuzzy (plural)

        found = self.heard('AI 编程工具 Amp 的工程师，Thorsten Ball', 'is AI. amp the Sostenbo.')
        self.assertEqual([item['word'] for item in found['words'] if not item['heard']], ['thorsten', 'ball'])
        self.assertEqual(self.heard('只有中文的一句', '')['words'], [])

    def test_a_line_with_english_tries_more_seeds_and_names_words_no_seed_says(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / '000.wav'
            chunk = {'text': '处理 bug 报告，给用户发邮件。', 'units': [], 'join': None}
            asr = lambda wav: {'text': chunk['text'], 'words': [{'text': chunk['text'], 'start': .1, 'end': 1.8}]}
            seen = []
            def english(wav, line):
                seen.append(1)
                return {'status': 'pass' if len(seen) >= 4 else 'fail', 'words': [{'word': 'bug', 'heard': len(seen) >= 4}], 'heard': ''}
            attempts = narration_master.generate_chunk(fake_tts, chunk, path, 7, asr, english)
            self.assertEqual(len(attempts), 4)                 # seeds 1-3 not heard, the 4th is
            never = lambda wav, line: {'status': 'fail', 'words': [{'word': 'bug', 'heard': False}], 'heard': 'the'}
            with self.assertRaisesRegex(ValueError, r'never heard in 6 seeds: bug.*Chinese term'):
                narration_master.generate_chunk(fake_tts, chunk, path, 7, asr, never)

    def test_chunks_made_before_the_kept_audio_check_are_checked_and_remade(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = manifest([['一个模型十次错一次，结果每件事都得多试几遍。']])
            calls = []
            synthesize = lambda text, seed: calls.append(seed) or fake_tts(text, seed)
            narration_master.build_master(root, job, synthesize, {'engine': 'fixture'}, {'asr': False})
            record_path = root / 'inputs/narration/master-chunks/000.json'
            record = json.loads(record_path.read_text())
            record.pop('checks_version')                     # as a chunk made before this check existed
            record_path.write_text(json.dumps(record))
            (root / 'inputs/narration/master.voice.json').unlink()
            (root / 'inputs/narration/master.wav').unlink()
            clipped = lambda wav: {'text': '一个模型十次错一次结果每件事都得多试几', 'words': [{'text': '一个', 'start': .1, 'end': 3.}]}
            good = lambda wav: {'text': '一个模型十次错一次结果每件事都得多试几遍', 'words': [{'text': '一个', 'start': .1, 'end': 3.}]}
            with mock.patch('align.transcribe', side_effect=[clipped(None)] + [good(None)] * 4):
                narration_master.build_master(root, job, synthesize, {'engine': 'fixture'}, {'asr': True})
            record = json.loads(record_path.read_text())
            self.assertEqual(record['checks_version'], narration_master.CHECKS_VERSION)
            self.assertIn('replaced', record)                # the clipped chunk was made again
            self.assertEqual(len(calls), 2)

    def test_a_cached_chunk_cut_inside_its_last_word_is_remade_even_when_recognition_hears_the_word(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = manifest([['一个模型十次错一次，结果每件事都得多试几遍。']])
            calls = []
            synthesize = lambda text, seed: calls.append(seed) or fake_tts(text, seed)
            narration_master.build_master(root, job, synthesize, {'engine': 'fixture'}, {'asr': False})
            record_path = root / 'inputs/narration/master-chunks/000.json'
            record = json.loads(record_path.read_text())
            # As the v3 film's chunk 005: checked at version 3, but its tail was cut by the old rule 0.14 s into 遍.
            record['checks_version'] = 3
            record['attempts'][-1].update(words=[{'text': '几', 'start': 2.62, 'end': 2.76}, {'text': '遍', 'start': 2.76, 'end': 2.94}],
                                          tail={'status': 'artifact_dropped', 'speech_end': 2.78, 'cut': 2.8, 'dropped_seconds': .48})
            record_path.write_text(json.dumps(record))
            (root / 'inputs/narration/master.voice.json').unlink()
            (root / 'inputs/narration/master.wav').unlink()
            heard = lambda wav: {'text': '一个模型十次错一次结果每件事都得多试几遍', 'words': [{'text': '一个', 'start': .1, 'end': 3.}]}
            with mock.patch('align.transcribe', side_effect=[heard(None)] * 5):
                narration_master.build_master(root, job, synthesize, {'engine': 'fixture'}, {'asr': True})
            record = json.loads(record_path.read_text())
            self.assertIn('inside the last word', ' '.join(record['replaced']['reason']))
            self.assertEqual(len(calls), 2)

    def test_a_cut_inside_the_last_word_is_named(self):
        words = [{'text': '遍', 'start': 4.76, 'end': 4.94}]
        self.assertIn('inside the last word', narration_master.clips_last_word({'cut': 4.8}, words))
        self.assertIsNone(narration_master.clips_last_word({'cut': 4.96}, words))
        self.assertIsNone(narration_master.clips_last_word({'status': 'cut_off'}, words))


if __name__ == '__main__':
    unittest.main()
