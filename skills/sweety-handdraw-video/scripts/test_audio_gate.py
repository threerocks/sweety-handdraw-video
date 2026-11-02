"""Objective noise gate, reference cleanup trimming and the film audio requirement."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy import signal
from audio_gate import measure, require_clean, SAMPLE_RATE

BAND = signal.butter(4, (150, 3500), 'bandpass', fs=SAMPLE_RATE, output='sos')


def speech_like(segments, seed=0, floor_db=-70.):
    """(speech_seconds, pause_seconds) pairs: band-limited syllables (4 per second) over a quiet floor."""
    rng = np.random.default_rng(seed)
    parts = []
    for speech, pause in segments:
        time = np.arange(round(speech * SAMPLE_RATE)) / SAMPLE_RATE
        parts += [signal.sosfilt(BAND, rng.standard_normal(len(time))) * np.sin(np.pi * 4 * time) ** 2, np.zeros(round(pause * SAMPLE_RATE))]
    samples = np.concatenate(parts)
    samples = .5 * samples / np.abs(samples).max()
    return samples + 10 ** (floor_db / 20) * rng.standard_normal(len(samples))


def save(path, samples):
    from narration_master import write_pcm
    write_pcm(Path(path), samples)
    return Path(path)


class AudioGateTests(unittest.TestCase):
    def test_tied_persistence_reports_the_tone_peak_instead_of_an_adjacent_bin(self):
        clean = speech_like([(2.5, .5)] * 4)
        time = np.arange(len(clean)) / SAMPLE_RATE
        for frequency in (50, 235):
            report = measure(clean + .01 * np.sin(2 * np.pi * frequency * time))
            self.assertEqual(report['tonal_lines_hz_persistence'][0][0], frequency)

    def test_clean_speech_passes_and_each_defect_fails(self):
        clean = speech_like([(2.5, .5)] * 4)
        self.assertEqual(measure(clean)['status'], 'PASS')
        time = np.arange(len(clean)) / SAMPLE_RATE
        hum = measure(clean + .01 * np.sin(2 * np.pi * 50 * time))
        self.assertTrue(any('50 Hz' in failure for failure in hum['failures']))
        bed = measure(clean + .01 * np.random.default_rng(1).standard_normal(len(clean)))
        self.assertTrue(any('pauses not quiet' in failure for failure in bed['failures']))
        self.assertTrue(any('DC offset' in failure for failure in measure(clean + .02)['failures']))
        self.assertTrue(any('clipping' in failure for failure in measure(np.clip(clean * 4, -1, 1))['failures']))
        self.assertEqual(measure(np.zeros(SAMPLE_RATE))['status'], 'FAIL')

    def test_one_window_of_harmonics_is_not_hum(self):
        time = np.arange(round(1.8 * SAMPLE_RATE)) / SAMPLE_RATE
        vowel = speech_like([(1.8, .3)])
        vowel[:len(time)] += .2 * np.sin(2 * np.pi * 247 * time)
        self.assertEqual(measure(vowel, short_clip=True)['tonal_lines_hz_persistence'], [])

    def test_the_voices_own_pitch_is_not_hum_but_a_tone_through_the_pauses_is(self):
        # The v3 film: a long list read evenly holds a male pitch near 120 Hz for seconds; seven takes failed as "hum".
        speech = speech_like([(2.5, .4)] * 3)
        time = np.arange(len(speech)) / SAMPLE_RATE
        voiced = np.abs(speech) > 1e-3
        envelope = np.convolve(voiced.astype(float), np.ones(480) / 480, mode='same')
        pitch = speech + .15 * envelope * np.sin(2 * np.pi * 120 * time)          # only while speaking
        report = measure(pitch, short_clip=True)
        self.assertEqual(report['status'], 'PASS', report['failures'])
        self.assertTrue(report.get('voice_pitch_lines'))
        hum = measure(speech + .01 * np.sin(2 * np.pi * 120 * time), short_clip=True)   # the same frequency, through the pauses
        self.assertTrue(any('120 Hz' in failure for failure in hum['failures']))

    def test_dirty_reference_is_rejected_with_cleanup_hint(self):
        with tempfile.TemporaryDirectory() as directory:
            clean = speech_like([(2.5, .5)] * 4)
            path = save(Path(directory) / 'dirty.wav', clean + .01 * np.random.default_rng(2).standard_normal(len(clean)))
            with self.assertRaisesRegex(ValueError, 'clean_reference.py'):
                require_clean(path, 'reference', short_clip=True)

    def test_reference_cleanup_trims_to_sentence_end_with_tail_and_record(self):
        from clean_reference import clean
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = save(root / 'source.wav', speech_like([(3., .7)] * 6, seed=3))
            with self.assertRaisesRegex(ValueError, 'sentence boundaries'):
                clean(source, 1., 11.5, '文本。', root / 'mid-sentence', separate=False, denoise=False, asr=False)
            record = clean(source, 3.8, 14.2, '一二三四五六七八九十。' * 5, root / 'reference', separate=False, denoise=False, asr=False)
            self.assertTrue(8 <= record['duration_seconds'] <= 15)
            self.assertAlmostEqual(record['used_seconds'][0], 3.55, delta=.05)
            self.assertAlmostEqual(record['used_seconds'][1], 14.12, delta=.05)
            self.assertGreaterEqual(record['tail']['natural_seconds'] + record['tail']['padded_seconds'], .5)
            self.assertEqual(record['gate_after']['status'], 'PASS')
            self.assertEqual(json.loads((root / 'reference/clean-record.json').read_text())['reference_sha256'], record['reference_sha256'])
            with self.assertRaisesRegex(ValueError, 'earlier files stay intact'):
                clean(source, 3.8, 14.2, '文本。', root / 'reference', separate=False, denoise=False, asr=False)

    def test_finalize_requires_passing_film_audio(self):
        from video_verifier import finalize_job
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in ('work', 'outputs'):
                (root / folder).mkdir()
            (root / 'job.json').write_text('{}')
            (root / 'outputs/film.mp4').write_bytes(b'film')
            (root / 'review.json').write_text('{}')
            (root / 'work/verification.json').write_text(json.dumps({'status': 'engineering_passed_review_required',
                'audio_gate': {'status': 'FAIL'}, 'caption_sync': {'status': 'PASS'}}))
            with self.assertRaisesRegex(ValueError, 'audio gate'):
                finalize_job(root)


if __name__ == '__main__':
    unittest.main()
