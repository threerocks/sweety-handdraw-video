"""A model-chosen style follows its content: adult, serious films get adult style families and proportions, prompts do
not ask for baby-schema faces, and a new film does not repeat the look of the recent ones. A user's own pick is kept."""
import json
import tempfile
import unittest
from pathlib import Path

import style_fit
from test_style_and_plan import CONTENT_FIT


def resolved(**changes):
    record = {'style_name': 'Engineer notebook', 'style_spec': {'line': '0.4 mm even pen line, ruled edges', 'palette': 'warm grey #EFEAE0, graphite #3A3F47, amber #E39B2D',
                                                                 'shading': 'one hard-edged shadow tone', 'texture': 'none', 'background': 'dot-grid page'},
              **json.loads(json.dumps(CONTENT_FIT))}
    record['content_analysis'].update(genre='tech_essay')
    record.update(style_family='engineer_notebook', directions=[{'family': 'engineer_notebook', 'idea': 'pen notebook', 'score': 9},
                                                                 {'family': 'editorial_concept', 'idea': 'one metaphor per point', 'score': 7},
                                                                 {'family': 'new_yorker_cartoon', 'idea': 'single-panel wit', 'score': 6}],
                  characters=[{'id': 'engineer', 'age': 36, 'head_to_body': 6.8, 'face': 'adult jaw, short beard, glasses'}])
    for key, value in changes.items():
        record[key] = value
    return record


INFERRED = {'mode': 'inferred', 'confirmation': 'user_delegated'}


class StyleFitTests(unittest.TestCase):
    def job(self, siblings=()):
        root = Path(tempfile.mkdtemp())
        for name, record in siblings:
            (root / name / 'style').mkdir(parents=True)
            (root / name / 'style/resolved.json').write_text(json.dumps(record))
        (root / 'new').mkdir()
        return root / 'new'

    def test_a_fitting_adult_style_passes(self):
        report = style_fit.check(self.job(), resolved(), INFERRED)
        self.assertTrue(report['checked'])
        self.assertEqual(report['style_family'], 'engineer_notebook')

    def test_chibi_for_an_adult_tech_essay_is_refused(self):
        record = resolved(style_family='chibi', directions=[{'family': 'chibi', 'idea': 'q', 'score': 9}] + resolved()['directions'][1:])
        with self.assertRaisesRegex(ValueError, 'reads as for children'):
            style_fit.check(self.job(), record, INFERRED)
        # A mostly comic script may say why.
        record['content_analysis']['tone']['humor_density'] = .7
        record['comedic_reason'] = 'a gag-a-line parody for a comedy account'
        record['characters'][0]['head_to_body'] = 6.8
        style_fit.check(self.job(), record, INFERRED)

    def test_adults_drawn_with_child_proportions_and_baby_faces_are_refused(self):
        with self.assertRaisesRegex(ValueError, 'adult .* heads tall'):
            style_fit.check(self.job(), resolved(characters=[{'id': 'lao-liu', 'age': 35, 'head_to_body': 3, 'face': 'adult'}]), INFERRED)
        with self.assertRaisesRegex(ValueError, 'baby-schema'):
            style_fit.check(self.job(), resolved(characters=[{'id': 'lao-liu', 'age': 35, 'head_to_body': 6.5, 'face': '圆脸、豆豆眼'}]), INFERRED)
        record = resolved()
        record['content_analysis']['genre'] = 'history_modern'
        with self.assertRaisesRegex(ValueError, 'at least 6.5 heads'):
            style_fit.check(self.job(), {**record, 'characters': [{'id': 'officer', 'age': 40, 'head_to_body': 6, 'face': 'lined, moustache'}]}, INFERRED)

    def test_three_directions_and_script_evidence_are_required(self):
        with self.assertRaisesRegex(ValueError, '3 structurally different directions'):
            style_fit.check(self.job(), resolved(directions=resolved()['directions'][:2]), INFERRED)
        record = resolved()
        record['content_analysis']['evidence'] = ['only one']
        with self.assertRaisesRegex(ValueError, 'evidence'):
            style_fit.check(self.job(), record, INFERRED)

    def test_repeating_a_recent_film_is_refused(self):
        previous = resolved()
        with self.assertRaisesRegex(ValueError, 'read 100% the same'):
            style_fit.check(self.job([('earlier', previous)]), resolved(style_family='editorial_concept'), INFERRED)
        other = resolved(style_spec={'line': 'brush line with pressure', 'palette': 'gold #C8A24A, cinnabar #B33A2B, indigo #2C3E6B',
                                     'shading': 'mineral flat colour', 'texture': 'silk', 'background': 'paper scroll'})
        with self.assertRaisesRegex(ValueError, 'already used engineer_notebook'):
            style_fit.check(self.job([('earlier', previous)]), other, INFERRED)
        other['series_reason'] = 'episode 2 of the same series keeps its look'
        style_fit.check(self.job([('earlier', previous)]), other, INFERRED)

    def test_the_engine_default_ink_needs_a_reason(self):
        record = resolved(style_spec={**resolved()['style_spec'], 'line': 'navy ink #1E2733'})
        with self.assertRaisesRegex(ValueError, 'engine default ink'):
            style_fit.check(self.job(), record, INFERRED)

    def test_a_style_the_user_picked_is_only_recorded(self):
        report = style_fit.check(self.job(), {'style_family': 'chibi'}, {'mode': 'custom', 'confirmation': 'explicit_user_request'})
        self.assertFalse(report['checked'])

    def test_prompts_of_an_adult_film_must_not_ask_for_baby_faces(self):
        found = style_fit.check_prompts(resolved(), [('i01.txt', 'A cute chibi programmer with big eyes'), ('i02.txt', 'An engineer, 36, glasses'),
                                                     ('i03.txt', 'Adult proportions, never cute (no chibi, no big round eyes, no blush)'),
                                                     ('i04.txt', '成人比例，不要大头和圆脸'), ('i05.txt', 'no glasses, cute face')])
        self.assertEqual(found, [('i01.txt', 'cute'), ('i05.txt', 'cute')])


if __name__ == '__main__':
    unittest.main()
