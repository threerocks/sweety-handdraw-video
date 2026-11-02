"""Rule gaps the aunt-doctor film found (2026-11-02): drawing budgets that follow the real process, web motion that
takes no pen time, a pen limit a strict mode can actually meet, and reading time on pages without a drawn picture."""
import unittest

from storyboard_lint import (FIGURE_STROKES, PROCESS_RATE, STROKES_PER_SECOND, finished_picture_hold_share, motion_only_visuals,
                             pen_remedies, plan, stroke_rate)


def beats(*seconds):
    cursor, result = 1., []
    for index, length in enumerate(seconds):
        result.append({'id': f'b{index}', 'start': cursor, 'end': cursor + length})
        cursor += length + .2
    return result


def board(shots, **extra):
    return {'version': 1, 'shots': [{'id': f's{index}', 'purpose': 'shows the story', **shot} for index, shot in enumerate(shots)], **extra}


def person(identity, **extra):
    return {'id': identity, 'kind': 'illustration', 'subject': 'the doctor at work', 'characters': ['doctor'], **extra}


def job(drawing=None, overlays=(), pictures=()):
    """A job.json with one page: the pictures as pen actions, the overlays as web motion."""
    return {'drawing': drawing or {}, 'characters': [{'id': 'doctor'}],
            'pages': [{'beats': [{'id': 'b0', 'actions': [{'type': 'illustration', 'visual': name} for name in pictures]}],
                       'overlays': [{'recipe': 'flow', 'visual': name} for name in overlays]}]}


class StrokeRateTests(unittest.TestCase):
    def test_rate_follows_colouring_construction_and_speed(self):
        self.assertEqual(stroke_rate({}), STROKES_PER_SECOND)                       # markers with construction: the default
        self.assertEqual(stroke_rate({'sketch': False}), PROCESS_RATE[('markers', False)])
        self.assertAlmostEqual(stroke_rate({'colouring': 'parts', 'sketch': False, 'speed': 1.4}), 6.1 * 1.4)
        # The aunt-doctor film's parts pictures drew 8.5 strokes/s on average at speed 1.4.
        self.assertAlmostEqual(stroke_rate({'colouring': 'parts', 'sketch': False, 'speed': 1.4}), 8.5, delta=.2)


class PenWindowTests(unittest.TestCase):
    def setUp(self):
        self.shots = board([{'beats': ['b0'], 'type': 'character_action', 'visuals': [
            person('v-doctor'), {'id': 'v-title', 'kind': 'text_group', 'subject': 'title'},
            {'id': 'v-flow', 'kind': 'diagram', 'subject': 'flow'}, {'id': 'v-age', 'kind': 'text_group', 'subject': 'ticker'}]}],
            characters=[{'id': 'doctor'}])
        self.timed = beats(10)

    def test_web_motion_visuals_take_no_pen_time(self):
        alone = {row['visual']: row for row in plan(self.shots, self.timed)['visuals']}
        manifest = job(overlays=['v-title', 'v-flow', 'v-age'], pictures=['v-doctor'])
        self.assertEqual(motion_only_visuals(self.shots, manifest), {'v-title', 'v-flow', 'v-age'})
        shared = {row['visual']: row for row in plan(self.shots, self.timed, manifest)['visuals']}
        # Without job.json the picture shares the window with three visuals (weights 3:1:1:1); the overlays play beside it.
        self.assertAlmostEqual(alone['v-doctor']['seconds'], (10 - .54 - .6 - 3 * .08) / 2)
        self.assertAlmostEqual(shared['v-doctor']['seconds'], 10 - .54 - .6)
        self.assertEqual(shared['v-flow']['seconds'], 0.)
        self.assertEqual(shared['v-flow']['complexity'], 'web motion beside the pen')

    def test_storyboard_recipe_marks_web_motion_before_job_json_has_overlays(self):
        for visual in self.shots['shots'][0]['visuals'][1:]:
            visual['recipe'] = visual['subject']
        self.assertEqual(motion_only_visuals(self.shots, None), {'v-title', 'v-flow', 'v-age'})

    def test_a_person_in_too_short_a_window_is_flagged_at_planning(self):
        warnings = '\n'.join(plan(self.shots, self.timed)['warnings'])
        self.assertIn(f'fewer than the ~{FIGURE_STROKES} a recognizable person needs', warnings)
        self.assertIn('"recipe"', warnings)
        # The same window at the aunt-doctor film's settings (parts colouring, speed 1.4, overlays beside the pen) fits.
        manifest = job({'colouring': 'parts', 'sketch': False, 'speed': 1.4}, ['v-title', 'v-flow', 'v-age'], ['v-doctor'])
        report = plan(self.shots, self.timed, manifest)
        self.assertNotIn('recognizable person', '\n'.join(report['warnings']))
        self.assertAlmostEqual(report['visuals'][0]['strokes_per_second'], 8.54)


class StrictModeTests(unittest.TestCase):
    def comic(self, mode):
        # Four pictures with a person, each across two 9 s sentences (73 s of narration).
        shots = [{'beats': [f'b{2 * index}', f'b{2 * index + 1}'], 'type': 'character_action' if index % 2 else 'closeup',
                  'visuals': [person(f'v{index}')]} for index in range(4)]
        return board(shots, characters=[{'id': 'doctor'}], presentation={'mode': mode, 'reason': 'story told in panels'})

    def test_forecast_warns_a_strict_mode_before_any_image_and_names_the_way_out(self):
        timed = beats(*[9] * 8)
        manifest = job({'speed': 1.4})
        comic = plan(self.comic('narrative_comic'), timed, manifest)
        warning = next(item for item in comic['warnings'] if 'usual size' in item)
        self.assertIn('mode narrative_comic allows 60%', warning)
        self.assertIn('"colouring": "parts"', warning)
        self.assertIn('"sketch": false', warning)
        self.assertIn('that is focus_diagram', warning)
        self.assertGreater(comic['stats']['pen_forecast_share'], .7)
        # Parts colouring without construction draws the same pictures in ~8.8 s each: 48% (+10%) fits the comic's 60%.
        cheap = plan(self.comic('narrative_comic'), timed, job({'colouring': 'parts', 'sketch': False, 'speed': 1.4}))
        self.assertAlmostEqual(cheap['stats']['pen_forecast_share'], .48, delta=.01)
        self.assertFalse([item for item in cheap['warnings'] if 'usual size' in item])

    def test_remedies_offer_only_what_the_job_does_not_do_yet(self):
        rules = {'drawing_share': .7}
        cheap = pen_remedies('focus_diagram', rules, [{'colouring': 'parts', 'sketch': False}])
        self.assertNotIn('colour by parts', cheap)
        self.assertNotIn('construction', cheap)
        self.assertNotIn('focus_diagram (pen limit', cheap)
        costly = pen_remedies('narrative_comic', {'drawing_share': .6}, [{}])
        for advice in ('colour by parts', 'construction sketch', 'that is focus_diagram'):
            self.assertIn(advice, costly)


class ReadingTimeTests(unittest.TestCase):
    def test_a_chapter_card_is_read_once_its_words_are_written(self):
        # Page: 1-11 s; the pen writes the chapter words 1.5-3 s; nothing else is on the page.
        timeline = {'pages': [{'beats': [{'start': 1}], 'speech_end': 11, 'end': 11,
                               'events': [{'asset': 'words', 'visual': 'v-chapter', 'caption': False, 'start': 1.5, 'end': 3}]}]}
        self.assertAlmostEqual(finished_picture_hold_share(timeline), .8)

    def test_an_explanation_form_is_read_once_its_last_item_arrives(self):
        page = {'beats': [{'start': 1}], 'speech_end': 11, 'end': 11, 'events': [],
                'overlays': [{'recipe': 'flow', 'visual': 'v-flow', 'start': 2, 'end': 4, 'moments': [2, 3]}]}
        # Last item at 3 s, arrived at 3.6 s; the form holds its last frame to the page end.
        self.assertAlmostEqual(finished_picture_hold_share({'pages': [page]}), (11 - 3.6) / 10)
        page['overlays'][0]['exit'] = {'start': 9, 'end': 9.5, 'style': 'fade'}
        self.assertAlmostEqual(finished_picture_hold_share({'pages': [page]}), (9 - 3.6) / 10)

    def test_the_pen_drawing_still_is_not_reading_time(self):
        page = {'beats': [{'start': 1}], 'speech_end': 11, 'end': 11,
                'events': [{'asset': 'figure', 'caption': False, 'image': ['figure.png', None], 'start': 4, 'end': 8, 'hide': 11}],
                'overlays': [{'recipe': 'checklist', 'visual': 'v-list', 'start': 1, 'end': 2, 'moments': [1.2]},
                             {'recipe': 'kinetic_title', 'visual': 'v-title', 'start': 1, 'end': 2, 'moments': [1]}]}
        # The list is readable 1.8-4 s and 8-11 s; the title is not a picture on a page that has one.
        self.assertAlmostEqual(finished_picture_hold_share({'pages': [page]}), (4 - 1.8 + 11 - 8) / 10)


if __name__ == '__main__':
    unittest.main()
