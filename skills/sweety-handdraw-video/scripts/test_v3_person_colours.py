"""A person comes out in the colours the plan names (aunt-doctor film, 2026-11-02: a mother painted with another
character's skin, a retired auntie painted back into the reference's white coat)."""
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from drawing_contract import colour_failures, colour_report, person_colours, validate_person_colours

SKIN, TEAL, BROWN, WHITE_COAT = '#EDC3A0', '#2F8F87', '#7A4B33', '#FBF7EF'


def plan(colours):
    return {'id': 'p1', 'outline_strategy': 'line_first',
            'subjects': [{'id': 'doctor', 'kind': 'human', **({'colours': colours} if colours is not None else {})}],
            'parts': [{'id': 'doctor_head', 'subject': 'doctor', 'role': 'head'}, {'id': 'doctor_torso', 'subject': 'doctor', 'role': 'torso'}]}


def figure(root, skin, cloth, name='figure.png'):
    """A head and a torso in flat colours with black outlines on white paper, and its outline mask."""
    image, mask = Image.new('RGB', (240, 360), 'white'), Image.new('L', (240, 360), 0)
    for draw, ink in ((ImageDraw.Draw(image), 'black'), (ImageDraw.Draw(mask), 255)):
        draw.ellipse((70, 20, 170, 120), fill=skin if ink == 'black' else None, outline=ink, width=4)
        draw.rectangle((50, 130, 190, 340), fill=cloth if ink == 'black' else None, outline=ink, width=4)
    image.save(root / name)
    mask.save(root / 'figure-outline.png')
    return name


ACTION = {'type': 'illustration', 'source': 'figure.png', 'outline_mask': 'figure-outline.png', 'drawing_plan_id': 'p1',
          'parts': [{'id': 'doctor_head', 'polygon': [[0, 0], [1, 0], [1, .35], [0, .35]]},
                    {'id': 'doctor_torso', 'polygon': [[0, .35], [1, .35], [1, 1], [0, 1]]}]}


class PersonColourTests(unittest.TestCase):
    def test_plan_must_name_skin_hair_and_clothing_for_each_person(self):
        with self.assertRaisesRegex(ValueError, 'skin, hair, clothing'):
            validate_person_colours(plan(None))
        with self.assertRaisesRegex(ValueError, 'clothing'):
            validate_person_colours(plan({'skin': SKIN, 'hair': '#2B2422', 'clothing': 'teal'}))
        # Hair may be null (a cap, a bald head); clothing may list the garments.
        self.assertTrue(validate_person_colours(plan({'skin': SKIN, 'hair': None, 'clothing': [TEAL, WHITE_COAT]})))

    def test_named_colours_pass_and_a_swapped_skin_or_outfit_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            figure(root, SKIN, TEAL)
            named = plan({'skin': SKIN, 'hair': None, 'clothing': TEAL})
            rows = {row['colour']: row for row in person_colours(root, ACTION, named)}
            self.assertGreater(rows['skin']['share'], .5)
            self.assertGreater(rows['clothing']['share'], .5)
            self.assertEqual(colour_failures(root, ACTION, named), [])
            # The colour edit swapped the skin for another character's, and the cardigan for the reference's white coat.
            wrong = figure(root, BROWN, WHITE_COAT, 'wrong.png')
            failures = '\n'.join(colour_failures(root, ACTION, named, root / wrong))
            self.assertIn('named skin #EDC3A0', failures)
            self.assertIn('named clothing #2F8F87', failures)
            self.assertIn('leave the character reference out', failures)

    def test_report_covers_every_placed_picture_with_named_colours(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            figure(root, SKIN, TEAL)
            (root / 'plan.json').write_text(json.dumps({'illustrations': [plan({'skin': SKIN, 'hair': None, 'clothing': [TEAL]})]}))
            (root / 'job.json').write_text(json.dumps({'drawing_plan': 'plan.json', 'pages': [{'beats': [{'actions': [ACTION]}]}]}))
            report = colour_report(root)
            self.assertEqual(report['status'], 'pass')
            self.assertEqual(len(report['people']), 2)
            figure(root, BROWN, TEAL, 'wrong.png')
            self.assertEqual(colour_report(root, 'p1', root / 'wrong.png')['status'], 'fail')
            with self.assertRaisesRegex(ValueError, '--plan-id'):
                colour_report(root, None, root / 'wrong.png')


if __name__ == '__main__':
    unittest.main()
