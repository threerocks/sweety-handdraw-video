"""Storyboard gate: monotony, lazy skipping, text-only films and drawing time are caught by scripts."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image, ImageDraw
from narration_master import script_text, sha256
from storyboard_lint import MAX_DRAWING_SHARE, check_build, drawing_time, pen_busy_share, pen_time, plan, timing, usage, finished_picture_hold_share
from video_job import FilmBuilder, save_wave
import test_style_and_plan as style_tests

WHOLE = [{'id': 'object', 'polygon': [[0, 0], [1, 0], [1, 1], [0, 1]]}]


def beats(*seconds):
    cursor, result = 1., []
    for index, length in enumerate(seconds):
        result.append({'id': f'b{index}', 'start': cursor, 'end': cursor + length})
        cursor += length + .2
    return result


def board(shots, **extra):
    return {'version': 1, 'shots': [{'id': f's{index}', 'purpose': 'shows the mechanism', **shot} for index, shot in enumerate(shots)], **extra}


def picture(identity, **extra):
    return {'id': identity, 'kind': 'illustration', 'subject': 'grape cluster', **extra}


class StoryboardPlanTests(unittest.TestCase):
    def test_grape_like_film_is_flagged_for_monotony_and_missing_visuals(self):
        timed = beats(*[10.5] * 6)
        report = plan(board([{'beats': [beat['id']], 'type': 'closeup', 'visuals': [picture(f'v{index}')]} for index, beat in enumerate(timed)]), timed)
        text = '\n'.join(report['errors'])
        self.assertEqual(report['stats']['required_visuals'], 8)
        self.assertIn('6 planned visuals', text)
        self.assertIn('1 shot type', text)
        self.assertIn('closeup fills 100%', text)
        self.assertTrue(report['warnings'])

    def test_varied_plan_passes_and_prints_complexity_tiers(self):
        timed = beats(16, 12, 12, 12)
        shots = [{'beats': ['b0'], 'type': 'scene_with_background', 'visuals': [picture('v0', background='vineyard at dawn')]},
                 {'beats': ['b1'], 'type': 'comparison', 'visuals': [picture('v1'), picture('v2')]},
                 {'beats': ['b2'], 'type': 'process_sequence', 'visuals': [picture('v3'), {'id': 'v4', 'kind': 'diagram', 'subject': 'arrows'}]},
                 {'beats': ['b3'], 'type': 'key_points', 'visuals': [picture('v5'), {'id': 'v6', 'kind': 'text_group', 'subject': 'three steps'}]}]
        report = plan(board(shots), timed)
        self.assertEqual(report['errors'], [])
        rows = {row['visual']: row for row in report['visuals']}
        self.assertIn('scene with background', rows['v0']['complexity'])
        self.assertIn('simple icon', rows['v1']['complexity'])
        self.assertEqual(report['stats']['required_visuals'], 7)

    def test_text_only_runtime_needs_a_declared_reason(self):
        timed = beats(6, 6, 6)
        shots = [{'beats': ['b0'], 'type': 'closeup', 'visuals': [picture('v0')]},
                 {'beats': ['b1', 'b2'], 'type': 'key_points', 'visuals': [{'id': f't{index}', 'kind': 'text_group', 'subject': 'point'} for index in range(2)]}]
        errors = '\n'.join(plan(board(shots), timed)['errors'])
        self.assertIn('Text-only shots fill', errors)
        self.assertIn('stop and tell the user', errors)
        self.assertEqual(plan(board(shots, text_only_reason='quote-only chapter'), timed)['errors'], [])

    def test_shot_types_need_their_structure_and_beats_must_be_complete(self):
        timed = beats(6)
        for kind, visuals in [('comparison', [picture('v0')]), ('scene_with_background', [picture('v0')]),
                              ('character_action', [picture('v0')])]:
            errors = '\n'.join(plan(board([{'beats': ['b0'], 'type': kind, 'visuals': visuals}]), timed)['errors'])
            self.assertIn(f'a {kind} shot needs', errors)
        errors = plan(board([{'beats': ['b0'], 'type': 'character_action', 'visuals': [picture('v0', characters=['fox'])]}]), timed)['errors']
        self.assertIn('undeclared character', errors[0])
        with self.assertRaisesRegex(ValueError, 'every narration beat'):
            plan(board([{'beats': ['b0'], 'type': 'closeup', 'visuals': [picture('v0')]}]), beats(4, 4))


class StoryboardTimingTests(unittest.TestCase):
    def timeline(self, images):
        pages, assets = [], {}
        for index, (beat, image) in enumerate(zip(beats(5, 5, 5), images)):
            name = f'e{index}'
            assets[name] = {}
            pages.append({'beats': [beat], 'events': [{'asset': name, 'caption': False, 'visual': f'v{index}', 'image': image,
                                                       'start': beat['start'] + .5, 'end': beat['end'] - .6}]})
        return {'pages': pages, 'assets': assets}

    def test_reuse_needs_a_new_illustration_or_a_callback_reason(self):
        shots = [{'beats': [f'b{index}'], 'type': 'closeup', 'visuals': [picture(f'v{index}')]} for index in range(3)]
        timeline = self.timeline([['same.png', None]] * 3)
        errors, _ = timing(board(shots), timeline)
        self.assertIn('more than 2 consecutive beats', '\n'.join(errors))
        shots[2]['callback'] = 'returns to the opening grape to close the loop'
        self.assertEqual(timing(board(shots), timeline)[0], [])
        atlas = self.timeline([['a.png', [0, 0, 5, 5]], ['a.png', [5, 0, 9, 5]], ['a.png', None]])
        self.assertEqual(timing(board([dict(shot, callback='') for shot in shots]), atlas)[0], [])

    def test_a_picture_each_sentence_annotates_may_carry_up_to_five_beats(self):
        timed = beats(*[3] * 6)
        shot = {'beats': [beat['id'] for beat in timed], 'type': 'closeup',
                'visuals': [picture('v0')] + [{'id': f't{index}', 'kind': 'text_group', 'subject': 'keyword'} for index in range(1, 6)]}
        events = [{'asset': 'e', 'caption': False, 'visual': 'v0', 'image': ['a.png', None], 'start': 1.5, 'end': 4}]
        overlays = [{'visual': f't{index}', 'start': beat['start'] + .3, 'end': beat['start'] + 1.3} for index, beat in enumerate(timed[1:5], 1)]
        page = {'beats': timed[:5], 'events': events, 'overlays': overlays}
        self.assertEqual(timing(board([shot]), {'pages': [page], 'assets': {'e': {}}})[0], [])
        page['overlays'] = overlays[:1] + overlays[2:]   # the third sentence adds nothing: the picture lingers
        self.assertIn('more than 2 consecutive beats (through b2', '\n'.join(timing(board([shot]), {'pages': [page], 'assets': {'e': {}}})[0]))
        # one form that adds a row in each later sentence (its scheduled moments) carries the picture as well
        page['overlays'] = [{'visual': 't1', 'start': timed[1]['start'] + .3, 'end': timed[4]['end'],
                             'moments': [beat['start'] + .3 for beat in timed[1:5]]}]
        self.assertEqual(timing(board([shot]), {'pages': [page], 'assets': {'e': {}}})[0], [])
        page.update(beats=timed, overlays=overlays + [{'visual': 't4', 'start': timed[5]['start'] + .3, 'end': timed[5]['start'] + 1.3}])   # a sixth beat is too many
        self.assertIn('through b5', '\n'.join(timing(board([shot]), {'pages': [page], 'assets': {'e': {}}})[0]))

    def test_cadence_counts_scene_parts_but_not_a_slow_single_object(self):
        timed = beats(12)
        timeline = {'pages': [{'beats': timed, 'events': [{'asset': 'e', 'caption': False, 'visual': 'v0', 'image': ['a.png', None],
                    'start': 1.5, 'end': 12.4}]}], 'assets': {'e': {'parts': [{'outline_start': step / 10} for step in range(6)]}}}
        single = board([{'beats': ['b0'], 'type': 'closeup', 'visuals': [picture('v0')]}])
        errors, stats = timing(single, timeline)
        self.assertIn('Nothing new is drawn for 11.5s', errors[0])
        self.assertEqual(stats['longest_static_range'], [1.5, 13.0])
        scene = board([{'beats': ['b0'], 'type': 'scene_with_background', 'visuals': [picture('v0', background='street')]}])
        self.assertEqual(timing(scene, timeline)[0], [])

    def test_drawing_time_minimum_is_an_error_and_required_is_a_warning(self):
        # 'required' is real human pace; video pace is a third of it, so 15 s needs 5 s and a 4 s window warns.
        timeline = {'pages': [{'events': [{'asset': 'e', 'caption': False, 'visual': 'v0', 'start': 1, 'end': 5}]}],
                    'assets': {'e': {'drawability': {'minimum_seconds': 6, 'required_seconds': 15, 'stroke_count': 140}}}}
        self.assertIn('simplify the image or extend its window across beats', drawing_time(timeline)[0][0])
        timeline['assets']['e']['drawability']['minimum_seconds'] = 3
        errors, warnings = drawing_time(timeline)
        self.assertEqual(errors, [])
        self.assertIn('the pen hurries', warnings[0])

    def test_pen_busy_share_counts_pen_time_once_and_pen_time_names_what_it_did(self):
        # The finished pictures must get time to hold: overlapping events count once, captions not at all.
        page = lambda start, end, events: {'beats': [{'start': start}], 'speech_end': end,
                                           'events': [{'caption': False, 'start': a, 'end': b, 'asset': name} for a, b, name in events]
                                           + [{'caption': True, 'start': start, 'end': end}]}
        drawn = {'type': 'illustration', 'segments': [{'kind': 'ink', 'start': 0, 'end': .4}, {'kind': 'travel', 'start': .4, 'end': .5},
                                                       {'kind': 'color', 'start': .5, 'end': 1}]}
        timeline = {'pages': [page(1, 11, [(1.5, 4, 'title'), (3, 6, 'drawn')]), page(11, 21, [(11.5, 20.5, 'revealed')])],
                    'assets': {'title': {'type': 'handwritten_text', 'segments': [{'kind': 'ink', 'start': 0, 'end': 1}]}, 'drawn': drawn,
                               'revealed': {'type': 'illustration', 'mode': 'reveal', 'segments': [{'kind': 'color', 'start': 0, 'end': 1}]}}}
        self.assertAlmostEqual(pen_busy_share(timeline), (4.5 + 9) / 20)
        self.assertGreater(pen_busy_share(timeline), MAX_DRAWING_SHARE - .1)
        # A revealed picture is pen time but not drawing: the report keeps it apart from lines and colour.
        self.assertEqual(pen_time(timeline), {'line': .06, 'colour': .075, 'reveal': .45, 'handwriting': .125, 'marker': 0., 'pen_up': .015})

    def test_production_build_requires_a_storyboard(self):
        with self.assertRaisesRegex(ValueError, 'Production requires a storyboard'):
            check_build('.', {'pages': []}, {'pages': [], 'assets': {}})

    def test_blank_lead_in_and_closing_tail_are_not_picture_reading_time(self):
        timeline = {'pages': [{'beats': [{'start': 1}], 'speech_end': 11, 'end': 16,
            'events': [{'asset': 'figure', 'caption': False, 'image': ['figure.png', None], 'start': 5, 'end': 9, 'hide': 16}]}]}
        self.assertAlmostEqual(pen_busy_share(timeline), .4)
        self.assertAlmostEqual(finished_picture_hold_share(timeline), .2)

    def test_diagram_is_complete_only_after_its_labels_are_finished(self):
        timeline = {'pages': [{'beats': [{'start': 1}], 'speech_end': 11,
            'events': [{'asset': 'arrow', 'visual': 'mechanism', 'caption': False, 'start': 1, 'end': 2},
                       {'asset': 'label', 'visual': 'mechanism', 'caption': False, 'start': 3, 'end': 4}]}]}
        board_doc = {'shots': [{'visuals': [{'id': 'mechanism', 'kind': 'diagram'}]}]}
        self.assertAlmostEqual(finished_picture_hold_share(timeline, board_doc), .7)

    def test_accumulating_board_can_reuse_the_scene_when_new_labels_are_added(self):
        timed = beats(4, 4, 4)
        board_doc = board([{'beats': ['b0', 'b1', 'b2'], 'type': 'closeup', 'board': 'main',
                            'visuals': [picture('scene'), {'id': 'label1'}, {'id': 'label2'}]}],
                          presentation={'mode': 'accumulating_board', 'reason': 'explain on one board'})
        timeline = {'pages': [{'beats': timed, 'end': 14, 'events': [
            {'asset': 'scene', 'image': ['scene.png', None], 'visual': 'scene', 'start': 1.5, 'end': 2.5, 'caption': False},
            {'asset': 'label', 'visual': 'label1', 'start': 5.5, 'end': 6, 'caption': False},
            {'asset': 'label', 'visual': 'label2', 'start': 9.5, 'end': 10, 'caption': False}]}], 'assets': {}}
        self.assertEqual(timing(board_doc, timeline)[0], [])


class StoryboardBuildTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        Image.new('RGB', (640, 360), 'white').save(self.root / 'cover.png')
        for name, color in (('icon', '#da934f'), ('leaf', '#9fb568')):
            image = Image.new('RGBA', (120, 100))
            ImageDraw.Draw(image).rounded_rectangle((10, 10, 110, 90), radius=12, fill=color, outline='#282520', width=4)
            image.save(self.root / f'{name}.png')
        time = np.arange(12 * 48000) / 48000
        save_wave(self.root / 'master.wav', (np.sin(time * 2 * np.pi * 220) * 7000).astype(np.int16))
        self.manifest = {'version': 1, 'runtime_fixture': True, 'title': '分镜检查', 'storyboard': 'storyboard.json',
            'canvas': {'width': 640, 'height': 360, 'fps': 24}, 'cover': {'source': 'cover.png'},
            'narration': {'source': 'master.wav', 'alignment': 'alignment.json'},
            'pages': [{'beats': [{'id': 'a', 'text': '先看果串。'}, {'id': 'b', 'text': '再看要点。'}],
                       'visual_groups': [{'start_beat': 'a', 'end_beat': 'b', 'actions': [
                           {'type': 'illustration', 'visual': 'v1', 'source': 'icon.png', 'parts': WHOLE, 'x': 40, 'y': 60, 'width': 120, 'height': 100},
                           {'type': 'text', 'visual': 'v2', 'from_beat': 'b', 'text': '要点', 'x': 300, 'y': 60, 'size': 40}]}]},
                      {'beats': [{'id': 'c', 'text': '最后看叶子。', 'actions': [
                           {'type': 'illustration', 'visual': 'v3', 'source': 'leaf.png', 'parts': WHOLE, 'x': 40, 'y': 60, 'width': 120, 'height': 100}]}]}]}
        self.board = board([{'beats': ['a', 'b'], 'type': 'closeup', 'visuals': [picture('v1'), {'id': 'v2', 'kind': 'text_group', 'subject': 'key point'}]},
                            {'beats': ['c'], 'type': 'closeup', 'visuals': [picture('v3')]}])
        self.write()

    def tearDown(self):
        self.temporary.cleanup()

    def write(self):
        alignment = {'source_sha256': sha256(self.root / 'master.wav'),
                     'script_sha256': hashlib.sha256(script_text(self.manifest).encode()).hexdigest(),
                     'beats': [{'id': beat_id, 'text': text, 'start': start, 'end': end} for beat_id, text, start, end in
                               [('a', '先看果串。', 0, 4), ('b', '再看要点。', 4.2, 8), ('c', '最后看叶子。', 8.2, 11.8)]]}
        for name, value in (('job.json', self.manifest), ('alignment.json', alignment), ('storyboard.json', self.board)):
            (self.root / name).write_text(json.dumps(value, ensure_ascii=False))

    def test_group_continues_across_sentences_and_anchored_action_starts_with_its_sentence(self):
        timeline = FilmBuilder(self.root).build()
        events = {event['visual']: event for page in timeline['pages'] for event in page['events'] if not event['caption']}
        self.assertEqual(events['v2']['start'], 5.2)
        self.assertLess(events['v1']['end'], 5.2)
        self.assertEqual(events['v3']['image'], ['leaf.png', None])
        self.assertEqual(timeline['storyboard']['stats']['required_visuals'], 2)
        self.assertLess(timeline['storyboard']['stats']['longest_static_seconds'], 8)

    def test_skipped_visual_and_unplanned_image_stop_the_build(self):
        self.manifest['pages'][0]['visual_groups'][0]['actions'].pop()
        self.manifest['pages'][1]['beats'][0]['actions'][0].pop('visual')
        self.write()
        errors = '\n'.join(usage(self.root, self.manifest, self.board))
        self.assertIn('leaf.png is not in the storyboard', errors)
        with self.assertRaisesRegex(ValueError, 'v2.*missing from job.json'):
            FilmBuilder(self.root).build()

    def test_only_a_progress_strip_may_be_a_navigation_visual(self):
        strip = {'recipe': 'roadmap', 'visual': 'where', 'params': {'mode': 'strip', 'groups': [{'label': 'g', 'count': 3}]}}
        self.manifest['pages'][0]['overlays'] = [strip]
        self.board['shots'][0]['visuals'].append({'id': 'where', 'kind': 'diagram', 'subject': 'progress', 'role': 'navigation'})
        self.assertFalse([error for error in usage(self.root, self.manifest, self.board) if 'navigation' in error])
        strip['params']['mode'] = 'full'                                # a full map explains the list: it is not navigation
        self.assertTrue([error for error in usage(self.root, self.manifest, self.board) if 'navigation' in error])

    def test_build_records_page_composition_and_production_rejects_tiny_pictures(self):
        timeline = FilmBuilder(self.root).build()
        composition = timeline['storyboard']['composition']
        self.assertEqual(len(composition['pages']), 2)
        self.assertFalse(composition['visual_acceptance_claimed'])
        self.assertTrue(any('content extent' in warning for warning in timeline['storyboard']['warnings']))
        with self.assertRaisesRegex(ValueError, 'visible picture extent'):
            check_build(self.root, {**self.manifest, 'runtime_fixture': False}, timeline)

    def test_shortest_picture_hold_warns_even_when_the_median_is_readable(self):
        timeline = FilmBuilder(self.root).build()
        events = {event.get('visual'): event for event in timeline['pages'][0]['events'] if not event['caption']}
        events['v2']['start'] = events['v1']['end'] + .2
        report = check_build(self.root, self.manifest, timeline)
        self.assertGreater(report['stats']['finished_hold_median_seconds'], 1.)
        self.assertTrue(any('below the 1.00s' in warning for warning in report['warnings']))

    def test_same_image_for_three_beats_stops_the_build(self):
        self.manifest['pages'][1]['beats'][0]['actions'][0]['source'] = 'icon.png'
        self.write()
        with self.assertRaisesRegex(ValueError, 'icon.png stays the main visual for more than 2 consecutive beats'):
            FilmBuilder(self.root).build()

    def test_plan_status_lists_remaining_work_by_shot(self):
        from plan_status import status
        self.manifest['pages'][0]['visual_groups'][0]['actions'].pop()
        self.write()
        report = status(self.root)
        items = [entry['item'] for entry in report['todo']]
        self.assertEqual(report['next_shot'], 's0')
        self.assertTrue(any(item.startswith('v2: place the planned text_group') for item in items))
        self.assertTrue(any('art-direction review missing' in item for item in items))
        self.assertIn('missing from job.json', '\n'.join(report['lint_errors']))

    def test_overlapping_pen_blocks_explain_how_to_merge(self):
        self.manifest['pages'][0]['beats'][1]['actions'] = [{'type': 'text', 'text': '冲突', 'x': 300, 'y': 200, 'size': 30}]
        self.write()
        with self.assertRaisesRegex(ValueError, 'one pen.*from_beat'):
            FilmBuilder(self.root).build()

    def test_characters_need_locked_references_cited_by_drawing_plans(self):
        Image.new('RGB', (20, 20), 'white').save(self.root / 'fox.png')
        self.board['characters'] = [{'id': 'fox', 'reference': 'fox.png', 'sha256': sha256(self.root / 'fox.png')}]
        self.board['shots'][1]['visuals'][0]['characters'] = ['fox']
        self.manifest['drawing_plan'] = 'plan.json'
        self.manifest['pages'][1]['beats'][0]['actions'][0]['drawing_plan_id'] = 'leaf'
        (self.root / 'plan.json').write_text(json.dumps({'illustrations': [{'id': 'leaf'}]}))
        self.assertIn('locked reference image', '\n'.join(usage(self.root, self.manifest, self.board)))
        (self.root / 'plan.json').write_text(json.dumps({'illustrations': [{'id': 'leaf', 'references': ['fox.png']}]}))
        self.assertEqual(usage(self.root, self.manifest, self.board), [])
        Image.new('RGB', (20, 20), 'black').save(self.root / 'fox.png')
        self.assertIn('locked sha256', '\n'.join(usage(self.root, self.manifest, self.board)))


class RecognizabilityTests(unittest.TestCase):
    def test_storyboarded_jobs_need_identity_cues_and_a_thumbnail_test(self):
        from drawing_contract import preflight, verify_drawing_plans, thumbnails
        from style_choice import digest, verify_style
        fixture = style_tests.StyleAndPlanTests('test_all_four_routes_are_valid_and_distinct')
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        root = fixture.root
        fixture.choose('inferred')
        plan_entry = fixture.human_plan()
        plan_entry['narrative_requirements'] = [{'id': 'pose', 'narration_quote': '人物站立。', 'observable': '人物完整站立'}]
        plan_path = root / 'plan.json'
        plan_path.write_text(json.dumps({'illustrations': [plan_entry]}))
        (root / 'job.json').write_text(json.dumps({'storyboard': 'storyboard.json'}))
        with self.assertRaisesRegex(ValueError, 'identity_cues'):
            preflight(root, plan_path)
        plan_entry['subjects'][0]['identity_cues'] = ['short hair', 'white shirt']
        plan_path.write_text(json.dumps({'illustrations': [plan_entry]}))
        preflight(root, plan_path)
        source = root / 'figure.png'
        Image.new('RGB', (10, 10), 'white').save(source)
        entry = dict(plan_id='figure', status='pass', source_sha256=digest(source), crop=None,
                     narrative_checks=[{'id': 'pose', 'status': 'pass', 'evidence': 'Fixture standing pose'}],
                     **dict.fromkeys(('anatomy_and_occlusion', 'part_boundaries', 'stroke_breaks', 'outline_then_fill', 'spatial_order'), 'Inspected'))
        review = {'style_sha256': verify_style(root)['resolved_sha256'], 'plan_sha256': digest(plan_path), 'illustrations': [entry]}
        (root / 'review.json').write_text(json.dumps(review))
        action = {'type': 'illustration', 'source': 'figure.png', 'drawing_plan_id': 'figure', 'parts': plan_entry['parts']}
        manifest = {'drawing_plan': 'plan.json', 'drawing_review': 'review.json', 'pages': [{'beats': [{'text': '人物站立。', 'actions': [action]}]}]}
        style = verify_style(root)
        self.assertEqual(verify_drawing_plans(root, manifest, style)['reviewed_illustrations'], 1)
        with self.assertRaisesRegex(ValueError, 'recognizability'):
            verify_drawing_plans(root, {**manifest, 'storyboard': 'storyboard.json'}, style)
        entry['recognizability'] = {'status': 'pass', 'named_as': 'a man in a white shirt', 'evidence': '160px sheet, named in 1s without captions'}
        (root / 'review.json').write_text(json.dumps(review))
        verify_drawing_plans(root, {**manifest, 'storyboard': 'storyboard.json'}, style)
        (root / 'job.json').write_text(json.dumps(manifest))
        self.assertTrue(Path(thumbnails(root)['sheet']).is_file())



def add_dense_composition(shots):
    for shot in shots:
        focus = next(visual['id'] for visual in shot['visuals'] if visual['kind'] == 'illustration')
        shot['composition'] = {'layout': 'scene', 'focus': focus, 'context': [
            {'visual': focus, 'element': 'vine', 'explains': 'shows where the fruit grows'},
            {'visual': focus, 'element': 'basket', 'explains': 'shows the harvesting result'}]}
    return shots


class PresentationDensityTests(unittest.TestCase):
    def test_picture_book_rejects_multi_hero_shots(self):
        timed = beats(8, 8)
        board_doc = board([
            {'beats': ['b0'], 'type': 'comparison', 'visuals': [picture('v0'), picture('v1')]},
            {'beats': ['b1'], 'type': 'closeup', 'visuals': [picture('v2')]},
        ], presentation={'mode': 'picture_book', 'reason': '儿童绘本，一页一图'})
        errors = '\n'.join(plan(board_doc, timed)['errors'])
        self.assertIn('picture_book', errors)
        self.assertIn('at most 1', errors)

    def test_accumulating_board_allows_longer_hold_and_fewer_visuals(self):
        timed = beats(20, 20, 20)
        # 60s narration; accumulating_board cadence_factor 0.55, hold 12 → required ~ ceil(60/12*0.55)=3
        shots = [
            {'beats': ['b0'], 'type': 'scene_with_background', 'board': 'main',
             'visuals': [picture('v0', background='chalkboard frame'), {'id': 't0', 'kind': 'text_group', 'subject': 'title'}]},
            {'beats': ['b1'], 'type': 'process_sequence', 'board': 'main',
             'visuals': [picture('v1'), {'id': 't1', 'kind': 'text_group', 'subject': 'step'}]},
            {'beats': ['b2'], 'type': 'key_points', 'board': 'main',
             'visuals': [picture('v2'), {'id': 't2', 'kind': 'text_group', 'subject': 'recap'}]},
        ]
        layout = {'id': 'main', 'purpose': 'explain three related steps on one board', 'regions': [
            {'id': f'row{index}', 'purpose': f'explain step {index}', 'box': [.05, .05 + index * .28, .95, .30 + index * .28],
             'visuals': [visual['id'] for visual in shot['visuals']]} for index, shot in enumerate(shots)]}
        report = plan(board(add_dense_composition(shots), boards=[layout], presentation={'mode': 'accumulating_board', 'reason': '课堂板书累积'}, max_hold_seconds=12), timed)
        self.assertEqual(report['stats']['presentation_mode'], 'accumulating_board')
        self.assertLessEqual(report['stats']['required_visuals'], 4)
        self.assertEqual(report['errors'], [])

    def test_dense_infographic_allows_multi_module_shot(self):
        timed = beats(12)
        shots = [{'beats': ['b0'], 'type': 'overview_recap',
                  'visuals': [picture('v0'), picture('v1'), picture('v2'), {'id': 't0', 'kind': 'text_group', 'subject': 'title'}]}]
        report = plan(board(add_dense_composition(shots), presentation={'mode': 'dense_infographic', 'reason': '成人科普信息图'}), timed)
        self.assertEqual(report['stats']['presentation_density'], 'dense')
        self.assertEqual(report['errors'], [])

    def test_a_progress_strip_is_navigation_not_a_picture_of_the_shot(self):
        timed = beats(10)
        strip = {'id': 'where', 'kind': 'diagram', 'subject': 'progress strip'}
        shots = [{'beats': ['b0'], 'type': 'closeup', 'visuals': [picture('v0'), {'id': 'form', 'kind': 'diagram', 'subject': 'table'}, strip]}]
        focus = {'mode': 'focus_diagram', 'reason': '一图一形式'}
        self.assertIn('at most 2', '\n'.join(plan(board(shots, presentation=focus), timed)['errors']))
        strip['role'] = 'navigation'
        self.assertFalse([e for e in plan(board(shots, presentation=focus), timed)['errors'] if 'at most' in e])


if __name__ == '__main__':
    unittest.main()
