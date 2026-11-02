"""Require drawable art plans before generation and bind semantic reviews to inputs."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from style_choice import digest, verify_style

HUMAN_ORDER = ('head', 'neck', 'torso', 'left_arm', 'right_arm', 'lower_body')
# The colours a person must come out in (aunt-doctor film, 2026-11-02). A colour edit given the character reference copies
# its colours (the retired auntie came back in the reference's white coat and dark hair); one left to choose picks from
# the film's palette (a Chinese mother and baby were painted with the brown skin listed for other people in the film).
# So a line_first plan names each person's skin, hair and main garment, the colour prompt states them, and the coloured
# picture is measured: the share of the person's head paint near the skin colour, and of the torso paint near the garment.
PERSON_COLOURS = ('skin', 'hair', 'clothing')
MEASURED_COLOURS = {'skin': ('head', .2), 'clothing': ('torso', .3)}   # hair is often near-black ink and too few pixels
COLOUR_TOLERANCE = 16.   # CIE76 delta E from the named colour: the flat coat as painted, not its darker shadow step


def validate_plan(plan):
    if plan.get('outline_strategy') not in ('line_first', 'visible_source_ink', 'explicit_mask'):
        raise ValueError('Choose line_first (a line plate, then a colour edit of it), visible source contours or an aligned explicit '
                         'outline mask before generation')
    if not plan.get('composition') or not plan.get('drawability_reason'):
        raise ValueError('Plan the composition and explain drawing feasibility before generation')
    parts, subjects = plan.get('parts', []), plan.get('subjects', [])
    identities = [part.get('id') for part in parts]
    if not parts or not all(identities) or len(set(identities)) != len(identities):
        raise ValueError('A drawing plan needs unique semantic parts')
    if plan.get('outline_order') != identities or plan.get('fill_order') != identities:
        raise ValueError('Outline and fill must use the same planned part order')
    subject_map = {subject['id']: subject for subject in subjects}
    if not subjects or len(subject_map) != len(subjects):
        raise ValueError('Declare each visible subject and its kind')
    for part in parts:
        if part.get('subject') not in subject_map or not part.get('role') or not part.get('stroke_groups'):
            raise ValueError('Each part needs a subject, semantic role and independently lifted stroke groups')
        if not isinstance(part['stroke_groups'], list) or not all(isinstance(group,str) and group.strip() for group in part['stroke_groups']):
            raise ValueError('Describe real strokes, not one whole-frame reveal')
        if part['role'] in ('whole', 'whole_frame', 'person', 'body') and subject_map[part['subject']].get('kind') == 'human':
            raise ValueError('A human figure cannot be a single semantic part')
    for subject in subjects:
        if subject.get('kind') not in ('human', 'animal', 'object', 'environment'):
            raise ValueError('Unknown subject kind')
        selected = [part for part in parts if part['subject'] == subject['id']]
        if not selected:
            raise ValueError('Every declared subject needs parts')
        if subject['kind'] == 'human':
            roles = [part['role'] for part in selected]
            omitted = subject.get('omitted_roles', {})
            if any(role not in roles and not omitted.get(role) for role in HUMAN_ORDER):
                raise ValueError('Human plan must account for head, neck, torso, left/right arms and lower body, including occlusion')
            indices = [HUMAN_ORDER.index(role) for role in roles if role in HUMAN_ORDER]
            if indices != sorted(indices) and not subject.get('order_exception'):
                raise ValueError('Human part order is head, neck, torso, left arm, right arm, lower body; explain any actual composition exception')
        elif len(selected) == 1 and subject['kind'] == 'animal':
            raise ValueError('Animals also need separate anatomical parts')
    if plan.get('spatial_order') != 'top_to_bottom_left_to_right' and not plan.get('order_exception'):
        raise ValueError('Use top-to-bottom, left-to-right order or explain the composition exception')
    return True


def validate_narrative_requirements(plan, review, narration):
    requirements = plan.get('narrative_requirements', [])
    if not requirements or len({item.get('id') for item in requirements}) != len(requirements):
        raise ValueError('Every production illustration needs distinct observable narrative requirements')
    checks = {item.get('id'): item for item in review.get('narrative_checks', [])}
    for requirement in requirements:
        quote = requirement.get('narration_quote', '')
        if not requirement.get('id') or not quote or quote not in narration or not requirement.get('observable'):
            raise ValueError('Narrative requirements must cite actual narration and state what must be visibly present')
        check = checks.get(requirement['id'], {})
        if check.get('status') != 'pass' or not check.get('evidence'):
            raise ValueError('An illustration has an unverified or failed narrative requirement: ' + requirement['id'])
    return len(requirements)


def validate_identity(plan, job):
    """Storyboarded jobs: every subject names what makes it recognizable at thumbnail size; cited references exist."""
    for subject in plan.get('subjects', []):
        cues = subject.get('identity_cues')
        if not isinstance(cues, list) or not cues or not all(isinstance(cue, str) and cue.strip() for cue in cues):
            raise ValueError(f'{plan.get("id")}/{subject.get("id")}: list identity_cues that make it nameable at thumbnail size '
                             '(e.g. grapes: the cluster and a vine leaf, not one magnified berry)')
    if not all((Path(job) / path).is_file() for path in plan.get('references', [])):
        raise ValueError(f'{plan.get("id")}: a cited reference image is missing')
    return True


def hex_list(value):
    values = value if isinstance(value, list) else [value]
    if not values or not all(isinstance(item, str) and len(item) == 7 and item.startswith('#') for item in values):
        return None
    try:
        return [tuple(int(item[index:index + 2], 16) for index in (1, 3, 5)) for item in values]
    except ValueError:
        return None


def validate_person_colours(plan):
    """A line_first plan names, for every person, the colours their skin, hair and main garment must come out in."""
    for subject in plan.get('subjects', []):
        if subject.get('kind') != 'human':
            continue
        colours = subject.get('colours')
        missing = [key for key in PERSON_COLOURS if not isinstance(colours, dict) or key not in colours]
        wrong = [key for key in PERSON_COLOURS if isinstance(colours, dict) and colours.get(key) is not None and hex_list(colours[key]) is None]
        if missing or wrong:
            raise ValueError(f'{plan.get("id")}/{subject.get("id")}: name the person\'s colours before the colour edit, '
                             '"colours": {"skin": "#RRGGBB", "hair": "#RRGGBB" or null when not visible, "clothing": "#RRGGBB" or a list} '
                             f'({", ".join(missing + wrong)}), and write them into the colour prompt: a colour edit copies the character '
                             'reference or guesses from the palette')
    return True


def lab_array(rgb):
    """CIE L*a*b* (D65) of sRGB colours, shape (..., 3)."""
    value = np.asarray(rgb, float) / 255.
    linear = np.where(value <= .04045, value / 12.92, ((value + .055) / 1.055) ** 2.4)
    xyz = linear @ np.array([[.4124, .3576, .1805], [.2126, .7152, .0722], [.0193, .1192, .9505]]).T / [.95047, 1., 1.08883]
    f = np.where(xyz > .008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def person_colours(job, action, plan, image=None, size=720):
    """Measure the colours the plan names for each person in the coloured picture (the action's source, or `image`: a
    colour edit registered on the same plate). Parts come from the action, as the build assigns them."""
    import stroke_assets
    from PIL import Image
    from scipy import ndimage
    job = Path(job)
    source = job / action['source']
    mask = job / action['outline_mask'] if action.get('outline_mask') else None
    art = stroke_assets.prepare_artwork(source, size, size, action.get('crop'), mask)
    if action.get('part_regions'):
        names, owners, _ = stroke_assets.region_owners(job / action['region_map'], action['part_regions'], art['alpha'], source)
    else:
        names, owners = stroke_assets.polygon_owners(action['parts'], art['artwork'], art['alpha'])
    pixels, alpha = art['pixels'], art['alpha']
    if image is not None:
        other = stroke_assets.prepare_artwork(image, size, size, action.get('crop'), None)
        pixels = np.array(Image.fromarray(other['pixels']).resize(alpha.shape[::-1], Image.Resampling.NEAREST))
    paint = (alpha > 0) & ~ndimage.binary_dilation(art['outline'], iterations=2)
    colour = lab_array(pixels[:, :, :3])
    rows = []
    for subject in plan.get('subjects', []):
        for key, (role, needed) in MEASURED_COLOURS.items():
            named = hex_list((subject.get('colours') or {}).get(key)) if (subject.get('colours') or {}).get(key) is not None else None
            parts = [part['id'] for part in plan['parts'] if part['subject'] == subject.get('id') and part.get('role') == role]
            area = paint & np.isin(owners, [names.index(part) for part in parts if part in names])
            if not named or area.sum() < 50:
                continue
            distance = np.linalg.norm(colour[area][:, None, :] - lab_array(named)[None], axis=2).min(1)
            share = float(np.mean(distance < COLOUR_TOLERANCE))
            rows.append({'subject': subject['id'], 'colour': key, 'named': subject['colours'][key], 'parts': parts,
                         'share': round(share, 3), 'needed': needed, 'status': 'pass' if share >= needed else 'fail'})
    return rows


def colour_failures(job, action, plan, image=None):
    rows = person_colours(job, action, plan, image)
    return [f'{plan["id"]}/{row["subject"]}: only {row["share"]:.0%} of the {"/".join(row["parts"])} paint is the named {row["colour"]} '
            f'{row["named"]} (needs {row["needed"]:.0%}); state each person\'s skin, hair and clothing in the colour prompt, '
            'and leave the character reference out of a colour edit whose outfit differs from it' for row in rows if row['status'] == 'fail']


def storyboarded(job):
    path = Path(job) / 'job.json'
    return path.is_file() and bool(json.loads(path.read_text()).get('storyboard'))


def preflight(job, plan_path):
    job, plan_path = Path(job).resolve(), Path(plan_path).resolve()
    style = verify_style(job)
    plan_document = json.loads(plan_path.read_text())
    plans = plan_document.get('illustrations', [])
    if not plans or len({plan['id'] for plan in plans}) != len(plans):
        raise ValueError('Drawing plans need unique illustration ids')
    for plan in plans:
        validate_plan(plan)
        if storyboarded(job):
            validate_identity(plan, job)
        if plan.get('outline_strategy') == 'line_first':
            validate_person_colours(plan)
        requirements = plan.get('narrative_requirements', [])
        if not requirements or any(not all(item.get(key) for key in ('id','narration_quote','observable')) for item in requirements):
            raise ValueError('Define quoted narration and observable narrative requirements before generating the illustration')
    receipt = {'status': 'ready_for_generation', 'plan_sha256': digest(plan_path),
               'style_sha256': style['resolved_sha256'], 'illustrations': [plan['id'] for plan in plans]}
    path = plan_path.with_suffix('.preflight.json')
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
    return receipt


def illustration_actions(manifest):
    for page in manifest['pages']:
        for group in [*page['beats'], *page.get('visual_groups', [])]:
            for action in group.get('actions', []):
                if action['type'] == 'illustration':
                    yield action


def verify_drawing_plans(job, manifest, style):
    reference = manifest.get('drawing_plan')
    if not reference:
        raise ValueError('A pre-generation drawing_plan is mandatory for every production job')
    path = Path(job) / reference
    document = json.loads(path.read_text())
    receipt = json.loads(path.with_suffix('.preflight.json').read_text())
    if receipt.get('status') != 'ready_for_generation' or receipt.get('plan_sha256') != digest(path) or receipt.get('style_sha256') != style['resolved_sha256']:
        raise ValueError('Drawing preflight is missing or stale; repair the plan before generating assets')
    plans = {plan['id']: plan for plan in document['illustrations']}
    for plan in plans.values():
        validate_plan(plan)
    reviews = json.loads((Path(job) / manifest['drawing_review']).read_text())
    if reviews.get('style_sha256') != style['resolved_sha256']:
        raise ValueError('Drawing review belongs to another resolved style')
    if reviews.get('plan_sha256') != digest(path):
        raise ValueError('Drawing review belongs to another plan')
    inspected = {entry['plan_id']: entry for entry in reviews.get('illustrations', [])}
    checked = 0
    narration = ''.join(beat['text'] for page in manifest['pages'] for beat in page['beats'])
    for action in illustration_actions(manifest):
        identity = action.get('drawing_plan_id')
        if identity not in plans:
            raise ValueError('Illustration has no pre-generation drawing plan')
        plan, review = plans[identity], inspected.get(identity, {})
        expected = [part['id'] for part in plan['parts']]
        actual = [part['id'] for part in action.get('part_regions') or action.get('parts', [])]
        if expected != actual:
            raise ValueError('Actual part masks must match the planned outline/fill order')
        source = Path(action['source'])
        source = source if source.is_absolute() else Path(job) / source
        if review.get('status') != 'pass' or review.get('source_sha256') != digest(source) or review.get('crop') != action.get('crop'):
            raise ValueError('Semantic drawing review is missing, failed, or belongs to another image/crop')
        for field in ('anatomy_and_occlusion', 'part_boundaries', 'stroke_breaks', 'outline_then_fill', 'spatial_order'):
            if not review.get(field):
                raise ValueError('Actual-image drawing review needs evidence for ' + field)
        validate_narrative_requirements(plan, review, narration)
        if manifest.get('storyboard'):
            validate_identity(plan, job)
            test = review.get('recognizability', {})
            if test.get('status') != 'pass' or not str(test.get('named_as', '')).strip() or not str(test.get('evidence', '')).strip():
                raise ValueError(f'{identity}: thumbnail recognizability test is missing or failed; redraw around its identity_cues, '
                                 'then record status/named_as/evidence (drawing_contract.py JOB --thumbnails)')
        if plan['outline_strategy'] in ('explicit_mask', 'line_first') and not action.get('outline_mask'):
            raise ValueError('This style needs its planned aligned outline mask')
        if plan['outline_strategy'] == 'line_first':
            validate_line_first(job, action, source)
            failures = colour_failures(job, action, plan)
            if failures:
                raise ValueError('A person came out in the wrong colours:\n- ' + '\n- '.join(failures))
        if action.get('outline_mask'):
            mask_path = Path(job) / action['outline_mask']
            if review.get('outline_mask_sha256') != digest(mask_path):
                raise ValueError('Outline mask changed after semantic review')
            from PIL import Image
            with Image.open(source) as image, Image.open(mask_path) as mask:
                if image.size != mask.size:
                    raise ValueError('Outline mask must align with the complete source image before cropping')
        checked += 1
    return {'plan_sha256': digest(path), 'reviewed_illustrations': checked}


def validate_line_first(job, action, source):
    """A line-first picture is the registered composite line_first.py wrote, with the plate as its outline mask."""
    if not action.get('line_first'):
        raise ValueError('A line_first illustration needs "line_first": the report line_first.py wrote next to the source')
    report_path = Path(job) / action['line_first']
    if not report_path.is_file():
        raise ValueError('Missing line-first report: ' + action['line_first'])
    report = json.loads(report_path.read_text())
    if report.get('status') != 'pass':
        raise ValueError('The colour edit failed line-first registration; regenerate it with the report prompt_fix: ' + action['line_first'])
    hashes = report.get('sha256', {})
    if hashes.get('source') != digest(source) or hashes.get('outline_mask') != digest(Path(job) / action['outline_mask']):
        raise ValueError('The source or outline mask is not the one line_first.py registered; rerun it after changing either plate')


def validate_strokes(metadata):
    segments = metadata['segments']
    previous_ink = None
    for segment in segments:
        if segment['end'] <= segment['start']:
            raise ValueError('Every stroke and pen lift needs a positive duration')
        if segment['kind'] != 'ink':
            continue
        points = np.asarray(segment['points'], dtype=float)
        if len(points) < 2 or not np.isfinite(points).all() or (np.linalg.norm(np.diff(points, axis=0), axis=1) > 1.5).any():
            raise ValueError('A pen stroke bridges disconnected outline points; split and lift the pen')
        if previous_ink is not None and segment['start'] <= previous_ink['end'] + 1e-10:
            raise ValueError('Separate strokes require a pen-up interval, even at a shared endpoint')
        previous_ink = segment
    return True


def colour_report(job, plan_id=None, image=None):
    """Person colours of every placed picture whose plan names them (or one plan id, optionally on another image)."""
    job = Path(job).resolve()
    manifest = json.loads((job / 'job.json').read_text())
    plans = {plan['id']: plan for plan in json.loads((job / manifest['drawing_plan']).read_text())['illustrations']}
    if image is not None and not plan_id:
        raise ValueError('--image needs --plan-id: the picture whose parts it is measured with')
    rows = []
    for action in illustration_actions(manifest):
        identity = action.get('drawing_plan_id')
        if identity in plans and (plan_id is None or identity == plan_id):
            rows += [{'plan_id': identity, **row} for row in person_colours(job, action, plans[identity], image)]
    return {'status': 'fail' if any(row['status'] == 'fail' for row in rows) else 'pass', 'tolerance_delta_e': COLOUR_TOLERANCE, 'people': rows}


def thumbnails(job, size=160):
    """Numbered caption-free thumbnails for the 1-second naming test; names stay out of the sheet."""
    from PIL import Image, ImageDraw
    job = Path(job).resolve()
    actions = list(illustration_actions(json.loads((job / 'job.json').read_text())))
    if not actions:
        raise ValueError('No illustrations to test')
    sheet = Image.new('RGB', (min(len(actions), 6) * (size + 24), math.ceil(len(actions) / 6) * (size + 44)), 'white')
    for index, action in enumerate(actions):
        with Image.open(job / action['source']) as source:
            image = source.convert('RGBA').crop(tuple(action['crop'])) if action.get('crop') else source.convert('RGBA')
        image.thumbnail((size, size))
        left, top = index % 6 * (size + 24) + 12, index // 6 * (size + 44) + 32
        sheet.paste(image, (left, top), image)
        ImageDraw.Draw(sheet).text((left, top - 24), str(index + 1), fill='#202020')
    path = job / 'work/review/recognizability.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return {'sheet': str(path), 'size_px': size, 'cells': [{'number': index + 1, 'drawing_plan_id': action.get('drawing_plan_id'),
            'visual': action.get('visual')} for index, action in enumerate(actions)]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', type=Path)
    parser.add_argument('plan', type=Path, nargs='?')
    parser.add_argument('--thumbnails', action='store_true', help='Write a numbered caption-free thumbnail sheet for the recognizability test')
    parser.add_argument('--colours', action='store_true', help='Measure the colours the plan names for each person in the coloured pictures')
    parser.add_argument('--image', type=Path, help='with --colours and --plan-id: a colour edit to measure before it is registered')
    parser.add_argument('--plan-id', help='with --colours: only this drawing plan id')
    arguments = parser.parse_args()
    if arguments.thumbnails:
        print(json.dumps(thumbnails(arguments.job), ensure_ascii=False, indent=1))
    elif arguments.colours:
        report = colour_report(arguments.job, arguments.plan_id, arguments.image)
        print(json.dumps(report, ensure_ascii=False, indent=1))
        raise SystemExit(1 if report['status'] == 'fail' else 0)
    elif arguments.plan:
        print(json.dumps(preflight(arguments.job, arguments.plan), ensure_ascii=False))
    else:
        parser.error('Give a drawing plan to preflight or --thumbnails')
