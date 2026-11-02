"""Reject unreviewed image inputs or stale art-direction records before rendering."""
from pathlib import Path
from hosted_assets import resolve_asset
import hashlib
import json

PACKAGE=Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def image_inputs(manifest, root):
    inputs=[(manifest['cover']['source'],None)]
    for page in manifest['pages']:
        actions=[action for beat in page['beats'] for action in beat.get('actions',[])]
        actions.extend(action for group in page.get('visual_groups',[]) for action in group['actions'])
        inputs.extend((action['source'],action.get('crop')) for action in actions if action['type']=='illustration')
    hand=manifest.get('hand',{})
    if hand.get('source'):
        inputs.append((hand['source'],None))
    return [(Path(path).expanduser().resolve() if Path(path).expanduser().is_absolute() else (root/path).resolve(),crop) for path,crop in inputs]


def check_record(record_path, required, dependency_hash, style_hash=None):
    record_path=Path(record_path)
    record=json.loads(record_path.read_text())
    if record.get('dependency_sha256')!=dependency_hash:
        raise ValueError('Art-direction record does not match the bundled mandatory dependency')
    if style_hash is not None and record.get('style_sha256') != style_hash:
        raise ValueError('Art-direction review belongs to another resolved style')
    entries={Path(entry['source']).resolve() if Path(entry['source']).is_absolute() else (record_path.parent/entry['source']).resolve():entry for entry in record['images']}
    for source,crop in required:
        entry=entries.get(source)
        if not entry or entry.get('status')!='pass' or not isinstance(entry.get('evidence'), str) or not entry['evidence'].strip():
            raise ValueError('Image has no completed art-direction review: '+str(source))
        if entry.get('sha256')!=digest(resolve_asset(source)):
            raise ValueError('Image changed after art-direction review: '+str(source))
        # library_reuse: an approved image from the local library (asset_library.py), reviewed again for this film.
        if entry.get('method') not in ('generated_under_guidance','composed_with_verified_layers','library_reuse'):
            raise ValueError('Image lacks a supported guided-generation/composition method')
        regions=entry.get('approved_crops')
        if regions is not None and (crop is None or list(crop) not in regions):
            raise ValueError('This illustration crop was not approved: '+str(source))
        for field in ('brief_file','prompt_file'):
            evidence_path = record_path.parent / entry.get(field, '')
            if not evidence_path.is_file() or not evidence_path.read_text().strip():
                raise ValueError('Missing or empty pre-generation art-direction evidence: '+field)
            hash_field = field.replace('_file', '_sha256')
            if entry.get(hash_field) != digest(evidence_path):
                raise ValueError('Art-direction evidence changed or was not bound before generation: '+field)
    return len(required)


def verify_art_direction(root, manifest):
    if manifest.get('runtime_fixture'):
        return {'fixture_only':True,'delivery_allowed':False}
    from style_choice import verify_style
    from drawing_contract import verify_drawing_plans
    style_review = verify_style(root)
    if manifest.get('drawing', {}).get('mode', 'draw') != style_review['drawing_mode']:
        raise ValueError(f'job.json drawing.mode must follow the chosen style: set "drawing": {{"mode": "{style_review["drawing_mode"]}"}}')
    drawing_review = verify_drawing_plans(root, manifest, style_review)
    design_path=root/manifest.get('design_file','design.json')
    if not design_path.is_file():
        raise ValueError('A model-authored design.json is required before production')
    design=json.loads(design_path.read_text())
    route=design.get('narrative_route',{})
    if not route.get('decision') or not route.get('reason') or not route.get('director_review'):
        raise ValueError('Narrative route requires a model decision, reason and director review')
    if route['decision']!='default_life_concept' and not route.get('director_plan'):
        raise ValueError('A non-default narrative requires an omniailab director plan')
    from presentation import validate_presentation
    if not design.get('presentation'):
        raise ValueError('design.json needs presentation.mode (picture_book|focus_diagram|dense_infographic|accumulating_board|step_sequence|narrative_comic) and reason — director density, not a user option')
    presentation = validate_presentation(design['presentation'], 'design.json')
    from composition import check_presentation_match
    from storyboard_lint import load as load_storyboard
    board = load_storyboard(root, manifest)
    if board is not None:
        check_presentation_match(presentation, board)
    dependency=PACKAGE/'dependencies/sweety-image-art-direction/GUIDE.md'
    if not dependency.is_file():
        raise ValueError('Mandatory sweety-image-art-direction dependency is missing')
    dependency_hash=digest(dependency)
    reference=manifest.get('art_direction',{}).get('record')
    if not reference:
        raise ValueError('All images require an art_direction.record before build')
    count=check_record(root/reference,image_inputs(manifest,root),dependency_hash,style_review['resolved_sha256'])
    # A serious, adult film's prompts must not ask the image model for baby-schema features (style_fit.CUTE).
    import style_fit
    record=json.loads((root/reference).read_text())
    resolved=json.loads((root/'style/resolved.json').read_text())
    prompts=[(entry.get('prompt_file'),((root/reference).parent/entry['prompt_file']).read_text()) for entry in record['images'] if entry.get('prompt_file')]
    cute=style_fit.check_prompts(resolved,prompts)
    if cute:
        raise ValueError('Image prompts ask for childlike features in an adult film: '+', '.join(f'{name} ({word})' for name,word in cute))
    hand=manifest.get('hand',{})
    if not hand.get('source'):
        catalog=json.loads((PACKAGE/'assets/hands/catalog.json').read_text())
        identity=hand.get('preset',catalog['default'])
        profile=next((entry for entry in catalog['presets'] if entry['id']==identity),None)
        if profile is None:
            raise ValueError('Unknown hand preset: '+identity)
        count+=check_record(PACKAGE/'assets/hands/art-direction.json',[(PACKAGE/'assets/hands'/profile['source'],None)],dependency_hash)
    return {'dependency_sha256':dependency_hash,'reviewed_image_uses':count, 'style':style_review, 'drawing_plan':drawing_review, 'presentation':presentation}


def bind_evidence(record_path):
    """Bind existing preparation files; never fabricate image review or pass status."""
    record_path = Path(record_path)
    record = json.loads(record_path.read_text())
    if not record.get('images'):
        raise ValueError('At least one planned image is required')
    for entry in record['images']:
        for field in ('brief_file', 'prompt_file'):
            path = record_path.parent / entry.get(field, '')
            if not path.is_file() or not path.read_text().strip():
                raise ValueError('Missing or empty preparation file: ' + field)
            entry[field.replace('_file', '_sha256')] = digest(path)
    temporary = record_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(record_path)
    return len(record['images'])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['bind-evidence'])
    parser.add_argument('record', type=Path)
    arguments = parser.parse_args()
    print(json.dumps({'bound_images': bind_evidence(arguments.record), 'review_status_unchanged': True}))
