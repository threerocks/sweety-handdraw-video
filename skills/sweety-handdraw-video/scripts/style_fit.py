"""Hold a model-chosen style to its content: the subject, era, audience and tone decide the style family and how old the
characters look, and a new film must not quietly repeat the look of the last ones.

Why: left free, the image model and the director converge on one cute flat look (four films in a row shared shading,
texture, background and palette word for word; a 35-year-old became a round-faced boy; a sober commentary got smiling
mascots). A mismatch breaks immersion, and an unimmersed viewer swipes away. A style the user picks is theirs and is
only recorded, never second-guessed.

resolved.json (model-chosen routes) adds:
  content_analysis: {genre, era, audience_band, tone: {primary, humor_density 0..1}, sensitivity, narrative_stance,
                     evidence: [>= 3 quotes or beat ids from the script]}
  style_family, directions: [>= 3 candidates {family, idea, score}], characters: [{id, age, head_to_body, face}]
  (or no_characters_reason), avoid: [...], diversity: {differs_on: [>= 3 of family/proportion/palette/line/background]}
  optional: comedic_reason, series_reason, ink_reason
"""
import json
import math
import re
from difflib import SequenceMatcher
from pathlib import Path

GENRES = {'history_ancient', 'history_modern', 'biography', 'true_crime', 'finance_business', 'commentary', 'science',
          'tech_essay', 'warm_adult_story', 'workplace_growth', 'guofeng_myth', 'medical_health', 'kids', 'comedy', 'other'}
FAMILIES = {'ink_brush_mineral', 'ligne_claire', 'woodcut_flat', 'semi_realistic_clear_line', 'noir_red', 'editorial_concept',
            'editorial_cartoon', 'science_atlas', 'engineer_notebook', 'adult_picturebook', 'new_yorker_cartoon',
            'baimiao_nianhua', 'medical_diagram', 'chibi', 'kids_storybook', 'doodle_blob', 'other'}
AUDIENCES = {'kids', 'teens', 'adults', 'seniors', 'general'}
CHILDISH = {'chibi', 'kids_storybook', 'doodle_blob'}
SERIOUS = {'history_ancient', 'history_modern', 'biography', 'true_crime', 'finance_business', 'commentary', 'tech_essay',
           'warm_adult_story', 'medical_health'}
GRAVE = {'history_ancient', 'history_modern', 'biography', 'true_crime', 'commentary', 'medical_health'}
# Words that pull an image model toward baby schema (round head, dot eyes, blush): wrong for an adult, serious film.
CUTE = re.compile(r'chibi|kawaii|\bcute\b|big[- ]?eyes?|bobble|Q版|萌|大头|圆脸|豆豆眼|腮红|娃娃脸|婴儿肥', re.I)
DIMENSIONS = {'family', 'proportion', 'palette', 'line', 'background'}
ENGINE_INK = '#1E2733'   # the engine's default ink: a style that names it is usually echoing the tool, not choosing
RECENT = 5


def hex_colours(text):
    return [tuple(int(code[k:k + 2], 16) for k in (0, 2, 4)) for code in re.findall(r'#([0-9A-Fa-f]{6})\b', text)]


def lab(rgb):
    """CIE L*a*b* (D65) of an sRGB colour."""
    def linear(value):
        value /= 255.
        return value / 12.92 if value <= .04045 else ((value + .055) / 1.055) ** 2.4
    r, g, b = (linear(float(value)) for value in rgb)
    x, y, z = (.4124 * r + .3576 * g + .1805 * b) / .95047, .2126 * r + .7152 * g + .0722 * b, (.0193 * r + .1192 * g + .9505 * b) / 1.08883
    f = lambda t: t ** (1 / 3) if t > .008856 else 7.787 * t + 16 / 116
    return 116 * f(y) - 16, 500 * (f(x) - f(y)), 200 * (f(y) - f(z))


def palette_overlap(first, second):
    """Share of the first palette's colours that have a near twin (CIE76 delta E < 5) in the second."""
    if not first or not second:
        return 0.
    near = lambda a, b: math.dist(lab(a), lab(b)) < 5
    return sum(any(near(colour, other) for other in second) for colour in first) / len(first)


def spec_text(resolved):
    spec = resolved.get('style_spec', {})
    return ' '.join(str(spec.get(key, '')) for key in ('line', 'shading', 'texture', 'background'))


def recent_styles(job, limit=RECENT):
    """resolved.json of the most recent other films beside this one (same parent folder), newest first."""
    job = Path(job).resolve()
    found = []
    for path in job.parent.glob('*/style/resolved.json'):
        if path.parents[1] == job:
            continue
        try:
            found.append((path.stat().st_mtime, path.parents[1].name, json.loads(path.read_text())))
        except (OSError, ValueError):
            continue
    return [(name, data) for _, name, data in sorted(found, key=lambda item: -item[0])[:limit]]


def check(job, resolved, selection):
    """Raise ValueError when a model-chosen style does not follow its content or repeats recent films. Returns a report."""
    user_chosen = selection.get('mode') == 'custom' or (selection.get('mode') == 'catalog' and selection.get('confirmation') in ('user_dialog', 'explicit_user_request'))
    if user_chosen:
        return {'checked': False, 'reason': 'style chosen by the user'}
    if resolved.get('legacy_style_record') and (Path(job) / 'outputs/film.mp4').is_file():
        return {'checked': False, 'reason': 'film delivered before the content-fit record existed'}
    analysis = resolved.get('content_analysis') or {}
    genre, audience, family = analysis.get('genre'), analysis.get('audience_band'), resolved.get('style_family')
    tone = analysis.get('tone') or {}
    problems = []
    if genre not in GENRES or audience not in AUDIENCES or not analysis.get('era') or not tone.get('primary') or not analysis.get('sensitivity') or not analysis.get('narrative_stance'):
        problems.append('content_analysis needs genre (' + '/'.join(sorted(GENRES)) + '), era, audience_band (' + '/'.join(sorted(AUDIENCES))
                        + '), tone.primary, sensitivity and narrative_stance, decided from the script before any style')
    if len([item for item in analysis.get('evidence', []) if str(item).strip()]) < 3:
        problems.append('content_analysis.evidence must quote at least 3 lines or beat ids from the script that set genre, era, audience and tone')
    if family not in FAMILIES:
        problems.append('style_family must be one of: ' + ', '.join(sorted(FAMILIES)))
    directions = resolved.get('directions') or []
    if len({item.get('family') for item in directions if item.get('family')}) < 3 or not all(item.get('idea') for item in directions):
        problems.append('compare at least 3 structurally different directions (different style_family, each with an idea and a score) before choosing')
    elif family not in {item.get('family') for item in directions}:
        problems.append('style_family must be one of the compared directions')
    humour = float(tone.get('humor_density', 0) or 0)
    if family in CHILDISH and audience != 'kids' and genre in SERIOUS and not (humour >= .5 and str(resolved.get('comedic_reason', '')).strip()):
        problems.append(f'{family} reads as for children; a {genre} film for {audience} needs an adult style family '
                        '(only a mostly comic script with comedic_reason may use it)')
    characters = resolved.get('characters')
    if characters is None and not str(resolved.get('no_characters_reason', '')).strip():
        problems.append('declare characters [{id, age, head_to_body, face}] or no_characters_reason')
    for character in characters or []:
        age, ratio = character.get('age'), character.get('head_to_body')
        if not isinstance(age, (int, float)) or not isinstance(ratio, (int, float)) or not str(character.get('face', '')).strip():
            problems.append(f'character {character.get("id")} needs a numeric age, head_to_body and a face description')
            continue
        floor = 6.5 if genre in GRAVE else 5.5
        if age >= 18 and ratio < floor and not (family in CHILDISH and humour >= .5):
            problems.append(f'character {character.get("id")} is an adult ({age}) drawn {ratio} heads tall: adults need at least {floor} heads in a {genre} film')
        if age <= 10 and ratio > 4.5:
            problems.append(f'character {character.get("id")} is a child ({age}) but {ratio} heads tall: children read as children at 4.5 heads or fewer')
        if age >= 18 and asks_for_cute(str(character.get('face', ''))) and genre in SERIOUS:
            problems.append(f'character {character.get("id")} face description uses baby-schema words: {asks_for_cute(character["face"])}')
    if len([item for item in resolved.get('avoid', []) if str(item).strip()]) < 2:
        problems.append('avoid must list at least 2 looks this content must not take (e.g. chibi programmers, glowing brains)')
    spec_words = json.dumps(resolved.get('style_spec', {}), ensure_ascii=False) + str(resolved.get('style_name', ''))
    if genre in SERIOUS and audience != 'kids' and asks_for_cute(spec_words) and not str(resolved.get('comedic_reason', '')).strip():
        problems.append(f'style_spec of a {genre} film asks for baby-schema features ({asks_for_cute(spec_words)})')
    if ENGINE_INK.lower() in spec_words.lower() and not str(resolved.get('ink_reason', '')).strip():
        problems.append(f'{ENGINE_INK} is the engine default ink, not a choice: pick the ink from the style, or give ink_reason')
    differs = set(resolved.get('diversity', {}).get('differs_on', [])) & DIMENSIONS
    if len(differs) < 3:
        problems.append('diversity.differs_on must name at least 3 of family/proportion/palette/line/background in which this film differs from the recent ones')
    recent = recent_styles(job)
    mine = hex_colours(json.dumps(resolved.get('style_spec', {})))
    compared = []
    for name, other in recent:
        text = SequenceMatcher(None, spec_text(resolved), spec_text(other)).ratio()
        colours = palette_overlap(mine, hex_colours(json.dumps(other.get('style_spec', {}))))
        compared.append({'film': name, 'spec_similarity': round(text, 2), 'palette_overlap': round(colours, 2), 'family': other.get('style_family')})
        if text >= .8:
            problems.append(f'line/shading/texture/background read {text:.0%} the same as {name}: describe this film\'s own look')
        if colours >= .6 and len(mine) >= 3:
            problems.append(f'{colours:.0%} of the palette repeats {name} (delta E < 5): choose colours from this content')
    if recent and recent[0][1].get('style_family') == family and family and not str(resolved.get('series_reason', '')).strip():
        problems.append(f'the last film ({recent[0][0]}) already used {family}: choose another family or give series_reason')
    if problems:
        raise ValueError('Style does not follow the content yet:\n- ' + '\n- '.join(problems))
    return {'checked': True, 'genre': genre, 'audience_band': audience, 'style_family': family, 'compared': compared}


# A forbidding word just before a baby-schema word, in the same clause: “never cute”, “no big eyes”, “不要大头和圆脸”.
NEGATION = re.compile(r'\b(never|not|no|avoid|without)\b([ -]+\w+){0,2}[ (-]*$|(禁止|避免|不要|不能|不画|别)[^。；，,;!?！？\n]{0,6}$', re.I)


def asks_for_cute(text):
    """The first baby-schema word the text asks for; a word it forbids (“never cute, no chibi”, “不要大头”) is not asked."""
    for match in CUTE.finditer(text):
        if not NEGATION.search(text[max(0, match.start() - 40):match.start()]):
            return match.group(0)
    return None


def check_prompts(resolved, prompt_texts):
    """Image prompts of a serious, adult film must not ask for baby-schema features."""
    genre = (resolved.get('content_analysis') or {}).get('genre')
    audience = (resolved.get('content_analysis') or {}).get('audience_band')
    if genre not in SERIOUS or audience == 'kids' or str(resolved.get('comedic_reason', '')).strip():
        return []
    return [(name, asks_for_cute(text)) for name, text in prompt_texts if asks_for_cute(text)]
