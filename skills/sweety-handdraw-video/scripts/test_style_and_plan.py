"""Behavioral checks for style route isolation and physically separate strokes."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image, ImageDraw
from style_choice import prepare_catalog, digest, verify_style
from drawing_contract import validate_plan, validate_strokes, preflight, verify_drawing_plans
import stroke_assets
from video_job import find_font


# A model-chosen style must record how it follows the content (style_fit.py).
CONTENT_FIT = {'content_analysis': {'genre': 'science', 'era': 'present day', 'audience_band': 'adults', 'tone': {'primary': 'curious', 'humor_density': .1},
                                    'sensitivity': 'low', 'narrative_stance': 'explainer', 'evidence': ['b1 sets the topic', 'b2 names the audience', 'b3 sets the tone']},
               'style_family': 'science_atlas', 'characters': [], 'avoid': ['chibi faces', 'neon glow'],
               'directions': [{'family': 'science_atlas', 'idea': 'cutaway of the object', 'score': 8},
                              {'family': 'editorial_concept', 'idea': 'one visual metaphor', 'score': 6},
                              {'family': 'ligne_claire', 'idea': 'clear-line scene', 'score': 5}],
               'diversity': {'differs_on': ['family', 'palette', 'line']}}


class StyleAndPlanTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        source = self.root/'catalog-source'
        (source/'examples').mkdir(parents=True)
        (source/'STYLES.md').write_text('## 1. Test lines\nComplete recipe one.\n## 2. Test colors\nComplete recipe two.\n')
        for identity in ('01','02'):
            Image.new('RGB',(20,20),'white').save(source/'examples'/(identity+'-test.png'))
        self.catalog = prepare_catalog(self.root,source)
        font,index = find_font({})
        stroke_assets.configure_assets(self.root/'assets',font,index)

    def tearDown(self):
        self.temporary.cleanup()

    def choose(self, mode, identity=None, confirmation='explicit_user_request'):
        selection={'mode':mode,'confirmation':confirmation,'evidence':'Isolated test selection'}
        if mode=='catalog':
            selection['catalog_id']=identity
        if mode=='custom':
            selection['user_description']='Red pencil, white background, sparse marks'
        path=self.root/'style/selection.json'
        path.write_text(json.dumps(selection))
        resolved={'status':'resolved','mode':mode,'selection_sha256':digest(path),'style_name':'Test style',
                  'rationale':'Test rationale','style_spec':dict.fromkeys(('line','palette','shading','texture','background'),'specified')}
        if mode in ('catalog','catalog_auto'):
            entry=next(entry for entry in self.catalog['styles'] if entry['id']==identity)
            resolved.update(catalog_id=identity,recipe_sha256=entry['recipe_sha256'])
            if 'drawing_mode' not in entry:
                resolved['drawing_mode']='draw'
        if mode in ('catalog_auto','inferred'):
            resolved['content_fit']='Matches the explanation in this fixture'
        if mode in ('catalog_auto','inferred') or confirmation in ('timeout_default','user_delegated'):
            resolved.update(json.loads(json.dumps(CONTENT_FIT)))
        if mode=='custom':
            resolved['style_basis']=selection['user_description']
        if mode in ('inferred','custom'):
            resolved['drawing_mode']='draw'
        self.save_resolution(resolved)
        return resolved

    def save_resolution(self,resolved):
        (self.root/'style/resolved.json').write_text(json.dumps(resolved))

    def test_all_four_routes_are_valid_and_distinct(self):
        for mode in ('catalog','catalog_auto','inferred','custom'):
            self.choose(mode,'1' if mode in ('catalog','catalog_auto') else None)
            self.assertEqual(verify_style(self.root)['mode'],mode)

    def test_drawing_mode_is_declared_and_follows_the_bundled_catalog(self):
        resolved=self.choose('inferred')
        del resolved['drawing_mode'];self.save_resolution(resolved)
        with self.assertRaisesRegex(ValueError,'drawing_mode'):
            verify_style(self.root)
        bundled=prepare_catalog(self.root,Path(__file__).resolve().parents[1]/'dependencies/hand-drawn-styles')
        modes={entry['id']:entry['drawing_mode'] for entry in bundled['styles']}
        self.assertEqual((modes['1'],modes['20'],modes['11'],modes['13'],modes['9']),('draw','draw','reveal','reveal','reveal'))
        self.catalog=bundled
        # Clicking a card the picker labels as revealed is the user's informed choice.
        resolved=self.choose('catalog','11','user_dialog')
        self.assertEqual(verify_style(self.root)['drawing_mode'],'reveal')
        resolved['drawing_mode']='draw';self.save_resolution(resolved)
        with self.assertRaisesRegex(ValueError,'catalog value'):
            verify_style(self.root)
        self.choose('catalog_auto','17')
        self.assertEqual(verify_style(self.root)['drawing_mode'],'draw')
        # A model-picked style is drawn and coloured; black-and-white style 1 needs a reason, a user's own pick does not.
        for mode,identity,confirmation in (('catalog_auto','1','explicit_user_request'),('catalog','1','timeout_default'),('catalog','1','user_delegated')):
            resolved=self.choose(mode,identity,confirmation)
            with self.assertRaisesRegex(ValueError,'monochrome_reason'):
                verify_style(self.root)
            resolved['monochrome_reason']='The user asked for an xkcd-like black line look';self.save_resolution(resolved)
            self.assertEqual(verify_style(self.root)['drawing_mode'],'draw')
        self.choose('catalog','1','user_dialog')
        self.assertEqual(verify_style(self.root)['drawing_mode'],'draw')
        # Every other route to reveal gives up drawing without the user knowing: it needs the user's own answer.
        for mode,identity,confirmation in (('catalog_auto','11','user_dialog'),('catalog','11','timeout_default'),('catalog','11','explicit_user_request'),
                                           ('inferred',None,'user_delegated'),('inferred',None,'explicit_user_request'),('custom',None,'user_dialog')):
            resolved=self.choose(mode,identity,confirmation)
            resolved['drawing_mode']='reveal';self.save_resolution(resolved)
            with self.assertRaisesRegex(ValueError,'only the user may choose it'):
                verify_style(self.root)
            resolved['reveal_accepted']='User: keep the ink-wash look even though it is brushed in, not drawn';self.save_resolution(resolved)
            self.assertEqual(verify_style(self.root)['drawing_mode'],'reveal')

    def test_inferred_and_custom_cannot_borrow_a_catalog_recipe(self):
        for mode in ('inferred','custom'):
            resolved=self.choose(mode)
            resolved['catalog_id']='1';self.save_resolution(resolved)
            with self.assertRaisesRegex(ValueError,'independent'):
                verify_style(self.root)

    def test_direct_choice_cannot_be_replaced_by_model(self):
        resolved=self.choose('catalog','1')
        resolved.update(catalog_id='2',recipe_sha256=self.catalog['styles'][1]['recipe_sha256'])
        self.save_resolution(resolved)
        with self.assertRaisesRegex(ValueError,'replace'):
            verify_style(self.root)

    def test_custom_description_cannot_be_replaced_by_content(self):
        resolved=self.choose('custom');resolved['style_basis']='The story seems emotional, use watercolor'
        self.save_resolution(resolved)
        with self.assertRaisesRegex(ValueError,'exact user description'):
            verify_style(self.root)

    def test_missing_and_stale_selection_fail_closed(self):
        with self.assertRaisesRegex(ValueError,'required'):
            verify_style(self.root)
        self.choose('inferred')
        path=self.root/'style/selection.json';path.write_text(path.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'stale'):
            verify_style(self.root)

    def human_plan(self):
        roles=['head','neck','torso','left_arm','right_arm','lower_body']
        return {'id':'figure','composition':'One standing person','drawability_reason':'Sparse outlines with visible separated limbs',
                'outline_strategy':'visible_source_ink','subjects':[{'id':'person','kind':'human'}],
                'parts':[{'id':role,'subject':'person','role':role,'stroke_groups':['outer contour','interior marks']} for role in roles],
                'outline_order':roles,'fill_order':roles,'spatial_order':'top_to_bottom_left_to_right'}

    def test_whole_human_and_unexplained_missing_parts_are_rejected(self):
        plan=self.human_plan();self.assertTrue(validate_plan(plan))
        plan['parts']=[{'id':'whole','subject':'person','role':'whole','stroke_groups':['single continuous line']}]
        plan['outline_order']=plan['fill_order']=['whole']
        with self.assertRaisesRegex(ValueError,'single semantic part'):
            validate_plan(plan)
        plan=self.human_plan();plan['parts']=plan['parts'][:-1];plan['outline_order']=plan['fill_order']=[part['id'] for part in plan['parts']]
        with self.assertRaisesRegex(ValueError,'account for'):
            validate_plan(plan)
        plan['subjects'][0]['omitted_roles']={'lower_body':'Cropped below waist'}
        self.assertTrue(validate_plan(plan))

    def test_outline_and_fill_cannot_reorder_parts(self):
        plan=self.human_plan();plan['fill_order']=list(reversed(plan['fill_order']))
        with self.assertRaisesRegex(ValueError,'same planned part order'):
            validate_plan(plan)
        plan=self.human_plan();plan['parts'][0],plan['parts'][1]=plan['parts'][1],plan['parts'][0]
        plan['outline_order']=plan['fill_order']=[part['id'] for part in plan['parts']]
        with self.assertRaisesRegex(ValueError,'Human part order'):
            validate_plan(plan)

    def test_post_generation_review_is_bound_to_plan_crop_and_image(self):
        self.choose('inferred');plan=self.human_plan()
        plan['narrative_requirements']=[{'id':'pose','narration_quote':'人物站立。','observable':'人物完整站立'}]
        plan_path=self.root/'plan.json';plan_path.write_text(json.dumps({'illustrations':[plan]}))
        preflight(self.root,plan_path)
        source=self.root/'figure.png';Image.new('RGB',(10,10),'white').save(source)
        review={'style_sha256':verify_style(self.root)['resolved_sha256'],'plan_sha256':digest(plan_path),'illustrations':[dict(plan_id='figure',status='pass',source_sha256=digest(source),crop=None,
                **dict.fromkeys(('anatomy_and_occlusion','part_boundaries','stroke_breaks','outline_then_fill','spatial_order'),'Inspected in fixture'))]}
        review['illustrations'][0]['narrative_checks']=[{'id':'pose','status':'pass','evidence':'Fixture standing pose'}]
        (self.root/'review.json').write_text(json.dumps(review))
        action={'type':'illustration','source':'figure.png','drawing_plan_id':'figure','parts':plan['parts']}
        manifest={'drawing_plan':'plan.json','drawing_review':'review.json','pages':[{'beats':[{'text':'人物站立。','actions':[action]}]}]}
        style=verify_style(self.root)
        self.assertEqual(verify_drawing_plans(self.root,manifest,style)['reviewed_illustrations'],1)
        action['crop']=[0,0,9,9]
        with self.assertRaisesRegex(ValueError,'another image/crop'):
            verify_drawing_plans(self.root,manifest,style)

        action.pop('crop')
        mask=self.root/'mask.png';Image.new('L',(10,10),255).save(mask)
        action['outline_mask']='mask.png'
        review['illustrations'][0]['outline_mask_sha256']=digest(mask)
        (self.root/'review.json').write_text(json.dumps(review))
        verify_drawing_plans(self.root,manifest,style)
        Image.new('L',(10,10),0).save(mask)
        with self.assertRaisesRegex(ValueError,'mask changed'):
            verify_drawing_plans(self.root,manifest,style)
        action.pop('outline_mask')
        plan_path.write_text(plan_path.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'stale'):
            verify_drawing_plans(self.root,manifest,style)

    def test_preflight_rejects_missing_narrative_facts(self):
        self.choose('inferred')
        path=self.root/'plan.json'
        path.write_text(json.dumps({'illustrations':[self.human_plan()]}))
        with self.assertRaisesRegex(ValueError,'before generating'):
            preflight(self.root,path)

    def test_disconnected_components_never_become_one_ink_stroke(self):
        mask=np.zeros((70,100),dtype=bool)
        mask[10:13,4:15]=True;mask[10:13,70:90]=True
        mask[30:55,20:22]=True;mask[40:42,10:40]=True
        paths=stroke_assets.ordered_paths(mask)
        reveal,segments=stroke_assets.time_paths(paths,mask.shape)
        self.assertGreater(len(paths),2)
        self.assertTrue(validate_strokes({'segments':segments}))
        for path in paths:
            coordinates=np.rint(path).astype(int)
            self.assertTrue(mask[coordinates[:,1],coordinates[:,0]].all())
        invalid=copy.deepcopy(segments);invalid[0]['points']=[[4,11],[70,11]]
        with self.assertRaisesRegex(ValueError,'disconnected'):
            validate_strokes({'segments':invalid})

    def test_shared_endpoint_still_has_a_lift(self):
        paths=[np.array([[0.,0.],[1.,0.]]),np.array([[1.,0.],[1.,1.]])]
        _,segments=stroke_assets.time_paths(paths,(3,3))
        self.assertEqual([segment['kind'] for segment in segments],['ink','travel','ink'])
        self.assertGreater(segments[1]['end'],segments[1]['start'])
        self.assertTrue(validate_strokes({'segments':segments}))
        from video_renderer import point_on_asset
        halfway=(segments[1]['start']+segments[1]['end'])/2
        self.assertLess(point_on_asset({'segments':segments},halfway)[1],-9)

    def test_color_only_art_requires_explicit_outline_and_scales_to_requested_box(self):
        image=Image.new('RGBA',(60,80));ImageDraw.Draw(image).rectangle((10,10,50,70),fill='#ef8971')
        source=self.root/'colored.png';image.save(source)
        parts=[{'id':'object','polygon':[[0,0],[1,0],[1,1],[0,1]]}]
        with self.assertRaisesRegex(ValueError,'no drawable outline'):
            stroke_assets.build_illustration('missing',source,120,160,parts=parts)
        mask=Image.new('L',image.size);ImageDraw.Draw(mask).rectangle((10,10,50,70),outline=255,width=2)
        mask_path=self.root/'outline.png';mask.save(mask_path)
        metadata=stroke_assets.build_illustration('outlined',source,120,160,parts=parts,outline_mask_path=mask_path)
        self.assertEqual((metadata['width'],metadata['height']),(120,160))
        self.assertTrue(validate_strokes(metadata))


if __name__=='__main__':
    unittest.main()
