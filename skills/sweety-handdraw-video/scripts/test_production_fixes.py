import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import wave
import numpy as np
from PIL import Image, ImageDraw
import align
import stroke_assets
from video_job import save_wave
from video_renderer import mix_audio
from video_verifier import match_reveal_to_tip

class NumberComparisonTests(unittest.TestCase):
    def test_mixed_magnitudes_and_decimal_values(self):
        for script, recognition in [('六亿元','6亿元'), ('一点五亿元','1.5亿元'), ('一亿两千三百万元','1亿2300万元'),
                                    ('一万亿元','1万亿元'), ('六亿零五万元','600050000元'), ('负三点五','-3.5'),
                                    ('百分之零点五','0.5%'), ('六亿元','600,000,000元')]:
            with self.subTest(script=script, recognition=recognition):
                self.assertEqual(align.cer(script, recognition), 0.)
    def test_different_values_are_never_equated(self):
        for first, second in [('六亿','六十亿'), ('三点五','三十五'), ('0.05','0.5'), ('-3.5','3.5'), ('两三万','23万'), ('四五十','450')]:
            with self.subTest(first=first, second=second):
                self.assertGreater(align.cer(first, second), 0)
    def test_asr_evidence_keeps_original_spelling(self):
        result=align.covers('预亏六亿元。','预亏6亿元。')
        self.assertEqual(result['status'],'pass')
        self.assertEqual(result['asr_text'],'预亏6亿元。')

class InkCornerTests(unittest.TestCase):
    def test_broad_corners_follow_the_nib_without_losing_source_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);stroke_assets.configure_assets(root,'/dev/null')
            image=Image.new('RGBA',(240,240));draw=ImageDraw.Draw(image)
            draw.line([(25,30),(210,30),(140,120),(210,210),(25,210)], fill='#202020',width=13,joint='curve')
            image.save(root/'source.png')
            mask=image.getchannel('A');mask.save(root/'outline.png')
            for name,texture in (('corners',False),('pressed',True)):
                metadata=stroke_assets.build_illustration(name,root/'source.png',240,240,
                    parts=[{'id':'contour','polygon':[[0,0],[1,0],[1,1],[0,1]]}],outline_mask_path=root/'outline.png',seconds=10,
                    drawing={'texture':texture})
                maps=np.load(root/f'{name}.npz');drawn=np.isfinite(maps['ink'])
                distances,_=match_reveal_to_tip(metadata,maps['ink'],drawn,{'ink'})
                self.assertLess(distances.max(),9.)
                self.assertEqual(metadata['drawability']['drawn_stroke_count'],metadata['drawability']['stroke_count'])
            original=np.array(image)[:,:,3]>0
            final=np.array(Image.open(root/'corners.png'))[:,:,3]>0
            self.assertGreater(np.count_nonzero(original & final)/original.sum(),.995)
            # With pressure ink the two free ends of this open line land and lift thin: source pixels may go only there,
            # within the taper (TAPER_WIDTHS line widths, plus the round tip) of an end; every other one is kept.
            import line_drawing
            pressed=np.array(Image.open(root/'pressed.png'))[:,:,3]>0
            ys,xs=np.nonzero(original)
            reach=(line_drawing.TAPER_WIDTHS+1)*13
            near_end=np.minimum(np.hypot(xs-25,ys-30),np.hypot(xs-25,ys-210))<=reach
            kept=pressed[ys,xs]
            self.assertGreater(kept[~near_end].mean(),.995)
            self.assertGreater(kept[near_end].mean(),.6)

    def test_adjacent_parts_use_their_nib_time_and_owner_together(self):
        points = np.c_[np.arange(15, 50), np.full(35, 20)]
        strokes = [{'points':points, 'times':np.linspace(.1,.3,35), 'part':0},
                   {'points':points + [0,8], 'times':np.linspace(.6,.8,35), 'part':1}]
        pixels=np.zeros((50,60,4),np.uint8);pixels[18:31,15:50]=[20,20,20,255]
        zone=pixels[:,:,3]>0
        _,alpha,times,owners=stroke_assets.source_ink(pixels,np.zeros_like(pixels),zone,zone,strokes)
        self.assertLess(times[21,30],.4)
        self.assertGreater(times[27,30],.5)
        self.assertEqual(owners[21,30],0)
        self.assertEqual(owners[27,30],1)
        self.assertTrue(np.isfinite(times[alpha>0]).all())

class MixLevelTests(unittest.TestCase):
    def test_silent_mono_or_stereo_effect_does_not_change_narration_gain(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'work').mkdir()
            signal=np.zeros(48000*3,dtype=np.int16)
            signal[10000:100000]=(5000*np.sin(np.arange(90000)*2*np.pi*220/48000)).astype(np.int16)
            save_wave(root/'work/narration.wav',signal)
            mix_audio(root,{'duration':3.})
            def read_mix():
                with wave.open(str(root/'work/mix.wav')) as stream:
                    return np.frombuffer(stream.readframes(stream.getnframes()),dtype='<i2').reshape(-1,2).astype(float)
            baseline=read_mix()
            for channels in (1,2):
                effect=root/'silence.wav'
                with wave.open(str(effect),'wb') as stream:
                    stream.setparams((channels,2,48000,0,'NONE','not compressed'))
                    stream.writeframes(np.zeros(24000*channels,dtype='<i2').tobytes())
                layer={'source':str(effect),'sha256':hashlib.sha256(effect.read_bytes()).hexdigest(),'gain':.08,'start':.5}
                mix_audio(root,{'duration':3.,'audio_policy':{'layers':[layer]}})
                mixed=read_mix()
                self.assertEqual(mixed.shape,baseline.shape)
                self.assertLess(np.abs(mixed-baseline).max(),3)
                self.assertLess(np.abs(mixed[-4000:]).max(),3)
    def test_stereo_effect_preserves_channels_without_changing_voice(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'work').mkdir()
            voice=np.zeros(144000,dtype=np.int16);voice[10000:130000]=(4000*np.sin(np.arange(120000)*2*np.pi*220/48000)).astype(np.int16)
            save_wave(root/'work/narration.wav',voice)
            effect=root/'left.wav';samples=np.zeros((9600,2),dtype='<i2');samples[:,0]=(2000*np.sin(np.arange(9600)*2*np.pi*660/48000)).astype(np.int16)
            with wave.open(str(effect),'wb') as stream:
                stream.setparams((2,2,48000,0,'NONE','not compressed'));stream.writeframes(samples.tobytes())
            mix_audio(root,{'duration':3.,'audio_policy':{'layers':[{'source':str(effect),'sha256':hashlib.sha256(effect.read_bytes()).hexdigest(),'gain':.2,'start':1.}]}})
            with wave.open(str(root/'work/mix.wav')) as stream: mixed=np.frombuffer(stream.readframes(stream.getnframes()),dtype='<i2').reshape(-1,2).astype(float)
            with wave.open(str(root/'work/normalized-narration.wav')) as stream: normalized=np.frombuffer(stream.readframes(stream.getnframes()),dtype='<i2').astype(float)
            self.assertLess(np.abs(mixed[:,1]-normalized).max(),3)
            self.assertGreater(np.std((mixed[:,0]-mixed[:,1])[48000:57600]),200)
            self.assertLess(np.abs(mixed).max(),32767)

if __name__=='__main__':unittest.main()
