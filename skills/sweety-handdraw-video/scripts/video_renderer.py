"""Render measured story timelines with path-based text, artwork and marker motion."""
from pathlib import Path
import bisect
import hashlib
import json
import math
import shutil
import subprocess
import wave
import numpy as np
from PIL import Image, ImageColor, ImageFilter
from watermark_assets import planned_watermark, watermark_position
from motion_profile import distance_fraction
import illustration_layers


ENTER_SECONDS, EXIT_SECONDS = .45, .5
BLUR_PX = 18.
TRANSITION_EFFECTS = ('wipe', 'zoom', 'erase', 'page_turn', 'slide_over', 'marker_wipe', 'ink_bleed', 'iris', 'card_flip', 'zoom_through', 'whip', 'push',
                      'paper_tear', 'brush_reveal', 'cube', 'shards', 'tiles', 'flash')  # camera travel per frame (screen px) above which a frame is exposed three times and blended
STAY_SECONDS = 1.  # shorter pen-up gaps keep the hand in frame
GESTURE_LIFT = 7.  # px the nib hovers above the paper while it points at an overlay
ZOOM_PARALLAX = .5  # a push-in spreads a layer at depth 1 by zoom**0.5 more than the page (its position, not its size)
MAX_TILT = 7  # degrees of wrist turn
SHADOW_OPACITY = .16
SHADOW_DIRECTION = np.array([.555, .832])  # shadows fall down and to the right (light from the upper left)
PEN_SHADOW_SLOPE, PEN_SHADOW_OPACITY = .2, .22  # a bare pen's shadow drifts this far per pixel of barrel from the nib


def ease(fraction):
    fraction=float(np.clip(fraction,0,1))
    return fraction*fraction*(3-2*fraction)


def lifted_move(origin,destination,fraction,height=10.):
    """Pen-up move: brief dwell at both ends, min-jerk glide and a small arc lift. Returns (point, lift)."""
    moving=float(np.clip((fraction-.15)/.75,0,1))
    lift=height*math.sin(math.pi*moving)
    return origin+distance_fraction(moving,'minjerk')*(destination-origin)+[0,-lift],lift


def asset_tip(metadata,progress,starts=None):
    """Pen tip on an asset at normalized progress, plus how far the pen is lifted off the paper."""
    segments=metadata['segments']
    if not segments:  # a mark too small to trace: the pen touches its centre
        return np.array([metadata.get('width',0)/2,metadata.get('height',0)/2],dtype=float),0.
    starts=starts if starts is not None else [segment['start'] for segment in segments]
    index=bisect.bisect_right(starts,progress)-1
    if index<0:
        return np.array(segments[0]['points'][0],dtype=float),0.
    segment=segments[index]
    points=np.array(segment['points'],dtype=float)
    if progress>segment['end']:
        if index+1==len(segments):
            return points[-1],0.
        following=segments[index+1]
        fraction=(progress-segment['end'])/max(1e-7,following['start']-segment['end'])
        return lifted_move(points[-1],np.array(following['points'][0],dtype=float),fraction)
    fraction=(progress-segment['start'])/max(1e-7,segment['end']-segment['start'])
    if segment['kind'] in ('travel', 'pause'):
        return lifted_move(points[0],points[-1],fraction)
    distances=np.array(segment['lengths'])
    target=distance_fraction(fraction,segment.get('profile','linear'))*distances[-1]
    return np.array([np.interp(target,distances,points[:,0]),np.interp(target,distances,points[:,1])]),0.


def point_on_asset(metadata,progress):
    return asset_tip(metadata,progress)[0]


ERASER_SIZE = (.028, .058)  # block eraser width and length, as shares of the frame height
ERASER_TILT = 28.           # degrees the top leans right, as a right hand holds it to rub


def eraser_sprite(height):
    """(image, contact point, shadow) of a plain block eraser in a paper sleeve, tilted for rubbing; the contact point is the
    middle of its rubbing end. Drawn at 4x and reduced."""
    from PIL import ImageDraw
    width, length = (max(8, round(height * share * 4)) for share in ERASER_SIZE)
    pad = width
    canvas = Image.new('RGBA', (width + 2 * pad, length + 2 * pad), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    body = [pad, pad, pad + width, pad + length]
    draw.rounded_rectangle(body, radius=width // 5, fill=(247, 244, 238, 255), outline=(150, 146, 140, 255), width=max(2, width // 18))
    sleeve = [pad, pad, pad + width, pad + round(length * .58)]
    draw.rectangle(sleeve, fill=(52, 104, 178, 255), outline=(36, 74, 130, 255), width=max(2, width // 18))
    band = pad + round(length * .5)
    draw.rectangle([pad, band, pad + width, band + max(2, length // 28)], fill=(240, 240, 236, 255))
    image = canvas.rotate(-ERASER_TILT, Image.Resampling.BICUBIC, expand=True)
    turn = math.radians(ERASER_TILT)
    center = np.array(canvas.size) / 2
    tip = np.array([pad + width / 2, pad + length]) - center
    contact = np.array([tip[0] * math.cos(turn) - tip[1] * math.sin(turn), tip[0] * math.sin(turn) + tip[1] * math.cos(turn)]) + np.array(image.size) / 2
    image = image.resize((max(1, image.width // 4), max(1, image.height // 4)), Image.Resampling.LANCZOS)
    shadow = Image.new('RGBA', image.size, (24, 22, 20, 0))
    shadow.putalpha(image.getchannel('A').filter(ImageFilter.GaussianBlur(max(1, height * .003))).point(lambda value: round(value * SHADOW_OPACITY)))
    return image, contact / 4, shadow


def bezier(origin,control,destination,fraction):
    return (1-fraction)**2*origin+2*(1-fraction)*fraction*control+fraction**2*destination


class LazyAssets(dict):
    """Asset images and time maps load on first use: a render chunk touches only the pages it shows."""
    def __init__(self, loader):
        super().__init__()
        self.loader = loader

    def __missing__(self, name):
        value = self[name] = self.loader(name)
        return value


class Renderer:
    def __init__(self, job_directory):
        self.root=Path(job_directory).resolve()
        self.work=self.root/'work'
        self.output=self.root/'outputs'
        self.timeline=json.loads((self.work/'timeline.json').read_text())
        self.width=self.timeline['canvas']['width']
        self.height=self.timeline['canvas']['height']
        self.fps=self.timeline['canvas']['fps']
        self.paper=ImageColor.getrgb(self.timeline['paper'])+(255,)
        self.starts=[page['start'] for page in self.timeline['pages']]
        assets=self.work/'assets'
        self.artwork=LazyAssets(lambda name:Image.open(assets/f'{name}.png').convert('RGBA'))
        self.pixels=LazyAssets(lambda name:np.array(self.artwork[name]))
        self.maps=LazyAssets(lambda name:np.load(assets/f'{name}.npz'))
        self.layered={name for name in self.timeline['assets'] if (assets/f'{name}-line.png').exists()}
        self.layers=LazyAssets(lambda name:illustration_layers.load(assets,name))
        self.cover=Image.open(self.output/'cover.png').convert('RGBA')
        self.backgrounds={}
        for page in self.timeline['pages']:
            if page.get('background'):
                size=self.board_size(page['index'])
                self.backgrounds[page['index']]=Image.open(page['background']).convert('RGBA').resize(size,Image.Resampling.LANCZOS)
        hand=Image.open(self.timeline['hand']['source']).convert('RGBA')
        scale=self.timeline['hand']['scale']
        self.hand=hand.resize((round(hand.width*scale),round(hand.height*scale)),Image.Resampling.LANCZOS)
        self.nib=np.array(self.timeline['hand']['nib'])*scale
        self.watermark = Image.open(self.timeline['watermark']['source']).convert('RGBA') if self.timeline.get('watermark') else None
        plan=self.work/'watermark-plan.json'
        self.watermark_plan=json.loads(plan.read_text()) if self.watermark is not None and plan.is_file() else None

    def scale(self,index):
        """Board resolution of a page relative to the screen: >1 when its camera zooms in, so a close-up stays sharp."""
        return float(self.timeline['pages'][index].get('scale',1.))

    def board_size(self,index):
        scale=self.scale(index)
        return (round(self.width*scale),round(self.height*scale))

    def layer(self,event,time):
        name=event['asset']
        if event.get('caption') or time>=event['end']:
            return self.artwork[name]
        progress=float(np.clip((time-event['start'])/(event['end']-event['start']),0,1))
        metadata=self.timeline['assets'][name]
        if metadata.get('type')=='counter':
            return self.counter_frame(metadata,progress)
        if name in self.layered:
            return Image.fromarray(illustration_layers.compose(self.layers[name],progress))
        maps=self.maps[name]
        pixels=self.pixels[name].copy()
        if 'ink' in maps:
            ink_fraction=np.clip((progress-maps['ink'])*250,0,1)
            color_fraction=np.clip((progress-maps['color'])*250,0,1)
            fraction=maps['ink_weight']*ink_fraction+(1-maps['ink_weight'])*color_fraction
        else:
            fraction=np.clip((progress-maps['reveal'])*250,0,1)
        pixels[:,:,3]=np.rint(pixels[:,:,3]*fraction).astype(np.uint8)
        return Image.fromarray(pixels)

    def counter_frame(self,metadata,progress):
        """Typeset value of a counting number at this point of its count (eased, so it settles on the final value)."""
        spec=metadata['counter']
        fonts=self.__dict__.setdefault('counter_fonts',{})
        key=(spec['font']['path'],spec['font']['index'],spec['size'])
        if key not in fonts:
            from PIL import ImageFont
            fonts[key]=ImageFont.truetype(spec['font']['path'],spec['size'],index=spec['font']['index'])
        value=spec['from']+(spec['to']-spec['from'])*float(distance_fraction(progress,'minjerk'))
        text=f"{spec['prefix']}{value:,.{spec['decimals']}f}{spec['suffix']}"
        image=Image.new('RGBA',(metadata['width'],metadata['height']))
        from PIL import ImageDraw
        ImageDraw.Draw(image).text(tuple(spec['origin']),text,font=fonts[key],fill=spec['color'])
        return image

    def event_state(self,event,progress):
        if event.get('gesture'):
            # Glide along the gesture path, nib hovering just above the paper (it points, it does not ink).
            points=np.asarray(event['points'],float)
            if len(points)<2:
                return points[0],GESTURE_LIFT
            arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
            along=float(distance_fraction(float(np.clip(progress,0,1)),'minjerk'))*arc[-1]
            return np.array([np.interp(along,arc,points[:,0]),np.interp(along,arc,points[:,1])]),GESTURE_LIFT
        cache=self.__dict__.setdefault('segment_starts',{})
        metadata=self.timeline['assets'][event['asset']]
        if event['asset'] not in cache:
            cache[event['asset']]=[segment['start'] for segment in metadata['segments']]
        point,lift=asset_tip(metadata,progress,cache[event['asset']])
        # Screen units: an asset of a zoomed page is drawn at its board resolution (event scale).
        scale=float(event.get('scale',1.))
        point=np.array([event['x'],event['y']])+point/scale
        page=self.warp_pages().get(id(event)) if event.get('depth') or event.get('visual') else None
        if page is not None:
            shift,factor,pivot=self.warp(event,page,event['start']+float(np.clip(progress,0,1))*(event['end']-event['start']))
            point=pivot+(point-pivot)*factor+shift
        return point,lift/scale

    def event_tip(self,event,progress):
        return self.event_state(event,progress)[0]

    def offstage(self,tip,fraction,leaving):
        """Arc between the tip and a rest point below the frame, bowing up and out like a withdrawing arm."""
        drop=self.height+self.nib[1]+self.hand.width*math.sin(math.radians(MAX_TILT))+40-tip[1]
        away=tip+np.array([.55*drop,drop])
        chord=away-tip
        control=(tip+away)/2+.22*np.array([chord[1],-chord[0]])
        moving=float(distance_fraction(fraction,'minjerk'))
        along=moving if leaving else 1-moving
        return bezier(tip,control,away,along),22*min(1.,3*along)

    def pen_events(self):
        """Every main drawing event of the film in time order with its page: one physical pen throughout."""
        if 'pen_event_list' not in self.__dict__:
            events=[(event,index) for index,page in enumerate(self.timeline['pages']) for event in page.get('events',[])
                    if not event['caption'] and event.get('pen',True)]
            # Pen gestures: the pen traces an overlay as it appears (no ink), so motion layers have a visible cause.
            events+=[(dict(gesture,caption=False,gesture=True),index) for index,page in enumerate(self.timeline['pages'])
                     for gesture in page.get('gestures',[])]
            self.pen_event_list=sorted(events,key=lambda item:item[0]['start'])
            self.pen_event_starts=[event['start'] for event,_ in self.pen_event_list]
        return self.pen_event_list

    def pen_state(self,time):
        """(tip, lift, page) of the pen, or None while the hand is out of frame."""
        events=self.pen_events()
        index=bisect.bisect_right(self.pen_event_starts,time)-1
        previous=events[index] if index>=0 else None
        following=events[index+1] if index+1<len(events) else None
        if previous and time<=previous[0]['end']:
            event=previous[0]
            return (*self.event_state(event,(time-event['start'])/(event['end']-event['start'])),previous[1])
        leave,enter=EXIT_SECONDS,ENTER_SECONDS
        if previous and following:
            gap=following[0]['start']-previous[0]['end']
            origin,target=self.event_tip(previous[0],1),self.event_tip(following[0],0)
            if gap<=STAY_SECONDS and previous[1]==following[1]:
                point,lift=lifted_move(origin,target,(time-previous[0]['end'])/gap,min(18.,8+.02*np.linalg.norm(target-origin)))
                return point,lift,following[1]
            # Across a page turn the hand always leaves and returns; squeeze both arcs into short gaps.
            squeeze=min(1.,gap/(leave+enter))
            leave,enter=leave*squeeze,enter*squeeze
        elif following and not previous:
            enter=float(np.clip(following[0]['start']-self.voice_start(),.15,enter))
        if previous and time<previous[0]['end']+leave:
            return (*self.offstage(self.event_tip(previous[0],1),(time-previous[0]['end'])/leave,True),previous[1])
        if following and following[0]['start']-time<enter:
            return (*self.offstage(self.event_tip(following[0],0),1-(following[0]['start']-time)/enter,False),following[1])
        return None

    def pen_kind(self,time):
        """Kind of the segment the pen is on now ('erase' while it rubs the construction out), or None."""
        events=self.pen_events()
        index=bisect.bisect_right(self.pen_event_starts,time)-1
        if index<0:
            return None
        event=events[index][0]
        if time>event['end'] or event.get('gesture') or event.get('asset') not in self.timeline['assets']:
            return None
        metadata=self.timeline['assets'][event['asset']]
        segments=metadata.get('segments') or []
        cache=self.__dict__.setdefault('segment_starts',{})
        if event['asset'] not in cache:
            cache[event['asset']]=[segment['start'] for segment in segments]
        progress=(time-event['start'])/max(1e-9,event['end']-event['start'])
        found=max(0,bisect.bisect_right(cache[event['asset']],progress)-1)
        # Moving between strokes the hand already carries the tool of the next stroke.
        while found<len(segments) and segments[found]['kind'] in ('travel','pause','slide'):
            found+=1
        return segments[found]['kind'] if found<len(segments) else None

    def pen(self,page,time):
        state=self.pen_state(time)
        return None if state is None or state[2]!=page['index'] else state[0]

    def sampled_tip(self,time):
        """Quarter-frame cached (tip, page) samples for smoothing wrist rotation and camera follow."""
        cache=self.__dict__.setdefault('tip_cache',{})
        key=round(time*self.fps*4)
        if key not in cache:
            if len(cache)>8192:
                cache.clear()
            state=self.pen_state(key/(self.fps*4))
            cache[key]=None if state is None else (state[0],state[2])
        return cache[key]

    def hand_pose(self,angle,scale=1.):
        """Hand, soft shadow, nib and cut-edge points rotated about the nib; cached per whole degree and board scale."""
        cache=self.__dict__.setdefault('hand_poses',{})
        if scale!=1.:
            if (angle,scale) not in cache:
                hand,shadow,nib,pad,cut=self.hand_pose(angle)
                size=lambda image:(max(1,round(image.width*scale)),max(1,round(image.height*scale)))
                cache[(angle,scale)]=(hand.resize(size(hand),Image.Resampling.LANCZOS),shadow.resize(size(shadow),Image.Resampling.BILINEAR),
                                      nib*scale,round(pad*scale),cut*scale)
            return cache[(angle,scale)]
        if angle not in cache:
            hand=self.hand.rotate(angle,Image.Resampling.BICUBIC,expand=True) if angle else self.hand
            radians=math.radians(angle)
            turn=np.array([[math.cos(radians),-math.sin(radians)],[math.sin(radians),math.cos(radians)]])
            nib=(self.nib-np.array(self.hand.size)/2)@turn+np.array(hand.size)/2
            alpha=np.array(self.hand.getchannel('A'))
            ys,xs=np.nonzero(alpha>128)
            border=(ys==0)|(xs==0)|(ys==alpha.shape[0]-1)|(xs==alpha.shape[1]-1)
            cut=(np.c_[xs[border],ys[border]][::6]-self.nib)@turn
            radius=max(2,round(self.height*.006))
            pad=3*radius
            if self.timeline.get('hand',{}).get('kind')=='pen':
                shadow=self.pen_shadow(hand,nib,radius,pad)
            else:
                shadow=Image.new('RGBA',(hand.width+2*pad,hand.height+2*pad),(24,22,20,0))
                mask=Image.new('L',shadow.size)
                mask.paste(hand.getchannel('A'),(pad,pad))
                shadow.putalpha(mask.filter(ImageFilter.GaussianBlur(radius)).point(lambda value:round(value*SHADOW_OPACITY)))
            cache[angle]=(hand,shadow,nib,pad,cut)
        return cache[angle]

    def pen_shadow(self,pen,nib,radius,pad):
        """A held pen touches the paper only at its nib: the shadow starts there and falls further off as the barrel rises."""
        ys,xs=np.nonzero(np.array(pen.getchannel('A'))>128)
        axis=np.array([xs.mean(),ys.mean()])-nib
        axis/=np.linalg.norm(axis)
        reach=PEN_SHADOW_SLOPE*float(np.max((np.c_[xs,ys]-nib)@axis))
        extra=np.ceil(reach*SHADOW_DIRECTION).astype(int)
        mask=Image.new('L',(pen.width+2*pad+int(extra[0]),pen.height+2*pad+int(extra[1])))
        mask.paste(pen.getchannel('A'),(pad,pad))
        # Shear about the nib: each point moves along the light by its distance up the barrel (inverse map for PIL).
        origin=nib+pad
        inverse=np.linalg.inv(np.eye(2)+PEN_SHADOW_SLOPE*np.outer(SHADOW_DIRECTION,axis))
        shift=origin-inverse@origin
        mask=mask.transform(mask.size,Image.Transform.AFFINE,(*inverse[0],shift[0],*inverse[1],shift[1]),Image.Resampling.BILINEAR)
        shadow=Image.new('RGBA',mask.size,(24,22,20,0))
        shadow.putalpha(mask.filter(ImageFilter.GaussianBlur(radius)).point(lambda value:round(value*PEN_SHADOW_OPACITY)))
        return shadow

    def hand_angle(self,time,point):
        """Small wrist turn following the smoothed horizontal pen velocity, never exposing the arm's cut edge."""
        # Gaussian-weighted slope of x(t): quick pen jumps blend in gradually instead of flipping the wrist.
        offsets=np.arange(-15,16)/self.fps
        samples=[(offset,tip[0][0]) for offset,tip in ((offset,self.sampled_tip(time+offset)) for offset in offsets) if tip is not None]
        tilt=0.
        if len(samples)>=6:
            offset,x=np.array(samples).T
            weight=np.exp(-.5*(offset/.25)**2)
            offset=offset-np.average(offset,weights=weight)
            velocity=np.sum(weight*offset*(x-np.average(x,weights=weight)))/np.sum(weight*offset**2)
            tilt=MAX_TILT*math.tanh(velocity/(.6*self.width))
        angle=-int(round(tilt))
        for candidate in range(angle,0,-1 if angle>0 else 1):
            cut=point+self.hand_pose(candidate)[4]
            if not ((cut[:,0]>=0)&(cut[:,0]<self.width)&(cut[:,1]>=0)&(cut[:,1]<self.height)).any():
                return candidate
        return 0

    def draw_hand(self,frame,time,page=None,scale=1.):
        """Draw the pen if it is in frame (and, when a page is given, only if it belongs to that page).
        On a board drawn at a higher resolution (scale) the hand is drawn at that resolution too."""
        state=self.pen_state(time)
        if state is None or (page is not None and state[2]!=page):
            return
        point,lift,_=state
        if self.pen_kind(time)=='erase':
            # The hand puts the pen down for an eraser while it rubs the construction lines out.
            key=round(self.height*scale)
            sprites=self.__dict__.setdefault('eraser_sprites',{})
            if key not in sprites:
                sprites[key]=eraser_sprite(self.height*scale)
            eraser,contact,eraser_shadow=sprites[key]
            corner=np.rint(point*scale-contact).astype(int)
            frame.alpha_composite(eraser_shadow,tuple(corner+np.rint(np.array([4.,6.])*scale).astype(int)))
            frame.alpha_composite(eraser,tuple(corner))
            return
        hand,shadow,nib,pad,_=self.hand_pose(self.hand_angle(time,point),scale)
        corner=np.rint(point*scale-nib).astype(int)
        # The shadow stays on the paper while the pen lifts, so it separates slightly from the hand; a bare pen's meets its nib.
        resting=0. if self.timeline.get('hand',{}).get('kind')=='pen' else self.height*np.array([.008,.012])
        offset=np.rint((resting+lift*np.array([.5,1.4]))*scale).astype(int)
        frame.alpha_composite(shadow,tuple(corner-pad+offset))
        frame.alpha_composite(hand,tuple(corner))

    def draw_captions(self,frame,index,time):
        for event in sorted(self.timeline['pages'][index]['events'],key=lambda event:(event['z'],event['start'])):
            if event['caption'] and event['start']<=time<event['hide']:
                frame.alpha_composite(self.layer(event,time),(event['x'],event['y']))

    def page_frame(self,index,time,include_hand=True,include_captions=True):
        """The page board at its own resolution (screen size times page scale). Finished elements below the first one
        still drawing are composited once and reused while they do not change."""
        page=self.timeline['pages'][index]
        scale=self.scale(index)
        events=self.__dict__.setdefault('sorted_events',{})
        if index not in events:
            events[index]=sorted((event for event in page['events'] if not event['caption']),key=lambda event:(event['z'],event['start']))
        visible=[event for event in events[index] if event['start']<=time<event['hide']]
        warped=self.warped(index)
        flat=[event for event in visible if id(event) not in warped]
        moving=sorted((event for event in visible if id(event) in warped),key=lambda event:(float(event.get('depth',0.)),event['z'],event['start']))
        behind=any(float(event.get('depth',0.))<0 for event in warped.values())  # far layers go under the page's flat ink
        settled=0
        while settled<len(flat) and time>=flat[settled]['end'] and not self.animated(flat[settled],time):
            settled+=1
        key=tuple(id(event) for event in flat[:settled])
        cache=self.__dict__.setdefault('settled_cache',{})  # one base per page: a transition shows two pages at once
        if cache.get(index,(None,))[0]!=key:
            base=Image.new('RGBA',self.board_size(index),(0,0,0,0) if behind else self.paper)
            if not behind:
                self.draw_ground(base,index)
            for event in flat[:settled]:
                base.alpha_composite(self.layer(event,time),self.anchor(event,scale))
            if len(cache)>4:
                cache.clear()
            cache[index]=(key,base)
        if behind:
            frame=Image.new('RGBA',self.board_size(index),self.paper)
            self.draw_ground(frame,index)
            for event in moving:
                if float(event.get('depth',0.))<0:
                    self.place(frame,event,time,scale,index)
            frame.alpha_composite(cache[index][1])
        else:
            frame=cache[index][1].copy()
        for event in flat[settled:]:
            self.place(frame,event,time,scale)
        for event in moving:
            if float(event.get('depth',0.))>=0 or not behind:
                self.place(frame,event,time,scale,index)
        self.draw_overlays(frame,index,time,'board',scale)
        if include_hand:
            self.draw_hand(frame,time,index,scale)
        if include_captions:
            if scale!=1.:
                frame=self.apply_camera(frame,np.array([self.width/2,self.height/2]),1.,scale)
            self.draw_overlays(frame,index,time,'screen')
            self.draw_captions(frame,index,time)
        return frame

    def draw_ground(self,frame,index):
        background=getattr(self,'backgrounds',{}).get(index)  # renderers built by tests may skip __init__
        if background is not None:
            frame.alpha_composite(background,(0,0))

    def warped(self,index):
        """{id: event} of the page's elements that move against its camera: a parallax depth, or the subject of a dolly zoom."""
        cache=self.__dict__.setdefault('warped_cache',{})
        if index not in cache:
            page=self.timeline['pages'][index]
            subjects={key['subject'] for key in page.get('camera') or () if key.get('subject')}
            cache[index]={id(event):event for event in page.get('events',()) if page.get('camera') and not event['caption']
                          and (float(event.get('depth',0.)) or event.get('visual') in subjects)}
        return cache[index]

    def warp_pages(self):
        if 'warp_page_map' not in self.__dict__:
            self.warp_page_map={key:index for index in range(len(self.timeline.get('pages',()))) for key in self.warped(index)}
        return self.warp_page_map

    def warp(self,event,index,time):
        """(shift, factor, pivot) in screen units of an element that moves against the camera.
        Parallax: a layer at depth d (-1 far .. 1 near) travels (1+d) times as far as the page when the camera pans away
        from the page's opening view, and a push-in spreads near layers out from the frame centre. Dolly zoom: the subject is scaled by factor about its
        centre (pivot) so it keeps its screen size while the camera zooms past it."""
        center,zoom,_,subject,factor=self.camera_track(index,time,steady=True)
        opening=self.__dict__.setdefault('opening_views',{})
        if index not in opening:  # layers sit where the layout put them in the page's opening view
            opening[index]=self.camera_track(index,self.timeline['pages'][index]['start'],steady=True)[:2]
        start,start_zoom=opening[index]
        if 'width' in event:  # a board overlay
            size=np.array([event['width'],event['height']],float)
        else:
            metadata=self.timeline['assets'][event['asset']]
            size=np.array([metadata['width'],metadata['height']],float)/float(event.get('scale',1.))
        pivot=np.array([event['x'],event['y']],float)+size/2
        depth=float(np.clip(event.get('depth',0.),-1,1))
        shift=-depth*(center-start)+(pivot-center)*((zoom/start_zoom)**(ZOOM_PARALLAX*depth)-1)
        return shift,(factor if subject is not None and event.get('visual')==subject else 1.),pivot

    @staticmethod
    def paste(frame,layer,corner):
        """alpha_composite at any integer corner: whatever falls outside the frame is cut off."""
        x,y=int(corner[0]),int(corner[1])
        left,top=max(0,-x),max(0,-y)
        right,bottom=min(layer.width,frame.width-x),min(layer.height,frame.height-y)
        if right>left and bottom>top:
            frame.alpha_composite(layer.crop((left,top,right,bottom)) if (left,top,right,bottom)!=(0,0,layer.width,layer.height) else layer,(x+left,y+top))

    def overlay_frame(self,overlay,time):
        """Current frame of a web motion layer (held on its last frame), with its exit applied; None when hidden."""
        if time<overlay['start']:
            return None
        exit_spec=overlay.get('exit')
        if exit_spec and time>=exit_spec['end']:
            return None
        index=min(overlay['count']-1,int((time-overlay['start'])*self.fps+1e-6))
        cache=self.__dict__.setdefault('overlay_frames',{})
        key=(overlay['frames'],index)
        if key not in cache:
            if len(cache)>240:
                cache.clear()
            cache[key]=Image.open(Path(overlay['frames'])/f'{index:05d}.png').convert('RGBA')
        image=cache[key]
        if exit_spec and time>=exit_spec['start']:
            import motion
            fraction=(time-exit_spec['start'])/max(1e-6,exit_spec['end']-exit_spec['start'])
            total=motion.identity()
            total['opacity']=1-fraction
            if exit_spec['style']=='shrink':
                total['sx']=total['sy']=1-.6*fraction
            image,_=motion.apply(image,total)
            return image,True
        return image,False

    def draw_overlays(self,frame,index,time,space,scale=1.):
        for overlay in sorted(self.timeline['pages'][index].get('overlays',()),key=lambda item:item['z']):
            if overlay['space']!=space:
                continue
            shown=self.overlay_frame(overlay,time)
            if shown is None:
                continue
            image,padded=shown
            x,y=round(overlay['x']*scale),round(overlay['y']*scale)
            if padded:  # motion.apply pads the layer; centre it on the overlay box
                x-=(image.width-round(overlay['width']*overlay['scale']))//2
                y-=(image.height-round(overlay['height']*overlay['scale']))//2
            if space=='board' and overlay.get('depth') and self.timeline['pages'][index].get('camera'):
                shift,_,_=self.warp(overlay,index,time)
                x,y=x+round(shift[0]*scale),y+round(shift[1]*scale)
                self.paste(frame,image,(x,y))
                continue
            frame.alpha_composite(image,(x,y))

    @staticmethod
    def anchor(event,scale):
        return (round(event['x']*scale),round(event['y']*scale))

    def animated(self,event,time):
        """A finished element playing a motion (or looping an idle one) cannot come from the settled cache."""
        import motion
        for effect in event.get('motions',()):
            if effect['type'] in motion.IDLE and time>=effect['start']:
                return True
            if effect['start']<=time<effect['end'] or (effect['type'] in motion.EXIT and time>=effect['start']):
                return True
        return any(effect['start']<=time<effect['end'] for effect in event.get('emphasis',()))

    def rig(self,event,part):
        cache=self.__dict__.setdefault('rigs',{})
        key=(event['asset'],part)
        if key not in cache:
            import motion
            cache[key]=motion.part_rig(self.artwork[event['asset']],np.asarray(self.maps[event['asset']]['owners']),part)
        return cache[key]

    def boil(self,event,variant):
        cache=self.__dict__.setdefault('boils',{})
        if event['asset'] not in cache:
            import motion
            if len(cache)>24:
                cache.clear()
            cache[event['asset']]=motion.boil_variants(self.artwork[event['asset']],1.1*float(event.get('scale',1.)))
        return cache[event['asset']][variant]

    def place(self,frame,event,time,scale,index=None):
        """Composite one element at this time: its drawing progress, motions, and (given its page index) the parallax
        shift or dolly-zoom scale of an element that moves against the camera."""
        layer=self.layer(event,time)
        corner=np.array(self.anchor(event,scale),float)
        motions=event.get('motions') or [dict(effect,type=effect['type']) for effect in event.get('emphasis',())]
        if motions and time>=event['end']:
            import motion
            for effect in motions:
                if effect['type'] in motion.PUPPET:
                    pose=motion.puppet_state(effect,time)
                    if pose:
                        layer=motion.puppet_frame(self.rig(event,effect['part']),*pose,float(event.get('scale',1.)))
            states=[state for state in (motion.state(effect,time) for effect in motions if effect['type'] not in motion.PUPPET) if state]
            if states:
                total=motion.combine(states)
                if 'boil' in total:
                    layer=self.boil(event,total.pop('boil'))
                if not motion.is_identity(total):
                    layer,offset=motion.apply(layer,total,float(event.get('scale',1.)))
                    corner=corner+np.asarray(offset,float)
        if index is not None and id(event) in self.warped(index):
            shift,factor,pivot=self.warp(event,index,time)
            if abs(factor-1)>1e-3:
                size=(max(1,round(layer.width*factor)),max(1,round(layer.height*factor)))
                center=pivot*scale
                corner=center+(corner-center)*np.array([size[0]/layer.width,size[1]/layer.height])
                layer=layer.resize(size,Image.Resampling.LANCZOS if factor<1 else Image.Resampling.BICUBIC)
            self.paste(frame,layer,np.rint(corner+shift*scale))
            return
        frame.alpha_composite(layer,tuple(int(value) for value in np.rint(corner)))

    def camera_keys(self,index):
        cache=self.__dict__.setdefault('camera_cache',{})
        if index not in cache:
            page=self.timeline['pages'][index]
            beats={beat['id']:beat['start']-page['start'] for beat in page['beats']}
            keys=[]
            import motion
            for key in page.get('camera') or []:
                key=motion.camera_key(key)
                at=key.get('at',0)
                keys.append({'at':beats[at[5:]] if isinstance(at,str) else float(at),'zoom':float(key.get('zoom',1)),
                             'center':np.array(key.get('center',[self.width/2,self.height/2]),dtype=float),
                             'seconds':float(key.get('seconds',.8)),'ease':key.get('ease','inout'),'follow':bool(key.get('follow')),
                             'rotate':float(key.get('rotate',0)),'shake':float(key.get('shake',0)),
                             'impact':float(key.get('impact',0)),'hop':float(key.get('hop',0)),'subject':key.get('subject')})
            cache[index]=sorted(keys,key=lambda key:key['at'])
        return cache[index]

    def clamp_view(self,center,zoom):
        if zoom<1:
            return center
        half=np.array([self.width,self.height])/(2*zoom)
        return np.clip(center,half,np.array([self.width,self.height])-half)

    def camera_state(self,index,time):
        """(centre, zoom, rotation degrees) of the page camera: a key at 0 s frames the page from its start, later keys
        glide in with their easing (any motion.ease name; inout = min-jerk). Zoom moves in log space (every doubling
        takes the same time); a hop dips the zoom mid-move; an impact shakes the frame on arrival."""
        return self.camera_track(index,time)[:3]

    def camera_track(self,index,time,steady=False):
        """camera_state plus the dolly zoom: (centre, zoom, rotation, subject visual or None, subject scale factor).
        steady leaves out pen following, hand-held drift and arrival shakes: the path parallax layers move against
        (it never asks where the pen is, so a pen drawing on a parallax layer can follow it)."""
        import motion
        page=self.timeline['pages'][index]
        local=time-page['start']
        center,zoom,rotate,shake=np.array([self.width/2,self.height/2]),1.,0.,0. if steady else float(page.get('handheld',0.))
        subject,factor=None,1.
        jolt=np.zeros(3)
        for key in self.camera_keys(index):
            if local<key['at']:
                break
            target=key['center']
            if key['follow'] and not steady:
                tips=[tip[0] for tip in (self.sampled_tip(time+.15*step) for step in range(-4,3))
                      if tip is not None and tip[1]==index and 0<=tip[0][1]<=self.height]
                target=target+.5*(np.mean(tips,axis=0)-target) if tips else target
            fraction=1. if key['seconds']<=0 or key['at']<=0 else float(np.clip((local-key['at'])/key['seconds'],0,1))
            weight=motion.ease(key['ease'],fraction)
            moved=zoom*(key['zoom']/zoom)**weight
            if key['hop'] and 0<fraction<1:
                moved=max(1.,moved/(1+key['hop']*4*fraction*(1-fraction)))
            if key['subject']:
                # The subject keeps the screen size it had when this key began; a later key eases it back.
                factor=(factor if subject==key['subject'] else 1.)*zoom/moved
                subject=key['subject']
            elif subject is not None:
                factor=factor**(1-weight)
            center,zoom,rotate=center+weight*(target-center),moved,rotate+weight*(key['rotate']-rotate)
            if not steady:
                shake=key['shake'] or shake
                if key['impact']:
                    jolt=np.array(motion.impact_shake(key['impact'],local-key['at']-key['seconds']))
        if shake:
            # Hand-held drift: two incommensurate slow sines per axis, a few pixels, never a jitter.
            center=center+shake*np.array([math.sin(1.3*time)+.5*math.sin(3.7*time+1),math.sin(1.1*time+2)+.5*math.sin(2.9*time)])/1.5
        if jolt.any():
            # A shake needs room to move: zoom in just enough that the frame never shows past the board.
            zoom=max(zoom,1+2.2*float(np.abs(jolt[:2]).max())/min(self.width,self.height))
            center=self.clamp_view(center,zoom)+jolt[:2]/zoom
        return self.clamp_view(center,zoom),zoom,rotate+jolt[2],subject,factor

    def camera(self,index,time):
        """Camera centre and zoom (see camera_state for rotation)."""
        center,zoom,_=self.camera_state(index,time)
        return center,zoom

    def apply_camera(self,frame,center,zoom,scale=1.,rotate=0.):
        """Screen view of a board: center in screen units, the board drawn at scale x screen resolution, so zooming in
        up to the page scale is a sharp crop instead of an enlargement. A rotated camera samples the board through
        one affine map (paper colour outside the board)."""
        if abs(rotate)>1e-3:
            board=frame.convert('RGB')
            if scale/zoom>1.6:  # downsample first so the affine map does not alias
                factor=int(scale/zoom)
                board=board.reduce(factor)
                scale/=factor
            angle=math.radians(rotate)
            cos,sin=math.cos(angle),math.sin(angle)
            k=scale/zoom
            cx,cy=self.width/2,self.height/2
            a,b,d,e=k*cos,k*sin,-k*sin,k*cos
            c=scale*center[0]-a*cx-b*cy
            f=scale*center[1]-d*cx-e*cy
            return board.transform((self.width,self.height),Image.Transform.AFFINE,(a,b,c,d,e,f),Image.Resampling.BICUBIC,
                                   fillcolor=self.paper[:3]).convert('RGBA')
        if abs(zoom-1)<1e-4 and scale==1.:
            return frame
        size=np.array([self.width,self.height])
        box=tuple(np.clip(np.r_[center-size/(2*zoom),center+size/(2*zoom)],0,np.r_[size,size])*scale)
        return frame.convert('RGB').resize((self.width,self.height),Image.Resampling.BICUBIC if zoom>scale else Image.Resampling.LANCZOS,
                                           box=box,reducing_gap=None if zoom>scale else 2.).convert('RGBA')

    def view(self,index,time,include_captions=True):
        if not self.timeline['pages'][index].get('camera') and self.scale(index)==1. and include_captions and not self.foreign_hand(index,time):
            return self.page_frame(index,time)
        board=self.page_frame(index,time,include_captions=False)
        center,zoom,rotate=self.camera_state(index,time)
        # Motion blur on fast camera moves: average three exposures across the frame interval.
        step=1/(3*self.fps)
        before,after=self.camera_state(index,time-step),self.camera_state(index,time+step)
        travel=(np.linalg.norm(after[0]-before[0])*zoom+abs(after[1]-before[1])*self.height+abs(after[2]-before[2])*self.height/60)*1.5
        if travel>BLUR_PX:
            shots=[self.apply_camera(board,*state[:2],self.scale(index),state[2]).convert('RGB') for state in (before,(center,zoom,rotate),after)]
            frame=Image.fromarray(np.mean([np.asarray(shot,np.float32) for shot in shots],axis=0).round().astype(np.uint8)).convert('RGBA')
        else:
            frame=self.apply_camera(board,center,zoom,self.scale(index),rotate)
        self.draw_overlays(frame,index,time,'screen')
        if include_captions:
            if self.foreign_hand(index,time):
                self.draw_hand(frame,time)
            self.draw_captions(frame,index,time)
        return frame

    def foreign_hand(self,index,time):
        state=self.pen_state(time) if self.timeline['pages'][index].get('events') else None
        return state is not None and state[2]!=index

    def transition(self,index):
        """Window of the move from page index-1 into page index; pages without a transition keep the legacy slide."""
        pages=self.timeline['pages']
        if not 0<index<len(pages):
            return None
        boundary=pages[index-1]['end']
        spec=pages[index].get('transition')
        if not spec:
            return {'type':'legacy','start':boundary-.24,'end':boundary}
        if spec['type']=='cut':
            return None
        seconds=float(spec.get('seconds',{'erase':1.2,'page_turn':1.1,'marker_wipe':1.,'ink_bleed':1.,'iris':.9,'card_flip':.9,'zoom_through':1.,
                                           'slide_over':.8,'whip':.45,'paper_tear':1.2,'brush_reveal':1.1,'cube':.9,'shards':1.1,'tiles':1.,
                                           'flash':.4}.get(spec['type'],.6)))
        drawn=[event['start'] for event in pages[index]['events'] if not event['caption']]
        # No ink is laid while the board moves; a leaving or arriving hand rides along with its page.
        room_in=max(0.,min(drawn)-boundary) if drawn else seconds
        room_out=max(0.,boundary-pages[index-1]['last_drawing_end'])
        # An eraser clears the old page before the new one starts: most of its time comes from the old page's hold.
        incoming=min(seconds*(.2 if spec['type']=='erase' else .5),room_in)
        outgoing=min(seconds-incoming,room_out)
        incoming=min(seconds-outgoing,room_in)
        if outgoing+incoming<2/self.fps:
            return None
        window={'type':spec['type'],'start':boundary-outgoing,'end':boundary+incoming,'direction':spec.get('direction','left')}
        window.update({key:spec[key] for key in ('target','color','angle','style') if key in spec})
        return window

    def pan(self,index,window,time,moving):
        """Glide across one continuous board: the next page waits to the right of the current one."""
        offset=round(self.width*1.06)
        start_center,start_zoom=self.camera(index-1,window['start'])
        end_center,end_zoom=self.camera(index,window['end'])
        center=start_center+moving*(end_center+[offset,0]-start_center)
        zoom=(start_zoom+moving*(end_zoom-start_zoom))*(1-.14*math.sin(math.pi*moving))
        board=Image.new('RGB',(offset+self.width,self.height),self.paper[:3])
        board.paste(self.screen_board(index-1,time).convert('RGB'),(0,0))
        board.paste(self.screen_board(index,time).convert('RGB'),(offset,0))
        left,top=center-np.array([self.width,self.height])/(2*zoom)
        return board.transform((self.width,self.height),Image.Transform.AFFINE,(1/zoom,0,left,0,1/zoom,top),
                               Image.Resampling.BILINEAR,fillcolor=self.paper[:3]).convert('RGBA')

    def screen_board(self,index,time,include_hand=True):
        """Whole page at screen resolution, without the camera and captions."""
        frame=self.page_frame(index,time,include_hand,include_captions=False)
        return frame if self.scale(index)==1. else self.apply_camera(frame,np.array([self.width/2,self.height/2]),1.,self.scale(index))

    def blend(self,index,window,time):
        fraction=(time-window['start'])/(window['end']-window['start'])
        if window['type']=='legacy':
            frame=self.page_frame(index-1,time) if self.scale(index-1)==1. else self.view(index-1,time)
            canvas=Image.new('RGBA',(self.width,self.height),self.paper)
            canvas.alpha_composite(frame,(-round(self.width*ease(fraction)),0))
            self.draw_hand(canvas,time,index)
            return canvas
        moving=float(distance_fraction(fraction,'minjerk'))
        if window['type']=='pan':
            frame=self.pan(index,window,time,moving)
        elif window['type'] in TRANSITION_EFFECTS:
            import transitions
            old,new=self.view(index-1,time,False),self.view(index,time,False)
            frame=getattr(transitions,window['type'])(self,old,new,fraction,window)
        else:
            old,new=self.view(index-1,time,False),self.view(index,time,False)
            if window['type']=='fade':
                frame=Image.blend(old,new,moving)
            else:
                frame=Image.new('RGBA',(self.width,self.height),self.paper)
                shift=round(self.width*moving)
                frame.alpha_composite(old,(-shift,0))
                frame.alpha_composite(new,(self.width-shift,0))
        self.draw_captions(frame,index if time>=self.timeline['pages'][index-1]['end'] else index-1,time)
        return frame

    def scene(self,index,time):
        for target in (index+1,index):
            window=self.transition(target)
            if window and window['start']<=time<window['end']:
                return self.blend(target,window,time)
        return self.view(index,time)

    def voice_start(self):
        return float(self.timeline.get('voice_start_seconds',self.timeline['cover_seconds']))

    def under_cover(self,time):
        """What the cover fades into: blank paper, or the first page once the voice has started under the fade."""
        if time>=self.voice_start() and self.timeline.get('pages'):
            return self.scene(0,time).convert('RGBA')
        return Image.new('RGBA',(self.width,self.height),self.paper)

    def frame(self,time):
        if time<self.timeline['cover_seconds']:
            frame=self.cover.copy()
            if time>.5:
                frame=Image.blend(frame,self.under_cover(time),(time-.5)/.5)
            return frame.convert('RGB')
        index=min(len(self.starts)-1,max(0,bisect.bisect_right(self.starts,time)-1))
        page=self.timeline['pages'][index]
        frame=self.scene(index,time)
        remaining=page['end']-time
        fade = self.timeline.get('ending', {}).get('fade_seconds', 0)
        if index == len(self.starts) - 1 and fade > 0 and remaining < fade:
            frame = Image.blend(frame, Image.new('RGBA', frame.size, self.paper), ease(1 - remaining / fade))
        if getattr(self, 'watermark', None) is not None and time >= self.timeline['watermark']['start_seconds']:
            plan=getattr(self,'watermark_plan',None)
            if plan and len(plan['pages'])==len(self.timeline['pages']):
                position,alpha=planned_watermark(plan,self.timeline['watermark'],self.width,self.height,time)
                if alpha>=.999:
                    frame.alpha_composite(self.watermark,position)
                elif alpha>0:
                    sprite=self.watermark.copy()
                    sprite.putalpha(sprite.getchannel('A').point(lambda value:round(value*alpha)))
                    frame.alpha_composite(sprite,position)
            else:
                frame.alpha_composite(self.watermark, watermark_position(self.timeline['watermark'], self.width, self.height, time))
        return frame.convert('RGB')


def mix_audio(root,timeline):
    root=Path(root)/'work'
    with wave.open(str(root/'narration.wav'),'rb') as stream:
        rate=stream.getframerate()
        samples=np.frombuffer(stream.readframes(stream.getnframes()),dtype='<i2')
    policy = timeline.get('audio_policy', {'background_music': False, 'layers': []})
    normalized_path=root/'normalized-narration.wav'
    # 70 Hz high-pass removes rumble/DC; two-pass linear loudnorm applies one measured gain instead of
    # dynamic pumping. Duplicating mono into two full-level channels adds 3 LU to stereo loudness.
    target='highpass=f=70,loudnorm=I=-20:TP=-2:LRA=7'
    analysis=subprocess.run(['ffmpeg','-hide_banner','-nostats','-i',str(root/'narration.wav'),'-af',target+':print_format=json','-f','null','-'],
                            capture_output=True,text=True,check=True).stderr
    measured=json.loads(analysis[analysis.rindex('{'):analysis.rindex('}')+1])
    second=target+(':measured_I={input_i}:measured_TP={input_tp}:measured_LRA={input_lra}:measured_thresh={input_thresh}'
                   ':offset={target_offset}:linear=true:print_format=json').format(**measured)
    applied=subprocess.run(['ffmpeg','-y','-hide_banner','-nostats','-i',str(root/'narration.wav'),'-af',second,
                            '-ar',str(rate),'-ac','1','-c:a','pcm_s16le',str(normalized_path)],capture_output=True,text=True,check=True).stderr
    result=json.loads(applied[applied.rindex('{'):applied.rindex('}')+1])
    layers = policy.get('layers', [])
    if not layers:
        # No implicit soundtrack and no buffered amix/limiter tail on narration-only output.
        subprocess.run(['ffmpeg','-y','-v','error','-i',str(normalized_path),'-af','pan=stereo|c0=c0|c1=c0',
                        '-c:a','pcm_s16le',str(root/'mix.wav')],check=True)
    else:
        from narration_master import sha256
        command=['ffmpeg','-y','-v','error','-i',str(normalized_path)]
        # Keep the same full-level dual-mono narration with or without effects.
        # Implicit mono-to-stereo negotiation otherwise attenuates it by 3 dB.
        filters=['[0:a]pan=stereo|c0=c0|c1=c0[narration]']
        for index, layer in enumerate(layers, 1):
            if sha256(layer['source']) != layer['sha256']:
                raise ValueError('Audio layer changed after build')
            command.extend(['-i',layer['source']])
            filters.append(f"[{index}:a]aformat=channel_layouts=stereo,volume={layer['gain']},adelay={round(layer['start']*1000)}:all=1[layer{index}]")
        inputs='[narration]'+''.join(f'[layer{index}]' for index in range(1,len(layers)+1))
        filters.append(inputs+f'amix=inputs={len(layers)+1}:duration=first:normalize=0,alimiter=limit=0.89:level=false:latency=true[out]')
        command.extend(['-filter_complex',';'.join(filters),'-map','[out]','-ar',str(rate),'-ac','2',str(root/'mix.wav')])
        subprocess.run(command,check=True)
    (root/'audio-mix-report.json').write_text(json.dumps({'background_music':policy.get('background_music',False),
        'layers':layers,'narration_only':not layers,'narration_filter':target,'loudnorm_first_pass':measured,
        'loudnorm_mode':result['normalization_type'],'delivery_notice':policy.get('delivery_notice',
        '默认未添加背景音乐；可自行合成，或在发布平台选择配乐。')},ensure_ascii=False,indent=2))
    with wave.open(str(root/'mix.wav'),'rb') as mixed_stream:
        mixed_seconds=mixed_stream.getnframes()/mixed_stream.getframerate()
    if abs(mixed_seconds-len(samples)/rate)>.01:
        raise ValueError('Audio mixing changed narration duration; refusing to encode a truncated track')


def aigc_arguments(produce_id):
    """Implicit AI-generated-content label (GB 45438 style keys); platforms may strip it, so delivery notes still ask for an explicit declaration."""
    label={'Label':'1','ContentProducer':'sweety-handdraw-video','ProduceID':produce_id,'ReservedCode1':'','ContentPropagator':'','PropagateID':'','ReservedCode2':''}
    return ['-metadata','AIGC='+json.dumps(label),'-movflags','+faststart+use_metadata_tags']


def x264_arguments():
    return ['-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p']


def render_workers():
    """Parallel render processes: HANDDRAW_RENDER_WORKERS, else all but two cores (at most 8)."""
    import os
    configured=os.environ.get('HANDDRAW_RENDER_WORKERS')
    if configured:
        return max(1,int(configured))
    return max(1,min(8,(os.cpu_count() or 2)-2))


def render_chunk(job_directory,path,start,first,count):
    """One worker: frames [first, first+count) of the interval beginning at start, video only."""
    renderer=Renderer(job_directory)
    command=['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{renderer.width}x{renderer.height}','-r',str(renderer.fps),
             '-i','-',*x264_arguments(),'-video_track_timescale',str(renderer.fps),str(path)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE)
    for index in range(first,first+count):
        process.stdin.write(renderer.frame(start+index/renderer.fps).tobytes())
        if (index-first)%(15*renderer.fps)==0:
            print('RENDER',round(start+index/renderer.fps,2),flush=True)
    process.stdin.close()
    if process.wait()!=0:
        raise RuntimeError('ffmpeg failed to encode a render chunk')
    return str(path)


def encode(renderer,path,start=0.,duration=None,workers=None):
    duration=renderer.timeline['duration']-start if duration is None else duration
    if start < 0 or duration <= 0 or start+duration > renderer.timeline['duration']+.001:
        raise ValueError('Preview interval must stay within the narration timeline')
    temporary_path=path.with_name(path.stem+'.rendering'+path.suffix)
    frames=math.ceil(duration*renderer.fps)
    workers=render_workers() if workers is None else workers
    workers=min(workers,max(1,frames//(8*renderer.fps)))  # chunks of at least ~8 s; process start-up is not free
    metadata=['-map_metadata','-1',*aigc_arguments(hashlib.sha256((renderer.root/'job.json').read_bytes()).hexdigest()[:16])]
    # Cut on a whole frame (the last frame keeps its full length, so the stream reports exactly fps) and pad the audio
    # with silence up to that cut: at most one frame past the narration timeline.
    length=str(frames/renderer.fps)
    if workers<=1:
        command=['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{renderer.width}x{renderer.height}','-r',str(renderer.fps),'-i','-','-ss',str(start),'-i',str(renderer.work/'mix.wav'),*x264_arguments(),'-af','apad','-c:a','aac','-b:a','192k','-t',length,*metadata,str(temporary_path)]
        process=subprocess.Popen(command,stdin=subprocess.PIPE)
        for index in range(frames):
            process.stdin.write(renderer.frame(start+index/renderer.fps).tobytes())
            if index%(15*renderer.fps)==0:
                print('RENDER',index/renderer.fps,'/',round(duration,2),flush=True)
        process.stdin.close()
        if process.wait()!=0:
            raise RuntimeError('ffmpeg failed to encode the video')
    else:
        # Frames are independent functions of time, so contiguous chunks render in parallel and join without re-encoding.
        # One timestamp tick per frame: chunk joins cannot round frame durations (which made the stream report 29.9997 fps).
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        chunks=renderer.work/'render-chunks'
        shutil.rmtree(chunks,ignore_errors=True)
        chunks.mkdir(parents=True)
        bounds=np.linspace(0,frames,workers+1).round().astype(int)
        jobs=[(str(renderer.root),chunks/f'chunk-{index:02d}.mp4',start,int(bounds[index]),int(bounds[index+1]-bounds[index])) for index in range(workers)]
        print('RENDER',workers,'workers for',frames,'frames',flush=True)
        with ProcessPoolExecutor(workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            parts=list(pool.map(render_chunk,*zip(*jobs)))
        listing=chunks/'list.txt'
        listing.write_text(''.join(f"file '{Path(part).name}'\n" for part in parts))
        command=['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',str(listing),'-ss',str(start),'-i',str(renderer.work/'mix.wav'),
                 '-map','0:v','-map','1:a','-c:v','copy','-video_track_timescale',str(renderer.fps),'-af','apad','-c:a','aac','-b:a','192k',
                 '-t',length,*metadata,str(temporary_path)]
        subprocess.run(command,check=True)
        shutil.rmtree(chunks,ignore_errors=True)
    temporary_path.replace(path)


def export_storyboard(renderer):
    review=renderer.work/'review'
    review.mkdir(parents=True,exist_ok=True)
    rows=math.ceil((len(renderer.timeline['pages'])+1)/2)
    sheet=Image.new('RGB',(1280,360*rows),renderer.paper[:3])
    frames=[renderer.cover.convert('RGB')]
    for page in renderer.timeline['pages']:
        frame=renderer.screen_board(page['index'],page['speech_end']-.15,False).convert('RGB')
        frame.save(review/f"page-{page['index']:02d}.jpg",quality=94)
        frames.append(frame)
    for index,frame in enumerate(frames):
        frame.thumbnail((640,360))
        sheet.paste(frame,((index%2)*640,(index//2)*360))
    sheet.save(review/'storyboard-contact.jpg',quality=95)


def render_job(job_directory,start=0.,duration=None,preview=False):
    from video_verifier import build_fingerprint, file_hash, verify_assets
    from art_direction import verify_art_direction
    renderer=Renderer(job_directory)
    verify_art_direction(renderer.root,json.loads((renderer.root/'job.json').read_text()))
    verify_assets(renderer.root,renderer.timeline)
    from narration_master import verify_compiled_master
    verify_compiled_master(renderer.root, renderer.timeline)
    from timeline_contract import validate_timeline,validate_cover
    validate_timeline(renderer.timeline)
    validate_cover(renderer)
    export_storyboard(renderer)
    plan=renderer.watermark_plan=None
    if renderer.watermark is not None and renderer.timeline['watermark'].get('position','legacy')!='legacy':
        from watermark_assets import plan_watermark
        plan=renderer.watermark_plan=plan_watermark(renderer)
        print('WATERMARK',plan['status'],'worst corner ink',plan['worst_busy'],[entry['anchor'] for entry in plan['pages']],flush=True)
    else:
        (renderer.work/'watermark-plan.json').unlink(missing_ok=True)
    mix_audio(renderer.root,renderer.timeline)
    path=renderer.work/'review'/'preview.mp4' if preview else renderer.output/'film.mp4'
    if not preview and (start or duration is not None):
        raise ValueError('Use --preview for partial renders; delivery film must include the complete timeline')
    encode(renderer,path,start,duration)
    from subtitle_contract import validate_subtitle_delivery
    validate_subtitle_delivery(path)
    if not preview:
        record={'video_sha256':file_hash(path),'build_fingerprint':build_fingerprint(renderer.root)}
        (renderer.work/'render-record.json').write_text(json.dumps(record,indent=2))
    print(str(path))
    return path
