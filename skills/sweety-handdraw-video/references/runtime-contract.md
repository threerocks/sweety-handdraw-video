# 运行格式与命令

任务目录包含 `job.json`、模型编写的 `design.json`、`design/art-direction.json`、`inputs/`、`work/` 和 `outputs/`。`review.json` 由 verify 初始化，执行者实际检查后填写。脚本不代替模型写内容、选路线或设计素材。

## 最小字段示例

以下只是格式说明；须先生成、检查、清理图片，写入有效美术记录并锁定完整旁白和时间戳后才能运行。

```json
{
  "version": 1,
  "title": "本次主题",
  "design_file": "design.json",
  "art_direction": {"record": "design/art-direction.json"},
  "canvas": {"width": 1920, "height": 1080, "fps": 30},
  "cover": {"source": "inputs/cover.png", "seconds": 1.0},
  "hand": {"preset": "navy", "pen_text": ""},
  "voice_studio": {"voice": "magnetic-narrator"},
  "narration": {"source": "inputs/narration/master.wav", "alignment": "inputs/narration/alignment.json", "say": {"Grok": "格罗克"}},
  "audio": {"end_hold_seconds": 0.45},
  "pages": [{"beats": [{
    "id": "opening",
    "text": "与音频逐字对应的讲解。",
    "actions": [
      {"type": "text", "text": "本段短标题", "x": 120, "y": 60, "size": 80},
      {"type": "illustration", "source": "inputs/object.png", "x": 200, "y": 230,
       "width": 680, "height": 580, "weight": 3,
       "parts": [{"id": "single_object", "polygon": [[0,0],[1,0],[1,1],[0,1]]}]}
    ]
  }]}]
}
```

单一部位示例只适用于一个简单物体。人物必须根据实际图像划出头、颈、躯干、左右臂、下半身等有语义的部位，不能用整图一个矩形敷衍。

## 设计和图片记录

`design.json` 必须有 `narrative_route.decision/reason/director_review`。默认路线 decision 为 `default_life_concept`；其他路线还要有 `director_plan`，记录 omniailab 的具体导演规划。完整内容字段见 [创作与分镜](story-design.md)。

美术记录格式：

```json
{
  "dependency_sha256": "随包美术指导 GUIDE.md 的实际 SHA-256",
  "images": [{
    "source": "../inputs/object.png",
    "sha256": "清理后文件的实际 SHA-256",
    "method": "generated_under_guidance",
    "brief_file": "briefs/object.md",
    "brief_sha256": "出图前简报文件的实际 SHA-256",
    "prompt_file": "prompts/object.txt",
    "prompt_sha256": "本次实际完整提示词文件的 SHA-256",
    "status": "pass",
    "evidence": "实际查看后的主体、接触、构图、边缘和可绘制性判断"
  }]
}
```

所有路径相对记录文件。封面、正文源图、自定义手均要覆盖；包内六款手有自己的记录。合成封面 method 为 `composed_with_verified_layers`，保存可编辑文字图层说明。图集用 `approved_crops` 数组列出已查看的像素裁切，每项是 `[left,top,right,bottom]`；未批准格子不能参与成片。更换图像后旧哈希失效，必须重新检查。提示词文件存在不是“指导已经执行”的充分证据。

## 绘制和时序

- `text`：`text/x/y/size/color/weight`，一行一项。每个字形在独立区域内描线，后字不会提前露出。图层按各字真实边界计算，并保留至少5px空白；禁止用字号估算高度。
- `marker`：`style` 为 `highlight/underline/circle/box`，使用 `width/height/color/line_width`。
- `illustration`：`source/x/y/width/height`，可设像素裁切 `crop`。保持比例缩入框。
- `parts`：有序数组，唯一 `id` 与归一化 `polygon` 坐标，或方框 `"box": [左, 上, 右, 下]`（0—1）。坐标相对裁切后的图。墨线围成的同色色块整块归给占它最多的部位，所以方框通常就够，不必描出形状；`python3 scripts/part_grid.py JOB 图` 输出带 0.05 网格、物体框和色块编号的 `design/parts/<图>.png`/`.json`，照着写方框。区域重叠时先声明者拥有像素；所有可见像素必须有归属。模型看图后确定边界，避免切碎手指、轮廓或道具。轮廓按数组顺序完成全部部位，再按同一顺序填色。笔划间抬笔。
- 前景手臂、持物或其他遮挡可用 `exclude_polygons` 给出归一化排除多边形数组（或 `exclude_boxes` 方框数组）；从头部或躯干排除的像素必须由后续正确部位接管。不要因为手臂横过胸前就让它在躯干阶段提前显现。
- 复杂遮挡可用部位的 `exclude_hsv:[[hue_min,hue_max,saturation_min,value_min]]` 排除透过发丝等空隙可见的背景颜色（色相0—360，饱和度/亮度0—1）。排除的像素必须由后续正确部位接管，需实际看图验证，不能仅以颜色自动推断语义。
- `weight` 只分配对应音频窗口里的绘制时间，不是秒数。插画和按笔顺手写的文字画到视频节奏（插画约3倍人速，即 `max(minimum_seconds, required_seconds/3)`；文字为自然速度的1/1.8）就停，窗口余下的时间留给观看，下一个动作仍按原窗口起笔。所有元素可设 `z`；底纹默认位于文字下面。
- 编号区域（可替代多边形）：`python3 scripts/region_marks.py JOB <drawing_plan_id>` 把线稿的闭合区域（大区域内再按颜色细分）编号，写出 `design/regions/<id>.png` 叠图、`.npz` 区域图（绑定源图哈希）和 `.json`。看叠图后在 action 写 `"region_map":"design/regions/<id>.npz","part_regions":[{"id":"head","regions":[3,4,7]},...]`，顺序即绘制顺序、id 与计划 parts 一致。每个编号只能属于一个部位；漏掉的可见区域会报错并列出编号；换图后须重跑。线稿在部位交界处不闭合（叶基通向茎、衣摆通向裤腿）时一个编号会跨两个部位，此时改用 `parts` 多边形。部位可加 `"key": true`，与 id 含 head/face/eye/mouth/hand/paw 的部位一样优先保留真实描线。
- 线稿门槛：`python3 scripts/lineart_gate.py SOURCE --width W --height H [--crop ...] [--outline-mask ...]` 测交叉点密度、小闭合孔、碎点和短纹理线占比；构建时同样检查。纹理型线稿（如边缘检测出的布纹）无法一笔一笔画，build 报错。按[可绘制约定](style-and-drawing.md)简化或重新出图。`lineart_exception` 只记录有意保留的纹理，不是出口：笔仍要一笔一笔画完每条短线，通常会超出窗口。
- 绘制方式：`"drawing": {"mode": "draw" | "reveal"}` 写在 job 顶层，须与 `style/resolved.json` 的 `drawing_mode` 一致；逐笔绘制的片子里单个插画动作不能改用 `reveal`。揭示只由用户选，见[绘制方式](style-and-drawing.md)。`reveal` 不提取线稿、不过线稿门槛，按部位用斜向软笔刷显出成图，元数据 `mode: "reveal"`，`drawability.minimum_seconds` 很小。见[绘制方式](style-and-drawing.md)。
- 出图检查：`python3 scripts/image_gate.py JOB 图 [--visual id] [--width W]` 在生成后几秒内测透明窟窿、光晕、线稿门槛和规划尺寸下的笔画数与最短秒数，对照故事板给这个视觉的秒数；不过时退出码为 1，并给出 `prompt_fix`（加进提示词的一句话，复杂度超标时写明目标笔画数）。它不改图也不改记录。
- 规划尺寸：笔画、顺序和时间在长边不超过 720 px 的规划尺寸上计算（`drawing.plan_side`，360—1600），更大的显示尺寸由同一份规划按显示分辨率重新渲染清晰版。所以 `minimum_seconds` 不随显示尺寸增大，图可以按版式需要画大。`plan_side` 作用在版式框上：裁切让图的一部分在同一个框里显示得更大，所以按更高的密度规划，笔画更多（阿姨片开场图只裁左右，最短绘制时间就从 7.9 s 变成 9.1 s），但线条也更不容易挤在一起、过不了线稿检查（阿姨片三张裁切图不裁就过不了）。需要省时间就调低 `plan_side`，别靠取消裁切。
- 素材缓存：每个绘制元素按内容哈希（动作去掉位置类字段、窗口秒数、画板倍数、输入文件哈希、字体和引擎代码）缓存在 `work/asset-cache/`。只改 `x/y/z/visual/weight/from_beat/motion/pen/depth`、镜头、转场或动效层时，重建直接复用，几秒完成；改了引擎代码、图片或绘制参数会自动重画。
- `drawing`：job 顶层或 action 内 `{"speed":1.0,"sketch":false}`，action 覆盖顶层。
  - `speed`（0.25—4）缩放人手基准速度及其上限：描线 4 倍、上色 6 倍人速；描线笔尖每帧不超过约 80px，上色宽笔每帧不超过约 120px。
  - `sketch:true` 先画淡色构图线，描线后淡出。`sketch` 也可以是模型规划的起形列表（ellipse/box/line/polygon，见[拟人绘制步骤](style-and-drawing.md)）：在墨线前画出，墨线后由橡皮擦掉（`erase` 笔段），再上色。
  - `colouring`：`"markers"`（默认，按马克笔：同色一起涂，先浅后深，每种颜色涂完底色再压阴影）或 `"parts"`（按模块）。
- 插画引擎（draw-v3）：只有真实墨线才描线——比旁边填充更暗、又接近整张图墨色的细线；深色填充（卫衣、头发、夜空、青色屏幕）的边缘带属于上色，黑色剪影只描细边缘作轮廓；墨色与底色太接近处判定为"墨"或"底色"，不出半透明噪点；离线条很远、只有几个像素的孤立墨点留给上色。轮廓经清理、骨架化后沿切线连续穿过交叉点成整笔，每笔整体属于一个部位。**笔顺按对象规划**：部位内先按图像实心区块分"对象"（眼睛属于它所在的脸），一个对象画完再画下一个，对象按阅读顺序（先上后左）、大的优先并兼顾笔尖距离；对象内先外轮廓，再画从已画线条上"长出来"的线，就近、先长后短、尽量不回头往上；下一笔起点紧贴刚画完的线时笔不抬起、直接滑过去。每笔 min-jerk 速度、转弯处放慢；时间紧时较长的笔改用两端缓起缓收、中段匀速的平台曲线。人速时每笔、每次抬笔至少2帧，延时绘制（快于人速1.5倍以上）时短笔和抬笔可为1帧；空中移动每帧不超过约120px。**时长跟随规划图幅而非屏幕**：以400px为基准，速度、每帧位移上限和马克笔宽度随图幅放大（最多1.8倍），换手长度、色块拆分阈值同样随图幅缩放；draw-v3 实测约每秒5–6笔，v3（加起形、擦除、按马克笔上色）约每秒3.9笔，都与图幅基本无关。再小的色块也有自己的一笔马克笔，不会随部位整体显出。上色按部位、从离马克笔最近的色块开始就近推进，用中位色以连续之字形大笔涂（区域内不抬笔），再用短暂的阴影笔从平涂过渡到原图。笔画只决定每个像素何时出现：线稿图层用原图自己的墨线像素（压在色块上的墨线先从底色中分离出来），每个像素在笔尖经过它最近的一笔时出现，不另画一套带笔压的线，终帧与原图一致；若成图出现断线、毛边，先查引擎，不要先怪原图。资产为 `{name}.png` 终帧加 `-line/-paint/-shade(-sketch).png` 图层和 `.npz` 时间图，逐帧合成，终帧与最后一帧一致。
- 时长：元数据 `drawability.required_seconds` 是全部按人速画完所需，`minimum_seconds` 是全部线条（含细节和纹理线）按上限速度描完、再用最宽笔快速上色所需。窗口低于 `minimum_seconds` 时故事板检查报错，只能简化画面或延长窗口；介于两者之间时笔加速、顺序改为就近，但每一笔都画出来，没有淡入或整块显现（`drawn_stroke_count` 等于 `stroke_count`）。`python3 scripts/engine_metrics.py work/assets element-0007` 输出笔画长度、抬笔路径、跳笔、反向、速度变化和“首次可见即终色”等客观指标。

汉字按规范笔顺手写：字形与笔画中线来自 Make Me a Hanzi（`assets/hanzi`，Arphic Public License），笔尖沿中线移动，笔画按经过的位置显现，逐笔抬笔，逐字从左到右；字形按所选字体的汉字高度和中心对齐，并略加粗以接近马克笔。包内含 GB2312 全部6763字；其他字首次使用时从 jsDelivr 固定版本下载到素材缓存，离线或无数据不报错。只要一行中有一个汉字缺数据，整行改用字体骨架描线（保持同一观感），资产元数据 `handwriting.mode/reason` 记录原因。拉丁字母、数字和标点沿字体路径，横向笔画从左到右、其余从上到下。每笔至少2帧、限速；`handwriting.natural_seconds/minimum_seconds` 给出自然与最低时长，分到的时间不足时整体压缩并在 build 时提示。笔顺要看得清，标题每字约需0.3—0.6秒，用 `weight` 给足时间。

跨多句的一个完整画面可以使用页面 `visual_groups`，每组为 `{"start_beat":0,"end_beat":2,"lead":0.54,"hold":0.72,"actions":[...]}`，索引从0开始。此时 beats 中的 actions 可以为空或省略。起止索引必须有序且在该页范围内，lead 非负，hold 至少0.45秒；多个组不能造成主笔事件重叠。不要给每句机械安排一幅新图。一个画面讲多句时，按句拆成多个动作并用 `from_beat` 让每个动作随它那句起笔，不要让一个动作铺满整组。

生产任务必须使用 `narration.source/alignment` 指定完整旁白母带和对应时间戳，字段见[连续旁白约定](continuous-narration.md)。`audio.end_hold_seconds` 只控制整篇声音结束后的画面停留；生产任务拒绝非零 `sentence_pause_seconds/page_pause_seconds`、`pace` 非1和 `trim_silence:true`。绘制窗口太短则简化画面或合并绘制窗口，不能截短呼吸、插入空白或拉慢旁白来容纳画面。封面只接受1秒，前半秒静止、后半秒渐隐进第一页；旁白和正文从 `1 - cover.voice_overlap` 秒开始（默认0.5秒，范围0–0.5，测试夹具默认0），时间线记 `voice_start_seconds`。

## 镜头与转场

### 开场的编排

按[黄金5秒开场](opening-design.md)把 `design.json.opening_hook` 落到现有动作与时间线；该字段是导演设计记录，不是引擎会自动执行的新参数。`alignment` 的母带时间加上1秒封面才是成片时间，页面 `camera.at` 仍从本页开始计时，不能混用。

封面用短标题和可辨认的主视觉承载问题、反差或真实结果；正文第一笔优先画与首句有关的简单对象或关系，避免先写长标题、铺边框或画完整背景。按 `visual_groups`、动作顺序、`from_beat` 和预算安排后续细节；`weight` 不是指定秒数，须用构建后的事件起止与实际预览核对关键画面何时可读。小于5秒的开场窗口不代表整幅复杂插画必须画完，不能为卡点超速、跳笔或改用揭示。

本版封面固定1秒，正文与旁白从1秒后开始；不要在 JSON 中臆造封面旁白、预先完成正文插画等能力。需要结果前置时，封面可以展示经核实的结果主视觉，正文依然按绘制约定形成画面。先运行下方15秒预览命令并检查0–5秒和正文衔接；同一段足够覆盖代表段要求时可合并检查。

### 页面设置

页面可加 `camera` 关键帧数组，在整张画面上推近、移动：`{"at": 秒数或 "beat:<本页beat id>", "center": [x, y], "zoom": 1.0—3.0, "move": 命名运镜, "seconds": 0—3, "ease": 缓动名, "follow": false, "impact": 0—40, "hop": 0—0.6, "rotate", "shake", "subject": 视觉 id}`。命名运镜有 cut、push_in、pull_out、pan、drift、hop、whip、snap、slam、dolly_zoom，给出默认时长与缓动；缩放在对数空间插值。`dolly_zoom` 必须写 `subject`（本页某个元素的 `visual`），`at` 不能为 0：主体保持这一刻的屏幕大小，画板照常推拉，下一个关键帧再把主体缓回原大。元素和 `space: board` 的动效层可设视差 `depth`（-1 远 … 1 近，0 是画板本身）：远层画在本页墨线下面，镜头离开开页取景时，各层按 (1+depth) 倍的距离移动，推近时近层向外散开；笔尖随所画的那一层移动。**大画板**：本页素材按本页最大 zoom 的倍数生成（`board.scale` 可显式指定，1—3），推近到该倍数仍是清晰的原尺寸画面，不是放大糊图。坐标仍按屏幕像素写，zoom 1 时整块画板缩进屏幕。典型用法：先以 1.6—2 倍逐个推近讲解，最后 `pull_out` 到 zoom 1，让观众看到整块"壁画"。`at` 从本页开始计时，是移动开始时刻；`at` 为 0 的关键帧直接作为开页取景。取景永远不出画布。`follow:true` 时镜头在该缩放内温和跟随笔尖。字幕和水印属于屏幕层，不随镜头缩放。

**取景检查**（`scripts/framing.py`，`build` 自动运行）：
- **检查什么。** 每个正在画或写的元素，在绘制窗口内每 0.25 秒检查一次。`space: board` 的动效层检查它的出场过程。
  - 文字、线条、标记：整块要在画面内，离画面边和字幕带至少 24 px（画面边已到画布边时不要求）。
  - 插画：笔尖要在画面内。
  - 动效层：整个框要在画面内。
- **怎么修。** 不在画面内时，依次尝试下面几种修法，选出整页效果最好的那个。先比笔在画面外写画的量，再比动效在画面外的量。
  - 镜头等笔：画到一半就开始的镜头，推迟到这一笔画完。开页取景（`at` 为 0）不推迟。后续镜头保留原来对应的台词时间，只在前一个镜头还没走完时顺延；页尾放不下的镜头去掉，最后一个保留。
  - 插入取景关键帧：在元素开始前 0.5 秒插入，带 `"auto": "framing"`，只有 build 会写这个字段。取法有三种：
    - 平移一点；
    - 把元素居中；
    - 拉远到同时装下原画面和元素。

    随后 3 秒内开始画的元素，只要缩放不变也能装下，就一起框进来。
- **修不了时。** 哪种修法都不能让整页变好（比如在字幕下、在画布外、在 `dolly_zoom` 中），就保留原镜头，在 `build` 警告里写明哪一页、哪个元素、偏出多少像素。结果记录在 `timeline.framing`。

页面可加 `transition` 描述进入本页的方式，首页不能设置：`{"type": …, "seconds": …, "direction": "left"|"right"|"up"|"down", "style": …, "target": [x, y], "color": "#hex", "angle": 度}`。可用类型分五类：轻转场 cut、fade、slide、push、pan、wipe、zoom；纸张类 page_turn（四个方向）、slide_over、paper_tear、card_flip、cube、tiles、shards；擦除类 erase（style 为 felt 或 rubber）；墨与笔类 brush_reveal、marker_wipe、ink_bleed、iris（style 为 round 或 blot）；镜头和点缀类 zoom_through、whip、flash。转场只放在上一页最后一笔之后、本页第一笔之前，空间不够时自动缩短；正在离场或入场的手跟随所属页面移动。需要完整转场时，给本页 visual group 的 `lead` 和上一页的 `hold` 各留至少一半转场时长。每张卡的用途、参数和"不要用于"见[运镜、转场与动效目录](motion-catalog.md)；`build` 会按密度规则提示镜头过度和转场重复。

`job.direction` 控制自动导演（`{"camera", "transitions", "motion": "auto"|"off", "boil": true}`，默认全 auto）：它只补模型没写的镜头、转场和画完后的轻动效，模型写了的一律保留。

页面可加 `overlays` 网页动效层：大字、手绘图表、地图路线、形状接力、物理道具（`physics`）、转动的立体物件（`spin3d`）、手写公式（`formula`）、标注、气泡、滚动数字等，字段与配方见同一目录。`space: board` 的动效层出现时，笔尖沿它描一遍（悬在纸面上方、不出墨，笔正忙时跳过；`"pen": false` 关闭）。屏幕上同一位置的两条 `kinetic_title` 会轮流出场：后一条出现前，前一条先淡出，不会被连读成一句。写了 `visual` 的动效层在故事板检查里算作那个视觉。需要先运行 `bash scripts/setup.sh --web-motion`。

全片只有一支笔：同页相邻两笔间隔不超过1秒时手留在画面内，抬笔、停顿后以最小加加速度轨迹滑到下一笔；更长的停顿或换页时，手沿弧线从右下方离开、再沿弧线回到下一笔起点。手腕随平滑后的横向笔速最多转动约7°（绕笔尖旋转；会让前臂裁切边入画的转动自动收回），并带随抬笔轻微分离的柔和投影；`pen-only` 的投影从笔尖起、沿笔杆逐渐偏离，表示笔斜立在纸上。同组动作之间只隔约0.08秒，相距很远的两个动作之间笔仍会移动很快，排版时让相邻动作就近。

### 强调动作、图表、箭头与数字

- 任一元素可加 `motion: [{"type": …, "at": "beat:<本页beat id>" 或秒数, "seconds": 0.2—8, "strength": 1, "part": 部位 id}]`（旧字段 `emphasis` 仍可用）。类型分四族：强调、待机循环、退场、部位木偶，完整列表见[动效目录](motion-catalog.md)。强调和部位动作只在画完之后播放：对应句开始时还没画完，会自动推到最后一笔之后。
- `{"type": "arrow"|"line", "points": [[x, y], ...], "curve": 0, "line_width": 5, "color": "#302C26", "head": "end"|"both"|"none", "pressure": 0.6}`：由笔尖画出的线或箭头，先画杆、再画两侧箭头笔；两点时 `curve` 让它弯曲（正负决定方向）。`pressure`（0—1）是笔压起伏：落笔渐粗、中段运笔最快处略细、收笔变尖，箭头倒钩是一下轻挑；0 是等宽的针管笔线。
- `{"type": "bar_chart", "x", "y", "width", "height", "values": [...], "labels": [...], "unit": "", "color" 或 "colors", "max"}`：构建时展开为"先画坐标轴，每根柱子由马克笔从底部涂上去并写上数值，最后写类别"，各部分按顺序分到时间。只用真实数据，不为好看编造数字。
- `{"type": "counter", "from": 0, "to": 1.4, "decimals": 1, "prefix": "", "suffix": "万+", "x", "y", "size", "color"}`：数字从起点滚到终值后停住；它是排字动画，不经过画笔，可以代替一个手写数字标签。

## 手、字体和字幕

六款手臂及仅画笔的 `hand.preset` 标识见 `assets/hands/catalog.json`。未选时使用模型根据内容准备的默认方案。`pen_text` 最多16个字符，这是格式上限。保留用户原文，按正常视频尺寸检查可读性；必要时调整笔杆排版，不擅自缩写广告名称。`pen_text_color` 可选。自定义手设置 `source/nib/wrist/label_segment/label_height`，提供实测锚点，源图前臂须延伸到底边；运行时自动限制最小尺度，使裁切边留在画外。新手图须同样经过美术指导和隐私清理。

`font` 为 `{"path":"/actual/font.ttc","index":2}`。字幕按[字幕与片尾设计](subtitle-design.md)由模型独立决定。`captions` 支持 `font:{path,index}/font_size/color/bottom_margin/max_width_ratio/max_lines/stroke_width/stroke_color`，以及 `strip_punctuation/keywords/highlight_numbers/highlight_color/max_chars`（beat 级 `caption_keywords`），默认值与竖屏安全区见字幕设计；`background:null` 不生成底板；可选 `background:{color,opacity}` 轻度底板；不继承顶层手写字体。优先单行，最多两行。beat 可增加 `caption_cues:[{"text":"完整短句。","start":0,"end":2.5}]`，秒数相对母带对应 beat 的开始时间；生产字幕时间写在 alignment 对应 beat 中，所有文本拼接须与 beat 文本逐字一致；不需要为字幕重新生成或切割语音。字幕作为静态排字显示，不执行主笔描写。`paper` 为十六进制颜色。默认管线适合白底或透明底的清晰线稿，不保证复杂深色背景自动分离成功。

`ending.settle_seconds`（默认2.5秒，0–3.5）在 `end_hold_seconds` 之后再让完成画面停留。可用 `ending:{"fade_seconds":0.6}` 指定片尾渐隐到纸色；渐隐须完全位于最后一句结束后的停留区间内。`audio.end_hold_seconds` 默认0.45秒、范围0.2—0.8秒，默认不渐隐。总尾部静音不得超过1.2秒；详见[声音与收尾硬约束](audio-and-ending.md)。

## 命令

替换实际路径后运行：

```bash
python3 /path/to/skill/scripts/voice_studio.py /path/to/job --transcribe
python3 /path/to/skill/scripts/handdraw_video.py doctor /path/to/job
python3 /path/to/skill/scripts/handdraw_video.py build /path/to/job
python3 /path/to/skill/scripts/handdraw_video.py verify /path/to/job --assets-only
python3 /path/to/skill/scripts/handdraw_video.py render /path/to/job --preview --duration 15
python3 /path/to/skill/scripts/handdraw_video.py render /path/to/job
python3 /path/to/skill/scripts/handdraw_video.py verify /path/to/job
```

`doctor`中的 `ready` 表示已有任务输入可合成；`voice_studio.available` 单独表示新旁白服务是否可用，二者不能混用。

渲染默认按时间切块多进程并行（`HANDDRAW_RENDER_WORKERS` 可指定进程数，默认核心数减2、最多8，每块不少于约8秒），各块独立编码后无重编码拼接；11分钟竖屏约2.5分钟渲完。

先实际查看短段再导出整片。verify 完整解码并生成成片抽帧，只代表工程检查。按 [检查与返工](quality-gates.md) 填写检查证据后，才运行 `finalize`。它会拒绝未检查记录、哈希过期、用户已拒绝的视频和运行测试夹具。连续播放、听音没做时回执状态为 `delivery_review_completed_with_open_checks`。用户不接受成片时运行 `reject JOB --reason "…"`。

输出 `film.mp4/cover.png/captions.srt/script.txt/verification.json/verification.txt`。另存中文名时保留原件与哈希。脚本、图片、音频、设计或布局变化均需重渲染和复核。`runtime_fixture:true` 仅供包内离线测试，永远不能生成成片交付回执。

## 分镜

生产任务在 `job.json` 设 `"storyboard": "storyboard.json"`（格式见[创作与分镜](story-design.md)），每个属于计划视觉的动作加 `"visual": "v01"`；插画必须带，没带的文字/马克笔视为标签。

- `visual_groups` 的 `start_beat/end_beat` 可写页内索引或 beat id。组内动作可加 `"from_beat": "b2"`：前面的动作画到该句开始前，这个动作随该句起笔；组跨句连续绘制，下一句开始时可以继续画。
- 同页两个绘制块（beat 动作或组）时间重叠时，构建直接报错，提示合并为一个组并用 `from_beat` 对齐；不再等到渲染才发现。
- 构建把 `visual` 与插画 `image:[source,crop]` 写入事件，运行分镜检查：错误中止，统计与警告写入 `timeline.storyboard`。
- 有 storyboard 时，drawing-plan 的 subject 须有 `identity_cues`，drawing-review 须有 `recognizability`；出现已声明角色的插画在 drawing-plan 写 `references`。

```bash
python3 /path/to/skill/scripts/storyboard_lint.py /path/to/job --budget   # 生图前：预算与复杂度档位
python3 /path/to/skill/scripts/storyboard_lint.py /path/to/job            # 写完 job.json：计划是否全部落地
python3 /path/to/skill/scripts/plan_status.py /path/to/job                # 按镜头列出剩余工作
python3 /path/to/skill/scripts/drawing_contract.py /path/to/job --thumbnails  # 缩略图识别测试
```

## 画风与绘制计划门槛

生产任务必须完成[画风选择与可绘制约定](style-and-drawing.md)。`job.json` 增加 `drawing_plan`、`drawing_review`；每个插画 action 增加 `drawing_plan_id`，需要独立轮廓时增加 `outline_mask`；线稿先行（`line_first`）的插画另加 `line_first`：`line_first.py` 写出的报告路径。旧任务不能直接绕过新门槛。

## v1.2 可选制作设置

开始时按[可选制作设置](production-options.md)收集偏好。`hand.preset` 新增 `pen-only`：`navy` 同一支笔去掉手臂，阴影从笔尖起，不套用前臂长度规则。`hand.pen_text` 上限为16个字符。字幕可设 `background:{"color":"#243039","opacity":0.76}` 并配白字；空背景仍可用。上述新约定替代旧示例中的20字符与固定无背景。

`watermark:{"text":"用户原文，可换行","opacity":0.24,"position":"auto"}` 添加低对比水印；不配置或空文字不加。`position` 默认 `auto`：渲染前对每页抽6帧，在四角和左右边缘的18个位置里挑墨迹最少的一处（写入 `work/watermark-plan.json`，换位置时在翻页处淡出淡入），最佳位置仍压墨超过6%时 `verify` 失败；可固定为 `top_right/top_left/bottom_right/bottom_left`，`legacy` 恢复旧的浮动位置。默认1080p约27px，可指定 `font/color/font_size`，不透明度上限0.35。字外有一圈同不透明度的纸色描边（`halo`，默认 `#F2EFE8`，`false` 关闭），漂到深色画面（夜景）上也看得见。固定封面静止阶段不漂移，从1秒正文开始显示，位置始终在画内并避开底部字幕。

`ending.caption_hold_seconds` 可延长最后一句的阅读停留；必须小于或等于末句后停留减去渐隐的时间。字幕文字不变，SRT和实际画面一起延长。

## v1.3 整篇声音主时间线

旧生产任务先按连续旁白约定迁移，不能仅把旧分段文件拼接后声称已解决语调断裂。`runtime_fixture:true` 保留分段合成只为离线测试，永远不能作为交付路径。

字幕交付使用单一烧录层。`timeline.captions` 固定为 `{"presentation":"burned","owner":"caption_events"}`；额外字幕事件、字幕轨和成片同目录的字幕文件均拒绝交付。

v1.2.1 起简报和提示词必须非空且与绑定哈希一致。调用图像工具前运行 `python scripts/art_direction.py bind-evidence design/art-direction.json`；该命令不会把图片标记为通过。封面、正文、角色、道具、自定义手部和所有候选/编辑遵守同一流程。
