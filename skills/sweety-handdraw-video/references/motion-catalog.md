# 运镜、转场与动效目录（2.0.1）

镜头和转场决定成片的观感，模型应逐镜设计，不交给默认值。本目录是可用的"卡组"：每张卡写明做什么、默认参数、适合什么、不适合什么。没写的页面由自动导演兜底（见末节），模型写了的一律以模型为准。

设计顺序：先定每页讲什么，再给每一镜选**镜头**（看哪里、怎么移），再选**进页方式**（转场），最后加**动效**（元素动作或网页动效层）。动效必须有画面里的起因：口播点到、数据揭示、因果推进。没有起因的特效会让画面像套模板。

## 一、镜头 `pages[].camera`

关键帧数组：`{"at": 秒 或 "beat:<id>", "center": [x, y], "zoom": 1–3, "move": 名称, "seconds": 0–3, "ease": 缓动, "follow": 布尔, "rotate": -20–20, "shake": 0–20, "impact": 0–40, "hop": 0–0.6, "subject": 视觉 id}`。

**基本规则**
- `at` 从本页开始计时；`at: 0` 的关键帧是开页取景。
- 本页素材按本页最大 zoom 生成，所以推近后画面仍然清晰。
- 缩放在对数空间插值：放大一倍和再放大一倍用时相同，推拉更像真实镜头。
- `move` 给出这一运镜的默认时长和缓动，显式写的字段优先。

| move | 默认 | 用途 | 不要用于 |
|---|---|---|---|
| `cut` | 0 s | 页内硬切到新取景（跳切）；开页取景 | 连续两次 |
| `push_in` | 1.0 s, inout_cubic | 推近讲解一个部位、一个数字 | 连续三次都往里推 |
| `pull_out` | 1.4 s, out_cubic，长尾落定 | 揭示全貌、整块壁画；页末总结 | 每页都用 |
| `pan` | 0.75 s, inout_sine | 同一缩放下移到相邻元素 | 横跨整块画板（用 hop） |
| `drift` | 3 s, inout_sine | 长停留时慢慢推近，给画面呼吸 | 快节奏段落 |
| `hop` | 1.2 s，途中拉远约 30% | 在推近状态下跨远距离到另一元素：先退一点看清去向，再推进 | 近距离移动 |
| `whip` | 0.28 s，自动运动模糊 | 话题急转、对比的另一边 | 平静叙述；超过镜头数的 1/3 |
| `snap` | 0.3 s, out_expo | 突然盯住一个细节 | 连续使用 |
| `slam` | 0.18 s, out_back，到位震动 10 px | 金句、关键数字砸进画面，每段最多一次 | 普通强调 |
| `dolly_zoom` | 2.4 s, inout_sine，需 `subject` | 希区柯克变焦：主体保持原大，背景冲过来（推近）或退远。"意识到了"、压迫感、命运转折，全片最多一次 | 普通讲解；`at: 0` |

**附加字段**
- `follow: true`：在该缩放内温和跟随笔尖，适合大图边画边看。
- `impact`：到位时的衰减震动，单位是屏幕像素。
- `hop`：途中拉远的比例，可以加在任何运镜上。
- `shake`：持续手持漂移（几像素的慢速摆动）。
- `rotate`：倾斜角度，只用于刻意的情绪镜头。
- `subject`：dolly_zoom 的主体，写元素或动效层的 `visual` id。从这一帧开始，主体保持开始时的屏幕大小，下一个关键帧再缓回原大。`center` 一般对准主体。

**视差 `depth`**（写在 `actions[]` 或 `space: board` 的动效层上，-1 到 1）
- 0 或不写就是画板本身。负数是远景：镜头移动时它只跟着走一部分，画在本页墨线下面。正数是近景：比画板移动得更多，推近时向外散开。
- 位置以本页开页取景为准：镜头不动时，元素就在版式写的位置。
- 只在镜头真的移动时才看得出来，所以要配 `pan`、`drift`、`hop` 或 `dolly_zoom`。一页用两到三层就够：远景 -0.4 到 -0.7，主体 0，近景 0.4 到 0.8。
- 笔尖会跟着它所画的那一层走。

**密度门禁**（`build` 报警告，依据花叔《镜头语言》和 anidoodle 运镜语法）
- 快速运镜（whip、snap、slam，以及时长不足 0.4 s 的移动）不超过全部运镜的 1/3。
- 不要连续三次推近。
- 推近每分钟超过 8 次，观众会坐立不安。

## 二、进页方式 `pages[].transition`

`{"type": …, "seconds": …, "direction": "left|right|up|down", "style": …, "target": [x, y], "color": "#hex", "angle": 度}`，首页不能设。

**时长从哪里来**：转场时长取自上一页最后一笔之后和本页第一笔之前的空档。空档不够时自动缩短；要完整播放，给两侧各留一半时长。

| 类 | type | 秒数范围 | 参数 | 适合 |
|---|---|---|---|---|
| 轻 | `cut` / `fade` / `slide` / `push` / `pan` | 0.3–1 | push、wipe 可设 direction | 同一章节内连续讲解。`pan` 让两页像同一块大白板相邻 |
| 纸张 | `page_turn` | 0.6–1.8 | direction：left 向前翻、right 往回翻、up 像便签本向上翻、down；angle 设翻页斜度 | **章节切换的招牌**、故事翻页 |
| 纸张 | `slide_over` | 0.3–1 | direction | 新的一张纸压上来，带投影 |
| 纸张 | `paper_tear` | 0.7–1.8 | — | 推翻旧观念、"撕掉过去"、强转折 |
| 纸张 | `card_flip` / `cube` | 0.5–1.4 | cube 可设 direction | 正反对比、两种方案、前后 |
| 纸张 | `tiles` | 0.6–1.6 | — | 拼图式汇总、多要素合成 |
| 纸张 | `shards` | 0.6–1.6 | target：从哪里先碎 | 打破、崩塌、"碎了"，全片最多一次 |
| 擦除 | `erase` | 0.6–2 | style：felt（白板擦，默认）、rubber（橡皮擦，留下碎屑） | **章节切换的招牌**、"清空重来"、黑板讲解 |
| 墨与笔 | `brush_reveal` | 0.6–1.6 | angle | 用毛刷把下一页刷出来，有干笔飞白，适合手绘感的过渡 |
| 墨与笔 | `marker_wipe` | 0.6–1.6 | color | 马克笔色带扫过，适合轻快的列表推进 |
| 墨与笔 | `ink_bleed` | 0.6–1.6 | target、color | 墨滴晕开再从墨里打开新页：情绪转折、回忆 |
| 墨与笔 | `iris` | 0.5–1.4 | style：round 或 blot（墨渍边）；target | 收束到一个点再打开：聚焦一个人或物 |
| 镜头 | `zoom` / `zoom_through` | 0.3–1.6 | target：穿越点 | 匹配剪辑：钻进某个元素，进入它的内部或下一层 |
| 镜头 | `whip` | 0.25–0.7 | direction | 快速转场到对立面 |
| 点缀 | `flash` | 0.2–0.7 | color | 闪白切：顿悟、爆点。半秒内不要连闪两次 |

**招牌转场**：一部片选 1–2 种作为章节转场（推荐翻页或板擦），其余用轻转场；设计感强的转场是点缀。同一种设计转场连续出现三次就像预设模板，`build` 会提示。

## 三、元素动作 `actions[].motion`

`[{"type": …, "at": "beat:<id>" 或秒数, "seconds": …, "strength": 1, "part": 部位 id}]`。强调和部位动作等最后一笔画完再播；待机循环到翻页为止；退场在 `at` 开始。

| 家族 | 类型 | 用途 |
|---|---|---|
| 强调（口播回到它） | pop、pulse、wiggle、shake、nod、glow、jelly、tada、bounce、heartbeat、rubber_band、swing | 点名一个元素；glow 用于"这个最重要" |
| 待机（让画完的画活着） | float、breathe、sway、boil | boil 是手绘线条的轻微抖动；sway 用于植物、旗子；float 用于气球、云 |
| 退场 | fade_out、shrink_out、pop_out、fly_out | 讲完的元素让位 |
| 部位木偶 | part_nod、part_tilt、part_wave、part_bob（需 `part`） | 人物点头、挥手；部位用插画的语义部位 |

## 四、网页动效层 `pages[].overlays`

`{"recipe": …, "params": {…}, "x", "y", "width", "height", "at", "seconds", "space": "board|screen", "exit": {"style": "fade|shrink", "at", "seconds"}, "z", "seed", "bleed": false, "visual": 视觉 id, "pen": true, "depth": 0}`。

**运行方式**
- 在无头 Chromium 里按虚拟时钟逐帧渲染，结果完全可复现，按内容哈希缓存。
- `board` 层随镜头移动；`screen` 层不动，用于标题、下沿字。
- 动画播完后停在最后一帧，直到退场或翻页。
- 大字会自动缩小到放得进框；任何配方的最终帧碰到框边（内容被截断），`build` 都会报错。确实要出血到框外的内容写 `"bleed": true`。
- **笔先动，画面才动**：`board` 层出现时，笔尖会描一下它：
  - 圈注：绕一圈；下划线：划过去；图表和大字：从下沿扫过；公式：逐字跟着写；3D 物体：绕一圈轨道；倒入的道具：从入口点下去；其他：一个小圈。
  - 笔尖悬在纸面上方，不出墨。笔正在画别的东西时不描。
  - 不想要时写 `"pen": false`。
- 写了 `visual` 的动效层算作故事板里的那个视觉（图表、地图、标题、公式……），`storyboard_lint` 会把它计入。
- 声音：`burst` 带一声轻铃，slam 大字和镜头 `slam` 带一声闷响（见 `audio-and-ending.md` 的音效层）。
- 需要先运行 `bash scripts/setup.sh --web-motion`。
- 第三方库和许可见 `assets/web-motion/THIRD_PARTY_NOTICES.md`。matter-js、Zdog、MathJax 只在用到 physics、spin3d、formula 时加载。

| recipe | 关键参数 | 用途 |
|---|---|---|
| `kinetic_title` | `text`、`keywords`（主色强调）、`style`、size、font（hand、sans、serif）、underline、badge（标题前的编号圈：条目标题用它，和章节标题分开层级） | 开场标题、金句、章节名。style 有八种：<br>• glyph_in：整字逐个淡入、轻微上浮落定，任何时刻都没有被遮掉半截的字；关键词保留强调色，英文单词和数字整体出现；停下后和 mask_up 的位置一模一样。开场标题首选它<br>• blur_up：模糊上浮，产品站质感<br>• spring_pop：逐字弹出<br>• mask_up：遮罩上推（中途只露出半个字，旧片保留，新片开场用 glyph_in）<br>• typewriter：打字机<br>• slam：逐词砸入并震屏，每段最多一次<br>• scramble：乱码收敛成字，适合数字和科技感<br>• word_by_word：逐词出现，关键词先大后缩、挤开邻词<br>blur_up、mask_up、typewriter、glyph_in 会把逐词（逐字）间隔压进这一层自己的 `seconds` 里，长句在结束前两帧全部落定，不会定格在半模糊上<br>换行：只在词与词之间断（数字连单位、人名连“·”、标点不落到行首）；要换行时两行尽量等长，优先断在逗号、问号后，末行不会只剩“哪一条？”这样的短词 |
| `rough_annotate` | type（circle、box、underline、highlight、strike-through、crossed-off、bracket） | 在已画好的元素上画圈、划线 |
| `sketch_shapes` | shapes：arrow、curve_arrow、line、rect、ellipse、polygon、check、cross、bracket、star；`flow` 为流动虚线 | 手绘箭头、勾叉、流向 |
| `callout` | text、tail | 弹出的手绘气泡或标签 |
| `number_ticker` | to、prefix、suffix、decimals | 里程表式滚动数字 |
| `progress_ring` | value、max | 圆环百分比 |
| `burst` | kind：sparkle、heart、confetti | 揭晓、达成。只在有起因时用 |
| `morph` | shapes：[{shape：circle、dot、rect、card、star、heart、triangle、diamond、arrow 或自定 path，box，fill，stroke，text，hold_ms}] | "持续主角"：一个点变成卡片、再变成下一个概念。先沿路径拉伸，后半程才变形，颜色跟着过渡 |
| `rough_chart` | kind 见下 | 手绘图表：先画坐标轴，再出数据，每张图只强调一件事（`highlight`）。数据必须真实 |
| `map_route` | route：[[经度, 纬度], …]；stops：[{at, label}]；projection：globe（转动的地球，跟随标记）或 flat（fit 设为 route 或 world）；travel_ms | 路线、航线、海缆、迁徙。只画海岸线，不画国界 |
| `physics` | scene：pour（倒入，默认）、topple（多米诺）、stack（球撞倒一摞方块）、bounce（弹跳）；container：jar、tray、floor、none；items：[{shape：circle、box、pill，size 或 w/h，fill，label，count}]；gravity；drop_ms | 真实物理（matter-js）："钱越攒越多"、"一个倒下全部倒下"、"一推就塌"。每个物体保持一份手绘轮廓，跟着刚体转动，不抖线 |
| `spin3d` | shape：box、coin、cylinder、cone、sphere、pyramid、gem；colors：[主色, 侧面, 顶面]；turns；tilt；axis（y 或 x） | 转动的立体物件（Zdog），有墨线外轮廓和平涂明暗：硬币翻面、产品转一圈、晶体、积木。先带一点过冲弹出，再转 |
| `formula` | tex（TeX）；size（每 em 像素，默认撑满框）；color；glyph_ms；highlight：{from, to（字形序号），style：circle、box、underline} | 手写公式（MathJax）：每个字形先描轮廓再填实，按书写顺序走，分数线先于分母，根号先于根式里的内容。写完可以圈出一部分。TeX 写错时 `build` 直接报错 |
| `roadmap` | groups：[{label, count, numbers（可选：这一组自己的条目编号）}]；done（已完成条数）；current（这一条的序号，从 0 起）；active（[起, 止) 一起点亮的一段，整组讲时用）；focus（点亮的组）；mode：full（章节卡、开场地图）或 strip（每条页面顶部的进度条，可加 caption）；enter: "settled"（只用于 strip：翻页后第一帧整条已在，只有本页这一条（组）点亮、caption 出现；第一页用默认的逐点弹出，之后各页用 settled，进度条跨页“停”在原处，不再每页从白纸长出来） | 长清单的结构与进度：观众知道在哪一章、第几条、还剩几条 |
| `table` | columns；rows（单元格写“✓”“✗”会画成勾叉）；widths；row_ms（每行出现的时刻）；highlight：[{row, col, ms}] | 对比与取舍：逐行写出，关键格打底色 |
| `flow` | nodes：[{id, text, at:[x,y] 0–1, size:[w,h], shape：round、box、circle, ms}]；edges：[{from, to, text, ms, bend, dashed}]；tokens：[{path:[id…], ms, edge_ms, loop}]；shake：[{id, ms, mark}] | 流程、循环、因果树、架构：方框依次画出，一个点沿箭头跑；冲突处抖动打叉 |
| `swimlane` | lanes：[{label}]；blocks：[{lane, from, to, text, kind：work 或 wait}]；span；sweep_ms；title | 谁在做、谁在等：同步与异步、批量交代 |
| `checklist` | items：[{text, ms, mark：check、cross、strike, mark_ms}]；columns | 列举：逐项写出，说到时打勾、打叉、划掉 |
| `cards` | count、columns、card_ms；flicker：{indices, step_ms, ms, question}；groups：[{indices, label, x, ms}]；highlight | 很多个：版本、并行 Agent、十次运行；错误标记跳动、分成对照组 |
| `pulse` | beats：[{ms, label}]；sweep_ms | 定时、心跳、周期任务 |
| `custom` | html、css、script（可用 anime、rough、RoughNotation、confetti、d3），element、explains，moments | 以上都不够时由模型现写。动画必须由 anime.js、CSS 或 element.animate 驱动（都受虚拟时钟控制），背景保持透明。写明 element（画的是什么）和 explains（讲清什么机制），才算一种讲解形式；`moments` 列出它新出现东西的时刻（毫秒，从图层开始算），画面节奏检查据此知道“这几秒在动”。 |

`rough_chart` 的六种 kind：

| kind | 参数 |
|---|---|
| bar | values、labels、highlight、suffix |
| line | values、labels、highlight |
| pie | values、labels |
| dots | total、marked，例如"100 个里有 99 个" |
| tally | count，按笔顺画"正"字计数；style 设为 bars 时画四竖一斜 |
| timeline | events：[{when, label}]，竖屏自动竖排 |

## 五、自动导演 `job.direction`

`{"camera": "auto|off", "transitions": "auto|off", "motion": "auto|off", "boil": true}`，默认全部 auto。只填补模型没写的部分。

- **镜头**
  - 每幅图开画前推近框住它，画下一幅时移过去；远距离用 hop。
  - 各页轮流用三种方式拿住正在画的图：静止、慢推、跟随笔尖。
  - 画完后、翻页前拉远看全页。
  - 页面本身已经撑满画面时，只做一次慢推。
- **转场**
  - 新章节（`chapter: true`，或以"第N章、第N步"等开头的标题）轮流用招牌转场：翻页、便签上翻、板擦、撕纸。
  - 其余页面用轻转场；每三页放一个设计感转场（毛刷、墨渍光圈、马克笔、穿越、拼砖、墨晕、立方体等）。
  - 相邻两页不会重复同一种。
- **动效**：插画最后一笔落下时轻弹一下，之后线条轻微 boil。

## 六、不要做

- 故障风（glitch）、光扫、镜头光晕、彩虹渐变。
- 没有起因的粒子爆发。
- 交叉淡化叠交叉淡化。
- 半秒内连续闪白。
- 每镜都推近、每页都换一种炫技转场。
- 画到一半就让元素做动作。
- 用动效遮掩画得不好的图：先改图。

## 七、常用组合

- **开场钩子**：大标题 `kinetic_title` 用 slam 或 blur_up，配一幅图从画面中心画起，镜头 `cut` 近景。第一句口播结束时 `pull_out` 看全貌。
- **数据揭示**：先画场景，再用 `rough_chart` 或 `number_ticker` 给出真实数字，镜头 `snap` 盯住数字。如有必要，在关键数字上用一次 `slam`。
- **因果链**：`morph` 让"原因"的形状变成"结果"，再用 `sketch_shapes` 画箭头加 `flow`；镜头用 `pan` 沿因果方向移动。
- **旅程或地理**：`map_route` 的地球随标记转动，配合站点口播；转场用 `zoom_through` 钻进某一站。
- **章节切换**：`page_turn` 或 `erase` 二选一作为全片招牌，标题页配 `kinetic_title`。
- **结尾回顾**：整块画板 `pull_out`，用 `rough_annotate` 圈出三处重点，最后一页 `iris`（blot）收束。
- **有纵深的开场**：远景 `depth: -0.6`、主体 0、近景文字 `depth: 0.7`，画完后用 `drift` 横移，层次自己出来。情绪顶点用一次 `dolly_zoom` 盯住主体。
- **公式推导**：`formula` 写出公式，口播讲到哪一项就 `highlight` 圈出哪一项；镜头 `push_in` 到公式，推导完 `pull_out`。
- **积少成多或连锁反应**：`physics` 的 pour 或 topple 配一个真实数字（`number_ticker`），镜头静止，让物理自己演。
