# 画风选择与可绘制约定

## 风格选择（文字列表，默认模型推理）

每个新任务先确定画风路径。默认 **模型推理**（`inferred`）：按主题与内容独立推理线条、色板、明暗、材质、背景；风格目录只是软参考，**不约束模型**。

主动选择时用宿主原生选项或终端数字菜单展示文字列表（**不展示效果图**）：

0. **模型推理（默认）** — `inferred`
1…N. 目录中的风格编号 — `catalog`（软选项）
N+1. **自定义描述** — `custom`

仍支持 `catalog_auto`（模型从目录匹配）以兼容旧任务。不使用 HTML 预览墙作为主路径。

```bash
python3 scripts/style_choice.py prepare JOB --catalog-source /path/to/hand-drawn
python3 scripts/production_options.py JOB --defaults JOB/options/defaults.json
```

**模型推理审美自由，绘制物理不自由。** 目录与编号是软参考。推理结果若无法形成干净连续墨线+平涂，应投影到可绘制的执行面（`drawing_mode: draw`），或取得用户对 reveal 的明确同意（`reveal_accepted`）。模型推理路径不得自行改用 reveal。若选目录中的纯黑白 1 号，写 `monochrome_reason`。

用户明确指出某个安装路径时，以该路径的实际目录为准；不要混进另一个版本的编号。否则默认使用本包固定依赖。目录从 `STYLES.md` 动态提取，不手写固定数量；快照保存完整配方和哈希。目录以文字名称为准；预览图可选生成但不进入选择 UI。更换目录来源需要重新准备快照。

选择过程只写本任务目录，不自动生图或发布。

## 风格决策文件

`style/selection.json` 由原生选项流程或模型写入。用户在聊天中明确选择/委托时可手工记录，`confirmation` 用 `explicit_user_request` 或 `user_delegated`；等待结束后用 `timeout_default`，说明实际等待时间和模型默认方案，不制造用户确认记录。用户说“你来选”“全部自动”时，由模型选出的画风一律记为 `user_delegated`，不能写成 `explicit_user_request`。

```json
{"mode":"inferred","confirmation":"explicit_user_request","evidence":"用户明确选择不选择列表中的，模型推理产出"}
```

模型随后写 `style/resolved.json`：

```json
{
  "status":"resolved",
  "mode":"inferred",
  "selection_sha256":"selection.json 的实际 SHA-256",
  "style_name":"本片实际画风名称",
  "rationale":"为什么采用此画法",
  "content_fit":"与本片具体内容的对应关系",
  "style_spec":{"line":"可分辨的独立轮廓","palette":"具体配色","shading":"具体明暗","texture":"具体质感","background":"背景处理"},
  "drawing_mode":"draw",
  "content_analysis":{"genre":"tech_essay","era":"2026，当下","audience_band":"adults","tone":{"primary":"理性、略带挑衅","humor_density":0.15},
                      "sensitivity":"低；涉及真实作者，只转述其观点","narrative_stance":"第一人称工程随笔的转述",
                      "evidence":["p01b1：AI 写的代码，他基本不审了","p05b2：工具调用止住了胡编","p16b4：他只问四件事"]},
  "style_family":"engineer_notebook",
  "directions":[{"family":"engineer_notebook","idea":"针管笔工程笔记，桌面与流程图","score":9},
                {"family":"editorial_concept","idea":"每条一个报刊隐喻","score":7},
                {"family":"new_yorker_cartoon","idea":"单幅讽刺漫画","score":5}],
  "characters":[{"id":"engineer","age":36,"head_to_body":6.8,"face":"成人下颌、短胡茬、细框眼镜"}],
  "avoid":["Q 版程序员","白壳机器人与发光大脑"],
  "diversity":{"differs_on":["family","proportion","palette","line"]}
}
```

（模型推理和自动匹配的画风必须写 `content_analysis` 到 `diversity` 这几项，规则见下节。）

### 模型推理画风：先读懂内容，再选画风（v3 硬门槛）

放任模型自由选择时，生图模型和导演都会收敛到同一种"可爱平涂"：四部片的明暗、质感、背景描述一字不差，色板几乎相同；35 岁的人物被画成圆脸少年，严肃时评里人人同款微笑。画风和内容错位，观众就出戏，出戏就划走。所以模型选的画风（`inferred`、`catalog_auto`、超时默认、"你来选"）必须按内容推出来，`scripts/style_fit.py` 在 `style_choice.py verify` 时检查。用户自己选的画风只记录，不检查。

1. **先读内容。** 写 `content_analysis`：
   - `genre`：history_ancient / history_modern / biography / true_crime / finance_business / commentary / science / tech_essay / warm_adult_story / workplace_growth / guofeng_myth / medical_health / kids / comedy / other；
   - `era`；
   - `audience_band`：kids / teens / adults / seniors / general；
   - `tone`：`{primary, humor_density 0–1}`；
   - `sensitivity`；
   - `narrative_stance`：第一人称随笔、报道、故事或教学；
   - `evidence`：至少 3 处原稿引文或 beat id。
2. **比较三个方向。** `directions` 至少写 3 个画风家族不同的方向，各写一句构想和一个分数。每个方向自问："只看一帧，观众会猜这是什么题材、什么年代、给谁看？"选中的写入 `style_family`。
3. **人物按年龄画。** `characters` 每人写 `age`、`head_to_body`、`face`：
   - 成人至少 5.5 头身；历史、传记、罪案、时评、医疗至少 6.5；
   - 儿童不超过 4.5；
   - 成人不用"圆脸、豆豆眼、腮红、大头、chibi"之类婴儿特征。

   没有人物时写 `no_characters_reason`。
4. **写清要避开什么。** `avoid` 至少两项。
5. **和最近的片子拉开。** 脚本读取同目录最近 5 部片的 `resolved.json`，以下情况拒绝：
   - 线条、明暗、质感、背景描述相似度达到 80%；
   - 六成颜色几乎相同（ΔE<5）；
   - 与上一部是同一画风家族却没写 `series_reason`；
   - 写了引擎默认墨色 `#1E2733` 却没写 `ink_reason`。

   另外，`diversity.differs_on` 至少写出画风家族、比例、色板、线条、背景中的 3 项差异。
6. **成人严肃题材不用低龄画风。** 受众不是儿童、题材属于历史、传记、罪案、财经、时评、科技随笔、成人温情、医疗时，不能用 chibi、kids_storybook、doodle_blob。只有幽默密度 ≥0.5 且写了 `comedic_reason` 的喜剧稿可以例外。这类片子的生图提示词里出现婴儿特征词，构建会拒绝。

画风家族都能画成墨线加平涂。参考映射如下（推断，依据 TED-Ed 按题材定画风、婴儿图式研究和 2026 年开源手绘技能的做法）：

| 题材与基调 | 画风家族 | 人物 | 色与线 | 避免 |
|---|---|---|---|---|
| 古代史、王朝 | `ink_brush_mineral`：白描勾线加矿物色平涂 | 成人 1:7 左右，服饰考据 | 绢黄、赭石、石青、朱砂；有提按的长线 | Q 版、糖果色、现代服装 |
| 近现代史、战争、灾难 | `ligne_claire` 或 `woodcut_flat` | 1:7，表情克制 | 土黄、铁锈红、灰绿；等宽清线或硬边阴影块 | 笑脸、亮色、把伤亡画可爱 |
| 传记 | `semi_realistic_clear_line` | 1:6.5–7，随章节变老 | 随年代 | 萌化或丑化真人 |
| 真实罪案 | `noir_red` | 1:7，受害者不画脸 | 黑、冷灰加一个红 | 暖色、圆润角色 |
| 财经、商业 | `editorial_concept`：报刊概念插画加手绘图表 | 1:6–6.5 | 藏青、石墨、米白加一个信号色 | 吉祥物、长脸金币 |
| 时评、舆论 | `editorial_cartoon` | 成人 1:6，原创人物 | 新闻纸米白、朱红、墨 | 与论点无关的统一微笑 |
| 科普 | `science_atlas`：图鉴、剖面线描 | 人物少 | 按对象取色；等宽技术线 | 给细胞、离子加脸 |
| 科技随笔、工程实践 | `engineer_notebook`：针管笔线描加 2–3 色平涂、正交的桌面与流程图 | 30–40 岁，1:6.5–7，多用手、屏幕、过肩镜头；Agent 等抽象对象不拟人 | 暖灰纸白、石墨、蓝图蓝加一个强调色；0.3–0.5mm 等宽线、直尺线、虚线框 | Q 版程序员、白壳机器人、发光大脑、赛博蓝紫 |
| 成人温情、家庭 | `adult_picturebook`：成人版墨线绘本 | 成人 1:6–7，儿童 1:4–5，老人有年龄感 | 暖橙、奶油、暮金；悲伤段降饱和 | 全员娃娃脸 |
| 职场、成长、观点 | `new_yorker_cartoon` | 1:5.5–6.5 | 3–4 色加一个强调色 | 默认卫衣主角、Corporate Memphis 扁平人 |
| 国风、神话 | `baimiao_nianhua` | 修长 | 朱、青、金、墨 | 日系萌 |
| 医疗健康 | `medical_diagram` | 成人 1:7 | 冷静低饱和加一个警示色 | 可爱病毒脸 |
| 儿童，或明确的喜剧 | `chibi` / `kids_storybook` / `doodle_blob` | 1:2.5–4 | 高饱和、圆润粗线 | 需 `comedic_reason` 或 audience=kids |

出第一张代表图后，做一次**盲猜测试**：只看 3 张 160px 缩略图（不读稿子），猜题材、年代、受众和情绪；与 `content_analysis` 有两项以上不符就重做。结论写入 drawing-review。线稿先行的线稿提示词也按画风家族写线条规格（等宽针管、提按勾线、硬边版画），不统一写"一样粗"。

库内两模式还须 `catalog_id` 和快照中的 `recipe_sha256`；库内自动匹配须 `content_fit`；模型代选 1 号须 `monochrome_reason`；`drawing_mode` 为 `reveal` 而不是用户主动选择了揭示时，须 `reveal_accepted`（用户同意的原话）。独立推理禁止这些库内字段。自定义还须 `style_basis`，逐字保留用户描述。先运行 `python3 scripts/style_choice.py verify JOB`。

库内配方的锚点、多阶段生成等既有要求仍然适用。保留所选风格的外观；若当前工具无法兼顾，要在生图前说明具体冲突，不能偷偷换画风。

### 绘制方式：逐笔绘制或揭示

- **逐笔绘制**（`drawing_mode: "draw"`）：只用于有清晰连续墨线、平涂上色的画风。引擎按部位整笔描线、再用马克笔上色，受线稿门槛和 `minimum_seconds` 约束。随包画风中 1、4、5、17、19、20 属于此类。
- **揭示**（`drawing_mode: "reveal"`）：水彩、水粉、纸雕、像素、水墨、蜡笔、厚涂、3D 等没有墨线轮廓的画风。手带宽软笔刷按部位斜向扫过，画面只在刷过处显出成图；不做线稿提取，也不受线稿门槛约束，整幅最短约2.5–3.6秒（实测2–7个部位的插画），远短于逐笔绘制。不要为这类画风硬造轮廓层。它放弃了逐笔绘制：观众看到的是成图被刷出来，不是被画出来。

库内画风的绘制方式由目录固定（`style/catalog.json` 的 `drawing_mode`，文字列表里会标明）；独立推理和自定义画风在 `resolved.json` 写明 `drawing_mode`。`job.json` 写 `"drawing": {"mode": "reveal"}`（默认 `draw`）与之保持一致，构建会核对。逐笔绘制的片子里，单张插画不能改用 `reveal`，构建会拒绝；画不出来就简化或重新生成那张图。

**揭示只由用户选。** 用户在主动选择里明确点了揭示风格，或书面同意后写入 reveal_accepted，就是知情选择。其他路径（自动匹配、超时默认、用户说“你来选”、独立推理、自定义描述）得出 `reveal` 时，`style_choice.py verify` 会拒绝，除非先告诉用户“这个画风只能刷出、不能画出”，用户同意后把答复写进 `resolved.json` 的 `reveal_accepted`。通常更好的做法是保留观感、改成可画的版本：例如“立体人物＋水墨环境”画成干净墨线轮廓＋平涂，用少量灰蓝色块和留白表现水墨气氛，`drawing_mode` 仍为 `draw`。

## 出图前的可绘制计划

每次调用图像生成或编辑前，先写计划并运行：

```bash
python3 scripts/drawing_contract.py JOB JOB/design/drawing-plan.json
```

输入含 `illustrations` 数组，每项有唯一 `id`，并包含：

- `composition`：实际主体、动作、遮挡和留白安排。
- `drawability_reason`：为什么这些部分能独立描线和上色，线条复杂度是否适合口播时长。
- `outline_strategy`：逐笔绘制的画风默认 `line_first`（先出线稿、再以线稿为输入上色，见下节）；也可以用 `visible_source_ink`（只出一张彩图，引擎从彩图里找线）或 `explicit_mask`（action 提供同尺寸的灰度 `outline_mask`，白色为轮廓、黑色为填色）。选 `visible_source_ink` 时在 `drawability_reason` 写明理由，例如图中几乎没有落在色块上的线。
- `subjects`：每个对象的 `id` 与 `kind`（human/animal/object/environment）。人物须覆盖 `head / neck / torso / left_arm / right_arm / lower_body`；不可见部分放入 `omitted_roles` 并说明裁切或遮挡，不能借此略过实际可见部位。左右指画面左右；不同观察角度保持解释一致。
- `parts`：按实际绘制顺序，每项写 `id / subject / role / stroke_groups`。`stroke_groups` 是独立笔画的简述数组，例如头部的外形弧线、左眼、右眼、鼻、嘴；不能一句“整个头连续画完”替代。
- `outline_order`、`fill_order`：均为同一有序 part id 数组。
- `spatial_order`：通常 `top_to_bottom_left_to_right`；构图/遮挡例外记录 `order_exception`。人物默认头、脖子、躯干、左臂、右臂、下半身，非默认在该 subject 写 `order_exception`。

校验通过生成同名 `.preflight.json`，绑定风格和计划哈希。计划改变后须重新预检；若图已生成，先重新评估受影响部分，不能把新计划伪称为旧图的出图前证据。

## 线稿先行（`line_first`）

一张插画分两次生图：先出纯黑线稿，检查、简化都在线稿上做，这时改图最便宜；再把线稿作为输入图做一次上色编辑。上色编辑几乎不动线（v3 实验：99.5–100% 的线在 3 px 内，偏移 0–1 px），所以线稿就是笔要画的线，不再从彩图里猜。从彩图里猜线时，落在色块上的线会漏掉，最后被马克笔当颜色涂出来：实测三张图分别只有 96.3%、97.2%、77.9% 的线是笔画的（笔记本电脑的屏幕和键盘几乎整块漏掉），线稿先行是 100%。

1. **线稿**。按美术指导写简报和提示词并绑定记录，再生图。提示词骨架（主体和构图按本片写）：
   > Black-and-white line art for a hand-drawn explainer video, drawn so a pen can trace it stroke by stroke. Pure black (#000000) lines on a pure white background, line quality per the style family (e.g. 0.4 mm even technical pen for engineer_notebook, brush lines with pressure for ink_brush_mineral). Every shape is closed. No fills, no grey, no shading, no hatching, no texture, no gradient, no glow, no text or letters. Subject: …
2. **先卡预算，再上色**。`image_gate.py JOB 线稿 --visual <id> --plate` 给封闭区域填上临时色，估算整张图（线和颜色）的 `minimum_seconds`，与真实构建相差约 4%。超出分镜窗口时有两条路：
   - 重新生成线稿，按 `prompt_fix` 写明轮廓数；
   - 更省的是编辑线稿：按 `edit_fix`，由模型看图逐项写出要擦掉的细节（如"头发里的发丝、帽衫上的褶皱线、窗外的云"），其余线不动。

   不要靠提示词里的笔数上限：模型不遵守，"最多约 35 笔"实测出来 87–99 笔。也不要按几何自动删线：哪条短线重要只有看图才知道，程序会把写字的手和杯子一起删掉。
3. **上色编辑**。输入图是线稿，提示词骨架：
   > Colour this exact line drawing (image 1). Keep every black line exactly where it is, with the same thickness: do not add, remove, move or redraw any line, and do not change the composition or proportions. Fill each closed region with one flat colour in …（本片色板）, then add one hard-edged, slightly darker shadow tone on the side of each object away from the light. No gradients, no texture, no glow, no outlines in other colours. The background stays pure white.

   画面里有人时，按 drawing-plan 中该人物的 `colours`，在提示词里逐人写出肤色、发色和衣服颜色，例如“The young mother: light skin #EDC3A0, black hair #2B2422, warm-white gown #FBF7EF”。只给色板不够：色板里有两种肤色时，模型会随便挑一种。服装和角色参考图不同时（换了便装、退休在家），上色这一步不要附参考图，否则会被带回参考图的服装和发色。build 会实测人物颜色，见[检查与返工](quality-gates.md)。
4. **配准**。`line_first.py JOB 线稿 上色图 --out inputs/art/<名>.png` 做四件事：
   - 把上色图对齐到线稿。上色图与线稿比例不同时（出图时线稿被补边成了方图或竖图），既试整张拉伸、也试从正中裁回线稿那一块，取保留线条多的一种，报告里的 `colour_fit` 写 `stretched` 或 `centre`；
   - 量线条保留率、偏移和新增线条，不合格就退出，并给出上色编辑的 `prompt_fix`；
   - 把线稿的线叠到颜色上，写出源图；
   - 写出同尺寸的 `<名>-outline.png`，以及 `<名>.line-first.json` 报告和 `<名>.line-first.jpg` 核对图（线稿 | 上色图 | 重合：黑 = 保留，红 = 丢失，蓝 = 新增 | 源图）。

   深色填充（黑发、黑鞋）是颜色不是线，不算新增。
5. **action**：`source` 为配准后的源图，`outline_mask` 为 `<名>-outline.png`，`line_first` 为报告路径。之后的区域编号、部位、门槛和构建都与 `explicit_mask` 相同。构建会核对报告为通过，且源图和遮罩的哈希与报告一致；换了任何一张图都要重跑 `line_first.py`。
6. **美术记录**：线稿和上色图都按 `generated_under_guidance` 记录；配准后的源图按 `composed_with_verified_layers` 记录，`evidence` 写明两张输入图和报告结论。

## 成图后的核对与执行

实际看图后，为 action 写 `drawing_plan_id`、按计划同顺序的 `parts` 多边形及必要排除区域，所有可见像素只能归属一个部位。多边形随实际解剖与遮挡，不用从上到下的等高横条代替语义部位。

线稿闭合良好时，可改用编号区域：运行 `region_marks.py`，看叠图把编号分配给部位（`region_map` + `part_regions`，见[运行约定](runtime-contract.md)），比手写多边形更准。

### 出图要求（让图真能被一笔一笔画出来）

引擎会用 `lineart_gate.py` 拒绝纹理型线稿，并按真实笔画计算 `minimum_seconds`。出图提示与编辑时：

- 连续、干净、粗细稳定的墨线轮廓；不要排线、交叉阴影、布纹/毛发/树皮纹理线和边缘检测感的碎线。
- 部位交界处把轮廓闭合（叶基、衣摆、袖口、领口画出完整分界线），区域才能分开编号。
- 平涂色块加柔和明暗；明暗不要用线条表达。
- 控制线条数量：默认画法按约每秒3.9笔估算（v3 实测，含起形、擦除和按马克笔上色，与图幅大小基本无关；按部位上色、不起形、调高 `speed` 都更快，各画法的速度见[创作与分镜](story-design.md)），45—50笔的单一主体全部画完最短约12—14秒（`minimum_seconds`）；口播窗口更短就简化画面或用 `visual_groups` 跨句延长窗口。引擎不淡入任何线条或色块，每一笔、每块颜色都由笔尖画出；窗口不够只会报错，不会偷偷省笔。

`job.json` 指定 `drawing_plan` 和 `drawing_review`。使用 `outline_mask` 时，审核中还须 `outline_mask_sha256`，且遮罩尺寸必须与裁切前的完整原图一致。后者结构：

```json
{"style_sha256":"resolved.json 的实际哈希","plan_sha256":"实际计划哈希","illustrations":[{
  "plan_id":"figure-01","status":"pass","source_sha256":"实际图像哈希","crop":null,
  "anatomy_and_occlusion":"看到了什么、哪些部位被遮挡",
  "part_boundaries":"各区域如何对应实际部位",
  "stroke_breaks":"哪里起笔、转折、收笔与抬笔",
  "outline_then_fill":"轮廓是否齐全且可独立追踪",
  "spatial_order":"确认部位顺序或说明构图例外"
}]}
```

`art_direction.record` 的顶层也须写入相同 `style_sha256`，绑定封面和生成素材到本次画风。换画风后不得沿用旧的图片审核。

所有插画整幅轮廓完成后才能填色。描线按部位顺序；上色默认按马克笔颜色跨部位（见下文“拟人绘制步骤”），`colouring: parts` 时按部位顺序。黑白线稿可以没有填色阶段，但不可反过来省略轮廓。笔画只能沿连续像素前进，断开的笔画必须抬笔，不能用最近点跳连；颜色只在笔刷经过范围出现。短到看不出起落、画面必须疯狂跳笔才能赶上口播时，减少非必要线条或重新分镜，不能靠缩短抬笔假装自然。

### 拟人绘制步骤（引擎）

引擎按人类习惯执行，而不是匀速显影。人画画靠思考，由简到繁：先用几笔简单形状定下整张图的大关系，再一个模块一个模块往深里画，最后擦掉辅助线上色。

0. **起形（v3，模型规划）**：模型看图后在 action 或顶层的 `drawing.sketch` 写一组简单形状（规范化坐标，`part` 可选），引擎在任何墨线之前用浅蓝铅笔画出：
   - `{"shape":"ellipse","center":[x,y],"radius":[rx,ry],"angle":度}`：从顶端起逆时针画一圈多一点；
   - `{"shape":"box","box":[左,上,右,下]}`：一笔绕四角；
   - `{"shape":"line","points":[[x,y],…]}`：过这些点的动势线，两端略出头；
   - `{"shape":"polygon","points":[…]}`：闭合多边形。

   先定大关系：头用圆加十字中线和眼线，身体用梯形加脊柱和肩线，手臂用动势线，再到道具和背景。顺序按路径连贯排，不在画面里来回跳。每个形状至少 3 帧，约按视频速度画；起形最多占全图时间的 18% 或 2.5 秒。墨线画完后，橡皮沿每条起形线来回小幅擦过，起形线在橡皮经过处消失（约占 9%，0.5–2 秒），然后才上色。擦除时画面上的笔换成一块橡皮。`drawing.sketch: true` 仍是旧做法：按部位轮廓自动简化出起形，描线后淡出。
1. **按模块由简到繁**：部位（模块）按 `parts` / `part_regions` 的顺序画，模块之间有短暂停顿。每个模块内分四步：
   - 外轮廓：它与别的模块或纸面的交界，按墨线围出的范围判断，不受生成图外圈浅色光晕的影响；
   - 长结构线：下颌、发际线、衣身的长线；
   - 关键部位（脸、手）里独立的五官：眼、眉、嘴，实心的眼睛圆点也算；
   - 短细节：发丝、褶皱、指缝。

   每一步内部都从已画的线上长出来、就近走。
2. 线：压力起收笔、拐弯降速、抬笔快甩；每笔都由笔尖走过源图墨线像素。
3. 色（2.0）：上色跟着画面自己的闭合区域走，不按部位多边形切块。模型画的部位框只用来给笔画分组。每块颜色先沿区域内侧描一圈边，再用斜向排笔填满：长条形横跨着画，圆形用右手自然的斜度。接近的几笔连成一笔不抬笔，漏掉的角落最后补涂。刚涂上的颜色是带毡头纹理的半透明湿笔触，下一笔覆盖上来或稍后变实，所以不会整块显现。窗口很紧时才退回最宽的马克笔，用长行快涂，不描边。部位之间短暂停顿。
   **按马克笔上色（v3，`drawing.colouring: "markers"`）**：像人拿着一套马克笔，一支笔把同色的地方都涂完再换下一支，先浅后深；换笔时稍停。同一支笔的区域按就近路径走，不管属于哪个模块。上色区域按墨线围出的范围划分，底色整块铺满；涂完一种颜色的底色后，换深一号的笔压这种颜色的暗面（`shade` 笔段），所以阴影是"后压上去的"，不是和底色并列的色块。v3 起默认按马克笔上色；`"parts"` 恢复按模块上色。
   **上色前沿（v3 定稿）**：颜色涂到哪里的那条边在显示分辨率下画，带 1 像素抗锯齿。规划图上每个像素记下笔头到它的有符号距离（被碰到之前多近、碰到时多深），近景重渲染时在相邻规划像素之间插值，所以放大 3–4.5 倍也不会出现规划像素的台阶；两块颜色一样、只有线稿上才有界线的区域，前沿跨过界线连续走，界线本身先平滑。软阴影条带（按模块上色时）在显示分辨率下按条带几何重算。
   **画材质感（v3，`drawing.texture`）**：`{"ink": "pressure" | "none", "marker": "flat" | "layered"}`。默认（`true`）是压感墨线 + 平涂马克笔：完成色与原图逐像素一致，只有上色进行中那一笔是湿的、带毡头纹，随即变实。`false` 回到 2.x 的匀速针管笔（完成图与改动前逐像素一致）。`layered` 只在明确写出 `{"marker": "layered"}` 时启用：它会在每块完成色上留下斜向笔道，成片里整张图都是一条条的，不纯净（3.0.0 曾默认开启，3.0.1 起改为默认平涂，前后对比见 `output/evidence/v3.0.1/flat-marker/`）。`pressure`：长线条的自由端（不接任何别的墨）落笔收笔变细，墨色在线内有很轻的浓淡（最多淡 8%）；短笔画（虚线、字形笔画）和画家已经收尖的端头保持原样，线的外轮廓始终是光滑的抗锯齿边，不加毛边，不加墨。`layered`：完成色按真实笔头宽度（不是延时摄影里那支宽笔）沿手扫的方向排出笔道，两道重叠处深约 9%（正片叠底），笔道边缘略积墨，顺着笔道有很淡的毡头纹；每块区域平均色不变，单像素明度变化限制在 −14%～+8%，接近白色的区域不加。实测三张 v3 插画：成色平均 ΔE 1.1–2.1，墨线核心保留 ≥99.96%，构建耗时 +10%。证据见 `output/evidence/v3/final/media/`。
4. 时间预算在分镜/出图前用 presentation.mode 卡住；窗口不够就简化图，不淡入、不改 reveal。

出图时尽量提供可分离的 line 层与平涂 fill（干净墨线、部位交界闭合）；`stroke_groups` 写清独立笔画，便于引擎与门禁对齐。

构建、渲染、验证会拒绝缺少风格/计划/审核的生产任务，并校验实际路径。检查不能自动证明模型标对了人体或遵守了画风；代表段必须实际检查轮廓末帧、填色起始、关键部位切换和连续动作。旧任务没有这些证据时需按真实来源重新评估，不批量补写“通过”。运行测试 fixture 仅用于工程回归，禁止最终交付。
