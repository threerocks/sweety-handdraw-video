# 制作工具包：用数据写一部片子

一部片子不再需要在 `work/` 里手写十几个 Python 文件。通用的部分都在 Skill 里，片子只写**数据**：

| 文件（任务目录内） | 写什么 | 由谁执行 |
|---|---|---|
| `inputs/script.json` | 口播（页、句；句可带 `say` 读法表） | 配音、`film_kit` |
| `design/image-specs.json` | 每张图的用途、场景、配色说明、要删的细节；全片画风提示词 | `art_pipeline.py` |
| `design/film.json` | 画面编排：每屏的图、标题、短句、讲解形式、进度条、章节页、转场、镜头，以及每张图的绘制计划 | `film_kit.py` |
| `design/cover.json` | 封面：底图与文字图层 | `cover.py` |
| `design/anim/*.js`（可选） | 自定义 SVG 动画脚本 | `film_kit.py`（custom） |

`job.json` 的 `pages`、`storyboard.json`、`design/drawing-plan.json`、简报、提示词、`design/cover-layers.json` 都由脚本生成，不手改。

## 流程

1. 写 `inputs/script.json`、配音锁定（`alignment.json` 与 `master-chunks/` 的逐字时间）。英文名、数字写法的读法用 `say` 表（见下）。
2. 写 `image-specs.json` 与 `film.json` 的 `pictures`（构图、主体、部位、叙事要求；方框可先省略）。`film_kit.py plan JOB` 写出绘制计划，`drawing_contract.py JOB design/drawing-plan.json` 做出图前检查。
3. `art_pipeline.py JOB prompts`：写简报和提示词并绑定哈希。简报自带**绘制预算**：本页口播秒数 × 70%（画笔上限，画完的图至少留 30% 给人看）× 每秒约 3.5 笔（一笔 = 一条连续轮廓线），规格要的轮廓数超了会标 `OVER BUDGET`。这里的每秒 3.5 是**有意保守的“向模型要多少条轮廓”**，不是引擎的绘制速度：3.0 测试片里生图模型交回的线稿笔画数是所要轮廓数的 1.3–6 倍（要 10–22 条，实测 14–85 笔）。真正的窗口以 `film_kit.py measure` 的实测秒数和 `storyboard_lint.py --budget` 按每张图画法给出的 `strokes_per_second` 为准（见 story-design.md 的速度表）。2:3 画布默认竖幅构图，提示词加一句 "Compose for a tall 2:3 portrait page"（横幅图在竖屏上会缩小，6/7 部片子因此触发“内容范围”警告）。
4. `art_pipeline.py JOB gen 名字…`：线稿 → `image_gate.py --plate`（超预算就停下，不浪费上色）→ 上色编辑 → `line_first.py` 配准，三张并行。超预算时把门禁给出的 `edit_fix` 写进规格的 `edit`，再 `prompts`、`gen --edit 名字`（旧线稿移到 `inputs/art/rejected/<名>-line.vN.png`，从它点名删细节）。看图后 `art_pipeline.py JOB record 名字 "已看原图：……"`。
5. `part_grid.py JOB 图` 看网格，把部位方框写进 `pictures.<名>.parts`，再 `film_kit.py plan JOB`、出图前检查。`film_kit.py measure JOB 名字…` 按页面尺寸实测最短绘制秒数（`名字@360` 试别的规划尺寸，不保存）。
6. 写 `film.json` 的 `screens`，`film_kit.py compile JOB`，再 `storyboard_lint.py JOB`、`director_lint.py JOB`、`handdraw_video.py build JOB`。build 之后 `art_pipeline.py JOB sync-minimums` 用引擎实测值更新窗口，再 compile 一次。
7. `cover.py JOB`（出图前先写简报、提示词并绑定），看图后 `art_pipeline.py JOB record cover "……"`；看完每张源图写 `review-notes.json`，`art_pipeline.py JOB review review-notes.json` 写 `design/drawing-review.json`。

已通过审核的图，`prompts` 不会改写它的简报和提示词（`--force` 才改，并把状态打回 pending）。

## 时间提示（cue）

所有时刻都取自配音的逐字时间，不手填秒数：

- `"@p03b4"`：这句开始；`"@p03b4:推特在国内"`：说到这几个字时；`"@p03b4:推特在国内+0.8"`：之后 0.8 秒（可为负）。
- 元素的 `at` 是页面秒数：标题、短句提前 0.1 秒，讲解形式提前 0.15 秒（写了偏移就按偏移）。形式参数里的提示（`ms`、`row_ms`、`mark_ms`、`moments`、自定义参数……）换算成**相对形式开始的毫秒**。
- 算式：`{"=": "max(3000, ms('p05b2', '下一段', 1))"}`，可用 `ms() at() max min round int abs float range` 与 `start`（形式开始秒数）、`length`（本屏秒数）。
- 识别把“五十四”写成 54、“三百六十五”拆成 “36”+“5”：查找时按中文读音字数计（四位及以上按逐位读，如年份）。
- `say`：`job.json narration.say` 是全片读法表（`{"Grok": "格罗克", "70万": "七十万"}`），句子的 `say` 在其上补充或覆盖。字幕显示 `text`，配音读替换后的文本；提示里的短语同样替换后在读法文本里查找，所以写 `@p01b1:Grok` 或 `@p01b1:格罗克` 都行。`film.json` 的 `say` / `beat_say` 原样写进 `job.json`。
- `film_kit.py at JOB p03b4 推特在国内` 打印该短语的页面秒数。

## film.json

顶层（都可省略，有默认值；省略的嵌套键沿用默认）：

| 键 | 默认 | 说明 |
|---|---|---|
| `palette` | ink/accent/accent2/grey | 任何地方写 `"$ink"` 等取色；`"$roadmap"` 取路线图分组 |
| `areas` | header/strip/band/picture/map（1280×1920） | `[x, y, 宽, 高]`；可加 `picture_portrait/landscape/square` |
| `titles` | `glyph_in`、0.8 秒、84 号、accent2 | 页头标题**随页面出现**（`at` 默认 0），整字逐个出现；写 `at` 才改成跟口播 |
| `lines` | `blur_up`、0.9 秒、60 号、accent | 形式带里的短句，必须写 `at` |
| `drawing` / `picture_drawing` | `{mode: draw, speed: 1.4, plan_side: 540}` / `{sketch: false}` | 前者写进 `job.json`，后者是每张图的默认；图里 `construct: true` 恢复引擎起形 |
| `timing` | lead .25、hook_lead .05、minimum_factor 1.08（首屏 1.02）、hold_min .6 | 绘制窗口：图从页首（或 `from_beat`）起画，实测最短秒数 × 系数后停笔，剩下都是看图时间 |
| `transitions` / `cameras` | chapter=page_turn、quiet=slide、end=fade / still | 屏上写名字或完整对象 |
| `roadmap` | groups、headers、marks…… | 长清单的地图、章节页、进度条；`strip_enter` 默认 `settled`（第二个进度条起，之前的点直接画好） |
| `characters` | — | `[{id, reference}]`，故事板锁定参考图哈希 |
| `plan` | look、neck、drawability_reason | 绘制计划的默认文字（人物头、上衣的笔画组） |
| `pictures` | — | 每张图：布局与绘制设置 + 绘制计划（见下） |
| `screens` | 每个口播页一屏 | 见下 |
| `scripts` | — | 自定义动画脚本 `{名字: JS}`（也可写文件路径） |
| `job` | — | 原样并入 `job.json` 顶层（audio、captions、direction……） |
| `storyboard` | max_hold_seconds 9 | 其余键并入故事板；`explains` 按配方覆盖说明文字 |

**图（`pictures.<名>`）**：`area`、`crop`（像素或 0–1 比例）、`plan_side`、`colouring`、`construct`、`drawing`（其余绘制键，`null` 删除）、`visual`、`characters`、`background`；绘制计划：`composition`、`subjects`（`{"human": id, "cues": […], "omitted": {…}}`、`{"object"|"animal"|"environment": id, "cues": […]}` 或完整对象）、`parts`（`{"id", "groups": [笔画组], "box": [左,上,右,下], 其他键给动作部位}`，或人物简写 `{"person": id, "head": 框, "torso": 框, "left_arm": 框, "right_arm": 框, "lower_body": 框, "groups": {…}}`；最后一个部位接住剩余像素）、`order_exception`、`narrative`（`[[口播原句, 必须看得见的东西]]`）。

**屏（`screens[]`）**：`id`、`page`（取整个口播页）或 `beats`（任意连续句）、`title`、`picture`（名字或 `{picture, visual, weight, from_beat}`）、`actions`（再加图或 `{"text", "x": "center"|数, "y", "size", "color"}` 笔写字）、`overlays`、`transition`、`camera`、`chapter`（组号：笔写组名 + 页头 + 全片地图，自带转场、静止镜头和镜头说明）、`shot`（`[类型, 目的]`，章节页可省）、`context`、`density_exception`、`negative_space_reason`、`focus_note`、`lead`、`hold_min`、`minimum_factor`。

**动效层（`overlays[]`）**，每项按出现的键识别：

- `{"title": "…"}`：页头。可加 `at`、`keywords`、`badge`（条目编号）、`style`、`size`、`visual`（默认 `vNN-title`）、`until`、`exit_at`、`area`/`x`/`y`/`on`。
- `{"line": "…", "at": cue}`：形式带里的短句（默认在 band 顶部）。
- `{"form": 配方, "at": cue, "params": {…}, "area": …, "until": cue}`：任何网页动效配方（table、flow、swimlane、checklist、cards、pulse、rough_chart、number_ticker、map_route、physics、formula……），时长按参数里最晚的时刻自动算。清单项可写 `{"text", "at": cue, "mark": "check"|"cross"|"strike"}`：落笔 = 说到时，打勾 = 0.8 秒后（`mark_at` 可改），划掉用强调色。
- `{"strip": {"group": g}}` / `{"strip": {"item": n}}`：顶部进度条（整组点亮 / 第 n 条）。
- `{"custom": 脚本名或路径, "at": cue, "element": "…", "explains": "…", "moments": [cue…], "params": {…}}`：自定义 SVG 动画，前面自动加上 `add/word/show` 小工具（`svg_kit: false` 关闭），参数里的提示同样换成毫秒。
- `{"annotate": 图名, "box": [u0,v0,u1,v1], "at": cue, "kind": "circle"|"box"}`：圈出画好的图的一部分；`{"callout": "…", "x", "y", "at"}`：气泡；`{"overlay": {…}}`：原样的动效层（其中的提示照样换算）。
- 任一项可加 `"explains": [元素, 讲清什么]` 覆盖故事板里的说明。

故事板自动生成：图是主角，讲解形式与 `context` 是辅助；没有图的页以第一个非导航形式为焦点；进度条标为 `navigation`。

## 最小示例

```json
{
 "palette": {"ink": "#22312B", "accent": "#C4553E", "accent2": "#3E8E6A", "grey": "#8C949C"},
 "characters": [{"id": "youth", "reference": "inputs/art/char-ref.png"}],
 "pictures": {
  "d01-list": {"area": [77, 340, 1126, 900], "plan_side": 600,
   "composition": "年轻人举着一张长到拖地的清单。",
   "subjects": [{"human": "youth", "cues": ["年轻人", "举着长清单"]}, {"object": "list", "cues": ["长清单"]}],
   "parts": [{"id": "list", "groups": ["清单外形", "方框"], "box": [0.66, 0.14, 0.97, 0.95]},
             {"person": "youth", "head": [0.33, 0.02, 0.53, 0.16], "torso": [0.21, 0.15, 0.52, 0.43], "left_arm": [0.5, 0.11, 0.7, 0.36],
              "right_arm": [0.19, 0.17, 0.33, 0.53], "lower_body": [0.24, 0.4, 0.58, 0.92]}],
   "narrative": [["二十九件事", "年轻人举着一张长到拖地的清单"]]}},
 "screens": [
  {"id": "p01", "camera": "still", "picture": "d01-list", "overlays": [
    {"title": "存款不到七十万，请照做？", "keywords": ["照做？"], "size": 92},
    {"line": "29 件事 · 坚持一年", "at": "@p01b2:二十九件事", "accent": "$grey", "until": "@p01b3"},
    {"form": "checklist", "at": "@p01b3:前五条", "visual": "v01-items", "params": {"size": 40, "items": [
      {"text": "1 · 苹果手机", "at": "@p01b3:买一部苹果手机", "mark": "check"}]}}],
   "shot": ["character_action", "钩子：长到拖地的清单"]}],
 "job": {"captions": {"color": "$ink", "stroke_width": 4, "stroke_color": "#F2EFE8", "background": null, "highlight_color": "$accent"}}
}
```

`image-specs.json`：`{"vars": {"YOUTH": "…"}, "character": {"ref", "note", "colour_phrase"}, "style": {"brief": [简报里的画风行], "one_shot", "plate", "colour", "edit", "line_art", "palette"}, "images": {"名": {"page", "use", "shape"（默认按画布：2:3 为 portrait）, "n"（默认 = 预算）, "ref", "plate"（false = 一次成图，如角色参考）, "colour_ref", "narration": [[原句, 画面]], "scene", "notes", "edit", "specks", "width", "seconds"}}, "model", "effort"}`。场景里写 `{YOUTH}` 引用 `vars`。`plate`/`colour`/`edit` 模板可用 `{scene} {n} {compose} {ref} {notes} {erase} {palette} {line_art}`，不写就用内置模板。

`cover.json`：`{"base": "inputs/art/d01-list.png", "width": 720, "top": 790, "paper": "#F2EFE8", "layers": {"title_1": {"text", "font": "hand"|"sans"|{path, index}, "size", "color", "y", "x"(可选，默认居中)}}}`。字体经 `fonts.py` 选（手写链 / 字幕链），不写死系统路径。

## Codex 出图路线

Claude Code 没有生图工具时，`codex_image.py PROMPT OUT --shape portrait --ref 图=说明` 调本机 Codex CLI 的内置生图：

- `stdin` 接 `/dev/null`（否则 codex 会等输入）；提示词参数放在所有 `-i` 之前（`-i` 吃多个值）；只取本次线程目录 `$CODEX_HOME/generated_images/<thread>/` 里的图，并行时不会拿到别人的图；失败自动重试。
- 返回的画布跟着第一张参考图的比例走：每张参考图先用边缘色补边到所要的画幅（竖 2:3、方、横 3:2）再传。
- 模型用 `--model`（默认取环境变量 `CODEX_IMAGE_MODEL`，`""` = codex 自己的默认），推理强度 `--effort`，可执行文件 `--codex` / `CODEX_BIN`。
