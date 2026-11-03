# sweety-handdraw-video · 手绘口播

**给一个主题、一篇文章或完整故事，做成边画边讲的有声视频。**

Agent 负责理解内容、写口播、设计分镜和生成插画；本地工具负责逐笔描线、填色、手写文字、配音对齐与导出。画什么、用什么画风、讲多久，按内容决定。

[![那条死于愤怒的蛇](docs/media/snake-cover.jpg)](https://github.com/threerocks/sweety-handdraw-video/releases/download/v1.2/docs--media--snake-story.mp4)

[看有声样片](#精选样片) · [安装与环境](#安装与环境) · [开始制作](#开始制作) · [全部制作选项](#全部制作选项) · [常见问题](#常见问题) · [关注作者](https://x.com/leo_xiaolei)

> **当前版本：v3.0.1（2026-11-03）**。修复 3.0.0 上色后每块颜色都带斜向笔道的问题：完成色默认平涂，与原图一致，详见 [v3.0.1 发布记录](docs/releases/v3.0.1.md)。3.0 是手绘线的收官版本：先出线稿再上色，笔先起形、按模块描线、用橡皮擦掉起形，再拿马克笔由浅到深上色，墨线有压感、完成色平整干净；水印每页自动避开画面，旁白在封面渐隐时就开口，框里的字按词断行，讲解形式（清单、表格、流程、地图、公式、机制动画）按口播的字出现；一部片只写数据，不再手写脚本（制作工具包）。详见 [v3.0.0 发布记录](docs/releases/v3.0.0.md)；3.0.0 的进展和本版修复的问题见 [#19](https://github.com/threerocks/sweety-handdraw-video/issues/19)。2.0 系列见 [v2.0.2](docs/releases/v2.0.2.md)、[v2.0.1](docs/releases/v2.0.1.md) 与 [v2.0.0](docs/releases/v2.0.0.md) 记录，v1.5 见[其发布记录](docs/releases/v1.5.0.md)。以包内 `VERSION` 和 Git 标签为准。

## 精选样片

### 《那条死于愤怒的蛇》 · 49 秒

从蛇与锯子的冲突，讲到愤怒如何放大伤害。适合先看短片的叙事密度、画面部位的出现顺序，以及旁白和落笔的配合。使用淡彩叙事插画、毛笔手部与无底板字幕。

[![蛇：实际描绘过程](docs/media/snake-drawing.webp)](https://github.com/threerocks/sweety-handdraw-video/releases/download/v1.2/docs--media--snake-story.mp4)

<!-- 在线播放：把 snake-story-720p.mp4 拖进 GitHub 编辑器上传，用生成的 user-attachments 链接替换本行 -->

[下载完整 1080p MP4（7.7 MB）](https://github.com/threerocks/sweety-handdraw-video/releases/download/v1.2/docs--media--snake-story.mp4)

### 《快乐囚禁你的方式》 · 1 分 52 秒

冷色棱角插画，女性手部，连续女声旁白。展示同一条故事内的画面一致性，以及完整旁白先生成、画面随后对齐的制作方式。当前女声暂时保留，听感仍待后续调整；原片音色参考继续保存在本地。本次只修字幕，音轨字节保持不变。

[![快乐囚禁你的方式](docs/media/wolf-cover.jpg)](https://github.com/threerocks/sweety-handdraw-video/releases/download/v1.2/docs--media--wolf-story.mp4)

[![狼：实际描绘过程](docs/media/wolf-drawing.webp)](https://github.com/threerocks/sweety-handdraw-video/releases/download/v1.2/docs--media--wolf-story.mp4)

<!-- 在线播放：把 wolf-story-720p.mp4 拖进 GitHub 编辑器上传，用生成的 user-attachments 链接替换本行 -->

[下载完整 1080p MP4（15.4 MB）](https://github.com/threerocks/sweety-handdraw-video/releases/download/v1.2/docs--media--wolf-story.mp4)

这两条 MP4 均为 1920×1080、30fps，由 v1.2 制作（逐笔绘制、分段配音等 v1.3 改动不在其中），保留制作时的预告水印；水印文字不代表当前发布时间。蛇片 v1.2 原版旁白带 235 Hz 持续嗡声，v1.3 门禁会拦下这种声音。动图是成片的无声节选（随仓库提供，国内访问不依赖 Release 下载）；判断音色和节奏请下载完整视频。GitHub Release 附件只能下载，不能在页面内播放。

<details>
<summary>更多题材：概念讲解与较长叙事</summary>

**《墨菲定律》**：约 2 分 05 秒，8 个画面、24 段口播，解释准备如何降低失误的代价。使用淡彩速写、女性碎钻长甲手及笔杆文字。它是公开 v1.0.0 的历史样片，不代表最新的连续旁白流程。

[![墨菲定律](docs/media/murphy-cover.jpg)](https://github.com/threerocks/sweety-handdraw-video/releases/download/v1.0.0/murphy-law.mp4)

<!-- 在线播放：把 murphy-law-720p.mp4 拖进 GitHub 编辑器上传，用生成的 user-attachments 链接替换本行 -->

[下载《墨菲定律》1080p MP4（14.4 MB）](https://github.com/threerocks/sweety-handdraw-video/releases/download/v1.0.0/murphy-law.mp4) · [查看《不是你在努力，是有人想让你努力》的作者展示帖](https://x.com/leo_xiaolei/status/2109880459664752867)

较长样片用于展示叙事跨度，不代表短观点也应做成长视频。

</details>

## 它适合做什么

| 输入 | 交付方向 |
|---|---|
| 一个主题 | 整理讲解路线，设计生活场景、机制与行动建议 |
| 一篇文章或参考资料 | 保留重要事实、数字和限定，改成适合听与看的口播 |
| 完整故事或定稿 | 按原文制作；要求保留原文时，不擅自重写 |
| 已认可的图、声音和稿件 | 复用素材，针对字幕、手部、画面或时间线进行修订 |

适合知识讲解、故事寓言、观点表达、产品原理和教学内容。优势在于把口播、画面与绘制过程放在同一条制作流程里：

- **画风跟随内容**：21 个库内入口，也可让模型独立设计，或提供自定义方向。
- **声音先行、按段生成**：口播按语义段落生成，每段过嗡声 / 底噪 / 削波门禁，拼接停顿实测，再按时间戳安排画面与字幕。
- **像人一样画，而且有条理**：只描真实墨线，深色填充不再以噪点提前出现；复杂画面按对象逐个画完，对象内先外轮廓、再从已画线条上"长出"内部线条；马克笔就近之字形上色；无轮廓画风由笔刷按部位揭示；汉字按标准笔顺书写。
- **先算预算再出图**：分镜按旁白算出每幅画能画几秒、最多几笔（按每幅画的画法算速度：默认起形 + 马克笔约每秒 3.9 笔，按部位上色、不起形约每秒 6 笔，与图幅基本无关），图按预算设计，不靠压缩时间硬塞；`layout_plan.py` 给出平台安全区版式并实测指定尺寸的最少绘制时间。
- **可以定位修改**：保留脚本、素材、声音、时间戳和检查记录，修字幕不必重新生图。
- **运行与观感分别检查**：文件能播放、声音清楚、剧情画对，是不同检查项。

它仍是需要 Agent、图像生成与本地运行环境的制作 Skill；成片不使用外部图生视频模型。当前主渲染实现是 Python / Pillow / SciPy / FFmpeg。

## 导演如何参与制作

内置 `omniailab-ai-director` 的导演方法，用在创作规划、分镜和最终审片。模型要把判断落实到具体画面与时间线：

| 导演判断 | 在手绘口播中的执行方式 |
|---|---|
| 受众与目标 | 明确观众看完应理解什么，决定切入点与信息密度 |
| 叙事路线 | 故事保留行动与因果；原理可用对照、变化或操作过程；用户定稿与原文要求优先 |
| 分镜与画面作用 | 每幅图说明它比直接显示这句话多解释了什么，列出必须可见的动作、接触与结果 |
| 构图和一致性 | 确定主次、角色与道具参考、色彩、留白，避免跨画面随意漂移 |
| 节奏和声画关系 | 一个完整意思讲完再换场；按实际旁白决定描线、填色、关键字和停留时间 |
| 成片审查 | 查看缺失的剧情动作、重复信息、遮挡、阅读时间与声音对应，失败时修改受影响部分 |

可以在需求里指定导演处理方式，例如“直接从冲突开始”“按老师讲题的顺序做对照”“关键结果出来后留时间看清”“保留原文结尾”。未指定时，模型根据内容规划，不硬套固定页数和每页时长。

导演判断写进创作方案，由本地渲染执行；不是一个独立运行的导演模型，也不自动启用上游影视生成平台。更广泛的导演资料按需取用，不能等同于本项目已验证了上游的全部影视能力。详见 [创作与分镜](skills/sweety-handdraw-video/references/story-design.md) 与 [依赖分工](skills/sweety-handdraw-video/references/dependencies.md)。

## 安装与环境

### 1. 安装 Skill

本轮支持 **macOS 与 Linux**，宿主为 **Codex 与 Claude Code**；Windows 暂不承诺。具备 Node.js / npm 时一条命令装到两个宿主：

```bash
npx skills add threerocks/sweety-handdraw-video --skill sweety-handdraw-video --agent claude-code codex --global --copy --yes
```

安装位置：Claude Code 为 `~/.claude/skills/sweety-handdraw-video`，Codex 为 `~/.agents/skills/sweety-handdraw-video`（该命令写入的通用目录）。只装一个宿主时去掉另一个 `--agent` 值；不加 `--global` 则装进当前项目（Claude Code 为 `.claude/skills/`）。此命令安装仓库当前公开内容。需要固定本版时，使用 [v3.0.1 安装包](https://github.com/threerocks/sweety-handdraw-video/releases/download/v3.0.1/sweety-handdraw-video-v3.0.1.zip)；[全部版本](https://github.com/threerocks/sweety-handdraw-video/releases)。下载同页的 `SHA256SUMS.txt`，执行 `shasum -a 256 -c SHA256SUMS.txt` 校验。更新仓库不会自动更新已安装的副本；替换完整 Skill 目录后，读取其中的 `VERSION`，本版应为 `3.0.0`。

手动安装：将包内的 `sweety-handdraw-video` **整个文件夹**放入宿主的 Skills 目录，例如 `~/.claude/skills/sweety-handdraw-video`（Claude Code）或 `~/.codex/skills/sweety-handdraw-video`（Codex）。更新前备份旧目录；仅复制 `SKILL.md` 会缺少脚本、素材和内置依赖。手部、画风预览与试听已随包提供（约 38 MB），装好即可离线使用，不再依赖首次下载。

在 Claude Code 中直接说“用 sweety-handdraw-video 把……做成手绘视频”或输入 `/sweety-handdraw-video`；在 Codex 中使用 `$sweety-handdraw-video`。

### 2. 准备运行环境

```bash
bash ~/.claude/skills/sweety-handdraw-video/scripts/setup.sh          # 或 Codex 安装目录下的同名脚本
bash ~/.claude/skills/sweety-handdraw-video/scripts/setup.sh --check  # 只检查，不安装
```

`setup.sh` 安装或检查 FFmpeg（macOS 用 Homebrew；Linux 用 apt，同时装 `fonts-noto-cjk` 与 `fontconfig`，并给出 dnf / pacman 对应命令），在 Skill 目录之外的 `~/.local/share/sweety-handdraw-video/venv` 建 Python 虚拟环境并安装 `scripts/requirements.txt`，然后核验安装包并报告实际选用的字体。它不会下载配音模型；只有显式传入 `--voxcpm`（Apple Silicon）才会下载约 3.2 GB 的 VoxCPM2-8bit。

| 条件 | 用途与边界 |
|---|---|
| 能读取 Skill、执行本地命令的 Agent | Codex 与 Claude Code。两者都已在 macOS 上完成整片实测（Claude Code 通过本机 Codex CLI 出图） |
| 可用的图像生成能力 | Codex 内置生图；Claude Code 需要接入生图工具 / MCP，或由用户提供插画。脚本无法探测宿主能力：没有生图能力时必须停下说明，不能交付只有文字的视频 |
| Python 3.9+ | 本地绘制与合成；依赖 Pillow、NumPy、SciPy、requests、fontTools，由 `setup.sh` 装入独立虚拟环境 |
| FFmpeg 与 ffprobe | 合成、编码、解码与音视频检查 |
| 中文字体 | 手写字、字幕、笔杆文字与水印分别选字体，并逐字核对字形覆盖；缺字时构建失败并列出缺失字符与已尝试字体。macOS 优先 Hannotate，Linux 使用 fontconfig 与 Noto CJK；不随包分发系统商业字体 |
| 一种可用配音路径 | 已配置的本地 VoxCPM2、兼容的 VoiceStudio，或已核对的完整旁白与时间戳 |
| Node.js / npm（可选） | 仅用于上面的安装命令；手动安装不依赖它 |

图片隐私清理已改为纯 Python（`scripts/clean_image.py`），不再需要 Bun 或 ExifTool。制作前运行 `python scripts/handdraw_video.py doctor JOB`：分别报告“能否渲染”“能否生成旁白”与生图能力提示，并对每个缺失项给出修复命令。

### 3. 选择配音路径

**音色与配音后端分开选择。** 六款内置音色提供真实的参考音频与同文试听，后端负责把本次正文生成声音。也可以导入自己的参考，或沿用已验收的本机主力声音。

| 路径 | 需要准备 | 已验证范围 |
|---|---|---|
| **VoxCPM2 本地适配器** | Apple Silicon Mac、独立 MLX 环境、`mlx-audio` / `mlx` / `soundfile`、已下载模型与真实路径 | 本项目使用 VoxCPM2-8bit MLX；当前适配器不是 Windows / NVIDIA 部署方案 |
| **VoiceStudio 本机服务** | 能提供兼容 HTTP API 的本机服务，已配置 TTS 引擎 | 默认 `http://127.0.0.1:3900`；已对接 `mlx-audio / Qwen3-TTS-12Hz-1.7B-Base-bf16` |
| **现成整篇旁白** | 已检查的音频及其时间戳 | 可离线编排和渲染；本次无需重新合成声音 |
| **火山豆包 / Gemini TTS（可选）** | 用户自己的密钥，放在环境变量 | 按官方文档实现，只做了模拟 HTTP 测试，本仓库未真实调用；不是默认，本地失败时也不自动改用 |

VoiceStudio 需要 `/engines/tts`、`/generate` 及可选 `/transcribe` 接口。任意同名软件不一定兼容。所有生成路径都按语义段落逐段提交（每段不超过约100字，只在句末切），逐段核对杂音、语速和漏读，不通过换种子重试，再拼成母带并自动写时间戳。

**为什么保留这几条路径？** 本项目引入 VoxCPM2 是为了改善长段旁白的自然度与参考音色延续，并在作者的 Mac 上实际测试 MLX 8-bit 路径；当前新版女声暂用，作者仍认为听感有差距，后续会继续调整。VoiceStudio 则保留已经可用的本机服务，避免要求所有用户更换声音环境。已有满意的整篇旁白时，直接复用可以省去再次合成。这里没有宣称一个后端在所有声音、稿件和机器上都更优。

本地合成便于保留参考、参数和生成记录，这条路径无需另接云端 TTS 账号；代价是需要自行准备模型与计算资源。VoiceStudio 是服务接口，Qwen3-TTS / VoxCPM2 才是模型，不能把服务名称、音色名称和模型名称混为一谈。

模型权重不在 Skill 包内，运行脚本不会自动购买服务、下载模型或切换全局声音引擎。带噪参考可能额外需要人声分离或降噪；HTDemucs / DeepFilterNet3 是本次女声修复使用的可选工具，不是所有用户都必须安装的基础依赖。

**硬件实测**：当前作者环境为 M3 Max、36GB 内存。使用的 VoxCPM2-8bit 权重约 3.23GB；权重大小不等于运行内存需求。本项目尚未测定 8GB / 16GB 设备的最低可用条件，也不据此给出全机型耗时承诺。

详细接入参数见 [本地运行环境](skills/sweety-handdraw-video/references/local-runtime.md) 与 [环境检查表](docs/environment.md)。

## 开始制作

先用几十秒的短主题确认本机环境与画风，再处理长稿。可直接发给 Agent：

```text
使用 $sweety-handdraw-video，把“为什么越生气，越容易做错决定”
做成一段面向普通成年人的手绘讲解。先讲清楚一个机制，控制在约一分钟。
画风、手部、声音和字幕由你根据内容选择，不加广告和水印。
```

```text
使用 $sweety-handdraw-video，把下面的完整故事做成视频。
正文按原文，不改写。使用自定义淡彩速写、毛笔手部、无底板字幕。
声音使用我已确认的本机参考，不加背景音乐。

【粘贴故事】
```

```text
使用 $sweety-handdraw-video，把这篇文章做成手绘讲解视频。
保留数字、来源和适用条件。画面用风格 17，手部用仅画笔。
笔杆文字：@leo_xiaolei。水印：@leo_xiaolei。
先检查一个代表场景，确认文字可读后完成整片。

【粘贴文章】
```

未指定时采用简体中文、竖屏 1280×1920（2:3，2.0 起；此前为 9:16）、30fps。时长由实际口播决定；“约一分钟”是创作目标，不靠拉长空白凑时间。默认有 1 秒封面，前 0.5 秒静止、后 0.5 秒渐隐。

开始时提供六项选择：画风、手部、声音、笔杆广告、水印、字幕。已明确的项目跳过；每个未选项目最多等 60 秒，随后采用模型根据内容拟定的方案。说“全部自动选择”会直接继续。广告与水印默认不加。

## 全部制作选项

| 项目 | 可以怎么选 |
|---|---|
| 内容 | 主题、资料、文章、完整故事；明确要求原文时保留原文 |
| 导演与叙事 | 指定受众、切入点、叙事顺序、信息密度、强调与收尾；也可委托模型根据内容规划 |
| 画风 | 指定库内编号、模型从库内匹配、模型独立设计、自定义描述；每种画风标明逐笔绘制或揭示 |
| 手部 | 六款手臂预设、仅画笔；支持有定位参数的自定义素材 |
| 声音 | 逼真男声 / 逼真女声（需本机原片参考）、四款随包合成声音、声明本人或授权的自定义声音、现成旁白；云端 TTS 只用你自己的密钥 |
| 笔杆文字 | 最多 16 个 Unicode 字符；空白即不加；长文字需检查实际可读性 |
| 水印 | 最多 180 字符，支持换行；低对比、缓慢浮动；空白即不加 |
| 字幕 | 默认白字黑边；`quiet` 无底板 / `plate` 有底板；全程只允许一套字幕 |
| 正文文字 | 手写关键句与插画分别编排；字形四边预留余量 |
| 画布与帧率 | 默认竖屏 1280×1920（2:3）、30fps；其他尺寸需重新检查构图、字号与遮挡 |
| 节奏 | 根据旁白时间戳安排描线、填色、文字、抬笔和停留；可提出快慢要求 |
| 音乐与音效 | 默认只在手写文字时有与笔尖同步的柔和笔声，描线和上色静音（可关，也可设为画画也发声）；背景音乐不自动添加，可提供自有音乐或选内置合成配乐 `builtin:calm/warm/curious/focused/tense`，构建自动循环接续、淡入淡出并随旁白压低 |
| 封面与片尾 | 1 秒封面；默认片尾停留 0.45 秒；结束语须写入完整旁白 |
| 修订 | 可复用已通过的图和声音；相关素材改变后重新检查受影响部分 |

以上是制作请求可以表达的选择，不代表每一项都有独立可视化滑块。详细字段见 [制作选项](skills/sweety-handdraw-video/references/production-options.md)、[字幕与片尾](skills/sweety-handdraw-video/references/subtitle-design.md)。

### 21 个库内画风入口

| 选择方式 | 依据与执行 |
|---|---|
| 直接选编号 | 用户看真实效果预览并指定，沿用对应完整配方 |
| 库内自动匹配 | 模型根据内容从库内选择，记录理由与完整配方；优先逐笔绘制、上色的 4、5、17、19、20 |
| 模型独立推理 | 独立设计线条、颜色、明暗、材质与背景，不以某个编号兜底 |
| 自定义描述 | 用户描述是画风依据；模型只补足可执行细节，不能因内容偏好擅自换风格 |

库内为 **20 个编号风格 + 3.1 稳定变体**，每项附包内真实效果预览（选择器离线显示）与完整配方。它们是静态美术方向，不表示每一种都已经做过相同强度的视频验证。

**绘制方式**：1、4、5、17、19、20 有清晰墨线轮廓和平涂，**逐笔绘制**（先描线、再上色）；其余画风没有可描的轮廓，用**揭示**（手带宽软笔刷按部位扫出成图）。选择器在每张卡片上标明。模型推理或自定义的画风按同一标准判定，写进 `style/resolved.json` 的 `drawing_mode`。没指定画风时，模型默认从上色的 4、5、17、19、20 里按内容选，纯黑白的 1 号要写明理由。揭示放弃了逐笔绘制，只能由用户亲自选（点揭示卡，或听过“只能刷出、不能画出”的说明后同意）；模型不会因超时、委托或某张图画不出来而改用揭示。

| 编号 | 画风 | 编号 | 画风 |
|---|---|---|---|
| 1 | 极简黑白线条讲解漫画 | 11 | 二维水彩 / 复古动画概念稿 |
| 2 | 蜡笔童涂 | 12 | 暖光童画 |
| 3 | 吉卜力风 | 13 | 北欧纸雕 |
| 4 | 小豆人黑色涂鸦信息图 | 14 | 北欧绘本水粉 |
| 5 | MS Paint 烂涂鸦 | 15 | 大鼻软偶 |
| 6 | 圆珠笔单线涂鸦 | 16 | 聚光水粉立绘 |
| 7 | 蜡笔实拍纸张 | 17 | 墨线绘本 |
| 8 | 水墨写意 | 18 | 暖色扁平绘本 |
| 9 | 复古像素 | 19 | 圆头红线极简童画 |
| 10 | 情绪叙事淡彩速写 | 20 | 暖黄墨线情绪小剧场 |
| 3.1 | 蜡笔童涂·潦草自画稳定变体 | | |

不想受风格库限制时，选择“模型推理产出”或给自定义描述。模型仍须写出画风方案、审查实际插画，再制作绘制路径。完整定义见 [STYLES.md](skills/sweety-handdraw-video/dependencies/hand-drawn-styles/STYLES.md)。


<details>
<summary>展开 21 种实际画风预览</summary>

<table>
<tr><td align="center" width="33%"><img src="skills/sweety-handdraw-video/assets/style-previews/1.jpg" width="220" alt="1 极简黑白线条"><br>1 · 极简黑白线条</td><td align="center" width="33%"><img src="skills/sweety-handdraw-video/assets/style-previews/2.jpg" width="220" alt="2 蜡笔童涂"><br>2 · 蜡笔童涂</td><td align="center" width="33%"><img src="skills/sweety-handdraw-video/assets/style-previews/3.jpg" width="220" alt="3 吉卜力风"><br>3 · 吉卜力风</td></tr>
<tr><td align="center" width="33%"><img src="skills/sweety-handdraw-video/assets/style-previews/3.1.jpg" width="220" alt="3.1 潦草自画稳定变体"><br>3.1 · 潦草自画稳定变体</td><td align="center" width="33%"><img src="skills/sweety-handdraw-video/assets/style-previews/4.jpg" width="220" alt="4 小豆人涂鸦信息图"><br>4 · 小豆人涂鸦信息图</td><td align="center" width="33%"><img src="skills/sweety-handdraw-video/assets/style-previews/5.jpg" width="220" alt="5 MS Paint 烂涂鸦"><br>5 · MS Paint 烂涂鸦</td></tr>
<tr><td align="center" width="33%"><img src="skills/sweety-handdraw-video/assets/style-previews/6.jpg" width="220" alt="6 圆珠笔单线涂鸦"><br>6 · 圆珠笔单线涂鸦</td><td align="center" width="33%"><img src="skills/sweety-handdraw-video/assets/style-previews/7.jpg" width="220" alt="7 蜡笔实拍"><br>7 · 蜡笔实拍</td><td align="center" width="33%"><img src="skills/sweety-handdraw-video/assets/style-previews/8.jpg" width="220" alt="8 水墨写意"><br>8 · 水墨写意</td></tr>
<tr><td align="center" width="33%"><img src="skills/sweety-handdraw-video/assets/style-previews/9.jpg" width="220" alt="9 复古像素"><br>9 · 复古像素</td><td align="center" width="33%"><img src="skills/sweety-handdraw-video/assets/style-previews/10.jpg" width="220" alt="10 淡彩速写"><br>10 · 淡彩速写</td><td align="center" width="33%"><img src="skills/sweety-handdraw-video/assets/style-previews/11.jpg" width="220" alt="11 二维水彩"><br>11 · 二维水彩</td></tr>
<tr><td align="center" width="33%"><img src="skills/sweety-handdraw-video/assets/style-previews/12.jpg" width="220" alt="12 暖光童画"><br>12 · 暖光童画</td><td align="center" width="33%"><img src="skills/sweety-handdraw-video/assets/style-previews/13.jpg" width="220" alt="13 北欧纸雕"><br>13 · 北欧纸雕</td><td align="center" width="33%"><img src="skills/sweety-handdraw-video/assets/style-previews/14.jpg" width="220" alt="14 北欧绘本水粉"><br>14 · 北欧绘本水粉</td></tr>
<tr><td align="center" width="33%"><img src="skills/sweety-handdraw-video/assets/style-previews/15.jpg" width="220" alt="15 大鼻软偶"><br>15 · 大鼻软偶</td><td align="center" width="33%"><img src="skills/sweety-handdraw-video/assets/style-previews/16.jpg" width="220" alt="16 聚光水粉"><br>16 · 聚光水粉</td><td align="center" width="33%"><img src="skills/sweety-handdraw-video/assets/style-previews/17.jpg" width="220" alt="17 墨线绘本"><br>17 · 墨线绘本</td></tr>
<tr><td align="center" width="33%"><img src="skills/sweety-handdraw-video/assets/style-previews/18.jpg" width="220" alt="18 暖色扁平绘本"><br>18 · 暖色扁平绘本</td><td align="center" width="33%"><img src="skills/sweety-handdraw-video/assets/style-previews/19.jpg" width="220" alt="19 圆头红线童画"><br>19 · 圆头红线童画</td><td align="center" width="33%"><img src="skills/sweety-handdraw-video/assets/style-previews/20.jpg" width="220" alt="20 暖黄墨线小剧场"><br>20 · 暖黄墨线小剧场</td></tr>
</table>

这些预览展示静态画风；实际绘制节奏以成片为准。

</details>

### 六款手臂与仅画笔

![六款手部预设](docs/media/hand-presets.jpg)

| 预设 | 标识 |
|---|---|
| 深蓝袖马克笔 | `navy` |
| 女性柔粉碎钻长甲 | `diamond-woman` |
| 儿童彩铅 | `child-pencil` |
| 上官婉儿灵感毛笔 | `waner-brush` |
| 白衬衫中性笔 | `teacher-pen` |
| 针织袖钢笔 | `cardigan-fountain` |
| 仅画笔，不显示手臂 | `pen-only` |

手臂素材按笔尖位置与手腕距离校准，前臂裁切边留在画外。`pen-only` 是 `navy` 同一支笔去掉手臂，握笔角度不变，阴影从笔尖起，看起来仍是被握着在纸上画。自定义素材需要提供笔尖定位等参数，不能将任意图片直接视为已经适配。笔杆广告只放在笔杆，不自动加品牌或版权声明。

### 笔杆广告与浮动水印

两个独立可选项，默认均为空。样片中的作者宣传文字不属于安装后的强制内容。

| 选项 | 位置与用途 | 规则 |
|---|---|---|
| 笔杆广告 | 跟随画笔，适合品牌短名或账号 | 最多 16 个字符；保留原文；长文字优先宽笔杆并检查实际可读性 |
| 浮动水印 | 位于画面内，用于账号、归属或简短发布信息 | 最多 180 字符，可换行；默认低对比、缓慢移动，避开底部字幕 |

例如：“笔杆写 @leo_xiaolei；水印写『手绘口播 · @leo_xiaolei』，淡一点，避开字幕。”也可以明确说“不加广告和水印”。

### 声音目录与参考

选择器和文字引导使用相同名称与顺序。试听文件随包提供，可离线播放。

| 名称 | 试听 | 生产参考 |
|---|---|---|
| 逼真男声 | [《那条死于愤怒的蛇》节选](skills/sweety-handdraw-video/assets/voices/previews/realistic-man.wav) | 沿用该片原始参考；公共安装需自行导入 |
| 逼真女声 | [《快乐囚禁你的方式》节选](skills/sweety-handdraw-video/assets/voices/previews/bilibili-outro.wav) | 沿用当前原片提取人声；公共安装需自行导入 |
| 磁性解说男声（默认） | [同文试听](skills/sweety-handdraw-video/assets/voices/previews/magnetic-narrator.wav) | 随包合成参考，可直接生成 |
| 儒雅青年 | [同文试听](skills/sweety-handdraw-video/assets/voices/previews/scholarly-young-man.wav) | 随包合成参考，可直接生成 |
| 温柔知性女声 | [同文试听](skills/sweety-handdraw-video/assets/voices/previews/gentle-intellectual-woman.wav) | 随包合成参考，可直接生成 |
| 悬疑故事解说 | [同文试听](skills/sweety-handdraw-video/assets/voices/previews/suspense-storyteller.wav) | 随包合成参考，可直接生成 |

四款合成声音由本机 VoxCPM2 仅凭文字描述设计，没有使用任何真人、主播或平台音色作参考，风格名只描述听感方向。也可以添加任意数量的自定义声音：本人录音或已获授权的声音，导入时声明授权，参考须通过杂音门禁。所有声音当前都待听感验收。

两款逼真声音播放对应成片节选。成片节选只用于试听，不能反过来当作原片参考继续克隆。逼真女声目前暂用，听感仍待调整；改名不代表新增听感验收。

作者本机已保存两款逼真声音的原始参考，可以直接选择。公共安装包不含这些录音：未导入时可以试听，但不能选择它们生成。导入方法见[本地声音运行](skills/sweety-handdraw-video/references/local-runtime.md)。个人目录和任务目录只能补充目录内各项的参考，不能把其他音色或旧名称加回选择器。

可直接指定上表名称或自定义声音，也可委托模型根据稿件选择。已有本机默认声音时优先沿用，本次明确选择优先。声音参考、逐字稿与生成参数分别保留；更换后端或稿件后仍需试听。

## 从输入到成片

1. **确定讲什么。** 明确受众、原文约束、关键事实和讲解路线，避免把简单观点拖成长片。
2. **先把声音做完整。** 写完整口播稿，按语义段落逐段生成并过门禁，拼成母带并自动对齐。
3. **用分镜算预算，再出图。** 按旁白算出视觉数量、每幅可画秒数和复杂度上限；美术指导后生图，实际检查关键剧情、接触关系和缩略图可认度。
4. **用声音安排画面。** 按时间戳描线或揭示插画、按笔顺写关键句、填色、排字幕与换场。
5. **检查代表场景，再导出全片。** 工程检查（含声音门禁、字幕同步）与实际看图、听音分开记录，失败项修复后再交付。

交付包含 **MP4、封面 PNG、SRT、口播稿与检查记录**；制作目录保留素材、配置与时间戳，方便追溯和修订。MP4 已写入 AIGC 隐式标识元数据；发布到平台时仍须按平台要求声明“AI 生成”。

## 常见问题

**一条视频要多久、多少钱？**

作者此前对一条约两分半样片的回答是“差不多 10 分钟”；在自己的 200 美元订阅环境中，观察到额度使用“不到 1%”。这是特定作品与账号的自述，未做统一计量，不能换算成固定单价或服务承诺。首次装环境、模型下载、排队、出图重试、配音修复和人工检查都会改变耗时。[耗时原回复](https://x.com/leo_xiaolei/status/2109590999672725537) · [额度原回复](https://x.com/leo_xiaolei/status/2109591088327749905)

开源不等于全链路免费。模型宿主与生图能力可能收费；本地配音仍消耗硬件、电力和时间。当前成片流程不额外依赖按视频秒数收费的图生视频服务。

**改一句话，要全部重做吗？**

只改字幕、笔杆文字、水印或画面布局，可复用未变化的图和声音，重新渲染并检查。修改口播正文时，只重新生成文字变了的语义段落，分段位置不变时其余段落直接复用，再重新拼接母带、对齐并更新受影响的时间线；换段处的语气衔接仍需试听。

**能稳定做系列、批量出片吗？**

同一系列可以固定画风配方、生图端、角色锚点、手部与声音参考，并复用已认可素材。仍可能出现角色、剧情或音色漂移，因此每条都要检查。当前不是承诺任意输入稳定通过的无人审核批量系统。

**支持哪些系统和宿主？Windows / Gemini 能用吗？**

本轮支持 macOS 与 Linux、Codex 与 Claude Code。完整成片流程实测来自 Apple Silicon Mac + Codex；Linux 由 CI 与 Ubuntu 容器验证安装、字体发现、回归与离线短片合成，本地 TTS 仍以 Apple Silicon 为主。Claude Code 需要接入生图工具 / MCP 或由你提供插画。Windows 与 Gemini 等其他组合暂不承诺。运行 `bash scripts/setup.sh --check` 与 `doctor` 先确认执行、生图、字体与配音能力。

**为什么没有背景音乐，仍要检查杂音？**

参考音频可能带配乐或噪声，声音合成也可能产生伪影。不添加音乐只说明没有额外混音轨，不证明旁白干净。程序指标和 ASR 不能代替听感验收。

**会按规范汉字笔顺写吗？有橡皮擦转场吗？**

会。GB2312 范围内的汉字按标准笔顺逐笔书写（笔画数据来自 hanzi-writer-data，Arphic 公共许可）；其余字符按字形轮廓描线。v1.5 起有板擦擦黑板转场（`transition: {"type": "erase"}`）；2.0 增加橡皮擦（`"style": "rubber"`，会留下碎屑）、四个方向的翻页、撕纸、毛刷刷出等，全部转场见[动效目录](skills/sweety-handdraw-video/references/motion-catalog.md)。

**发布到平台要注意什么？**

成片 MP4 写入了 AIGC 隐式标识（`AIGC` 元数据：生成标记、制作工具和作品 ID）。这只是文件里的隐式标识，发布时仍须按平台规则勾选或声明“AI 生成”。声音、人物和第三方内容的使用权利需要自行确认。

## 内置依赖、开发与验证

安装目录只注册一个 Skill。以下五项以固定快照内置，不需要分别安装；运行时不会自动追踪上游更新：

| 内置项目 | 用途 |
|---|---|
| `sweety-image-art-direction` | 主体、接触关系、构图、比例与观看主次 |
| `hand-drawn-styles` | 风格配方、参考锚点与预览 |
| `omniailab-ai-director` | 导演规划、叙事、声画关系与终审方法 |
| `huashu-art-motion` | 适用题材下的艺术动效方法 |
| `sweety-image-privacy` | 图像隐私元数据清理与审计；本项目随后移除模拟相机字段 |

来源、许可证和固定快照见 [依赖说明](skills/sweety-handdraw-video/references/dependencies.md) 与 [第三方声明](THIRD_PARTY_NOTICES.md)。

```bash
python -m unittest discover -s skills/sweety-handdraw-video/scripts -p 'test*.py' -v
python skills/sweety-handdraw-video/scripts/smoke_test.py --output /tmp/handdraw-smoke-new
python3 tools/package.py --output dist
python3 tools/release.py check
python3 -m unittest discover -s tools -p 'test*.py' -v
```

v3.0.1 修复上色条纹，见 [发布记录](docs/releases/v3.0.1.md)；v3.0.0 的变更、前后对比与验收边界见 [其记录](docs/releases/v3.0.0.md)；v2.0.2 见[其记录](docs/releases/v2.0.2.md)，v2.0.1 见[其记录](docs/releases/v2.0.1.md)（包含未公开发布的 [v2.0.0](docs/releases/v2.0.0.md)），v1.5.0 见[其记录](docs/releases/v1.5.0.md)；v1.4.1 新增发布一致性和实际画面范围检查，见 [v1.4.1 记录](docs/releases/v1.4.1.md)。标签触发发布工作流，先完成 macOS / Linux 检查，再从标签提交打包上传 ZIP 与校验文件；稳定标签不移动。

v1.3 记录：158 项回归通过（1 项平台用例跳过）、离线冒烟短片通过、安装包 843 个文件、49.4 MB 通过核验；蛇故事以 v1.3 流程端到端重做（揭示模式），软件工具片以修复后的引擎重做（18 幅插画全部逐笔绘制、无淡入，终帧与原图一致）；2026-10-17 增补宽墨线角点覆盖、混音电平、数字比较与填充最短时长修复，并新增配乐自动准备。相应成片工程检查、声音门禁与字幕同步均通过。该结果不能替代视频观感、故事事实、听感或学习效果验收。更多记录见 [验证范围](docs/validation.md)。

v1.2 封板包含整篇连续旁白、纯旁白默认、剧情检查与片尾保护、原片参考导入、声音来源追踪、可选离线降噪、本机主力声音目录，以及字幕唯一性和素材缓存检查。此前本地 1.3/1.4/1.4.1 为未公开的开发迭代号；公开标签按本次指定固定为 v1.2。

## 致谢与许可

主项目代码与文档采用 [MIT](LICENSE)。内置项目保留各自的版权与许可证；AI 生成手部、参考音频等素材不因此获得真人身份或官方角色授权。第三方内容与声音的使用仍需自行确认权利。

感谢 [花叔的 huashu-art-motion](https://github.com/alchaincyf/huashu-art-motion)、OmniAiLab AI 导演、手绘风格库与美术指导提供的方法，也感谢评论区关于停顿、字幕、女声、改稿和运行环境的具体反馈。

觉得有用，给仓库点一个 Star；制作结果与问题欢迎附上版本、运行环境和可复现片段提交 Issue。演示和更新见 [@leo_xiaolei](https://x.com/leo_xiaolei)。

## v1.2 素材与字幕规则

v1.3 起手部、画风预览和试听重新随包提供（原始字节，SHA-256 与清单一致），装好即可离线使用；文件缺失或损坏时才依次从镜像（`HANDDRAW_MIRROR` → jsDelivr → GitHub Release）修复并校验。README 图片随仓库提供，完整 1080p 成片仍是 Release 下载。Netlify 因额度耗尽已停用，内置画风库的参考图链接改为 jsDelivr 固定上游提交。原片声音参考仍只存本地。详见[托管与缓存](skills/sweety-handdraw-video/references/hosted-assets.md)。

字幕只由一个图层负责。禁止重复事件、额外内嵌字幕轨和同目录自动加载字幕；编辑文件放入 `editing/`，预览页不挂字幕轨。最终实际画面仍须检查，不能以程序检查替代观看。

### 每次出图都先做美术指导

v1.2.1 增加简报和实际提示词的非空检查、内容哈希绑定，以及 7 项针对性回归。封面、正文、角色、道具、新手部、候选、编辑和返工均须先应用内置 `sweety-image-art-direction`；缺少指导记录、空文件、提示词改变、图片改变或视觉审查未通过都会停止生产。它约束流程与证据，不能代替模型实际看图。
