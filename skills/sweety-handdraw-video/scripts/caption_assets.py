"""Typeset quiet subtitles independently of the hand-drawn artwork font.

Lines and cues break between words: never inside a Latin word, a number with its unit, a keyword or a name joined by
a middle dot (托尔斯滕·鲍尔), never before
closing punctuation or a particle, never after an opening bracket (kinsoku); clause punctuation is the preferred break,
then particles (after) and conjunctions/prepositions (before); two-line captions are balanced. English breaks only at
spaces. jieba refines Chinese word boundaries when it is installed; the built-in rules work without it.
On vertical feeds the caption sits above the platform UI band (account, description, music, right-hand buttons).
"""
from pathlib import Path
import math
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor

MAX_CUE_CHARS = 24
HIGHLIGHT_COLOR = '#FFD43B'
# Full-screen 9:16 feed UI (Douyin, Channels/视频号, Xiaohongshu, TikTok). No platform publishes a pixel map; measured
# working margins on 1080x1920 put the account/description/music/progress block in the bottom 18-22 % (~384 px), the
# avatar/like/comment/share rail in the right 12-13 % between ~42 % and ~80 % of the height, tabs in the top ~7 %.
PLATFORM_UI = {'top': .07, 'bottom': .20, 'left': .04, 'right': .125, 'rail': (.42, .80)}
UI_GAP = .02  # caption bottom edge sits this fraction of the height above the bottom band
USE_JIEBA = True

INF = math.inf
NO_START = set('，。、；：！？）」』】》〉”’…·—～%‰,.;:!?)]}>')
NO_END = set('（「『【《〈“‘([{<')
NAME_JOINERS = set('·・‧')
CLOSERS = set('）」』】》〉”’)]}')
CLAUSE_END = set('，。、；：！？…,;:!?')
SENTENCE_END = set('。！？!?…')
DROPPABLE = set('，。、；：,;:')
FULLWIDTH_DROPPABLE = set('，。、；：')
PARTICLES = set('的了着过地得们吗呢吧啊呀嘛么')
LEADS = set('和与及或而但且并把被从对向给让跟比为在将由往朝于用以按据趁')
ADVERBS = set('就也都还又才再却便已曾正很最更太')
NEGATIONS = set('不没')
DEMONSTRATIVES = set('这那每各该某')
PRONOUNS = set('我你他她它咱')
COMPLEMENTS = set('来去起出上下进回开完住掉')
LOCALIZERS = set('上下里中内外前后间')
UNITS = set('年月日号点时分秒岁倍天周个条篇份件位名次元块米斤克张本部台集期层代级种项款家套场轮届章节页段万亿千百')
CN_DIGITS = set('零〇一二两三四五六七八九十百千万亿')
CONNECTIVES = set('''因为 所以 但是 而且 如果 虽然 可是 然而 不过 于是 甚至 或者 并且 只要 只有 除非 即使 哪怕 尽管 无论 不管 因此
    否则 然后 同时 以及 还是 就是 而是 不是 一旦 毕竟 其实 尤其 比如 例如 包括 直到 就像 好像 也就是 而不是 不仅 以便'''.split())
# Common words that contain a particle, preposition, adverb or complement character: never split, and their inner
# character does not count as a break cue.
WORDS = CONNECTIVES | set('''了解 为了 除了 罢了 了不起 的确 目的 得到 得出 觉得 记得 值得 获得 取得 难得 懂得 显得 舍得 晓得 免得
    使得 赢得 地方 地区 地球 地位 地图 地址 地面 地下 地铁 地产 地点 地域 地理 地震 天地 土地 当地 各地 场地 基地 本地 外地
    过去 过程 过来 过年 过度 过于 过分 过滤 过期 过往 经过 通过 超过 难过 错过 度过 着急 着手 着想 着实 着重 着迷 接着 跟着
    随着 看着 意味着 现在 正在 存在 实在 所在 好在 自在 在于 在意 还在 和平 和谐 总和 温和 缓和 暖和 柔和 因而 从而 进而 反而
    或许 把握 被动 从来 从此 从前 从事 自从 服从 对于 对手 对方 对象 对比 对话 对待 面对 绝对 相对 针对 反对 应对 不对 方向
    面向 倾向 走向 导向 向往 一向 交给 带给 送给 供给 让步 转让 跟进 比较 比例 比赛 相比 好比 无比 作为 成为 认为 以为 行为
    称为 更为 极为 较为 最为 为何 为什么 将来 即将 由于 理由 自由 往往 以往 前往 往来 等于 终于 至于 属于 关于 用户 使用 作用
    利用 应用 费用 有用 用于 以后 以前 可以 难以 以上 以下 加以 予以 足以 成就 早就 也许 再也 首都 大都 全都 还有 归还 偿还
    才能 人才 刚才 天才 方才 再次 再见 不再 冷却 忘却 已经 曾经 正是 正常 真正 正式 正好 很多 最后 最近 最好 最终 最大 更加
    更多 太阳 太多 不会 不能 不要 不同 不断 不少 不用 不错 不出 没有 没想到 别人 我们 你们 他们 她们 它们 人们 咱们 自己 这样
    那样 怎样 一样 同样 这些 那些 这个 那个 这种 那种 这里 那里 这么 那么 什么 怎么 多么 只是 总是 也是 都是 要是 若是 凡是
    算是 像是 是否 来自 来说 起来 出来 上来 下来 进来 回来 原来 后来 未来 本来 看来 一起 出现 出去 出发 输出 突出 提出 做出
    付出 杰出 回到 达到 到达 到底 到处 感到 看到 想到 遇到 受到 做到 收到 来到 找到 听到 见到 拿到 刷到 学到 用到 碰到 走到
    提到 上班 上家 上市 上限 上线 上传 网上 线上 晚上 早上 马上 加上 下去 下午 线下 一下 之后 之前 之一 之间 之中 总之 一切
    一直 一定 一些 一般 统一 唯一 万一 第一 开始 开发 打开 离开 公开 展开 完成 完全 完美 成长 成功 成本 形成 变成 构成 一边
    一点 一份 下来 出海 上家 上手 手上 身上 世上 路上 东西 时候 一旦 不仅 并不 并非 而已 以免 起点 发起 引起 一手 二手 发给 寄给
    还给 留给 卖给 递给 传给 嫁给 输给 接近 国内 国外 海外 境内 境外 互联网 自媒体 公众号 小红书 人工智能 大模型 程序员 工程师
    机器人 为止 年轻人 有的人 每个人 所有人 一辈子 这辈子 朋友圈 短视频 直播间 生产力 竞争力 影响力 注意力 安全感 价值观
    核对 配对 比对 派对 横向 纵向 转向 指向 意向 动向 志向 双向 单向 反向 正向 逆向 采用 运用 适用 通用 调用 引用 实用 常用
    试用 享用 专用 得以 用以 借以 类比 位于 处于 善于 敢于 基于 趋于 便于 易于 难于 忙于 急于 内在 外在 平和 随和 顺从 跟从
    听从 必将 分成 变为 改为 视为 成为 全部 部分 注册 登录 模型 成果 结果 效果 如何 任何 之所以'''.split())
# Frequent general words, so an unmarked run of characters is not split inside them.
COMMON = set('''工具 技术 内容 行业 思考 产品 用户 市场 数据 平台 能力 时间 问题 方法 系统 机会 资源 认知 效率 价值 经验 生活
    工作 学习 商业 逻辑 代码 程序 视频 文章 粉丝 流量 账号 收入 成本 风险 资产 公司 团队 项目 世界 时代 信息 知识 社会 经济
    发展 目标 原因 选择 机构 政策 企业 客户 服务 设计 功能 体验 质量 速度 规模 增长 趋势 竞争 优势 策略 计划 决定 判断 理解
    分析 研究 实验 测试 训练 算法 智能 网络 手机 电脑 软件 硬件 芯片 屏幕 文件 图片 声音 画面 字幕 故事 朋友 家人 孩子 父母
    老师 学生 身体 健康 情绪 压力 焦虑 安全 成长 改变 坚持 努力 习惯 规则 标准 答案 关系 环境 条件 情况 阶段 整体 方式 模式
    结构 框架 概念 观点 角度 证据 案例 例子 细节 重点 核心 关键 本质 意义 作品 主页 参考 反馈 比例 收藏 关注 浏览 点赞 评论
    转发 分享 发布 运营 推广 品牌 创业 投资 赚钱 副业 职业 工资 简历 面试 老板 同事 领导 员工 合作 交流 沟通 表达 写作 阅读
    英语 中文 语言 翻译 自然 简单 复杂 重要 主要 需要 应该 可能 能够 只能 必须 已经 突然 终于 同时 然后 最后 首先 其次 一般
    普通 特别 非常 真的 确实 当然 其实 直接 主动 自动 手动 完全 彻底 继续 开始 结束 准备 解决 处理 提高 降低 增加 减少 影响
    帮助 支持 提供 使用 获取 生成 创造 制作 绘制 记录 检查 确认 发现 证明 显示 出现 消失 存在 保持 保存 打开 关闭 中间'''.split())
WORDS |= COMMON
# Words that close the phrase before them (像…一样, 三年以后): a line or cue never starts with them.
POSTPOSITIONS = set('一样 似的 而已 之后 之前 以后 以前 以来 为止 的话 左右 之间 之中 当中 罢了 起来 下去 出来 上来 下来'.split())
CONJUNCTIONS_EN = set('and but or nor so because although though while when whenever if unless until since that which who whom whose '
                      'where to for with from of in on at by as than into about over after before'.split())
WEAK_ENDS_EN = set('a an the to of and or but my your his her its our their this that these those in on at for with by '
                   'from as is are was were be very'.split())
ARABIC_NUMBER = (r'(?<![A-Za-z0-9_.\-])(?:[-+±](?=[$¥￥€£]?\d))?[$¥￥€£]?\d+(?:[.,:]\d+)*(?:\s?[%‰])?'
                 r'(?:[多余几]?(?:[万亿千百]|倍|岁|年|月|日|号|点|时|分|秒|天|周|个|条|篇|份|件|位|名|次|元|块|米|公里|斤|克|张|本|部|台|集|期|层|代|级|K|k|M|B|x|X|W|w)+)?')
CHINESE_NUMBER = (r'(?:第|百分之)?[零〇一二两三四五六七八九十百千万亿]+(?:点[零〇一二三四五六七八九]+)?[多余几来]?'
                  r'(?:倍|岁|年|月|日|号|时|分|秒|天|周|个|条|篇|份|件|位|名|次|元|块|米|公里|斤|克|张|本|部|台|集|期|层|代|级)?')
LATIN_UNITS = {**dict.fromkeys("ilIj.,'!|:;", .25), **dict.fromkeys(' ', .33), **dict.fromkeys('frt-()[]/"', .35),
               **dict.fromkeys('svxzJ', .5), **dict.fromkeys('BKPRSVXYZ0123456789', .7), **dict.fromkeys('wACDGHNOQU', .8),
               **dict.fromkeys('m%', .9), **dict.fromkeys('MW', 1.)}
NOT_NUMBERS = {'十分', '千万', '万万', '万一', '一一', '一一对应'}
_jieba = None


def caption_font(options):
    """Regular caption/watermark font: options.font.path, else the platform chain (coverage is checked by fonts.check_coverage)."""
    from fonts import select
    try:
        return select('caption', '', options.get('font'))
    except FileNotFoundError as error:
        raise FileNotFoundError(f'{error}; set captions.font to a legible regular Chinese font') from error


def resolve_cues(beat):
    source = beat.get('caption_cues') or [{'text': beat['text'], 'start': 0, 'end': beat['duration']}]
    if ''.join(cue['text'] for cue in source) != beat['text']:
        raise ValueError('Caption cues must preserve every narration character in order')
    previous = 0.
    result = []
    for cue in source:
        start, end = float(cue['start']), float(cue['end'])
        if not cue['text'].strip() or not math.isfinite(start + end) or not 0 <= previous <= start < end <= beat['duration'] + .001:
            raise ValueError('Caption cues must be nonempty, ordered and inside the measured beat')
        result.append({'text': cue['text'], 'start': beat['start'] + start, 'end': beat['start'] + end})
        previous = end
    return result


# ---------------------------------------------------------------- canvas-aware defaults and the platform safe zone

def vertical_factor(canvas_width, canvas_height):
    """0 for landscape/square/3:4, 1 for 9:16 and taller, linear between: how much of the feed UI covers the frame."""
    ratio = canvas_height / canvas_width
    return min(1., max(0., (ratio - 4 / 3) / (16 / 9 - 4 / 3)))


def caption_safe_zone(canvas_width, canvas_height):
    """Insets (pixels) a vertical feed's UI covers; all zero on landscape. rail = (top, bottom) of the right button rail."""
    factor = vertical_factor(canvas_width, canvas_height)
    return {'top': round(canvas_height * PLATFORM_UI['top'] * factor), 'bottom': round(canvas_height * PLATFORM_UI['bottom'] * factor),
            'left': round(canvas_width * PLATFORM_UI['left'] * factor), 'right': round(canvas_width * PLATFORM_UI['right'] * factor),
            'rail': tuple(round(canvas_height * value) for value in PLATFORM_UI['rail'])}


def caption_safe_violations(box, canvas_width, canvas_height):
    """Problems of a caption box (x, y, width, height) against the canvas and the platform UI; [] when it is clear."""
    x, y, width, height = box
    zone, problems = caption_safe_zone(canvas_width, canvas_height), []
    if x < 0 or y < 0 or x + width > canvas_width or y + height > canvas_height:
        problems.append('outside the canvas')
    if zone['bottom'] and y + height > canvas_height - zone['bottom']:
        problems.append(f"bottom platform UI (account/description/music) covers {y + height - (canvas_height - zone['bottom'])} px; "
                        f"keep the caption bottom at or above y={canvas_height - zone['bottom']}")
    if zone['right'] and x + width > canvas_width - zone['right'] and y < zone['rail'][1] and y + height > zone['rail'][0]:
        problems.append(f"right button rail covers {x + width - (canvas_width - zone['right'])} px; narrow the caption")
    if zone['left'] and x < zone['left']:
        problems.append(f"left edge inset is {zone['left']} px")
    if zone['top'] and y < zone['top']:
        problems.append(f"top tabs cover {zone['top'] - y} px")
    return problems


def default_bottom_margin(canvas_width, canvas_height):
    """4 % of the height on landscape (old default); on vertical feeds above the platform UI band plus a small gap."""
    factor = vertical_factor(canvas_width, canvas_height)
    return max(round(canvas_height * .04), round(canvas_height * (PLATFORM_UI['bottom'] + UI_GAP) * factor))


def default_font_size(canvas_width, canvas_height):
    # Phones show a vertical film at full width: 52 px per 1080 px of width is about 19 pt on a 390 pt wide phone.
    return round(canvas_width * 52 / 1080) if canvas_height > canvas_width else max(18, round(canvas_height * 38 / 1080))


def default_width_ratio(canvas_width, canvas_height):
    """.76 of the width, narrowed on vertical feeds so a centred caption clears the right button rail."""
    zone = caption_safe_zone(canvas_width, canvas_height)
    return min(.76, 1 - 2 * zone['right'] / canvas_width - .01)


def default_stroke(size):
    """2 px for 1080p landscape text (~38 px); 4 px from ~44 px up, the phone-size outline on light paper."""
    return 2 if size < 44 else min(4, round(size / 13))


def caption_position(metadata, options, canvas_width, canvas_height):
    """Top-left (x, y) of a built caption: centred; explicit captions.bottom_margin wins, else default_bottom_margin."""
    margin = options.get('bottom_margin')
    margin = default_bottom_margin(canvas_width, canvas_height) if margin is None else int(margin)
    x, y = (canvas_width - metadata['width']) // 2, canvas_height - metadata['height'] - margin
    if min(x, y) < 0 or y + metadata['height'] > canvas_height:
        raise ValueError('Caption lies outside the canvas')
    return x, y


def caption_char_limit(options, canvas_width, canvas_height):
    """Cue length in CJK characters (Latin letters ~0.6): captions.max_chars, else what one caption line holds, in
    half-character steps, at most MAX_CUE_CHARS (vertical 1080x1920 default: 14.5; 16:9: 24)."""
    if options.get('max_chars') is not None:
        limit = int(options['max_chars'])
        if not 4 <= limit <= 60:
            raise ValueError('captions.max_chars must be within 4..60')
        return limit
    size = int(options.get('font_size', default_font_size(canvas_width, canvas_height)))
    stroke = int(options.get('stroke_width', default_stroke(size)))
    ratio = float(options.get('max_width_ratio', default_width_ratio(canvas_width, canvas_height)))
    # Half-character steps: 14.5 keeps 14 Chinese characters, or 13 plus a short Latin word, on one line.
    return max(8, min(MAX_CUE_CHARS, math.floor((canvas_width * ratio - 16 - stroke * 2) / max(1, size) * 2) / 2))


def cue_char_limit(manifest):
    """caption_char_limit for a job.json, resolving its canvas the way video_job does (2:3 when unspecified)."""
    canvas = manifest.get('canvas', {})
    width, height = canvas.get('width'), canvas.get('height')
    if not width or not height:
        try:
            from aspect_ratio import resolve_canvas, DEFAULT_RATIO
            resolved = resolve_canvas(canvas.get('aspect_ratio', DEFAULT_RATIO), width, height, canvas.get('fps', 30))
            width, height = resolved['width'], resolved['height']
        except ImportError:
            width, height = 1280, 1920
    return caption_char_limit(manifest.get('captions', {}), int(width), int(height))


# ---------------------------------------------------------------- word-aware breaking

def _kind(character):
    code = ord(character)
    if character.isspace():
        return ' '
    if (0x3400 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF or 0x3040 <= code <= 0x30FF or 0xAC00 <= code <= 0xD7AF
            or code == 0x3007 or 0x20000 <= code <= 0x2FFFF):
        return 'h'
    return 'a' if character.isalnum() else 'p'


def is_latin(text):
    """English/Latin-script captions break at spaces only; Chinese with Latin words (我用Claude Code…) does not count."""
    letters = sum(_kind(character) == 'a' and character.isalpha() for character in text)
    return letters > 8 * sum(_kind(character) == 'h' for character in text)


def _jieba_edges(text):
    global _jieba
    if not USE_JIEBA:
        return None
    if _jieba is None:
        try:
            import jieba
            jieba.setLogLevel(60)
            _jieba = jieba
        except ImportError:
            _jieba = False
    if not _jieba:
        return None
    edges, cursor = set(), 0
    for token in _jieba.cut(text):
        cursor += len(token)
        edges.add(cursor)
    return edges


def number_spans(text):
    """(start, end) of numbers worth keeping whole and highlighting: Arabic with unit, or clear Chinese numerals."""
    spans = [match.span() for match in re.finditer(ARABIC_NUMBER, text)]
    for match in re.finditer(CHINESE_NUMBER, text):
        (start, end), value = match.span(), match[0]
        digits = value.removeprefix('第').removeprefix('百分之').rstrip('倍岁年月日号时分秒天周个条篇份件位名次元块米公里斤克张本部台集期层代级多余几来')
        if value in NOT_NUMBERS or not digits or all(character == '一' for character in digits):
            continue
        if digits[-1] in '万亿' and value[-1] in '年月日号时分秒天周':
            end -= 1  # 七十万年薪: 年 starts the next word
        if len(digits) >= 2 or end - start > len(digits):
            spans.append((start, end))
    spans.sort()
    merged = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:  # a date such as 二零二六年十月二十六日 stays one unit
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def name_spans(text):
    """(start, end) of names joined by a middle dot (托尔斯滕·鲍尔, 马丁·路德·金): a line or cue never breaks inside one.
    A part is the Latin word or number next to the dot, or two to four Chinese characters: before the dot the longest
    such jieba word run (jieba cuts unknown names short: 托尔斯|滕), after it the shortest (鲍尔|今年); four without jieba."""
    if not any(dot in text for dot in NAME_JOINERS):
        return []
    edges = _jieba_edges(text)
    edges = None if edges is None else edges | {0, len(text)}
    spans = []
    for dot, character in enumerate(text):
        if character not in NAME_JOINERS or not 0 < dot < len(text) - 1:
            continue
        left, right = text[dot - 1], text[dot + 1]
        if {_kind(left), _kind(right)} - {'h', 'a'}:  # "说明文件 · 技能": a spaced dot separates, it does not join
            continue
        start, end = dot - 1, dot + 2
        while start > 0 and _kind(text[start - 1]) == _kind(left) and text[start - 1] not in NAME_JOINERS:
            start -= 1
        while end < len(text) and _kind(text[end]) == _kind(right) and text[end] not in NAME_JOINERS:
            end += 1
        if _kind(left) == 'h':
            choices = [edge for edge in (edges or ()) if 2 <= dot - edge <= 4 and edge >= start]
            start = min(choices) if choices else max(start, dot - 4)
        if _kind(right) == 'h':
            choices = [edge for edge in (edges or ()) if 2 <= edge - dot - 1 <= 4 and edge <= end]
            end = min(choices) if choices else min(end, dot + 5)
        spans.append((start, end))
    return spans


def keyword_spans(text, keywords):
    """Non-overlapping occurrences of the keywords, longest first."""
    taken = []
    for word in sorted({str(word) for word in keywords if str(word).strip()}, key=len, reverse=True):
        for match in re.finditer(re.escape(word), text):
            if all(match.end() <= start or match.start() >= end for start, end in taken):
                taken.append(match.span())
    return sorted(taken)


def break_costs(text, keywords=()):
    """Cost of a line/cue break before text[i] (index 1..len-1; INF = never). Lower is a better break."""
    count = len(text)
    costs, plain = [INF] * (count + 1), [False] * (count + 1)
    atoms = number_spans(text) + keyword_spans(text, keywords) + name_spans(text)
    locked = {index for start, end in atoms for index in range(start + 1, end)}
    if is_latin(text):
        for index in range(1, count):
            left, right = text[index - 1], text[index]
            if index in locked or not left.isspace() or right.isspace() or right in NO_START:
                if left == '-' and index not in locked and text[index - 2:index - 1].isalpha() and right.isalpha():
                    costs[index] = 8.
                continue
            before = text[:index].split()[-1] if text[:index].split() else ''
            after = text[index:].split()[0].lower().strip('"\'(')
            ended = before.rstrip('"\')]}”’')[-1:]
            costs[index] = 0. if ended in CLAUSE_END or ended == '.' else 1.
            if after in CONJUNCTIONS_EN:
                costs[index] = min(costs[index], .5)
            if before.lower().strip('"\'(') in WEAK_ENDS_EN:
                costs[index] += 6.
        return costs
    inside, word_edges, word_starts = set(), set(), set()
    for word in WORDS | POSTPOSITIONS:
        start = text.find(word)
        while start >= 0:
            inside.update(range(start + 1, start + len(word)))
            word_edges.update((start, start + len(word)))
            word_starts.add(start)
            start = text.find(word, start + 1)
    closing = {index for index in range(count) if any(text.startswith(word, index) for word in POSTPOSITIONS)}
    clause_start = 0
    for index in range(1, count):
        left, right = text[index - 1], text[index]
        kinds = _kind(left), _kind(right)
        if left in CLAUSE_END:
            clause_start = index
        if right in NO_START or left in NO_END or index in locked or kinds == ('a', 'a'):
            continue
        if left in '.,:/-_+&@#\'' and index >= 2 and _kind(text[index - 2]) == 'a' and kinds[1] == 'a':
            continue
        if kinds[0] == 'a' and right in UNITS:
            continue
        if right in PARTICLES and index not in word_starts and kinds[0] != ' ':
            costs[index] = 40.  # a particle never opens a line, whatever precedes it
            continue
        if index in closing:
            costs[index] = 35.
            continue
        if left in CLAUSE_END:
            costs[index] = 1. if left in '、：:' else 0.
        elif left in CLOSERS:
            costs[index] = 1. if text[index - 2:index - 1] and text[index - 2] in CLAUSE_END else 3.
        elif right in NO_END:
            costs[index] = 3.
        elif kinds[0] == ' ':
            costs[index] = 45. if index >= 2 and _kind(text[index - 2]) == 'a' and kinds[1] == 'a' else 2.  # Grok Bot
        elif kinds[1] == ' ':
            continue  # break after the space, never before it
        elif 'p' in kinds:
            costs[index] = 6.
        elif kinds[0] != kinds[1]:
            costs[index] = 12. if kinds[0] == 'a' and right in LOCALIZERS else 4.
            if kinds[0] == 'h' and left in LEADS | DEMONSTRATIVES:
                costs[index] = 14.
        elif left in CN_DIGITS and (right in CN_DIGITS or right in UNITS):
            costs[index] = 50.
        elif index in inside:
            costs[index] = 45.
        elif right in PARTICLES:
            costs[index] = 40.
        elif any(text.startswith(word, index) for word in CONNECTIVES):
            costs[index] = 2.
        elif left == '得' and right in COMPLEMENTS | set('到了动完懂清见好'):
            costs[index] = 30.  # 用得上, 看得懂: potential complement
        elif left in PARTICLES:
            # 会用AI的人: a one-character head noun stays with its 的
            costs[index] = 14. if left == '的' and right in '人事话时样' and _kind(text[index + 1:index + 2] or '，') != 'h' else 2.
        elif right == '到' and index not in word_starts:
            costs[index] = 5. if '从' in text[clause_start:index] else 16.
        elif left == '到' and index - 1 not in inside and '从' in text[clause_start:index - 1]:
            costs[index] = 16.  # 从A到B: 到 opens B like a preposition
        elif right in LEADS:
            costs[index] = 5.
        elif left in LEADS:
            costs[index] = 16.
        elif left in NEGATIONS:
            costs[index] = 30.
        elif right in NEGATIONS and text[index + 1:index + 2] and text[index + 1] in COMPLEMENTS | set('到了动见懂清完成着'):
            costs[index] = 40.  # 卖不出去, 做不到: verb, negation and complement stay together
        elif right in ADVERBS or right in NEGATIONS:
            costs[index] = 4.
        elif left in ADVERBS:
            costs[index] = 15.
        elif left == '是':
            costs[index] = 4.
        elif right == '是':
            costs[index] = 7.
        elif left in DEMONSTRATIVES:
            costs[index] = 18.
        elif right in DEMONSTRATIVES or right in PRONOUNS:
            costs[index] = 6.
        elif right in COMPLEMENTS and index not in word_starts:
            costs[index] = 14.
        elif left in PRONOUNS:
            costs[index] = 9.
        else:
            costs[index], plain[index] = 12., True
    # Inside an unmarked run of Chinese characters two-character words dominate: even offsets from both ends of the run
    # are likely word edges, odd ones likely split a word.
    edges = [index for index in range(count + 1) if index in (0, count) or index not in inside and (not plain[index] or index in word_edges)]
    for index in range(1, count):
        if plain[index]:
            start = max(edge for edge in edges if edge < index)
            end = min(edge for edge in edges if edge > index)
            first, second = index - start, end - index
            costs[index] = 9. if first % 2 == 0 and second % 2 == 0 else 15. if first % 2 and second % 2 else 12.
            if min(first, second) == 1:
                costs[index] += 3.
    jieba_edges = _jieba_edges(text)
    if jieba_edges is not None:
        for index in range(1, count):
            if costs[index] < INF:
                costs[index] = max(costs[index], 30.) if index not in jieba_edges else min(costs[index], 7.) if plain[index] else costs[index]
    return costs


def _shown(character, strip):
    return ' ' if strip == 'all' and character in FULLWIDTH_DROPPABLE else character


def _droppable_end(text, end, strip):
    """Index after trimming trailing spaces and, when stripping, dropped punctuation before trailing closers."""
    if not strip:
        while end and text[end - 1].isspace():
            end -= 1
        return end, end
    closers = end
    while closers and (text[closers - 1].isspace() or text[closers - 1] in CLOSERS):
        closers -= 1
    cut = closers
    while cut and (text[cut - 1].isspace() or text[cut - 1] in DROPPABLE or text[cut - 1] == '.' and cut >= 2
                   and (text[cut - 2].islower() or text[cut - 2].isdigit() or _kind(text[cut - 2]) == 'h') and text[cut - 2] != '.'):
        cut -= 1
    return cut, closers


def display_line(text, strip=True):
    """What a caption line shows: trailing ，。、；：,;: (and a sentence period) dropped when stripping; ？！… kept;
    strip='all' also turns inner Chinese ，。、；： into spaces."""
    text = text.strip()
    if not strip:
        return text
    cut, closers = _droppable_end(text, len(text), strip)
    while closers < len(text) and text[closers].isspace():
        closers += 1
    shown = text[:cut] + text[closers:].strip()
    if strip == 'all':
        shown = re.sub(r'\s+', ' ', ''.join(_shown(character, strip) for character in shown)).strip()
    return shown or text


def display_units(character):
    """Width in CJK ems (CJK = 1); Latin from typical sans advances, so cue sizes track the rendered line."""
    kind = _kind(character)
    if kind == 'h':
        return 1.
    if character.isascii():
        return LATIN_UNITS.get(character, .6)
    return .6 if kind == 'a' and character.isalpha() and ord(character) < 0x2E80 else 1.


def _segments(text, widths, costs, limit, strip, max_parts=None, balance=40.):
    """Best (start, end) split of text: fewest parts whose shown width fits, then lowest break cost + imbalance."""
    count = len(text)
    prefix = [0.]
    for width in widths:
        prefix.append(prefix[-1] + width)

    def span(start, end):
        while start < end and text[start].isspace():
            start += 1
        cut, closers = _droppable_end(text, end, strip)
        shown = prefix[max(start, cut)] - prefix[start]
        return shown + sum(widths[index] for index in range(max(start, closers), end) if not text[index].isspace())

    total = span(0, count)
    if total <= limit:
        return [(0, count)]
    points = [0] + [index for index in range(1, count) if costs[index] < INF] + [count]
    fewest = max(2, math.ceil(total / limit))
    weight = [cost if cost < 30 else cost * 4 for cost in costs]  # one bad break outweighs several fair ones
    found = []
    for parts in range(fewest, min(max_parts or count, fewest + 6) + 1):
        if found and parts > found[0][1] + 2:
            break
        target, best = total / parts, {0: (0., [])}
        for step in range(parts):
            following = {}
            for end in points[1:]:
                if (end == count) != (step == parts - 1):
                    continue
                for start, (score, cuts) in best.items():
                    width = span(start, end) if start < end else INF
                    if width > limit:
                        continue
                    value = score + (weight[start] if start else 0.) + balance * ((width - target) / limit) ** 2
                    if width < .3 * limit:
                        value += (.3 * limit - width) / limit * 20
                    if end not in following or value < following[end][0]:
                        following[end] = (value, cuts + [start] if start else cuts)
            best = following
            if not best:
                break
        if count in best:
            cuts = best[count][1]
            found.append((max(costs[cut] for cut in cuts), parts, cuts))
            # Fewest parts, unless one or two more parts avoid splitting a word or a Latin phrase (cost >= 30).
            if found[-1][0] < 30:
                break
    if not found:
        return None
    good = [item for item in found if item[0] < 30]
    _, _, cuts = good[0] if good else min(found, key=lambda item: item[:2])
    edges = [0] + cuts + [count]
    return list(zip(edges, edges[1:]))


def split_text(text, limit, width_of=display_units, max_parts=None, strip=True, keywords=()):
    """Word-aware split of text into the fewest pieces whose shown width is <= limit; raw pieces join back to text.
    width_of measures one character (pixels or CJK units). Breaks inside a word only when an unbreakable run (long
    word, keyword) is wider than the limit; None when max_parts cannot hold it."""
    if not text:
        return [text]
    widths = [width_of(_shown(character, strip)) for character in text]
    costs = break_costs(text, keywords)
    segments = _segments(text, widths, costs, limit, strip, max_parts)
    if segments is None:  # only a run that cannot fit between its own break points may break inside
        points = [0] + [index for index in range(1, len(text)) if costs[index] < INF] + [len(text)]
        relaxed = list(costs)
        for start, end in zip(points, points[1:]):
            if sum(widths[start:end]) > limit:
                for index in range(start + 1, end):
                    if text[index] not in NO_START and not text[index].isspace():
                        relaxed[index] = 150. if text[index - 1] in NAME_JOINERS else 200.  # a name too long: after its dot
        segments = _segments(text, widths, relaxed, limit, strip, max_parts)
    if segments is None:  # max_parts too few for whole words: break inside one rather than before punctuation
        relaxed = [cost if cost < INF else 200. if 0 < index < len(text) and text[index] not in NO_START
                   and not text[index].isspace() else INF for index, cost in enumerate(costs)]
        segments = _segments(text, widths, relaxed, limit, strip, max_parts)
    return None if segments is None else [text[start:end] for start, end in segments]


def character_wrap(text, fits, kinsoku=True):
    """Last-resort wrap, one character at a time while fits(line); with kinsoku a line never starts with closing
    punctuation (，。？）…): the character before it goes down with it."""
    lines, current = [], ''
    for character in text:
        if character == '\n' or (current and not fits(current + character)):
            carry = current[-1] if kinsoku and character in NO_START and len(current) > 1 and current[-1] not in NO_START else ''
            lines.append(current[:len(current) - len(carry)])
            current = carry
        if character != '\n':
            current += character
    return lines + [current] if current else lines


def caption_lines(text, width_of, limit, max_lines=None, strip=True, keywords=()):
    """Displayed caption lines (explicit newlines kept); None when the text cannot fit in max_lines."""
    lines = []
    for paragraph in text.split('\n'):
        pieces = split_text(paragraph, limit, width_of, max_lines, strip, keywords)
        if pieces is None:
            return None
        lines += [display_line(piece, strip) for piece in pieces]
    lines = [line for line in lines if line]
    return lines if lines and (max_lines is None or len(lines) <= max_lines) else None


def clauses(text):
    """Text split after each run of clause punctuation (with closing quotes and spaces); 1,000 and 10:30 stay whole."""
    pieces, start, index = [], 0, 0
    while index < len(text):
        character = text[index]
        numeric = character in ',:' and 0 < index < len(text) - 1 and text[index - 1].isdigit() and text[index + 1].isdigit()
        if character in CLAUSE_END and not numeric:
            while index + 1 < len(text) and (text[index + 1] in CLAUSE_END or text[index + 1] in CLOSERS or text[index + 1].isspace()):
                index += 1
            pieces.append(text[start:index + 1])
            start = index + 1
        index += 1
    return pieces + ([text[start:]] if start < len(text) else [])


def ends_sentence(text):
    text = text.rstrip()
    while text and text[-1] in CLOSERS:
        text = text[:-1]
    return text[-1:] in SENTENCE_END


def shown_width(text, strip=True):
    return sum(display_units(character) for character in display_line(text, strip))


def cue_pieces(text, limit=MAX_CUE_CHARS, keywords=()):
    """Caption cues for one narration text, joining back to it exactly. Clauses longer than limit CJK characters
    (Latin letters count ~0.6) split at word-aware positions into the fewest balanced parts; clauses then pack up to
    the limit without crossing a sentence end; a tiny piece (<= 4 characters) joins a neighbour that has room."""
    pieces = []
    for clause in clauses(text):
        split = split_text(clause, limit, display_units, None, True, keywords) if shown_width(clause) > limit else None
        pieces += [(part, False) for part in split[:-1]] + [(split[-1], True)] if split else [(clause, True)]
    packed = []
    for piece, closes_clause in pieces:
        if not re.search(r'\w', piece) and packed:
            packed[-1][0] += piece
            continue
        previous = packed[-1] if packed else None
        if (previous and previous[1] and not ends_sentence(previous[0])
                and shown_width(previous[0] + piece) <= limit):
            previous[0] += piece
            previous[1] = closes_clause
        else:
            packed.append([piece, closes_clause])
    merged = []
    for position, (piece, _) in enumerate(packed):
        if merged and shown_width(piece) <= 4 and not ends_sentence(merged[-1]) \
                and shown_width(merged[-1] + piece) <= limit:
            merged[-1] += piece
        elif (position + 1 < len(packed) and shown_width(piece) <= 4 and not ends_sentence(piece)
              and shown_width(piece + packed[position + 1][0]) <= limit):
            packed[position + 1][0] = piece + packed[position + 1][0]
        else:
            merged.append(piece)
    return merged


def highlight_spans(line, keywords=(), numbers=False):
    spans = keyword_spans(line, keywords)
    if numbers:
        spans += [span for span in number_spans(line) if all(span[1] <= start or span[0] >= end for start, end in spans)]
    return sorted(spans)


# ---------------------------------------------------------------- rendering

def build_caption(directory, name, text, options, canvas_width, canvas_height, keywords=None):
    """Render one cue. keywords: extra highlight words for this cue (e.g. the beat's caption_keywords)."""
    path, index = caption_font(options)
    size = int(options.get('font_size', default_font_size(canvas_width, canvas_height)))
    if size <= 0:
        raise ValueError('Captions require a positive font size')
    font = ImageFont.truetype(path, size, index=index)
    stroke = int(options.get('stroke_width', default_stroke(size)))
    ratio = float(options.get('max_width_ratio', default_width_ratio(canvas_width, canvas_height)))
    if not 0 < ratio <= .95 or not 0 <= stroke <= 4:
        raise ValueError('Invalid caption width or outline')
    strip = options.get('strip_punctuation', True)
    if strip not in (True, False, 'all'):
        raise ValueError('captions.strip_punctuation must be true, false or "all"')
    def listed(value):
        return [value] if isinstance(value, str) else list(value or [])
    words = [str(word) for word in listed(options.get('keywords')) + listed(keywords)]
    highlight = options.get('highlight_color', HIGHLIGHT_COLOR)
    ImageColor.getrgb(highlight)
    width_limit = int(canvas_width * ratio)
    line_limit = width_limit - 16 - stroke * 2
    max_lines = int(options.get('max_lines', 2))
    advance = {}

    def width_of(character):
        if character not in advance:
            advance[character] = font.getlength(character)
        return advance[character]

    lines = None
    for attempt in range(3):  # kerning can make a joined line a few pixels wider than its characters
        lines = caption_lines(text, width_of, line_limit - attempt * size * .25, max_lines, strip, words)
        if lines is None or all(font.getlength(line) <= line_limit for line in lines):
            break
    if lines is None or any(font.getlength(line) > line_limit for line in lines):
        # the previous character wrap: anything it accepted is still accepted; kinsoku when the lines allow it
        wrapped = [character_wrap(text, lambda line: font.getlength(line) <= line_limit, kinsoku) for kinsoku in (True, False)]
        lines = wrapped[0] if len(wrapped[0]) <= max_lines else wrapped[1]
    if not lines or len(lines) > max_lines:
        raise ValueError('Split long subtitles into speech-aligned caption_cues')
    margin = 6 + stroke
    boxes = [font.getbbox(line, stroke_width=stroke) for line in lines]
    line_height = max(box[3] - box[1] for box in boxes)
    line_gap = round(size * .18)
    width = max(box[2] - box[0] for box in boxes) + margin * 2
    height = line_height * len(lines) + line_gap * (len(lines) - 1) + margin * 2
    image = Image.new('RGBA', (width, height))
    draw = ImageDraw.Draw(image)
    background = options.get('background')
    if background is not None:
        if not isinstance(background, dict) or not 0 < float(background.get('opacity', .76)) <= 1:
            raise ValueError('Caption background requires color and opacity within 0..1')
        color = ImageColor.getrgb(background.get('color', '#243039')) + (round(255 * background.get('opacity', .76)),)
        draw.rounded_rectangle((5, 5, width-6, height-6), radius=round(size*.2), fill=color)
    fill, outline = options.get('color', '#FFFFFF'), options.get('stroke_color', '#000000')
    highlights = []
    for row, (line, box) in enumerate(zip(lines, boxes)):
        x = (width - (box[2] - box[0])) // 2 - box[0]
        y = margin + row * (line_height + line_gap) - box[1]
        spans = highlight_spans(line, words, bool(options.get('highlight_numbers', False)))
        if not spans:
            draw.text((x, y), line, font=font, fill=fill, stroke_width=stroke, stroke_fill=outline)
            continue
        # One outline pass for the whole line, then the fills: colored words keep the same outline and baseline.
        if stroke:
            draw.text((x, y), line, font=font, fill=outline, stroke_width=stroke, stroke_fill=outline)
        cursor = 0
        for start, end in spans + [(len(line), len(line))]:
            for left, right, color in ((cursor, start, fill), (start, end, highlight)):
                if right > left:
                    draw.text((x + font.getlength(line[:left]), y), line[left:right], font=font, fill=color)
            cursor = end
        highlights += [{'line': row, 'start': start, 'end': end, 'text': line[start:end]} for start, end in spans]
    image.save(Path(directory) / f'{name}.png')
    np.savez_compressed(Path(directory) / f'{name}.npz', reveal=np.zeros((height, width), dtype=np.float32))
    return {'type': 'typeset_caption', 'text': text, 'lines': lines, 'width': width, 'height': height,
            'font': {'path': path, 'index': index, 'family': font.getname()[0], 'style': font.getname()[1]},
            'font_size': size, 'color': fill, 'stroke_width': stroke,
            'stroke_color': outline, 'background': background, 'safe_margin': 5, 'segments': [],
            'strip_punctuation': strip, 'highlights': highlights, 'highlight_color': highlight if highlights else None}
