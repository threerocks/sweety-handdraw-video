// Motion recipes for web overlay layers. Each recipe builds its DOM inside `root` (a transparent box the size of
// the overlay) and starts its animations at virtual time 0; the capture then steps the clock frame by frame.
// Libraries: anime.js (MIT), rough.js (MIT), rough-notation (MIT), canvas-confetti (ISC), d3-geo + d3-array (ISC), topojson-client (ISC),
// world-atlas land outlines (ISC; Natural Earth, public domain); for their recipes only, matter-js (MIT), Zdog (MIT) and
// MathJax (Apache-2.0) — see vendor/*.LICENSE and THIRD_PARTY_NOTICES.md.
(() => {
  const { animate, stagger, createSpring, svg, createTimeline } = anime;
  const NS = 'http://www.w3.org/2000/svg';
  const fonts = {
    hand: '"Hannotate SC", "HanziPen SC", "Noto Sans CJK SC", "PingFang SC", sans-serif',
    sans: '"PingFang SC", "Hiragino Sans GB", "Noto Sans CJK SC", "Helvetica Neue", sans-serif',
    serif: '"Songti SC", "Noto Serif CJK SC", serif',
  };
  const family = (value) => fonts[value] || value || fonts.sans;
  const spring = (stiffness = 180, damping = 14, mass = 1) => createSpring({ stiffness, damping, mass });

  // ---------------------------------------------------------------- text that wraps between words
  // Chinese has no spaces, so a browser breaks a line between any two glyphs ("对面司/机", "援非医/疗队", "Opus/4.5？").
  // Wrapped text is cut into word units instead and lines break only between units: Intl.Segmenter words (or the
  // jieba words Python passes), with ICU's leftovers paired (it returns unknown words one glyph at a time: 援|非,
  // 托|尔|斯|滕); a number keeps its unit and a name its version ("50 米", "70%", "≥ 12 lx", "Opus 4.5"); a name
  // joined by · stays whole; closing punctuation sticks to the word before it and opening punctuation to the word
  // after (kinsoku: no line starts with ，。？）); a particle (的了吗…) never opens a line.
  const NO_START = '，。、；：！？）」』】》〉”’…·—～%‰,.;:!?)]}>';
  const NO_END = '（「『【《〈“‘([{<';
  const JOINERS = '·・‧';
  const PARTICLES = '的了着过地得们吗呢吧啊呀嘛么';
  const LEADS = '和与及或而但且并把被从对向给让跟比为在将由往朝于用以按据趁各每第就也都还又才再却便已曾正很最更太不没';
  const SUFFIXES = '队者性化家员式型率度法器机局院所站品业界区组团会类学论感力量值费证表图号版线网点层级款门店厂场馆楼';
  const UNIT_GLYPHS = '年月日号点时分秒岁倍天周个条篇份件位名次元块米斤克张本部台集期层代级种项款家套场轮届章节页段万亿千百';
  const LATIN_UNIT = /^(?:[%‰×x]|°[CF]?|[kKMGT]?(?:B|b|bps|Hz|W|Wh)|[mkμc]?[gms]|m[lL]|[kcm]m|lx|px|pt|fps|rpm|dB|mAh|min|h|L)$/;
  const isSpace = (s) => !s.trim();
  const isHan = (s) => /^[\u3400-\u9FFF\uF900-\uFAFF\u3007\u{20000}-\u{2FFFF}]+$/u.test(s);
  const isLatin = (s) => /^[\x21-\x7E\u00A2-\u00A5\u00B0\u00B1\u00B5\u00D7\u2030\u20AC\uFFE5]+$/.test(s);
  const only = (s, set) => !!s && [...s].every((c) => set.includes(c));

  // Units that join back to text exactly; whitespace units are where lines may break. words: a segmentation to start
  // from (used when it joins back to text), else Intl.Segmenter.
  function wordUnits(text, words) {
    text = String(text ?? '');
    const given = Array.isArray(words) && words.join('') === text;
    let t = (given ? words : window.Intl && Intl.Segmenter ? [...new Intl.Segmenter('zh', { granularity: 'word' }).segment(text)].map((s) => s.segment) : [...text])
      .flatMap((s) => s.split(/([·・‧]|\s+)/)).filter((s) => s);
    const glue = (i, j) => t.splice(i, j - i + 1, t.slice(i, j + 1).join(''));
    // GPT-5, 3.5, 70%, v2.0.1: adjacent Latin/number pieces are one word
    t = t.reduce((out, s) => { if (out.length && isLatin(s) && isLatin(out[out.length - 1])) out[out.length - 1] += s; else out.push(s); return out; }, []);
    // a number and its unit, a name and its version, a comparison and its number (across at most one space)
    for (let i = 0; i < t.length; i++) {
      const k = t[i + 1] === ' ' ? i + 2 : i + 1, next = t[k];
      if (next === undefined || isSpace(next)) continue;
      const number = /\d$/.test(t[i]);
      if ((number && ((isHan(next) && next.length <= 2 && only(next, UNIT_GLYPHS)) || LATIN_UNIT.test(next)))
          || (/^[A-Za-z][A-Za-z-]*$/.test(t[i]) && /^\d/.test(next) && isLatin(next)) || (/^[≤≥<>=≈~±＜＞]$/.test(t[i]) && /^\d/.test(next))) {
        const more = number && next.length === 1 && isHan(next) && (t[k + 1] || '').length === 1 && UNIT_GLYPHS.includes(t[k + 1]);   // 3.5 万|元
        glue(i, more ? k + 1 : k); i -= 1;
      }
    }
    // 托尔斯滕·鲍尔, 马丁·路德·金: up to four glyphs of the name on each side of the dot (the word before it, the
    // single glyphs after it: 鲍|尔|今年 keeps 今年 out)
    const namePart = (s) => /^[\p{Script=Han}A-Za-z·・‧]+$/u.test(s);
    for (let i = 1; i < t.length - 1; i++) {
      if (!only(t[i], JOINERS) || !namePart(t[i - 1]) || !namePart(t[i + 1])) continue;
      let a = i - 1, b = i + 1, size = t[a].length;
      while (a > 0 && namePart(t[a - 1]) && !only(t[a - 1], JOINERS) && size + t[a - 1].length <= 4) size += t[--a].length;
      for (size = t[b].length; b + 1 < t.length && t[b].length === 1 && t[b + 1].length === 1 && isHan(t[b + 1]) && size < 4; size++) b++;
      glue(a, b); i = a;
    }
    if (!given) {   // ICU's single glyphs in a row: two-glyph words dominate Chinese, so pair them from the left
      const out = [];
      for (let i = 0; i < t.length;) {
        let j = i;
        while (j < t.length && t[j].length === 1 && isHan(t[j]) && !PARTICLES.includes(t[j])) j++;
        if (j - i < 2) { out.push(t[i]); i = Math.max(i + 1, j); continue; }
        for (let k = i; k + 1 < j; k += 2) out.push(t[k] + t[k + 1]);
        if ((j - i) % 2) out.push(t[j - 1]);
        i = j;
      }
      t = out;
    }
    // a Chinese count and its measure word: 这两|小时, 三|个月, 十万|元
    for (let i = 0; i + 1 < t.length; i++) {
      if (isHan(t[i]) && /[一二两三四五六七八九十百千万几半]$/.test(t[i]) && (UNIT_GLYPHS.includes(t[i + 1][0]) || /^(?:小时|分钟|公里|公斤|星期|成)/.test(t[i + 1]))) {
        glue(i, i + 1); i -= 1;
      }
    }
    // a lone glyph: a particle or a suffix (医疗|队) joins the word before it, a preposition or an adverb (把, 各, 最,
    // 不) the word after
    for (let i = 0; i < t.length; i++) {
      const s = t[i], before = t[i - 1], after = t[i + 1];
      if (s.length !== 1 || !isHan(s)) continue;
      if (before && !isSpace(before) && (PARTICLES.includes(s) || (SUFFIXES.includes(s) && isHan(before.slice(-1))))) { glue(i - 1, i); i -= 1; }
      else if (after && !isSpace(after) && LEADS.includes(s) && !only(after, NO_START)) glue(i, i + 1);
    }
    const units = [];
    let open = '';
    for (const s of t) {
      if (only(s, NO_START) && units.length) {
        if (isSpace(units[units.length - 1]) && units.length > 1) { const space = units.pop(); units[units.length - 1] += space; }
        units[units.length - 1] += s;
      } else if (only(s, NO_END) || (open && isSpace(s))) open += s;
      else { units.push(open + s); open = ''; }
    }
    if (open) units.push(open);
    return units;
  }

  // Balance wrapped lines. narrow(cut) sets a block `cut` px narrower; keep() says its lines still hold (same count).
  // The narrowest such setting evens the lines, so the last is not one lone word ("…会先做 / 哪一条？"); but a wider
  // one whose lines all end at a clause mark or a space wins if it is still fairly even (its shortest line at least a
  // third of the longest): "13:30　拍下这两小时， / 会怎么看我？", not "13:30　拍下这两 / 小时，会怎么看我？". items: the block's
  // units in order (their own boxes); spaced(item): a space comes before it.
  const CLAUSE_END = /[，。、；：？！…,.;:?!][）」』】》”’)\]]*$/;
  function balance(items, limit, narrow, keep, spaced) {
    let fits = 0, fails = Math.max(0, limit) + 1;
    for (let i = 0; i < 14 && fails - fits > 1; i++) {
      const cut = (fits + fails) / 2;
      narrow(cut);
      if (keep()) fits = cut; else fails = cut;
    }
    const rows = () => items.reduce((list, item) => {
      const row = list[list.length - 1];
      if (!row || item.offsetTop > row.top + item.offsetHeight / 2) list.push({ top: item.offsetTop, left: item.offsetLeft, right: 0, first: item, last: item });
      const current = list[list.length - 1];
      current.left = Math.min(current.left, item.offsetLeft);
      current.right = Math.max(current.right, item.offsetLeft + item.offsetWidth);
      current.last = item;
      return list;
    }, []);
    const tidy = (list) => list.length > 1 && list.slice(1).every((row, k) => CLAUSE_END.test(list[k].last.textContent.trim()) || spaced(row.first))
      && 3 * Math.min(...list.map((row) => row.right - row.left)) >= Math.max(...list.map((row) => row.right - row.left));
    let best = fits;
    for (let k = 0; k <= 24 && fits > 0; k++) {
      const cut = fits * (1 - k / 24);
      narrow(cut);
      if (keep() && tidy(rows())) { best = cut; break; }
    }
    narrow(best);
  }

  // Fill a block (already in the page, `width` px wide) with word units that each stay on one line. A unit wider than
  // the block (a long URL) may still break inside rather than run out of the box. Then its lines are balanced (above);
  // Chromium's own text-wrap: balance is a heuristic that left "现在谁还 / 在用 Opus 4.5？".
  function setWords(node, text, width) {
    node.textContent = '';
    for (const unit of wordUnits(text)) {
      if (isSpace(unit)) { node.appendChild(document.createTextNode(unit)); continue; }
      const span = document.createElement('span');
      span.textContent = unit;
      span.style.whiteSpace = 'nowrap';
      node.appendChild(span);
    }
    Object.assign(node.style, { width: `${width}px`, boxSizing: 'border-box' });
    const units = [...node.children];
    units.forEach((span) => { if (span.offsetWidth > width + 0.5) Object.assign(span.style, { whiteSpace: 'normal', overflowWrap: 'anywhere' }); });
    const height = node.offsetHeight, left = ['left', 'start'].includes(getComputedStyle(node).textAlign);
    balance(units, width - Math.max(0, ...units.map((span) => span.offsetWidth)),
            (cut) => Object.assign(node.style, { paddingLeft: left ? '0' : `${cut / 2}px`, paddingRight: `${left ? cut : cut / 2}px` }),
            () => node.offsetHeight <= height, (span) => !!span.previousSibling && span.previousSibling.nodeType === 3);
    return node;
  }

  function svgLayer(root, width, height) {
    const layer = document.createElementNS(NS, 'svg');
    layer.setAttribute('width', width); layer.setAttribute('height', height);
    layer.setAttribute('viewBox', `0 0 ${width} ${height}`);
    layer.style.position = 'absolute'; layer.style.left = '0'; layer.style.top = '0'; layer.style.overflow = 'visible';
    root.appendChild(layer);
    return layer;
  }

  // Draw-on for every path of a rough.js node: stroke-dashoffset from its length to 0, one after another.
  function drawOn(node, start, duration) {
    const paths = [...node.querySelectorAll('path')];
    const lengths = paths.map((path) => path.getTotalLength());
    const total = lengths.reduce((a, b) => a + b, 0) || 1;
    let cursor = start;
    paths.forEach((path, index) => {
      const share = duration * lengths[index] / total;
      if (path.getAttribute('fill') && path.getAttribute('fill') !== 'none' && !path.getAttribute('stroke')) {
        path.animate([{ opacity: 0 }, { opacity: 1 }], { delay: cursor, duration: Math.max(60, share), fill: 'both' });
      } else {
        path.style.strokeDasharray = `${lengths[index]}`;
        path.animate([{ strokeDashoffset: lengths[index] }, { strokeDashoffset: 0 }],
                     { delay: cursor, duration: Math.max(40, share), easing: 'cubic-bezier(.45,.05,.4,1)', fill: 'both' });
      }
      cursor += share;
    });
    return cursor;
  }

  const Recipes = {};

  // Headline words arrive like a product site: blurred and low, then sharp and in place, staggered (cal.com feel).
  Recipes.kinetic_title = (root, p, ctx) => {
    const box = document.createElement('div');
    Object.assign(box.style, { position: 'absolute', inset: '0', display: 'flex', flexWrap: 'wrap', alignContent: 'center',
      justifyContent: p.align === 'left' ? 'flex-start' : 'center', gap: `0 ${0.18 * (p.size || 96)}px`,
      fontFamily: family(p.font || 'hand'), fontSize: `${p.size || 96}px`, fontWeight: p.weight || 700, color: p.color || '#1E2733',
      lineHeight: 1.15, letterSpacing: p.tracking || '0' });
    root.appendChild(box);
    const style = p.style || 'blur_up';
    // Words come segmented from Python (jieba + keyword atoms); accents are [start, end) character ranges.
    // Glyph styles split Chinese one character per part, but keep a Latin word whole with a space after it: split per
    // character, an English quote lost its spaces and wrapped mid-word ("It'slikedrivingacaratnight").
    const latin = /^[\x21-\x7E\u00C0-\u024F\u2010-\u201F]+$/;
    const parts = ['spring_pop', 'typewriter', 'scramble'].includes(style)
      ? (p.text.match(/[\x21-\x7E\u00C0-\u024F\u2010-\u201F]+|\s+|[^]/gu) || []) : (p.words || [p.text]);
    // Key words take the accent colour; keyword_colors ({"红的": "#E04B3C", "蓝的": "#3E7CC4"}) gives a key word its own.
    // A word listed only in keyword_colors is still found in the text (list it in keywords too, so no line breaks inside it).
    const accent = p.accent || '#E9784E', named = p.keyword_colors || {}, tint = new Map();
    (p.accents || []).forEach(([a, b]) => { for (let k = a; k < b; k++) tint.set(k, named[p.text.slice(a, b)] || accent); });
    Object.keys(named).filter((word) => word).sort((a, b) => b.length - a.length).forEach((word) => {
      for (let at = p.text.indexOf(word); at >= 0; at = p.text.indexOf(word, at + 1)) {
        const span = [...Array(word.length).keys()].map((k) => at + k);
        if (!span.some((k) => tint.has(k))) span.forEach((k) => tint.set(k, named[word]));
      }
    });
    // A line may break only where a word unit ends (see wordUnits): the parts of one unit (a word's glyphs, "Opus" and
    // "4.5？", "50" and "米") share one nowrap group, so a title never wraps inside a word or before its punctuation.
    const breaks = new Set([0]);
    wordUnits(p.text, p.words).reduce((at, unit) => { breaks.add(at + unit.length); return at + unit.length; }, 0);
    let offset = 0, group = null;
    const spans = [], pieces = [];   // pieces: what glyph_in moves, a Chinese glyph or a whole Latin word
    if (p.badge != null) {  // the item's number in a ring before the words: a point title, one level under a chapter title
      const badge = document.createElement('span');
      badge.textContent = String(p.badge);
      Object.assign(badge.style, { display: 'inline-flex', alignItems: 'center', justifyContent: 'center', minWidth: '1.25em', height: '1.25em',
        padding: '0 0.12em', boxSizing: 'border-box', borderRadius: '0.7em', border: `0.07em solid ${p.accent || '#E9784E'}`,
        color: p.accent || '#E9784E', fontSize: '0.78em', transform: 'rotate(-4deg)' });
      box.appendChild(badge);
      spans.push(badge);
      pieces.push(badge);
    }
    for (const part of parts) {
      const begin = offset;
      offset += part.length;
      if (breaks.has(begin)) group = null;
      if (!part.trim()) continue;
      if (!group) {
        group = document.createElement('span');
        Object.assign(group.style, { display: 'inline-flex', columnGap: 'inherit', whiteSpace: 'nowrap' });
        if (/\s$/.test(p.text.slice(0, begin))) group.dataset.spaced = '1';   // a line may end at this space (balance)
        box.appendChild(group);
      }
      const span = document.createElement('span');
      span.style.display = 'inline-block';
      // a space's width after a Latin word that a space followed (the flex gap alone is a CJK letter gap)
      if (latin.test(part) && /^\s/.test(p.text.slice(offset, offset + 1))) span.style.marginRight = '0.1em';
      let piece = span;
      [...part].forEach((character, index, characters) => {
        if (style === 'glyph_in' && !(index && latin.test(character) && latin.test(characters[index - 1]))) {
          piece = document.createElement('span');
          piece.style.display = 'inline-block';
          span.appendChild(piece);
          pieces.push(piece);
        }
        const glyph = document.createElement('span');
        glyph.textContent = character;
        if (tint.has(begin + index)) glyph.style.color = tint.get(begin + index);
        piece.appendChild(glyph);
      });
      group.appendChild(span);
      spans.push(span);
    }
    // Fit the line(s) inside the box at rest, before any motion: a title that overflows its box is cut off in the film.
    let size = p.size || 96;
    const overflows = () => spans.some((span) => {
      const r = span.getBoundingClientRect();
      return r.left < 1 || r.top < 1 || r.right > ctx.width - 1 || r.bottom > ctx.height - 1;
    });
    for (let i = 0; i < 16 && overflows(); i++) {
      size *= 0.92;
      box.style.fontSize = `${size}px`;
      box.style.gap = `0 ${0.18 * size}px`;
    }
    // Balanced lines: a title that wraps at rest is set in the narrowest box that keeps its line count, so its last
    // line is not one lone word ("…会先做 / 哪一条？" becomes two even lines). word_by_word lands a key word at 1.45x:
    // the box may only narrow as far as that larger word still adds no line, so nothing jumps while it shrinks.
    const items = [...box.children];
    const swell = style === 'word_by_word' ? spans.filter((span) => [...span.children].some((glyph) => glyph.style.color)) : [];
    const lines = (big = false) => {
      swell.forEach((span) => { span.style.fontSize = big ? '1.45em' : ''; });
      const count = new Set(items.map((item) => Math.round(item.offsetTop))).size;
      swell.forEach((span) => { span.style.fontSize = ''; });
      return count;
    };
    const wrapped = lines(), landing = lines(true);
    if (wrapped > 1) {
      balance(items, ctx.width - Math.max(...items.map((item) => item.offsetWidth)), (cut) => {
        box.style.paddingLeft = p.align === 'left' ? '0' : `${cut / 2}px`;
        box.style.paddingRight = `${p.align === 'left' ? cut : cut / 2}px`;
      }, () => lines() === wrapped && lines(true) <= landing, (item) => item.dataset.spaced === '1');
    }
    const delay = p.delay || 0;
    // Every word settles inside the layer's own time (its last frame is held): a long line spreads its stagger thinner
    // and, if it must, shortens each word's move, instead of freezing its last words mid-blur.
    const fit = (each, length, count = spans.length) => {
      const room = Math.max(1, (ctx.duration || Infinity) - delay - 70);   // settled two frames before the end
      const move = Math.min(length, room);
      return { each: count > 1 ? Math.min(each, Math.max(0, (room - move) / (count - 1))) : each, move };
    };
    if (style === 'blur_up') {
      const timing = fit(p.stagger || 70, 900);
      animate(spans, { opacity: [0, 1], filter: ['blur(14px)', 'blur(0px)'], translateY: ['0.45em', '0em'],
                       delay: stagger(timing.each, { start: delay }), duration: timing.move, ease: 'outExpo' });
    } else if (style === 'spring_pop') {
      animate(spans, { opacity: [0, 1], scale: [0.2, 1], translateY: ['0.3em', '0em'], delay: stagger(p.stagger || 45, { start: delay }),
                       ease: spring(260, 12) });
    } else if (style === 'mask_up') {
      spans.forEach((span) => { const mask = document.createElement('span'); mask.style.display = 'inline-block'; mask.style.overflow = 'hidden';
        span.replaceWith(mask); mask.appendChild(span); });
      const timing = fit(p.stagger || 80, 700);
      animate(spans, { translateY: ['110%', '0%'], delay: stagger(timing.each, { start: delay }), duration: timing.move, ease: 'outQuart' });
    } else if (style === 'glyph_in') {
      // Whole glyphs one after another, each fading in with a small rise that settles: nothing is masked or clipped,
      // so no glyph is ever seen cut in half (mask_up shows half glyphs mid-move). Laid out by words like mask_up, so
      // it rests in the same place; a Latin word or a number arrives as one piece; the stagger spreads over the
      // layer's own time, up to `stagger` ms a glyph.
      const timing = fit(p.stagger || 150, p.duration_ms || 420, pieces.length);
      pieces.forEach((piece, index) => {
        const base = piece.style.transform || '';   // the badge keeps its tilt
        piece.animate([{ opacity: 0, transform: `${base} translateY(0.16em) scale(0.86)` },
                       { opacity: 1, transform: `${base} translateY(-0.02em) scale(1.02)`, offset: 0.6 },
                       { opacity: 1, transform: `${base} translateY(0) scale(1)` }],
                      { delay: delay + index * timing.each, duration: timing.move, easing: 'cubic-bezier(.25,.7,.3,1)', fill: 'backwards' });
      });   // nothing held after it lands: at rest the title is the same pixels as mask_up's
    } else if (style === 'typewriter') {
      // character by character, also inside a whole Latin word
      const glyphs = spans.flatMap((span) => (span.children.length ? [...span.children] : [span]));
      const timing = fit(p.stagger || 60, 1, glyphs.length);
      glyphs.forEach((glyph, index) => glyph.animate([{ opacity: 0 }, { opacity: 1 }], { delay: delay + index * timing.each, duration: 1, fill: 'both' }));
    } else if (style === 'slam') {
      // Each word lands hard, 1.55x -> 0.94x -> 1x in under 0.3 s (the "砸入" beat), and the frame jolts as it lands.
      spans.forEach((span, index) => {
        const at = delay + index * (p.stagger || 260);
        span.animate([{ transform: 'scale(1.55)', opacity: 0 }, { transform: 'scale(0.94)', opacity: 1, offset: 0.6 }, { transform: 'scale(1)', opacity: 1 }],
                     { delay: at, duration: 290, easing: 'cubic-bezier(.2,.9,.3,1)', fill: 'both' });
        if (p.shake !== false) {
          const k = size / 24;
          box.animate([{ transform: 'translate(0,0)' }, { transform: `translate(${-k}px,${k / 2}px)` }, { transform: `translate(${k * 0.6}px,${-k / 3}px)` },
                       { transform: `translate(${-k / 4}px,${k / 6}px)` }, { transform: 'translate(0,0)' }], { delay: at + 170, duration: 240 });
        }
      });
    } else if (style === 'scramble') {
      // Glyphs flicker through look-alikes and resolve left to right into the line (decoder / terminal feel).
      const glyphs = spans.flatMap((span) => [...span.children]);
      const pool = p.pool ? [...p.pool] : [...new Set([...(p.text || '')].filter((c) => c.trim()))].concat([...'#%&*+=?<>/0123456789']);
      const each = p.stagger || 45, settle = p.duration_ms || 700;
      glyphs.forEach((glyph) => { glyph.dataset.final = glyph.textContent; glyph.dataset.color = glyph.style.color; glyph.style.opacity = 0; });
      const tick = () => {
        const t = performance.now() - delay;
        glyphs.forEach((glyph, index) => {
          if (t < index * each * 0.4) { glyph.style.opacity = 0; return; }
          glyph.style.opacity = 1;
          const done = t >= index * each + settle;
          glyph.textContent = done ? glyph.dataset.final : pool[Math.floor(Math.random() * pool.length)];
          glyph.style.color = done ? glyph.dataset.color : (p.scramble_color || '#8C96A3');
        });
        requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    } else if (style === 'word_by_word') {
      // One word at a time on the beat; a key word arrives big and shrinks into the line, pushing its neighbours.
      spans.forEach((span, index) => {
        const at = delay + index * (p.stagger || 240);
        span.animate([{ opacity: 0, transform: 'translateY(0.25em)' }, { opacity: 1, transform: 'translateY(0)' }],
                     { delay: at, duration: 220, easing: 'cubic-bezier(.2,.8,.2,1)', fill: 'both' });
        if ([...span.children].some((glyph) => glyph.style.color)) {
          span.animate([{ fontSize: '1.45em' }, { fontSize: '1.45em', offset: 0.45 }, { fontSize: '1em' }],
                       { delay: at, duration: 760, easing: 'cubic-bezier(.3,1.35,.5,1)', fill: 'both' });
        }
      });
    }
    if (p.underline) {
      const layer = svgLayer(root, ctx.width, ctx.height);
      const y = Math.min(ctx.height - 6, ctx.height / 2 + 0.62 * size);
      const node = rough.svg(layer).line(ctx.width * 0.12, y, ctx.width * 0.88, y + 4,
        { stroke: p.accent || '#E9784E', strokeWidth: Math.max(4, size / 14), roughness: 1.6, bowing: 2, seed: p.seed || 7 });
      layer.appendChild(node);
      drawOn(node, delay + spans.length * (p.stagger || 70) + 300, 500);
    }
  };

  // A hand-drawn annotation around the overlay box: underline, box, circle, highlight, strike-through, bracket...
  Recipes.rough_annotate = (root, p) => {
    // The mark is drawn around the target with padding and a wobbly stroke: keep the target inset so it stays in the box.
    const inset = p.inset ?? ((p.padding ?? 8) + 2 * (p.stroke_width || 5) + 10);
    const target = document.createElement('div');
    Object.assign(target.style, { position: 'absolute', left: `${inset}px`, top: `${inset}px`, right: `${inset}px`, bottom: `${inset}px` });
    root.appendChild(target);
    const annotation = RoughNotation.annotate(target, { type: p.type || 'circle', color: p.color || '#E9784E',
      strokeWidth: p.stroke_width || 5, padding: p.padding ?? 8, iterations: p.iterations || 2,
      animationDuration: p.duration_ms || 800, brackets: p.brackets || ['left', 'right'], multiline: false });
    setTimeout(() => annotation.show(), p.delay || 0);
  };

  // Sketchy shapes drawn on in order: arrows, curves, boxes, ellipses, brackets, checks, crosses, stars.
  Recipes.sketch_shapes = (root, p, ctx) => {
    const layer = svgLayer(root, ctx.width, ctx.height);
    const pen = rough.svg(layer);
    let cursor = p.delay || 0;
    const each = (p.duration_ms || 900) / Math.max(1, (p.shapes || []).length);
    (p.shapes || []).forEach((shape, index) => {
      const o = { stroke: shape.color || p.color || '#1E2733', strokeWidth: shape.width || p.width || 5, roughness: shape.roughness ?? 1.4,
                  bowing: shape.bowing ?? 1.5, seed: (p.seed || 3) + index, fill: shape.fill, fillStyle: shape.fill_style || 'hachure',
                  hachureGap: shape.hachure_gap || 10, fillWeight: shape.fill_weight || 3 };
      const [x1, y1, x2, y2] = shape.box || [0, 0, ctx.width, ctx.height];
      let node;
      const k = shape.kind;
      if (k === 'arrow' || k === 'line' || k === 'curve_arrow') {
        const [[ax, ay], [bx, by]] = shape.points;
        const bend = k === 'curve_arrow' ? (shape.bend ?? 0.25) : 0;
        const mx = (ax + bx) / 2 - (by - ay) * bend, my = (ay + by) / 2 + (bx - ax) * bend;
        node = document.createElementNS(NS, 'g');
        const shaft = bend ? pen.curve([[ax, ay], [mx, my], [bx, by]], o) : pen.line(ax, ay, bx, by, o);
        node.appendChild(shaft);
        if (k !== 'line') {
          const angle = Math.atan2(by - (bend ? my : ay), bx - (bend ? mx : ax));
          const size = shape.head || 28;
          for (const side of [-1, 1]) {
            const a = angle + Math.PI + side * 0.5;
            node.appendChild(pen.line(bx, by, bx + size * Math.cos(a), by + size * Math.sin(a), o));
          }
        }
      } else if (k === 'rect') node = pen.rectangle(x1, y1, x2 - x1, y2 - y1, o);
      else if (k === 'ellipse' || k === 'circle') node = pen.ellipse((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, o);
      else if (k === 'polygon') node = pen.polygon(shape.points, o);
      else if (k === 'check') node = pen.linearPath([[x1, (y1 + y2) / 2], [x1 + (x2 - x1) * 0.4, y2], [x2, y1]], o);
      else if (k === 'cross') { node = document.createElementNS(NS, 'g'); node.appendChild(pen.line(x1, y1, x2, y2, o)); node.appendChild(pen.line(x2, y1, x1, y2, o)); }
      else if (k === 'bracket') node = pen.linearPath([[x1 + 20, y1], [x1, y1], [x1, y2], [x1 + 20, y2]], o);
      else if (k === 'star') {
        const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2, r = Math.min(x2 - x1, y2 - y1) / 2;
        node = pen.polygon([...Array(10)].map((_, i) => { const a = -Math.PI / 2 + i * Math.PI / 5, rr = i % 2 ? r * 0.45 : r;
          return [cx + rr * Math.cos(a), cy + rr * Math.sin(a)]; }), o);
      } else throw new Error('Unknown sketch shape: ' + k);
      layer.appendChild(node);
      cursor = drawOn(node, cursor, shape.duration_ms || each) + (p.gap_ms ?? 60);
      if (shape.flow) {  // marching dashes along the arrow after it is drawn: a flow, a signal, a path taken
        node.querySelectorAll('path').forEach((path) => {
          path.animate([{ strokeDasharray: '14 12', strokeDashoffset: 0 }, { strokeDasharray: '14 12', strokeDashoffset: -52 }],
                       { delay: cursor, duration: 900, iterations: Infinity, easing: 'linear' });
        });
      }
    });
  };

  // Particles: confetti, sparkle stars or hearts bursting from the box centre (deterministic: seeded Math.random).
  Recipes.burst = (root, p, ctx) => {
    const kind = p.kind || 'sparkle';
    const canvas = document.createElement('canvas');
    canvas.width = ctx.width * ctx.scale; canvas.height = ctx.height * ctx.scale;
    Object.assign(canvas.style, { position: 'absolute', inset: '0', width: `${ctx.width}px`, height: `${ctx.height}px` });
    root.appendChild(canvas);
    if (kind === 'confetti') {
      const fire = confetti.create(canvas, { resize: false, useWorker: false });
      setTimeout(() => fire({ particleCount: p.count || 90, spread: p.spread || 80, startVelocity: p.velocity || 38,
        origin: { x: p.origin_x ?? 0.5, y: p.origin_y ?? 0.6 }, colors: p.colors || ['#E9784E', '#F2C14E', '#4FA3A5', '#1E2733'],
        scalar: (p.particle_scale || 1.1) * ctx.scale, ticks: p.ticks || 220, gravity: p.gravity ?? 1, disableForReducedMotion: false }), p.delay || 0);
      return;
    }
    const g = canvas.getContext('2d');
    const count = p.count || 14;
    const colors = p.colors || ['#F2C14E', '#E9784E', '#4FA3A5'];
    const parts = [...Array(count)].map((_, i) => ({ a: (i / count) * Math.PI * 2 + Math.random() * 0.4, v: 0.6 + Math.random() * 0.6,
      s: 0.6 + Math.random() * 0.8, c: colors[i % colors.length], r: Math.random() * Math.PI }));
    const start = p.delay || 0, life = p.life_ms || 900;
    const cx = canvas.width / 2, cy = canvas.height / 2, reach = Math.min(canvas.width, canvas.height) * 0.45;
    const shape = (x, y, size, rotation) => {
      g.save(); g.translate(x, y); g.rotate(rotation); g.beginPath();
      if (kind === 'heart') { g.moveTo(0, size * 0.3); g.bezierCurveTo(size, -size * 0.6, size * 0.4, -size, 0, -size * 0.35);
        g.bezierCurveTo(-size * 0.4, -size, -size, -size * 0.6, 0, size * 0.3); }
      else { for (let i = 0; i < 8; i++) { const a = i * Math.PI / 4, rr = i % 2 ? size * 0.25 : size; g.lineTo(rr * Math.cos(a), rr * Math.sin(a)); } }
      g.closePath(); g.fill(); g.restore();
    };
    const draw = (time) => {
      g.clearRect(0, 0, canvas.width, canvas.height);
      const t = (time - start) / life;
      if (t >= 0 && t <= 1) {
        const ease = 1 - Math.pow(1 - t, 3);
        for (const q of parts) {
          g.globalAlpha = Math.min(1, 4 * (1 - t)); g.fillStyle = q.c;
          shape(cx + Math.cos(q.a) * reach * q.v * ease, cy + Math.sin(q.a) * reach * q.v * ease,
                (p.size || 18) * ctx.scale * q.s * (t < 0.2 ? t / 0.2 : 1 - 0.5 * t), q.r + t * 2);
        }
      }
      requestAnimationFrame(draw);
    };
    requestAnimationFrame(draw);
  };

  // An odometer: each digit column rolls to its value with a spring, staggered right to left.
  Recipes.number_ticker = (root, p) => {
    const value = Number(p.to), decimals = p.decimals || 0;
    const text = `${p.prefix || ''}${value.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}${p.suffix || ''}`;
    const row = document.createElement('div');
    Object.assign(row.style, { position: 'absolute', inset: '0', display: 'flex', alignItems: 'center',
      justifyContent: p.align === 'left' ? 'flex-start' : 'center', fontFamily: family(p.font || 'hand'), fontSize: `${p.size || 120}px`,
      fontWeight: p.weight || 800, color: p.color || '#1E2733', fontVariantNumeric: 'tabular-nums', lineHeight: 1 });
    root.appendChild(row);
    const columns = [];
    [...text].forEach((character) => {
      const cell = document.createElement('span');
      Object.assign(cell.style, { display: 'inline-block', height: '1.1em', overflow: 'hidden', position: 'relative' });
      if (/\d/.test(character)) {
        const strip = document.createElement('span');
        Object.assign(strip.style, { display: 'flex', flexDirection: 'column' });
        const turns = p.spins ?? 1;  // full turns of the wheel before it settles
        for (let d = 0; d < 10 * (turns + 1); d++) { const digit = document.createElement('span'); digit.textContent = d % 10; digit.style.height = '1.1em'; strip.appendChild(digit); }
        cell.appendChild(strip); columns.push([strip, Number(character) + 10 * turns]);
      } else { cell.textContent = character; if (p.accent_suffix && !/[\d.,]/.test(character)) cell.style.color = p.accent || '#E9784E'; }
      row.appendChild(cell);
    });
    columns.reverse().forEach(([strip, index], order) => {
      animate(strip, { translateY: ['0em', `${-1.1 * index}em`], delay: (p.delay || 0) + order * (p.stagger || 90), ease: spring(120, 14) });
    });
  };

  // A rough speech bubble or tag that springs out of its anchor with its text.
  Recipes.callout = (root, p, ctx) => {
    const layer = svgLayer(root, ctx.width, ctx.height);
    const pad = 10, tail = p.tail || 'bottom-left';
    const w = ctx.width - 2 * pad, h = ctx.height - 2 * pad - (p.tail ? 30 : 0);
    const o = { stroke: p.stroke || '#1E2733', strokeWidth: p.stroke_width || 4, fill: p.fill || '#FFFFFF', fillStyle: 'solid', roughness: 1.2, seed: p.seed || 5 };
    const group = document.createElementNS(NS, 'g');
    group.appendChild(rough.svg(layer).rectangle(pad, pad, w, h, o));
    if (p.tail) {
      const x = tail.endsWith('right') ? pad + w * 0.75 : pad + w * 0.2;
      group.appendChild(rough.svg(layer).polygon([[x, pad + h - 2], [x + 34, pad + h - 2], [x - 6, pad + h + 30]], o));
    }
    layer.appendChild(group);
    const label = document.createElement('div');
    Object.assign(label.style, { position: 'absolute', left: `${pad}px`, top: `${pad}px`, width: `${w}px`, height: `${h}px`, display: 'flex',
      alignItems: 'center', justifyContent: 'center', textAlign: 'center', fontFamily: family(p.font || 'hand'),
      fontSize: `${p.size || 54}px`, color: p.color || '#1E2733', padding: '0 0.4em', boxSizing: 'border-box', lineHeight: 1.2 });
    const words = document.createElement('div');   // the flex item whose lines wrap between words
    words.style.minWidth = '0';
    label.appendChild(words);
    root.appendChild(label);
    setWords(words, p.text || '', w - 0.8 * (p.size || 54));
    root.style.transformOrigin = tail.endsWith('right') ? '75% 100%' : '20% 100%';
    animate(root, { scale: [0.3, 1], opacity: [0, 1], rotate: [p.tilt ?? -6, 0], delay: p.delay || 0, ease: spring(220, 13) });
  };

  // A hand-drawn ring that is inked to a percentage while its number counts up (rough.js, drawn on like a pen stroke).
  Recipes.progress_ring = (root, p, ctx) => {
    const layer = svgLayer(root, ctx.width, ctx.height), pen = rough.svg(layer);
    const size = Math.min(ctx.width, ctx.height), r = size * 0.36, cx = ctx.width / 2, cy = ctx.height / 2;
    const share = Math.max(0, Math.min(1, (p.value || 0) / (p.max || 100)));
    const track = pen.circle(cx, cy, 2 * r, { stroke: p.track || '#D8D2C4', strokeWidth: size * 0.05, roughness: 1.1, seed: p.seed || 4 });
    layer.appendChild(track);
    const start = -Math.PI / 2, end = start + 2 * Math.PI * Math.max(0.001, share);
    const arc = pen.arc(cx, cy, 2 * r, 2 * r, start, end, false, { stroke: p.color || '#E9784E', strokeWidth: size * 0.075, roughness: 1.3, seed: (p.seed || 4) + 1 });
    layer.appendChild(arc);
    drawOn(track, p.delay || 0, 260);
    drawOn(arc, (p.delay || 0) + 200, p.duration_ms || 1100);
    const label = document.createElement('div');
    Object.assign(label.style, { position: 'absolute', inset: '0', display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontFamily: family(p.font || 'hand'), fontWeight: 700, fontSize: `${size * 0.22}px`, color: p.text_color || '#1E2733' });
    root.appendChild(label);
    const counter = { v: 0 };
    animate(counter, { v: p.value || 0, delay: (p.delay || 0) + 200, duration: p.duration_ms || 1100, ease: 'outCubic',
      onUpdate: () => { label.textContent = `${counter.v.toFixed(p.decimals || 0)}${p.suffix ?? '%'}`; } });
  };

  // ---------------------------------------------------------------- 2.0 recipes

  const PALETTE = ['#E9784E', '#4FA3A5', '#F2C14E', '#1E2733', '#8C6BB1', '#6BA368'];
  const ink = (p) => p.ink || '#1E2733';
  function label(root, text, x, y, size, p, align = 'center', color) {
    const node = document.createElement('div');
    node.textContent = text;
    Object.assign(node.style, { position: 'absolute', left: `${x}px`, top: `${y}px`, transform: `translate(${align === 'center' ? '-50%' : align === 'right' ? '-100%' : '0'}, -50%)`,
      fontFamily: family(p.font || 'hand'), fontSize: `${size}px`, color: color || ink(p), whiteSpace: 'nowrap', lineHeight: 1 });
    root.appendChild(node);
    return node;
  }
  const appear = (node, at, duration = 260) => node.animate([{ opacity: 0, transform: `${node.style.transform} translateY(0.3em)` }, { opacity: 1, transform: node.style.transform }],
                                                            { delay: at, duration, easing: 'cubic-bezier(.2,.8,.2,1)', fill: 'both' });
  const ease = (t) => (t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t));

  // Outline of a shape as N points, evenly spaced along its edge and starting at its top (so shapes morph without twisting).
  function outline(shape, count) {
    const [x1, y1, x2, y2] = shape.box, w = x2 - x1, h = y2 - y1, cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
    const kind = shape.shape || 'circle';
    let d;
    if (kind === 'circle' || kind === 'dot' || kind === 'ellipse') d = `M ${cx} ${y1} A ${w / 2} ${h / 2} 0 1 1 ${cx - 0.01} ${y1} Z`;
    else if (kind === 'rect' || kind === 'card') {
      const r = Math.min(shape.radius ?? Math.min(w, h) * 0.12, w / 2, h / 2);
      d = `M ${cx} ${y1} H ${x2 - r} Q ${x2} ${y1} ${x2} ${y1 + r} V ${y2 - r} Q ${x2} ${y2} ${x2 - r} ${y2} H ${x1 + r} Q ${x1} ${y2} ${x1} ${y2 - r} V ${y1 + r} Q ${x1} ${y1} ${x1 + r} ${y1} Z`;
    } else if (kind === 'star') {
      const pts = [...Array(10)].map((_, i) => { const a = -Math.PI / 2 + i * Math.PI / 5, k = i % 2 ? 0.45 : 1; return [cx + k * w / 2 * Math.cos(a), cy + k * h / 2 * Math.sin(a)]; });
      d = 'M ' + pts.map((q) => q.join(' ')).join(' L ') + ' Z';
    } else if (kind === 'heart') {
      d = `M ${cx} ${y1 + h * 0.28} C ${cx + w * 0.1} ${y1 - h * 0.05}, ${x2 + w * 0.05} ${y1 + h * 0.1}, ${x2 - w * 0.02} ${y1 + h * 0.38} C ${x2 - w * 0.1} ${y1 + h * 0.62}, ${cx + w * 0.2} ${y1 + h * 0.8}, ${cx} ${y2} C ${cx - w * 0.2} ${y1 + h * 0.8}, ${x1 + w * 0.1} ${y1 + h * 0.62}, ${x1 + w * 0.02} ${y1 + h * 0.38} C ${x1 - w * 0.05} ${y1 + h * 0.1}, ${cx - w * 0.1} ${y1 - h * 0.05}, ${cx} ${y1 + h * 0.28} Z`;
    } else if (kind === 'triangle') d = `M ${cx} ${y1} L ${x2} ${y2} L ${x1} ${y2} Z`;
    else if (kind === 'diamond') d = `M ${cx} ${y1} L ${x2} ${cy} L ${cx} ${y2} L ${x1} ${cy} Z`;
    else if (kind === 'arrow') d = `M ${x1 + w * 0.6} ${y1} L ${x2} ${cy} L ${x1 + w * 0.6} ${y2} L ${x1 + w * 0.6} ${cy + h * 0.2} L ${x1} ${cy + h * 0.2} L ${x1} ${cy - h * 0.2} L ${x1 + w * 0.6} ${cy - h * 0.2} Z`;
    else if (kind === 'path') d = shape.d;
    else throw new Error('Unknown morph shape: ' + kind);
    const probe = document.createElementNS(NS, 'path');
    probe.setAttribute('d', d);
    const length = probe.getTotalLength();
    let points = [...Array(count)].map((_, i) => { const q = probe.getPointAtLength(length * i / count); return [q.x, q.y]; });
    const top = points.reduce((best, q, i) => (q[1] < points[best][1] - 0.5 || (Math.abs(q[1] - points[best][1]) <= 0.5 && q[0] < points[best][0]) ? i : best), 0);
    points = points.slice(top).concat(points.slice(0, top));
    // Clockwise in screen space, so every shape winds the same way.
    const area = points.reduce((sum, q, i) => { const r = points[(i + 1) % count]; return sum + q[0] * r[1] - r[0] * q[1]; }, 0);
    return area < 0 ? [points[0]].concat(points.slice(1).reverse()) : points;
  }

  // One shape becomes the next, travelling as it changes (a dot grows into a card, the card into a star...): it
  // stretches along its path first and takes the new form in the second half; colours carry across.
  // shapes: [{shape: circle|dot|rect|card|star|heart|triangle|diamond|arrow|path, box: [x1,y1,x2,y2], fill, stroke, text, hold_ms}]
  Recipes.morph = (root, p, ctx) => {
    const layer = svgLayer(root, ctx.width, ctx.height);
    const count = 120, shapes = p.shapes || [];
    if (shapes.length < 1) throw new Error('morph needs shapes');
    const forms = shapes.map((shape) => outline(shape, count));
    const centres = forms.map((points) => points.reduce((a, q) => [a[0] + q[0] / count, a[1] + q[1] / count], [0, 0]));
    const path = document.createElementNS(NS, 'path');
    path.setAttribute('stroke-width', p.stroke_width || 5); path.setAttribute('stroke-linejoin', 'round');
    layer.appendChild(path);
    const texts = shapes.map((shape) => { if (!shape.text) return null; const [x1, y1, x2, y2] = shape.box;
      const node = label(root, shape.text, (x1 + x2) / 2, (y1 + y2) / 2, shape.text_size || Math.min(x2 - x1, y2 - y1) * 0.3, p, 'center', shape.text_color || '#FFFFFF');
      node.style.opacity = 0; return node; });
    const morphMs = p.morph_ms || 700, start = p.delay || 0;
    const plan = []; let cursor = start;
    shapes.forEach((shape, index) => { plan.push({ index, from: cursor, hold: shape.hold_ms ?? 500 }); cursor += (shape.hold_ms ?? 500) + morphMs; });
    const color = (shape, key, fallback) => shape[key] || p[key] || fallback;
    const draw = () => {
      const now = performance.now();
      let index = 0;
      while (index + 1 < plan.length && now >= plan[index + 1].from) index += 1;
      const stay = plan[index], next = Math.min(index + 1, shapes.length - 1);
      const t = index + 1 < shapes.length ? Math.max(0, Math.min(1, (now - stay.from - stay.hold) / morphMs)) : 0;
      const move = ease(t), form = ease(Math.max(0, (t - 0.3) / 0.7));
      const [ax, ay] = centres[index], [bx, by] = centres[next];
      const cx = ax + (bx - ax) * move, cy = ay + (by - ay) * move;
      const dx = bx - ax, dy = by - ay, distance = Math.hypot(dx, dy) || 1, stretch = 0.28 * Math.sin(Math.PI * t) * Math.min(1, distance / 200);
      const ux = dx / distance, uy = dy / distance;
      const points = forms[index].map((q, i) => {
        const r = forms[next][i];
        let x = (q[0] - ax) + ((r[0] - bx) - (q[0] - ax)) * form, y = (q[1] - ay) + ((r[1] - by) - (q[1] - ay)) * form;
        const along = x * ux + y * uy;
        x += ux * along * stretch; y += uy * along * stretch;
        return [cx + x, cy + y];
      });
      path.setAttribute('d', 'M ' + points.map((q) => `${q[0].toFixed(2)} ${q[1].toFixed(2)}`).join(' L ') + ' Z');
      const mix = (key, fallback) => {
        const [a, b] = [color(shapes[index], key, fallback), color(shapes[next], key, fallback)].map((hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16)));
        return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * form)).join(',')})`;
      };
      path.setAttribute('fill', mix('fill', '#E9784E')); path.setAttribute('stroke', mix('stroke', ink(p)));
      const appearAt = now < start ? 0 : Math.min(1, (now - start) / 220);
      path.setAttribute('opacity', appearAt);
      texts.forEach((node, i) => { if (node) node.style.opacity = i === index && t === 0 && now >= stay.from + 80 ? 1 : i === next && form > 0.85 ? (form - 0.85) / 0.15 : 0; });
      requestAnimationFrame(draw);
    };
    requestAnimationFrame(draw);
  };

  // Hand-drawn charts that draw themselves: axes first, then the data, then one highlighted fact.
  // kind: bar | line | pie | dots | tally | timeline (see references/motion-catalog.md for fields).
  Recipes.rough_chart = (root, p, ctx) => {
    const layer = svgLayer(root, ctx.width, ctx.height), pen = rough.svg(layer);
    const W = ctx.width, H = ctx.height, kind = p.kind || 'bar', colors = p.colors || PALETTE, size = p.label_size || Math.max(26, Math.min(W, H) * 0.068);
    const o = (extra = {}) => ({ stroke: ink(p), strokeWidth: p.stroke_width || 4, roughness: 1.3, bowing: 1.2, seed: p.seed || 9, ...extra });
    const start = p.delay || 0;
    const left = W * 0.1, right = W * 0.96, top = H * 0.1, bottom = H * (p.labels ? 0.84 : 0.92);
    const values = p.values || [], highlight = p.highlight ?? -1;
    const fmt = (v) => `${p.prefix || ''}${Number(v).toLocaleString('en-US', { maximumFractionDigits: p.decimals || 0 })}${p.suffix || ''}`;
    const count = (node, value, at, duration) => { const c = { v: 0 }; node.textContent = fmt(0);
      animate(c, { v: value, delay: at, duration, ease: 'outCubic', onUpdate: () => { node.textContent = fmt(c.v); } }); };
    if (kind === 'bar' || kind === 'line') {
      const axes = document.createElementNS(NS, 'g');
      axes.appendChild(pen.line(left, bottom, right, bottom, o())); axes.appendChild(pen.line(left, bottom, left, top, o()));
      layer.appendChild(axes);
      let cursor = drawOn(axes, start, 500) + 80;
      const max = p.max || Math.max(...values) * 1.12 || 1, slot = (right - left) / Math.max(1, values.length);
      const yOf = (v) => bottom - (bottom - top) * v / max;
      if (kind === 'bar') {
        values.forEach((value, i) => {
          const x = left + slot * (i + 0.2), w = slot * 0.6, y = yOf(value);
          const bar = document.createElementNS(NS, 'g');
          bar.appendChild(pen.rectangle(x, y, w, bottom - y, o({ fill: colors[(i === highlight ? 0 : (highlight >= 0 ? 3 : i)) % colors.length], fillStyle: 'hachure', hachureGap: 9, fillWeight: 3 })));
          bar.style.transformBox = 'view-box'; bar.style.transformOrigin = `0px ${bottom}px`;
          layer.appendChild(bar);
          const at = cursor + i * (p.stagger || 220);
          bar.animate([{ transform: 'scaleY(0)' }, { transform: 'scaleY(1)' }], { delay: at, duration: 650, easing: 'cubic-bezier(.3,1.25,.5,1)', fill: 'both' });
          const tag = label(root, '', x + w / 2, y - size * 0.8, size, p, 'center', i === highlight ? colors[0] : ink(p));
          appear(tag, at + 200); count(tag, value, at, 800);
          if (p.labels) appear(label(root, p.labels[i], x + w / 2, bottom + size * 0.9, size * 0.9, p), at);
        });
      } else {
        const pts = values.map((value, i) => [left + slot * (i + 0.5), yOf(value)]);
        const line = pen.curve(pts, o({ stroke: colors[0], strokeWidth: (p.stroke_width || 4) * 1.4 }));
        layer.appendChild(line);
        const end = drawOn(line, cursor, p.duration_ms || 1400);
        pts.forEach(([x, y], i) => {
          const at = cursor + (end - cursor) * i / Math.max(1, pts.length - 1);
          const dot = pen.circle(x, y, size * 0.6, o({ fill: i === highlight ? colors[0] : '#FFFFFF', fillStyle: 'solid' }));
          dot.style.transformBox = 'fill-box'; dot.style.transformOrigin = 'center'; layer.appendChild(dot);
          dot.animate([{ transform: 'scale(0)' }, { transform: 'scale(1)' }], { delay: at, duration: 380, easing: 'cubic-bezier(.3,1.6,.5,1)', fill: 'both' });
          if (p.show_values !== false) appear(label(root, fmt(values[i]), x, y - size * 1.1, size * 0.85, p, 'center', i === highlight ? colors[0] : ink(p)), at + 100);
          if (p.labels) appear(label(root, p.labels[i], x, bottom + size * 0.9, size * 0.9, p), at);
        });
      }
    } else if (kind === 'pie') {
      const cx = W / 2, cy = H / 2, r = Math.min(W, H) * 0.36, total = values.reduce((a, b) => a + b, 0) || 1;
      let angle = -Math.PI / 2;
      values.forEach((value, i) => {
        const sweep = 2 * Math.PI * value / total, at = start + i * (p.stagger || 320);
        const slice = pen.arc(cx, cy, 2 * r, 2 * r, angle, angle + sweep, true, o({ fill: colors[i % colors.length], fillStyle: i === highlight ? 'solid' : 'hachure', hachureGap: 8 }));
        slice.style.transformBox = 'view-box'; slice.style.transformOrigin = `${cx}px ${cy}px`; layer.appendChild(slice);
        const pop = i === highlight ? 1.08 : 1;
        slice.animate([{ transform: 'scale(0) rotate(-20deg)' }, { transform: `scale(${pop})` }], { delay: at, duration: 600, easing: 'cubic-bezier(.3,1.3,.5,1)', fill: 'both' });
        const middle = angle + sweep / 2;
        const text = (p.labels ? p.labels[i] + ' ' : '') + (p.show_values === false ? '' : fmt(value));
        appear(label(root, text, cx + Math.cos(middle) * r * 1.28, cy + Math.sin(middle) * r * 1.22, size, p), at + 250);
        angle += sweep;
      });
    } else if (kind === 'dots') {
      const total = p.total || 100, marked = p.marked ?? 0, columns = p.columns || Math.ceil(Math.sqrt(total * W / H));
      const rows = Math.ceil(total / columns), cell = Math.min((right - left) / columns, (bottom - top) / rows), r = cell * 0.32;
      const x0 = (W - cell * columns) / 2 + cell / 2, y0 = (H - cell * rows) / 2 + cell / 2;
      for (let i = 0; i < total; i++) {
        const x = x0 + (i % columns) * cell, y = y0 + Math.floor(i / columns) * cell, on = i < marked;
        const dot = pen.circle(x, y, 2 * r, o({ strokeWidth: 2, fill: on ? colors[0] : '#D8D2C4', fillStyle: on ? 'cross-hatch' : 'solid',
                                                 hachureGap: Math.max(2.5, r * 0.45), fillWeight: Math.max(1, r * 0.18), roughness: 1.4,
                                                 stroke: on ? colors[0] : '#B9B1A1', seed: (p.seed || 9) + i }));
        dot.style.transformBox = 'fill-box'; dot.style.transformOrigin = 'center'; layer.appendChild(dot);
        const at = start + (on ? 400 + i * (p.stagger || 25) : i * 4);
        dot.animate([{ transform: 'scale(0)' }, { transform: 'scale(1)' }], { delay: at, duration: 300, easing: 'cubic-bezier(.3,1.6,.5,1)', fill: 'both' });
      }
    } else if (kind === 'tally') {
      // Counting the Chinese way: every 正 is five strokes, in stroke order (or four bars and a slash with style "bars").
      const total = p.count || 5, groups = Math.ceil(total / 5), g = Math.min((right - left) / Math.max(1, groups) * 0.8, (bottom - top) * 0.8);
      let cursor = start;
      for (let k = 0; k < groups; k++) {
        const x = left + (right - left) * (k + 0.5) / groups - g / 2, y = (H - g) / 2;
        const strokes = p.style === 'bars'
          ? [[x + g * 0.1, y, x + g * 0.1, y + g], [x + g * 0.33, y, x + g * 0.33, y + g], [x + g * 0.56, y, x + g * 0.56, y + g], [x + g * 0.79, y, x + g * 0.79, y + g], [x - g * 0.05, y + g * 0.75, x + g * 0.95, y + g * 0.25]]
          : [[x + g * 0.1, y + g * 0.05, x + g * 0.9, y + g * 0.05], [x + g * 0.5, y + g * 0.05, x + g * 0.5, y + g * 0.95], [x + g * 0.5, y + g * 0.48, x + g * 0.85, y + g * 0.48],
             [x + g * 0.22, y + g * 0.42, x + g * 0.22, y + g * 0.95], [x, y + g * 0.95, x + g, y + g * 0.95]];
        strokes.slice(0, Math.min(5, total - 5 * k)).forEach((q, i) => {
          const node = pen.line(...q, o({ strokeWidth: g * 0.07, stroke: p.color || ink(p), roughness: 0.9 }));
          layer.appendChild(node); cursor = drawOn(node, cursor, p.stroke_ms || 180) + 40;
        });
        cursor += 120;
      }
    } else if (kind === 'timeline') {
      const events = p.events || [], vertical = H > W * 1.2;
      const axis = vertical ? pen.line(W * 0.2, top, W * 0.2, bottom, o()) : pen.line(left, H / 2, right, H / 2, o());
      layer.appendChild(axis);
      let cursor = drawOn(axis, start, 600);
      events.forEach((event, i) => {
        const f = events.length > 1 ? i / (events.length - 1) : 0.5;
        // horizontal: the end dots sit far enough in that a centred label (up to ~6 glyphs) stays inside the box
        const x = vertical ? W * 0.2 : left + (right - left) * (0.1 + 0.8 * f), y = vertical ? top + (bottom - top) * (0.06 + 0.88 * f) : H / 2;
        const at = cursor + i * (p.stagger || 380);
        const dot = pen.circle(x, y, size * 0.8, o({ fill: i === highlight ? colors[0] : '#FFFFFF', fillStyle: 'solid' }));
        dot.style.transformBox = 'fill-box'; dot.style.transformOrigin = 'center'; layer.appendChild(dot);
        dot.animate([{ transform: 'scale(0)' }, { transform: 'scale(1)' }], { delay: at, duration: 360, easing: 'cubic-bezier(.3,1.6,.5,1)', fill: 'both' });
        const above = i % 2 === 0;
        if (event.when) appear(label(root, event.when, vertical ? x - size * 0.9 : x, vertical ? y : y + (above ? size * 1.3 : -size * 1.3), size * 0.8, p, vertical ? 'right' : 'center', colors[0]), at + 80);
        appear(label(root, event.label || '', vertical ? x + size * 0.9 : x, vertical ? y : y + (above ? -size * 1.3 : size * 1.3), size, p, vertical ? 'left' : 'center'), at + 140);
      });
    } else throw new Error('Unknown chart kind: ' + kind);
  };

  // A route over the world drawn as it is travelled: land outlines only (no borders), a great-circle line inked up to
  // a moving marker, stops popping as the marker passes them. projection "globe" turns to follow the marker; "flat" fits
  // the route. route: [[lon, lat], ...]; stops: [{at: [lon, lat], label}].
  // A short route zooms in (the history film's trip across Central Asia was a 100 px stub on a half globe, its four
  // labels one black knot): when the route would cover under half the box's short side, the camera flies from the whole
  // globe down to the route during the intro and holds there, the sphere cut by a rounded frame just inside the box.
  // zoom: a number sets the factor, false keeps the turning globe; max_zoom caps auto (40).
  Recipes.map_route = (root, p, ctx) => {
    const W = ctx.width, H = ctx.height, scale = ctx.scale;
    const canvas = document.createElement('canvas');
    canvas.width = W * scale; canvas.height = H * scale;
    Object.assign(canvas.style, { position: 'absolute', inset: '0', width: `${W}px`, height: `${H}px` });
    root.appendChild(canvas);
    const g = canvas.getContext('2d'); g.scale(scale, scale);
    const land = topojson.feature(WORLD_LAND, WORLD_LAND.objects.land);
    const route = p.route || [];
    if (route.length < 2) throw new Error('map_route needs a route of at least two [lon, lat] points');
    const legs = route.slice(1).map((q, i) => ({ from: route[i], to: q, length: d3.geoDistance(route[i], q), at: d3.geoInterpolate(route[i], q) }));
    const total = legs.reduce((a, leg) => a + leg.length, 0) || 1;
    const along = (f) => { let rest = f * total; for (const leg of legs) { if (rest <= leg.length || leg === legs[legs.length - 1]) return leg.at(Math.min(1, rest / (leg.length || 1))); rest -= leg.length; } };
    const samples = (f) => { const n = Math.max(2, Math.ceil(240 * f)); return [...Array(n + 1)].map((_, i) => along(f * i / n)); };
    const globe = (p.projection || 'globe') === 'globe';
    const font = family(p.font || 'hand');
    const mark = p.label_size || Math.max(20, Math.min(W, H) * 0.045);   // stop dots and the marker keep their 2.0 size
    const size = p.label_size || Math.max(20, Math.min(W, H) * 0.045, W * 0.032);   // labels: 36 px in a 1126 px box, readable on a phone
    g.font = `${size}px ${font}`;
    const line = samples(1), shape = { type: 'MultiPoint', coordinates: line.concat((p.stops || []).map((stop) => stop.at)) };
    const whole = Math.min(W, H) * 0.44;
    let zoom = 1, fitted = null;
    if (globe && p.zoom !== false) {
      // the globe turned to face the route, scaled so the route and its stops fill the box with room around them for labels
      const widest = Math.max(0, ...(p.stops || []).map((stop) => (stop.label ? g.measureText(stop.label).width : 0)));
      const mx = Math.min(W * 0.3, Math.max(W * 0.14, widest * 0.6 + size)), my = Math.min(H * 0.3, Math.max(H * 0.16, size * 1.9));
      const [[x0, y0], [x1, y1]] = d3.geoBounds(shape);
      fitted = d3.geoOrthographic().clipAngle(90).rotate([-(x0 + (x1 < x0 ? x1 + 360 : x1)) / 2, -(y0 + y1) / 2])
        .fitExtent([[mx, my], [W - mx, H - my]], shape);
      if (!Number.isFinite(fitted.scale())) fitted.scale(whole * (p.max_zoom || 40)).translate([W / 2, H / 2]);   // a route of one point
      const auto = fitted.scale() / whole, [[a0, b0], [a1, b1]] = d3.geoPath(fitted).bounds(shape);
      const share = Math.max(a1 - a0, b1 - b0) / auto / Math.min(W, H);   // of the box's short side, on the whole globe
      zoom = typeof p.zoom === 'number' ? Math.max(1, p.zoom) : share < 0.5 ? Math.max(1, Math.min(auto, p.max_zoom || 40)) : 1;
    }
    const zoomed = zoom > 1.05;
    const projection = globe ? d3.geoOrthographic().clipAngle(90).scale(whole).translate([W / 2, H / 2])
                             : d3.geoNaturalEarth1().fitExtent([[W * 0.08, H * 0.08], [W * 0.92, H * 0.92]], p.fit === 'world' ? { type: 'Sphere' } : shape);
    const path = d3.geoPath(projection, g);
    const start = p.delay || 0, intro = p.intro_ms ?? (zoomed ? 1000 : 600), travel = p.travel_ms || 2600;
    const colors = { land: p.land || '#E4D9C2', coast: p.coast || '#5C5346', sea: p.sea || '#EEF3F2', route: p.color || '#E9784E', ink: ink(p),
                     halo: typeof p.label_halo === 'string' ? p.label_halo : 'rgba(250, 247, 240, 0.9)' };
    const inset = 5, frame = () => { g.beginPath(); g.roundRect(inset, inset, W - 2 * inset, H - 2 * inset, Math.min(W, H) * 0.04); };
    // the camera k of the way (0..1, in log scale) from the whole globe to the zoomed view
    const aim = (target, k) => {
      const s = whole * Math.pow(zoom, k), r = s / fitted.scale(), [tx, ty] = fitted.translate();
      return target.rotate(fitted.rotate()).scale(s).translate([W / 2 - (W / 2 - tx) * r, H / 2 - (H / 2 - ty) * r]);
    };
    // zoomed in, the graticule also crosses the land (inland there may be nothing else), with finer lines as it closes in
    const span = Math.min(W, H) / (whole * zoom) * 180 / Math.PI, step = [0.5, 1, 2, 5].find((s) => span / s <= 6);
    const fine = zoomed && step ? (() => {
      const [[x0, y0], [x1, y1]] = d3.geoBounds(shape), m = 3 * span;
      return d3.geoGraticule().extent([[x0 - m, Math.max(-89, y0 - m)], [(x1 < x0 ? x1 + 360 : x1) + m, Math.min(89, y1 + m)]]).step([step, step]).precision(step / 4)();
    })() : null;
    // Deep in (3x and more), a small whole globe in the emptiest corner shows where the view is, its patch outlined in
    // the route colour; it fades in as the camera arrives. locator: false leaves it out.
    let locator = null;
    if (zoomed && zoom >= 3 && p.locator !== false) {
      const view = aim(d3.geoOrthographic().clipAngle(90), 1), R = Math.max(36, Math.min(W, H) * 0.12), m = inset + 12 + R;
      const marks = shape.coordinates.map((at) => view(at));
      const clear = (c) => Math.min(...marks.map(([x, y]) => Math.hypot(x - c[0], y - c[1])));
      const c = [[W - m, H - m], [m, H - m], [W - m, m], [m, m]].reduce((a, b) => (clear(b) > clear(a) + 1 ? b : a));
      if (clear(c) > R + size * 2) {
        const pin = d3.geoOrthographic().clipAngle(90).rotate(fitted.rotate()).scale(R).translate(c);
        const patch = [[inset, inset], [W - inset, inset], [W - inset, H - inset], [inset, H - inset], [inset, inset]].map((xy) => view.invert(xy));
        const corners = patch.every((at) => at && at.every(Number.isFinite)) ? patch.map((at) => pin(at)) : [];
        const across = corners.length ? Math.hypot(corners[0][0] - corners[2][0], corners[0][1] - corners[2][1]) : 0;
        locator = { x: c[0], y: c[1], R, pin, patch: corners.length ? patch : null, ring: across < 16 ? pin(view.invert([W / 2, H / 2])) : null };
      }
    }
    // Labels never pile up. In route order, each stop's label takes the cheapest free spot around its dot: above, below,
    // the sides, the diagonals, then the same further out with a thin leader line back to the dot. A spot never leaves
    // the box and costs heavily if it covers another label, any stop's dot or the marker (where it is and 250 ms on);
    // covering the route costs extra, so does sitting nearer another stop's dot than its own, and leaving last frame's
    // spot a little, so labels do not hop as the globe turns.
    const halo = p.label_halo === false ? 0 : Math.max(2, size * 0.1);
    const dotR = mark * 0.32 * 1.4 + 2, markerR = mark * 0.42 + 4, lane = (p.route_width || 5) / 2 + 3, edge = (zoomed ? inset : 0) + halo + 3;
    const AROUND = [[0, -1, 0], [0, 1, 1], [1, 0, 1.5], [-1, 0, 1.5], [1, -1, 2], [-1, -1, 2], [1, 1, 2.5], [-1, 1, 2.5]];   // dx, dy, rank
    const stops = (p.stops || []).map((stop, index) => {
      let f = 0, bestD = Infinity;
      for (let i = 0; i <= 200; i++) { const d = d3.geoDistance(along(i / 200), stop.at); if (d < bestD) { bestD = d; f = i / 200; } }
      let t = 1; for (let i = 0; i <= 400; i++) if (ease(i / 400) >= f - 1e-9) { t = i / 400; break; }  // when the marker gets there
      const m = stop.label ? g.measureText(stop.label) : null;
      const box = m && { left: m.actualBoundingBoxLeft, right: m.actualBoundingBoxRight, ascent: m.actualBoundingBoxAscent, descent: m.actualBoundingBoxDescent };
      if (box) { box.w = box.left + box.right + 2 * halo; box.h = box.ascent + box.descent + 2 * halo; }
      return { ...stop, index, f, t, box, spot: null, target: null, u: null, v: null, x: 0, y: 0 };
    });
    const gap0 = dotR + size * 0.12;   // from a dot to its nearest label spot
    const meets = (a, b, gap) => a[0] < b[2] + gap && b[0] < a[2] + gap && a[1] < b[3] + gap && b[1] < a[3] + gap;
    const touches = (b, x, y, r) => Math.hypot(Math.max(b[0] - x, 0, x - b[2]), Math.max(b[1] - y, 0, y - b[3])) < r;
    const place = (dots, marker, trail) => {
      const taken = [];
      for (const dot of [...dots].sort((a, b) => a.stop.f - b.stop.f || a.stop.index - b.stop.index)) {
        const L = dot.stop.box;
        if (!L) continue;
        let best = null;
        for (let ring = 0; ring < 6; ring++) {
          const gap = gap0 + ring * size * 0.8;
          AROUND.forEach(([dx, dy, rank], k) => {
            const cx = dot.x + dx * (gap + L.w / 2), cy = dot.y + dy * (gap + L.h / 2);
            const box = [cx - L.w / 2, cy - L.h / 2, cx + L.w / 2, cy + L.h / 2];
            if (box[0] < edge || box[1] < edge || box[2] > W - edge || box[3] > H - edge) return;
            const covers = taken.filter((other) => meets(box, other, size * 0.15)).length + dots.filter((other) => touches(box, other.x, other.y, dotR)).length
              + (marker.some(([x, y]) => touches(box, x, y, markerR)) ? 1 : 0) + (locator && touches(box, locator.x, locator.y, locator.R + 8) ? 1 : 0);
            const crossed = trail.filter(([x, y]) => x > box[0] - lane && x < box[2] + lane && y > box[1] - lane && y < box[3] + lane).length;
            const near = (other) => Math.hypot(Math.max(box[0] - other.x, 0, other.x - box[2]), Math.max(box[1] - other.y, 0, other.y - box[3]));
            const misleads = dots.some((other) => other !== dot && near(other) < near(dot) + 2);   // reads as another stop's label
            const cost = 1000 * covers + 10 * ring + rank + (crossed ? 12 + Math.min(20, crossed * 0.25) : 0) + (misleads ? 15 : 0)
              + (dot.stop.spot === ring * 8 + k ? 0 : 8);
            if (!best || cost < best.cost) best = { cost, box, spot: ring * 8 + k };
          });
        }
        if (!best) {   // a box too small for any spot: above the dot, pushed inside
          const cx = Math.max(L.w / 2, Math.min(W - L.w / 2, dot.x)), cy = Math.max(L.h / 2, dot.y - dotR - L.h / 2);
          best = { box: [cx - L.w / 2, cy - L.h / 2, cx + L.w / 2, cy + L.h / 2], spot: -1 };
        }
        dot.stop.spot = best.spot;
        dot.stop.target = best.box;
        taken.push(best.box);
      }
    };
    root.__map = { zoom, size, dot: dotR, marker: markerR, locator: locator && [locator.x, locator.y, locator.R] };   // read by the tests
    let last = null;
    const draw = () => {
      const now = performance.now() - start, dt = last === null ? 0 : Math.max(0, now - last);
      last = now;
      const raw = (now - intro) / travel, f = ease(Math.max(0, Math.min(1, raw)));
      const shown = Math.max(0, Math.min(1, now / Math.max(1, zoomed ? intro * 0.3 : intro)));
      const close = zoomed ? ease((now / Math.max(1, intro) - 0.15) / 0.85) : 0;   // how far the camera has flown in
      const marker = along(f);
      if (zoomed) aim(projection, close);
      else if (globe) {
        const lead = along(Math.min(1, f + 0.08)), lon = marker[0] + ((((lead[0] - marker[0]) % 360) + 540) % 360 - 180) * 0.4;
        projection.rotate([-lon, -Math.max(-50, Math.min(50, marker[1])) * 0.55]);   // lead unwrapped across the antimeridian
      }
      const facing = [-projection.rotate()[0], -projection.rotate()[1]];
      const seen = (at) => !globe || d3.geoDistance(at, facing) < Math.PI / 2;
      g.clearRect(0, 0, W, H);
      g.save();
      if (zoomed) { frame(); g.clip(); }
      g.globalAlpha = shown;
      if (globe) { g.beginPath(); path({ type: 'Sphere' }); g.fillStyle = colors.sea; g.fill(); g.lineWidth = 3; g.strokeStyle = colors.coast; g.stroke();
        g.beginPath(); path(d3.geoGraticule10()); g.lineWidth = 0.8; g.strokeStyle = 'rgba(92,83,70,0.18)'; g.stroke(); }
      g.beginPath(); path(land); g.fillStyle = colors.land; g.fill();
      // A sketchy coast: two passes a hair apart.
      g.lineJoin = 'round'; g.strokeStyle = colors.coast; g.lineWidth = 1.6; g.stroke();
      g.save(); g.translate(0.9, -0.6); g.globalAlpha = shown * 0.45; g.beginPath(); path(land); g.stroke(); g.restore();
      if (zoomed && close > 0) {
        g.globalAlpha = shown * close; g.lineWidth = 0.8;
        g.beginPath(); path(d3.geoGraticule10()); g.strokeStyle = 'rgba(92,83,70,0.16)'; g.stroke();
        if (fine) { g.beginPath(); path(fine); g.strokeStyle = 'rgba(92,83,70,0.11)'; g.stroke(); }
      }
      g.restore();
      // zoomed in, the box's rounded edge takes over from the sphere's rim wherever the sphere is cut
      if (zoomed) { g.save(); g.globalAlpha = shown; g.beginPath(); path({ type: 'Sphere' }); g.clip(); frame(); g.lineWidth = 3; g.strokeStyle = colors.coast; g.stroke(); g.restore(); }
      if (locator && close > 0.6) {
        const pin = d3.geoPath(locator.pin, g);
        g.save(); g.globalAlpha = shown * Math.min(1, (close - 0.6) / 0.4);
        g.beginPath(); pin({ type: 'Sphere' }); g.fillStyle = colors.sea; g.fill(); g.lineWidth = 7; g.strokeStyle = colors.halo; g.stroke();
        g.beginPath(); pin(land); g.fillStyle = colors.land; g.fill(); g.lineWidth = 1; g.strokeStyle = colors.coast; g.stroke();
        g.beginPath(); pin({ type: 'Sphere' }); g.lineWidth = 2.5; g.strokeStyle = colors.coast; g.stroke();
        g.lineWidth = 2.5; g.strokeStyle = colors.route; g.lineJoin = 'round';
        if (locator.patch) { g.beginPath(); pin({ type: 'LineString', coordinates: locator.patch }); g.stroke(); }
        if (locator.ring) { g.beginPath(); g.arc(locator.ring[0], locator.ring[1], 9, 0, 2 * Math.PI); g.stroke(); }
        g.restore();
      }
      g.globalAlpha = 1;
      const drawn = [];
      if (now > intro) {
        g.beginPath(); path({ type: 'LineString', coordinates: samples(f) });
        g.strokeStyle = colors.route; g.lineWidth = p.route_width || 5; g.lineCap = 'round'; g.setLineDash([]); g.stroke();
        const dots = stops.filter((stop) => seen(stop.at)).map((stop) => { const [x, y] = projection(stop.at); return { stop, x, y }; });
        const xy = projection(marker);
        const ahead = [0.5, 1].map((k) => projection(along(ease(Math.max(0, Math.min(1, raw + k * 250 / travel))))));
        place(dots, [xy, ...ahead].filter(Boolean), line.filter(seen).map((at) => projection(at)));
        const glide = 1 - Math.exp(-dt / 90);   // a label that has to move slides there
        for (const { stop, x, y } of dots) {
          if (stop.f > f + 1e-6) continue;
          const pop = Math.max(0.0001, Math.min(1, (raw - stop.t) * travel / 260)), r = mark * 0.32 * (1 + 0.4 * Math.sin(Math.PI * pop));
          g.beginPath(); g.arc(x, y, r, 0, 2 * Math.PI); g.fillStyle = '#FFFFFF'; g.fill(); g.lineWidth = 3; g.strokeStyle = colors.ink; g.stroke();
          if (!stop.box) continue;
          // The offset from the dot glides, so a label rides with its dot as the globe turns. It is measured in units of
          // the label's half size plus the nearest gap (every spot is at least 1 out in x or y there) and swings round the
          // dot by angle and that max-norm radius, so a label changing sides never passes over its own dot.
          const nx = stop.box.w / 2 + gap0, ny = stop.box.h / 2 + gap0;
          const u = ((stop.target[0] + stop.target[2]) / 2 - x) / nx, v = ((stop.target[1] + stop.target[3]) / 2 - y) / ny;
          if (stop.u === null) { stop.u = u; stop.v = v; } else {
            let a = Math.atan2(stop.v, stop.u), r = Math.max(Math.abs(stop.u), Math.abs(stop.v));
            a += (((Math.atan2(v, u) - a) % (2 * Math.PI) + 3 * Math.PI) % (2 * Math.PI) - Math.PI) * glide;
            r += (Math.max(Math.abs(u), Math.abs(v)) - r) * glide;
            const c = Math.cos(a), s = Math.sin(a), m = Math.max(Math.abs(c), Math.abs(s));
            [stop.u, stop.v] = [c / m * r, s / m * r];
          }
          stop.x = stop.u * nx; stop.y = stop.v * ny;
          drawn.push({ stop, x, y, pop, home: [u * nx, v * ny] });
        }
        // a slide that would cover another label, a dot or the marker jumps to its spot instead (the spots are clear)
        const boxOf = ({ stop, x, y }) => [x + stop.x - stop.box.w / 2, y + stop.y - stop.box.h / 2, x + stop.x + stop.box.w / 2, y + stop.y + stop.box.h / 2];
        for (let pass = 0; pass <= drawn.length; pass++) {
          const blocked = drawn.filter((item) => {
            const box = boxOf(item);
            return (item.stop.x !== item.home[0] || item.stop.y !== item.home[1])
              && (drawn.some((other) => other !== item && meets(box, boxOf(other), 0)) || dots.some((dot) => touches(box, dot.x, dot.y, dotR))
                  || (xy && touches(box, xy[0], xy[1], markerR)));
          });
          if (!blocked.length) break;
          blocked.forEach((item) => { [item.stop.x, item.stop.y] = item.home; item.stop.u = item.home[0] / (item.stop.box.w / 2 + gap0); item.stop.v = item.home[1] / (item.stop.box.h / 2 + gap0); });
        }
        if (xy) {
          const pulse = (now % 900) / 900;
          g.beginPath(); g.arc(xy[0], xy[1], mark * (0.45 + 0.8 * pulse), 0, 2 * Math.PI); g.strokeStyle = colors.route; g.globalAlpha = 1 - pulse; g.lineWidth = 3; g.stroke(); g.globalAlpha = 1;
          g.beginPath(); g.arc(xy[0], xy[1], mark * 0.42, 0, 2 * Math.PI); g.fillStyle = colors.route; g.fill(); g.lineWidth = 3; g.strokeStyle = '#FFFFFF'; g.stroke();
        }
        root.__mapMarker = xy && [xy[0], xy[1], markerR];
      }
      for (const stop of stops) if (!drawn.some((item) => item.stop === stop)) stop.u = stop.v = null;
      // leaders and labels over everything, each on a paper-coloured halo
      g.font = `${size}px ${font}`; g.textAlign = 'center'; g.textBaseline = 'alphabetic';
      root.__mapLabels = drawn.map(({ stop, x, y, pop }) => {
        const L = stop.box, lx = x + stop.x, ly = y + stop.y, box = [lx - L.w / 2, ly - L.h / 2, lx + L.w / 2, ly + L.h / 2];
        const nx = Math.max(box[0], Math.min(box[2], x)), ny = Math.max(box[1], Math.min(box[3], y)), d = Math.hypot(nx - x, ny - y);
        const leader = d > dotR + size * 0.45;
        if (leader) {   // moved away from its dot: a thin line back to it
          const from = mark * 0.32 + 2, to = d - 3;
          g.beginPath(); g.moveTo(x + (nx - x) * from / d, y + (ny - y) * from / d); g.lineTo(x + (nx - x) * to / d, y + (ny - y) * to / d);
          g.globalAlpha = 0.75 * pop; g.lineWidth = Math.max(1.5, size * 0.045); g.lineCap = 'round'; g.strokeStyle = colors.ink; g.stroke();
        }
        g.globalAlpha = pop;
        const tx = lx - (L.right - L.left) / 2, ty = ly + (L.ascent - L.descent) / 2;
        if (halo) { g.lineJoin = 'round'; g.lineWidth = 2 * halo; g.strokeStyle = colors.halo; g.strokeText(stop.label, tx, ty); }
        g.fillStyle = colors.ink; g.fillText(stop.label, tx, ty);
        g.globalAlpha = 1;
        return { label: stop.label, box, dot: [x, y], leader, pop };
      });
      requestAnimationFrame(draw);
    };
    requestAnimationFrame(draw);
  };

  // ---------------------------------------------------------------- 2.0.1 recipes

  // Props under real physics (matter-js): coins pour into a jar, dominoes topple, a tower is knocked over, balls bounce
  // and settle. Each body keeps one rough.js outline (a fixed seed) and turns rigidly with the body instead of boiling.
  // p: {scene: pour|topple|stack|bounce, container: jar|tray|floor|none, items: [{shape: circle|box|pill, size | w,h,
  //     fill, label, count}], gravity (1 = 1000 px/s^2), bounce (0..1), drop_ms, intro_ms, delay, ink, label_size}
  Recipes.physics = (root, p, ctx) => {
    if (!window.Matter) throw new Error('physics needs matter-js (vendor/matter.js)');
    const { Engine, Bodies, Body, Composite } = Matter;
    const W = ctx.width, H = ctx.height, scale = ctx.scale, M = 8;  // bodies stay M px inside the box
    const canvas = document.createElement('canvas');
    canvas.width = W * scale; canvas.height = H * scale;
    Object.assign(canvas.style, { position: 'absolute', inset: '0', width: `${W}px`, height: `${H}px` });
    root.appendChild(canvas);
    const g = canvas.getContext('2d'); g.scale(scale, scale);
    const rc = rough.canvas(canvas), gen = rc.generator;
    // Toppling reads best faster than the free fall of a small prop (2.5x), so dominoes run as a chain, not a slow lean.
    const engine = Engine.create({ gravity: { x: 0, y: p.gravity ?? (p.scene === 'topple' ? 2.5 : 1), scale: 0.001 } });
    engine.positionIterations = 10; engine.velocityIterations = 8;
    const scene = p.scene || 'pour', kind = p.container || (scene === 'pour' ? 'jar' : 'floor');
    const start = p.delay || 0, intro = p.intro_ms ?? (kind === 'none' ? 0 : 500), lineInk = ink(p);
    const wall = (x, y, w, h, angle = 0) => Bodies.rectangle(x, y, w, h, { isStatic: true, angle, friction: 0.6 });
    // Invisible bounds a little inside the box, so nothing settles against (and gets cut by) the overlay edge.
    const statics = [wall(W / 2, H - M + 30, W * 2, 60), wall(M - 30, H / 2, 60, H * 3), wall(W - M + 30, H / 2, 60, H * 3)];
    let outlinePath = null, floorY = H - M;
    if (kind === 'jar' || kind === 'tray') {
      const jw = Math.min(W - 2 * M - 8, kind === 'jar' ? Math.max(W * 0.55, 160) : W * 0.8), jh = kind === 'jar' ? Math.min(H * 0.62, jw * 1.25) : Math.min(H * 0.22, 90);
      const left = (W - jw) / 2, right = left + jw, bottom = H - M - 4, top = bottom - jh, t = 16, r = Math.min(26, jw * 0.12);
      statics.push(wall(left - t / 2 + 2, bottom - jh / 2, t, jh), wall(right + t / 2 - 2, bottom - jh / 2, t, jh), wall(W / 2, bottom + t / 2 - 2, jw, t));
      outlinePath = `M ${left - 6} ${top} Q ${left} ${top + 6} ${left} ${top + 18} L ${left} ${bottom - r} Q ${left} ${bottom} ${left + r} ${bottom} L ${right - r} ${bottom} Q ${right} ${bottom} ${right} ${bottom - r} L ${right} ${top + 18} Q ${right} ${top + 6} ${right + 6} ${top}`;
      floorY = bottom;
      statics.opening = [left + 10, right - 10];
    } else if (kind === 'floor') {
      outlinePath = `M ${M + 4} ${H - M - 2} L ${W - M - 4} ${H - M - 2}`;
      floorY = H - M - 4;
      statics[0] = wall(W / 2, floorY + 30, W * 2, 60);
    }
    Composite.add(engine.world, statics);
    const outline = outlinePath && gen.path(outlinePath, { stroke: lineInk, strokeWidth: 3.2, roughness: 1.1, seed: 7 });
    const items = [];
    const specs = (p.items && p.items.length ? p.items : [{ shape: 'circle', fill: '#F2C14E', label: '¥', count: 14 }]);
    for (const spec of specs) for (let i = 0; i < (spec.count || 1); i++) items.push({ ...spec });
    const size = (spec) => spec.size || Math.max(22, Math.min(W, H) * 0.085);
    const FILLS = PALETTE.filter((color) => color !== '#1E2733');  // no ink-dark bodies: they read as holes
    const make = (spec, x, y, angle = 0) => {
      const options = { restitution: spec.bounce ?? p.bounce ?? (scene === 'bounce' ? 0.62 : 0.2), friction: spec.friction ?? 0.4, frictionAir: 0.004,
                        density: spec.density || 0.002, angle };
      const s = size(spec), w = spec.w || s, h = spec.h || s;
      if ((spec.shape || 'circle') === 'circle') return Bodies.circle(x, y, s / 2, options);
      if (spec.shape === 'pill') return Bodies.rectangle(x, y, w * 1.6, h * 0.7, { ...options, chamfer: { radius: h * 0.34 } });
      return Bodies.rectangle(x, y, w, h, { ...options, chamfer: { radius: Math.min(w, h) * 0.08 } });
    };
    const drawable = (spec, index) => {
      const s = size(spec), w = spec.w || s, h = spec.h || s, fill = spec.fill || FILLS[index % FILLS.length];
      const style = { stroke: lineInk, strokeWidth: 2.4, roughness: 0.9, fill, fillStyle: 'solid', seed: 11 + index };
      if ((spec.shape || 'circle') === 'circle') return gen.circle(0, 0, s, style);
      if (spec.shape === 'pill') return gen.path(`M ${-w * 0.8 + h * 0.35} ${-h * 0.35} H ${w * 0.8 - h * 0.35} A ${h * 0.35} ${h * 0.35} 0 0 1 ${w * 0.8 - h * 0.35} ${h * 0.35} H ${-w * 0.8 + h * 0.35} A ${h * 0.35} ${h * 0.35} 0 0 1 ${-w * 0.8 + h * 0.35} ${-h * 0.35} Z`, style);
      return gen.rectangle(-w / 2, -h / 2, w, h, style);
    };
    const bodies = [];  // {body, shape, spec, at}
    const release = (spec, index, at, x, y, angle = 0, kick = null) => bodies.push({ spec, shape: drawable(spec, index), at, x, y, angle, kick, body: null });
    const dropMs = p.drop_ms ?? 1400;
    const [lo, hi] = statics.opening || [M + 20, W - M - 20];
    if (scene === 'topple' || scene === 'stack') {
      const s = size(items[0]);
      if (scene === 'topple') {
        // Dominoes a little over half their height apart: each one falls far enough to tip the next.
        const n = Math.max(3, items.length), h = Math.min(s * 2.4, (floorY - M) * 0.8), w = Math.max(8, h * 0.17), gap = Math.min(h * 0.58, (W - 2 * M - 2 * w) / n);
        const x0 = W / 2 - gap * (n - 1) / 2;
        for (let i = 0; i < n; i++) release({ ...items[i % items.length], shape: 'box', w, h, friction: 0.08 }, i, 0, x0 + i * gap, floorY - h / 2 - 1, 0, i === 0 ? { angular: 0.05 } : null);
      } else {
        // A two-wide tower and a heavy ball that rolls in and knocks it over.
        const rows = Math.max(3, Math.ceil(items.length / 2)), w = s * 1.1, h = Math.min(s * 0.8, (floorY - M - 20) / rows);
        let index = 0;
        for (let r = 0; r < rows; r++) for (let c = 0; c < 2; c++) release({ ...items[index % items.length], shape: 'box', w, h }, index++, 0,
          W * 0.62 + (c - 0.5) * (w + 1) + (r % 2 ? 3 : -3), floorY - h / 2 - r * (h + 0.5) - 1);
        release({ shape: 'circle', size: s * 1.2, fill: '#E9784E', density: 0.02, bounce: 0.1 }, index, intro + 150, M + s * 0.7, floorY - rows * h * 0.62, 0, { vx: 22 });
      }
    } else {
      items.forEach((spec, i) => {
        const x = lo + (hi - lo) * (scene === 'bounce' ? (i + 0.5) / items.length : (0.5 + 0.42 * Math.sin(i * 2.399)));
        release(spec, i, intro + dropMs * i / Math.max(1, items.length), x, -size(spec) - 4 - (scene === 'bounce' ? i * 18 : 0), Math.random() * 6.28);
      });
    }
    const step = 1000 / 120;
    let sim = 0;
    const draw = () => {
      const now = performance.now() - start;
      for (; sim + step <= now; sim += step) {
        for (const entry of bodies) {
          if (entry.body || sim < entry.at) continue;
          entry.body = make(entry.spec, entry.x, entry.y, entry.angle);
          Composite.add(engine.world, entry.body);
          if (entry.kick && entry.kick.angular) Body.setAngularVelocity(entry.body, entry.kick.angular);
          if (entry.kick && entry.kick.vx) Body.setVelocity(entry.body, { x: entry.kick.vx, y: -2 });
        }
        if (sim >= Math.min(...bodies.map((entry) => entry.at), intro)) Engine.update(engine, step);
      }
      g.clearRect(0, 0, W, H);
      if (outline) {
        // The container is drawn on first: its path revealed with a moving clip from left to right.
        const f = intro ? ease(Math.min(1, now / intro)) : 1;
        g.save(); g.beginPath(); g.rect(0, 0, W * f, H); g.clip(); rc.draw(outline); g.restore();
      }
      const font = family(p.font || 'hand');
      for (const entry of bodies) {
        const shown = entry.body || (scene !== 'pour' && scene !== 'bounce');
        if (!shown) continue;
        const x = entry.body ? entry.body.position.x : entry.x, y = entry.body ? entry.body.position.y : entry.y, a = entry.body ? entry.body.angle : entry.angle;
        g.save(); g.translate(x, y); g.rotate(a);
        if (!entry.body && intro) g.globalAlpha = ease(Math.min(1, now / intro));
        rc.draw(entry.shape);
        if (entry.spec.label) {
          const s = size(entry.spec);
          g.fillStyle = lineInk; g.font = `${Math.round(p.label_size || s * 0.55)}px ${font}`; g.textAlign = 'center'; g.textBaseline = 'middle';
          g.fillText(entry.spec.label, 0, s * 0.03);
        }
        g.restore();
      }
      requestAnimationFrame(draw);
    };
    requestAnimationFrame(draw);
  };

  // A 3D object turning in place (Zdog): flat-shaded faces inside an ink silhouette, popping in with a small overshoot.
  // p: {shape: box|coin|cylinder|cone|sphere|pyramid|gem, size, colors: [main, side, top], turns (1), spin_ms, tilt (deg),
  //     axis: y|x, appear_ms, delay, ink, outline (px)}
  Recipes.spin3d = (root, p, ctx) => {
    if (!window.Zdog) throw new Error('spin3d needs Zdog (vendor/zdog.js)');
    const W = ctx.width, H = ctx.height, TAU = Zdog.TAU;
    const shape = p.shape || 'box', s = p.size || Math.min(W, H) * 0.5, line = p.outline ?? 4, inkColor = ink(p);
    const colors = p.colors || [PALETTE[0], '#C8603A', '#F4A47F'];
    const [main, side, top] = [colors[0], colors[1] || colors[0], colors[2] || colors[0]];
    // Two canvases, one model each: an ink silhouette below (every surface in ink, strokes thickened by the outline) and
    // the colours above (Zdog clears its canvas on every render, so the passes cannot share one).
    const build = (silhouette) => {
      // Sized in CSS pixels: Zdog itself up-rezes by devicePixelRatio (the capture's board scale), so no extra zoom here.
      const canvas = document.createElement('canvas');
      canvas.width = W; canvas.height = H;
      Object.assign(canvas.style, { position: 'absolute', inset: '0', width: `${W}px`, height: `${H}px` });
      root.appendChild(canvas);
      const pen = (color) => (silhouette ? inkColor : color), grow = silhouette ? 2 * line : 0;
      const scene = new Zdog.Illustration({ element: canvas, resize: false, centered: true });
      const body = new Zdog.Anchor({ addTo: scene });
      const face = (points, color, front) => new Zdog.Shape({ addTo: body, path: points, fill: true, stroke: 2 + grow, color: pen(color), front, backface: silhouette ? pen(color) : false });
      if (shape === 'box') {
        new Zdog.Box({ addTo: body, width: s * 0.78, height: s * 0.78, depth: s * 0.78, stroke: 2 + grow, color: pen(main),
          leftFace: pen(side), rightFace: pen(side), topFace: pen(top), bottomFace: pen(side), frontFace: pen(main), rearFace: pen(main) });
      } else if (shape === 'coin' || shape === 'cylinder') {
        const coin = shape === 'coin', d = coin ? s * 0.92 : s * 0.62, length = coin ? s * 0.12 : s * 0.85;
        new Zdog.Cylinder({ addTo: body, diameter: d, length, stroke: 2 + grow, color: pen(side), frontFace: pen(main), backface: pen(main),
          rotate: coin ? {} : { x: TAU / 4 } });
        if (coin && !silhouette) for (const z of [1, -1]) new Zdog.Ellipse({ addTo: body, diameter: d * 0.7, stroke: Math.max(2, s * 0.025), color: top, translate: { z: z * (length / 2 + 0.5) } });
        // A can's label stripe shows that it turns (a plain cylinder spinning on its axis looks still).
        if (!coin && !silhouette) new Zdog.Rect({ addTo: body, width: d * 0.28, height: length * 0.86, stroke: 1, fill: true, color: top, backface: false,
                                                   translate: { z: d / 2 + 1 } });
      } else if (shape === 'cone') {
        new Zdog.Cone({ addTo: body, diameter: s * 0.8, length: s * 0.9, stroke: 2 + grow, color: pen(main), backface: pen(side), rotate: { x: TAU / 4 }, translate: { y: s * 0.38 } });
        if (!silhouette) new Zdog.Shape({ addTo: body, path: [{ y: -s * 0.5 }, { y: s * 0.38, z: s * 0.4 }], stroke: Math.max(4, s * 0.05), color: top });  // a seam that turns
      } else if (shape === 'sphere') {
        const half = (color, turn) => new Zdog.Hemisphere({ addTo: body, diameter: s * 0.9, stroke: 2 + grow, color: pen(color), backface: pen(color), rotate: { y: turn } });
        half(main, 0); half(top, TAU / 2);
      } else if (shape === 'pyramid' || shape === 'gem') {
        const r = s * 0.48, apex = { x: 0, y: -s * 0.5, z: 0 }, tip = shape === 'gem' ? { x: 0, y: s * 0.62, z: 0 } : null;
        const base = [0, 1, 2, 3].map((i) => ({ x: r * Math.cos(TAU * i / 4 + TAU / 8), y: s * 0.3, z: r * Math.sin(TAU * i / 4 + TAU / 8) }));
        const corners = [apex, ...base, ...(tip ? [tip] : [])];
        const center = ['x', 'y', 'z'].reduce((c, k) => ({ ...c, [k]: corners.reduce((sum, q) => sum + q[k], 0) / corners.length }), {});
        // Zdog hides a face when its "front" point (relative to the face origin) turns away: give it the true outward normal.
        const normal = (points) => {
          const [a, b, c] = points, u = { x: b.x - a.x, y: b.y - a.y, z: b.z - a.z }, v = { x: c.x - a.x, y: c.y - a.y, z: c.z - a.z };
          const n = { x: u.y * v.z - u.z * v.y, y: u.z * v.x - u.x * v.z, z: u.x * v.y - u.y * v.x };
          const out = ['x', 'y', 'z'].reduce((sum, k) => sum + n[k] * (points.reduce((t, q) => t + q[k], 0) / points.length - center[k]), 0);
          return out < 0 ? { x: -n.x, y: -n.y, z: -n.z } : n;
        };
        const palette = [main, side, top, side];
        base.forEach((a, i) => {
          const b = base[(i + 1) % 4];
          face([apex, a, b], palette[i], normal([apex, a, b]));
          if (tip) face([tip, b, a], palette[(i + 1) % 4], normal([tip, b, a]));
        });
        if (!tip) face(base, side, normal(base));
      } else throw new Error('spin3d shape is box, coin, cylinder, cone, sphere, pyramid or gem');
      return { scene, body };
    };
    const passes = [build(true), build(false)];
    const start = p.delay || 0, appear = p.appear_ms ?? 420, spin = p.spin_ms || Math.max(600, ctx.duration - start - appear - 200);
    const tilt = (p.tilt ?? 18) * TAU / 360, turns = p.turns ?? 1, axis = p.axis || 'y';
    const draw = () => {
      const now = performance.now() - start;
      const grow = now <= 0 ? 0 : now >= appear ? 1 : 1 - Math.cos(now / appear * Math.PI * 1.25) * Math.exp(-4 * now / appear);  // overshoot, then settle
      const turn = TAU * turns * ease(Math.max(0, Math.min(1, (now - appear) / spin))) + TAU / 10;
      passes.forEach(({ scene, body }) => {
        body.scale = Math.max(0.0001, grow);
        body.rotate = axis === 'x' ? { x: turn, y: TAU / 10 } : { x: -tilt, y: turn };
        scene.updateRenderGraph();
      });
      requestAnimationFrame(draw);
    };
    requestAnimationFrame(draw);
  };

  // A formula written by hand (MathJax TeX -> SVG glyph outlines): glyph by glyph in reading order, each outline traced
  // like a pen, then filled; fraction bars and root lines are ruled left to right. A highlight circles or underlines a
  // part afterwards. p: {tex, size (px per em; default fits the box), color, glyph_ms, delay, align: center|left,
  //     highlight: {from, to (glyph indexes), style: circle|box|underline, color, at_ms}}
  Recipes.formula = (root, p, ctx) => {
    if (!window.MathJax || !MathJax.tex2svg) throw new Error('formula needs MathJax (vendor/mathjax-tex-svg.js)');
    if (!p.tex) throw new Error('formula needs "tex"');
    const W = ctx.width, H = ctx.height;
    const holder = MathJax.tex2svg(p.tex, { display: true });
    const svgNode = holder.querySelector('svg');
    if (!svgNode) throw new Error('MathJax produced no SVG for ' + p.tex);
    const error = svgNode.querySelector('[data-mml-node="merror"]');
    if (error) throw new Error('TeX error in formula: ' + (error.getAttribute('title') || error.textContent));
    const box = svgNode.viewBox.baseVal, pad = Math.min(W, H) * 0.06;
    const em = p.size || Math.min((W - 2 * pad) / (box.width / 1000), (H - 2 * pad) / (box.height / 1000), Math.max(40, H * 0.5));
    const width = box.width / 1000 * em, height = box.height / 1000 * em;
    svgNode.setAttribute('width', width); svgNode.setAttribute('height', height);
    Object.assign(svgNode.style, { position: 'absolute', top: `${(H - height) / 2}px`, left: `${p.align === 'left' ? pad : (W - width) / 2}px`,
                                   color: p.color || ink(p), overflow: 'visible' });
    root.appendChild(svgNode);
    // Reading order as a hand writes it: a fraction's bar before its denominator, a root's sign and bar before its contents.
    for (const frac of svgNode.querySelectorAll('[data-mml-node="mfrac"]')) {
      const bar = [...frac.children].find((child) => child.tagName === 'rect');
      if (bar && frac.children.length > 2) frac.insertBefore(bar, frac.children[1]);
    }
    for (const root2 of svgNode.querySelectorAll('[data-mml-node="msqrt"], [data-mml-node="mroot"]')) {
      for (const child of [...root2.children].filter((c) => c.tagName === 'rect' || c.getAttribute('data-mml-node') === 'mo').reverse()) root2.insertBefore(child, root2.firstChild);
    }
    const marks = [...svgNode.querySelectorAll('path, rect')].filter((node) => !(node.tagName === 'rect' && node.closest('[data-mml-node="merror"]')));
    const start = p.delay || 0, slot = p.glyph_ms || Math.max(90, Math.min(240, (ctx.duration - start - 500) / Math.max(1, marks.length)));
    const color = p.color || ink(p);
    marks.forEach((node, index) => {
      const at = start + index * slot;
      if (node.tagName === 'rect') {
        node.animate([{ clipPath: 'inset(0 100% 0 0)' }, { clipPath: 'inset(0 0% 0 0)' }], { delay: at, duration: slot * 1.2, easing: 'cubic-bezier(.45,.05,.4,1)', fill: 'both' });
        return;
      }
      const length = node.getTotalLength();
      Object.assign(node.style, { stroke: 'currentColor', strokeWidth: `${p.pen ?? 1.4}px`, vectorEffect: 'non-scaling-stroke', strokeLinejoin: 'round', strokeDasharray: `${length}` });
      node.animate([{ strokeDashoffset: length }, { strokeDashoffset: 0 }], { delay: at, duration: slot * 1.25, easing: 'cubic-bezier(.45,.05,.4,1)', fill: 'both' });
      node.animate([{ fillOpacity: 0 }, { fillOpacity: 1 }], { delay: at + slot * 0.75, duration: slot * 1.1, easing: 'ease-out', fill: 'both' });
    });
    const mark = p.highlight;
    if (mark) {
      const glyphs = marks.slice(mark.from ?? 0, (mark.to ?? marks.length - 1) + 1);
      const origin = root.getBoundingClientRect();
      const rects = glyphs.map((node) => node.getBoundingClientRect()).filter((r) => r.width || r.height);
      if (!rects.length) throw new Error('formula highlight covers no glyphs');
      const x1 = Math.min(...rects.map((r) => r.left)) - origin.left, y1 = Math.min(...rects.map((r) => r.top)) - origin.top;
      const x2 = Math.max(...rects.map((r) => r.right)) - origin.left, y2 = Math.max(...rects.map((r) => r.bottom)) - origin.top;
      const layer = svgLayer(root, W, H), rs = rough.svg(layer), m = em * 0.12, style = { stroke: mark.color || '#E9784E', strokeWidth: mark.width || 3.4, roughness: 1.3, seed: 5 };
      const kind = mark.style || 'circle';
      const node = kind === 'underline' ? rs.line(x1, y2 + m * 0.6, x2, y2 + m * 0.4, style)
        : kind === 'box' ? rs.rectangle(x1 - m, y1 - m, x2 - x1 + 2 * m, y2 - y1 + 2 * m, style)
        : rs.ellipse((x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * 1.18 + 2 * m, (y2 - y1) * 1.3 + 2 * m, style);
      layer.appendChild(node);
      drawOn(node, mark.at_ms ?? start + marks.length * slot + 250, mark.ms || 520);
    }
  };

  // Anything else: the model writes the body and a script; it must animate with anime.js, CSS or element.animate
  // (all driven by the virtual clock) and keep a transparent background.
  Recipes.custom = (root, p, ctx) => {
    if (p.css) { const style = document.createElement('style'); style.textContent = p.css; document.head.appendChild(style); }
    root.innerHTML = p.html || '';
    if (p.script) new Function('root', 'p', 'ctx', 'anime', 'rough', 'RoughNotation', 'confetti', 'd3', p.script)(root, p, ctx, anime, rough, RoughNotation, confetti, d3);
  };

  // ---------------------------------------------------------------- 3.0 recipes: forms that explain
  // Each picks the form the content has (a list's structure, a comparison, a process, who does what when) and moves
  // only when the narration gets there: times are milliseconds from the layer's start, computed from the words.
  const muted = (p) => p.muted || '#A7ABB0';
  const accentOf = (p) => p.accent || '#E9784E';
  const roughOf = (p, extra = {}) => ({ stroke: ink(p), strokeWidth: p.stroke_width || 3.5, roughness: 1.1, bowing: 1, seed: p.seed || 11, ...extra });
  const popIn = (node, delay, duration = 320) => {
    node.style.transformBox = 'fill-box'; node.style.transformOrigin = 'center';
    node.animate([{ transform: 'scale(0)', opacity: 0 }, { transform: 'scale(1)', opacity: 1 }], { delay, duration, easing: 'cubic-bezier(.3,1.5,.5,1)', fill: 'both' });
  };
  const show = (node, delay, duration = 220) => node.animate([{ opacity: 0 }, { opacity: 1 }], { delay, duration, fill: 'both' });
  const roughCheck = (pen, x, y, s, o) => pen.linearPath([[x - s * 0.5, y], [x - s * 0.1, y + s * 0.42], [x + s * 0.55, y - s * 0.45]], o);
  const roughCross = (pen, x, y, s, o) => { const g = document.createElementNS(NS, 'g');
    g.appendChild(pen.line(x - s / 2, y - s / 2, x + s / 2, y + s / 2, o)); g.appendChild(pen.line(x + s / 2, y - s / 2, x - s / 2, y + s / 2, o)); return g; };
  // A label that wraps inside a width (for table cells and flow nodes), between words only; centred unless align says
  // otherwise.
  function block(root, text, x, y, width, size, p, color, align = 'center') {
    const node = document.createElement('div');
    Object.assign(node.style, { position: 'absolute', left: `${x}px`, top: `${y}px`, width: `${width}px`, transform: 'translateY(-50%)', textAlign: align,
      fontFamily: family(p.font || 'hand'), fontSize: `${size}px`, color: color || ink(p), lineHeight: 1.15, whiteSpace: 'normal' });
    root.appendChild(node);
    return setWords(node, text, width);
  }
  // A one-line label wider than its room wraps between words instead of running out of the box.
  function wrapWide(node, width, align = 'left') {
    if (node.offsetWidth <= width) return node;
    Object.assign(node.style, { whiteSpace: 'normal', width: `${width}px`, lineHeight: 1.15, textAlign: align });
    return setWords(node, node.textContent, width);
  }

  // The map of a long list: groups of numbered items. "full" is a chapter card (boxes, numbered dots, names, the
  // current group lit); "strip" is the thin progress line on every item's page (done, current, still to come).
  // groups: [{label, count}], done: items finished before this one, current: global index of this item (or -1),
  // focus: group index lit, caption: text after the strip, enter: "settled" (strip only) when it stays across pages.
  Recipes.roadmap = (root, p, ctx) => {
    const layer = svgLayer(root, ctx.width, ctx.height), pen = rough.svg(layer);
    const W = ctx.width, H = ctx.height, groups = p.groups || [], total = groups.reduce((a, g) => a + g.count, 0);
    const done = p.done || 0, current = p.current ?? -1, focus = p.focus ?? -1, start = p.delay || 0, accent = accentOf(p);
    // active: [from, to) item positions lit together (a page that covers a whole group); numbers: a group's own item numbers
    const active = p.active || [-1, -1];
    const state = (index) => (index === current || (index >= active[0] && index < active[1]) ? 'current' : index < done ? 'done' : 'todo');
    const numberOf = (g, k, index) => String(groups[g].numbers ? groups[g].numbers[k] : index + 1);
    if (p.mode === 'strip') {
      // A long list (29 items) and its caption must fit the box: the dots shrink, the caption keeps its size.
      const size = p.size || H * 0.5;
      const captionWidth = p.caption ? [...p.caption].reduce((w, c) => w + (c.charCodeAt(0) > 255 ? 1.02 : 0.6) * size, 0) : 0;
      const units = 0.75 + total * 1.55 + groups.length * 0.85 + (p.caption ? 0.4 : 0);   // in dots: lead, dot+gap, group gaps
      const dot = Math.min(H * 0.36, (W - 12 - captionWidth) / units), gap = dot * 0.55, groupGap = dot * 1.4;
      let x = dot * 0.75 + 6, index = 0, lit = 0;   // room for the current item's larger dot
      // enter "settled": the strip stays across a page turn: every other dot is in place at t=0, only this page's
      // item moves (and the caption arrives); the default draws the whole strip in, dot by dot.
      const settled = p.enter === 'settled';
      groups.forEach((group, g) => {
        for (let k = 0; k < group.count; k++, index++) {
          const s = state(index), r = s === 'current' ? dot * 0.62 : dot / 2;
          const node = pen.circle(x, H / 2, 2 * r, roughOf(p, { strokeWidth: 2.5, stroke: s === 'todo' ? muted(p) : s === 'current' ? accent : ink(p),
            fill: s === 'done' ? ink(p) : s === 'current' ? accent : 'none', fillStyle: 'solid', seed: (p.seed || 11) + index }));
          layer.appendChild(node);
          if (!settled) popIn(node, start + index * 18, s === 'current' ? 420 : 200);
          else if (s === 'current') {
            // the strip as the last page left it: this item still an open circle (same outline), filled as it pops in
            const was = pen.circle(x, H / 2, dot, roughOf(p, { strokeWidth: 2.5, stroke: muted(p), fill: 'none', fillStyle: 'solid', seed: (p.seed || 11) + index }));
            layer.insertBefore(was, node);
            popIn(node, start + 40 * lit++, 420);
          }
          if (s === 'current' && (ctx.duration || Infinity) - start >= 1370) node.animate([{ transform: 'scale(1)' }, { transform: 'scale(1.35)' }, { transform: 'scale(1)' }],
                                           { delay: start + 600, duration: 700, easing: 'ease-in-out', fill: 'none' });
          x += dot + gap;
        }
        x += groupGap - gap;
      });
      if (p.caption) appear(label(root, p.caption, x + dot * 0.4, H / 2, p.size || H * 0.5, p, 'left', p.caption_color || accent), start + 200);
      return;
    }
    const columns = groups.length, cell = W / Math.max(1, columns), boxH = H * 0.62, top = H * 0.06;
    let index = 0;
    groups.forEach((group, g) => {
      const x0 = cell * g + cell * 0.06, width = cell * 0.88, at = start + g * (p.group_ms || 380), lit = g === focus;
      const frame = pen.rectangle(x0, top, width, boxH, roughOf(p, { stroke: lit ? accent : ink(p), strokeWidth: lit ? 5 : 3,
        fill: lit ? accent : undefined, fillStyle: 'hachure', hachureGap: 14, fillWeight: 1.5, seed: (p.seed || 11) + g }));
      if (lit) frame.querySelectorAll('path').forEach((path) => { if (path.getAttribute('stroke') && path.getAttribute('fill') === 'none') path.setAttribute('stroke-opacity', '1'); });
      layer.appendChild(frame);
      drawOn(frame, at, 380);
      const perRow = Math.min(group.count, 2), rows = Math.ceil(group.count / perRow);
      const d = Math.min(width / (perRow * 1.6), boxH / (rows * 1.5));
      for (let k = 0; k < group.count; k++, index++) {
        const cx = x0 + width * ((k % perRow) + 0.5) / perRow, cy = top + boxH * (Math.floor(k / perRow) + 0.5) / rows, s = state(index);
        const done_ = s === 'done';
        const node = pen.circle(cx, cy, d, roughOf(p, { stroke: s === 'todo' && focus >= 0 && !lit ? muted(p) : ink(p), fill: done_ ? ink(p) : s === 'current' ? accent : '#FFFFFF',
          fillStyle: 'solid', strokeWidth: 3, seed: (p.seed || 11) + 40 + index }));
        layer.appendChild(node);
        const when = at + 200 + k * (p.item_ms || 90);
        popIn(node, when);
        const number = label(root, numberOf(g, k, index), cx, cy, d * 0.5, p, 'center', done_ || s === 'current' ? '#FFFFFF' : (s === 'todo' && focus >= 0 && !lit ? muted(p) : ink(p)));
        show(number, when + 120);
        // pulse only as many times as fit before the layer is held, so no dot freezes half enlarged
        const pulses = Math.max(0, Math.min(3, Math.floor(((ctx.duration || Infinity) - when - 570) / 900)));
        if (s === 'current' && pulses) node.animate([{ transform: 'scale(1)' }, { transform: 'scale(1.18)' }, { transform: 'scale(1)' }],
                                                    { delay: when + 500, duration: 900, iterations: pulses, easing: 'ease-in-out' });
      }
      const name = block(root, group.label, x0 - cell * 0.04, top + boxH + H * 0.17, width + cell * 0.08, p.size || Math.min(H * 0.13, cell * 0.16), p,
                         lit ? accent : (focus >= 0 ? muted(p) : ink(p)));
      appear(name, at + 260);
    });
  };

  // A hand-ruled table filled row by row as the narration reaches each row. columns: [header...], rows: [[cell...]],
  // widths: column shares, row_ms: [reveal time per row], highlight: [{row, col, ms, color}]. A cell "✓" or "✗" is drawn.
  Recipes.table = (root, p, ctx) => {
    const layer = svgLayer(root, ctx.width, ctx.height), pen = rough.svg(layer);
    const W = ctx.width, H = ctx.height, columns = p.columns || [], rows = p.rows || [], start = p.delay || 0;
    const shares = p.widths || columns.map(() => 1 / columns.length);
    const edges = shares.reduce((list, share) => list.concat(list[list.length - 1] + share * (W - 24)), [12]);
    const headH = H * (p.header_share || 0.2), rowH = (H - headH - 6) / Math.max(1, rows.length), size = p.size || Math.min(rowH * 0.42, 44);
    const grid = document.createElementNS(NS, 'g');
    grid.appendChild(pen.line(12, 10, W - 12, 10, roughOf(p)));
    grid.appendChild(pen.line(12, headH, W - 12, headH, roughOf(p, { strokeWidth: 4.5 })));
    edges.slice(1, -1).forEach((x) => grid.appendChild(pen.line(x, 10, x, H - 10, roughOf(p))));
    layer.appendChild(grid);
    drawOn(grid, start, 520);
    columns.forEach((text, c) => appear(block(root, text, edges[c] + 8, headH / 2 + 2, edges[c + 1] - edges[c] - 16, size * 1.05, p, p.header_color || accentOf(p)), start + 300 + c * 80));
    // highlight: [{row, col, color, ms}]; anything else is ignored (a bare number made rough.js hatch NaN forever)
    (Array.isArray(p.highlight) ? p.highlight : []).filter((h) => h && Number.isInteger(h.col) && Number.isInteger(h.row) && edges[h.col + 1] != null).forEach((h) => {
      const x = edges[h.col], y = headH + rowH * h.row;
      const mark = pen.rectangle(x + 6, y + 6, edges[h.col + 1] - x - 12, rowH - 12, roughOf(p, { stroke: 'none', fill: h.color || accentOf(p), fillStyle: 'hachure', hachureGap: 9, fillWeight: 2.2 }));
      mark.style.opacity = 0.45; layer.insertBefore(mark, layer.firstChild);
      show(mark, h.ms ?? start + 900, 300);
    });
    rows.forEach((row, r) => {
      const at = (p.row_ms && p.row_ms[r] != null) ? p.row_ms[r] : start + 700 + r * (p.stagger || 500), y = headH + rowH * (r + 0.5);
      const rule = pen.line(12, headH + rowH * (r + 1) - 4, W - 12, headH + rowH * (r + 1) - 4, roughOf(p, { strokeWidth: 2, stroke: muted(p) }));
      layer.appendChild(rule); drawOn(rule, at, 260);
      row.forEach((text, c) => {
        const cx = (edges[c] + edges[c + 1]) / 2;
        if (text === '✓' || text === '✗') {
          const s = Math.min(rowH * 0.5, 46), mark = text === '✓' ? roughCheck(pen, cx, y, s, roughOf(p, { stroke: p.yes_color || '#3D8B5A', strokeWidth: 6 }))
                                                                   : roughCross(pen, cx, y, s, roughOf(p, { stroke: p.no_color || '#C2452D', strokeWidth: 6 }));
          layer.appendChild(mark); drawOn(mark, at + c * 120, 260);
        } else appear(block(root, text, edges[c] + 10, y, edges[c + 1] - edges[c] - 20, size, p, c === 0 ? ink(p) : (p.cell_color || ink(p)), c === 0 && p.first_left ? 'left' : 'center'), at + c * 120);
      });
    });
  };

  // A diagram of boxes and arrows: a flowchart, a loop, a tree of guesses, an architecture. nodes: [{id, text, at: [x, y]
  // (0..1 of the box), size: [w, h] (0..1), shape: box|round|circle, color, ms}], edges: [{from, to, text, ms, bend,
  // dashed}], tokens: [{path: [ids], ms, edge_ms, loop, color}] a dot travelling the path; shake: [{id, ms}] a clash.
  Recipes.flow = (root, p, ctx) => {
    const layer = svgLayer(root, ctx.width, ctx.height), pen = rough.svg(layer);
    const W = ctx.width, H = ctx.height, start = p.delay || 0, nodes = {}, size = p.size || Math.min(W, H) * 0.075;
    (p.nodes || []).forEach((node, i) => {
      const [u, v] = node.at, [sw, sh] = node.size || p.node_size || [0.22, 0.2];
      const cx = u * W, cy = v * H, w = sw * W, h = sh * H, at = node.ms ?? start + i * (p.stagger || 260), shape = node.shape || 'round';
      const color = node.color || ink(p);
      const o = roughOf(p, { stroke: color, seed: (p.seed || 11) + i, fill: node.fill, fillStyle: node.fill ? 'hachure' : undefined, hachureGap: 10, fillWeight: 1.8 });
      const group = document.createElementNS(NS, 'g');
      if (shape !== 'none')   // "none": a plain label (a row's name), no frame
        group.appendChild(shape === 'circle' ? pen.ellipse(cx, cy, w, h, o) : pen.path(roundedPath(cx - w / 2, cy - h / 2, w, h, shape === 'box' ? 4 : Math.min(w, h) * 0.22), o));
      layer.appendChild(group);
      drawOn(group, at, 380);
      const text = block(root, node.text || '', cx - w / 2 + 6, cy, w - 12, node.text_size || size, p, node.text_color || (shape === 'none' ? color : ink(p)));
      appear(text, at + 220);
      nodes[node.id] = { cx, cy, w, h, shape, group, text, at };
    });
    const rim = (n, dx, dy) => {   // where a line toward (dx, dy) leaves the node's outline
      const len = Math.hypot(dx, dy) || 1;
      if (n.shape === 'circle') return [n.cx + dx / len * n.w / 2, n.cy + dy / len * n.h / 2];
      const t = Math.min(Math.abs(n.w / 2 / (dx || 1e-6)), Math.abs(n.h / 2 / (dy || 1e-6)));
      return [n.cx + dx * t, n.cy + dy * t];
    };
    const routes = {};
    (p.edges || []).forEach((edge, i) => {
      const a = nodes[edge.from], b = nodes[edge.to];
      if (!a || !b) throw new Error(`flow edge ${edge.from} -> ${edge.to}: unknown node`);
      const [ax, ay] = rim(a, b.cx - a.cx, b.cy - a.cy), [bx, by] = rim(b, a.cx - b.cx, a.cy - b.cy);
      const pad = 8, len = Math.hypot(bx - ax, by - ay) || 1, ux = (bx - ax) / len, uy = (by - ay) / len;
      const sx = ax + ux * pad, sy = ay + uy * pad, ex = bx - ux * pad, ey = by - uy * pad, bend = edge.bend || 0;
      const mx = (sx + ex) / 2 - (ey - sy) * bend, my = (sy + ey) / 2 + (ex - sx) * bend;
      const o = roughOf(p, { stroke: edge.color || ink(p), strokeWidth: 3, seed: (p.seed || 11) + 100 + i });
      const g = document.createElementNS(NS, 'g');
      g.appendChild(bend ? pen.curve([[sx, sy], [mx, my], [ex, ey]], o) : pen.line(sx, sy, ex, ey, o));
      const angle = Math.atan2(ey - (bend ? my : sy), ex - (bend ? mx : sx)), head = edge.head || 20;
      for (const side of [-1, 1]) { const q = angle + Math.PI + side * 0.5; g.appendChild(pen.line(ex, ey, ex + head * Math.cos(q), ey + head * Math.sin(q), o)); }
      if (edge.dashed) g.querySelectorAll('path').forEach((path) => path.setAttribute('stroke-dasharray', '10 9'));
      layer.appendChild(g);
      const at = edge.ms ?? Math.max(a.at, b.at) + 300;
      drawOn(g, at, 360);
      if (edge.text) {
        // Beside the line, not on it: offset along the edge's normal (the side facing up), on a paper-coloured pad, so a
        // diagonal arrow never runs through its own label like a strike-through.
        const len = Math.hypot(ex - sx, ey - sy) || 1;
        let nx = -(ey - sy) / len, ny = (ex - sx) / len;
        if (ny > 0) { nx = -nx; ny = -ny; }
        const tag = label(root, edge.text, (sx + ex) / 2 - (ey - sy) * bend * 0.5 + nx * size * 0.9 + (edge.text_dx || 0),
                          (sy + ey) / 2 + (ex - sx) * bend * 0.5 + ny * size * 0.9 + (edge.text_dy || 0), size * 0.8, p, 'center', edge.text_color || accentOf(p));
        Object.assign(tag.style, { background: p.label_bg || 'rgba(248, 245, 238, 0.94)', padding: '0.08em 0.3em', borderRadius: '0.3em' });
        appear(tag, at + 200);
      }
      routes[`${edge.from}>${edge.to}`] = bend ? [[sx, sy], [mx, my], [ex, ey]] : [[sx, sy], [ex, ey]];
    });
    (p.tokens || []).forEach((token, i) => {
      const points = [];
      token.path.slice(1).forEach((id, k) => {
        const from = token.path[k], route = routes[`${from}>${id}`];
        const leg = route || [[nodes[from].cx, nodes[from].cy], [nodes[id].cx, nodes[id].cy]];
        (points.length ? leg.slice(1) : leg).forEach((q) => points.push(q));
      });
      if (points.length < 2) return;
      const dot = document.createElementNS(NS, 'circle');
      dot.setAttribute('r', token.radius || size * 0.32); dot.setAttribute('fill', token.color || accentOf(p));
      dot.setAttribute('stroke', ink(p)); dot.setAttribute('stroke-width', 2.5);
      layer.appendChild(dot);
      const lengths = points.slice(1).map((q, k) => Math.hypot(q[0] - points[k][0], q[1] - points[k][1]));
      const total = lengths.reduce((a, b) => a + b, 0) || 1;
      let run = 0;
      const frames = points.map((q, k) => { if (k) run += lengths[k - 1]; return { transform: `translate(${q[0]}px, ${q[1]}px)`, offset: run / total }; });
      const legs = token.path.length - 1, duration = (token.edge_ms || 700) * legs;
      dot.animate([{ opacity: 0 }, { opacity: 1 }], { delay: token.ms ?? start + 1200, duration: 150, fill: 'both' });
      dot.animate(frames, { delay: token.ms ?? start + 1200, duration, iterations: token.loop ? Infinity : 1, easing: 'linear', fill: 'both' });
    });
    (p.shake || []).forEach((s) => {
      const n = nodes[s.id];
      [[n.group, ''], [n.text, ' translateY(-50%)']].forEach(([node, rest]) => node.animate(
        [0, -9, 8, -6, 5, 0].map((dx) => ({ transform: `translateX(${dx}px)${rest}` })),
        { delay: s.ms, duration: s.duration || 520, iterations: s.times || 2 }));
      if (s.mark) { const m = roughCross(pen, n.cx + n.w * 0.42, n.cy - n.h * 0.42, size * 0.9, roughOf(p, { stroke: '#C2452D', strokeWidth: 6 }));
        layer.appendChild(m); drawOn(m, s.ms + 200, 260); }
    });
  };
  function roundedPath(x, y, w, h, r) {
    return `M ${x + r} ${y} H ${x + w - r} Q ${x + w} ${y} ${x + w} ${y + r} V ${y + h - r} Q ${x + w} ${y + h} ${x + w - r} ${y + h} H ${x + r} Q ${x} ${y + h} ${x} ${y + h - r} V ${y + r} Q ${x} ${y} ${x + r} ${y} Z`;
  }

  // Who does what over time: lanes (people, agents) and blocks on a shared clock, drawn as a playhead sweeps left to
  // right. lanes: [{label}], blocks: [{lane, from, to, text, kind: work|wait}], span: clock units across, sweep_ms.
  Recipes.swimlane = (root, p, ctx) => {
    const layer = svgLayer(root, ctx.width, ctx.height), pen = rough.svg(layer);
    const W = ctx.width, H = ctx.height, lanes = p.lanes || [], start = p.delay || 0, sweep = p.sweep_ms || 4000, span = p.span || 10;
    const titleH = p.title ? H * 0.2 : 0, left = W * (p.label_share || 0.17), laneH = (H - titleH) / Math.max(1, lanes.length);
    const size = p.size || Math.min(laneH * 0.32, 40), xOf = (t) => left + (W - left - 8) * t / span;
    if (p.title) appear(label(root, p.title, 16, titleH / 2, size * 1.05, p, 'left', p.title_color || accentOf(p)), start);
    lanes.forEach((lane, i) => {
      const y = titleH + laneH * (i + 0.5);
      appear(label(root, lane.label, left - 14, y, size, p, 'right'), start + 100 + i * 80);
      const rule = pen.line(left, titleH + laneH * (i + 1) - 8, W - 10, titleH + laneH * (i + 1) - 8, roughOf(p, { strokeWidth: 1.6, stroke: muted(p) }));
      layer.appendChild(rule); drawOn(rule, start + 100, 400);
    });
    (p.blocks || []).forEach((b, i) => {
      const x = xOf(b.from), w = xOf(b.to) - x, y = titleH + laneH * b.lane + laneH * 0.16, h = laneH * 0.68;
      const work = (b.kind || 'work') === 'work';
      const node = pen.rectangle(x + 2, y, Math.max(6, w - 4), h, roughOf(p, { stroke: work ? (b.color || ink(p)) : muted(p), strokeWidth: work ? 3 : 2.2,
        fill: work ? (b.color || accentOf(p)) : undefined, fillStyle: 'hachure', hachureGap: 8, fillWeight: 2, seed: (p.seed || 11) + i }));
      if (!work) node.querySelectorAll('path').forEach((path) => path.setAttribute('stroke-dasharray', '9 8'));
      node.style.transformBox = 'view-box'; node.style.transformOrigin = `${x}px 0px`;
      layer.appendChild(node);
      const at = start + 300 + sweep * b.from / span, grow = Math.max(160, sweep * (b.to - b.from) / span);
      node.animate([{ transform: 'scaleX(0)' }, { transform: 'scaleX(1)' }], { delay: at, duration: grow, easing: 'linear', fill: 'both' });
      if (b.text) appear(block(root, b.text, x + 4, y + h / 2, Math.max(40, w - 8), b.text_size || size * 0.8, p, work ? (b.text_color || ink(p)) : muted(p)), at + grow * 0.4);
    });
    const head = pen.line(0, titleH + 4, 0, H - 8, roughOf(p, { stroke: accentOf(p), strokeWidth: 3 }));
    layer.appendChild(head);
    head.animate([{ transform: `translateX(${left}px)`, opacity: 1 }, { transform: `translateX(${W - 6}px)`, opacity: 1 }, { transform: `translateX(${W - 6}px)`, opacity: 0 }],
                 { delay: start + 300, duration: sweep + 300, easing: 'linear', fill: 'both' });
  };

  // Items written one by one and ticked, crossed or struck through when the narration settles them.
  // items: [{text, ms, mark: check|cross|strike, mark_ms}].
  Recipes.checklist = (root, p, ctx) => {
    const layer = svgLayer(root, ctx.width, ctx.height), pen = rough.svg(layer);
    const W = ctx.width, H = ctx.height, items = p.items || [], start = p.delay || 0;
    const columns = p.columns || 1, rows = Math.ceil(items.length / columns), rowH = H / Math.max(1, rows), colW = W / columns;
    const size = p.size || Math.min(rowH * 0.5, 50), box = size * 0.95;
    items.forEach((item, i) => {
      const x = colW * Math.floor(i / rows) + 8, y = rowH * (i % rows + 0.5), at = item.ms ?? start + i * (p.stagger || 420);
      const square = pen.rectangle(x, y - box / 2, box, box, roughOf(p, { strokeWidth: 3, seed: (p.seed || 11) + i }));
      layer.appendChild(square); drawOn(square, at, 240);
      const text = wrapWide(label(root, item.text, x + box * 1.45, y, size, p, 'left', item.color), colW * (Math.floor(i / rows) + 1) - x - box * 1.45 - 8);
      appear(text, at + 120);
      const when = item.mark_ms ?? at + 600;
      if (item.mark === 'check') { const m = roughCheck(pen, x + box / 2, y, box * 1.05, roughOf(p, { stroke: p.yes_color || '#3D8B5A', strokeWidth: 6 })); layer.appendChild(m); drawOn(m, when, 260); }
      else if (item.mark === 'cross') { const m = roughCross(pen, x + box / 2, y, box * 0.8, roughOf(p, { stroke: p.no_color || '#C2452D', strokeWidth: 6 })); layer.appendChild(m); drawOn(m, when, 260); }
      if (item.mark === 'strike' || item.strike) {
        const width = text.getBoundingClientRect().width;
        const line = pen.line(x + box * 1.3, y + 2, x + box * 1.6 + width, y - 4, roughOf(p, { stroke: p.no_color || '#C2452D', strokeWidth: 5 }));
        layer.appendChild(line); drawOn(line, when, 320);
        text.animate([{ opacity: 1 }, { opacity: 0.45 }], { delay: when + 200, duration: 300, fill: 'forwards' });
      }
    });
  };

  // Many small agent windows (amber title bar, a few lines): versions, parallel agents, runs. count, columns, card_ms
  // (stagger), labels, flicker: {indices, step_ms, ms} an error mark hopping between cards (which one failed?),
  // groups: [{indices, label, x: centre 0..1, ms}] the cards slide into labelled groups, highlight: [{indices, ms}].
  Recipes.cards = (root, p, ctx) => {
    const layer = svgLayer(root, ctx.width, ctx.height), pen = rough.svg(layer);
    const W = ctx.width, H = ctx.height, count = p.count || 6, columns = p.columns || Math.min(count, 5), rows = Math.ceil(count / columns);
    const start = p.delay || 0, top = p.groups ? H * 0.2 : 0, cw = W / columns, ch = (H - top) / rows;
    const w = Math.min(cw * 0.8, ch * 1.25), h = Math.min(ch * 0.78, w * 0.78), bar = p.bar || accentOf(p);
    const cards = [];
    for (let i = 0; i < count; i++) {
      const cx = cw * (i % columns + 0.5), cy = top + ch * (Math.floor(i / columns) + 0.5);
      const slot = document.createElementNS(NS, 'g'), g = document.createElementNS(NS, 'g');   // slot moves, g pops
      slot.appendChild(g);
      g.appendChild(pen.rectangle(cx - w / 2, cy - h / 2, w, h, roughOf(p, { strokeWidth: 2.6, fill: '#FFFFFF', fillStyle: 'solid', seed: (p.seed || 11) + i })));
      g.appendChild(pen.rectangle(cx - w / 2, cy - h / 2, w, h * 0.2, roughOf(p, { strokeWidth: 2.6, fill: bar, fillStyle: 'solid', seed: (p.seed || 11) + 50 + i })));
      [0.42, 0.6, 0.78].forEach((f, k) => g.appendChild(pen.line(cx - w * 0.36, cy - h / 2 + h * f, cx - w * 0.36 + w * (0.72 - 0.18 * k), cy - h / 2 + h * f, roughOf(p, { strokeWidth: 2, stroke: muted(p) }))));
      layer.appendChild(slot);
      popIn(g, start + i * (p.card_ms || 60), 300);
      if (p.labels && p.labels[i]) appear(wrapWide(label(root, p.labels[i], cx, cy + h * 0.12, Math.min(h * 0.3, 34), p), w * 0.9, 'center'), start + i * (p.card_ms || 60) + 150);
      cards.push({ g: slot, cx, cy });
    }
    (p.groups || []).forEach((group, k) => {
      const n = group.indices.length, gcols = Math.min(n, group.columns || 3), gx = group.x * W, at = group.ms ?? start + 1500;
      appear(label(root, group.label, gx, top * 0.45, p.size || Math.min(top * 0.55, 44), p, 'center', group.color || (k ? accentOf(p) : ink(p))), at + 200);
      let x1 = Infinity, y1 = Infinity, x2 = -Infinity, y2 = -Infinity;
      const k_ = p.group_scale || 0.78, gw = w * k_, gh = h * k_;   // grouped cards shrink a little, so the groups stand apart
      group.indices.forEach((index, j) => {
        const card = cards[index], tx = gx + (j % gcols - (gcols - 1) / 2) * gw * 1.15, ty = top + ch * 0.5 + Math.floor(j / gcols) * gh * 1.15;
        // scale about the origin, then move: the card's centre lands on (tx, ty)
        card.g.animate([{ transform: 'translate(0px, 0px) scale(1)' }, { transform: `translate(${tx - k_ * card.cx}px, ${ty - k_ * card.cy}px) scale(${k_})` }],
                       { delay: at, duration: 700, easing: 'cubic-bezier(.5,0,.2,1)', fill: 'forwards' });
        card.cx = tx; card.cy = ty;
        x1 = Math.min(x1, tx - gw / 2); x2 = Math.max(x2, tx + gw / 2); y1 = Math.min(y1, ty - gh / 2); y2 = Math.max(y2, ty + gh / 2);
      });
      // The group's own frame, drawn once its cards have arrived: the split reads as two sets, not one row.
      const pad = Math.min(gw, gh) * 0.18;
      const frame = pen.path(roundedPath(x1 - pad, y1 - pad, x2 - x1 + 2 * pad, y2 - y1 + 2 * pad, pad * 1.5),
                             roughOf(p, { stroke: group.color || (k ? accentOf(p) : ink(p)), strokeWidth: 3, seed: (p.seed || 11) + 90 + k }));
      frame.querySelectorAll('path').forEach((path) => path.setAttribute('stroke-dasharray', '12 9'));
      layer.insertBefore(frame, layer.firstChild);
      drawOn(frame, at + 650, 400);
    });
    (p.highlight || []).forEach((hl) => hl.indices.forEach((index) => {
      const card = cards[index];
      const ring = pen.rectangle(card.cx - w * 0.6, card.cy - h * 0.62, w * 1.2, h * 1.24, roughOf(p, { stroke: hl.color || accentOf(p), strokeWidth: 5 }));
      layer.appendChild(ring); drawOn(ring, hl.ms ?? start + 1500, 300);
    }));
    if (p.flicker) {
      const f = p.flicker, mark = roughCross(pen, 0, 0, Math.min(w, h) * 0.55, roughOf(p, { stroke: '#C2452D', strokeWidth: 7 }));
      layer.appendChild(mark);
      const frames = f.indices.map((index) => ({ transform: `translate(${cards[index].cx}px, ${cards[index].cy}px)` }));
      mark.animate([{ opacity: 0 }, { opacity: 1 }], { delay: f.ms ?? start + 900, duration: 120, fill: 'both' });
      mark.animate(frames, { delay: f.ms ?? start + 900, duration: (f.step_ms || 380) * frames.length, easing: `steps(${frames.length}, end)`, fill: 'both' });
      if (f.question) {   // the hopping stops on a question: which one was it?
        const last = cards[f.indices[f.indices.length - 1]];
        const q = label(root, '?', last.cx + w * 0.42, last.cy - h * 0.3, Math.min(w, h) * 0.7, p, 'center', '#C2452D');
        show(q, (f.ms ?? start + 900) + (f.step_ms || 380) * frames.length);
      }
    }
  };

  // A heartbeat on a clock: the line runs left to right and spikes at each beat, where the beat's label pops.
  // beats: [{ms, label}], sweep_ms.
  Recipes.pulse = (root, p, ctx) => {
    const layer = svgLayer(root, ctx.width, ctx.height), pen = rough.svg(layer);
    const W = ctx.width, H = ctx.height, start = p.delay || 0, sweep = p.sweep_ms || 6000, base = H * 0.66, size = p.size || H * 0.13;
    // spikes stay 40 px inside the box; labels are clamped so a late beat's label does not run off the right edge
    const beats = (p.beats || []).map((b) => ({ ...b, x: 40 + (W - 100) * Math.min(1, Math.max(0, (b.ms - start) / sweep)) }));
    const points = [[10, base]];
    beats.forEach((b) => points.push([b.x - 26, base], [b.x - 12, base - H * 0.08], [b.x, base + H * 0.12], [b.x + 12, base - H * 0.42], [b.x + 26, base + H * 0.06], [b.x + 38, base]));
    points.push([W - 10, base]);
    const line = pen.linearPath(points, roughOf(p, { stroke: p.color || accentOf(p), strokeWidth: 4.5, roughness: 0.6 }));
    layer.appendChild(line);
    drawOn(line, start, sweep);
    beats.forEach((b, i) => {
      // labels alternate above the spike and under the line, so beats close in time do not overprint each other
      const left = Math.min(W - 8 - W * 0.32, Math.max(8, b.x + 12 - W * 0.16)), below = i % 2 === 1;
      appear(block(root, b.label, left, below ? base + H * 0.2 : base - H * 0.42 - size * 1.1, W * 0.32, size, p, b.color || ink(p)), b.ms);
      const dot = pen.circle(b.x + 12, base - H * 0.42, 14, roughOf(p, { fill: p.color || accentOf(p), fillStyle: 'solid', strokeWidth: 2 }));
      layer.appendChild(dot); popIn(dot, b.ms);
    });
  };

  window.Recipes = Recipes;
  window.RecipeText = { wordUnits, setWords };   // for custom recipes (text that wraps between words) and tests
})();
