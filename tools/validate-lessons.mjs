// Checks every lesson in backend/content (or the folder given as argv[2]) against the lesson format.
//   node tools/validate-lessons.mjs [dir] [week-NN.md ...]
// Exit code 1 if any lesson has an error. Warnings do not fail the run.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import * as M from '../frontend/src/dc/lessonMath.js';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const dir = process.argv[2] && !process.argv[2].endsWith('.md') ? path.resolve(process.argv[2]) : path.join(root, 'backend', 'content');
const only = process.argv.slice(2).filter((a) => a.endsWith('.md'));

const HOOK_KINDS = ['text', 'image', 'speech', 'tokens', 'denoise', 'conv', 'bars', 'loss'];
let totalErrors = 0;

function check(file) {
  const src = fs.readFileSync(path.join(dir, file), 'utf8');
  const n = Number(/week-(\d\d)\.md/.exec(file)[1]);
  const errors = [];
  const warns = [];
  const err = (m) => errors.push(m);
  const warn = (m) => warns.push(m);
  const isStr = (x, min = 1) => typeof x === 'string' && x.trim().length >= min;
  const isNum = (x) => typeof x === 'number' && Number.isFinite(x);
  const rect = (m) => Array.isArray(m) && m.length > 0 && m.every((r) => Array.isArray(r) && r.length === m[0].length && r.every(isNum));

  // ---- markdown structure
  if (!src.startsWith('> ')) err('first line must be the "> Content based on …" source note');
  const h1 = /^# Week (\d+) — (.+)$/m.exec(src);
  if (!h1) err('missing "# Week N — Title" heading');
  else if (Number(h1[1]) !== n) err(`h1 says Week ${h1[1]} but the file is week-${String(n).padStart(2, '0')}.md`);
  if (!/^\*\*Purpose:\*\* .+==.+==.+/m.test(src)) err('missing "**Purpose:**" line containing a ==highlight==');
  if (!/^\*\*Term:\*\* .+\*\*Month:\*\* \d.*\*\*Time:\*\*/m.test(src)) err('missing "**Term:** … · **Month:** n · **Time:** …" line');
  const month = Math.ceil(n / 4);
  if (!new RegExp(`\\*\\*Month:\\*\\* ${month}\\b`).test(src)) warn(`Month should be ${month} for week ${n}`);
  const h2 = [...src.matchAll(/^## (.+)$/gm)].map((m) => m[1]);
  if (!h2.some((h) => /^By the end/.test(h))) err('missing "## By the end, you can…" section');
  if (!h2.some((h) => /^Key takeaway/.test(h))) err('missing "## Key takeaways" section');
  if (!h2.some((h) => /^Check your understanding/.test(h))) err('missing "## Check your understanding" section');
  const rails = h2.map((h) => /\{rail=([^}]+)\}\s*$/.exec(h)).filter(Boolean).map((m) => m[1]);
  if (rails.length < 5) err(`needs at least 5 {rail=…} headings, found ${rails.length}`);
  if (rails.length > 8) warn(`rail has ${rails.length} items; 8 is the most that fits`);
  if (new Set(rails).size !== rails.length) err('duplicate {rail=…} labels');
  rails.forEach((r) => r.length > 18 && warn(`rail label "${r}" is long (>18 chars)`));
  if (!/^\{lede\} /m.test(src)) err('use {lede} at the start of the main-idea paragraph of the first concept section');
  if ((src.match(/\{lede\}/g) || []).length < 2) warn('use {lede} in more than one section');
  // prose only: fenced blocks (JSON figures, code) may legitimately contain [[ or ==
  const prose = src.replace(/^```[\s\S]*?^```/gm, '');
  const marks = (prose.match(/==[^=\n]+==/g) || []).length;
  if (marks < 8) err(`only ${marks} ==highlights==; use at least 8 on key ideas`);
  if ((prose.match(/==/g) || []).length % 2) err('unbalanced == highlight markers');
  const terms = [...prose.matchAll(/\[\[([^\]|\n]+)\|([^\]\n]+)\]\]/g)];
  if (terms.length < 4) err(`only ${terms.length} [[term|definition]] items; use at least 4`);
  if ((prose.match(/\[\[/g) || []).length !== terms.length) err('a [[term|definition]] is malformed (no "|", a "]" inside, or a line break)');
  if ((src.match(/^```/gm) || []).length % 2) err('unclosed ``` code fence');
  if (/\bTODO\b|lorem ipsum|FIXME|XXX/i.test(src)) err('contains a TODO/placeholder');
  const bad = src.match(/<(script|iframe|style|img|object)\b/i);
  if (bad) err(`raw <${bad[1]}> is not allowed`);

  // ---- fenced blocks
  const blocks = [...src.matchAll(/^```(viz|quiz)\n([\s\S]*?)\n```/gm)].map((m) => ({ kind: m[1], text: m[2] }));
  let hooks = 0;
  let figs = 0;
  let quizzes = 0;
  blocks.forEach((b, i) => {
    let s;
    try {
      s = JSON.parse(b.text);
    } catch (e) {
      err(`block ${i + 1} (${b.kind}) is not valid JSON: ${e.message}`);
      return;
    }
    const at = (m) => err(`block ${i + 1} (${b.kind}${s.type ? ' ' + s.type : ''}): ${m}`);
    if (b.kind === 'quiz') {
      quizzes++;
      if (!/^[a-z0-9-]{1,34}$/.test(s.id || '')) at('id must match [a-z0-9-] and be at most 34 characters');
      if (!isNum(s.passMark)) at('passMark must be a number');
      const q = s.questions || [];
      if (q.length < 6 || q.length > 15) at(`needs 6–15 questions, has ${q.length}`);
      const pos = [0, 0, 0, 0];
      q.forEach((x, k) => {
        if (!isStr(x.q, 12)) at(`question ${k + 1} has no text`);
        if (!Array.isArray(x.options) || x.options.length < 3 || x.options.length > 4) at(`question ${k + 1} needs 3–4 options`);
        else if (new Set(x.options.map((o) => o.trim().toLowerCase())).size !== x.options.length) at(`question ${k + 1} has duplicate options`);
        if (!Number.isInteger(x.answer) || x.answer < 0 || x.answer >= (x.options || []).length) at(`question ${k + 1} has an invalid answer index`);
        else pos[x.answer]++;
        if (!isStr(x.why, 25)) at(`question ${k + 1} needs a "why" explanation`);
      });
      if (q.length >= 6 && Math.max(...pos) / q.length > 0.5) warn(`quiz answers are lopsided (${pos.join('/')}); vary the correct position`);
      return;
    }
    if (!isStr(s.title)) at('missing title');
    if (s.type === 'hook') {
      hooks++;
      const sc = s.scenes || [];
      if (sc.length < 3 || sc.length > 5) at('hook needs 3–5 scenes');
      if (!isStr(s.caption, 40)) at('hook needs a caption of at least 40 characters');
      sc.forEach((x, k) => {
        if (!isStr(x.name) || !HOOK_KINDS.includes(x.kind)) at(`scene ${k + 1} needs name and kind (${HOOK_KINDS.join('|')})`);
        if (!isStr(x.prompt)) at(`scene ${k + 1} needs a prompt`);
        if (x.kind === 'text' && !isStr(x.output)) at(`scene ${k + 1} (text) needs output`);
        if (x.kind === 'tokens' && !(Array.isArray(x.tokens) && x.tokens.length >= 2)) at(`scene ${k + 1} (tokens) needs tokens[]`);
        if (x.kind === 'bars' && !(Array.isArray(x.bars) && x.bars.length >= 2 && x.bars.every((y) => isStr(y.label) && y.value >= 0 && y.value <= 1))) at(`scene ${k + 1} (bars) needs bars[{label, value 0..1}]`);
        if (x.kind === 'denoise' && !(Array.isArray(x.image) && x.image.length >= 6 && x.image.every((r) => /^[0-9]+$/.test(r) && r.length === x.image.length))) at(`scene ${k + 1} (denoise) needs a square image of digit strings`);
      });
      return;
    }
    figs++;
    const T = {
      flow: () => {
        if (!(s.modes || []).length || s.modes.length < 2) at('needs at least 2 modes');
        (s.modes || []).forEach((m, k) => { if (!isStr(m.name) || !isStr(m.tagline) || !(m.examples || []).length) at(`mode ${k + 1} needs name, tagline, examples`); (m.examples || []).forEach((e, j) => { if (!isStr(e.input) || !isStr(e.output)) at(`mode ${k + 1} example ${j + 1} needs input and output`); }); });
      },
      sorter: () => {
        const bins = (s.bins || [{ id: 'generative' }, { id: 'discriminative' }]).map((x) => x.id);
        if ((s.cards || []).length < 6) at('needs at least 6 cards');
        (s.cards || []).forEach((c, k) => { if (!isStr(c.name) || !isStr(c.why, 20)) at(`card ${k + 1} needs name and why`); if (c.answer !== 'either' && !bins.includes(c.answer)) at(`card ${k + 1} answer "${c.answer}" is not one of ${bins.join(', ')}, either`); });
        if (s.bins && (s.bins.length < 2 || s.bins.some((x) => !isStr(x.id) || !isStr(x.label)))) at('bins need id and label');
        const used = new Set((s.cards || []).map((c) => c.answer));
        bins.forEach((x) => !used.has(x) && warn(`block ${i + 1}: no card uses bin "${x}"`));
      },
      timeline: () => { if ((s.events || []).length < 4) at('needs at least 4 events'); (s.events || []).forEach((e, k) => { if (!isStr(e.year) || !isStr(e.title) || !isStr(e.text, 30)) at(`event ${k + 1} needs year, title, text`); }); },
      temperature: () => { if (!isStr(s.prompt) || (s.tokens || []).length < 4 || s.tokens.some((t) => !isStr(t.t) || !isNum(t.logit))) at('needs prompt and ≥4 tokens {t, logit}'); },
      pipeline: () => {
        const ids = (s.nodes || []).map((x) => x.id);
        if (ids.length < 3 || new Set(ids).size !== ids.length || (s.nodes || []).some((x) => !isStr(x.label))) at('needs ≥3 nodes with unique id and a label');
        if ((s.steps || []).length < 3) at('needs ≥3 steps');
        (s.steps || []).forEach((st, k) => { if (!isStr(st.title) || !isStr(st.body, 30)) at(`step ${k + 1} needs title and body`); (st.active || []).forEach((a) => !ids.includes(a) && at(`step ${k + 1} activates unknown node "${a}"`)); if (!(st.active || []).length) at(`step ${k + 1} needs an active list`); });
      },
      calc: () => { if (!['fc_vs_conv', 'lora', 'memory', 'scaling', 'split'].includes(s.kind)) at('kind must be fc_vs_conv, lora, memory, scaling or split'); },
      network: () => {
        const L = s.layers;
        if (!Array.isArray(L) || L.length < 3 || L.some((n) => !(n >= 1 && n <= 5))) return at('layers must be ≥3 numbers, each 1..5');
        if (!Array.isArray(s.weights) || s.weights.length !== L.length - 1 || s.weights.some((m, l) => m.length !== L[l + 1] || m.some((r) => r.length !== L[l]))) return at('weights must be [layer][to][from] matching layers');
        if (!Array.isArray(s.biases) || s.biases.length !== L.length - 1 || s.biases.some((r, l) => r.length !== L[l + 1])) return at('biases must be [layer][unit] matching layers');
        if (!Array.isArray(s.inputs) || s.inputs.length !== L[0]) return at('inputs length must equal layers[0]');
        (s.checks || []).forEach((c, k) => {
          const { a } = M.forwardNet(L, s.weights, s.biases, c.inputs, s.activation || 'relu', s.outputActivation || 'linear');
          const o = a[a.length - 1];
          if (o.some((v, i) => Math.abs(v - c.output[i]) > 0.01)) at(`check ${k + 1}: expected ${c.output}, computed ${o.map((v) => v.toFixed(3))}`);
        });
      },
      fit: () => {
        const ok = (a) => Array.isArray(a) && a.length >= 4 && a.every((p) => typeof p.x === 'number' && typeof p.y === 'number');
        if (!ok(s.train) || !ok(s.test)) at('train and test need ≥4 points {x, y}');
        else if (s.train.length > 30) at('train has too many points');
      },
      trainer: () => {
        if (s.dataset && !['xor', 'circle', 'moons', 'line'].includes(s.dataset)) at('dataset must be xor, circle, moons or line');
        (s.checks || []).forEach((c, k) => {
          const r = M.trainNet({ dataset: c.dataset, hidden: c.hidden, lr: c.lr, epochs: c.epochs });
          if (c.minAcc != null && r.acc < c.minAcc) at(`check ${k + 1}: accuracy ${r.acc.toFixed(3)} < ${c.minAcc}`);
          if (c.maxAcc != null && r.acc > c.maxAcc) at(`check ${k + 1}: accuracy ${r.acc.toFixed(3)} > ${c.maxAcc}`);
        });
      },
      backprop: () => {
        const L = s.layers;
        if (!Array.isArray(L) || L.length < 3 || L.length > 4 || L.some((n) => !(n >= 1 && n <= 4))) return at('layers must be 3–4 numbers, each 1..4');
        if (!Array.isArray(s.weights) || s.weights.length !== L.length - 1 || s.weights.some((m, l) => m.length !== L[l + 1] || m.some((r) => r.length !== L[l]))) return at('weights must be [layer][to][from] matching layers');
        if (!Array.isArray(s.biases) || s.biases.length !== L.length - 1 || s.biases.some((r, l) => r.length !== L[l + 1])) return at('biases must match layers');
        if (!Array.isArray(s.inputs) || s.inputs.length !== L[0] || !Array.isArray(s.target) || s.target.length !== L[L.length - 1]) return at('inputs and target lengths must match layers');
        const r = M.backwardNet(L, s.weights, s.biases, s.inputs, s.target, s.activation || 'sigmoid', s.outputActivation || 'sigmoid', s.loss || 'mse');
        (s.checks || []).forEach((c, k) => {
          const got = c.kind === 'loss' ? r.loss : c.kind === 'delta' ? r.delta[c.layer][c.unit] : c.kind === 'dW' ? r.dW[c.layer][c.unit][c.from] : null;
          if (got == null || Math.abs(got - c.value) > (c.tol || 0.002)) at(`check ${k + 1} (${c.kind}): expected ${c.value}, computed ${got}`);
        });
      },
      gradflow: () => {
        (s.checks || []).forEach((c, k) => {
          const n = M.gradNorms({ depth: c.depth, width: s.width || 16, act: c.activation, scale: c.scale, seed: s.seed || 5 });
          const ratio = n[0] / n[n.length - 1];
          if (c.maxRatio != null && ratio > c.maxRatio) at(`check ${k + 1}: ratio ${ratio.toExponential(2)} > ${c.maxRatio}`);
          if (c.minRatio != null && ratio < c.minRatio) at(`check ${k + 1}: ratio ${ratio.toExponential(2)} < ${c.minRatio}`);
        });
      },
      fold: () => { if (s.layers != null && !(s.layers >= 1 && s.layers <= 6)) at('layers must be 1..6'); },
      mlp: () => { if (!Array.isArray(s.layers) || s.layers.length < 3 || s.layers.some((n) => !(n >= 1))) at('layers must be ≥3 positive numbers'); },
      curve: () => {
        if (!(s.series || []).length || s.series.some((x) => !M.FN[x.fn])) at(`series fn must be one of ${Object.keys(M.FN).join(', ')}`);
        if (!Array.isArray(s.xRange) || s.xRange.length !== 2 || s.xRange[0] >= s.xRange[1]) at('xRange must be [min, max]');
        if (s.logY && !Array.isArray(s.yRange)) at('logY needs yRange in log10 units, e.g. [-8, 0]');
        if ((s.series || []).some((x) => x.fn === 'decay') && !s.param) at('decay needs a param slider');
        if ((s.series || []).some((x) => x.fn === 'leaky_relu') && !s.param) at('leaky_relu needs a param slider');
      },
      descent: () => { if (s.fn && !M.LOSS[s.fn]) at(`fn must be one of ${Object.keys(M.LOSS).join(', ')}`); },
      neuron: () => { if (!(s.datasets || []).length) at('needs datasets'); (s.datasets || []).forEach((d, k) => { if (!isStr(d.name) || (d.points || []).length < 4 || d.points.some((p) => !isNum(p.x) || !isNum(p.y) || ![0, 1].includes(p.c))) at(`dataset ${k + 1} needs name and ≥4 points {x,y,c}`); }); },
      conv: () => {
        if (!rect(s.input) || !rect(s.kernel) || s.kernel.length !== s.kernel[0].length) at('input and kernel must be rectangular numeric matrices (kernel square)');
        else if (s.mode === 'transpose') { if (s.input.length > 4) warn(`block ${i + 1}: transpose input larger than 4×4 is hard to read`); }
        else { const o = M.convolve(s.input, s.kernel, s.stride || 1, s.padding || 0); if (!o.length || !o[0].length) at('output would be empty'); if (s.input.length > 8) warn(`block ${i + 1}: input larger than 8×8 is hard to read`); }
      },
      bpe: () => {
        if (!(s.corpus || []).length || s.corpus.some((c) => !/^[a-z]+$/.test(c[0]) || !Number.isInteger(c[1]))) at('corpus needs [["word", count], …] with lowercase a–z words');
        else if (M.learnBPE(s.corpus, s.maxMerges || 10).learned.length < 3) at('corpus yields fewer than 3 merges; use repeated word pieces');
        if (s.tryWord && !/^[a-z]+$/.test(s.tryWord)) at('tryWord must be lowercase a–z');
      },
      ngram: () => { const w = (s.text || '').split(/\s+/).filter(Boolean); if (w.length < 25) at('text needs ≥25 words'); const m = M.bigrams(s.text || ''); if (s.start && !m.has(s.start)) at(`start word "${s.start}" is not in the text`); },
      scatter: () => {
        const p = s.points || [];
        if (p.length < 6 || new Set(p.map((x) => x.label)).size !== p.length || p.some((x) => !isStr(x.label) || !isNum(x.x) || !isNum(x.y))) at('needs ≥6 points with unique labels and numeric x, y');
        (s.analogies || []).forEach((a, k) => {
          const by = Object.fromEntries(p.map((x) => [x.label, x]));
          if (![a.a, a.b, a.c].every((l) => by[l])) { at(`analogy ${k + 1} uses an unknown label`); return; }
          if (!isStr(a.expect)) { at(`analogy ${k + 1} needs "expect" (the label it should land on)`); return; }
          const v = [by[a.a].x - by[a.b].x + by[a.c].x, by[a.a].y - by[a.b].y + by[a.c].y];
          const top = M.nearest(p, v, [a.a, a.b, a.c], s.metric || 'cosine', 1)[0].p.label;
          if (top !== a.expect) at(`analogy ${k + 1} (${a.a} − ${a.b} + ${a.c}) lands on "${top}", not "${a.expect}"`);
        });
      },
      heatmap: () => {
        const tabs = s.tabs || [{ W: s.W }];
        tabs.forEach((t, k) => { if (!rect(t.W) || t.W.length !== (s.rows || []).length || t.W[0].length !== (s.cols || []).length) at(`matrix ${k + 1} does not match rows × cols`); else if (s.rowsSumToOne && t.W.some((r) => Math.abs(r.reduce((a, v) => a + v, 0) - 1) > 0.011)) at(`matrix ${k + 1}: rows must sum to 1 (±0.01)`); });
      },
      posenc: () => {},
      attention: () => {
        const n = (s.tokens || []).length;
        if (n < 3 || s.tokens.some((t) => !isStr(t))) at('needs ≥3 tokens');
        if (!rect(s.keys) || s.keys.length !== n) at('keys must be a numeric matrix with one row per token');
        if (!rect(s.values) || s.values.length !== n) at('values must be a numeric matrix with one row per token');
        const d = rect(s.keys) ? s.keys[0].length : 0;
        if (d > 4 || (rect(s.values) && s.values[0].length > 4)) warn(`block ${i + 1}: vectors longer than 4 numbers are hard to read`);
        if (!(s.queries || []).length || s.queries.some((x) => !isStr(x.label) || !Array.isArray(x.q) || x.q.length !== d || !x.q.every(isNum))) at('queries need label and q (same length as a key)');
        else if (rect(s.keys) && rect(s.values) && s.keys.length === n && s.values.length === n) s.queries.forEach((x, k) => { const r = M.attend(x.q, s.keys, s.values, s.scale !== false); if (Math.abs(r.alpha.reduce((a, v) => a + v, 0) - 1) > 1e-6 || r.out.some((v) => !Number.isFinite(v))) at(`query ${k + 1} gives invalid weights`); });
      },
      equation: () => {
        const parts = s.parts || [];
        const ids = parts.map((p) => p.id);
        if (parts.length < 2 || new Set(ids).size !== ids.length || parts.some((p) => !isStr(p.id) || !isStr(p.name) || !isStr(p.meaning, 15))) at('needs ≥2 parts with unique id, name and meaning');
        if (!Array.isArray(s.formula) || !s.formula.length) at('formula must be an array of strings and {t, id} terms');
        else (s.formula).forEach((f) => { if (typeof f !== 'string' && (!isStr(f.t) || !ids.includes(f.id))) at(`formula term "${f && f.t}" does not match a part id`); });
        (s.steps || []).forEach((st, k) => { if (!isStr(st.title) || !isStr(st.expr) || !isStr(st.note, 15)) at(`step ${k + 1} needs title, expr and note`); });
      },
      patch: () => { const im = s.image || []; if (im.length < 4 || !im.every((r) => Array.isArray(r) && r.length === im.length && r.every((v) => Number.isInteger(v) && v >= 0 && v <= 9))) at('image must be a square matrix of integers 0–9'); else if (!(s.patchSizes || [2, 4, 8]).some((p) => im.length % p === 0)) at('no patch size divides the image'); },
      diffusion: () => { const im = s.image || []; if (im.length < 6 || im.length > 16 || !im.every((r) => /^[0-9]+$/.test(r) && r.length === im.length)) at('image must be a square array of digit strings, 6–16 per side'); },
      nested: () => { if ((s.levels || []).length < 3 || s.levels.some((l) => !isStr(l.name) || !isStr(l.blurb, 30))) at('needs ≥3 levels (outermost first) with name and blurb'); },
      sampling: () => { if (!isStr(s.prompt) || (s.tokens || []).length < 5 || s.tokens.some((t) => !isStr(t.t) || !isNum(t.logit))) at('needs prompt and ≥5 tokens {t, logit}'); else if (new Set(s.tokens.map((t) => t.t)).size !== s.tokens.length) at('token labels must be unique'); },
      chain: () => {
        const ss = s.sentences || [];
        if (ss.length < 2 || ss.some((x) => !isStr(x.label) || (x.tokens || []).length < 3 || x.tokens.some((t) => !isStr(t.t) || !(t.p > 0 && t.p <= 1)))) at('needs ≥2 sentences, each with label and ≥3 tokens {t, p} where 0 < p ≤ 1');
      },
      logbars: () => { if ((s.items || []).length < 4 || s.items.some((x) => !isStr(x.label) || !(x.value > 0))) at('needs ≥4 items {label, value > 0}'); },
      retrieval: () => {
        const d = s.docs || [];
        if (d.length < 4 || d.some((x) => !isStr(x.id) || !isStr(x.title) || !isStr(x.text, 30))) at('needs ≥4 docs {id,title,text}');
        (s.checks || []).forEach((c, k) => { const top = M.tfidfRank(d, c.query)[0]; if (top.d.id !== c.top) at(`check ${k + 1} ("${c.query}") ranks "${top.d.id}" first, not "${c.top}"`); });
        if (!(s.checks || []).length) at('add "checks": [{"query": "...", "top": "doc-id"}] so the ranking is verified');
      },
    };
    if (!T[s.type]) at(`unknown viz type "${s.type}"`);
    else T[s.type]();
  });
  if (hooks !== 1) err(`needs exactly one hook block, found ${hooks}`);
  if (figs < 3) err(`needs at least 3 figures (viz blocks other than the hook), found ${figs}`);
  if (quizzes !== 1) err(`needs exactly one quiz block, found ${quizzes}`);
  const types = blocks.filter((b) => b.kind === 'viz').map((b) => { try { return JSON.parse(b.text).type; } catch (e) { return '?'; } });
  if (new Set(types.filter((t) => t !== 'hook')).size < 2) warn('use at least two different figure types');

  return { errors, warns, figs, marks, terms: terms.length, rails: rails.length };
}

const files = fs.readdirSync(dir).filter((f) => /^week-\d\d\.md$/.test(f) && (!only.length || only.includes(f))).sort();
if (!files.length) {
  console.log('No lesson files found in', dir);
  process.exit(1);
}
for (const f of files) {
  const r = check(f);
  totalErrors += r.errors.length;
  console.log(`${r.errors.length ? '✗' : '✓'} ${f}  figures ${r.figs} · highlights ${r.marks} · terms ${r.terms} · rail ${r.rails}`);
  r.errors.forEach((e) => console.log('    ERROR  ' + e));
  r.warns.forEach((w) => console.log('    warn   ' + w));
}
console.log(totalErrors ? `\n${totalErrors} error(s)` : '\nAll lessons valid');
process.exit(totalErrors ? 1 : 0);
