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
    if (/\[\[[^\]\n]*\|/.test(b.text)) at('contains a [[term|definition]]: glossary terms are only supported in prose, not inside figure JSON');
    if (b.kind === 'quiz') {
      quizzes++;
      if (!/^[a-z0-9-]{1,34}$/.test(s.id || '')) at('id must match [a-z0-9-] and be at most 34 characters');
      if (!isNum(s.passMark)) at('passMark must be a number');
      const q = s.questions || [];
      if (s.exam) {
        if (q.length < 20 || q.length > 60) at(`an exam needs 20–60 questions, has ${q.length}`);
        if (!isNum(s.minutes) || s.minutes < 10 || s.minutes > 180) at('an exam needs "minutes" between 10 and 180');
        q.forEach((x, k) => { if (!isStr(x.topic, 3)) at(`exam question ${k + 1} needs a "topic" (used for the score breakdown)`); });
      } else if (q.length < 6 || q.length > 15) at(`needs 6–15 questions, has ${q.length}`);
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
      calc: () => { if (!['fc_vs_conv', 'lora', 'memory', 'scaling', 'split', 'ngram_space', 'attn_cost', 'vit_tokens'].includes(s.kind)) at('kind must be fc_vs_conv, lora, memory, scaling, split, ngram_space, attn_cost or vit_tokens'); },
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
      convgrad: () => {
        const sq = (m) => Array.isArray(m) && m.length >= 1 && m.every((r) => Array.isArray(r) && r.length === m[0].length && r.every((q) => typeof q === 'number'));
        if (!sq(s.input) || !sq(s.kernel) || !sq(s.upstream)) return at('input, kernel and upstream must be number grids');
        if (s.kernel.length !== s.kernel[0].length) return at('kernel must be square');
        const y = M.convolve(s.input, s.kernel, s.stride || 1, s.padding || 0);
        if (y.length !== s.upstream.length || y[0].length !== s.upstream[0].length) return at(`upstream must be ${y.length}×${y[0].length} (the output size)`);
        const g = M.convGrad(s.input, s.kernel, s.upstream, s.stride || 1, s.padding || 0);
        (s.checks || []).forEach((c, k) => {
          const got = c.kind === 'dx' ? g.dx[c.i][c.j] : c.kind === 'dK' ? g.dK[c.i][c.j] : c.kind === 'y' ? y[c.i][c.j] : null;
          if (got == null || Math.abs(got - c.value) > 0.005) at(`check ${k + 1} (${c.kind}): expected ${c.value}, computed ${got}`);
        });
      },
      filters: () => {
        if (!Array.isArray(s.image) || s.image.length < 6 || s.image.some((r) => !/^[0-9]+$/.test(r) || r.length !== s.image[0].length)) at('image must be ≥6 rows of equal-length digit strings');
        if (!Array.isArray(s.presets) || !s.presets.length || s.presets.some((p) => !isStr(p.name) || !Array.isArray(p.kernel) || p.kernel.length !== 3 || p.kernel.some((r) => r.length !== 3))) at('presets need {name, kernel 3×3}');
      },
      pool: () => {
        if (!Array.isArray(s.input) || s.input.length < 4 || s.input.some((r) => r.length !== s.input[0].length)) at('input must be a grid of at least 4 rows');
        else { const k = s.size || 2, st = s.stride || k; if ((s.input.length - k) % st || (s.input[0].length - k) % st) at('window and stride must tile the input exactly'); }
      },
      cnnstack: () => {
        if (!Array.isArray(s.input) || s.input.length !== 3 || !Array.isArray(s.layers) || s.layers.length < 3) return at('needs input [h,w,c] and ≥3 layers');
        const rows = M.convShapes(s.input, s.layers);
        if (rows.some((r) => !r.valid)) at('a layer produces an empty output');
        (s.checks || []).forEach((c, k) => {
          if (c.total != null) { if (c.total !== rows.reduce((q, z) => q + z.params, 0)) at(`check ${k + 1}: total expected ${c.total}`); if (c.layer == null) return; }
          const r = rows[c.layer];
          if (!r) return at(`check ${k + 1}: no layer ${c.layer}`);
          if (c.shape && c.shape.join() !== r.shape.join()) at(`check ${k + 1}: shape expected ${c.shape}, computed ${r.shape}`);
          if (c.params != null && c.params !== r.params) at(`check ${k + 1}: params expected ${c.params}, computed ${r.params}`);
          if (c.rf != null && c.rf !== r.rf) at(`check ${k + 1}: receptive field expected ${c.rf}, computed ${r.rf}`);
        });
      },
      vocabtrade: () => {
        if (!Array.isArray(s.corpus) || s.corpus.length < 4 || s.corpus.some((c) => !Array.isArray(c) || !/^[a-z]+$/.test(c[0]) || !(c[1] >= 1))) return at('corpus must be ≥4 [lowercase word, count] pairs');
        if (!isStr(s.text) || !/^[a-z ]+$/.test(s.text)) return at('text must be a lowercase sentence (letters and spaces)');
        (s.checks || []).forEach((c, k) => {
          const r = M.tokenStats(s.corpus, c.text != null ? c.text : s.text, c.merges);
          if (c.vocabSize != null && r.vocabSize !== c.vocabSize) at(`check ${k + 1}: vocabulary expected ${c.vocabSize}, computed ${r.vocabSize}`);
          if (c.textTokens != null && r.textTokens !== c.textTokens) at(`check ${k + 1}: text tokens expected ${c.textTokens}, computed ${r.textTokens}`);
          if (c.unk != null && r.wordLevel.unk !== c.unk) at(`check ${k + 1}: unknown words expected ${c.unk}, computed ${r.wordLevel.unk}`);
          if (c.wordVocab != null && r.wordLevel.vocabSize !== c.wordVocab) at(`check ${k + 1}: word vocabulary expected ${c.wordVocab}, computed ${r.wordLevel.vocabSize}`);
        });
      },
      vecsim: () => {
        if (!Array.isArray(s.words) || s.words.length < 3 || s.words.some((w) => !isStr(w.label) || !Array.isArray(w.v) || w.v.length < 2 || w.v.length !== s.words[0].v.length)) return at('words need ≥3 {label, v[]} with equal-length vectors');
        (s.checks || []).forEach((c, k) => {
          const a = s.words.find((w) => w.label === c.a), b = s.words.find((w) => w.label === c.b);
          if (!a || !b) return at(`check ${k + 1}: unknown word label`);
          const r = M.cosineSim(a.v, b.v);
          if (c.cos != null && Math.abs(r.cos - c.cos) > 0.005) at(`check ${k + 1}: cosine(${c.a}, ${c.b}) expected ${c.cos}, computed ${r.cos.toFixed(3)}`);
          if (c.dot != null && Math.abs(r.dot - c.dot) > 0.005) at(`check ${k + 1}: dot(${c.a}, ${c.b}) expected ${c.dot}, computed ${r.dot.toFixed(3)}`);
        });
      },
      seqmemory: () => {
        (s.checks || []).forEach((c, k) => {
          const r = M.rnnInfluence({ T: c.length, w: c.w, seed: s.seed || 3 }).influence[0];
          if (c.rnnFirstMax != null && r > c.rnnFirstMax) at(`check ${k + 1}: RNN influence of step 1 is ${r.toExponential(2)} > ${c.rnnFirstMax}`);
          if (c.rnnFirstMin != null && r < c.rnnFirstMin) at(`check ${k + 1}: RNN influence of step 1 is ${r.toExponential(2)} < ${c.rnnFirstMin}`);
          if (c.forget != null) { const l = M.lstmInfluence(c.length, c.forget)[0]; if (Math.abs(l - c.lstmFirst) > Math.max(1e-4, c.lstmFirst * 0.02)) at(`check ${k + 1}: LSTM influence is ${l.toExponential(3)}, not ${c.lstmFirst}`); }
        });
      },
      lstmcell: () => {
        const p = s.params;
        if (!p || ['f', 'i', 'g', 'o'].some((q) => !Array.isArray(p[q]) || p[q].length !== 3)) return at('params need f, i, g, o, each [w_x, w_h, bias]');
        (s.checks || []).forEach((c, k) => {
          const r = M.lstmStep(p, c.x, c.hPrev, c.cPrev);
          ['f', 'i', 'o', 'g', 'c', 'h'].forEach((q) => { if (c[q] != null && Math.abs(r[q] - c[q]) > 0.002) at(`check ${k + 1}: ${q} expected ${c[q]}, computed ${r[q].toFixed(4)}`); });
        });
      },
      hardsoft: () => {
        if (!Array.isArray(s.scores) || s.scores.length < 3 || !Array.isArray(s.labels) || s.labels.length !== s.scores.length || !Array.isArray(s.values) || s.values.length !== s.scores.length || s.values.some((v) => !Array.isArray(v) || v.length !== s.values[0].length)) return at('scores, labels and values need the same length (≥3); values are equal-length vectors');
        (s.checks || []).forEach((c, k) => {
          const r = M.softHard(s.scores, s.values, c.temperature || 1);
          (c.soft || []).forEach((q, i) => { if (Math.abs(r.soft[i] - q) > 0.002) at(`check ${k + 1}: soft weight ${i + 1} expected ${q}, computed ${r.soft[i].toFixed(4)}`); });
          (c.softOut || []).forEach((q, i) => { if (Math.abs(r.softOut[i] - q) > 0.002) at(`check ${k + 1}: soft output ${i + 1} expected ${q}, computed ${r.softOut[i].toFixed(4)}`); });
          (c.hardOut || []).forEach((q, i) => { if (Math.abs(r.hardOut[i] - q) > 0.002) at(`check ${k + 1}: hard output ${i + 1} expected ${q}, computed ${r.hardOut[i].toFixed(4)}`); });
        });
      },
      tfblock: () => {
        if (!(s.d % s.heads === 0)) at('d must be divisible by heads');
        if (![s.vocab, s.d, s.dff, s.layers, s.heads].every((q) => q >= 1)) at('vocab, d, dff, layers and heads are required');
        (s.checks || []).forEach((c, k) => {
          const P = M.tfParams({ vocab: c.vocab ?? s.vocab, d: c.d ?? s.d, dff: c.dff ?? s.dff, layers: c.layers ?? s.layers, ctx: c.ctx ?? s.ctx ?? 0, posLearned: c.posLearned ?? s.posLearned ?? true, tied: c.tied ?? s.tied ?? true });
          if (c.total != null && P.total !== c.total) at(`check ${k + 1}: total expected ${c.total}, computed ${P.total}`);
          if (c.block != null && P.block !== c.block) at(`check ${k + 1}: block expected ${c.block}, computed ${P.block}`);
          if (c.attn != null && P.attn !== c.attn) at(`check ${k + 1}: attention expected ${c.attn}, computed ${P.attn}`);
          if (c.ffn != null && P.ffn !== c.ffn) at(`check ${k + 1}: ffn expected ${c.ffn}, computed ${P.ffn}`);
        });
      },
      causalmask: () => {
        const n = (s.tokens || []).length;
        if (n < 3 || !Array.isArray(s.scores) || s.scores.length !== n || s.scores.some((r) => !Array.isArray(r) || r.length !== n)) return at('tokens (≥3) and an n×n scores matrix are required');
        (s.checks || []).forEach((c, k) => {
          const W = M.maskedSoftmax(s.scores, !!c.causal);
          if (Math.abs(W[c.row][c.col] - c.value) > 0.002) at(`check ${k + 1}: weight [${c.row}][${c.col}] expected ${c.value}, computed ${W[c.row][c.col].toFixed(4)}`);
        });
      },
      shiftloss: () => {
        if (!Array.isArray(s.tokens) || s.tokens.length < 4 || !Array.isArray(s.p) || s.p.length !== s.tokens.length - 1 || s.p.some((q) => !(q > 0 && q <= 1))) return at('tokens (≥4) and p (one probability in (0,1] per position, length tokens−1) are required');
        const mean = s.p.reduce((a, q) => a - Math.log(q), 0) / s.p.length;
        (s.checks || []).forEach((c, k) => {
          if (c.loss != null && Math.abs(mean - c.loss) > 0.002) at(`check ${k + 1}: mean loss expected ${c.loss}, computed ${mean.toFixed(4)}`);
          if (c.perplexity != null && Math.abs(Math.exp(mean) - c.perplexity) > 0.01) at(`check ${k + 1}: perplexity expected ${c.perplexity}, computed ${Math.exp(mean).toFixed(3)}`);
        });
      },
      llmfamily: () => {
        if (!Array.isArray(s.tokens) || s.tokens.length < 4 || s.tokens.length > 10) return at('tokens needs 4–10 items');
        if (!Array.isArray(s.families) || s.families.length < 2) return at('families needs ≥2 entries');
        s.families.forEach((f, k) => {
          if (!isStr(f.name) || !['bidirectional', 'causal', 'prefix'].includes(f.mask) || !isStr(f.maskName) || !isStr(f.objective) || !isStr(f.input) || !isStr(f.target) || !isStr(f.uses) || !isStr(f.examples)) at(`family ${k + 1} needs name, mask (bidirectional|causal|prefix), maskName, objective, input, target, uses, examples`);
        });
      },
      moe: () => {
        const sc = s.scores;
        if (!Array.isArray(s.tokens) || !Array.isArray(sc) || sc.length !== s.tokens.length || sc.some((r) => !Array.isArray(r) || r.length !== sc[0].length) || sc[0].length < 3) return at('tokens and a scores matrix (tokens × ≥3 experts) are required');
        (s.checks || []).forEach((c, k) => {
          const r = M.moeRoute(sc, c.k)[c.token];
          if (c.experts && r.map((q) => q.expert).join() !== c.experts.join()) at(`check ${k + 1}: experts expected ${c.experts}, computed ${r.map((q) => q.expert)}`);
          if (c.weights) c.weights.forEach((w, j) => { if (Math.abs(r[j].weight - w) > 0.002) at(`check ${k + 1}: weight ${j + 1} expected ${w}, computed ${r[j].weight.toFixed(4)}`); });
          if (c.total != null || c.active != null) { const P = M.moeParams(sc[0].length, c.k, s.d ?? 512, s.dff ?? 2048); if (c.total != null && P.total !== c.total) at(`check ${k + 1}: total expected ${c.total}, computed ${P.total}`); if (c.active != null && P.active !== c.active) at(`check ${k + 1}: active expected ${c.active}, computed ${P.active}`); }
        });
      },
      scalinglaw: () => {
        (s.checks || []).forEach((c, k) => {
          if (c.N != null && c.D != null && c.loss != null) { const L = M.chinLoss(c.N, c.D); if (Math.abs(L - c.loss) > 0.002) at(`check ${k + 1}: loss(N, D) expected ${c.loss}, computed ${L.toFixed(4)}`); }
          if (c.C != null) { const o = M.chinOptimal(c.C); if (c.optN != null && Math.abs(o.N / c.optN - 1) > 0.03) at(`check ${k + 1}: optimal N expected ${c.optN}, computed ${o.N.toExponential(3)}`); if (c.optRatio != null && Math.abs(o.ratio - c.optRatio) > 0.6) at(`check ${k + 1}: tokens per parameter expected ${c.optRatio}, computed ${o.ratio.toFixed(2)}`); }
        });
      },
      emergent: () => {
        (s.checks || []).forEach((c, k) => {
          const r = M.emergence(s.curve || {}, c.logN, c.length);
          if (c.p != null && Math.abs(r.p - c.p) > 0.005) at(`check ${k + 1}: p expected ${c.p}, computed ${r.p.toFixed(3)}`);
          if (c.em != null && Math.abs(r.em - c.em) > 0.005) at(`check ${k + 1}: exact match expected ${c.em}, computed ${r.em.toFixed(3)}`);
        });
      },
      clipsim: () => {
        const ok = (a) => Array.isArray(a) && a.length >= 2 && a.every((x) => isStr(x.label) && Array.isArray(x.v) && x.v.length >= 2 && x.v.length === a[0].v.length);
        if (!ok(s.images) || !ok(s.texts) || s.images[0].v.length !== s.texts[0].v.length) return at('images and texts need ≥2 {label, v[]} entries with vectors of one common length');
        (s.checks || []).forEach((c, k) => {
          const R = M.clipMatrix(s.images.map((x) => x.v), s.texts.map((x) => x.v), c.scale ?? s.scale ?? 10);
          if (c.cos) c.cos.forEach(([i, j, q]) => { if (Math.abs(R.cos[i][j] - q) > 0.005) at(`check ${k + 1}: cosine(image ${i}, text ${j}) expected ${q}, computed ${R.cos[i][j].toFixed(4)}`); });
          if (c.prob) c.prob.forEach(([i, j, q]) => { if (Math.abs(R.rows[i][j] - q) > 0.005) at(`check ${k + 1}: probability(image ${i}, text ${j}) expected ${q}, computed ${R.rows[i][j].toFixed(4)}`); });
          if (c.loss != null && Math.abs(R.loss - c.loss) > 0.005) at(`check ${k + 1}: loss expected ${c.loss}, computed ${R.loss.toFixed(4)}`);
        });
      },
      patchshuffle: () => {
        const n = (s.image || []).length;
        if (n < 8 || s.image.some((r) => !Array.isArray(r) || r.length !== n)) return at('image must be a square matrix of integers 0–9, at least 8×8');
        if (n % (s.patch ?? 4)) at('patch must divide the image side');
      },
      fusion: () => {
        if (!Array.isArray(s.patterns) || s.patterns.length < 2) return at('patterns needs ≥2 entries');
        s.patterns.forEach((p, k) => { if (!isStr(p.name) || !Array.isArray(p.inputs) || !Array.isArray(p.stages) || !isStr(p.output) || !isStr(p.how) || !isStr(p.examples) || !isStr(p.good) || !isStr(p.bad)) at(`pattern ${k + 1} needs name, inputs[], stages[], output, how, examples, good, bad`); });
      },
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
