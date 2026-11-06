// Pure maths behind the lesson figures (no React), shared by lessonViz.jsx and tools/validate-lessons.mjs.

export function mulberry32(a) {
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussNoise(n, seed = 7) {
  const r = mulberry32(seed);
  return Array.from({ length: n }, () => {
    const u = Math.max(r(), 1e-9);
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r());
  });
}

export const BYTES = { FP32: 4, 'FP16 / BF16': 2, INT8: 1, INT4: 0.5 };

export function convParams(h, cin, units, filters, k) {
  return { fc: h * h * cin * units + units, conv: k * k * cin * filters + filters };
}

export function loraParams(d, r) {
  return { full: d * d, lora: 2 * d * r, frac: (2 * d * r) / (d * d) };
}

export function chinchilla(logC) {
  const C = 10 ** logC;
  const N = Math.sqrt(C / 120); // C = 6 N D and D = 20 N  ->  N = sqrt(C / 120)
  return { C, N, D: 20 * N };
}

export const sig = (x) => 1 / (1 + Math.exp(-x));

export const FN = {
  sigmoid: { f: (x) => sig(x), label: 'Sigmoid' },
  tanh: { f: (x) => Math.tanh(x), label: 'Tanh' },
  relu: { f: (x) => Math.max(0, x), label: 'ReLU' },
  leaky_relu: { f: (x, p) => (x > 0 ? x : p * x), label: 'Leaky ReLU' },
  swish: { f: (x) => x * sig(x), label: 'Swish' },
  gelu: { f: (x) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3))), label: 'GELU' },
  sigmoid_grad: { f: (x) => sig(x) * (1 - sig(x)), label: 'Sigmoid gradient' },
  tanh_grad: { f: (x) => 1 - Math.tanh(x) ** 2, label: 'Tanh gradient' },
  relu_grad: { f: (x) => (x > 0 ? 1 : 0), label: 'ReLU gradient' },
  decay: { f: (x, p) => p ** x, label: 'Gradient after n layers' },
};

export const LOSS = {
  quad: { f: (w) => (w - 3) ** 2, g: (w) => 2 * (w - 3), range: [-1, 7], label: 'L(w) = (w − 3)²' },
  double_well: { f: (w) => w ** 4 / 4 - w ** 2 / 2 + 0.1 * w, g: (w) => w ** 3 - w + 0.1, range: [-2, 2], label: 'L(w) = w⁴/4 − w²/2 + 0.1w' },
};

export function neuronAccuracy(points, w1, w2, b) {
  const ok = points.filter((p) => (w1 * p.x + w2 * p.y + b > 0 ? 1 : 0) === p.c).length;
  return { ok, n: points.length };
}

export function convolve(inp, ker, stride = 1, pad = 0) {
  const n = inp.length, m = inp[0].length, k = ker.length;
  const get = (i, j) => (i < 0 || j < 0 || i >= n || j >= m ? 0 : inp[i][j]);
  const oh = Math.floor((n + 2 * pad - k) / stride) + 1, ow = Math.floor((m + 2 * pad - k) / stride) + 1;
  const out = Array.from({ length: oh }, () => Array(ow).fill(0));
  for (let i = 0; i < oh; i++) for (let j = 0; j < ow; j++) {
    let s = 0;
    for (let a = 0; a < k; a++) for (let c = 0; c < k; c++) s += get(i * stride - pad + a, j * stride - pad + c) * ker[a][c];
    out[i][j] = s;
  }
  return out;
}

export function transposeConv(inp, ker, stride = 2) {
  const n = inp.length, m = inp[0].length, k = ker.length;
  const oh = (n - 1) * stride + k, ow = (m - 1) * stride + k;
  const out = Array.from({ length: oh }, () => Array(ow).fill(0));
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) for (let a = 0; a < k; a++) for (let c = 0; c < k; c++) out[i * stride + a][j * stride + c] += inp[i][j] * ker[a][c];
  return out;
}

export function learnBPE(corpus, merges) {
  let words = corpus.map(([w, c]) => ({ t: [...w, '</w>'], c }));
  const learned = [];
  for (let m = 0; m < merges; m++) {
    const counts = new Map();
    words.forEach(({ t, c }) => { for (let i = 0; i < t.length - 1; i++) { const k = t[i] + '\u0001' + t[i + 1]; counts.set(k, (counts.get(k) || 0) + c); } });
    let best = null, bc = 0;
    counts.forEach((c, k) => { if (c > bc) { bc = c; best = k; } });
    if (!best || bc < 2) break;
    const [a, b] = best.split('\u0001');
    learned.push({ a, b, count: bc });
    words = words.map(({ t, c }) => {
      const out = [];
      for (let i = 0; i < t.length; i++) { if (i < t.length - 1 && t[i] === a && t[i + 1] === b) { out.push(a + b); i++; } else out.push(t[i]); }
      return { t: out, c };
    });
  }
  return { words, learned };
}

export function applyBPE(word, learned) {
  let t = [...word, '</w>'];
  learned.forEach(({ a, b }) => {
    const out = [];
    for (let i = 0; i < t.length; i++) { if (i < t.length - 1 && t[i] === a && t[i + 1] === b) { out.push(a + b); i++; } else out.push(t[i]); }
    t = out;
  });
  return t;
}

export function bigrams(text) {
  const w = text.toLowerCase().replace(/[^a-z' ]+/g, ' ').split(/\s+/).filter(Boolean);
  const m = new Map();
  for (let i = 0; i < w.length - 1; i++) { if (!m.has(w[i])) m.set(w[i], new Map()); const t = m.get(w[i]); t.set(w[i + 1], (t.get(w[i + 1]) || 0) + 1); }
  return m;
}

export const cosine = (a, b) => (a[0] * b[0] + a[1] * b[1]) / (Math.hypot(a[0], a[1]) * Math.hypot(b[0], b[1]) || 1);

export function nearest(points, vec, exclude, metric = 'cosine', k = 3) {
  return points.filter((p) => !exclude.includes(p.label)).map((p) => ({ p, s: metric === 'cosine' ? cosine([p.x, p.y], vec) : -Math.hypot(p.x - vec[0], p.y - vec[1]) })).sort((a, b) => b.s - a.s).slice(0, k);
}

export function posEnc(pos, d) {
  return Array.from({ length: d }, (_, i) => { const k = Math.floor(i / 2); const a = pos / 10000 ** ((2 * k) / d); return i % 2 === 0 ? Math.sin(a) : Math.cos(a); });
}

export function alphaBar(t, T = 1000, b0 = 1e-4, b1 = 0.02) {
  let ab = 1;
  for (let i = 1; i <= t; i++) ab *= 1 - (b0 + ((b1 - b0) * (i - 1)) / (T - 1));
  return ab;
}

export const STOP = new Set('a an the of and or to in on for is are was were be by with as at from that this it its into how what which who why when where do does can will'.split(' '));

export const tokens = (s) => s.toLowerCase().replace(/[^a-z0-9 ]+/g, ' ').split(/\s+/).filter((w) => w && !STOP.has(w));

export function tfidfRank(docs, query) {
  const toks = docs.map((d) => tokens(d.title + ' ' + d.text));
  const N = docs.length;
  const df = new Map();
  toks.forEach((t) => new Set(t).forEach((w) => df.set(w, (df.get(w) || 0) + 1)));
  const idf = (w) => Math.log((N + 1) / ((df.get(w) || 0) + 1)) + 1;
  const vec = (t) => { const m = new Map(); t.forEach((w) => m.set(w, (m.get(w) || 0) + 1)); m.forEach((c, w) => m.set(w, c * idf(w))); return m; };
  const norm = (m) => Math.sqrt([...m.values()].reduce((a, v) => a + v * v, 0)) || 1;
  const q = vec(tokens(query)), qn = norm(q);
  return docs.map((d, i) => { const v = vec(toks[i]); let dot = 0; q.forEach((c, w) => { dot += c * (v.get(w) || 0); }); return { d, score: dot / (qn * norm(v)), hits: [...q.keys()].filter((w) => v.has(w)) }; }).sort((a, b) => b.score - a.score);
}


// Scaled dot-product attention for one query: scores q·kᵢ (÷ √d when scale), softmax weights α, output Σ αᵢ vᵢ.
export function attend(q, K, V, scale = true) {
  const d = q.length;
  const raw = K.map((k) => k.reduce((s, x, j) => s + x * q[j], 0));
  const scores = raw.map((s) => (scale ? s / Math.sqrt(d) : s));
  const m = Math.max(...scores);
  const e = scores.map((s) => Math.exp(s - m));
  const z = e.reduce((a, b) => a + b, 0);
  const alpha = e.map((x) => x / z);
  const out = V[0].map((_, j) => V.reduce((s, v, i) => s + alpha[i] * v[j], 0));
  return { raw, scores, alpha, out };
}
// ---- sampling (temperature, top-k, top-p) ----
export function softmaxT(logits, T) {
  const m = Math.max(...logits);
  const ex = logits.map((l) => Math.exp((l - m) / T));
  const s = ex.reduce((a, b) => a + b, 0);
  return ex.map((x) => x / s);
}
// Keep the topK most likely tokens (0 = off), then the smallest set whose probability adds up to at least topP (1 = off).
// Returns { keep: boolean[], probs: renormalised probabilities (0 for removed tokens) }.
export function filterSampling(probs, topK = 0, topP = 1) {
  const order = probs.map((p, i) => i).sort((a, b) => probs[b] - probs[a]);
  const keep = probs.map(() => false);
  let cum = 0;
  for (let r = 0; r < order.length; r++) {
    if (topK > 0 && r >= topK) break;
    keep[order[r]] = true;
    cum += probs[order[r]];
    if (topP < 1 && cum >= topP - 1e-12) break;
  }
  const z = probs.reduce((a, p, i) => a + (keep[i] ? p : 0), 0);
  return { keep, probs: probs.map((p, i) => (keep[i] ? p / z : 0)) };
}
// ---- chain rule: P(w1..wn) = prod P(wi | w<i) ----
export function chainStats(ps) {
  const logp = ps.reduce((a, p) => a + Math.log(p), 0);
  return { P: Math.exp(logp), bits: -logp / Math.LN2, perplexity: Math.exp(-logp / ps.length), running: ps.map((_, i) => ps.slice(0, i + 1).reduce((a, p) => a * p, 1)) };
}
