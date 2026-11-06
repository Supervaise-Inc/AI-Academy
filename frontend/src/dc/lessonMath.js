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
  neg_log: { f: (x) => -Math.log(Math.max(x, 1e-6)), label: 'Cross-entropy loss −ln p' },
  squared: { f: (x) => x * x, label: 'Squared error' },
  abs: { f: (x) => Math.abs(x), label: 'Absolute error' },
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

// ---- a small fully connected network, forward pass ----
export const ACT = { relu: (x) => Math.max(0, x), tanh: Math.tanh, sigmoid: (x) => 1 / (1 + Math.exp(-x)), linear: (x) => x, step: (x) => (x > 0 ? 1 : 0) };
// layers: [2, 3, 1]; W[l][j][i] = weight from unit i of layer l to unit j of layer l+1; B[l][j] = bias of that unit.
export function forwardNet(layers, W, B, inputs, act = 'relu', outAct = 'linear') {
  const a = [inputs.slice()];
  const z = [null];
  for (let l = 0; l < layers.length - 1; l++) {
    const f = ACT[l === layers.length - 2 ? outAct : act];
    const zl = W[l].map((row, j) => row.reduce((s, w, i) => s + w * a[l][i], 0) + B[l][j]);
    z.push(zl);
    a.push(zl.map(f));
  }
  return { a, z };
}
export const countParams = (layers) => layers.slice(1).reduce((s, n, i) => s + layers[i] * n + n, 0);

// ---- polynomial least squares (x should lie in [-1, 1]) ----
export function polyFit(xs, ys, deg) {
  const n = deg + 1;
  const A = Array.from({ length: n }, () => Array(n).fill(0));
  const b = Array(n).fill(0);
  xs.forEach((x, k) => {
    const pw = [1];
    for (let i = 1; i < 2 * n; i++) pw.push(pw[i - 1] * x);
    for (let i = 0; i < n; i++) {
      b[i] += pw[i] * ys[k];
      for (let j = 0; j < n; j++) A[i][j] += pw[i + j];
    }
  });
  for (let i = 0; i < n; i++) A[i][i] += 1e-9;
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]];
    [b[c], b[p]] = [b[p], b[c]];
    for (let r = c + 1; r < n; r++) {
      const f = A[r][c] / A[c][c];
      for (let k = c; k < n; k++) A[r][k] -= f * A[c][k];
      b[r] -= f * b[c];
    }
  }
  const coef = Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = b[i];
    for (let j = i + 1; j < n; j++) s -= A[i][j] * coef[j];
    coef[i] = s / A[i][i];
  }
  return coef;
}
export const polyEval = (c, x) => c.reduceRight((a, ci) => a * x + ci, 0);
export const polyMse = (c, pts) => pts.reduce((s, p) => s + (polyEval(c, p.x) - p.y) ** 2, 0) / pts.length;

// ---- tiny trainable network: 2 inputs -> H tanh units -> 1 sigmoid output, binary cross-entropy, full-batch gradient descent ----
export function makeDataset(name, n = 60, seed = 3) {
  const r = mulberry32(seed);
  const g = () => { const u = Math.max(r(), 1e-9); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); };
  const X = [];
  const Y = [];
  for (let i = 0; i < n; i++) {
    if (name === 'xor') {
      const sx = i % 2 ? 1 : -1;
      const sy = (i >> 1) % 2 ? 1 : -1;
      X.push([sx + 0.28 * g(), sy + 0.28 * g()]);
      Y.push(sx * sy > 0 ? 1 : 0);
    } else if (name === 'circle') {
      const inner = i % 2 === 0;
      const rad = inner ? 0.55 * Math.sqrt(r()) : 1.0 + 0.5 * r();
      const t = 2 * Math.PI * r();
      X.push([rad * Math.cos(t) + 0.05 * g(), rad * Math.sin(t) + 0.05 * g()]);
      Y.push(inner ? 1 : 0);
    } else if (name === 'moons') {
      const up = i % 2 === 0;
      const t = Math.PI * r();
      X.push(up ? [Math.cos(t) - 0.5 + 0.1 * g(), Math.sin(t) - 0.25 + 0.1 * g()] : [0.5 - Math.cos(t) + 0.1 * g(), 0.25 - Math.sin(t) + 0.1 * g()]);
      Y.push(up ? 1 : 0);
    } else {
      const x = 2.4 * r() - 1.2;
      const y = 2.4 * r() - 1.2;
      X.push([x, y]);
      Y.push(x + y + 0.15 * g() > 0 ? 1 : 0);
    }
  }
  return { X, Y };
}
export function initNet(H, seed = 1) {
  const r = mulberry32(seed * 7919 + H);
  const g = () => { const u = Math.max(r(), 1e-9); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); };
  return { H, W1: Array.from({ length: H }, () => [0.7 * g(), 0.7 * g()]), b1: Array(H).fill(0), w2: Array.from({ length: H }, () => 0.7 * g()), b2: 0 };
}
export function predictNet(net, x) {
  const h = net.W1.map((w, j) => Math.tanh(w[0] * x[0] + w[1] * x[1] + net.b1[j]));
  const z = h.reduce((s, v, j) => s + v * net.w2[j], net.b2);
  return 1 / (1 + Math.exp(-z));
}
export function netLossAcc(net, X, Y) {
  let L = 0;
  let ok = 0;
  X.forEach((x, i) => {
    const p = Math.min(Math.max(predictNet(net, x), 1e-9), 1 - 1e-9);
    L += -(Y[i] * Math.log(p) + (1 - Y[i]) * Math.log(1 - p));
    if ((p > 0.5 ? 1 : 0) === Y[i]) ok++;
  });
  return { loss: L / X.length, acc: ok / X.length };
}
export function trainEpoch(net, X, Y, lr) {
  const N = X.length;
  const H = net.H;
  const gW1 = Array.from({ length: H }, () => [0, 0]);
  const gb1 = Array(H).fill(0);
  const gw2 = Array(H).fill(0);
  let gb2 = 0;
  X.forEach((x, i) => {
    const h = net.W1.map((w, j) => Math.tanh(w[0] * x[0] + w[1] * x[1] + net.b1[j]));
    const z = h.reduce((s, v, j) => s + v * net.w2[j], net.b2);
    const dz2 = 1 / (1 + Math.exp(-z)) - Y[i];
    gb2 += dz2;
    for (let j = 0; j < H; j++) {
      gw2[j] += dz2 * h[j];
      const dz1 = dz2 * net.w2[j] * (1 - h[j] * h[j]);
      gb1[j] += dz1;
      gW1[j][0] += dz1 * x[0];
      gW1[j][1] += dz1 * x[1];
    }
  });
  for (let j = 0; j < H; j++) {
    net.W1[j][0] -= (lr * gW1[j][0]) / N;
    net.W1[j][1] -= (lr * gW1[j][1]) / N;
    net.b1[j] -= (lr * gb1[j]) / N;
    net.w2[j] -= (lr * gw2[j]) / N;
  }
  net.b2 -= (lr * gb2) / N;
}
export function trainNet({ dataset = 'xor', hidden = 4, lr = 0.5, epochs = 1000, seed = 1, n = 60 }) {
  const { X, Y } = makeDataset(dataset, n);
  const net = initNet(hidden, seed);
  for (let e = 0; e < epochs; e++) trainEpoch(net, X, Y, lr);
  return netLossAcc(net, X, Y);
}

// ---- Week 3: backpropagation and gradient flow
// derivative of each activation written in terms of z and a = f(z)
export const DACT = { relu: (z) => (z > 0 ? 1 : 0), tanh: (z, a) => 1 - a * a, sigmoid: (z, a) => a * (1 - a), linear: () => 1 };

// Forward and backward pass. Loss is 'mse' (L = 1/2 * sum (a - y)^2) or 'bce' (needs a sigmoid output; L = -sum [y ln a + (1-y) ln(1-a)]).
// delta[l][j] = dL/dz of unit j in layer l (layers 1..L). dW[l][j][i] = delta[l+1][j] * a[l][i]; dB[l][j] = delta[l+1][j].
export function backwardNet(layers, W, B, inputs, target, act = 'sigmoid', outAct = 'sigmoid', loss = 'mse') {
  const { a, z } = forwardNet(layers, W, B, inputs, act, outAct);
  const L = layers.length - 1;
  const out = a[L];
  let lossV = 0;
  if (loss === 'bce') lossV = out.reduce((s, p, j) => s - (target[j] * Math.log(p) + (1 - target[j]) * Math.log(1 - p)), 0);
  else lossV = out.reduce((s, p, j) => s + 0.5 * (p - target[j]) ** 2, 0);
  const delta = Array(L + 1).fill(null);
  delta[L] = out.map((p, j) => (loss === 'bce' && outAct === 'sigmoid' ? p - target[j] : (p - target[j]) * DACT[outAct](z[L][j], p)));
  for (let l = L - 1; l >= 1; l--) delta[l] = Array.from({ length: layers[l] }, (_, i) => {
    let s = 0;
    for (let j = 0; j < layers[l + 1]; j++) s += W[l][j][i] * delta[l + 1][j];
    return s * DACT[act](z[l][i], a[l][i]);
  });
  const dW = W.map((m, l) => m.map((row, j) => row.map((_, i) => delta[l + 1][j] * a[l][i])));
  const dB = B.map((r, l) => r.map((_, j) => delta[l + 1][j]));
  return { a, z, loss: lossV, delta, dW, dB };
}

// Gradient size per layer of a random deep network (width units per layer, `depth` weight layers, biases 0).
// Weights ~ N(0, (scale^2 / width)); the loss is the sum of the outputs, so the gradient entering the last layer is 1.
// Returns the Frobenius norm of dL/dW for each weight layer, from layer 1 (nearest the input) to layer `depth`.
export function gradNorms({ depth = 10, width = 16, act = 'sigmoid', scale = 1, seed = 5 }) {
  const r = mulberry32(seed);
  const gauss = () => { let u = 0, v = 0; while (u === 0) u = r(); while (v === 0) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  const layers = [width, ...Array(depth - 1).fill(width), 1];
  const W = layers.slice(1).map((n, l) => Array.from({ length: n }, () => Array.from({ length: layers[l] }, () => (gauss() * scale) / Math.sqrt(layers[l]))));
  const B = layers.slice(1).map((n) => Array(n).fill(0));
  const x = Array.from({ length: width }, () => gauss());
  const { a, z } = forwardNet(layers, W, B, x, act, 'linear');
  const L = layers.length - 1;
  let delta = [1];
  const norms = Array(L);
  for (let l = L; l >= 1; l--) {
    let s = 0;
    for (let j = 0; j < layers[l]; j++) for (let i = 0; i < layers[l - 1]; i++) s += (delta[j] * a[l - 1][i]) ** 2;
    norms[l - 1] = Math.sqrt(s);
    if (l > 1) delta = Array.from({ length: layers[l - 1] }, (_, i) => {
      let t = 0;
      for (let j = 0; j < layers[l]; j++) t += W[l - 1][j][i] * delta[j];
      return t * DACT[act](z[l - 1][i], a[l - 1][i]);
    });
  }
  return norms;
}

// ---- Week 4: convolutional networks
export const outSize = (n, k, s = 1, p = 0) => Math.floor((n + 2 * p - k) / s) + 1;

// Gradients of y = conv(x, K) (cross-correlation, as in deep learning) given G = dL/dy.
// dx[i][j] sums G[p][q] * K[a][c] over every output cell (p,q) and kernel cell (a,c) that touched x[i][j]; dK[a][c] sums G[p][q] * x[...].
export function convGrad(inp, ker, G, stride = 1, pad = 0) {
  const n = inp.length, m = inp[0].length, k = ker.length;
  const dx = inp.map((r) => r.map(() => 0));
  const dK = ker.map((r) => r.map(() => 0));
  for (let p = 0; p < G.length; p++) for (let q = 0; q < G[0].length; q++) for (let a = 0; a < k; a++) for (let c = 0; c < k; c++) {
    const i = p * stride - pad + a, j = q * stride - pad + c;
    if (i < 0 || j < 0 || i >= n || j >= m) continue;
    dx[i][j] += G[p][q] * ker[a][c];
    dK[a][c] += G[p][q] * inp[i][j];
  }
  return { dx, dK };
}

export function pool2d(inp, size = 2, stride = 2, mode = 'max') {
  const oh = outSize(inp.length, size, stride), ow = outSize(inp[0].length, size, stride);
  return Array.from({ length: oh }, (_, i) => Array.from({ length: ow }, (_, j) => {
    const vals = [];
    for (let a = 0; a < size; a++) for (let c = 0; c < size; c++) vals.push(inp[i * stride + a][j * stride + c]);
    return mode === 'avg' ? vals.reduce((s, v) => s + v, 0) / vals.length : Math.max(...vals);
  }));
}

// layers: {type:'conv',k,filters,stride,pad} | {type:'pool',k,stride} | {type:'flatten'} | {type:'fc',units}
// Returns one row per layer with output shape [h,w,c], parameter count, receptive field (of one unit, in input pixels) and jump.
export function convShapes(input, layers) {
  let [h, w, c] = input;
  let rf = 1, jump = 1;
  let flat = null;
  return layers.map((L) => {
    let params = 0;
    if (L.type === 'conv') {
      const s = L.stride || 1, p = L.pad || 0;
      params = L.k * L.k * c * L.filters + L.filters;
      rf += (L.k - 1) * jump; jump *= s;
      h = outSize(h, L.k, s, p); w = outSize(w, L.k, s, p); c = L.filters;
    } else if (L.type === 'pool') {
      const s = L.stride || L.k;
      rf += (L.k - 1) * jump; jump *= s;
      h = outSize(h, L.k, s); w = outSize(w, L.k, s);
    } else if (L.type === 'convT') {
      const s = L.stride || 1, p = L.pad || 0;
      params = L.k * L.k * c * L.filters + L.filters;
      rf = null; jump = 1;
      h = outSizeT(h, L.k, s, p, L.outputPadding || 0); w = outSizeT(w, L.k, s, p, L.outputPadding || 0); c = L.filters;
    } else if (L.type === 'reshape') {
      const tot = L.shape[0] * L.shape[1] * L.shape[2];
      h = L.shape[0]; w = L.shape[1]; c = L.shape[2]; flat = null;
      params = 0; rf = null; if (tot !== (L._from || tot)) params = 0;
    } else if (L.type === 'flatten') {
      flat = h * w * c; h = 1; w = 1; c = flat;
    } else if (L.type === 'fc') {
      const inN = flat != null ? flat : h * w * c;
      params = inN * L.units + L.units;
      flat = L.units; h = 1; w = 1; c = L.units;
    }
    return { type: L.type, shape: [h, w, c], params, rf: L.type === 'fc' || L.type === 'flatten' || L.type === 'convT' || L.type === 'reshape' ? null : rf, valid: h >= 1 && w >= 1 };
  });
}

// ---- Week 6: tokens, vocabularies and embeddings
// For a corpus of [word, count] pairs and a sample text: how big the vocabulary is and how many tokens the text needs when words are
// split into characters (0 merges), into BPE pieces (m merges), or kept whole (word level; unseen words become <unk>).
export function tokenStats(corpus, text, merges) {
  const { words, learned } = learnBPE(corpus, merges);
  const vocab = new Set();
  words.forEach((w) => w.t.forEach((t) => vocab.add(t)));
  const sample = text.toLowerCase().split(/\s+/).filter(Boolean);
  const pieces = sample.map((w) => applyBPE(w, learned));
  const known = new Set(corpus.map(([w]) => w));
  const totalCount = corpus.reduce((s, [, c]) => s + c, 0);
  const corpusTokens = words.reduce((s, w) => s + w.t.length * w.c, 0);
  return {
    vocabSize: vocab.size,
    learned,
    pieces,
    textTokens: pieces.reduce((s, p) => s + p.length, 0),
    wordLevel: { vocabSize: known.size, textTokens: sample.length, unk: sample.filter((w) => !known.has(w)).length },
    charLevel: { vocabSize: new Set(corpus.flatMap(([w]) => [...w]).concat(['</w>'])).size, textTokens: sample.reduce((s, w) => s + w.length + 1, 0) },
    tokensPerWord: corpusTokens / totalCount,
  };
}
export function cosineSim(a, b) {
  const dot = a.reduce((s, x, i) => s + x * b[i], 0);
  const na = Math.sqrt(a.reduce((s, x) => s + x * x, 0)), nb = Math.sqrt(b.reduce((s, x) => s + x * x, 0));
  return { dot, na, nb, cos: na && nb ? dot / (na * nb) : 0 };
}

// ---- Week 7: sequences, LSTM and attention
// One-unit RNN h_t = tanh(w*h_(t-1) + u*x_t) on a fixed pseudo-random input. influence[t-1] = d h_T / d h_t = product over k = t+1..T of w*(1 - h_k^2):
// how much a change at step t still matters at the last step T. influence[T-1] = 1.
export function rnnInfluence({ T = 20, w = 0.9, u = 0.8, seed = 3 }) {
  const r = mulberry32(seed);
  let h = 0;
  const hs = [];
  for (let t = 0; t < T; t++) { const x = r() < 0.5 ? -1 : 1; h = Math.tanh(w * h + u * x); hs.push(h); }
  const infl = Array(T).fill(1);
  for (let t = T - 2; t >= 0; t--) infl[t] = infl[t + 1] * w * (1 - hs[t + 1] * hs[t + 1]);
  return { influence: infl, states: hs };
}
// LSTM cell state path: c_t = f_t * c_(t-1) + ...  so d c_T / d c_t = product of the forget gates (constant f here).
export const lstmInfluence = (T, f) => Array.from({ length: T }, (_, t) => Math.pow(f, T - 1 - t));
// Scalar LSTM cell. p = { f:[wx,wh,b], i:[…], g:[…], o:[…] }
export function lstmStep(p, x, hPrev, cPrev) {
  const sig = (z) => 1 / (1 + Math.exp(-z));
  const lin = (q) => q[0] * x + q[1] * hPrev + q[2];
  const f = sig(lin(p.f)), i = sig(lin(p.i)), g = Math.tanh(lin(p.g)), o = sig(lin(p.o));
  const c = f * cPrev + i * g;
  const h = o * Math.tanh(c);
  return { f, i, g, o, c, h };
}
// Soft attention over scores: weights = softmax(scores / T); output = weighted sum of values. Hard attention: one-hot at the arg max.
export function softHard(scores, values, temp = 1) {
  const z = scores.map((s) => s / temp);
  const m = Math.max(...z);
  const e = z.map((q) => Math.exp(q - m));
  const s = e.reduce((a, b) => a + b, 0);
  const soft = e.map((q) => q / s);
  const arg = scores.indexOf(Math.max(...scores));
  const hard = scores.map((_, i) => (i === arg ? 1 : 0));
  const mix = (w) => values[0].map((_, d) => w.reduce((a, wi, i) => a + wi * values[i][d], 0));
  return { soft, hard, softOut: mix(soft), hardOut: mix(hard), arg };
}

// ---- Week 8: the Transformer block and autoregressive training
// Parameter counts of a decoder-only Transformer. posLearned: n_ctx*d learned position vectors (0 for sinusoidal). tied: the output layer reuses the embedding matrix.
// Biases are counted (as in GPT-2): attention Q,K,V,O each d*d + d; FFN d*dff + dff + dff*d + d; two LayerNorms per block (2*d each) and one final LayerNorm.
export function tfParams({ vocab, d, dff, layers, ctx = 0, posLearned = false, tied = true, bias = true }) {
  const b = bias ? 1 : 0;
  const emb = vocab * d;
  const pos = posLearned ? ctx * d : 0;
  const attn = 4 * d * d + 4 * d * b;
  const ffn = 2 * d * dff + (dff + d) * b;
  const ln = 2 * (2 * d);
  const block = attn + ffn + ln;
  const finalLn = 2 * d;
  const head = tied ? 0 : vocab * d;
  return { emb, pos, attn, ffn, ln, block, blocks: block * layers, finalLn, head, total: emb + pos + block * layers + finalLn + head };
}
// Row-wise softmax of a score matrix; with causal = true, entries above the diagonal are masked out (set to -infinity first).
export function maskedSoftmax(S, causal = false) {
  return S.map((row, i) => {
    const z = row.map((v, j) => (causal && j > i ? -Infinity : v));
    const m = Math.max(...z);
    const e = z.map((v) => (v === -Infinity ? 0 : Math.exp(v - m)));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  });
}

// ---- Week 9: LLM families, scaling laws, mixture of experts, emergent abilities
// Parametric loss of the Chinchilla form L(N, D) = E + A / N^alpha + B / D^beta, with the constants of the re-fit by Besiroglu et al. (2024).
// (The constants printed in the original paper for this form give a compute-optimal ratio far above 20 tokens per parameter, which does not match the paper's own headline result.)
export const CHIN = { E: 1.8172, A: 482.01, B: 2085.43, alpha: 0.3478, beta: 0.3658 };
export const chinLoss = (N, D) => CHIN.E + CHIN.A / Math.pow(N, CHIN.alpha) + CHIN.B / Math.pow(D, CHIN.beta);
// For a compute budget C = 6 N D, the N that minimises chinLoss (ternary search on log N); returns {N, D, loss, ratio}
export function chinOptimal(C) {
  let lo = Math.log(1e6), hi = Math.log(C / 6 / 1e6);
  for (let i = 0; i < 200; i++) {
    const m1 = lo + (hi - lo) / 3, m2 = hi - (hi - lo) / 3;
    const f = (x) => chinLoss(Math.exp(x), C / 6 / Math.exp(x));
    if (f(m1) < f(m2)) hi = m2; else lo = m1;
  }
  const N = Math.exp((lo + hi) / 2), D = C / 6 / N;
  return { N, D, loss: chinLoss(N, D), ratio: D / N };
}
// Mixture of experts: for every token keep the k highest router scores and soften them with a softmax over the kept scores.
export function moeRoute(scores, k) {
  return scores.map((row) => {
    const idx = row.map((s, i) => [s, i]).sort((a, b) => b[0] - a[0] || a[1] - b[1]).slice(0, k);
    const m = Math.max(...idx.map((q) => q[0]));
    const e = idx.map((q) => Math.exp(q[0] - m));
    const sum = e.reduce((a, b) => a + b, 0);
    return idx.map((q, j) => ({ expert: q[1], score: q[0], weight: e[j] / sum }));
  });
}
export const moeParams = (E, k, d, dff) => { const one = 2 * d * dff + dff + d; return { one, total: E * one, active: k * one }; };
// Per-token accuracy p grows smoothly with scale (a logistic in log10 of the parameter count); exact match on an answer of L tokens is p^L.
export function emergence({ pMin = 0.3, pMax = 0.99, mid = 9.5, width = 0.9 }, logN, L) {
  const p = pMin + (pMax - pMin) / (1 + Math.exp(-(logN - mid) / width));
  return { p, em: Math.pow(p, L) };
}

// ---- Week 11: multimodal models, Vision Transformers and contrastive learning
// A square image of side `img` with `c` channels cut into patch x patch squares: the token counts and sizes of a ViT.
export function vitTokens({ img = 224, patch = 16, c = 3, d = 768, cls = true }) {
  const side = Math.floor(img / patch);
  const patches = side * side;
  const patchDim = patch * patch * c;
  const seq = patches + (cls ? 1 : 0);
  return { side, patches, patchDim, seq, projParams: patchDim * d + d, attn: seq * seq, pixelSeq: img * img, pixelAttn: img * img * img * img, exact: img % patch === 0 };
}
// CLIP-style similarity: logits S_ij = scale * cos(image_i, text_j); softmax over texts for each image (rows) and over images for each text (columns);
// the symmetric contrastive loss is the mean of -ln(diagonal probabilities) over both directions.
export function clipMatrix(images, texts, scale = 10) {
  const cos = images.map((a) => texts.map((b) => cosineSim(a, b).cos));
  const logits = cos.map((r) => r.map((q) => q * scale));
  const sm = (row) => { const m = Math.max(...row); const e = row.map((q) => Math.exp(q - m)); const s = e.reduce((x, y) => x + y, 0); return e.map((q) => q / s); };
  const rows = logits.map(sm);
  const cols = texts.map((_, j) => sm(logits.map((r) => r[j]))); // cols[j][i]
  const n = Math.min(images.length, texts.length);
  let li = 0, lt = 0;
  for (let i = 0; i < n; i++) { li -= Math.log(rows[i][i]); lt -= Math.log(cols[i][i]); }
  return { cos, logits, rows, cols, lossImage: li / n, lossText: lt / n, loss: (li + lt) / (2 * n) };
}
// A fixed pseudo-random permutation of the patches of a g x g grid (Fisher-Yates with mulberry32).
export function patchPermutation(g, seed = 4) {
  const r = mulberry32(seed);
  const p = Array.from({ length: g * g }, (_, i) => i);
  for (let i = p.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [p[i], p[j]] = [p[j], p[i]]; }
  return p;
}

// ---- Week 12: GANs and transpose convolution
// A one-dimensional GAN. Real data ~ N(mu, sigma^2). Generator G(z) = a*z + b with z ~ N(0,1) (a learnable scale and shift).
// Discriminator D(x) = sigmoid(w1*x + w2*x^2 + c). Alternating gradient steps with batch size `batch`, learning rate `lr` for both.
export const sigmoid = (u) => 1 / (1 + Math.exp(-u));
export function ganInit() { return { a: 0.5, b: 0, w1: 0, w2: 0, c: 0, step: 0 }; }
function gaussStream(seed) {
  const r = mulberry32(seed);
  return () => { let u = 0, v = 0; while (u === 0) u = r(); while (v === 0) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
}
// One alternating step (one D step then one G step). `nonSat` chooses the generator loss: true = -ln D(G(z)), false = ln(1 - D(G(z))).
export function ganStep(s, { mu = 4, sigma = 1, lr = 0.05, lrG = null, batch = 64, nonSat = true, seed = 11, dSteps = 1 } = {}) {
  const lg = lrG == null ? lr : lrG;
  const g = gaussStream(seed * 7919 + s.step);
  const n = batch;
  const out = { ...s };
  let nd = 0;
  // --- discriminator: dSteps gradient steps on fresh real and fake samples
  for (let k = 0; k < dSteps; k++) {
    const xr = Array.from({ length: n }, () => mu + sigma * g());
    const zz = Array.from({ length: n }, () => g());
    const xf0 = zz.map((q) => out.a * q + out.b);
    let gw1 = 0, gw2 = 0, gc = 0, lossD = 0;
    xr.forEach((x) => { const u = out.w1 * x + out.w2 * x * x + out.c; const p = sigmoid(u); lossD -= Math.log(p + 1e-12); const d = p - 1; gw1 += d * x; gw2 += d * x * x; gc += d; });
    xf0.forEach((x) => { const u = out.w1 * x + out.w2 * x * x + out.c; const p = sigmoid(u); lossD -= Math.log(1 - p + 1e-12); const d = p; gw1 += d * x; gw2 += d * x * x; gc += d; });
    nd = lossD / (2 * n);
    out.w1 -= (lr * gw1) / (2 * n); out.w2 -= (lr * gw2) / (2 * n); out.c -= (lr * gc) / (2 * n);
  }
  // --- generator step (against the updated discriminator)
  const z = Array.from({ length: n }, () => g());
  let ga = 0, gb = 0, lossG = 0;
  z.forEach((q) => {
    const x = out.a * q + out.b;
    const u = out.w1 * x + out.w2 * x * x + out.c; const p = sigmoid(u);
    const dLdu = nonSat ? p - 1 : -p;
    lossG += nonSat ? -Math.log(p + 1e-12) : Math.log(1 - p + 1e-12);
    const dudx = out.w1 + 2 * out.w2 * x;
    ga += dLdu * dudx * q; gb += dLdu * dudx;
  });
  out.a -= (lg * ga) / n; out.b -= (lg * gb) / n;
  out.step = s.step + 1;
  return { state: out, lossD: nd, lossG: lossG / n };
}
export function ganRun(steps, opts = {}) {
  let s = ganInit(), last = null;
  for (let i = 0; i < steps; i++) { last = ganStep(s, opts); s = last.state; }
  return { state: s, lossD: last ? last.lossD : null, lossG: last ? last.lossG : null };
}
// Output size of a transpose convolution (PyTorch convention): (n - 1) * stride - 2 * pad + kernel + outputPadding
export const outSizeT = (n, k, s = 1, p = 0, op = 0) => (n - 1) * s - 2 * p + k + op;

// Upsampling a small grid by 2: nearest neighbour, bilinear (PyTorch's align_corners=False) or a transpose convolution (kernel k, stride 2, padding p).
export function upsampleNearest(g) { const n = g.length, m = g[0].length; return Array.from({ length: 2 * n }, (_, i) => Array.from({ length: 2 * m }, (_, j) => g[i >> 1][j >> 1])); }
export function upsampleBilinear(g) {
  const n = g.length, m = g[0].length;
  const src = (i, size) => Math.min(size - 1, Math.max(0, (i + 0.5) / 2 - 0.5));
  const at = (a, b) => g[Math.min(n - 1, Math.max(0, a))][Math.min(m - 1, Math.max(0, b))];
  return Array.from({ length: 2 * n }, (_, i) => Array.from({ length: 2 * m }, (_, j) => {
    const y = src(i, n), x = src(j, m), y0 = Math.floor(y), x0 = Math.floor(x), fy = y - y0, fx = x - x0;
    return at(y0, x0) * (1 - fy) * (1 - fx) + at(y0, x0 + 1) * (1 - fy) * fx + at(y0 + 1, x0) * fy * (1 - fx) + at(y0 + 1, x0 + 1) * fy * fx;
  }));
}
export function transposeConvPad(inp, ker, stride = 2, pad = 0) {
  const full = transposeConv(inp, ker, stride);
  const oh = full.length - 2 * pad, ow = full[0].length - 2 * pad;
  return Array.from({ length: oh }, (_, i) => Array.from({ length: ow }, (_, j) => full[i + pad][j + pad]));
}

// ---- Week 13: diffusion models
// beta schedule and cumulative signal products. kind 'linear': beta_t from b0 to b1 over T steps; 'cosine' (Nichol and Dhariwal 2021): abar_t = f(t)/f(0), f(t) = cos^2(((t/T) + s)/(1 + s) * pi/2).
export function schedule(T = 1000, kind = 'linear', b0 = 1e-4, b1 = 0.02, s = 0.008) {
  const abar = [1], beta = [0];
  if (kind === 'cosine') {
    const f = (t) => Math.cos((((t / T) + s) / (1 + s)) * Math.PI / 2) ** 2;
    for (let t = 1; t <= T; t++) { abar.push(f(t) / f(0)); beta.push(Math.min(0.999, 1 - abar[t] / abar[t - 1])); }
  } else {
    for (let t = 1; t <= T; t++) { const b = b0 + ((b1 - b0) * (t - 1)) / (T - 1); beta.push(b); abar.push(abar[t - 1] * (1 - b)); }
  }
  return { T, beta, abar };
}
// Exact DDPM reverse process for 1-D data from a Gaussian mixture, using the exact noise predictor (so no network is needed).
// Returns the particles at every step: traj[t] is the array of n values at step t (traj[T] = pure noise, traj[0] = the generated samples).
export function diffSample1d({ means = [-2, 3], weights = [0.5, 0.5], sd0 = 0.4, T = 200, b0 = 1e-3, b1 = 0.06, n = 400, seed = 5, stochastic = true } = {}) {
  const { beta, abar } = schedule(T, 'linear', b0, b1);
  const g = gaussStream(seed);
  let x = Array.from({ length: n }, () => g());
  const traj = Array(T + 1);
  traj[T] = x.slice();
  const epsHat = (xv, t) => {
    const ab = abar[t];
    const m = means.map((mu) => Math.sqrt(ab) * mu), v = sd0 * sd0 * ab + (1 - ab);
    const lw = weights.map((w, i) => Math.log(w) - 0.5 * ((xv - m[i]) ** 2) / v);
    const mx = Math.max(...lw), e = lw.map((q) => Math.exp(q - mx)), z = e.reduce((a, b) => a + b, 0);
    let s = 0; e.forEach((q, i) => { s += (q / z) * (xv - m[i]) / v; });
    return Math.sqrt(1 - ab) * s;
  };
  for (let t = T; t >= 1; t--) {
    const a = 1 - beta[t];
    x = x.map((xv) => {
      const mean = (xv - (beta[t] / Math.sqrt(1 - abar[t])) * epsHat(xv, t)) / Math.sqrt(a);
      return mean + (stochastic && t > 1 ? Math.sqrt(beta[t]) * g() : 0);
    });
    traj[t - 1] = x.slice();
  }
  return { traj, beta, abar };
}
// Shapes and parameter counts of a small UNet: `levels` resolutions, channels base * 2^i, two 3x3 convs per block plus a time-embedding projection.
export function unetShapes({ size = 16, inCh = 1, base = 16, levels = 3, timeDim = 32 }) {
  const rows = [];
  const conv = (i, o) => 9 * i * o + o;
  const block = (i, o) => conv(i, o) + conv(o, o) + timeDim * o + o;
  rows.push({ name: 'time embedding (sinusoidal, then MLP)', shape: `${timeDim}`, params: 2 * (timeDim * timeDim + timeDim), skip: null });
  let cin = inCh, res = size;
  const enc = [];
  for (let i = 0; i < levels; i++) {
    const c = base * 2 ** i;
    rows.push({ name: i === levels - 1 ? `bottleneck block ${i + 1}` : `down block ${i + 1}`, shape: `${c} × ${res} × ${res}`, params: block(cin, c), skip: i === levels - 1 ? null : `saved for the skip at level ${i + 1}` });
    enc.push({ c, res });
    if (i < levels - 1) { rows.push({ name: `downsample ${i + 1} (conv 3×3, stride 2)`, shape: `${c} × ${res / 2} × ${res / 2}`, params: conv(c, c) * 1 + 0, skip: null }); res /= 2; }
    cin = c;
  }
  for (let i = levels - 2; i >= 0; i--) {
    const c = enc[i].c, up = enc[i + 1].c;
    rows.push({ name: `upsample ${i + 1} (transpose conv 2×2, stride 2)`, shape: `${c} × ${enc[i].res} × ${enc[i].res}`, params: 4 * up * c + c, skip: null });
    rows.push({ name: `concatenate skip ${i + 1}`, shape: `${2 * c} × ${enc[i].res} × ${enc[i].res}`, params: 0, skip: 'channels double' });
    rows.push({ name: `up block ${i + 1}`, shape: `${c} × ${enc[i].res} × ${enc[i].res}`, params: block(2 * c, c), skip: null });
  }
  rows.push({ name: 'output conv 1×1', shape: `${inCh} × ${size} × ${size}`, params: base * inCh + inCh, skip: null });
  return { rows, total: rows.reduce((s, r) => s + r.params, 0) };
}
