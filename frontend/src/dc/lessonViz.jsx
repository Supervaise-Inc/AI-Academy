import React, { useEffect, useMemo, useRef, useState } from 'react';
import { softmaxT, filterSampling, chainStats, mulberry32, gaussNoise, BYTES, convParams, loraParams, chinchilla, FN, LOSS, neuronAccuracy, convolve, transposeConv, learnBPE, applyBPE, bigrams, cosine, nearest, posEnc, alphaBar, tokens, tfidfRank, attend, ACT, forwardNet, countParams, polyFit, polyEval, polyMse, makeDataset, initNet, predictNet, netLossAcc, trainEpoch, DACT, backwardNet, gradNorms, outSize, convGrad, pool2d, convShapes, tokenStats, cosineSim, rnnInfluence, lstmInfluence, lstmStep, softHard, tfParams, maskedSoftmax, CHIN, chinLoss, chinOptimal, moeRoute, moeParams, emergence, vitTokens, clipMatrix, patchPermutation, ganInit, ganStep, sigmoid, upsampleNearest, upsampleBilinear, transposeConvPad, outSizeT, schedule, diffSample1d, unetShapes, prefProb, prefLoss, klObjective, klOptimum, softmaxArr, svdSmall, lowRankApprox, ftMemory, chunkWords, nChunks, spanIntact, rankMetrics, ragBudget, indexSize, normCdf, railPassThrough, fgsmAttack, thresholdRates, agentCost, traceCost } from './lessonMath.js';
export { convolve, gaussNoise };

// Data-driven figures for lessons. Each takes the JSON from a ```viz block in a week-NN.md file and does a
// real calculation with it (no canned results), so what students see is what the maths gives.
// The spec shape for each type is documented in docs/LESSON_WIDGETS.md and checked by tools/validate-lessons.mjs.
// Exported: VIZ (type -> component) and the pure helpers used by the validator.

const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const f2 = (x, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : '–');

const Title = ({ children, fig }) => (
  <div className="wg-head">
    {fig ? <span className="wg-fig">Figure {fig}</span> : null}
    <div className="wg-t">{children}</div>
  </div>
);


// ---------------------------------------------------------------- pipeline (step through a process)
function Pipeline({ spec, fig }) {
  const [i, setI] = useState(0);
  const [play, setPlay] = useState(false);
  const steps = spec.steps;
  useEffect(() => {
    if (!play) return undefined;
    const id = setInterval(() => setI((x) => (x + 1 < steps.length ? x + 1 : (setPlay(false), x))), 1800);
    return () => clearInterval(id);
  }, [play, steps.length]);
  const st = steps[i];
  const active = new Set(st.active || []);
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <ol className={'vz-chain' + (spec.vertical ? ' v' : '')} aria-label="Stages">
        {spec.nodes.map((n) => (
          <li key={n.id} className={'vz-node' + (active.has(n.id) ? ' on' : '')} aria-current={active.has(n.id) ? 'step' : undefined}>
            <span className="vz-node-l">{n.label}</span>
            {n.sub ? <span className="vz-node-s">{n.sub}</span> : null}
          </li>
        ))}
      </ol>
      <div className="wg-detail" aria-live="polite">
        <div className="wg-detail-y">{i + 1}/{steps.length}</div>
        <div>
          <div className="wg-detail-t">{st.title}</div>
          <p className="wg-p" style={{ margin: 0 }}>{st.body}</p>
          {st.example ? <p className="vz-ex">{st.example}</p> : null}
        </div>
      </div>
      <div className="wg-foot">
        <button type="button" className="wg-btn ghost" onClick={() => { setPlay(false); setI(Math.max(0, i - 1)); }} disabled={i === 0}>Back</button>
        <span className="wg-row">
          <button type="button" className="wg-btn ghost" onClick={() => { if (i + 1 >= steps.length) setI(0); setPlay(!play); }}>{play ? 'Pause' : 'Play'}</button>
          <button type="button" className="wg-btn" onClick={() => { setPlay(false); setI(Math.min(steps.length - 1, i + 1)); }} disabled={i === steps.length - 1}>Next</button>
        </span>
      </div>
      {spec.note ? <p className="wg-note">{spec.note}</p> : null}
    </div>
  );
}

// ---------------------------------------------------------------- calc (parameter / memory calculators)
const human = (x) => {
  const a = Math.abs(x);
  if (a >= 1e15) return `10^${Math.round(Math.log10(a))}`.replace(/\^(\d+)/, (m, e) => '^' + e);
  if (a >= 1e12) return f2(x / 1e12, 2) + ' T';
  if (a >= 1e9) return f2(x / 1e9, 2) + ' B';
  if (a >= 1e6) return f2(x / 1e6, 2) + ' M';
  if (a >= 1e3) return f2(x / 1e3, 1) + ' K';
  return String(Math.round(x));
};

function Slider({ label, min, max, step, value, onChange, show }) {
  return (
    <label className="wg-slide">
      <span>
        {label} <strong>{show ? show(value) : value}</strong>
      </span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  );
}

function Calc({ spec, fig }) {
  const [a, setA] = useState(spec.kind === 'fc_vs_conv' ? 28 : spec.kind === 'lora' ? 4096 : spec.kind === 'memory' ? 7 : spec.kind === 'split' ? 1000 : spec.kind === 'ngram_space' ? 10000 : spec.kind === 'attn_cost' ? 512 : spec.kind === 'vit_tokens' ? 224 : spec.kind === 'gan_loss' ? -3 : spec.kind === 'ft_memory' ? 7 : spec.kind === 'rag_budget' ? 8192 : spec.kind === 'index_size' ? 10 : spec.kind === 'agent_cost' ? 800 : 24);
  const [b, setB] = useState(spec.kind === 'fc_vs_conv' ? 100 : spec.kind === 'lora' ? 8 : spec.kind === 'split' ? 70 : spec.kind === 'ngram_space' ? 2 : spec.kind === 'attn_cost' ? 64 : spec.kind === 'vit_tokens' ? 16 : spec.kind === 'gan_loss' ? 3 : spec.kind === 'ft_memory' ? 1 : spec.kind === 'rag_budget' ? 4 : spec.kind === 'index_size' ? 20 : spec.kind === 'agent_cost' ? 6 : 0);
  const [c3, setC3] = useState(spec.kind === 'ngram_space' ? 1000 : spec.kind === 'vit_tokens' ? (spec.channels ?? 3) : spec.kind === 'ft_memory' ? 1 : spec.kind === 'rag_budget' ? 300 : spec.kind === 'index_size' ? 384 : spec.kind === 'agent_cost' ? 200 : 15);
  let body = null;
  if (spec.kind === 'fc_vs_conv') {
    const cin = spec.channels || 1;
    const filters = spec.filters || 8;
    const k = spec.kernel || 3;
    const p = convParams(a, cin, b, filters, k);
    body = (
      <>
        <Slider label="Image side (pixels)" min={8} max={224} step={4} value={a} onChange={setA} show={(v) => `${v}×${v}×${cin}`} />
        <Slider label="Units in the fully connected layer" min={10} max={1000} step={10} value={b} onChange={setB} />
        <div className="vz-cmp">
          <div><span className="wg-k">Fully connected layer</span><span className="vz-big">{human(p.fc)}</span><span className="wg-note">{a}×{a}×{cin}×{b} weights + {b} biases</span></div>
          <div className="out"><span className="wg-k">Convolution: {filters} filters, {k}×{k}</span><span className="vz-big">{human(p.conv)}</span><span className="wg-note">{k}×{k}×{cin}×{filters} weights + {filters} biases, whatever the image size</span></div>
        </div>
        <p className="wg-note">The convolution reuses the same small kernels at every position (weight sharing), so its count does not grow with the image. The fully connected layer needs {f2(p.fc / p.conv, 0)}× more parameters here.</p>
      </>
    );
  } else if (spec.kind === 'lora') {
    const r = b;
    const p = loraParams(a, r);
    body = (
      <>
        <Slider label="Model dimension d (a d × d weight matrix)" min={512} max={8192} step={512} value={a} onChange={setA} />
        <Slider label="LoRA rank r" min={1} max={64} step={1} value={b} onChange={setB} />
        <div className="vz-cmp">
          <div><span className="wg-k">Full fine-tuning of this matrix</span><span className="vz-big">{human(p.full)}</span><span className="wg-note">d × d = {a} × {a}</span></div>
          <div className="out"><span className="wg-k">LoRA adapter (A: d×r, B: r×d)</span><span className="vz-big">{human(p.lora)}</span><span className="wg-note">2 × d × r = {human(p.lora)} trainable, {f2(p.frac * 100, 2)}% of the matrix</span></div>
        </div>
        <p className="wg-note">The original weights stay frozen; only A and B are trained. Per matrix, LoRA trains a fraction 2r/d of the parameters.</p>
      </>
    );
  } else if (spec.kind === 'memory') {
    const rows = Object.entries(BYTES).map(([name, by]) => ({ name, gb: (a * 1e9 * by) / 1e9 }));
    const max = Math.max(...rows.map((r) => r.gb), a * 16);
    body = (
      <>
        <Slider label="Model size (billions of parameters)" min={1} max={180} step={1} value={a} onChange={setA} show={(v) => `${v} B`} />
        <ul className="wg-bars">
          {rows.map((r) => (
            <li key={r.name} style={{ gridTemplateColumns: '110px minmax(0,1fr) 84px' }}>
              <span className="wg-bar-l" style={{ fontFamily: 'inherit' }}>{r.name}</span>
              <span className="wg-bar-track"><span className="wg-bar-fill" style={{ width: `${(r.gb / max) * 100}%` }} /></span>
              <span className="wg-bar-p">{f2(r.gb, r.gb < 10 ? 1 : 0)} GB</span>
            </li>
          ))}
          <li style={{ gridTemplateColumns: '110px minmax(0,1fr) 84px' }}>
            <span className="wg-bar-l" style={{ fontFamily: 'inherit' }}>Training, Adam</span>
            <span className="wg-bar-track"><span className="wg-bar-fill warn" style={{ width: `${((a * 16) / max) * 100}%` }} /></span>
            <span className="wg-bar-p">{f2(a * 16, 0)} GB</span>
          </li>
        </ul>
        <p className="wg-note">Weights only: parameters × bytes per parameter (GB = 10⁹ bytes). “Training, Adam” uses the common mixed-precision estimate of 16 bytes per parameter (weights, gradients, FP32 master copy and two Adam moments). Activations, the KV cache and framework overhead come on top.</p>
      </>
    );
  } else if (spec.kind === 'ft_memory') {
    const methods = [['full', 'Full fine-tuning'], ['lora', 'LoRA (16-bit base)'], ['qlora', 'QLoRA (4-bit base)']];
    const mi = Math.max(0, Math.min(2, b - 1));
    const m = methods[mi][0];
    const frac = c3 / 100;
    const R = ftMemory({ N: a, method: m, trainable: frac });
    const gb = (x) => (x / 1e9 >= 100 ? (x / 1e9).toFixed(0) : (x / 1e9).toFixed(1)) + ' GB';
    body = (
      <>
        <Slider label="Model size N (billions of parameters)" min={0.1} max={70} step={0.1} value={a} onChange={setA} show={(q) => `${q.toFixed(1)} B`} />
        <div className="wg-row" style={{ gap: 8, margin: '6px 0 10px' }}>{methods.map(([k, t], i) => <button key={k} type="button" className={'wg-chip' + (mi === i ? ' on' : '')} aria-pressed={mi === i} onClick={() => setB(i + 1)}>{t}</button>)}</div>
        {m !== 'full' ? <Slider label="Trainable share of the parameters (LoRA adapters)" min={0.1} max={5} step={0.1} value={Math.min(c3, 5)} onChange={setC3} show={(q) => `${q.toFixed(1)} %`} /> : null}
        <div className="vz-cmp" style={{ gridTemplateColumns: 'repeat(4,minmax(0,1fr))' }}>
          <div><span className="wg-k">Weights</span><span className="vz-big">{gb(R.weights)}</span><span className="wg-note">{m === 'qlora' ? '0.5 byte each' : '2 bytes each (16-bit)'}</span></div>
          <div><span className="wg-k">Gradients</span><span className="vz-big">{gb(R.grads)}</span><span className="wg-note">only for trained numbers</span></div>
          <div><span className="wg-k">Adam state + master copy</span><span className="vz-big">{gb(R.optimizer)}</span><span className="wg-note">12 bytes per trained number</span></div>
          <div className="out"><span className="wg-k">Total (without activations)</span><span className="vz-big">{gb(R.total)}</span><span className="wg-note">{m === 'full' ? '16 bytes per parameter' : `${(R.total / (a * 1e9)).toFixed(2)} bytes per parameter`}</span></div>
        </div>
        <p className="wg-note">Activations (the memory used by a forward pass: grows with batch size and sequence length) come on top. The 16 bytes of full fine-tuning are 2 (16-bit weights) + 2 (gradients) + 4 (32-bit master copy) + 8 (Adam's two moments). {spec.note || ''}</p>
      </>
    );
  } else if (spec.kind === 'rag_budget') {
    const R = ragBudget({ W: a, k: b, chunk: c3, overhead: spec.overhead ?? 150 });
    const reserve = spec.reserve ?? 300;
    const ok = R.left >= reserve;
    body = (
      <>
        <Slider label="Context window of the model (tokens)" min={1024} max={32768} step={1024} value={a} onChange={setA} show={(q) => q.toLocaleString('en-US')} />
        <Slider label="Passages retrieved (k)" min={1} max={20} step={1} value={b} onChange={setB} />
        <Slider label="Tokens per passage (chunk size)" min={50} max={1000} step={50} value={c3} onChange={setC3} />
        <div className="vz-cmp" style={{ gridTemplateColumns: 'repeat(4,minmax(0,1fr))' }}>
          <div><span className="wg-k">Instructions + question</span><span className="vz-big">{(spec.overhead ?? 150).toLocaleString('en-US')}</span><span className="wg-note">fixed overhead (assumed)</span></div>
          <div><span className="wg-k">Retrieved context</span><span className="vz-big">{R.context.toLocaleString('en-US')}</span><span className="wg-note">{b} × {c3} tokens</span></div>
          <div><span className="wg-k">Left for the answer</span><span className="vz-big">{R.left.toLocaleString('en-US')}</span><span className="wg-note">window − prompt</span></div>
          <div className="out"><span className="wg-k">Does it fit?</span><span className="vz-big">{ok ? 'Yes' : 'No'}</span><span className="wg-note">{ok ? `at least ${reserve} tokens left for the reply` : `needs ${reserve} tokens for the reply`}</span></div>
        </div>
        <p className="wg-note">Prompt tokens = overhead + k × chunk size. Attention cost also grows with prompt length (Week 7), and long prompts can bury the useful passage, so retrieving more is not always better. The overhead and the reply reserve are assumptions for the demonstration.</p>
      </>
    );
  } else if (spec.kind === 'index_size') {
    const R = indexSize({ docs: a * 1000, chunksPer: b, dim: c3 });
    const bytesTxt = (x) => (x >= 1e9 ? (x / 1e9).toFixed(2) + ' GB' : (x / 1e6).toFixed(1) + ' MB');
    body = (
      <>
        <Slider label="Documents (thousands)" min={1} max={1000} step={1} value={a} onChange={setA} show={(q) => `${q.toLocaleString('en-US')} thousand`} />
        <Slider label="Chunks per document" min={1} max={100} step={1} value={b} onChange={setB} />
        <div className="wg-row" style={{ gap: 8, margin: '6px 0 10px' }}>{[384, 768, 1536].map((d) => <button key={d} type="button" className={'wg-chip' + (c3 === d ? ' on' : '')} aria-pressed={c3 === d} onClick={() => setC3(d)}>{d} numbers per vector</button>)}</div>
        <div className="vz-cmp" style={{ gridTemplateColumns: 'repeat(3,minmax(0,1fr))' }}>
          <div><span className="wg-k">Vectors in the index</span><span className="vz-big">{R.vectors.toLocaleString('en-US')}</span><span className="wg-note">documents × chunks</span></div>
          <div className="out"><span className="wg-k">Memory (32-bit floats)</span><span className="vz-big">{bytesTxt(R.bytes)}</span><span className="wg-note">vectors × {c3} × 4 bytes</span></div>
          <div><span className="wg-k">One exact search</span><span className="vz-big">{human(R.flops)}</span><span className="wg-note">multiply-adds: vectors × {c3}</span></div>
        </div>
        <p className="wg-note">Exact (brute-force) search compares the question with every vector. Large systems use approximate nearest-neighbour indexes to avoid that, and sometimes store 16-bit or 8-bit numbers. Typical embedding sizes: 384 (small models) to 1,536 (large ones).</p>
      </>
    );
  } else if (spec.kind === 'agent_cost') {
    const W = spec.window ?? 8192;
    const R = agentCost({ C0: a, s: c3, n: b });
    const single = a;
    const fits = R.final <= W;
    body = (
      <>
        <Slider label="Starting prompt: instructions, tool list, goal (tokens)" min={100} max={4000} step={100} value={a} onChange={setA} show={(q) => q.toLocaleString('en-US')} />
        <Slider label="Loops of the agent (model calls)" min={1} max={30} step={1} value={b} onChange={setB} />
        <Slider label="Tokens added per loop (thought + action + observation)" min={50} max={1000} step={50} value={c3} onChange={setC3} />
        <div className="vz-cmp" style={{ gridTemplateColumns: 'repeat(4,minmax(0,1fr))' }}>
          <div><span className="wg-k">Final context</span><span className="vz-big">{R.final.toLocaleString('en-US')}</span><span className="wg-note">{a} + {b} × {c3} tokens</span></div>
          <div className="out"><span className="wg-k">Tokens read in total</span><span className="vz-big">{R.read.toLocaleString('en-US')}</span><span className="wg-note">{b}·{a} + {c3}·{b}({b} − 1)/2</span></div>
          <div><span className="wg-k">Compared with one call</span><span className="vz-big">{f2(R.read / single, 1)}×</span><span className="wg-note">of the {single.toLocaleString('en-US')}-token prompt alone</span></div>
          <div><span className="wg-k">Fits in {W.toLocaleString('en-US')} tokens?</span><span className="vz-big">{fits ? 'Yes' : 'No'}</span><span className="wg-note">{fits ? `${(W - R.final).toLocaleString('en-US')} tokens to spare` : 'the oldest steps would be cut or the run fails'}</span></div>
        </div>
        <p className="wg-note">Every loop sends the whole history to the model again, so the work grows with the square of the number of loops while the context grows in a straight line. The window size is {W.toLocaleString('en-US')} tokens (an assumption for the demonstration).</p>
      </>
    );
  } else if (spec.kind === 'gan_loss') {
    const dF = sigmoid(a), dR = sigmoid(b);
    const f = (q) => Number(q.toFixed(3)).toString();
    body = (
      <>
        <Slider label="Discriminator's score (logit) for a FAKE picture" min={-8} max={8} step={0.5} value={a} onChange={setA} show={(q) => `${q.toFixed(1)}  →  D = ${f(sigmoid(q))}`} />
        <Slider label="Discriminator's score (logit) for a REAL picture" min={-8} max={8} step={0.5} value={b} onChange={setB} show={(q) => `${q.toFixed(1)}  →  D = ${f(sigmoid(q))}`} />
        <div className="vz-cmp" style={{ gridTemplateColumns: 'repeat(3,minmax(0,1fr))' }}>
          <div><span className="wg-k">Discriminator loss, real: −ln D(x)</span><span className="vz-big">{f(-Math.log(dR))}</span><span className="wg-note">small when D is sure it is real</span></div>
          <div><span className="wg-k">Discriminator loss, fake: −ln(1 − D(G(z)))</span><span className="vz-big">{f(-Math.log(1 - dF))}</span><span className="wg-note">small when D is sure it is fake</span></div>
          <div className="out"><span className="wg-k">Value V = ln D(x) + ln(1 − D(G(z)))</span><span className="vz-big">{f(Math.log(dR) + Math.log(1 - dF))}</span><span className="wg-note">D wants V high, G wants it low; at equilibrium V = −ln 4 = −1.386</span></div>
        </div>
        <div className="vz-cmp" style={{ marginTop: 10, gridTemplateColumns: 'repeat(2,minmax(0,1fr))' }}>
          <div><span className="wg-k">Generator, original loss ln(1 − D(G(z)))</span><span className="vz-big">{f(Math.log(1 - dF))}</span><span className="wg-note">gradient w.r.t. the logit = −D = {f(-dF)}: tiny when D is winning (D ≈ 0)</span></div>
          <div className="out"><span className="wg-k">Generator, non-saturating loss −ln D(G(z))</span><span className="vz-big">{f(-Math.log(dF))}</span><span className="wg-note">gradient w.r.t. the logit = D − 1 = {f(dF - 1)}: large when D is winning</span></div>
        </div>
        <p className="wg-note">Drag the fake score to −6: the original generator loss barely moves (gradient about 0), so the generator learns nothing exactly when it needs to learn most. {spec.note || ''}</p>
      </>
    );
  } else if (spec.kind === 'vit_tokens') {
    const sizes = [8, 14, 16, 32];
    const t = vitTokens({ img: a, patch: b, c: c3, d: spec.d || 768 });
    const fmtN = (x) => (x >= 1e9 ? (x / 1e9).toFixed(2) + ' billion' : x >= 1e6 ? (x / 1e6).toFixed(2) + ' million' : x.toLocaleString());
    body = (
      <>
        <Slider label="Image side (pixels)" min={32} max={512} step={32} value={a} onChange={setA} show={(q) => `${q} × ${q}`} />
        <div className="wg-row" style={{ gap: 8, margin: '6px 0 10px' }}>{sizes.map((q) => <button key={q} type="button" className={'wg-chip' + (b === q ? ' on' : '')} aria-pressed={b === q} onClick={() => setB(q)}>patch {q}×{q}</button>)}<button type="button" className={'wg-chip' + (c3 === 3 ? ' on' : '')} aria-pressed={c3 === 3} onClick={() => setC3(c3 === 3 ? 1 : 3)}>{c3 === 3 ? 'colour (3 channels)' : 'greyscale (1 channel)'}</button></div>
        {!t.exact ? <p className="wg-why bad" role="alert">{a} is not divisible by {b}: a ViT needs the patches to tile the image exactly (it would be resized or cropped).</p> : null}
        <div className="vz-cmp" style={{ gridTemplateColumns: 'repeat(3,minmax(0,1fr))' }}>
          <div><span className="wg-k">Patches (tokens)</span><span className="vz-big">{t.side} × {t.side} = {t.patches.toLocaleString()}</span><span className="wg-note">+ 1 CLS token = a sequence of {t.seq.toLocaleString()}</span></div>
          <div><span className="wg-k">Numbers in one flattened patch</span><span className="vz-big">{t.patchDim.toLocaleString()}</span><span className="wg-note">{b} × {b} × {c3}</span></div>
          <div className="out"><span className="wg-k">Patch-embedding parameters (d = {spec.d || 768})</span><span className="vz-big">{t.projParams.toLocaleString()}</span><span className="wg-note">patch numbers × d + d biases</span></div>
        </div>
        <div className="vz-cmp" style={{ marginTop: 10, gridTemplateColumns: 'repeat(2,minmax(0,1fr))' }}>
          <div><span className="wg-k">Attention scores with patch tokens</span><span className="vz-big">{fmtN(t.attn)}</span><span className="wg-note">one table of {t.seq} × {t.seq} per head</span></div>
          <div><span className="wg-k">If every pixel were a token</span><span className="vz-big">{fmtN(t.pixelAttn)}</span><span className="wg-note">{t.pixelSeq.toLocaleString()} tokens: {(t.pixelAttn / t.attn).toFixed(0)}× more scores</span></div>
        </div>
        <p className="wg-note">Patching is what makes self-attention affordable on images: n² grows with the number of tokens (Week 7). {spec.note || ''}</p>
      </>
    );
  } else if (spec.kind === 'attn_cost') {
    const n = a, d = b;
    const scores = n * n;
    const flops = 2 * n * n * d * 2;
    body = (
      <>
        <Slider label="Sequence length n (tokens)" min={16} max={32768} step={16} value={a} onChange={setA} show={(q) => q.toLocaleString()} />
        <Slider label="Head size d" min={16} max={256} step={16} value={b} onChange={setB} />
        <div className="vz-cmp" style={{ gridTemplateColumns: 'repeat(3,minmax(0,1fr))' }}>
          <div><span className="wg-k">Attention scores (n × n)</span><span className="vz-big">{scores >= 1e6 ? (scores / 1e6).toFixed(1) + ' M' : scores.toLocaleString()}</span><span className="wg-note">one number for every pair of tokens</span></div>
          <div><span className="wg-k">Multiply-adds for QKᵀ and weights × V</span><span className="vz-big">{flops >= 1e9 ? (flops / 1e9).toFixed(2) + ' G' : (flops / 1e6).toFixed(1) + ' M'}</span><span className="wg-note">about 4 · n² · d</span></div>
          <div className="out"><span className="wg-k">Steps that must run one after another</span><span className="vz-big">1 vs {n.toLocaleString()}</span><span className="wg-note">attention (all pairs at once) vs an RNN (one step per token)</span></div>
        </div>
        <p className="wg-note">Double the length and the score table gets four times bigger: attention trades the RNN's long chain of steps for a quadratic table. {spec.note || ''}</p>
      </>
    );
  } else if (spec.kind === 'ngram_space') {
    const possible = Math.pow(a, b);
    const corpus = c3 * 1e6;
    const seen = Math.min(possible, corpus);
    const sci = (x) => (x >= 1e6 ? x.toExponential(1).replace('e+', ' × 10^') : x.toLocaleString());
    body = (
      <>
        <Slider label="Vocabulary size V (different words)" min={1000} max={200000} step={1000} value={a} onChange={setA} show={(q) => q.toLocaleString()} />
        <Slider label="n (length of the n-gram)" min={1} max={5} step={1} value={b} onChange={setB} />
        <Slider label="Training text (millions of words)" min={1} max={10000} step={1} value={c3} onChange={setC3} show={(q) => q.toLocaleString()} />
        <div className="vz-cmp" style={{ gridTemplateColumns: 'repeat(3,minmax(0,1fr))' }}>
          <div><span className="wg-k">Possible {b}-grams (V^{b})</span><span className="vz-big">{sci(possible)}</span><span className="wg-note">the table you would need</span></div>
          <div><span className="wg-k">Words of training text</span><span className="vz-big">{sci(corpus)}</span><span className="wg-note">at most this many different {b}-grams can ever be seen</span></div>
          <div className="out"><span className="wg-k">Share of all {b}-grams that can ever appear</span><span className="vz-big">{seen / possible >= 0.01 ? (100 * seen / possible).toFixed(1) + '%' : (100 * seen / possible).toExponential(1) + '%'}</span><span className="wg-note">the rest are unseen: their count is 0</span></div>
        </div>
        <p className="wg-note">Most possible n-grams never occur in any text, so an n-gram model gives them probability 0 unless you smooth. This is the sparsity problem. {spec.note || ''}</p>
      </>
    );
  } else if (spec.kind === 'split') {
    const tr = Math.round((a * b) / 100);
    const va = Math.round((a * c3) / 100);
    const te = a - tr - va;
    body = (
      <>
        <Slider label="Examples in the whole dataset" min={50} max={100000} step={50} value={a} onChange={setA} show={(v) => v.toLocaleString()} />
        <Slider label="Training share" min={50} max={90} step={5} value={b} onChange={(v) => { setB(v); if (v + c3 > 95) setC3(95 - v); }} show={(v) => `${v}%`} />
        <Slider label="Validation share" min={0} max={Math.min(30, 95 - b)} step={5} value={Math.min(c3, 95 - b)} onChange={setC3} show={(v) => `${v}%`} />
        <div className="vz-split" role="img" aria-label={`Training ${tr}, validation ${va}, test ${te}`}>
          <span style={{ width: `${(tr / a) * 100}%` }} className="tr">train</span>
          {va > 0 ? <span style={{ width: `${(va / a) * 100}%` }} className="va">val</span> : null}
          <span style={{ width: `${(te / a) * 100}%` }} className="te">test</span>
        </div>
        <div className="vz-cmp" style={{ gridTemplateColumns: 'repeat(3,minmax(0,1fr))' }}>
          <div><span className="wg-k">Training set</span><span className="vz-big">{tr.toLocaleString()}</span><span className="wg-note">the model learns from these</span></div>
          <div><span className="wg-k">Validation set</span><span className="vz-big">{va.toLocaleString()}</span><span className="wg-note">you tune choices on these</span></div>
          <div className="out"><span className="wg-k">Test set</span><span className="vz-big">{te.toLocaleString()}</span><span className="wg-note">touched once, at the very end</span></div>
        </div>
        <p className="wg-note">The test set must stay untouched until the end; if you keep adjusting the model to do well on it, it stops being an honest exam.</p>
      </>
    );
  } else if (spec.kind === 'scaling') {
    const c = chinchilla(a);
    body = (
      <>
        <Slider label="Training compute C (FLOPs)" min={20} max={26} step={0.25} value={a} onChange={setA} show={(v) => `10^${v}`} />
        <div className="vz-cmp">
          <div><span className="wg-k">Compute-optimal parameters N</span><span className="vz-big">{human(c.N)}</span><span className="wg-note">N = √(C / 120)</span></div>
          <div className="out"><span className="wg-k">Compute-optimal training tokens D</span><span className="vz-big">{human(c.D)}</span><span className="wg-note">D ≈ 20 × N</span></div>
        </div>
        <p className="wg-note">Rule of thumb from the Chinchilla paper (Hoffmann et al., 2022): training cost C ≈ 6·N·D FLOPs, and for a fixed budget the best results come from about 20 training tokens per parameter. This is an approximation, not an exact law.</p>
      </>
    );
  }
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      {body}
    </div>
  );
}

// ---------------------------------------------------------------- curve (activation functions, gradients, decay)

function Curve({ spec, fig }) {
  const [p, setP] = useState(spec.param ? spec.param.value : 0);
  const [x0, x1] = spec.xRange || [-6, 6];
  const [y0, y1] = spec.yRange || [-0.5, 1.5];
  const W = 560;
  const H = 240;
  const pad = { l: 40, r: 12, t: 12, b: 28 };
  const sx = (x) => pad.l + ((x - x0) / (x1 - x0)) * (W - pad.l - pad.r);
  const ly = (y) => (spec.logY ? Math.log10(Math.max(y, 1e-12)) : y);
  const [ya, yb] = spec.logY ? [spec.yRange ? spec.yRange[0] : -8, spec.yRange ? spec.yRange[1] : 0] : [y0, y1];
  const sy = (y) => H - pad.b - ((ly(y) - ya) / (yb - ya)) * (H - pad.t - pad.b);
  const N = 160;
  const colors = ['var(--accent)', 'var(--cyan)', 'var(--amber)', 'var(--indigo)'];
  const series = spec.series.map((s, k) => {
    const fn = FN[s.fn];
    const pts = Array.from({ length: N + 1 }, (_, i) => {
      const x = x0 + ((x1 - x0) * i) / N;
      return [sx(x), sy(fn.f(x, p))];
    });
    return { label: s.label || fn.label, color: colors[k % colors.length], d: pts.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(1)} ${clamp(y, -50, H + 50).toFixed(1)}`).join(' ') };
  });
  const xt = [x0, (x0 + x1) / 2, x1];
  const yt = spec.logY ? [ya, (ya + yb) / 2, yb] : [y0, (y0 + y1) / 2, y1];
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      {spec.param ? <Slider label={spec.param.label} min={spec.param.min} max={spec.param.max} step={spec.param.step} value={p} onChange={setP} /> : null}
      <svg viewBox={`0 0 ${W} ${H}`} className="vz-plot" role="img" aria-label={spec.alt || spec.title}>
        {xt.map((x) => <g key={'x' + x}><line x1={sx(x)} x2={sx(x)} y1={pad.t} y2={H - pad.b} className="vz-grid" /><text x={sx(x)} y={H - 8} className="vz-tick" textAnchor="middle">{f2(x, x % 1 ? 1 : 0)}</text></g>)}
        {yt.map((y) => <g key={'y' + y}><line x1={pad.l} x2={W - pad.r} y1={sy(spec.logY ? 10 ** y : y)} y2={sy(spec.logY ? 10 ** y : y)} className="vz-grid" /><text x={pad.l - 6} y={sy(spec.logY ? 10 ** y : y) + 4} className="vz-tick" textAnchor="end">{spec.logY ? `1e${f2(y, 0)}` : f2(y, 1)}</text></g>)}
        <clipPath id="vz-clip"><rect x={pad.l} y={pad.t} width={W - pad.l - pad.r} height={H - pad.t - pad.b} /></clipPath>
        <g clipPath="url(#vz-clip)">{series.map((s) => <path key={s.label} d={s.d} fill="none" stroke={s.color} strokeWidth="2.4" strokeLinejoin="round" />)}</g>
      </svg>
      <div className="wg-row vz-legend">
        {series.map((s) => <span key={s.label} className="vz-key"><i style={{ background: s.color }} />{s.label}</span>)}
      </div>
      {spec.note ? <p className="wg-note">{spec.note}</p> : null}
    </div>
  );
}

// ---------------------------------------------------------------- descent (gradient descent on a 1-D loss)
function Descent({ spec, fig }) {
  const L = LOSS[spec.fn || 'quad'];
  const [lr, setLr] = useState(spec.lr ?? 0.1);
  const [w0, setW0] = useState(spec.start ?? L.range[0] + 0.5);
  const [path, setPath] = useState([spec.start ?? L.range[0] + 0.5]);
  const w = path[path.length - 1];
  const reset = (s = w0) => setPath([s]);
  const step = (n) => setPath((p) => {
    const out = [...p];
    for (let i = 0; i < n; i++) out.push(clamp(out[out.length - 1] - lr * L.g(out[out.length - 1]), -1e3, 1e3));
    return out;
  });
  const W = 560, H = 230, pad = 28;
  const [a, b] = L.range;
  const ys = Array.from({ length: 121 }, (_, i) => L.f(a + ((b - a) * i) / 120));
  const ymin = Math.min(...ys), ymax = Math.max(...ys);
  const sx = (x) => pad + ((x - a) / (b - a)) * (W - 2 * pad);
  const sy = (y) => H - pad - ((clamp(y, ymin, ymax) - ymin) / (ymax - ymin)) * (H - 2 * pad);
  const d = ys.map((y, i) => `${i ? 'L' : 'M'}${sx(a + ((b - a) * i) / 120).toFixed(1)} ${sy(y).toFixed(1)}`).join(' ');
  const diverged = Math.abs(w) > 50 || !Number.isFinite(w);
  const settled = path.length > 1 && Math.abs(L.g(w)) < 1e-3;
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <p className="wg-note" style={{ marginTop: 0 }}>{L.label}</p>
      <Slider label="Learning rate" min={0.01} max={1.2} step={0.01} value={lr} onChange={setLr} />
      <Slider label="Starting weight" min={a} max={b} step={0.1} value={w0} onChange={(v) => { setW0(v); reset(v); }} />
      <svg viewBox={`0 0 ${W} ${H}`} className="vz-plot" role="img" aria-label={`Loss curve with the path of gradient descent. Current weight ${f2(w)}, loss ${f2(L.f(w))}.`}>
        <path d={d} fill="none" stroke="var(--text3)" strokeWidth="2" />
        {path.slice(-40).map((x, i, arr) => <circle key={i} cx={sx(x)} cy={sy(L.f(x))} r={i === arr.length - 1 ? 6 : 3.5} fill={i === arr.length - 1 ? 'var(--accent)' : 'var(--accent-soft-border)'} />)}
      </svg>
      <p className="wg-p" aria-live="polite" style={{ margin: '8px 0' }}>
        Step <strong>{path.length - 1}</strong> · w = <strong>{diverged ? '∞' : f2(w, 3)}</strong> · loss = <strong>{diverged ? '∞' : f2(L.f(w), 3)}</strong> · gradient = <strong>{diverged ? '∞' : f2(L.g(w), 3)}</strong>
      </p>
      <p className="wg-note" style={{ marginTop: 0 }}>{diverged ? 'Diverging: the steps are so large that the loss grows. Lower the learning rate.' : settled ? 'Converged: the gradient is close to zero here.' : 'Each step moves w by −learning rate × gradient.'}</p>
      <div className="wg-row">
        <button type="button" className="wg-btn" onClick={() => step(1)}>Step</button>
        <button type="button" className="wg-btn ghost" onClick={() => step(20)}>Run 20 steps</button>
        <button type="button" className="wg-btn ghost" onClick={() => reset()}>Reset</button>
      </div>
      {spec.note ? <p className="wg-note">{spec.note}</p> : null}
    </div>
  );
}

// ---------------------------------------------------------------- neuron (a single neuron in 2-D)
function Neuron({ spec, fig }) {
  const sets = spec.datasets;
  const [k, setK] = useState(0);
  const [w1, setW1] = useState(spec.w ? spec.w[0] : 0.5);
  const [w2, setW2] = useState(spec.w ? spec.w[1] : -0.5);
  const [b, setB] = useState(spec.b ?? 0);
  const pts = sets[k].points;
  const { ok, n } = neuronAccuracy(pts, w1, w2, b);
  const [r0, r1] = spec.range || [-3, 3];
  const S = 280;
  const sx = (x) => ((x - r0) / (r1 - r0)) * S;
  const sy = (y) => S - ((y - r0) / (r1 - r0)) * S;
  let line = null;
  if (Math.abs(w2) > 1e-6) {
    const ya = (-b - w1 * r0) / w2, yb = (-b - w1 * r1) / w2;
    line = [sx(r0), sy(ya), sx(r1), sy(yb)];
  } else if (Math.abs(w1) > 1e-6) {
    const x = -b / w1;
    line = [sx(x), sy(r0), sx(x), sy(r1)];
  }
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      {sets.length > 1 ? (
        <div className="wg-tabs" role="tablist">
          {sets.map((s, i) => <button key={s.name} type="button" role="tab" aria-selected={i === k} className={'wg-tab' + (i === k ? ' on' : '')} onClick={() => setK(i)}>{s.name}</button>)}
        </div>
      ) : null}
      <div className="vz-two">
        <div>
          <Slider label="Weight w₁" min={-3} max={3} step={0.1} value={w1} onChange={setW1} show={(v) => f2(v, 1)} />
          <Slider label="Weight w₂" min={-3} max={3} step={0.1} value={w2} onChange={setW2} show={(v) => f2(v, 1)} />
          <Slider label="Bias b" min={-3} max={3} step={0.1} value={b} onChange={setB} show={(v) => f2(v, 1)} />
          <p className="wg-p" aria-live="polite" style={{ margin: '6px 0' }}>Output = 1 when <strong>{f2(w1, 1)}·x + {f2(w2, 1)}·y + {f2(b, 1)} &gt; 0</strong></p>
          <p className={'wg-score' + (ok === n ? ' ok' : '')} style={{ padding: '10px 14px' }} aria-live="polite"><span className="wg-score-n" style={{ fontSize: 22 }}>{ok}/{n}</span> correct{ok === n ? ': the line separates the classes.' : '.'}</p>
        </div>
        <svg viewBox={`0 0 ${S} ${S}`} className="vz-sq" role="img" aria-label={`Two classes of points and the decision line. ${ok} of ${n} classified correctly.`}>
          <rect width={S} height={S} fill="var(--surface2)" rx="10" />
          <line x1={sx(0)} x2={sx(0)} y1="0" y2={S} className="vz-grid" /><line x1="0" x2={S} y1={sy(0)} y2={sy(0)} className="vz-grid" />
          {line ? <line x1={line[0]} y1={line[1]} x2={line[2]} y2={line[3]} stroke="var(--text)" strokeWidth="2" /> : null}
          {pts.map((p, i) => {
            const good = (w1 * p.x + w2 * p.y + b > 0 ? 1 : 0) === p.c;
            return p.c ? <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r="7" fill="var(--accent)" stroke={good ? 'none' : 'var(--red-text)'} strokeWidth="3" /> : <rect key={i} x={sx(p.x) - 6} y={sy(p.y) - 6} width="12" height="12" fill="var(--cyan)" stroke={good ? 'none' : 'var(--red-text)'} strokeWidth="3" />;
          })}
        </svg>
      </div>
      <p className="wg-note">Circles are class 1, squares are class 0. A point with a red outline is on the wrong side of the line.{spec.note ? ' ' + spec.note : ''}</p>
    </div>
  );
}

// ---------------------------------------------------------------- conv (convolution and transpose convolution by hand)
function Grid({ data, mark, onPick, cls, tone }) {
  return (
    <div className="vz-grid-m" style={{ gridTemplateColumns: `repeat(${data[0].length}, minmax(0, 34px))` }}>
      {data.map((row, i) => row.map((v, j) => {
        const on = mark && mark(i, j);
        return (
          <button key={i + '-' + j} type="button" className={'vz-cell ' + (cls || '') + (on ? ' on' : '')} disabled={!onPick} onClick={() => onPick && onPick(i, j)} aria-label={`row ${i + 1}, column ${j + 1}: ${v}`} style={tone ? { background: tone(v) } : undefined}>
            {Number.isInteger(v) ? v : f2(v, 1)}
          </button>
        );
      }))}
    </div>
  );
}
function Conv({ spec, fig }) {
  const transpose = spec.mode === 'transpose';
  const stride = spec.stride || (transpose ? 2 : 1);
  const pad = spec.padding || 0;
  const out = useMemo(() => (transpose ? transposeConv(spec.input, spec.kernel, stride) : convolve(spec.input, spec.kernel, stride, pad)), [spec, transpose, stride, pad]);
  const [sel, setSel] = useState([0, 0]);
  const k = spec.kernel.length;
  const [si, sj] = sel;
  let calc = '';
  let inMark = () => false, outMark = () => false;
  if (!transpose) {
    const terms = [];
    for (let a = 0; a < k; a++) for (let c = 0; c < k; c++) {
      const i = si * stride - pad + a, j = sj * stride - pad + c;
      const v = i < 0 || j < 0 || i >= spec.input.length || j >= spec.input[0].length ? 0 : spec.input[i][j];
      terms.push(`${v}×${spec.kernel[a][c]}`);
    }
    calc = `${terms.join(' + ')} = ${out[si][sj]}`;
    inMark = (i, j) => i >= si * stride - pad && i < si * stride - pad + k && j >= sj * stride - pad && j < sj * stride - pad + k;
    outMark = (i, j) => i === si && j === sj;
  } else {
    const v = spec.input[si][sj];
    calc = `input ${v} × the ${k}×${k} kernel, added into the output starting at row ${si * stride + 1}, column ${sj * stride + 1}`;
    inMark = (i, j) => i === si && j === sj;
    outMark = (i, j) => i >= si * stride && i < si * stride + k && j >= sj * stride && j < sj * stride + k;
  }
  const pick = transpose ? (i, j) => setSel([i, j]) : (i, j) => setSel([i, j]);
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <div className="vz-conv">
        <div><span className="wg-k">Input {spec.input.length}×{spec.input[0].length}</span><Grid data={spec.input} mark={inMark} onPick={transpose ? pick : null} /></div>
        <span className="vz-op">{transpose ? '⊛ᵀ' : '∗'}</span>
        <div><span className="wg-k">Kernel {k}×{k}</span><Grid data={spec.kernel} cls="ker" /></div>
        <span className="vz-op">=</span>
        <div><span className="wg-k">Output {out.length}×{out[0].length} (stride {stride}{pad ? `, padding ${pad}` : ''})</span><Grid data={out} mark={outMark} onPick={transpose ? null : pick} cls="out" /></div>
      </div>
      <p className="wg-p" aria-live="polite" style={{ margin: '12px 0 0' }}><strong>{transpose ? `Input cell (${si + 1}, ${sj + 1})` : `Output cell (${si + 1}, ${sj + 1})`}:</strong> <span className="vz-calc">{calc}</span></p>
      <p className="wg-note">{transpose ? 'Select an input cell to see where its scaled copy of the kernel lands. Where copies overlap, the values add.' : 'Select an output cell to see which input window produced it. The same kernel slides over every position.'} {spec.note || ''}</p>
    </div>
  );
}

// ---------------------------------------------------------------- bpe (byte-pair encoding, learned live)
function BPE({ spec, fig }) {
  const maxM = spec.maxMerges || 10;
  const full = useMemo(() => learnBPE(spec.corpus, maxM), [spec, maxM]);
  const [m, setM] = useState(0);
  const [word, setWord] = useState(spec.tryWord || '');
  const cur = useMemo(() => learnBPE(spec.corpus, m), [spec, m]);
  const vocab = new Set(cur.words.flatMap((w) => w.t));
  const last = cur.learned[cur.learned.length - 1];
  const show = (t) => t.replace('</w>', '·');
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <Slider label="Merges learned" min={0} max={full.learned.length} step={1} value={Math.min(m, full.learned.length)} onChange={setM} />
      <p className="wg-p" aria-live="polite" style={{ margin: '4px 0 10px' }}>
        {last ? <>Last merge: <strong>{show(last.a)}</strong> + <strong>{show(last.b)}</strong> → <strong>{show(last.a + last.b)}</strong> (seen {last.count} times)</> : 'Start: every word is a sequence of single characters plus an end-of-word mark (·).'}
        {' '}Vocabulary size: <strong>{vocab.size}</strong>.
      </p>
      <ul className="vz-words">
        {cur.words.map((w, i) => (
          <li key={i}><span className="vz-wc">{spec.corpus[i][0]} <em>×{spec.corpus[i][1]}</em></span><span className="vz-toks">{w.t.map((t, k) => <span key={k} className={'vz-tok' + (last && t === last.a + last.b ? ' new' : '')}>{show(t)}</span>)}</span></li>
        ))}
      </ul>
      <label className="vz-try"><span>Try a word with the {cur.learned.length} merges learned so far:</span>
        <input type="text" value={word} onChange={(e) => setWord(e.target.value.toLowerCase().replace(/[^a-z]/g, ''))} placeholder="e.g. lowest" aria-label="Word to tokenise" />
      </label>
      {word ? <p className="vz-toks" style={{ margin: '8px 0 0' }} aria-live="polite">{applyBPE(word, cur.learned).map((t, k) => <span key={k} className="vz-tok">{show(t)}</span>)}</p> : null}
      {spec.note ? <p className="wg-note">{spec.note}</p> : null}
    </div>
  );
}

// ---------------------------------------------------------------- ngram (a tiny statistical language model)
function NGram({ spec, fig }) {
  const model = useMemo(() => bigrams(spec.text), [spec]);
  const keys = [...model.keys()].sort();
  const [prev, setPrev] = useState(spec.start && model.has(spec.start) ? spec.start : keys[0]);
  const [gen, setGen] = useState([]);
  const next = model.get(prev) || new Map();
  const total = [...next.values()].reduce((a, b) => a + b, 0);
  const rows = [...next.entries()].sort((a, b) => b[1] - a[1]);
  const generate = () => {
    let w = prev; const out = [w];
    for (let i = 0; i < 9; i++) {
      const t = model.get(w); if (!t) break;
      const tot = [...t.values()].reduce((a, b) => a + b, 0); let r = Math.random() * tot; let pick = null;
      for (const [k, c] of t) { r -= c; if (r <= 0) { pick = k; break; } }
      w = pick || [...t.keys()][0]; out.push(w);
    }
    setGen((g) => [...g.slice(-3), out.join(' ')]);
  };
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <p className="vz-corpus">{spec.text}</p>
      <label className="vz-try"><span>The previous word is:</span>
        <select value={prev} onChange={(e) => setPrev(e.target.value)} aria-label="Previous word">{keys.map((k) => <option key={k}>{k}</option>)}</select>
      </label>
      <ul className="wg-bars" aria-live="polite">
        {rows.length ? rows.map(([w, c]) => (
          <li key={w} style={{ gridTemplateColumns: '110px minmax(0,1fr) 64px' }}><span className="wg-bar-l">{w}</span><span className="wg-bar-track"><span className="wg-bar-fill" style={{ width: `${(c / total) * 100}%` }} /></span><span className="wg-bar-p">{c}/{total}</span></li>
        )) : <li className="wg-note">“{prev}” never has a next word in this text.</li>}
      </ul>
      <div className="wg-row"><button type="button" className="wg-btn" onClick={generate}>Generate a sentence from “{prev}”</button><button type="button" className="wg-btn ghost" onClick={() => setGen([])} disabled={!gen.length}>Clear</button></div>
      {gen.length ? <ul className="vz-gen">{gen.map((g, i) => <li key={i}>{g}</li>)}</ul> : null}
      <p className="wg-note">Probabilities are counts from the text above: P(next | previous) = count(previous, next) / count(previous). The model sees only one previous word.{spec.note ? ' ' + spec.note : ''}</p>
    </div>
  );
}

// ---------------------------------------------------------------- scatter (embedding space)
const GROUP_COLORS = ['var(--accent)', 'var(--cyan)', 'var(--amber)', 'var(--indigo)', '#E0709B'];
function Scatter({ spec, fig }) {
  const metric = spec.metric || 'cosine';
  const [sel, setSel] = useState(spec.points[0].label);
  const [an, setAn] = useState(-1);
  const pts = spec.points;
  const groups = [...new Set(pts.map((p) => p.group || ''))];
  const S = 360;
  const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
  const x0 = Math.min(...xs, 0) - 0.5, x1 = Math.max(...xs) + 0.5, y0 = Math.min(...ys, 0) - 0.5, y1 = Math.max(...ys) + 0.5;
  const sx = (x) => 14 + ((x - x0) / (x1 - x0)) * (S - 28), sy = (y) => S - 14 - ((y - y0) / (y1 - y0)) * (S - 28);
  const by = Object.fromEntries(pts.map((p) => [p.label, p]));
  let near, label;
  if (an >= 0) {
    const a = spec.analogies[an];
    const v = [by[a.a].x - by[a.b].x + by[a.c].x, by[a.a].y - by[a.b].y + by[a.c].y];
    near = nearest(pts, v, [a.a, a.b, a.c], metric, 3);
    label = `${a.a} − ${a.b} + ${a.c} ≈`;
  } else {
    near = nearest(pts, [by[sel].x, by[sel].y], [sel], metric, 3);
    label = `Nearest to “${sel}” by ${metric === 'cosine' ? 'cosine similarity' : 'distance'}:`;
  }
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <div className="vz-two">
        <svg viewBox={`0 0 ${S} ${S}`} className="vz-sq" role="img" aria-label="A 2-D map of words or items; closer points are more similar.">
          <rect width={S} height={S} rx="10" fill="var(--surface2)" />
          {an >= 0 ? (() => { const a = spec.analogies[an]; return <><line x1={sx(by[a.b].x)} y1={sy(by[a.b].y)} x2={sx(by[a.a].x)} y2={sy(by[a.a].y)} stroke="var(--accent)" strokeWidth="2" strokeDasharray="5 4" /><line x1={sx(by[a.c].x)} y1={sy(by[a.c].y)} x2={sx(by[a.c].x + by[a.a].x - by[a.b].x)} y2={sy(by[a.c].y + by[a.a].y - by[a.b].y)} stroke="var(--accent)" strokeWidth="2" strokeDasharray="5 4" /></>; })() : null}
          {[...pts].sort((u, v) => (u.hideLabel ? 1 : 0) - (v.hideLabel ? 1 : 0)).map((p) => {
            const on = p.label === sel && an < 0;
            const small = p.hideLabel && !on;
            return (
              <g key={p.label} onClick={() => { setAn(-1); setSel(p.label); }} style={{ cursor: 'pointer' }}>
                <circle cx={sx(p.x)} cy={sy(p.y)} r={on ? 9 : small ? 5 : 6} fill={GROUP_COLORS[groups.indexOf(p.group || '') % 5]} stroke={on ? 'var(--text)' : small ? '#fff' : 'none'} strokeWidth={small ? 1.5 : 2} />
                {small ? null : <text x={sx(p.x) + (sx(p.x) > S * 0.72 ? -10 : 10)} y={sy(p.y) + 4} textAnchor={sx(p.x) > S * 0.72 ? 'end' : 'start'} className="vz-lab" style={{ paintOrder: 'stroke', stroke: 'var(--surface2)', strokeWidth: 3 }}>{p.label}</text>}
              </g>
            );
          })}
        </svg>
        <div>
          <div className="wg-row" style={{ marginBottom: 10 }}>
            {pts.map((p) => <button key={p.label} type="button" className={'wg-chip' + (an < 0 && p.label === sel ? ' on' : '')} onClick={() => { setAn(-1); setSel(p.label); }}>{p.label}</button>)}
          </div>
          {spec.analogies ? <div className="wg-row" style={{ marginBottom: 10 }}>{spec.analogies.map((a, i) => <button key={i} type="button" className={'wg-chip' + (an === i ? ' on' : '')} onClick={() => setAn(i)}>{a.a} − {a.b} + {a.c}</button>)}</div> : null}
          <p className="wg-p" style={{ margin: '0 0 6px' }}><strong>{label}</strong></p>
          <ol className="vz-near" aria-live="polite">{near.map(({ p, s }) => <li key={p.label}><strong>{p.label}</strong> <span>{metric === 'cosine' ? f2(s, 2) : f2(-s, 2)}</span></li>)}</ol>
        </div>
      </div>
      <p className="wg-note">{spec.note || 'Illustrative 2-D coordinates authored for teaching, not taken from a trained model.'}</p>
    </div>
  );
}

// ---------------------------------------------------------------- heatmap (any labelled matrix, optional tabs)
function Heatmap({ spec, fig }) {
  const tabs = spec.tabs || [{ name: '', W: spec.W }];
  const [t, setT] = useState(0);
  const [cell, setCell] = useState(null);
  const [row, setRow] = useState(null);
  const W = tabs[t].W;
  const max = spec.max || Math.max(...W.flat());
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      {tabs.length > 1 ? <div className="wg-tabs" role="tablist">{tabs.map((x, i) => <button key={x.name} type="button" role="tab" aria-selected={i === t} className={'wg-tab' + (i === t ? ' on' : '')} onClick={() => setT(i)}>{x.name}</button>)}</div> : null}
      {tabs[t].caption ? <p className="wg-p">{tabs[t].caption}</p> : null}
      <div className="vz-hm-wrap">
        <table className="vz-hm" aria-label={spec.title}>
          <thead><tr><th scope="col">{spec.corner || ''}</th>{spec.cols.map((c) => <th key={c} scope="col">{c}</th>)}</tr></thead>
          <tbody>
            {W.map((r, i) => (
              <tr key={i} className={row === i ? 'hi' : ''}>
                <th scope="row"><button type="button" className="vz-rowbtn" onClick={() => setRow(row === i ? null : i)}>{spec.rows[i]}</button></th>
                {r.map((v, j) => <td key={j}><button type="button" className="vz-hmc" style={{ '--v': clamp(v / max, 0, 1) }} onClick={() => setCell([i, j])} aria-label={`${spec.rows[i]}, ${spec.cols[j]}: ${f2(v, 2)}`}>{spec.showValues ? f2(v, 2).replace(/^0\./, '.') : ''}</button></td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="wg-p" aria-live="polite" style={{ margin: '8px 0 0' }}>{cell ? <><strong>{spec.rows[cell[0]]}</strong> → <strong>{spec.cols[cell[1]]}</strong>: {f2(W[cell[0]][cell[1]], 3)}</> : (spec.hint || 'Select a cell to read its value. Select a row name to highlight that row.')}{row !== null ? ` · row sum = ${f2(W[row].reduce((a, b) => a + b, 0), 2)}` : ''}</p>
      {spec.note ? <p className="wg-note">{spec.note}</p> : null}
    </div>
  );
}

// ---------------------------------------------------------------- posenc (sinusoidal positional encoding)
function PosEnc({ spec, fig }) {
  const [d, setD] = useState(spec.dModel || 16);
  const [n, setN] = useState(spec.positions || 24);
  const [p, setP] = useState(0);
  const rows = useMemo(() => Array.from({ length: n }, (_, i) => posEnc(i, d)), [n, d]);
  const sel = rows[Math.min(p, n - 1)];
  const dots = rows.map((r) => r.reduce((a, v, i) => a + v * sel[i], 0));
  const maxDot = Math.max(...dots.map(Math.abs), 1);
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <Slider label="Embedding size d" min={8} max={64} step={8} value={d} onChange={setD} />
      <Slider label="Positions shown" min={8} max={48} step={4} value={n} onChange={(v) => { setN(v); setP(Math.min(p, v - 1)); }} />
      <div className="vz-pe" style={{ gridTemplateColumns: `repeat(${d}, 1fr)` }} role="img" aria-label="Heatmap of sinusoidal positional encodings: rows are positions, columns are embedding dimensions.">
        {rows.map((r, i) => r.map((v, j) => <button key={i + '-' + j} type="button" tabIndex={j === 0 ? 0 : -1} onClick={() => setP(i)} className={'vz-pe-c' + (i === p ? ' sel' : '')} style={{ background: v >= 0 ? `color-mix(in srgb, var(--accent) ${Math.round(v * 100)}%, var(--surface2))` : `color-mix(in srgb, var(--cyan) ${Math.round(-v * 100)}%, var(--surface2))` }} aria-label={`position ${i}, dimension ${j}: ${f2(v, 2)}`} />))}
      </div>
      <p className="wg-note" style={{ marginTop: 6 }}>Rows are positions 0–{n - 1}; columns are dimensions 0–{d - 1}. Green is positive, blue is negative. Left columns oscillate fast, right columns slowly. Select a row.</p>
      <p className="wg-p" style={{ margin: '10px 0 6px' }}><strong>Position {p}</strong>: dot product with every position (larger = more similar)</p>
      <div className="vz-dots" aria-label="Similarity of the selected position to each position">
        {dots.map((v, i) => <span key={i} className={'vz-dot' + (i === p ? ' sel' : '')} style={{ height: `${Math.max(4, (Math.abs(v) / maxDot) * 60)}px` }} title={`position ${i}: ${f2(v, 2)}`} />)}
      </div>
      <p className="wg-note">PE(pos, 2i) = sin(pos / 10000^(2i/d)), PE(pos, 2i+1) = cos(pos / 10000^(2i/d)). Nearby positions get similar vectors, which lets a model that has no built-in sense of order use word position.{spec.note ? ' ' + spec.note : ''}</p>
    </div>
  );
}

// ---------------------------------------------------------------- patch (image -> patches -> tokens, as in a Vision Transformer)
function Patch({ spec, fig }) {
  const img = spec.image;
  const n = img.length;
  const opts = (spec.patchSizes || [2, 4, 8]).filter((p) => n % p === 0);
  const [ps, setPs] = useState(opts[Math.min(1, opts.length - 1)]);
  const [sel, setSel] = useState(0);
  const g = n / ps;
  const sI = Math.min(sel, g * g - 1);
  const pr = Math.floor(sI / g), pc = sI % g;
  const flat = [];
  for (let i = 0; i < ps; i++) for (let j = 0; j < ps; j++) flat.push(img[pr * ps + i][pc * ps + j]);
  const C = spec.channels || 1;
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <div className="wg-tabs" role="tablist" aria-label="Patch size">{opts.map((p) => <button key={p} type="button" role="tab" aria-selected={p === ps} className={'wg-tab' + (p === ps ? ' on' : '')} onClick={() => { setPs(p); setSel(0); }}>{p}×{p} patches</button>)}</div>
      <div className="vz-two">
        <div className="vz-patchimg" style={{ gridTemplateColumns: `repeat(${g}, 1fr)` }}>
          {Array.from({ length: g * g }, (_, k) => {
            const r = Math.floor(k / g), c = k % g;
            return (
              <button key={k} type="button" className={'vz-patch' + (k === sI ? ' sel' : '')} onClick={() => setSel(k)} aria-label={`Patch ${k + 1} of ${g * g}`}>
                <span className="vz-px" style={{ gridTemplateColumns: `repeat(${ps}, 1fr)` }}>
                  {Array.from({ length: ps * ps }, (_, q) => { const v = img[r * ps + Math.floor(q / ps)][c * ps + (q % ps)]; return <i key={q} style={{ background: `rgb(${255 - v * 28},${255 - v * 28},${255 - v * 28})` }} />; })}
                </span>
                <b>{k + 1}</b>
              </button>
            );
          })}
        </div>
        <div>
          <p className="wg-p" style={{ margin: '0 0 6px' }}><strong>{g * g} patches</strong> → a sequence of <strong>{g * g}</strong> tokens, each a flattened vector of length {ps}×{ps}×{C} = <strong>{ps * ps * C}</strong>.</p>
          <p className="wg-p" style={{ margin: '0 0 6px' }} aria-live="polite">Patch <strong>{sI + 1}</strong> (row {pr + 1}, column {pc + 1}) flattened:</p>
          <p className="vz-calc">[{flat.join(', ')}]</p>
          <p className="wg-note">A learned linear layer turns each vector into an embedding. Because attention ignores order, a position embedding (patch index {sI + 1}) is added so the model knows where the patch came from.</p>
        </div>
      </div>
      {spec.note ? <p className="wg-note">{spec.note}</p> : null}
    </div>
  );
}

// ---------------------------------------------------------------- diffusion (forward noising and an idealised reverse)
function Diffusion({ spec, fig }) {
  const img = spec.image;
  const n = img.length;
  const x0 = useMemo(() => img.flatMap((r) => [...String(r)].map((c) => (Number(c) / 9) * 2 - 1)), [img]);
  const eps = useMemo(() => gaussNoise(x0.length, spec.seed || 11), [x0, spec.seed]);
  const [t, setT] = useState(spec.t ?? 300);
  const [play, setPlay] = useState(false);
  const ref = useRef(t);
  ref.current = t;
  useEffect(() => {
    if (!play) return undefined;
    const id = setInterval(() => { if (ref.current <= 0) { setPlay(false); return; } setT(Math.max(0, ref.current - 20)); }, 90);
    return () => clearInterval(id);
  }, [play]);
  const ab = alphaBar(t);
  const xt = x0.map((v, i) => Math.sqrt(ab) * v + Math.sqrt(1 - ab) * eps[i]);
  const shade = (v) => { const g = Math.round(clamp((v + 1) / 2, 0, 1) * 255); return `rgb(${g},${g},${g})`; };
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <Slider label="Noise step t (of 1000)" min={0} max={1000} step={10} value={t} onChange={(v) => { setPlay(false); setT(v); }} />
      <div className="vz-two">
        <div className="vz-dimg" style={{ gridTemplateColumns: `repeat(${n}, 1fr)` }} role="img" aria-label={`The image at noise step ${t}. Signal strength ${f2(Math.sqrt(ab) * 100, 0)} percent.`}>
          {xt.map((v, i) => <i key={i} style={{ background: shade(v) }} />)}
        </div>
        <div>
          <p className="wg-p" aria-live="polite" style={{ margin: '0 0 8px' }}>At <strong>t = {t}</strong>: signal kept <strong>√ᾱₜ = {f2(Math.sqrt(ab), 3)}</strong>, noise added <strong>√(1−ᾱₜ) = {f2(Math.sqrt(1 - ab), 3)}</strong>.</p>
          <p className="vz-calc">xₜ = √ᾱₜ · x₀ + √(1−ᾱₜ) · ε</p>
          <div className="wg-row" style={{ marginTop: 10 }}>
            <button type="button" className="wg-btn" onClick={() => { if (t <= 0) setT(1000); setPlay(!play); }}>{play ? 'Pause' : t <= 0 ? 'Replay from noise' : 'Play reverse (denoise)'}</button>
            <button type="button" className="wg-btn ghost" onClick={() => { setPlay(false); setT(0); }}>Clean image</button>
          </div>
        </div>
      </div>
      <p className="wg-note">Forward process: a linear noise schedule (β from 0.0001 to 0.02 over 1000 steps, as in the original DDPM paper) mixes the image with fixed random noise. The “reverse” button replays the same path backwards as if the denoiser predicted the noise perfectly. A real model has to learn to predict ε at each step.{spec.note ? ' ' + spec.note : ''}</p>
    </div>
  );
}

// ---------------------------------------------------------------- retrieval (TF-IDF search over a few documents)
function Retrieval({ spec, fig }) {
  const [q, setQ] = useState(spec.query || '');
  const k = spec.topK || 2;
  const ranked = useMemo(() => tfidfRank(spec.docs, q), [spec, q]);
  const used = ranked.slice(0, k).filter((r) => r.score > 0);
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <label className="vz-try"><span>Question:</span><input type="text" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Question to search with" style={{ flex: 1, minWidth: 220 }} /></label>
      {spec.examples ? <div className="wg-row" style={{ margin: '8px 0' }}>{spec.examples.map((e) => <button key={e} type="button" className="wg-chip" onClick={() => setQ(e)}>{e}</button>)}</div> : null}
      <ol className="vz-rank" aria-live="polite">
        {ranked.map((r, i) => (
          <li key={r.d.id} className={i < k && r.score > 0 ? 'top' : ''}>
            <div className="vz-rank-h"><strong>{r.d.title}</strong><span>{f2(r.score, 2)}</span></div>
            <div className="wg-bar-track" style={{ height: 8 }}><span className="wg-bar-fill" style={{ width: `${clamp(r.score, 0, 1) * 100}%` }} /></div>
            <p className="wg-note" style={{ margin: '6px 0 0' }}>{r.d.text}{r.hits.length ? <> <em>Matches: {r.hits.join(', ')}</em></> : null}</p>
          </li>
        ))}
      </ol>
      <p className="wg-p" style={{ margin: '12px 0 4px' }}><strong>Prompt sent to the model</strong> (top {k} passages as context):</p>
      <pre className="vz-prompt">{`Answer using only the context below. If it is not there, say you do not know.\n\nContext:\n${used.length ? used.map((r) => '- ' + r.d.text).join('\n') : '(nothing matched)'}\n\nQuestion: ${q}`}</pre>
      <p className="wg-note">Passages are ranked by TF-IDF cosine similarity (words that are rare across the documents count for more). Real RAG systems usually use dense embeddings instead, but the pipeline is the same: retrieve, add to the prompt, generate.{spec.note ? ' ' + spec.note : ''}</p>
    </div>
  );
}


// ---------------------------------------------------------------- attention (query, keys, values -> output)
// One vector drawn as a stack of shaded cells with the numbers inside.
function VecCells({ x, y, vals, color, w = 40, h = 17 }) {
  return (
    <g>
      {vals.map((v, i) => (
        <g key={i}>
          <rect x={x} y={y + i * h} width={w} height={h} fill={color} fillOpacity={0.28 + 0.62 * clamp(Math.abs(v) / 2, 0, 1)} stroke={color} strokeWidth="1" rx="3" />
          <text x={x + w / 2} y={y + i * h + h / 2 + 3.5} textAnchor="middle" className="vz-vn">{f2(v, 1)}</text>
        </g>
      ))}
    </g>
  );
}

function Attention({ spec, fig }) {
  const [qi, setQi] = useState(0);
  const [scale, setScale] = useState(spec.scale !== false);
  const [hi, setHi] = useState(null);
  const n = spec.tokens.length;
  const q = spec.queries[qi];
  const { raw, scores, alpha, out } = useMemo(() => attend(q.q, spec.keys, spec.values, scale), [spec, qi, scale]);
  const d = q.q.length;
  const dv = spec.values[0].length;
  const colW = 118;
  const W = n * colW + 150;
  const cx = (i) => 50 + i * colW + colW / 2 - 20;
  const qh = d * 17, kh = d * 17, vh = dv * 17;
  const qY = 8, kY = 112, vY = kY + kh + 44, tokY = vY + vh + 14;
  const H = tokY + 38;
  const qx = (n * colW) / 2 + 30 - 20;
  const outX = n * colW + 62;
  const best = alpha.indexOf(Math.max(...alpha));
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <div className="wg-row" role="group" aria-label="Choose the query" style={{ marginBottom: 10 }}>
        {spec.queries.map((x, i) => <button key={x.label} type="button" className={'wg-chip' + (i === qi ? ' on' : '')} onClick={() => setQi(i)}>{x.label}</button>)}
        <label className="vz-check"><input type="checkbox" checked={scale} onChange={(e) => setScale(e.target.checked)} /> divide by √d</label>
      </div>
      <div className="vz-hm-wrap">
        <svg viewBox={`0 0 ${W} ${H}`} className="vz-att" role="img" aria-label={`Attention for the query “${q.label}”: the query is compared with each key, the scores become weights that sum to 1, and the output is the weighted sum of the values. The largest weight, ${f2(alpha[best], 2)}, is on “${spec.tokens[best]}”.`}>
          <text x="6" y={qY + qh / 2 + 4} className="vz-rl">Query</text>
          <text x="6" y={kY + kh / 2 + 4} className="vz-rl">Keys</text>
          <text x="6" y={vY + vh / 2 + 4} className="vz-rl">Values</text>
          <VecCells x={qx} y={qY} vals={q.q} color="var(--accent)" />
          {spec.tokens.map((t, i) => {
            const a = alpha[i];
            const dotY = kY + kh / 2;
            const x = cx(i);
            return (
              <g key={i} onMouseEnter={() => setHi(i)} onMouseLeave={() => setHi(null)} opacity={hi === null || hi === i ? 1 : 0.45}>
                <path d={`M ${qx + 20} ${qY + qh} C ${qx + 20} ${qY + qh + 60}, ${x + 62} ${dotY - 62}, ${x + 62} ${dotY - 11}`} fill="none" stroke="var(--text2)" strokeWidth={1 + 4 * a} strokeOpacity={0.3 + 0.7 * a} strokeDasharray="5 4" />
                <VecCells x={x} y={kY} vals={spec.keys[i]} color="var(--amber)" />
                <circle cx={x + 62} cy={dotY} r="10" fill="var(--surface2)" stroke="var(--text2)" />
                <circle cx={x + 62} cy={dotY} r="1.8" fill="var(--text)" />
                <text x={x + 62} y={dotY + 25} textAnchor="middle" className="vz-sc">{f2(scores[i], 2)}</text>
                <path d={`M ${x + 62} ${dotY + 33} L ${x + 62} ${vY + 6}`} stroke="var(--text2)" strokeWidth="1.5" strokeDasharray="4 3" fill="none" markerEnd="url(#vz-arr)" />
                <VecCells x={x} y={vY} vals={spec.values[i]} color="var(--indigo)" />
                <text x={x + 45} y={vY + vh / 2 + 4} className="vz-al">·α{i + 1}={f2(a, 2)}</text>
                {i < n - 1 ? <text x={x + colW - 14} y={vY + vh / 2 + 4} className="vz-al">+</text> : null}
                <rect x={x - 12} y={tokY} width={colW - 6} height="28" rx="6" fill={i === best ? 'var(--accent-soft-bg)' : 'var(--surface2)'} stroke={i === best ? 'var(--accent)' : 'var(--surface2-border)'} />
                <text x={x - 12 + (colW - 6) / 2} y={tokY + 18} textAnchor="middle" className="vz-tk">{t}</text>
              </g>
            );
          })}
          <defs><marker id="vz-arr" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="var(--text2)" /></marker></defs>
          <text x={outX - 18} y={vY + vh / 2 + 5} className="vz-eq">=</text>
          <VecCells x={outX} y={vY + (vh - dv * 17) / 2} vals={out} color="var(--text3)" />
          <text x={outX + 20} y={vY + vh / 2 + dv * 8.5 + 20} textAnchor="middle" className="vz-rl">Output</text>
        </svg>
      </div>
      <div className="vz-hm-wrap">
        <table className="vz-att-t" aria-label="Scores and attention weights">
          <thead><tr><th scope="row">Token</th>{spec.tokens.map((t) => <th key={t} scope="col">{t}</th>)}</tr></thead>
          <tbody>
            <tr><th scope="row">score q·k</th>{raw.map((x, i) => <td key={i}>{f2(x, 2)}</td>)}</tr>
            {scale ? <tr><th scope="row">÷ √{d}</th>{scores.map((x, i) => <td key={i}>{f2(x, 2)}</td>)}</tr> : null}
            <tr><th scope="row">weight α</th>{alpha.map((x, i) => <td key={i} className={i === best ? 'hi' : ''}>{f2(x, 2)}</td>)}</tr>
          </tbody>
        </table>
      </div>
      <p className="wg-p" aria-live="polite" style={{ margin: '10px 0 0' }}>
        The weights sum to <strong>{f2(alpha.reduce((a, b) => a + b, 0), 2)}</strong>. “{q.label}” attends most to <strong>{spec.tokens[best]}</strong> ({f2(alpha[best] * 100, 0)}%). Output = {alpha.map((a, i) => `${f2(a, 2)}·v${i + 1}`).join(' + ')} = <strong>[{out.map((x) => f2(x, 2)).join(', ')}]</strong>.
      </p>
      <p className="wg-note">{spec.note || 'Illustrative numbers authored for teaching, not taken from a trained model. Hover a column to focus it.'}</p>
    </div>
  );
}

// ---------------------------------------------------------------- equation (formula with named parts, plus a worked example)
function Equation({ spec, fig }) {
  const [part, setPart] = useState(spec.parts[0].id);
  const [step, setStep] = useState(0);
  const steps = spec.steps || [];
  const sel = spec.parts.find((p) => p.id === part) || spec.parts[0];
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <div className="vz-formula" role="math" aria-label={spec.aria || spec.formula.map((f) => (typeof f === 'string' ? f : f.t)).join('')}>
        {spec.formula.map((f, i) => {
          if (typeof f === 'string') return <span key={i}>{f}</span>;
          return <button key={i} type="button" className={'vz-term' + (f.id === part ? ' on' : '')} onClick={() => setPart(f.id)} aria-pressed={f.id === part}>{f.t}</button>;
        })}
      </div>
      <div className="vz-part" aria-live="polite"><strong>{sel.name}</strong> {sel.meaning}</div>
      {steps.length ? (
        <div className="vz-steps">
          <div className="wg-row" style={{ marginBottom: 8 }}>
            {steps.map((s, i) => <button key={i} type="button" className={'wg-chip' + (i === step ? ' on' : '')} onClick={() => setStep(i)}>{i + 1}. {s.title}</button>)}
          </div>
          <div className="vz-step-b"><code>{steps[step].expr}</code><p className="wg-p" style={{ margin: '6px 0 0' }}>{steps[step].note}</p></div>
        </div>
      ) : null}
      {spec.note ? <p className="wg-note">{spec.note}</p> : null}
    </div>
  );
}

// ---------------------------------------------------------------- nested (AI > machine learning > deep learning > generative AI)
function Nested({ spec, fig }) {
  const L = spec.levels;
  const [i, setI] = useState(L.length - 1);
  const sel = L[i];
  const N = L.length;
  const R = 150;
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <div className="vz-two">
        <svg viewBox="0 0 320 320" className="vz-sq" role="img" aria-label={`Nested circles: ${L.map((l) => l.name).join(' contains ')}`}>
          {L.map((l, k) => {
            const r = R - k * (R / (N + 0.8));
            return (
              <g key={l.name} onClick={() => setI(k)} style={{ cursor: 'pointer' }}>
                <circle cx="160" cy={310 - r} r={r} fill={k === i ? 'var(--accent-soft-bg)' : 'var(--surface2)'} stroke={k === i ? 'var(--accent)' : 'var(--surface2-border)'} strokeWidth={k === i ? 3 : 1.5} />
                <text x="160" y={310 - 2 * r + 24} textAnchor="middle" className="vz-lab" style={{ fontSize: 13 }}>{l.name}</text>
              </g>
            );
          })}
        </svg>
        <div>
          <div className="wg-row" style={{ marginBottom: 10 }}>
            {L.map((l, k) => <button key={l.name} type="button" className={'wg-chip' + (k === i ? ' on' : '')} aria-pressed={k === i} onClick={() => setI(k)}>{l.name}</button>)}
          </div>
          <p className="wg-p" aria-live="polite" style={{ margin: '0 0 8px' }}><strong>{sel.name}.</strong> {sel.blurb}</p>
          {sel.examples ? <p className="wg-note" style={{ marginTop: 0 }}>Examples: {sel.examples.join(' · ')}</p> : null}
        </div>
      </div>
      {spec.note ? <p className="wg-note">{spec.note}</p> : null}
    </div>
  );
}

// ---------------------------------------------------------------- sampling (softmax with temperature, then top-k and top-p)
function Sampling({ spec, fig }) {
  const toks = spec.tokens;
  const [T, setT] = useState(1);
  const [k, setK] = useState(0);
  const [p, setP] = useState(1);
  const [text, setText] = useState([]);
  const [tally, setTally] = useState(null);
  const base = useMemo(() => softmaxT(toks.map((x) => x.logit), T), [toks, T]);
  const f = useMemo(() => filterSampling(base, k, p), [base, k, p]);
  const top = base.indexOf(Math.max(...base));
  const z = toks.reduce((a, x) => a + Math.exp((x.logit - Math.max(...toks.map((y) => y.logit))) / T), 0);
  const draw = () => {
    let r = Math.random();
    for (let i = 0; i < f.probs.length; i++) { r -= f.probs[i]; if (r <= 0) return i; }
    return f.probs.findIndex((x) => x > 0);
  };
  const reset = () => { setTally(null); };
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <p className="wg-p" style={{ margin: '0 0 8px' }}>After <strong>“{spec.prompt}”</strong> the model gives every word a score (a logit z). Three dials decide which word is picked.</p>
      <Slider label="Temperature T" min={0.1} max={2} step={0.05} value={T} onChange={(v) => { setT(v); reset(); }} show={(v) => v.toFixed(2)} />
      <Slider label="Top-k (0 = off)" min={0} max={toks.length} step={1} value={k} onChange={(v) => { setK(v); reset(); }} show={(v) => (v === 0 ? 'off' : v)} />
      <Slider label="Top-p (1 = off)" min={0.1} max={1} step={0.05} value={p} onChange={(v) => { setP(v); reset(); }} show={(v) => (v >= 1 ? 'off' : v.toFixed(2))} />
      <div className="vz-formula" role="math" aria-label="softmax with temperature">
        <span>p(</span><span className="vz-term static">{toks[top].t}</span><span>) = e</span><sup>z/T</sup><span> / Σ e</span><sup>z/T</sup><span> = 1 / Σ e</span><sup>(z − z</sup><sub>top</sub><sup>)/T</sup><span> = 1 / {f2(z, 2)} = <strong>{f2(base[top], 3)}</strong></span>
      </div>
      <ul className="wg-bars vz-samp" aria-live="polite">
        {toks.map((x, i) => (
          <li key={x.t} className={f.keep[i] ? '' : 'cut'} style={{ gridTemplateColumns: '84px 62px minmax(0,1fr) 62px 40px' }}>
            <span className="wg-bar-l">{x.t}</span>
            <span className="wg-bar-p" title="logit z">z={x.logit}</span>
            <span className="wg-bar-track"><span className="vz-ghost" style={{ width: `${base[i] * 100}%` }} /><span className="wg-bar-fill" style={{ width: `${f.probs[i] * 100}%` }} /></span>
            <span className="wg-bar-p">{f.keep[i] ? f2(f.probs[i] * 100, 1) + '%' : 'cut'}</span>
            {tally ? <span className="wg-bar-c">{tally[i]}×</span> : <span />}
          </li>
        ))}
      </ul>
      <div className="wg-row">
        <button type="button" className="wg-btn" onClick={() => setText((t) => [...t, toks[draw()].t])}>Pick the next word</button>
        <button type="button" className="wg-btn ghost" onClick={() => { const c = toks.map(() => 0); for (let n = 0; n < 100; n++) c[draw()]++; setTally(c); }}>Draw 100 times</button>
        <button type="button" className="wg-btn ghost" onClick={() => { setText([]); setTally(null); }} disabled={!text.length && !tally}>Clear</button>
      </div>
      <p className="wg-sent" aria-live="polite">{spec.prompt}{text.map((w, i) => <span key={i} className="wg-picked"> {w}</span>)}…</p>
      <p className="wg-note">The faint bar is the probability before top-k/top-p; the solid bar is what remains after cutting and renormalising so the kept words add up to 100%. {spec.note || 'The scores are made up for teaching; they are not from a real model.'}</p>
    </div>
  );
}

// ---------------------------------------------------------------- chain (P(sentence) = product of next-word probabilities)
function Chain({ spec, fig }) {
  const [s, setS] = useState(0);
  const S = spec.sentences[s];
  const ps = S.tokens.map((t) => t.p);
  const st = chainStats(ps);
  const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
  const sci = (x) => {
    if (x >= 0.001) return f2(x, 4);
    const [m, e] = x.toExponential(2).split('e');
    return `${m} × 10${[...String(Number(e))].map((c) => SUP[c] || c).join('')}`;
  };
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <div className="wg-tabs" role="tablist">{spec.sentences.map((x, i) => <button key={x.label} type="button" role="tab" aria-selected={i === s} className={'wg-tab' + (i === s ? ' on' : '')} onClick={() => setS(i)}>{x.label}</button>)}</div>
      <div className="vz-formula" role="math" aria-label="chain rule"><span>P(w</span><sub>1</sub><span> … w</span><sub>n</sub><span>) = P(w</span><sub>1</sub><span>) × P(w</span><sub>2</sub><span> | w</span><sub>1</sub><span>) × … × P(w</span><sub>n</sub><span> | w</span><sub>1…n−1</sub><span>)</span></div>
      <ol className="vz-chain-w" aria-label="Words and their probabilities">
        {S.tokens.map((t, i) => (
          <li key={i}>
            <span className="vz-chain-bar" style={{ height: `${Math.max(6, t.p * 90)}px` }} />
            <span className="vz-chain-t">{t.t}</span>
            <span className="vz-chain-p">{f2(t.p, 2)}</span>
            <span className="vz-chain-r" title="running product">× → {sci(st.running[i])}</span>
          </li>
        ))}
      </ol>
      <div className="vz-cmp">
        <div><span className="wg-k">Probability of the whole sentence</span><span className="vz-big">{sci(st.P)}</span><span className="wg-note">the product of the {ps.length} numbers above</span></div>
        <div className="out"><span className="wg-k">Perplexity (lower = less surprised)</span><span className="vz-big">{f2(st.perplexity, 2)}</span><span className="wg-note">exp(−average ln p) = “about {f2(st.perplexity, 1)} equally likely choices per word”</span></div>
      </div>
      {S.note ? <p className="wg-p">{S.note}</p> : null}
      <p className="wg-note">{spec.note || 'Probabilities are illustrative numbers authored for teaching, not measured from a real model.'}</p>
    </div>
  );
}

// ---------------------------------------------------------------- logbars (sizes on a log scale)
function LogBars({ spec, fig }) {
  const items = [...spec.items].sort((a, b) => a.value - b.value);
  const [sel, setSel] = useState(items.length - 1);
  const lo = Math.log10(items[0].value) - 0.4;
  const hi = Math.log10(items[items.length - 1].value);
  const it = items[sel];
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <ul className="vz-lb" aria-label="Sizes on a logarithmic scale">
        {items.map((x, i) => (
          <li key={x.label}>
            <button type="button" className={'vz-lb-b' + (i === sel ? ' on' : '')} aria-pressed={i === sel} onClick={() => setSel(i)}>
              <span className="vz-lb-l">{x.label}{x.year ? <em> {x.year}</em> : null}</span>
              <span className="wg-bar-track"><span className="wg-bar-fill" style={{ width: `${((Math.log10(x.value) - lo) / (hi - lo)) * 100}%` }} /></span>
              <span className="vz-lb-v">{human(x.value)}</span>
            </button>
          </li>
        ))}
      </ul>
      <p className="wg-p" aria-live="polite" style={{ margin: '10px 0 0' }}><strong>{it.label}</strong>: {human(it.value)} {spec.unit || ''}. {it.note || ''}</p>
      <p className="wg-note">Each step along the axis is ten times bigger, so a short bar can still be a thousand times larger than a shorter one. {spec.note || ''}</p>
    </div>
  );
}

// ---------------------------------------------------------------- network (a small neural network with a live forward pass)
function Network({ spec, fig }) {
  const L = spec.layers;
  const [W, setW] = useState(() => spec.weights.map((m) => m.map((r) => r.slice())));
  const [B, setB] = useState(() => spec.biases.map((r) => r.slice()));
  const [x, setX] = useState(spec.inputs.slice());
  const [act, setAct] = useState(spec.activation || 'relu');
  const [sel, setSel] = useState([L.length - 1, 0]);
  const out = spec.outputActivation || 'linear';
  const { a, z } = useMemo(() => forwardNet(L, W, B, x, act, out), [L, W, B, x, act, out]);
  const colW = 640 / L.length;
  const pos = (l, j) => [colW * l + colW / 2 - 40 + 40, 30 + ((j + 0.5) * 230) / L[l]];
  const [sl, sj] = sel;
  const incoming = sl > 0 ? W[sl - 1][sj] : null;
  const setWeight = (i, v) => setW((w) => w.map((m, l) => (l === sl - 1 ? m.map((r, j) => (j === sj ? r.map((q, ii) => (ii === i ? v : q)) : r)) : m)));
  const setBias = (v) => setB((bb) => bb.map((r, l) => (l === sl - 1 ? r.map((q, j) => (j === sj ? v : q)) : r)));
  const fmt = (n) => (Math.abs(n) < 1e-9 ? '0' : Number(n.toFixed(2)).toString());
  const reset = () => { setW(spec.weights.map((m) => m.map((r) => r.slice()))); setB(spec.biases.map((r) => r.slice())); };
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <div className="wg-row" style={{ marginBottom: 10 }}>
        {(spec.presets || []).map((p) => <button key={p.name} type="button" className={'wg-chip' + (p.inputs.join() === x.join() ? ' on' : '')} onClick={() => setX(p.inputs.slice())}>{p.name}</button>)}
        {['relu', 'tanh', 'sigmoid', 'linear'].map((k) => <button key={k} type="button" className={'wg-chip' + (act === k ? ' on' : '')} aria-pressed={act === k} onClick={() => setAct(k)}>hidden: {k}</button>)}
      </div>
      <svg viewBox="0 0 640 290" className="vz-netsvg" role="img" aria-label={`Neural network with layers ${L.join(', ')}. Output ${a[a.length - 1].map((v) => v.toFixed(2)).join(', ')}.`}>
        {W.map((m, l) => m.map((row, j) => row.map((w, i) => {
          const [x1, y1] = pos(l, i);
          const [x2, y2] = pos(l + 1, j);
          const hot = sl === l + 1 && sj === j;
          return (
            <g key={`${l}-${j}-${i}`}>
              <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={w >= 0 ? 'var(--accent)' : 'var(--cyan)'} strokeWidth={Math.min(7, 0.8 + Math.abs(w) * 2)} opacity={hot ? 1 : 0.55} />
              {spec.showWeights !== false && L[l] * L[l + 1] <= 9 ? <text x={(x1 + x2) / 2 + (i - (row.length - 1) / 2) * 4} y={(y1 + y2) / 2 - 3 + (i - (row.length - 1) / 2) * 7} className="vz-tick" textAnchor="middle" style={{ fontSize: 10 }}>{fmt(w)}</text> : null}
            </g>
          );
        })))}
        {L.map((n, l) => Array.from({ length: n }, (_, j) => {
          const [cx, cy] = pos(l, j);
          const v = a[l][j];
          const on = sl === l && sj === j;
          return (
            <g key={`${l}-${j}`} onClick={() => setSel([l, j])} style={{ cursor: 'pointer' }} role="button" tabIndex={0} aria-label={`Unit ${j + 1} of layer ${l + 1}, value ${fmt(v)}`} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setSel([l, j]); }}>
              <circle cx={cx} cy={cy} r="24" fill={`color-mix(in srgb, var(--accent) ${Math.round(Math.min(1, Math.abs(v)) * 60)}%, var(--card-bg))`} stroke={on ? 'var(--text)' : 'var(--surface2-border)'} strokeWidth={on ? 3 : 1.5} />
              <text x={cx} y={cy + 4} textAnchor="middle" className="vz-lab" style={{ fontSize: 12.5 }}>{fmt(v)}</text>
              {l === 0 && spec.inputLabels ? <text x={cx - 38} y={cy + 4} textAnchor="end" className="vz-tick">{spec.inputLabels[j]}</text> : null}
              {l === L.length - 1 && spec.outputLabels ? <text x={cx + 38} y={cy + 4} className="vz-tick">{spec.outputLabels[j]}</text> : null}
            </g>
          );
        }))}
        {L.map((n, l) => <text key={l} x={colW * l + colW / 2} y="284" textAnchor="middle" className="vz-tick">{l === 0 ? 'input' : l === L.length - 1 ? 'output' : `hidden ${l}`}</text>)}
      </svg>
      <div className="vz-two">
        <div>
          <p className="wg-p" style={{ margin: '0 0 6px' }}><strong>Inputs</strong> (drag to change)</p>
          {x.map((v, i) => <Slider key={i} label={spec.inputLabels ? spec.inputLabels[i] : `x${i + 1}`} min={-2} max={2} step={0.1} value={v} onChange={(q) => setX((xx) => xx.map((o, k) => (k === i ? q : o)))} show={(q) => q.toFixed(1)} />)}
          <p className="wg-note">Parameters in this network: <strong>{countParams(L)}</strong> = {L.slice(1).map((n, i) => `${L[i]}×${n}+${n}`).join(' + ')}.</p>
        </div>
        <div aria-live="polite">
          {sl === 0 ? <p className="wg-p"><strong>Input unit {sj + 1}</strong>: its value is the number you set with the slider: {fmt(x[sj])}.</p> : (
            <>
              <p className="wg-p" style={{ margin: '0 0 6px' }}><strong>Unit {sj + 1} of {sl === L.length - 1 ? 'the output layer' : `hidden layer ${sl}`}</strong></p>
              <p className="vz-calc">z = {incoming.map((w, i) => `${fmt(w)}×${fmt(a[sl - 1][i])}`).join(' + ')} + {fmt(B[sl - 1][sj])} = {fmt(z[sl][sj])}</p>
              <p className="vz-calc">a = {sl === L.length - 1 ? out : act}({fmt(z[sl][sj])}) = <strong>{fmt(a[sl][sj])}</strong></p>
              {spec.editable !== false ? (
                <>
                  {incoming.map((w, i) => <Slider key={i} label={`weight from unit ${i + 1}`} min={-3} max={3} step={0.1} value={w} onChange={(q) => setWeight(i, q)} show={(q) => q.toFixed(1)} />)}
                  <Slider label="bias" min={-3} max={3} step={0.1} value={B[sl - 1][sj]} onChange={setBias} show={(q) => q.toFixed(1)} />
                  <button type="button" className="wg-btn ghost" onClick={reset}>Reset weights</button>
                </>
              ) : null}
            </>
          )}
        </div>
      </div>
      <p className="wg-note">Select any circle to see its arithmetic. Green lines are positive weights, blue are negative; thicker means larger. {spec.note || ''}</p>
    </div>
  );
}

// ---------------------------------------------------------------- fit (underfitting and overfitting with a polynomial)
function Fit({ spec, fig }) {
  const [deg, setDeg] = useState(spec.degree ?? 1);
  const maxD = spec.maxDegree || 9;
  const fits = useMemo(() => Array.from({ length: maxD + 1 }, (_, d) => { const c = polyFit(spec.train.map((p) => p.x), spec.train.map((p) => p.y), d); return { c, tr: polyMse(c, spec.train), te: polyMse(c, spec.test) }; }), [spec, maxD]);
  const best = fits.reduce((bi, f, i) => (f.te < fits[bi].te ? i : bi), 0);
  const cur = fits[deg];
  const xs = [...spec.train, ...spec.test].map((p) => p.x), ys = [...spec.train, ...spec.test].map((p) => p.y);
  const x0 = Math.min(...xs, -1), x1 = Math.max(...xs, 1);
  const y0 = Math.min(...ys) - 0.5, y1 = Math.max(...ys) + 0.5;
  const W = 560, H = 250, pad = 28;
  const sx = (x) => pad + ((x - x0) / (x1 - x0)) * (W - 2 * pad);
  const sy = (y) => H - pad - ((Math.max(y0 - 1, Math.min(y1 + 1, y)) - y0) / (y1 - y0)) * (H - 2 * pad);
  const curve = Array.from({ length: 121 }, (_, i) => { const x = x0 + ((x1 - x0) * i) / 120; return `${i ? 'L' : 'M'}${sx(x).toFixed(1)} ${sy(polyEval(cur.c, x)).toFixed(1)}`; }).join(' ');
  const mx = Math.max(cur.te, cur.tr, 1e-6);
  const verdict = deg < best ? 'Underfitting: the curve is too simple to follow the pattern.' : deg === best ? 'A good fit: it follows the pattern and also does well on new points.' : 'Overfitting: it chases the noise in the training points, so it does worse on new points.';
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <Slider label="Model complexity: polynomial degree" min={0} max={maxD} step={1} value={deg} onChange={setDeg} />
      <svg viewBox={`0 0 ${W} ${H}`} className="vz-plot" role="img" aria-label={`A degree ${deg} curve fitted to the training points. Training error ${cur.tr.toFixed(3)}, test error ${cur.te.toFixed(3)}.`}>
        <line x1={pad} x2={W - pad} y1={sy(0)} y2={sy(0)} className="vz-grid" />
        <path d={curve} fill="none" stroke="var(--accent)" strokeWidth="2.6" />
        {spec.train.map((p, i) => <circle key={'t' + i} cx={sx(p.x)} cy={sy(p.y)} r="5" fill="var(--text)" />)}
        {spec.test.map((p, i) => <circle key={'e' + i} cx={sx(p.x)} cy={sy(p.y)} r="5" fill="none" stroke="var(--amber)" strokeWidth="2.2" />)}
      </svg>
      <div className="wg-row vz-legend"><span className="vz-key"><i style={{ background: 'var(--text)', height: 10, width: 10, borderRadius: 99 }} />training points (the model sees these)</span><span className="vz-key"><i style={{ border: '2px solid var(--amber)', height: 10, width: 10, borderRadius: 99, background: 'none' }} />test points (hidden from the model)</span></div>
      <ul className="wg-bars" style={{ marginTop: 10 }}>
        <li style={{ gridTemplateColumns: '120px minmax(0,1fr) 70px' }}><span className="wg-bar-l" style={{ fontFamily: 'inherit' }}>Training error</span><span className="wg-bar-track"><span className="wg-bar-fill" style={{ width: `${(cur.tr / mx) * 100}%` }} /></span><span className="wg-bar-p">{cur.tr.toFixed(3)}</span></li>
        <li style={{ gridTemplateColumns: '120px minmax(0,1fr) 70px' }}><span className="wg-bar-l" style={{ fontFamily: 'inherit' }}>Test error</span><span className="wg-bar-track"><span className="wg-bar-fill warn" style={{ width: `${(cur.te / mx) * 100}%` }} /></span><span className="wg-bar-p">{cur.te.toFixed(3)}</span></li>
      </ul>
      <p className="wg-p" aria-live="polite" style={{ margin: '8px 0 0' }}><strong>Degree {deg}.</strong> {verdict}</p>
      <p className="wg-note">Error is the mean squared error. The lowest test error here is at degree {best}. {spec.note || 'The points are authored for teaching.'}</p>
    </div>
  );
}

// ---------------------------------------------------------------- trainer (a tiny network learning in your browser)
const SETS = ['xor', 'circle', 'moons', 'line'];
function Trainer({ spec, fig }) {
  const [ds, setDs] = useState(spec.dataset || 'xor');
  const [H, setH] = useState(spec.hidden ?? 4);
  const [lr, setLr] = useState(spec.lr ?? 0.5);
  const [run, setRun] = useState(false);
  const [, force] = useState(0);
  const data = useMemo(() => makeDataset(ds), [ds]);
  const st = useRef({ net: initNet(H, 1), epoch: 0, losses: [] });
  const reset = (h = H) => { st.current = { net: initNet(h, 1), epoch: 0, losses: [] }; force((n) => n + 1); };
  useEffect(() => { reset(H); }, [ds, H]); // eslint-disable-line react-hooks/exhaustive-deps
  const lrRef = useRef(lr); lrRef.current = lr;
  const step = (k) => {
    const s = st.current;
    for (let i = 0; i < k; i++) { trainEpoch(s.net, data.X, data.Y, lrRef.current); s.epoch++; }
    s.losses.push(netLossAcc(s.net, data.X, data.Y).loss);
    if (s.losses.length > 300) s.losses = s.losses.filter((_, i) => i % 2 === 0);
    force((n) => n + 1);
  };
  useEffect(() => { if (!run) return undefined; const id = setInterval(() => step(10), 40); return () => clearInterval(id); }, [run, ds]); // eslint-disable-line react-hooks/exhaustive-deps
  const { net, epoch, losses } = st.current;
  const { loss, acc } = netLossAcc(net, data.X, data.Y);
  const S = 300, G = 30;
  const toPx = (v) => ((v + 2) / 4) * S;
  const cells = [];
  for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) { const gx = -2 + ((i + 0.5) * 4) / G, gy = 2 - ((j + 0.5) * 4) / G; cells.push([i, j, predictNet(net, [gx, gy])]); }
  const lmax = Math.max(...losses, 0.7);
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <div className="wg-tabs" role="tablist">{SETS.map((k) => <button key={k} type="button" role="tab" aria-selected={ds === k} className={'wg-tab' + (ds === k ? ' on' : '')} onClick={() => { setRun(false); setDs(k); }}>{k}</button>)}</div>
      <div className="vz-two">
        <svg viewBox={`0 0 ${S} ${S}`} className="vz-sq" role="img" aria-label={`Decision regions of the network on the ${ds} data. Accuracy ${Math.round(acc * 100)} percent.`}>
          {cells.map(([i, j, p]) => <rect key={i + '-' + j} x={(i * S) / G} y={(j * S) / G} width={S / G + 0.5} height={S / G + 0.5} fill={`color-mix(in srgb, var(--accent) ${Math.round(p * 70)}%, var(--cyan-soft-bg))`} />)}
          {data.X.map((p, i) => (data.Y[i] ? <circle key={i} cx={toPx(p[0])} cy={S - toPx(p[1])} r="5" fill="var(--accent)" stroke="var(--text)" strokeWidth="1.5" /> : <rect key={i} x={toPx(p[0]) - 4.5} y={S - toPx(p[1]) - 4.5} width="9" height="9" fill="var(--cyan)" stroke="var(--text)" strokeWidth="1.5" />))}
        </svg>
        <div>
          <Slider label="Hidden units" min={1} max={8} step={1} value={H} onChange={(v) => { setRun(false); setH(v); }} />
          <Slider label="Learning rate" min={0.05} max={3} step={0.05} value={lr} onChange={setLr} show={(v) => v.toFixed(2)} />
          <p className="wg-p" aria-live="polite" style={{ margin: '6px 0' }}>Epoch <strong>{epoch}</strong> · loss <strong>{loss.toFixed(3)}</strong> · accuracy <strong>{Math.round(acc * 100)}%</strong></p>
          <svg viewBox="0 0 300 80" className="vz-plot" role="img" aria-label="Training loss over time" style={{ maxWidth: 320 }}>
            <line x1="4" x2="296" y1="76" y2="76" className="vz-grid" />
            <path d={losses.map((l, i) => `${i ? 'L' : 'M'}${4 + (i / Math.max(1, losses.length - 1)) * 292} ${76 - (l / lmax) * 70}`).join(' ')} fill="none" stroke="var(--accent)" strokeWidth="2.2" />
          </svg>
          <div className="wg-row">
            <button type="button" className="wg-btn" onClick={() => setRun((r) => !r)}>{run ? 'Pause' : 'Train'}</button>
            <button type="button" className="wg-btn ghost" onClick={() => step(1)}>1 epoch</button>
            <button type="button" className="wg-btn ghost" onClick={() => { setRun(false); reset(); }}>Reset</button>
          </div>
        </div>
      </div>
      <p className="wg-note">Circles are class 1, squares are class 0. The background shows what the network predicts everywhere: green means “class 1”. The network is 2 inputs → hidden tanh units → 1 sigmoid output, trained by full-batch gradient descent on the cross-entropy loss, all running in your browser. {spec.note || ''}</p>
    </div>
  );
}

// ---------------------------------------------------------------- mlp (count the parameters of a stack of layers)
function Mlp({ spec, fig }) {
  const [hid, setHid] = useState(spec.layers.slice(1, -1));
  const layers = [spec.layers[0], ...hid, spec.layers[spec.layers.length - 1]];
  const rows = layers.slice(1).map((n, i) => ({ from: layers[i], to: n, w: layers[i] * n, b: n }));
  const total = countParams(layers);
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      {hid.map((n, i) => <Slider key={i} label={`Hidden layer ${i + 1} width`} min={2} max={1024} step={2} value={n} onChange={(v) => setHid((h) => h.map((o, k) => (k === i ? v : o)))} />)}
      <table className="vz-mlp" aria-label="Parameters per layer">
        <thead><tr><th>Layer</th><th>Shape</th><th>Weights in × out</th><th>Biases</th><th>Parameters</th></tr></thead>
        <tbody>
          {rows.map((r, i) => <tr key={i}><td>{i + 1}</td><td>{r.from} → {r.to}</td><td>{r.from} × {r.to} = {r.w.toLocaleString()}</td><td>{r.b}</td><td><strong>{(r.w + r.b).toLocaleString()}</strong></td></tr>)}
        </tbody>
      </table>
      <div className="vz-cmp" style={{ marginTop: 12 }}><div className="out"><span className="wg-k">Total trainable parameters</span><span className="vz-big">{total.toLocaleString()}</span><span className="wg-note">{(total * 4 / 1e6).toFixed(2)} MB as 32-bit numbers</span></div><div><span className="wg-k">Rule</span><span className="wg-v">each layer has (inputs × outputs) weights plus one bias per output</span></div></div>
      {spec.note ? <p className="wg-note">{spec.note}</p> : null}
    </div>
  );
}

// ---------------------------------------------------------------- backprop (forward pass, backward pass, chain rule on a small network)
function Backprop({ spec, fig }) {
  const L = spec.layers;
  const init = () => [spec.weights.map((m) => m.map((r) => r.slice())), spec.biases.map((r) => r.slice())];
  const [[W, B], setWB] = useState(init);
  const [x] = useState(spec.inputs.slice());
  const [mode, setMode] = useState('forward');
  const [sel, setSel] = useState([L.length - 1, 0]);
  const [lr, setLr] = useState(spec.lr ?? 0.5);
  const [hist, setHist] = useState([]);
  const act = spec.activation || 'sigmoid', out = spec.outputActivation || 'sigmoid', loss = spec.loss || 'mse';
  const r = useMemo(() => backwardNet(L, W, B, x, spec.target, act, out, loss), [L, W, B, x, spec, act, out, loss]);
  const fmt = (n) => (Math.abs(n) < 5e-5 ? '0' : Math.abs(n) < 0.01 ? n.toExponential(1) : Number(n.toFixed(3)).toString());
  const colW = 640 / L.length;
  const pos = (l, j) => [colW * l + colW / 2, 30 + ((j + 0.5) * 230) / L[l]];
  const [sl, sj] = sel;
  const bwd = mode === 'backward';
  const maxD = Math.max(1e-9, ...r.delta.slice(1).flat().map(Math.abs));
  const step = () => { setHist((h) => [...h, r.loss]); setWB(([w, b]) => [w.map((m, l) => m.map((row, j) => row.map((q, i) => q - lr * r.dW[l][j][i]))), b.map((row, l) => row.map((q, j) => q - lr * r.dB[l][j]))]); };
  const reset = () => { setWB(init()); setHist([]); };
  const hl = sl > 0 ? (() => {
    const l = sl, j = sj;
    if (!bwd) return (<><p className="vz-calc">z = {W[l - 1][j].map((w, i) => `${fmt(w)}×${fmt(r.a[l - 1][i])}`).join(' + ')} + {fmt(B[l - 1][j])} = {fmt(r.z[l][j])}</p><p className="vz-calc">a = {l === L.length - 1 ? out : act}({fmt(r.z[l][j])}) = <strong>{fmt(r.a[l][j])}</strong></p></>);
    if (l === L.length - 1) return (<><p className="vz-calc">δ = dL/dz = (a − y) × φ′(z)</p><p className="vz-calc">δ = ({fmt(r.a[l][j])} − {fmt(spec.target[j])}) × {fmt(DACT[out](r.z[l][j], r.a[l][j]))} = <strong>{fmt(r.delta[l][j])}</strong></p></>);
    const terms = r.delta[l + 1].map((d, k) => `${fmt(W[l][k][j])}×${fmt(d)}`).join(' + ');
    return (<><p className="vz-calc">δ = (Σ w × δ_next) × φ′(z)</p><p className="vz-calc">δ = ({terms}) × {fmt(DACT[act](r.z[l][j], r.a[l][j]))} = <strong>{fmt(r.delta[l][j])}</strong></p></>);
  })() : <p className="wg-p">An input unit has no δ: its value is just the number you feed in ({fmt(x[sj])}).</p>;
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <div className="wg-tabs" role="tablist">{['forward', 'backward'].map((k) => <button key={k} type="button" role="tab" aria-selected={mode === k} className={'wg-tab' + (mode === k ? ' on' : '')} onClick={() => setMode(k)}>{k === 'forward' ? 'Forward pass →' : '← Backward pass'}</button>)}</div>
      <svg viewBox="0 0 640 290" className="vz-netsvg" role="img" aria-label={`${mode} pass through a network with layers ${L.join(', ')}; loss ${fmt(r.loss)}`}>
        <defs><marker id="bp-ar" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L8 4L0 8z" fill="context-stroke" /></marker></defs>
        {W.map((m, l) => m.map((row, j) => row.map((w, i) => {
          const [x1, y1] = pos(l, i), [x2, y2] = pos(l + 1, j);
          const g = r.dW[l][j][i];
          const hot = sl === l + 1 && sj === j;
          const wid = bwd ? 0.8 + Math.min(6, (Math.abs(g) / Math.max(1e-9, ...r.dW.flat(2).map(Math.abs))) * 6) : Math.min(7, 0.8 + Math.abs(w) * 2);
          const col = bwd ? 'var(--amber)' : w >= 0 ? 'var(--accent)' : 'var(--cyan)';
          return <line key={`${l}-${j}-${i}`} x1={bwd ? x2 : x1} y1={bwd ? y2 : y1} x2={bwd ? x1 : x2} y2={bwd ? y1 : y2} stroke={col} strokeWidth={wid} opacity={hot ? 1 : 0.5} />;
        })))}
        {L.map((n, l) => Array.from({ length: n }, (_, j) => {
          const [cx, cy] = pos(l, j);
          const val = bwd ? (l ? r.delta[l][j] : null) : r.a[l][j];
          const on = sl === l && sj === j;
          const inten = bwd ? (l ? Math.abs(val) / maxD : 0) : Math.min(1, Math.abs(val));
          return (
            <g key={`${l}-${j}`} onClick={() => setSel([l, j])} style={{ cursor: 'pointer' }} role="button" tabIndex={0} aria-label={`Unit ${j + 1} of layer ${l + 1}`} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setSel([l, j]); }}>
              <circle cx={cx} cy={cy} r="25" fill={`color-mix(in srgb, ${bwd ? 'var(--amber)' : 'var(--accent)'} ${Math.round(inten * 60)}%, var(--card-bg))`} stroke={on ? 'var(--text)' : 'var(--surface2-border)'} strokeWidth={on ? 3 : 1.5} />
              <text x={cx} y={cy + 4} textAnchor="middle" className="vz-lab" style={{ fontSize: 12 }}>{val == null ? fmt(x[j]) : fmt(val)}</text>
              {l === 0 && spec.inputLabels ? <text x={cx - 38} y={cy + 4} textAnchor="end" className="vz-tick">{spec.inputLabels[j]}</text> : null}
            </g>
          );
        }))}
        {L.map((n, l) => <text key={l} x={colW * l + colW / 2} y="284" textAnchor="middle" className="vz-tick">{l === 0 ? 'input' : l === L.length - 1 ? 'output' : `hidden ${l}`}</text>)}
        <text x="636" y="14" textAnchor="end" className="vz-tick">{bwd ? 'circle = δ = dL/dz, line thickness = |dL/dw|' : 'circle = a, line = weight'}</text>
      </svg>
      <div className="vz-two">
        <div>
          <p className="wg-p" style={{ margin: '0 0 6px' }}><strong>Prediction</strong> {fmt(r.a[L.length - 1][0])} · <strong>target</strong> {fmt(spec.target[0])} · <strong>loss</strong> {fmt(r.loss)}</p>
          <p className="wg-p" style={{ margin: '0 0 6px' }}>Loss: {loss === 'bce' ? 'cross-entropy' : '½ (a − y)²'} · activation {act}</p>
          {hist.length ? <p className="wg-p">Loss after each step: {[...hist.slice(1), r.loss].map((h) => fmt(h)).join(' → ')}</p> : null}
          <Slider label="Learning rate η" min={0.1} max={3} step={0.1} value={lr} onChange={setLr} show={(q) => q.toFixed(1)} />
          <div className="wg-row"><button type="button" className="wg-btn" onClick={step}>Update: w ← w − η·dL/dw</button><button type="button" className="wg-btn ghost" onClick={reset}>Reset</button></div>
        </div>
        <div aria-live="polite">
          <p className="wg-p" style={{ margin: '0 0 6px' }}><strong>{sl === 0 ? 'Input' : `Unit ${sj + 1} of ${sl === L.length - 1 ? 'the output layer' : `hidden layer ${sl}`}`}</strong></p>
          {hl}
          {sl > 0 ? <p className="vz-calc">{W[sl - 1][sj].map((_, i) => `dL/dw${i + 1} = δ × a_in${i + 1} = ${fmt(r.delta[sl][sj])}×${fmt(r.a[sl - 1][i])} = ${fmt(r.dW[sl - 1][sj][i])}`).join('\n')}</p> : null}
        </div>
      </div>
      <p className="wg-note">Switch to the backward pass and watch the blame travel from the loss back toward the inputs. {spec.note || ''}</p>
    </div>
  );
}

// ---------------------------------------------------------------- gradflow (vanishing and exploding gradients by layer)
function GradFlow({ spec, fig }) {
  const [depth, setDepth] = useState(spec.depth ?? 10);
  const [act, setAct] = useState(spec.activation || 'sigmoid');
  const [scale, setScale] = useState(spec.scale ?? 1);
  const norms = useMemo(() => gradNorms({ depth, width: spec.width || 16, act, scale, seed: spec.seed || 5 }), [depth, act, scale, spec]);
  const lo = -8, hi = 4;
  const lg = norms.map((n) => Math.max(lo, Math.min(hi, Math.log10(Math.max(n, 1e-30)))));
  const ratio = norms[0] / norms[norms.length - 1];
  const top = Math.max(...norms), bottom = Math.min(...norms);
  const state = ratio < 0.01 ? 'Vanishing: layers near the input get almost no signal, so they barely learn.' : ratio > 100 ? 'Exploding: layers near the input get huge gradients, so the weights can jump wildly.' : top > 20 ? 'Exploding: the gradients are huge in every layer, so each update is a giant jump.' : top < 0.02 ? 'Vanishing: the gradients are tiny in every layer, so learning crawls.' : bottom < 1e-3 ? 'Vanishing in some layers: parts of the network get almost no signal.' : 'Healthy: the gradient keeps a similar, moderate size in every layer.';
  const bw = 640 / norms.length;
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <div className="wg-row" style={{ marginBottom: 8 }}>
        {['sigmoid', 'tanh', 'relu'].map((k) => <button key={k} type="button" className={'wg-chip' + (act === k ? ' on' : '')} aria-pressed={act === k} onClick={() => setAct(k)}>{k}</button>)}
        {(spec.presets || [{ name: 'small init (0.7)', scale: 0.7 }, { name: 'Xavier (1.0)', scale: 1 }, { name: 'He (1.41)', scale: 1.41 }, { name: 'too large (2.5)', scale: 2.5 }]).map((p) => <button key={p.name} type="button" className={'wg-chip' + (p.scale === scale ? ' on' : '')} onClick={() => setScale(p.scale)}>{p.name}</button>)}
      </div>
      <Slider label="Depth (number of layers)" min={2} max={30} step={1} value={depth} onChange={setDepth} />
      <Slider label="Weight scale" min={0.25} max={3} step={0.05} value={scale} onChange={setScale} show={(q) => q.toFixed(2)} />
      <svg viewBox="0 0 640 220" className="vz-plot" role="img" aria-label={`Gradient size per layer for a ${depth}-layer ${act} network. Layer 1 over last layer ratio ${ratio.toExponential(1)}.`}>
        {[-6, -4, -2, 0, 2, 4].map((e) => <g key={e}><line x1="40" x2="636" y1={190 - ((e - lo) / (hi - lo)) * 175} y2={190 - ((e - lo) / (hi - lo)) * 175} className="vz-grid" /><text x="36" y={194 - ((e - lo) / (hi - lo)) * 175} textAnchor="end" className="vz-tick">1e{e}</text></g>)}
        {lg.map((g, i) => <rect key={i} x={40 + (i * 596) / norms.length + 1} width={Math.max(2, 596 / norms.length - 2)} y={190 - ((g - lo) / (hi - lo)) * 175} height={Math.max(1, ((g - lo) / (hi - lo)) * 175)} fill={g < -3 ? 'var(--cyan)' : g > 1.5 ? 'var(--amber)' : 'var(--accent)'} rx="2"><title>{`Layer ${i + 1}: ${norms[i].toExponential(2)}`}</title></rect>)}
        <text x="40" y="212" className="vz-tick">layer 1 (next to the input)</text>
        <text x="636" y="212" textAnchor="end" className="vz-tick">last layer (next to the loss)</text>
      </svg>
      <p className="wg-p" aria-live="polite"><strong>Layer 1 gradient ÷ last-layer gradient = {ratio >= 1e4 || ratio < 1e-3 ? ratio.toExponential(1) : ratio.toFixed(3)}.</strong> {state}</p>
      <p className="wg-note">Each bar is the size (Frobenius norm) of dL/dW for one layer, on a log scale, in one random {spec.width || 16}-unit-wide network with zero biases. The weights are drawn with standard deviation scale/√(inputs). {spec.note || ''}</p>
    </div>
  );
}

// ---------------------------------------------------------------- fold (why depth helps: pieces double with every layer)
function Fold({ spec, fig }) {
  const [k, setK] = useState(spec.layers ?? 2);
  const tent = (x) => 2 * Math.min(x, 1 - x);
  const f = (x) => { let y = x; for (let i = 0; i < k; i++) y = tent(y); return y; };
  const N = 600;
  const pts = Array.from({ length: N + 1 }, (_, i) => { const x = i / N; return `${i ? 'L' : 'M'}${(20 + x * 600).toFixed(1)} ${(130 - f(x) * 110).toFixed(1)}`; }).join(' ');
  const pieces = 2 ** k;
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <Slider label="Number of stacked layers" min={1} max={6} step={1} value={k} onChange={setK} />
      <svg viewBox="0 0 640 150" className="vz-plot" role="img" aria-label={`A zig-zag function with ${pieces} straight pieces made by ${k} stacked layers`}>
        <line x1="20" x2="620" y1="130" y2="130" className="vz-grid" />
        <path d={pts} fill="none" stroke="var(--accent)" strokeWidth="2.4" />
      </svg>
      <div className="vz-cmp" style={{ gridTemplateColumns: 'repeat(3,minmax(0,1fr))' }}>
        <div><span className="wg-k">Straight pieces</span><span className="vz-big">{pieces}</span><span className="wg-note">2 to the power {k}</span></div>
        <div className="out"><span className="wg-k">ReLU units, deep and narrow</span><span className="vz-big">{2 * k}</span><span className="wg-note">2 per layer</span></div>
        <div><span className="wg-k">ReLU units, one wide layer</span><span className="vz-big">≥ {pieces - 1}</span><span className="wg-note">m units give at most m + 1 pieces</span></div>
      </div>
      <p className="wg-note">Each layer applies the same “tent” (two ReLU units) to the output of the layer before, which folds the line in half again. {spec.note || ''}</p>
    </div>
  );
}

// ---------------------------------------------------------------- convgrad (backward pass of a convolution)
function ConvGrad({ spec, fig }) {
  const stride = spec.stride || 1, pad = spec.padding || 0;
  const k = spec.kernel.length;
  const y = useMemo(() => convolve(spec.input, spec.kernel, stride, pad), [spec, stride, pad]);
  const { dx, dK } = useMemo(() => convGrad(spec.input, spec.kernel, spec.upstream, stride, pad), [spec, stride, pad]);
  const [sel, setSel] = useState({ kind: 'x', i: 1, j: 1 });
  const n = spec.input.length, m = spec.input[0].length;
  const terms = [];
  let gMark = () => false;
  if (sel.kind === 'x') {
    for (let p = 0; p < spec.upstream.length; p++) for (let q = 0; q < spec.upstream[0].length; q++) {
      const a = sel.i - (p * stride - pad), c = sel.j - (q * stride - pad);
      if (a >= 0 && a < k && c >= 0 && c < k) terms.push({ p, q, text: `${spec.upstream[p][q]}×${spec.kernel[a][c]}` });
    }
  } else {
    for (let p = 0; p < spec.upstream.length; p++) for (let q = 0; q < spec.upstream[0].length; q++) {
      const i = p * stride - pad + sel.i, j = q * stride - pad + sel.j;
      if (i >= 0 && j >= 0 && i < n && j < m) terms.push({ p, q, text: `${spec.upstream[p][q]}×${spec.input[i][j]}` });
    }
  }
  gMark = (p, q) => terms.some((t) => t.p === p && t.q === q);
  const total = sel.kind === 'x' ? dx[sel.i][sel.j] : dK[sel.i][sel.j];
  const tone = (v) => (v === 0 ? undefined : `color-mix(in srgb, ${v > 0 ? 'var(--accent)' : 'var(--cyan)'} ${Math.min(55, 12 + Math.abs(v) * 7)}%, var(--surface2))`);
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <div className="vz-conv">
        <div><span className="wg-k" style={{ textTransform: "none" }}>Input x</span><Grid data={spec.input} mark={(i, j) => sel.kind === 'x' && sel.i === i && sel.j === j} onPick={(i, j) => setSel({ kind: 'x', i, j })} /></div>
        <span className="vz-op">∗</span>
        <div><span className="wg-k" style={{ textTransform: "none" }}>Kernel K</span><Grid data={spec.kernel} cls="ker" mark={(i, j) => sel.kind === 'k' && sel.i === i && sel.j === j} onPick={(i, j) => setSel({ kind: 'k', i, j })} /></div>
        <span className="vz-op">=</span>
        <div><span className="wg-k" style={{ textTransform: "none" }}>Output y</span><Grid data={y} cls="out" /></div>
      </div>
      <div className="vz-conv" style={{ marginTop: 14 }}>
        <div><span className="wg-k" style={{ textTransform: "none" }}>Upstream gradient G = dL/dy</span><Grid data={spec.upstream} mark={gMark} tone={tone} /></div>
        <span className="vz-op">→</span>
        <div><span className="wg-k" style={{ textTransform: "none" }}>dL/dK</span><Grid data={dK} cls="ker" tone={tone} mark={(i, j) => sel.kind === 'k' && sel.i === i && sel.j === j} /></div>
        <div><span className="wg-k" style={{ textTransform: "none" }}>dL/dx</span><Grid data={dx} tone={tone} mark={(i, j) => sel.kind === 'x' && sel.i === i && sel.j === j} /></div>
      </div>
      <p className="wg-p" aria-live="polite" style={{ margin: '12px 0 0' }}>
        <strong>{sel.kind === 'x' ? `dL/dx(${sel.i + 1}, ${sel.j + 1})` : `dL/dK(${sel.i + 1}, ${sel.j + 1})`}</strong> = <span className="vz-calc">{terms.length ? terms.map((t) => t.text).join(' + ') : '0 (no output used this cell)'} = {Number.isInteger(total) ? total : total.toFixed(2)}</span>
      </p>
      <p className="wg-note">Select an input cell to see which output gradients flowed back into it (each multiplied by the kernel weight that connected them), or a kernel cell to see which input values it met (each multiplied by the output gradient there). The kernel is shared, so its gradient is a sum over every position. {spec.note || ''}</p>
    </div>
  );
}

// ---------------------------------------------------------------- filters (apply a kernel to a small picture)
function Filters({ spec, fig }) {
  const img = useMemo(() => spec.image.map((r) => r.split('').map(Number)), [spec]);
  const presets = spec.presets;
  const [pi, setPi] = useState(0);
  const [custom, setCustom] = useState([[0, 0, 0], [0, 1, 0], [0, 0, 0]]);
  const [relu, setRelu] = useState(!!spec.relu);
  const isCustom = pi === presets.length;
  const ker = isCustom ? custom : presets[pi].kernel;
  const raw = useMemo(() => convolve(img, ker, 1, spec.padding || 0), [img, ker, spec]);
  const out = relu ? raw.map((r) => r.map((x) => Math.max(0, x))) : raw;
  const mx = Math.max(1, ...out.flat().map(Math.abs));
  const px = (val, signed) => (signed ? (val >= 0 ? `color-mix(in srgb, var(--accent) ${Math.round((val / mx) * 85)}%, var(--surface2))` : `color-mix(in srgb, var(--cyan) ${Math.round((-val / mx) * 85)}%, var(--surface2))`) : `color-mix(in srgb, var(--text) ${Math.round((val / 9) * 85)}%, var(--surface2))`);
  const cyc = (i, j) => setCustom((c) => c.map((r, a) => r.map((q, b) => (a === i && b === j ? (q >= 2 ? -2 : q + 1) : q))));
  const Pix = ({ data, signed }) => (
    <div className="vz-pix" style={{ gridTemplateColumns: `repeat(${data[0].length}, 1fr)` }} role="img" aria-label="picture as a grid of numbers">
      {data.map((r, i) => r.map((q, j) => <i key={i + '-' + j} style={{ background: px(q, signed) }} title={String(Number(q.toFixed(1)))} />))}
    </div>
  );
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <div className="wg-row" style={{ marginBottom: 10 }}>
        {presets.map((p, i) => <button key={p.name} type="button" className={'wg-chip' + (pi === i ? ' on' : '')} aria-pressed={pi === i} onClick={() => setPi(i)}>{p.name}</button>)}
        <button type="button" className={'wg-chip' + (isCustom ? ' on' : '')} aria-pressed={isCustom} onClick={() => setPi(presets.length)}>your own</button>
        <button type="button" className={'wg-chip' + (relu ? ' on' : '')} aria-pressed={relu} onClick={() => setRelu(!relu)}>ReLU after</button>
      </div>
      <div className="vz-conv" style={{ alignItems: 'flex-start' }}>
        <div><span className="wg-k">Picture {img.length}×{img[0].length}</span><Pix data={img} signed={false} /></div>
        <span className="vz-op">∗</span>
        <div><span className="wg-k">{isCustom ? 'Kernel (click a cell to change it)' : 'Kernel 3×3'}</span><Grid data={ker} cls="ker" onPick={isCustom ? cyc : null} /></div>
        <span className="vz-op">=</span>
        <div><span className="wg-k">Feature map {out.length}×{out[0].length}</span><Pix data={out} signed /></div>
      </div>
      <p className="wg-note">{isCustom ? 'Click kernel cells to cycle each weight from −2 to 2. ' : presets[pi].note ? presets[pi].note + ' ' : ''}Green means a strong positive response, blue a negative one, grey no response. {spec.note || ''}</p>
    </div>
  );
}

// ---------------------------------------------------------------- pool (max and average pooling by hand)
function Pool({ spec, fig }) {
  const [mode, setMode] = useState('max');
  const size = spec.size || 2, stride = spec.stride || size;
  const out = useMemo(() => pool2d(spec.input, size, stride, mode), [spec, size, stride, mode]);
  const [sel, setSel] = useState([0, 0]);
  const [si, sj] = sel;
  const win = [];
  for (let a = 0; a < size; a++) for (let c = 0; c < size; c++) win.push(spec.input[si * stride + a][sj * stride + c]);
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <div className="wg-tabs" role="tablist">{['max', 'avg'].map((k) => <button key={k} type="button" role="tab" aria-selected={mode === k} className={'wg-tab' + (mode === k ? ' on' : '')} onClick={() => setMode(k)}>{k === 'max' ? 'Max pooling' : 'Average pooling'}</button>)}</div>
      <div className="vz-conv" style={{ marginTop: 10 }}>
        <div><span className="wg-k">Input {spec.input.length}×{spec.input[0].length}</span><Grid data={spec.input} mark={(i, j) => i >= si * stride && i < si * stride + size && j >= sj * stride && j < sj * stride + size} /></div>
        <span className="vz-op">→</span>
        <div><span className="wg-k">Output {out.length}×{out[0].length} (window {size}×{size}, stride {stride})</span><Grid data={out} cls="out" mark={(i, j) => i === si && j === sj} onPick={(i, j) => setSel([i, j])} /></div>
      </div>
      <p className="wg-p" aria-live="polite" style={{ margin: '12px 0 0' }}><strong>Output cell ({si + 1}, {sj + 1}):</strong> <span className="vz-calc">{mode === 'max' ? `max(${win.join(', ')}) = ${Math.max(...win)}` : `(${win.join(' + ')}) ÷ ${win.length} = ${Number((win.reduce((s, q) => s + q, 0) / win.length).toFixed(2))}`}</span></p>
      <p className="wg-note">Select an output cell to see its window. Pooling has no weights to learn: it only shrinks the map and keeps the strongest (or the average) response. {spec.note || ''}</p>
    </div>
  );
}

// ---------------------------------------------------------------- cnnstack (shapes, parameters and receptive field through a CNN)
function CnnStack({ spec, fig }) {
  const [layers, setLayers] = useState(spec.layers.map((l) => ({ ...l })));
  const rows = useMemo(() => convShapes(spec.input, layers), [spec, layers]);
  const total = rows.reduce((s, r) => s + r.params, 0);
  const set = (i, patch) => setLayers((ls) => ls.map((l, k) => (k === i ? { ...l, ...patch } : l)));
  const fmt = (n) => n.toLocaleString();
  const first = rows.findIndex((r) => r.type === 'flatten');
  const flatN = first >= 0 ? rows[first].shape[2] : null;
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <div className="vz-scroll">
        <table className="vz-mlp" aria-label="Layer by layer shapes and parameters">
          <thead><tr><th>Layer</th><th>Settings</th><th>Output shape (H × W × C)</th><th>Parameters</th><th>Receptive field</th></tr></thead>
          <tbody>
            <tr><td>input</td><td></td><td>{spec.input[0]} × {spec.input[1]} × {spec.input[2]}</td><td>0</td><td>1 × 1</td></tr>
            {layers.map((L, i) => (
              <tr key={i} className={rows[i].valid ? '' : 'bad'}>
                <td>{L.type === 'conv' ? 'conv' : L.type === 'convT' ? 'transpose conv' : L.type === 'pool' ? 'max pool' : L.type}</td>
                <td>
                  {L.type === 'conv' || L.type === 'convT' ? (<>
                    <label className="vz-sel">k <select value={L.k} onChange={(e) => set(i, { k: +e.target.value })}>{[1, 3, 5, 7].map((q) => <option key={q}>{q}</option>)}</select></label>{' '}
                    <label className="vz-sel">stride <select value={L.stride || 1} onChange={(e) => set(i, { stride: +e.target.value })}>{[1, 2].map((q) => <option key={q}>{q}</option>)}</select></label>{' '}
                    <label className="vz-sel">pad <select value={L.pad || 0} onChange={(e) => set(i, { pad: +e.target.value })}>{[0, 1, 2, 3].map((q) => <option key={q}>{q}</option>)}</select></label>{' '}
                    <label className="vz-sel">filters <select value={L.filters} onChange={(e) => set(i, { filters: +e.target.value })}>{[4, 6, 8, 16, 32, 64].map((q) => <option key={q}>{q}</option>)}</select></label>
                  </>) : L.type === 'pool' ? `${L.k}×${L.k}, stride ${L.stride || L.k}` : L.type === 'fc' ? `${L.units} units` : ''}
                </td>
                <td>{rows[i].valid ? `${rows[i].shape[0]} × ${rows[i].shape[1]} × ${rows[i].shape[2]}` : 'too small!'}</td>
                <td>{fmt(rows[i].params)}</td>
                <td>{rows[i].rf ? `${rows[i].rf} × ${rows[i].rf}` : '–'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="vz-cmp" style={{ marginTop: 12, gridTemplateColumns: 'repeat(3,minmax(0,1fr))' }}>
        <div className="out"><span className="wg-k">Total parameters</span><span className="vz-big">{fmt(total)}</span><span className="wg-note">the sum of the column above</span></div>
        <div><span className="wg-k">Pixels in the input</span><span className="vz-big">{fmt(spec.input[0] * spec.input[1] * spec.input[2])}</span><span className="wg-note">a dense layer sees all of them at once</span></div>
        <div><span className="wg-k">Numbers fed to the first dense layer</span><span className="vz-big">{flatN != null ? fmt(flatN) : '–'}</span><span className="wg-note">after convolution and pooling</span></div>
      </div>
      <p className="wg-note">Change a layer’s settings and watch every shape after it update. Output size is ⌊(n + 2p − k) ÷ s⌋ + 1; if it reaches 0 the layer does not fit. A conv layer’s parameters are k × k × (input channels) × filters + filters. {spec.note || ''}</p>
    </div>
  );
}

// ---------------------------------------------------------------- vocabtrade (characters, sub-words or whole words as tokens)
function VocabTrade({ spec, fig }) {
  const maxM = spec.maxMerges || 12;
  const [m, setM] = useState(spec.merges ?? 4);
  const [text, setText] = useState(spec.text || '');
  const all = useMemo(() => Array.from({ length: maxM + 1 }, (_, k) => tokenStats(spec.corpus, spec.text || '', k)), [spec, maxM]);
  const cur = useMemo(() => tokenStats(spec.corpus, text, m), [spec, text, m]);
  const maxV = Math.max(...all.map((a) => a.vocabSize), cur.wordLevel.vocabSize);
  const show = (t) => t.replace('</w>', '·');
  const W = 560, H = 150;
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <Slider label="BPE merges learned (0 = characters)" min={0} max={Math.min(maxM, all[maxM].learned.length)} step={1} value={Math.min(m, all[maxM].learned.length)} onChange={setM} />
      <label className="vz-try"><span>Sentence to split (try a word the corpus never contained):</span>
        <input type="text" value={text} onChange={(e) => setText(e.target.value.toLowerCase().replace(/[^a-z ]/g, ''))} aria-label="Sentence to tokenise" />
      </label>
      <div className="vz-cmp" style={{ margin: '12px 0', gridTemplateColumns: 'repeat(3,minmax(0,1fr))' }}>
        <div><span className="wg-k">Characters</span><span className="vz-big">{cur.charLevel.textTokens} tokens</span><span className="wg-note">vocabulary {cur.charLevel.vocabSize} symbols</span></div>
        <div className="out"><span className="wg-k">BPE, {Math.min(m, cur.learned.length)} merges</span><span className="vz-big">{cur.textTokens} tokens</span><span className="wg-note">vocabulary {cur.vocabSize} pieces</span></div>
        <div><span className="wg-k">Whole words</span><span className="vz-big">{cur.wordLevel.textTokens} tokens</span><span className="wg-note">vocabulary {cur.wordLevel.vocabSize} words{cur.wordLevel.unk ? ` · ${cur.wordLevel.unk} unknown` : ''}</span></div>
      </div>
      <p className="vz-toks" style={{ margin: '0 0 10px' }} aria-live="polite">{cur.pieces.map((p, i) => <span key={i} className="vz-wpiece">{p.map((t, k) => <span key={k} className="vz-tok">{show(t)}</span>)}</span>)}</p>
      <svg viewBox={`0 0 ${W} ${H}`} className="vz-plot" role="img" aria-label="Vocabulary size and tokens for the sentence as merges grow">
        <line x1="40" x2={W - 10} y1={H - 20} y2={H - 20} className="vz-grid" />
        {all.map((a, k) => { const x = 40 + (k / maxM) * (W - 60); const y = H - 20 - (a.vocabSize / maxV) * (H - 40); return <circle key={k} cx={x} cy={y} r={k === Math.min(m, maxM) ? 6 : 3.5} fill={k === Math.min(m, maxM) ? 'var(--accent)' : 'var(--text3)'} />; })}
        <path d={all.map((a, k) => `${k ? 'L' : 'M'}${40 + (k / maxM) * (W - 60)} ${H - 20 - (a.vocabSize / maxV) * (H - 40)}`).join(' ')} fill="none" stroke="var(--accent)" strokeWidth="2" />
        <text x="40" y={H - 4} className="vz-tick">0 merges</text>
        <text x={W - 10} y={H - 4} textAnchor="end" className="vz-tick">{maxM} merges</text>
        <text x="44" y="14" className="vz-tick">vocabulary size (grows with merges)</text>
      </svg>
      <p className="wg-note">More merges give a bigger vocabulary but shorter sequences. A word-level vocabulary has no way to write a word it has never seen, while characters and BPE pieces always can. {spec.note || ''}</p>
    </div>
  );
}

// ---------------------------------------------------------------- vecsim (dot product, norms and cosine of two word vectors)
function VecSim({ spec, fig }) {
  const words = spec.words;
  const [ia, setIa] = useState(spec.a ?? 0);
  const [ib, setIb] = useState(spec.b ?? 1);
  const A = words[ia], B = words[ib];
  const r = cosineSim(A.v, B.v);
  const dims = spec.dims || A.v.map((_, i) => `d${i + 1}`);
  const mx = Math.max(...words.flatMap((w) => w.v.map(Math.abs)), 1e-9);
  const f = (n) => Number(n.toFixed(2)).toString();
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <div className="wg-row" style={{ gap: 14, marginBottom: 8 }}>
        <label className="vz-sel">word 1 <select value={ia} onChange={(e) => setIa(+e.target.value)}>{words.map((w, i) => <option key={i} value={i}>{w.label}</option>)}</select></label>
        <label className="vz-sel">word 2 <select value={ib} onChange={(e) => setIb(+e.target.value)}>{words.map((w, i) => <option key={i} value={i}>{w.label}</option>)}</select></label>
      </div>
      <div className="vz-scroll">
        <table className="vz-mlp" aria-label="Vector entries and products">
          <thead><tr><th></th>{dims.map((d, i) => <th key={i}>{d}</th>)}</tr></thead>
          <tbody>
            <tr><td><strong>{A.label}</strong></td>{A.v.map((x, i) => <td key={i}><span className="vz-vbar" style={{ width: `${Math.abs(x) / mx * 46}px`, background: x >= 0 ? 'var(--accent)' : 'var(--cyan)' }} />{f(x)}</td>)}</tr>
            <tr><td><strong>{B.label}</strong></td>{B.v.map((x, i) => <td key={i}><span className="vz-vbar" style={{ width: `${Math.abs(x) / mx * 46}px`, background: x >= 0 ? 'var(--accent)' : 'var(--cyan)' }} />{f(x)}</td>)}</tr>
            <tr><td>product</td>{A.v.map((x, i) => <td key={i}>{f(x * B.v[i])}</td>)}</tr>
          </tbody>
        </table>
      </div>
      <div className="vz-cmp" style={{ marginTop: 12, gridTemplateColumns: 'repeat(3,minmax(0,1fr))' }}>
        <div><span className="wg-k">Dot product (sum of products)</span><span className="vz-big">{f(r.dot)}</span></div>
        <div><span className="wg-k">Lengths</span><span className="vz-big">{f(r.na)} · {f(r.nb)}</span></div>
        <div className="out"><span className="wg-k">Cosine similarity</span><span className="vz-big">{f(r.cos)}</span><span className="wg-note">{f(r.dot)} ÷ ({f(r.na)} × {f(r.nb)}); 1 = same direction, 0 = unrelated, −1 = opposite</span></div>
      </div>
      <p className="wg-note">{spec.note || 'Illustrative vectors, written for teaching; real embeddings have hundreds of dimensions learned from text.'}</p>
    </div>
  );
}

// ---------------------------------------------------------------- seqmemory (how much of step t survives to the end: RNN, LSTM, attention)
function SeqMemory({ spec, fig }) {
  const [T, setT] = useState(spec.length ?? 20);
  const [w, setW] = useState(spec.w ?? 0.9);
  const [f, setF] = useState(spec.forget ?? 0.95);
  const rnn = useMemo(() => rnnInfluence({ T, w, seed: spec.seed || 3 }).influence, [T, w, spec]);
  const lstm = useMemo(() => lstmInfluence(T, f), [T, f]);
  const lo = -12, hi = 1;
  const yy = (x) => 150 - ((Math.max(lo, Math.min(hi, Math.log10(Math.max(x, 1e-30)))) - lo) / (hi - lo)) * 140;
  const W_ = 600;
  const line = (arr, col) => <path d={arr.map((q, i) => `${i ? 'L' : 'M'}${30 + (i / (T - 1)) * (W_ - 40)} ${yy(q)}`).join(' ')} fill="none" stroke={col} strokeWidth="2.4" />;
  const fmt = (x) => (x >= 0.01 ? x.toFixed(2) : x.toExponential(1));
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <Slider label="Sequence length T" min={5} max={80} step={1} value={T} onChange={setT} />
      <Slider label="RNN recurrent weight w" min={0.2} max={1.6} step={0.05} value={w} onChange={setW} show={(q) => q.toFixed(2)} />
      <Slider label="LSTM forget gate f" min={0.5} max={1} step={0.01} value={f} onChange={setF} show={(q) => q.toFixed(2)} />
      <svg viewBox={`0 0 ${W_} 175`} className="vz-plot" role="img" aria-label={`How much of each step still reaches the last step. After ${T} steps the first step keeps ${fmt(rnn[0])} in the RNN, ${fmt(lstm[0])} in the LSTM and 1 with attention.`}>
        {[-10, -8, -6, -4, -2, 0].map((e) => <g key={e}><line x1="30" x2={W_ - 10} y1={yy(10 ** e)} y2={yy(10 ** e)} className="vz-grid" /><text x="26" y={yy(10 ** e) + 4} textAnchor="end" className="vz-tick">1e{e}</text></g>)}
        {line(rnn, 'var(--red)')}
        {line(lstm, 'var(--accent)')}
        <line x1="30" x2={W_ - 10} y1={yy(1)} y2={yy(1)} stroke="var(--cyan)" strokeWidth="2.4" strokeDasharray="6 4" />
        <text x="30" y="170" className="vz-tick">first step</text>
        <text x={W_ - 10} y="170" textAnchor="end" className="vz-tick">last step</text>
      </svg>
      <div className="wg-row vz-legend">
        <span className="vz-key"><i style={{ background: 'var(--red)' }} />RNN (red): {fmt(rnn[0])} left from step 1</span>
        <span className="vz-key"><i style={{ background: 'var(--accent)' }} />LSTM (green): {fmt(lstm[0])}</span>
        <span className="vz-key"><i style={{ background: 'var(--cyan)' }} />attention (blue, dashed): 1, a direct path</span>
      </div>
      <p className="wg-note">The curves show d(final state) ÷ d(state at step t): the gradient signal that travels from the end back to step t, on a log scale. The RNN is a one-unit tanh network on a fixed ±1 input; the LSTM line is the product of constant forget gates; attention links the last position straight to every earlier one. {spec.note || ''}</p>
    </div>
  );
}

// ---------------------------------------------------------------- lstmcell (one LSTM step with its four gates)
function LstmCell({ spec, fig }) {
  const [x, setX] = useState(spec.x ?? 1);
  const [h, setH] = useState(spec.h ?? 0);
  const [c, setC] = useState(spec.c ?? 0.5);
  const r = useMemo(() => lstmStep(spec.params, x, h, c), [spec, x, h, c]);
  const f2_ = (q) => Number(q.toFixed(3)).toString();
  const Bar = ({ name, v, lo = 0, hi = 1, hint }) => (
    <li style={{ gridTemplateColumns: '150px minmax(0,1fr) 60px' }}>
      <span className="wg-bar-l" style={{ fontFamily: 'inherit' }}>{name}</span>
      <span className="wg-bar-track"><span className="wg-bar-fill" style={{ width: `${((v - lo) / (hi - lo)) * 100}%` }} /></span>
      <span className="wg-bar-p">{f2_(v)}</span>
    </li>
  );
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <div className="vz-two">
        <div>
          <Slider label="Input x_t" min={-2} max={2} step={0.1} value={x} onChange={setX} show={(q) => q.toFixed(1)} />
          <Slider label="Previous hidden state h_(t−1)" min={-1} max={1} step={0.1} value={h} onChange={setH} show={(q) => q.toFixed(1)} />
          <Slider label="Previous cell state c_(t−1)" min={-2} max={2} step={0.1} value={c} onChange={setC} show={(q) => q.toFixed(1)} />
        </div>
        <ul className="wg-bars" style={{ margin: 0 }}>
          <Bar name="forget gate f (keep old memory)" v={r.f} />
          <Bar name="input gate i (let new in)" v={r.i} />
          <Bar name="output gate o (reveal memory)" v={r.o} />
          <Bar name="candidate g (new content)" v={(r.g + 1) / 2} />
        </ul>
      </div>
      <p className="vz-calc">c_t = f × c_(t−1) + i × g = {f2_(r.f)} × {f2_(c)} + {f2_(r.i)} × {f2_(r.g)} = <strong>{f2_(r.c)}</strong>{'\n'}h_t = o × tanh(c_t) = {f2_(r.o)} × {f2_(Math.tanh(r.c))} = <strong>{f2_(r.h)}</strong></p>
      <p className="wg-note">The old memory is only multiplied by f, not squashed through a weight matrix and a tanh, so with f near 1 the memory (and its gradient) survives many steps. {spec.note || 'The gate weights are authored for teaching, not learned.'}</p>
    </div>
  );
}

// ---------------------------------------------------------------- hardsoft (hard attention picks one, soft attention blends)
function HardSoft({ spec, fig }) {
  const [scores, setScores] = useState(spec.scores.slice());
  const [temp, setTemp] = useState(spec.temperature ?? 1);
  const [mode, setMode] = useState('soft');
  const r = useMemo(() => softHard(scores, spec.values, temp), [scores, spec, temp]);
  const bump = useMemo(() => { const s2 = scores.map((q, i) => (i === r.arg ? q : q)); s2[(r.arg + 1) % scores.length] += 0.1; return softHard(s2, spec.values, temp); }, [scores, spec, temp, r.arg]);
  const w = mode === 'soft' ? r.soft : r.hard;
  const out = mode === 'soft' ? r.softOut : r.hardOut;
  const outB = mode === 'soft' ? bump.softOut : bump.hardOut;
  const d = out.map((q, k) => outB[k] - q);
  const fmt = (q) => Number(q.toFixed(3)).toString();
  const nb = (r.arg + 1) % scores.length;
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <div className="wg-tabs" role="tablist">{['soft', 'hard'].map((k) => <button key={k} type="button" role="tab" aria-selected={mode === k} className={'wg-tab' + (mode === k ? ' on' : '')} onClick={() => setMode(k)}>{k === 'soft' ? 'Soft attention' : 'Hard attention'}</button>)}</div>
      {scores.map((s, i) => (
        <div key={i} className="vz-hsrow">
          <Slider label={`score for “${spec.labels[i]}”`} min={-3} max={5} step={0.1} value={s} onChange={(q) => setScores((a) => a.map((o, k) => (k === i ? q : o)))} show={(q) => q.toFixed(1)} />
          <span className="wg-bar-track" aria-label={`weight ${fmt(w[i])}`}><span className={'wg-bar-fill' + (mode === 'hard' ? ' warn' : '')} style={{ width: `${w[i] * 100}%` }} /></span>
          <span className="wg-bar-p">{fmt(w[i])}</span>
        </div>
      ))}
      {mode === 'soft' ? <Slider label="Temperature T (small T sharpens towards hard)" min={0.1} max={3} step={0.1} value={temp} onChange={setTemp} show={(q) => q.toFixed(1)} /> : null}
      <p className="vz-calc">output = Σ weight × value = ({fmt(out[0])}{out.length > 1 ? `, ${fmt(out[1])}` : ''})</p>
      <p className="wg-p" aria-live="polite" style={{ margin: '6px 0 0' }}><strong>Sensitivity.</strong> Nudge the score of “{spec.labels[nb]}” up by 0.1: the output changes by ({fmt(d[0])}{d.length > 1 ? `, ${fmt(d[1])}` : ''}). {mode === 'soft' ? 'Every score moves the output a little, so gradients can flow back to every score.' : (d.every((q) => Math.abs(q) < 1e-9) ? 'The output does not move at all, so there is no gradient to learn from: hard attention is a jump, not a slope.' : 'The output jumped because a different position became the maximum: hard attention changes in jumps, so it has no useful slope.')}</p>
      <p className="wg-note">{spec.note || 'Scores and values are made up for the demonstration.'}</p>
    </div>
  );
}

// ---------------------------------------------------------------- tfblock (parts, shapes and parameters of a decoder-only Transformer)
const TF_PARTS = {
  emb: ['Token embedding', 'Looks up one d-dimensional vector per token id (Week 6). Table of V × d numbers.'],
  pos: ['Positional encoding', 'Adds a vector that says where in the sequence each token is. Sinusoidal: no parameters. Learned: n_ctx × d numbers.'],
  attn: ['Multi-head self-attention', 'Every token mixes in information from the earlier tokens: W_Q, W_K, W_V and W_O, each d × d (plus biases).'],
  ffn: ['Feed-forward network', 'Processes each token on its own: d → d_ff → d, with a non-linearity (ReLU or GELU) in between.'],
  ln: ['Layer norm (two per block)', 'Re-centres and re-scales each token vector; learns a scale and a shift of d numbers each.'],
  res: ['Residual connections', 'Add the input of a sub-layer to its output: x + F(x). No parameters, but they keep gradients flowing.'],
  fin: ['Final layer norm', 'One more layer norm after the last block.'],
  head: ['Output layer (unembedding)', 'A d × V matrix turning each final vector into one score per vocabulary token; often tied to (shared with) the embedding table.'],
};
function TfBlock({ spec, fig }) {
  const [cfg, setCfg] = useState({ vocab: spec.vocab ?? 50257, d: spec.d ?? 768, heads: spec.heads ?? 12, dff: spec.dff ?? 3072, layers: spec.layers ?? 12, ctx: spec.ctx ?? 1024, posLearned: spec.posLearned ?? true, tied: spec.tied ?? true });
  const [sel, setSel] = useState('attn');
  const set = (p) => setCfg((c) => ({ ...c, ...p }));
  const P = tfParams(cfg);
  const fmt = (n) => n.toLocaleString();
  const dh = cfg.d / cfg.heads;
  const ok = Number.isInteger(dh);
  const rows = [
    ['emb', `V × d = ${fmt(cfg.vocab)} × ${fmt(cfg.d)}`, `(n, d)`, P.emb, 1],
    ['pos', cfg.posLearned ? `n_ctx × d = ${fmt(cfg.ctx)} × ${fmt(cfg.d)} (learned)` : 'sinusoidal: computed, nothing to learn', `(n, d)`, P.pos, 1],
    ['attn', `${cfg.heads} heads of ${ok ? dh : '?'} · 4·d² + 4·d`, `(n, d)`, P.attn, cfg.layers],
    ['ffn', `d → ${fmt(cfg.dff)} → d`, `(n, d)`, P.ffn, cfg.layers],
    ['ln', '2 × (2·d)', `(n, d)`, P.ln, cfg.layers],
    ['fin', '2·d', `(n, d)`, P.finalLn, 1],
    ['head', cfg.tied ? 'tied to the embedding table' : 'd × V', `(n, V)`, P.head, 1],
  ];
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <div className="wg-row" style={{ gap: 12, marginBottom: 10 }}>
        <label className="vz-sel">d_model <select value={cfg.d} onChange={(e) => set({ d: +e.target.value })}>{[64, 128, 256, 512, 768, 1024, 2048].map((q) => <option key={q}>{q}</option>)}</select></label>
        <label className="vz-sel">heads <select value={cfg.heads} onChange={(e) => set({ heads: +e.target.value })}>{[1, 2, 4, 8, 12, 16, 32].map((q) => <option key={q}>{q}</option>)}</select></label>
        <label className="vz-sel">d_ff <select value={cfg.dff} onChange={(e) => set({ dff: +e.target.value })}>{[256, 512, 1024, 2048, 3072, 4096, 8192].map((q) => <option key={q}>{q}</option>)}</select></label>
        <label className="vz-sel">blocks <select value={cfg.layers} onChange={(e) => set({ layers: +e.target.value })}>{[1, 2, 4, 6, 12, 24, 48].map((q) => <option key={q}>{q}</option>)}</select></label>
        <label className="vz-sel">vocabulary <select value={cfg.vocab} onChange={(e) => set({ vocab: +e.target.value })}>{[65, 1000, 8000, 32000, 50257, 100000].map((q) => <option key={q} value={q}>{fmt(q)}</option>)}</select></label>
        <button type="button" className={'wg-chip' + (cfg.posLearned ? ' on' : '')} aria-pressed={cfg.posLearned} onClick={() => set({ posLearned: !cfg.posLearned })}>learned positions</button>
        <button type="button" className={'wg-chip' + (cfg.tied ? ' on' : '')} aria-pressed={cfg.tied} onClick={() => set({ tied: !cfg.tied })}>tied output</button>
      </div>
      {!ok ? <p className="wg-why bad" role="alert">d_model must divide evenly by the number of heads (d_model ÷ heads = {Number(dh.toFixed(2))}).</p> : null}
      <div className="vz-scroll">
        <table className="vz-mlp" aria-label="Parts, shapes and parameters">
          <thead><tr><th>Part</th><th>Size</th><th>Output shape</th><th>Parameters each</th><th>Copies</th><th>Total</th></tr></thead>
          <tbody>
            {rows.map(([k, size, shape, p, n]) => (
              <tr key={k} className={sel === k ? 'sel' : ''} onClick={() => setSel(k)} style={{ cursor: 'pointer' }}>
                <td><button type="button" className="vz-link" onClick={() => setSel(k)}>{TF_PARTS[k][0]}</button></td><td>{size}</td><td>{shape}</td><td>{fmt(p)}</td><td>× {n}</td><td>{fmt(p * n)}</td>
              </tr>
            ))}
            <tr className="tot"><td colSpan={5}><strong>Total trainable parameters</strong></td><td><strong>{fmt(P.total)}</strong></td></tr>
          </tbody>
        </table>
      </div>
      <p className="wg-p" aria-live="polite" style={{ margin: '10px 0 0' }}><strong>{TF_PARTS[sel][0]}.</strong> {TF_PARTS[sel][1]}</p>
      <p className="wg-note">Shapes are for one sequence of n tokens; the (n, d) shape stays the same through every block, which is what lets residual connections add the input to the output and blocks be stacked. {spec.note || ''}</p>
    </div>
  );
}

// ---------------------------------------------------------------- causalmask (the mask that stops a token looking at the future)
function CausalMask({ spec, fig }) {
  const toks = spec.tokens;
  const [masked, setMasked] = useState(true);
  const [row, setRow] = useState(spec.row ?? toks.length - 1);
  const W = useMemo(() => maskedSoftmax(spec.scores, masked), [spec, masked]);
  const f = (q) => Number(q.toFixed(2)).toString();
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <div className="wg-tabs" role="tablist">{[true, false].map((k) => <button key={String(k)} type="button" role="tab" aria-selected={masked === k} className={'wg-tab' + (masked === k ? ' on' : '')} onClick={() => setMasked(k)}>{k ? 'With the causal mask' : 'Without the mask'}</button>)}</div>
      <div className="vz-scroll">
        <table className="vz-mlp vz-attn" aria-label="Attention weights">
          <thead><tr><th>query ↓ key →</th>{toks.map((t, j) => <th key={j}>{t}</th>)}</tr></thead>
          <tbody>
            {W.map((r, i) => (
              <tr key={i} className={i === row ? 'sel' : ''} onClick={() => setRow(i)} style={{ cursor: 'pointer' }}>
                <td><button type="button" className="vz-link" onClick={() => setRow(i)}>{toks[i]}</button></td>
                {r.map((q, j) => <td key={j} style={{ background: j > i && masked ? 'repeating-linear-gradient(45deg,var(--surface2),var(--surface2) 4px,transparent 4px,transparent 8px)' : `color-mix(in srgb, var(--accent) ${Math.round(q * 70)}%, var(--surface2))` }}>{j > i && masked ? '−∞→0' : f(q)}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="vz-calc" aria-live="polite">row “{toks[row]}”: scores ({spec.scores[row].map(f).join(', ')}){masked ? ` → future positions set to −∞` : ''} → softmax = ({W[row].map(f).join(', ')}) · sum {f(W[row].reduce((a, b) => a + b, 0))}</p>
      <p className="wg-note">With the mask, row i can only use tokens 1…i, so the whole sentence can be trained in one pass without cheating. Without it, each token sees the word it is supposed to predict. {spec.note || 'The scores are made up for the demonstration.'}</p>
    </div>
  );
}

// ---------------------------------------------------------------- shiftloss (teacher forcing: inputs, shifted targets and the loss at each position)
function ShiftLoss({ spec, fig }) {
  const toks = spec.tokens;
  const p = spec.p;
  const [sel, setSel] = useState(0);
  const losses = p.map((q) => -Math.log(q));
  const mean = losses.reduce((a, b) => a + b, 0) / losses.length;
  const mx = Math.max(...losses, 1e-9);
  const f = (q) => Number(q.toFixed(3)).toString();
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <div className="vz-scroll">
        <table className="vz-mlp" aria-label="Inputs, targets and losses">
          <thead><tr><th>Position</th>{p.map((_, i) => <th key={i}>{i + 1}</th>)}</tr></thead>
          <tbody>
            <tr><td>input (what the model sees)</td>{p.map((_, i) => <td key={i} className={sel === i ? 'sel' : ''}><button type="button" className="vz-link" onClick={() => setSel(i)}>{toks[i]}</button></td>)}</tr>
            <tr><td>target (the next token)</td>{p.map((_, i) => <td key={i} className={sel === i ? 'sel' : ''}><strong>{toks[i + 1]}</strong></td>)}</tr>
            <tr><td>probability given to the target</td>{p.map((q, i) => <td key={i} className={sel === i ? 'sel' : ''}>{f(q)}</td>)}</tr>
            <tr><td>loss −ln p</td>{losses.map((l, i) => <td key={i} className={sel === i ? 'sel' : ''}><span className="vz-vbar" style={{ width: `${(l / mx) * 40}px`, background: 'var(--amber)' }} />{f(l)}</td>)}</tr>
          </tbody>
        </table>
      </div>
      <p className="wg-p" aria-live="polite" style={{ margin: '10px 0 0' }}><strong>Position {sel + 1}:</strong> having read “{toks.slice(0, sel + 1).join(' ')}”, the model must predict “{toks[sel + 1]}”. It gave that token probability {f(p[sel])}, so the loss is −ln {f(p[sel])} = {f(losses[sel])}.</p>
      <div className="vz-cmp" style={{ marginTop: 10, gridTemplateColumns: 'repeat(3,minmax(0,1fr))' }}>
        <div className="out"><span className="wg-k">Training loss (mean)</span><span className="vz-big">{f(mean)}</span><span className="wg-note">the average of the {p.length} losses</span></div>
        <div><span className="wg-k">Perplexity</span><span className="vz-big">{f(Math.exp(mean))}</span><span className="wg-note">exp of the loss</span></div>
        <div><span className="wg-k">Tokens trained in one pass</span><span className="vz-big">{p.length}</span><span className="wg-note">one prediction per position</span></div>
      </div>
      <p className="wg-note">The target sequence is the input sequence shifted left by one: each position learns to predict the token after it, and all positions are trained at once. {spec.note || 'The probabilities are made up for the demonstration.'}</p>
    </div>
  );
}

// ---------------------------------------------------------------- llmfamily (encoder-only, decoder-only, encoder-decoder: masks and training tasks)
function LlmFamily({ spec, fig }) {
  const fams = spec.families;
  const [i, setI] = useState(spec.start ?? 0);
  const f = fams[i];
  const toks = spec.tokens;
  const n = toks.length;
  const allowed = (r, c) => (f.mask === 'causal' ? c <= r : f.mask === 'prefix' ? (c < (f.prefix ?? Math.floor(n / 2)) || c <= r) : true);
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <div className="wg-tabs" role="tablist">{fams.map((x, k) => <button key={k} type="button" role="tab" aria-selected={i === k} className={'wg-tab' + (i === k ? ' on' : '')} onClick={() => setI(k)}>{x.name}</button>)}</div>
      <div className="vz-two" style={{ marginTop: 10 }}>
        <div>
          <span className="wg-k">Who may look at whom ({f.maskName})</span>
          <div className="vz-maskgrid" style={{ gridTemplateColumns: `34px repeat(${n}, 1fr)` }} role="img" aria-label={`${f.maskName}: row = the token being read, column = the token it may use`}>
            <i />
            {toks.map((t, c) => <b key={c}>{t}</b>)}
            {toks.map((t, r) => (
              <React.Fragment key={r}>
                <b>{t}</b>
                {toks.map((_, c) => <i key={c} className={allowed(r, c) ? 'on' : ''} />)}
              </React.Fragment>
            ))}
          </div>
          <p className="wg-note" style={{ margin: '6px 0 0' }}>Green = the row token may use the column token.</p>
        </div>
        <div>
          <p className="wg-p" style={{ margin: '0 0 6px' }}><strong>{f.objective}</strong></p>
          <p className="vz-calc">input:  {f.input}{'\n'}target: {f.target}</p>
          <p className="wg-p" style={{ margin: '6px 0 0' }}><strong>Typical uses:</strong> {f.uses}</p>
          <p className="wg-p" style={{ margin: '4px 0 0' }}><strong>Examples:</strong> {f.examples}</p>
        </div>
      </div>
      <p className="wg-note">{spec.note || ''}</p>
    </div>
  );
}

// ---------------------------------------------------------------- moe (a router sends each token to a few experts)
function Moe({ spec, fig }) {
  const E = spec.scores[0].length;
  const [k, setK] = useState(spec.k ?? 2);
  const [d] = useState(spec.d ?? 512);
  const [dff, setDff] = useState(spec.dff ?? 2048);
  const route = useMemo(() => moeRoute(spec.scores, k), [spec, k]);
  const load = Array(E).fill(0);
  route.forEach((r) => r.forEach((q) => { load[q.expert] += 1; }));
  const P = moeParams(E, k, d, dff);
  const dense = moeParams(1, 1, d, dff);
  const f = (q) => Number(q.toFixed(2)).toString();
  const mx = Math.max(...load, 1);
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <Slider label={`Experts used per token (top-k) out of ${E}`} min={1} max={E} step={1} value={k} onChange={setK} />
      <div className="vz-scroll">
        <table className="vz-mlp vz-attn" aria-label="Router choices">
          <thead><tr><th>token</th>{Array.from({ length: E }, (_, j) => <th key={j}>expert {j + 1}</th>)}</tr></thead>
          <tbody>
            {route.map((r, i) => (
              <tr key={i}>
                <td><strong>{spec.tokens[i]}</strong></td>
                {Array.from({ length: E }, (_, j) => { const q = r.find((z) => z.expert === j); return <td key={j} style={{ background: q ? `color-mix(in srgb, var(--accent) ${Math.round(q.weight * 70)}%, var(--surface2))` : undefined, color: q ? 'var(--text)' : 'var(--text3)' }}>{q ? f(q.weight) : `· (${f(spec.scores[i][j])})`}</td>; })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="wg-note" style={{ margin: '6px 0 8px' }}>A number is the weight given to that expert's answer (softmax over the chosen experts' scores). A dot means the expert is skipped for that token; the router score is in brackets.</p>
      <ul className="wg-bars">
        {load.map((q, j) => <li key={j} style={{ gridTemplateColumns: '90px minmax(0,1fr) 70px' }}><span className="wg-bar-l" style={{ fontFamily: 'inherit' }}>expert {j + 1}</span><span className="wg-bar-track"><span className={'wg-bar-fill' + (q === 0 ? ' warn' : '')} style={{ width: `${(q / mx) * 100}%` }} /></span><span className="wg-bar-p">{q} tokens</span></li>)}
      </ul>
      <div className="vz-cmp" style={{ marginTop: 12, gridTemplateColumns: 'repeat(3,minmax(0,1fr))' }}>
        <div><span className="wg-k">Dense feed-forward (1 expert)</span><span className="vz-big">{(dense.total / 1e6).toFixed(2)} M</span><span className="wg-note">d = {d}, d_ff = {dff}</span></div>
        <div><span className="wg-k">All {E} experts stored</span><span className="vz-big">{(P.total / 1e6).toFixed(2)} M</span><span className="wg-note">memory you must hold</span></div>
        <div className="out"><span className="wg-k">Used per token (top-{k})</span><span className="vz-big">{(P.active / 1e6).toFixed(2)} M</span><span className="wg-note">compute per token ≈ {k} experts, not {E}</span></div>
      </div>
      <p className="wg-note">{spec.note || 'Router scores are made up. Real routers are small learned linear layers, trained with a loss that spreads tokens across experts.'}</p>
    </div>
  );
}

// ---------------------------------------------------------------- scalinglaw (Chinchilla's fitted loss and the compute-optimal size)
function ScalingLaw({ spec, fig }) {
  const [logC, setLogC] = useState(spec.logC ?? 23.77);
  const [logN, setLogN] = useState(spec.logN ?? 10.8);
  const C = 10 ** logC;
  const opt = useMemo(() => chinOptimal(C), [C]);
  const N = 10 ** logN, D = C / 6 / N;
  const lo = Math.log10(opt.N) - 2, hi = Math.log10(opt.N) + 2;
  const pts = Array.from({ length: 81 }, (_, i) => { const x = lo + ((hi - lo) * i) / 80; const n = 10 ** x; return [x, chinLoss(n, C / 6 / n)]; });
  const ys = pts.map((p) => p[1]);
  const y0 = Math.min(...ys), y1 = Math.min(Math.max(...ys), y0 + 1.2);
  const W = 560, H = 170;
  const sx = (x) => 30 + ((x - lo) / (hi - lo)) * (W - 50);
  const sy = (y) => H - 24 - ((Math.min(y, y1) - y0) / (y1 - y0)) * (H - 44);
  const L = chinLoss(N, D);
  const sci = (x) => x.toExponential(2).replace('e+', ' × 10^');
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <Slider label="Compute budget C (log10 FLOPs)" min={20} max={26} step={0.05} value={logC} onChange={setLogC} show={(q) => `10^${q.toFixed(2)}`} />
      <Slider label="Your model size N (log10 parameters)" min={Math.max(7, lo)} max={Math.min(13, hi)} step={0.02} value={Math.min(Math.max(logN, lo), hi)} onChange={setLogN} show={(q) => `${(10 ** q / 1e9).toFixed(2)} B`} />
      <svg viewBox={`0 0 ${W} ${H}`} className="vz-plot" role="img" aria-label="Loss against model size at a fixed compute budget; the minimum is the compute-optimal size">
        <line x1="30" x2={W - 20} y1={H - 24} y2={H - 24} className="vz-grid" />
        <path d={pts.map((p, i) => `${i ? 'L' : 'M'}${sx(p[0]).toFixed(1)} ${sy(p[1]).toFixed(1)}`).join(' ')} fill="none" stroke="var(--accent)" strokeWidth="2.4" />
        <line x1={sx(Math.log10(opt.N))} x2={sx(Math.log10(opt.N))} y1="8" y2={H - 24} stroke="var(--cyan)" strokeDasharray="5 4" />
        <circle cx={sx(Math.min(Math.max(logN, lo), hi))} cy={sy(L)} r="6" fill="var(--amber)" />
        <text x="30" y={H - 6} className="vz-tick">smaller model, more data</text>
        <text x={W - 20} y={H - 6} textAnchor="end" className="vz-tick">bigger model, less data</text>
      </svg>
      <div className="vz-cmp" style={{ marginTop: 8, gridTemplateColumns: 'repeat(3,minmax(0,1fr))' }}>
        <div className="out"><span className="wg-k">Compute-optimal (fitted law)</span><span className="vz-big">{(opt.N / 1e9).toFixed(1)} B</span><span className="wg-note">{(opt.D / 1e9).toFixed(0)} B tokens · {opt.ratio.toFixed(0)} tokens per parameter · loss {opt.loss.toFixed(3)}</span></div>
        <div><span className="wg-k">The 20-tokens-per-parameter rule</span><span className="vz-big">{(Math.sqrt(C / 120) / 1e9).toFixed(1)} B</span><span className="wg-note">N = √(C ÷ 120), D = 20·N</span></div>
        <div><span className="wg-k">Your choice</span><span className="vz-big">{L.toFixed(3)}</span><span className="wg-note">N = {(N / 1e9).toFixed(2)} B, D = {(D / 1e9).toFixed(0)} B tokens</span></div>
      </div>
      <p className="wg-note">C = 6·N·D FLOPs is the usual estimate of training compute. The loss comes from the formula L = {CHIN.E} + {CHIN.A}/N^{CHIN.alpha} + {CHIN.B}/D^{CHIN.beta}, the Chinchilla form with constants from a published re-fit (Besiroglu et al., 2024); treat it as a fitted curve, not a law of nature. {spec.note || ''}</p>
    </div>
  );
}

// ---------------------------------------------------------------- emergent (smooth skill, sharp-looking metric)
function Emergent({ spec, fig }) {
  const [L, setL] = useState(spec.length ?? 8);
  const cfg = spec.curve || {};
  const xs = Array.from({ length: 61 }, (_, i) => 7 + (i / 60) * 5);
  const rows = xs.map((x) => ({ x, ...emergence(cfg, x, L) }));
  const W = 560, H = 170;
  const sx = (x) => 36 + ((x - 7) / 5) * (W - 56);
  const sy = (y) => H - 24 - y * (H - 40);
  const half = rows.find((r) => r.em >= 0.5);
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <Slider label="Answer length L (tokens that must all be right)" min={1} max={30} step={1} value={L} onChange={setL} />
      <svg viewBox={`0 0 ${W} ${H}`} className="vz-plot" role="img" aria-label="Per-token accuracy and exact-match accuracy against model size">
        {[0, 0.5, 1].map((y) => <g key={y}><line x1="36" x2={W - 20} y1={sy(y)} y2={sy(y)} className="vz-grid" /><text x="32" y={sy(y) + 4} textAnchor="end" className="vz-tick">{y}</text></g>)}
        <path d={rows.map((r, i) => `${i ? 'L' : 'M'}${sx(r.x).toFixed(1)} ${sy(r.p).toFixed(1)}`).join(' ')} fill="none" stroke="var(--cyan)" strokeWidth="2.4" />
        <path d={rows.map((r, i) => `${i ? 'L' : 'M'}${sx(r.x).toFixed(1)} ${sy(r.em).toFixed(1)}`).join(' ')} fill="none" stroke="var(--amber)" strokeWidth="2.8" />
        {[7, 8, 9, 10, 11, 12].map((x) => <text key={x} x={sx(x)} y={H - 6} textAnchor="middle" className="vz-tick">1e{x}</text>)}
      </svg>
      <div className="wg-row vz-legend">
        <span className="vz-key"><i style={{ background: 'var(--cyan)' }} />per-token accuracy p (partial credit): rises smoothly</span>
        <span className="vz-key"><i style={{ background: 'var(--amber)' }} />exact match p^{L} (all {L} tokens right)</span>
      </div>
      <p className="wg-p" aria-live="polite" style={{ margin: '8px 0 0' }}>{half ? `Exact match first passes 50% at about ${(10 ** half.x / 1e9).toFixed(1)} billion parameters.` : 'Exact match never reaches 50% in this range.'} At L = 1 the two curves are the same line; the longer the answer, the more the amber curve looks like a sudden jump even though the skill underneath improves smoothly.</p>
      <p className="wg-note">{spec.note || 'Illustrative curves, not measurements of any real model: parameters on the horizontal axis (log scale), a logistic skill curve underneath.'}</p>
    </div>
  );
}

// ---------------------------------------------------------------- clipsim (a similarity matrix between images and texts)
function ClipSim({ spec, fig }) {
  const [scale, setScale] = useState(spec.scale ?? 10);
  const [sel, setSel] = useState(0);
  const R = useMemo(() => clipMatrix(spec.images.map((x) => x.v), spec.texts.map((x) => x.v), scale), [spec, scale]);
  const f = (q) => Number(q.toFixed(2)).toString();
  const row = R.rows[sel];
  const best = row.indexOf(Math.max(...row));
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <Slider label="Logit scale (1 ÷ temperature)" min={1} max={50} step={1} value={scale} onChange={setScale} />
      <div className="vz-scroll">
        <table className="vz-mlp vz-attn" aria-label="Cosine similarity of every image with every text">
          <thead><tr><th>image ↓ text →</th>{spec.texts.map((t, j) => <th key={j}>{t.label}</th>)}</tr></thead>
          <tbody>
            {spec.images.map((im, i) => (
              <tr key={i} className={i === sel ? 'sel' : ''} onClick={() => setSel(i)} style={{ cursor: 'pointer' }}>
                <td><button type="button" className="vz-link" onClick={() => setSel(i)}>{im.label}</button></td>
                {R.cos[i].map((q, j) => <td key={j} style={{ background: `color-mix(in srgb, ${i === j ? 'var(--accent)' : 'var(--cyan)'} ${Math.round(Math.max(0, q) * 55)}%, var(--surface2))`, fontWeight: i === j ? 800 : 500 }}>{f(q)}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="wg-note" style={{ margin: '4px 0 8px' }}>Cosine similarity; the diagonal (bold, green) holds the true pairs. Select an image to see its zero-shot probabilities.</p>
      <p className="wg-p" aria-live="polite" style={{ margin: '0 0 6px' }}><strong>Zero-shot for “{spec.images[sel].label}”</strong>: softmax(scale × cosine) over the texts →</p>
      <ul className="wg-bars">
        {row.map((p, j) => <li key={j} style={{ gridTemplateColumns: '170px minmax(0,1fr) 56px' }}><span className="wg-bar-l" style={{ fontFamily: 'inherit' }}>{spec.texts[j].label}</span><span className="wg-bar-track"><span className={'wg-bar-fill' + (j === best ? '' : ' warn')} style={{ width: `${p * 100}%` }} /></span><span className="wg-bar-p">{f(p)}</span></li>)}
      </ul>
      <div className="vz-cmp" style={{ marginTop: 10, gridTemplateColumns: 'repeat(3,minmax(0,1fr))' }}>
        <div><span className="wg-k">Image → text loss</span><span className="vz-big">{f(R.lossImage)}</span><span className="wg-note">mean −ln p(true text)</span></div>
        <div><span className="wg-k">Text → image loss</span><span className="vz-big">{f(R.lossText)}</span><span className="wg-note">the same, down the columns</span></div>
        <div className="out"><span className="wg-k">Contrastive loss (average)</span><span className="vz-big">{f(R.loss)}</span><span className="wg-note">training pushes this down</span></div>
      </div>
      <p className="wg-note">{spec.note || 'Illustrative vectors written for teaching; CLIP learns its vectors from hundreds of millions of image–text pairs.'}</p>
    </div>
  );
}

// ---------------------------------------------------------------- patchshuffle (a bag of patches has no layout: why position embeddings are needed)
function PatchShuffle({ spec, fig }) {
  const img = spec.image;
  const n = img.length;
  const ps = spec.patch ?? 4;
  const g = n / ps;
  const [shuffled, setShuffled] = useState(false);
  const [withPos, setWithPos] = useState(false);
  const perm = useMemo(() => patchPermutation(g, spec.seed ?? 4), [g, spec]);
  const order = shuffled ? perm : Array.from({ length: g * g }, (_, i) => i);
  const moved = order.filter((q, i) => q !== i).length;
  const patchCells = (k) => Array.from({ length: ps * ps }, (_, q) => { const r = Math.floor(k / g), c = k % g; const v = img[r * ps + Math.floor(q / ps)][c * ps + (q % ps)]; return <i key={q} style={{ background: `rgb(${255 - v * 28},${255 - v * 28},${255 - v * 28})` }} />; });
  const Tile = ({ k, pos }) => (
    <span className="vz-sp-tile" title={`patch ${k + 1}${withPos ? `, position ${pos + 1}` : ''}`}>
      <span className="vz-px" style={{ gridTemplateColumns: `repeat(${ps}, 1fr)` }}>{patchCells(k)}</span>
      {withPos ? <b>+p{pos + 1}</b> : null}
    </span>
  );
  const sameBag = true;
  const seqDiff = moved;
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <div className="wg-row" style={{ marginBottom: 10 }}>
        <button type="button" className={'wg-chip' + (!shuffled ? ' on' : '')} aria-pressed={!shuffled} onClick={() => setShuffled(false)}>original picture</button>
        <button type="button" className={'wg-chip' + (shuffled ? ' on' : '')} aria-pressed={shuffled} onClick={() => setShuffled(true)}>patches shuffled</button>
        <button type="button" className={'wg-chip' + (withPos ? ' on' : '')} aria-pressed={withPos} onClick={() => setWithPos(!withPos)}>add position embeddings</button>
      </div>
      <div className="vz-two">
        <div>
          <span className="wg-k">The picture the model is given</span>
          <div className="vz-sp-grid" style={{ gridTemplateColumns: `repeat(${g}, 1fr)` }}>{order.map((k, i) => <Tile key={i} k={k} pos={i} />)}</div>
        </div>
        <div>
          <p className="wg-p" style={{ margin: '0 0 6px' }}><strong>As a set of tokens:</strong> {sameBag ? 'the same patches, just in another order.' : ''} Reading them in the order shown, {shuffled ? `${moved} of ${g * g}` : `0 of ${g * g}`} patches are not where they started.</p>
          {withPos ? <p className="wg-p" style={{ margin: '0 0 6px' }}><strong>With position embeddings</strong> each token is “patch content + p<sub>position</sub>”. After the shuffle {seqDiff} of the {g * g} tokens are different vectors from the original, so the model can tell the two pictures apart.</p> : <p className="wg-p" style={{ margin: '0 0 6px' }}><strong>Without position embeddings</strong> attention treats the tokens as a bag: it gives the same set of outputs for the original and for the shuffle (just reordered). The model cannot tell where anything was.</p>}
          <p className="wg-note">Switch “add position embeddings” on and compare the two pictures.</p>
        </div>
      </div>
      <p className="wg-note">{spec.note || ''}</p>
    </div>
  );
}

// ---------------------------------------------------------------- fusion (how a model can combine modalities)
function Fusion({ spec, fig }) {
  const [i, setI] = useState(0);
  const f = spec.patterns[i];
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <div className="wg-tabs" role="tablist">{spec.patterns.map((p, k) => <button key={k} type="button" role="tab" aria-selected={i === k} className={'wg-tab' + (i === k ? ' on' : '')} onClick={() => setI(k)}>{p.name}</button>)}</div>
      <div className="vz-fusion" style={{ marginTop: 10 }}>
        {f.inputs.map((x, k) => <span key={'i' + k} className="vz-fchip in">{x}</span>)}
        <span className="vz-farrow" aria-hidden="true">→</span>
        {f.stages.map((x, k) => <React.Fragment key={'s' + k}><span className="vz-fchip st">{x}</span>{k < f.stages.length - 1 ? <span className="vz-farrow" aria-hidden="true">→</span> : null}</React.Fragment>)}
        <span className="vz-farrow" aria-hidden="true">→</span>
        <span className="vz-fchip out">{f.output}</span>
      </div>
      <p className="wg-p" style={{ margin: '10px 0 4px' }}>{f.how}</p>
      <p className="wg-p" style={{ margin: '4px 0' }}><strong>Examples:</strong> {f.examples}</p>
      <p className="wg-p" style={{ margin: '4px 0' }}><strong>Good at:</strong> {f.good} <strong>Watch out:</strong> {f.bad}</p>
      <p className="wg-note">{spec.note || ''}</p>
    </div>
  );
}

// ---------------------------------------------------------------- ganlab (a one-dimensional GAN trained live)
function GanLab({ spec, fig }) {
  const mu = spec.mu ?? 4, sigma = spec.sigma ?? 1;
  const [opts, setOpts] = useState({ lr: spec.lr ?? 0.1, lrG: spec.lrG ?? 0.02, dSteps: spec.dSteps ?? 5, nonSat: spec.nonSat ?? true });
  const [run, setRun] = useState(false);
  const st = useRef({ s: ganInit(), lossD: null, lossG: null });
  const [, force] = useState(0);
  const optsRef = useRef(opts); optsRef.current = opts;
  const step = (k) => { for (let i = 0; i < k; i++) { const r = ganStep(st.current.s, { mu, sigma, batch: 64, seed: spec.seed ?? 11, ...optsRef.current }); st.current = { s: r.state, lossD: r.lossD, lossG: r.lossG }; } force((n) => n + 1); };
  useEffect(() => { if (!run) return undefined; const id = setInterval(() => step(8), 40); return () => clearInterval(id); }, [run]); // eslint-disable-line react-hooks/exhaustive-deps
  const reset = () => { setRun(false); st.current = { s: ganInit(), lossD: null, lossG: null }; force((n) => n + 1); };
  const { s, lossD, lossG } = st.current;
  const pdf = (x, m, sd) => Math.exp(-((x - m) ** 2) / (2 * sd * sd)) / (sd * Math.sqrt(2 * Math.PI));
  const lo = -2, hi = 10, W = 560, H = 190;
  const sx = (x) => 30 + ((x - lo) / (hi - lo)) * (W - 50);
  const sa = Math.max(Math.abs(s.a), 0.05);
  const ymax = Math.max(0.45, pdf(0, 0, sa) * 1.05);
  const sy = (p) => H - 24 - (Math.min(p, ymax) / ymax) * (H - 44);
  const xs = Array.from({ length: 121 }, (_, i) => lo + ((hi - lo) * i) / 120);
  const path = (f) => xs.map((x, i) => `${i ? 'L' : 'M'}${sx(x).toFixed(1)} ${sy(f(x)).toFixed(1)}`).join(' ');
  const dPath = xs.map((x, i) => `${i ? 'L' : 'M'}${sx(x).toFixed(1)} ${(H - 24 - sigmoid(s.w1 * x + s.w2 * x * x + s.c) * (H - 44)).toFixed(1)}`).join(' ');
  const set = (p) => setOpts((o) => ({ ...o, ...p }));
  const f2 = (q) => (q == null ? '–' : Number(q.toFixed(2)).toString());
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <svg viewBox={`0 0 ${W} ${H}`} className="vz-plot" role="img" aria-label={`Real data centred at ${mu}, generated data centred at ${f2(s.b)} with spread ${f2(sa)}, after ${s.step} steps`}>
        <line x1="30" x2={W - 20} y1={H - 24} y2={H - 24} className="vz-grid" />
        <path d={path((x) => pdf(x, mu, sigma))} fill="none" stroke="var(--text)" strokeWidth="2.4" />
        <path d={path((x) => pdf(x, s.b, sa))} fill="none" stroke="var(--accent)" strokeWidth="2.6" />
        <path d={dPath} fill="none" stroke="var(--amber)" strokeWidth="2" strokeDasharray="6 4" />
        {[0, 2, 4, 6, 8].map((x) => <text key={x} x={sx(x)} y={H - 6} textAnchor="middle" className="vz-tick">{x}</text>)}
      </svg>
      <div className="wg-row vz-legend">
        <span className="vz-key"><i style={{ background: 'var(--text)' }} />real data N({mu}, {sigma}²)</span>
        <span className="vz-key"><i style={{ background: 'var(--accent)' }} />generator: N({f2(s.b)}, {f2(sa)}²)</span>
        <span className="vz-key"><i style={{ background: 'var(--amber)' }} />discriminator D(x) (0 to 1)</span>
      </div>
      <p className="wg-p" aria-live="polite" style={{ margin: '8px 0' }}>Step <strong>{s.step}</strong> · generator mean <strong>{f2(s.b)}</strong> (target {mu}) · generator spread <strong>{f2(sa)}</strong> (target {sigma}) · D loss <strong>{f2(lossD)}</strong> · G loss <strong>{f2(lossG)}</strong></p>
      <Slider label="Discriminator learning rate" min={0.01} max={0.3} step={0.01} value={opts.lr} onChange={(q) => set({ lr: q })} show={(q) => q.toFixed(2)} />
      <Slider label="Generator learning rate" min={0.005} max={0.2} step={0.005} value={opts.lrG} onChange={(q) => set({ lrG: q })} show={(q) => q.toFixed(3)} />
      <Slider label="Discriminator steps per generator step" min={1} max={10} step={1} value={opts.dSteps} onChange={(q) => set({ dSteps: q })} />
      <div className="wg-row" style={{ marginTop: 6 }}>
        <button type="button" className="wg-btn" onClick={() => setRun((r) => !r)}>{run ? 'Pause' : 'Train'}</button>
        <button type="button" className="wg-btn ghost" onClick={() => step(10)}>+10 steps</button>
        <button type="button" className="wg-btn ghost" onClick={reset}>Reset</button>
        <button type="button" className={'wg-chip' + (opts.nonSat ? ' on' : '')} aria-pressed={opts.nonSat} onClick={() => set({ nonSat: !opts.nonSat })}>non-saturating generator loss</button>
      </div>
      <p className="wg-note">Generator: G(z) = a·z + b with z ~ N(0, 1). Discriminator: D(x) = σ(w₁x + w₂x² + c). Each step is a few discriminator updates followed by one generator update, on fresh samples. Training is a tug of war, not a smooth descent: the curves overshoot and wobble. {spec.note || ''}</p>
    </div>
  );
}

// ---------------------------------------------------------------- upsample (nearest, bilinear and transpose convolutions)
function Upsample({ spec, fig }) {
  const g = spec.input;
  const methods = [
    { id: 'nearest', label: 'Nearest neighbour (no learning)' },
    { id: 'bilinear', label: 'Bilinear (no learning)' },
    { id: 't2', label: 'Transpose conv k=2, s=2', k: 2, s: 2, p: 0 },
    { id: 't3', label: 'Transpose conv k=3, s=2', k: 3, s: 2, p: 0 },
    { id: 't4', label: 'Transpose conv k=4, s=2, p=1', k: 4, s: 2, p: 1 },
  ];
  const [mi, setMi] = useState(spec.start ?? 4);
  const m = methods[mi];
  const kernelFor = (k) => Array.from({ length: k }, () => Array(k).fill(1));
  const out = useMemo(() => (m.id === 'nearest' ? upsampleNearest(g) : m.id === 'bilinear' ? upsampleBilinear(g) : transposeConvPad(g, kernelFor(m.k), m.s, m.p)), [g, m]);
  const cover = useMemo(() => (m.k ? transposeConvPad(g.map((r) => r.map(() => 1)), kernelFor(m.k), m.s, m.p) : null), [g, m]);
  const inner = cover && cover.length > 3 ? cover.slice(1, -1).map((r) => r.slice(1, -1)) : cover;
  const uneven = cover && new Set(inner.flat()).size > 1;
  const n = g.length;
  const tone = (v) => `color-mix(in srgb, var(--accent) ${Math.round(Math.min(1, Math.abs(v) / (spec.max || 9)) * 70)}%, var(--surface2))`;
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <div className="wg-tabs" role="tablist">{methods.map((x, k) => <button key={x.id} type="button" role="tab" aria-selected={mi === k} className={'wg-tab' + (mi === k ? ' on' : '')} onClick={() => setMi(k)}>{x.label}</button>)}</div>
      <div className="vz-conv" style={{ marginTop: 10, alignItems: 'flex-start' }}>
        <div><span className="wg-k">Input {n}×{n}</span><Grid data={g} tone={tone} /></div>
        <span className="vz-op">→</span>
        <div><span className="wg-k">Output {out.length}×{out[0].length}{m.k ? ` = (${n} − 1)·${m.s} − 2·${m.p} + ${m.k}` : ' = 2 × the input'}</span><Grid data={out.map((r) => r.map((q) => Math.round(q * 100) / 100))} tone={tone} cls="out" /></div>
        {cover ? <div><span className="wg-k">How many kernel copies reach each output cell</span><Grid data={cover} tone={(v) => `color-mix(in srgb, var(--amber) ${Math.round((v / Math.max(...cover.flat())) * 70)}%, var(--surface2))`} /></div> : null}
      </div>
      <p className="wg-p" aria-live="polite" style={{ margin: '10px 0 0' }}>{!m.k ? 'A fixed rule: nothing here can be learned. It only copies or blends neighbouring values.' : uneven ? `Uneven overlap: some output cells receive more copies of the kernel than others (${Math.min(...cover.flat())} to ${Math.max(...cover.flat())}), so even a constant input gives a pattern of bright and dark cells. This is the checkerboard artefact.` : `Even overlap in the middle: every inner output cell receives ${inner[0][0]} copies of the kernel, so a constant input stays constant there (the border cells see fewer copies because the output is cropped).`} {m.k ? 'The kernel here is all ones to show the overlap; in a real layer its numbers are learned.' : ''}</p>
      <p className="wg-note">{spec.note || ''}</p>
    </div>
  );
}

// ---------------------------------------------------------------- schedule (the noise schedule and the signal-to-noise trade)
function Schedule({ spec, fig }) {
  const T = spec.T || 1000;
  const [kind, setKind] = useState(spec.kind || 'linear');
  const [t, setT] = useState(spec.t ?? 300);
  const sch = useMemo(() => ({ linear: schedule(T, 'linear'), cosine: schedule(T, 'cosine') }), [T]);
  const cur = sch[kind], other = sch[kind === 'linear' ? 'cosine' : 'linear'];
  const ab = cur.abar[t];
  const W = 560, H = 170;
  const sx = (i) => 36 + (i / T) * (W - 56), sy = (y) => H - 24 - y * (H - 40);
  const path = (arr, f) => arr.map((q, i) => `${i ? 'L' : 'M'}${sx(i).toFixed(1)} ${sy(f(q)).toFixed(1)}`).join(' ');
  const f = (q) => (q >= 0.01 ? Number(q.toFixed(4)).toString() : q.toExponential(2));
  const snr = ab / (1 - ab);
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <div className="wg-tabs" role="tablist">{['linear', 'cosine'].map((k) => <button key={k} type="button" role="tab" aria-selected={kind === k} className={'wg-tab' + (kind === k ? ' on' : '')} onClick={() => setKind(k)}>{k === 'linear' ? 'Linear schedule (β from 0.0001 to 0.02)' : 'Cosine schedule'}</button>)}</div>
      <Slider label="Step t" min={0} max={T} step={1} value={t} onChange={setT} />
      <svg viewBox={`0 0 ${W} ${H}`} className="vz-plot" role="img" aria-label={`At step ${t} the signal share ᾱ is ${f(ab)}`}>
        {[0, 0.5, 1].map((y) => <g key={y}><line x1="36" x2={W - 20} y1={sy(y)} y2={sy(y)} className="vz-grid" /><text x="32" y={sy(y) + 4} textAnchor="end" className="vz-tick">{y}</text></g>)}
        <path d={path(other.abar, (q) => q)} fill="none" stroke="var(--text3)" strokeWidth="1.6" strokeDasharray="5 4" />
        <path d={path(cur.abar, (q) => Math.sqrt(q))} fill="none" stroke="var(--accent)" strokeWidth="2.4" />
        <path d={path(cur.abar, (q) => Math.sqrt(1 - q))} fill="none" stroke="var(--amber)" strokeWidth="2.4" />
        <path d={path(cur.abar, (q) => q)} fill="none" stroke="var(--cyan)" strokeWidth="2.4" />
        <line x1={sx(t)} x2={sx(t)} y1="8" y2={H - 24} stroke="var(--text)" strokeDasharray="3 3" />
        <text x="36" y={H - 6} className="vz-tick">t = 0 (clean)</text>
        <text x={W - 20} y={H - 6} textAnchor="end" className="vz-tick">t = {T} (noise)</text>
      </svg>
      <div className="wg-row vz-legend">
        <span className="vz-key"><i style={{ background: 'var(--cyan)' }} />ᾱ_t, the signal share</span>
        <span className="vz-key"><i style={{ background: 'var(--accent)' }} />√ᾱ_t, the weight of the picture</span>
        <span className="vz-key"><i style={{ background: 'var(--amber)' }} />√(1 − ᾱ_t), the weight of the noise</span>
        <span className="vz-key"><i style={{ background: 'var(--text3)' }} />ᾱ_t of the other schedule</span>
      </div>
      <div className="vz-cmp" style={{ marginTop: 10, gridTemplateColumns: 'repeat(3,minmax(0,1fr))' }}>
        <div className="out"><span className="wg-k" style={{ textTransform: "none" }}>ᾱ_t at t = {t}</span><span className="vz-big">{f(ab)}</span><span className="wg-note">fraction of the original signal power that is left</span></div>
        <div><span className="wg-k" style={{ textTransform: "none" }}>x_t = {f(Math.sqrt(ab))}·x₀ + {f(Math.sqrt(1 - ab))}·ε</span><span className="vz-big">{f(Math.sqrt(ab))} | {f(Math.sqrt(1 - ab))}</span><span className="wg-note">weight of the picture | weight of the noise</span></div>
        <div><span className="wg-k" style={{ textTransform: "none" }}>Signal-to-noise ratio ᾱ ÷ (1 − ᾱ)</span><span className="vz-big">{f(snr)}</span><span className="wg-note">above 1 the picture still dominates</span></div>
      </div>
      <p className="wg-note">{spec.note || ''}</p>
    </div>
  );
}

// ---------------------------------------------------------------- sampler1d (the reverse process run with the exact denoiser)
function Sampler1d({ spec, fig }) {
  const opt = { means: spec.means || [-2, 3], weights: spec.weights || [0.5, 0.5], sd0: spec.sd0 ?? 0.4, T: spec.T || 200, n: spec.n || 600, seed: spec.seed || 5 };
  const [stoch, setStoch] = useState(spec.stochastic ?? true);
  const R = useMemo(() => diffSample1d({ ...opt, stochastic: stoch }), [stoch, spec]); // eslint-disable-line react-hooks/exhaustive-deps
  const [k, setK] = useState(0);
  const [play, setPlay] = useState(false);
  const kRef = useRef(0); kRef.current = k;
  useEffect(() => {
    if (!play) return undefined;
    const id = setInterval(() => { if (kRef.current >= opt.T) { setPlay(false); return; } setK(Math.min(opt.T, kRef.current + 3)); }, 60);
    return () => clearInterval(id);
  }, [play]); // eslint-disable-line react-hooks/exhaustive-deps
  const t = opt.T - k;
  const xs = R.traj[t];
  const lo = -5, hi = 6, bins = 44, W = 560, H = 170;
  const counts = Array(bins).fill(0);
  xs.forEach((q) => { const b = Math.floor(((q - lo) / (hi - lo)) * bins); if (b >= 0 && b < bins) counts[b]++; });
  const mx = Math.max(...counts, 1);
  const dens = (x) => opt.weights.reduce((s, w, i) => s + w * Math.exp(-((x - opt.means[i]) ** 2) / (2 * opt.sd0 ** 2)) / (opt.sd0 * Math.sqrt(2 * Math.PI)), 0);
  const dmax = Math.max(...Array.from({ length: 200 }, (_, i) => dens(lo + ((hi - lo) * i) / 199)));
  const sx = (x) => 20 + ((x - lo) / (hi - lo)) * (W - 40);
  const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
  const sd = Math.sqrt(xs.reduce((a, b) => a + (b - mean) ** 2, 0) / xs.length);
  const f2_ = (q) => Number(q.toFixed(2)).toString();
  const ab = R.abar[t];
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <Slider label="Reverse steps taken" min={0} max={opt.T} step={1} value={k} onChange={setK} show={(q) => `${q} of ${opt.T}  (now at t = ${opt.T - q})`} />
      <svg viewBox={`0 0 ${W} ${H}`} className="vz-plot" role="img" aria-label={`Histogram of ${xs.length} particles at step t = ${t}: mean ${f2_(mean)}, spread ${f2_(sd)}`}>
        <line x1="20" x2={W - 20} y1={H - 24} y2={H - 24} className="vz-grid" />
        {counts.map((c, i) => <rect key={i} x={20 + (i / bins) * (W - 40) + 1} width={(W - 40) / bins - 2} y={H - 24 - (c / mx) * (H - 44)} height={(c / mx) * (H - 44)} fill="var(--accent)" opacity="0.75" />)}
        <path d={Array.from({ length: 200 }, (_, i) => { const x = lo + ((hi - lo) * i) / 199; return `${i ? 'L' : 'M'}${sx(x).toFixed(1)} ${(H - 24 - (dens(x) / dmax) * (H - 44)).toFixed(1)}`; }).join(' ')} fill="none" stroke="var(--text)" strokeWidth="2" strokeDasharray="5 4" />
        {[-4, -2, 0, 2, 4].map((x) => <text key={x} x={sx(x)} y={H - 6} textAnchor="middle" className="vz-tick">{x}</text>)}
      </svg>
      <div className="wg-row vz-legend">
        <span className="vz-key"><i style={{ background: 'var(--accent)' }} />{xs.length} particles now</span>
        <span className="vz-key"><i style={{ background: 'var(--text)' }} />the data distribution to recover</span>
      </div>
      <p className="wg-p" aria-live="polite" style={{ margin: '8px 0' }}>Step t = <strong>{t}</strong> · signal share ᾱ_t = <strong>{ab >= 0.01 ? f2_(ab) : ab.toExponential(1)}</strong> · mean <strong>{f2_(mean)}</strong> · spread <strong>{f2_(sd)}</strong>. {t === opt.T ? 'Pure noise: a bell curve centred at 0.' : t === 0 ? 'Done: two separate groups appeared out of one bell curve.' : 'The denoiser is nudging every particle towards where data are likely.'}</p>
      <div className="wg-row">
        <button type="button" className="wg-btn" onClick={() => { if (k >= opt.T) setK(0); setPlay(!play); }}>{play ? 'Pause' : k >= opt.T ? 'Replay' : 'Play the reverse process'}</button>
        <button type="button" className="wg-btn ghost" onClick={() => { setPlay(false); setK(0); }}>Back to noise</button>
        <button type="button" className={'wg-chip' + (stoch ? ' on' : '')} aria-pressed={stoch} onClick={() => setStoch(!stoch)}>add fresh noise at each step</button>
      </div>
      <p className="wg-note">Here the denoiser is the exact one for this simple data set (a mixture of two bell curves), so no network is needed: the reverse step is x_(t−1) = (x_t − β_t/√(1−ᾱ_t)·ε̂)/√(1−β_t) + σ_t·z. In a real model a neural network learns ε̂ from examples. {spec.note || ''}</p>
    </div>
  );
}

// ---------------------------------------------------------------- unet (shapes, skips and parameters of a small UNet denoiser)
function Unet({ spec, fig }) {
  const [cfg, setCfg] = useState({ base: spec.base ?? 16, levels: spec.levels ?? 3, timeDim: spec.timeDim ?? 32 });
  const size = spec.size ?? 16, inCh = spec.inCh ?? 1;
  const ok = size % 2 ** (cfg.levels - 1) === 0;
  const U = useMemo(() => unetShapes({ size, inCh, base: cfg.base, levels: cfg.levels, timeDim: cfg.timeDim }), [size, inCh, cfg]);
  const L = cfg.levels;
  const W = 560, H = 40 + L * 52;
  const colW = (W - 40) / (2 * L - 1);
  const pos = (col, lvl) => [20 + col * colW + colW / 2, 30 + lvl * 52];
  const boxes = [];
  for (let i = 0; i < L; i++) boxes.push({ col: i, lvl: i, label: `${cfg.base * 2 ** i}c · ${size / 2 ** i}²`, kind: i === L - 1 ? 'bn' : 'down' });
  for (let i = L - 2; i >= 0; i--) boxes.push({ col: 2 * L - 2 - i, lvl: i, label: `${cfg.base * 2 ** i}c · ${size / 2 ** i}²`, kind: 'up' });
  const fmt = (n) => n.toLocaleString();
  const set = (p) => setCfg((c) => ({ ...c, ...p }));
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <div className="wg-row" style={{ gap: 12, marginBottom: 8 }}>
        <label className="vz-sel">base channels <select value={cfg.base} onChange={(e) => set({ base: +e.target.value })}>{[8, 16, 32, 64].map((q) => <option key={q}>{q}</option>)}</select></label>
        <label className="vz-sel">levels <select value={cfg.levels} onChange={(e) => set({ levels: +e.target.value })}>{[2, 3, 4].map((q) => <option key={q}>{q}</option>)}</select></label>
        <label className="vz-sel">time-embedding size <select value={cfg.timeDim} onChange={(e) => set({ timeDim: +e.target.value })}>{[16, 32, 64, 128].map((q) => <option key={q}>{q}</option>)}</select></label>
      </div>
      {!ok ? <p className="wg-why bad" role="alert">{size} cannot be halved {L - 1} times: use fewer levels.</p> : null}
      <svg viewBox={`0 0 ${W} ${H}`} className="vz-netsvg" role="img" aria-label="A U-shaped network: the picture is shrunk step by step, then grown back, with skip connections across the U">
        {boxes.filter((b) => b.kind === 'up').map((b, k) => { const [x1, y1] = pos(2 * L - 2 - b.lvl, b.lvl), [x0] = pos(b.lvl, b.lvl); return <line key={'s' + k} x1={x0 + 34} y1={y1} x2={x1 - 34} y2={y1} stroke="var(--amber)" strokeWidth="2" strokeDasharray="5 4" />; })}
        {boxes.slice(0, -1).map((b, k) => { const n = boxes[k + 1]; const [x1, y1] = pos(b.col, b.lvl), [x2, y2] = pos(n.col, n.lvl); return <line key={'a' + k} x1={x1} y1={y1} x2={x2} y2={y2} stroke="var(--text3)" strokeWidth="1.6" />; })}
        {boxes.map((b, k) => { const [x, y] = pos(b.col, b.lvl); return <g key={k}><rect x={x - 34} y={y - 15} width="68" height="30" rx="7" fill="var(--card-bg)" /><rect x={x - 34} y={y - 15} width="68" height="30" rx="7" fill={b.kind === 'bn' ? 'var(--amber-soft-bg)' : b.kind === 'down' ? 'var(--cyan-soft-bg)' : 'var(--accent-soft-bg)'} stroke="var(--surface2-border)" /><text x={x} y={y + 4} textAnchor="middle" className="vz-lab" style={{ fontSize: 10.5 }}>{b.label}</text></g>; })}
        <text x="20" y={H - 4} className="vz-tick">input picture + time t</text>
        <text x={W - 20} y={H - 4} textAnchor="end" className="vz-tick">predicted noise (same shape as the input)</text>
      </svg>
      <p className="wg-note" style={{ margin: '0 0 8px' }}>Blue boxes shrink the picture, the orange box is the bottleneck, green boxes grow it back; dashed orange lines are the skip connections that copy the early, detailed maps across to the decoder.</p>
      <div className="vz-scroll">
        <table className="vz-mlp" aria-label="Layers, shapes and parameters of the UNet">
          <thead><tr><th>Layer</th><th>Output shape (channels × H × W)</th><th>Parameters</th></tr></thead>
          <tbody>{U.rows.map((r, i) => <tr key={i}><td>{r.name}</td><td>{r.shape}</td><td>{fmt(r.params)}</td></tr>)}<tr className="tot"><td colSpan={2}><strong>Total parameters</strong></td><td><strong>{fmt(U.total)}</strong></td></tr></tbody>
        </table>
      </div>
      <p className="wg-note">Every block adds a learned projection of the time embedding to its feature maps, so the same network knows how noisy the input is. The 16 × 16 greyscale pictures are an example size. {spec.note || ''}</p>
    </div>
  );
}

// ---------------------------------------------------------------- prefpair (reward model: which answer do people prefer?)
function PrefPair({ spec, fig }) {
  const [scores, setScores] = useState(spec.pairs.map((p) => [p.chosenScore ?? 0, p.rejectedScore ?? 0]));
  const f = (q) => Number(q.toFixed(3)).toString();
  const rows = scores.map(([rw, rl]) => ({ p: prefProb(rw, rl), l: prefLoss(rw, rl), ok: rw > rl }));
  const mean = rows.reduce((s, r) => s + r.l, 0) / rows.length;
  const acc = rows.filter((r) => r.ok).length / rows.length;
  const set = (i, j, q) => setScores((s) => s.map((r, k) => (k === i ? r.map((x, jj) => (jj === j ? q : x)) : r)));
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      {spec.pairs.map((p, i) => (
        <div key={i} className="vz-pref">
          <p className="wg-p" style={{ margin: '0 0 4px' }}><strong>Prompt:</strong> {p.prompt}</p>
          <div className="vz-two">
            <div><span className="wg-k" style={{ textTransform: 'none' }}>✓ Chosen by the human</span><p className="wg-p" style={{ margin: '2px 0 4px' }}>{p.chosen}</p><Slider label="reward r(chosen)" min={-4} max={4} step={0.1} value={scores[i][0]} onChange={(q) => set(i, 0, q)} show={(q) => q.toFixed(1)} /></div>
            <div><span className="wg-k" style={{ textTransform: 'none' }}>✗ Rejected</span><p className="wg-p" style={{ margin: '2px 0 4px' }}>{p.rejected}</p><Slider label="reward r(rejected)" min={-4} max={4} step={0.1} value={scores[i][1]} onChange={(q) => set(i, 1, q)} show={(q) => q.toFixed(1)} /></div>
          </div>
          <p className="vz-calc" aria-live="polite">P(chosen preferred) = σ({f(scores[i][0])} − {f(scores[i][1])}) = <strong>{f(rows[i].p)}</strong> · loss = −ln {f(rows[i].p)} = <strong>{f(rows[i].l)}</strong> · {rows[i].ok ? 'ranked correctly' : 'ranked wrongly'}</p>
        </div>
      ))}
      <div className="vz-cmp" style={{ marginTop: 10, gridTemplateColumns: 'repeat(2,minmax(0,1fr))' }}>
        <div className="out"><span className="wg-k">Mean preference loss</span><span className="vz-big">{f(mean)}</span><span className="wg-note">training pushes this down; 0.693 = ln 2 means "no idea"</span></div>
        <div><span className="wg-k">Pairs ranked correctly</span><span className="vz-big">{Math.round(acc * 100)}%</span><span className="wg-note">{rows.filter((r) => r.ok).length} of {rows.length}</span></div>
      </div>
      <p className="wg-note">{spec.note || 'Prompts, answers and scores are made up for the demonstration. A real reward model is a language model with a scoring head, trained on many thousands of comparisons.'}</p>
    </div>
  );
}

// ---------------------------------------------------------------- chunker (cut a document into overlapping chunks)
function Chunker({ spec, fig }) {
  const words = useMemo(() => spec.text.trim().split(/\s+/), [spec]);
  const L = words.length;
  const [size, setSize] = useState(spec.size ?? 20);
  const [ov, setOv] = useState(spec.overlap ?? 4);
  const o = Math.min(ov, size - 1);
  const chunks = useMemo(() => chunkWords(words, size, o), [words, size, o]);
  const aw = spec.answer.trim().split(/\s+/);
  let a0 = -1;
  for (let i = 0; i + aw.length <= L; i++) if (aw.every((w, j) => words[i + j] === w)) { a0 = i; break; }
  const a1 = a0 + aw.length;
  const intact = a0 >= 0 && spanIntact(chunks, a0, a1);
  const home = intact ? chunks.findIndex(([s, e]) => s <= a0 && e >= a1) : -1;
  const hues = ['var(--accent-soft-bg)', 'var(--surface2)', 'var(--amber-soft-bg, #fdf1d8)'];
  const wordChunks = words.map((_, i) => chunks.map(([s, e], k) => (i >= s && i < e ? k : -1)).filter((k) => k >= 0));
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <Slider label="Chunk size (words)" min={4} max={Math.max(8, L)} step={1} value={size} onChange={(v) => setSize(v)} show={(q) => `${q} words`} />
      <Slider label="Overlap (words shared with the next chunk)" min={0} max={Math.max(0, size - 1)} step={1} value={o} onChange={setOv} show={(q) => `${q} words`} />
      <p className="wg-p" style={{ lineHeight: 2.1 }} aria-label="The document with each word shaded by its chunk">
        {words.map((w, i) => {
          const ks = wordChunks[i];
          const inAns = i >= a0 && i < a1;
          return <span key={i} title={`chunk ${ks.map((k) => k + 1).join(' and ')}`} style={{ background: hues[ks[0] % 3], borderBottom: ks.length > 1 ? '3px double var(--accent)' : '3px solid transparent', outline: inAns ? '1.5px solid var(--text2)' : 'none', borderRadius: 4, padding: '1px 3px', margin: '0 1px', fontWeight: inAns ? 700 : 400 }}>{w} </span>;
        })}
      </p>
      <div className="vz-cmp" style={{ gridTemplateColumns: 'repeat(3,minmax(0,1fr))' }}>
        <div><span className="wg-k">Words in the document</span><span className="vz-big">{L}</span><span className="wg-note">to be cut into chunks</span></div>
        <div className="out"><span className="wg-k">Chunks</span><span className="vz-big">{chunks.length}</span><span className="wg-note">{L <= size ? 'the whole text fits in one chunk' : `⌈(${L} − ${o}) ÷ (${size} − ${o})⌉ = ⌈${f2((L - o) / (size - o), 2)}⌉`}</span></div>
        <div><span className="wg-k">Words embedded in total</span><span className="vz-big">{chunks.reduce((q, [s, e]) => q + (e - s), 0)}</span><span className="wg-note">{f2(chunks.reduce((q, [s, e]) => q + (e - s), 0) / L, 2)}× the document (overlap repeats words)</span></div>
      </div>
      <p className="vz-calc" aria-live="polite" style={{ marginTop: 10 }}>{spec.question ? <><strong>Question:</strong> {spec.question} </> : null}→ <strong>{intact ? `The answer (outlined words) sits whole inside chunk ${home + 1}, so retrieval can find it.` : 'The answer (outlined words) is cut between chunks: no single chunk holds all of it.'}</strong></p>
      <p className="wg-note">{spec.note || 'Words stand in for tokens to keep the arithmetic readable; real systems count tokens, and often cut at sentence or paragraph boundaries first. Words with a double underline are repeated in two chunks.'}</p>
    </div>
  );
}

// ---------------------------------------------------------------- rankmetrics (precision, recall and reciprocal rank of a ranked list)
function RankMetrics({ spec, fig }) {
  const [qi, setQi] = useState(0);
  const q = spec.queries[qi];
  const [k, setK] = useState(spec.k ?? 3);
  const kk = Math.min(k, q.results.length);
  const m = rankMetrics(q.results.map((r) => r.rel), kk, q.totalRelevant);
  const f = (x) => Number(x.toFixed(3)).toString();
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <div className="wg-row" role="group" aria-label="Choose the question" style={{ marginBottom: 8 }}>
        {spec.queries.map((x, i) => <button key={i} type="button" className={'wg-chip' + (i === qi ? ' on' : '')} aria-pressed={i === qi} onClick={() => setQi(i)}>{x.label}</button>)}
      </div>
      <p className="wg-p"><strong>Question:</strong> {q.question} <em>({q.totalRelevant} passage{q.totalRelevant === 1 ? '' : 's'} in the whole collection can answer it)</em></p>
      <Slider label="Passages passed to the model (k)" min={1} max={q.results.length} step={1} value={kk} onChange={setK} />
      <ol className="vz-rank">
        {q.results.map((r, i) => (
          <li key={i} className={i < kk ? 'top' : ''} style={{ opacity: i < kk ? 1 : 0.55 }}>
            <div className="vz-rank-h"><strong>Rank {i + 1}</strong><span>{r.rel ? '✓ relevant' : '✗ not relevant'}</span></div>
            <p className="wg-note" style={{ margin: '4px 0 0' }}>{r.text}</p>
          </li>
        ))}
      </ol>
      <div className="vz-cmp" style={{ marginTop: 10, gridTemplateColumns: 'repeat(3,minmax(0,1fr))' }}>
        <div><span className="wg-k">Precision@{kk}</span><span className="vz-big">{f(m.precision)}</span><span className="wg-note">{m.hits} relevant ÷ {kk} retrieved</span></div>
        <div><span className="wg-k">Recall@{kk}</span><span className="vz-big">{f(m.recall)}</span><span className="wg-note">{m.hits} found ÷ {q.totalRelevant} that exist</span></div>
        <div className="out"><span className="wg-k">Reciprocal rank</span><span className="vz-big">{f(m.rr)}</span><span className="wg-note">{m.rr ? `1 ÷ ${Math.round(1 / m.rr)} (first relevant at rank ${Math.round(1 / m.rr)})` : 'no relevant passage in the top ' + kk}</span></div>
      </div>
      <p className="wg-note">{spec.note || 'The passages and their labels are made up for the demonstration. Raising k can only raise recall, but it usually lowers precision: more noise reaches the model.'}</p>
    </div>
  );
}

// ---------------------------------------------------------------- railstack (layers of guardrails, each missing some attacks)
function RailStack({ spec, fig }) {
  const [c, setC] = useState(spec.rails.map((r) => r.catch));
  const pass = railPassThrough(c);
  let rem = 1;
  const stages = [{ name: 'Attack arrives', share: 1, where: '' }, ...spec.rails.map((r, i) => { rem *= 1 - c[i]; return { name: r.name, share: rem, where: r.where }; })];
  const pct = (x) => (x * 100).toFixed(x < 0.1 ? 2 : 1) + ' %';
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      {spec.rails.map((r, i) => (
        <Slider key={i} label={`${r.name}${r.where ? ' (' + r.where + ')' : ''}: share of attacks it catches`} min={0} max={0.99} step={0.01} value={c[i]} onChange={(q) => setC((x) => x.map((v, k) => (k === i ? q : v)))} show={(q) => `${Math.round(q * 100)} %`} />
      ))}
      <div className="vz-scroll">
        {stages.map((st, i) => (
          <div key={i} style={{ display: 'grid', gridTemplateColumns: '170px 1fr 70px', gap: 8, alignItems: 'center', margin: '6px 0' }}>
            <span className="wg-note" style={{ margin: 0 }}>{i === 0 ? st.name : `After ${st.name}`}</span>
            <div className="wg-bar-track" style={{ height: 14 }}><span className="wg-bar-fill" style={{ width: `${Math.max(1, st.share * 100)}%`, background: i === 0 ? 'var(--amber)' : undefined }} /></div>
            <strong style={{ textAlign: 'right' }}>{pct(st.share)}</strong>
          </div>
        ))}
      </div>
      <div className="vz-cmp" style={{ gridTemplateColumns: 'repeat(2,minmax(0,1fr))' }}>
        <div className="out"><span className="wg-k">Attacks that get through every layer</span><span className="vz-big">{pct(pass)}</span><span className="wg-note">{c.map((q) => `(1 − ${f2(q, 2)})`).join(' × ')}</span></div>
        <div><span className="wg-k">Attacks stopped</span><span className="vz-big">{pct(1 - pass)}</span><span className="wg-note">of {spec.attacks ?? 1000} attacks, about {Math.round((1 - pass) * (spec.attacks ?? 1000))} stopped</span></div>
      </div>
      <p className="wg-note">{spec.note || 'The product assumes the layers miss independently of each other. Real layers often miss the same cleverly worded attacks, so the true number is higher; the catch rates here are made up for the demonstration.'}</p>
    </div>
  );
}

// ---------------------------------------------------------------- fgsm (a tiny signed-gradient attack on a linear image classifier)
function Fgsm({ spec, fig }) {
  const [eps, setEps] = useState(spec.eps ?? 0.1);
  const n = spec.w.length;
  const x = spec.image.map((row) => row.split('').map((d) => Number(d) / 9));
  const R = useMemo(() => fgsmAttack({ w: spec.w, b: spec.b, x, eps }), [spec, eps]);
  const cell = 20;
  const Grid = ({ vals, title, mode }) => (
    <div>
      <p className="wg-k" style={{ textTransform: 'none', margin: '0 0 4px' }}>{title}</p>
      <svg viewBox={`0 0 ${n * cell} ${n * cell}`} width={n * cell} height={n * cell} role="img" aria-label={title}>
        {vals.map((v, i) => {
          const r = Math.floor(i / n), q = i % n;
          const fill = mode === 'sign' ? (v > 0 ? 'var(--accent)' : v < 0 ? 'var(--amber)' : 'var(--surface2)') : `rgb(${Math.round(255 * (1 - v))},${Math.round(255 * (1 - v))},${Math.round(255 * (1 - v))})`;
          return <rect key={i} x={q * cell} y={r * cell} width={cell - 1} height={cell - 1} fill={fill} fillOpacity={mode === 'sign' ? 0.75 : 1} stroke="var(--divider)" strokeWidth="0.5" />;
        })}
      </svg>
    </div>
  );
  const sgn = spec.w.flat().map((v) => Math.sign(v));
  const p0 = sigmoid(R.s0), p1 = sigmoid(R.s1);
  const names = spec.labels || ['class A', 'class B'];
  const cls = (sc) => (sc >= 0 ? names[0] : names[1]);
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <Slider label="Attack strength ε (how far each pixel may move, 0 to 1)" min={0} max={0.6} step={0.01} value={eps} onChange={setEps} show={(q) => q.toFixed(2)} />
      <div className="wg-row" style={{ gap: 18, alignItems: 'flex-start', flexWrap: 'wrap' }}>
        <Grid vals={x.flat()} title="Original picture x" mode="img" />
        <Grid vals={sgn} title="sign of the weights (the direction that matters)" mode="sign" />
        <Grid vals={R.xAdv} title="Attacked picture x′" mode="img" />
      </div>
      <div className="vz-cmp" style={{ marginTop: 10, gridTemplateColumns: 'repeat(3,minmax(0,1fr))' }}>
        <div><span className="wg-k">Score before: s = w·x + b</span><span className="vz-big">{f2(R.s0, 2)}</span><span className="wg-note">{cls(R.s0)}, probability {f2(Math.max(p0, 1 - p0), 2)}</span></div>
        <div className={R.flipped ? 'out' : ''}><span className="wg-k">Score after the attack</span><span className="vz-big">{f2(R.s1, 2)}</span><span className="wg-note">{cls(R.s1)}, probability {f2(Math.max(p1, 1 - p1), 2)}{R.flipped ? ' → the label flipped' : ''}</span></div>
        <div><span className="wg-k">Largest possible change</span><span className="vz-big">{f2(eps * R.l1, 2)}</span><span className="wg-note">ε × Σ|w| = {f2(eps, 2)} × {f2(R.l1, 2)}</span></div>
      </div>
      <p className="wg-note">{spec.note || 'A made-up linear classifier on an 8 × 8 picture. Moving every pixel by ε in the direction of sign(w) changes the score by up to ε × Σ|w|; many tiny changes add up. The weights and the picture are illustrative.'}</p>
    </div>
  );
}

// ---------------------------------------------------------------- railthreshold (a detector's threshold trades missed attacks against false alarms)
function RailThreshold({ spec, fig }) {
  const [t, setT] = useState(spec.threshold ?? 0.5);
  const A = spec.attack, L = spec.legit;
  const R = thresholdRates({ muA: A.mean, sdA: A.sd, muL: L.mean, sdL: L.sd, t, prevalence: spec.prevalence ?? 0.1 });
  const W = 560, H = 170, x0 = 20, x1 = W - 20;
  const lo = Math.min(A.mean - 3.5 * A.sd, L.mean - 3.5 * L.sd), hi = Math.max(A.mean + 3.5 * A.sd, L.mean + 3.5 * L.sd);
  const X = (v) => x0 + ((v - lo) / (hi - lo)) * (x1 - x0);
  const dens = (v, m, sd) => Math.exp(-0.5 * ((v - m) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI));
  const ymax = Math.max(dens(A.mean, A.mean, A.sd), dens(L.mean, L.mean, L.sd));
  const Y = (d) => H - 24 - (d / ymax) * (H - 44);
  const pts = (m, sd) => Array.from({ length: 121 }, (_, i) => { const v = lo + ((hi - lo) * i) / 120; return [X(v), Y(dens(v, m, sd))]; });
  const line = (m, sd) => pts(m, sd).map(([a, b], i) => `${i ? 'L' : 'M'}${a.toFixed(1)} ${b.toFixed(1)}`).join(' ');
  const tail = (m, sd) => { const p = pts(m, sd).filter(([a]) => a >= X(t)); return p.length ? `M${X(t)} ${H - 24} ` + p.map(([a, b]) => `L${a.toFixed(1)} ${b.toFixed(1)}`).join(' ') + ` L${p[p.length - 1][0]} ${H - 24} Z` : ''; };
  const pc = (x) => (x * 100).toFixed(1) + ' %';
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <Slider label="Threshold: block a message when its risk score is at least" min={lo} max={hi} step={(hi - lo) / 100} value={t} onChange={setT} show={(q) => q.toFixed(2)} />
      <div className="vz-hm-wrap">
        <svg viewBox={`0 0 ${W} ${H}`} className="vz-att" role="img" aria-label="Two bell curves of risk scores, legitimate messages and attacks, with a threshold line">
          <path d={tail(L.mean, L.sd)} fill="var(--amber)" fillOpacity="0.35" />
          <path d={tail(A.mean, A.sd)} fill="var(--accent)" fillOpacity="0.30" />
          <path d={line(L.mean, L.sd)} fill="none" stroke="var(--amber)" strokeWidth="2" />
          <path d={line(A.mean, A.sd)} fill="none" stroke="var(--accent)" strokeWidth="2" />
          <line x1={X(t)} x2={X(t)} y1="8" y2={H - 24} stroke="var(--text)" strokeWidth="1.5" strokeDasharray="5 4" />
          <text x={x0} y={H - 6} className="vz-rl">{L.label || 'legitimate'} (amber)</text>
          <text x={x1} y={H - 6} textAnchor="end" className="vz-rl">{A.label || 'attacks'} (green)</text>
        </svg>
      </div>
      <div className="vz-cmp" style={{ gridTemplateColumns: 'repeat(4,minmax(0,1fr))' }}>
        <div className="out"><span className="wg-k">Attacks caught (recall)</span><span className="vz-big">{pc(R.tpr)}</span><span className="wg-note">green area right of the line</span></div>
        <div><span className="wg-k">Attacks missed</span><span className="vz-big">{pc(R.fnr)}</span><span className="wg-note">1 − recall</span></div>
        <div><span className="wg-k">Legitimate blocked (false alarms)</span><span className="vz-big">{pc(R.fpr)}</span><span className="wg-note">amber area right of the line</span></div>
        <div><span className="wg-k">Blocked messages that were attacks</span><span className="vz-big">{pc(R.precision)}</span><span className="wg-note">when {Math.round((spec.prevalence ?? 0.1) * 100)} % of messages are attacks</span></div>
      </div>
      <p className="wg-note">{spec.note || 'The two bell curves are made up. Moving the threshold right lets fewer legitimate messages through the filter wrongly but misses more attacks; no setting removes both errors when the curves overlap.'}</p>
    </div>
  );
}

// ---------------------------------------------------------------- agenttrace (the loop of an agent, step by step, with the context growing)
function AgentTrace({ spec, fig }) {
  const [i, setI] = useState(0);
  const steps = spec.steps;
  const W = spec.window ?? 4096;
  const C = traceCost(spec.C0, steps, i);
  const shown = steps.slice(0, i);
  const done = i >= steps.length;
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <p className="wg-p"><strong>Goal:</strong> {spec.goal}</p>
      {spec.tools ? <p className="wg-note" style={{ margin: '0 0 8px' }}><strong>Tools:</strong> {spec.tools.join(' · ')}</p> : null}
      <div className="wg-row" style={{ gap: 8, marginBottom: 8 }}>
        <button type="button" className="wg-chip" disabled={i === 0} onClick={() => setI(i - 1)}>← Back</button>
        <button type="button" className="wg-chip on" disabled={done} onClick={() => setI(i + 1)}>{done ? 'Finished' : i === 0 ? 'Start the loop' : 'Next loop →'}</button>
        <button type="button" className="wg-chip" onClick={() => setI(0)}>Reset</button>
        <span className="wg-note" style={{ margin: 0 }}>model call {Math.min(i, steps.length)} of {steps.length}</span>
      </div>
      <ol className="vz-rank">
        {shown.map((st, k) => (
          <li key={k} className={k === shown.length - 1 ? 'top' : ''}>
            <div className="vz-rank-h"><strong>{st.answer ? 'Final answer' : `Loop ${k + 1}`}</strong><span>{(st.outTokens || 0) + (st.obsTokens || 0)} tokens added</span></div>
            {st.thought ? <p className="wg-note" style={{ margin: '4px 0 0' }}><strong>Thought:</strong> {st.thought}</p> : null}
            {st.action ? <p className="wg-note" style={{ margin: '2px 0 0' }}><strong>Action:</strong> <code>{st.action}</code></p> : null}
            {st.observation ? <p className="wg-note" style={{ margin: '2px 0 0' }}><strong>Observation:</strong> {st.observation}</p> : null}
            {st.answer ? <p className="wg-note" style={{ margin: '4px 0 0' }}>{st.answer}</p> : null}
          </li>
        ))}
      </ol>
      <div className="vz-cmp" style={{ gridTemplateColumns: 'repeat(3,minmax(0,1fr))' }}>
        <div><span className="wg-k">Context the next call reads</span><span className="vz-big">{C.context.toLocaleString('en-US')}</span><span className="wg-note">{spec.C0} (prompt) + the loops so far</span></div>
        <div className="out"><span className="wg-k">Tokens read in total</span><span className="vz-big">{C.read.toLocaleString('en-US')}</span><span className="wg-note">the sum of the context at every call</span></div>
        <div><span className="wg-k">Share of the window used</span><span className="vz-big">{Math.round((C.context / W) * 100)} %</span><span className="wg-note">window {W.toLocaleString('en-US')} tokens</span></div>
      </div>
      <p className="wg-note">{spec.note || 'A scripted example; the token counts are made up but add up exactly as shown. Each model call re-reads everything written so far.'}</p>
    </div>
  );
}

// ---------------------------------------------------------------- klreward (reward versus staying close to the original model)
function KlReward({ spec, fig }) {
  const ref = spec.answers.map((a) => a.ref);
  const r = spec.answers.map((a) => a.reward);
  const [beta, setBeta] = useState(spec.beta ?? 1);
  const [z, setZ] = useState(() => ref.map((q) => Math.log(q)));
  const pi = softmaxArr(z);
  const cur = klObjective({ pi, ref, r, beta });
  const opt = klOptimum({ ref, r, beta });
  const optV = klObjective({ pi: opt, ref, r, beta });
  const f = (q) => Number(q.toFixed(3)).toString();
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <Slider label="KL penalty β (how strongly to stay close to the original model)" min={0.1} max={10} step={0.1} value={beta} onChange={setBeta} show={(q) => q.toFixed(1)} />
      <div className="vz-scroll">
        <table className="vz-mlp" aria-label="Answers with reward, reference probability and policy probability">
          <thead><tr><th>Answer</th><th>Reward r</th><th>Original model π_ref</th><th>Your policy π</th><th>Best policy π* for this β</th></tr></thead>
          <tbody>
            {spec.answers.map((a, i) => (
              <tr key={i}>
                <td>{a.text}</td><td>{a.reward}</td><td>{f(a.ref)}</td>
                <td><input type="range" min={-6} max={6} step={0.1} value={z[i]} aria-label={`Logit of answer ${i + 1}`} onChange={(e) => setZ((zz) => zz.map((q, k) => (k === i ? +e.target.value : q)))} style={{ width: 90, verticalAlign: 'middle' }} /> <strong>{f(pi[i])}</strong></td>
                <td>{f(opt[i])}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="vz-cmp" style={{ marginTop: 10, gridTemplateColumns: 'repeat(4,minmax(0,1fr))' }}>
        <div><span className="wg-k">Expected reward E[r]</span><span className="vz-big">{f(cur.expR)}</span></div>
        <div><span className="wg-k">KL(π ‖ π_ref)</span><span className="vz-big">{f(cur.kl)}</span></div>
        <div className="out"><span className="wg-k">Objective J = E[r] − β·KL</span><span className="vz-big">{f(cur.J)}</span></div>
        <div><span className="wg-k">Best possible J (at π*)</span><span className="vz-big">{f(optV.J)}</span></div>
      </div>
      <div className="wg-row" style={{ marginTop: 8 }}>
        <button type="button" className="wg-btn" onClick={() => setZ(opt.map((q) => Math.log(Math.max(q, 1e-9))))}>Jump to the best policy π*</button>
        <button type="button" className="wg-btn ghost" onClick={() => setZ(ref.map((q) => Math.log(q)))}>Back to the original model</button>
      </div>
      <p className="wg-note">Small β: the best policy puts almost everything on the highest-reward answer, even if that reward is an imperfect score (“reward hacking”). Large β: it barely moves from the original model. π* ∝ π_ref · exp(r/β). {spec.note || 'The answers and rewards are made up for the demonstration.'}</p>
    </div>
  );
}

// ---------------------------------------------------------------- lowrank (why a low-rank update can be enough)
function LowRank({ spec, fig }) {
  const A = spec.matrix;
  const n = A.length, m = A[0].length;
  const svd = useMemo(() => svdSmall(A), [A]);
  const [r, setR] = useState(spec.rank ?? 2);
  const approx = useMemo(() => lowRankApprox(A, r), [A, r]);
  const err = Math.sqrt(A.reduce((s, row, i) => s + row.reduce((t, x, j) => t + (x - approx[i][j]) ** 2, 0), 0));
  const tot = Math.sqrt(A.reduce((s, row) => s + row.reduce((t, x) => t + x * x, 0), 0));
  const maxS = svd.S[0] || 1;
  const tone = (v) => (v >= 0 ? `color-mix(in srgb, var(--accent) ${Math.round(Math.min(1, Math.abs(v) / (spec.max || 3)) * 75)}%, var(--surface2))` : `color-mix(in srgb, var(--cyan) ${Math.round(Math.min(1, Math.abs(v) / (spec.max || 3)) * 75)}%, var(--surface2))`);
  const round = (M) => M.map((row) => row.map((x) => Math.round(x * 10) / 10));
  const f = (q) => Number(q.toFixed(3)).toString();
  return (
    <div className="wg">
      <Title fig={fig}>{spec.title}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <Slider label="Rank r of the approximation" min={0} max={Math.min(n, m)} step={1} value={r} onChange={setR} />
      <div className="vz-conv" style={{ alignItems: 'flex-start', marginTop: 6 }}>
        <div><span className="wg-k" style={{ textTransform: 'none' }}>The full update ΔW ({n}×{m} = {n * m} numbers)</span><Grid data={round(A)} tone={tone} /></div>
        <span className="vz-op">≈</span>
        <div><span className="wg-k" style={{ textTransform: 'none' }}>Rank-{r} approximation B·A (uses {r * (n + m)} numbers)</span><Grid data={round(approx)} tone={tone} cls="out" /></div>
      </div>
      <ul className="wg-bars" style={{ marginTop: 10 }}>
        {svd.S.map((s, i) => <li key={i} style={{ gridTemplateColumns: '120px minmax(0,1fr) 60px' }}><span className="wg-bar-l" style={{ fontFamily: 'inherit' }}>singular value {i + 1}</span><span className="wg-bar-track"><span className={'wg-bar-fill' + (i >= r ? ' warn' : '')} style={{ width: `${(s / maxS) * 100}%` }} /></span><span className="wg-bar-p">{f(s)}</span></li>)}
      </ul>
      <div className="vz-cmp" style={{ marginTop: 10, gridTemplateColumns: 'repeat(3,minmax(0,1fr))' }}>
        <div><span className="wg-k">Numbers to store: full vs rank {r}</span><span className="vz-big">{n * m} vs {r * (n + m)}</span></div>
        <div className="out"><span className="wg-k">Error ‖ΔW − BA‖ ÷ ‖ΔW‖</span><span className="vz-big">{tot ? f(err / tot) : '0'}</span><span className="wg-note">0 = exact</span></div>
        <div><span className="wg-k">Energy kept (Σσ² of the first {r})</span><span className="vz-big">{f(svd.S.slice(0, r).reduce((s, q) => s + q * q, 0) / svd.S.reduce((s, q) => s + q * q, 0))}</span></div>
      </div>
      <p className="wg-note">Green bars are the singular values kept by the approximation, amber ones are dropped. When a few singular values dominate, a thin product B·A (an n×r matrix times an r×m matrix) is almost as good as the whole matrix: that is the bet LoRA makes about fine-tuning updates. {spec.note || 'The matrix is written for teaching.'}</p>
    </div>
  );
}

export const VIZ = { railstack: RailStack, fgsm: Fgsm, railthreshold: RailThreshold, agenttrace: AgentTrace, chunker: Chunker, rankmetrics: RankMetrics, prefpair: PrefPair, klreward: KlReward, lowrank: LowRank, schedule: Schedule, sampler1d: Sampler1d, unet: Unet, ganlab: GanLab, upsample: Upsample, clipsim: ClipSim, patchshuffle: PatchShuffle, fusion: Fusion, llmfamily: LlmFamily, moe: Moe, scalinglaw: ScalingLaw, emergent: Emergent, tfblock: TfBlock, causalmask: CausalMask, shiftloss: ShiftLoss, seqmemory: SeqMemory, lstmcell: LstmCell, hardsoft: HardSoft, vocabtrade: VocabTrade, vecsim: VecSim, convgrad: ConvGrad, filters: Filters, pool: Pool, cnnstack: CnnStack, backprop: Backprop, gradflow: GradFlow, fold: Fold, network: Network, fit: Fit, trainer: Trainer, mlp: Mlp, nested: Nested, sampling: Sampling, chain: Chain, logbars: LogBars, attention: Attention, equation: Equation, pipeline: Pipeline, calc: Calc, curve: Curve, descent: Descent, neuron: Neuron, conv: Conv, bpe: BPE, ngram: NGram, scatter: Scatter, heatmap: Heatmap, posenc: PosEnc, patch: Patch, diffusion: Diffusion, retrieval: Retrieval };
