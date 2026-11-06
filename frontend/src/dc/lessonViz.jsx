import React, { useEffect, useMemo, useRef, useState } from 'react';
import { softmaxT, filterSampling, chainStats, mulberry32, gaussNoise, BYTES, convParams, loraParams, chinchilla, FN, LOSS, neuronAccuracy, convolve, transposeConv, learnBPE, applyBPE, bigrams, cosine, nearest, posEnc, alphaBar, tokens, tfidfRank, attend } from './lessonMath.js';
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
  const [a, setA] = useState(spec.kind === 'fc_vs_conv' ? 28 : spec.kind === 'lora' ? 4096 : spec.kind === 'memory' ? 7 : 24);
  const [b, setB] = useState(spec.kind === 'fc_vs_conv' ? 100 : spec.kind === 'lora' ? 8 : 0);
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
          {pts.map((p) => (
            <g key={p.label} onClick={() => { setAn(-1); setSel(p.label); }} style={{ cursor: 'pointer' }}>
              <circle cx={sx(p.x)} cy={sy(p.y)} r={p.label === sel && an < 0 ? 9 : 6} fill={GROUP_COLORS[groups.indexOf(p.group || '') % 5]} stroke={p.label === sel && an < 0 ? 'var(--text)' : 'none'} strokeWidth="2" />
              <text x={sx(p.x) + (sx(p.x) > S * 0.72 ? -10 : 10)} y={sy(p.y) + 4} textAnchor={sx(p.x) > S * 0.72 ? 'end' : 'start'} className="vz-lab">{p.label}</text>
            </g>
          ))}
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

export const VIZ = { nested: Nested, sampling: Sampling, chain: Chain, logbars: LogBars, attention: Attention, equation: Equation, pipeline: Pipeline, calc: Calc, curve: Curve, descent: Descent, neuron: Neuron, conv: Conv, bpe: BPE, ngram: NGram, scatter: Scatter, heatmap: Heatmap, posenc: PosEnc, patch: Patch, diffusion: Diffusion, retrieval: Retrieval };
