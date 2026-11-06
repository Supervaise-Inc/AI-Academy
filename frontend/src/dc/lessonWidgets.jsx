import React, { useEffect, useMemo, useRef, useState } from 'react';
import * as session from './session.js';
import { VIZ, gaussNoise, convolve } from './lessonViz.jsx';

// Interactive blocks a lesson's Markdown can embed:
//
//   ```viz            { "type": "sorter" | "flow" | "timeline" | "temperature", ... }
//   ```quiz           { "id": "week-01-check", "passMark": 70, "questions": [ ... ] }
//
// Their data lives in the lesson file itself, so a lesson is edited in one place.

export function Widget({ kind, spec, error, fig }) {
  if (error) {
    return (
      <div className="wg wg-err" role="alert">
        <strong>This interactive block could not be shown.</strong> {error}
      </div>
    );
  }
  if (kind === 'quiz') return spec.exam ? <ExamQuiz spec={spec} /> : <Quiz spec={spec} />;
  spec = { ...spec, fig };
  switch (spec.type) {
    case 'sorter':
      return <Sorter spec={spec} />;
    case 'flow':
      return <Flow spec={spec} />;
    case 'timeline':
      return <Timeline spec={spec} />;
    case 'temperature':
      return <Temperature spec={spec} />;
    default: {
      const V = VIZ[spec.type];
      if (V) return <V spec={spec} fig={fig} />;
      return <div className="wg wg-err" role="alert">Unknown visual type “{String(spec.type)}”.</div>;
    }
  }
}

const Title = ({ children, fig }) => (
  <div className="wg-head">
    {fig ? <span className="wg-fig">Figure {fig}</span> : null}
    <div className="wg-t">{children}</div>
  </div>
);

// ---------- sorter: discriminative or generative? ----------
function Sorter({ spec }) {
  const [picked, setPicked] = useState({});
  const bins = spec.bins || [
    { id: 'generative', label: 'Generates new data' },
    { id: 'discriminative', label: 'Classifies or predicts' },
  ];
  const correct = spec.cards.filter((c, i) => picked[i] && (c.answer === 'either' || picked[i] === c.answer)).length;
  const done = Object.keys(picked).length;
  return (
    <div className="wg">
      <Title fig={spec.fig}>{spec.title || 'Sort the examples'}</Title>
      {spec.intro ? <p className="wg-p">{spec.intro}</p> : null}
      <div className="wg-cards">
        {spec.cards.map((c, i) => {
          const p = picked[i];
          const ok = p && (c.answer === 'either' || p === c.answer);
          return (
            <div key={c.name} className={'wg-card' + (p ? (ok ? ' ok' : ' bad') : '')}>
              <div className="wg-card-n">{c.name}</div>
              <div className="wg-row">
                {bins.map((b) => (
                  <button key={b.id} type="button" className={'wg-btn' + (p === b.id ? ' on' : '')} aria-pressed={p === b.id} disabled={!!p} onClick={() => setPicked((prev) => ({ ...prev, [i]: b.id }))}>
                    {b.label}
                  </button>
                ))}
              </div>
              {p ? (
                <p className="wg-why" aria-live="polite">
                  <strong>{ok ? (c.answer === 'either' ? 'Either answer works. ' : 'Correct. ') : 'Not quite. '}</strong>
                  {c.why}
                </p>
              ) : null}
            </div>
          );
        })}
      </div>
      <div className="wg-foot">
        <span aria-live="polite">{done === spec.cards.length ? `Done: ${correct} of ${spec.cards.length} correct.` : `${done} of ${spec.cards.length} sorted`}</span>
        <button type="button" className="wg-btn ghost" onClick={() => setPicked({})} disabled={!done}>
          Start over
        </button>
      </div>
    </div>
  );
}

// ---------- flow: input -> model -> output, for two kinds of model ----------
function Flow({ spec }) {
  const [m, setM] = useState(0);
  const [e, setE] = useState(0);
  const mode = spec.modes[m];
  const ex = mode.examples[Math.min(e, mode.examples.length - 1)];
  return (
    <div className="wg">
      <Title fig={spec.fig}>{spec.title || 'How the model is used'}</Title>
      <div className="wg-tabs" role="tablist">
        {spec.modes.map((x, i) => (
          <button key={x.name} type="button" role="tab" aria-selected={m === i} className={'wg-tab' + (m === i ? ' on' : '')} onClick={() => { setM(i); setE(0); }}>
            {x.name}
          </button>
        ))}
      </div>
      <p className="wg-p">{mode.tagline}</p>
      <div className="wg-flow">
        <div className="wg-box">
          <span className="wg-k">{mode.inputLabel || 'Input'}</span>
          <span className="wg-v">{ex.input}</span>
        </div>
        <svg className="wg-arrow" viewBox="0 0 60 24" aria-hidden="true">
          <path d="M2 12h50M44 4l10 8-10 8" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <div className="wg-box model">
          <span className="wg-k">Model</span>
          <span className="wg-v">{mode.modelLabel || 'learned from data'}</span>
        </div>
        <svg className="wg-arrow" viewBox="0 0 60 24" aria-hidden="true">
          <path d="M2 12h50M44 4l10 8-10 8" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <div className="wg-box out" aria-live="polite">
          <span className="wg-k">{mode.outputLabel || 'Output'}</span>
          <span className="wg-v">{ex.output}</span>
        </div>
      </div>
      <div className="wg-row" style={{ marginTop: 14 }}>
        {mode.examples.map((x, i) => (
          <button key={x.input} type="button" className={'wg-chip' + (e === i ? ' on' : '')} onClick={() => setE(i)}>
            {x.label || `Example ${i + 1}`}
          </button>
        ))}
      </div>
      {mode.note ? <p className="wg-note">{mode.note}</p> : null}
    </div>
  );
}

// ---------- timeline ----------
function Timeline({ spec }) {
  const [i, setI] = useState(0);
  const ev = spec.events[i];
  return (
    <div className="wg">
      <Title fig={spec.fig}>{spec.title || 'Timeline'}</Title>
      <ol className="wg-tl" aria-label="Events">
        {spec.events.map((x, k) => (
          <li key={x.year + x.title}>
            <button type="button" className={'wg-node' + (k === i ? ' on' : '') + (k < i ? ' past' : '')} aria-current={k === i} onClick={() => setI(k)}>
              <span className="wg-yr">{x.year}</span>
              <span className="wg-nt">{x.title}</span>
            </button>
          </li>
        ))}
      </ol>
      <div className="wg-detail" aria-live="polite">
        <div className="wg-detail-y">{ev.year}</div>
        <div>
          <div className="wg-detail-t">{ev.title}</div>
          <p className="wg-p" style={{ margin: 0 }}>{ev.text}</p>
        </div>
      </div>
      <div className="wg-foot">
        <button type="button" className="wg-btn ghost" onClick={() => setI(Math.max(0, i - 1))} disabled={i === 0}>
          Earlier
        </button>
        <button type="button" className="wg-btn ghost" onClick={() => setI(Math.min(spec.events.length - 1, i + 1))} disabled={i === spec.events.length - 1}>
          Later
        </button>
      </div>
    </div>
  );
}

// ---------- temperature: next-word probabilities ----------
function softmax(logits, t) {
  const m = Math.max(...logits);
  const ex = logits.map((l) => Math.exp((l - m) / t));
  const s = ex.reduce((a, b) => a + b, 0);
  return ex.map((x) => x / s);
}
function draw(p) {
  let r = Math.random();
  for (let i = 0; i < p.length; i++) {
    r -= p[i];
    if (r <= 0) return i;
  }
  return p.length - 1;
}

function Temperature({ spec }) {
  const [t, setT] = useState(1);
  const [text, setText] = useState([]);
  const [tally, setTally] = useState(null);
  const probs = useMemo(() => softmax(spec.tokens.map((x) => x.logit), t), [spec, t]);
  const mood = t < 0.5 ? 'Very focused: almost always the top word.' : t <= 1.05 ? 'Balanced.' : t < 1.6 ? 'More varied: unlikely words get a real chance.' : 'Very adventurous: expect odd choices.';
  const sample = () => setText((prev) => [...prev, spec.tokens[draw(probs)].t]);
  const sample50 = () => {
    const c = spec.tokens.map(() => 0);
    for (let i = 0; i < 50; i++) c[draw(probs)]++;
    setTally(c);
  };
  const setTemp = (v) => {
    setT(v);
    setTally(null);
  };
  return (
    <div className="wg">
      <Title fig={spec.fig}>{spec.title || 'Temperature'}</Title>
      <p className="wg-p">
        The model has read: <strong>“{spec.prompt}…”</strong> and scores every possible next word.
      </p>
      <label className="wg-slide">
        <span>
          Temperature <strong>{t.toFixed(1)}</strong>
        </span>
        <input type="range" min="0.2" max="2" step="0.1" value={t} onChange={(e) => setTemp(Number(e.target.value))} aria-valuetext={`${t.toFixed(1)}: ${mood}`} />
      </label>
      <p className="wg-note" style={{ marginTop: 0 }}>{mood}</p>
      <ul className="wg-bars">
        {spec.tokens.map((x, i) => (
          <li key={x.t}>
            <span className="wg-bar-l">{x.t}</span>
            <span className="wg-bar-track">
              <span className="wg-bar-fill" style={{ width: `${(probs[i] * 100).toFixed(1)}%` }} />
            </span>
            <span className="wg-bar-p">{(probs[i] * 100).toFixed(1)}%</span>
            {tally ? <span className="wg-bar-c" title="Times chosen in 50 draws">{tally[i]}×</span> : null}
          </li>
        ))}
      </ul>
      <div className="wg-row">
        <button type="button" className="wg-btn" onClick={sample}>
          Pick the next word
        </button>
        <button type="button" className="wg-btn ghost" onClick={sample50}>
          Draw 50 times
        </button>
        <button type="button" className="wg-btn ghost" onClick={() => { setText([]); setTally(null); }} disabled={!text.length && !tally}>
          Clear
        </button>
      </div>
      <p className="wg-sent" aria-live="polite">
        {spec.prompt}
        {text.map((w, i) => (
          <span key={i} className="wg-picked"> {w}</span>
        ))}
        …
      </p>
      {spec.note ? <p className="wg-note">{spec.note}</p> : null}
    </div>
  );
}

// ---------- quiz ----------
function Quiz({ spec }) {
  const qs = spec.questions;
  const pass = spec.passMark ?? 70;
  const key = `quiz-${spec.id}`;
  const [i, setI] = useState(0);
  const [sel, setSel] = useState(null);
  const [checked, setChecked] = useState(false);
  const [answers, setAnswers] = useState([]);
  const [finished, setFinished] = useState(false);
  const [best, setBest] = useState(null);

  useEffect(() => {
    let alive = true;
    session
      .loadPageState(key)
      .then((d) => alive && d && setBest(d))
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [key]);

  const q = qs[i];
  const check = () => {
    setAnswers([...answers, sel]);
    setChecked(true);
  };
  const next = () => {
    if (i + 1 < qs.length) {
      setI(i + 1);
      setSel(null);
      setChecked(false);
    } else {
      const score = answers.filter((a, k) => a === qs[k].answer).length;
      const prev = best || { best: 0, attempts: 0 };
      const rec = { best: Math.max(prev.best, score), attempts: prev.attempts + 1, total: qs.length };
      setBest(rec);
      session.savePageState(key, rec).catch(() => {});
      setFinished(true);
    }
  };
  const restart = () => {
    setI(0);
    setSel(null);
    setChecked(false);
    setAnswers([]);
    setFinished(false);
  };

  if (finished) {
    const score = answers.filter((a, k) => a === qs[k].answer).length;
    const pct = Math.round((score / qs.length) * 100);
    const passed = pct >= pass;
    return (
      <div className="wg">
        <Title fig={spec.fig}>{spec.title || 'Check your understanding'}</Title>
        <div className={'wg-score' + (passed ? ' ok' : '')} aria-live="polite">
          <div className="wg-score-n">
            {score}/{qs.length}
          </div>
          <div>
            <strong>{passed ? 'Passed' : 'Not yet'}</strong> · {pct}% (pass mark {pass}%)
            <div className="wg-note" style={{ margin: '2px 0 0' }}>
              {best ? `Your best: ${best.best}/${best.total || qs.length} in ${best.attempts} attempt${best.attempts === 1 ? '' : 's'}.` : ''}
            </div>
          </div>
        </div>
        <ol className="wg-review">
          {qs.map((x, k) => {
            const ok = answers[k] === x.answer;
            return (
              <li key={k} className={ok ? 'ok' : 'bad'}>
                <div className="wg-rq">{x.q}</div>
                <div className="wg-ra">
                  {ok ? 'Correct: ' : `You chose “${x.options[answers[k]]}”. Answer: `}
                  <strong>{x.options[x.answer]}</strong>
                </div>
                {!ok ? <div className="wg-note" style={{ margin: '4px 0 0' }}>{x.why}</div> : null}
              </li>
            );
          })}
        </ol>
        <div className="wg-foot">
          <span />
          <button type="button" className="wg-btn" onClick={restart}>
            Try again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="wg">
      <Title fig={spec.fig}>{spec.title || 'Check your understanding'}</Title>
      <div className="wg-prog" aria-label={`Question ${i + 1} of ${qs.length}`}>
        <span className="wg-prog-fill" style={{ width: `${((i + (checked ? 1 : 0)) / qs.length) * 100}%` }} />
      </div>
      <div className="wg-note" style={{ margin: '6px 0 10px' }}>
        Question {i + 1} of {qs.length}
        {best ? ` · best so far ${best.best}/${best.total || qs.length}` : ''}
      </div>
      <fieldset className="wg-q">
        <legend>{q.q}</legend>
        {q.options.map((o, k) => {
          const state = !checked ? (sel === k ? ' sel' : '') : k === q.answer ? ' right' : sel === k ? ' wrong' : '';
          return (
            <label key={k} className={'wg-opt' + state}>
              <input type="radio" name={`${key}-${i}`} checked={sel === k} disabled={checked} onChange={() => setSel(k)} />
              <span>{o}</span>
            </label>
          );
        })}
      </fieldset>
      {checked ? (
        <p className={'wg-why ' + (sel === q.answer ? 'ok' : 'bad')} aria-live="polite">
          <strong>{sel === q.answer ? 'Correct. ' : 'Not quite. '}</strong>
          {q.why}
        </p>
      ) : null}
      <div className="wg-foot">
        <span />
        {!checked ? (
          <button type="button" className="wg-btn" onClick={check} disabled={sel === null}>
            Check answer
          </button>
        ) : (
          <button type="button" className="wg-btn" onClick={next}>
            {i + 1 < qs.length ? 'Next question' : 'See my result'}
          </button>
        )}
      </div>
    </div>
  );
}

// ---------- exam mode quiz: timed, no feedback until submit, free navigation, topic breakdown ----------
function ExamQuiz({ spec }) {
  const qs = spec.questions;
  const pass = spec.passMark ?? 70;
  const key = `quiz-${spec.id}`;
  const total = (spec.minutes || 60) * 60;
  const [phase, setPhase] = useState('intro');
  const [i, setI] = useState(0);
  const [ans, setAns] = useState(() => qs.map(() => null));
  const [flag, setFlag] = useState(() => qs.map(() => false));
  const [left, setLeft] = useState(total);
  const [best, setBest] = useState(null);
  const [confirm, setConfirm] = useState(false);
  const ansRef = useRef(ans);
  ansRef.current = ans;

  useEffect(() => {
    let alive = true;
    session.loadPageState(key).then((d) => alive && d && setBest(d)).catch(() => {});
    return () => { alive = false; };
  }, [key]);

  const finish = (a) => {
    const score = a.filter((x, k) => x === qs[k].answer).length;
    const prev = best || { best: 0, attempts: 0 };
    const rec = { best: Math.max(prev.best, score), attempts: prev.attempts + 1, total: qs.length };
    setBest(rec);
    session.savePageState(key, rec).catch(() => {});
    setPhase('done');
  };
  useEffect(() => {
    if (phase !== 'run') return undefined;
    const id = setInterval(() => setLeft((t) => t - 1), 1000);
    return () => clearInterval(id);
  }, [phase]);
  useEffect(() => { if (phase === 'run' && left <= 0) finish(ansRef.current); }, [left, phase]); // eslint-disable-line react-hooks/exhaustive-deps

  const mm = (t) => `${Math.floor(Math.max(0, t) / 60)}:${String(Math.max(0, t) % 60).padStart(2, '0')}`;
  const start = () => { setAns(qs.map(() => null)); setFlag(qs.map(() => false)); setI(0); setLeft(total); setConfirm(false); setPhase('run'); };

  if (phase === 'intro') {
    return (
      <div className="wg">
        <Title fig={spec.fig}>{spec.title || 'Written exam'}</Title>
        <ul className="wg-rules">
          <li><strong>{qs.length}</strong> multiple-choice questions, <strong>{spec.minutes || 60} minutes</strong>. The clock starts when you press Start and the exam is submitted for you when it reaches zero.</li>
          <li>You can move back and forth, change answers and flag questions to come back to. Nothing is marked until you submit.</li>
          <li>Closed book: no notes, no other tabs, no AI assistants. A calculator or paper is fine.</li>
          {(spec.rules || []).map((r, k) => <li key={k}>{r}</li>)}
          <li>When you submit you see your score, a breakdown by topic and the explanation for every question you missed.</li>
        </ul>
        {best ? <p className="wg-note">Your best so far: {best.best}/{best.total || qs.length} in {best.attempts} attempt{best.attempts === 1 ? '' : 's'}.</p> : null}
        <div className="wg-foot"><span /><button type="button" className="wg-btn" onClick={start}>Start the exam</button></div>
      </div>
    );
  }

  if (phase === 'done') {
    const score = ans.filter((a, k) => a === qs[k].answer).length;
    const pct = Math.round((score / qs.length) * 100);
    const passed = pct >= pass;
    const topics = {};
    qs.forEach((x, k) => { const t = x.topic || 'General'; topics[t] = topics[t] || { n: 0, ok: 0 }; topics[t].n++; if (ans[k] === x.answer) topics[t].ok++; });
    const unanswered = ans.filter((a) => a === null).length;
    return (
      <div className="wg">
        <Title fig={spec.fig}>{spec.title || 'Written exam'}</Title>
        <div className={'wg-score' + (passed ? ' ok' : '')} aria-live="polite">
          <div className="wg-score-n">{score}/{qs.length}</div>
          <div>
            <strong>{passed ? 'Passed' : 'Not yet'}</strong> · {pct}% (pass mark {pass}%)
            <div className="wg-note" style={{ margin: '2px 0 0' }}>{unanswered ? `${unanswered} unanswered. ` : ''}Time used {mm(total - Math.max(0, left))} of {spec.minutes || 60}:00.{best ? ` Your best: ${best.best}/${best.total || qs.length} in ${best.attempts} attempt${best.attempts === 1 ? '' : 's'}.` : ''}</div>
          </div>
        </div>
        <p className="wg-p" style={{ margin: '4px 0 6px' }}><strong>By topic</strong></p>
        <ul className="wg-bars">
          {Object.entries(topics).map(([t, v]) => (
            <li key={t} style={{ gridTemplateColumns: '170px minmax(0,1fr) 56px' }}><span className="wg-bar-l" style={{ fontFamily: 'inherit' }}>{t}</span><span className="wg-bar-track"><span className={'wg-bar-fill' + (v.ok / v.n < 0.6 ? ' warn' : '')} style={{ width: `${(v.ok / v.n) * 100}%` }} /></span><span className="wg-bar-p">{v.ok}/{v.n}</span></li>
          ))}
        </ul>
        <p className="wg-note">Topics under 60% are shown in amber: start your revision there.</p>
        <ol className="wg-review">
          {qs.map((x, k) => {
            const ok = ans[k] === x.answer;
            return (
              <li key={k} className={ok ? 'ok' : 'bad'}>
                <div className="wg-rq">{x.q}</div>
                <div className="wg-ra">{ok ? 'Correct: ' : ans[k] === null ? 'Not answered. Answer: ' : `You chose “${x.options[ans[k]]}”. Answer: `}<strong>{x.options[x.answer]}</strong></div>
                {!ok ? <div className="wg-note" style={{ margin: '4px 0 0' }}>{x.why}</div> : null}
              </li>
            );
          })}
        </ol>
        <div className="wg-foot"><span /><button type="button" className="wg-btn" onClick={() => setPhase('intro')}>Take it again</button></div>
      </div>
    );
  }

  const q = qs[i];
  const answered = ans.filter((a) => a !== null).length;
  return (
    <div className="wg">
      <Title fig={spec.fig}>{spec.title || 'Written exam'}</Title>
      <div className="wg-examhead">
        <span>Question {i + 1} of {qs.length} · {answered} answered{q.topic ? ` · ${q.topic}` : ''}</span>
        <span className={'wg-clock' + (left < 300 ? ' low' : '')} role="timer" aria-label="Time left">{mm(left)}</span>
      </div>
      <div className="wg-nav" role="group" aria-label="Question navigator">
        {qs.map((_, k) => <button key={k} type="button" className={'wg-nb' + (k === i ? ' cur' : '') + (ans[k] !== null ? ' done' : '') + (flag[k] ? ' flag' : '')} onClick={() => setI(k)} aria-label={`Question ${k + 1}${ans[k] !== null ? ', answered' : ''}${flag[k] ? ', flagged' : ''}`}>{k + 1}</button>)}
      </div>
      <fieldset className="wg-q">
        <legend>{q.q}</legend>
        {q.options.map((o, k) => (
          <label key={k} className={'wg-opt' + (ans[i] === k ? ' sel' : '')}>
            <input type="radio" name={`${key}-${i}`} checked={ans[i] === k} onChange={() => setAns((a) => a.map((x, j) => (j === i ? k : x)))} />
            <span>{o}</span>
          </label>
        ))}
      </fieldset>
      <div className="wg-foot" style={{ flexWrap: 'wrap', gap: 8 }}>
        <span style={{ display: 'flex', gap: 8 }}>
          <button type="button" className="wg-btn ghost" onClick={() => setI(Math.max(0, i - 1))} disabled={i === 0}>Previous</button>
          <button type="button" className={'wg-btn ghost'} aria-pressed={flag[i]} onClick={() => setFlag((f) => f.map((x, j) => (j === i ? !x : x)))}>{flag[i] ? 'Unflag' : 'Flag for review'}</button>
          {ans[i] !== null ? <button type="button" className="wg-btn ghost" onClick={() => setAns((a) => a.map((x, j) => (j === i ? null : x)))}>Clear answer</button> : null}
        </span>
        <span style={{ display: 'flex', gap: 8 }}>
          {i + 1 < qs.length ? <button type="button" className="wg-btn" onClick={() => setI(i + 1)}>Next</button> : null}
          <button type="button" className={i + 1 < qs.length ? 'wg-btn ghost' : 'wg-btn'} onClick={() => (confirm || answered === qs.length ? finish(ans) : setConfirm(true))}>Submit exam</button>
        </span>
      </div>
      {confirm ? <p className="wg-why bad" role="alert">You have {qs.length - answered} unanswered question{qs.length - answered === 1 ? '' : 's'}{flag.some(Boolean) ? ` and ${flag.filter(Boolean).length} flagged` : ''}. Press Submit exam again to hand in anyway, or keep going.</p> : null}
    </div>
  );
}

// ---------- hook: animated opener that shows what "generative" means ----------
const clamp01 = (x) => Math.max(0, Math.min(1, x));
const typed = (s, f) => s.slice(0, Math.floor(s.length * clamp01(f)));
const SCENE_TICKS = 120; // 50 ms each: 6 s to play, then a short hold
const HOLD_TICKS = 30;

const BUILDINGS = [
  [8, 54, 70], [34, 36, 92], [62, 50, 60], [92, 30, 104], [124, 46, 76], [154, 38, 88], [184, 52, 64], [212, 34, 98],
];

function CityArt({ p, tick }) {
  return (
    <svg viewBox="0 0 250 150" preserveAspectRatio="xMidYMid slice" className="hk-art" role="img" aria-label="A futuristic city at dusk with flying cars">
      <defs>
        <linearGradient id="hkSky" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#0f1b33" />
          <stop offset="1" stopColor="#2c5d7c" />
        </linearGradient>
        <filter id="hkNoise" x="0" y="0" width="100%" height="100%">
          <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="2" seed="4" />
          <feColorMatrix type="saturate" values="0" />
        </filter>
      </defs>
      <g style={{ filter: `blur(${((1 - p) * 9).toFixed(1)}px)` }}>
        <rect width="250" height="150" fill="url(#hkSky)" />
        {BUILDINGS.map(([x, w, h], i) => (
          <g key={x}>
            <rect x={x} y={150 - h} width={w - 6} height={h} fill="#0a1220" />
            {[0, 1, 2, 3].map((r) => (
              <rect key={r} x={x + 5 + (r % 2) * 12} y={150 - h + 8 + Math.floor(r / 2) * 14 + (i % 3) * 6} width="5" height="6" fill="#fcd34d" opacity={(i + r) % 3 === 0 ? 0.25 : 0.85} />
            ))}
          </g>
        ))}
        {[0, 1, 2].map((c) => {
          const x = ((tick * (1.4 + c * 0.5) + c * 90) % 290) - 20;
          return (
            <g key={c} transform={`translate(${x.toFixed(1)} ${28 + c * 22})`}>
              <rect x="-18" y="1" width="16" height="2" rx="1" fill="#76B900" opacity="0.35" />
              <rect width="14" height="5" rx="2.5" fill="#B6F23A" />
            </g>
          );
        })}
      </g>
      <rect width="250" height="150" filter="url(#hkNoise)" opacity={(1 - p) * 0.9} />
    </svg>
  );
}

function WaveArt({ p, tick }) {
  const bars = 30;
  const talking = p > 0.02 && p < 1;
  return (
    <svg viewBox="0 0 250 150" preserveAspectRatio="xMidYMid slice" className="hk-art" role="img" aria-label="An audio waveform of a synthetic voice">
      {Array.from({ length: bars }, (_, i) => {
        const amp = talking ? 0.35 + 0.65 * Math.abs(Math.sin(tick * 0.45 + i * 0.7) * Math.cos(i * 0.31 + tick * 0.12)) : 0.12;
        const h = 8 + amp * 96;
        const played = i / bars <= p;
        return <rect key={i} x={10 + i * 7.8} y={75 - h / 2} width="4.6" height={h} rx="2.3" fill={played ? 'var(--accent)' : 'var(--track)'} />;
      })}
    </svg>
  );
}

function DenoiseArt({ image, p }) {
  const n = image.length; // digit strings: one per row
  const noise = useMemo(() => gaussNoise(n * n, 5), [n]);
  const flat = image.flatMap((r) => [...String(r)].map(Number));
  return (
    <div className="hk-art vz-dimg" style={{ display: 'grid', gridTemplateColumns: `repeat(${n}, 1fr)`, aspectRatio: '1', width: 150, height: 150, margin: '0 auto', borderRadius: 8, overflow: 'hidden' }} role="img" aria-label="A pixel image emerging from noise">
      {flat.map((v, i) => {
        const x = p * (v / 9) + (1 - p) * clamp01((noise[i] + 2) / 4);
        const g = Math.round(clamp01(x) * 255);
        return <i key={i} style={{ background: `rgb(${g},${g},${g})`, display: 'block' }} />;
      })}
    </div>
  );
}

const KERNEL = [[1, 0, -1], [1, 0, -1], [1, 0, -1]];
function ConvArt({ grid, p }) {
  const g = grid || [[0, 0, 0, 9, 9, 9], [0, 0, 0, 9, 9, 9], [0, 0, 0, 9, 9, 9], [0, 0, 0, 9, 9, 9], [0, 0, 0, 9, 9, 9], [0, 0, 0, 9, 9, 9]];
  const out = convolve(g, KERNEL, 1, 0);
  const total = out.length * out[0].length;
  const k = Math.min(total - 1, Math.floor(p * total));
  const wr = Math.floor(k / out[0].length), wc = k % out[0].length;
  return (
    <div className="hk-art" style={{ display: 'flex', gap: 16, alignItems: 'center', justifyContent: 'center', height: 150 }} role="img" aria-label="A 3 by 3 filter sliding over a grid and filling in an output grid">
      <div style={{ display: 'grid', gridTemplateColumns: `repeat(${g[0].length}, 20px)`, gap: 2 }}>
        {g.map((r, i) => r.map((v, j) => {
          const inWin = i >= wr && i < wr + 3 && j >= wc && j < wc + 3;
          return <span key={i + '-' + j} style={{ height: 20, borderRadius: 3, background: inWin ? 'var(--accent)' : `rgb(${255 - v * 20},${255 - v * 20},${255 - v * 20})`, opacity: inWin ? 0.9 : 1, border: '1px solid var(--surface2-border)' }} />;
        }))}
      </div>
      <span style={{ fontSize: 20, color: 'var(--text3)' }}>→</span>
      <div style={{ display: 'grid', gridTemplateColumns: `repeat(${out[0].length}, 26px)`, gap: 2 }}>
        {out.map((r, i) => r.map((v, j) => {
          const idx = i * out[0].length + j;
          return <span key={i + '-' + j} style={{ height: 26, borderRadius: 4, fontSize: 10, display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: 'JetBrains Mono, monospace', background: idx <= k ? 'var(--accent-soft-bg)' : 'var(--surface2)', border: '1px solid var(--surface2-border)', color: 'var(--text)' }}>{idx <= k ? v : ''}</span>;
        }))}
      </div>
    </div>
  );
}

function LossArt({ p }) {
  const N = 60;
  const f = (t) => 0.12 + 1.8 * Math.exp(-3.2 * t) + 0.05 * Math.sin(t * 40) * Math.exp(-2 * t);
  const pts = Array.from({ length: N + 1 }, (_, i) => [10 + (i / N) * 230, 130 - (f(i / N) / 2) * 120]).filter((_, i) => i / N <= p);
  const last = pts[pts.length - 1] || [10, 10];
  return (
    <svg viewBox="0 0 250 150" className="hk-art" role="img" aria-label="A training loss curve going down">
      <line x1="10" y1="130" x2="245" y2="130" stroke="var(--divider)" /><line x1="10" y1="8" x2="10" y2="130" stroke="var(--divider)" />
      <path d={pts.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(1)} ${y.toFixed(1)}`).join(' ')} fill="none" stroke="var(--accent)" strokeWidth="2.6" strokeLinejoin="round" />
      <circle cx={last[0]} cy={last[1]} r="5" fill="var(--accent)" />
      <text x="14" y="20" className="vz-tick">loss</text><text x="212" y="144" className="vz-tick">steps</text>
    </svg>
  );
}

export function HookFigure({ spec }) {
  const scenes = spec.scenes;
  const reduce = useMemo(() => typeof window !== 'undefined' && !!window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches, []);
  const [scene, setScene] = useState(0);
  const [tick, setTick] = useState(reduce ? SCENE_TICKS : 0);
  const [paused, setPaused] = useState(reduce);

  useEffect(() => {
    if (paused) return undefined;
    const id = setInterval(() => setTick((t) => t + 1), 50);
    return () => clearInterval(id);
  }, [paused]);
  useEffect(() => {
    if (tick > SCENE_TICKS + HOLD_TICKS) {
      setScene((s) => (s + 1) % scenes.length);
      setTick(0);
    }
  }, [tick, scenes.length]);

  const sc = scenes[scene];
  const p = clamp01(tick / SCENE_TICKS);
  const promptP = clamp01(p / 0.25);
  const outP = clamp01((p - 0.35) / 0.55);
  const state = p < 0.3 ? 'waiting' : outP < 1 ? 'generating' : 'done';
  const pick = (i) => {
    setScene(i);
    setTick(reduce ? SCENE_TICKS : 0);
  };

  return (
    <figure className="hook">
      <p className="hook-k">{spec.title || 'Watch a model make something new'}</p>
      <div className="hk-stage" role="group" aria-label="Animated examples of generative AI">
        <div className="hk-card">
          <span className="hk-lab">Your prompt</span>
          <span className="hk-txt">
            {typed(sc.prompt, promptP)}
            {promptP < 1 ? <span className="hk-caret" /> : null}
          </span>
        </div>
        <div className={'hk-model ' + state} aria-hidden="true">
          <svg viewBox="0 0 24 24" className="hk-arr">
            <path d="M3 12h15M13 6l6 6-6 6" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <span className="hk-chip">Model</span>
          <span className="hk-dots">
            <i />
            <i />
            <i />
          </span>
        </div>
        <div className="hk-card out">
          <span className="hk-lab">{sc.outputLabel || 'New content'}</span>
          {sc.kind === 'image' ? (
            <CityArt p={outP} tick={tick} />
          ) : sc.kind === 'speech' ? (
            <WaveArt p={outP} tick={tick} />
          ) : sc.kind === 'denoise' ? (
            <DenoiseArt image={sc.image} p={outP} />
          ) : sc.kind === 'conv' ? (
            <ConvArt grid={sc.grid} p={outP} />
          ) : sc.kind === 'loss' ? (
            <LossArt p={outP} />
          ) : sc.kind === 'tokens' ? (
            <span className="hk-toks" aria-live="off">
              {sc.tokens.slice(0, Math.ceil(sc.tokens.length * outP)).map((t, i) => <span key={i} className="hk-tokc">{t}</span>)}
            </span>
          ) : sc.kind === 'bars' ? (
            <span className="hk-bars">
              {sc.bars.map((b) => (
                <span key={b.label} className="hk-bar"><b>{b.label}</b><span className="t"><i style={{ width: `${b.value * 100 * outP}%` }} /></span><span>{Math.round(b.value * 100 * outP)}%</span></span>
              ))}
            </span>
          ) : (
            <span className="hk-txt strong" aria-live="off">
              {typed(sc.output, outP)}
              {outP > 0 && outP < 1 ? <span className="hk-caret" /> : null}
            </span>
          )}
        </div>
      </div>
      <div className="hk-ctl">
        <div className="hk-dotsrow" role="tablist" aria-label="Examples">
          {scenes.map((s, i) => (
            <button key={s.name} type="button" role="tab" aria-selected={i === scene} className={'hk-tab' + (i === scene ? ' on' : '')} onClick={() => pick(i)}>
              {s.name}
            </button>
          ))}
        </div>
        <button type="button" className="hk-pause" onClick={() => setPaused((x) => !x)} aria-pressed={paused}>
          {paused ? 'Play' : 'Pause'}
        </button>
      </div>
      {spec.caption ? <figcaption className="hook-cap">{spec.caption}</figcaption> : null}
    </figure>
  );
}
