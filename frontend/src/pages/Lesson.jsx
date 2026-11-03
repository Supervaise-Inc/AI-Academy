// Generated from design-source/Lesson.dc.html by tools/convert.mjs.
// Re-running the converter regenerates (and overwrites) this file, so edit design-source/ + tools/patches.mjs instead,
// or stop using the converter and edit this file directly.
/* eslint-disable */
import React from 'react';
import DCLogic from '../dc/DCLogic.jsx';
import DcLink from '../dc/DcLink.jsx';
import { css } from '../dc/css.js';
import pageCss from './Lesson.css?inline';
import planCss from '../dc/lessonPlan.css?inline';
import { prepare, PlanView, PlanStatus } from '../dc/lessonPlan.jsx';
import * as session from '../dc/session.js';
import { TOTAL_WEEKS, getWeek } from '../data/courseWeeks.js';

const THEME_KEY = 'supervaise-theme';
const THEME_EVENT = 'supervaise-theme-change';
const DARK_VARS =
  '--nv-dark:block;--nv-light:none;--bg:#090A0F;--text:#FFFFFF;--text2:#94A3B8;--text3:#6B7385;--card-bg:rgba(18,20,28,0.72);--card-border:rgba(255,255,255,0.08);--card-shadow:rgba(0,0,0,0.35);--card-shadow-hover:rgba(0,0,0,0.45);--elev:0 8px 24px rgba(0,0,0,0.35);--scrollbar:#1A1D28;--accent:#76B900;--accent-strong:#8FD400;--accent-bright:#B6F23A;--accent-cta-bg:#76B900;--accent-cta-text:#090A0F;--accent-soft-bg:rgba(118,185,0,0.14);--accent-soft-border:rgba(118,185,0,0.3);--accent-soft-text:#B6F23A;--accent-ghost-bg:rgba(118,185,0,0.08);--accent-ghost-border:rgba(118,185,0,0.25);--accent-glow:rgba(118,185,0,0.55);--accent-glow-soft:rgba(118,185,0,0.6);--accent-pulse:rgba(118,185,0,0.55);--accent-pulse-soft:rgba(118,185,0,0.12);--surface2:rgba(255,255,255,0.05);--surface2-border:rgba(255,255,255,0.08);--divider:rgba(255,255,255,0.08);--cyan:#00E5FF;--cyan-text:#67E8F9;--cyan-chip-bg:#00E5FF;--cyan-chip-text:#06202A;--cyan-glow:rgba(0,229,255,0.45);--cyan-soft-bg:rgba(0,229,255,0.07);--cyan-soft-border:rgba(0,229,255,0.22);--indigo:#8B83F0;--indigo-soft-bg:rgba(91,81,216,0.14);--indigo-soft-border:rgba(91,81,216,0.35);--amber:#FFA000;--amber-text:#FFB84D;--amber-soft-bg:rgba(255,160,0,0.14);--amber-soft-border:rgba(255,160,0,0.3);--amber-ghost-bg:rgba(255,160,0,0.08);--amber-ghost-border:rgba(255,160,0,0.25);--amber-pulse:rgba(255,160,0,0.5);--amber-pulse-soft:rgba(255,160,0,0.14);--red-soft-bg:rgba(239,68,68,0.14);--red-border:rgba(239,68,68,0.45);--red-text:#FCA5A5;--neutral-chip-bg:transparent;--neutral-chip-text:#A0A5B5;--neutral-chip-border:rgba(255,255,255,0.15);--mono-chip-bg:rgba(255,255,255,0.06);--mono-chip-text:#A0A5B5;--mono-chip-today-bg:rgba(255,160,0,0.14);--mono-chip-today-text:#FFB84D;--track:rgba(255,255,255,0.08);--footer-text:#6B7385;--footer-bg:rgba(255,255,255,0.04);--footer-border:rgba(255,255,255,0.08);--avatar-ring:rgba(118,185,0,0.6);--toggle-bg:rgba(255,255,255,0.05);--toggle-border:rgba(255,255,255,0.08);--spot1:rgba(118,185,0,0.22);--spot2:rgba(0,229,255,0.14);--spot3:rgba(91,81,216,0.16);';
const LIGHT_VARS =
  '--nv-dark:none;--nv-light:block;--bg:#F8FAFC;--text:#0F172A;--text2:#475569;--text3:#64748B;--card-bg:#FFFFFF;--card-border:#E2E8F0;--card-shadow:rgba(15,23,42,0.06);--card-shadow-hover:rgba(15,23,42,0.1);--elev:0 4px 6px -1px rgba(0,0,0,0.05),0 2px 4px -2px rgba(0,0,0,0.05);--scrollbar:#CBD5E1;--accent:#659B00;--accent-strong:#4F7A00;--accent-bright:#3D5F00;--accent-cta-bg:#4F7A00;--accent-cta-text:#FFFFFF;--accent-soft-bg:rgba(101,155,0,0.12);--accent-soft-border:rgba(101,155,0,0.35);--accent-soft-text:#4F7A00;--accent-ghost-bg:rgba(101,155,0,0.08);--accent-ghost-border:rgba(101,155,0,0.3);--accent-glow:rgba(101,155,0,0.35);--accent-glow-soft:rgba(101,155,0,0.3);--accent-pulse:rgba(101,155,0,0.4);--accent-pulse-soft:rgba(101,155,0,0.1);--surface2:#F1F5F9;--surface2-border:#E2E8F0;--divider:#E2E8F0;--cyan:#0284C7;--cyan-text:#0369A1;--cyan-chip-bg:#0284C7;--cyan-chip-text:#FFFFFF;--cyan-glow:rgba(2,132,199,0.3);--cyan-soft-bg:rgba(2,132,199,0.08);--cyan-soft-border:rgba(2,132,199,0.28);--indigo:#4F46E5;--indigo-soft-bg:rgba(79,70,229,0.08);--indigo-soft-border:rgba(79,70,229,0.25);--amber:#D97706;--amber-text:#B45309;--amber-soft-bg:rgba(217,119,6,0.12);--amber-soft-border:rgba(217,119,6,0.32);--amber-ghost-bg:rgba(217,119,6,0.08);--amber-ghost-border:rgba(217,119,6,0.28);--amber-pulse:rgba(217,119,6,0.35);--amber-pulse-soft:rgba(217,119,6,0.12);--red-soft-bg:#FEF2F2;--red-border:#FECACA;--red-text:#B91C1C;--neutral-chip-bg:#F1F5F9;--neutral-chip-text:#475569;--neutral-chip-border:#E2E8F0;--mono-chip-bg:#F1F5F9;--mono-chip-text:#475569;--mono-chip-today-bg:rgba(217,119,6,0.12);--mono-chip-today-text:#B45309;--track:#E2E8F0;--footer-text:#64748B;--footer-bg:#F1F5F9;--footer-border:#E2E8F0;--avatar-ring:rgba(101,155,0,0.45);--toggle-bg:#F1F5F9;--toggle-border:#E2E8F0;--spot1:rgba(101,155,0,0.10);--spot2:rgba(2,132,199,0.07);--spot3:rgba(79,70,229,0.08);';
function readTheme() {
  try {
    const s = localStorage.getItem(THEME_KEY);
    if (s === 'light' || s === 'dark') return s;
  } catch (e) {}
  return 'light';
}
function saveTheme(t) {
  try {
    localStorage.setItem(THEME_KEY, t);
  } catch (e) {}
  try {
    window.dispatchEvent(new CustomEvent(THEME_EVENT, { detail: t }));
  } catch (e) {}
}
function bindThemeSync(comp) {
  const onStorage = (e) => {
    if (e.key === THEME_KEY && (e.newValue === 'light' || e.newValue === 'dark') && e.newValue !== comp.state.theme)
      comp.setState({ theme: e.newValue });
  };
  const onLocal = (e) => {
    const t = e.detail;
    if ((t === 'light' || t === 'dark') && t !== comp.state.theme) comp.setState({ theme: t });
  };
  try {
    window.addEventListener('storage', onStorage);
    window.addEventListener(THEME_EVENT, onLocal);
  } catch (e) {}
  return () => {
    try {
      window.removeEventListener('storage', onStorage);
      window.removeEventListener(THEME_EVENT, onLocal);
    } catch (e) {}
  };
}
function themeVals(dark) {
  return {
    themeVars: dark ? DARK_VARS : LIGHT_VARS,
    lightSegBg: dark ? 'transparent' : '#FFFFFF',
    lightSegFg: dark ? '#94A3B8' : '#0F172A',
    lightSegShadow: dark ? 'none' : '0 1px 2px rgba(15,23,42,0.12)',
    darkSegBg: dark ? 'rgba(118,185,0,0.2)' : 'transparent',
    darkSegFg: dark ? '#B6F23A' : '#64748B',
  };
}

const TOKENS = ['The', 'animal', "didn't", 'cross', 'the', 'street', 'because', 'it', 'was', 'too', 'tired', '.'];
// Illustrative, hand-authored attention weights for teaching. Not taken from a trained model.
// Each row is one query token; each row sums to 1.00.
const HEADS = [
  {
    name: 'Head A',
    short: 'long-range',
    W: [
      [0.3, 0.32, 0.04, 0.05, 0.04, 0.08, 0.03, 0.05, 0.03, 0.02, 0.03, 0.01],
      [0.1, 0.15, 0.18, 0.22, 0.04, 0.08, 0.05, 0.1, 0.03, 0.01, 0.03, 0.01],
      [0.03, 0.28, 0.12, 0.3, 0.02, 0.08, 0.04, 0.05, 0.03, 0.02, 0.02, 0.01],
      [0.02, 0.1, 0.14, 0.12, 0.12, 0.3, 0.06, 0.05, 0.04, 0.01, 0.03, 0.01],
      [0.06, 0.04, 0.02, 0.2, 0.18, 0.38, 0.03, 0.03, 0.02, 0.01, 0.02, 0.01],
      [0.02, 0.06, 0.04, 0.3, 0.2, 0.24, 0.04, 0.04, 0.02, 0.01, 0.02, 0.01],
      [0.02, 0.1, 0.18, 0.12, 0.02, 0.06, 0.14, 0.12, 0.08, 0.04, 0.11, 0.01],
      [0.03, 0.46, 0.02, 0.03, 0.03, 0.03, 0.03, 0.14, 0.07, 0.03, 0.12, 0.01],
      [0.02, 0.12, 0.04, 0.03, 0.01, 0.03, 0.05, 0.3, 0.2, 0.06, 0.13, 0.01],
      [0.01, 0.04, 0.02, 0.02, 0.01, 0.02, 0.04, 0.1, 0.16, 0.2, 0.37, 0.01],
      [0.01, 0.06, 0.01, 0.02, 0.01, 0.03, 0.08, 0.28, 0.22, 0.15, 0.12, 0.01],
      [0.04, 0.1, 0.05, 0.08, 0.02, 0.08, 0.06, 0.12, 0.07, 0.05, 0.18, 0.15],
    ],
  },
  {
    name: 'Head B',
    short: 'local',
    W: [
      [0.7, 0.2, 0.04, 0.02, 0.01, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0, 0.01],
      [0.58, 0.24, 0.08, 0.03, 0.02, 0.01, 0.01, 0.01, 0.01, 0.0, 0.0, 0.01],
      [0.06, 0.56, 0.24, 0.07, 0.02, 0.01, 0.01, 0.01, 0.0, 0.0, 0.01, 0.01],
      [0.02, 0.07, 0.55, 0.25, 0.06, 0.02, 0.01, 0.01, 0.0, 0.0, 0.0, 0.01],
      [0.03, 0.02, 0.06, 0.57, 0.24, 0.05, 0.01, 0.01, 0.0, 0.0, 0.0, 0.01],
      [0.01, 0.02, 0.02, 0.08, 0.58, 0.22, 0.04, 0.01, 0.01, 0.0, 0.0, 0.01],
      [0.01, 0.01, 0.01, 0.02, 0.06, 0.6, 0.23, 0.04, 0.01, 0.0, 0.0, 0.01],
      [0.01, 0.02, 0.01, 0.01, 0.01, 0.05, 0.57, 0.24, 0.05, 0.01, 0.01, 0.01],
      [0.0, 0.01, 0.01, 0.01, 0.0, 0.01, 0.06, 0.58, 0.24, 0.05, 0.02, 0.01],
      [0.0, 0.01, 0.0, 0.01, 0.0, 0.01, 0.02, 0.06, 0.56, 0.25, 0.07, 0.01],
      [0.0, 0.01, 0.0, 0.0, 0.0, 0.01, 0.01, 0.02, 0.07, 0.57, 0.27, 0.04],
      [0.01, 0.01, 0.0, 0.01, 0.0, 0.01, 0.01, 0.02, 0.03, 0.08, 0.56, 0.26],
    ],
  },
];
const ROW_H = 28;
function rowY(i) {
  return ROW_H / 2 + i * ROW_H;
}
function linkPath(i, j) {
  const a = rowY(i),
    b = rowY(j);
  return 'M0 ' + a + ' C50 ' + a + ' 50 ' + b + ' 100 ' + b;
}
function linkWidth(w) {
  return (0.6 + w * 9).toFixed(2) + 'px';
}
function linkOpacity(w) {
  return w <= 0 ? 0 : +(0.12 + 0.88 * Math.min(1, w / 0.45)).toFixed(3);
}
function cellOpacity(w) {
  return +Math.min(1, w / 0.45).toFixed(3);
}
function q(word) {
  return '“' + word + '”';
}

// Figure 3: the Laboratory 3 example. One query, three keys, three values, d_k = 2.
const F3 = {
  q: [1, 0],
  K: [
    [1, 0],
    [0, 1],
    [1, 1],
  ],
  V: [
    [1, 0],
    [0, 1],
    [2, 2],
  ],
  dk: 2,
};
const SUB = ['₁', '₂', '₃'];
const F3_STAGES = [
  { label: '1 Scores', qty: 'score q·kᵢ' },
  { label: '2 Scale', qty: 'scaled score' },
  { label: '3 Softmax', qty: 'weight wᵢ' },
  { label: '4 Weighted sum', qty: 'weight wᵢ' },
];
const F3_AXIS = 1.2;
const PLOT_MAX = 2.4;
function f3Compute(scaleOn) {
  const raw = F3.K.map((k) => k[0] * F3.q[0] + k[1] * F3.q[1]);
  const div = scaleOn ? Math.sqrt(F3.dk) : 1;
  const scaled = raw.map((s) => s / div);
  const ex = scaled.map((s) => Math.exp(s));
  const Z = ex.reduce((a, b) => a + b, 0);
  const w = ex.map((e) => e / Z);
  const out = [0, 1].map((d) => w.reduce((a, wi, i) => a + wi * F3.V[i][d], 0));
  return { raw, scaled, ex, Z, w, out, div };
}
function vec(v, n) {
  return '(' + v.map((x) => (n === undefined ? String(x) : x.toFixed(n))).join(', ') + ')';
}
function pct(v) {
  return (Math.max(0, Math.min(1, v / F3_AXIS)) * 100).toFixed(2) + '%';
}

const QUESTIONS = [
  {
    id: 'GA-3.1-Q01',
    question: 'Which objective does an autoregressive language model optimise during pre-training?',
    options: [
      'Classification of each text into a fixed set of categories',
      'Prediction of the next token given the preceding tokens',
      'Compression of token embeddings into fewer dimensions',
      'Reduction of the time required for training',
    ],
    answer_index: 1,
    answer_text:
      'Prediction of the next token given the preceding tokens. Every other capability of the model is learned in service of this objective.',
  },
  {
    id: 'GA-3.1-Q02',
    question: 'Why does a recurrent network struggle to relate words that are far apart in a sentence?',
    options: [
      'Information about early tokens must pass through every intermediate hidden state and degrades on the way',
      'It cannot process sentences of different lengths',
      'It requires a labelled example for every token',
      'It reads future tokens before past ones',
    ],
    answer_index: 0,
    answer_text:
      'Information about early tokens must pass through every intermediate hidden state and degrades on the way. Attention lets any token consult any other token directly.',
  },
  {
    id: 'GA-3.1-Q03',
    question: 'Which statement correctly distinguishes soft attention from hard attention?',
    options: [
      'Soft attention selects one token; hard attention blends all tokens',
      'Soft attention is used only in recurrent networks',
      'Soft attention forms a weighted average of all values and is differentiable; hard attention selects a single token',
      'Hard attention always yields higher accuracy than soft attention',
    ],
    answer_index: 2,
    answer_text:
      'Soft attention forms a weighted average of all values and is differentiable; hard attention selects a single token, which gives no useful gradient.',
  },
  {
    id: 'GA-3.1-Q04',
    question: 'Why are the attention scores divided by √dₖ before the softmax?',
    options: [
      'To make each row of weights sum to one',
      'To keep dot products from growing with the key dimension and saturating the softmax',
      'To reduce the number of trainable parameters',
      'To allow the computation to run in parallel across tokens',
    ],
    answer_index: 1,
    answer_text:
      'To keep dot products from growing with the key dimension and saturating the softmax. Softmax alone already makes each row sum to one.',
  },
];

// Key terms: one source for the inline definitions and the Key terms list.
const TERMS = [
  {
    id: 'hidden',
    name: 'Hidden state',
    def: 'The fixed-length vector that a recurrent network passes from one time step to the next. It is the network’s only memory of earlier input.',
  },
  { id: 'query', name: 'Query', def: 'What a token is looking for in the rest of the sentence.' },
  { id: 'key', name: 'Key', def: 'What a token offers to be matched against other tokens’ queries.' },
  { id: 'value', name: 'Value', def: 'The content a token passes on once it has been matched.' },
  {
    id: 'softmax',
    name: 'Softmax',
    def: 'A function that turns any list of real numbers into positive weights that sum to one, while exaggerating the gap between large and small inputs.',
  },
  {
    id: 'head',
    name: 'Attention head',
    def: 'One independent set of query, key and value projections. Transformers run several heads in parallel so that each can specialise.',
  },
  {
    id: 'scale',
    name: 'Scaling by √dₖ',
    def: 'Dot products grow with the length of the key vectors. Dividing by √dₖ keeps the scores in a range where softmax does not saturate and still passes useful gradients.',
  },
  {
    id: 'hardsoft',
    name: 'Hard and soft attention',
    def: 'Soft attention blends every value by its weight and can be differentiated; hard attention selects a single token, which gives no useful gradient. The distinction is Part A of Laboratory 3.',
  },
];
// Five numbered lesson sections, used by the progress rail and the header.
const SECTIONS = [
  { id: 'concept', label: 'Concept' },
  { id: 'figures', label: 'Figures' },
  { id: 'takeaway', label: 'Takeaway' },
  { id: 'lab', label: 'Laboratory' },
  { id: 'check', label: 'Check' },
];
// Where to look again after each question (only where this lesson covers the concept).
const REVISIT = {
  'GA-3.1-Q02': { label: 'The bottleneck in recurrent networks', href: '#concept' },
  'GA-3.1-Q03': { label: 'Soft weights and parallel computation', href: '#soft' },
  'GA-3.1-Q04': { label: 'Figure 3, scaled dot-product attention', href: '#fig3' },
};

class Component extends DCLogic {
  constructor(props) {
    super(props);
    const saved = readTheme();
    this.state = {
      token: 7,
      hoverTok: null,
      hoverKey: null,
      head: 0,
      nnStep: 0,
      playing: false,
      scaleOn: true,
      f3Hover: null,
      qIndex: 0,
      selectedOpt: null,
      submitted: false,
      theme: saved,
      term: null,
      sec: 1,
      // Week lesson plan (backend/content/week-NN.md) shown instead of the built-in Week 7 lesson
      plan: { status: this.props.week ? 'loading' : 'none', data: null, view: null },
      nb: null, // Colab notebook for the built-in Week 7 lab
    };
    this.toggleTheme = this.toggleTheme.bind(this);
    this.onScroll = this.onScroll.bind(this);
  }
  componentDidMount() {
    this.loadPlan();
    if (!this.props.week || Number(this.props.week) === 7) {
      session
        .api('/notebooks/week-07')
        .then((nb) => this.setState({ nb }))
        .catch(() => {});
    }
    this._unbindTheme = bindThemeSync(this);
    try {
      document.addEventListener('scroll', this.onScroll, { capture: true, passive: true });
      window.addEventListener('resize', this.onScroll);
    } catch (e) {}
    this.updateSec();
  }
  componentWillUnmount() {
    if (this._timer) clearInterval(this._timer);
    if (this._unbindTheme) this._unbindTheme();
    if (this._secT) clearTimeout(this._secT);
    try {
      document.removeEventListener('scroll', this.onScroll, { capture: true });
      window.removeEventListener('resize', this.onScroll);
    } catch (e) {}
  }
  // /lesson/:week -> backend/content/week-NN.md. No file: Week 7 keeps the built-in lesson, other weeks say "not published".
  loadPlan() {
    const week = this.props.week;
    if (!week) return;
    const n = Number(week);
    if (!Number.isInteger(n) || n < 1 || n > 20) {
      this.setState({ plan: { status: 'missing', data: null, view: null } });
      return;
    }
    session
      .api('/content/week-' + String(n).padStart(2, '0'))
      .then((data) => this.setState({ plan: { status: 'ready', data, view: prepare(data.markdown) } }))
      .catch((e) => this.setState({ plan: { status: e.status === 404 ? 'missing' : 'error', data: null, view: null } }));
  }
  onScroll() {
    if (this._secT) return;
    this._secT = setTimeout(() => {
      this._secT = null;
      this.updateSec();
    }, 80);
  }
  updateSec() {
    try {
      const vh = window.innerHeight || 800;
      let cur = 1;
      SECTIONS.forEach((s, i) => {
        const el = document.getElementById(s.id);
        if (el && el.getBoundingClientRect().top <= vh * 0.4) cur = i + 1;
      });
      if (cur !== this.state.sec) this.setState({ sec: cur });
    } catch (e) {}
  }
  toggleTheme() {
    const next = this.state.theme === 'dark' ? 'light' : 'dark';
    this.setState({ theme: next });
    saveTheme(next);
  }
  togglePlay() {
    if (this.state.playing) {
      clearInterval(this._timer);
      this.setState({ playing: false });
    } else {
      this._timer = setInterval(() => {
        this.setState({ nnStep: (this.state.nnStep + 1) % 4 });
      }, 1800);
      this.setState({ playing: true });
    }
  }
  stopPlay() {
    if (this._timer) clearInterval(this._timer);
    this._timer = null;
  }
  hkVals() {
    const cur = this.state.hkOpen || '';
    const tog = (k) => () => this.setState({ hkOpen: (this.state.hkOpen || '') === k ? '' : k });
    const cls = (k) => (cur === k ? 'hk is-open' : 'hk');
    const ex = (k) => (cur === k ? 'true' : 'false');
    return {
      hkTogS: tog('s'),
      hkTogU: tog('u'),
      hkTogA: tog('a'),
      hkTogT: tog('t'),
      hkClsS: cls('s'),
      hkClsU: cls('u'),
      hkClsA: cls('a'),
      hkClsT: cls('t'),
      hkExS: ex('s'),
      hkExU: ex('u'),
      hkExA: ex('a'),
      hkExT: ex('t'),
      hkScrim: cur ? 'block' : 'none',
      hkClose: () => this.setState({ hkOpen: '' }),
      hkKey: (e) => {
        if (e && e.key === 'Escape') this.setState({ hkOpen: '' });
      },
    };
  }
  planNode() {
    const week = this.props.week;
    const { status, data, view } = this.state.plan;
    if (status === 'none' || (status === 'missing' && Number(week) === 7)) return null; // built-in lesson
    if (status === 'ready') return <PlanView plan={data} view={view} />;
    return <PlanStatus week={Number(week)} status={status} />;
  }
  renderVals() {
    const wk = getWeek(this.props.week);
    return {
      ...this.renderVals0(),
      ...this.hkVals(),
      planNode: this.planNode(),
      nbColab: this.state.nb && this.state.nb.colabUrl,
      nbDownload: this.state.nb && session.apiUrl(this.state.nb.downloadUrl),
      ctxWeek: 'Week ' + wk.week + ' of ' + TOTAL_WEEKS,
      ctxTerm: wk.termLabel,
    };
  }
  renderVals0() {
    const dark = this.state.theme === 'dark';

    // ---- Figures 1 and 2: shared selection ----
    const head = HEADS[this.state.head];
    const W = head.W;
    const pinned = this.state.token;
    const qi = this.state.hoverTok === null ? pinned : this.state.hoverTok;
    const kj = this.state.hoverKey;
    const row = W[qi];
    const enterTok = (i) => () => {
      if (this.state.hoverTok !== i || this.state.hoverKey !== null) this.setState({ hoverTok: i, hoverKey: null });
    };
    const pickTok = (i) => () => this.setState({ token: i, hoverTok: null, hoverKey: null });

    const f1Queries = TOKENS.map((w, i) => ({
      word: w,
      pressed: i === qi ? 'true' : 'false',
      aria: 'Show attention from ' + q(w) + ', word ' + (i + 1) + ' of ' + TOKENS.length,
      enter: enterTok(i),
      pick: pickTok(i),
    }));
    const f1Links = TOKENS.map((_, j) => ({ d: linkPath(qi, j), sw: linkWidth(row[j]), op: kj === j ? 0 : linkOpacity(row[j]) }));
    const f1Hl =
      kj === null ? { d: linkPath(qi, qi), sw: '0px', op: 0 } : { d: linkPath(qi, kj), sw: linkWidth(Math.max(row[kj], 0.06)), op: 1 };
    const maxW = Math.max.apply(null, row);
    const f1Keys = TOKENS.map((w, j) => ({
      word: w,
      w: row[j].toFixed(2),
      fg: j === kj || row[j] === maxW ? 'var(--text)' : row[j] >= 0.1 ? 'var(--text2)' : 'var(--text3)',
      fw: j === kj || row[j] === maxW ? 700 : 400,
      enter: () => {
        if (this.state.hoverKey !== j) this.setState({ hoverKey: j });
      },
    }));
    const ranked = row.map((w, j) => ({ w, j })).sort((a, b) => b.w - a.w);
    const top = ranked.slice(0, 3);
    const selfRank = ranked.findIndex((r) => r.j === qi) + 1;
    const f1 = {
      head: head.name,
      q: q(TOKENS[qi]),
      w1: top[0].w.toFixed(2),
      k1: q(TOKENS[top[0].j]),
      w2: top[1].w.toFixed(2),
      k2: q(TOKENS[top[1].j]),
      w3: top[2].w.toFixed(2),
      k3: q(TOKENS[top[2].j]),
      focus:
        kj !== null
          ? 'Highlighted link: ' + q(TOKENS[qi]) + ' to ' + q(TOKENS[kj]) + ', weight ' + row[kj].toFixed(2) + '.'
          : top[0].w >= 0.4
            ? 'Weight is concentrated: the strongest link alone carries ' +
              Math.round(top[0].w * 100) +
              ' per cent of the total. The word itself ranks ' +
              selfRank +
              ' of ' +
              TOKENS.length +
              '.'
            : 'Weight is spread out: no single link carries more than ' +
              Math.round(top[0].w * 100) +
              ' per cent of the total. The word itself ranks ' +
              selfRank +
              ' of ' +
              TOKENS.length +
              '.',
    };

    const hmCols = TOKENS.map((w, j) => ({ word: w, gc: String(j + 2), fg: j === kj ? 'var(--text)' : 'var(--text3)' }));
    const hmRows = TOKENS.map((w, i) => ({
      word: w,
      gr: String(i + 2),
      pressed: i === qi ? 'true' : 'false',
      aria: 'Select row ' + q(w),
      enter: enterTok(i),
      pick: pickTok(i),
    }));
    const hmCells = [];
    for (let i = 0; i < TOKENS.length; i++) {
      for (let j = 0; j < TOKENS.length; j++) {
        const on = i === qi && j === kj;
        hmCells.push({
          gr: String(i + 2),
          gc: String(j + 2),
          op: cellOpacity(W[i][j]),
          ol: on ? '2px solid var(--text)' : 'none',
          z: on ? 3 : 'auto',
          enter: () => {
            if (this.state.hoverTok !== i || this.state.hoverKey !== j) this.setState({ hoverTok: i, hoverKey: j });
          },
        });
      }
    }
    const hm =
      kj === null
        ? {
            lead: 'Point to a cell to read its weight. The outlined row is ' + q(TOKENS[qi]) + '; its largest entry is',
            val: maxW.toFixed(2),
            tail: ', in the column for ' + q(TOKENS[top[0].j]) + '.',
          }
        : {
            lead: 'Row ' + q(TOKENS[qi]) + ', column ' + q(TOKENS[kj]) + ': weight',
            val: W[qi][kj].toFixed(2),
            tail:
              '. That is ' +
              Math.round(W[qi][kj] * 100) +
              ' per cent of the attention ' +
              q(TOKENS[qi]) +
              ' distributes under ' +
              head.name +
              '.',
          };

    // ---- Figure 3: scaled dot-product, staged (reuses nnStep / playing) ----
    const st = this.state.nnStep;
    const scaleOn = this.state.scaleOn;
    const c = f3Compute(scaleOn);
    const cur = st === 0 ? c.raw : st === 1 ? c.scaled : c.w;
    const prev = st === 1 ? c.raw : st === 2 ? c.scaled : null;
    const fh = this.state.f3Hover;
    const hoverBg = 'var(--surface2)';
    const f3Rows = F3.K.map((k, i) => ({
      name: 'k' + SUB[i],
      k: vec(k),
      v: vec(F3.V[i]),
      width: pct(cur[i]),
      val: cur[i].toFixed(3),
      ghost: prev ? pct(prev[i]) : '0%',
      ghostOp: prev ? 1 : 0,
      bg: fh === i ? hoverBg : 'transparent',
      aria: 'Key ' + (i + 1) + ', ' + F3_STAGES[st].qty + ' ' + cur[i].toFixed(3),
      enter: () => {
        if (this.state.f3Hover !== i) this.setState({ f3Hover: i });
      },
    }));
    const r3 = (x) => x.toFixed(3);
    const calcFor = (i) => {
      if (st === 0) return 'q·k' + SUB[i] + ' = (1)(' + F3.K[i][0] + ') + (0)(' + F3.K[i][1] + ') = ' + r3(c.raw[i]);
      if (st === 1)
        return scaleOn
          ? r3(c.raw[i]) + ' / √2 = ' + r3(c.raw[i]) + ' / 1.414 = ' + r3(c.scaled[i])
          : r3(c.raw[i]) + ' (unchanged, scaling off)';
      if (st === 2)
        return 'w' + SUB[i] + ' = exp(' + r3(c.scaled[i]) + ') / ' + r3(c.Z) + ' = ' + r3(c.ex[i]) + ' / ' + r3(c.Z) + ' = ' + r3(c.w[i]);
      return (
        'w' +
        SUB[i] +
        'v' +
        SUB[i] +
        ' = ' +
        r3(c.w[i]) +
        ' × ' +
        vec(F3.V[i]) +
        ' = ' +
        vec(
          F3.V[i].map((x) => x * c.w[i]),
          3
        )
      );
    };
    const notes = [
      'Score each key by its dot product with the query. Keys k₁ and k₃ share the query’s first component and score 1; k₂ is orthogonal to q and scores 0.',
      scaleOn
        ? 'Divide every score by √dₖ = √2 ≈ 1.414. The ranking is unchanged; only the spread between scores shrinks.'
        : 'Scaling is switched off, so the scores pass to the softmax unchanged. Compare the next stage with scaling on.',
      'Apply softmax: exponentiate each score and divide by the sum, ' + r3(c.Z) + '. The three weights are positive and sum to 1.000.',
      'Multiply each value vector by its weight and add the results. The output, ' +
        vec(c.out, 3) +
        ', lies between v₁ and v₃ and is pulled slightly toward v₂.',
    ];
    const f3 = {
      qty: F3_STAGES[st].qty,
      step: 'Stage ' + (st + 1) + ' of 4.',
      note: notes[st],
      calc: fh === null ? (st === 3 ? 'output = ' + vec(c.out, 3) : 'Point to a row to see its arithmetic.') : calcFor(fh),
      ox: ((c.out[0] / PLOT_MAX) * 100).toFixed(2) + '%',
      oy: (100 - (c.out[1] / PLOT_MAX) * 100).toFixed(2) + '%',
      olx: 'calc(' + ((c.out[0] / PLOT_MAX) * 100).toFixed(2) + '% + 10px)',
      oly: 'calc(' + (100 - (c.out[1] / PLOT_MAX) * 100).toFixed(2) + '% - 22px)',
      oOp: st === 3 ? 1 : 0,
    };
    const f3Pts = F3.V.map((v, i) => {
      const d = st >= 2 ? Math.round(8 + 26 * Math.sqrt(c.w[i])) : 9;
      const x = (v[0] / PLOT_MAX) * 100,
        y = 100 - (v[1] / PLOT_MAX) * 100;
      return {
        name: 'v' + SUB[i],
        d: d + 'px',
        op: fh === null || fh === i ? 1 : 0.35,
        x: x.toFixed(2) + '%',
        y: y.toFixed(2) + '%',
        lx: 'calc(' + x.toFixed(2) + '% + ' + (d / 2 + 4) + 'px)',
        ly: 'calc(' + y.toFixed(2) + '% + 2px)',
      };
    });
    const on = 'var(--accent-strong)',
      off = 'var(--text3)',
      clear = 'transparent';
    const eq = {
      qk: st === 0 ? on : 'var(--text2)',
      qkB: st === 0 ? on : clear,
      sc: st === 1 ? on : scaleOn ? 'var(--text2)' : off,
      scB: st === 1 ? on : clear,
      sm: st === 2 ? on : 'var(--text2)',
      smB: st === 2 ? on : clear,
      v: st === 3 ? on : 'var(--text2)',
      vB: st === 3 ? on : clear,
    };
    const f3Stages = F3_STAGES.map((s, i) => ({
      label: s.label,
      pressed: i === st ? 'true' : 'false',
      pick: () => {
        this.stopPlay();
        this.setState({ nnStep: i, playing: false });
      },
    }));

    // ---- Key terms (inline popovers and list) ----
    const tm = {};
    TERMS.forEach((t) => {
      const open = this.state.term === t.id;
      tm[t.id] = {
        name: t.name,
        def: t.def,
        open,
        exp: open ? 'true' : 'false',
        toggle: () => this.setState({ term: this.state.term === t.id ? null : t.id }),
        key: (e) => {
          if (e && e.key === 'Escape' && this.state.term === t.id) this.setState({ term: null });
        },
      };
    });
    const termList = TERMS.map((t) => ({ name: t.name, def: t.def }));

    // ---- Section rail ----
    const sec = this.state.sec;
    const rail = SECTIONS.map((s, i) => ({
      label: s.label,
      href: '#' + s.id,
      cls: i + 1 < sec ? 'done' : i + 1 === sec ? 'cur' : '',
      cur: i + 1 === sec ? 'step' : 'false',
      go: () => {
        if (this.state.sec !== i + 1) this.setState({ sec: i + 1 });
      },
    }));

    // ---- Check for understanding ----
    const qn = QUESTIONS[this.state.qIndex];
    const submitted = this.state.submitted;
    const selected = this.state.selectedOpt;
    const opts = qn.options.map((opt, i) => {
      const sel = selected === i;
      let cls = 'qopt',
        mark = '';
      if (submitted) {
        cls += ' is-done';
        if (i === qn.answer_index) {
          cls += ' is-right';
          mark = sel ? 'Your answer, correct' : 'Correct answer';
        } else if (sel) {
          cls += ' is-wrong';
          mark = 'Your answer';
        }
      } else if (sel) {
        cls += ' is-sel';
      }
      return {
        text: opt,
        cls,
        mark,
        pressed: sel ? 'true' : 'false',
        locked: submitted ? 'true' : 'false',
        pick: () => {
          if (!this.state.submitted) this.setState({ selectedOpt: i });
        },
      };
    });
    const right = submitted && selected === qn.answer_index;
    const revisit = REVISIT[qn.id] || { label: '', href: '#check' };

    return {
      ...themeVals(dark),
      toggleTheme: this.toggleTheme,
      tm,
      termList,
      closeTerm: () => this.setState({ term: null }),
      rail,
      secNum: sec,
      secTotal: SECTIONS.length,
      headAPressed: this.state.head === 0 ? 'true' : 'false',
      headBPressed: this.state.head === 1 ? 'true' : 'false',
      pickHeadA: () => this.setState({ head: 0 }),
      pickHeadB: () => this.setState({ head: 1 }),
      clearHover: () => {
        if (this.state.hoverTok !== null || this.state.hoverKey !== null) this.setState({ hoverTok: null, hoverKey: null });
      },
      f1Queries,
      f1Links,
      f1Hl,
      f1Keys,
      f1,
      hmCols,
      hmRows,
      hmCells,
      hmBandRow: String(qi + 2),
      hm,
      f3Rows,
      f3Pts,
      f3Stages,
      f3,
      eq,
      clearF3: () => {
        if (this.state.f3Hover !== null) this.setState({ f3Hover: null });
      },
      scalePressed: scaleOn ? 'true' : 'false',
      scaleLabel: scaleOn ? 'on' : 'off',
      toggleScale: () => this.setState({ scaleOn: !scaleOn }),
      playLabel: this.state.playing ? 'Pause' : 'Play',
      togglePlay: () => this.togglePlay(),
      stepFwd: () => {
        this.stopPlay();
        this.setState({ nnStep: (st + 1) % 4, playing: false });
      },
      stepBack: () => {
        this.stopPlay();
        this.setState({ nnStep: (st + 3) % 4, playing: false });
      },
      reset: () => {
        this.stopPlay();
        this.setState({ nnStep: 0, playing: false, scaleOn: true, f3Hover: null });
      },
      qText: qn.question,
      qNum: this.state.qIndex + 1,
      qTotal: QUESTIONS.length,
      opts,
      submitted,
      notSubmitted: !submitted,
      noSelection: !submitted && selected === null,
      fbRight: right,
      fbWrong: submitted && !right,
      hasRevisit: !!REVISIT[qn.id],
      revisit,
      submit: () => {
        if (this.state.selectedOpt === null) return;
        this.setState({ submitted: true });
      },
      correctText: qn.answer_text,
      next: () => {
        this.setState({ qIndex: (this.state.qIndex + 1) % QUESTIONS.length, selectedOpt: null, submitted: false });
      },
    };
  }
}

function template(v) {
  return (
    <>
      <div
        style={css(
          `${v.themeVars}background:var(--bg);color:var(--text);transition:background-color .3s ease,color .3s ease;width:100%;min-height:100vh;display:flex;flex-wrap:wrap`
        )}
      >
        <div
          className="sidebar"
          style={{
            flex: '0 0 240px',
            minWidth: '240px',
            borderRight: '1px solid rgba(255,255,255,0.08)',
            background: 'linear-gradient(180deg,#0D1117,#0A0D12)',
            boxShadow: 'inset -1px 0 0 rgba(118,185,0,0.06)',
            padding: '24px 16px',
            display: 'flex',
            flexDirection: 'column',
            gap: '4px',
          }}
        >
          <div className="sb-course" style={{ padding: '8px 10px 22px' }}>
            <div
              style={{
                fontSize: '10.5px',
                fontWeight: '700',
                letterSpacing: '.12em',
                textTransform: 'uppercase',
                color: '#7C857A',
                lineHeight: '1.3',
                marginBottom: '6px',
              }}
            >
              {'Course'}
            </div>
            <div style={{ fontSize: '15px', fontWeight: '800', color: '#FFFFFF', lineHeight: '1.3' }}>{'GenAI & LLMs with NVIDIA'}</div>
            <div
              style={{ width: '28px', height: '3px', borderRadius: '999px', background: 'rgba(255,255,255,0.16)', marginTop: '10px' }}
            ></div>
          </div>
          <DcLink className="navlink" href="Main.dc.html">
            <span>{'●'}</span>
            <span>{'Overview'}</span>
          </DcLink>
          <DcLink className="navlink" href="CourseMap.dc.html">
            <span>{'◉'}</span>
            <span>{'Course Map'}</span>
          </DcLink>
          <DcLink className="navlink active" href="Lesson.dc.html" aria-current="page">
            <span>{'◇'}</span>
            <span>{'Lesson Workspace'}</span>
          </DcLink>
          <DcLink className="navlink" href="PromptWorkshop.dc.html">
            <span>{'✎'}</span>
            <span>{'Prompt Workshop'}</span>
          </DcLink>
          <DcLink className="navlink" href="Capstone.dc.html">
            <span>{'⚐'}</span>
            <span>{'Capstone Project'}</span>
          </DcLink>
          <DcLink className="navlink" href="PracticeExam.dc.html">
            <span>{'⏱'}</span>
            <span>{'Mock Exam Prep'}</span>
          </DcLink>
          <DcLink className="navlink" href="Certificates.dc.html">
            <span>{'▲'}</span>
            <span>{'Certification Center'}</span>
          </DcLink>
          <div className="sb-spacer" style={{ flex: '1' }}></div>
          <div className="sb-profile" style={{ borderTop: '1px solid rgba(255,255,255,0.08)', margin: '12px 0 0', paddingTop: '12px' }}>
            <div className="profilecard" style={{ display: 'flex', alignItems: 'flex-start', gap: '10px', marginBottom: '6px' }}>
              <div
                style={{
                  flex: '0 0 auto',
                  width: '34px',
                  height: '34px',
                  borderRadius: '999px',
                  background: 'linear-gradient(135deg,#76B900,#5B51D8)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '12px',
                  fontWeight: '800',
                  color: '#FFFFFF',
                }}
              >
                {v.userInitials}
              </div>
              <div style={{ minWidth: '0' }}>
                <div style={{ fontSize: '13px', fontWeight: '700', color: '#FFFFFF', lineHeight: '1.3' }}>{v.userName}</div>
                <div style={{ fontSize: '11.5px', color: '#9BA3AF', lineHeight: '1.4' }}>{'Student · Week 7 of 20'}</div>
                <div style={{ marginTop: '6px' }}>
                  <span
                    style={{
                      display: 'inline-block',
                      padding: '2px 8px',
                      borderRadius: '999px',
                      background: 'rgba(255,255,255,0.06)',
                      color: '#9BA3AF',
                      fontSize: '11px',
                      fontWeight: '700',
                      lineHeight: '1.5',
                    }}
                  >
                    {'Section A'}
                  </span>
                </div>
              </div>
            </div>
            <DcLink className="navlink" href="Landing.dc.html" onClick={v.signOut} style={{ color: '#7C857A' }}>
              <span>{'⎋'}</span>
              <span>{'Sign out'}</span>
            </DcLink>
          </div>
        </div>
        <div
          className="cm"
          style={{
            flex: '999 1 560px',
            minWidth: '0',
            background: 'var(--bg)',
            color: 'var(--text)',
            padding: '36px 0 64px',
            transition: 'background-color .3s ease,color .3s ease',
          }}
        >
          <div className="lw">
            <div className="utilrow" role="toolbar" aria-label="Course tools" onKeyDown={v.hkKey}>
              <div className="hk-scrim" style={css(`display:${v.hkScrim}`)} onClick={v.hkClose}></div>
              <div className="ctx">
                <span>{'A.Y. 2026/2027'}</span> <span className="ctx-s" aria-hidden="true"></span> <span>{'Second Semester'}</span>{' '}
                <span className="ctx-s" aria-hidden="true"></span> <span className="ctx-now">{v.ctxWeek}</span>{' '}
                <span className="ctx-s" aria-hidden="true"></span> <span>{v.ctxTerm}</span>
              </div>
              <div className={v.hkClsS}>
                <button
                  type="button"
                  className="hkbtn"
                  aria-haspopup="true"
                  aria-expanded={v.hkExS}
                  onClick={v.hkTogS}
                  aria-label="Schedule"
                  title="Schedule"
                >
                  <svg className="hkic" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                    <rect x="3.2" y="4.8" width="17.6" height="15.4" rx="2.6" stroke="currentColor" strokeWidth="1.7"></rect>
                    <path d="M3.2 9.4h17.6M8.2 2.9v3.6M15.8 2.9v3.6" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round"></path>
                  </svg>
                </button>
                <div className="hkpop">
                  <div className="hkpop-h">{'Today · Wednesday, 14 October 2026'}</div>
                  <ul className="hkl">
                    <li>
                      <span className="hkt">{'08:00'}</span>
                      <span className="hkd">
                        <span className="hki">{'Lecture 7.1: Attention mechanisms'}</span>
                        <span className="hkm">{'Room CS‑204 · Dr. J. A. Jose'}</span>
                      </span>
                    </li>
                    <li>
                      <span className="hkt">{'10:30'}</span>
                      <span className="hkd">
                        <span className="hki">{'Laboratory 3: Scaled dot‑product attention'}</span>
                        <span className="hkm">{'Computing Laboratory 2 · Attendance required'}</span>
                      </span>
                    </li>
                    <li>
                      <span className="hkt">{'14:00'}</span>
                      <span className="hkd">
                        <span className="hki">{'Consultation hours'}</span>
                        <span className="hkm">{'By appointment · Faculty Room 5'}</span>
                      </span>
                    </li>
                  </ul>
                  <DcLink className="hkpop-f" href="CourseMap.dc.html">
                    {'View the full course schedule'}
                  </DcLink>
                </div>
              </div>
              <div className={v.hkClsU}>
                <button
                  type="button"
                  className="hkbtn"
                  aria-haspopup="true"
                  aria-expanded={v.hkExU}
                  onClick={v.hkTogU}
                  aria-label="Upcoming requirements"
                  title="Upcoming requirements"
                >
                  <svg className="hkic" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                    <circle cx="12" cy="12" r="8.8" stroke="currentColor" strokeWidth="1.7"></circle>
                    <path
                      d="M12 6.9V12l3.5 2.1"
                      stroke="currentColor"
                      strokeWidth="1.7"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    ></path>
                  </svg>
                </button>
                <div className="hkpop">
                  <div className="hkpop-h">{'Upcoming requirements'}</div>
                  <ul className="hkl">
                    <li>
                      <span className="hkt">{'17 Oct'}</span>
                      <span className="hkd">
                        <span className="hki">{'Laboratory 3 report'}</span>
                        <span className="hkm">{'10% of final grade · Due 11:59 PM PHT'}</span>
                      </span>
                    </li>
                    <li>
                      <span className="hkt">{'24 Oct'}</span>
                      <span className="hkd">
                        <span className="hki">{'Quiz 4: Transformer architecture'}</span>
                        <span className="hkm">{'5% of final grade · Two attempts permitted'}</span>
                      </span>
                    </li>
                    <li>
                      <span className="hkt">{'07 Nov'}</span>
                      <span className="hkd">
                        <span className="hki">{'Midterm examination'}</span>
                        <span className="hkm">{'25% of final grade · Weeks 6 to 10 covered'}</span>
                      </span>
                    </li>
                  </ul>
                  <DcLink className="hkpop-f" href="Certificates.dc.html">
                    {'View all assessments and weights'}
                  </DcLink>
                </div>
              </div>
              <div className={v.hkClsA}>
                <button
                  type="button"
                  className="hkbtn"
                  aria-haspopup="true"
                  aria-expanded={v.hkExA}
                  onClick={v.hkTogA}
                  aria-label="Announcements, 3 new"
                  title="Announcements"
                >
                  <svg className="hkic" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                    <path
                      d="M18.3 16.2V10.6a6.3 6.3 0 1 0-12.6 0v5.6l-1.6 2.3h15.8z"
                      stroke="currentColor"
                      strokeWidth="1.7"
                      strokeLinejoin="round"
                    ></path>
                    <path d="M9.9 21.1h4.2" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round"></path>
                  </svg>
                  <span className="hkn" aria-hidden="true">
                    {'3'}
                  </span>
                </button>
                <div className="hkpop">
                  <div className="hkpop-h">{'Announcements'}</div>
                  <ul className="hkl">
                    <li>
                      <span className="hkt">{'Today'}</span>
                      <span className="hkd">
                        <span className="hki">{'Laboratory 3 rubric published'}</span>
                        <span className="hkm">{'Posted by Dr. J. A. Jose'}</span>
                      </span>
                    </li>
                    <li>
                      <span className="hkt">{'13 Oct'}</span>
                      <span className="hkd">
                        <span className="hki">{'Quiz 3 results released'}</span>
                        <span className="hkm">{'Class mean 82% · Review in consultation hours'}</span>
                      </span>
                    </li>
                    <li>
                      <span className="hkt">{'11 Oct'}</span>
                      <span className="hkd">
                        <span className="hki">{'Reading added for Week 8'}</span>
                        <span className="hkm">{'Vaswani et al., sections 3.2 to 3.4'}</span>
                      </span>
                    </li>
                  </ul>
                  <DcLink className="hkpop-f" href="#">
                    {'View all announcements'}
                  </DcLink>
                </div>
              </div>
              <div className={v.hkClsT}>
                <button
                  type="button"
                  className="hkbtn"
                  aria-haspopup="true"
                  aria-expanded={v.hkExT}
                  onClick={v.hkTogT}
                  aria-label="Settings"
                  title="Settings"
                >
                  <svg className="hkic" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                    <circle cx="12" cy="12" r="3.1" stroke="currentColor" strokeWidth="1.7"></circle>
                    <path
                      d="M19.3 14.4a1.6 1.6 0 0 0 .3 1.8l.1.1a1.9 1.9 0 1 1-2.7 2.7l-.1-.1a1.6 1.6 0 0 0-2.7 1.1v.3a1.9 1.9 0 0 1-3.8 0v-.2a1.6 1.6 0 0 0-2.8-1.1l-.1.1a1.9 1.9 0 1 1-2.7-2.7l.1-.1a1.6 1.6 0 0 0-1.1-2.7h-.3a1.9 1.9 0 0 1 0-3.8h.2a1.6 1.6 0 0 0 1.1-2.8l-.1-.1a1.9 1.9 0 1 1 2.7-2.7l.1.1a1.6 1.6 0 0 0 1.8.3h.1a1.6 1.6 0 0 0 1-1.5v-.3a1.9 1.9 0 0 1 3.8 0v.2a1.6 1.6 0 0 0 2.7 1.1l.1-.1a1.9 1.9 0 1 1 2.7 2.7l-.1.1a1.6 1.6 0 0 0 1.1 2.7h.3a1.9 1.9 0 0 1 0 3.8h-.2a1.6 1.6 0 0 0-1.5 1z"
                      stroke="currentColor"
                      strokeWidth="1.5"
                      strokeLinejoin="round"
                    ></path>
                  </svg>
                </button>
                <div className="hkpop">
                  <div className="hkpop-h">{'Settings'}</div>
                  <div className="setrow">
                    <span className="setlab">{'Appearance'}</span>{' '}
                    <button
                      type="button"
                      className="themepill"
                      onClick={v.toggleTheme}
                      aria-label="Switch between light and dark theme"
                      title="Switch between light and dark theme"
                    >
                      <span
                        className="themeseg"
                        style={css(`background:${v.lightSegBg};color:${v.lightSegFg};box-shadow:${v.lightSegShadow}`)}
                      >
                        {'Light'}
                      </span>
                      <span className="themeseg" style={css(`background:${v.darkSegBg};color:${v.darkSegFg}`)}>
                        {'Dark'}
                      </span>
                    </button>
                  </div>
                  <div className="setrow">
                    <span className="setlab">{'Notifications'}</span>
                    <span className="setval">{'Email and in‑app'}</span>
                  </div>
                  <div className="setrow">
                    <span className="setlab">{'Time zone'}</span>
                    <span className="setval">
                      {'Philippine Standard Time'}
                      <br />
                      {'(UTC+8)'}
                    </span>
                  </div>
                  <div className="setrow">
                    <span className="setlab">{'Enrolled section'}</span>
                    <span className="setval">{'Section A'}</span>
                  </div>
                  <DcLink className="hkpop-f" href="#">
                    {'Account and privacy settings'}
                  </DcLink>
                </div>
              </div>
            </div>
            {v.planNode ? (
              v.planNode
            ) : (
              <>
                <header className="lhead hx hx-cyan">
                  <div className="hx-body">
                    <h1 className="lh-title">{'Attention mechanisms'}</h1>
                    <p className="lh-purpose">
                      {'Learn how a Transformer lets '}
                      <mark className="mk">{'every token weigh every other token'}</mark>
                      {', and why that replaced recurrence.'}
                    </p>
                    <details className="disc">
                      <summary>{'Lesson details'}</summary>
                      <div className="disc-b">
                        <dl className="kv">
                          <dt>{'Lesson'}</dt>
                          <dd>{'Week 7 · Lesson 7.1 · about 20 minutes'}</dd>
                          <dt>{'Progress'}</dt>
                          <dd aria-live="polite">
                            {'Section '}
                            {v.secNum}
                            {' of '}
                            {v.secTotal}
                          </dd>
                          <dt>{'Due'}</dt>
                          <dd>{'Laboratory 3 report, 17 October 2026, 11:59 PM PHT'}</dd>
                          <dt>{'Instructor'}</dt>
                          <dd>{'Dr. J. A. Jose'}</dd>
                          <dt>{'Unit'}</dt>
                          <dd>{'GA‑3.1 · Lecture 7.1'}</dd>
                          <dt>{'Term'}</dt>
                          <dd>{'A.Y. 2026/2027 · Second Semester · Midterm Term · Week 7 of 20'}</dd>
                          <dt>{'Required reading'}</dt>
                          <dd>
                            {'Vaswani et al. (2017), section 3.2 '}
                            <DcLink className="cite" href="#ref-1">
                              {'1'}
                            </DcLink>
                          </dd>
                        </dl>
                      </div>
                    </details>
                  </div>
                  <div className="hx-ico">
                    <svg viewBox="0 0 24 24" aria-hidden="true">
                      <path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z"></path>
                      <circle cx="12" cy="12" r="3"></circle>
                    </svg>
                  </div>
                </header>
                <figure className="hook">
                  <p className="hook-k">{'Watch one word read its sentence'}</p>
                  <svg
                    viewBox="0 0 960 280"
                    role="img"
                    aria-label="Animated diagram. In the sentence The animal was tired so it slept, the word it draws arcs to every other word. The arc to animal is thickest, about 55 percent, then tired at 20 percent, and the rest are thin."
                  >
                    <rect className="hk-ring" x="660" y="204" width="120" height="54" rx="14"></rect>
                    <path
                      className="hk-arc"
                      pathLength="1"
                      d="M720 204 Q395 -129 70 204"
                      strokeWidth="2.0"
                      style={{ '--o': '0.45', '--d': '0.00s' }}
                    ></path>
                    <path
                      className="hk-arc strong"
                      pathLength="1"
                      d="M720 204 Q460 -74 200 204"
                      strokeWidth="10.3"
                      style={{ '--o': '1', '--d': '0.90s' }}
                    ></path>
                    <path
                      className="hk-arc"
                      pathLength="1"
                      d="M720 204 Q525 -19 330 204"
                      strokeWidth="2.1"
                      style={{ '--o': '0.45', '--d': '0.30s' }}
                    ></path>
                    <path
                      className="hk-arc"
                      pathLength="1"
                      d="M720 204 Q590 34 460 204"
                      strokeWidth="4.7"
                      style={{ '--o': '0.8', '--d': '0.45s' }}
                    ></path>
                    <path
                      className="hk-arc"
                      pathLength="1"
                      d="M720 204 Q655 89 590 204"
                      strokeWidth="2.3"
                      style={{ '--o': '0.45', '--d': '0.60s' }}
                    ></path>
                    <path
                      className="hk-arc"
                      pathLength="1"
                      d="M720 204 Q785 89 850 204"
                      strokeWidth="2.3"
                      style={{ '--o': '0.45', '--d': '0.75s' }}
                    ></path>
                    <rect className="hk-chip" x="10" y="204" width="120" height="54" rx="14"></rect>
                    <rect className="hk-chip is-hit" x="140" y="204" width="120" height="54" rx="14"></rect>
                    <rect className="hk-chip" x="270" y="204" width="120" height="54" rx="14"></rect>
                    <rect className="hk-chip" x="400" y="204" width="120" height="54" rx="14"></rect>
                    <rect className="hk-chip" x="530" y="204" width="120" height="54" rx="14"></rect>
                    <rect className="hk-chip is-q" x="660" y="204" width="120" height="54" rx="14"></rect>
                    <rect className="hk-chip" x="790" y="204" width="120" height="54" rx="14"></rect>
                    <text className="hk-w" x="70" y="240">
                      {'The'}
                    </text>
                    <text className="hk-w" x="200" y="240">
                      {'animal'}
                    </text>
                    <text className="hk-w" x="330" y="240">
                      {'was'}
                    </text>
                    <text className="hk-w" x="460" y="240">
                      {'tired'}
                    </text>
                    <text className="hk-w" x="590" y="240">
                      {'so'}
                    </text>
                    <text className="hk-w q" x="720" y="240">
                      {'it'}
                    </text>
                    <text className="hk-w" x="850" y="240">
                      {'slept'}
                    </text>
                    <text className="hk-pct strong" x="460" y="50" style={{ '--d': '0.90s' }}>
                      {'55%'}
                    </text>
                    <text className="hk-pct" x="590" y="105" style={{ '--d': '0.45s' }}>
                      {'20%'}
                    </text>
                  </svg>
                  <figcaption className="hook-cap">
                    {
                      'Reading “it”, the model gives most of its weight to “animal” (55%) and “tired” (20%). Every word does this for every other word at the same time. That is attention.'
                    }
                  </figcaption>
                </figure>
                <nav className="rail" aria-label="Lesson sections">
                  <ol>
                    {(v.rail || []).map((s, $index) => (
                      <React.Fragment key={$index}>
                        <li>
                          <DcLink className={s?.cls} href={s?.href} aria-current={s?.cur} onClick={s?.go}>
                            <i aria-hidden="true"></i>
                            <span>{s?.label}</span>
                          </DcLink>
                        </li>
                      </React.Fragment>
                    ))}
                  </ol>
                </nav>
                <details className="disc outc" id="outcomes" open>
                  <summary>{'By the end, you can...'}</summary>
                  <ul className="chk">
                    <li>
                      <svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
                        <path
                          d="M3 8.4l3.1 3L13 4.6"
                          stroke="currentColor"
                          strokeWidth="1.8"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        ></path>
                      </svg>
                      <span>
                        {'Explain why recurrent networks lose information over long sequences, and how attention removes that bottleneck.'}
                      </span>
                    </li>
                    <li>
                      <svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
                        <path
                          d="M3 8.4l3.1 3L13 4.6"
                          stroke="currentColor"
                          strokeWidth="1.8"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        ></path>
                      </svg>
                      <span>{'Read an attention-weight matrix, each row a distribution over the tokens of a sentence.'}</span>
                    </li>
                    <li>
                      <svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
                        <path
                          d="M3 8.4l3.1 3L13 4.6"
                          stroke="currentColor"
                          strokeWidth="1.8"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        ></path>
                      </svg>
                      <span>{'Compute scaled dot-product attention by hand for one query, three keys and three values.'}</span>
                    </li>
                    <li>
                      <svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
                        <path
                          d="M3 8.4l3.1 3L13 4.6"
                          stroke="currentColor"
                          strokeWidth="1.8"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        ></path>
                      </svg>
                      <span>
                        {'Implement it in PyTorch and evaluate the effect of the √d'}
                        <sub>{'k'}</sub>
                        {' scaling term.'}
                      </span>
                    </li>
                  </ul>
                  <p className="chk-n">{'Assessed in Laboratory 3 and Quiz 4.'}</p>
                </details>
                <section className="sec" id="concept" aria-labelledby="concept-h">
                  <h2 className="sec-h" id="concept-h">
                    {'Why attention replaced recurrence'}
                  </h2>
                  <h3 className="h3">{'The bottleneck in recurrent networks'}</h3>
                  <p className="prose">
                    {'A recurrent network reads a sentence one token at a time and folds everything it has read into a single '}
                    <button
                      type="button"
                      className="term"
                      aria-expanded={v.tm?.hidden?.exp}
                      onClick={v.tm?.hidden?.toggle}
                      onKeyDown={v.tm?.hidden?.key}
                    >
                      {'hidden state'}
                    </button>
                    {v.tm?.hidden?.open ? (
                      <>
                        <span className="tpop" role="note">
                          <span className="tpop-t">{v.tm?.hidden?.name}</span>
                          {v.tm?.hidden?.def}
                          <button type="button" className="tpop-x" aria-label="Close definition" onClick={v.closeTerm}>
                            <svg viewBox="0 0 14 14" fill="none" aria-hidden="true">
                              <path d="M3 3l8 8M11 3l-8 8" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round"></path>
                            </svg>
                          </button>
                        </span>
                      </>
                    ) : null}
                    {' of fixed size.'}
                  </p>
                  <p className="prose">
                    {
                      'Whatever the network knows about the first word must survive every later update to that state. In practice it does not survive intact: by the end of a long sentence, the opening words leave only a faint trace, and the model struggles to connect a pronoun to the noun it refers to several words earlier.'
                    }
                    <DcLink className="cite" href="#ref-2">
                      {'2'}
                    </DcLink>
                  </p>
                  <h3 className="h3">{'Every token consults every other token'}</h3>
                  <p className="prose lede">
                    {'Attention removes this bottleneck. Instead of relaying information step by step, '}
                    <mark className="mk">{'every token consults every other token directly'}</mark>
                    {' and decides how much each one should contribute to its own representation.'}
                  </p>
                  <p className="prose">
                    {'To do so, each token is projected into three vectors: a '}
                    <button
                      type="button"
                      className="term"
                      aria-expanded={v.tm?.query?.exp}
                      onClick={v.tm?.query?.toggle}
                      onKeyDown={v.tm?.query?.key}
                    >
                      {'query'}
                    </button>
                    {v.tm?.query?.open ? (
                      <>
                        <span className="tpop" role="note">
                          <span className="tpop-t">{v.tm?.query?.name}</span>
                          {v.tm?.query?.def}
                          <button type="button" className="tpop-x" aria-label="Close definition" onClick={v.closeTerm}>
                            <svg viewBox="0 0 14 14" fill="none" aria-hidden="true">
                              <path d="M3 3l8 8M11 3l-8 8" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round"></path>
                            </svg>
                          </button>
                        </span>
                      </>
                    ) : null}
                    {', a '}
                    <button
                      type="button"
                      className="term"
                      aria-expanded={v.tm?.key?.exp}
                      onClick={v.tm?.key?.toggle}
                      onKeyDown={v.tm?.key?.key}
                    >
                      {'key'}
                    </button>
                    {v.tm?.key?.open ? (
                      <>
                        <span className="tpop" role="note">
                          <span className="tpop-t">{v.tm?.key?.name}</span>
                          {v.tm?.key?.def}
                          <button type="button" className="tpop-x" aria-label="Close definition" onClick={v.closeTerm}>
                            <svg viewBox="0 0 14 14" fill="none" aria-hidden="true">
                              <path d="M3 3l8 8M11 3l-8 8" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round"></path>
                            </svg>
                          </button>
                        </span>
                      </>
                    ) : null}
                    {' and a '}
                    <button
                      type="button"
                      className="term"
                      aria-expanded={v.tm?.value?.exp}
                      onClick={v.tm?.value?.toggle}
                      onKeyDown={v.tm?.value?.key}
                    >
                      {'value'}
                    </button>
                    {v.tm?.value?.open ? (
                      <>
                        <span className="tpop" role="note">
                          <span className="tpop-t">{v.tm?.value?.name}</span>
                          {v.tm?.value?.def}
                          <button type="button" className="tpop-x" aria-label="Close definition" onClick={v.closeTerm}>
                            <svg viewBox="0 0 14 14" fill="none" aria-hidden="true">
                              <path d="M3 3l8 8M11 3l-8 8" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round"></path>
                            </svg>
                          </button>
                        </span>
                      </>
                    ) : null}
                    {'. Then:'}
                  </p>
                  <ol className="plist">
                    <li>{"One token's query is compared with the key of every token in the sentence."}</li>
                    <li>
                      {'The comparison scores are converted into weights by a '}
                      <button
                        type="button"
                        className="term"
                        aria-expanded={v.tm?.softmax?.exp}
                        onClick={v.tm?.softmax?.toggle}
                        onKeyDown={v.tm?.softmax?.key}
                      >
                        {'softmax'}
                      </button>
                      {v.tm?.softmax?.open ? (
                        <>
                          <span className="tpop" role="note">
                            <span className="tpop-t">{v.tm?.softmax?.name}</span>
                            {v.tm?.softmax?.def}
                            <button type="button" className="tpop-x" aria-label="Close definition" onClick={v.closeTerm}>
                              <svg viewBox="0 0 14 14" fill="none" aria-hidden="true">
                                <path d="M3 3l8 8M11 3l-8 8" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round"></path>
                              </svg>
                            </button>
                          </span>
                        </>
                      ) : null}
                      {'.'}
                    </li>
                    <li>
                      {"The token's new representation is the weighted average of all the values."}
                      <DcLink className="cite" href="#ref-1">
                        {'1'}
                      </DcLink>
                      <DcLink className="cite" href="#ref-3">
                        {'3'}
                      </DcLink>
                    </li>
                  </ol>
                </section>
                <section className="sec" id="figures" aria-labelledby="figures-h">
                  <h2 className="sec-h" id="figures-h">
                    {'Explore the figures'}
                  </h2>
                  <p className="sec-sub">{'Three interactive figures. Figures 1 and 2 share one selection.'}</p>
                  <p className="prose">
                    {'A Transformer learns many '}
                    <button
                      type="button"
                      className="term"
                      aria-expanded={v.tm?.head?.exp}
                      onClick={v.tm?.head?.toggle}
                      onKeyDown={v.tm?.head?.key}
                    >
                      {'attention heads'}
                    </button>
                    {v.tm?.head?.open ? (
                      <>
                        <span className="tpop" role="note">
                          <span className="tpop-t">{v.tm?.head?.name}</span>
                          {v.tm?.head?.def}
                          <button type="button" className="tpop-x" aria-label="Close definition" onClick={v.closeTerm}>
                            <svg viewBox="0 0 14 14" fill="none" aria-hidden="true">
                              <path d="M3 3l8 8M11 3l-8 8" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round"></path>
                            </svg>
                          </button>
                        </span>
                      </>
                    ) : null}
                    {' side by side, each with its own pattern of weights. Figure 1 shows two of them for a single sentence.'}
                  </p>
                  <figure className="fig" id="fig1" style={{ marginLeft: '0', marginRight: '0' }} role="group" aria-labelledby="fig1-t">
                    <h3 className="fig-t" id="fig1-t">
                      {'Figure 1. Self-attention as weighted links'}
                    </h3>
                    <p className="fig-i">
                      {'Select a word on the left to see where it directs its attention, then switch heads and compare.'}
                    </p>
                    <div className="fig-body">
                      <div className="fig-ctl">
                        <span className="segs" role="group" aria-label="Attention head">
                          <button type="button" className="seg" aria-pressed={v.headAPressed} onClick={v.pickHeadA}>
                            {'Head A: long-range'}
                          </button>{' '}
                          <button type="button" className="seg" aria-pressed={v.headBPressed} onClick={v.pickHeadB}>
                            {'Head B: local'}
                          </button>
                        </span>
                      </div>
                      <div className="f1wrap">
                        <div style={{ flex: '1 1 380px', maxWidth: '500px', minWidth: '0' }}>
                          <div className="f1axis">
                            <span>{'Query'}</span>
                            <span>{'Key and weight'}</span>
                          </div>
                          <div className="f1" onMouseLeave={v.clearHover}>
                            <div className="f1col">
                              {(v.f1Queries || []).map((t, $index) => (
                                <React.Fragment key={$index}>
                                  <button
                                    type="button"
                                    className="qtok"
                                    aria-pressed={t?.pressed}
                                    aria-label={t?.aria}
                                    onMouseEnter={t?.enter}
                                    onFocus={t?.enter}
                                    onClick={t?.pick}
                                  >
                                    {t?.word}
                                  </button>
                                </React.Fragment>
                              ))}
                            </div>
                            <div className="f1mid" aria-hidden="true">
                              {(v.f1Links || []).map((l, $index) => (
                                <React.Fragment key={$index}>
                                  <svg className="f1svg" viewBox="0 0 100 336" preserveAspectRatio="none">
                                    <path d={l?.d} style={css(`stroke:var(--accent-strong);stroke-width:${l?.sw};opacity:${l?.op}`)}></path>
                                  </svg>
                                </React.Fragment>
                              ))}{' '}
                              <svg className="f1svg" viewBox="0 0 100 336" preserveAspectRatio="none">
                                <path
                                  d={v.f1Hl?.d}
                                  style={css(`stroke:var(--text);stroke-width:${v.f1Hl?.sw};opacity:${v.f1Hl?.op}`)}
                                ></path>
                              </svg>
                            </div>
                            <div className="f1col">
                              {(v.f1Keys || []).map((k, $index) => (
                                <React.Fragment key={$index}>
                                  <div className="ktok" onMouseEnter={k?.enter} style={css(`color:${k?.fg};font-weight:${k?.fw}`)}>
                                    {k?.word}
                                    <span className="kw">{k?.w}</span>
                                  </div>
                                </React.Fragment>
                              ))}
                            </div>
                          </div>
                        </div>
                        <div className="f1side">
                          <p className="readout" aria-live="polite">
                            {'Under '}
                            <span className="rk">{v.f1?.head}</span>
                            {', '}
                            <span className="q">{v.f1?.q}</span>
                            {' places '}
                            <span className="num">{v.f1?.w1}</span>
                            {' of its attention on '}
                            <span className="rk">{v.f1?.k1}</span>
                            {', '}
                            <span className="num">{v.f1?.w2}</span>
                            {' on '}
                            <span className="rk">{v.f1?.k2}</span>
                            {' and '}
                            <span className="num">{v.f1?.w3}</span>
                            {' on '}
                            <span className="rk">{v.f1?.k3}</span>
                            {'.'}
                          </p>
                          <p className="readout" style={{ fontSize: '13.5px', color: 'var(--text3)' }}>
                            {v.f1?.focus}
                          </p>
                          <div className="legend">
                            <div className="legend-h">{'Link weight'}</div>
                            <div className="legend-r">
                              <svg viewBox="0 0 46 10" aria-hidden="true">
                                <line x1="2" y1="5" x2="44" y2="5" style={{ strokeWidth: '1.05px', opacity: '.25' }}></line>
                              </svg>
                              <span>{'0.05'}</span>
                            </div>
                            <div className="legend-r">
                              <svg viewBox="0 0 46 10" aria-hidden="true">
                                <line x1="2" y1="5" x2="44" y2="5" style={{ strokeWidth: '2.4px', opacity: '.51' }}></line>
                              </svg>
                              <span>{'0.20'}</span>
                            </div>
                            <div className="legend-r">
                              <svg viewBox="0 0 46 10" aria-hidden="true">
                                <line x1="2" y1="5" x2="44" y2="5" style={{ strokeWidth: '4.65px', opacity: '1' }}></line>
                              </svg>
                              <span>{'0.45 or more'}</span>
                            </div>
                            <div className="legend-r">
                              <span
                                style={{
                                  display: 'inline-block',
                                  width: '7px',
                                  height: '7px',
                                  borderRadius: '999px',
                                  background: 'var(--accent)',
                                  margin: '0 19px 0 20px',
                                }}
                              ></span>
                              <span>{'selected query'}</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                    <p className="fig-ill">{'Illustrative weights, authored for teaching; not taken from a trained model.'}</p>
                    <details className="disc">
                      <summary>{'More detail'}</summary>
                      <div className="disc-b">
                        <p>
                          {
                            'Each query token on the left distributes a total weight of 1.00 across every key token on the right, itself included. Link width and opacity are proportional to weight; the number beside each key is its exact weight. Point to a word to inspect it, or select it to keep it in view.'
                          }
                        </p>
                        <p>
                          {'Under Head A, the pronoun '}
                          <i>{'it'}</i>
                          {' sends its strongest link to '}
                          <i>{'animal'}</i>
                          {
                            ', which is the reading a fluent speaker would give. Under Head B, the same word attends almost entirely to the word before it. Both patterns are useful.'
                          }
                        </p>
                      </div>
                    </details>
                  </figure>
                  <p className="prose">
                    {'Drawn for every word at once, the same weights form a square matrix. Row '}
                    <i>{'i'}</i>
                    {' holds the weights that token '}
                    <i>{'i'}</i>
                    {' assigns to each token in the sentence, so every row sums to one.'}
                  </p>
                  <figure className="fig" id="fig2" style={{ marginLeft: '0', marginRight: '0' }} role="group" aria-labelledby="fig2-t">
                    <h3 className="fig-t" id="fig2-t">
                      {'Figure 2. The attention matrix'}
                    </h3>
                    <p className="fig-i">
                      {
                        'Point to or tap any cell to read its weight; the matching link is drawn in Figure 1. Select a row label to change the query.'
                      }
                    </p>
                    <div className="fig-body">
                      <div className="fig-ctl">
                        <span className="hm-ax">
                          {'Showing '}
                          {v.f1?.head}
                          {' · rows: query · columns: key'}
                        </span>
                      </div>
                      <div className="hmwrap">
                        <div className="hm" onMouseLeave={v.clearHover}>
                          {(v.hmCols || []).map((c, $index) => (
                            <React.Fragment key={$index}>
                              <span className="hm-col" style={css(`grid-column:${c?.gc};color:${c?.fg}`)}>
                                {c?.word}
                              </span>
                            </React.Fragment>
                          ))}{' '}
                          {(v.hmRows || []).map((r, $index) => (
                            <React.Fragment key={$index}>
                              <button
                                type="button"
                                className="hm-row"
                                aria-pressed={r?.pressed}
                                aria-label={r?.aria}
                                onFocus={r?.enter}
                                onMouseEnter={r?.enter}
                                onClick={r?.pick}
                                style={css(`grid-row:${r?.gr}`)}
                              >
                                {r?.word}
                              </button>
                            </React.Fragment>
                          ))}{' '}
                          {(v.hmCells || []).map((c, $index) => (
                            <React.Fragment key={$index}>
                              <div
                                className="hm-c"
                                onMouseEnter={c?.enter}
                                onClick={c?.enter}
                                style={css(`grid-row:${c?.gr};grid-column:${c?.gc};outline:${c?.ol};z-index:${c?.z}`)}
                              >
                                <span className="hm-f" style={css(`opacity:${c?.op}`)}></span>
                              </div>
                            </React.Fragment>
                          ))}
                          <div className="hm-band" style={css(`grid-row:${v.hmBandRow}`)} aria-hidden="true"></div>
                        </div>
                        <div className="f1side">
                          <p className="readout" aria-live="polite">
                            {v.hm?.lead} <span className="num">{v.hm?.val}</span>
                            {v.hm?.tail}
                          </p>
                          <div className="legend">
                            <div className="legend-h">{'Weight'}</div>
                            <div className="sw" aria-hidden="true">
                              <span>
                                <i className="z"></i>
                                {'0'}
                              </span>{' '}
                              <span>
                                <i style={{ opacity: '.22' }}></i>
                                {'0.10'}
                              </span>{' '}
                              <span>
                                <i style={{ opacity: '.44' }}></i>
                                {'0.20'}
                              </span>{' '}
                              <span>
                                <i style={{ opacity: '.67' }}></i>
                                {'0.30'}
                              </span>{' '}
                              <span>
                                <i style={{ opacity: '1' }}></i>
                                {'0.45+'}
                              </span>
                            </div>
                            <div className="legend-r" style={{ marginTop: '6px' }}>
                              <span
                                style={{
                                  display: 'inline-block',
                                  width: '26px',
                                  height: '12px',
                                  border: '1.5px solid var(--text)',
                                  borderRadius: '3px',
                                }}
                              ></span>
                              <span>{'row selected in Figure 1'}</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                    <p className="fig-ill">{'Illustrative weights, authored for teaching; not taken from a trained model.'}</p>
                    <details className="disc">
                      <summary>{'More detail'}</summary>
                      <div className="disc-b">
                        <p>
                          {
                            'Rows are queries and columns are keys; shading is proportional to weight on a linear scale that saturates at 0.45. The outlined row is the query selected in Figure 1, and the two figures share one selection.'
                          }
                        </p>
                        <p>
                          {
                            'Read down a column instead and the matrix shows which tokens the sentence consults most. Compare the two heads: Head A scatters weight across distant words, while Head B concentrates it in a narrow band just left of the diagonal, the signature of a head that tracks word order.'
                          }
                        </p>
                      </div>
                    </details>
                  </figure>
                  <p className="prose">
                    {'Where do the weights come from? Scaled dot-product attention computes them in four steps.'}
                    <DcLink className="cite" href="#ref-1">
                      {'1'}
                    </DcLink>
                    {
                      ' Figure 3 runs them on the Laboratory 3 example: one query, three keys and three values, each with two components, so d'
                    }
                    <sub>{'k'}</sub>
                    {' = 2. The scores are divided by '}
                    <button
                      type="button"
                      className="term"
                      aria-expanded={v.tm?.scale?.exp}
                      onClick={v.tm?.scale?.toggle}
                      onKeyDown={v.tm?.scale?.key}
                    >
                      {'√d'}
                      <sub>{'k'}</sub>
                    </button>
                    {v.tm?.scale?.open ? (
                      <>
                        <span className="tpop" role="note">
                          <span className="tpop-t">{v.tm?.scale?.name}</span>
                          {v.tm?.scale?.def}
                          <button type="button" className="tpop-x" aria-label="Close definition" onClick={v.closeTerm}>
                            <svg viewBox="0 0 14 14" fill="none" aria-hidden="true">
                              <path d="M3 3l8 8M11 3l-8 8" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round"></path>
                            </svg>
                          </button>
                        </span>
                      </>
                    ) : null}
                    {' before the softmax.'}
                  </p>
                  <figure className="fig" id="fig3" style={{ marginLeft: '0', marginRight: '0' }} role="group" aria-labelledby="fig3-t">
                    <h3 className="fig-t" id="fig3-t">
                      {'Figure 3. Scaled dot-product attention, step by step'}
                    </h3>
                    <p className="fig-i">
                      {'Step through the four stages, then switch √d'}
                      <sub>{'k'}</sub>
                      {' scaling off and compare the weights.'}
                    </p>
                    <div className="fig-body">
                      <div className="fig-ctl">
                        <p className="eq" aria-label="Attention of Q, K and V equals softmax of Q K transpose over root d k, times V">
                          {'Attention('}
                          <i>{'Q'}</i>
                          {', '}
                          <i>{'K'}</i>
                          {', '}
                          <i>{'V'}</i>
                          {') = '}
                          <span style={css(`color:${v.eq?.sm};border-bottom-color:${v.eq?.smB}`)}>{'softmax('}</span>
                          <span style={css(`color:${v.eq?.qk};border-bottom-color:${v.eq?.qkB}`)}>
                            <i>{'QK'}</i>
                            <sup>{'T'}</sup>
                          </span>
                          <span style={css(`color:${v.eq?.sc};border-bottom-color:${v.eq?.scB}`)}>
                            {' / √'}
                            <i>{'d'}</i>
                            <sub>{'k'}</sub>
                          </span>
                          <span style={css(`color:${v.eq?.sm};border-bottom-color:${v.eq?.smB}`)}>{')'}</span>{' '}
                          <span style={css(`color:${v.eq?.v};border-bottom-color:${v.eq?.vB}`)}>
                            <i>{'V'}</i>
                          </span>
                        </p>
                        <button type="button" className="seg" aria-pressed={v.scalePressed} onClick={v.toggleScale}>
                          <span>
                            {'√d'}
                            <sub>{'k'}</sub>
                            {' scaling: '}
                            {v.scaleLabel}
                          </span>
                        </button>
                      </div>
                      <div className="f3ctl">
                        <span className="segs" role="group" aria-label="Stage">
                          {(v.f3Stages || []).map((s, $index) => (
                            <React.Fragment key={$index}>
                              <button type="button" className="seg" aria-pressed={s?.pressed} onClick={s?.pick}>
                                {s?.label}
                              </button>
                            </React.Fragment>
                          ))}
                        </span>{' '}
                        <span style={{ display: 'inline-flex', flexWrap: 'wrap', gap: '6px' }}>
                          <button type="button" className="ghostbtn" onClick={v.stepBack} aria-label="Previous stage">
                            {'← Back'}
                          </button>
                          <button type="button" className="ghostbtn" onClick={v.stepFwd} aria-label="Next stage">
                            {'Next →'}
                          </button>
                          <button type="button" className="ghostbtn" onClick={v.togglePlay}>
                            {v.playLabel}
                          </button>
                          <button
                            type="button"
                            className="ghostbtn"
                            onClick={v.reset}
                            style={{ background: 'none', borderColor: 'transparent' }}
                          >
                            {'Reset'}
                          </button>
                        </span>
                      </div>
                      <div className="f3wrap">
                        <div className="f3tab" onMouseLeave={v.clearF3}>
                          <div className="f3g f3hd">
                            <span></span>
                            <span>
                              {'k'}
                              <sub>{'i'}</sub>
                            </span>
                            <span>
                              {'v'}
                              <sub>{'i'}</sub>
                            </span>
                            <span>{v.f3?.qty}</span>
                            <span style={{ textAlign: 'right' }}>{'value'}</span>
                          </div>
                          {(v.f3Rows || []).map((r, $index) => (
                            <React.Fragment key={$index}>
                              <button
                                type="button"
                                className="f3g f3r"
                                aria-label={r?.aria}
                                onMouseEnter={r?.enter}
                                onFocus={r?.enter}
                                onClick={r?.enter}
                                style={css(`background:${r?.bg}`)}
                              >
                                <span className="kl">{r?.name}</span>
                                <span>{r?.k}</span>
                                <span>{r?.v}</span>
                                <span className="f3track">
                                  <span className="f3ghost" style={css(`left:${r?.ghost};opacity:${r?.ghostOp}`)}></span>
                                  <span className="f3bar" style={css(`width:${r?.width}`)}></span>
                                </span>
                                <span className="f3val">{r?.val}</span>
                              </button>
                            </React.Fragment>
                          ))}
                          <div className="f3g">
                            <span></span>
                            <span></span>
                            <span></span>
                            <span className="f3ticks">
                              <span style={{ left: '0%' }}>{'0'}</span>
                              <span style={{ left: '41.67%' }}>{'0.5'}</span>
                              <span style={{ left: '83.33%' }}>{'1.0'}</span>
                            </span>
                            <span></span>
                          </div>
                          <p className="f3note" aria-live="polite">
                            <span className="rk">{v.f3?.step}</span> {v.f3?.note}
                            <br />
                            <span className="mono">{v.f3?.calc}</span>
                          </p>
                        </div>
                        <div className="f3plotw">
                          <div className="fig-n">{'Value vectors and output'}</div>
                          <div className="f3plot" aria-hidden="true">
                            <svg viewBox="0 0 100 100" preserveAspectRatio="none">
                              <line x1="41.67" y1="0" x2="41.67" y2="100"></line>
                              <line x1="83.33" y1="0" x2="83.33" y2="100"></line>
                              <line x1="0" y1="58.33" x2="100" y2="58.33"></line>
                              <line x1="0" y1="16.67" x2="100" y2="16.67"></line>
                            </svg>{' '}
                            {(v.f3Pts || []).map((p, $index) => (
                              <React.Fragment key={$index}>
                                <span
                                  className="f3pt"
                                  style={css(`left:${p?.x};top:${p?.y};width:${p?.d};height:${p?.d};opacity:${p?.op}`)}
                                ></span>{' '}
                                <span className="f3pl" style={css(`left:${p?.lx};top:${p?.ly}`)}>
                                  {p?.name}
                                </span>
                              </React.Fragment>
                            ))}{' '}
                            <span className="f3out" style={css(`left:${v.f3?.ox};top:${v.f3?.oy};opacity:${v.f3?.oOp}`)}></span>{' '}
                            <span
                              className="f3pl"
                              style={css(`left:${v.f3?.olx};top:${v.f3?.oly};opacity:${v.f3?.oOp};font-style:normal;color:var(--text)`)}
                            >
                              {'output'}
                            </span>{' '}
                            <span className="f3tk" style={{ left: '41.67%', bottom: '-24px', transform: 'translateX(-50%)' }}>
                              {'1'}
                            </span>{' '}
                            <span className="f3tk" style={{ left: '83.33%', bottom: '-24px', transform: 'translateX(-50%)' }}>
                              {'2'}
                            </span>{' '}
                            <span className="f3tk" style={{ left: '-22px', top: '58.33%', transform: 'translateY(-50%)' }}>
                              {'1'}
                            </span>{' '}
                            <span className="f3tk" style={{ left: '-22px', top: '16.67%', transform: 'translateY(-50%)' }}>
                              {'2'}
                            </span>
                          </div>
                          <div className="legend" style={{ maxWidth: '240px' }}>
                            <div className="legend-r">
                              <span
                                style={{
                                  display: 'inline-block',
                                  width: '10px',
                                  height: '10px',
                                  borderRadius: '999px',
                                  background: 'var(--text2)',
                                  margin: '0 4px',
                                }}
                              ></span>
                              <span>{'value vector, area grows with its weight'}</span>
                            </div>
                            <div className="legend-r">
                              <span
                                style={{
                                  display: 'inline-block',
                                  width: '9px',
                                  height: '9px',
                                  transform: 'rotate(45deg)',
                                  background: 'var(--accent)',
                                  border: '1.5px solid var(--text)',
                                  margin: '0 4px',
                                }}
                              ></span>
                              <span>{'output, the weighted average'}</span>
                            </div>
                            <div className="legend-r">
                              <span
                                style={{
                                  display: 'inline-block',
                                  width: '0',
                                  height: '12px',
                                  borderLeft: '1.5px dashed var(--text3)',
                                  margin: '0 8px',
                                }}
                              ></span>
                              <span>{'value at the previous stage'}</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                    <p className="fig-ill">
                      {'Values are computed exactly from the vectors shown, with query '}
                      <i>{'q'}</i>
                      {' = (1, 0).'}
                    </p>
                    <details className="disc">
                      <summary>{'More detail'}</summary>
                      <div className="disc-b">
                        <p>
                          {
                            'Each row follows one key through the computation; bars share a linear axis from 0 to 1.2 and the dashed mark records the previous stage. Point to a row to see its arithmetic. The plot places the three value vectors in the plane; once the weights are known, the output is their weighted average.'
                          }
                        </p>
                        <p>
                          {
                            'With scaling switched off, the weights move further apart. You will measure this effect at scale in Part C of the laboratory.'
                          }
                        </p>
                      </div>
                    </details>
                  </figure>
                  <h3 className="h3" id="soft">
                    {'Soft weights and parallel computation'}
                  </h3>
                  <p className="prose">
                    {'Two consequences follow from Figure 3. First, the output is a '}
                    <em>{'soft'}</em>
                    {
                      ' mixture: every value contributes in proportion to its weight, so the whole computation can be differentiated and trained by gradient descent. '
                    }
                    <button
                      type="button"
                      className="term"
                      aria-expanded={v.tm?.hardsoft?.exp}
                      onClick={v.tm?.hardsoft?.toggle}
                      onKeyDown={v.tm?.hardsoft?.key}
                    >
                      {'Hard attention'}
                    </button>
                    {v.tm?.hardsoft?.open ? (
                      <>
                        <span className="tpop" role="note">
                          <span className="tpop-t">{v.tm?.hardsoft?.name}</span>
                          {v.tm?.hardsoft?.def}
                          <button type="button" className="tpop-x" aria-label="Close definition" onClick={v.closeTerm}>
                            <svg viewBox="0 0 14 14" fill="none" aria-hidden="true">
                              <path d="M3 3l8 8M11 3l-8 8" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round"></path>
                            </svg>
                          </button>
                        </span>
                      </>
                    ) : null}
                    {' would instead select only the highest-scoring key, an operation with no useful gradient.'}
                  </p>
                  <p className="prose">
                    {
                      'Second, no step depends on the result for the previous token, so the computation for every token in a sentence can run at once. That parallelism, rather than any single accuracy gain, is what allowed the Transformer to replace recurrent models in practice.'
                    }
                    <DcLink className="cite" href="#ref-1">
                      {'1'}
                    </DcLink>
                  </p>
                </section>
                <section className="sec" id="takeaway" aria-labelledby="takeaway-h">
                  <h2 className="sec-h is-label" id="takeaway-h">
                    {'Key takeaway'}
                  </h2>
                  <div className="take">
                    <p>
                      {'Attention replaces the fixed-size hidden state of a recurrent network with '}
                      <mark className="mk">{'direct, weighted connections between every pair of tokens'}</mark>
                      {'. Each weight comes from comparing a query with a key, scaling by √d'}
                      <sub>{'k'}</sub>
                      {
                        ' and applying softmax, and the output is the weighted average of the values. Because that mixture is soft and no token waits for the one before it, attention can be trained by gradient descent and computed in parallel.'
                      }
                    </p>
                  </div>
                  <details className="disc">
                    <summary>{'Key terms'}</summary>
                    <div className="disc-b">
                      <dl className="terms">
                        {(v.termList || []).map((t, $index) => (
                          <React.Fragment key={$index}>
                            <div>
                              <dt>{t?.name}</dt>
                              <dd>{t?.def}</dd>
                            </div>
                          </React.Fragment>
                        ))}
                      </dl>
                    </div>
                  </details>
                </section>
                <section className="sec" id="lab" aria-labelledby="lab-h">
                  <h2 className="sec-h" id="lab-h">
                    {'Laboratory 3: Scaled dot-product attention'}
                  </h2>
                  <p className="labmeta">
                    <span>{'Part 1'}</span>
                    <span className="dot" aria-hidden="true"></span>
                    <span className="lm-status">{'Not started'}</span>
                    <span className="dot" aria-hidden="true"></span>
                    <span>{'Due 17 October 2026, 11:59 PM PHT'}</span>
                    <span className="dot" aria-hidden="true"></span>
                    <span>{'10% of final grade'}</span>
                  </p>
                  <p className="labobj">
                    {'Compute the attention output by hand, then '}
                    <mark className="mk">{'implement and verify it in PyTorch'}</mark>
                    {'.'}
                  </p>
                  <div className="lgrp">
                    <h3 className="h3">{'What to do'}</h3>
                    <ol className="lsteps">
                      <li>
                        <span className="runin">{'Set up.'}</span>
                        {
                          ' Open the Laboratory 3 notebook from the course workspace; the required libraries are pre-installed. Select the device with '
                        }
                        <code className="icode">{'device = "cuda" if torch.cuda.is_available() else "cpu"'}</code>
                        {'.'}
                      </li>
                      <li>
                        <span className="runin">{'Part A: Concepts.'}</span>
                        {
                          ' In a short written answer, explain why soft attention can be trained by gradient descent and hard attention cannot.'
                        }
                      </li>
                      <li>
                        <span className="runin">{'Part B: By hand, then in code.'}</span>
                        {
                          ' Using the query, keys and values in Figure 3, compute the scores, apply softmax and form the weighted sum. Confirm each intermediate result in PyTorch.'
                        }
                      </li>
                      <li>
                        <span className="runin">{'Part C: Experiments.'}</span>
                        <ul>
                          <li>
                            {'Scaling ablation: remove the √d'}
                            <sub>{'k'}</sub>
                            {' term, increase d'}
                            <sub>{'k'}</sub>
                            {' to 64, and report how the weights change.'}
                          </li>
                          <li>{'Hard and soft attention: compare argmax selection with the softmax blend on the same inputs.'}</li>
                        </ul>
                      </li>
                    </ol>
                  </div>
                  <div className="codebox">
                    <div className="codebar">
                      <span>{'attention_lab.py'}</span>
                      <span>{'Python 3 · PyTorch'}</span>
                    </div>
                    <pre className="code">
                      <span className="c">{'# Laboratory 3, Part 1: scaled dot-product attention'}</span>{' '}
                      <span className="k">{'import'}</span>
                      {' torch, torch.nn.functional '}
                      <span className="k">{'as'}</span>
                      {' F\n\nq = torch.'}
                      <span className="f">{'tensor'}</span>
                      {'([['}
                      <span className="n">{'1.'}</span>
                      {', '}
                      <span className="n">{'0.'}</span>
                      {']])                       '}
                      <span className="c">{'# 1 query'}</span>
                      {'\nK = torch.'}
                      <span className="f">{'tensor'}</span>
                      {'([['}
                      <span className="n">{'1.'}</span>
                      {', '}
                      <span className="n">{'0.'}</span>
                      {'], ['}
                      <span className="n">{'0.'}</span>
                      {', '}
                      <span className="n">{'1.'}</span>
                      {'], ['}
                      <span className="n">{'1.'}</span>
                      {', '}
                      <span className="n">{'1.'}</span>
                      {']])   '}
                      <span className="c">{'# 3 keys'}</span>
                      {'\nV = torch.'}
                      <span className="f">{'tensor'}</span>
                      {'([['}
                      <span className="n">{'1.'}</span>
                      {', '}
                      <span className="n">{'0.'}</span>
                      {'], ['}
                      <span className="n">{'0.'}</span>
                      {', '}
                      <span className="n">{'1.'}</span>
                      {'], ['}
                      <span className="n">{'2.'}</span>
                      {', '}
                      <span className="n">{'2.'}</span>
                      {']])   '}
                      <span className="c">{'# 3 values'}</span>
                      {'\n\nscores = q @ K.T / ('}
                      <span className="n">{'2'}</span>
                      {' ** '}
                      <span className="n">{'0.5'}</span>
                      {')\nw = F.'}
                      <span className="f">{'softmax'}</span>
                      {'(scores, dim='}
                      <span className="n">{'-1'}</span>
                      {')\nout = w @ V\n'}
                      <span className="k">{'print'}</span>
                      {'(scores, w, out)\n\n'}
                      <span className="c">{'# reusable version for Part C'}</span> <span className="k">{'def'}</span>{' '}
                      <span className="f">{'scaled_dot_product_attention'}</span>
                      {'(query, key, value):\n    d_k = query.shape['}
                      <span className="n">{'-1'}</span>
                      {']\n    scores = torch.'}
                      <span className="f">{'matmul'}</span>
                      {'(query, key.'}
                      <span className="f">{'transpose'}</span>
                      {'('}
                      <span className="n">{'-2'}</span>
                      {', '}
                      <span className="n">{'-1'}</span>
                      {')) / (d_k ** '}
                      <span className="n">{'0.5'}</span>
                      {')\n    weights = F.'}
                      <span className="f">{'softmax'}</span>
                      {'(scores, dim='}
                      <span className="n">{'-1'}</span>
                      {')\n    '}
                      <span className="k">{'return'}</span>
                      {' torch.'}
                      <span className="f">{'matmul'}</span>
                      {'(weights, value), weights'}
                    </pre>
                    <div className="outbox">
                      <p className="outlab">{'Expected output'}</p>
                      <pre>
                        {
                          'scores:  tensor([[0.7071, 0.0000, 0.7071]])\nweights: tensor([[0.4001, 0.1998, 0.4001]])\noutput:  tensor([[1.2003, 1.0000]])'
                        }
                      </pre>
                      <p className="outnote">
                        {'Shown for reference. Run the notebook to reproduce these values before continuing to Part C.'}
                      </p>
                    </div>
                  </div>
                  <div className="lgrp">
                    <h3 className="h3">{'What to submit'}</h3>
                    <p className="lp">
                      {'The notebook with Parts A to C executed in order. As a checkpoint, your weights should be close to '}
                      <span className="icode">{'[0.401, 0.198, 0.401]'}</span>
                      {' and the output close to '}
                      <span className="icode">{'[1.203, 1.000]'}</span>
                      {'.'}
                    </p>
                    <h3 className="h3">{'How it is assessed'}</h3>
                    <p className="lp">
                      {'Marked against the published Laboratory 3 rubric; the report counts for 10% of the final grade.'}
                    </p>
                  </div>
                  <details className="disc">
                    <summary>{'Optional extension and setup notes'}</summary>
                    <div className="disc-b">
                      <p>
                        <span className="runin">{'Optional extension.'}</span>
                        {' Introduce a temperature parameter and plot its effect on the weight distribution.'}
                      </p>
                      <p>
                        <span className="runin">{'Go further.'}</span>
                        {' The first sections of '}
                        <DcLink className="textlink" href="https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html" target="_blank" rel="noopener noreferrer">
                          {'UvA Deep Learning Tutorial 6: Transformers and Multi-Head Attention'}
                        </DcLink>
                        {' ("What is Attention?" and "Scaled Dot Product Attention") rebuild this laboratory in PyTorch, with a worked numeric example and the same softmax(QKᵀ/√d_k)V formula. Tutorial by Phillip Lippe, University of Amsterdam (MIT licence). It downloads data and checkpoints, so run it with internet access, not on the offline Spark. Week 8 continues with the rest of that tutorial.'}
                      </p>
                      <p>{'Laboratory sessions run on NVIDIA-accelerated infrastructure provided by the programme.'}</p>
                    </div>
                  </details>
                  <div className="actions">
                    {v.nbColab ? (
                      <a className="solidbtn" href={v.nbColab} target="_blank" rel="noopener noreferrer">
                        {'Open Laboratory 3 in Colab'}
                      </a>
                    ) : null}{' '}
                    {v.nbDownload ? (
                      <a className={v.nbColab ? 'textlink' : 'solidbtn'} href={v.nbDownload} download>
                        {'Download notebook (.ipynb)'}
                      </a>
                    ) : null}{' '}
                    <DcLink className="textlink" href="#">
                      {'View marking rubric'}
                    </DcLink>
                  </div>
                  <p className="note">{'Submitted work must be your own. Disclose any AI assistance as set out in the course policy.'}</p>
                </section>
                <section className="sec" id="check" aria-labelledby="check-h">
                  <h2 className="sec-h" id="check-h">
                    {'Check for understanding'}
                  </h2>
                  <p className="sec-sub">{'Formative · Not graded · Unlimited attempts'}</p>
                  <div className="quiz">
                    <p className="qmeta">
                      {'Question '}
                      {v.qNum}
                      {' of '}
                      {v.qTotal}
                    </p>
                    <p className="qtext" id="qtext">
                      {v.qText}
                    </p>
                    <div role="group" aria-labelledby="qtext">
                      {(v.opts || []).map((o, $index) => (
                        <React.Fragment key={$index}>
                          <button type="button" className={o?.cls} aria-pressed={o?.pressed} aria-disabled={o?.locked} onClick={o?.pick}>
                            <span className="rad" aria-hidden="true"></span>
                            <span className="qt">{o?.text}</span>
                            <span className="qm">{o?.mark}</span>
                          </button>
                        </React.Fragment>
                      ))}
                    </div>
                    {v.submitted ? (
                      <>
                        <div className="qfb" role="status">
                          {v.fbRight ? (
                            <>
                              <p className="qfb-h">
                                <svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
                                  <path
                                    d="M3 8.4l3.1 3L13 4.6"
                                    stroke="currentColor"
                                    strokeWidth="2"
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                  ></path>
                                </svg>
                                {'Correct.'}
                              </p>
                            </>
                          ) : null}{' '}
                          {v.fbWrong ? (
                            <>
                              <p className="qfb-h">
                                <svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
                                  <path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" strokeWidth="2" strokeLinecap="round"></path>
                                </svg>
                                {'Not quite.'}
                              </p>
                            </>
                          ) : null}
                          <p>{v.correctText}</p>
                          {v.hasRevisit ? (
                            <>
                              <p className="rv">
                                {'Revisit: '}
                                <DcLink href={v.revisit?.href}>{v.revisit?.label}</DcLink>
                              </p>
                            </>
                          ) : null}
                        </div>
                      </>
                    ) : null}
                    <div className="qact">
                      {v.notSubmitted ? (
                        <>
                          <button type="button" className="solidbtn" onClick={v.submit} disabled={v.noSelection}>
                            {'Check answer'}
                          </button>
                        </>
                      ) : null}{' '}
                      {v.submitted ? (
                        <>
                          <button type="button" className="solidbtn" onClick={v.next}>
                            {'Next question →'}
                          </button>
                        </>
                      ) : null}{' '}
                      {v.noSelection ? (
                        <>
                          <span className="qhint">{'Choose an answer to continue.'}</span>
                        </>
                      ) : null}
                    </div>
                  </div>
                </section>
                <section className="sec refsec" id="references" aria-labelledby="references-h">
                  <h2 className="sec-h" id="references-h">
                    {'References'}
                  </h2>
                  <ol className="refs">
                    <li id="ref-1">
                      <span className="rn">{'[1]'}</span>
                      <span>
                        {
                          'Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). '
                        }
                        <i>{'Attention is all you need.'}</i>
                        {' In Advances in Neural Information Processing Systems 30 (NeurIPS 2017), pp. 5998–6008. '}
                        <DcLink href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noopener">
                          {'arXiv:1706.03762'}
                        </DcLink>
                      </span>
                    </li>
                    <li id="ref-2">
                      <span className="rn">{'[2]'}</span>
                      <span>
                        {'Bahdanau, D., Cho, K., & Bengio, Y. (2015). '}
                        <i>{'Neural machine translation by jointly learning to align and translate.'}</i>
                        {' In Proceedings of the 3rd International Conference on Learning Representations (ICLR 2015). '}
                        <DcLink href="https://arxiv.org/abs/1409.0473" target="_blank" rel="noopener">
                          {'arXiv:1409.0473'}
                        </DcLink>
                      </span>
                    </li>
                    <li id="ref-3">
                      <span className="rn">{'[3]'}</span>
                      <span>
                        {'Olah, C., & Carter, S. (2016). '}
                        <i>{'Attention and augmented recurrent neural networks.'}</i>
                        {' Distill, 1(9). '}
                        <DcLink href="https://doi.org/10.23915/distill.00001" target="_blank" rel="noopener">
                          {'doi:10.23915/distill.00001'}
                        </DcLink>
                      </span>
                    </li>
                  </ol>
                </section>
              </>
            )}
            <footer className="sitefoot">
              <div className="sf-top">
                <div className="sf-brand">
                  <svg viewBox="0 0 360 310" aria-hidden="true">
                    <polygon points="5,308 140,6 275,308 225,308 142,116 55,308" fill="var(--accent)"></polygon>
                    <circle cx="250" cy="73" r="40" fill="var(--accent)"></circle>
                    <rect x="308" y="13" width="47" height="295" fill="var(--accent)"></rect>
                  </svg>{' '}
                  <span className="sf-bt">
                    <b>{'AI Academy'}</b>
                    <span>{'A Supervaise and NVIDIA programme in generative artificial intelligence.'}</span>
                  </span>
                </div>
                <nav className="sf-cols" aria-label="Footer">
                  <div className="sf-col">
                    <h4>{'Programme'}</h4>
                    <DcLink href="CourseMap.dc.html">{'Course outline'}</DcLink> <DcLink href="Lesson.dc.html">{'Lesson workspace'}</DcLink>{' '}
                    <DcLink href="Capstone.dc.html">{'Capstone project'}</DcLink>
                  </div>
                  <div className="sf-col">
                    <h4>{'Assessment'}</h4>
                    <DcLink href="PracticeExam.dc.html">{'Examination preparation'}</DcLink>{' '}
                    <DcLink href="Certificates.dc.html">{'Credentials and records'}</DcLink>{' '}
                    <DcLink href="PromptWorkshop.dc.html">{'Practical workshops'}</DcLink>
                  </div>
                  <div className="sf-col">
                    <h4>{'Institution'}</h4>
                    <DcLink href="Educator.dc.html">{'Faculty console'}</DcLink>{' '}
                    <DcLink href="mailto:anthony.jose@supervaise.io">{'Programme office'}</DcLink>{' '}
                    <DcLink href="#">{'Academic integrity policy'}</DcLink>
                  </div>
                </nav>
              </div>
              <div className="sf-bot">
                <span>{'© 2026 Supervaise Inc. All rights reserved.'}</span>{' '}
                <span className="sf-soc">
                  <DcLink href="https://instagram.com/supervaise_ph" target="_blank" rel="noopener" aria-label="Supervaise on Instagram">
                    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                      <rect x="3.2" y="3.2" width="17.6" height="17.6" rx="5.2" stroke="currentColor" strokeWidth="1.7"></rect>
                      <circle cx="12" cy="12" r="4.1" stroke="currentColor" strokeWidth="1.7"></circle>
                      <circle cx="17.1" cy="6.9" r="1.25" fill="currentColor"></circle>
                    </svg>
                  </DcLink>{' '}
                  <DcLink href="https://linkedin.com/company/supervaise" target="_blank" rel="noopener" aria-label="Supervaise on LinkedIn">
                    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                      <rect x="3.2" y="3.2" width="17.6" height="17.6" rx="4.4" stroke="currentColor" strokeWidth="1.7"></rect>
                      <circle cx="7.8" cy="8.1" r="1.35" fill="currentColor"></circle>
                      <rect x="6.75" y="10.5" width="2.1" height="7" fill="currentColor"></rect>
                      <path
                        d="M11.2 17.5v-7h2.05v1.03a2.75 2.75 0 0 1 2.35-1.23c1.76 0 2.8 1.14 2.8 3.16v4.04h-2.1v-3.76c0-1.04-.45-1.65-1.37-1.65-.95 0-1.63.7-1.63 1.82v3.59z"
                        fill="currentColor"
                      ></path>
                    </svg>
                  </DcLink>{' '}
                  <DcLink href="https://supervaise.io" target="_blank" rel="noopener" aria-label="supervaise.io">
                    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                      <circle cx="12" cy="12" r="8.8" stroke="currentColor" strokeWidth="1.7"></circle>
                      <path d="M3.2 12h17.6" stroke="currentColor" strokeWidth="1.7"></path>
                      <path
                        d="M12 3.2c2.3 2.4 3.5 5.5 3.5 8.8s-1.2 6.4-3.5 8.8c-2.3-2.4-3.5-5.5-3.5-8.8S9.7 5.6 12 3.2z"
                        stroke="currentColor"
                        strokeWidth="1.7"
                      ></path>
                    </svg>
                  </DcLink>{' '}
                  <DcLink href="mailto:anthony.jose@supervaise.io" aria-label="Email the programme office">
                    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                      <rect x="2.8" y="5" width="18.4" height="14" rx="3.2" stroke="currentColor" strokeWidth="1.7"></rect>
                      <path
                        d="M4.2 7.6 12 13l7.8-5.4"
                        stroke="currentColor"
                        strokeWidth="1.7"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      ></path>
                    </svg>
                  </DcLink>
                </span>{' '}
                <span className="sf-power">
                  <span className="sf-pw-l">{'Powered by'}</span>
                  <img className="sf-nv nvl-d" src="/assets/nvidia-dark.png" alt="NVIDIA" />
                  <img className="sf-nv nvl-l" src="/assets/nvidia-light.png" alt="NVIDIA" />
                  <strong>{'DGX Spark'}</strong>
                </span>
              </div>
            </footer>
          </div>
        </div>
      </div>
    </>
  );
}

Component.template = template;
Component.css = pageCss + planCss;
Component.pageKey = 'lesson';
Component.persist = ['qIndex', 'selectedOpt', 'submitted'];

export default Component;
