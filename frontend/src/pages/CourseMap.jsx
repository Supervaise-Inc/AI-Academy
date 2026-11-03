// Generated from design-source/CourseMap.dc.html by tools/convert.mjs.
// Re-running the converter regenerates (and overwrites) this file, so edit design-source/ + tools/patches.mjs instead,
// or stop using the converter and edit this file directly.
/* eslint-disable */
import React from 'react';
import DCLogic from '../dc/DCLogic.jsx';
import DcLink from '../dc/DcLink.jsx';
import { css } from '../dc/css.js';
import pageCss from './CourseMap.css?inline';

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

const WEEKS = [
  {
    week: 1,
    term: 'Prelim',
    title: 'Course orientation & Intro to Generative AI',
    lab: 'GA Lab 1',
    exam: false,
    kc: 19,
    outs: [
      'Describe what Generative AI is and how it differs from classification or regression models.',
      'Place key events on the history of AI (Dartmouth 1956, Perceptron 1958, deep learning, GPT-3, ChatGPT) and say where the field is heading.',
    ],
  },
  {
    week: 2,
    term: 'Prelim',
    title: 'Machine Learning & Neural Networks',
    lab: 'DL Lab 1',
    exam: false,
    kc: 20,
    outs: [
      'Explain supervised learning: data, labels, model, loss, training, testing, generalization.',
      'Describe an artificial neuron and a layer: weights, bias, activation, and how layers stack.',
    ],
  },
  {
    week: 3,
    term: 'Prelim',
    title: 'Deep Learning & Backpropagation',
    lab: 'DL Lab 2B',
    exam: false,
    kc: 20,
    outs: [
      'Explain why depth helps and why deep networks are hard to train (vanishing and exploding gradients).',
      'Apply backpropagation: forward pass, backward pass, chain rule, and gradient flow through each layer.',
    ],
  },
  {
    week: 4,
    term: 'Prelim',
    title: 'Convolutional Neural Networks',
    lab: 'DL Lab 4B Part 1',
    exam: false,
    kc: 20,
    outs: [
      'Explain why fully connected layers are a poor fit for images and how convolution uses local connectivity and weight sharing.',
      'Compute a convolution (output size, values) and its input gradient by hand.',
    ],
  },
  { week: 5, term: 'Prelim', title: 'PRELIM EXAM', lab: null, exam: true, kc: 0, outs: ['Cumulative examination for the Prelim term.'] },
  {
    week: 6,
    term: 'Midterm',
    title: 'NLP: Tokens & Word Embeddings',
    lab: 'GA Lab 2',
    exam: false,
    kc: 60,
    outs: [
      'Explain what a language model is and trace the history from rules, to statistics (n-grams), to neural models.',
      'Explain tokenization, vocabularies, and why modern LLMs use sub-word tokens (BPE) rather than words or characters.',
    ],
  },
  {
    week: 7,
    term: 'Midterm',
    title: 'Language Models & Attention',
    lab: 'GA Lab 3 Part 1',
    exam: false,
    kc: 20,
    outs: [
      'Explain why sequence models (RNN, LSTM) struggle with long sequences, and why attention replaced them.',
      'Distinguish hard attention from soft attention.',
    ],
  },
  {
    week: 8,
    term: 'Midterm',
    title: 'The Transformer & Autoregressive Training',
    lab: 'GA Lab 3 Part 2',
    exam: false,
    kc: 40,
    outs: [
      'Name and explain the parts of a Transformer block: embeddings, positional encoding, multi-head self-attention, feed-forward, residuals, layer norm.',
      'Explain why positional encoding is needed and implement the sinusoidal version.',
    ],
  },
  {
    week: 9,
    term: 'Midterm',
    title: 'LLM Families, Scaling Laws & Prompt Engineering',
    lab: 'GA Lab 4',
    exam: false,
    kc: 60,
    outs: [
      'Compare encoder-only (BERT), decoder-only (GPT), encoder-decoder (T5) and Mixture-of-Experts models.',
      'Explain scaling laws, Chinchilla compute-optimal training, and emergent abilities.',
    ],
  },
  {
    week: 10,
    term: 'Midterm',
    title: 'MIDTERM EXAM',
    lab: null,
    exam: true,
    kc: 0,
    outs: ['Cumulative examination for the Midterm term.'],
  },
  {
    week: 11,
    term: 'Pre-final',
    title: 'Multimodal AI',
    lab: 'GA Lab 5',
    exam: false,
    kc: 60,
    outs: [
      'Explain what a multimodal model is and give three real applications.',
      'Explain how a Vision Transformer (ViT) turns an image into "tokens" and why position embeddings are needed.',
    ],
  },
  {
    week: 12,
    term: 'Pre-final',
    title: 'Image Generation & GANs',
    lab: 'GA demo 6.1',
    exam: false,
    kc: 25,
    outs: [
      'Explain the GAN idea: a generator and a discriminator trained against each other.',
      'Explain transpose convolution as learnable upsampling.',
    ],
  },
  {
    week: 13,
    term: 'Pre-final',
    title: 'Diffusion Models',
    lab: 'GA Lab 6',
    exam: false,
    kc: 50,
    outs: [
      'Describe the forward (add noise) and reverse (denoise) processes in diffusion.',
      'Explain why a UNet with time embedding is used as the denoiser.',
    ],
  },
  {
    week: 14,
    term: 'Pre-final',
    title: 'Training LLMs: Fine-tuning, RLHF, PEFT',
    lab: 'GA Lab 7',
    exam: false,
    kc: 60,
    outs: [
      'Distinguish pre-training, continued pre-training, and task-specific fine-tuning.',
      'Explain instruction tuning and RLHF, and why chat models need them.',
    ],
  },
  {
    week: 15,
    term: 'Pre-final',
    title: 'PRE-FINAL EXAM + Capstone proposal',
    lab: null,
    exam: true,
    kc: 0,
    outs: ['Cumulative examination for the Pre-final term, plus capstone proposal session.'],
  },
  {
    week: 16,
    term: 'Final',
    title: 'LLM Components & RAG',
    lab: 'GA Lab 8 Part 1',
    exam: false,
    kc: 50,
    outs: [
      'Name the building blocks of an LLM application: template, tokenizer, model, parser, chain.',
      'Explain why a plain LLM hallucinates and how RAG reduces this.',
    ],
  },
  {
    week: 17,
    term: 'Final',
    title: 'AI Agents, Guardrails & Trustworthy AI',
    lab: 'GA Lab 8 Part 2',
    exam: false,
    kc: 25,
    outs: [
      'Explain what an LLM agent is: model + tools + a reasoning loop + memory.',
      'Apply input, dialog and output rails with NeMo Guardrails.',
    ],
  },
  {
    week: 18,
    term: 'Final',
    title: 'Scaling & Deploying AI',
    lab: 'GA Lab 9',
    exam: false,
    kc: 58,
    outs: [
      'Explain data, model (tensor) and pipeline parallelism and when each is used.',
      'Use mixed precision and quantization and measure the effect on speed and memory.',
    ],
  },
  {
    week: 19,
    term: 'Final',
    title: 'Capstone Build Week + Mock NCA-GENL Exam',
    lab: null,
    exam: false,
    kc: 0,
    outs: [
      'Apply course skills to build a working application for a real problem.',
      'Complete a 50-question, 60-minute mock certification exam (70% = exam-ready).',
    ],
  },
  {
    week: 20,
    term: 'Final',
    title: 'FINAL EXAM & Demo Day',
    lab: null,
    exam: true,
    kc: 0,
    outs: ['Cumulative final examination, followed by capstone demos.'],
  },
];
const CURRENT_WEEK = 7;
const SHORT = {
  1: 'Intro to GenAI',
  2: 'Neural networks',
  3: 'Backprop',
  4: 'CNNs',
  5: 'Prelim exam',
  6: 'Tokens & embeddings',
  7: 'Attention',
  8: 'Transformers',
  9: 'LLMs & prompting',
  10: 'Midterm exam',
  11: 'Multimodal',
  12: 'GANs',
  13: 'Diffusion',
  14: 'Fine-tuning & RLHF',
  15: 'Pre-final & proposal',
  16: 'RAG',
  17: 'Agents & guardrails',
  18: 'Scale & deploy',
  19: 'Capstone build',
  20: 'Final & Demo Day',
};
const LANES = [
  { term: 'Prelim', name: 'Prelim', theme: 'How machines learn', glyph: '\u2699', color: 'var(--accent)' },
  { term: 'Midterm', name: 'Midterm', theme: 'Language and Transformers', glyph: '\u275D', color: 'var(--cyan)' },
  { term: 'Pre-final', name: 'Pre-final', theme: 'Images and generation', glyph: '\u25D0', color: 'var(--indigo)' },
  { term: 'Final', name: 'Final', theme: 'Apps, agents and deployment', glyph: '\u2726', color: '#E0709B' },
];

class Component extends DCLogic {
  constructor(props) {
    super(props);
    const saved = readTheme();
    this.state = {
      selected: CURRENT_WEEK,
      theme: saved,
      mapOpen: false,
      openTerms: { Prelim: false, Midterm: true, 'Pre-final': false, Final: false },
    };
    this.toggleMap = () => this.setState({ mapOpen: !this.state.mapOpen });
    this.toggleTermPrelim = () => this.toggleTerm('Prelim');
    this.toggleTermMidterm = () => this.toggleTerm('Midterm');
    this.toggleTermPrefinal = () => this.toggleTerm('Pre-final');
    this.toggleTermFinal = () => this.toggleTerm('Final');
    this.toggleTheme = this.toggleTheme.bind(this);
  }
  componentDidMount() {
    this._unbindTheme = bindThemeSync(this);
  }
  componentWillUnmount() {
    if (this._unbindTheme) this._unbindTheme();
  }
  toggleTheme() {
    const next = this.state.theme === 'dark' ? 'light' : 'dark';
    this.setState({ theme: next });
    saveTheme(next);
  }
  palette() {
    const dark = this.state.theme === 'dark';
    return dark
      ? {
          green: '#B6F23A',
          greenBg: 'rgba(118,185,0,0.16)',
          greenBorder: 'rgba(118,185,0,0.35)',
          amber: '#FFB84D',
          amberBg: 'rgba(255,160,0,0.16)',
          amberBorder: 'rgba(255,160,0,0.45)',
          red: '#FCA5A5',
          redBg: 'rgba(239,68,68,0.16)',
          redBorder: 'rgba(239,68,68,0.4)',
          neutralBg: 'rgba(255,255,255,0.06)',
          neutralText: '#A0A5B5',
          neutralBorder: 'rgba(255,255,255,0.12)',
          neutralDot: '#6B7280',
        }
      : {
          green: '#4F7A00',
          greenBg: '#EAF5D1',
          greenBorder: '#CFE3B0',
          amber: '#8A5A12',
          amberBg: '#FCEBD2',
          amberBorder: '#8A5A12',
          red: '#C23B3B',
          redBg: '#FBE3E2',
          redBorder: '#E8A8A4',
          neutralBg: '#EEF0E8',
          neutralText: '#5B6358',
          neutralBorder: '#E1E4DE',
          neutralDot: '#8A9280',
        };
  }
  styleFor(w, pal) {
    if (w.exam) return { border: pal.redBorder, dot: pal.red, glow: 'none' };
    if (w.week < CURRENT_WEEK) return { border: pal.greenBorder, dot: pal.green, glow: 'none' };
    if (w.week === CURRENT_WEEK) return { border: pal.amberBorder, dot: pal.amber, glow: '0 0 0 1px ' + pal.amberBorder + ' inset' };
    return { border: pal.neutralBorder, dot: pal.neutralDot, glow: 'none' };
  }
  statusOf(w, pal) {
    if (w.week === CURRENT_WEEK) return { label: 'In progress', bg: pal.amberBg, color: pal.amber };
    if (w.week < CURRENT_WEEK)
      return w.exam ? { label: 'Passed', bg: pal.redBg, color: pal.red } : { label: 'Graded', bg: pal.greenBg, color: pal.green };
    return w.exam
      ? { label: 'Not started', bg: pal.redBg, color: pal.red }
      : { label: 'Not started', bg: pal.neutralBg, color: pal.neutralText };
  }
  toggleTerm(t) {
    this.setState({ openTerms: { ...this.state.openTerms, [t]: !this.state.openTerms[t] } });
  }
  reveal(open, shift) {
    return {
      expanded: open ? 'true' : 'false',
      rows: open ? '1fr' : '0fr',
      op: open ? '1' : '0',
      vis: open ? 'visible' : 'hidden',
      vt: open ? '0s' : '0s linear .5s',
      rot: open ? 'rotate(180deg)' : 'rotate(0deg)',
      tf: open ? 'none' : 'translateY(-' + (shift || 0) + 'px)',
    };
  }
  withHandlers(list, pal) {
    return list.map((w) => {
      const s = this.styleFor(w, pal);
      return {
        ...w,
        border: s.border,
        dot: s.dot,
        glow: s.glow,
        bg: w.week === this.state.selected ? 'var(--accent-soft-bg)' : 'transparent',
        tag: w.exam ? 'Exam' : w.lab || 'Capstone',
        pick: () => this.setState({ selected: w.week }),
      };
    });
  }
  detailOf(w, pal) {
    const st = this.statusOf(w, pal);
    return {
      week: w.week,
      term: w.term,
      title: w.title,
      outs: w.outs,
      lab: w.lab || 'None (exam week)',
      kcLabel: w.kc > 0 ? w.kc + ' questions' : 'Not applicable',
      dgx: w.exam ? 'Not used' : 'JupyterLab session',
      statusLabel: st.label,
      statusBg: st.bg,
      statusColor: st.color,
    };
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
  renderVals() {
    return { ...this.renderVals0(), ...this.hkVals() };
  }
  renderVals0() {
    const dark = this.state.theme === 'dark';
    const pal = this.palette();
    const selectedWeek = WEEKS.find((w) => w.week === this.state.selected) || WEEKS[CURRENT_WEEK - 1];
    const sel = this.detailOf(selectedWeek, pal);
    const cur = this.detailOf(WEEKS[CURRENT_WEEK - 1], pal);
    const segs = WEEKS.map((w) => ({
      tip: 'Week ' + w.week + ': ' + w.title,
      bg: w.week < CURRENT_WEEK ? pal.green : w.week === CURRENT_WEEK ? pal.amber : pal.neutralBorder,
      cls: w.week === CURRENT_WEEK ? 'pulsedot' : '',
      mark: w.exam ? '\u25C6' : '',
    }));
    const upnext = [8, 9, 10].map((n) => {
      const w = WEEKS[n - 1];
      const s = this.styleFor(w, pal);
      return {
        eyebrow: w.exam ? 'Week ' + n + ' \u00B7 Examination' : 'Week ' + n + ' \u00B7 Next',
        title: w.exam ? 'Midterm examination \u00B7 Week ' + n : w.title,
        sub: w.exam ? 'Cumulative Midterm examination' : w.lab,
        border: s.border,
        dot: s.dot,
        pick: () => this.setState({ selected: n, mapOpen: true }),
      };
    });
    const lanes = LANES.map((L) => {
      const ws = WEEKS.filter((w) => w.term === L.term);
      return {
        name: L.name,
        theme: L.theme,
        glyph: L.glyph,
        color: L.color,
        done: ws.filter((w) => w.week < CURRENT_WEEK).length,
        total: ws.length,
        weeks: ws.map((w) => {
          const state = w.week < CURRENT_WEEK ? 'n-done' : w.week === CURRENT_WEEK ? 'n-cur' : 'n-next';
          const isSel = w.week === this.state.selected;
          return {
            cls: 'mn ' + state + (w.exam ? ' n-exam' : '') + (isSel ? ' sel' : ''),
            tag: w.week === CURRENT_WEEK ? 'You are here' : '',
            mark: w.week < CURRENT_WEEK && !w.exam ? '\u2713' : String(w.week),
            short: SHORT[w.week],
            pressed: isSel ? 'true' : 'false',
            aria:
              'Week ' +
              w.week +
              ': ' +
              SHORT[w.week] +
              (w.week < CURRENT_WEEK ? ', done' : w.week === CURRENT_WEEK ? ', current week' : ''),
            pick: () => this.setState({ selected: w.week }),
          };
        }),
      };
    });
    const ot = this.state.openTerms;
    return {
      ...themeVals(dark),
      toggleTheme: this.toggleTheme,
      pal,
      prelim: this.withHandlers(
        WEEKS.filter((w) => w.term === 'Prelim'),
        pal
      ),
      midterm: this.withHandlers(
        WEEKS.filter((w) => w.term === 'Midterm'),
        pal
      ),
      prefinal: this.withHandlers(
        WEEKS.filter((w) => w.term === 'Pre-final'),
        pal
      ),
      final: this.withHandlers(
        WEEKS.filter((w) => w.term === 'Final'),
        pal
      ),
      sel: { ...sel, top: sel.outs.slice(0, 2) },
      cur,
      segs,
      upnext,
      lanes,
      toggleMap: this.toggleMap,
      toggleTermPrelim: this.toggleTermPrelim,
      toggleTermMidterm: this.toggleTermMidterm,
      toggleTermPrefinal: this.toggleTermPrefinal,
      toggleTermFinal: this.toggleTermFinal,
      mapExpanded: this.state.mapOpen ? 'true' : 'false',
      mapLabel: this.state.mapOpen ? 'Hide full course map' : 'View full course map (20 weeks)',
      mapRot: this.state.mapOpen ? 'rotate(180deg)' : 'rotate(0deg)',
      mapR: this.reveal(this.state.mapOpen, 8),
      tPrelim: this.reveal(!!ot['Prelim']),
      tMidterm: this.reveal(!!ot['Midterm']),
      tPrefinal: this.reveal(!!ot['Pre-final']),
      tFinal: this.reveal(!!ot['Final']),
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
          <div style={{ padding: '8px 10px 22px' }}>
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
            <div style={{ width: '28px', height: '3px', borderRadius: '999px', background: '#76B900', marginTop: '10px' }}></div>
          </div>
          <DcLink className="navlink" href="Main.dc.html">
            <span>{'●'}</span>
            <span>{'Overview'}</span>
          </DcLink>
          <DcLink className="navlink active" href="CourseMap.dc.html">
            <span>{'◉'}</span>
            <span>{'Course Map'}</span>
          </DcLink>
          <DcLink className="navlink" href="Lesson.dc.html">
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
          <div
            style={{
              padding: '16px 10px 4px',
              fontSize: '10.5px',
              fontWeight: '700',
              letterSpacing: '.12em',
              textTransform: 'uppercase',
              color: '#7C857A',
              lineHeight: '1.3',
            }}
          >
            {'Activities'}
          </div>
          <DcLink className="navlink" href="/activities">
            <span>{'▤'}</span>
            <span>{'Activities'}</span>
          </DcLink>
          <DcLink className="navlink" href="/announcements">
            <span>{'✉'}</span>
            <span>{'Announcements'}</span>
          </DcLink>
          <DcLink className="navlink" href="/grades">
            <span>{'✓'}</span>
            <span>{'My grades'}</span>
          </DcLink>
          <div style={{ flex: '1' }}></div>
          <div style={{ borderTop: '1px solid rgba(255,255,255,0.08)', margin: '12px 0 0', paddingTop: '12px' }}>
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
                <div style={{ fontSize: '11.5px', color: '#9BA3AF', lineHeight: '1.4' }}>{'Student · Week 7'}</div>
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
            padding: '36px 48px 64px',
            display: 'flex',
            flexWrap: 'wrap',
            gap: '28px',
            transition: 'background-color .3s ease,color .3s ease',
          }}
        >
          <div className="utilbar cm-w" style={{ flex: '1 1 100%' }}>
            <div className="ctx">
              <span className="ctx-i">{'A.Y. 2026/2027'}</span> <span className="ctx-s" aria-hidden="true"></span>{' '}
              <span className="ctx-i">{'Second Semester'}</span> <span className="ctx-s" aria-hidden="true"></span>{' '}
              <span className="ctx-i ctx-now">{'Week 7 of 20'}</span> <span className="ctx-s" aria-hidden="true"></span>{' '}
              <span className="ctx-i">{'Midterm Term'}</span>
            </div>
            <div className="hkbar" role="toolbar" aria-label="Course tools" onKeyDown={v.hkKey}>
              <div className="hk-scrim" style={css(`display:${v.hkScrim}`)} onClick={v.hkClose}></div>
              <div className={v.hkClsS}>
                <button
                  type="button"
                  className="hkbtn"
                  aria-haspopup="true"
                  aria-expanded={v.hkExS}
                  onClick={v.hkTogS}
                  aria-label="Schedule"
                  title="Schedule (S)"
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
                        <b>{'Lecture 7.1: Attention mechanisms'}</b>
                        <span className="hkm">{'Room CS‑204 · Dr. J. A. Jose'}</span>
                      </span>
                    </li>
                    <li>
                      <span className="hkt">{'10:30'}</span>
                      <span className="hkd">
                        <b>{'Laboratory 3: Scaled dot‑product attention'}</b>
                        <span className="hkm">{'Computing Laboratory 2 · Attendance required'}</span>
                      </span>
                    </li>
                    <li>
                      <span className="hkt">{'14:00'}</span>
                      <span className="hkd">
                        <b>{'Consultation hours'}</b>
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
                  title="Upcoming requirements (U)"
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
                        <b>{'Laboratory 3 report'}</b>
                        <span className="hkm">{'10% of final grade · Due 11:59 PM PHT'}</span>
                      </span>
                    </li>
                    <li>
                      <span className="hkt">{'24 Oct'}</span>
                      <span className="hkd">
                        <b>{'Quiz 4: Transformer architecture'}</b>
                        <span className="hkm">{'5% of final grade · Two attempts permitted'}</span>
                      </span>
                    </li>
                    <li>
                      <span className="hkt">{'07 Nov'}</span>
                      <span className="hkd">
                        <b>{'Midterm examination'}</b>
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
                  aria-label="Alerts, 3 unread"
                  title="Alerts"
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
                  <span className="hkn">{'3'}</span>
                </button>
                <div className="hkpop">
                  <div className="hkpop-h">{'Announcements'}</div>
                  <ul className="hkl">
                    <li>
                      <span className="hkt">{'Today'}</span>
                      <span className="hkd">
                        <b>{'Laboratory 3 rubric published'}</b>
                        <span className="hkm">{'Posted by Dr. J. A. Jose'}</span>
                      </span>
                    </li>
                    <li>
                      <span className="hkt">{'13 Oct'}</span>
                      <span className="hkd">
                        <b>{'Quiz 3 results released'}</b>
                        <span className="hkm">{'Class mean 82% · Review in consultation hours'}</span>
                      </span>
                    </li>
                    <li>
                      <span className="hkt">{'11 Oct'}</span>
                      <span className="hkd">
                        <b>{'Reading added for Week 8'}</b>
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
                  className="hkbtn hkbtn-icon"
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
          </div>
          <div className="cm-w" style={{ flex: '1 1 100%', minWidth: '0', display: 'flex', flexDirection: 'column', gap: '0' }}>
            <header className="hx hx-green" style={{ marginBottom: 'var(--s5)' }}>
              <div className="hx-body">
                <div className="eyebrow" style={{ marginBottom: 'var(--s2)' }}>
                  {'Course map'}
                </div>
                <h1 className="cm-title">{'Generative AI and LLMs with NVIDIA'}</h1>
                <p className="cm-lede cm-r" style={{ marginTop: 'var(--s3)' }}>
                  {'Four terms. Twenty weeks. '}
                  <mark className="mk">{'One working AI application'}</mark>
                  {' at the end.'}
                </p>
                <div className="mstats" role="list" aria-label="Course in numbers">
                  <div className="ms" role="listitem">
                    <b>{'20'}</b>
                    <span>{'weeks'}</span>
                  </div>
                  <div className="ms" role="listitem">
                    <b>{'4'}</b>
                    <span>{'terms'}</span>
                  </div>
                  <div className="ms" role="listitem">
                    <b>{'15'}</b>
                    <span>{'lab weeks'}</span>
                  </div>
                  <div className="ms" role="listitem">
                    <b>{'1'}</b>
                    <span>{'capstone'}</span>
                  </div>
                </div>
              </div>
              <div className="hx-ico">
                <svg viewBox="0 0 24 24" aria-hidden="true">
                  <circle cx="6" cy="18" r="2.4"></circle>
                  <circle cx="18" cy="6" r="2.4"></circle>
                  <path d="M8.3 17.2c4.2 0 .9-5.2 4-5.2s0-4.6 3.4-4.8"></path>
                </svg>
              </div>
            </header>
            <div className="sect" style={{ marginBottom: 'var(--s5)' }}>
              <div className="mdet card" aria-live="polite">
                <div className="md-l">
                  <div className="md-wk">
                    <span className="md-w">{'Week'}</span>
                    <span className="md-n">{v.sel?.week}</span>
                  </div>
                  <span className="pill" style={css(`background:${v.sel?.statusBg};color:${v.sel?.statusColor}`)}>
                    {v.sel?.statusLabel}
                  </span>
                </div>
                <div className="md-m">
                  <h3 className="wk-t" style={{ margin: '0 0 var(--s2)' }}>
                    {v.sel?.title}
                  </h3>
                  <ul className="md-o">
                    {(v.sel?.top || []).map((o, $index) => (
                      <React.Fragment key={$index}>
                        <li>{o}</li>
                      </React.Fragment>
                    ))}
                  </ul>
                  <div className="md-c">
                    <span className="chip-i">
                      <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                        <path
                          d="M9.5 3.5h5M10.5 3.5v5.6L5.2 18a1.8 1.8 0 0 0 1.6 2.7h10.4a1.8 1.8 0 0 0 1.6-2.7l-5.3-8.9V3.5"
                          stroke="currentColor"
                          strokeWidth="1.7"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        ></path>
                      </svg>
                      {v.sel?.lab}
                    </span>{' '}
                    <span className="chip-i">
                      <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                        <circle cx="12" cy="12" r="8.8" stroke="currentColor" strokeWidth="1.7"></circle>
                        <path
                          d="m8.2 12.3 2.7 2.7 5-5.4"
                          stroke="currentColor"
                          strokeWidth="1.7"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        ></path>
                      </svg>
                      {v.sel?.kcLabel}
                    </span>
                  </div>
                </div>
                <DcLink className="ctabtn md-go" href={`/lesson/${v.sel?.week}`}>
                  {'Open lesson'}
                </DcLink>
                <DcLink className="ctabtn md-go" href={`/prompt-workshop/${v.sel?.week}`}>
                  {'Open workshop'}
                </DcLink>
                <DcLink className="ctabtn md-go" href={`/dashboard/${v.sel?.week}`}>
                  {'Week overview'}
                </DcLink>
              </div>
            </div>
            <div className="sect journey" style={{ marginBottom: 'var(--s5)' }}>
              <div className="sect-h">
                <h2 className="sect-t">{'Your journey'}</h2>
                <div className="mleg" aria-label="Legend">
                  <span className="ml">
                    <i className="ml-d ml-done"></i>
                    {'Done'}
                  </span>{' '}
                  <span className="ml">
                    <i className="ml-d ml-cur"></i>
                    {'You are here'}
                  </span>{' '}
                  <span className="ml">
                    <i className="ml-d ml-next"></i>
                    {'Ahead'}
                  </span>{' '}
                  <span className="ml">
                    <i className="ml-d ml-exam"></i>
                    {'Exam'}
                  </span>
                </div>
              </div>
              <div className="lanes">
                {(v.lanes || []).map((lane, $index) => (
                  <React.Fragment key={$index}>
                    <section className="lane" style={css(`--lane:${lane?.color}`)} aria-label={`${lane?.name} term`}>
                      <div className="lane-h">
                        <span className="lane-ic" aria-hidden="true">
                          {lane?.glyph}
                        </span>
                        <div className="lane-t">
                          <h3 className="lane-n">{lane?.name}</h3>
                          <p className="lane-s">{lane?.theme}</p>
                        </div>
                        <span className="lane-p">
                          {lane?.done}
                          <span>
                            {' of '}
                            {lane?.total}
                          </span>
                        </span>
                      </div>
                      <div className="nodes">
                        {(lane?.weeks || []).map((n, $index) => (
                          <React.Fragment key={$index}>
                            <button type="button" className={n?.cls} onClick={n?.pick} aria-label={n?.aria} aria-pressed={n?.pressed}>
                              <span className="mn-tag">{n?.tag}</span> <span className="mn-dot">{n?.mark}</span>{' '}
                              <span className="mn-l">{n?.short}</span>
                            </button>
                          </React.Fragment>
                        ))}
                      </div>
                    </section>
                  </React.Fragment>
                ))}
              </div>
            </div>
            <div className="sect" style={{ marginBottom: 'var(--s5)' }}>
              <div className="sect-h">
                <h2 className="sect-t">{'By the end, you can'}</h2>
              </div>
              <ul className="skills">
                <li className="sk">
                  <span className="sk-n">{'1'}</span>
                  {'Train and explain neural networks and CNNs'}
                </li>
                <li className="sk">
                  <span className="sk-n">{'2'}</span>
                  {'Build the core of a Transformer'}
                </li>
                <li className="sk">
                  <span className="sk-n">{'3'}</span>
                  {'Compare LLMs, prompt and fine-tune them'}
                </li>
                <li className="sk">
                  <span className="sk-n">{'4'}</span>
                  {'Explain multimodal, GAN and diffusion models'}
                </li>
                <li className="sk">
                  <span className="sk-n">{'5'}</span>
                  {'Design a RAG pipeline with guardrails'}
                </li>
                <li className="sk">
                  <span className="sk-n">{'6'}</span>
                  {'Ship a capstone and pass the mock NCA-GENL exam'}
                </li>
              </ul>
            </div>
            <details className="more">
              <summary>
                <span>{'Grading and course rules'}</span>
                <span className="chev" aria-hidden="true">
                  {'▾'}
                </span>
              </summary>
              <div className="more-b">
                <div className="sect">
                  <div className="sect-h">
                    <h2 className="sect-t">{'Assessment scheme'}</h2>
                    <span className="sect-n">{'Component weights are not stated in this outline'}</span>
                  </div>
                  <div className="card" style={{ padding: '6px 18px', overflowX: 'auto' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '15px', minWidth: '460px' }}>
                      <thead>
                        <tr style={{ textAlign: 'left', color: 'var(--text3)', fontSize: '13px' }}>
                          <th style={{ padding: '10px 12px 10px 0', fontWeight: '500' }}>{'Component'}</th>
                          <th style={{ padding: '10px 12px', fontWeight: '500' }}>{'Weight'}</th>
                          <th style={{ padding: '10px 0 10px 12px', fontWeight: '500' }}>{'Timing'}</th>
                        </tr>
                      </thead>
                      <tbody style={{ color: 'var(--text)' }}>
                        <tr style={{ borderTop: '1px solid var(--divider)' }}>
                          <td style={{ padding: '12px 12px 12px 0', fontWeight: '500', lineHeight: '1.5' }}>
                            {'Hands-on laboratories (GA Lab and DL Lab series)'}
                          </td>
                          <td style={{ padding: '12px', color: 'var(--text2)', whiteSpace: 'nowrap' }}>{'Not specified'}</td>
                          <td style={{ padding: '12px 0 12px 12px', color: 'var(--text2)' }}>{'Weeks 1 to 18, except exam weeks'}</td>
                        </tr>
                        <tr style={{ borderTop: '1px solid var(--divider)' }}>
                          <td style={{ padding: '12px 12px 12px 0', fontWeight: '500', lineHeight: '1.5' }}>{'Weekly knowledge checks'}</td>
                          <td style={{ padding: '12px', color: 'var(--text2)', whiteSpace: 'nowrap' }}>{'Not specified'}</td>
                          <td style={{ padding: '12px 0 12px 12px', color: 'var(--text2)' }}>
                            {'Weeks 1 to 4, 6 to 9, 11 to 14 and 16 to 18'}
                          </td>
                        </tr>
                        <tr style={{ borderTop: '1px solid var(--divider)' }}>
                          <td style={{ padding: '12px 12px 12px 0', fontWeight: '500', lineHeight: '1.5' }}>{'Prelim examination'}</td>
                          <td style={{ padding: '12px', color: 'var(--text2)', whiteSpace: 'nowrap' }}>{'Not specified'}</td>
                          <td style={{ padding: '12px 0 12px 12px', color: 'var(--text2)' }}>{'Week 5'}</td>
                        </tr>
                        <tr style={{ borderTop: '1px solid var(--divider)' }}>
                          <td style={{ padding: '12px 12px 12px 0', fontWeight: '500', lineHeight: '1.5' }}>{'Midterm examination'}</td>
                          <td style={{ padding: '12px', color: 'var(--text2)', whiteSpace: 'nowrap' }}>{'Not specified'}</td>
                          <td style={{ padding: '12px 0 12px 12px', color: 'var(--text2)' }}>{'Week 10'}</td>
                        </tr>
                        <tr style={{ borderTop: '1px solid var(--divider)' }}>
                          <td style={{ padding: '12px 12px 12px 0', fontWeight: '500', lineHeight: '1.5' }}>
                            {'Pre-final examination and capstone proposal'}
                          </td>
                          <td style={{ padding: '12px', color: 'var(--text2)', whiteSpace: 'nowrap' }}>{'Not specified'}</td>
                          <td style={{ padding: '12px 0 12px 12px', color: 'var(--text2)' }}>{'Week 15'}</td>
                        </tr>
                        <tr style={{ borderTop: '1px solid var(--divider)' }}>
                          <td style={{ padding: '12px 12px 12px 0', fontWeight: '500', lineHeight: '1.5' }}>
                            {'Capstone build and mock NCA-GENL examination (70% is the readiness threshold)'}
                          </td>
                          <td style={{ padding: '12px', color: 'var(--text2)', whiteSpace: 'nowrap' }}>{'Not specified'}</td>
                          <td style={{ padding: '12px 0 12px 12px', color: 'var(--text2)' }}>{'Week 19'}</td>
                        </tr>
                        <tr style={{ borderTop: '1px solid var(--divider)' }}>
                          <td style={{ padding: '12px 12px 12px 0', fontWeight: '500', lineHeight: '1.5' }}>
                            {'Final examination and Demo Day'}
                          </td>
                          <td style={{ padding: '12px', color: 'var(--text2)', whiteSpace: 'nowrap' }}>{'Not specified'}</td>
                          <td style={{ padding: '12px 0 12px 12px', color: 'var(--text2)' }}>{'Week 20'}</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>
                <div className="sect">
                  <div className="sect-h">
                    <h2 className="sect-t">{'Course policies'}</h2>
                    <DcLink className="sect-n" href="#">
                      {'Academic integrity policy'}
                    </DcLink>
                  </div>
                  <ul className="cm-list cm-r" style={{ color: 'var(--text2)' }}>
                    <li>
                      <span style={{ color: 'var(--text)', fontWeight: '500' }}>{'Attendance:'}</span>
                      {' Attendance is required at laboratory sessions and examinations.'}
                    </li>
                    <li>
                      <span style={{ color: 'var(--text)', fontWeight: '500' }}>{'Academic integrity:'}</span>
                      {' Submitted work must be your own. Disclose any AI assistance as set out in the course policy.'}
                    </li>
                    <li>
                      <span style={{ color: 'var(--text)', fontWeight: '500' }}>{'Late submissions:'}</span>
                      {' Work submitted after its deadline is marked Overdue and is handled under the course policy.'}
                    </li>
                  </ul>
                </div>
              </div>
            </details>
          </div>
          <footer className="sitefoot" style={{ flex: '1 1 100%' }}>
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
    </>
  );
}

Component.template = template;
Component.css = pageCss;
Component.pageKey = 'course-map';
Component.persist = ['selected', 'openTerms'];

export default Component;
