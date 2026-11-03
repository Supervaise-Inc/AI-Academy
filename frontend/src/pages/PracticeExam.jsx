// Generated from design-source/PracticeExam.dc.html by tools/convert.mjs.
// Re-running the converter regenerates (and overwrites) this file, so edit design-source/ + tools/patches.mjs instead,
// or stop using the converter and edit this file directly.
/* eslint-disable */
import React from 'react';
import DCLogic from '../dc/DCLogic.jsx';
import DcLink from '../dc/DcLink.jsx';
import { css } from '../dc/css.js';
import pageCss from './PracticeExam.css?inline';

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

const HISTORY_KEY = 'supervaise-exam-history';
const MODULES = [
  {
    key: 'mid',
    code: 'MID',
    tag: 'Midterm review',
    title: 'Weeks 6 to 9: knowledge check review',
    seconds: 240,
    desc: 'Tokens, embeddings, scaling laws and data quality: the core of the Week 10 midterm.',
    tagBg: 'var(--cyan-soft-bg)',
    tagFg: 'var(--cyan-text)',
    tagBorder: 'var(--cyan-soft-border)',
    questions: [
      {
        id: 'GA-2.2-Q03',
        domain: 'Week 6 · Tokens',
        question: 'Which tokenization method divides text into characters?',
        options: ['Word-level tokenization', 'Character-level tokenization', 'Sub-word tokenization', 'Byte-level tokenization'],
        answer_index: 1,
        explain:
          'Character-level tokenization splits text into individual characters. The vocabulary is tiny, but sequences become much longer, which is one reason modern LLMs prefer sub-word tokens.',
      },
      {
        id: 'GA-2.2-Q07',
        domain: 'Week 6 · Tokens',
        question: 'What is an advantage of sub-word tokenization?',
        options: [
          'Handles rare words effectively',
          'Minimizes memory usage',
          'Supports larger vocabulary sizes',
          'Balances rare and frequent words',
        ],
        answer_index: 0,
        explain:
          'Sub-word schemes such as BPE break a rare or unseen word into known pieces, so the model almost never meets a true out-of-vocabulary token.',
      },
      {
        id: 'GA-2.3-Q08',
        domain: 'Week 6 · Embeddings',
        question: "What is a limitation of Word2Vec's static embeddings?",
        options: [
          'Fixed meaning for each word',
          'Requires large datasets',
          'Requires labeled training data',
          'Fails to generalize across languages',
        ],
        answer_index: 0,
        explain:
          'Word2Vec gives every word one fixed vector, so "bank" (river) and "bank" (money) share an embedding. Contextual embeddings from Transformers fix this.',
      },
      {
        id: 'GA-4.2-Q02',
        domain: 'Week 9 · Scaling laws',
        question: 'What is a key insight from LLM scaling laws?',
        options: [
          'Adding more layers always improves accuracy',
          'Scaling must balance model size, dataset size, and compute',
          'Models should prioritize dataset size over compute',
          'Data must always be larger than model parameters',
        ],
        answer_index: 1,
        explain:
          'Loss falls predictably only when parameters, training tokens and compute grow together. Chinchilla showed many large models were under-trained for their size.',
      },
      {
        id: 'GA-4.3-Q09',
        domain: 'Week 9 · Data',
        question: 'Why is deduplication important in training datasets?',
        options: [
          'It increases the model’s training time',
          'It helps avoid redundant training data',
          'It removes essential training data',
          'It slows down inference speeds',
        ],
        answer_index: 1,
        explain:
          'Duplicate text wastes compute and pushes the model toward memorising repeated passages. Deduplication keeps the training signal diverse.',
      },
    ],
  },
  {
    key: 'kc',
    code: 'KC',
    tag: 'Knowledge check',
    title: 'GA-3.1: Language models and attention',
    seconds: 120,
    desc: "This week's knowledge check, timed: sequence models and attention.",
    tagBg: 'var(--amber-soft-bg)',
    tagFg: 'var(--amber-text)',
    tagBorder: 'var(--amber-soft-border)',
    questions: [
      {
        id: 'GA-3.1-Q01',
        domain: 'Week 7 · Language models',
        question: 'What is the main goal of a language model in deep learning?',
        options: [
          'To classify text into categories',
          'To predict the next word in a sequence',
          'To improve token embeddings',
          'To reduce training time',
        ],
        answer_index: 1,
        explain: 'A language model assigns probabilities to sequences. In practice, it predicts the next token from the tokens before it.',
      },
      {
        id: 'GA-3.1-Q02',
        domain: 'Week 7 · Sequence models',
        question: 'What is a key challenge of modeling sequences with standard neural networks?',
        options: [
          'They lack memory for previous inputs',
          'They ignore future tokens',
          'They require labeled data for training',
          'They have high computational costs',
        ],
        answer_index: 0,
        explain: 'A plain feed-forward network treats each input independently, so it has no memory of the earlier tokens in a sequence.',
      },
      {
        id: 'GA-3.1-Q03',
        domain: 'Week 7 · RNNs',
        question: 'What feature distinguishes Recurrent Neural Networks (RNNs) from feedforward networks?',
        options: [
          'They process sequences without recurrence',
          'They use recurrence to process sequential data',
          'They store all previous outputs',
          'They require no sequence processing',
        ],
        answer_index: 1,
        explain:
          'An RNN feeds its hidden state back in at every step. That recurrence carries information along the sequence, though it fades over long inputs, which is what attention addresses.',
      },
    ],
  },
  {
    key: 'nca',
    code: 'NCA',
    tag: 'Certification preparation',
    title: 'Mock NCA-GENL: sample set',
    seconds: 180,
    desc: 'Five items in the style of the NCA-GENL exam, across its blueprint domains.',
    tagBg: 'var(--indigo-soft-bg)',
    tagFg: 'var(--indigo)',
    tagBorder: 'var(--indigo-soft-border)',
    questions: [
      {
        id: 'MOCK-Q01',
        domain: 'Core ML & AI Knowledge',
        question: 'What is the purpose of the causal (look-ahead) mask in a decoder-only transformer?',
        options: [
          'It hides only the padding tokens at the end of a batch',
          'It prevents each position from attending to tokens that come after it',
          'It randomly drops tokens to regularize training',
          'It masks out rare words that are not in the vocabulary',
        ],
        answer_index: 1,
        explain:
          'Position i may only attend to positions ≤ i. The mask sets future scores to −∞ before the softmax, so generation can never peek at tokens it has not produced yet.',
      },
      {
        id: 'MOCK-Q02',
        domain: 'Experimentation',
        question: 'You are tuning the learning rate and dropout of a model. Which data split should you use to compare the settings?',
        options: [
          'The validation set, keeping the test set untouched for a final, unbiased estimate',
          'All three splits merged together',
          'The training set, because it is the largest',
          'The test set, because it gives the most realistic score',
        ],
        answer_index: 0,
        explain:
          'Hyperparameters are compared on the validation set. Touching the test set while tuning leaks information and inflates the final score.',
      },
      {
        id: 'MOCK-Q03',
        domain: 'Core ML & AI Knowledge',
        question: 'What makes CLIP useful for zero-shot image classification?',
        options: [
          'It converts speech into text before classifying images',
          'It learns a shared embedding space for images and text through contrastive training, so an image can be matched to text descriptions of classes it was never explicitly trained on',
          'It is a CNN trained only on the 1,000 ImageNet classes',
          'It generates new images by gradually removing noise',
        ],
        answer_index: 1,
        explain:
          'Contrastive pre-training places images and captions in one embedding space, so you classify by matching an image against prompts such as "a photo of a {class}".',
      },
      {
        id: 'MOCK-Q04',
        domain: 'Software Development',
        question:
          'Using the Hugging Face Transformers library, which pair of calls loads a tokenizer and a model for text generation from a model ID?',
        options: [
          'torch.load(model_id) and torch.save(model_id)',
          "pipeline('image-classification', model=model_id) only",
          'AutoTokenizer.from_pretrained(model_id) and AutoModelForSequenceClassification.from_pretrained(model_id)',
          'AutoTokenizer.from_pretrained(model_id) and AutoModelForCausalLM.from_pretrained(model_id)',
        ],
        answer_index: 3,
        explain:
          'AutoModelForCausalLM loads a decoder with a language-modelling head for generation; AutoModelForSequenceClassification adds a classifier head instead.',
      },
      {
        id: 'MOCK-Q05',
        domain: 'Trustworthy AI',
        question:
          'A RAG chatbot occasionally answers confidently with information not found in any retrieved document. What is this called, and what is one mitigation?',
        options: [
          'Overfitting: mitigate with more training epochs',
          'Hallucination: mitigate by instructing the model to answer only from retrieved context and to say when it is unsure',
          'Mode collapse: mitigate with label smoothing',
          'Catastrophic forgetting: mitigate with a larger learning rate',
        ],
        answer_index: 1,
        explain:
          'This is hallucination. Grounding answers in the retrieved context, and letting the model say it does not know, reduces it. Guardrails add a further check.',
      },
    ],
  },
];
const KEYS = ['A', 'B', 'C', 'D'];

function loadHistory() {
  return [];
}
function storeHistory(arr) {}
function fmt(sec) {
  const m = Math.floor(sec / 60),
    s = sec % 60;
  return m + ':' + (s < 10 ? '0' : '') + s;
}

class Component extends DCLogic {
  constructor(props) {
    super(props);
    this.state = {
      theme: readTheme(),
      view: 'home',
      mod: 0,
      qIndex: 0,
      answers: [],
      flags: [],
      secondsLeft: 0,
      modal: null,
      history: loadHistory(),
    };
    this.toggleTheme = this.toggleTheme.bind(this);
  }
  componentDidMount() {
    this._unbindTheme = bindThemeSync(this);
  }
  componentWillUnmount() {
    if (this._timer) clearInterval(this._timer);
    if (this._unbindTheme) this._unbindTheme();
  }
  toggleTheme() {
    const next = this.state.theme === 'dark' ? 'light' : 'dark';
    this.setState({ theme: next });
    saveTheme(next);
  }

  start(mi) {
    if (this._timer) clearInterval(this._timer);
    const n = MODULES[mi].questions.length;
    this.setState({
      view: 'exam',
      mod: mi,
      qIndex: 0,
      answers: new Array(n).fill(null),
      flags: new Array(n).fill(false),
      secondsLeft: MODULES[mi].seconds,
      modal: null,
    });
    this._timer = setInterval(() => {
      const t = this.state.secondsLeft - 1;
      if (t <= 0) {
        this.setState({ secondsLeft: 0 });
        this.finish();
      } else this.setState({ secondsLeft: t });
    }, 1000);
  }
  finish() {
    if (this._timer) {
      clearInterval(this._timer);
      this._timer = null;
    }
    if (this.state.view !== 'exam') return;
    const m = MODULES[this.state.mod];
    const score = m.questions.filter((q, i) => this.state.answers[i] === q.answer_index).length;
    const history = this.state.history.concat([{ m: m.key, s: score, t: m.questions.length, at: Date.now() }]).slice(-8);
    storeHistory(history);
    this.setState({ view: 'result', history });
  }
  quit() {
    if (this._timer) {
      clearInterval(this._timer);
      this._timer = null;
    }
    this.setState({ view: 'home', modal: null });
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
    const st = this.state;
    const mod = MODULES[st.mod];
    const qs = mod.questions;
    const qi = Math.min(st.qIndex, qs.length - 1);
    const q = qs[qi];
    const hist = st.history;

    const modules = MODULES.map((m, i) => {
      const mine = hist.filter((h) => h.m === m.key);
      const best = mine.length
        ? Math.max.apply(
            null,
            mine.map((h) => Math.round((h.s / h.t) * 100))
          )
        : null;
      return {
        tag: m.tag,
        title: m.title,
        desc: m.desc,
        count: m.questions.length,
        timeLabel: fmt(m.seconds),
        best: best === null ? 'No attempt recorded' : 'Best score ' + best + '%',
        tagBg: m.tagBg,
        tagFg: m.tagFg,
        tagBorder: m.tagBorder,
        start: () => this.start(i),
      };
    });
    const fmtDate = (t) => {
      try {
        return (
          new Date(t).toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric', timeZone: 'Asia/Manila' }) + ' PHT'
        );
      } catch (e) {
        return 'Date not recorded';
      }
    };
    const history = hist
      .slice()
      .reverse()
      .map((h) => {
        const pct = Math.round((h.s / h.t) * 100);
        const m = MODULES.find((x) => x.key === h.m);
        return {
          pct: pct + '%',
          pctLabel: pct + '%',
          code: m ? m.code : 'n/a',
          color: pct >= 70 ? 'var(--accent-soft-text)' : 'var(--amber-text)',
          dateLabel: fmtDate(h.at),
          moduleLabel: m ? m.tag : 'Module',
          scoreLabel: h.s + ' of ' + h.t + ' (' + pct + '%)',
          resultLabel: pct >= 70 ? 'Passed' : 'Not passed',
        };
      });
    const pcts = hist.map((h) => Math.round((h.s / h.t) * 100));
    const kcRecord = [1, 2, 3, 4, 5, 6, 7].map((w) => ({
      week: w,
      color: w === 7 ? 'var(--amber)' : w === 5 ? 'var(--cyan)' : 'var(--accent)',
    }));

    const ans = st.answers;
    const opts = q.options.map((text, i) => {
      const sel = ans[qi] === i;
      return {
        key: KEYS[i],
        text,
        bg: sel ? 'var(--accent-soft-bg)' : 'var(--card-bg)',
        border: sel ? 'var(--accent)' : 'var(--card-border)',
        keyBg: sel ? 'var(--accent-cta-bg)' : 'var(--surface2)',
        keyFg: sel ? 'var(--accent-cta-text)' : 'var(--text2)',
        pick: () => {
          const a = st.answers.slice();
          a[qi] = i;
          this.setState({ answers: a });
        },
      };
    });
    const navs = qs.map((_, i) => {
      const answered = ans[i] !== null && ans[i] !== undefined,
        flagged = !!st.flags[i],
        cur = i === qi;
      let bg = 'var(--surface2)',
        fg = 'var(--text2)',
        border = '1px solid var(--surface2-border)';
      if (answered) {
        bg = 'var(--accent-cta-bg)';
        fg = 'var(--accent-cta-text)';
        border = '1px solid var(--accent-cta-bg)';
      }
      if (flagged) {
        bg = 'var(--amber-soft-bg)';
        fg = 'var(--amber-text)';
        border = '1px solid var(--amber)';
      }
      if (cur) border = '2px solid var(--text)';
      return { label: String(i + 1), bg, fg, border, go: () => this.setState({ qIndex: i }) };
    });
    const answeredCount = ans.filter((a) => a !== null && a !== undefined).length;

    const review = qs.map((qq, i) => {
      const correct = ans[i] === qq.answer_index,
        skipped = ans[i] === null || ans[i] === undefined;
      return {
        n: i + 1,
        domain: qq.domain,
        question: qq.question,
        mark: correct ? 'Correct' : skipped ? 'Not answered' : 'Incorrect',
        icon: correct ? '✓' : skipped ? '–' : '✕',
        markBg: correct ? 'var(--accent-soft-bg)' : skipped ? 'var(--surface2)' : 'var(--red-soft-bg)',
        markFg: correct ? 'var(--accent-soft-text)' : skipped ? 'var(--text3)' : 'var(--red-text)',
        explain: () => this.setState({ modal: i }),
      };
    });
    const score = review.filter((r) => r.mark === 'Correct').length;
    const pct = Math.round((score / qs.length) * 100);
    const md = st.modal !== null ? qs[st.modal] : null;
    const mdRow = st.modal !== null ? review[st.modal] : null;

    return {
      ...themeVals(dark),
      toggleTheme: this.toggleTheme,
      homeView: st.view === 'home',
      examView: st.view === 'exam',
      resultView: st.view === 'result',
      modules,
      history,
      noHistory: hist.length === 0,
      hasHistory: hist.length > 0,
      hCount: hist.length,
      hAvg: pcts.length ? Math.round(pcts.reduce((a, b) => a + b, 0) / pcts.length) + '%' : 'n/a',
      hBest: pcts.length ? Math.max.apply(null, pcts) + '%' : 'n/a',
      clearHistory: () => {
        storeHistory([]);
        this.setState({ history: [] });
      },
      kcRecord,
      domains: [
        { name: 'Core Machine Learning & AI Knowledge', weight: '30%' },
        { name: 'Software Development', weight: '24%' },
        { name: 'Experimentation', weight: '22%' },
        { name: 'Data Analysis', weight: '14%' },
        { name: 'Trustworthy AI', weight: '10%' },
      ],
      modTag: mod.tag,
      modTitle: mod.title,
      timeLeft: fmt(st.secondsLeft),
      timerColor: st.secondsLeft <= 30 ? 'var(--red-text)' : 'var(--accent-soft-text)',
      timerBorder: st.secondsLeft <= 30 ? 'var(--red-border)' : 'var(--card-border)',
      qNum: qi + 1,
      qTotal: qs.length,
      qDomain: q.domain,
      qId: q.id,
      qText: q.question,
      opts,
      navs,
      answeredCount,
      isLast: qi === qs.length - 1,
      notLast: qi !== qs.length - 1,
      next: () => this.setState({ qIndex: Math.min(qi + 1, qs.length - 1) }),
      prev: () => this.setState({ qIndex: Math.max(qi - 1, 0) }),
      toggleFlag: () => {
        const f = st.flags.slice();
        f[qi] = !f[qi];
        this.setState({ flags: f });
      },
      flagLabel: st.flags[qi] ? 'Flagged' : 'Flag for review',
      flagFg: st.flags[qi] ? 'var(--amber-text)' : 'var(--text)',
      flagBorder: st.flags[qi] ? 'var(--amber)' : 'var(--surface2-border)',
      finish: () => this.finish(),
      quit: () => this.quit(),
      scorePct: pct + '%',
      scoreLine: score + ' / ' + qs.length,
      ringOffset: (289.03 * (1 - pct / 100)).toFixed(2),
      ringColor: pct >= 70 ? 'var(--accent)' : 'var(--amber)',
      verdict: 'Score: ' + pct + '%. Passing standard: 70%. Result: ' + (pct >= 70 ? 'Passed.' : 'Not passed.'),
      verdictSub:
        pct >= 70
          ? 'Review the explanations for any incorrect or unanswered items.'
          : 'Review the missed topics in the Lesson workspace, then repeat the module.',
      review,
      retry: () => this.start(st.mod),
      goHome: () => this.setState({ view: 'home', modal: null }),
      modalOpen: md !== null,
      closeModal: () => this.setState({ modal: null }),
      mdQuestion: md ? md.question : '',
      mdCorrect: md ? md.options[md.answer_index] : '',
      mdYours: md ? (ans[st.modal] === null || ans[st.modal] === undefined ? 'No answer' : md.options[ans[st.modal]]) : '',
      mdExplain: md ? md.explain : '',
      mdSource: md ? 'Source: ' + md.id + ' · ' + md.domain : '',
      mdMark: mdRow ? mdRow.mark : '',
      mdBg: mdRow ? mdRow.markBg : 'transparent',
      mdFg: mdRow ? mdRow.markFg : 'var(--text)',
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
        <div className="sidebar">
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
          </DcLink>{' '}
          <DcLink className="navlink" href="CourseMap.dc.html">
            <span>{'◉'}</span>
            <span>{'Course Map'}</span>
          </DcLink>{' '}
          <DcLink className="navlink" href="Lesson.dc.html">
            <span>{'◇'}</span>
            <span>{'Lesson Workspace'}</span>
          </DcLink>{' '}
          <DcLink className="navlink" href="PromptWorkshop.dc.html">
            <span>{'✎'}</span>
            <span>{'Prompt Workshop'}</span>
          </DcLink>{' '}
          <DcLink className="navlink" href="Capstone.dc.html">
            <span>{'⚐'}</span>
            <span>{'Capstone Project'}</span>
          </DcLink>{' '}
          <DcLink className="navlink active" href="PracticeExam.dc.html">
            <span>{'⏱'}</span>
            <span>{'Mock Exam Prep'}</span>
          </DcLink>{' '}
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
                <div style={{ fontSize: '11.5px', color: '#9BA3AF', lineHeight: '1.4' }}>{'Student'}</div>
              </div>
            </div>
            <DcLink className="navlink" href="Landing.dc.html" onClick={v.signOut} style={{ color: '#7C857A' }}>
              <span>{'⎋'}</span>
              <span>{'Sign out'}</span>
            </DcLink>
          </div>
        </div>
        <div className="main cm" style={{}}>
          <div className="cm-w">
            <div className="utilbar">
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
                  <button type="button" className="hkbtn" aria-haspopup="true" aria-expanded={v.hkExU} onClick={v.hkTogU}>
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
                  <button type="button" className="hkbtn" aria-haspopup="true" aria-expanded={v.hkExA} onClick={v.hkTogA}>
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
            {v.homeView ? (
              <>
                <header className="hx hx-rose cp-head">
                  <div className="hx-body">
                    <div className="eyebrow" style={{ marginBottom: 'var(--s2)' }}>
                      {'Midterm examination · Week 10'}
                    </div>
                    <h1 className="cm-title">{'Mock Exam Prep'}</h1>
                    <p className="cm-lede" style={{ margin: 'var(--s3) 0 0' }}>
                      {'Practise with timed questions. '}
                      <mark className="mk">{'Answers unlock only after you submit'}</mark>
                      {', each with an explanation.'}
                    </p>
                    <div className="cp-chips">
                      <span className="chip-i">{'Practice sets of 3 to 5 questions'}</span>
                      <span className="chip-i">{'Pass mark 70%'}</span>
                      <span className="chip-i">{'Closed book'}</span>
                    </div>
                  </div>
                  <div className="hx-ico">
                    <svg viewBox="0 0 24 24" aria-hidden="true">
                      <circle cx="12" cy="13.5" r="7"></circle>
                      <path d="M12 10v3.8l2.4 1.4M9.5 3h5M12 3v3.5"></path>
                    </svg>
                  </div>
                </header>
                <div className="sect">
                  <div className="sect-h">
                    <h2 className="sect-t">{'How it works'}</h2>
                  </div>
                  <ol className="flow f3">
                    <li>
                      <span className="fl-n">{'1'}</span>
                      <span className="fl-t">
                        {'Pick a module'}
                        <small>{'Three timed sets, below.'}</small>
                      </span>
                    </li>
                    <li>
                      <span className="fl-n">{'2'}</span>
                      <span className="fl-t">
                        {'Answer before time runs out'}
                        <small>{'Flag any question to revisit it.'}</small>
                      </span>
                    </li>
                    <li>
                      <span className="fl-n">{'3'}</span>
                      <span className="fl-t">
                        {'Submit and review'}
                        <small>{'Select Explain on any question.'}</small>
                      </span>
                    </li>
                  </ol>
                </div>
                <div className="sect">
                  <div className="sect-h">
                    <h2 className="sect-t">{'Practice modules'}</h2>
                    <span className="sect-n">{'Timed · repeat as often as you like'}</span>
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(250px,1fr))', gap: '16px' }}>
                    {(v.modules || []).map((m, $index) => (
                      <React.Fragment key={$index}>
                        <div className="card modcard">
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '8px' }}>
                            <span
                              className="badge"
                              style={css(`background:${m?.tagBg};color:${m?.tagFg};border:1px solid ${m?.tagBorder}`)}
                            >
                              {m?.tag}
                            </span>
                            <span className="mono" style={{ fontSize: '14px', fontWeight: '500', color: 'var(--text3)' }}>
                              {m?.timeLabel}
                              {' limit'}
                            </span>
                          </div>
                          <div style={{ fontFamily: 'var(--ff-read)', fontSize: 'var(--fs-h3)', fontWeight: '700', lineHeight: '1.3' }}>
                            {m?.title}
                          </div>
                          <div style={{ fontSize: 'var(--fs-detail)', color: 'var(--text2)', lineHeight: '1.55', flex: '1' }}>
                            {m?.desc}
                          </div>
                          <div
                            style={{
                              display: 'flex',
                              justifyContent: 'space-between',
                              alignItems: 'center',
                              fontSize: 'var(--fs-small)',
                              color: 'var(--text3)',
                            }}
                          >
                            <span>
                              {m?.count}
                              {' questions'}
                            </span>
                            <span>{m?.best}</span>
                          </div>
                          <button type="button" className="solidbtn" onClick={m?.start} style={{ textAlign: 'center' }}>
                            {'Start module'}
                          </button>
                        </div>
                      </React.Fragment>
                    ))}
                  </div>
                </div>
                <div className="sect">
                  <div className="sect-h">
                    <h2 className="sect-t">{'Previous attempts'}</h2>
                    <span className="sect-n">{'Last 8 · saved in this browser only'}</span>
                  </div>
                  <div className="card" style={{ padding: '18px 22px' }}>
                    {v.noHistory ? (
                      <>
                        <div style={{ fontSize: 'var(--fs-detail)', color: 'var(--text2)' }}>
                          {'No attempts yet. Finished modules appear here.'}
                        </div>
                      </>
                    ) : null}{' '}
                    {v.hasHistory ? (
                      <>
                        <div style={{ fontSize: 'var(--fs-detail)', color: 'var(--text2)', marginBottom: '12px' }}>
                          {'Attempts '}
                          <b style={{ color: 'var(--text)' }}>{v.hCount}</b>
                          {' · Average '}
                          <b style={{ color: 'var(--text)' }}>{v.hAvg}</b>
                          {' · Best '}
                          <b style={{ color: 'var(--text)' }}>{v.hBest}</b>
                        </div>
                        <div
                          style={{
                            display: 'grid',
                            gridTemplateColumns: 'minmax(120px,1.3fr) minmax(110px,1.2fr) minmax(90px,1fr) minmax(70px,.7fr)',
                            gap: '0 14px',
                            fontSize: 'var(--fs-detail)',
                            alignItems: 'center',
                          }}
                        >
                          <div
                            style={{
                              fontSize: '13px',
                              fontWeight: '500',
                              color: 'var(--text3)',
                              paddingBottom: '8px',
                              borderBottom: '1px solid var(--divider)',
                            }}
                          >
                            {'Date'}
                          </div>
                          <div
                            style={{
                              fontSize: '13px',
                              fontWeight: '500',
                              color: 'var(--text3)',
                              paddingBottom: '8px',
                              borderBottom: '1px solid var(--divider)',
                            }}
                          >
                            {'Module'}
                          </div>
                          <div
                            style={{
                              fontSize: '13px',
                              fontWeight: '500',
                              color: 'var(--text3)',
                              paddingBottom: '8px',
                              borderBottom: '1px solid var(--divider)',
                            }}
                          >
                            {'Score'}
                          </div>
                          <div
                            style={{
                              fontSize: '13px',
                              fontWeight: '500',
                              color: 'var(--text3)',
                              paddingBottom: '8px',
                              borderBottom: '1px solid var(--divider)',
                            }}
                          >
                            {'Result'}
                          </div>
                          {(v.history || []).map((h, $index) => (
                            <React.Fragment key={$index}>
                              <div style={{ padding: '9px 0', borderBottom: '1px solid var(--divider)', color: 'var(--text2)' }}>
                                {h?.dateLabel}
                              </div>
                              <div style={{ padding: '9px 0', borderBottom: '1px solid var(--divider)', color: 'var(--text2)' }}>
                                {h?.moduleLabel}
                              </div>
                              <div
                                className="mono"
                                style={{ padding: '9px 0', borderBottom: '1px solid var(--divider)', fontWeight: '500' }}
                              >
                                {h?.scoreLabel}
                              </div>
                              <div style={css(`padding:9px 0;border-bottom:1px solid var(--divider);font-weight:500;color:${h?.color}`)}>
                                {h?.resultLabel}
                              </div>
                            </React.Fragment>
                          ))}
                        </div>
                        <div style={{ marginTop: '12px', textAlign: 'right' }}>
                          <button
                            type="button"
                            onClick={v.clearHistory}
                            style={{ all: 'unset', cursor: 'pointer', fontSize: '13px', fontWeight: '500', color: 'var(--text3)' }}
                          >
                            {'Clear attempt history'}
                          </button>
                        </div>
                      </>
                    ) : null}
                  </div>
                </div>
                <div className="sect">
                  <div className="sect-h">
                    <h2 className="sect-t">{'Know the rules'}</h2>
                    <span className="sect-n">{'Applies to every timed attempt'}</span>
                  </div>
                  <div className="cp2e">
                    <div className="card rules">
                      <h3>{'Full mock exam'}</h3>
                      <div className="setrow">
                        <span className="setlab">{'Items'}</span>
                        <span className="setval">{'50 multiple choice'}</span>
                      </div>
                      <div className="setrow">
                        <span className="setlab">{'Time'}</span>
                        <span className="setval">{'60 minutes'}</span>
                      </div>
                      <div className="setrow">
                        <span className="setlab">{'Pass mark'}</span>
                        <span className="setval">{'35 of 50 (70%)'}</span>
                      </div>
                      <div className="setrow">
                        <span className="setlab">{'Materials'}</span>
                        <span className="setval">{'None (closed book)'}</span>
                      </div>
                      <div className="setrow">
                        <span className="setlab">{'Opens'}</span>
                        <span className="setval">{'Week 19, released by your instructor'}</span>
                      </div>
                    </div>
                    <div className="card rules">
                      <h3>{'Your responsibilities'}</h3>
                      <div className="setrow">
                        <span className="setlab">{'Integrity'}</span>
                        <span className="setval">{'Work on your own. Disclose any AI help, as the course policy sets out.'}</span>
                      </div>
                      <div className="setrow">
                        <span className="setlab">{'Attempts'}</span>
                        <span className="setval">{'Practice modules: unlimited. Full midterm: once.'}</span>
                      </div>
                      <div className="setrow">
                        <span className="setlab">{'Accommodations'}</span>
                        <span className="setval">
                          {'Ask the '}
                          <DcLink href="mailto:anthony.jose@supervaise.io">{'programme office'}</DcLink>
                          {' before the exam date.'}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </>
            ) : null}{' '}
            {v.examView ? (
              <>
                <div
                  style={{
                    display: 'flex',
                    flexWrap: 'wrap',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    gap: '12px',
                    marginBottom: '18px',
                  }}
                >
                  <div>
                    <div className="eyebrow">{v.modTag}</div>
                    <div style={{ fontSize: '19px', fontWeight: '600', lineHeight: '1.3' }}>{v.modTitle}</div>
                  </div>
                  <div
                    className="card mono"
                    style={css(`padding:9px 18px;font-size:18px;font-weight:500;color:${v.timerColor};border-color:${v.timerBorder}`)}
                  >
                    {'Time remaining: '}
                    {v.timeLeft}
                  </div>
                </div>
                <div
                  className="examgrid"
                  style={{ display: 'grid', gridTemplateColumns: 'minmax(0,1fr) 230px', gap: '18px', alignItems: 'start' }}
                >
                  <div>
                    <div className="card" style={{ padding: '26px', marginBottom: '14px' }}>
                      <div
                        style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          gap: '10px',
                          marginBottom: '12px',
                          fontSize: '14px',
                          color: 'var(--text3)',
                        }}
                      >
                        <span>
                          {'Question '}
                          {v.qNum}
                          {' of '}
                          {v.qTotal}
                          {' · '}
                          {v.qDomain}
                        </span>
                        <span className="mono">{v.qId}</span>
                      </div>
                      <div style={{ fontSize: '16px', fontWeight: '500', marginBottom: '18px', lineHeight: '1.55' }}>{v.qText}</div>
                      {(v.opts || []).map((o, $index) => (
                        <React.Fragment key={$index}>
                          <button
                            type="button"
                            className="optline"
                            onClick={o?.pick}
                            style={css(`background:${o?.bg};border:1px solid ${o?.border}`)}
                          >
                            <span className="optkey" style={css(`background:${o?.keyBg};color:${o?.keyFg}`)}>
                              {o?.key}
                            </span>
                            <span>{o?.text}</span>
                          </button>
                        </React.Fragment>
                      ))}
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', gap: '10px', flexWrap: 'wrap' }}>
                      <button type="button" className="ghostbtn" onClick={v.prev}>
                        {'Previous question'}
                      </button>
                      <button
                        type="button"
                        className="ghostbtn"
                        onClick={v.toggleFlag}
                        style={css(`color:${v.flagFg};border-color:${v.flagBorder}`)}
                      >
                        {v.flagLabel}
                      </button>
                      {v.notLast ? (
                        <>
                          <button type="button" className="ghostbtn" onClick={v.next}>
                            {'Next question'}
                          </button>
                        </>
                      ) : null}
                      {v.isLast ? (
                        <>
                          <button type="button" className="solidbtn" onClick={v.finish}>
                            {'Submit module for scoring'}
                          </button>
                        </>
                      ) : null}
                    </div>
                  </div>
                  <div className="card" style={{ padding: '18px' }}>
                    <div style={{ fontSize: '14px', fontWeight: '600', marginBottom: '10px' }}>{'Question navigator'}</div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '7px', marginBottom: '12px' }}>
                      {(v.navs || []).map((n, $index) => (
                        <React.Fragment key={$index}>
                          <button
                            type="button"
                            className="navq"
                            onClick={n?.go}
                            style={css(`background:${n?.bg};color:${n?.fg};border:${n?.border}`)}
                          >
                            {n?.label}
                          </button>
                        </React.Fragment>
                      ))}
                    </div>
                    <div
                      style={{
                        display: 'flex',
                        flexDirection: 'column',
                        gap: '5px',
                        fontSize: '13px',
                        color: 'var(--text2)',
                        marginBottom: '14px',
                      }}
                    >
                      <span>
                        <span
                          style={{
                            display: 'inline-block',
                            width: '10px',
                            height: '10px',
                            borderRadius: '3px',
                            background: 'var(--accent-cta-bg)',
                            marginRight: '6px',
                          }}
                        ></span>
                        {'Answered · '}
                        {v.answeredCount}
                        {'/'}
                        {v.qTotal}
                      </span>
                      <span>
                        <span
                          style={{
                            display: 'inline-block',
                            width: '10px',
                            height: '10px',
                            borderRadius: '3px',
                            background: 'var(--amber-soft-bg)',
                            border: '1px solid var(--amber)',
                            marginRight: '6px',
                          }}
                        ></span>
                        {'Flagged for review'}
                      </span>
                      <span>
                        <span
                          style={{
                            display: 'inline-block',
                            width: '10px',
                            height: '10px',
                            borderRadius: '3px',
                            background: 'var(--surface2)',
                            border: '1px solid var(--surface2-border)',
                            marginRight: '6px',
                          }}
                        ></span>
                        {'Not answered'}
                      </span>
                    </div>
                    <button
                      type="button"
                      className="solidbtn"
                      onClick={v.finish}
                      style={{ width: '100%', boxSizing: 'border-box', textAlign: 'center' }}
                    >
                      {'Submit module'}
                    </button>{' '}
                    <button
                      type="button"
                      onClick={v.quit}
                      style={{
                        all: 'unset',
                        cursor: 'pointer',
                        display: 'block',
                        textAlign: 'center',
                        width: '100%',
                        fontSize: '14px',
                        fontWeight: '500',
                        color: 'var(--text3)',
                        marginTop: '10px',
                      }}
                    >
                      {'Exit without submitting'}
                    </button>
                  </div>
                </div>
                <div style={{ fontSize: '13px', color: 'var(--text3)', marginTop: '16px' }}>
                  {
                    'Answers remain locked until submission. The live platform enforces this on the server; this preview runs in the browser only.'
                  }
                </div>
              </>
            ) : null}{' '}
            {v.resultView ? (
              <>
                <div
                  className="card"
                  style={{ padding: '26px', marginBottom: '20px', display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '24px' }}
                >
                  <div style={{ position: 'relative', width: '110px', height: '110px', flexShrink: '0' }}>
                    <svg width="110" height="110" viewBox="0 0 110 110" style={{ transform: 'rotate(-90deg)' }}>
                      <circle cx="55" cy="55" r="46" fill="none" strokeWidth="10" style={{ stroke: 'var(--track)' }}></circle>
                      <circle
                        cx="55"
                        cy="55"
                        r="46"
                        fill="none"
                        strokeWidth="10"
                        strokeLinecap="round"
                        strokeDasharray="289.03"
                        strokeDashoffset={v.ringOffset}
                        style={css(`stroke:${v.ringColor};transition:stroke-dashoffset .6s ease`)}
                      ></circle>
                    </svg>
                    <div
                      style={{
                        position: 'absolute',
                        inset: '0',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center',
                      }}
                    >
                      <div className="mono" style={{ fontSize: '24px', fontWeight: '500' }}>
                        {v.scorePct}
                      </div>
                      <div style={{ fontSize: '13px', color: 'var(--text3)' }}>{v.scoreLine}</div>
                    </div>
                  </div>
                  <div style={{ flex: '1 1 320px', minWidth: '0' }}>
                    <div className="eyebrow">
                      {v.modTag}
                      {' · Results'}
                    </div>
                    <div style={{ fontSize: '20px', fontWeight: '600', margin: '2px 0 6px', lineHeight: '1.3' }}>{v.verdict}</div>
                    <div style={{ fontSize: '15px', color: 'var(--text2)' }}>
                      {v.verdictSub}
                      {' Select Explain on any question for its explanation.'}
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                    <button type="button" className="solidbtn" onClick={v.retry}>
                      {'Repeat module'}
                    </button>
                    <button type="button" className="ghostbtn" onClick={v.goHome}>
                      {'Return to examination preparation'}
                    </button>
                  </div>
                </div>
                <div className="sect-h">
                  <h2 className="sect-t">{'Question review'}</h2>
                  <span className="sect-n">
                    {'Score '}
                    {v.scoreLine}
                  </span>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                  {(v.review || []).map((r, $index) => (
                    <React.Fragment key={$index}>
                      <div className="card reviewrow">
                        <span className="optkey" style={css(`background:${r?.markBg};color:${r?.markFg}`)}>
                          {r?.icon}
                        </span>
                        <div style={{ flex: '1', minWidth: '0' }}>
                          <div style={{ fontSize: '13px', color: 'var(--text3)', marginBottom: '2px' }}>
                            {'Q'}
                            {r?.n}
                            {' · '}
                            {r?.domain}
                            {' · '}
                            <span style={css(`color:${r?.markFg};font-weight:500`)}>{r?.mark}</span>
                          </div>
                          <div style={{ fontSize: '15px', fontWeight: '500', lineHeight: '1.5' }}>{r?.question}</div>
                        </div>
                        <button
                          type="button"
                          className="ghostbtn"
                          onClick={r?.explain}
                          style={{ flexShrink: '0', padding: '8px 14px', fontSize: '14px' }}
                        >
                          {'Explain'}
                        </button>
                      </div>
                    </React.Fragment>
                  ))}
                </div>
              </>
            ) : null}{' '}
            {v.modalOpen ? (
              <>
                <div
                  style={{
                    position: 'fixed',
                    inset: '0',
                    zIndex: '50',
                    background: 'rgba(9,10,15,0.55)',
                    backdropFilter: 'blur(3px)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    padding: '20px',
                  }}
                >
                  <div
                    className="card"
                    style={{
                      width: '100%',
                      maxWidth: '560px',
                      padding: '26px',
                      background: 'var(--card-bg)',
                      boxShadow: '0 24px 60px rgba(0,0,0,0.35)',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                      <span className="badge" style={css(`background:${v.mdBg};color:${v.mdFg};border:1px solid ${v.mdFg}`)}>
                        {v.mdMark}
                      </span>
                      <button
                        type="button"
                        onClick={v.closeModal}
                        aria-label="Close"
                        style={{ all: 'unset', cursor: 'pointer', fontSize: '20px', lineHeight: '1', color: 'var(--text3)' }}
                      >
                        {'×'}
                      </button>
                    </div>
                    <div style={{ fontSize: '15px', fontWeight: '500', lineHeight: '1.55', marginBottom: '14px' }}>{v.mdQuestion}</div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '14px', fontSize: '15px' }}>
                      <div
                        style={{
                          padding: '10px 12px',
                          borderRadius: '9px',
                          background: 'var(--surface2)',
                          border: '1px solid var(--surface2-border)',
                        }}
                      >
                        <span style={{ color: 'var(--text3)', fontSize: '13px', fontWeight: '500', display: 'block' }}>
                          {'Your answer'}
                        </span>
                        {v.mdYours}
                      </div>
                      <div
                        style={{
                          padding: '10px 12px',
                          borderRadius: '9px',
                          background: 'var(--accent-soft-bg)',
                          border: '1px solid var(--accent-soft-border)',
                        }}
                      >
                        <span style={{ color: 'var(--accent-soft-text)', fontSize: '13px', fontWeight: '500', display: 'block' }}>
                          {'Correct answer'}
                        </span>
                        {v.mdCorrect}
                      </div>
                    </div>
                    <div style={{ fontSize: '15px', color: 'var(--text2)', lineHeight: '1.7', marginBottom: '12px' }}>{v.mdExplain}</div>
                    <div className="mono" style={{ fontSize: '13px', color: 'var(--text3)' }}>
                      {v.mdSource}
                    </div>
                  </div>
                </div>
              </>
            ) : null}
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
Component.css = pageCss;
Component.pageKey = 'mock-exam';
Component.persist = ['history'];

export default Component;
