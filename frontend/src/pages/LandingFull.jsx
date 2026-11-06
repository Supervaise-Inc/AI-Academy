// Generated from design-source/LandingFull.dc.html by tools/convert.mjs.
// Re-running the converter regenerates (and overwrites) this file, so edit design-source/ + tools/patches.mjs instead,
// or stop using the converter and edit this file directly.
/* eslint-disable */
import React from 'react';
import DCLogic from '../dc/DCLogic.jsx';
import DcLink from '../dc/DcLink.jsx';
import { css } from '../dc/css.js';
import pageCss from './LandingFull.css?inline';

// Reveal beats (ms from mount). The phrase owns the room first, the supporting
// line answers it, and the partnership credit settles in last.
const PHRASE_AT = 400;
const BLURB_AT = 1500;
const LOCK_AT = 2500;

// Below the hero: tokeniser demo, programme rail, credentials, welcome closer.
const DEMO = 'Students in Cebu are learning generative AI.';
const STEP_MS = 1800;
const COMMON = { the: 1, a: 1, in: 1, are: 1, is: 1, and: 1, of: 1, to: 1, ai: 1, students: 1, learning: 1 };
const PALETTE = ['#2E4A12', '#123A4A', '#3D2A4A', '#4A3412', '#1E3D2A', '#3A1E2A'];
const STEPS = [
  { title: 'Split', body: 'Text breaks into tokens.' },
  { title: 'Number', body: 'Each token gets an ID.' },
  { title: 'Predict', body: 'The model guesses the next ID.' },
];
// Welcome line, one entry per revealed word; "after" is the space that follows it.
const WORDS = [
  ['Welcome', ' '],
  ['to', ' '],
  ['your', ' '],
  ['first', ' '],
  ['step', ' '],
  ['into', ' '],
  ['the', ' '],
  ['Age', ' '],
  ['of', ' '],
  ['AI', '', true],
  ['.', ''],
];
const RV_KEYS = [
  'tryHead',
  'tryP1',
  'tryP2',
  'progHead',
  'terms',
  'term1',
  'term2',
  'term3',
  'term4',
  'credHead',
  'cred1',
  'bridge',
  'cred2',
  'fine',
  'welcome',
];
const NODE_KEYS = ['n1', 'n2', 'n3', 'n4'];
const RV_THRESHOLD = 0.18;
const LAB_THRESHOLD = 0.4;

function hashStr(s) {
  let h = 0;
  for (let k = 0; k < s.length; k++) {
    h = (h * 31 + s.charCodeAt(k)) >>> 0;
  }
  return h;
}
function tokenize(t) {
  const out = [];
  (t.match(/[A-Za-z]+|\d+|[^\sA-Za-z\d]/g) || []).forEach((w, j) => {
    const pre = j > 0 && /[A-Za-z\d]/.test(w) ? ' ' : '';
    if (w.length <= 6 || COMMON[w.toLowerCase()]) {
      out.push(pre + w);
      return;
    }
    const c = Math.ceil(w.length / 2);
    out.push(pre + w.slice(0, c));
    out.push(w.slice(c));
  });
  return out;
}
// Real measurements from GPT-2 (small, 124 million parameters), the model used in the Week 1 lab:
// the prompt's tokens with their token IDs, and the model's five most likely next tokens with their probabilities.
const SAMPLES = [
  {
    prompt: 'The capital of France is',
    toks: [['The', 464], [' capital', 3139], [' of', 286], [' France', 4881], [' is', 318]],
    out: [[' the', 0.085], [' now', 0.048], [' a', 0.046], [' France', 0.032], [' Paris', 0.032]],
    note: 'Fluent, but not the right answer: “ Paris” is only fifth. The model predicts likely words, not true ones. Week 1 · Lab Part 1.',
  },
  {
    prompt: 'To make tea, first boil the',
    toks: [['To', 2514], [' make', 787], [' tea', 8887], [',', 11], [' first', 717], [' boil', 20667], [' the', 262]],
    out: [[' water', 0.323], [' milk', 0.104], [' tea', 0.066], [' beans', 0.033], [' sugar', 0.024]],
    note: 'Here the context is clear, so the model is much surer: “ water” gets 32%. Week 1 · Lab Part 1.',
  },
  {
    prompt: 'Once upon a time there was a',
    toks: [['Once', 7454], [' upon', 2402], [' a', 257], [' time', 640], [' there', 612], [' was', 373], [' a', 257]],
    out: [[' great', 0.04], [' man', 0.039], [' war', 0.012], [' woman', 0.011], [' certain', 0.011]],
    note: 'Many continuations are fine, so no word gets more than 4%. Sampling picks among them. Week 1 · temperature and top-k.',
  },
  {
    prompt: '2 + 2 =',
    toks: [['2', 17], [' +', 1343], [' 2', 362], [' =', 796]],
    out: [[' 3', 0.102], [' 1', 0.092], [' 2', 0.084], [' 0', 0.075], [' 4', 0.055]],
    note: 'A small language model is not a calculator: “ 4” is only fifth. Week 1 · Lab Part 1.',
  },
];
const TYPE_MS = 55;
const HOLD_MS = 5200;
// Seconds (after the prompt has been typed) at which each stage of the sample animation starts.
function animTimes(x) {
  const tok = 0.28;
  const think = x.toks.length * tok + 0.15;
  const out = think + 0.7;
  return { tok, think, out, end: out + x.out.length * 0.2 + 0.7 };
}
function visibleFraction(r, vh) {
  if (!r || r.height <= 0) return 0;
  const seen = Math.min(r.bottom, vh) - Math.max(r.top, 0);
  return seen <= 0 ? 0 : seen / r.height;
}
function allTrue(keys) {
  const o = {};
  keys.forEach((k) => {
    o[k] = true;
  });
  return o;
}

class Component extends DCLogic {
  constructor(props) {
    super(props);
    let reduce = false;
    try {
      reduce = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
    } catch (e) {}
    this._reduce = reduce;
    this.state = {
      showPhrase: reduce,
      showBlurb: reduce,
      showLock: reduce,
      panel: null,
      text: reduce ? DEMO : '',
      step: -1,
      sample: 0,
      typed: reduce ? 999 : 0,
      run: 0,
      rv: reduce ? allTrue(RV_KEYS) : {},
      lit: reduce ? allTrue(NODE_KEYS) : {},
      guideH: 0,
      fillH: 0,
    };
    this.openLogin = () => this.openPanel('login');
    this.openSignup = () => this.openPanel('signup');
    this.closePanel = () => this.setState({ panel: null });
    this.switchToLogin = () => this.openPanel('login');
    this._key = (e) => {
      if (e.key === 'Escape' && this.state.panel) this.closePanel();
    };
    this._timers = [];
    this._stepTimer = null;
    this._typed = false;
    this._raf = 0;
    this.onText = (e) => this.setState({ text: e.target.value });
    this.pickSample = (i) => (this._typed ? this._play(i) : this.setState({ sample: i, typed: 999 }));
    this.onFocus = () => {
      const t = this.state.text;
      if (t !== DEMO && t.length < DEMO.length) this.setState({ text: DEMO });
    };
    this._onScroll = () => {
      if (this._raf) return;
      try {
        this._raf = requestAnimationFrame(() => {
          this._raf = 0;
          this._measure();
        });
      } catch (e) {
        this._raf = 0;
        this._measure();
      }
    };
  }
  componentDidMount() {
    try {
      document.addEventListener('keydown', this._key);
    } catch (e) {}
    try {
      window.addEventListener('scroll', this._onScroll, { passive: true, capture: true });
      window.addEventListener('resize', this._onScroll, { passive: true });
    } catch (e) {}
    this._onScroll();
    // Re-measure once fonts and images have settled the layout.
    this._timers.push(setTimeout(this._onScroll, 300));
    this._timers.push(setTimeout(this._onScroll, 1200));
    if (this.state.showLock) return;
    this._timers.push(setTimeout(() => this.setState({ showPhrase: true }), PHRASE_AT));
    this._timers.push(setTimeout(() => this.setState({ showBlurb: true }), BLURB_AT));
    this._timers.push(setTimeout(() => this.setState({ showLock: true }), LOCK_AT));
  }
  componentWillUnmount() {
    this._timers.forEach((id) => clearTimeout(id));
    if (this._stepTimer) clearInterval(this._stepTimer);
    clearInterval(this._typeTimer);
    clearTimeout(this._nextTimer);
    try {
      if (this._raf) cancelAnimationFrame(this._raf);
    } catch (e) {}
    try {
      window.removeEventListener('scroll', this._onScroll, { capture: true });
      window.removeEventListener('resize', this._onScroll);
    } catch (e) {}
    try {
      document.removeEventListener('keydown', this._key);
    } catch (e) {}
  }
  openPanel(panel) {
    this.setState({ panel });
    setTimeout(() => {
      try {
        const i = document.querySelector('.panel input');
        if (i) i.focus();
      } catch (e) {}
    }, 60);
  }
  // Read-only layout pass: positions of the guide nodes and reveal blocks go into state.
  _measure() {
    let flow = null;
    try {
      flow = document.querySelector('#lf-flow');
    } catch (e) {}
    if (!flow) return;
    const vh = window.innerHeight || 0;
    const ft = flow.getBoundingClientRect().top;
    const ys = {};
    document.querySelectorAll('[data-node]').forEach((n) => {
      if (!n.getClientRects().length) return;
      ys[n.getAttribute('data-node')] = n.getBoundingClientRect().top - ft + 7;
    });
    const guideH = Math.max(0, Math.round(ys.n4 || 0));
    const fillH = this._reduce ? guideH : Math.round(Math.max(0, Math.min(guideH, vh * 0.62 - ft)));
    const next = {};
    let changed = guideH !== this.state.guideH || fillH !== this.state.fillH;
    if (!this._reduce) {
      const lit = {};
      NODE_KEYS.forEach((k) => {
        lit[k] = ys[k] != null && fillH >= ys[k] - 3;
        if (lit[k] !== !!this.state.lit[k]) changed = true;
      });
      next.lit = lit;
    }
    const rv = Object.assign({}, this.state.rv);
    document.querySelectorAll('[data-rv]').forEach((el) => {
      const k = el.getAttribute('data-rv');
      if (rv[k]) return;
      if (visibleFraction(el.getBoundingClientRect(), vh) >= RV_THRESHOLD) {
        rv[k] = true;
        changed = true;
      }
    });
    next.rv = rv;
    if (!this._typed) {
      const lab = document.querySelector('[data-lab]');
      if (lab && visibleFraction(lab.getBoundingClientRect(), vh) >= LAB_THRESHOLD) this._startDemo();
    }
    if (!changed) return;
    next.guideH = guideH;
    next.fillH = fillH;
    this.setState(next);
  }
  _cycleSteps() {
    this.setState({ step: (this.state.step + 1) % STEPS.length });
  }
  _inputFocused() {
    try {
      const a = document.activeElement;
      return !!(a && a.id === 'lf-t');
    } catch (e) {
      return false;
    }
  }
  // Plays the sample: the prompt types in, its tokens pop in one by one, the model "thinks", then the output bars grow.
  // After a pause it moves on to the next sample.
  _play(i) {
    clearInterval(this._typeTimer);
    clearTimeout(this._nextTimer);
    const x = SAMPLES[i];
    this.setState({ sample: i, typed: 0, run: this.state.run + 1 });
    let n = 0;
    this._typeTimer = setInterval(() => {
      n += 1;
      this.setState({ typed: n });
      if (n >= x.prompt.length) clearInterval(this._typeTimer);
    }, TYPE_MS);
    const total = x.prompt.length * TYPE_MS + animTimes(x).end * 1000 + HOLD_MS;
    this._nextTimer = setTimeout(() => this._play((i + 1) % SAMPLES.length), total);
  }
  _startDemo() {
    if (this._typed) return;
    this._typed = true;
    if (this._reduce) {
      this.setState({ typed: 999 });
      return;
    }
    this._play(this.state.sample);
  }
  renderVals() {
    const text = this.state.text || '';
    const tk = tokenize(text);
    const tokens = tk.map((t) => {
      const h = hashStr(t.trim().toLowerCase());
      return { text: t.replace(/ /g, '·'), idStr: String(1000 + (h % 49000)), style: 'background:' + PALETTE[h % PALETTE.length] };
    });
    const on = {};
    RV_KEYS.forEach((k) => {
      on[k] = this.state.rv[k] ? 'on' : '';
    });
    const lit = {};
    NODE_KEYS.forEach((k) => {
      lit[k] = this.state.lit[k] ? 'lit' : '';
    });
    return {
      phraseClass: this.state.showPhrase ? 'show' : '',
      blurbClass: this.state.showBlurb ? 'show' : '',
      lockClass: this.state.showLock ? 'show' : '',
      isLogin: this.state.panel === 'login',
      isSignup: this.state.panel === 'signup',
      openLogin: this.openLogin,
      openSignup: this.openSignup,
      closePanel: this.closePanel,
      switchToLogin: this.switchToLogin,
      text: text,
      onText: this.onText,
      onFocus: this.onFocus,
      tokens: tokens,
      chars: String(text.length),
      tokCount: String(tk.length),
      ratio: tk.length ? (text.length / tk.length).toFixed(1) : '0',
      samples: SAMPLES.map((x, i) => ({ prompt: x.prompt, cls: i === this.state.sample ? 'chip on' : 'chip', pick: () => this.pickSample(i) })),
      sample: (() => {
        const x = SAMPLES[this.state.sample];
        const chars = x.prompt.length;
        const anim = !this._reduce && this.state.typed < 999;
        const t0 = anim ? chars * (TYPE_MS / 1000) : 0;
        const tm = animTimes(x);
        const shown = Math.min(this.state.typed, chars);
        return {
          prompt: x.prompt.slice(0, shown),
          typing: shown < chars,
          note: x.note,
          chars: String(chars),
          count: String(x.toks.length),
          ratio: (chars / x.toks.length).toFixed(1),
          run: this.state.run,
          animCls: anim || (!this._reduce && this.state.run > 0) ? 'anim' : '',
          thinkStyle: '--td:' + (t0 + tm.think).toFixed(2) + 's',
          full: x.prompt,
          next: x.out[0][0],
          nextStyle: '--td:' + (t0 + tm.out + 0.5).toFixed(2) + 's',
          toks: x.toks.map(([t, id], k) => ({ text: t.replace(/ /g, '·'), idStr: String(id), style: 'background:' + PALETTE[hashStr(t.trim().toLowerCase()) % PALETTE.length] + ';--td:' + (t0 + k * tm.tok).toFixed(2) + 's' })),
          out: x.out.map(([w, pr], k) => ({ word: w.replace(/^ /, '·'), pct: (pr * 100).toFixed(pr < 0.1 ? 1 : 0) + '%', style: 'width:' + Math.max(3, Math.round(pr * 100 * 2.6)) + '%;--td:' + (t0 + tm.out + k * 0.2).toFixed(2) + 's', cls: k === 0 ? 'mbar top' : 'mbar' })),
        };
      })(),
      steps: STEPS.map((s, i) => ({ title: s.title, body: s.body, cls: i === this.state.step ? 'act' : '' })),
      words: WORDS.map((w, i) => ({ text: w[0], after: w[1], cls: w[2] ? 'w hl' : 'w', style: '--i:' + i })),
      on: on,
      lit: lit,
      guideStyle: 'height:' + this.state.guideH + 'px',
      fillStyle: 'height:' + this.state.fillH + 'px',
      headStyle: 'top:' + this.state.fillH + 'px',
    };
  }
}

function template(v) {
  return (
    <>
      <div className="lf-page">
        <div className="stage" id="top">
          <div className="floor"></div>
          <div className="waves">
            <svg width="4320" height="420" viewBox="0 0 4320 420" aria-hidden="true">
              <g fill="none" stroke="#76B900" strokeLinecap="round">
                <path
                  d="M0 70.0 L20 77.6 L40 84.7 L60 90.9 L80 95.7 L100 98.9 L120 100.4 L140 100.3 L160 98.7 L180 96.0 L200 92.5 L220 88.6 L240 84.6 L260 81.0 L280 77.8 L300 75.1 L320 73.0 L340 71.4 L360 70.0 L380 68.6 L400 67.0 L420 64.9 L440 62.2 L460 59.0 L480 55.4 L500 51.4 L520 47.5 L540 44.0 L560 41.3 L580 39.7 L600 39.6 L620 41.1 L640 44.3 L660 49.1 L680 55.3 L700 62.4 L720 70.0 L740 77.6 L760 84.7 L780 90.9 L800 95.7 L820 98.9 L840 100.4 L860 100.3 L880 98.7 L900 96.0 L920 92.5 L940 88.6 L960 84.6 L980 81.0 L1000 77.8 L1020 75.1 L1040 73.0 L1060 71.4 L1080 70.0 L1100 68.6 L1120 67.0 L1140 64.9 L1160 62.2 L1180 59.0 L1200 55.4 L1220 51.4 L1240 47.5 L1260 44.0 L1280 41.3 L1300 39.7 L1320 39.6 L1340 41.1 L1360 44.3 L1380 49.1 L1400 55.3 L1420 62.4 L1440 70.0 L1460 77.6 L1480 84.7 L1500 90.9 L1520 95.7 L1540 98.9 L1560 100.4 L1580 100.3 L1600 98.7 L1620 96.0 L1640 92.5 L1660 88.6 L1680 84.6 L1700 81.0 L1720 77.8 L1740 75.1 L1760 73.0 L1780 71.4 L1800 70.0 L1820 68.6 L1840 67.0 L1860 64.9 L1880 62.2 L1900 59.0 L1920 55.4 L1940 51.4 L1960 47.5 L1980 44.0 L2000 41.3 L2020 39.7 L2040 39.6 L2060 41.1 L2080 44.3 L2100 49.1 L2120 55.3 L2140 62.4 L2160 70.0 L2180 77.6 L2200 84.7 L2220 90.9 L2240 95.7 L2260 98.9 L2280 100.4 L2300 100.3 L2320 98.7 L2340 96.0 L2360 92.5 L2380 88.6 L2400 84.6 L2420 81.0 L2440 77.8 L2460 75.1 L2480 73.0 L2500 71.4 L2520 70.0 L2540 68.6 L2560 67.0 L2580 64.9 L2600 62.2 L2620 59.0 L2640 55.4 L2660 51.4 L2680 47.5 L2700 44.0 L2720 41.3 L2740 39.7 L2760 39.6 L2780 41.1 L2800 44.3 L2820 49.1 L2840 55.3 L2860 62.4 L2880 70.0 L2900 77.6 L2920 84.7 L2940 90.9 L2960 95.7 L2980 98.9 L3000 100.4 L3020 100.3 L3040 98.7 L3060 96.0 L3080 92.5 L3100 88.6 L3120 84.6 L3140 81.0 L3160 77.8 L3180 75.1 L3200 73.0 L3220 71.4 L3240 70.0 L3260 68.6 L3280 67.0 L3300 64.9 L3320 62.2 L3340 59.0 L3360 55.4 L3380 51.4 L3400 47.5 L3420 44.0 L3440 41.3 L3460 39.7 L3480 39.6 L3500 41.1 L3520 44.3 L3540 49.1 L3560 55.3 L3580 62.4 L3600 70.0 L3620 77.6 L3640 84.7 L3660 90.9 L3680 95.7 L3700 98.9 L3720 100.4 L3740 100.3 L3760 98.7 L3780 96.0 L3800 92.5 L3820 88.6 L3840 84.6 L3860 81.0 L3880 77.8 L3900 75.1 L3920 73.0 L3940 71.4 L3960 70.0 L3980 68.6 L4000 67.0 L4020 64.9 L4040 62.2 L4060 59.0 L4080 55.4 L4100 51.4 L4120 47.5 L4140 44.0 L4160 41.3 L4180 39.7 L4200 39.6 L4220 41.1 L4240 44.3 L4260 49.1 L4280 55.3 L4300 62.4 L4320 70.0"
                  style={{ opacity: '0.18' }}
                  strokeWidth="1.2"
                ></path>
                <path
                  d="M0 156.7 L20 158.4 L40 158.0 L60 155.5 L80 151.6 L100 146.7 L120 141.3 L140 135.9 L160 130.9 L180 126.7 L200 123.2 L220 120.5 L240 118.3 L260 116.4 L280 114.4 L300 112.1 L320 109.1 L340 105.4 L360 100.9 L380 95.8 L400 90.3 L420 85.0 L440 80.4 L460 76.9 L480 75.0 L500 75.2 L520 77.6 L540 82.4 L560 89.3 L580 97.9 L600 107.8 L620 118.3 L640 128.7 L660 138.3 L680 146.4 L700 152.7 L720 156.7 L740 158.4 L760 158.0 L780 155.5 L800 151.6 L820 146.7 L840 141.3 L860 135.9 L880 130.9 L900 126.7 L920 123.2 L940 120.5 L960 118.3 L980 116.4 L1000 114.4 L1020 112.1 L1040 109.1 L1060 105.4 L1080 100.9 L1100 95.8 L1120 90.3 L1140 85.0 L1160 80.4 L1180 76.9 L1200 75.0 L1220 75.2 L1240 77.6 L1260 82.4 L1280 89.3 L1300 97.9 L1320 107.8 L1340 118.3 L1360 128.7 L1380 138.3 L1400 146.4 L1420 152.7 L1440 156.7 L1460 158.4 L1480 158.0 L1500 155.5 L1520 151.6 L1540 146.7 L1560 141.3 L1580 135.9 L1600 130.9 L1620 126.7 L1640 123.2 L1660 120.5 L1680 118.3 L1700 116.4 L1720 114.4 L1740 112.1 L1760 109.1 L1780 105.4 L1800 100.9 L1820 95.8 L1840 90.3 L1860 85.0 L1880 80.4 L1900 76.9 L1920 75.0 L1940 75.2 L1960 77.6 L1980 82.4 L2000 89.3 L2020 97.9 L2040 107.8 L2060 118.3 L2080 128.7 L2100 138.3 L2120 146.4 L2140 152.7 L2160 156.7 L2180 158.4 L2200 158.0 L2220 155.5 L2240 151.6 L2260 146.7 L2280 141.3 L2300 135.9 L2320 130.9 L2340 126.7 L2360 123.2 L2380 120.5 L2400 118.3 L2420 116.4 L2440 114.4 L2460 112.1 L2480 109.1 L2500 105.4 L2520 100.9 L2540 95.8 L2560 90.3 L2580 85.0 L2600 80.4 L2620 76.9 L2640 75.0 L2660 75.2 L2680 77.6 L2700 82.4 L2720 89.3 L2740 97.9 L2760 107.8 L2780 118.3 L2800 128.7 L2820 138.3 L2840 146.4 L2860 152.7 L2880 156.7 L2900 158.4 L2920 158.0 L2940 155.5 L2960 151.6 L2980 146.7 L3000 141.3 L3020 135.9 L3040 130.9 L3060 126.7 L3080 123.2 L3100 120.5 L3120 118.3 L3140 116.4 L3160 114.4 L3180 112.1 L3200 109.1 L3220 105.4 L3240 100.9 L3260 95.8 L3280 90.3 L3300 85.0 L3320 80.4 L3340 76.9 L3360 75.0 L3380 75.2 L3400 77.6 L3420 82.4 L3440 89.3 L3460 97.9 L3480 107.8 L3500 118.3 L3520 128.7 L3540 138.3 L3560 146.4 L3580 152.7 L3600 156.7 L3620 158.4 L3640 158.0 L3660 155.5 L3680 151.6 L3700 146.7 L3720 141.3 L3740 135.9 L3760 130.9 L3780 126.7 L3800 123.2 L3820 120.5 L3840 118.3 L3860 116.4 L3880 114.4 L3900 112.1 L3920 109.1 L3940 105.4 L3960 100.9 L3980 95.8 L4000 90.3 L4020 85.0 L4040 80.4 L4060 76.9 L4080 75.0 L4100 75.2 L4120 77.6 L4140 82.4 L4160 89.3 L4180 97.9 L4200 107.8 L4220 118.3 L4240 128.7 L4260 138.3 L4280 146.4 L4300 152.7 L4320 156.7"
                  style={{ opacity: '0.35' }}
                  strokeWidth="2.1"
                ></path>
                <path
                  d="M0 188.8 L20 184.1 L40 179.4 L60 175.2 L80 171.6 L100 168.6 L120 166.3 L140 164.5 L160 162.8 L180 161.1 L200 159.0 L220 156.3 L240 152.9 L260 148.9 L280 144.3 L300 139.6 L320 135.0 L340 131.0 L360 128.2 L380 126.8 L400 127.3 L420 129.7 L440 134.2 L460 140.5 L480 148.2 L500 157.0 L520 166.2 L540 175.2 L560 183.5 L580 190.3 L600 195.5 L620 198.7 L640 199.9 L660 199.2 L680 196.9 L700 193.3 L720 188.8 L740 184.1 L760 179.4 L780 175.2 L800 171.6 L820 168.6 L840 166.3 L860 164.5 L880 162.8 L900 161.1 L920 159.0 L940 156.3 L960 152.9 L980 148.9 L1000 144.3 L1020 139.6 L1040 135.0 L1060 131.0 L1080 128.2 L1100 126.8 L1120 127.3 L1140 129.7 L1160 134.2 L1180 140.5 L1200 148.2 L1220 157.0 L1240 166.2 L1260 175.2 L1280 183.5 L1300 190.3 L1320 195.5 L1340 198.7 L1360 199.9 L1380 199.2 L1400 196.9 L1420 193.3 L1440 188.8 L1460 184.1 L1480 179.4 L1500 175.2 L1520 171.6 L1540 168.6 L1560 166.3 L1580 164.5 L1600 162.8 L1620 161.1 L1640 159.0 L1660 156.3 L1680 152.9 L1700 148.9 L1720 144.3 L1740 139.6 L1760 135.0 L1780 131.0 L1800 128.2 L1820 126.8 L1840 127.3 L1860 129.7 L1880 134.2 L1900 140.5 L1920 148.2 L1940 157.0 L1960 166.2 L1980 175.2 L2000 183.5 L2020 190.3 L2040 195.5 L2060 198.7 L2080 199.9 L2100 199.2 L2120 196.9 L2140 193.3 L2160 188.8 L2180 184.1 L2200 179.4 L2220 175.2 L2240 171.6 L2260 168.6 L2280 166.3 L2300 164.5 L2320 162.8 L2340 161.1 L2360 159.0 L2380 156.3 L2400 152.9 L2420 148.9 L2440 144.3 L2460 139.6 L2480 135.0 L2500 131.0 L2520 128.2 L2540 126.8 L2560 127.3 L2580 129.7 L2600 134.2 L2620 140.5 L2640 148.2 L2660 157.0 L2680 166.2 L2700 175.2 L2720 183.5 L2740 190.3 L2760 195.5 L2780 198.7 L2800 199.9 L2820 199.2 L2840 196.9 L2860 193.3 L2880 188.8 L2900 184.1 L2920 179.4 L2940 175.2 L2960 171.6 L2980 168.6 L3000 166.3 L3020 164.5 L3040 162.8 L3060 161.1 L3080 159.0 L3100 156.3 L3120 152.9 L3140 148.9 L3160 144.3 L3180 139.6 L3200 135.0 L3220 131.0 L3240 128.2 L3260 126.8 L3280 127.3 L3300 129.7 L3320 134.2 L3340 140.5 L3360 148.2 L3380 157.0 L3400 166.2 L3420 175.2 L3440 183.5 L3460 190.3 L3480 195.5 L3500 198.7 L3520 199.9 L3540 199.2 L3560 196.9 L3580 193.3 L3600 188.8 L3620 184.1 L3640 179.4 L3660 175.2 L3680 171.6 L3700 168.6 L3720 166.3 L3740 164.5 L3760 162.8 L3780 161.1 L3800 159.0 L3820 156.3 L3840 152.9 L3860 148.9 L3880 144.3 L3900 139.6 L3920 135.0 L3940 131.0 L3960 128.2 L3980 126.8 L4000 127.3 L4020 129.7 L4040 134.2 L4060 140.5 L4080 148.2 L4100 157.0 L4120 166.2 L4140 175.2 L4160 183.5 L4180 190.3 L4200 195.5 L4220 198.7 L4240 199.9 L4260 199.2 L4280 196.9 L4300 193.3 L4320 188.8"
                  style={{ opacity: '0.51' }}
                  strokeWidth="3.0"
                ></path>
                <path
                  d="M0 213.0 L20 211.6 L40 210.5 L60 209.5 L80 208.4 L100 207.1 L120 205.4 L140 203.2 L160 200.7 L180 197.9 L200 195.0 L220 192.2 L240 189.8 L260 188.2 L280 187.5 L300 188.0 L320 189.7 L340 192.6 L360 196.7 L380 201.5 L400 207.0 L420 212.7 L440 218.1 L460 223.1 L480 227.2 L500 230.1 L520 231.9 L540 232.5 L560 231.9 L580 230.3 L600 228.0 L620 225.2 L640 222.3 L660 219.5 L680 216.9 L700 214.7 L720 213.0 L740 211.6 L760 210.5 L780 209.5 L800 208.4 L820 207.1 L840 205.4 L860 203.2 L880 200.7 L900 197.9 L920 195.0 L940 192.2 L960 189.8 L980 188.2 L1000 187.5 L1020 188.0 L1040 189.7 L1060 192.6 L1080 196.7 L1100 201.5 L1120 207.0 L1140 212.7 L1160 218.1 L1180 223.1 L1200 227.2 L1220 230.1 L1240 231.9 L1260 232.5 L1280 231.9 L1300 230.3 L1320 228.0 L1340 225.2 L1360 222.3 L1380 219.5 L1400 216.9 L1420 214.7 L1440 213.0 L1460 211.6 L1480 210.5 L1500 209.5 L1520 208.4 L1540 207.1 L1560 205.4 L1580 203.2 L1600 200.7 L1620 197.9 L1640 195.0 L1660 192.2 L1680 189.8 L1700 188.2 L1720 187.5 L1740 188.0 L1760 189.7 L1780 192.6 L1800 196.7 L1820 201.5 L1840 207.0 L1860 212.7 L1880 218.1 L1900 223.1 L1920 227.2 L1940 230.1 L1960 231.9 L1980 232.5 L2000 231.9 L2020 230.3 L2040 228.0 L2060 225.2 L2080 222.3 L2100 219.5 L2120 216.9 L2140 214.7 L2160 213.0 L2180 211.6 L2200 210.5 L2220 209.5 L2240 208.4 L2260 207.1 L2280 205.4 L2300 203.2 L2320 200.7 L2340 197.9 L2360 195.0 L2380 192.2 L2400 189.8 L2420 188.2 L2440 187.5 L2460 188.0 L2480 189.7 L2500 192.6 L2520 196.7 L2540 201.5 L2560 207.0 L2580 212.7 L2600 218.1 L2620 223.1 L2640 227.2 L2660 230.1 L2680 231.9 L2700 232.5 L2720 231.9 L2740 230.3 L2760 228.0 L2780 225.2 L2800 222.3 L2820 219.5 L2840 216.9 L2860 214.7 L2880 213.0 L2900 211.6 L2920 210.5 L2940 209.5 L2960 208.4 L2980 207.1 L3000 205.4 L3020 203.2 L3040 200.7 L3060 197.9 L3080 195.0 L3100 192.2 L3120 189.8 L3140 188.2 L3160 187.5 L3180 188.0 L3200 189.7 L3220 192.6 L3240 196.7 L3260 201.5 L3280 207.0 L3300 212.7 L3320 218.1 L3340 223.1 L3360 227.2 L3380 230.1 L3400 231.9 L3420 232.5 L3440 231.9 L3460 230.3 L3480 228.0 L3500 225.2 L3520 222.3 L3540 219.5 L3560 216.9 L3580 214.7 L3600 213.0 L3620 211.6 L3640 210.5 L3660 209.5 L3680 208.4 L3700 207.1 L3720 205.4 L3740 203.2 L3760 200.7 L3780 197.9 L3800 195.0 L3820 192.2 L3840 189.8 L3860 188.2 L3880 187.5 L3900 188.0 L3920 189.7 L3940 192.6 L3960 196.7 L3980 201.5 L4000 207.0 L4020 212.7 L4040 218.1 L4060 223.1 L4080 227.2 L4100 230.1 L4120 231.9 L4140 232.5 L4160 231.9 L4180 230.3 L4200 228.0 L4220 225.2 L4240 222.3 L4260 219.5 L4280 216.9 L4300 214.7 L4320 213.0"
                  style={{ opacity: '0.68' }}
                  strokeWidth="1.2"
                ></path>
                <path
                  d="M0 253.8 L20 252.2 L40 250.3 L60 247.9 L80 245.4 L100 242.8 L120 240.3 L140 238.3 L160 237.0 L180 236.5 L200 237.1 L220 238.8 L240 241.6 L260 245.3 L280 249.8 L300 254.8 L320 259.8 L340 264.7 L360 269.0 L380 272.6 L400 275.1 L420 276.5 L440 276.8 L460 276.1 L480 274.6 L500 272.4 L520 269.9 L540 267.3 L560 264.8 L580 262.5 L600 260.7 L620 259.1 L640 258.0 L660 257.0 L680 256.1 L700 255.1 L720 253.8 L740 252.2 L760 250.3 L780 247.9 L800 245.4 L820 242.8 L840 240.3 L860 238.3 L880 237.0 L900 236.5 L920 237.1 L940 238.8 L960 241.6 L980 245.3 L1000 249.8 L1020 254.8 L1040 259.8 L1060 264.7 L1080 269.0 L1100 272.6 L1120 275.1 L1140 276.5 L1160 276.8 L1180 276.1 L1200 274.6 L1220 272.4 L1240 269.9 L1260 267.3 L1280 264.8 L1300 262.5 L1320 260.7 L1340 259.1 L1360 258.0 L1380 257.0 L1400 256.1 L1420 255.1 L1440 253.8 L1460 252.2 L1480 250.3 L1500 247.9 L1520 245.4 L1540 242.8 L1560 240.3 L1580 238.3 L1600 237.0 L1620 236.5 L1640 237.1 L1660 238.8 L1680 241.6 L1700 245.3 L1720 249.8 L1740 254.8 L1760 259.8 L1780 264.7 L1800 269.0 L1820 272.6 L1840 275.1 L1860 276.5 L1880 276.8 L1900 276.1 L1920 274.6 L1940 272.4 L1960 269.9 L1980 267.3 L2000 264.8 L2020 262.5 L2040 260.7 L2060 259.1 L2080 258.0 L2100 257.0 L2120 256.1 L2140 255.1 L2160 253.8 L2180 252.2 L2200 250.3 L2220 247.9 L2240 245.4 L2260 242.8 L2280 240.3 L2300 238.3 L2320 237.0 L2340 236.5 L2360 237.1 L2380 238.8 L2400 241.6 L2420 245.3 L2440 249.8 L2460 254.8 L2480 259.8 L2500 264.7 L2520 269.0 L2540 272.6 L2560 275.1 L2580 276.5 L2600 276.8 L2620 276.1 L2640 274.6 L2660 272.4 L2680 269.9 L2700 267.3 L2720 264.8 L2740 262.5 L2760 260.7 L2780 259.1 L2800 258.0 L2820 257.0 L2840 256.1 L2860 255.1 L2880 253.8 L2900 252.2 L2920 250.3 L2940 247.9 L2960 245.4 L2980 242.8 L3000 240.3 L3020 238.3 L3040 237.0 L3060 236.5 L3080 237.1 L3100 238.8 L3120 241.6 L3140 245.3 L3160 249.8 L3180 254.8 L3200 259.8 L3220 264.7 L3240 269.0 L3260 272.6 L3280 275.1 L3300 276.5 L3320 276.8 L3340 276.1 L3360 274.6 L3380 272.4 L3400 269.9 L3420 267.3 L3440 264.8 L3460 262.5 L3480 260.7 L3500 259.1 L3520 258.0 L3540 257.0 L3560 256.1 L3580 255.1 L3600 253.8 L3620 252.2 L3640 250.3 L3660 247.9 L3680 245.4 L3700 242.8 L3720 240.3 L3740 238.3 L3760 237.0 L3780 236.5 L3800 237.1 L3820 238.8 L3840 241.6 L3860 245.3 L3880 249.8 L3900 254.8 L3920 259.8 L3940 264.7 L3960 269.0 L3980 272.6 L4000 275.1 L4020 276.5 L4040 276.8 L4060 276.1 L4080 274.6 L4100 272.4 L4120 269.9 L4140 267.3 L4160 264.8 L4180 262.5 L4200 260.7 L4220 259.1 L4240 258.0 L4260 257.0 L4280 256.1 L4300 255.1 L4320 253.8"
                  style={{ opacity: '0.51' }}
                  strokeWidth="2.1"
                ></path>
                <path
                  d="M0 279.9 L20 275.9 L40 272.8 L60 270.8 L80 270.3 L100 271.6 L120 274.6 L140 279.5 L160 285.8 L180 293.3 L200 301.5 L220 309.8 L240 317.7 L260 324.6 L280 330.1 L300 334.0 L320 336.0 L340 336.3 L360 334.9 L380 332.2 L400 328.6 L420 324.4 L440 320.1 L460 316.0 L480 312.4 L500 309.4 L520 307.1 L540 305.2 L560 303.7 L580 302.2 L600 300.5 L620 298.3 L640 295.6 L660 292.3 L680 288.4 L700 284.1 L720 279.9 L740 275.9 L760 272.8 L780 270.8 L800 270.3 L820 271.6 L840 274.6 L860 279.5 L880 285.8 L900 293.3 L920 301.5 L940 309.8 L960 317.7 L980 324.6 L1000 330.1 L1020 334.0 L1040 336.0 L1060 336.3 L1080 334.9 L1100 332.2 L1120 328.6 L1140 324.4 L1160 320.1 L1180 316.0 L1200 312.4 L1220 309.4 L1240 307.1 L1260 305.2 L1280 303.7 L1300 302.2 L1320 300.5 L1340 298.3 L1360 295.6 L1380 292.3 L1400 288.4 L1420 284.1 L1440 279.9 L1460 275.9 L1480 272.8 L1500 270.8 L1520 270.3 L1540 271.6 L1560 274.6 L1580 279.5 L1600 285.8 L1620 293.3 L1640 301.5 L1660 309.8 L1680 317.7 L1700 324.6 L1720 330.1 L1740 334.0 L1760 336.0 L1780 336.3 L1800 334.9 L1820 332.2 L1840 328.6 L1860 324.4 L1880 320.1 L1900 316.0 L1920 312.4 L1940 309.4 L1960 307.1 L1980 305.2 L2000 303.7 L2020 302.2 L2040 300.5 L2060 298.3 L2080 295.6 L2100 292.3 L2120 288.4 L2140 284.1 L2160 279.9 L2180 275.9 L2200 272.8 L2220 270.8 L2240 270.3 L2260 271.6 L2280 274.6 L2300 279.5 L2320 285.8 L2340 293.3 L2360 301.5 L2380 309.8 L2400 317.7 L2420 324.6 L2440 330.1 L2460 334.0 L2480 336.0 L2500 336.3 L2520 334.9 L2540 332.2 L2560 328.6 L2580 324.4 L2600 320.1 L2620 316.0 L2640 312.4 L2660 309.4 L2680 307.1 L2700 305.2 L2720 303.7 L2740 302.2 L2760 300.5 L2780 298.3 L2800 295.6 L2820 292.3 L2840 288.4 L2860 284.1 L2880 279.9 L2900 275.9 L2920 272.8 L2940 270.8 L2960 270.3 L2980 271.6 L3000 274.6 L3020 279.5 L3040 285.8 L3060 293.3 L3080 301.5 L3100 309.8 L3120 317.7 L3140 324.6 L3160 330.1 L3180 334.0 L3200 336.0 L3220 336.3 L3240 334.9 L3260 332.2 L3280 328.6 L3300 324.4 L3320 320.1 L3340 316.0 L3360 312.4 L3380 309.4 L3400 307.1 L3420 305.2 L3440 303.7 L3460 302.2 L3480 300.5 L3500 298.3 L3520 295.6 L3540 292.3 L3560 288.4 L3580 284.1 L3600 279.9 L3620 275.9 L3640 272.8 L3660 270.8 L3680 270.3 L3700 271.6 L3720 274.6 L3740 279.5 L3760 285.8 L3780 293.3 L3800 301.5 L3820 309.8 L3840 317.7 L3860 324.6 L3880 330.1 L3900 334.0 L3920 336.0 L3940 336.3 L3960 334.9 L3980 332.2 L4000 328.6 L4020 324.4 L4040 320.1 L4060 316.0 L4080 312.4 L4100 309.4 L4120 307.1 L4140 305.2 L4160 303.7 L4180 302.2 L4200 300.5 L4220 298.3 L4240 295.6 L4260 292.3 L4280 288.4 L4300 284.1 L4320 279.9"
                  style={{ opacity: '0.35' }}
                  strokeWidth="3.0"
                ></path>
                <path
                  d="M0 309.8 L20 314.1 L40 320.6 L60 329.0 L80 338.8 L100 349.4 L120 359.9 L140 369.8 L160 378.4 L180 385.2 L200 389.8 L220 392.0 L240 392.0 L260 389.9 L280 386.2 L300 381.4 L320 376.1 L340 370.6 L360 365.5 L380 361.0 L400 357.3 L420 354.4 L440 352.1 L460 350.1 L480 348.2 L500 345.9 L520 343.1 L540 339.5 L560 335.1 L580 330.1 L600 324.6 L620 319.2 L640 314.3 L660 310.4 L680 308.2 L700 307.9 L720 309.8 L740 314.1 L760 320.6 L780 329.0 L800 338.8 L820 349.4 L840 359.9 L860 369.8 L880 378.4 L900 385.2 L920 389.8 L940 392.0 L960 392.0 L980 389.9 L1000 386.2 L1020 381.4 L1040 376.1 L1060 370.6 L1080 365.5 L1100 361.0 L1120 357.3 L1140 354.4 L1160 352.1 L1180 350.1 L1200 348.2 L1220 345.9 L1240 343.1 L1260 339.5 L1280 335.1 L1300 330.1 L1320 324.6 L1340 319.2 L1360 314.3 L1380 310.4 L1400 308.2 L1420 307.9 L1440 309.8 L1460 314.1 L1480 320.6 L1500 329.0 L1520 338.8 L1540 349.4 L1560 359.9 L1580 369.8 L1600 378.4 L1620 385.2 L1640 389.8 L1660 392.0 L1680 392.0 L1700 389.9 L1720 386.2 L1740 381.4 L1760 376.1 L1780 370.6 L1800 365.5 L1820 361.0 L1840 357.3 L1860 354.4 L1880 352.1 L1900 350.1 L1920 348.2 L1940 345.9 L1960 343.1 L1980 339.5 L2000 335.1 L2020 330.1 L2040 324.6 L2060 319.2 L2080 314.3 L2100 310.4 L2120 308.2 L2140 307.9 L2160 309.8 L2180 314.1 L2200 320.6 L2220 329.0 L2240 338.8 L2260 349.4 L2280 359.9 L2300 369.8 L2320 378.4 L2340 385.2 L2360 389.8 L2380 392.0 L2400 392.0 L2420 389.9 L2440 386.2 L2460 381.4 L2480 376.1 L2500 370.6 L2520 365.5 L2540 361.0 L2560 357.3 L2580 354.4 L2600 352.1 L2620 350.1 L2640 348.2 L2660 345.9 L2680 343.1 L2700 339.5 L2720 335.1 L2740 330.1 L2760 324.6 L2780 319.2 L2800 314.3 L2820 310.4 L2840 308.2 L2860 307.9 L2880 309.8 L2900 314.1 L2920 320.6 L2940 329.0 L2960 338.8 L2980 349.4 L3000 359.9 L3020 369.8 L3040 378.4 L3060 385.2 L3080 389.8 L3100 392.0 L3120 392.0 L3140 389.9 L3160 386.2 L3180 381.4 L3200 376.1 L3220 370.6 L3240 365.5 L3260 361.0 L3280 357.3 L3300 354.4 L3320 352.1 L3340 350.1 L3360 348.2 L3380 345.9 L3400 343.1 L3420 339.5 L3440 335.1 L3460 330.1 L3480 324.6 L3500 319.2 L3520 314.3 L3540 310.4 L3560 308.2 L3580 307.9 L3600 309.8 L3620 314.1 L3640 320.6 L3660 329.0 L3680 338.8 L3700 349.4 L3720 359.9 L3740 369.8 L3760 378.4 L3780 385.2 L3800 389.8 L3820 392.0 L3840 392.0 L3860 389.9 L3880 386.2 L3900 381.4 L3920 376.1 L3940 370.6 L3960 365.5 L3980 361.0 L4000 357.3 L4020 354.4 L4040 352.1 L4060 350.1 L4080 348.2 L4100 345.9 L4120 343.1 L4140 339.5 L4160 335.1 L4180 330.1 L4200 324.6 L4220 319.2 L4240 314.3 L4260 310.4 L4280 308.2 L4300 307.9 L4320 309.8"
                  style={{ opacity: '0.18' }}
                  strokeWidth="1.2"
                ></path>
              </g>
            </svg>
          </div>
          <div className="waves w2">
            <svg width="4320" height="420" viewBox="0 0 4320 420" aria-hidden="true">
              <g fill="none" stroke="#76B900" strokeLinecap="round">
                <path
                  d="M0 70.0 L20 75.7 L40 81.3 L60 86.4 L80 90.9 L100 94.6 L120 97.5 L140 99.4 L160 100.4 L180 100.5 L200 99.7 L220 98.1 L240 96.0 L260 93.4 L280 90.6 L300 87.6 L320 84.6 L340 81.8 L360 79.3 L380 77.0 L400 75.1 L420 73.5 L440 72.2 L460 71.0 L480 70.0 L500 69.0 L520 67.8 L540 66.5 L560 64.9 L580 63.0 L600 60.7 L620 58.2 L640 55.4 L660 52.4 L680 49.4 L700 46.6 L720 44.0 L740 41.9 L760 40.3 L780 39.5 L800 39.6 L820 40.6 L840 42.5 L860 45.4 L880 49.1 L900 53.6 L920 58.7 L940 64.3 L960 70.0 L980 75.7 L1000 81.3 L1020 86.4 L1040 90.9 L1060 94.6 L1080 97.5 L1100 99.4 L1120 100.4 L1140 100.5 L1160 99.7 L1180 98.1 L1200 96.0 L1220 93.4 L1240 90.6 L1260 87.6 L1280 84.6 L1300 81.8 L1320 79.3 L1340 77.0 L1360 75.1 L1380 73.5 L1400 72.2 L1420 71.0 L1440 70.0 L1460 69.0 L1480 67.8 L1500 66.5 L1520 64.9 L1540 63.0 L1560 60.7 L1580 58.2 L1600 55.4 L1620 52.4 L1640 49.4 L1660 46.6 L1680 44.0 L1700 41.9 L1720 40.3 L1740 39.5 L1760 39.6 L1780 40.6 L1800 42.5 L1820 45.4 L1840 49.1 L1860 53.6 L1880 58.7 L1900 64.3 L1920 70.0 L1940 75.7 L1960 81.3 L1980 86.4 L2000 90.9 L2020 94.6 L2040 97.5 L2060 99.4 L2080 100.4 L2100 100.5 L2120 99.7 L2140 98.1 L2160 96.0 L2180 93.4 L2200 90.6 L2220 87.6 L2240 84.6 L2260 81.8 L2280 79.3 L2300 77.0 L2320 75.1 L2340 73.5 L2360 72.2 L2380 71.0 L2400 70.0 L2420 69.0 L2440 67.8 L2460 66.5 L2480 64.9 L2500 63.0 L2520 60.7 L2540 58.2 L2560 55.4 L2580 52.4 L2600 49.4 L2620 46.6 L2640 44.0 L2660 41.9 L2680 40.3 L2700 39.5 L2720 39.6 L2740 40.6 L2760 42.5 L2780 45.4 L2800 49.1 L2820 53.6 L2840 58.7 L2860 64.3 L2880 70.0 L2900 75.7 L2920 81.3 L2940 86.4 L2960 90.9 L2980 94.6 L3000 97.5 L3020 99.4 L3040 100.4 L3060 100.5 L3080 99.7 L3100 98.1 L3120 96.0 L3140 93.4 L3160 90.6 L3180 87.6 L3200 84.6 L3220 81.8 L3240 79.3 L3260 77.0 L3280 75.1 L3300 73.5 L3320 72.2 L3340 71.0 L3360 70.0 L3380 69.0 L3400 67.8 L3420 66.5 L3440 64.9 L3460 63.0 L3480 60.7 L3500 58.2 L3520 55.4 L3540 52.4 L3560 49.4 L3580 46.6 L3600 44.0 L3620 41.9 L3640 40.3 L3660 39.5 L3680 39.6 L3700 40.6 L3720 42.5 L3740 45.4 L3760 49.1 L3780 53.6 L3800 58.7 L3820 64.3 L3840 70.0 L3860 75.7 L3880 81.3 L3900 86.4 L3920 90.9 L3940 94.6 L3960 97.5 L3980 99.4 L4000 100.4 L4020 100.5 L4040 99.7 L4060 98.1 L4080 96.0 L4100 93.4 L4120 90.6 L4140 87.6 L4160 84.6 L4180 81.8 L4200 79.3 L4220 77.0 L4240 75.1 L4260 73.5 L4280 72.2 L4300 71.0 L4320 70.0"
                  style={{ opacity: '0.18' }}
                  strokeWidth="1.2"
                ></path>
                <path
                  d="M0 180.1 L20 181.6 L40 181.8 L60 180.9 L80 178.9 L100 176.0 L120 172.6 L140 168.7 L160 164.6 L180 160.6 L200 156.7 L220 153.1 L240 150.0 L260 147.3 L280 145.1 L300 143.2 L320 141.6 L340 140.2 L360 138.8 L380 137.2 L400 135.4 L420 133.3 L440 130.7 L460 127.7 L480 124.2 L500 120.4 L520 116.4 L540 112.3 L560 108.4 L580 104.8 L600 101.8 L620 99.6 L640 98.3 L660 98.3 L680 99.4 L700 101.9 L720 105.7 L740 110.7 L760 116.7 L780 123.6 L800 131.2 L820 139.0 L840 146.9 L860 154.5 L880 161.6 L900 167.9 L920 173.2 L940 177.2 L960 180.1 L980 181.6 L1000 181.8 L1020 180.9 L1040 178.9 L1060 176.0 L1080 172.6 L1100 168.7 L1120 164.6 L1140 160.6 L1160 156.7 L1180 153.1 L1200 150.0 L1220 147.3 L1240 145.1 L1260 143.2 L1280 141.6 L1300 140.2 L1320 138.8 L1340 137.2 L1360 135.4 L1380 133.3 L1400 130.7 L1420 127.7 L1440 124.2 L1460 120.4 L1480 116.4 L1500 112.3 L1520 108.4 L1540 104.8 L1560 101.8 L1580 99.6 L1600 98.3 L1620 98.3 L1640 99.4 L1660 101.9 L1680 105.7 L1700 110.7 L1720 116.7 L1740 123.6 L1760 131.2 L1780 139.0 L1800 146.9 L1820 154.5 L1840 161.6 L1860 167.9 L1880 173.2 L1900 177.2 L1920 180.1 L1940 181.6 L1960 181.8 L1980 180.9 L2000 178.9 L2020 176.0 L2040 172.6 L2060 168.7 L2080 164.6 L2100 160.6 L2120 156.7 L2140 153.1 L2160 150.0 L2180 147.3 L2200 145.1 L2220 143.2 L2240 141.6 L2260 140.2 L2280 138.8 L2300 137.2 L2320 135.4 L2340 133.3 L2360 130.7 L2380 127.7 L2400 124.2 L2420 120.4 L2440 116.4 L2460 112.3 L2480 108.4 L2500 104.8 L2520 101.8 L2540 99.6 L2560 98.3 L2580 98.3 L2600 99.4 L2620 101.9 L2640 105.7 L2660 110.7 L2680 116.7 L2700 123.6 L2720 131.2 L2740 139.0 L2760 146.9 L2780 154.5 L2800 161.6 L2820 167.9 L2840 173.2 L2860 177.2 L2880 180.1 L2900 181.6 L2920 181.8 L2940 180.9 L2960 178.9 L2980 176.0 L3000 172.6 L3020 168.7 L3040 164.6 L3060 160.6 L3080 156.7 L3100 153.1 L3120 150.0 L3140 147.3 L3160 145.1 L3180 143.2 L3200 141.6 L3220 140.2 L3240 138.8 L3260 137.2 L3280 135.4 L3300 133.3 L3320 130.7 L3340 127.7 L3360 124.2 L3380 120.4 L3400 116.4 L3420 112.3 L3440 108.4 L3460 104.8 L3480 101.8 L3500 99.6 L3520 98.3 L3540 98.3 L3560 99.4 L3580 101.9 L3600 105.7 L3620 110.7 L3640 116.7 L3660 123.6 L3680 131.2 L3700 139.0 L3720 146.9 L3740 154.5 L3760 161.6 L3780 167.9 L3800 173.2 L3820 177.2 L3840 180.1 L3860 181.6 L3880 181.8 L3900 180.9 L3920 178.9 L3940 176.0 L3960 172.6 L3980 168.7 L4000 164.6 L4020 160.6 L4040 156.7 L4060 153.1 L4080 150.0 L4100 147.3 L4120 145.1 L4140 143.2 L4160 141.6 L4180 140.2 L4200 138.8 L4220 137.2 L4240 135.4 L4260 133.3 L4280 130.7 L4300 127.7 L4320 124.2"
                  style={{ opacity: '0.43' }}
                  strokeWidth="2.1"
                ></path>
                <path
                  d="M0 235.5 L20 232.0 L40 228.4 L60 225.0 L80 221.9 L100 219.1 L120 216.7 L140 214.7 L160 213.0 L180 211.6 L200 210.3 L220 209.1 L240 207.7 L260 206.2 L280 204.4 L300 202.2 L320 199.6 L340 196.6 L360 193.3 L380 189.8 L400 186.2 L420 182.8 L440 179.6 L460 176.9 L480 174.8 L500 173.7 L520 173.5 L540 174.4 L560 176.4 L580 179.5 L600 183.8 L620 188.9 L640 194.9 L660 201.4 L680 208.3 L700 215.2 L720 221.9 L740 228.2 L760 233.8 L780 238.5 L800 242.2 L820 244.8 L840 246.2 L860 246.6 L880 245.9 L900 244.2 L920 241.8 L940 238.9 L960 235.5 L980 232.0 L1000 228.4 L1020 225.0 L1040 221.9 L1060 219.1 L1080 216.7 L1100 214.7 L1120 213.0 L1140 211.6 L1160 210.3 L1180 209.1 L1200 207.7 L1220 206.2 L1240 204.4 L1260 202.2 L1280 199.6 L1300 196.6 L1320 193.3 L1340 189.8 L1360 186.2 L1380 182.8 L1400 179.6 L1420 176.9 L1440 174.8 L1460 173.7 L1480 173.5 L1500 174.4 L1520 176.4 L1540 179.5 L1560 183.8 L1580 188.9 L1600 194.9 L1620 201.4 L1640 208.3 L1660 215.2 L1680 221.9 L1700 228.2 L1720 233.8 L1740 238.5 L1760 242.2 L1780 244.8 L1800 246.2 L1820 246.6 L1840 245.9 L1860 244.2 L1880 241.8 L1900 238.9 L1920 235.5 L1940 232.0 L1960 228.4 L1980 225.0 L2000 221.9 L2020 219.1 L2040 216.7 L2060 214.7 L2080 213.0 L2100 211.6 L2120 210.3 L2140 209.1 L2160 207.7 L2180 206.2 L2200 204.4 L2220 202.2 L2240 199.6 L2260 196.6 L2280 193.3 L2300 189.8 L2320 186.2 L2340 182.8 L2360 179.6 L2380 176.9 L2400 174.8 L2420 173.7 L2440 173.5 L2460 174.4 L2480 176.4 L2500 179.5 L2520 183.8 L2540 188.9 L2560 194.9 L2580 201.4 L2600 208.3 L2620 215.2 L2640 221.9 L2660 228.2 L2680 233.8 L2700 238.5 L2720 242.2 L2740 244.8 L2760 246.2 L2780 246.6 L2800 245.9 L2820 244.2 L2840 241.8 L2860 238.9 L2880 235.5 L2900 232.0 L2920 228.4 L2940 225.0 L2960 221.9 L2980 219.1 L3000 216.7 L3020 214.7 L3040 213.0 L3060 211.6 L3080 210.3 L3100 209.1 L3120 207.7 L3140 206.2 L3160 204.4 L3180 202.2 L3200 199.6 L3220 196.6 L3240 193.3 L3260 189.8 L3280 186.2 L3300 182.8 L3320 179.6 L3340 176.9 L3360 174.8 L3380 173.7 L3400 173.5 L3420 174.4 L3440 176.4 L3460 179.5 L3480 183.8 L3500 188.9 L3520 194.9 L3540 201.4 L3560 208.3 L3580 215.2 L3600 221.9 L3620 228.2 L3640 233.8 L3660 238.5 L3680 242.2 L3700 244.8 L3720 246.2 L3740 246.6 L3760 245.9 L3780 244.2 L3800 241.8 L3820 238.9 L3840 235.5 L3860 232.0 L3880 228.4 L3900 225.0 L3920 221.9 L3940 219.1 L3960 216.7 L3980 214.7 L4000 213.0 L4020 211.6 L4040 210.3 L4060 209.1 L4080 207.7 L4100 206.2 L4120 204.4 L4140 202.2 L4160 199.6 L4180 196.6 L4200 193.3 L4220 189.8 L4240 186.2 L4260 182.8 L4280 179.6 L4300 176.9 L4320 174.8"
                  style={{ opacity: '0.68' }}
                  strokeWidth="3.0"
                ></path>
                <path
                  d="M0 283.0 L20 281.9 L40 281.1 L60 280.3 L80 279.5 L100 278.7 L120 277.8 L140 276.7 L160 275.4 L180 273.8 L200 272.0 L220 270.0 L240 267.9 L260 265.7 L280 263.5 L300 261.6 L320 259.8 L340 258.5 L360 257.7 L380 257.6 L400 258.0 L420 259.2 L440 261.0 L460 263.5 L480 266.7 L500 270.3 L520 274.2 L540 278.4 L560 282.7 L580 286.8 L600 290.7 L620 294.2 L640 297.2 L660 299.5 L680 301.2 L700 302.2 L720 302.5 L740 302.1 L760 301.2 L780 299.8 L800 298.0 L820 295.9 L840 293.8 L860 291.6 L880 289.5 L900 287.5 L920 285.8 L940 284.3 L960 283.0 L980 281.9 L1000 281.1 L1020 280.3 L1040 279.5 L1060 278.7 L1080 277.8 L1100 276.7 L1120 275.4 L1140 273.8 L1160 272.0 L1180 270.0 L1200 267.9 L1220 265.7 L1240 263.5 L1260 261.6 L1280 259.8 L1300 258.5 L1320 257.7 L1340 257.6 L1360 258.0 L1380 259.2 L1400 261.0 L1420 263.5 L1440 266.7 L1460 270.3 L1480 274.2 L1500 278.4 L1520 282.7 L1540 286.8 L1560 290.7 L1580 294.2 L1600 297.2 L1620 299.5 L1640 301.2 L1660 302.2 L1680 302.5 L1700 302.1 L1720 301.2 L1740 299.8 L1760 298.0 L1780 295.9 L1800 293.8 L1820 291.6 L1840 289.5 L1860 287.5 L1880 285.8 L1900 284.3 L1920 283.0 L1940 281.9 L1960 281.1 L1980 280.3 L2000 279.5 L2020 278.7 L2040 277.8 L2060 276.7 L2080 275.4 L2100 273.8 L2120 272.0 L2140 270.0 L2160 267.9 L2180 265.7 L2200 263.5 L2220 261.6 L2240 259.8 L2260 258.5 L2280 257.7 L2300 257.6 L2320 258.0 L2340 259.2 L2360 261.0 L2380 263.5 L2400 266.7 L2420 270.3 L2440 274.2 L2460 278.4 L2480 282.7 L2500 286.8 L2520 290.7 L2540 294.2 L2560 297.2 L2580 299.5 L2600 301.2 L2620 302.2 L2640 302.5 L2660 302.1 L2680 301.2 L2700 299.8 L2720 298.0 L2740 295.9 L2760 293.8 L2780 291.6 L2800 289.5 L2820 287.5 L2840 285.8 L2860 284.3 L2880 283.0 L2900 281.9 L2920 281.1 L2940 280.3 L2960 279.5 L2980 278.7 L3000 277.8 L3020 276.7 L3040 275.4 L3060 273.8 L3080 272.0 L3100 270.0 L3120 267.9 L3140 265.7 L3160 263.5 L3180 261.6 L3200 259.8 L3220 258.5 L3240 257.7 L3260 257.6 L3280 258.0 L3300 259.2 L3320 261.0 L3340 263.5 L3360 266.7 L3380 270.3 L3400 274.2 L3420 278.4 L3440 282.7 L3460 286.8 L3480 290.7 L3500 294.2 L3520 297.2 L3540 299.5 L3560 301.2 L3580 302.2 L3600 302.5 L3620 302.1 L3640 301.2 L3660 299.8 L3680 298.0 L3700 295.9 L3720 293.8 L3740 291.6 L3760 289.5 L3780 287.5 L3800 285.8 L3820 284.3 L3840 283.0 L3860 281.9 L3880 281.1 L3900 280.3 L3920 279.5 L3940 278.7 L3960 277.8 L3980 276.7 L4000 275.4 L4020 273.8 L4040 272.0 L4060 270.0 L4080 267.9 L4100 265.7 L4120 263.5 L4140 261.6 L4160 259.8 L4180 258.5 L4200 257.7 L4220 257.6 L4240 258.0 L4260 259.2 L4280 261.0 L4300 263.5 L4320 266.7"
                  style={{ opacity: '0.43' }}
                  strokeWidth="1.2"
                ></path>
                <path
                  d="M0 347.2 L20 346.0 L40 344.6 L60 343.0 L80 341.3 L100 339.4 L120 337.4 L140 335.5 L160 333.7 L180 332.1 L200 330.9 L220 330.1 L240 329.8 L260 330.2 L280 331.1 L300 332.7 L320 334.9 L340 337.6 L360 340.8 L380 344.4 L400 348.1 L420 351.9 L440 355.7 L460 359.2 L480 362.4 L500 365.1 L520 367.3 L540 368.9 L560 369.8 L580 370.2 L600 369.9 L620 369.1 L640 367.9 L660 366.3 L680 364.5 L700 362.6 L720 360.6 L740 358.7 L760 356.9 L780 355.4 L800 354.0 L820 352.8 L840 351.9 L860 351.0 L880 350.3 L900 349.7 L920 348.9 L940 348.1 L960 347.2 L980 346.0 L1000 344.6 L1020 343.0 L1040 341.3 L1060 339.4 L1080 337.4 L1100 335.5 L1120 333.7 L1140 332.1 L1160 330.9 L1180 330.1 L1200 329.8 L1220 330.2 L1240 331.1 L1260 332.7 L1280 334.9 L1300 337.6 L1320 340.8 L1340 344.4 L1360 348.1 L1380 351.9 L1400 355.7 L1420 359.2 L1440 362.4 L1460 365.1 L1480 367.3 L1500 368.9 L1520 369.8 L1540 370.2 L1560 369.9 L1580 369.1 L1600 367.9 L1620 366.3 L1640 364.5 L1660 362.6 L1680 360.6 L1700 358.7 L1720 356.9 L1740 355.4 L1760 354.0 L1780 352.8 L1800 351.9 L1820 351.0 L1840 350.3 L1860 349.7 L1880 348.9 L1900 348.1 L1920 347.2 L1940 346.0 L1960 344.6 L1980 343.0 L2000 341.3 L2020 339.4 L2040 337.4 L2060 335.5 L2080 333.7 L2100 332.1 L2120 330.9 L2140 330.1 L2160 329.8 L2180 330.2 L2200 331.1 L2220 332.7 L2240 334.9 L2260 337.6 L2280 340.8 L2300 344.4 L2320 348.1 L2340 351.9 L2360 355.7 L2380 359.2 L2400 362.4 L2420 365.1 L2440 367.3 L2460 368.9 L2480 369.8 L2500 370.2 L2520 369.9 L2540 369.1 L2560 367.9 L2580 366.3 L2600 364.5 L2620 362.6 L2640 360.6 L2660 358.7 L2680 356.9 L2700 355.4 L2720 354.0 L2740 352.8 L2760 351.9 L2780 351.0 L2800 350.3 L2820 349.7 L2840 348.9 L2860 348.1 L2880 347.2 L2900 346.0 L2920 344.6 L2940 343.0 L2960 341.3 L2980 339.4 L3000 337.4 L3020 335.5 L3040 333.7 L3060 332.1 L3080 330.9 L3100 330.1 L3120 329.8 L3140 330.2 L3160 331.1 L3180 332.7 L3200 334.9 L3220 337.6 L3240 340.8 L3260 344.4 L3280 348.1 L3300 351.9 L3320 355.7 L3340 359.2 L3360 362.4 L3380 365.1 L3400 367.3 L3420 368.9 L3440 369.8 L3460 370.2 L3480 369.9 L3500 369.1 L3520 367.9 L3540 366.3 L3560 364.5 L3580 362.6 L3600 360.6 L3620 358.7 L3640 356.9 L3660 355.4 L3680 354.0 L3700 352.8 L3720 351.9 L3740 351.0 L3760 350.3 L3780 349.7 L3800 348.9 L3820 348.1 L3840 347.2 L3860 346.0 L3880 344.6 L3900 343.0 L3920 341.3 L3940 339.4 L3960 337.4 L3980 335.5 L4000 333.7 L4020 332.1 L4040 330.9 L4060 330.1 L4080 329.8 L4100 330.2 L4120 331.1 L4140 332.7 L4160 334.9 L4180 337.6 L4200 340.8 L4220 344.4 L4240 348.1 L4260 351.9 L4280 355.7 L4300 359.2 L4320 362.4"
                  style={{ opacity: '0.18' }}
                  strokeWidth="2.1"
                ></path>
              </g>
            </svg>
          </div>
          <div className="field" style={{ left: '-180px', bottom: '-60px' }}>
            <svg viewBox="0 0 620 520" width="620" height="520" aria-hidden="true" style={{ overflow: 'visible' }}>
              <defs>
                <radialGradient id="wmAg" cx="50%" cy="50%" r="50%">
                  <stop offset="0" stopColor="#fff" stopOpacity="1"></stop>
                  <stop offset="0.65" stopColor="#fff" stopOpacity="0.55"></stop>
                  <stop offset="1" stopColor="#fff" stopOpacity="0"></stop>
                </radialGradient>
                <mask id="wmA">
                  <rect x="-60" y="-60" width="740" height="640" fill="url(#wmAg)"></rect>
                </mask>
              </defs>
              <g mask="url(#wmA)" fill="none" stroke="#76B900" strokeWidth="2.2" strokeLinecap="round" strokeDasharray="0.1 7">
                <path
                  className="flow"
                  d="M0.0 0.0 L8.1 11.6 L16.0 23.1 L23.7 34.7 L30.8 46.2 L37.2 57.8 L42.9 69.3 L47.7 80.9 L51.5 92.4 L54.3 104.0 L55.9 115.6 L56.5 127.1 L55.9 138.7 L54.2 150.2 L51.4 161.8 L47.7 173.3 L43.2 184.9 L37.9 196.4 L31.9 208.0 L25.6 219.6 L18.9 231.1 L12.0 242.7 L5.2 254.2 L-1.4 265.8 L-7.7 277.3 L-13.5 288.9 L-18.7 300.4 L-23.2 312.0 L-26.8 323.6 L-29.4 335.1 L-31.1 346.7 L-31.8 358.2 L-31.4 369.8 L-30.1 381.3 L-27.8 392.9 L-24.6 404.4 L-20.6 416.0 L-15.9 427.6 L-10.7 439.1 L-5.0 450.7 L1.0 462.2 L7.0 473.8 L13.1 485.3 L18.9 496.9 L24.4 508.4 L29.3 520.0"
                  style={{ opacity: '0.35', animationDelay: '-0.0s' }}
                ></path>
                <path
                  className="flow"
                  d="M29.2 8.1 L37.1 19.6 L44.7 31.2 L51.8 42.7 L58.2 54.3 L63.8 65.8 L68.5 77.4 L72.3 88.9 L75.0 100.5 L76.5 112.1 L76.9 123.6 L76.2 135.2 L74.4 146.7 L71.6 158.3 L67.7 169.8 L63.0 181.4 L57.6 192.9 L51.5 204.5 L45.0 216.1 L38.1 227.6 L31.1 239.2 L24.1 250.7 L17.3 262.3 L10.9 273.8 L4.9 285.4 L-0.5 296.9 L-5.1 308.5 L-8.8 320.1 L-11.6 331.6 L-13.5 343.2 L-14.3 354.7 L-14.1 366.3 L-12.9 377.8 L-10.8 389.4 L-7.7 400.9 L-3.8 412.5 L0.8 424.1 L5.9 435.6 L11.5 447.2 L17.4 458.7 L23.4 470.3 L29.4 481.8 L35.2 493.4 L40.6 504.9 L45.6 516.5 L49.9 528.1"
                  style={{ opacity: '0.41', animationDelay: '-0.6s' }}
                ></path>
                <path
                  className="flow"
                  d="M58.1 14.8 L65.6 26.4 L72.6 38.0 L79.0 49.5 L84.5 61.1 L89.2 72.6 L92.8 84.2 L95.4 95.7 L96.8 107.3 L97.1 118.8 L96.3 130.4 L94.4 142.0 L91.4 153.5 L87.4 165.1 L82.6 176.6 L77.0 188.2 L70.8 199.7 L64.1 211.3 L57.0 222.8 L49.9 234.4 L42.7 246.0 L35.8 257.5 L29.1 269.1 L23.0 280.6 L17.5 292.2 L12.7 303.7 L8.8 315.3 L5.8 326.8 L3.8 338.4 L2.8 350.0 L2.9 361.5 L4.0 373.1 L6.0 384.6 L9.0 396.2 L12.8 407.7 L17.3 419.3 L22.3 430.8 L27.9 442.4 L33.7 454.0 L39.7 465.5 L45.7 477.1 L51.4 488.6 L56.9 500.2 L61.8 511.7 L66.2 523.3 L69.8 534.8"
                  style={{ opacity: '0.47', animationDelay: '-1.2s' }}
                ></path>
                <path
                  className="flow"
                  d="M86.4 19.3 L93.3 30.8 L99.6 42.4 L105.0 54.0 L109.5 65.5 L113.1 77.1 L115.5 88.6 L116.8 100.2 L117.0 111.7 L116.1 123.3 L114.0 134.8 L110.8 146.4 L106.7 158.0 L101.7 169.5 L96.0 181.1 L89.6 192.6 L82.7 204.2 L75.6 215.7 L68.2 227.3 L60.9 238.8 L53.8 250.4 L47.0 262.0 L40.7 273.5 L35.1 285.1 L30.1 296.6 L26.1 308.2 L23.0 319.7 L20.8 331.3 L19.7 342.8 L19.7 354.4 L20.7 366.0 L22.6 377.5 L25.5 389.1 L29.2 400.6 L33.6 412.2 L38.7 423.7 L44.2 435.3 L50.0 446.8 L55.9 458.4 L61.9 470.0 L67.7 481.5 L73.2 493.1 L78.2 504.6 L82.6 516.2 L86.2 527.7 L89.1 539.3"
                  style={{ opacity: '0.53', animationDelay: '-1.8s' }}
                ></path>
                <path
                  className="flow"
                  d="M113.7 20.7 L119.9 32.2 L125.2 43.8 L129.6 55.4 L133.0 66.9 L135.3 78.5 L136.5 90.0 L136.6 101.6 L135.4 113.1 L133.2 124.7 L129.9 136.2 L125.7 147.8 L120.5 159.4 L114.6 170.9 L108.1 182.5 L101.1 194.0 L93.7 205.6 L86.3 217.1 L78.8 228.7 L71.5 240.2 L64.6 251.8 L58.1 263.4 L52.3 274.9 L47.3 286.5 L43.1 298.0 L39.9 309.6 L37.6 321.1 L36.4 332.7 L36.3 344.2 L37.2 355.8 L39.0 367.4 L41.9 378.9 L45.5 390.5 L49.9 402.0 L54.9 413.6 L60.4 425.1 L66.2 436.7 L72.2 448.2 L78.2 459.8 L84.0 471.4 L89.6 482.9 L94.6 494.5 L99.1 506.0 L102.8 517.6 L105.8 529.1 L107.8 540.7"
                  style={{ opacity: '0.59', animationDelay: '-2.4s' }}
                ></path>
                <path
                  className="flow"
                  d="M139.9 18.8 L145.1 30.4 L149.3 41.9 L152.6 53.5 L154.8 65.1 L155.8 76.6 L155.7 88.2 L154.5 99.7 L152.1 111.3 L148.7 122.8 L144.2 134.4 L139.0 145.9 L132.9 157.5 L126.2 169.1 L119.0 180.6 L111.6 192.2 L103.9 203.7 L96.3 215.3 L88.9 226.8 L81.8 238.4 L75.3 249.9 L69.3 261.5 L64.2 273.1 L59.9 284.6 L56.5 296.2 L54.2 307.7 L52.9 319.3 L52.7 330.8 L53.5 342.4 L55.4 353.9 L58.2 365.5 L61.8 377.1 L66.2 388.6 L71.2 400.2 L76.7 411.7 L82.5 423.3 L88.5 434.8 L94.6 446.4 L100.5 457.9 L106.1 469.5 L111.2 481.1 L115.8 492.6 L119.7 504.2 L122.7 515.7 L124.9 527.3 L126.0 538.8"
                  style={{ opacity: '0.65', animationDelay: '-3.0s' }}
                ></path>
                <path
                  className="flow"
                  d="M164.6 14.0 L168.7 25.6 L171.8 37.1 L173.9 48.7 L174.8 60.2 L174.5 71.8 L173.1 83.3 L170.6 94.9 L167.0 106.4 L162.5 118.0 L157.0 129.6 L150.8 141.1 L144.0 152.7 L136.7 164.2 L129.1 175.8 L121.3 187.3 L113.6 198.9 L106.0 210.4 L98.8 222.0 L92.2 233.6 L86.1 245.1 L80.9 256.7 L76.5 268.2 L73.1 279.8 L70.7 291.3 L69.3 302.9 L69.1 314.4 L69.9 326.0 L71.7 337.6 L74.4 349.1 L78.1 360.7 L82.5 372.2 L87.5 383.8 L93.0 395.3 L98.9 406.9 L105.0 418.4 L111.1 430.0 L117.1 441.6 L122.8 453.1 L128.1 464.7 L132.8 476.2 L136.7 487.8 L139.9 499.3 L142.2 510.9 L143.5 522.4 L143.9 534.0"
                  style={{ opacity: '0.70', animationDelay: '-3.6s' }}
                ></path>
                <path
                  className="flow"
                  d="M187.7 7.0 L190.7 18.5 L192.6 30.1 L193.3 41.6 L192.9 53.2 L191.4 64.7 L188.7 76.3 L185.0 87.8 L180.3 99.4 L174.7 111.0 L168.4 122.5 L161.4 134.1 L154.0 145.6 L146.3 157.2 L138.4 168.7 L130.6 180.3 L122.9 191.8 L115.6 203.4 L108.9 215.0 L102.8 226.5 L97.4 238.1 L93.0 249.6 L89.5 261.2 L87.1 272.7 L85.7 284.3 L85.4 295.8 L86.2 307.4 L88.0 319.0 L90.8 330.5 L94.4 342.1 L98.8 353.6 L103.9 365.2 L109.5 376.7 L115.5 388.3 L121.6 399.8 L127.9 411.4 L134.0 423.0 L139.8 434.5 L145.2 446.1 L150.0 457.6 L154.1 469.2 L157.4 480.7 L159.9 492.3 L161.4 503.8 L161.9 515.4 L161.4 527.0"
                  style={{ opacity: '0.75', animationDelay: '-0.0s' }}
                ></path>
                <path
                  className="flow"
                  d="M209.2 -1.2 L210.9 10.4 L211.6 21.9 L211.0 33.5 L209.3 45.0 L206.5 56.6 L202.7 68.2 L197.8 79.7 L192.1 91.3 L185.7 102.8 L178.6 114.4 L171.1 125.9 L163.2 137.5 L155.3 149.0 L147.3 160.6 L139.6 172.2 L132.2 183.7 L125.4 195.3 L119.2 206.8 L113.9 218.4 L109.4 229.9 L105.9 241.5 L103.4 253.0 L102.0 264.6 L101.7 276.2 L102.5 287.7 L104.4 299.3 L107.2 310.8 L110.9 322.4 L115.4 333.9 L120.5 345.5 L126.2 357.0 L132.2 368.6 L138.5 380.2 L144.8 391.7 L151.1 403.3 L157.0 414.8 L162.5 426.4 L167.5 437.9 L171.8 449.5 L175.3 461.0 L177.9 472.6 L179.6 484.2 L180.2 495.7 L179.9 507.3 L178.6 518.8"
                  style={{ opacity: '0.80', animationDelay: '-0.6s' }}
                ></path>
                <path
                  className="flow"
                  d="M228.9 -9.1 L229.4 2.4 L228.7 14.0 L226.9 25.5 L224.0 37.1 L220.0 48.6 L215.1 60.2 L209.2 71.8 L202.7 83.3 L195.5 94.9 L187.9 106.4 L180.0 118.0 L171.9 129.5 L163.9 141.1 L156.1 152.6 L148.7 164.2 L141.9 175.8 L135.7 187.3 L130.3 198.9 L125.8 210.4 L122.2 222.0 L119.8 233.5 L118.4 245.1 L118.2 256.6 L119.0 268.2 L120.9 279.8 L123.7 291.3 L127.5 302.9 L132.1 314.4 L137.3 326.0 L143.1 337.5 L149.3 349.1 L155.7 360.6 L162.1 372.2 L168.5 383.8 L174.6 395.3 L180.3 406.9 L185.4 418.4 L189.8 430.0 L193.5 441.5 L196.3 453.1 L198.1 464.6 L199.0 476.2 L198.8 487.8 L197.6 499.3 L195.5 510.9"
                  style={{ opacity: '0.84', animationDelay: '-1.2s' }}
                ></path>
                <path
                  className="flow"
                  d="M246.9 -15.6 L246.1 -4.1 L244.2 7.5 L241.2 19.0 L237.1 30.6 L232.0 42.1 L226.1 53.7 L219.5 65.2 L212.2 76.8 L204.5 88.4 L196.6 99.9 L188.5 111.5 L180.4 123.0 L172.6 134.6 L165.2 146.1 L158.3 157.7 L152.1 169.2 L146.7 180.8 L142.2 192.4 L138.7 203.9 L136.2 215.5 L134.9 227.0 L134.7 238.6 L135.6 250.1 L137.5 261.7 L140.5 273.2 L144.4 284.8 L149.0 296.4 L154.4 307.9 L160.3 319.5 L166.6 331.0 L173.1 342.6 L179.7 354.1 L186.2 365.7 L192.5 377.2 L198.3 388.8 L203.6 400.4 L208.3 411.9 L212.1 423.5 L215.1 435.0 L217.1 446.6 L218.1 458.1 L218.1 469.7 L217.1 481.2 L215.1 492.8 L212.2 504.4"
                  style={{ opacity: '0.88', animationDelay: '-1.8s' }}
                ></path>
                <path
                  className="flow"
                  d="M263.2 -19.7 L261.2 -8.1 L258.1 3.4 L253.9 15.0 L248.8 26.5 L242.8 38.1 L236.1 49.6 L228.8 61.2 L221.1 72.8 L213.1 84.3 L204.9 95.9 L196.9 107.4 L189.0 119.0 L181.6 130.5 L174.7 142.1 L168.5 153.6 L163.1 165.2 L158.7 176.8 L155.2 188.3 L152.9 199.9 L151.6 211.4 L151.4 223.0 L152.4 234.5 L154.5 246.1 L157.5 257.6 L161.5 269.2 L166.3 280.8 L171.8 292.3 L177.8 303.9 L184.2 315.4 L190.9 327.0 L197.7 338.5 L204.3 350.1 L210.7 361.6 L216.8 373.2 L222.2 384.8 L227.0 396.3 L231.0 407.9 L234.2 419.4 L236.4 431.0 L237.6 442.5 L237.8 454.1 L236.9 465.6 L235.1 477.2 L232.4 488.8 L228.8 500.3"
                  style={{ opacity: '0.91', animationDelay: '-2.4s' }}
                ></path>
                <path
                  className="flow"
                  d="M278.0 -20.6 L274.8 -9.1 L270.5 2.5 L265.4 14.0 L259.3 25.6 L252.6 37.2 L245.3 48.7 L237.5 60.3 L229.5 71.8 L221.4 83.4 L213.3 94.9 L205.5 106.5 L198.1 118.0 L191.2 129.6 L185.1 141.2 L179.7 152.7 L175.3 164.3 L171.9 175.8 L169.6 187.4 L168.5 198.9 L168.4 210.5 L169.5 222.0 L171.6 233.6 L174.8 245.2 L178.9 256.7 L183.8 268.3 L189.4 279.8 L195.6 291.4 L202.2 302.9 L209.0 314.5 L216.0 326.0 L222.8 337.6 L229.4 349.2 L235.6 360.7 L241.2 372.3 L246.2 383.8 L250.4 395.4 L253.7 406.9 L256.0 418.5 L257.4 430.0 L257.7 441.6 L257.1 453.2 L255.4 464.7 L252.8 476.3 L249.3 487.8 L245.1 499.4"
                  style={{ opacity: '0.94', animationDelay: '-3.0s' }}
                ></path>
                <path
                  className="flow"
                  d="M291.3 -18.3 L287.0 -6.8 L281.8 4.8 L275.8 16.4 L269.0 27.9 L261.7 39.5 L254.0 51.0 L245.9 62.6 L237.8 74.1 L229.8 85.7 L222.0 97.2 L214.7 108.8 L207.8 120.4 L201.8 131.9 L196.5 143.5 L192.2 155.0 L188.9 166.6 L186.7 178.1 L185.6 189.7 L185.7 201.2 L186.8 212.8 L189.1 224.4 L192.4 235.9 L196.6 247.5 L201.7 259.0 L207.5 270.6 L213.8 282.1 L220.5 293.7 L227.5 305.2 L234.6 316.8 L241.6 328.4 L248.4 339.9 L254.7 351.5 L260.6 363.0 L265.7 374.6 L270.0 386.1 L273.5 397.7 L276.0 409.2 L277.5 420.8 L278.0 432.4 L277.5 443.9 L276.0 455.5 L273.5 467.0 L270.1 478.6 L266.0 490.1 L261.2 501.7"
                  style={{ opacity: '0.96', animationDelay: '-3.6s' }}
                ></path>
                <path
                  className="flow"
                  d="M303.5 -13.1 L298.2 -1.5 L292.2 10.0 L285.5 21.6 L278.1 33.1 L270.4 44.7 L262.4 56.2 L254.4 67.8 L246.4 79.3 L238.7 90.9 L231.4 102.5 L224.7 114.0 L218.7 125.6 L213.5 137.1 L209.3 148.7 L206.1 160.2 L204.0 171.8 L203.0 183.3 L203.2 194.9 L204.5 206.5 L206.9 218.0 L210.4 229.6 L214.7 241.1 L220.0 252.7 L225.9 264.2 L232.4 275.8 L239.3 287.3 L246.4 298.9 L253.7 310.5 L260.8 322.0 L267.8 333.6 L274.3 345.1 L280.2 356.7 L285.5 368.2 L290.1 379.8 L293.7 391.3 L296.3 402.9 L298.0 414.5 L298.6 426.0 L298.2 437.6 L296.8 449.1 L294.4 460.7 L291.1 472.2 L287.1 483.8 L282.4 495.3 L277.1 506.9"
                  style={{ opacity: '0.98', animationDelay: '-0.0s' }}
                ></path>
                <path
                  className="flow"
                  d="M314.7 -5.8 L308.6 5.7 L301.9 17.3 L294.7 28.8 L287.0 40.4 L279.1 51.9 L271.1 63.5 L263.2 75.1 L255.5 86.6 L248.3 98.2 L241.7 109.7 L235.8 121.3 L230.7 132.8 L226.6 144.4 L223.6 155.9 L221.6 167.5 L220.8 179.1 L221.1 190.6 L222.5 202.2 L225.1 213.7 L228.7 225.3 L233.2 236.8 L238.6 248.4 L244.7 259.9 L251.3 271.5 L258.4 283.1 L265.7 294.6 L273.1 306.2 L280.4 317.7 L287.5 329.3 L294.1 340.8 L300.3 352.4 L305.7 363.9 L310.4 375.5 L314.1 387.1 L316.9 398.6 L318.7 410.2 L319.4 421.7 L319.1 433.3 L317.8 444.8 L315.5 456.4 L312.3 467.9 L308.3 479.5 L303.7 491.1 L298.4 502.6 L292.8 514.2"
                  style={{ opacity: '0.99', animationDelay: '-0.6s' }}
                ></path>
                <path
                  className="flow"
                  d="M325.1 2.4 L318.5 13.9 L311.3 25.5 L303.7 37.0 L295.9 48.6 L288.0 60.1 L280.2 71.7 L272.6 83.2 L265.5 94.8 L259.0 106.4 L253.2 117.9 L248.3 129.5 L244.3 141.0 L241.4 152.6 L239.6 164.1 L238.9 175.7 L239.3 187.2 L240.9 198.8 L243.6 210.4 L247.4 221.9 L252.0 233.5 L257.6 245.0 L263.8 256.6 L270.6 268.1 L277.8 279.7 L285.3 291.2 L292.8 302.8 L300.3 314.4 L307.5 325.9 L314.3 337.5 L320.6 349.0 L326.2 360.6 L330.9 372.1 L334.8 383.7 L337.7 395.2 L339.6 406.8 L340.4 418.4 L340.2 429.9 L338.9 441.5 L336.7 453.0 L333.6 464.6 L329.6 476.1 L325.0 487.7 L319.8 499.2 L314.1 510.8 L308.2 522.4"
                  style={{ opacity: '1.00', animationDelay: '-1.2s' }}
                ></path>
                <path
                  className="flow"
                  d="M335.2 10.2 L328.1 21.7 L320.6 33.3 L312.9 44.8 L305.1 56.4 L297.4 68.0 L290.0 79.5 L283.0 91.1 L276.6 102.6 L271.0 114.2 L266.2 125.7 L262.3 137.3 L259.6 148.8 L257.9 160.4 L257.3 172.0 L257.9 183.5 L259.7 195.1 L262.5 206.6 L266.4 218.2 L271.3 229.7 L276.9 241.3 L283.3 252.8 L290.2 264.4 L297.6 276.0 L305.2 287.5 L312.9 299.1 L320.5 310.6 L327.9 322.2 L334.8 333.7 L341.2 345.3 L346.9 356.8 L351.7 368.4 L355.7 380.0 L358.7 391.5 L360.7 403.1 L361.5 414.6 L361.4 426.2 L360.2 437.7 L358.0 449.3 L354.9 460.8 L350.9 472.4 L346.3 484.0 L341.0 495.5 L335.4 507.1 L329.4 518.6 L323.4 530.2"
                  style={{ opacity: '1.00', animationDelay: '-1.8s' }}
                ></path>
                <path
                  className="flow"
                  d="M345.2 16.4 L337.8 27.9 L330.2 39.5 L322.5 51.1 L314.9 62.6 L307.7 74.2 L300.8 85.7 L294.6 97.3 L289.1 108.8 L284.4 120.4 L280.7 131.9 L278.1 143.5 L276.6 155.1 L276.2 166.6 L276.9 178.2 L278.8 189.7 L281.8 201.3 L285.8 212.8 L290.8 224.4 L296.6 235.9 L303.1 247.5 L310.2 259.1 L317.7 270.6 L325.5 282.2 L333.3 293.7 L341.0 305.3 L348.5 316.8 L355.5 328.4 L362.0 339.9 L367.8 351.5 L372.7 363.1 L376.8 374.6 L379.8 386.2 L381.8 397.7 L382.8 409.3 L382.6 420.8 L381.4 432.4 L379.3 443.9 L376.1 455.5 L372.2 467.1 L367.5 478.6 L362.3 490.2 L356.5 501.7 L350.5 513.3 L344.4 524.8 L338.3 536.4"
                  style={{ opacity: '0.99', animationDelay: '-2.4s' }}
                ></path>
                <path
                  className="flow"
                  d="M355.2 20.0 L347.7 31.6 L340.2 43.1 L332.8 54.7 L325.7 66.2 L319.0 77.8 L312.9 89.4 L307.5 100.9 L303.0 112.5 L299.5 124.0 L297.0 135.6 L295.6 147.1 L295.4 158.7 L296.3 170.2 L298.3 181.8 L301.5 193.4 L305.6 204.9 L310.7 216.5 L316.7 228.0 L323.3 239.6 L330.5 251.1 L338.1 262.7 L346.0 274.2 L353.9 285.8 L361.7 297.4 L369.3 308.9 L376.4 320.5 L383.0 332.0 L388.8 343.6 L393.8 355.1 L397.9 366.7 L401.0 378.2 L403.1 389.8 L404.0 401.4 L403.9 412.9 L402.7 424.5 L400.5 436.0 L397.4 447.6 L393.4 459.1 L388.6 470.7 L383.3 482.2 L377.5 493.8 L371.4 505.4 L365.2 516.9 L359.0 528.5 L352.9 540.0"
                  style={{ opacity: '0.98', animationDelay: '-3.0s' }}
                ></path>
                <path
                  className="flow"
                  d="M365.7 20.5 L358.3 32.0 L351.0 43.6 L344.0 55.2 L337.5 66.7 L331.6 78.3 L326.4 89.8 L322.0 101.4 L318.6 112.9 L316.3 124.5 L315.0 136.0 L314.9 147.6 L316.0 159.2 L318.2 170.7 L321.4 182.3 L325.7 193.8 L330.9 205.4 L337.0 216.9 L343.8 228.5 L351.1 240.0 L358.8 251.6 L366.7 263.2 L374.7 274.7 L382.6 286.3 L390.3 297.8 L397.4 309.4 L404.1 320.9 L409.9 332.5 L415.0 344.0 L419.1 355.6 L422.2 367.2 L424.3 378.7 L425.2 390.3 L425.1 401.8 L423.9 413.4 L421.6 424.9 L418.4 436.5 L414.4 448.0 L409.6 459.6 L404.2 471.2 L398.3 482.7 L392.1 494.3 L385.7 505.8 L379.4 517.4 L373.2 528.9 L367.3 540.5"
                  style={{ opacity: '0.96', animationDelay: '-3.6s' }}
                ></path>
                <path
                  className="flow"
                  d="M376.7 17.7 L369.6 29.3 L362.8 40.8 L356.4 52.4 L350.6 63.9 L345.6 75.5 L341.4 87.1 L338.1 98.6 L335.9 110.2 L334.8 121.7 L334.8 133.3 L336.0 144.8 L338.3 156.4 L341.7 167.9 L346.1 179.5 L351.4 191.1 L357.6 202.6 L364.4 214.2 L371.8 225.7 L379.6 237.3 L387.6 248.8 L395.7 260.4 L403.7 271.9 L411.3 283.5 L418.6 295.1 L425.2 306.6 L431.1 318.2 L436.2 329.7 L440.3 341.3 L443.4 352.8 L445.4 364.4 L446.4 375.9 L446.2 387.5 L444.9 399.1 L442.6 410.6 L439.3 422.2 L435.2 433.7 L430.3 445.3 L424.8 456.8 L418.8 468.4 L412.5 479.9 L406.0 491.5 L399.5 503.1 L393.2 514.6 L387.2 526.2 L381.6 537.7"
                  style={{ opacity: '0.94', animationDelay: '-0.0s' }}
                ></path>
                <path
                  className="flow"
                  d="M388.6 12.2 L381.9 23.7 L375.7 35.3 L370.0 46.8 L365.1 58.4 L361.1 69.9 L358.0 81.5 L355.9 93.1 L354.9 104.6 L355.1 116.2 L356.4 127.7 L358.8 139.3 L362.2 150.8 L366.7 162.4 L372.2 173.9 L378.4 185.5 L385.3 197.1 L392.8 208.6 L400.6 220.2 L408.7 231.7 L416.8 243.3 L424.8 254.8 L432.5 266.4 L439.7 277.9 L446.4 289.5 L452.3 301.1 L457.3 312.6 L461.4 324.2 L464.5 335.7 L466.5 347.3 L467.4 358.8 L467.1 370.4 L465.8 381.9 L463.4 393.5 L460.0 405.1 L455.8 416.6 L450.8 428.2 L445.2 439.7 L439.0 451.3 L432.6 462.8 L425.9 474.4 L419.3 485.9 L412.8 497.5 L406.6 509.1 L400.9 520.6 L395.8 532.2"
                  style={{ opacity: '0.91', animationDelay: '-0.6s' }}
                ></path>
                <path
                  className="flow"
                  d="M401.4 4.7 L395.3 16.2 L389.8 27.8 L385.1 39.4 L381.1 50.9 L378.1 62.5 L376.2 74.0 L375.3 85.6 L375.6 97.1 L377.0 108.7 L379.5 120.2 L383.0 131.8 L387.6 143.4 L393.1 154.9 L399.4 166.5 L406.4 178.0 L413.9 189.6 L421.7 201.1 L429.8 212.7 L437.9 224.2 L445.9 235.8 L453.6 247.4 L460.9 258.9 L467.5 270.5 L473.4 282.0 L478.4 293.6 L482.4 305.1 L485.4 316.7 L487.4 328.2 L488.2 339.8 L487.8 351.4 L486.4 362.9 L483.9 374.5 L480.4 386.0 L476.1 397.6 L471.0 409.1 L465.2 420.7 L458.9 432.2 L452.3 443.8 L445.5 455.4 L438.7 466.9 L432.1 478.5 L425.7 490.0 L419.8 501.6 L414.5 513.1 L410.0 524.7"
                  style={{ opacity: '0.88', animationDelay: '-1.2s' }}
                ></path>
                <path
                  className="flow"
                  d="M415.3 -3.5 L409.9 8.0 L405.3 19.6 L401.5 31.1 L398.6 42.7 L396.8 54.2 L396.0 65.8 L396.3 77.4 L397.8 88.9 L400.4 100.5 L404.0 112.0 L408.6 123.6 L414.1 135.1 L420.5 146.7 L427.5 158.2 L435.0 169.8 L442.9 181.4 L451.0 192.9 L459.1 204.5 L467.0 216.0 L474.7 227.6 L481.9 239.1 L488.5 250.7 L494.3 262.2 L499.3 273.8 L503.3 285.4 L506.2 296.9 L508.0 308.5 L508.7 320.0 L508.3 331.6 L506.8 343.1 L504.1 354.7 L500.6 366.2 L496.1 377.8 L490.8 389.4 L484.9 400.9 L478.5 412.5 L471.7 424.0 L464.8 435.6 L457.8 447.1 L451.0 458.7 L444.5 470.2 L438.4 481.8 L432.9 493.4 L428.2 504.9 L424.3 516.5"
                  style={{ opacity: '0.84', animationDelay: '-1.8s' }}
                ></path>
                <path
                  className="flow"
                  d="M430.3 -11.2 L425.8 0.4 L422.1 11.9 L419.3 23.5 L417.6 35.0 L416.9 46.6 L417.3 58.1 L418.8 69.7 L421.4 81.3 L425.1 92.8 L429.7 104.4 L435.3 115.9 L441.6 127.5 L448.6 139.0 L456.1 150.6 L464.0 162.1 L472.1 173.7 L480.1 185.3 L488.1 196.8 L495.7 208.4 L502.8 219.9 L509.3 231.5 L515.1 243.0 L520.0 254.6 L523.9 266.1 L526.7 277.7 L528.4 289.3 L529.0 300.8 L528.5 312.4 L526.8 323.9 L524.1 335.5 L520.3 347.0 L515.7 358.6 L510.3 370.1 L504.2 381.7 L497.7 393.3 L490.7 404.8 L483.6 416.4 L476.5 427.9 L469.5 439.5 L462.8 451.0 L456.6 462.6 L451.0 474.1 L446.0 485.7 L442.0 497.3 L438.8 508.8"
                  style={{ opacity: '0.80', animationDelay: '-2.4s' }}
                ></path>
                <path
                  className="flow"
                  d="M446.6 -17.1 L443.0 -5.5 L440.3 6.0 L438.6 17.6 L437.9 29.1 L438.4 40.7 L439.9 52.2 L442.6 63.8 L446.3 75.4 L450.9 86.9 L456.5 98.5 L462.8 110.0 L469.8 121.6 L477.2 133.1 L485.1 144.7 L493.1 156.2 L501.1 167.8 L509.0 179.4 L516.5 190.9 L523.6 202.5 L530.0 214.0 L535.6 225.6 L540.4 237.1 L544.2 248.7 L546.9 260.2 L548.5 271.8 L549.0 283.4 L548.3 294.9 L546.5 306.5 L543.6 318.0 L539.8 329.6 L535.0 341.1 L529.4 352.7 L523.2 364.2 L516.5 375.8 L509.4 387.4 L502.1 398.9 L494.8 410.5 L487.7 422.0 L480.8 433.6 L474.4 445.1 L468.6 456.7 L463.6 468.2 L459.4 479.8 L456.0 491.4 L453.7 502.9"
                  style={{ opacity: '0.75', animationDelay: '-3.0s' }}
                ></path>
                <path
                  className="flow"
                  d="M464.0 -20.3 L461.4 -8.7 L459.7 2.8 L459.1 14.4 L459.6 25.9 L461.2 37.5 L463.8 49.0 L467.5 60.6 L472.1 72.2 L477.6 83.7 L483.9 95.3 L490.8 106.8 L498.2 118.4 L506.0 129.9 L513.9 141.5 L521.9 153.0 L529.6 164.6 L537.1 176.2 L544.1 187.7 L550.4 199.3 L555.9 210.8 L560.6 222.4 L564.2 233.9 L566.8 245.5 L568.3 257.0 L568.6 268.6 L567.8 280.2 L565.9 291.7 L562.8 303.3 L558.8 314.8 L553.9 326.4 L548.2 337.9 L541.8 349.5 L534.9 361.0 L527.7 372.6 L520.2 384.2 L512.8 395.7 L505.5 407.3 L498.5 418.8 L492.0 430.4 L486.0 441.9 L480.8 453.5 L476.5 465.0 L473.0 476.6 L470.6 488.2 L469.2 499.7"
                  style={{ opacity: '0.70', animationDelay: '-3.6s' }}
                ></path>
                <path
                  className="flow"
                  d="M482.6 -20.3 L481.0 -8.7 L480.4 2.8 L480.8 14.4 L482.4 25.9 L485.0 37.5 L488.6 49.0 L493.2 60.6 L498.7 72.2 L504.9 83.7 L511.7 95.3 L519.1 106.8 L526.7 118.4 L534.6 129.9 L542.4 141.5 L550.1 153.0 L557.4 164.6 L564.3 176.2 L570.5 187.7 L575.9 199.3 L580.4 210.8 L583.9 222.4 L586.4 233.9 L587.7 245.5 L587.9 257.0 L586.9 268.6 L584.8 280.2 L581.7 291.7 L577.5 303.3 L572.4 314.8 L566.6 326.4 L560.0 337.9 L553.0 349.5 L545.6 361.0 L538.0 372.6 L530.4 384.2 L523.0 395.7 L515.8 407.3 L509.2 418.8 L503.1 430.4 L497.8 441.9 L493.3 453.5 L489.8 465.0 L487.2 476.6 L485.7 488.2 L485.3 499.7"
                  style={{ opacity: '0.65', animationDelay: '-0.0s' }}
                ></path>
                <path
                  className="flow"
                  d="M502.2 -17.1 L501.6 -5.5 L502.0 6.0 L503.6 17.6 L506.1 29.1 L509.7 40.7 L514.2 52.2 L519.6 63.8 L525.7 75.4 L532.5 86.9 L539.7 98.5 L547.3 110.0 L555.0 121.6 L562.7 133.1 L570.3 144.7 L577.5 156.2 L584.2 167.8 L590.2 179.4 L595.5 190.9 L599.9 202.5 L603.3 214.0 L605.6 225.6 L606.8 237.1 L606.8 248.7 L605.7 260.2 L603.5 271.8 L600.1 283.4 L595.8 294.9 L590.6 306.5 L584.6 318.0 L577.9 329.6 L570.7 341.1 L563.2 352.7 L555.5 364.2 L547.7 375.8 L540.2 387.4 L532.9 398.9 L526.1 410.5 L520.0 422.0 L514.6 433.6 L510.0 445.1 L506.4 456.7 L503.7 468.2 L502.2 479.8 L501.7 491.4 L502.2 502.9"
                  style={{ opacity: '0.59', animationDelay: '-0.6s' }}
                ></path>
                <path
                  className="flow"
                  d="M522.8 -11.2 L523.2 0.4 L524.6 11.9 L527.1 23.5 L530.6 35.0 L535.0 46.6 L540.3 58.1 L546.3 69.7 L553.0 81.3 L560.1 92.8 L567.5 104.4 L575.1 115.9 L582.7 127.5 L590.1 139.0 L597.2 150.6 L603.8 162.1 L609.7 173.7 L614.8 185.3 L619.0 196.8 L622.3 208.4 L624.4 219.9 L625.5 231.5 L625.4 243.0 L624.1 254.6 L621.7 266.1 L618.2 277.7 L613.8 289.3 L608.4 300.8 L602.3 312.4 L595.4 323.9 L588.1 335.5 L580.5 347.0 L572.7 358.6 L564.8 370.1 L557.1 381.7 L549.8 393.3 L542.9 404.8 L536.6 416.4 L531.1 427.9 L526.5 439.5 L522.8 451.0 L520.1 462.6 L518.5 474.1 L518.0 485.7 L518.6 497.3 L520.1 508.8"
                  style={{ opacity: '0.53', animationDelay: '-1.2s' }}
                ></path>
                <path
                  className="flow"
                  d="M544.2 -3.5 L545.6 8.0 L548.0 19.6 L551.3 31.1 L555.7 42.7 L560.8 54.2 L566.7 65.8 L573.2 77.4 L580.2 88.9 L587.5 100.5 L594.9 112.0 L602.4 123.6 L609.6 135.1 L616.6 146.7 L623.0 158.2 L628.7 169.8 L633.7 181.4 L637.8 192.9 L640.9 204.5 L642.9 216.0 L643.8 227.6 L643.5 239.1 L642.1 250.7 L639.6 262.2 L636.0 273.8 L631.4 285.4 L625.9 296.9 L619.6 308.5 L612.7 320.0 L605.3 331.6 L597.5 343.1 L589.6 354.7 L581.6 366.2 L573.9 377.8 L566.4 389.4 L559.5 400.9 L553.2 412.5 L547.6 424.0 L542.9 435.6 L539.2 447.1 L536.5 458.7 L534.9 470.2 L534.3 481.8 L534.9 493.4 L536.5 504.9 L539.1 516.5"
                  style={{ opacity: '0.47', animationDelay: '-1.8s' }}
                ></path>
                <path
                  className="flow"
                  d="M566.3 4.7 L568.6 16.2 L571.8 27.8 L576.0 39.4 L581.0 50.9 L586.8 62.5 L593.1 74.0 L600.0 85.6 L607.1 97.1 L614.4 108.7 L621.7 120.2 L628.8 131.8 L635.6 143.4 L641.8 154.9 L647.4 166.5 L652.3 178.0 L656.2 189.6 L659.1 201.1 L661.0 212.7 L661.7 224.2 L661.3 235.8 L659.8 247.4 L657.2 258.9 L653.4 270.5 L648.7 282.0 L643.1 293.6 L636.7 305.1 L629.7 316.7 L622.2 328.2 L614.3 339.8 L606.3 351.4 L598.3 362.9 L590.5 374.5 L583.0 386.0 L576.0 397.6 L569.6 409.1 L564.0 420.7 L559.3 432.2 L555.6 443.8 L552.9 455.4 L551.3 466.9 L550.7 478.5 L551.3 490.0 L553.0 501.6 L555.6 513.1 L559.2 524.7"
                  style={{ opacity: '0.41', animationDelay: '-2.4s' }}
                ></path>
                <path
                  className="flow"
                  d="M588.9 12.2 L592.0 23.7 L596.1 35.3 L600.9 46.8 L606.5 58.4 L612.7 69.9 L619.4 81.5 L626.4 93.1 L633.5 104.6 L640.6 116.2 L647.6 127.7 L654.2 139.3 L660.3 150.8 L665.8 162.4 L670.4 173.9 L674.2 185.5 L677.0 197.1 L678.7 208.6 L679.4 220.2 L678.8 231.7 L677.2 243.3 L674.4 254.8 L670.6 266.4 L665.8 277.9 L660.1 289.5 L653.6 301.1 L646.5 312.6 L638.9 324.2 L631.0 335.7 L622.9 347.3 L614.8 358.8 L606.9 370.4 L599.4 381.9 L592.4 393.5 L586.0 405.1 L580.4 416.6 L575.7 428.2 L572.0 439.7 L569.3 451.3 L567.7 462.8 L567.2 474.4 L567.9 485.9 L569.6 497.5 L572.3 509.1 L575.9 520.6 L580.4 532.2"
                  style={{ opacity: '0.35', animationDelay: '-3.0s' }}
                ></path>
              </g>
            </svg>
          </div>
          <div className="field f2" style={{ right: '-200px', top: '-80px' }}>
            <svg viewBox="0 0 600 500" width="600" height="500" aria-hidden="true" style={{ overflow: 'visible' }}>
              <defs>
                <radialGradient id="wmBg" cx="50%" cy="50%" r="50%">
                  <stop offset="0" stopColor="#fff" stopOpacity="1"></stop>
                  <stop offset="0.65" stopColor="#fff" stopOpacity="0.55"></stop>
                  <stop offset="1" stopColor="#fff" stopOpacity="0"></stop>
                </radialGradient>
                <mask id="wmB">
                  <rect x="-60" y="-60" width="720" height="620" fill="url(#wmBg)"></rect>
                </mask>
              </defs>
              <g mask="url(#wmB)" fill="none" stroke="#76B900" strokeWidth="2.2" strokeLinecap="round" strokeDasharray="0.1 7">
                <path
                  className="flow"
                  d="M39.7 17.8 L39.0 29.5 L36.9 41.1 L33.5 52.7 L29.1 64.4 L23.7 76.0 L17.6 87.6 L11.0 99.2 L4.3 110.9 L-2.4 122.5 L-8.7 134.1 L-14.4 145.8 L-19.3 157.4 L-23.2 169.0 L-25.9 180.6 L-27.4 192.3 L-27.4 203.9 L-26.1 215.5 L-23.4 227.2 L-19.5 238.8 L-14.5 250.4 L-8.5 262.0 L-1.7 273.7 L5.5 285.3 L13.0 296.9 L20.4 308.5 L27.5 320.2 L34.1 331.8 L39.8 343.4 L44.5 355.1 L48.1 366.7 L50.2 378.3 L50.9 389.9 L50.2 401.6 L47.9 413.2 L44.2 424.8 L39.2 436.5 L33.1 448.1 L26.0 459.7 L18.2 471.3 L9.9 483.0 L1.3 494.6 L-7.1 506.2 L-15.2 517.8"
                  style={{ opacity: '0.35', animationDelay: '-0.0s' }}
                ></path>
                <path
                  className="flow"
                  d="M60.9 15.3 L59.0 26.9 L55.8 38.6 L51.4 50.2 L46.1 61.8 L40.0 73.4 L33.5 85.1 L26.6 96.7 L19.8 108.3 L13.3 120.0 L7.4 131.6 L2.2 143.2 L-2.0 154.8 L-5.0 166.5 L-6.8 178.1 L-7.2 189.7 L-6.3 201.3 L-4.0 213.0 L-0.4 224.6 L4.3 236.2 L10.0 247.9 L16.5 259.5 L23.5 271.1 L30.7 282.7 L38.0 294.4 L45.1 306.0 L51.6 317.6 L57.3 329.3 L62.0 340.9 L65.6 352.5 L67.8 364.1 L68.6 375.8 L68.0 387.4 L65.8 399.0 L62.2 410.6 L57.4 422.3 L51.3 433.9 L44.2 445.5 L36.5 457.2 L28.1 468.8 L19.6 480.4 L11.1 492.0 L2.9 503.7 L-4.7 515.3"
                  style={{ opacity: '0.42', animationDelay: '-0.6s' }}
                ></path>
                <path
                  className="flow"
                  d="M81.0 10.0 L77.9 21.6 L73.6 33.3 L68.3 44.9 L62.3 56.5 L55.6 68.2 L48.7 79.8 L41.8 91.4 L35.1 103.0 L28.9 114.7 L23.4 126.3 L18.9 137.9 L15.5 149.6 L13.4 161.2 L12.6 172.8 L13.2 184.4 L15.1 196.1 L18.3 207.7 L22.7 219.3 L28.1 230.9 L34.3 242.6 L41.1 254.2 L48.2 265.8 L55.3 277.5 L62.3 289.1 L68.7 300.7 L74.4 312.3 L79.2 324.0 L82.8 335.6 L85.1 347.2 L86.1 358.9 L85.5 370.5 L83.5 382.1 L80.1 393.7 L75.3 405.4 L69.3 417.0 L62.4 428.6 L54.6 440.3 L46.4 451.9 L37.8 463.5 L29.3 475.1 L21.1 486.8 L13.4 498.4 L6.5 510.0"
                  style={{ opacity: '0.48', animationDelay: '-1.2s' }}
                ></path>
                <path
                  className="flow"
                  d="M99.8 3.0 L95.6 14.6 L90.3 26.2 L84.2 37.8 L77.5 49.5 L70.5 61.1 L63.4 72.7 L56.5 84.3 L50.1 96.0 L44.3 107.6 L39.5 119.2 L35.8 130.9 L33.3 142.5 L32.1 154.1 L32.3 165.7 L33.9 177.4 L36.8 189.0 L40.8 200.6 L45.9 212.3 L51.8 223.9 L58.4 235.5 L65.3 247.1 L72.3 258.8 L79.1 270.4 L85.5 282.0 L91.3 293.6 L96.1 305.3 L99.8 316.9 L102.2 328.5 L103.2 340.2 L102.8 351.8 L101.0 363.4 L97.7 375.0 L93.1 386.7 L87.2 398.3 L80.4 409.9 L72.8 421.6 L64.6 433.2 L56.1 444.8 L47.5 456.4 L39.3 468.1 L31.6 479.7 L24.6 491.3 L18.7 503.0"
                  style={{ opacity: '0.54', animationDelay: '-1.8s' }}
                ></path>
                <path
                  className="flow"
                  d="M117.3 -4.6 L112.0 7.0 L105.9 18.6 L99.1 30.2 L92.0 41.9 L84.7 53.5 L77.6 65.1 L70.9 76.8 L64.9 88.4 L59.8 100.0 L55.7 111.6 L52.8 123.3 L51.3 134.9 L51.1 146.5 L52.3 158.1 L54.8 169.8 L58.6 181.4 L63.3 193.0 L69.0 204.7 L75.3 216.3 L82.1 227.9 L88.9 239.5 L95.7 251.2 L102.1 262.8 L107.8 274.4 L112.7 286.1 L116.5 297.7 L119.0 309.3 L120.2 320.9 L120.0 332.6 L118.3 344.2 L115.2 355.8 L110.8 367.4 L105.1 379.1 L98.4 390.7 L90.9 402.3 L82.8 414.0 L74.4 425.6 L65.9 437.2 L57.7 448.8 L49.9 460.5 L42.9 472.1 L36.9 483.7 L32.0 495.4"
                  style={{ opacity: '0.61', animationDelay: '-2.4s' }}
                ></path>
                <path
                  className="flow"
                  d="M133.4 -11.4 L127.2 0.2 L120.4 11.8 L113.1 23.5 L105.7 35.1 L98.4 46.7 L91.4 58.4 L85.1 70.0 L79.7 81.6 L75.3 93.2 L72.0 104.9 L70.1 116.5 L69.6 128.1 L70.4 139.8 L72.6 151.4 L76.0 163.0 L80.5 174.6 L85.9 186.3 L92.0 197.9 L98.6 209.5 L105.4 221.1 L112.1 232.8 L118.4 244.4 L124.2 256.0 L129.1 267.7 L133.0 279.3 L135.7 290.9 L137.1 302.5 L137.1 314.2 L135.6 325.8 L132.7 337.4 L128.5 349.1 L123.0 360.7 L116.5 372.3 L109.1 383.9 L101.2 395.6 L92.8 407.2 L84.4 418.8 L76.2 430.5 L68.5 442.1 L61.4 453.7 L55.4 465.3 L50.5 477.0 L46.9 488.6"
                  style={{ opacity: '0.67', animationDelay: '-3.0s' }}
                ></path>
                <path
                  className="flow"
                  d="M148.2 -16.1 L141.3 -4.5 L133.9 7.1 L126.3 18.7 L118.8 30.4 L111.6 42.0 L105.0 53.6 L99.2 65.3 L94.5 76.9 L90.9 88.5 L88.7 100.1 L87.8 111.8 L88.2 123.4 L90.1 135.0 L93.2 146.7 L97.4 158.3 L102.5 169.9 L108.5 181.5 L114.9 193.2 L121.6 204.8 L128.2 216.4 L134.6 228.0 L140.4 239.7 L145.4 251.3 L149.5 262.9 L152.3 274.6 L153.9 286.2 L154.1 297.8 L152.9 309.4 L150.2 321.1 L146.2 332.7 L141.0 344.3 L134.6 356.0 L127.5 367.6 L119.7 379.2 L111.5 390.8 L103.1 402.5 L95.0 414.1 L87.3 425.7 L80.3 437.3 L74.2 449.0 L69.2 460.6 L65.6 472.2 L63.3 483.9"
                  style={{ opacity: '0.72', animationDelay: '-3.6s' }}
                ></path>
                <path
                  className="flow"
                  d="M161.8 -18.0 L154.3 -6.4 L146.5 5.3 L138.8 16.9 L131.4 28.5 L124.5 40.1 L118.5 51.8 L113.4 63.4 L109.5 75.0 L106.9 86.7 L105.7 98.3 L105.8 109.9 L107.3 121.5 L110.1 133.2 L114.1 144.8 L119.0 156.4 L124.7 168.1 L131.0 179.7 L137.6 191.3 L144.2 202.9 L150.6 214.6 L156.5 226.2 L161.7 237.8 L165.9 249.5 L168.9 261.1 L170.7 272.7 L171.1 284.3 L170.2 296.0 L167.8 307.6 L164.0 319.2 L159.0 330.8 L153.0 342.5 L146.0 354.1 L138.4 365.7 L130.3 377.4 L122.1 389.0 L114.1 400.6 L106.4 412.2 L99.4 423.9 L93.3 435.5 L88.4 447.1 L84.7 458.8 L82.3 470.4 L81.5 482.0"
                  style={{ opacity: '0.77', animationDelay: '-0.0s' }}
                ></path>
                <path
                  className="flow"
                  d="M174.3 -16.6 L166.4 -5.0 L158.5 6.6 L150.8 18.3 L143.7 29.9 L137.4 41.5 L132.0 53.1 L127.8 64.8 L124.9 76.4 L123.3 88.0 L123.1 99.6 L124.3 111.3 L126.9 122.9 L130.6 134.5 L135.3 146.2 L140.9 157.8 L147.0 169.4 L153.6 181.0 L160.2 192.7 L166.6 204.3 L172.6 215.9 L177.9 227.6 L182.3 239.2 L185.5 250.8 L187.6 262.4 L188.3 274.1 L187.6 285.7 L185.5 297.3 L182.0 309.0 L177.3 320.6 L171.5 332.2 L164.8 343.8 L157.3 355.5 L149.5 367.1 L141.4 378.7 L133.5 390.3 L125.9 402.0 L118.9 413.6 L112.9 425.2 L107.9 436.9 L104.1 448.5 L101.8 460.1 L100.9 471.7 L101.5 483.4"
                  style={{ opacity: '0.82', animationDelay: '-0.6s' }}
                ></path>
                <path
                  className="flow"
                  d="M185.9 -12.3 L177.8 -0.7 L170.0 11.0 L162.6 22.6 L156.0 34.2 L150.4 45.8 L145.9 57.5 L142.6 69.1 L140.8 80.7 L140.3 92.3 L141.2 104.0 L143.5 115.6 L147.0 127.2 L151.5 138.9 L156.9 150.5 L163.0 162.1 L169.5 173.7 L176.1 185.4 L182.6 197.0 L188.7 208.6 L194.1 220.3 L198.7 231.9 L202.2 243.5 L204.5 255.1 L205.5 266.8 L205.1 278.4 L203.3 290.0 L200.2 301.7 L195.8 313.3 L190.3 324.9 L183.8 336.5 L176.6 348.2 L168.9 359.8 L161.1 371.4 L153.2 383.0 L145.7 394.7 L138.8 406.3 L132.8 417.9 L127.8 429.6 L124.0 441.2 L121.6 452.8 L120.7 464.4 L121.2 476.1 L123.3 487.7"
                  style={{ opacity: '0.86', animationDelay: '-1.2s' }}
                ></path>
                <path
                  className="flow"
                  d="M196.9 -5.8 L188.9 5.8 L181.3 17.5 L174.5 29.1 L168.6 40.7 L163.8 52.4 L160.2 64.0 L158.1 75.6 L157.3 87.2 L158.0 98.9 L160.0 110.5 L163.3 122.1 L167.7 133.8 L173.0 145.4 L179.0 157.0 L185.5 168.6 L192.1 180.3 L198.7 191.9 L204.9 203.5 L210.5 215.2 L215.3 226.8 L219.1 238.4 L221.7 250.0 L223.0 261.7 L222.9 273.3 L221.4 284.9 L218.6 296.5 L214.5 308.2 L209.3 319.8 L203.1 331.4 L196.2 343.1 L188.7 354.7 L181.0 366.3 L173.3 377.9 L166.0 389.6 L159.1 401.2 L153.1 412.8 L148.1 424.5 L144.3 436.1 L141.8 447.7 L140.8 459.3 L141.3 471.0 L143.3 482.6 L146.8 494.2"
                  style={{ opacity: '0.90', animationDelay: '-1.8s' }}
                ></path>
                <path
                  className="flow"
                  d="M207.5 1.8 L199.7 13.4 L192.7 25.0 L186.5 36.7 L181.5 48.3 L177.7 59.9 L175.3 71.5 L174.3 83.2 L174.8 94.8 L176.6 106.4 L179.7 118.1 L183.9 129.7 L189.1 141.3 L195.1 152.9 L201.5 164.6 L208.2 176.2 L214.9 187.8 L221.2 199.4 L227.0 211.1 L232.1 222.7 L236.1 234.3 L239.0 246.0 L240.7 257.6 L240.9 269.2 L239.8 280.8 L237.3 292.5 L233.6 304.1 L228.7 315.7 L222.8 327.4 L216.1 339.0 L208.9 350.6 L201.4 362.2 L193.8 373.9 L186.6 385.5 L179.8 397.1 L173.8 408.8 L168.8 420.4 L165.0 432.0 L162.5 443.6 L161.4 455.3 L161.8 466.9 L163.7 478.5 L167.1 490.1 L171.8 501.8"
                  style={{ opacity: '0.93', animationDelay: '-2.4s' }}
                ></path>
                <path
                  className="flow"
                  d="M218.0 9.0 L210.8 20.6 L204.4 32.3 L199.2 43.9 L195.2 55.5 L192.5 67.2 L191.3 78.8 L191.5 90.4 L193.1 102.0 L196.1 113.7 L200.2 125.3 L205.3 136.9 L211.2 148.5 L217.7 160.2 L224.5 171.8 L231.2 183.4 L237.8 195.1 L243.8 206.7 L249.1 218.3 L253.4 229.9 L256.6 241.6 L258.6 253.2 L259.2 264.8 L258.5 276.5 L256.4 288.1 L253.0 299.7 L248.4 311.3 L242.8 323.0 L236.4 334.6 L229.4 346.2 L222.1 357.9 L214.7 369.5 L207.5 381.1 L200.8 392.7 L194.9 404.4 L189.9 416.0 L186.0 427.6 L183.4 439.2 L182.3 450.9 L182.6 462.5 L184.4 474.1 L187.7 485.8 L192.3 497.4 L198.1 509.0"
                  style={{ opacity: '0.96', animationDelay: '-3.0s' }}
                ></path>
                <path
                  className="flow"
                  d="M228.8 14.6 L222.3 26.3 L216.8 37.9 L212.6 49.5 L209.8 61.2 L208.3 72.8 L208.4 84.4 L209.8 96.0 L212.6 107.7 L216.7 119.3 L221.7 130.9 L227.6 142.5 L234.1 154.2 L240.9 165.8 L247.8 177.4 L254.5 189.1 L260.8 200.7 L266.4 212.3 L271.0 223.9 L274.5 235.6 L276.9 247.2 L277.8 258.8 L277.5 270.5 L275.7 282.1 L272.7 293.7 L268.4 305.3 L263.2 317.0 L257.0 328.6 L250.3 340.2 L243.2 351.9 L236.0 363.5 L228.9 375.1 L222.3 386.7 L216.3 398.4 L211.3 410.0 L207.4 421.6 L204.7 433.2 L203.5 444.9 L203.7 456.5 L205.4 468.1 L208.5 479.8 L213.0 491.4 L218.7 503.0 L225.5 514.6"
                  style={{ opacity: '0.98', animationDelay: '-3.6s' }}
                ></path>
                <path
                  className="flow"
                  d="M240.1 17.7 L234.5 29.3 L230.1 40.9 L227.1 52.5 L225.5 64.2 L225.4 75.8 L226.7 87.4 L229.4 99.1 L233.3 110.7 L238.3 122.3 L244.2 133.9 L250.8 145.6 L257.7 157.2 L264.7 168.8 L271.6 180.4 L278.1 192.1 L283.9 203.7 L288.9 215.3 L292.8 227.0 L295.4 238.6 L296.8 250.2 L296.8 261.8 L295.4 273.5 L292.7 285.1 L288.8 296.7 L283.9 308.4 L278.0 320.0 L271.5 331.6 L264.6 343.2 L257.5 354.9 L250.6 366.5 L244.0 378.1 L238.1 389.8 L233.0 401.4 L229.0 413.0 L226.3 424.6 L224.9 436.3 L225.0 447.9 L226.6 459.5 L229.6 471.1 L233.9 482.8 L239.5 494.4 L246.1 506.0 L253.6 517.7"
                  style={{ opacity: '0.99', animationDelay: '-0.0s' }}
                ></path>
                <path
                  className="flow"
                  d="M252.3 17.5 L247.8 29.2 L244.6 40.8 L242.8 52.4 L242.6 64.0 L243.8 75.7 L246.4 87.3 L250.2 98.9 L255.2 110.5 L261.1 122.2 L267.7 133.8 L274.7 145.4 L281.9 157.1 L289.0 168.7 L295.7 180.3 L301.8 191.9 L307.1 203.6 L311.3 215.2 L314.4 226.8 L316.1 238.5 L316.5 250.1 L315.5 261.7 L313.1 273.3 L309.6 285.0 L304.9 296.6 L299.4 308.2 L293.1 319.8 L286.4 331.5 L279.4 343.1 L272.6 354.7 L266.0 366.4 L260.1 378.0 L255.0 389.6 L251.0 401.2 L248.1 412.9 L246.6 424.5 L246.5 436.1 L247.9 447.8 L250.7 459.4 L254.9 471.0 L260.4 482.6 L266.9 494.3 L274.2 505.9 L282.3 517.5"
                  style={{ opacity: '1.00', animationDelay: '-0.6s' }}
                ></path>
                <path
                  className="flow"
                  d="M265.6 14.3 L262.3 25.9 L260.4 37.5 L260.0 49.1 L261.1 60.8 L263.6 72.4 L267.5 84.0 L272.4 95.7 L278.3 107.3 L285.0 118.9 L292.1 130.5 L299.5 142.2 L306.8 153.8 L313.7 165.4 L320.1 177.1 L325.7 188.7 L330.2 200.3 L333.6 211.9 L335.7 223.6 L336.5 235.2 L335.8 246.8 L333.9 258.4 L330.6 270.1 L326.3 281.7 L321.0 293.3 L314.9 305.0 L308.4 316.6 L301.6 328.2 L294.8 339.8 L288.3 351.5 L282.4 363.1 L277.2 374.7 L273.1 386.4 L270.1 398.0 L268.5 409.6 L268.2 421.2 L269.4 432.9 L272.0 444.5 L276.0 456.1 L281.2 467.8 L287.6 479.4 L294.8 491.0 L302.7 502.6 L311.0 514.3"
                  style={{ opacity: '1.00', animationDelay: '-1.2s' }}
                ></path>
                <path
                  className="flow"
                  d="M280.2 8.5 L278.3 20.1 L277.8 31.7 L278.8 43.3 L281.3 55.0 L285.0 66.6 L290.0 78.2 L295.9 89.9 L302.6 101.5 L309.9 113.1 L317.4 124.7 L324.9 136.4 L332.1 148.0 L338.7 159.6 L344.6 171.2 L349.5 182.9 L353.2 194.5 L355.7 206.1 L356.8 217.8 L356.5 229.4 L354.9 241.0 L352.0 252.6 L347.9 264.3 L342.9 275.9 L337.1 287.5 L330.7 299.2 L324.0 310.8 L317.3 322.4 L310.8 334.0 L304.8 345.7 L299.6 357.3 L295.4 368.9 L292.2 380.5 L290.4 392.2 L289.9 403.8 L290.9 415.4 L293.3 427.1 L297.1 438.7 L302.1 450.3 L308.2 461.9 L315.3 473.6 L323.0 485.2 L331.2 496.8 L339.6 508.5"
                  style={{ opacity: '0.99', animationDelay: '-1.8s' }}
                ></path>
                <path
                  className="flow"
                  d="M296.5 1.1 L295.9 12.8 L296.8 24.4 L299.2 36.0 L303.0 47.7 L307.9 59.3 L313.9 70.9 L320.7 82.5 L328.0 94.2 L335.7 105.8 L343.3 117.4 L350.8 129.0 L357.7 140.7 L363.8 152.3 L369.0 163.9 L373.1 175.6 L375.9 187.2 L377.4 198.8 L377.5 210.4 L376.2 222.1 L373.6 233.7 L369.8 245.3 L365.0 257.0 L359.4 268.6 L353.1 280.2 L346.6 291.8 L339.9 303.5 L333.5 315.1 L327.4 326.7 L322.1 338.3 L317.7 350.0 L314.4 361.6 L312.4 373.2 L311.7 384.9 L312.4 396.5 L314.5 408.1 L318.1 419.7 L322.8 431.4 L328.8 443.0 L335.6 454.6 L343.2 466.3 L351.2 477.9 L359.4 489.5 L367.6 501.1"
                  style={{ opacity: '0.98', animationDelay: '-2.4s' }}
                ></path>
                <path
                  className="flow"
                  d="M314.4 -6.4 L315.2 5.2 L317.6 16.9 L321.3 28.5 L326.2 40.1 L332.2 51.8 L339.1 63.4 L346.5 75.0 L354.3 86.6 L362.2 98.3 L369.8 109.9 L377.0 121.5 L383.4 133.2 L388.9 144.8 L393.3 156.4 L396.5 168.0 L398.3 179.7 L398.7 191.3 L397.7 202.9 L395.4 214.6 L391.9 226.2 L387.3 237.8 L381.9 249.4 L375.8 261.1 L369.3 272.7 L362.7 284.3 L356.2 295.9 L350.1 307.6 L344.6 319.2 L340.1 330.8 L336.6 342.5 L334.3 354.1 L333.4 365.7 L333.8 377.3 L335.7 389.0 L338.9 400.6 L343.4 412.2 L349.1 423.9 L355.7 435.5 L363.1 447.1 L370.9 458.7 L379.0 470.4 L387.1 482.0 L394.8 493.6"
                  style={{ opacity: '0.96', animationDelay: '-3.0s' }}
                ></path>
                <path
                  className="flow"
                  d="M334.0 -12.8 L336.3 -1.1 L340.0 10.5 L344.9 22.1 L351.0 33.8 L357.9 45.4 L365.4 57.0 L373.3 68.6 L381.3 80.3 L389.2 91.9 L396.6 103.5 L403.3 115.1 L409.1 126.8 L413.8 138.4 L417.2 150.0 L419.4 161.7 L420.1 173.3 L419.4 184.9 L417.4 196.5 L414.1 208.2 L409.7 219.8 L404.4 231.4 L398.5 243.1 L392.0 254.7 L385.4 266.3 L378.9 277.9 L372.7 289.6 L367.2 301.2 L362.4 312.8 L358.7 324.4 L356.2 336.1 L355.0 347.7 L355.1 359.3 L356.7 371.0 L359.6 382.6 L363.9 394.2 L369.2 405.8 L375.6 417.5 L382.7 429.1 L390.4 440.7 L398.3 452.4 L406.2 464.0 L413.9 475.6 L421.0 487.2"
                  style={{ opacity: '0.93', animationDelay: '-3.6s' }}
                ></path>
                <path
                  className="flow"
                  d="M355.5 -16.9 L359.1 -5.2 L364.0 6.4 L370.1 18.0 L377.0 29.6 L384.6 41.3 L392.7 52.9 L400.8 64.5 L408.8 76.2 L416.5 87.8 L423.4 99.4 L429.5 111.0 L434.5 122.7 L438.2 134.3 L440.6 145.9 L441.6 157.6 L441.2 169.2 L439.4 180.8 L436.4 192.4 L432.2 204.1 L427.1 215.7 L421.2 227.3 L414.8 239.0 L408.2 250.6 L401.6 262.2 L395.3 273.8 L389.6 285.5 L384.7 297.1 L380.7 308.7 L377.9 320.3 L376.4 332.0 L376.2 343.6 L377.5 355.2 L380.1 366.9 L384.0 378.5 L389.1 390.1 L395.2 401.7 L402.0 413.4 L409.5 425.0 L417.2 436.6 L425.0 448.3 L432.5 459.9 L439.6 471.5 L445.9 483.1"
                  style={{ opacity: '0.90', animationDelay: '-0.0s' }}
                ></path>
                <path
                  className="flow"
                  d="M378.6 -18.0 L383.5 -6.3 L389.5 5.3 L396.5 16.9 L404.2 28.6 L412.3 40.2 L420.6 51.8 L428.8 63.4 L436.6 75.1 L443.8 86.7 L450.1 98.3 L455.3 109.9 L459.3 121.6 L462.0 133.2 L463.3 144.8 L463.1 156.5 L461.6 168.1 L458.7 179.7 L454.7 191.3 L449.7 203.0 L443.9 214.6 L437.5 226.2 L430.9 237.9 L424.2 249.5 L417.8 261.1 L411.9 272.7 L406.7 284.4 L402.5 296.0 L399.4 307.6 L397.6 319.3 L397.1 330.9 L398.0 342.5 L400.3 354.1 L403.9 365.8 L408.6 377.4 L414.4 389.0 L421.0 400.6 L428.2 412.3 L435.8 423.9 L443.4 435.5 L450.8 447.2 L457.8 458.8 L464.0 470.4 L469.3 482.0"
                  style={{ opacity: '0.86', animationDelay: '-0.6s' }}
                ></path>
                <path
                  className="flow"
                  d="M403.3 -15.8 L409.4 -4.2 L416.3 7.4 L424.0 19.0 L432.2 30.7 L440.6 42.3 L448.9 53.9 L456.9 65.5 L464.3 77.2 L470.8 88.8 L476.3 100.4 L480.5 112.1 L483.4 123.7 L484.9 135.3 L485.0 146.9 L483.7 158.6 L481.0 170.2 L477.1 181.8 L472.2 193.5 L466.5 205.1 L460.1 216.7 L453.4 228.3 L446.7 240.0 L440.1 251.6 L434.0 263.2 L428.6 274.8 L424.1 286.5 L420.7 298.1 L418.6 309.7 L417.8 321.4 L418.3 333.0 L420.2 344.6 L423.4 356.2 L427.9 367.9 L433.3 379.5 L439.7 391.1 L446.6 402.8 L454.0 414.4 L461.5 426.0 L468.8 437.6 L475.6 449.3 L481.8 460.9 L487.1 472.5 L491.3 484.2"
                  style={{ opacity: '0.82', animationDelay: '-1.2s' }}
                ></path>
                <path
                  className="flow"
                  d="M429.5 -10.9 L436.4 0.7 L444.2 12.3 L452.4 24.0 L460.8 35.6 L469.3 47.2 L477.4 58.9 L484.9 70.5 L491.6 82.1 L497.3 93.7 L501.7 105.4 L504.9 117.0 L506.6 128.6 L506.8 140.3 L505.7 151.9 L503.2 163.5 L499.4 175.1 L494.6 186.8 L488.9 198.4 L482.5 210.0 L475.8 221.6 L468.9 233.3 L462.2 244.9 L455.9 256.5 L450.3 268.2 L445.5 279.8 L441.8 291.4 L439.3 303.0 L438.1 314.7 L438.3 326.3 L439.8 337.9 L442.7 349.6 L446.8 361.2 L451.9 372.8 L458.0 384.4 L464.7 396.1 L471.8 407.7 L479.1 419.3 L486.3 431.0 L493.1 442.6 L499.3 454.2 L504.6 465.8 L508.8 477.5 L511.7 489.1"
                  style={{ opacity: '0.77', animationDelay: '-1.8s' }}
                ></path>
                <path
                  className="flow"
                  d="M456.8 -4.0 L464.5 7.6 L472.7 19.2 L481.2 30.9 L489.7 42.5 L497.9 54.1 L505.6 65.7 L512.4 77.4 L518.3 89.0 L522.9 100.6 L526.2 112.3 L528.1 123.9 L528.6 135.5 L527.6 147.1 L525.2 158.8 L521.5 170.4 L516.8 182.0 L511.1 193.6 L504.7 205.3 L497.9 216.9 L490.9 228.5 L484.0 240.2 L477.5 251.8 L471.6 263.4 L466.5 275.0 L462.5 286.7 L459.6 298.3 L458.1 309.9 L457.9 321.6 L459.1 333.2 L461.6 344.8 L465.3 356.4 L470.1 368.1 L475.9 379.7 L482.4 391.3 L489.3 403.0 L496.4 414.6 L503.5 426.2 L510.2 437.8 L516.4 449.5 L521.7 461.1 L525.9 472.7 L528.9 484.3 L530.6 496.0"
                  style={{ opacity: '0.72', animationDelay: '-2.4s' }}
                ></path>
                <path
                  className="flow"
                  d="M484.9 3.6 L493.2 15.2 L501.7 26.8 L510.2 38.5 L518.5 50.1 L526.2 61.7 L533.2 73.3 L539.2 85.0 L544.0 96.6 L547.5 108.2 L549.6 119.9 L550.1 131.5 L549.3 143.1 L547.0 154.7 L543.4 166.4 L538.7 178.0 L533.0 189.6 L526.6 201.3 L519.7 212.9 L512.6 224.5 L505.5 236.1 L498.8 247.8 L492.6 259.4 L487.2 271.0 L482.9 282.6 L479.7 294.3 L477.8 305.9 L477.2 317.5 L478.0 329.2 L480.1 340.8 L483.5 352.4 L488.0 364.0 L493.5 375.7 L499.7 387.3 L506.4 398.9 L513.4 410.6 L520.4 422.2 L527.0 433.8 L533.1 445.4 L538.5 457.1 L542.8 468.7 L545.9 480.3 L547.7 492.0 L548.1 503.6"
                  style={{ opacity: '0.67', animationDelay: '-3.0s' }}
                ></path>
                <path
                  className="flow"
                  d="M513.7 10.5 L522.2 22.2 L530.7 33.8 L539.0 45.4 L546.8 57.1 L553.9 68.7 L560.0 80.3 L564.9 91.9 L568.6 103.6 L570.8 115.2 L571.5 126.8 L570.7 138.5 L568.5 150.1 L565.0 161.7 L560.4 173.3 L554.7 185.0 L548.2 196.6 L541.2 208.2 L534.0 219.8 L526.7 231.5 L519.7 243.1 L513.3 254.7 L507.6 266.4 L502.9 278.0 L499.4 289.6 L497.1 301.2 L496.1 312.9 L496.6 324.5 L498.3 336.1 L501.4 347.8 L505.6 359.4 L510.8 371.0 L516.8 382.6 L523.3 394.3 L530.1 405.9 L536.9 417.5 L543.6 429.1 L549.7 440.8 L555.0 452.4 L559.4 464.0 L562.7 475.7 L564.6 487.3 L565.2 498.9 L564.3 510.5"
                  style={{ opacity: '0.61', animationDelay: '-3.6s' }}
                ></path>
                <path
                  className="flow"
                  d="M542.6 15.6 L551.1 27.3 L559.5 38.9 L567.3 50.5 L574.5 62.1 L580.6 73.8 L585.7 85.4 L589.4 97.0 L591.7 108.6 L592.5 120.3 L591.9 131.9 L589.8 143.5 L586.3 155.2 L581.7 166.8 L576.0 178.4 L569.5 190.0 L562.4 201.7 L555.0 213.3 L547.5 224.9 L540.3 236.6 L533.6 248.2 L527.7 259.8 L522.6 271.4 L518.7 283.1 L516.1 294.7 L514.8 306.3 L514.8 318.0 L516.2 329.6 L519.0 341.2 L522.9 352.8 L527.8 364.5 L533.5 376.1 L539.9 387.7 L546.5 399.3 L553.3 411.0 L559.9 422.6 L566.0 434.2 L571.4 445.9 L575.9 457.5 L579.3 469.1 L581.4 480.7 L582.1 492.4 L581.4 504.0 L579.3 515.6"
                  style={{ opacity: '0.54', animationDelay: '-0.0s' }}
                ></path>
                <path
                  className="flow"
                  d="M571.5 17.9 L579.8 29.5 L587.6 41.2 L594.8 52.8 L601.0 64.4 L606.2 76.1 L610.0 87.7 L612.4 99.3 L613.3 110.9 L612.7 122.6 L610.7 134.2 L607.3 145.8 L602.6 157.5 L596.9 169.1 L590.3 180.7 L583.1 192.3 L575.6 204.0 L568.0 215.6 L560.6 227.2 L553.6 238.9 L547.3 250.5 L542.0 262.1 L537.7 273.7 L534.7 285.4 L533.1 297.0 L532.8 308.6 L533.9 320.2 L536.3 331.9 L539.9 343.5 L544.5 355.1 L550.0 366.8 L556.2 378.4 L562.8 390.0 L569.4 401.6 L576.0 413.3 L582.1 424.9 L587.6 436.5 L592.2 448.2 L595.8 459.8 L598.1 471.4 L599.0 483.0 L598.6 494.7 L596.7 506.3 L593.4 517.9"
                  style={{ opacity: '0.48', animationDelay: '-0.6s' }}
                ></path>
                <path
                  className="flow"
                  d="M599.9 17.0 L607.7 28.6 L614.9 40.3 L621.2 51.9 L626.3 63.5 L630.2 75.2 L632.7 86.8 L633.7 98.4 L633.2 110.0 L631.2 121.7 L627.8 133.3 L623.2 144.9 L617.4 156.6 L610.8 168.2 L603.5 179.8 L595.9 191.4 L588.1 203.1 L580.5 214.7 L573.2 226.3 L566.7 237.9 L561.0 249.6 L556.5 261.2 L553.1 272.8 L551.1 284.5 L550.5 296.1 L551.2 307.7 L553.3 319.3 L556.6 331.0 L561.1 342.6 L566.4 354.2 L572.4 365.9 L578.8 377.5 L585.5 389.1 L592.0 400.7 L598.2 412.4 L603.8 424.0 L608.5 435.6 L612.2 447.3 L614.7 458.9 L615.9 470.5 L615.7 482.1 L614.1 493.8 L611.1 505.4 L606.8 517.0"
                  style={{ opacity: '0.42', animationDelay: '-1.2s' }}
                ></path>
                <path
                  className="flow"
                  d="M627.5 13.1 L634.7 24.7 L641.0 36.3 L646.2 48.0 L650.1 59.6 L652.6 71.2 L653.7 82.8 L653.2 94.5 L651.3 106.1 L648.0 117.7 L643.4 129.4 L637.6 141.0 L630.9 152.6 L623.6 164.2 L615.8 175.9 L607.8 187.5 L600.0 199.1 L592.5 210.8 L585.7 222.4 L579.8 234.0 L574.9 245.6 L571.2 257.3 L568.9 268.9 L567.9 280.5 L568.4 292.1 L570.2 303.8 L573.3 315.4 L577.4 327.0 L582.6 338.7 L588.4 350.3 L594.8 361.9 L601.4 373.5 L608.0 385.2 L614.2 396.8 L619.9 408.4 L624.8 420.1 L628.7 431.7 L631.5 443.3 L632.9 454.9 L633.0 466.6 L631.7 478.2 L629.0 489.8 L624.9 501.4 L619.7 513.1"
                  style={{ opacity: '0.35', animationDelay: '-1.8s' }}
                ></path>
              </g>
            </svg>
          </div>
          <div className="topnav">
            <DcLink className="brandmark" href="#top" aria-label="AI Academy, by Supervaise">
              <span className="bm-line" aria-hidden="true">
                <span className="bm-grow bm-pre">
                  <span>{'SUPERV'}</span>
                </span>{' '}
                <svg className="bm-mark" viewBox="0 0 360 310" aria-hidden="true">
                  <polygon points="5,308 140,6 275,308 225,308 142,116 55,308" fill="#76B900"></polygon>
                  <circle cx="250" cy="73" r="40" fill="#76B900"></circle>
                  <rect x="308" y="13" width="47" height="295" fill="#76B900"></rect>
                </svg>{' '}
                <span className="bm-grow bm-post">
                  <span>{'SE'}</span>
                </span>{' '}
                <span className="bm-grow bm-acad">
                  <span>{'ACADEMY'}</span>
                </span>
              </span>
            </DcLink>
            <nav className="navauth" aria-label="Account">
              <button type="button" className="btn ghost sm" onClick={v.openLogin}>
                {'Log In'}
              </button>{' '}
              <button type="button" className="btn primary sm" onClick={v.openSignup}>
                {'Sign Up'}
              </button>
            </nav>
          </div>
          <section className="hero" aria-label="AI Academy by Supervaise and NVIDIA">
            <h1 className={`phrase ${v.phraseClass}`}>
              {'Adapting Together in the Age of '}
              <span className="hl">{'AI'}</span>
            </h1>
            <p className={`blurb ${v.blurbClass}`}>
              {
                'Generative AI within everyone’s reach: guided courses, hands-on labs on real NVIDIA compute, and mentors who take you from first prompt to shipped project.'
              }
            </p>
            <p className={`lockup ${v.lockClass}`}>
              <span className="lk-pre">{'In Partnership with'}</span>{' '}
              <span className="lk-brand">
                <img className="lk-logo" src="/assets/nvidia-dark.png" alt="NVIDIA" />
              </span>
            </p>
          </section>
          <DcLink className="cue" href="#try">
            <span className="cue-mouse" aria-hidden="true"></span>
            {'Scroll to explore'}
          </DcLink>
        </div>
        <div className="lf-flow" id="lf-flow">
          <div className="guide" aria-hidden="true" style={css(`${v.guideStyle}`)}>
            <div className="guide-track"></div>
            <div className="guide-fill" style={css(`${v.fillStyle}`)}></div>
            <div className="guide-head" style={css(`${v.headStyle}`)}></div>
          </div>
          <section className="s" id="try">
            <div className="wrap">
              <span className={`node ${v.lit?.n1}`} data-node="n1"></span>
              <div className={`s-head rv ${v.on?.tryHead}`} data-rv="tryHead">
                <p className="eyebrow">{'01 · Try it now'}</p>
                <h2>{'See text the way AI does'}</h2>
                <p className="lead">{'Pick a prompt. See the tokens GPT-2 reads and what it predicts next.'}</p>
              </div>
              <div className="lab" data-lab="1">
                <div className={`lab-panel rv ${v.on?.tryP1} ${v.sample?.animCls}`} data-rv="tryP1" style={{ '--d': '.12s' }}>
                  <h3>{'AI model'}</h3>
                  <p className="mname">
                    <b>{'GPT-2'}</b>
                    {' · a language model with 124 million parameters'}
                  </p>
                  <span className="fl">{'Input: pick a prompt'}</span>
                  <div className="chips">
                    {(v.samples || []).map((c, $index) => (
                      <button type="button" key={$index} className={c?.cls} onClick={c?.pick}>
                        {c?.prompt}
                      </button>
                    ))}
                  </div>
                  <div className="tx tx-ro" aria-label={v.sample?.full}>
                    {v.sample?.prompt}
                    <span className={v.sample?.typing ? 'caret' : 'caret idle'}></span>
                  </div>
                  <span className="fl">{'The model reads it as tokens, each with an ID number'}</span>
                  <div className="toks" key={'t' + v.sample?.run}>
                    {(v.sample?.toks || []).map((tk, $index) => (
                      <React.Fragment key={$index}>
                        <span className="tok" style={css(`${tk?.style}`)}>
                          {tk?.text}
                          <small>{tk?.idStr}</small>
                        </span>
                      </React.Fragment>
                    ))}
                  </div>
                  <div className="stats">
                    <div className="stat">
                      <b>{v.sample?.chars}</b>
                      <span>{'characters'}</span>
                    </div>
                    <div className="stat">
                      <b>{v.sample?.count}</b>
                      <span>{'tokens'}</span>
                    </div>
                    <div className="stat">
                      <b>{v.sample?.ratio}</b>
                      <span>{'chars per token'}</span>
                    </div>
                  </div>
                </div>
                <div className={`lab-panel rv ${v.on?.tryP2} ${v.sample?.animCls}`} data-rv="tryP2" style={{ '--d': '.24s' }}>
                  <h3>{'Output'}</h3>
                  <div className="mout" key={'o' + v.sample?.run} aria-live="polite">
                    <p className="mq">
                      {'GPT-2’s five most likely next tokens after '}
                      <b>{'“' + (v.sample?.prompt || '') + '”'}</b>
                    </p>
                    {(v.sample?.out || []).map((pr, $index) => (
                      <div className={pr?.cls} key={$index}>
                        <span className="mw">{pr?.word}</span>
                        <span className="mt"><i style={css(`${pr?.style}`)}></i></span>
                        <span className="mp">{pr?.pct}</span>
                      </div>
                    ))}
                  </div>
                  <p className="cont" key={'c' + v.sample?.run}>
                    <span className="cp">{v.sample?.full}</span>
                    <span className="cn" style={css(`${v.sample?.nextStyle}`)}>{v.sample?.next}</span>
                  </p>
                  <p className="mnote">{v.sample?.note}</p>
                  <p className="hint">{'Real output of GPT-2 small, measured in our Week 1 lab. “·” marks a space; the model scores every token in its 50,257-token vocabulary.'}</p>
                </div>
              </div>
            </div>
          </section>
          <section className="s">
            <div className="wrap">
              <span className={`node ${v.lit?.n2}`} data-node="n2"></span>
              <div className={`s-head rv ${v.on?.progHead}`} data-rv="progHead">
                <p className="eyebrow">{'02 · The programme'}</p>
                <h2>{'20 weeks. 4 terms. One path.'}</h2>
                <p className="lead">{'Lecture, lab and quiz every week.'}</p>
              </div>
              <div className={`terms-wrap ${v.on?.terms}`} data-rv="terms">
                <div className="rail" aria-hidden="true">
                  <i></i>
                </div>
                <div className="terms">
                  <div className={`term rv ${v.on?.term1}`} data-rv="term1" style={{ '--d': '.1s' }}>
                    <div className="dot" style={{ '--d': '.4s' }}>
                      {'01'}
                    </div>
                    <div className="card">
                      <span className="wk">{'Weeks 1 to 5'}</span>
                      <h3>{'Foundations'}</h3>
                      <ul>
                        <li>{'Tokens and embeddings'}</li>
                        <li>{'Training a network'}</li>
                      </ul>
                    </div>
                  </div>
                  <div className={`term rv ${v.on?.term2}`} data-rv="term2" style={{ '--d': '.25s' }}>
                    <div className="dot" style={{ '--d': '1s' }}>
                      {'02'}
                    </div>
                    <div className="card">
                      <span className="wk">{'Weeks 6 to 10'}</span>
                      <h3>{'Transformers'}</h3>
                      <ul>
                        <li>{'Attention'}</li>
                        <li>{'Prompt engineering'}</li>
                      </ul>
                    </div>
                  </div>
                  <div className={`term rv ${v.on?.term3}`} data-rv="term3" style={{ '--d': '.4s' }}>
                    <div className="dot" style={{ '--d': '1.6s' }}>
                      {'03'}
                    </div>
                    <div className="card">
                      <span className="wk">{'Weeks 11 to 15'}</span>
                      <h3>{'Fine-tuning'}</h3>
                      <ul>
                        <li>{'Fine-tuning'}</li>
                        <li>{'Responsible AI'}</li>
                      </ul>
                    </div>
                  </div>
                  <div className={`term rv ${v.on?.term4}`} data-rv="term4" style={{ '--d': '.55s' }}>
                    <div className="dot" style={{ '--d': '2.2s' }}>
                      {'04'}
                    </div>
                    <div className="card">
                      <span className="wk">{'Weeks 16 to 20'}</span>
                      <h3>{'Agents and capstone'}</h3>
                      <ul>
                        <li>{'RAG and agents'}</li>
                        <li>{'Capstone defence'}</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </section>
          <section className="s">
            <div className="wrap">
              <span className={`node ${v.lit?.n3}`} data-node="n3"></span>
              <div className={`s-head rv ${v.on?.credHead}`} data-rv="credHead">
                <p className="eyebrow">{'03 · Credentials'}</p>
                <h2>{'Two NVIDIA certifications'}</h2>
                <p className="lead">{'Start at Associate. Advance to Professional.'}</p>
              </div>
              <div className="creds">
                <div className={`cred rv ${v.on?.cred1}`} data-rv="cred1" style={{ '--d': '.1s' }}>
                  <img className="cred-logo" src="/assets/nvidia-dark.png" alt="NVIDIA" />{' '}
                  <span className="kind">{'Level 1 · Associate'}</span>
                  <h3>{'NVIDIA-Certified Associate: Generative AI LLMs'}</h3>
                  <p>{'The programme target. Mock examination in Week 19.'}</p>
                  <span className="code">{'NCA-GENL'}</span>
                </div>
                <div className={`bridge rv ${v.on?.bridge}`} data-rv="bridge" style={{ '--d': '.3s' }} aria-hidden="true">
                  <i></i>
                </div>
                <div className={`cred pro rv ${v.on?.cred2}`} data-rv="cred2" style={{ '--d': '.45s' }}>
                  <img className="cred-logo" src="/assets/nvidia-dark.png" alt="NVIDIA" />{' '}
                  <span className="kind">{'Level 2 · Professional'}</span>
                  <h3>{'NVIDIA-Certified Professional: Generative AI LLMs'}</h3>
                  <p>{'The next step after the Associate.'}</p>
                  <span className="code">{'NCP-GENL'}</span>
                </div>
              </div>
              <p className={`fine rv ${v.on?.fine}`} data-rv="fine" style={{ '--d': '.5s' }}>
                {'Both certifications are issued by NVIDIA. Supervaise prepares learners for the examinations.'}
              </p>
            </div>
          </section>
          <section className={`welcome ${v.on?.welcome}`} id="welcome" data-rv="welcome">
            <span className={`node ${v.lit?.n4}`} data-node="n4" style={{ top: '56px' }}></span>
            <p className="w-pre">{'Your journey starts here'}</p>
            <h2 className="w-line">
              {(v.words || []).map((w, $index) => (
                <React.Fragment key={$index}>
                  <span className={w?.cls} style={css(`${w?.style}`)}>
                    {w?.text}
                  </span>
                  {w?.after}
                </React.Fragment>
              ))}
            </h2>
            <div className="w-cta">
              <button type="button" className="btn primary glow" onClick={v.openSignup}>
                {'Sign Up'}
              </button>
              <button type="button" className="btn ghost" onClick={v.openLogin}>
                {'Log In'}
              </button>
            </div>
          </section>
        </div>
        <div className="foot">
          <footer className="studio-footer">
            <span className="sf-copy">{'© 2026 Supervaise Inc.'}</span>{' '}
            <span className="sf-social">
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
              <DcLink href="mailto:anthony.jose@supervaise.io" aria-label="Email Supervaise">
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
              <span>{'Powered by'}</span>
              <img className="sf-nv" src="/assets/nvidia-dark.png" alt="NVIDIA" />
              <strong>{'DGX Spark'}</strong>
            </span>
          </footer>
        </div>
        {v.isLogin ? (
          <>
            <div className="overlay">
              <div className="backdrop" onClick={v.closePanel}></div>
              <div className="panel" role="dialog" aria-modal="true" aria-label="Log in">
                <button type="button" className="close" onClick={v.closePanel} aria-label="Close">
                  {'×'}
                </button>
                <h2>{'Welcome back'}</h2>
                <p className="sub">{'Log in to continue your AI Academy journey.'}</p>
                <label className="pf">
                  <span>{'Email'}</span>
                  <input
                    type="email"
                    name="email"
                    placeholder="you@example.com"
                    autoComplete="email"
                    value={v.authEmail}
                    onChange={v.onField}
                    onKeyDown={v.onAuthKey}
                  />
                </label>{' '}
                <label className="pf">
                  <span>{'Password'}</span>
                  <input
                    type="password"
                    name="password"
                    placeholder="Password"
                    autoComplete="current-password"
                    value={v.authPassword}
                    onChange={v.onField}
                    onKeyDown={v.onAuthKey}
                  />
                </label>{' '}
                {v.authError ? (
                  <>
                    <p className="note" role="alert" style={{ color: '#F87171', fontSize: '12.5px', margin: '0 0 10px' }}>
                      {v.authError}
                    </p>
                  </>
                ) : null}
                <button type="button" className="btn primary" disabled={v.authBusy} onClick={v.submitLogin}>
                  {'Log in'}
                </button>
                <p className="quiet">
                  {'Faculty members are directed to the '}
                  <DcLink href="Educator.dc.html">{'Faculty Console'}</DcLink>
                  {'.'}
                </p>
                <button type="button" className="linkbtn cancel" onClick={v.closePanel}>
                  {'Cancel'}
                </button>
              </div>
            </div>
          </>
        ) : null}{' '}
        {v.isSignup ? (
          <>
            <div className="overlay">
              <div className="backdrop" onClick={v.closePanel}></div>
              <div className="panel" role="dialog" aria-modal="true" aria-label="Create your account">
                <button type="button" className="close" onClick={v.closePanel} aria-label="Close">
                  {'×'}
                </button>
                <h2>{'Create your account'}</h2>
                <p className="sub">{'Join the Supervaise × NVIDIA AI Academy.'}</p>
                <label className="pf">
                  <span>{'Full name'}</span>
                  <input
                    type="text"
                    name="name"
                    placeholder="Maya Santos"
                    autoComplete="name"
                    value={v.authName}
                    onChange={v.onField}
                    onKeyDown={v.onAuthKey}
                  />
                </label>{' '}
                <label className="pf">
                  <span>{'Email'}</span>
                  <input
                    type="email"
                    name="email"
                    placeholder="you@example.com"
                    autoComplete="email"
                    value={v.authEmail}
                    onChange={v.onField}
                    onKeyDown={v.onAuthKey}
                  />
                </label>{' '}
                <label className="pf">
                  <span>{'Password'}</span>
                  <input
                    type="password"
                    name="password"
                    placeholder="Create a password (8+ characters)"
                    autoComplete="new-password"
                    value={v.authPassword}
                    onChange={v.onField}
                    onKeyDown={v.onAuthKey}
                  />
                </label>{' '}
                {v.authError ? (
                  <>
                    <p className="note" role="alert" style={{ color: '#F87171', fontSize: '12.5px', margin: '0 0 10px' }}>
                      {v.authError}
                    </p>
                  </>
                ) : null}
                <button type="button" className="btn primary" disabled={v.authBusy} onClick={v.submitSignup}>
                  {'Create account'}
                </button>
                <p className="quiet">
                  {'Already have an account? '}
                  <button type="button" className="linkbtn" onClick={v.switchToLogin}>
                    {'Log in'}
                  </button>
                </p>
                <button type="button" className="linkbtn cancel" onClick={v.closePanel}>
                  {'Cancel'}
                </button>
              </div>
            </div>
          </>
        ) : null}
      </div>
    </>
  );
}

Component.template = template;
Component.css = pageCss;
Component.pageKey = 'landing';
Component.persist = [];

export default Component;
