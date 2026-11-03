// Generated from design-source/Main.dc.html by tools/convert.mjs.
// Re-running the converter regenerates (and overwrites) this file, so edit design-source/ + tools/patches.mjs instead,
// or stop using the converter and edit this file directly.
/* eslint-disable */
import React from 'react';
import DCLogic from '../dc/DCLogic.jsx';
import DcLink from '../dc/DcLink.jsx';
import { css } from '../dc/css.js';
import pageCss from './Main.css?inline';
import { CURRENT_WEEK, TOTAL_WEEKS, getWeek, activityStatus } from '../data/courseWeeks.js';

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

const COURSE_WEEKS = [
  { week: 1, term: 'Prelim', title: 'Course orientation & Intro to Generative AI', exam: false },
  { week: 2, term: 'Prelim', title: 'Machine Learning & Neural Networks', exam: false },
  { week: 3, term: 'Prelim', title: 'Deep Learning & Backpropagation', exam: false },
  { week: 4, term: 'Prelim', title: 'Convolutional Neural Networks', exam: false },
  { week: 5, term: 'Prelim', title: 'PRELIM EXAM', exam: true },
  { week: 6, term: 'Midterm', title: 'NLP: Tokens & Word Embeddings', exam: false },
  { week: 7, term: 'Midterm', title: 'Language Models & Attention', exam: false },
  { week: 8, term: 'Midterm', title: 'The Transformer & Autoregressive Training', exam: false },
  { week: 9, term: 'Midterm', title: 'LLM Families, Scaling Laws & Prompt Engineering', exam: false },
  { week: 10, term: 'Midterm', title: 'MIDTERM EXAM', exam: true },
  { week: 11, term: 'Pre-final', title: 'Multimodal AI', exam: false },
  { week: 12, term: 'Pre-final', title: 'Image Generation & GANs', exam: false },
  { week: 13, term: 'Pre-final', title: 'Diffusion Models', exam: false },
  { week: 14, term: 'Pre-final', title: 'Training LLMs: Fine-tuning, RLHF, PEFT', exam: false },
  { week: 15, term: 'Pre-final', title: 'PRE-FINAL EXAM + Capstone proposal', exam: true },
  { week: 16, term: 'Final', title: 'LLM Components & RAG', exam: false },
  { week: 17, term: 'Final', title: 'AI Agents, Guardrails & Trustworthy AI', exam: false },
  { week: 18, term: 'Final', title: 'Scaling & Deploying AI', exam: false },
  { week: 19, term: 'Final', title: 'Capstone Build Week + Mock NCA-GENL Exam', exam: false },
  { week: 20, term: 'Final', title: 'FINAL EXAM & Demo Day', exam: true },
];
const STAGES = [
  {
    name: 'Neural network foundations',
    weeks: 'Weeks 1 to 5',
    from: 1,
    to: 5,
    miles: ['Machine learning and neural networks', 'Backpropagation and convolutional networks', 'Prelim examination, Week 5: Passed'],
  },
  {
    name: 'Language models and transformers',
    weeks: 'Weeks 6 to 10',
    from: 6,
    to: 10,
    miles: ['Tokens, embeddings and attention', 'Transformer architecture and prompt engineering', 'Midterm examination, Week 10'],
  },
  {
    name: 'Multimodal models and fine-tuning',
    weeks: 'Weeks 11 to 15',
    from: 11,
    to: 15,
    miles: ['Multimodal AI, GANs and diffusion models', 'Fine-tuning, RLHF and PEFT', 'Pre-final examination and capstone proposal'],
  },
  {
    name: 'Retrieval, agents and capstone',
    weeks: 'Weeks 16 to 20',
    from: 16,
    to: 20,
    miles: [
      'Retrieval-augmented generation and AI agents',
      'Mock NCA-GENL examination, Week 19',
      'Final examination and capstone demonstration, Week 20',
    ],
  },
  {
    name: 'NVIDIA-Certified Associate: Generative AI LLMs',
    weeks: 'Level 1 · NCA-GENL',
    from: 21,
    to: 21,
    goal: true,
    miles: ['Mock examination, Week 19', 'Registration through NVIDIA', '1 hour, 50 to 60 multiple-choice questions'],
  },
  {
    name: 'NVIDIA-Certified Professional: Generative AI LLMs',
    weeks: 'Level 2 · NCP-GENL',
    from: 22,
    to: 22,
    goal: true,
    miles: ['Advanced level', 'Recommended after the Associate', 'Issued by NVIDIA'],
  },
];
function roadmapVals() {
  const n = STAGES.length;
  return STAGES.map((st, i) => {
    const done = CURRENT_WEEK > st.to;
    const now = CURRENT_WEEK >= st.from && CURRENT_WEEK <= st.to;
    const frac = done ? 100 : now ? Math.round(((CURRENT_WEEK - st.from + 0.5) / (st.to - st.from + 1)) * 100) : 0;
    const status = done ? 'Completed' : now ? 'In progress' : 'Not started';
    return {
      num: String(i + 1),
      name: st.name,
      weeks: st.weeks,
      status: status,
      stCls: 'rm-stage' + (st.goal ? ' rm-dest' : ''),
      nodeCls: 'rm-node ' + (st.goal ? 'rm-goal' : done ? 'rm-done' : now ? 'rm-now' : 'rm-todo'),
      statusColor: done ? 'var(--accent-soft-text)' : now ? 'var(--amber-text)' : 'var(--text3)',
      fill: frac + '%',
      segDisp: i === n - 1 ? 'none' : 'block',
      hereDisp: now ? 'inline-flex' : 'none',
      popCls: 'rm-pop' + (i === 0 ? ' rm-pop-l' : i >= n - 2 ? ' rm-pop-r' : ''),
      popHead: now ? 'In progress: ' + frac + '% of this stage' : st.goal ? 'Destination' : 'Milestones',
      label: st.name + ', ' + st.weeks + ', ' + status + '. Show milestones.',
      miles: st.miles.map((t) => ({ t: t })),
    };
  });
}

class Component extends DCLogic {
  constructor(props) {
    super(props);
    const saved = readTheme();
    this.state = { theme: saved, mapOpen: false };
    this.toggleMap = () => this.setState({ mapOpen: !this.state.mapOpen });
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
    const open = this.state.mapOpen;
    const line = (w) => {
      const cur = w.week === CURRENT_WEEK;
      return {
        week: w.week,
        title: w.title,
        color: cur ? 'var(--amber-text)' : w.exam ? 'var(--red-text)' : w.week < CURRENT_WEEK ? 'var(--text2)' : 'var(--text)',
        weight: cur || w.exam ? '700' : '500',
        dot: cur ? 'var(--amber)' : w.exam ? 'var(--red-text)' : w.week < CURRENT_WEEK ? 'var(--accent)' : 'var(--track)',
        bg: cur ? 'var(--amber-soft-bg)' : 'transparent',
      };
    };
    const term = (t) => COURSE_WEEKS.filter((w) => w.term === t).map(line);
    const segs = COURSE_WEEKS.map((w) => ({
      tip: 'Week ' + w.week + ': ' + w.title,
      bg: w.week < CURRENT_WEEK ? 'var(--accent)' : w.week === CURRENT_WEEK ? 'var(--amber)' : 'var(--track)',
      cls: w.week === CURRENT_WEEK ? 'pulseamber' : '',
      mark: w.exam ? '\u25C6' : '',
    }));
    const wk = getWeek(this.props.week);
    const BADGE = {
      Completed: { background: 'var(--accent-soft-bg)', color: 'var(--accent-soft-text)', border: '1px solid var(--accent-soft-border)' },
      'In progress': { background: 'var(--amber-soft-bg)', color: 'var(--amber-text)', border: '1px solid var(--amber-soft-border)' },
      'Not started': { background: 'var(--neutral-chip-bg)', color: 'var(--neutral-chip-text)', border: '1px solid var(--neutral-chip-border)' },
    };
    const isNow = wk.week === CURRENT_WEEK;
    const eyebrow = isNow ? 'Next action' : wk.week < CURRENT_WEEK ? 'Review' : 'Coming up';
    return {
      wk,
      weekOf: 'Week ' + wk.week + ' of ' + TOTAL_WEEKS,
      eyebrow: eyebrow + ' · Week ' + wk.week + ' of ' + TOTAL_WEEKS,
      scheduleHead: (isNow ? 'Today' : 'Week ' + wk.week) + ' · ' + wk.sessionsDate,
      sessionsHead: isNow ? 'Sessions today' : 'Sessions this week',
      studentLine: 'Student · Week ' + CURRENT_WEEK,
      activities: wk.activities.map((a) => {
        const status = activityStatus(wk.week, wk.activities.indexOf(a));
        return { ...a, status, badge: BADGE[status] };
      }),
      ...themeVals(dark),
      toggleTheme: this.toggleTheme,
      toggleMap: this.toggleMap,
      mapExpanded: open ? 'true' : 'false',
      mapLabel: open ? 'Hide all 20 weeks' : 'Show all 20 weeks',
      mapRot: open ? 'rotate(180deg)' : 'rotate(0deg)',
      mapR: { rows: open ? '1fr' : '0fr', op: open ? '1' : '0', vis: open ? 'visible' : 'hidden', vt: open ? '0s' : '0s linear .5s' },
      roadmap: roadmapVals(),
      segs,
      prelim: term('Prelim'),
      midterm: term('Midterm'),
      prefinal: term('Pre-final'),
      final: term('Final'),
      pColor: 'var(--text2)',
      mColor: 'var(--amber-text)',
      fColor: 'var(--text2)',
      lColor: 'var(--text2)',
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
          <DcLink className="navlink active" href="Main.dc.html">
            <span>{'●'}</span>
            <span>{'Overview'}</span>
          </DcLink>
          <DcLink className="navlink" href="CourseMap.dc.html">
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
                <div style={{ fontSize: '11.5px', color: '#9BA3AF', lineHeight: '1.4' }}>{v.studentLine}</div>
                <div style={{ marginTop: '6px' }}>
                  <span
                    style={{
                      display: 'inline-block',
                      padding: '2px 8px',
                      borderRadius: '999px',
                      background: 'rgba(255,255,255,0.06)',
                      color: '#9BA3AF',
                      fontSize: '11px',
                      fontWeight: '600',
                      lineHeight: '1.5',
                    }}
                  >
                    {'Module 2 of 4 in progress'}
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
          style={{ flex: '999 1 560px', minWidth: '0', padding: '32px 48px 64px', maxWidth: '75rem', marginInline: 'auto' }}
        >
          <div className="utilbar">
            <div className="ctx">
              <span className="ctx-i">{'A.Y. 2026/2027'}</span> <span className="ctx-s" aria-hidden="true"></span>{' '}
              <span className="ctx-i">{'Second Semester'}</span> <span className="ctx-s" aria-hidden="true"></span>{' '}
              <span className="ctx-i ctx-now">{v.weekOf}</span> <span className="ctx-s" aria-hidden="true"></span>{' '}
              <span className="ctx-i">{v.wk.termLabel}</span>
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
                  <div className="hkpop-h">{v.scheduleHead}</div>
                  <ul className="hkl">
                    {(v.wk.sessions || []).map((ss, $index) => (
                      <li key={$index}>
                        <span className="hkt">{ss.time}</span>
                        <span className="hkd">
                          <b>{ss.title}</b>
                          <span className="hkm">{ss.note}</span>
                        </span>
                      </li>
                    ))}
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
          <header className="hx hx-coral">
            <div className="hx-body">
              <div className="eyebrow" style={{ marginBottom: 'var(--s2)' }}>
                {v.eyebrow}
              </div>
              <h1 className="cm-title" style={{ fontSize: 'clamp(28px,3.4vw,42px)', lineHeight: '1.15', margin: '0 0 var(--s3)' }}>
                {v.wk.hero.lead}
                <mark className="mk">{v.wk.hero.mark}</mark>
                {v.wk.hero.rest}
              </h1>
              <div className="hx-meta">
                {v.wk.hero.chips.map((c, $index) => (
                  <span className="chip-i" key={$index}>
                    {c}
                  </span>
                ))}
              </div>
              <DcLink href={v.wk.lessonHref} className="hx-cta">
                {v.wk.hero.cta}{' '}
                <span aria-hidden="true">{'→'}</span>
              </DcLink>
            </div>
            <div className="hx-ico">
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <circle cx="12" cy="12" r="8.6"></circle>
                <circle cx="12" cy="12" r="4.6"></circle>
                <circle cx="12" cy="12" r="1.2" fill="currentColor"></circle>
                <path d="M12 1.8v3M12 19.2v3M1.8 12h3M19.2 12h3"></path>
              </svg>
            </div>
          </header>
          <div className="sect">
            <div className="sect-h">
              <h2 className="sect-t">{'This week'}</h2>
              <span className="sect-n">{v.weekOf + ' · ' + v.wk.short}</span>
            </div>
            <div className="ov-two" style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '16px' }}>
              <div className="glass" style={{ padding: '22px' }}>
                <div style={{ fontSize: '18.5px', fontWeight: '600', marginBottom: '12px', color: 'var(--text)' }}>
                  {'Week ' + v.wk.week + ' learning activities'}
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  {(v.activities || []).map((a, $index) => (
                    <div
                      key={$index}
                      className="lift"
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        gap: '12px',
                        padding: '12px 14px',
                        borderRadius: '10px',
                        border: '1px solid var(--divider)',
                      }}
                    >
                      <div>
                        <div style={{ fontSize: '16px', fontWeight: '600', color: 'var(--text)' }}>{a.title}</div>
                        <div style={{ fontSize: '14px', color: 'var(--text3)' }}>{a.note}</div>
                      </div>
                      <span className="badge" style={a.badge}>
                        {a.status}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="glass" style={{ padding: '20px' }}>
                <div style={{ fontSize: '18.5px', fontWeight: '600', marginBottom: '2px', color: 'var(--text)' }}>{v.sessionsHead}</div>
                <div className="mono" style={{ fontSize: '13px', color: 'var(--text3)', fontWeight: '600', marginBottom: '12px' }}>
                  {v.wk.sessionsDate}
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  {(v.wk.sessions || []).map((ss, $index) => (
                    <div key={$index} style={{ display: 'grid', gridTemplateColumns: '50px 1fr', gap: '10px', alignItems: 'start' }}>
                      <span className="mono" style={{ fontSize: '13px', fontWeight: '500', color: 'var(--text3)' }}>
                        {ss.time}
                      </span>
                      <span>
                        <span style={{ display: 'block', fontSize: '15px', fontWeight: '500', color: 'var(--text)', lineHeight: '1.4' }}>
                          {ss.title}
                        </span>
                        <span style={{ display: 'block', fontSize: '13px', color: 'var(--text3)' }}>{ss.note}</span>
                      </span>
                    </div>
                  ))}
                </div>
                <DcLink
                  href="CourseMap.dc.html"
                  style={{
                    display: 'block',
                    marginTop: '14px',
                    paddingTop: '12px',
                    borderTop: '1px solid var(--divider)',
                    fontSize: '14px',
                    fontWeight: '500',
                  }}
                >
                  {'View the full course schedule →'}
                </DcLink>
              </div>
            </div>
          </div>
          <div className="sect">
            <div className="sect-h">
              <h2 className="sect-t">{'Pathway to certification'}</h2>
              <span className="sect-n">{'Stage 2 of 6 · two NVIDIA certification levels'}</span>
            </div>
            <div className="glass rm">
              <div className="rm-track" aria-label="Route to NVIDIA certification">
                {(v.roadmap || []).map((st, $index) => (
                  <React.Fragment key={$index}>
                    <div className={st?.stCls}>
                      <span className="rm-here" style={css(`display:${st?.hereDisp}`)}>
                        {'You are here'}
                      </span>{' '}
                      <span className="rm-seg" style={css(`display:${st?.segDisp}`)} aria-hidden="true">
                        <span className="rm-fill" style={css(`width:${st?.fill}`)}></span>
                      </span>{' '}
                      <button type="button" className={st?.nodeCls} aria-label={st?.label}>
                        <span className="rm-num">{st?.num}</span>
                        <svg className="rm-ic rm-ic-ok" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                          <path
                            d="M5 12.5l4.5 4.5L19 7.5"
                            stroke="currentColor"
                            strokeWidth="2.6"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          ></path>
                        </svg>
                        <svg className="rm-ic rm-ic-goal" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                          <circle cx="12" cy="9.2" r="6.2" stroke="currentColor" strokeWidth="1.9"></circle>
                          <path
                            d="M12 6.3l.95 1.95 2.15.3-1.55 1.5.37 2.13L12 11.2l-1.92 1 .37-2.13-1.55-1.5 2.15-.3z"
                            fill="currentColor"
                          ></path>
                          <path
                            d="M8.4 14.2 7 21l5-2.6 5 2.6-1.4-6.8"
                            stroke="currentColor"
                            strokeWidth="1.9"
                            strokeLinejoin="round"
                          ></path>
                        </svg>
                      </button>
                      <div className="rm-txt">
                        <div className="rm-name">{st?.name}</div>
                        <div className="rm-wk">{st?.weeks}</div>
                        <div className="rm-st" style={css(`color:${st?.statusColor}`)}>
                          {st?.status}
                        </div>
                      </div>
                      <div className={st?.popCls} role="tooltip">
                        <div className="rm-pop-h">{st?.popHead}</div>
                        <ul className="rm-mil">
                          {(st?.miles || []).map((m, $index) => (
                            <React.Fragment key={$index}>
                              <li>{m?.t}</li>
                            </React.Fragment>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </React.Fragment>
                ))}
              </div>
              <div className="rm-stats">
                <div>
                  <div className="rm-k">{'32%'}</div>
                  <div className="rm-l">{'of the programme completed'}</div>
                </div>
                <div>
                  <div className="rm-k">{'7 / 20'}</div>
                  <div className="rm-l">{'weeks, Midterm term'}</div>
                </div>
                <div>
                  <div className="rm-k">{'6 / 7'}</div>
                  <div className="rm-l">{'knowledge checks completed'}</div>
                </div>
                <div>
                  <div className="rm-k">{'Week 19'}</div>
                  <div className="rm-l">{'mock NCA‑GENL examination'}</div>
                </div>
              </div>
              <p className="rm-note">
                {
                  'The NVIDIA-Certified Associate (NCA‑GENL) is the programme target, and the NVIDIA-Certified Professional (NCP‑GENL) is the next level. Both are issued by NVIDIA; the programme prepares learners for the examinations but cannot issue or guarantee them. '
                }
                <DcLink href="CourseMap.dc.html" style={{ fontWeight: '500' }}>
                  {'View the full course outline →'}
                </DcLink>
              </p>
            </div>
          </div>
          <div className="sect">
            <div className="sect-h">
              <h2 className="sect-t">{'Requirements and assessment'}</h2>
              <span className="sect-n">{'3 requirements due · 40% of the final grade'}</span>
            </div>
            <div className="glass" style={{ padding: '22px' }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '9px' }}>
                <div
                  className="lift"
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    gap: '12px',
                    padding: '13px 14px',
                    borderRadius: '10px',
                    border: '1px solid var(--divider)',
                  }}
                >
                  <div style={{ minWidth: '0' }}>
                    <div style={{ fontSize: '16px', fontWeight: '500', color: 'var(--text)' }}>{'Laboratory 3 report'}</div>
                    <div style={{ fontSize: '14px', color: 'var(--text2)' }}>{'10% of final grade · One submission permitted'}</div>
                  </div>
                  <div style={{ textAlign: 'right', flex: 'none' }}>
                    <div className="mono" style={{ fontSize: '14px', fontWeight: '500', color: 'var(--amber-text)' }}>
                      {'17 October 2026'}
                    </div>
                    <div style={{ fontSize: '13px', color: 'var(--text3)' }}>{'11:59 PM PHT · Not started'}</div>
                  </div>
                </div>
                <div
                  className="lift"
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    gap: '12px',
                    padding: '13px 14px',
                    borderRadius: '10px',
                    border: '1px solid var(--divider)',
                  }}
                >
                  <div style={{ minWidth: '0' }}>
                    <div style={{ fontSize: '16px', fontWeight: '500', color: 'var(--text)' }}>{'Quiz 4: Transformer architecture'}</div>
                    <div style={{ fontSize: '14px', color: 'var(--text2)' }}>{'5% of final grade · Two attempts permitted'}</div>
                  </div>
                  <div style={{ textAlign: 'right', flex: 'none' }}>
                    <div className="mono" style={{ fontSize: '14px', fontWeight: '500', color: 'var(--text)' }}>
                      {'24 October 2026'}
                    </div>
                    <div style={{ fontSize: '13px', color: 'var(--text3)' }}>{'Opens 08:00 AM PHT · Not started'}</div>
                  </div>
                </div>
                <div
                  className="lift"
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    gap: '12px',
                    padding: '13px 14px',
                    borderRadius: '10px',
                    border: '1px solid var(--divider)',
                  }}
                >
                  <div style={{ minWidth: '0' }}>
                    <div style={{ fontSize: '16px', fontWeight: '500', color: 'var(--text)' }}>{'Midterm examination'}</div>
                    <div style={{ fontSize: '14px', color: 'var(--text2)' }}>{'25% of final grade · Weeks 6 to 10 covered · Week 10'}</div>
                  </div>
                  <div style={{ textAlign: 'right', flex: 'none' }}>
                    <div className="mono" style={{ fontSize: '14px', fontWeight: '500', color: 'var(--text)' }}>
                      {'07 November 2026'}
                    </div>
                    <div style={{ fontSize: '13px', color: 'var(--text3)' }}>{'09:00 AM PHT · Not started'}</div>
                  </div>
                </div>
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    gap: '12px',
                    padding: '13px 14px',
                    borderRadius: '10px',
                    border: '1px solid var(--divider)',
                  }}
                >
                  <div style={{ minWidth: '0' }}>
                    <div style={{ fontSize: '16px', fontWeight: '500', color: 'var(--text)' }}>{'Quiz 3: Word embeddings'}</div>
                    <div style={{ fontSize: '14px', color: 'var(--text2)' }}>{'Results released 13 October 2026 · Class mean 82%'}</div>
                  </div>
                  <span
                    className="badge"
                    style={{
                      background: 'var(--neutral-chip-bg)',
                      color: 'var(--neutral-chip-text)',
                      border: '1px solid var(--neutral-chip-border)',
                      flex: 'none',
                    }}
                  >
                    {'Graded'}
                  </span>
                </div>
              </div>
              <p
                style={{
                  margin: '16px 0 0',
                  paddingTop: '14px',
                  borderTop: '1px solid var(--divider)',
                  fontSize: '14px',
                  color: 'var(--text3)',
                  lineHeight: '1.65',
                }}
              >
                {'Submitted work must be your own. Disclose any AI assistance as set out in the course policy. '}
                <DcLink href="Certificates.dc.html" style={{ fontWeight: '500' }}>
                  {'View all assessments and weights →'}
                </DcLink>
              </p>
            </div>
          </div>
          <div className="sect">
            <div className="sect-h">
              <h2 className="sect-t">{'Course materials'}</h2>
              <span className="sect-n">{'Reference screens for this programme'}</span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(230px,1fr))', gap: '16px' }}>
              <DcLink href="CourseMap.dc.html" className="glass lift" style={{ display: 'block', padding: '20px', color: 'inherit' }}>
                <div className="mono" style={{ fontSize: '13px', fontWeight: '500', color: 'var(--text3)', marginBottom: '8px' }}>
                  {'Course outline'}
                </div>
                <div style={{ fontSize: '18.5px', fontWeight: '600', color: 'var(--text)', marginBottom: '6px' }}>
                  {'All 20 weeks, four terms'}
                </div>
                <div style={{ fontSize: '14px', fontWeight: '500', color: 'var(--accent-strong)', marginTop: '12px' }}>
                  {'Open the course outline →'}
                </div>
              </DcLink>
              <DcLink href={v.wk.lessonHref} className="glass lift" style={{ display: 'block', padding: '20px', color: 'inherit' }}>
                <div className="mono" style={{ fontSize: '13px', fontWeight: '500', color: 'var(--text3)', marginBottom: '8px' }}>
                  {'Lesson workspace'}
                </div>
                <div style={{ fontSize: '18.5px', fontWeight: '600', color: 'var(--text)', marginBottom: '6px' }}>
                  {v.wk.lessonCard.heading}
                </div>
                <div style={{ fontSize: '14px', fontWeight: '500', color: 'var(--accent-strong)', marginTop: '12px' }}>
                  {v.wk.lessonCard.cta}
                </div>
              </DcLink>
              <DcLink href="PromptWorkshop.dc.html" className="glass lift" style={{ display: 'block', padding: '20px', color: 'inherit' }}>
                <div className="mono" style={{ fontSize: '13px', fontWeight: '500', color: 'var(--text3)', marginBottom: '8px' }}>
                  {'Practical workshops'}
                </div>
                <div style={{ fontSize: '18.5px', fontWeight: '600', color: 'var(--text)', marginBottom: '6px' }}>
                  {'Prompt engineering workshop'}
                </div>
                <div style={{ fontSize: '14px', fontWeight: '500', color: 'var(--accent-strong)', marginTop: '12px' }}>
                  {'Open the workshops →'}
                </div>
              </DcLink>
            </div>
          </div>
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
    </>
  );
}

Component.template = template;
Component.css = pageCss;
Component.pageKey = 'dashboard';
Component.persist = [];

export default Component;
