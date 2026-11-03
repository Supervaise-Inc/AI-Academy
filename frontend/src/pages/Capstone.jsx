// Generated from design-source/Capstone.dc.html by tools/convert.mjs.
// Re-running the converter regenerates (and overwrites) this file, so edit design-source/ + tools/patches.mjs instead,
// or stop using the converter and edit this file directly.
/* eslint-disable */
import React from 'react';
import DCLogic from '../dc/DCLogic.jsx';
import DcLink from '../dc/DcLink.jsx';
import { css } from '../dc/css.js';
import pageCss from './Capstone.css?inline';

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

const CURRENT_WEEK = 7;
const TEMPLATES = {
  proposal: {
    title: 'Proposal: required contents',
    items: [
      'Project title, team name and members with roles (team lead, technical lead, responsible-AI lead)',
      'Problem statement and intended users, approach and architecture diagram',
      'Data sources and permissions, models and computing resource estimate',
      'Evaluation plan and responsible-AI risk table',
      'Timeline and roles, plus a 3-minute in-class presentation',
    ],
  },
  progress: {
    title: 'Progress review: required contents',
    items: [
      'Status summary for data, core pipeline and demonstration notebook (done, in progress, not started or dropped)',
      'A running prototype demonstrated to the adviser',
      'First results and current blockers',
      'Updated plan for the remaining weeks',
      'Contribution table for every team member',
    ],
  },
  final: {
    title: 'Final report: required contents',
    items: [
      'Executive summary: problem, user, what was built, one key result with a number, main limitation',
      'About 6 to 10 pages plus appendices, submitted with the project folder',
      'Scored under Documentation (10%); also supports Technical implementation and Responsible AI',
      'Sources are credited; the README and report must match what the panel sees in the live demonstration',
    ],
  },
  judging: {
    title: 'Panel evaluation: how the final defence is scored',
    items: [
      'Problem and proposal (15%): clear problem, user and sensible scope',
      'Technical implementation (40%): live demonstration, evaluation against a baseline, handling of an unscripted panel input',
      'Responsible AI (15%): risks named, mitigations shown working, limitations stated',
      'Presentation (20%): clear structure, within time, every member speaks',
      'Documentation (10%): README and report available and consistent with the demonstration',
    ],
  },
};
const MILESTONES = [
  {
    week: 15,
    date: '9 December 2026',
    title: 'Proposal',
    weight: '15% of capstone grade',
    state: 'upcoming',
    bring: ['2 to 3 page proposal', 'Architecture diagram and risk table', '3-minute pitch to the class'],
    items: TEMPLATES.proposal.items,
  },
  {
    week: 18,
    date: '30 December 2026',
    title: 'Progress review',
    weight: 'Counts toward Technical implementation (40%)',
    state: 'upcoming',
    bring: ['A working prototype', '10-minute review with your adviser', 'Attendance is mandatory'],
    items: TEMPLATES.progress.items,
  },
  {
    week: 20,
    date: '13 January 2027',
    title: 'Final defence',
    weight: 'Presentation 20% · Documentation 10%',
    state: 'upcoming',
    bring: ['Live demo, 10-minute talk, 5-minute Q&A', 'Final report, 6 to 10 pages', 'Clean, runnable project folder'],
    items: TEMPLATES.final.items.concat(['Week 19 is reserved for building and the mock NCA-GENL exam']),
  },
];
const NOTES = [
  {
    week: 'Week 7',
    text: 'Pipeline sketch: photograph, defect classifier (crack, spalling, corrosion or none), retrieval of matching checklist items, then a draft summary for engineer review.',
  },
  {
    week: 'Week 7',
    text: 'Responsible-AI rule for version 1: the assistant triages and drafts, and does not issue a pass or fail safety sign-off.',
  },
  { week: 'Week 6', text: 'Backup topic required for team formation: a retrieval-augmented assistant over laboratory-equipment manuals.' },
];

class Component extends DCLogic {
  constructor(props) {
    super(props);
    this.state = { theme: readTheme(), tpl: 'proposal' };
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
  pick(key) {
    return this.state.tpl === key
      ? { bg: 'var(--accent-soft-bg)', border: 'var(--accent)' }
      : { bg: 'var(--card-bg)', border: 'var(--card-border)' };
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
    const milestones = MILESTONES.map((m) => {
      let status,
        chipBg,
        chipFg,
        chipBorder,
        bar = 'var(--accent)';
      if (m.state === 'done') {
        status = 'Submitted';
        chipBg = 'var(--accent-soft-bg)';
        chipFg = 'var(--accent-soft-text)';
        chipBorder = 'var(--accent-soft-border)';
      } else if (m.state === 'active') {
        status = 'In progress';
        chipBg = 'var(--amber-soft-bg)';
        chipFg = 'var(--amber-text)';
        chipBorder = 'var(--amber-soft-border)';
        bar = 'var(--amber)';
      } else {
        status = 'Not started';
        chipBg = 'var(--neutral-chip-bg)';
        chipFg = 'var(--neutral-chip-text)';
        chipBorder = 'var(--neutral-chip-border)';
      }
      return {
        week: m.week,
        date: m.date,
        title: m.title,
        weight: m.weight,
        bring: m.bring,
        items: m.items,
        status,
        chipBg,
        chipFg,
        chipBorder,
      };
    });
    const feedback = [
      {
        who: 'Faculty adviser',
        when: 'Week 7 consultation hours',
        initials: 'CI',
        avBg: 'var(--accent-soft-bg)',
        avFg: 'var(--accent-soft-text)',
        text: 'Strong, real-world problem. Keep a licensed engineer in the loop and scope version 1 to triage, not pass or fail decisions. Begin checking dataset licences now.',
      },
    ];
    const p = this.pick('proposal'),
      g = this.pick('progress'),
      f = this.pick('final'),
      j = this.pick('judging');
    const t = TEMPLATES[this.state.tpl];
    return {
      ...themeVals(dark),
      toggleTheme: this.toggleTheme,
      milestones,
      notes: NOTES,
      feedback,
      proposalBg: p.bg,
      proposalBorder: p.border,
      progressBg: g.bg,
      progressBorder: g.border,
      finalBg: f.bg,
      finalBorder: f.border,
      judgingBg: j.bg,
      judgingBorder: j.border,
      pickProposal: () => this.setState({ tpl: 'proposal' }),
      pickProgress: () => this.setState({ tpl: 'progress' }),
      pickFinal: () => this.setState({ tpl: 'final' }),
      pickJudging: () => this.setState({ tpl: 'judging' }),
      tplTitle: t.title,
      tplItems: t.items,
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
          <DcLink className="navlink active" href="Capstone.dc.html">
            <span>{'⚐'}</span>
            <span>{'Capstone Project'}</span>
          </DcLink>{' '}
          <DcLink className="navlink" href="PracticeExam.dc.html">
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
                <div style={{ fontSize: '11.5px', color: '#9BA3AF', lineHeight: '1.4' }}>{'Student · Week 7 of 20'}</div>
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
            <header className="hx hx-amber cp-head">
              <div className="hx-body">
                <div className="eyebrow" style={{ marginBottom: 'var(--s2)' }}>
                  {'Team capstone · Weeks 13 to 20'}
                </div>
                <h1 className="cm-title">{'Structural Inspection Assistant'}</h1>
                <p className="cm-lede" style={{ margin: 'var(--s3) 0 0' }}>
                  {'Build an AI assistant that '}
                  <mark className="mk">{'triages inspection photos'}</mark>
                  {' and drafts a summary for an engineer to review.'}
                </p>
                <div className="cp-chips">
                  <span className="chip-i">{'20% of final grade'}</span>
                  <span className="chip-i">{'Teams of students'}</span>
                  <span className="chip-i">{'Starts Week 13'}</span>
                </div>
              </div>
              <div className="hx-ico">
                <svg viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M5 21V4M5 4h11l-2 4 2 4H5"></path>
                </svg>
              </div>
            </header>
            <div className="sect">
              <div className="sect-h">
                <h2 className="sect-t">{'What to do now'}</h2>
                <span className="sect-n">{'Week 7 of 20 · the project itself opens in Week 13'}</span>
              </div>
              <p className="cm-lede cm-r" style={{ margin: '0 0 var(--s4)' }}>
                {'No team yet. '}
                <mark className="mk">{'Build the skills you will need'}</mark>
                {', and arrive in Week 13 with an idea.'}
              </p>
              <div className="now">
                <DcLink className="now-i first" href="Lesson.dc.html">
                  <span className="now-n">{'1'}</span> <span className="now-t">{'Finish the Week 7 attention laboratory'}</span>{' '}
                  <span className="now-d">{'It teaches the vision-transformer idea your image triage stage is built on.'}</span>{' '}
                  <span className="now-w">{'This week · Open the lesson workspace'}</span>
                </DcLink>
                <div className="now-i">
                  <span className="now-n">{'2'}</span> <span className="now-t">{'Keep scouting datasets'}</span>{' '}
                  <span className="now-d">
                    {'Your idea brief is in. Collect candidate photo sets of cracks, spalling and corrosion, and note each licence.'}
                  </span>{' '}
                  <span className="now-w">{'Before Week 13'}</span>
                </div>
                <div className="now-i">
                  <span className="now-n">{'3'}</span> <span className="now-t">{'Plan your team'}</span>{' '}
                  <span className="now-d">
                    {'Registration opens on 25 November 2026. Talk to classmates now about roles and strengths.'}
                  </span>{' '}
                  <span className="now-w">{'Week 13 · 25 November 2026'}</span>
                </div>
              </div>
            </div>
            <div className="sect">
              <div className="sect-h">
                <h2 className="sect-t">{'The project'}</h2>
                <span className="sect-n">{'Version 1: triage and draft only'}</span>
              </div>
              <div className="cp2">
                <div className="card" style={{ padding: '22px 24px' }}>
                  <p className="cp-prob">
                    {'Inspectors must sort photos of cracks, spalling and corrosion, then write a summary based on the right checklist.'}
                  </p>
                  <ol className="flow" aria-label="How the assistant works">
                    <li>
                      <span className="fl-n">{'1'}</span>
                      <span className="fl-t">{'Photo in'}</span>
                    </li>
                    <li>
                      <span className="fl-n">{'2'}</span>
                      <span className="fl-t">
                        {'Classify'}
                        <small>{'crack · spalling · corrosion · none'}</small>
                      </span>
                    </li>
                    <li>
                      <span className="fl-n">{'3'}</span>
                      <span className="fl-t">
                        {'Find checklist items'}
                        <small>{'RAG'}</small>
                      </span>
                    </li>
                    <li>
                      <span className="fl-n">{'4'}</span>
                      <span className="fl-t">
                        {'Draft summary'}
                        <small>{'for the engineer'}</small>
                      </span>
                    </li>
                  </ol>
                  <div className="callout" style={{ marginTop: 'var(--s3)' }}>
                    <p style={{ fontSize: '15px', color: 'var(--text)', margin: '0' }}>
                      <b style={{ fontWeight: '600' }}>{'AI-use rule:'}</b>{' '}
                      <mark className="mk">{'a licensed engineer makes every pass or fail decision.'}</mark>
                      {' The assistant never signs off on safety.'}
                    </p>
                  </div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginTop: '14px' }}>
                    <span
                      className="badge"
                      style={{ background: 'var(--cyan-soft-bg)', color: 'var(--cyan-text)', border: '1px solid var(--cyan-soft-border)' }}
                    >
                      {'Vision · ViT / CLIP'}
                    </span>
                    <span
                      className="badge"
                      style={{ background: 'var(--indigo-soft-bg)', color: 'var(--indigo)', border: '1px solid var(--indigo-soft-border)' }}
                    >
                      {'RAG over checklists'}
                    </span>
                    <span
                      className="badge"
                      style={{
                        background: 'var(--amber-soft-bg)',
                        color: 'var(--amber-text)',
                        border: '1px solid var(--amber-soft-border)',
                      }}
                    >
                      {'NeMo Guardrails'}
                    </span>
                  </div>
                </div>
                <div className="card" style={{ padding: '22px 24px' }}>
                  <div className="cp-k">{'Where you are'}</div>
                  <div className="cp-v">{'Pre-work · Week 7 of 20'}</div>
                  <div className="bar" style={{ margin: '12px 0 6px' }}>
                    <div className="barfill" style={{ width: '0%', background: 'var(--accent)' }}></div>
                  </div>
                  <div className="cp-s">{'Idea brief submitted. Dataset scouting in progress.'}</div>
                  <div className="cp-k" style={{ marginTop: 'var(--s4)' }}>
                    {'Next'}
                  </div>
                  <div className="cp-v">{'Team registration'}</div>
                  <div className="cp-s">{'Week 13 · 25 November 2026'}</div>
                </div>
              </div>
            </div>
            <div className="sect">
              <div className="sect-h">
                <h2 className="sect-t">{'Milestones'}</h2>
                <span className="sect-n">{'Three deadlines'}</span>
              </div>
              <div className="ms3">
                {(v.milestones || []).map((m, $index) => (
                  <React.Fragment key={$index}>
                    <article className="mscard">
                      <div className="ms-top">
                        <span className="ms-wk">
                          {'Week '}
                          {m?.week}
                        </span>
                        <span className="ms-d">{m?.date}</span>
                      </div>
                      <h3 className="ms-t">{m?.title}</h3>
                      <p className="ms-w">{m?.weight}</p>
                      <ul className="ms-l">
                        {(m?.bring || []).map((b, $index) => (
                          <React.Fragment key={$index}>
                            <li>{b}</li>
                          </React.Fragment>
                        ))}
                      </ul>
                      <details className="ms-more">
                        <summary>{'Full checklist'}</summary>
                        <ul>
                          {(m?.items || []).map((i, $index) => (
                            <React.Fragment key={$index}>
                              <li>{i}</li>
                            </React.Fragment>
                          ))}
                        </ul>
                      </details>
                      <span
                        className="badge ms-b"
                        style={css(`background:${m?.chipBg};color:${m?.chipFg};border:1px solid ${m?.chipBorder}`)}
                      >
                        {m?.status}
                      </span>
                    </article>
                  </React.Fragment>
                ))}
              </div>
            </div>
            <div className="sect">
              <div className="sect-h">
                <h2 className="sect-t">{'How you are graded'}</h2>
                <span className="sect-n">{'100% of the capstone grade'}</span>
              </div>
              <div className="card" style={{ padding: '22px 24px' }}>
                <div
                  style={{
                    display: 'flex',
                    height: '14px',
                    borderRadius: '999px',
                    overflow: 'hidden',
                    marginBottom: 'var(--s3)',
                    gap: '2px',
                  }}
                  aria-hidden="true"
                >
                  <div style={{ width: '15%', background: 'var(--accent)' }}></div>
                  <div style={{ width: '40%', background: 'var(--accent-strong)' }}></div>
                  <div style={{ width: '15%', background: 'var(--cyan)' }}></div>
                  <div style={{ width: '20%', background: 'var(--amber)' }}></div>
                  <div style={{ width: '10%', background: 'var(--indigo)' }}></div>
                </div>
                <ul className="rub">
                  <li>
                    <span className="rb-s" style={{ background: 'var(--accent)' }}></span>
                    <b>{'Problem and proposal'}</b>
                    <span className="rb-w">{'15%'}</span>
                    <span className="rb-d">{'Clear problem, user and scope'}</span>
                  </li>
                  <li>
                    <span className="rb-s" style={{ background: 'var(--accent-strong)' }}></span>
                    <b>{'Technical implementation'}</b>
                    <span className="rb-w">{'40%'}</span>
                    <span className="rb-d">{'Live demo, baseline comparison, an unscripted input'}</span>
                  </li>
                  <li>
                    <span className="rb-s" style={{ background: 'var(--cyan)' }}></span>
                    <b>{'Responsible AI'}</b>
                    <span className="rb-w">{'15%'}</span>
                    <span className="rb-d">{'Risks named, mitigations shown, limits stated'}</span>
                  </li>
                  <li>
                    <span className="rb-s" style={{ background: 'var(--amber)' }}></span>
                    <b>{'Presentation'}</b>
                    <span className="rb-w">{'20%'}</span>
                    <span className="rb-d">{'Clear, on time, every member speaks'}</span>
                  </li>
                  <li>
                    <span className="rb-s" style={{ background: 'var(--indigo)' }}></span>
                    <b>{'Documentation'}</b>
                    <span className="rb-w">{'10%'}</span>
                    <span className="rb-d">{'README and report match the demo'}</span>
                  </li>
                </ul>
              </div>
            </div>
            <div className="sect">
              <div className="sect-h">
                <h2 className="sect-t">{'Team and adviser'}</h2>
                <span className="sect-n">{'Registration opens Week 13'}</span>
              </div>
              <div className="cp2">
                <div className="card" style={{ padding: '8px 24px' }}>
                  <div className="logrow">
                    <span className="lg-k">{'Roles'}</span>
                    <span className="lg-v">{'Team lead · Technical lead · Responsible-AI lead'}</span>
                  </div>
                  <div className="logrow">
                    <span className="lg-k">{'Register'}</span>
                    <span className="lg-v">{'With a backup topic'}</span>
                  </div>
                  <div className="logrow">
                    <span className="lg-k">{'Adviser'}</span>
                    <span className="lg-v">{'Dr. J. A. Jose · Faculty Room 5, by appointment'}</span>
                  </div>
                  <div className="logrow">
                    <span className="lg-k">{'Repository'}</span>
                    <span className="lg-v">{'Created for your team at registration'}</span>
                  </div>
                </div>
                <div className="card" style={{ padding: '22px 24px' }}>
                  <div className="cp-k">{'Adviser note · Week 7'}</div>
                  <p className="cp-note">
                    {'Strong, real-world problem. Keep an engineer in the loop, scope version 1 to triage, and '}
                    <mark className="mk">{'check dataset licences now.'}</mark>
                  </p>
                </div>
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
      </div>
    </>
  );
}

Component.template = template;
Component.css = pageCss;
Component.pageKey = 'capstone';
Component.persist = [];

export default Component;
