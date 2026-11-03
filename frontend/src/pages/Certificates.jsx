// Generated from design-source/Certificates.dc.html by tools/convert.mjs.
// Re-running the converter regenerates (and overwrites) this file, so edit design-source/ + tools/patches.mjs instead,
// or stop using the converter and edit this file directly.
/* eslint-disable */
import React from 'react';
import DCLogic from '../dc/DCLogic.jsx';
import DcLink from '../dc/DcLink.jsx';
import { css } from '../dc/css.js';
import pageCss from './Certificates.css?inline';

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

class Component extends DCLogic {
  constructor(props) {
    super(props);
    const saved = readTheme();
    this.state = { theme: saved };
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
    const greenBg = dark ? 'rgba(118,185,0,0.16)' : '#EAF5D1';
    const greenText = dark ? '#B6F23A' : '#4F7A00';
    const amberBg = dark ? 'rgba(255,160,0,0.16)' : '#FCEBD2';
    const amberText = dark ? '#FFB84D' : '#8A5A12';
    const neutralBg = dark ? 'rgba(255,255,255,0.06)' : '#EEF0E8';
    const neutralText = dark ? '#A0A5B5' : '#5B6358';
    return {
      ...themeVals(dark),
      toggleTheme: this.toggleTheme,
      scores: [
        { name: 'Prelim Examination', type: 'Major exam · Weeks 1 to 5', detail: '', status: 'Passed', bg: greenBg, color: greenText },
        { name: 'Knowledge check GA-3.1', type: 'Practice · Week 7', detail: '', status: 'In progress', bg: amberBg, color: amberText },
        {
          name: 'GA Lab 3 Part 1',
          type: 'Laboratory · Week 7',
          detail: 'Due 17 Oct 2026',
          status: 'In progress',
          bg: amberBg,
          color: amberText,
        },
        {
          name: 'Laboratory exercises 1 to 18',
          type: 'Laboratories · Weeks 1 to 9',
          detail: '18 of 20 done',
          status: 'In progress',
          bg: amberBg,
          color: amberText,
        },
        {
          name: 'Midterm Examination',
          type: 'Major exam · Week 10',
          detail: '25% of final grade',
          status: 'Not started',
          bg: neutralBg,
          color: neutralText,
        },
      ],
      nPassed: 1,
      nProg: 3,
      nTodo: 1,
      domains: [
        { name: 'Core Machine Learning & AI Knowledge', weight: '30%', w: '30%', c: '#76B900' },
        { name: 'Software Development', weight: '24%', w: '24%', c: '#0EA5C6' },
        { name: 'Experimentation', weight: '22%', w: '22%', c: '#6C63F0' },
        { name: 'Data Analysis', weight: '14%', w: '14%', c: '#F59E0B' },
        { name: 'Trustworthy AI', weight: '10%', w: '10%', c: '#D6589F' },
      ],
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
          <DcLink className="navlink active" href="Certificates.dc.html">
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
        <div
          className="cm"
          style={{
            flex: '999 1 560px',
            minWidth: '0',
            background: 'var(--bg)',
            color: 'var(--text)',
            padding: '36px 48px 64px',
            maxWidth: '1160px',
            transition: 'background-color .3s ease,color .3s ease',
          }}
        >
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
          <header className="cx-hero">
            <div>
              <div className="cx-eye">{'Student record'}</div>
              <h1 className="cm-title">{'Credentials and records'}</h1>
              <p className="cm-lede">
                {'Your goal is the '}
                <mark className="mk">{'NVIDIA-Certified Associate'}</mark>
                {'. NVIDIA issues it after its own proctored exam; Supervaise cannot issue or guarantee it.'}
              </p>
              <div className="cp-chips">
                <span className="chip-i">{'2 certification levels'}</span>
                <span className="chip-i">{'Issued by NVIDIA'}</span>
                <span className="chip-i">{'Mock exam in Week 19'}</span>
              </div>
            </div>
            <div className="cx-seal">
              <svg viewBox="0 0 170 200" aria-hidden="true">
                <defs>
                  <linearGradient id="cxg" x1="0" y1="0" x2="1" y2="1">
                    <stop offset="0" stopColor="#C9F65A"></stop>
                    <stop offset="1" stopColor="#4F7A00"></stop>
                  </linearGradient>
                </defs>
                <path d="M52 138 38 192l32-16 18 20 4-52z" fill="#3D5F00"></path>
                <path d="M118 138l14 54-32-16-18 20-4-52z" fill="#4F7A00"></path>
                <path d="M85 6l68 38v78l-68 38-68-38V44z" fill="url(#cxg)"></path>
                <path d="M85 19l57 32v60l-57 32-57-32V51z" fill="#0D1117"></path>
                <path d="M85 19l57 32v60l-57 32-57-32V51z" fill="none" stroke="#B6F23A" strokeOpacity=".55" strokeWidth="1.5"></path>
                <image href="/assets/nvidia-eye.png" x="50" y="57.6" width="70" height="47"></image>
              </svg>
              <span>{'NCA-GENL · Not yet earned'}</span>
            </div>
          </header>
          <div className="sect">
            <div className="sect-h">
              <h2 className="sect-t">{'Your path to the Associate'}</h2>
              <span className="sect-n">{'NCA-GENL · 4 steps'}</span>
            </div>
            <ol className="cx-jr">
              <li className="cur">
                <span className="cx-nd">
                  <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                    <path
                      d="M4 5.5A2.5 2.5 0 0 1 6.5 3H20v15H6.5A2.5 2.5 0 0 0 4 20.5z"
                      stroke="currentColor"
                      strokeWidth="1.7"
                      strokeLinejoin="round"
                    ></path>
                    <path
                      d="M4 20.5A2.5 2.5 0 0 0 6.5 23H20v-5"
                      stroke="currentColor"
                      strokeWidth="1.7"
                      strokeLinejoin="round"
                      transform="translate(0 -3)"
                    ></path>
                    <path d="M9 8h7" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round"></path>
                  </svg>
                </span>
                <div>
                  <span className="cx-here">{'You are here'}</span>
                  <div className="cx-jt">
                    {'Practise'}
                    <small>
                      <DcLink href="PracticeExam.dc.html">{'Open Mock Exam Prep'}</DcLink>
                      {' and finish the course.'}
                    </small>
                  </div>
                </div>
              </li>
              <li>
                <span className="cx-nd">
                  <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                    <circle cx="12" cy="12" r="8.8" stroke="currentColor" strokeWidth="1.7"></circle>
                    <path
                      d="M12 6.9V12l3.5 2.1"
                      stroke="currentColor"
                      strokeWidth="1.7"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    ></path>
                  </svg>
                </span>
                <div>
                  <div className="cx-jt">
                    {'Sit the Week 19 mock'}
                    <small>{'50 questions in 60 minutes.'}</small>
                  </div>
                </div>
              </li>
              <li>
                <span className="cx-nd">
                  <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                    <circle cx="8" cy="15" r="4.2" stroke="currentColor" strokeWidth="1.7"></circle>
                    <path
                      d="M11 12.2 20 3.5M16.5 7l2.8 2.8M14 9.5l2 2"
                      stroke="currentColor"
                      strokeWidth="1.7"
                      strokeLinecap="round"
                    ></path>
                  </svg>
                </span>
                <div>
                  <div className="cx-jt">
                    {'Score 35 of 50 (70%)'}
                    <small>{'Your instructor sends your exam discount code, privately.'}</small>
                  </div>
                </div>
              </li>
              <li className="fin">
                <span className="cx-nd">
                  <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                    <path
                      d="M12 3 4.5 6v5.6c0 4.4 3 8.2 7.5 9.4 4.5-1.2 7.5-5 7.5-9.4V6z"
                      stroke="currentColor"
                      strokeWidth="1.7"
                      strokeLinejoin="round"
                    ></path>
                    <path
                      d="m8.8 12.1 2.3 2.3 4.2-4.6"
                      stroke="currentColor"
                      strokeWidth="1.7"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    ></path>
                  </svg>
                </span>
                <div>
                  <div className="cx-jt">
                    {'Take the NVIDIA exam'}
                    <small>{'Proctored and issued by NVIDIA.'}</small>
                  </div>
                </div>
              </li>
            </ol>
          </div>
          <div className="sect">
            <div className="sect-h">
              <h2 className="sect-t">{'Two levels'}</h2>
              <span className="sect-n">{'Both start as Not started'}</span>
            </div>
            <div className="cx-creds">
              <div className="cx-cred l1">
                <div className="cx-ctop">
                  <span className="cx-ico">
                    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                      <circle cx="12" cy="9" r="5.6" stroke="currentColor" strokeWidth="1.8"></circle>
                      <path d="m9.2 13.6-2 7.4 4.8-2.4 4.8 2.4-2-7.4" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round"></path>
                      <path d="m12 6.3.9 1.9 2 .3-1.5 1.4.4 2-1.8-1-1.8 1 .4-2-1.5-1.4 2-.3z" fill="currentColor"></path>
                    </svg>
                  </span>
                  <span className="cx-code">{'NCA-GENL'}</span>
                </div>
                <span
                  className="cx-tag"
                  style={{
                    alignSelf: 'flex-start',
                    background: 'var(--accent-soft-bg)',
                    color: 'var(--accent-soft-text)',
                    border: '1px solid var(--accent-soft-border)',
                  }}
                >
                  {'Level 1 · Your target'}
                </span>
                <h3>{'NVIDIA-Certified Associate: Generative AI LLMs'}</h3>
                <p>{'The programme prepares you for this exam, including the Week 19 mock.'}</p>
                <div className="cx-facts">
                  <span className="cx-f">
                    <b>{'50'}</b>
                    {' questions'}
                  </span>
                  <span className="cx-f">
                    <b>{'60'}</b>
                    {' minutes'}
                  </span>
                  <span className="cx-f">
                    {'Ready at '}
                    <b>{'35/50'}</b>
                  </span>
                </div>
                <div className="cx-cstat">
                  <span>{'Status'}</span>
                  <span
                    className="cx-tag"
                    style={{ background: 'var(--surface2)', color: 'var(--text2)', border: '1px solid var(--surface2-border)' }}
                  >
                    {'Not started'}
                  </span>
                </div>
              </div>
              <div className="cx-cred l2">
                <div className="cx-ctop">
                  <span className="cx-ico">
                    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                      <rect x="5" y="10.5" width="14" height="10" rx="2.6" stroke="currentColor" strokeWidth="1.8"></rect>
                      <path d="M8.2 10.5V8a3.8 3.8 0 0 1 7.6 0v2.5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"></path>
                    </svg>
                  </span>
                  <span className="cx-code">{'NCP-GENL'}</span>
                </div>
                <span
                  className="cx-tag"
                  style={{
                    alignSelf: 'flex-start',
                    background: 'var(--indigo-soft-bg)',
                    color: 'var(--indigo)',
                    border: '1px solid var(--indigo-soft-border)',
                  }}
                >
                  {'Level 2 · Optional next step'}
                </span>
                <h3>{'NVIDIA-Certified Professional: Generative AI LLMs'}</h3>
                <p>
                  {
                    'The advanced level. Optional preparation is offered once the Associate is done, and the Associate is the recommended prerequisite.'
                  }
                </p>
                <div className="cx-facts">
                  <span className="cx-f">
                    {'Take the '}
                    <b>{'Associate'}</b>
                    {' first'}
                  </span>
                </div>
                <div className="cx-cstat">
                  <span>{'Status'}</span>
                  <span
                    className="cx-tag"
                    style={{ background: 'var(--surface2)', color: 'var(--text2)', border: '1px solid var(--surface2-border)' }}
                  >
                    {'Not started'}
                  </span>
                </div>
              </div>
            </div>
          </div>
          <div className="sect">
            <div className="sect-h">
              <h2 className="sect-t">{'What the exam covers'}</h2>
              <span className="sect-n">{'NCA-GENL · readiness fills in after the Week 19 mock'}</span>
            </div>
            <div className="card" style={{ padding: '24px 26px' }}>
              <div className="cx-bp" aria-hidden="true">
                {(v.domains || []).map((d, $index) => (
                  <React.Fragment key={$index}>
                    <i style={css(`width:${d?.w};background:${d?.c}`)}></i>
                  </React.Fragment>
                ))}
              </div>
              <div className="cx-dg">
                {(v.domains || []).map((d, $index) => (
                  <React.Fragment key={$index}>
                    <div className="cx-dr">
                      <span className="cx-dot" style={css(`background:${d?.c}`)}></span>
                      <span>{d?.name}</span>
                      <span className="cx-dw">{d?.weight}</span>
                      <div className="cx-db">
                        <i style={css(`background:${d?.c}`)}></i>
                      </div>
                    </div>
                  </React.Fragment>
                ))}
              </div>
            </div>
            <div className="cx-key">
              <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <circle cx="8" cy="15" r="4.2" stroke="currentColor" strokeWidth="1.7"></circle>
                <path d="M11 12.2 20 3.5M16.5 7l2.8 2.8M14 9.5l2 2" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round"></path>
              </svg>
              <div>
                <b>{'Discount codes'}</b>
                {
                  ' are issued one by one by your instructor once you reach the readiness mark. They are never posted here or shared with the class.'
                }
              </div>
            </div>
          </div>
          <div className="sect">
            <div className="sect-h">
              <h2 className="sect-t">{'Assessment record'}</h2>
              <span className="sect-n">{'Scores appear when grading is released'}</span>
            </div>
            <div className="cx-st">
              <div className="cx-tile">
                <b style={{ color: 'var(--accent-soft-text)' }}>{v.nPassed}</b>
                <span>{'Passed'}</span>
              </div>
              <div className="cx-tile">
                <b style={{ color: 'var(--amber-text)' }}>{v.nProg}</b>
                <span>{'In progress'}</span>
              </div>
              <div className="cx-tile">
                <b style={{ color: 'var(--text3)' }}>{v.nTodo}</b>
                <span>{'Not started'}</span>
              </div>
            </div>
            <div className="card" style={{ padding: '8px 22px' }}>
              <ul className="rec">
                {(v.scores || []).map((s, $index) => (
                  <React.Fragment key={$index}>
                    <li style={css(`--rc:${s?.color}`)}>
                      <span className="rec-n">
                        {s?.name}
                        <small>{s?.type}</small>
                      </span>{' '}
                      <span className="rec-d">{s?.detail}</span>{' '}
                      <span className="rec-s" style={css(`background:${s?.bg};color:${s?.color}`)}>
                        {s?.status}
                      </span>
                    </li>
                  </React.Fragment>
                ))}
              </ul>
            </div>
          </div>
          <div className="sect">
            <div className="sect-h">
              <h2 className="sect-t">{'Verification'}</h2>
              <span className="sect-n">{'For schools and employers'}</span>
            </div>
            <div className="cx-ver">
              <div className="cx-vc">
                <span className="cx-vi">
                  <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                    <path d="M6 3h8l4 4v14H6z" stroke="currentColor" strokeWidth="1.7" strokeLinejoin="round"></path>
                    <path
                      d="M14 3v4h4M9 12h6M9 16h6"
                      stroke="currentColor"
                      strokeWidth="1.7"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    ></path>
                  </svg>
                </span>
                <div>
                  <h3>{'Programme records'}</h3>
                  <p>
                    {'Held by Supervaise. Send verification requests to the '}
                    <DcLink href="mailto:anthony.jose@supervaise.io">{'programme office'}</DcLink>
                    {'.'}
                  </p>
                </div>
              </div>
              <div className="cx-vc">
                <span className="cx-vi">
                  <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                    <path
                      d="M12 3 4.5 6v5.6c0 4.4 3 8.2 7.5 9.4 4.5-1.2 7.5-5 7.5-9.4V6z"
                      stroke="currentColor"
                      strokeWidth="1.7"
                      strokeLinejoin="round"
                    ></path>
                    <path
                      d="m8.8 12.1 2.3 2.3 4.2-4.6"
                      stroke="currentColor"
                      strokeWidth="1.7"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    ></path>
                  </svg>
                </span>
                <div>
                  <h3>{'NVIDIA certifications'}</h3>
                  <p>
                    {
                      'Issued and confirmed only by NVIDIA. Finishing this course is not the same as holding one, and the two are tracked separately.'
                    }
                  </p>
                </div>
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
    </>
  );
}

Component.template = template;
Component.css = pageCss;
Component.pageKey = 'certification';
Component.persist = [];

export default Component;
