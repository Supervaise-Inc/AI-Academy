import React, { useEffect, useId, useLayoutEffect, useRef, useState, useSyncExternalStore } from 'react';
import { NavLink, Link } from 'react-router-dom';
import { marked } from 'marked';
import DOMPurify from 'dompurify';
import * as session from '../dc/session.js';
import { CURRENT_WEEK } from '../data/courseWeeks.js';
import { DARK_VARS, LIGHT_VARS } from './themeVars.js';
import { formatDateTime } from './format.js';
import { UtilBar, SiteFooter } from './Chrome.jsx';
import './lms.css';
import './chrome.css';

const THEME_KEY = 'supervaise-theme';
const THEME_EVENT = 'supervaise-theme-change';

const COURSE_NAV = [
  { to: '/dashboard', label: 'Overview', icon: '●', end: true },
  { to: '/course-map', label: 'Course Map', icon: '◉' },
  { to: '/lesson', label: 'Lesson Workspace', icon: '◇' },
  { to: '/prompt-workshop', label: 'Prompt Workshop', icon: '✎' },
  { to: '/capstone', label: 'Capstone Project', icon: '⚐' },
  { to: '/mock-exam', label: 'Mock Exam Prep', icon: '⏱' },
  { to: '/certification', label: 'Certification Center', icon: '▲' },
];
const ACTIVITY_NAV = [
  { to: '/activities', label: 'Activities', icon: '▤' },
  { to: '/announcements', label: 'Announcements', icon: '✉' },
  { to: '/grades', label: 'My grades', icon: '✓' },
];
const GROUPS = [
  { label: 'Course', items: COURSE_NAV },
  { label: 'Activities', items: ACTIVITY_NAV },
];

function readTheme() {
  try {
    const s = localStorage.getItem(THEME_KEY);
    if (s === 'light' || s === 'dark') return s;
  } catch (e) {}
  return 'light';
}

function useTheme() {
  const [theme, setTheme] = useState(readTheme);
  useEffect(() => {
    const onLocal = (e) => (e.detail === 'light' || e.detail === 'dark') && setTheme(e.detail);
    window.addEventListener(THEME_EVENT, onLocal);
    return () => window.removeEventListener(THEME_EVENT, onLocal);
  }, []);
  const choose = (t) => {
    setTheme(t);
    try {
      localStorage.setItem(THEME_KEY, t);
    } catch (e) {}
    try {
      window.dispatchEvent(new CustomEvent(THEME_EVENT, { detail: t }));
    } catch (e) {}
  };
  return [theme, choose];
}

const initials = (name) =>
  String(name || '')
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((t) => t[0].toUpperCase())
    .join('') || '?';

function applyVars(el, theme) {
  if (!el) return;
  for (const decl of (theme === 'dark' ? DARK_VARS : LIGHT_VARS).split(';')) {
    const i = decl.indexOf(':');
    if (i > 0) el.style.setProperty(decl.slice(0, i).trim(), decl.slice(i + 1).trim());
  }
}

function Sidebar({ user }) {
  const [open, setOpen] = useState(false);
  const menuId = useId();
  return (
    <div className="sidebar">
      <div className="sb-brand" role="region" aria-label="Course">
        <div className="sb-k">Course</div>
        <Link className="sb-t" to="/dashboard">
          GenAI &amp; LLMs with NVIDIA<span className="lms-sr">, dashboard</span>
        </Link>
        <div className="sb-bar" />
      </div>
      <button type="button" className="sb-toggle" aria-expanded={open} aria-controls={menuId} onClick={() => setOpen(!open)}>
        <span>Menu</span>
        <span className="sb-caret" aria-hidden="true">
          ▾
        </span>
      </button>
      <div className="sb-menu" id={menuId} data-open={open}>
        <nav className="sb-nav" aria-label="Main">
          {GROUPS.map((g) => (
            <div className="sb-group" key={g.label}>
              <div className="sb-glabel">{g.label}</div>
              {g.items.map((it) => (
                <NavLink key={it.to} to={it.to} end={it.end} className={({ isActive }) => `navlink${isActive ? ' active' : ''}`}>
                  <span aria-hidden="true">{it.icon}</span>
                  <span>{it.label}</span>
                </NavLink>
              ))}
            </div>
          ))}
        </nav>
        <div className="sb-spacer" />
        <div className="sb-profile">
          <div className="profilecard">
            <div className="sb-avatar" aria-hidden="true">
              {initials(user && user.name)}
            </div>
            <div className="sb-who">
              <div className="sb-name">{user ? user.name : 'Signed in'}</div>
              <div className="sb-role">Student · Week {CURRENT_WEEK}</div>
            </div>
          </div>
          <button type="button" className="navlink sb-out" onClick={() => session.logout().then(() => session.navigate('/'))}>
            <span aria-hidden="true">⎋</span>
            <span>Sign out</span>
          </button>
        </div>
      </div>
    </div>
  );
}

// Page frame: sidebar + top bar. Wrap each student page in this.
export function Shell({ children, tone }) {
  const { user } = useSyncExternalStore(session.subscribe, session.getSession);
  const [theme, setTheme] = useTheme();
  const ref = useRef(null);
  useLayoutEffect(() => applyVars(ref.current, theme), [theme]);
  return (
    <div className="lms lshell" data-theme={theme} data-role="student" ref={ref}>
      <Sidebar user={user} />
      <main id="main-content" className="cm" tabIndex={-1} data-tone={tone}>
        <UtilBar theme={theme} onChoose={setTheme} />
        {children}
        <SiteFooter />
      </main>
    </div>
  );
}

const GLYPHS = {
  checklist: <path d="M12 17l4 4 7-8M30 19h18M12 31l4 4 7-8M30 33h18M12 45l4 4 7-8M30 47h18" />,
  megaphone: <path d="M12 24h8l22-12v36L20 36h-8zM20 36l3 12h6l-2-10M48 24c3 3 3 9 0 12" />,
  open: (
    <>
      <circle cx="30" cy="30" r="20" />
      <path d="M30 18v13l9 5" />
    </>
  ),
  late: <path className="ph-gl-warn" d="M30 10L52 50H8zM30 24v12M30 43v1" />,
  done: (
    <>
      <circle cx="30" cy="30" r="20" />
      <path d="M20 31l7 7 14-15" />
    </>
  ),
  graded: <path d="M30 8l6.5 14 15 1.8-11 10.4 2.9 15L30 41l-13.4 8.2 2.9-15-11-10.4 15-1.8z" />,
};

// Rounded square icon tile, the same shape as the hero icon on the other pages.
function Emblem({ name }) {
  return (
    <div className="ph-tile" data-emblem={GLYPHS[name] ? name : 'checklist'} aria-hidden="true">
      <svg viewBox="0 0 60 60" focusable="false">
        {GLYPHS[name] || GLYPHS.checklist}
      </svg>
    </div>
  );
}

// Big header card at the top of each page.
export function PageHeader({ eyebrow, title, intro, highlight, chips = [], emblem, headingRef }) {
  const shown = chips.filter((c) => c != null && c !== '' && c !== false);
  let lede = intro;
  if (intro && highlight && intro.includes(highlight)) {
    const i = intro.indexOf(highlight);
    lede = (
      <>
        {intro.slice(0, i)}
        <mark className="mk">{highlight}</mark>
        {intro.slice(i + highlight.length)}
      </>
    );
  }
  return (
    <header className={`ph${emblem ? ' ph-has-emblem' : ''}`}>
      <div className="ph-main">
        {eyebrow && <div className="ph-eye">{eyebrow}</div>}
        <h1 id="lms-title" className="cm-title" ref={headingRef} tabIndex={-1}>
          {title}
        </h1>
        {lede && <p className="cm-lede">{lede}</p>}
        {shown.length > 0 && (
          <ul className="ph-chips" aria-label="At a glance">
            {shown.map((c, i) => (
              <li className="chip-i" key={i}>
                {c}
              </li>
            ))}
          </ul>
        )}
      </div>
      {emblem && (
        <div className="ph-seal">
          <Emblem name={emblem} />
        </div>
      )}
    </header>
  );
}

export function Time({ value, now }) {
  return value ? <time dateTime={value}>{formatDateTime(value, now)}</time> : null;
}

export function Markdown({ text }) {
  const html = DOMPurify.sanitize(marked.parse(text || '', { breaks: true }));
  return <div className="lms-md" dangerouslySetInnerHTML={{ __html: html }} />;
}

export function SampleBanner() {
  return (
    <div className="edc-err lms-banner" role="note">
      <p>
        <strong>Sample data.</strong> This server does not have the student activities service yet, so you are seeing example items.
      </p>
    </div>
  );
}

// Load once, with loading / error / ready states.
export function useLoad(fn) {
  const [state, setState] = useState({ status: 'loading', data: null, error: null });
  const [tick, setTick] = useState(0);
  useEffect(() => {
    let alive = true;
    setState((s) => ({ ...s, status: s.data ? 'ready' : 'loading', error: null }));
    fn().then(
      (data) => alive && setState({ status: 'ready', data, error: null }),
      (error) => alive && setState((s) => ({ status: 'error', data: s.data, error })),
    );
    return () => {
      alive = false;
    };
  }, [fn, tick]);
  return [state, () => setTick((t) => t + 1)];
}

export function LoadState({ state, what, onRetry }) {
  if (state.status === 'loading')
    return (
      <div className="lms-list" aria-busy="true" aria-hidden="true">
        <div className="lms-skeleton" />
        <div className="lms-skeleton" />
        <div className="lms-skeleton" />
      </div>
    );
  if (state.status === 'error' && !state.data)
    return (
      <div className="card lms-state" role="alert">
        <p>
          <strong>We could not load {what}.</strong> {state.error && state.error.message}
        </p>
        <button type="button" className="ghostbtn" onClick={onRetry}>
          Try again
        </button>
      </div>
    );
  return null;
}
