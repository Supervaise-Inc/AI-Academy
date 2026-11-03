import React, { useState } from 'react';
import { Link } from 'react-router-dom';

// Top bar and footer, the same as on every other student page.
const icon = (children) => (
  <svg className="hkic" viewBox="0 0 24 24" fill="none" aria-hidden="true">
    {children}
  </svg>
);
const ICONS = {
  s: icon(
    <>
      <rect x="3.2" y="4.8" width="17.6" height="15.4" rx="2.6" stroke="currentColor" strokeWidth="1.7" />
      <path d="M3.2 9.4h17.6M8.2 2.9v3.6M15.8 2.9v3.6" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
    </>,
  ),
  u: icon(
    <>
      <circle cx="12" cy="12" r="8.8" stroke="currentColor" strokeWidth="1.7" />
      <path d="M12 6.9V12l3.5 2.1" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
    </>,
  ),
  a: icon(
    <>
      <path d="M18.3 16.2V10.6a6.3 6.3 0 1 0-12.6 0v5.6l-1.6 2.3h15.8z" stroke="currentColor" strokeWidth="1.7" strokeLinejoin="round" />
      <path d="M9.9 21.1h4.2" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
    </>,
  ),
  t: icon(
    <>
      <circle cx="12" cy="12" r="3.1" stroke="currentColor" strokeWidth="1.7" />
      <path
        d="M19.3 14.4a1.6 1.6 0 0 0 .3 1.8l.1.1a1.9 1.9 0 1 1-2.7 2.7l-.1-.1a1.6 1.6 0 0 0-2.7 1.1v.3a1.9 1.9 0 0 1-3.8 0v-.2a1.6 1.6 0 0 0-2.8-1.1l-.1.1a1.9 1.9 0 1 1-2.7-2.7l.1-.1a1.6 1.6 0 0 0-1.1-2.7h-.3a1.9 1.9 0 0 1 0-3.8h.2a1.6 1.6 0 0 0 1.1-2.8l-.1-.1a1.9 1.9 0 1 1 2.7-2.7l.1.1a1.6 1.6 0 0 0 1.8.3h.1a1.6 1.6 0 0 0 1-1.5v-.3a1.9 1.9 0 0 1 3.8 0v.2a1.6 1.6 0 0 0 2.7 1.1l.1-.1a1.9 1.9 0 1 1 2.7 2.7l-.1.1a1.6 1.6 0 0 0 1.1 2.7h.3a1.9 1.9 0 0 1 0 3.8h-.2a1.6 1.6 0 0 0-1.5 1z"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
    </>,
  ),
};

function Row({ time, title, note }) {
  return (
    <li>
      <span className="hkt">{time}</span>
      <span className="hkd">
        <b>{title}</b>
        <span className="hkm">{note}</span>
      </span>
    </li>
  );
}

export function UtilBar({ theme, onChoose }) {
  const [open, setOpen] = useState('');
  const dark = theme === 'dark';
  const toggle = (k) => () => setOpen(open === k ? '' : k);
  const panel = (k, label, title, button, body) => (
    <div className={open === k ? 'hk is-open' : 'hk'}>
      <button
        type="button"
        className={'hkbtn' + (k === 't' ? ' hkbtn-icon' : '')}
        aria-haspopup="true"
        aria-expanded={open === k}
        onClick={toggle(k)}
        aria-label={label}
        title={title}
      >
        {ICONS[k]}
        {button}
      </button>
      <div className="hkpop">{body}</div>
    </div>
  );
  return (
    <div className="utilbar">
      <div className="ctx">
        <span className="ctx-i">A.Y. 2026/2027</span> <span className="ctx-s" aria-hidden="true" />
        <span className="ctx-i">Second Semester</span> <span className="ctx-s" aria-hidden="true" />
        <span className="ctx-i ctx-now">Week 7 of 20</span> <span className="ctx-s" aria-hidden="true" />
        <span className="ctx-i">Midterm Term</span>
      </div>
      <div className="hkbar" role="toolbar" aria-label="Course tools" onKeyDown={(e) => e.key === 'Escape' && setOpen('')}>
        {open && <div className="hk-scrim" onClick={() => setOpen('')} />}
        {panel(
          's',
          'Schedule',
          'Schedule (S)',
          null,
          <>
            <div className="hkpop-h">Today · Wednesday, 14 October 2026</div>
            <ul className="hkl">
              <Row time="08:00" title="Lecture 7.1: Attention mechanisms" note="Room CS‑204 · Dr. J. A. Jose" />
              <Row time="10:30" title="Laboratory 3: Scaled dot‑product attention" note="Computing Laboratory 2 · Attendance required" />
              <Row time="14:00" title="Consultation hours" note="By appointment · Faculty Room 5" />
            </ul>
            <Link className="hkpop-f" to="/course-map" onClick={() => setOpen('')}>
              View the full course schedule
            </Link>
          </>,
        )}
        {panel(
          'u',
          'Upcoming requirements',
          'Upcoming requirements (U)',
          null,
          <>
            <div className="hkpop-h">Upcoming requirements</div>
            <ul className="hkl">
              <Row time="17 Oct" title="Laboratory 3 report" note="10% of final grade · Due 11:59 PM PHT" />
              <Row time="24 Oct" title="Quiz 4: Transformer architecture" note="5% of final grade · Two attempts permitted" />
              <Row time="07 Nov" title="Midterm examination" note="25% of final grade · Weeks 6 to 10 covered" />
            </ul>
            <Link className="hkpop-f" to="/activities" onClick={() => setOpen('')}>
              View all assessments and weights
            </Link>
          </>,
        )}
        {panel(
          'a',
          'Alerts, 3 unread',
          'Alerts',
          <span className="hkn">3</span>,
          <>
            <div className="hkpop-h">Announcements</div>
            <ul className="hkl">
              <Row time="Today" title="Laboratory 3 rubric published" note="Posted by Dr. J. A. Jose" />
              <Row time="13 Oct" title="Quiz 3 results released" note="Class mean 82% · Review in consultation hours" />
              <Row time="11 Oct" title="Reading added for Week 8" note="Vaswani et al., sections 3.2 to 3.4" />
            </ul>
            <Link className="hkpop-f" to="/announcements" onClick={() => setOpen('')}>
              View all announcements
            </Link>
          </>,
        )}
        {panel(
          't',
          'Settings',
          'Settings',
          null,
          <>
            <div className="hkpop-h">Settings</div>
            <div className="setrow">
              <span className="setlab">Appearance</span>{' '}
              <button
                type="button"
                className="themepill"
                onClick={() => onChoose(dark ? 'light' : 'dark')}
                aria-label="Switch between light and dark theme"
                title="Switch between light and dark theme"
              >
                <span
                  className="themeseg"
                  style={{
                    background: dark ? 'transparent' : '#FFFFFF',
                    color: dark ? '#94A3B8' : '#0F172A',
                    boxShadow: dark ? 'none' : '0 1px 2px rgba(15,23,42,0.12)',
                  }}
                >
                  Light
                </span>
                <span className="themeseg" style={{ background: dark ? 'rgba(118,185,0,0.2)' : 'transparent', color: dark ? '#B6F23A' : '#64748B' }}>
                  Dark
                </span>
              </button>
            </div>
            <div className="setrow">
              <span className="setlab">Notifications</span>
              <span className="setval">Email and in‑app</span>
            </div>
            <div className="setrow">
              <span className="setlab">Time zone</span>
              <span className="setval">
                Philippine Standard Time
                <br />
                (UTC+8)
              </span>
            </div>
            <div className="setrow">
              <span className="setlab">Enrolled section</span>
              <span className="setval">Section A</span>
            </div>
            <a className="hkpop-f" href="#">
              Account and privacy settings
            </a>
          </>,
        )}
      </div>
    </div>
  );
}

const SOCIAL = [
  [
    'https://instagram.com/supervaise_ph',
    'Supervaise on Instagram',
    <>
      <rect x="3.2" y="3.2" width="17.6" height="17.6" rx="5.2" stroke="currentColor" strokeWidth="1.7" />
      <circle cx="12" cy="12" r="4.1" stroke="currentColor" strokeWidth="1.7" />
      <circle cx="17.1" cy="6.9" r="1.25" fill="currentColor" />
    </>,
  ],
  [
    'https://linkedin.com/company/supervaise',
    'Supervaise on LinkedIn',
    <>
      <rect x="3.2" y="3.2" width="17.6" height="17.6" rx="4.4" stroke="currentColor" strokeWidth="1.7" />
      <circle cx="7.8" cy="8.1" r="1.35" fill="currentColor" />
      <rect x="6.75" y="10.5" width="2.1" height="7" fill="currentColor" />
      <path
        d="M11.2 17.5v-7h2.05v1.03a2.75 2.75 0 0 1 2.35-1.23c1.76 0 2.8 1.14 2.8 3.16v4.04h-2.1v-3.76c0-1.04-.45-1.65-1.37-1.65-.95 0-1.63.7-1.63 1.82v3.59z"
        fill="currentColor"
      />
    </>,
  ],
  [
    'https://supervaise.io',
    'supervaise.io',
    <>
      <circle cx="12" cy="12" r="8.8" stroke="currentColor" strokeWidth="1.7" />
      <path d="M3.2 12h17.6" stroke="currentColor" strokeWidth="1.7" />
      <path d="M12 3.2c2.3 2.4 3.5 5.5 3.5 8.8s-1.2 6.4-3.5 8.8c-2.3-2.4-3.5-5.5-3.5-8.8S9.7 5.6 12 3.2z" stroke="currentColor" strokeWidth="1.7" />
    </>,
  ],
  [
    'mailto:anthony.jose@supervaise.io',
    'Email the programme office',
    <>
      <rect x="2.8" y="5" width="18.4" height="14" rx="3.2" stroke="currentColor" strokeWidth="1.7" />
      <path d="M4.2 7.6 12 13l7.8-5.4" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
    </>,
  ],
];

export function SiteFooter() {
  return (
    <footer className="sitefoot">
      <div className="sf-top">
        <div className="sf-brand">
          <svg viewBox="0 0 360 310" aria-hidden="true">
            <polygon points="5,308 140,6 275,308 225,308 142,116 55,308" fill="var(--accent)" />
            <circle cx="250" cy="73" r="40" fill="var(--accent)" />
            <rect x="308" y="13" width="47" height="295" fill="var(--accent)" />
          </svg>{' '}
          <span className="sf-bt">
            <b>AI Academy</b>
            <span>A Supervaise and NVIDIA programme in generative artificial intelligence.</span>
          </span>
        </div>
        <nav className="sf-cols" aria-label="Footer">
          <div className="sf-col">
            <h4>Programme</h4>
            <Link to="/course-map">Course outline</Link> <Link to="/lesson">Lesson workspace</Link> <Link to="/capstone">Capstone project</Link>
          </div>
          <div className="sf-col">
            <h4>Assessment</h4>
            <Link to="/mock-exam">Examination preparation</Link> <Link to="/certification">Credentials and records</Link>{' '}
            <Link to="/prompt-workshop">Practical workshops</Link>
          </div>
          <div className="sf-col">
            <h4>Institution</h4>
            <Link to="/educator">Faculty console</Link> <a href="mailto:anthony.jose@supervaise.io">Programme office</a>{' '}
            <a href="#">Academic integrity policy</a>
          </div>
        </nav>
      </div>
      <div className="sf-bot">
        <span>© 2026 Supervaise Inc. All rights reserved.</span>{' '}
        <span className="sf-soc">
          {SOCIAL.map(([href, label, glyph]) => (
            <a key={href} href={href} target={href.startsWith('http') ? '_blank' : undefined} rel="noopener" aria-label={label}>
              <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                {glyph}
              </svg>
            </a>
          ))}
        </span>{' '}
        <span className="sf-power">
          <span className="sf-pw-l">Powered by</span>
          <img className="sf-nv nvl-d" src="/assets/nvidia-dark.png" alt="NVIDIA" />
          <img className="sf-nv nvl-l" src="/assets/nvidia-light.png" alt="NVIDIA" />
          <strong>DGX Spark</strong>
        </span>
      </div>
    </footer>
  );
}
