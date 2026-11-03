import React, { useEffect, useMemo, useRef, useState } from 'react';
import DcLink from './DcLink.jsx';
import Infographic from './Infographic.jsx';
import { overviewOf } from '../data/lessonOverview/index.js';
import { CURRENT_WEEK, TOTAL_WEEKS } from '../data/courseWeeks.js';

// "Lesson library": a sheet that covers the Lesson Workspace. Screen one shows every weekly lesson as a card with
// its infographic. Clicking a card opens that week's info board; Back returns to the cards. It is for choosing and
// previewing a lesson, so it does not repeat the Course Map (the schedule and progress view).
const TERMS = ['All', 'Prelim', 'Midterm', 'Pre-final', 'Final'];
const TABS = [
  ['ideas', 'Ideas'],
  ['skills', 'Skills'],
  ['terms', 'Terms'],
  ['time', 'Time'],
];

const statusOf = (o) =>
  o.exam
    ? { key: 'exam', label: 'Exam week' }
    : o.week < CURRENT_WEEK
      ? { key: 'done', label: 'Completed' }
      : o.week === CURRENT_WEEK
        ? { key: 'now', label: 'This week' }
        : { key: 'next', label: 'Up next' };

const minutes = (m) => (m >= 60 ? Math.floor(m / 60) + ' h' + (m % 60 ? ' ' + (m % 60) + ' min' : '') : m + ' min');

// The info board for one week.
function Board({ o, onOpen }) {
  const st = statusOf(o);
  const [tab, setTab] = useState('ideas');
  useEffect(() => setTab('ideas'), [o.week]);
  const total = o.path.reduce((a, p) => a + p.min, 0);
  return (
    <div className="ll-board" key={o.week}>
      <div className="ll-board-l">
        <div className="ll-prev-top">
          <span className="ll-eye">
            Week {o.week} · {o.term}
          </span>
          {st.key === 'now' || st.key === 'exam' ? <span className={'ll-chip ' + st.key}>{st.label}</span> : null}
        </div>
        <h3 className="ll-prev-t">{o.title}</h3>
        <p className="ll-prev-lede">{o.tagline}</p>
        {o.graphic ? (
          <div className="ll-prev-fig">
            <Infographic spec={o.graphic} />
          </div>
        ) : null}
      </div>
      <div className="ll-board-r">
        <div className="ll-tabs" role="tablist" aria-label="More about this lesson">
          {TABS.map(([k, l]) => (
            <button key={k} type="button" role="tab" aria-selected={tab === k} className={'ll-tab' + (tab === k ? ' on' : '')} onClick={() => setTab(k)}>
              {l}
            </button>
          ))}
        </div>
        <div className="ll-panel" role="tabpanel">
          {tab === 'ideas' && (
            <ul className="ll-ideas">
              {o.ideas.map((x, i) => (
                <li key={i}>{x}</li>
              ))}
            </ul>
          )}
          {tab === 'skills' && (
            <ol className="ll-outs">
              {o.outcomes.map((x, i) => (
                <li key={i}>{x}</li>
              ))}
            </ol>
          )}
          {tab === 'terms' && (
            <dl className="ll-terms">
              {o.terms.map((x, i) => (
                <div key={i}>
                  <dt>{x.t}</dt>
                  <dd>{x.d}</dd>
                </div>
              ))}
            </dl>
          )}
          {tab === 'time' && (
            <>
              <p className="ll-total">
                About <b>{minutes(total)}</b>
              </p>
              <ol className="ll-path" aria-label="Study path">
                {o.path.map((p, i) => (
                  <li key={i} style={{ flexGrow: Math.max(1, Math.round(Math.sqrt(p.min))) }}>
                    <span className="ll-seg" />
                    <b>{p.label}</b>
                    <span>{minutes(p.min)}</span>
                  </li>
                ))}
              </ol>
              {o.lab || o.kc ? <p className="ll-time">{[o.lab, o.kc ? o.kc + ' quiz questions' : ''].filter(Boolean).join(' · ')}</p> : null}
            </>
          )}
        </div>
        <div className="ll-actions">
          <DcLink className="ll-go" href={'/lesson/' + o.week} onClick={onOpen}>
            {o.exam ? 'Open revision lesson' : 'Open lesson'} <span aria-hidden="true">→</span>
          </DcLink>
          <DcLink className="ll-go ghost" href={'/prompt-workshop/' + o.week} onClick={onOpen}>
            Workshop
          </DcLink>
        </div>
      </div>
    </div>
  );
}

export default function LessonLibrary({ initialWeek, onClose }) {
  const [view, setView] = useState('grid'); // 'grid' = all the cards, 'board' = one week's info board
  const [q, setQ] = useState('');
  const [term, setTerm] = useState('All');
  const [sel, setSel] = useState(initialWeek || CURRENT_WEEK);
  const focusRef = useRef(null);
  const bodyRef = useRef(null);

  const all = useMemo(() => Array.from({ length: TOTAL_WEEKS }, (_, i) => overviewOf(i + 1)), []);
  const shown = useMemo(() => {
    const t = q.trim().toLowerCase();
    return all.filter(
      (o) =>
        (term === 'All' || o.term === term) &&
        (!t || [o.title, o.tagline, 'week ' + o.week, ...o.terms.map((x) => x.t)].join(' ').toLowerCase().includes(t)),
    );
  }, [all, q, term]);
  const groups = ['Prelim', 'Midterm', 'Pre-final', 'Final'].map((t) => ({ t, list: shown.filter((o) => o.term === t) })).filter((g) => g.list.length);
  const cur = all.find((o) => o.week === sel) || all[CURRENT_WEEK - 1];

  // Esc: from an info board go back to the cards; from the cards close the sheet.
  useEffect(() => {
    const onKey = (e) => {
      if (e.key !== 'Escape') return;
      if (view === 'board') setView('grid');
      else onClose();
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [view, onClose]);
  useEffect(() => {
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      document.body.style.overflow = prev;
    };
  }, []);
  // Move focus and scroll to the right place whenever the screen changes.
  useEffect(() => {
    try {
      if (view === 'grid') {
        const c = document.querySelector('.ll-card.on');
        if (c) c.scrollIntoView({ block: 'center' });
        if (c) c.focus({ preventScroll: true });
      } else {
        if (bodyRef.current) bodyRef.current.scrollTop = 0;
        if (focusRef.current) focusRef.current.focus();
      }
    } catch (e) {}
  }, [view]);

  const open = (n) => {
    setSel(n);
    setView('board');
  };
  const step = (d) => setSel((n) => Math.min(TOTAL_WEEKS, Math.max(1, n + d)));

  return (
    <div className="ll-scrim" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="ll" role="dialog" aria-modal="true" aria-labelledby="ll-title">
        {view === 'grid' ? (
          <>
            <header className="ll-head">
              <div>
                <div className="ll-eye">Lesson library</div>
                <h2 id="ll-title" className="ll-title">
                  Choose a weekly lesson
                </h2>
                <p className="ll-lede">Click a week to see what it covers.</p>
              </div>
              <button type="button" className="ll-close" onClick={onClose} aria-label="Close the lesson library">
                <span aria-hidden="true">✕</span> Close
              </button>
            </header>
            <div className="ll-tools">
              <input className="ll-search" type="search" placeholder="Search lessons or terms" aria-label="Search lessons" value={q} onChange={(e) => setQ(e.target.value)} />
              <div className="ll-filters" role="group" aria-label="Filter by term">
                {TERMS.map((t) => (
                  <button key={t} type="button" className={'ll-f' + (term === t ? ' on' : '')} aria-pressed={term === t} onClick={() => setTerm(t)}>
                    {t}
                  </button>
                ))}
              </div>
            </div>
            <div className="ll-body ll-cards-view" ref={bodyRef}>
              {groups.map((g) => (
                <section key={g.t} aria-label={g.t + ' term'}>
                  <h3 className="ll-term">
                    {g.t}{' '}
                    <span>
                      Weeks {g.list[0].week} to {g.list[g.list.length - 1].week}
                    </span>
                  </h3>
                  <div className="ll-grid">
                    {g.list.map((o) => {
                      const st = statusOf(o);
                      return (
                        <button
                          key={o.week}
                          type="button"
                          data-week={o.week}
                          className={'ll-card ' + st.key + (sel === o.week ? ' on' : '')}
                          onClick={() => open(o.week)}
                          aria-label={'Week ' + o.week + ': ' + o.short + (st.key === 'done' ? ', completed' : st.key === 'now' ? ', this week' : st.key === 'exam' ? ', exam week' : '')}
                        >
                          <span className="ll-card-top">
                            <span className="ll-card-n">{o.week}</span>
                            {st.key === 'done' ? (
                              <span className="ll-tick" aria-hidden="true">
                                ✓
                              </span>
                            ) : null}
                            {st.key === 'now' || st.key === 'exam' ? <span className={'ll-chip ' + st.key}>{st.label}</span> : null}
                          </span>
                          {o.graphic ? <Infographic spec={o.graphic} compact className="ll-thumb" /> : <span className="ll-thumb ll-nothumb" />}
                          <span className="ll-card-t">{o.short}</span>
                        </button>
                      );
                    })}
                  </div>
                </section>
              ))}
              {!groups.length ? <p className="ll-empty">No lessons match your search.</p> : null}
            </div>
          </>
        ) : (
          <>
            <header className="ll-head ll-head-board">
              <button type="button" className="ll-back" onClick={() => setView('grid')} ref={focusRef}>
                <span aria-hidden="true">←</span> All lessons
              </button>
              <span id="ll-title" className="ll-lede ll-crumb">
                Week {cur.week} of {TOTAL_WEEKS}
              </span>
              <span className="ll-stepper">
                <button type="button" className="ll-step" onClick={() => step(-1)} disabled={sel <= 1} aria-label="Previous week">
                  ‹
                </button>
                <button type="button" className="ll-step" onClick={() => step(1)} disabled={sel >= TOTAL_WEEKS} aria-label="Next week">
                  ›
                </button>
              </span>
              <button type="button" className="ll-close" onClick={onClose} aria-label="Close the lesson library">
                <span aria-hidden="true">✕</span> Close
              </button>
            </header>
            <div className="ll-body ll-board-view" ref={bodyRef}>
              <Board o={cur} onOpen={onClose} />
            </div>
          </>
        )}
      </div>
    </div>
  );
}
