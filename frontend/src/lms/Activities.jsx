import React, { useEffect, useRef, useState } from 'react';
import { listActivities } from './api.js';
import { Shell, PageHeader, Time, LoadState, SampleBanner, useLoad } from './Shell.jsx';
import { FILTERS, applyFilter, counts, notes, percent, points, relative, sortActivities, statusView, summaryLine, typeLine } from './format.js';

function ActivityCard({ a, now }) {
  const view = statusView(a, now);
  const upcomingOrDue = ['todo', 'overdue', 'upcoming'].includes(view.key);
  return (
    <li className="card lms-item">
      <div className="lms-item-head">
        <h2 className="cm-h3">{a.title}</h2>
        <span className="badge" data-tone={view.tone}>
          {view.label}
        </span>
      </div>
      <ul className="lms-meta">
        <li>{typeLine(a)}</li>
        <li>
          Due <Time value={a.dueAt} now={now} />
          {upcomingOrDue && <span> ({relative(a.dueAt, now)})</span>}
        </li>
        <li>
          {a.attempts > 0 ? `You have ${a.attempts} ${a.attempts === 1 ? 'submission' : 'submissions'}.` : 'You have not submitted yet.'}
          {a.attempts > 0 && a.receipt && (
            <>
              {' '}
              Last: <Time value={a.receipt.receivedAt} now={now} />
            </>
          )}
        </li>
        <li>Worth {points(a.maxPoints)} points</li>
        {a.grade && (
          <li className="lms-grade">
            Your grade: {points(a.grade.points)} of {points(a.grade.maxPoints)} points ({percent(a.grade.points, a.grade.maxPoints)})
          </li>
        )}
      </ul>
      {notes(a, now).length > 0 && (
        <ul className="lms-notes">
          {notes(a, now).map((n) => (
            <li key={n.id} className="callout lms-note" data-tone={n.tone}>
              {n.text}
            </li>
          ))}
        </ul>
      )}
    </li>
  );
}

export default function Activities() {
  const [state, reload] = useLoad(listActivities);
  const [filter, setFilter] = useState('all');
  const heading = useRef(null);
  useEffect(() => {
    document.title = 'Activities - AI Academy';
  }, []);
  const now = Date.now();
  const items = state.data ? state.data.items : [];
  const shown = sortActivities(applyFilter(items, filter, now), now);
  const c = state.data ? counts(items, now) : null;
  return (
    <Shell tone="indigo">
      <section aria-labelledby="lms-title">
        <PageHeader
          eyebrow="Your activities"
          title="Activities"
          intro="Everything you need to do, with due dates and what we have received from you."
          highlight="due dates"
          chips={[...(c ? [`${c.open} open`, `${c.late} late`, `${c.graded} graded`] : []), 'All times are Philippine time (PHT)']}
          emblem="checklist"
          headingRef={heading}
        />
        <p className="lms-sr" role="status" aria-live="polite" aria-atomic="true">
          {state.status === 'ready' ? summaryLine(items, now) : 'Loading activities.'}
        </p>
        {state.data && state.data.sample && <SampleBanner />}
        <LoadState state={state} what="activities" onRetry={reload} />
        {state.data && items.length === 0 && (
          <div className="card lms-state">
            <p>No activities yet. Your teacher will add them here.</p>
          </div>
        )}
        {state.data && items.length > 0 && (
          <>
            <div className="lms-filters" role="group" aria-label="Show activities">
              {FILTERS.map((f) => (
                <button key={f.id} type="button" className="chipbtn" aria-pressed={filter === f.id ? 'true' : 'false'} onClick={() => setFilter(f.id)}>
                  {f.label}
                </button>
              ))}
            </div>
            {shown.length === 0 ? (
              <div className="card lms-state">
                <p>{filter === 'todo' ? 'Nothing to do right now. Good job.' : 'You have not submitted any work yet.'}</p>
              </div>
            ) : (
              <ul className="lms-list" aria-label="Activities">
                {shown.map((a) => (
                  <ActivityCard key={a.key} a={a} now={now} />
                ))}
              </ul>
            )}
          </>
        )}
      </section>
    </Shell>
  );
}
