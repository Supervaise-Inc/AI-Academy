import React, { useEffect } from 'react';
import { Link } from 'react-router-dom';
import { listGrades } from './api.js';
import { Shell, PageHeader, Time, Markdown, LoadState, SampleBanner, useLoad } from './Shell.jsx';
import { humanKey, percent, points, typeLine } from './format.js';

function GradeCard({ g, now }) {
  return (
    <li className="card lms-item">
      <div className="lms-item-head">
        <h2 className="cm-h3">{g.title}</h2>
        <p className="lms-grade">
          <span className="lms-sr">Your grade: </span>
          {points(g.points)} of {points(g.maxPoints)} points ({percent(g.points, g.maxPoints)})
        </p>
      </div>
      <ul className="lms-meta">
        <li>{typeLine(g)}</li>
        <li>
          Grade released <Time value={g.releasedAt} now={now} />
        </li>
      </ul>
      {g.rubric.length > 0 && (
        <div className="edc-scroll" role="region" tabIndex={0} aria-label="Scores for each part">
          <table className="edc-table lms-stack">
            <caption>Scores for each part of {g.title}</caption>
            <thead>
              <tr>
                <th scope="col">Part</th>
                <th scope="col">How you did</th>
                <th scope="col" className="num">
                  Points
                </th>
              </tr>
            </thead>
            <tbody>
              {g.rubric.map((r) => (
                <tr key={r.criterionKey}>
                  <th scope="row" data-label="Part">
                    {r.title || humanKey(r.criterionKey)}
                  </th>
                  <td data-label="How you did">{r.levelTitle || humanKey(r.levelKey)}</td>
                  <td className="num" data-label="Points">
                    {points(r.points)}
                    {r.maxPoints !== null && ` of ${points(r.maxPoints)}`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {g.feedback && (
        <div>
          <h3>Your teacher's feedback</h3>
          <Markdown text={g.feedback} />
        </div>
      )}
    </li>
  );
}

export default function Grades() {
  const [state, reload] = useLoad(listGrades);
  useEffect(() => {
    document.title = 'My grades - AI Academy';
  }, []);
  const now = Date.now();
  const items = state.data ? state.data.items : [];
  return (
    <Shell tone="amber">
      <section aria-labelledby="lms-title">
        <PageHeader
          eyebrow="Your grades"
          title="My grades"
          intro="You see a grade here after your teacher releases it. You also see the scores for each part and your teacher's feedback."
          chips={[state.data && `${items.length} ready`]}
          emblem="graded"
        />
        <p className="lms-sr" role="status" aria-live="polite" aria-atomic="true">
          {state.status === 'ready' ? (items.length ? `${items.length} ${items.length === 1 ? 'grade' : 'grades'} ready.` : 'No grades yet.') : 'Loading your grades.'}
        </p>
        {state.data && state.data.sample && <SampleBanner />}
        <LoadState state={state} what="your grades" onRetry={reload} />
        {state.data && items.length === 0 && (
          <div className="card lms-state">
            <p>No grades yet. When your teacher releases one, you will see it here.</p>
            <Link className="ghostbtn" to="/activities">
              Go to Activities
            </Link>
          </div>
        )}
        {items.length > 0 && (
          <ul className="lms-list" aria-label="Your grades">
            {items.map((g) => (
              <GradeCard key={g.key} g={g} now={now} />
            ))}
          </ul>
        )}
      </section>
    </Shell>
  );
}
