import React, { useEffect } from 'react';
import { listAnnouncements } from './api.js';
import { Shell, PageHeader, Time, Markdown, LoadState, SampleBanner, useLoad } from './Shell.jsx';
import { plural } from './format.js';

export default function Announcements() {
  const [state, reload] = useLoad(listAnnouncements);
  useEffect(() => {
    document.title = 'Announcements - AI Academy';
  }, []);
  const now = Date.now();
  const items = state.data ? [...state.data.items].sort((a, b) => Date.parse(b.publishAt) - Date.parse(a.publishAt)) : [];
  return (
    <Shell tone="cyan">
      <section aria-labelledby="lms-title">
        <PageHeader
          eyebrow="From your teacher"
          title="Announcements"
          intro="Messages from your teacher, newest first."
          chips={[state.data && plural(items.length, 'announcement', 'announcements')]}
          emblem="megaphone"
        />
        <p className="lms-sr" role="status" aria-live="polite" aria-atomic="true">
          {state.status === 'ready' ? (items.length ? plural(items.length, 'announcement', 'announcements') + '.' : 'No announcements yet.') : 'Loading announcements.'}
        </p>
        {state.data && state.data.sample && <SampleBanner />}
        <LoadState state={state} what="announcements" onRetry={reload} />
        {state.data && items.length === 0 && (
          <div className="card lms-state">
            <p>No announcements yet. Your teacher will publish them here.</p>
          </div>
        )}
        {items.length > 0 && (
          <ul className="lms-list ann-list" aria-label="Announcements">
            {items.map((a) => (
              <li key={a.id} className="card lms-item ann-item">
                <h2 className="cm-h3">{a.title}</h2>
                <p className="lms-meta">
                  Published <Time value={a.publishAt} now={now} />
                </p>
                <Markdown text={a.bodyMd} />
              </li>
            ))}
          </ul>
        )}
      </section>
    </Shell>
  );
}
