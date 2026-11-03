import React, { useEffect, useMemo, useState } from 'react';
import { SECTIONS, UNASSIGNED, STATUS_RANK, lastActive, sectionName } from './educatorRoster.js';

// Section picker, section comparison and the learner monitor for the Faculty Console (demo data).

const PAGE = 20;
const STATUS = {
  on: ['On track', 'var(--accent-soft-bg)', 'var(--accent-soft-text)', 'var(--accent-soft-border)'],
  watch: ['Watch', 'var(--amber-soft-bg)', 'var(--amber-text)', 'var(--amber-soft-border)'],
  support: ['Needs support', 'var(--red-soft-bg)', 'var(--red-text)', 'var(--red-border)'],
};
const chipStyle = (on) =>
  on
    ? { background: 'var(--accent-soft-bg)', color: 'var(--accent-soft-text)', border: '1px solid var(--accent-soft-border)' }
    : { background: 'var(--surface2)', color: 'var(--text2)', border: '1px solid var(--surface2-border)' };

export function Badge({ status }) {
  const c = STATUS[status];
  return (
    <span className="badge" style={{ background: c[1], color: c[2], border: '1px solid ' + c[3] }}>
      {c[0]}
    </span>
  );
}

const Avatar = ({ ini, size = 28 }) => (
  <span
    style={{
      width: size,
      height: size,
      flexShrink: 0,
      borderRadius: 999,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontSize: 13,
      fontWeight: 600,
      background: 'var(--surface2)',
      color: 'var(--text2)',
      border: '1px solid var(--surface2-border)',
    }}
  >
    {ini}
  </span>
);

// "All sections | A 28 | B 31 | ... | Unassigned 6". Wraps onto more lines when there are many sections.
export function SectionBar({ all, value, onPick }) {
  const items = [{ id: 'all', label: 'All sections', list: all }].concat(
    SECTIONS.map((s) => ({ id: s.id, label: s.name, list: all.filter((x) => x.section === s.id) })),
    [{ id: UNASSIGNED, label: 'Unassigned', list: all.filter((x) => x.section === UNASSIGNED) }],
  );
  return (
    <div className="card mon-secbar" role="group" aria-label="Choose a section">
      <span className="mon-k">Section</span>
      <div className="mon-chips">
        {items.map((it) => {
          const flagged = it.list.filter((x) => x.status !== 'on').length;
          return (
            <button key={it.id} type="button" className="chipbtn mon-sec" style={chipStyle(value === it.id)} onClick={() => onPick(it.id)} aria-pressed={value === it.id}>
              {it.label}
              <span className="mono mon-n">{it.list.length}</span>
              {flagged > 0 && it.id !== 'all' && <span className="mon-flag" title={flagged + ' learners flagged'} />}
            </button>
          );
        })}
      </div>
    </div>
  );
}

// One row per section: how each is doing at a glance. Click a row to open that section.
export function SectionCompare({ all, gradedOf, value, onPick }) {
  const rows = SECTIONS.map((s) => {
    const l = all.filter((x) => x.section === s.id);
    const n = Math.max(1, l.length);
    return {
      id: s.id,
      name: s.name,
      n: l.length,
      prog: Math.round(l.reduce((a, x) => a + x.prog, 0) / n),
      on: Math.round((100 * l.filter((x) => x.status === 'on').length) / n),
      support: l.filter((x) => x.status === 'support').length,
      toGrade: l.filter((x) => x.sub && !gradedOf(x)).length,
    };
  });
  const cols = 'minmax(0,1.1fr) 70px minmax(0,1.2fr) 80px 90px 100px';
  return (
    <div className="card" style={{ padding: 22 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <div style={{ fontSize: 16, fontWeight: 600 }}>Sections at a glance</div>
        <span className="demo">Demo data</span>
      </div>
      <div className="mon-scroll">
        <div className="row mon-head" style={{ gridTemplateColumns: cols, minWidth: 560 }}>
          <span>Section</span>
          <span>Learners</span>
          <span>Course completion</span>
          <span>On track</span>
          <span>Need support</span>
          <span style={{ textAlign: 'right' }}>To grade</span>
        </div>
        {rows.map((r) => (
          <button key={r.id} type="button" className={'row mon-rowbtn' + (value === r.id ? ' on' : '')} style={{ gridTemplateColumns: cols, minWidth: 560 }} onClick={() => onPick(r.id)}>
            <span style={{ fontWeight: 600 }}>{r.name}</span>
            <span className="mono">{r.n}</span>
            <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span className="bar" style={{ flex: 1 }}>
                <span className="barfill" style={{ display: 'block', width: r.prog + '%', background: 'var(--accent)' }} />
              </span>
              <span className="mono" style={{ fontSize: 13, color: 'var(--text3)', width: 34 }}>
                {r.prog}%
              </span>
            </span>
            <span className="mono">{r.on}%</span>
            <span className="mono" style={{ color: r.support ? 'var(--red-text)' : 'var(--text2)' }}>
              {r.support}
            </span>
            <span className="mono" style={{ textAlign: 'right', color: r.toGrade ? 'var(--amber-text)' : 'var(--text3)' }}>
              {r.toGrade}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}

const SORTS = {
  name: (a, b) => a.name.localeCompare(b.name),
  prog: (a, b) => a.prog - b.prog,
  active: (a, b) => a.active - b.active,
  status: (a, b) => STATUS_RANK[a.status] - STATUS_RANK[b.status] || a.name.localeCompare(b.name),
};

function csvOf(list) {
  const q = (v) => '"' + String(v == null ? '' : v).replace(/"/g, '""') + '"';
  const head = ['Name', 'Section', 'Completion %', 'Labs', 'Checks', 'Preliminary', 'Last active', 'Status', 'Note'];
  const rows = list.map((s) => [s.name, sectionName(s.section), s.prog, s.labs, s.kc, s.prelim, lastActive(s.active), STATUS[s.status][0], s.flag || '']);
  return [head].concat(rows).map((r) => r.map(q).join(',')).join('\n');
}

function exportCsv(list, label) {
  try {
    const url = URL.createObjectURL(new Blob([csvOf(list)], { type: 'text/csv;charset=utf-8' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = 'learners-' + label.toLowerCase().replace(/[^a-z0-9]+/g, '-') + '.csv';
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (e) {}
}

// The learner table: search, status filter, sorting, paging, multi-select with "move to section", per-learner detail.
export function Monitor({ students, label, showSection, onMove, onGrade, gradedOf }) {
  const [q, setQ] = useState('');
  const [fil, setFil] = useState('all');
  const [sort, setSort] = useState('status');
  const [dir, setDir] = useState(1);
  const [page, setPage] = useState(0);
  const [picked, setPicked] = useState({});
  const [open, setOpen] = useState('');
  const [target, setTarget] = useState('');

  // New section or new filter: start from the first page and drop selections that are no longer in the list.
  useEffect(() => setPage(0), [q, fil, sort, dir, label]);
  useEffect(() => {
    setPicked({});
    setOpen('');
  }, [label]);

  const counts = useMemo(
    () => ({
      all: students.length,
      support: students.filter((s) => s.status === 'support').length,
      watch: students.filter((s) => s.status === 'watch').length,
      on: students.filter((s) => s.status === 'on').length,
      quiet: students.filter((s) => s.active >= 7).length,
    }),
    [students],
  );
  const list = useMemo(() => {
    const t = q.trim().toLowerCase();
    const f = students.filter(
      (s) =>
        (!t || s.name.toLowerCase().includes(t) || sectionName(s.section).toLowerCase().includes(t)) &&
        (fil === 'all' || (fil === 'quiet' ? s.active >= 7 : s.status === fil)),
    );
    return f.slice().sort((a, b) => dir * SORTS[sort](a, b));
  }, [students, q, fil, sort, dir]);

  const pages = Math.max(1, Math.ceil(list.length / PAGE));
  const cur = Math.min(page, pages - 1);
  const shown = list.slice(cur * PAGE, cur * PAGE + PAGE);
  const pickedIds = Object.keys(picked).filter((id) => picked[id]);
  const pageAll = shown.length > 0 && shown.every((s) => picked[s.id]);
  const toggle = (id) => setPicked({ ...picked, [id]: !picked[id] });
  const togglePage = () => {
    const next = { ...picked };
    shown.forEach((s) => (next[s.id] = !pageAll));
    setPicked(next);
  };
  const selectAll = () => {
    const next = {};
    list.forEach((s) => (next[s.id] = true));
    setPicked(next);
  };
  const move = (ids, section) => {
    if (!section || !ids.length) return;
    onMove(ids, section);
    setPicked({});
    setTarget('');
  };
  const head = (key, text) => (
    <button
      type="button"
      className={'mon-th' + (sort === key ? ' on' : '')}
      onClick={() => (sort === key ? setDir(-dir) : (setSort(key), setDir(1)))}
      aria-label={'Sort by ' + text}
    >
      {text}
      {sort === key ? (dir === 1 ? ' ↑' : ' ↓') : ''}
    </button>
  );
  const cols = (showSection ? '28px ' : '28px ') + 'minmax(0,1.5fr) minmax(0,1.1fr) 56px 56px 80px 96px 112px';
  const filters = [
    ['all', 'All', counts.all],
    ['support', 'Needs support', counts.support],
    ['watch', 'Watch', counts.watch],
    ['on', 'On track', counts.on],
    ['quiet', 'Inactive 7+ days', counts.quiet],
  ];
  const moveTargets = SECTIONS.map((s) => [s.id, s.name]).concat([[UNASSIGNED, 'Unassigned']]);

  return (
    <div className="card" style={{ padding: 22 }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'center', gap: 10, marginBottom: 12 }}>
        <div style={{ fontSize: 16, fontWeight: 600 }}>
          Learner monitor <span style={{ fontWeight: 500, color: 'var(--text3)', fontSize: 14 }}>· {label}</span>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <button type="button" className="ghostbtn" style={{ padding: '6px 12px', fontSize: 13 }} onClick={() => exportCsv(list, label)} disabled={!list.length}>
            Export CSV
          </button>
          <span className="demo">Demo data</span>
        </div>
      </div>
      <div className="mon-tools">
        <input className="mon-search" type="search" placeholder="Search learners" aria-label="Search learners" value={q} onChange={(e) => setQ(e.target.value)} />
        <div className="mon-chips">
          {filters.map(([k, l, n]) => (
            <button key={k} type="button" className="chipbtn" style={chipStyle(fil === k)} onClick={() => setFil(k)} aria-pressed={fil === k}>
              {l} <span className="mono mon-n">{n}</span>
            </button>
          ))}
        </div>
      </div>

      {pickedIds.length > 0 && (
        <div className="mon-bulk" role="region" aria-label="Selected learners">
          <b>{pickedIds.length}</b> selected
          {pageAll && list.length > shown.length && pickedIds.length < list.length && (
            <button type="button" className="mon-link" onClick={selectAll}>
              Select all {list.length} matching
            </button>
          )}
          <span className="mon-gap" />
          <label className="mon-lab" htmlFor="mon-move">
            Move to
          </label>
          <select id="mon-move" className="mon-select" value={target} onChange={(e) => setTarget(e.target.value)}>
            <option value="">Choose section</option>
            {moveTargets.map(([id, n]) => (
              <option key={id} value={id}>
                {n}
              </option>
            ))}
          </select>
          <button type="button" className="solidbtn" style={{ padding: '7px 14px', fontSize: 13 }} disabled={!target} onClick={() => move(pickedIds, target)}>
            Apply
          </button>
          <button type="button" className="mon-link" onClick={() => setPicked({})}>
            Clear
          </button>
        </div>
      )}

      <div className="mon-scroll">
        <div className="row mon-head" style={{ gridTemplateColumns: cols, minWidth: 820 }}>
          <input type="checkbox" aria-label="Select this page" checked={pageAll} onChange={togglePage} />
          {head('name', 'Learner')}
          {head('prog', 'Course completion')}
          <span>Labs</span>
          <span>Checks</span>
          <span>Preliminary</span>
          {head('active', 'Last active')}
          <span style={{ textAlign: 'right' }}>{head('status', 'Status')}</span>
        </div>
        {shown.map((s) => (
          <React.Fragment key={s.id}>
            <div className={'row mon-row' + (open === s.id ? ' open' : '')} style={{ gridTemplateColumns: cols, minWidth: 820 }}>
              <input type="checkbox" aria-label={'Select ' + s.name} checked={!!picked[s.id]} onChange={() => toggle(s.id)} />
              <button type="button" className="mon-name" onClick={() => setOpen(open === s.id ? '' : s.id)} aria-expanded={open === s.id}>
                <Avatar ini={s.ini} />
                <span className="mon-nm">{s.name}</span>
                {showSection && <span className="mon-tag">{s.section === UNASSIGNED ? 'No section' : s.section}</span>}
              </button>
              <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span className="bar" style={{ flex: 1 }}>
                  <span className="barfill" style={{ display: 'block', width: s.prog + '%', background: 'var(--accent)' }} />
                </span>
                <span className="mono" style={{ fontSize: 13, color: 'var(--text3)', width: 32 }}>
                  {s.prog}%
                </span>
              </span>
              <span className="mono" style={{ fontSize: 14 }}>
                {s.labs}
              </span>
              <span className="mono" style={{ fontSize: 14 }}>
                {s.kc}
              </span>
              <span style={{ fontSize: 14, fontWeight: 500, color: s.prelim === 'Passed' ? 'var(--accent-soft-text)' : 'var(--amber-text)' }}>{s.prelim}</span>
              <span style={{ fontSize: 13, color: s.active >= 7 ? 'var(--amber-text)' : 'var(--text3)' }}>{lastActive(s.active)}</span>
              <span style={{ textAlign: 'right' }}>
                <Badge status={s.status} />
              </span>
            </div>
            {open === s.id && (
              <div className="mon-detail" style={{ minWidth: 820 }}>
                <dl className="mon-dl">
                  <div>
                    <dt>Section</dt>
                    <dd>{sectionName(s.section)}</dd>
                  </div>
                  <div>
                    <dt>Laboratory 3 Part 1</dt>
                    <dd>{s.sub ? (gradedOf(s) ? 'Submitted and graded' : 'Submitted, awaiting grading') : 'Not submitted'}</dd>
                  </div>
                  <div>
                    <dt>Last active</dt>
                    <dd>{lastActive(s.active)}</dd>
                  </div>
                  <div>
                    <dt>Follow-up</dt>
                    <dd style={{ color: s.flag ? (s.status === 'support' ? 'var(--red-text)' : 'var(--amber-text)') : 'var(--text2)' }}>{s.flag || 'None'}</dd>
                  </div>
                </dl>
                <div className="mon-actions">
                  <button type="button" className="ghostbtn" style={{ padding: '7px 14px', fontSize: 13 }} onClick={() => onGrade(s.id)}>
                    {s.sub ? 'Open in grading' : 'Open in grading (reminder)'}
                  </button>
                  <label className="mon-lab" htmlFor={'mv-' + s.id}>
                    Section
                  </label>
                  <select id={'mv-' + s.id} className="mon-select" value={s.section} onChange={(e) => move([s.id], e.target.value)}>
                    {moveTargets.map(([id, n]) => (
                      <option key={id} value={id}>
                        {n}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            )}
          </React.Fragment>
        ))}
        {!shown.length && <div className="mon-empty">No learners match these filters.</div>}
      </div>

      <div className="mon-pager">
        <span style={{ color: 'var(--text3)', fontSize: 13 }}>
          {list.length ? 'Showing ' + (cur * PAGE + 1) + '–' + (cur * PAGE + shown.length) + ' of ' + list.length : '0 learners'}
        </span>
        <span className="mon-gap" />
        <button type="button" className="ghostbtn" style={{ padding: '6px 12px', fontSize: 13 }} disabled={cur === 0} onClick={() => setPage(cur - 1)}>
          Previous
        </button>
        <span className="mono" style={{ fontSize: 13, color: 'var(--text2)' }}>
          {cur + 1} / {pages}
        </span>
        <button type="button" className="ghostbtn" style={{ padding: '6px 12px', fontSize: 13 }} disabled={cur >= pages - 1} onClick={() => setPage(cur + 1)}>
          Next
        </button>
      </div>
    </div>
  );
}
