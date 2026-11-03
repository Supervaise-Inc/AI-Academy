// Date, status and wording helpers for the student Activities / Announcements / My grades pages.
// Rebuilt from the behaviour of the deployed Academy build. All times are shown in Philippine time.
const ZONE = 'Asia/Manila';
const DAY = 864e5;
const CLOSED_STATES = ['closed', 'collected', 'exported', 'purge_pending', 'purged'];

export const KIND_LABEL = {
  notebook: 'Lab',
  quiz: 'Quiz',
  capstone_milestone: 'Capstone step',
  offline: 'Work on paper',
  other: 'Activity',
};
const EXCUSE_REASON = {
  illness: 'being sick',
  school_event: 'a school event',
  technical: 'a computer problem',
  other: 'another reason',
};

const ms = (iso) => Date.parse(iso);
const yearOf = (d) => new Intl.DateTimeFormat('en-US', { timeZone: ZONE, year: 'numeric' }).format(d);

function parts(date, opts) {
  const out = {};
  for (const p of new Intl.DateTimeFormat('en-US', { timeZone: ZONE, ...opts }).formatToParts(date)) out[p.type] = p.value;
  return out;
}

// "Sun 1 Nov, 8:01 PM PHT"
export function formatDateTime(iso, now = Date.now(), { zone = true } = {}) {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '';
  const p = parts(d, { hour12: true, weekday: 'short', day: 'numeric', month: 'short', hour: 'numeric', minute: '2-digit' });
  const year = yearOf(d) === yearOf(new Date(now)) ? '' : ` ${yearOf(d)}`;
  return `${p.weekday} ${p.day} ${p.month}${year}, ${p.hour}:${p.minute} ${String(p.dayPeriod || '').toUpperCase()}${zone ? ' PHT' : ''}`;
}

// "Sun 1 Nov"
export function formatDate(iso, now = Date.now()) {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '';
  const p = parts(d, { weekday: 'short', day: 'numeric', month: 'short' });
  const year = yearOf(d) === yearOf(new Date(now)) ? '' : ` ${yearOf(d)}`;
  return `${p.weekday} ${p.day} ${p.month}${year}`;
}

// "in 2 days" / "3 hours ago"
export function relative(iso, now = Date.now()) {
  const diff = ms(iso) - now;
  if (!Number.isFinite(diff)) return '';
  const abs = Math.abs(diff);
  if (abs < 6e4) return diff >= 0 ? 'in under a minute' : 'under a minute ago';
  let n;
  let unit;
  if (abs < 36e5) [n, unit] = [Math.round(abs / 6e4), 'minute'];
  else if (abs < DAY) [n, unit] = [Math.round(abs / 36e5), 'hour'];
  else [n, unit] = [Math.round(abs / DAY), 'day'];
  const text = `${n} ${unit}${n === 1 ? '' : 's'}`;
  return diff >= 0 ? `in ${text}` : `${text} ago`;
}

export const points = (n) => (Number.isFinite(n) ? String(Math.round(n * 100) / 100) : '');
export const percent = (n, max) => (!Number.isFinite(n) || !Number.isFinite(max) || max <= 0 ? '' : `${Math.round((n / max) * 100)}%`);
export const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;
export const humanKey = (k) => {
  const t = String(k || '').replace(/[-_]+/g, ' ').trim();
  return t ? t.charAt(0).toUpperCase() + t.slice(1) : '';
};
export const typeLine = (a) => [a.week ? `Week ${a.week}` : null, KIND_LABEL[a.kind] || KIND_LABEL.other].filter(Boolean).join(' · ');

const isSubmitted = (a) => a.status === 'submitted' || a.status === 'submitted_late';

// 'upcoming' | 'open' | 'late_window' | 'closed'
export function phase(a, now) {
  if (CLOSED_STATES.includes(a.state)) return 'closed';
  if (now < ms(a.opensAt)) return 'upcoming';
  if (now <= ms(a.dueAt)) return 'open';
  if (a.latePolicy === 'accept_flag' && a.lateUntil && now <= ms(a.lateUntil)) return 'late_window';
  return 'closed';
}

// The badge shown on a card.
export function statusView(a, now) {
  if (a.status === 'excused' || (a.override && a.override.excused)) return { key: 'excused', label: 'Excused', tone: 'info' };
  if (a.status === 'submitted') return { key: 'submitted', label: 'Submitted', tone: 'ok' };
  if (a.status === 'submitted_late') return { key: 'submitted_late', label: 'Submitted late', tone: 'warn' };
  if (a.status === 'missing') return { key: 'missing', label: 'Closed, not submitted', tone: 'bad' };
  const p = phase(a, now);
  if (p === 'closed') return { key: 'missing', label: 'Closed, not submitted', tone: 'bad' };
  if (p === 'late_window') return { key: 'overdue', label: 'Late: you can still submit', tone: 'warn' };
  if (p === 'upcoming') return { key: 'upcoming', label: `Opens ${formatDate(a.opensAt, now)}`, tone: 'neutral' };
  return { key: 'todo', label: 'To do', tone: 'neutral' };
}

// The coloured notes under a card.
export function notes(a, now) {
  const out = [];
  const p = phase(a, now);
  const o = a.override;
  const when = (iso) => formatDateTime(iso, now);
  if (o && o.excused) {
    out.push({
      id: 'excused',
      tone: 'info',
      text: `Your teacher excused you from this activity${o.reason ? ` because of ${EXCUSE_REASON[o.reason]}` : ''}. You do not need to submit anything.`,
    });
    return out;
  }
  if (o && (o.extended || o.dueAt || o.lateUntil))
    out.push({ id: 'override', tone: 'info', text: `Your teacher gave you more time. Your new due date is ${when(a.dueAt)}.` });
  if (p === 'upcoming') out.push({ id: 'upcoming', tone: 'neutral', text: `This activity opens ${when(a.opensAt)}. You cannot submit before then.` });
  if (p === 'late_window' && !isSubmitted(a))
    out.push({
      id: 'late-window',
      tone: 'warn',
      text: `The due date has passed. You can still submit work until ${when(a.lateUntil)}. It will be counted as late.`,
    });
  if (a.status === 'submitted_late')
    out.push({ id: 'late', tone: 'warn', text: 'We received your work after the due date. It is counted as late.' });
  if (p === 'closed' && !isSubmitted(a))
    out.push({
      id: 'closed',
      tone: 'bad',
      text: 'This activity is closed. We did not receive any work from you. Ask your teacher if you need help.',
    });
  else if (p === 'closed' && isSubmitted(a) && a.state !== 'closed' && a.state !== 'purged')
    out.push({ id: 'collected', tone: 'neutral', text: 'This activity is closed. Your teacher is grading your work.' });
  return out;
}

export const FILTERS = [
  { id: 'all', label: 'All' },
  { id: 'todo', label: 'To do' },
  { id: 'done', label: 'Submitted' },
];

export function applyFilter(items, filter, now) {
  if (filter === 'done') return items.filter(isSubmitted);
  if (filter === 'todo') return items.filter((a) => !isSubmitted(a) && statusView(a, now).key !== 'excused' && phase(a, now) !== 'closed');
  return items;
}

// Work still to do first (soonest due first), then everything else (most recent first).
export function sortActivities(items, now) {
  const pending = (a) => !isSubmitted(a) && phase(a, now) !== 'closed' && !(a.override && a.override.excused);
  return [...items].sort((a, b) => {
    const pa = pending(a);
    const pb = pending(b);
    if (pa !== pb) return pa ? -1 : 1;
    const d = ms(a.dueAt) - ms(b.dueAt);
    if (d !== 0) return pa ? d : -d;
    return a.key < b.key ? -1 : a.key > b.key ? 1 : 0;
  });
}

export function counts(items, now) {
  let open = 0;
  let late = 0;
  let graded = 0;
  for (const a of items || []) {
    const k = statusView(a, now).key;
    if (k === 'todo') open += 1;
    if (k === 'overdue' || k === 'submitted_late') late += 1;
    if (a.grade) graded += 1;
  }
  return { open, late, graded };
}

export const summaryLine = (items, now) =>
  items.length ? `${plural(items.length, 'activity', 'activities')}. ${applyFilter(items, 'todo', now).length} to do.` : 'No activities yet.';

// Which hero emblem to show for one activity.
export function emblemFor(a, view) {
  if (a && a.grade) return { emblem: 'graded', caption: 'Graded' };
  const k = view ? view.key : 'todo';
  if (k === 'excused' || k === 'submitted') return { emblem: 'done', caption: view.label };
  if (k === 'submitted_late') return { emblem: 'late', caption: 'Submitted late' };
  if (k === 'overdue') return { emblem: 'late', caption: 'Late' };
  if (k === 'missing') return { emblem: 'late', caption: 'Closed, not submitted' };
  if (k === 'upcoming') return { emblem: 'open', caption: 'Not open yet' };
  return { emblem: 'open', caption: 'To do' };
}
