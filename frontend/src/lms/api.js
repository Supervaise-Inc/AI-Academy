// Data for the student Activities / Announcements / My grades pages.
//
// Reads the same endpoints the deployed Academy build uses:
//   GET /api/v1/me/activities       -> { items: [...] }
//   GET /api/v1/me/grades           -> { items: [...], next_after }
//   GET /api/v1/me/announcements    -> { items: [...] }
// If this server does not have those routes yet (404/501), the pages fall back to the bundled sample data in
// sample.js and show a "sample data" banner, so the UI can be built and reviewed before the backend exists.
import { api } from '../dc/session.js';
import { SAMPLE_ACTIVITIES, SAMPLE_GRADES, SAMPLE_ANNOUNCEMENTS } from './sample.js';

const KINDS = ['notebook', 'quiz', 'capstone_milestone', 'offline'];
const num = (v) => (v === null || v === undefined || v === '' ? null : Number(v));
const text = (v, max) => (typeof v === 'string' ? v.slice(0, max) : '');

function stateOf(phase, collected) {
  if (phase === 'scheduled') return 'scheduled';
  if (phase === 'closed') return collected ? 'collected' : 'closed';
  return collected ? 'collected' : 'open';
}

function statusOf({ excused, attempts, lastLate, phase }) {
  if (excused) return 'excused';
  if (attempts > 0) return lastLate ? 'submitted_late' : 'submitted';
  return phase === 'closed' ? 'missing' : 'not_submitted';
}

function gradeOf(g) {
  if (!g || g.released_at === null || g.released_at === undefined) return null;
  return { points: num(g.points), maxPoints: num(g.max_points), releasedAt: g.released_at };
}

function activity(r) {
  const attempts = Number.isInteger(r.attempts) ? r.attempts : 0;
  const lastLate = r.last_submission_late === true;
  return {
    key: r.assessment_key,
    title: text(r.title, 200),
    kind: KINDS.includes(r.kind) ? r.kind : 'other',
    week: r.week ?? null,
    maxPoints: num(r.max_points),
    state: stateOf(r.phase, !!r.collected),
    opensAt: r.opens_at,
    dueAt: r.due_at,
    lateUntil: r.late_until || null,
    latePolicy: r.late_policy,
    override: r.excused ? { extended: false, excused: true, reason: null, dueAt: null, lateUntil: null } : null,
    status: statusOf({ excused: !!r.excused, attempts, lastLate, phase: r.phase }),
    attempts,
    receipt: r.last_submitted_at ? { receivedAt: r.last_submitted_at, isLate: lastLate } : null,
    grade: gradeOf(r.grade),
  };
}

function grade(r) {
  if (!r || r.released_at === null || r.released_at === undefined) return null;
  const rubric = (r.rubric || []).map((c) => ({
    criterionKey: c.criterion_key,
    title: text(c.criterion_name, 200) || null,
    levelKey: c.level_key,
    levelTitle: text(c.level_label, 200) || null,
    points: num(c.points),
    maxPoints: num(c.max_points),
  }));
  return {
    key: r.assessment_key,
    title: text(r.title, 200).trim() || r.assessment_key,
    kind: KINDS.includes(r.kind) ? r.kind : 'other',
    week: null,
    points: num(r.points),
    maxPoints: r.max_points == null ? rubric.reduce((s, c) => s + (c.maxPoints || 0), 0) : num(r.max_points),
    releasedAt: r.released_at,
    feedback: typeof r.feedback === 'string' && r.feedback.trim() ? r.feedback.slice(0, 16384) : null,
    rubric,
  };
}

const announcement = (r) => ({
  id: String(r.id).toLowerCase(),
  title: text(r.title, 160),
  bodyMd: text(r.body_md, 4096),
  publishAt: r.publish_at,
  expiresAt: r.expires_at || null,
});

// True when the server simply does not have these routes (so sample data is appropriate).
const routeMissing = (e) => e && (e.status === 404 || e.status === 501);

async function load(path, sample, map) {
  try {
    const res = await api(path);
    return { ...map(res), sample: false };
  } catch (e) {
    if (routeMissing(e)) return { ...sample(), sample: true };
    throw e;
  }
}

export const listActivities = () =>
  load(
    '/v1/me/activities',
    () => ({ items: SAMPLE_ACTIVITIES() }),
    (r) => ({ items: (r.items || []).map(activity) }),
  );

export const listAnnouncements = () =>
  load(
    '/v1/me/announcements',
    () => ({ items: SAMPLE_ANNOUNCEMENTS() }),
    (r) => ({ items: (r.items || []).map(announcement) }),
  );

export const listGrades = () =>
  load(
    '/v1/me/grades',
    () => ({ items: SAMPLE_GRADES() }),
    (r) => {
      const items = (r.items || []).map(grade).filter(Boolean);
      items.sort((a, b) => Date.parse(b.releasedAt) - Date.parse(a.releasedAt) || (a.key < b.key ? -1 : 1));
      return { items };
    },
  );
