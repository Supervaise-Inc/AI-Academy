// Sample data used only when the server has no /v1/me/* routes. Fictional. Mirrors the pilot screenshots.
// Dates are fixed so the page always looks the same (times are UTC; the pages show Philippine time).
export const SAMPLE_ACTIVITIES = () => [
  {
    key: 'lab-0',
    title: 'Lab 0: Warm-up',
    kind: 'notebook',
    week: 1,
    maxPoints: 100,
    state: 'closed',
    opensAt: '2026-10-25T00:00:00Z',
    dueAt: '2026-11-01T12:01:00Z',
    lateUntil: null,
    latePolicy: 'refuse',
    override: null,
    status: 'submitted',
    attempts: 1,
    receipt: { receivedAt: '2026-10-31T12:01:00Z', isLate: false },
    grade: { points: 80, maxPoints: 100, releasedAt: '2026-11-02T01:00:00Z' },
  },
  {
    key: 'lab-2',
    title: 'Lab 2: Prompting basics',
    kind: 'notebook',
    week: 2,
    maxPoints: 100,
    state: 'closed',
    opensAt: '2026-10-25T00:00:00Z',
    dueAt: '2026-10-31T11:39:00Z',
    lateUntil: null,
    latePolicy: 'refuse',
    override: null,
    status: 'missing',
    attempts: 0,
    receipt: null,
    grade: null,
  },
  {
    key: 'lab-1',
    title: 'Lab 1: Tokens and text',
    kind: 'notebook',
    week: 1,
    maxPoints: 100,
    state: 'closed',
    opensAt: '2026-10-25T00:00:00Z',
    dueAt: '2026-10-29T12:41:00Z',
    lateUntil: null,
    latePolicy: 'refuse',
    override: null,
    status: 'missing',
    attempts: 0,
    receipt: null,
    grade: null,
  },
];

export const SAMPLE_GRADES = () => [
  {
    key: 'lab-0',
    title: 'Lab 0: Warm-up',
    kind: 'notebook',
    week: 1,
    points: 80,
    maxPoints: 100,
    releasedAt: '2026-11-02T01:00:00Z',
    feedback: 'Good start. Your prompts were clear. Next time, explain **why** you picked each setting.',
    rubric: [
      { criterionKey: 'setup', title: 'Setup', levelKey: 'full', levelTitle: 'Complete', points: 30, maxPoints: 30 },
      { criterionKey: 'prompts', title: 'Prompts', levelKey: 'good', levelTitle: 'Mostly clear', points: 30, maxPoints: 40 },
      { criterionKey: 'reflection', title: 'Reflection', levelKey: 'ok', levelTitle: 'Partly done', points: 20, maxPoints: 30 },
    ],
  },
];

export const SAMPLE_ANNOUNCEMENTS = () => [
  {
    id: '00000000-0000-4000-8000-000000000001',
    title: 'Welcome to the pilot',
    bodyMd: 'Thank you for testing. Please use **fictional** information only and tell us what confused you.',
    publishAt: '2026-10-26T01:00:00Z',
    expiresAt: null,
  },
];
