// Single source for the week-by-week overview: term, learning activities, sessions and the "next action".
// The Overview and Lesson pages read the selected week from here, so adding or changing a week is a one-file edit.
// Per-user progress is not here: it belongs with the signed-in user (see backend), this file is course content.

export const CURRENT_WEEK = 7;
export const TOTAL_WEEKS = 20;

const WEEK_ONE_WEDNESDAY = Date.UTC(2026, 8, 2); // Wednesday 2 September 2026
const DAY = 86400000;
const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
const DAYS = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];

const dateOf = (week, dayOffset = 0) => new Date(WEEK_ONE_WEDNESDAY + ((week - 1) * 7 + dayOffset) * DAY);
const longDate = (d) => `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
const fullDate = (d) => `${DAYS[d.getUTCDay()]}, ${longDate(d)}`;

// lab: null means no laboratory that week. kc: number of knowledge-check questions.
const WEEKS = [
  { week: 1, term: 'Prelim', title: 'Course orientation & Intro to Generative AI', short: 'Introduction to generative AI', lab: 'GA Lab 1', kc: 19 },
  { week: 2, term: 'Prelim', title: 'Machine Learning & Neural Networks', short: 'Machine learning and neural networks', lab: 'DL Lab 1', kc: 20 },
  { week: 3, term: 'Prelim', title: 'Deep Learning & Backpropagation', short: 'Deep learning and backpropagation', lab: 'DL Lab 2B', kc: 20 },
  { week: 4, term: 'Prelim', title: 'Convolutional Neural Networks', short: 'Convolutional neural networks', lab: 'DL Lab 4B, Part 1', kc: 20 },
  { week: 5, term: 'Prelim', title: 'Prelim examination', short: 'Prelim examination', exam: true, covers: 'Weeks 1 to 4' },
  { week: 6, term: 'Midterm', title: 'NLP: Tokens & Word Embeddings', short: 'Tokens and word embeddings', lab: 'GA Lab 2', kc: 60 },
  {
    week: 7,
    term: 'Midterm',
    title: 'Language Models & Attention',
    short: 'Language models and attention',
    lab: 'Laboratory 3, Part 1',
    labTitle: 'scaled dot-product attention',
    lessonTitle: 'Why attention replaced sequence models',
    lessonNote: 'Queries, keys and values',
    kc: 20,
    weight: '10% of final grade',
    extras: [{ title: 'Attention Explorer visualisation', note: 'Guided exercise · Self-attention weights' }],
    sessions: [
      { time: '08:00', title: 'Lecture 7.1: Attention mechanisms', note: 'Room CS-204' },
      { time: '10:30', title: 'Laboratory 3 session', note: 'Computing Laboratory 2 · Attendance required' },
      { time: '14:00', title: 'Consultation hours', note: 'By appointment · Faculty Room 5' },
    ],
  },
  { week: 8, term: 'Midterm', title: 'The Transformer & Autoregressive Training', short: 'The Transformer', lab: 'GA Lab 3, Part 2', kc: 40 },
  { week: 9, term: 'Midterm', title: 'LLM Families, Scaling Laws & Prompt Engineering', short: 'LLM families and prompt engineering', lab: 'GA Lab 4', kc: 60 },
  { week: 10, term: 'Midterm', title: 'Midterm examination', short: 'Midterm examination', exam: true, covers: 'Weeks 6 to 9' },
  { week: 11, term: 'Pre-final', title: 'Multimodal AI', short: 'Multimodal AI', lab: 'GA Demo 6.1', kc: 25 },
  { week: 12, term: 'Pre-final', title: 'Image Generation & GANs', short: 'Image generation and GANs', lab: 'GA Lab 6', kc: 50 },
  { week: 13, term: 'Pre-final', title: 'Diffusion Models', short: 'Diffusion models', lab: 'GA Lab 7', kc: 60 },
  { week: 14, term: 'Pre-final', title: 'Training LLMs: Fine-tuning, RLHF, PEFT', short: 'Fine-tuning, RLHF and PEFT', lab: null, kc: 0 },
  { week: 15, term: 'Pre-final', title: 'Pre-final examination and capstone proposal', short: 'Pre-final examination', exam: true, covers: 'Weeks 11 to 14' },
  { week: 16, term: 'Final', title: 'LLM Components & RAG', short: 'LLM components and RAG', lab: 'GA Lab 8, Part 1', kc: 50 },
  { week: 17, term: 'Final', title: 'AI Agents, Guardrails & Trustworthy AI', short: 'AI agents and guardrails', lab: 'GA Lab 8, Part 2', kc: 25 },
  { week: 18, term: 'Final', title: 'Scaling & Deploying AI', short: 'Scaling and deploying AI', lab: 'GA Lab 9', kc: 58 },
  { week: 19, term: 'Final', title: 'Capstone Build Week + Mock NCA-GENL Exam', short: 'Capstone build and mock exam', lab: null, kc: 0, capstone: true },
  { week: 20, term: 'Final', title: 'Final examination and Demo Day', short: 'Final examination and Demo Day', exam: true, covers: 'Weeks 16 to 19' },
];

const DUE_TIME = '11:59 PM PHT';

function build(w) {
  const n = w.week;
  const wed = dateOf(n);
  const sat = dateOf(n, 3);
  const due = `${longDate(sat)}, ${DUE_TIME}`;
  // Always a week address: the bare /lesson address opens the lesson library instead of a lesson.
  const lessonHref = `/lesson/${n}`;
  const lessonTitle = w.lessonTitle || w.title;
  const activities = [];
  const sessions = [];
  let hero;

  if (w.exam) {
    activities.push(
      { title: `Review: ${w.covers} content`, note: 'Lecture and reading · Cumulative review' },
      { title: 'Mock exam practice', note: 'Formative · Unweighted, unlimited attempts' },
      { title: w.title, note: `Examination · ${w.covers} covered` }
    );
    sessions.push(
      { time: '09:00', title: w.title, note: `Room CS-204 · ${w.covers} covered` },
      { time: '14:00', title: 'Consultation hours', note: 'By appointment · Faculty Room 5' }
    );
    hero = { lead: 'Prepare for the ', mark: w.title.toLowerCase(), rest: '', chips: [`${w.covers} covered`, `Exam on ${longDate(sat)}`], cta: 'Open the review lesson' };
  } else {
    activities.push({ title: `Lesson ${n}.1: ${lessonTitle}`, note: `Lecture and reading · ${w.lessonNote || w.short}` });
    (w.extras || []).forEach((e) => activities.push(e));
    if (w.lab) activities.push({ title: `${w.lab}${w.labTitle ? ': ' + w.labTitle : ''}`, note: `Laboratory · Report due ${due}` });
    if (w.kc) activities.push({ title: `Knowledge check ${n}.1: ${w.kc} questions`, note: 'Formative · Unweighted, unlimited attempts' });
    if (w.capstone) activities.push({ title: 'Capstone build', note: 'Project work · Mock NCA-GENL exam this week' });

    sessions.push(...(w.sessions || []));
    if (!w.sessions) {
      sessions.push({ time: '08:00', title: `Lecture ${n}.1: ${w.short}`, note: 'Room CS-204' });
      if (w.lab) sessions.push({ time: '10:30', title: `${w.lab} session`, note: 'Computing Laboratory 2 · Attendance required' });
      sessions.push({ time: '14:00', title: 'Consultation hours', note: 'By appointment · Faculty Room 5' });
    }

    hero = w.lab
      ? {
          lead: 'Complete ',
          mark: w.lab,
          rest: `: ${w.labTitle || w.short}`,
          chips: [...(w.weight ? [w.weight] : []), `Due ${due}`],
          cta: `Open ${w.lab}`,
        }
      : {
          lead: 'Work through ',
          mark: `Lesson ${n}.1`,
          rest: `: ${lessonTitle}`,
          chips: [w.capstone ? 'Capstone and mock exam week' : 'No laboratory this week'],
          cta: `Open Lesson ${n}.1`,
        };
  }

  return {
    week: n,
    term: w.term,
    termLabel: `${w.term} Term`,
    title: w.title,
    short: w.short,
    exam: !!w.exam,
    covers: w.covers || '',
    lab: w.lab || null,
    kc: w.kc || 0,
    lessonHref,
    lessonCard: { heading: `Week ${n} ${w.exam ? 'review' : 'lesson' + (w.lab ? ' and laboratory' : '')}`, cta: w.exam ? 'Continue: review lesson →' : `Continue: Lesson ${n}.1 →` },
    activities,
    sessions,
    sessionsDate: fullDate(wed),
    hero,
  };
}

export const WEEK_DATA = WEEKS.map(build);

// Accepts "7", 7, undefined; anything invalid falls back to the current week.
export function getWeek(value) {
  const n = Number(value);
  return WEEK_DATA[Number.isInteger(n) && n >= 1 && n <= TOTAL_WEEKS ? n - 1 : CURRENT_WEEK - 1];
}

// "Completed" before the current week, "Not started" after it; within it the first activity is done and the second under way.
export function activityStatus(week, index) {
  if (week < CURRENT_WEEK) return 'Completed';
  if (week > CURRENT_WEEK) return 'Not started';
  return index === 0 ? 'Completed' : index === 1 ? 'In progress' : 'Not started';
}
