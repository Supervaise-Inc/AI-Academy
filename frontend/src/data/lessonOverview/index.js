// Lesson library content, one file per week (w01.js ... w20.js). The Lesson Workspace opens a "lesson library"
// sheet with a card, an infographic and a preview for every week; adding or changing a week is a one-file edit.
// Title, term, lab and quiz counts come from ../courseWeeks.js; this folder holds what is specific to the library.
//
// A week file has one default export:
//
//   {
//     week: 4,
//     tagline: 'Small filters slide over an image to find edges and shapes.',   // one line, 90 characters at most
//     time: 'About 4.5 hours',                  // reading plus lab, from the lesson's own time estimate
//     ideas: ['...', '...', '...'],              // exactly 3 key ideas, 80 characters at most each, plain language
//     outcomes: ['...', '...', '...'],           // exactly 3, each starts with a verb ("Compute", "Explain") and
//                                                // follows the lesson's "By the end, you can" list
//     terms: [{ t: 'Kernel', d: 'A small grid of weights that slides over the image.' }],   // 4 to 6 terms, d <= 90 chars
//     path: [{ label: 'Read', min: 45 }, { label: 'Visuals', min: 20 }, { label: 'Lab', min: 180 }, { label: 'Check', min: 20 }],
//                                                // 2 to 5 study steps with minutes, taken from the lesson (never invented)
//     graphic: { type: '...', ... },             // the infographic, see below
//   }
//
// Exam weeks (5, 10, 15, 20) and the capstone week (19) are revision overviews: `ideas` are the topics to revise,
// `outcomes` start with "Recall", "Explain", "Apply", and `path` is the revision plan. Add `exam: true`.
//
// graphic types (keep every label short: 14 characters for node labels, 22 for sub-labels, 70 for a caption):
//   { type: 'flow',    steps: [{ label, sub? }], caption }                      3 to 5 steps, left to right
//   { type: 'cycle',   steps: [{ label }], caption }                            3 to 5 steps in a loop
//   { type: 'layers',  items: [{ label, sub? }], caption }                      3 to 5 layers, first = top
//   { type: 'bars',    unit: '%', items: [{ label, value }], caption }          3 to 5 bars, value is a number
//   { type: 'compare', left: { title, points }, right: { title, points }, caption }   2 to 3 points each, 28 chars
//   { type: 'grid',    rows: [['..', '..'], ...], hot: [[row, col], ...], caption }   up to 5 x 5 short cells
//   { type: 'curve',   xLabel, yLabel, series: [{ label, points: [[x, y], ...] }], caption }   1 or 2 series, 4+ points
// The graphic must show the week's central idea, with numbers taken from the lesson, never invented.

import { getWeek } from '../courseWeeks.js';

const modules = import.meta.glob('./w[0-9][0-9].js', { eager: true });

export const OVERVIEWS = {};
for (const m of Object.values(modules)) {
  if (m.default && m.default.week) OVERVIEWS[m.default.week] = m.default;
}

// Overview for any week; a week with no file still gets a plain card.
export function overviewOf(week) {
  const base = getWeek(week);
  const o = OVERVIEWS[week];
  return {
    week: base.week,
    title: base.title,
    short: base.short,
    term: base.term,
    exam: !!base.exam,
    covers: base.covers,
    lab: base.lab,
    kc: base.kc,
    ...(o || { tagline: base.short, time: '', ideas: [], outcomes: [], terms: [], path: [], graphic: null }),
    exam: o && o.exam !== undefined ? !!o.exam : !!base.exam,
  };
}
