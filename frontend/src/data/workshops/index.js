// Prompt Workshop content, one file per week (w01.js ... w20.js). The workshop page reads the week the
// student opened from the Course Map, so adding or changing a week's workshop is a one-file edit.
//
// A week file has one default export:
//
//   {
//     week: 4,
//     title: 'Convolution lab',                       // page heading
//     lede: 'Run the lab step by step ...',           // one sentence under the heading
//     lab: 'DL Lab 4B, Part 1',                       // name of the lab this workshop follows
//     chips: ['CPU is enough', '8×8 digits'],         // small facts shown under the heading (2 to 4)
//     exercises: [ ...see below... ],                 // 4 to 7 exercises, in lab order
//   }
//
// Exam weeks (5, 10, 15, 20) and the capstone week (19) have no lab. Their file is a revision workshop built from the
// weeks the exam covers: add `exam: true` and `covers: 'Weeks 1 to 4'` at the top level (the page then shows an
// "Open Mock Exam Prep" button), and use `prompt` exercises (self-quiz with the tutor) and `note` cells (revision cards).
//
// Every exercise has: tab (short tab label, like '1 · convolution'), title, min (minutes), goal (one sentence),
// tasks (2 to 4 short imperative steps), and a kind:
//
//   kind: 'code'    A lab code cell. `code` is the snippet (copy the lab's own code, trimmed to what fits on a
//                   screen). Pressing Run shows `out`, the output the lab says to expect. `expect` is the one-line
//                   "what to look for". The cell is not executed: it shows the recorded output.
//
//   kind: 'prompt'  The student writes a prompt to a simulated tutor model and presses Run. `sys: true` adds a
//                   system-prompt box. `presets` are starting prompts: [{ l: 'label', u: 'user text', s: 'system text' }].
//                   `rules` pick the reply: the FIRST rule whose tests all pass wins. Each rule is
//                   { u: 'regex', s: 'regex', not: 'regex', out: 'reply', note: 'short verdict', ok: true|false }.
//                   `u` is tested against the user prompt, `s` against the system prompt, `not` against both
//                   (the rule is skipped when it matches). All regexes are case-insensitive strings. `fallback`
//                   is { out, note, ok } for anything no rule matches. Every preset must match a rule.
//
//   kind: 'note'    A markdown notes cell. `note` is the starting text (a checklist or reflection card).
//
// Rules for content: outputs must come from the lesson or notebook ("What you should see") or be derivable by hand.
// Where a real value varies by run (accuracy, loss), write a range or "about", never a made-up exact number.

const modules = import.meta.glob('./w[0-9][0-9].js', { eager: true });

export const WORKSHOPS = {};
for (const m of Object.values(modules)) {
  if (m.default && m.default.week) WORKSHOPS[m.default.week] = m.default;
}

export { runRules } from './rules.js';
