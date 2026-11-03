// Checks every frontend/src/data/workshops/wNN.js: required fields, valid regexes, and that every preset
// prompt produces a rule reply (not the fallback, unless the preset is marked as meant to hit it).
// Usage: node tools/validate-workshops.mjs [week ...]      (no weeks = all files that exist)
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const dir = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'frontend', 'src', 'data', 'workshops');
const { runRules } = await import(pathToFileURL(path.join(dir, 'rules.js')));
const only = process.argv.slice(2).map(Number);
const files = fs.readdirSync(dir).filter((f) => /^w\d\d\.js$/.test(f)).filter((f) => !only.length || only.includes(Number(f.slice(1, 3))));
let problems = 0;
const bad = (f, m) => { problems++; console.log(`✗ ${f}: ${m}`); };

for (const f of files) {
  const w = (await import(pathToFileURL(path.join(dir, f)))).default;
  const n = Number(f.slice(1, 3));
  if (w.week !== n) bad(f, `week is ${w.week}, file name says ${n}`);
  for (const k of ['title', 'lede', 'lab']) if (!w[k] || typeof w[k] !== 'string') bad(f, `missing ${k}`);
  if (!Array.isArray(w.chips) || w.chips.length < 2 || w.chips.length > 4) bad(f, 'chips must be 2 to 4 strings');
  if (w.exam && !w.covers) bad(f, 'exam workshop needs covers');
  const ex = w.exercises || [];
  if (ex.length < 3 || ex.length > 8) bad(f, `has ${ex.length} exercises (want 3 to 8)`);
  ex.forEach((e, i) => {
    const id = `${f} exercise ${i + 1} (${e.tab || '?'})`;
    for (const k of ['tab', 'title', 'goal']) if (!e[k]) bad(id, `missing ${k}`);
    if (!Number.isFinite(e.min)) bad(id, 'min must be a number');
    if (!Array.isArray(e.tasks) || e.tasks.length < 1 || e.tasks.length > 5) bad(id, 'tasks must be 1 to 5 strings');
    if (!['code', 'prompt', 'note'].includes(e.kind)) return bad(id, `unknown kind ${e.kind}`);
    if (e.kind === 'code') for (const k of ['code', 'out', 'expect']) if (!e[k]) bad(id, `code exercise missing ${k}`);
    if (e.kind === 'note' && !e.note) bad(id, 'note exercise missing note');
    if (e.kind === 'prompt') {
      if (!e.presets || !e.presets.length) bad(id, 'prompt exercise needs presets');
      if (!e.rules || !e.rules.length) bad(id, 'prompt exercise needs rules');
      if (!e.fallback || !e.fallback.out) bad(id, 'prompt exercise needs fallback.out');
      (e.rules || []).forEach((r, j) => {
        if (!r.out) bad(id, `rule ${j + 1} has no out`);
        for (const k of ['u', 's', 'not']) if (r[k]) { try { new RegExp(r[k], 'i'); } catch (err) { bad(id, `rule ${j + 1} bad regex ${k}: ${err.message}`); } }
        if (!r.u && !r.s) bad(id, `rule ${j + 1} needs u or s`);
      });
      (e.presets || []).forEach((p) => {
        const hit = runRules(e, { usr: p.u, sys: p.s || '' });
        const isFallback = hit.out === e.fallback.out;
        if (isFallback && !p.l.match(/vague|weak|bad|naive|too short|no |missing/i)) bad(id, `preset "${p.l}" falls to the fallback; label it "Vague..." if intended, or fix the rules`);
      });
    }
  });
  console.log(`checked ${f}: ${ex.length} exercises`);
}
console.log(problems ? `\n${problems} problem(s)` : '\nall good');
process.exit(problems ? 1 : 0);
