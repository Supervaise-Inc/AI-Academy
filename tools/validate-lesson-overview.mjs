// Checks every frontend/src/data/lessonOverview/wNN.js against the schema in index.js (field counts and lengths).
// Usage: node tools/validate-lesson-overview.mjs [week ...]      (no weeks = every file that exists)
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const dir = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'frontend', 'src', 'data', 'lessonOverview');
const only = process.argv.slice(2).map(Number);
const files = fs.readdirSync(dir).filter((f) => /^w\d\d\.js$/.test(f)).filter((f) => !only.length || only.includes(Number(f.slice(1, 3))));
let problems = 0;
const bad = (f, m) => { problems++; console.log(`✗ ${f}: ${m}`); };
const len = (f, what, s, max) => { if (typeof s !== 'string' || !s.trim()) bad(f, `${what} is empty`); else if (s.length > max) bad(f, `${what} is ${s.length} chars (max ${max}): "${s}"`); };
const num = (v) => typeof v === 'number' && Number.isFinite(v);

for (const f of files) {
  const w = (await import(pathToFileURL(path.join(dir, f)))).default;
  const n = Number(f.slice(1, 3));
  if (w.week !== n) bad(f, `week is ${w.week}, file name says ${n}`);
  len(f, 'tagline', w.tagline, 90); len(f, 'time', w.time, 60);
  if (!Array.isArray(w.ideas) || w.ideas.length !== 3) bad(f, 'ideas must be exactly 3'); else w.ideas.forEach((x, i) => len(f, `idea ${i + 1}`, x, 80));
  if (!Array.isArray(w.outcomes) || w.outcomes.length !== 3) bad(f, 'outcomes must be exactly 3'); else w.outcomes.forEach((x, i) => len(f, `outcome ${i + 1}`, x, 90));
  if (!Array.isArray(w.terms) || w.terms.length < 4 || w.terms.length > 6) bad(f, 'terms must be 4 to 6'); else w.terms.forEach((x, i) => { len(f, `term ${i + 1} name`, x.t, 28); len(f, `term ${i + 1} definition`, x.d, 90); });
  if (!Array.isArray(w.path) || w.path.length < 2 || w.path.length > 5) bad(f, 'path must be 2 to 5 steps'); else w.path.forEach((x, i) => { len(f, `path ${i + 1} label`, x.label, 14); if (!num(x.min) || x.min <= 0) bad(f, `path ${i + 1} min must be a positive number`); });
  const g = w.graphic;
  if (!g) { bad(f, 'graphic missing'); continue; }
  len(f, 'graphic caption', g.caption, 80);
  const nodes = (arr, what, lo, hi, lmax = 14, smax = 22) => {
    if (!Array.isArray(arr) || arr.length < lo || arr.length > hi) return bad(f, `${what} needs ${lo} to ${hi} items`);
    arr.forEach((x, i) => { len(f, `${what} ${i + 1} label`, x.label, lmax); if (x.sub !== undefined) len(f, `${what} ${i + 1} sub`, x.sub, smax); });
  };
  switch (g.type) {
    case 'flow': nodes(g.steps, 'flow step', 3, 5); break;
    case 'cycle': nodes(g.steps, 'cycle step', 3, 5, 12); break;
    case 'layers': nodes(g.items, 'layer', 3, 5, 18, 26); break;
    case 'bars': nodes(g.items, 'bar', 3, 5, 14); (g.items || []).forEach((x, i) => { if (!num(x.value)) bad(f, `bar ${i + 1} value must be a number`); }); break;
    case 'compare': for (const side of ['left', 'right']) { const s = g[side]; if (!s) { bad(f, `compare ${side} missing`); continue; } len(f, `${side} title`, s.title, 18); if (!Array.isArray(s.points) || s.points.length < 2 || s.points.length > 3) bad(f, `${side} needs 2 or 3 points`); else s.points.forEach((p, i) => len(f, `${side} point ${i + 1}`, p, 28)); } break;
    case 'grid': if (!Array.isArray(g.rows) || g.rows.length < 2 || g.rows.length > 5 || g.rows.some((r) => !Array.isArray(r) || r.length < 2 || r.length > 5)) bad(f, 'grid needs 2 to 5 rows of 2 to 5 cells'); else { g.rows.forEach((r) => r.forEach((c) => { if (String(c).length > 6) bad(f, `grid cell "${c}" is longer than 6 characters`); })); (g.hot || []).forEach((h) => { if (!g.rows[h[0]] || g.rows[h[0]][h[1]] === undefined) bad(f, `hot cell ${h} is outside the grid`); }); } break;
    case 'curve': len(f, 'xLabel', g.xLabel, 22); len(f, 'yLabel', g.yLabel, 22); if (!Array.isArray(g.series) || g.series.length < 1 || g.series.length > 2) bad(f, 'curve needs 1 or 2 series'); else g.series.forEach((s, i) => { len(f, `series ${i + 1} label`, s.label, 16); if (!Array.isArray(s.points) || s.points.length < 4 || s.points.some((p) => !Array.isArray(p) || !num(p[0]) || !num(p[1]))) bad(f, `series ${i + 1} needs 4 or more [x, y] number pairs`); }); break;
    default: bad(f, `unknown graphic type ${g.type}`);
  }
  console.log(`checked ${f}`);
}
console.log(problems ? `\n${problems} problem(s)` : '\nall good');
process.exit(problems ? 1 : 0);
