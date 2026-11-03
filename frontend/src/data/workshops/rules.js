// Reply picker for the data-driven prompt exercises (see index.js for the rule format). Pure, so Node can test it.
const tests = (re, text) => {
  try {
    return new RegExp(re, 'i').test(text || '');
  } catch (e) {
    return false;
  }
};

// Pick the reply for a data-driven prompt exercise. Returns { out, note, ok }.
export function runRules(ex, d) {
  const usr = d.usr || '',
    sys = d.sys || '';
  for (const r of ex.rules || []) {
    if (r.u && !tests(r.u, usr)) continue;
    if (r.s && !tests(r.s, sys)) continue;
    if (r.not && (tests(r.not, usr) || tests(r.not, sys))) continue;
    return { out: r.out, note: r.note || '', ok: r.ok === undefined ? null : r.ok };
  }
  const f = ex.fallback || { out: 'The model replied, but not in a way this exercise is looking for. Re-read the task and try again.' };
  return { out: f.out, note: f.note || '', ok: f.ok === undefined ? null : f.ok };
}
