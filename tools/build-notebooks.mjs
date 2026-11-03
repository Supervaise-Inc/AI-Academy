// Builds Colab-ready Jupyter notebooks from percent-format Python sources.
//   notebooks/src/week-NN.py   ->   notebooks/week-NN.ipynb
//
//   node tools/build-notebooks.mjs            build every notebook
//   node tools/build-notebooks.mjs week-04    build one
//
// Source format (see docs/NOTEBOOK_FORMAT.md):
//   # accelerator: GPU         (optional, first line only: asks Colab for a GPU runtime)
//   # %% [markdown]            a markdown cell; the following "# " comment lines are its text
//   # %%                       a code cell; the following lines are its code
//   #!pip install -q foo       inside a code cell: becomes the shell command "!pip install -q foo"
//   #%matplotlib inline        inside a code cell: becomes the line magic "%matplotlib inline"
// Because magics stay comments in the .py file, the source also runs as an ordinary Python script.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const srcDir = path.join(root, 'notebooks', 'src');
const outDir = path.join(root, 'notebooks');

export function parse(text, file = 'source') {
  const lines = text.replace(/\r\n/g, '\n').split('\n');
  const meta = {};
  const cells = [];
  let cur = null;
  const errors = [];
  const flush = () => {
    if (!cur) return;
    while (cur.lines.length && cur.lines[cur.lines.length - 1].trim() === '') cur.lines.pop();
    while (cur.lines.length && cur.lines[0].trim() === '') cur.lines.shift();
    if (cur.lines.length) cells.push(cur);
    cur = null;
  };
  lines.forEach((line, i) => {
    const m = /^# %%(?:\s+\[(markdown)\])?\s*$/.exec(line);
    if (m) {
      flush();
      cur = { type: m[1] ? 'markdown' : 'code', lines: [], at: i + 1 };
      return;
    }
    if (!cur) {
      const h = /^# (accelerator):\s*(\w+)\s*$/.exec(line);
      if (h) meta[h[1]] = h[2];
      else if (line.trim() !== '') errors.push(`${file}:${i + 1}: text before the first "# %%" cell marker`);
      return;
    }
    if (cur.type === 'markdown') {
      if (line.trim() === '') cur.lines.push('');
      else if (line === '#') cur.lines.push('');
      else if (line.startsWith('# ')) cur.lines.push(line.slice(2));
      else errors.push(`${file}:${i + 1}: markdown cell lines must start with "# " (got: ${line.slice(0, 40)})`);
    } else if (/^#!/.test(line)) cur.lines.push('!' + line.slice(2));
    else if (/^#%/.test(line)) cur.lines.push('%' + line.slice(2));
    else cur.lines.push(line);
  });
  flush();
  return { cells, meta, errors };
}

const toSource = (lines) => lines.map((l, i) => (i < lines.length - 1 ? l + '\n' : l));

export function toNotebook({ cells, meta }) {
  const first = cells.find((c) => c.type === 'markdown');
  const title = (first && /^#\s+(.+)$/m.exec(first.lines.join('\n')))?.[1] || 'Notebook';
  const nb = {
    cells: cells.map((c) =>
      c.type === 'markdown'
        ? { cell_type: 'markdown', metadata: {}, source: toSource(c.lines) }
        : { cell_type: 'code', execution_count: null, metadata: {}, outputs: [], source: toSource(c.lines) }
    ),
    metadata: {
      colab: { name: title, provenance: [], toc_visible: true, ...(meta.accelerator ? { gpuType: 'T4' } : {}) },
      kernelspec: { display_name: 'Python 3', language: 'python', name: 'python3' },
      language_info: { name: 'python' },
      ...(meta.accelerator ? { accelerator: meta.accelerator } : {}),
    },
    nbformat: 4,
    nbformat_minor: 0,
  };
  return { nb, title };
}

export function check(parsed, file) {
  const errs = [...parsed.errors];
  const { cells } = parsed;
  const code = cells.filter((c) => c.type === 'code');
  const md = cells.filter((c) => c.type === 'markdown');
  if (!cells.length || cells[0].type !== 'markdown' || !/^# \S/.test(cells[0].lines[0] || '')) errs.push(`${file}: the first cell must be markdown starting with a "# Title" line`);
  if (code.length < 5) errs.push(`${file}: only ${code.length} code cells; a lesson notebook needs at least 5`);
  if (md.length < 5) errs.push(`${file}: only ${md.length} markdown cells; explain each step`);
  if (cells.length && cells[cells.length - 1].type !== 'markdown') errs.push(`${file}: the last cell must be markdown (a wrap-up)`);
  if (!code.some((c) => /\bassert\b/.test(c.lines.join('\n')))) errs.push(`${file}: add at least one self-check cell using assert`);
  if (!md.some((c) => /<details>/.test(c.lines.join('\n')))) errs.push(`${file}: add at least one <details> block with a worked answer`);
  if (!/Colab/i.test(cells[0]?.lines.join('\n') || '')) errs.push(`${file}: the title cell should tell students it is meant for Google Colab`);
  return errs;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const want = process.argv.slice(2).map((a) => a.replace(/\.(py|ipynb)$/, ''));
  const files = fs.existsSync(srcDir) ? fs.readdirSync(srcDir).filter((f) => /^week-\d\d\.py$/.test(f)).sort() : [];
  let bad = 0;
  for (const f of files) {
    const name = f.replace('.py', '');
    if (want.length && !want.includes(name)) continue;
    const parsed = parse(fs.readFileSync(path.join(srcDir, f), 'utf8'), f);
    const errs = check(parsed, f);
    if (errs.length) {
      bad += errs.length;
      console.log(`✗ ${name}`);
      errs.forEach((e) => console.log('    ERROR  ' + e));
      continue;
    }
    const { nb, title } = toNotebook(parsed);
    fs.writeFileSync(path.join(outDir, name + '.ipynb'), JSON.stringify(nb, null, 1) + '\n');
    console.log(`✓ ${name}.ipynb  ${nb.cells.filter((c) => c.cell_type === 'code').length} code · ${nb.cells.filter((c) => c.cell_type === 'markdown').length} markdown cells · "${title}"`);
  }
  if (!files.length) console.log('No sources in', srcDir);
  process.exit(bad ? 1 : 0);
}
