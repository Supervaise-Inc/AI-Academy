import React, { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { marked } from 'marked';
import DOMPurify from 'dompurify';
import { Widget, HookFigure } from './lessonWidgets.jsx';
import { navigate, apiUrl } from './session.js';

// Renders a lesson written in Markdown (backend/content/week-NN.md) with the same type system and
// components as the built-in Week 7 lesson: hero, hook figure, section rail, outcomes, takeaway, lab, check.
//
// Markdown conventions on top of GitHub-flavoured Markdown:
//   > text                       (first line of the file)  source note, shown under "Lesson details"
//   **Purpose:** ...             the one-sentence purpose shown in the hero
//   **Term:** a · **Time:** b    "Lesson details" key/value list
//   ==important idea==           highlighter marker
//   [[term|definition]]          dotted-underline term; click to show the definition
//   {lede} / {sub}               start of a paragraph: bigger main-idea line / small line under the heading
//   ## Heading {rail=Label}      short label for the sticky section rail (only marked sections appear there)
//   ## By the end, you can…      outcomes checklist     ## Key takeaways   takeaway blocks
//   ```viz {json}  ```quiz {json}   interactive figures and a scored quiz (see lessonWidgets.jsx)

marked.use({
  extensions: [
    {
      name: 'mark',
      level: 'inline',
      start: (src) => src.indexOf('=='),
      tokenizer(src) {
        const m = /^==(?!=)([^=\n]+?)==/.exec(src);
        if (m) return { type: 'mark', raw: m[0], text: m[1], tokens: this.lexer.inlineTokens(m[1]) };
        return undefined;
      },
      renderer(t) {
        return `<mark class="mk">${this.parser.parseInline(t.tokens)}</mark>`;
      },
    },
  ],
});

const slugify = (s) =>
  s
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 60);

const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

// [[term|definition]] -> button that toggles an inline definition
const withTerms = (md) =>
  md.replace(/\[\[([^\]|\n]+)\|([^\]\n]+)\]\]/g, (_, term, def) => `<button type="button" class="term" aria-expanded="false" data-def="${esc(def)}">${term}</button>`);

// CommonMark won't start a numbered list at "3." straight after a paragraph, which is how the lab steps
// ("3. Translation…") follow the indented "Exercise 2" lines. Put a blank line before such items.
function fixNumberedLists(md) {
  const lines = md.split('\n');
  const isItem = (l) => /^\s*(\d+\.|[-*+])\s/.test(l);
  return lines
    .map((l, i) => (/^\d+\.\s/.test(l) && i > 0 && lines[i - 1].trim() !== '' && !isItem(lines[i - 1]) ? '\n' + l : l))
    .join('\n');
}

// --- tiny Python highlighter in the design's token colours (.c comment, .k keyword, .f function, .n number/string)
const PY_KW = new Set('import from as def return with for in if else elif del print not and or None True False class try except lambda pass'.split(' '));
function highlightPython(code) {
  const out = [];
  const re = /(#[^\n]*)|("(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*')|(\b\d+(?:\.\d+)?\b)|([A-Za-z_][A-Za-z0-9_]*)(\()?/g;
  let last = 0;
  let m;
  while ((m = re.exec(code))) {
    if (m.index > last) out.push(esc(code.slice(last, m.index)));
    if (m[1]) out.push(`<span class="c">${esc(m[1])}</span>`);
    else if (m[2]) out.push(`<span class="n">${esc(m[2])}</span>`);
    else if (m[3]) out.push(`<span class="n">${m[3]}</span>`);
    else if (PY_KW.has(m[4])) out.push(`<span class="k">${m[4]}</span>${m[5] || ''}`);
    else if (m[5]) out.push(`<span class="f">${m[4]}</span>(`);
    else out.push(m[4]);
    last = m.index + m[0].length;
  }
  if (last < code.length) out.push(esc(code.slice(last)));
  return out.join('');
}

const CHECK_SVG =
  '<svg viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="M3 8.4l3.1 3L13 4.6" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"></path></svg>';

const take = (html) => {
  const d = document.createElement('div');
  d.className = 'take';
  d.innerHTML = `<p>${html}</p>`;
  return d;
};

function stripPrefix(el, re) {
  const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
  const n = walker.nextNode();
  if (n) n.nodeValue = n.nodeValue.replace(re, '');
}

export function prepare(markdown) {
  const raw = marked.parse(fixNumberedLists(withTerms(markdown)), { gfm: true, breaks: true });
  const clean = DOMPurify.sanitize(raw);
  const box = document.createElement('div');
  box.innerHTML = clean;

  const view = { purpose: '', details: [], note: '', hook: null, rail: [], html: '', widgets: [] };

  // header pieces
  const first = box.firstElementChild;
  if (first && first.tagName === 'BLOCKQUOTE') {
    view.note = first.textContent.trim();
    first.remove();
  }
  const h1 = box.querySelector('h1');
  if (h1) h1.remove();
  [...box.children].forEach((el) => {
    if (el.tagName !== 'P' || box.querySelector('h2')?.compareDocumentPosition(el) & Node.DOCUMENT_POSITION_FOLLOWING) return;
    const lead = el.firstElementChild;
    const label = lead && lead.tagName === 'STRONG' ? lead.textContent.trim() : '';
    if (/^Purpose:?$/i.test(label)) {
      lead.remove();
      view.purpose = el.innerHTML.trim();
      el.remove();
    } else if (label && label.endsWith(':')) {
      // "**Term:** a · **Time:** b" -> key/value pairs
      let key = null;
      let val = '';
      const flush = () => key && view.details.push({ k: key.replace(/:$/, ''), v: val.replace(/\s*·\s*$/, '').trim() });
      el.childNodes.forEach((n) => {
        if (n.nodeType === 1 && n.tagName === 'STRONG' && n.textContent.trim().endsWith(':')) {
          flush();
          key = n.textContent.trim();
          val = '';
        } else val += n.textContent;
      });
      flush();
      el.remove();
    }
  });

  // ```viz / ```quiz fenced blocks -> mount points for interactive widgets (a "hook" block goes in the hero area)
  let fig = 0;
  box.querySelectorAll('pre > code').forEach((code) => {
    const kind = /\blanguage-(viz|quiz)\b/.exec(code.className || '');
    if (!kind) return;
    let spec;
    let error;
    try {
      spec = JSON.parse(code.textContent);
    } catch (e) {
      error = 'The block is not valid JSON (' + e.message + ').';
    }
    if (spec && spec.type === 'hook') {
      view.hook = spec;
      code.parentElement.remove();
      return;
    }
    const host = document.createElement('div');
    host.className = 'wg-host';
    host.setAttribute('data-w', String(view.widgets.length));
    view.widgets.push({ kind: kind[1], spec, error, fig: kind[1] === 'viz' ? ++fig : 0 });
    code.parentElement.replaceWith(host);
  });

  // inline code, code blocks (design "codebox"), tables
  box.querySelectorAll('code').forEach((c) => {
    if (c.parentElement.tagName !== 'PRE') c.classList.add('icode');
  });
  box.querySelectorAll('pre').forEach((pre) => {
    const code = pre.textContent.replace(/\n$/, '');
    const lang = (/language-(\w+)/.exec(pre.firstElementChild?.className || '') || [])[1] || 'python';
    const wrap = document.createElement('div');
    wrap.className = 'codebox';
    wrap.innerHTML = `<div class="codebar"><span>${lang === 'python' ? 'Python' : esc(lang)}</span><button type="button" class="cp" data-copy>Copy</button></div><pre class="code">${
      lang === 'python' ? highlightPython(code) : esc(code)
    }</pre>`;
    wrap.querySelector('button').setAttribute('data-code', code);
    pre.replaceWith(wrap);
  });
  // external links open in a new tab
  box.querySelectorAll('a[href^="http"]').forEach((l) => {
    l.setAttribute('target', '_blank');
    l.setAttribute('rel', 'noopener noreferrer');
  });
  box.querySelectorAll('table').forEach((t) => {
    const wrap = document.createElement('div');
    wrap.className = 'wp-tbl';
    t.replaceWith(wrap);
    wrap.appendChild(t);
  });

  // split into sections at each h2
  const sections = [];
  let cur = null;
  [...box.children].forEach((el) => {
    if (el.tagName === 'H2') {
      cur = { h: el, nodes: [] };
      sections.push(cur);
    } else if (cur) cur.nodes.push(el);
  });

  const out = document.createElement('div');
  const used = new Set();
  const hasRailMarks = sections.some((s) => /\{rail=[^}]+\}\s*$/.test(s.h.textContent));
  sections.forEach(({ h, nodes }) => {
    let title = h.textContent.trim();
    let label = null;
    const m = /\s*\{rail=([^}]+)\}\s*$/.exec(title);
    if (m) {
      label = m[1].trim();
      title = title.replace(m[0], '');
    }
    h.textContent = title;

    if (/^By the end/i.test(title)) {
      const det = document.createElement('details');
      det.className = 'disc outc';
      det.open = true;
      det.id = 'outcomes';
      const list = nodes.find((n) => n.tagName === 'OL' || n.tagName === 'UL');
      det.innerHTML = `<summary>${esc(title)}</summary><ul class="chk">${[...(list ? list.children : [])]
        .map((li) => `<li>${CHECK_SVG}<span>${li.innerHTML}</span></li>`)
        .join('')}</ul>`;
      nodes
        .filter((n) => n !== list && n.tagName === 'P')
        .forEach((p) => {
          const n = document.createElement('p');
          n.className = 'chk-n';
          n.innerHTML = p.innerHTML;
          det.appendChild(n);
        });
      out.appendChild(det);
      return;
    }

    let id = slugify(title) || 'section';
    while (used.has(id)) id += '-2';
    used.add(id);
    const isTake = /^Key takeaway/i.test(title);
    const isLab = /^Lab/i.test(title);
    const isStudy = /^Study plan/i.test(title);
    const sec = document.createElement('section');
    sec.className = 'sec';
    sec.id = id;
    sec.setAttribute('aria-labelledby', `${id}-h`);
    h.className = 'sec-h' + (isTake ? ' is-label' : '');
    h.id = `${id}-h`;
    sec.appendChild(h);
    if (label || !hasRailMarks) view.rail.push({ id, label: label || title });

    let firstPara = true;
    let nbPlaced = false;
    nodes.forEach((el) => {
      const tag = el.tagName;
      if (tag === 'P') {
        const t = el.textContent;
        if (/^\{sub\}/.test(t)) {
          stripPrefix(el, /^\{sub\}\s*/);
          el.className = 'sec-sub';
        } else if (/^\{lede\}/.test(t)) {
          stripPrefix(el, /^\{lede\}\s*/);
          el.className = 'prose lede';
        } else if (isLab && firstPara) el.className = 'labobj';
        else el.className = 'prose';
        firstPara = false;
      } else if (tag === 'UL' || tag === 'OL') {
        if (isTake) {
          [...el.children].forEach((li) => sec.appendChild(take(li.innerHTML)));
          return;
        }
        el.className = isLab && tag === 'OL' ? 'lsteps' : 'plist';
        if (el.className === 'lsteps') el.style.counterReset = `step ${(Number(el.getAttribute('start')) || 1) - 1}`;
      } else if (tag === 'H3') {
        el.className = 'h3';
        el.id = `${id}-${slugify(el.textContent)}`;
      } else if (tag === 'BLOCKQUOTE') {
        sec.appendChild(take(el.textContent));
        return;
      }
      sec.appendChild(el);
      // the Colab notebook card goes right after the first paragraph of the lab (or study plan) section
      if ((isLab || isStudy) && !nbPlaced && tag === 'P') {
        const host = document.createElement('div');
        host.className = 'nb-host';
        sec.appendChild(host);
        nbPlaced = true;
      }
    });
    out.appendChild(sec);
  });

  // "*Exercise 3:* …" paragraphs and steps become a highlighted callout with a badge
  out.querySelectorAll('p.prose, .lsteps > li, .plist > li').forEach((el) => {
    const lead = el.firstElementChild;
    if (!lead || lead.tagName !== 'EM' || !/^Exercise\b/i.test(lead.textContent.trim())) return;
    el.classList.add('exercise');
    const badge = document.createElement('span');
    badge.className = 'ex-badge';
    badge.textContent = lead.textContent.trim().replace(/:$/, '');
    lead.replaceWith(badge);
  });

  view.html = out.innerHTML;
  return view;
}

const checksKey = (slug) => `aia-checks-${slug}`;
const loadChecks = (slug) => {
  try {
    return JSON.parse(localStorage.getItem(checksKey(slug)) || '[]');
  } catch (e) {
    return [];
  }
};

// Lesson Workspace content for a week whose lesson lives in backend/content/week-NN.md.
export function PlanView({ plan, view }) {
  const article = useRef(null);
  const [hosts, setHosts] = useState([]);
  const [cur, setCur] = useState(0);
  const [weekLabel, ...rest] = plan.title.split(/\s+[—–-]\s+/);
  const headline = rest.join(' — ') || plan.title;

  // mount points for the interactive figures inside the injected HTML
  useLayoutEffect(() => {
    setHosts(article.current ? [...article.current.querySelectorAll('.wg-host')] : []);
  }, [view]);

  // checklist ticks, term definitions and the code "Copy" buttons: the HTML is injected, so listen natively
  useEffect(() => {
    const el = article.current;
    if (!el) return undefined;
    const done = new Set(loadChecks(plan.slug));
    el.querySelectorAll('input[type="checkbox"]').forEach((c) => {
      c.checked = done.has(Number(c.dataset.i));
    });
    const onClick = (e) => {
      // links to other app pages (e.g. /prompt-workshop) navigate without a page reload
      const link = e.target.closest && e.target.closest('a[href^="/"]');
      if (link && !e.metaKey && !e.ctrlKey && !e.shiftKey && e.button === 0) {
        e.preventDefault();
        navigate(link.getAttribute('href'));
        return;
      }
      const t = e.target.closest && e.target.closest('button');
      if (!t) return;
      if (t.classList.contains('term')) {
        const block = t.closest('p, li');
        const open = t.getAttribute('aria-expanded') === 'true';
        const key = t.textContent;
        block.parentElement.querySelectorAll(`.defn[data-term="${CSS.escape(key)}"]`).forEach((d) => d.remove());
        t.setAttribute('aria-expanded', open ? 'false' : 'true');
        if (!open) {
          const d = document.createElement('div');
          d.className = 'defn';
          d.setAttribute('data-term', key);
          d.setAttribute('role', 'note');
          d.innerHTML = `<strong></strong> <span></span>`;
          d.firstChild.textContent = key;
          d.lastChild.textContent = '· ' + t.getAttribute('data-def');
          block.after(d);
        }
      } else if (t.hasAttribute('data-copy')) {
        const code = t.getAttribute('data-code') || '';
        const done2 = () => {
          t.textContent = 'Copied';
          setTimeout(() => (t.textContent = 'Copy'), 1400);
        };
        if (navigator.clipboard) navigator.clipboard.writeText(code).then(done2, () => {});
      }
    };
    const onChange = (e) => {
      const c = e.target;
      if (!(c instanceof HTMLInputElement) || c.type !== 'checkbox') return;
      const set = new Set(loadChecks(plan.slug));
      if (c.checked) set.add(Number(c.dataset.i));
      else set.delete(Number(c.dataset.i));
      try {
        localStorage.setItem(checksKey(plan.slug), JSON.stringify([...set]));
      } catch (err) {
        /* storage unavailable */
      }
    };
    el.addEventListener('click', onClick);
    el.addEventListener('change', onChange);
    return () => {
      el.removeEventListener('click', onClick);
      el.removeEventListener('change', onChange);
    };
  }, [plan, view]);

  // which section is the reader in? (drives the rail)
  useEffect(() => {
    let raf = 0;
    const update = () => {
      raf = 0;
      const vh = window.innerHeight || 800;
      let c = 0;
      view.rail.forEach((r, i) => {
        const s = document.getElementById(r.id);
        if (s && s.getBoundingClientRect().top <= vh * 0.4) c = i;
      });
      setCur(c);
    };
    const onScroll = () => {
      if (!raf) raf = requestAnimationFrame(update);
    };
    document.addEventListener('scroll', onScroll, { capture: true, passive: true });
    window.addEventListener('resize', onScroll);
    update();
    return () => {
      document.removeEventListener('scroll', onScroll, { capture: true });
      window.removeEventListener('resize', onScroll);
      if (raf) cancelAnimationFrame(raf);
    };
  }, [view]);

  const go = (id) => (e) => {
    e.preventDefault();
    const el = document.getElementById(id);
    if (el) el.scrollIntoView({ behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start' });
  };

  return (
    <div className="lp-root">
      <header className="lhead hx hx-cyan">
        <div className="hx-body">
          <h1 className="lh-title">{headline}</h1>
          {view.purpose ? <p className="lh-purpose" dangerouslySetInnerHTML={{ __html: view.purpose }} /> : <p className="lh-purpose">{weekLabel}</p>}
          <details className="disc">
            <summary>Lesson details</summary>
            <div className="disc-b">
              <dl className="kv">
                <dt>Lesson</dt>
                <dd>{weekLabel}</dd>
                {view.details.map((d) => (
                  <React.Fragment key={d.k}>
                    <dt>{d.k}</dt>
                    <dd>{d.v}</dd>
                  </React.Fragment>
                ))}
                {view.note ? (
                  <>
                    <dt>Source</dt>
                    <dd>{view.note}</dd>
                  </>
                ) : null}
              </dl>
            </div>
          </details>
        </div>
        <div className="hx-ico">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M12 3l1.8 4.7L18.5 9.5l-4.7 1.8L12 16l-1.8-4.7L5.5 9.5l4.7-1.8z" />
            <path d="M18.5 15l.8 2.2 2.2.8-2.2.8-.8 2.2-.8-2.2-2.2-.8 2.2-.8z" />
          </svg>
        </div>
      </header>

      {view.hook ? <HookFigure spec={view.hook} /> : null}

      {view.rail.length > 1 ? (
        <nav className="rail" aria-label="Lesson sections">
          <ol style={{ gridTemplateColumns: `repeat(${view.rail.length},minmax(0,1fr))` }}>
            {view.rail.map((r, i) => (
              <li key={r.id}>
                <a className={i < cur ? 'done' : i === cur ? 'cur' : ''} href={`#${r.id}`} aria-current={i === cur ? 'true' : undefined} onClick={go(r.id)}>
                  <i aria-hidden="true"></i>
                  <span>{r.label}</span>
                </a>
              </li>
            ))}
          </ol>
        </nav>
      ) : null}

      <article ref={article} className="wp-prose" dangerouslySetInnerHTML={{ __html: view.html }} />
      {plan.notebook && article.current
        ? [...article.current.querySelectorAll('.nb-host')].map((h, i) => createPortal(<NotebookCard nb={plan.notebook} week={plan.title} />, h, 'nb' + i))
        : null}
      {hosts.map((h) => {
        const w = view.widgets[Number(h.getAttribute('data-w'))];
        return w ? createPortal(<Widget kind={w.kind} spec={w.spec} error={w.error} fig={w.fig} />, h, h.getAttribute('data-w')) : null;
      })}
    </div>
  );
}

// "Open in Colab" / "Download" card shown at the start of a lesson's lab.
export function NotebookCard({ nb, week }) {
  return (
    <aside className="nb-card" aria-label="Notebook for this lab">
      <div className="nb-text">
        <div className="nb-t">Do this lab in your own notebook</div>
        <p className="nb-p">
          Open the hands-on notebook in Google Colab. It runs on Google's computers, so there is nothing to install. Choose <strong>File → Save a copy in Drive</strong> first, so your changes are yours. You need a Google account.
        </p>
      </div>
      <div className="nb-actions">
        {nb.colabUrl ? (
          <a className="nb-btn" href={nb.colabUrl} target="_blank" rel="noopener noreferrer" aria-label={`Open the notebook for ${week} in Google Colab (opens in a new tab)`}>
            Open in Colab
          </a>
        ) : null}
        <a className={'nb-btn' + (nb.colabUrl ? ' ghost' : '')} href={apiUrl(nb.downloadUrl)} download>
          Download .ipynb
        </a>
      </div>
      {!nb.colabUrl ? <p className="nb-p small">To use Colab, go to colab.research.google.com, choose <strong>Upload</strong>, and select the file you downloaded.</p> : null}
    </aside>
  );
}

export function PlanStatus({ week, status }) {
  if (status === 'loading') return <div style={{ color: 'var(--text3)', padding: '24px 0' }} aria-busy="true">Loading lesson…</div>;
  return (
    <div className="wp-empty">
      <h2>Week {week} lesson not published yet</h2>
      <p>
        {status === 'error' ? (
          'The lesson could not be loaded. Check that the server is running and try again.'
        ) : (
          <>
            Add <code>week-{String(week).padStart(2, '0')}.md</code> to <code>backend/content</code> and refresh.
          </>
        )}
      </p>
    </div>
  );
}
