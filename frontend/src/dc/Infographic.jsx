import React from 'react';

// Draws a lesson infographic from a small data spec (see data/lessonOverview/index.js for the spec types).
// Everything is plain SVG coloured with the page's theme variables, so it follows light and dark mode.
const W = 320;
const H = 150;

// Font size that fits `text` in `w` pixels.
const fit = (text, w, max = 11, min = 7.5) => Math.max(min, Math.min(max, w / (Math.max(1, String(text).length) * 0.57)));

// Split a label into at most two balanced lines.
function lines(label) {
  const t = String(label);
  if (t.length <= 9 || !t.includes(' ')) return [t];
  const words = t.split(' ');
  let best = null;
  for (let i = 1; i < words.length; i++) {
    const a = words.slice(0, i).join(' '),
      b = words.slice(i).join(' ');
    const d = Math.abs(a.length - b.length);
    if (!best || d < best.d) best = { d, a, b };
  }
  return [best.a, best.b];
}

function Label({ x, y, text, w, cls = 'ig-t', max = 11, anchor = 'middle', bold }) {
  // One line when it fits at a readable size, otherwise two balanced lines.
  const ls = fit(text, w, max) >= 8.8 ? [String(text)] : lines(text);
  const fs = fit(ls.reduce((a, b) => (b.length > a.length ? b : a), ''), w, max);
  const gap = fs * 1.12;
  const top = y - ((ls.length - 1) * gap) / 2;
  return (
    <text className={cls} x={x} y={top} fontSize={fs} textAnchor={anchor} dominantBaseline="central" fontWeight={bold ? 700 : 600}>
      {ls.map((l, i) => (
        <tspan key={i} x={x} dy={i === 0 ? 0 : gap}>
          {l}
        </tspan>
      ))}
    </text>
  );
}

const Arrow = ({ x1, y1, x2, y2 }) => {
  const a = Math.atan2(y2 - y1, x2 - x1),
    s = 5;
  const p = [
    [x2, y2],
    [x2 - s * Math.cos(a - 0.5), y2 - s * Math.sin(a - 0.5)],
    [x2 - s * Math.cos(a + 0.5), y2 - s * Math.sin(a + 0.5)],
  ];
  return (
    <g className="ig-arrow">
      <line x1={x1} y1={y1} x2={x2 - 3 * Math.cos(a)} y2={y2 - 3 * Math.sin(a)} />
      <polygon points={p.map((q) => q.join(',')).join(' ')} />
    </g>
  );
};

function Flow({ steps, compact }) {
  const n = steps.length,
    gap = 16,
    pad = 8;
  const w = (W - pad * 2 - gap * (n - 1)) / n;
  const h = compact ? 50 : 54;
  const y = compact ? 75 : 62;
  return (
    <>
      {steps.map((s, i) => {
        const x = pad + i * (w + gap);
        return (
          <g key={i}>
            <rect className={'ig-box' + (i === n - 1 ? ' hot' : '')} x={x} y={y - h / 2} width={w} height={h} rx="9" />
            <text className="ig-num" x={x + 7} y={y - h / 2 + 9} fontSize="7.5">
              {i + 1}
            </text>
            <Label x={x + w / 2} y={y + 2} text={s.label} w={w - 8} max={11.5} bold />
            {!compact && s.sub ? <Label x={x + w / 2} y={y + h / 2 + 18} text={s.sub} w={w + gap - 2} cls="ig-s" max={9.5} /> : null}
            {i < n - 1 ? <Arrow x1={x + w + 2} y1={y} x2={x + w + gap - 2} y2={y} /> : null}
          </g>
        );
      })}
    </>
  );
}

function Cycle({ steps }) {
  const n = steps.length,
    cx = W / 2,
    cy = H / 2,
    rx = 112,
    ry = 46;
  const at = (i) => {
    const t = (-90 + (360 / n) * i) * (Math.PI / 180);
    return { x: cx + rx * Math.cos(t), y: cy + ry * Math.sin(t), t };
  };
  const nodes = steps.map((s, i) => ({ s, ...at(i) }));
  const mids = steps.map((_, i) => {
    const t = (-90 + (360 / n) * (i + 0.5)) * (Math.PI / 180);
    const x = cx + rx * Math.cos(t),
      y = cy + ry * Math.sin(t);
    const dx = -rx * Math.sin(t),
      dy = ry * Math.cos(t),
      len = Math.hypot(dx, dy);
    return { x, y, ux: dx / len, uy: dy / len };
  });
  const bw = n > 4 ? 64 : 76,
    bh = 28;
  return (
    <>
      <ellipse className="ig-ring" cx={cx} cy={cy} rx={rx} ry={ry} />
      {mids.map((m, i) => (
        <polygon
          key={'m' + i}
          className="ig-chev"
          points={[
            [m.x + m.ux * 6, m.y + m.uy * 6],
            [m.x - m.ux * 4 - m.uy * 5, m.y - m.uy * 4 + m.ux * 5],
            [m.x - m.ux * 4 + m.uy * 5, m.y - m.uy * 4 - m.ux * 5],
          ]
            .map((q) => q.join(','))
            .join(' ')}
        />
      ))}
      {nodes.map((o, i) => (
        <g key={i}>
          <rect className={'ig-box' + (i === 0 ? ' hot' : '')} x={o.x - bw / 2} y={o.y - bh / 2} width={bw} height={bh} rx="8" />
          <Label x={o.x} y={o.y + 1} text={o.s.label} w={bw - 8} max={10.5} bold />
        </g>
      ))}
    </>
  );
}

function Layers({ items, compact }) {
  const n = items.length;
  const h = Math.min(30, 128 / n),
    gap = 5,
    total = n * h + (n - 1) * gap,
    y0 = (H - total) / 2;
  return (
    <>
      {items.map((it, i) => {
        const y = y0 + i * (h + gap),
          inset = (n - 1 - i) * 0 + 0;
        return (
          <g key={i}>
            <rect className={'ig-box' + (i === 0 ? ' hot' : '')} x={20 + inset} y={y} width={W - 40 - inset * 2} height={h} rx="7" />
            <Label x={compact || !it.sub ? W / 2 : 34} y={y + h / 2 + 1} text={it.label} w={compact || !it.sub ? W - 80 : 150} max={11.5} bold anchor={compact || !it.sub ? 'middle' : 'start'} />
            {!compact && it.sub ? <Label x={W - 34} y={y + h / 2 + 1} text={it.sub} w={118} cls="ig-s" max={9.5} anchor="end" /> : null}
          </g>
        );
      })}
    </>
  );
}

function Bars({ items, unit = '' }) {
  const n = items.length,
    max = Math.max(...items.map((i) => Number(i.value) || 0), 1);
  const h = Math.min(22, 118 / n),
    gap = Math.min(10, (128 - n * h) / Math.max(1, n - 1)),
    y0 = (H - (n * h + (n - 1) * gap)) / 2;
  return (
    <>
      {items.map((it, i) => {
        const y = y0 + i * (h + gap),
          bw = Math.max(4, ((W - 150) * (Number(it.value) || 0)) / max);
        return (
          <g key={i}>
            <Label x={100} y={y + h / 2 + 1} text={it.label} w={88} anchor="end" max={10.5} />
            <rect className="ig-track" x={108} y={y} width={W - 150} height={h} rx={h / 2} />
            <rect className={'ig-bar' + (i === items.length - 1 ? ' hot' : '')} x={108} y={y} width={bw} height={h} rx={h / 2} />
            <text className="ig-t" x={108 + bw + 6 > W - 40 ? 108 + bw - 6 : 108 + bw + 6} y={y + h / 2 + 1} fontSize="10" fontWeight="700" dominantBaseline="central" textAnchor={108 + bw + 6 > W - 40 ? 'end' : 'start'} fill={108 + bw + 6 > W - 40 ? '#fff' : undefined}>
              {it.value}
              {unit}
            </text>
          </g>
        );
      })}
    </>
  );
}

function Compare({ left, right }) {
  const bw = 148,
    side = (o, x, hot) => (
      <g>
        <rect className={'ig-box' + (hot ? ' hot' : '')} x={x} y={10} width={bw} height={130} rx="10" />
        <Label x={x + bw / 2} y={28} text={o.title} w={bw - 16} max={12} bold />
        <line className="ig-rule" x1={x + 12} y1={42} x2={x + bw - 12} y2={42} />
        {(o.points || []).map((p, i) => (
          <g key={i}>
            <circle className="ig-dot" cx={x + 14} cy={62 + i * 26} r="2.6" />
            <Label x={x + 22} y={62 + i * 26} text={p} w={bw - 30} anchor="start" cls="ig-s2" max={9.6} />
          </g>
        ))}
      </g>
    );
  return (
    <>
      {side(left, 6, false)}
      {side(right, W - bw - 6, true)}
      <text className="ig-s" x={W / 2} y={H / 2} fontSize="9" textAnchor="middle" dominantBaseline="central">
        vs
      </text>
    </>
  );
}

function Grid({ rows, hot = [] }) {
  const r = rows.length,
    c = Math.max(...rows.map((x) => x.length));
  const cell = Math.min(30, 124 / r, 280 / c),
    gx = 3;
  const x0 = (W - (c * cell + (c - 1) * gx)) / 2,
    y0 = (H - (r * cell + (r - 1) * gx)) / 2;
  const isHot = (i, j) => hot.some((h) => h[0] === i && h[1] === j);
  return (
    <>
      {rows.map((row, i) =>
        row.map((v, j) => (
          <g key={i + '-' + j}>
            <rect className={'ig-box' + (isHot(i, j) ? ' hot' : '')} x={x0 + j * (cell + gx)} y={y0 + i * (cell + gx)} width={cell} height={cell} rx="5" />
            <Label x={x0 + j * (cell + gx) + cell / 2} y={y0 + i * (cell + gx) + cell / 2 + 1} text={String(v)} w={cell - 4} max={10} bold={isHot(i, j)} />
          </g>
        )),
      )}
    </>
  );
}

function Curve({ series, xLabel, yLabel }) {
  const all = series.flatMap((s) => s.points);
  const xs = all.map((p) => p[0]),
    ys = all.map((p) => p[1]);
  const x0 = Math.min(...xs),
    x1 = Math.max(...xs),
    y0 = Math.min(...ys),
    y1 = Math.max(...ys);
  const L = 36,
    R = 308,
    T = 26,
    B = 118;
  const px = (x) => L + ((x - x0) / (x1 - x0 || 1)) * (R - L);
  const py = (y) => B - ((y - y0) / (y1 - y0 || 1)) * (B - T);
  return (
    <>
      <line className="ig-axis" x1={L} y1={T} x2={L} y2={B} />
      <line className="ig-axis" x1={L} y1={B} x2={R} y2={B} />
      {[0.25, 0.5, 0.75].map((f) => (
        <line key={f} className="ig-gridline" x1={L} y1={B - f * (B - T)} x2={R} y2={B - f * (B - T)} />
      ))}
      {series.map((s, k) => (
        <g key={k}>
          <polyline className={'ig-line k' + k} points={s.points.map((p) => px(p[0]) + ',' + py(p[1])).join(' ')} />
          {s.points.map((p, i) => (
            <circle key={i} className={'ig-pt k' + k} cx={px(p[0])} cy={py(p[1])} r="2.4" />
          ))}
        </g>
      ))}
      <text className="ig-s" x={(L + R) / 2} y={H - 12} fontSize="9" textAnchor="middle">
        {xLabel}
      </text>
      <text className="ig-s" x={10} y={(T + B) / 2} fontSize="9" textAnchor="middle" transform={`rotate(-90 10 ${(T + B) / 2})`}>
        {yLabel}
      </text>
      {series.length > 1 || series[0].label
        ? series.map((sr, k) => (
            <g key={'l' + k}>
              <rect className={'ig-key k' + k} x={L + k * 118} y={8} width="10" height="3.5" rx="1.7" />
              <text className="ig-s2" x={L + 14 + k * 118} y={10} fontSize="9.5" dominantBaseline="central">
                {sr.label}
              </text>
            </g>
          ))
        : null}
    </>
  );
}

const BODY = { flow: Flow, cycle: Cycle, layers: Layers, bars: Bars, compare: Compare, grid: Grid, curve: Curve };

export default function Infographic({ spec, compact = false, className = '' }) {
  if (!spec || !BODY[spec.type]) return null;
  const Body = BODY[spec.type];
  return (
    <figure className={'ig ' + className}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={spec.caption || 'Lesson infographic'} preserveAspectRatio="xMidYMid meet">
        <Body {...spec} compact={compact} />
      </svg>
      {!compact && spec.caption ? <figcaption>{spec.caption}</figcaption> : null}
    </figure>
  );
}
