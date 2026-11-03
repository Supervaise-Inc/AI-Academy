// Demo cohort for the Faculty Console. Not real learners: the console is labelled "Demo data".
// The first eight learners are the original sample roster (Section A). The rest are generated from a fixed seed,
// so the numbers are the same on every load. Instructors assign sections by hand, so a few learners are unassigned.

export const UNASSIGNED = 'none';
export const SECTIONS = [
  { id: 'A', name: 'Section A', room: 'Room CS-204', size: 28 },
  { id: 'B', name: 'Section B', room: 'Room CS-206', size: 31 },
  { id: 'C', name: 'Section C', room: 'Room CS-210', size: 29 },
  { id: 'D', name: 'Section D', room: 'Room CS-212', size: 26 },
];
const UNASSIGNED_COUNT = 6;

export const sectionName = (id) => (id === UNASSIGNED ? 'Unassigned' : (SECTIONS.find((s) => s.id === id) || {}).name || 'Section ' + id);

const ORIGINAL = [
  { id: 'kb', name: 'K. Bautista', when: '13 Oct, 20:41 PHT', sub: true, preset: [0, 0, 0, 1, 0, 0, 0], graded: true, prog: 34, labs: 6, kc: 6, prelim: 'Passed', status: 'on', active: 0 },
  { id: 'lg', name: 'L. Garcia', when: '13 Oct, 22:05 PHT', sub: true, preset: [0, 1, 0, 0, 1, 0, 0], graded: true, prog: 32, labs: 6, kc: 6, prelim: 'Passed', status: 'on', active: 1 },
  { id: 'am', name: 'A. Mendoza', when: '14 Oct, 09:12 PHT', sub: true, preset: null, graded: false, prog: 33, labs: 6, kc: 6, prelim: 'Passed', status: 'on', active: 0 },
  { id: 'pr', name: 'P. Reyes', when: '14 Oct, 14:30 PHT', sub: true, preset: null, graded: false, prog: 30, labs: 6, kc: 4, prelim: 'Passed', status: 'support', flag: 'Knowledge check mean 54%', active: 2 },
  { id: 'dv', name: 'D. Villanueva', when: '14 Oct, 23:58 PHT', sub: true, preset: null, graded: false, prog: 31, labs: 5, kc: 6, prelim: 'Passed', status: 'watch', flag: '1 laboratory late', active: 1 },
  { id: 'rd', name: 'R. Dela Cruz', when: '15 Oct, 07:45 PHT', sub: true, preset: null, graded: false, prog: 29, labs: 6, kc: 5, prelim: 'Retake', status: 'watch', flag: 'Preliminary exam retake', active: 0 },
  { id: 'ms', name: 'M. Santos', when: 'Not submitted · due 16 Oct, 11:59 PM PHT', sub: false, prog: 32, labs: 5, kc: 6, prelim: 'Passed', status: 'on', active: 3 },
  { id: 'ja', name: 'J. Alcantara', when: 'Not submitted · due 16 Oct, 11:59 PM PHT', sub: false, prog: 27, labs: 4, kc: 5, prelim: 'Passed', status: 'support', flag: '2 laboratories late', active: 6 },
];

const FIRST = 'Aaron Bea Carlo Dana Elijah Faith Gabriel Hannah Ian Jasmine Kyle Liza Marco Nina Oscar Paula Quinn Rafael Sofia Tomas Uma Vince Wendy Xavier Yna Zach Alyssa Bryan Camille Dennis Eliza Francis Grace Harold Isabel Jericho Kaye Lance Mika Noel Olivia Patrick Rhea Sean Trisha Ulysses Vanessa Warren Yvonne'.split(' ');
const LAST = 'Abad Aguilar Bernardo Castillo Domingo Enriquez Fernandez Gonzales Hernandez Ignacio Javier Laurel Magsaysay Navarro Ocampo Pascual Quizon Ramos Salazar Tan Ubaldo Valdez Yap Zamora Aquino Buenaventura Cruz Del Rosario Evangelista Flores'.split(' ');

function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const initials = (n) =>
  n
    .replace(/\./g, '')
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((p) => p[0].toUpperCase())
    .join('');

function classify(s) {
  if (s.kc <= 4) return { status: 'support', flag: 'Knowledge check mean ' + (40 + s.kc * 3) + '%' };
  if (s.labs <= 4) return { status: 'support', flag: s.labs === 4 ? '2 laboratories late' : '3 laboratories late' };
  if (s.active >= 7) return { status: 'watch', flag: 'No activity for ' + s.active + ' days' };
  if (s.labs === 5) return { status: 'watch', flag: '1 laboratory late' };
  if (s.prelim === 'Retake') return { status: 'watch', flag: 'Preliminary exam retake' };
  return { status: 'on' };
}

function build() {
  const rand = rng(20261014);
  const pick = (n) => Math.floor(rand() * n);
  const used = new Set(ORIGINAL.map((s) => s.name));
  const out = ORIGINAL.map((s) => ({ ...s, section: 'A' }));
  const make = (section, n) => {
    for (let i = 0; i < n; i++) {
      let name;
      do name = FIRST[pick(FIRST.length)][0] + '. ' + LAST[pick(LAST.length)];
      while (used.has(name));
      used.add(name);
      const r = rand();
      const labs = r < 0.07 ? 4 : r < 0.2 ? 5 : 6;
      const kc = rand() < 0.08 ? 3 + pick(2) : rand() < 0.25 ? 5 : 6 + (rand() < 0.4 ? 1 : 0);
      const active = rand() < 0.12 ? 7 + pick(6) : pick(4);
      const sub = rand() < 0.9;
      const graded = sub && rand() < 0.35;
      const day = 13 + pick(4),
        hh = String(7 + pick(16)).padStart(2, '0'),
        mm = String(pick(60)).padStart(2, '0');
      const s = {
        id: section + '-' + String(i + 1).padStart(2, '0') + '-' + out.length,
        name,
        section,
        when: sub ? day + ' Oct, ' + hh + ':' + mm + ' PHT' : 'Not submitted · due 16 Oct, 11:59 PM PHT',
        sub,
        graded,
        preset: graded ? Array.from({ length: 7 }, () => (rand() < 0.65 ? 0 : rand() < 0.7 ? 1 : 2)) : null,
        prog: Math.max(14, Math.min(40, 36 - (6 - labs) * 3 - (active > 6 ? 5 : 0) + pick(5) - 2)),
        labs,
        kc,
        prelim: rand() < 0.07 ? 'Retake' : 'Passed',
        active,
      };
      Object.assign(s, classify(s));
      out.push(s);
    }
  };
  SECTIONS.forEach((sec) => make(sec.id, sec.id === 'A' ? sec.size - ORIGINAL.length : sec.size));
  make(UNASSIGNED, UNASSIGNED_COUNT);
  return out.map((s) => ({
    ...s,
    ini: initials(s.name),
    labsN: s.labs,
    kcN: s.kc,
    labs: s.labs + '/6',
    kc: s.kc + '/7',
  }));
}

export const STUDENTS = build();

export const lastActive = (d) => (d === 0 ? 'Today' : d === 1 ? 'Yesterday' : d + ' days ago');
export const STATUS_RANK = { support: 0, watch: 1, on: 2 };
