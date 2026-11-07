// Week 10 · Midterm Exam, revision of Weeks 6 to 9 (backend/content/week-10.md)
export default {
  week: 10,
  exam: true,
  tagline: 'Part A: 40 questions in 60 minutes. Part B: five tasks in 120 minutes. 100 points.',
  time: 'Part A 60 minutes, Part B 120 minutes, plus your revision',
  ideas: [
    'Week 6: bigram and smoothed probabilities, perplexity, V^n, BPE, V*d, cosine.',
    'Week 7: gradient products, LSTM gates, scaled attention, the n x n cost.',
    'Weeks 8-9: Transformer counts, mask, KV cache, MoE, C = 6ND, Chinchilla.',
  ],
  outcomes: [
    'Recall the formulas of Weeks 6 to 9 and run them on small numbers.',
    'Explain why each design exists: sub-words, attention, scaling, masks.',
    'Apply the right model family, scale rule or prompt technique to a problem.',
  ],
  terms: [
    { t: 'Perplexity', d: 'P^(-1/n): average confusion per token; lower is better.' },
    { t: 'Scaled attention', d: 'softmax(QK^T / sqrt(d_k)) V: score, scale, softmax, mix.' },
    { t: 'Causal mask', d: 'Hides later tokens; trains every position in one pass.' },
    { t: 'Chinchilla rule', d: 'N = sqrt(C/120) and D = 20N for a compute budget C.' },
    { t: 'Standard error', d: 'sqrt(p(1-p)/n): how noisy an accuracy from n questions is.' },
  ],
  path: [
    { label: 'Day 1: Wk 6-7', min: 90 },
    { label: 'Day 2: Wk 8', min: 90 },
    { label: 'Day 3: Wk 9', min: 90 },
    { label: 'Day 4: all', min: 90 },
  ],
  graphic: {
    type: 'bars',
    unit: ' min',
    items: [
      { label: 'Task 1: Wk 6', value: 20 },
      { label: 'Task 2: Wk 7', value: 25 },
      { label: 'Task 3: Wk 8', value: 30 },
      { label: 'Task 4: Wk 9', value: 30 },
      { label: 'Task 5: mixed', value: 15 },
    ],
    caption: 'Part B time boxes add up to 120 minutes (60 points of 100).',
  },
};
