// Week 10 · Midterm Exam, revision of Weeks 6 to 9 (backend/content/week-10.md)
export default {
  week: 10,
  exam: true,
  tagline: 'Four weeks, one story: tokens, vectors, attention, the Transformer, then prompts.',
  time: 'About 30 minutes of review, then your own study time',
  ideas: [
    'Week 6: BPE merges, cosine similarity and the bigram model.',
    'Weeks 7 and 8: scaled attention, positional encoding, causal mask, the block.',
    'Week 9: BERT, GPT and T5, the Chinchilla rule, and prompting.',
  ],
  outcomes: [
    'Recall the formulas and run them on small numbers, by hand.',
    'Explain why each design exists: sub-words, attention, scaling, position, masks.',
    'Apply the right model family and prompting technique to a given problem.',
  ],
  terms: [
    { t: 'BPE', d: 'Builds sub-word pieces by repeatedly merging the most frequent pair.' },
    { t: 'Cosine similarity', d: 'A score from -1 to 1 for how closely two vectors point the same way.' },
    { t: 'Scaled attention', d: 'Score, scale by the square root of d_k, softmax, then mix the values.' },
    { t: 'Causal mask', d: 'Hides later tokens so each token sees only itself and earlier ones.' },
    { t: 'Chinchilla rule', d: 'About 20 training tokens per parameter, so N = square root of C / 120.' },
  ],
  path: [
    { label: 'Sort first', min: 15 },
    { label: 'Week 6', min: 30 },
    { label: 'Weeks 7-8', min: 60 },
    { label: 'Pos. encoding', min: 10 },
    { label: 'Week 9', min: 30 },
  ],
  graphic: {
    type: 'flow',
    steps: [
      { label: 'Tokens', sub: 'Week 6' },
      { label: 'Vectors', sub: 'Week 6' },
      { label: 'Attention', sub: 'Week 7' },
      { label: 'Transformer', sub: 'Week 8' },
      { label: 'Prompts', sub: 'Week 9' },
    ],
    caption: 'Text becomes tokens, tokens become vectors, attention mixes them, prompts steer.',
  },
};
