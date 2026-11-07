// Week 6 · NLP: Tokens & Word Embeddings (backend/content/week-06.md)
export default {
  week: 6,
  tagline: 'A language model scores the next token; text becomes tokens, and tokens become vectors.',
  time: 'About 60 minutes of reading, then a 3-hour lab',
  ideas: [
    'Counts give P(next | previous), but V^n n-grams means most are never seen.',
    'Sub-word tokens sit between words and characters; BPE learns them by merging.',
    'An embedding row E[id] is a learned vector; cosine similarity compares two.',
  ],
  outcomes: [
    'Compute a bigram probability, a sentence probability and a perplexity by hand.',
    'Compare word, character and sub-word tokens, and run BPE merges by hand.',
    'Compute a cosine similarity and explain what an embedding lookup does.',
  ],
  terms: [
    { t: 'Language model', d: 'Gives the probability of the next token, given the tokens before it.' },
    { t: 'Perplexity', d: 'P^(-1/n): the average confusion per token. Lower means a better fit.' },
    { t: 'Byte-pair encoding', d: 'Repeatedly merges the most frequent neighbouring pair into one symbol.' },
    { t: 'Embedding', d: 'A learned vector for a token: one row of a V x d table.' },
    { t: 'Cosine similarity', d: 'A number from -1 to 1 for how much two vectors point the same way.' },
  ],
  path: [
    { label: 'Read', min: 60 },
    { label: 'N-grams', min: 70 },
    { label: 'Tokens, BPE', min: 55 },
    { label: 'Embeddings', min: 30 },
    { label: 'Setup, design', min: 25 },
  ],
  graphic: {
    type: 'bars',
    unit: '%',
    items: [
      { label: 'students', value: 44 },
      { label: 'canteen', value: 33 },
      { label: 'teacher', value: 11 },
      { label: 'counter', value: 11 },
    ],
    caption: 'P(next word | "the") from counts in a 45-word text: "the" came 9 times.',
  },
};
