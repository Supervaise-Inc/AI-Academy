// Week 6 · NLP: Tokens & Word Embeddings (backend/content/week-06.md)
export default {
  week: 6,
  tagline: 'Split text into tokens, then give every token a vector a model can compute with.',
  time: 'About 30 minutes of reading, then a 3-hour lab',
  ideas: [
    'A bigram model scores the next word from counts, but sees only one word back.',
    'Byte-pair encoding builds sub-word tokens, so new words can still be read.',
    'Embeddings turn tokens into vectors; similar meaning points the same way.',
  ],
  outcomes: [
    'Build a bigram model from counts and use it to generate text.',
    'Explain why models use sub-word tokens and learn BPE merges by hand.',
    'Explain word embeddings, cosine similarity, analogies and their limits.',
  ],
  terms: [
    { t: 'Token', d: 'One piece of text a model reads as a unit: a word, part of a word, or a character.' },
    { t: 'N-gram model', d: 'Estimates the next word from counts of the last n-1 words in training text.' },
    { t: 'Byte-pair encoding', d: 'Repeatedly merges the most frequent neighbouring pair into a new symbol.' },
    { t: 'Embedding', d: 'A vector of numbers for a token; similar uses give similar vectors.' },
    { t: 'Cosine similarity', d: 'A number from -1 to 1 for how much two vectors point the same way.' },
  ],
  path: [
    { label: 'Read', min: 30 },
    { label: 'N-grams, BPE', min: 80 },
    { label: 'Tokenisers', min: 35 },
    { label: 'Embeddings', min: 45 },
    { label: 'Reflect', min: 20 },
  ],
  graphic: {
    type: 'flow',
    steps: [
      { label: 'Text', sub: 'unbelievably' },
      { label: 'Tokens', sub: 'un, believ, ably' },
      { label: 'Vectors', sub: 'one per token' },
      { label: 'Cosine', sub: 'compare meaning' },
    ],
    caption: 'Text becomes tokens, tokens become vectors. The split is an example.',
  },
};
