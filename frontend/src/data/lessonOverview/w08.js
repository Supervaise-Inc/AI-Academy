// Week 8 · The Transformer and Autoregressive Training (backend/content/week-08.md)
export default {
  week: 8,
  tagline: 'One block, repeated: attention mixes tokens, a feed-forward net transforms each one.',
  time: 'About 60 minutes of reading, then a 3-hour lab',
  ideas: [
    'A block keeps the shape (n, d_model), so identical blocks stack into a model.',
    'Attention cannot see order, so a positional encoding is added to each embedding.',
    'A causal mask and targets shifted by one train every position in one pass.',
  ],
  outcomes: [
    'Name the parts of a Transformer block and say what each one does.',
    'Compute a sinusoidal encoding, layer norm and a residual by hand.',
    'Count the parameters of an attention layer and of a small Transformer.',
  ],
  terms: [
    { t: 'Positional encoding', d: 'A vector per position, added to the embedding to give order.' },
    { t: 'Multi-head attention', d: 'Splits d_model into h heads; the layer has 4d^2 + 4d parameters.' },
    { t: 'Residual connection', d: 'y = x + F(x): gives the gradient a path with slope 1.' },
    { t: 'Layer norm', d: 'Rescales each token vector to a steady mean and spread.' },
    { t: 'Causal mask', d: 'Hides later tokens so no position can peek at its own answer.' },
  ],
  path: [
    { label: 'Read', min: 60 },
    { label: 'Pos., heads', min: 60 },
    { label: 'Mask, norm', min: 50 },
    { label: 'Tiny GPT', min: 55 },
    { label: 'Setup, scale', min: 25 },
  ],
  graphic: {
    type: 'layers',
    items: [
      { label: 'Output + softmax', sub: 'shares the token table' },
      { label: 'Final norm', sub: '1,536 parameters' },
      { label: '12 blocks', sub: '85,054,464 parameters' },
      { label: 'Positions', sub: '786,432 parameters' },
      { label: 'Token table', sub: '38,597,376 parameters' },
    ],
    caption: 'GPT-2 small: 124,439,808 parameters, 7,087,872 in each block.',
  },
};
