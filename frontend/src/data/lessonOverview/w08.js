// Week 8 · The Transformer & Autoregressive Training (backend/content/week-08.md)
export default {
  week: 8,
  tagline: 'Stack attention blocks, add word order, and train to predict the next token.',
  time: 'About 30 minutes of reading, then a 3-hour lab',
  ideas: [
    'Attention ignores order, so a positional encoding is added to the embeddings.',
    'A block is attention, add and normalise, feed-forward, add and normalise.',
    'Targets are shifted by one and a causal mask hides the future.',
  ],
  outcomes: [
    'Explain why a Transformer needs positional encoding and compute one value.',
    'Name the parts of a Transformer block in order and say what each does.',
    'Explain autoregressive training: shifted targets, the causal mask, teacher forcing.',
  ],
  terms: [
    { t: 'Positional encoding', d: 'A vector added to each embedding that tells the model where the token sits.' },
    { t: 'Multi-head attention', d: 'Several attention heads run in parallel, then are joined and mixed by W_O.' },
    { t: 'Residual connection', d: 'A shortcut that adds a layer’s input to its output, x + F(x).' },
    { t: 'Layer normalisation', d: 'Rescales each token’s vector to mean 0 and variance 1 for stable training.' },
    { t: 'Causal mask', d: 'Hides future tokens, so token t only attends to positions up to t.' },
    { t: 'Teacher forcing', d: 'Feeds the correct previous tokens as input so all positions train at once.' },
  ],
  path: [
    { label: 'Read', min: 30 },
    { label: 'Positions', min: 20 },
    { label: 'Attn and block', min: 70 },
    { label: 'Train', min: 40 },
    { label: 'Generate', min: 40 },
  ],
  graphic: {
    type: 'layers',
    items: [
      { label: 'Embed + position', sub: 'tokens plus order' },
      { label: 'Attention', sub: 'heads, then W_O' },
      { label: 'Add and norm', sub: 'residual shortcut' },
      { label: 'Feed-forward', sub: 'each token alone' },
      { label: 'Add and norm', sub: 'same shape out' },
    ],
    caption: 'One Transformer block, in order from input to output. Shapes in, same out.',
  },
};
