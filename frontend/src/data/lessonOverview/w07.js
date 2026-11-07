// Week 7 · Language Models and Attention (backend/content/week-07.md)
export default {
  week: 7,
  tagline: 'Recurrent models forget over long spans; attention lets every word look straight back.',
  time: 'About 60 minutes of reading, then a 3-hour lab',
  ideas: [
    'An RNN reuses one state, and its gradient is a product of factors below 1.',
    'An LSTM adds a cell, so forget gates (0.95) keep far more signal than an RNN.',
    'Attention is softmax(QK^T / sqrt(d_k)) V: score, softmax, weighted sum.',
  ],
  outcomes: [
    'Explain how an RNN reads a sequence and why its gradient shrinks.',
    'Explain what LSTM gates add and the encoder-decoder bottleneck.',
    'Compute attention scores, softmax weights and a weighted output by hand.',
  ],
  terms: [
    { t: 'RNN', d: 'Reads one token at a time, updating a single hidden state with shared weights.' },
    { t: 'LSTM', d: 'An RNN with gates and a cell state that protect memory over long spans.' },
    { t: 'Attention', d: 'A weighted mix of all earlier states; the path between positions is 1 step.' },
    { t: 'Soft vs hard', d: 'Soft attention averages and can be trained; hard attention picks one.' },
    { t: 'Scaling by sqrt(d_k)', d: 'Keeps scores steady so the softmax does not saturate.' },
  ],
  path: [
    { label: 'Read', min: 60 },
    { label: 'RNN, LSTM', min: 100 },
    { label: 'Attention', min: 55 },
    { label: 'Hard, design', min: 40 },
    { label: 'Setup', min: 10 },
  ],
  graphic: {
    type: 'curve',
    xLabel: 'steps back',
    yLabel: 'signal left',
    series: [
      { label: 'forget gate 0.95', points: [[1, 0.95], [5, 0.774], [10, 0.599], [20, 0.358], [29, 0.226]] },
      { label: 'forget gate 0.9', points: [[1, 0.9], [5, 0.59], [10, 0.349], [20, 0.122], [29, 0.047]] },
    ],
    caption: 'A plain RNN keeps 3.6e-11 over 30 steps; attention keeps 1.',
  },
};
