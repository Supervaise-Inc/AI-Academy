// Week 7 · Language Models & Attention (lesson built into frontend/src/pages/Lesson.jsx, Figure 3)
export default {
  week: 7,
  tagline: 'Every token looks at every other token and takes a weighted average.',
  time: 'About 20 minutes of reading, then a 3-hour lab',
  ideas: [
    'Recurrent networks pass information step by step, and it degrades on the way.',
    'Each token becomes a query, a key and a value; queries are matched to keys.',
    'Scores are scaled by the square root of d_k, then softmax makes weights.',
  ],
  outcomes: [
    'Explain why recurrent networks lose information and how attention removes that.',
    'Read an attention-weight matrix, each row a distribution over the tokens.',
    'Compute scaled dot-product attention by hand for one query and three keys.',
  ],
  terms: [
    { t: 'Query', d: 'What a token is looking for in the rest of the sentence.' },
    { t: 'Key', d: 'What a token offers to be matched against other tokens’ queries.' },
    { t: 'Value', d: 'The content a token passes on once it has been matched.' },
    { t: 'Softmax', d: 'Turns any list of numbers into positive weights that sum to one.' },
    { t: 'Scaling by √d_k', d: 'Keeps scores in a range where softmax does not saturate.' },
    { t: 'Soft attention', d: 'Blends every value by its weight, so it can be trained by gradient descent.' },
  ],
  path: [
    { label: 'Read', min: 20 },
    { label: 'Lab', min: 180 },
  ],
  graphic: {
    type: 'grid',
    rows: [
      ['Key', 'Score', 'Scaled', 'Weight'],
      ['k1', '1', '0.707', '0.401'],
      ['k2', '0', '0', '0.198'],
      ['k3', '1', '0.707', '0.401'],
    ],
    hot: [[1, 3], [3, 3]],
    caption: 'Lab example, query (1, 0), d_k = 2: scale by √2, softmax, weights sum to 1.',
  },
};
