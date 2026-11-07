// Week 3 · Deep Learning and Backpropagation (backend/content/week-03.md)
export default {
  week: 3,
  tagline: 'Backpropagation sends blame backwards with the chain rule; deep stacks can fade it.',
  time: 'About 60 minutes of reading, then a 3-hour lab',
  ideas: [
    'Depth builds hierarchies: simple features combine into complex ones.',
    'Backprop is the chain rule: local derivative times the gradient from above.',
    'A product of many factors shrinks (0.25¹⁰ ≈ 10⁻⁶) or grows (1.5¹⁰ ≈ 58).',
  ],
  outcomes: [
    'Explain why depth helps, with an example and a hierarchy of features.',
    'Compute a forward and backward pass by hand using the chain rule and δ equations.',
    'Explain vanishing and exploding gradients, and name the remedy for each.',
  ],
  terms: [
    { t: 'Forward pass', d: 'Computes z = Wa + b and a = φ(z) layer by layer and stores them.' },
    { t: 'Chain rule', d: 'Multiply the local derivatives along the path from the loss.' },
    { t: 'δ (blame)', d: 'How much a unit is responsible for the loss; sent back layer by layer.' },
    { t: 'Vanishing gradient', d: 'Factors below 1 shrink the gradient; sigmoid slope is at most 0.25.' },
    { t: 'Exploding gradient', d: 'Factors above 1 grow the gradient, giving huge weights or NaN.' },
    { t: 'Residual connection', d: 'A skip path that adds the input back, keeping the factor near 1.' },
  ],
  path: [
    { label: 'Read', min: 60 },
    { label: 'Lab 1 and 2', min: 55 },
    { label: 'Lab 3 and 4', min: 65 },
    { label: 'Lab 5 and 6', min: 55 },
    { label: 'Lab 7', min: 25 },
  ],
  graphic: {
    type: 'curve',
    xLabel: 'Layers back',
    yLabel: 'log10 gradient left',
    series: [
      {
        label: 'f = 0.25 sigmoid',
        points: [
          [0, 0],
          [5, -3.01],
          [10, -6.02],
          [15, -9.03],
          [20, -12.04],
        ],
      },
      {
        label: 'f = 0.9',
        points: [
          [0, 0],
          [5, -0.23],
          [10, -0.46],
          [15, -0.69],
          [20, -0.92],
        ],
      },
    ],
    caption: 'Sigmoid, f = 0.25: after 10 layers only 0.25¹⁰ ≈ one millionth is left.',
  },
};
