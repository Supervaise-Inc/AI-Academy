// Week 4 · Convolutional Neural Networks (backend/content/week-04.md)
export default {
  week: 4,
  tagline: 'One small shared filter slides over the picture, so a CNN needs far fewer weights.',
  time: 'About 60 minutes of reading, then a 3-hour lab',
  ideas: [
    'Dense layers fit pictures poorly: too many weights, and the layout is ignored.',
    'A small kernel looks locally and is shared, so it finds a pattern anywhere.',
    'Output size is ⌊(n + 2p − k) / s⌋ + 1; backward is the chain rule again.',
  ],
  outcomes: [
    'Explain why dense layers fit images poorly, and what local connectivity fixes.',
    'Compute a convolution, its output size and its parameter count by hand.',
    'Compute dL/dK and dL/dx by hand, and trace shapes through a small CNN.',
  ],
  terms: [
    { t: 'Kernel', d: 'A small grid of weights that slides over the picture, summing x · K + b.' },
    { t: 'Feature map', d: 'The grid of outputs one filter makes as it slides over the input.' },
    { t: 'Stride and padding', d: 'Step size of the window, and a border added to keep the size.' },
    { t: 'Pooling', d: 'Summarises each neighbourhood, e.g. its maximum, giving a smaller map.' },
    { t: 'Receptive field', d: 'How much of the picture one unit can see.' },
    { t: 'Weight sharing', d: 'One kernel reused at every position, so parameters do not grow with size.' },
  ],
  path: [
    { label: 'Read', min: 60 },
    { label: 'Lab 1 and 2', min: 60 },
    { label: 'Lab 3 and 4', min: 55 },
    { label: 'Lab 5 to 7', min: 65 },
  ],
  graphic: {
    type: 'compare',
    left: {
      title: 'Dense layer',
      points: ['Unrolls the picture', '224×224×3 into 1,000 units', '150,529,000 parameters'],
    },
    right: {
      title: 'Convolution',
      points: ['16 shared 3×3 filters', '3·3·3·16 + 16 = 448', 'Same at any picture size'],
    },
    caption: 'Same colour picture: 150,529,000 weights versus 448 for the convolution.',
  },
};
