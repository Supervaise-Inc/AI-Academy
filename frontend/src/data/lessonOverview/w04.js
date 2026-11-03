// Week 4 · Convolutional Neural Networks (backend/content/week-04.md)
export default {
  week: 4,
  tagline: 'Slide one small filter over the whole picture instead of wiring every pixel.',
  time: 'About 30 minutes of reading, then a 3-hour lab',
  ideas: [
    'A fully connected layer on an image needs huge numbers of weights.',
    'One small kernel is reused at every position: 80 weights, not 78,500.',
    'A CNN stacks convolution, ReLU and pooling, then a final linear layer.',
  ],
  outcomes: [
    'Explain why fully connected layers fit images poorly.',
    'Compute a convolution by hand, with stride and padding.',
    'Build and train a small CNN and compare it with a fully connected network.',
  ],
  terms: [
    { t: 'Kernel', d: 'A small grid of weights that slides over the image.' },
    { t: 'Stride', d: 'How many pixels the kernel moves at each step.' },
    { t: 'Padding', d: 'A border of zeros added so the output keeps its size.' },
    { t: 'Weight sharing', d: 'The same kernel weights are used at every position.' },
    { t: 'Pooling', d: 'Shrinks each feature map by keeping the strongest value.' },
  ],
  path: [
    { label: 'Read', min: 30 },
    { label: 'Figures', min: 20 },
    { label: 'Lab', min: 180 },
    { label: 'Quiz', min: 20 },
  ],
  graphic: {
    type: 'flow',
    steps: [
      { label: 'Image', sub: '8×8 digit' },
      { label: 'Conv + ReLU', sub: '8 filters' },
      { label: 'Pooling', sub: 'half the size' },
      { label: 'Flatten', sub: '64 numbers' },
      { label: 'Linear', sub: '10 classes' },
    ],
    caption: 'The small CNN from the lab: the shape shrinks while the channels grow.',
  },
};
