// Week 5 · Prelim examination, revision of Weeks 1 to 4 (backend/content/week-05.md)
export default {
  week: 5,
  exam: true,
  tagline: 'Revise Weeks 1 to 4 and practise the calculations you may do by hand.',
  time: 'About 30 minutes, then your study plan',
  ideas: [
    'Weeks 1 and 2: generative models, neurons and gradient descent.',
    'Week 3: backpropagation is the chain rule, and sigmoid gradients vanish.',
    'Week 4: convolution gives local connectivity and weight sharing.',
  ],
  outcomes: [
    'Recall the key AI dates and the difference between generative and discriminative models.',
    'Explain why deep sigmoid networks lose the gradient and what helps.',
    'Apply gradient descent, the chain rule and convolution formulas by hand.',
  ],
  terms: [
    { t: 'Loss', d: 'A number that says how wrong the model is; training lowers it with gradient descent.' },
    { t: 'Gradient descent', d: 'New weight = weight minus learning rate times gradient.' },
    { t: 'Chain rule', d: 'Backpropagation multiplies local derivatives along the path from the loss.' },
    { t: 'Vanishing gradient', d: 'Sigmoid slopes are at most 0.25, so deep stacks shrink the gradient.' },
    { t: 'Weight sharing', d: 'One kernel is reused at every position in the image.' },
  ],
  path: [
    { label: 'Basics', min: 20 },
    { label: 'Figures', min: 30 },
    { label: 'Paper sums', min: 45 },
    { label: 'Mock exam', min: 45 },
    { label: 'Labs and quiz', min: 45 },
  ],
  graphic: {
    type: 'flow',
    steps: [
      { label: 'Week 1', sub: 'AI and language' },
      { label: 'Week 2', sub: 'Neurons, descent' },
      { label: 'Week 3', sub: 'Backprop, gradients' },
      { label: 'Week 4', sub: 'Convolution' },
    ],
    caption: 'The Prelim is cumulative: each week builds on the one before it.',
  },
};
