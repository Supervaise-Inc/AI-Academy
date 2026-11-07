// Week 5 · Prelim examination, revision of Weeks 1 to 4 (backend/content/week-05.md)
export default {
  week: 5,
  exam: true,
  tagline: 'Part A is 40 questions, Part B is 120 minutes of notebook tasks, 100 points in all.',
  time: 'Part A 60 minutes, Part B 120 minutes',
  ideas: [
    'Week 1: temperature softmax, chain rule, perplexity, top-k, top-p, diffusion.',
    'Weeks 2 and 3: loss, neuron, parameters, gradient descent, δ equations, fⁿ.',
    'Week 4: convolution, output size, parameters, receptive field, dL/dK and dL/dx.',
  ],
  outcomes: [
    'Recall the formula sheet for Weeks 1 to 4 and the rules of the exam.',
    'Explain why gradients vanish or explode, and why a shared kernel needs fewer weights.',
    'Apply the formulas by hand, and diagnose a model from its numbers.',
  ],
  terms: [
    { t: 'Part A', d: '40 multiple-choice questions, 1 point each, closed book, 60 minutes.' },
    { t: 'Part B', d: 'Five notebook tasks worth 60 points, with time boxes adding to 120 minutes.' },
    { t: 'Pass mark', d: '70 points out of 100.' },
    { t: 'Formula sheet', d: 'Every formula of Weeks 1 to 4; learn it, as Part A is closed book.' },
    { t: 'Time box', d: 'The minutes set for each Part B task, 15 to 30; keep to them.' },
  ],
  path: [
    { label: 'Day 1: Wk 1-2', min: 90 },
    { label: 'Day 2: Week 3', min: 90 },
    { label: 'Day 3: Week 4', min: 90 },
    { label: 'Day 4: All', min: 90 },
  ],
  graphic: {
    type: 'bars',
    unit: ' Qs',
    items: [
      { label: 'Week 1', value: 8 },
      { label: 'Week 2', value: 10 },
      { label: 'Week 3', value: 11 },
      { label: 'Week 4', value: 11 },
    ],
    caption: 'Part A has 40 questions, so no week can hide. Part B adds 60 points.',
  },
};
