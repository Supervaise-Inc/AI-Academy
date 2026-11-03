// Week 2 · Machine Learning & Neural Networks (backend/content/week-02.md)
export default {
  week: 2,
  tagline: 'A neuron guesses, a loss measures the mistake, and gradient descent fixes it.',
  time: 'About 30 minutes of reading, then a 3-hour lab',
  ideas: [
    'Supervised learning trains on labelled examples and is judged on unseen data.',
    'One neuron draws one straight line, so it cannot solve XOR.',
    'Training is gradient descent: step each parameter against its gradient.',
  ],
  outcomes: [
    'Explain supervised learning, loss, and the training, validation and test sets.',
    'Compute a neuron output and a mean squared error by hand.',
    'Train a two-layer network in PyTorch that solves a noisy XOR problem.',
  ],
  terms: [
    { t: 'Feature and label', d: 'The input the model is given, and the correct answer for it.' },
    { t: 'Loss function', d: 'A number that measures how far the predictions are from the labels.' },
    { t: 'Overfitting', d: 'Fitting the training examples so closely that new data does worse.' },
    { t: 'Bias', d: 'A number added to the weighted sum that shifts when the neuron switches on.' },
    { t: 'Learning rate', d: 'How big a step gradient descent takes.' },
  ],
  path: [
    { label: 'Read', min: 30 },
    { label: 'Lab: neuron', min: 35 },
    { label: 'Lab: training', min: 70 },
    { label: 'Lab: tuning', min: 75 },
  ],
  graphic: {
    type: 'cycle',
    steps: [
      { label: 'Predict' },
      { label: 'Loss' },
      { label: 'Zero grads' },
      { label: 'Backward' },
      { label: 'Update' },
    ],
    caption: 'The five-step training loop you write in the lab, repeated every epoch.',
  },
};
