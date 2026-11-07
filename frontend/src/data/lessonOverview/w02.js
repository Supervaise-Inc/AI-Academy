// Week 2 · Supervised Learning and Neural Networks (backend/content/week-02.md)
export default {
  week: 2,
  tagline: 'A neuron guesses, a loss measures the mistake, and training nudges the weights.',
  time: 'About 60 minutes of reading, then a 3-hour lab',
  ideas: [
    'Supervised learning: labelled examples, a model, a loss, then training.',
    'A neuron is a = φ(Σwx + b); a layer is φ(Wx + b); a hidden layer solves XOR.',
    'Training is w ← w − η × gradient; then test on data the model has not seen.',
  ],
  outcomes: [
    'Explain supervised learning and split data into training, validation and test sets.',
    'Compute a neuron and a layer by hand, and count the parameters of a network.',
    'Explain why hidden layers and non-linearity are needed, and spot overfitting.',
  ],
  terms: [
    { t: 'Label', d: 'The correct answer attached to each training example.' },
    { t: 'Loss', d: 'One number for how wrong the model is; MSE for numbers, −ln p for classes.' },
    { t: 'Activation', d: 'The non-linear gate a neuron applies to its weighted sum.' },
    { t: 'Learning rate', d: 'The small number η that sets the size of each weight update.' },
    { t: 'Overfitting', d: 'Doing well on training data but badly on data it has not seen.' },
    { t: 'Test set', d: 'Data kept aside and used only once, to judge the final model.' },
  ],
  path: [
    { label: 'Read', min: 60 },
    { label: 'Lab 1 to 3', min: 80 },
    { label: 'Lab 4 and 5', min: 60 },
    { label: 'Lab 6 and 7', min: 50 },
  ],
  graphic: {
    type: 'curve',
    xLabel: 'Training step',
    yLabel: 'Weight w',
    series: [
      {
        label: 'w, ideal is 2',
        points: [
          [0, 0.5],
          [1, 1.7],
          [2, 1.94],
          [3, 1.988],
        ],
      },
    ],
    caption: 'w ← w − 0.1 × gradient: the loss falls from 9 to 0.36 in one step.',
  },
};
