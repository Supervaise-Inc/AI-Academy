// Week 3 · Deep Learning & Backpropagation (backend/content/week-03.md)
export default {
  week: 3,
  tagline: 'The blame for a mistake travels backwards, and in deep networks it can fade.',
  time: 'About 30 minutes of reading, then a 3-hour lab',
  ideas: [
    'Depth builds complex features from simple ones, layer by layer.',
    'Backpropagation is the chain rule from the loss back to each weight.',
    'Gradients are products of layer factors, so they can vanish or explode.',
  ],
  outcomes: [
    'Explain why depth helps, and run a forward pass through a tiny network.',
    'Compute a gradient with the chain rule and check it with autograd.',
    'Explain vanishing and exploding gradients and name the standard fixes.',
  ],
  terms: [
    { t: 'Chain rule', d: 'Multiply the rates of change of each step along the path.' },
    { t: 'Computation graph', d: 'A record of the forward pass so the backward pass can reuse it.' },
    { t: 'Kaiming initialisation', d: 'Starting weights for ReLU networks that keep signal size steady.' },
    { t: 'Residual connection', d: 'A shortcut that adds a layer input to its output.' },
    { t: 'Gradient clipping', d: 'Rescale the gradient down to a limit when it gets too large.' },
  ],
  path: [
    { label: 'Read', min: 30 },
    { label: 'Lab: autograd', min: 50 },
    { label: 'Lab: 10 layers', min: 90 },
    { label: 'Lab: clipping', min: 30 },
  ],
  graphic: {
    type: 'bars',
    unit: '×',
    items: [
      { label: 'Output layer', value: 1 },
      { label: '1 layer back', value: 0.25 },
      { label: '2 layers back', value: 0.0625 },
      { label: '3 layers back', value: 0.0156 },
    ],
    caption: 'Largest gradient reaching each layer, with sigmoids passing on at most ×0.25.',
  },
};
