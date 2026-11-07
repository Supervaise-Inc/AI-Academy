// Week 1 · Introduction to Generative AI (backend/content/week-01.md)
export default {
  week: 1,
  tagline: 'A generative model learns a distribution, then samples new content from it.',
  time: 'About 60 minutes of reading, then a 3-hour lab',
  ideas: [
    'Generative models create new data; discriminative models only label or score it.',
    'Learn a distribution, then sample: a language model writes one token at a time.',
    'Temperature, top-k and top-p set how adventurous it is. Fluent is not true.',
  ],
  outcomes: [
    'Explain generative versus discriminative AI and place it inside AI and deep learning.',
    'Compute a sentence probability with the chain rule, and explain perplexity.',
    'Use temperature, top-k and top-p, and apply the five trustworthy-AI principles.',
  ],
  terms: [
    { t: 'Generative model', d: 'Learns patterns in data, then produces new data that follows them.' },
    { t: 'Discriminative model', d: 'Gives back a label or a score for data that already exists.' },
    { t: 'Token', d: 'The piece of text a language model reads and predicts, one at a time.' },
    { t: 'Perplexity', d: 'A model\'s average confusion per token; lower is better.' },
    { t: 'Temperature', d: 'Low plays safe, high is adventurous when picking the next word.' },
    { t: 'Trustworthy AI', d: 'Fairness, transparency, privacy, safety and accountability.' },
  ],
  path: [
    { label: 'Read', min: 60 },
    { label: 'Lab 1 to 3', min: 90 },
    { label: 'Lab 4 and 5', min: 55 },
    { label: 'Lab 6 and 7', min: 40 },
  ],
  graphic: {
    type: 'bars',
    unit: '%',
    items: [
      { label: 'away', value: 73 },
      { label: 'beyond', value: 12 },
      { label: 'from', value: 9 },
      { label: 'off', value: 4 },
      { label: 'behind', value: 2 },
    ],
    caption: 'Next word after "far, far": the model scores every word, then samples one.',
  },
};
