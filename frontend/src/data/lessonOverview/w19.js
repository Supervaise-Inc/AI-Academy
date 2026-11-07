// Week 19 · Capstone Build Week + Mock NCA-GENL Exam (backend/content/week-19.md)
export default {
  week: 19,
  exam: true,
  tagline: 'Build, evaluate and ship the capstone, then sit the timed 50-question mock exam.',
  time: 'Lab about 3 hours, mock exam 60 minutes',
  ideas: [
    'Build: photo, classify, retrieve, draft, rails; the engineer decides.',
    'Evaluate: confusion matrix, per-class recall, and fix the weakest step first.',
    'Ship, then mock exam: 50 questions, 60 minutes, 35/50 = 70% to pass.',
  ],
  outcomes: [
    'Recall the capstone pipeline and why a licensed engineer makes every decision.',
    'Explain why a chain succeeds only as often as the product of its steps.',
    'Apply the time plan: 55 seconds a question first pass, then the flagged ones.',
  ],
  terms: [
    { t: 'Capstone chain', d: 'Photo, classify, retrieve, draft, rails; the engineer signs off.' },
    { t: 'Confusion matrix', d: 'True class against predicted class; mistakes sit off the diagonal.' },
    { t: 'Per-class recall', d: 'Share of each class found; accuracy can hide a class never found.' },
    { t: 'Latency budget', d: 'Time each step may use; 18 + 4 + 35 + 6 = 63 ms.' },
    { t: 'NCA-GENL', d: 'The NVIDIA associate exam, issued by NVIDIA after its own proctored test.' },
  ],
  path: [
    { label: 'Parts 1-4', min: 110 },
    { label: 'Parts 5-7', min: 95 },
    { label: 'First pass', min: 46 },
    { label: 'Flagged items', min: 14 },
  ],
  graphic: {
    type: 'bars',
    unit: '%',
    items: [
      { label: 'Classify', value: 92 },
      { label: 'Retrieve', value: 75 },
      { label: 'Draft', value: 95 },
      { label: 'Rails', value: 99 },
      { label: 'Whole chain', value: 64.9 },
    ],
    caption: '0.92 x 0.75 x 0.95 x 0.99 = 0.649: fix retrieval, the weakest step, first.',
  },
};
