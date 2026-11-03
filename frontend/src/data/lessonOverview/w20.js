// Week 20 · Final Exam & Demo Day (backend/content/week-20.md)
export default {
  week: 20,
  exam: true,
  tagline: 'Revise the whole course as one story, then show your capstone.',
  time: 'About 30 minutes for this page; your instructor sets dates',
  ideas: [
    'The final exam is cumulative: revise by term and by calculation.',
    'Each term builds on the last: learning, Transformers, images, applications.',
    'Demo Day: problem, live demo, how it works, responsible-AI choices, what next.',
  ],
  outcomes: [
    'Recall the standard calculations from each term, such as convolution size.',
    'Explain the main ideas of each term in your own words.',
    'Present your capstone in five parts and link each to the marking criteria.',
  ],
  terms: [
    { t: 'Vanishing gradient', d: 'A gradient that shrinks toward zero through many layers, so early ones learn slowly.' },
    { t: 'Positional encoding', d: 'Numbers added to token vectors so the model knows word order.' },
    { t: 'PEFT', d: 'Train a small number of added or selected parameters, not the whole model.' },
    { t: 'Retrieval', d: 'Finding the stored passages most similar to the question to add to the prompt.' },
    { t: 'NCA-GENL', d: 'The NVIDIA-Certified Associate exam, issued by NVIDIA, not by Supervaise.' },
  ],
  path: [
    { label: 'Review', min: 30 },
    { label: 'First pass', min: 46 },
    { label: 'Marked items', min: 14 },
  ],
  graphic: {
    type: 'layers',
    items: [
      { label: 'Final', sub: 'Apps and deployment' },
      { label: 'Pre-final', sub: 'Images and generation' },
      { label: 'Midterm', sub: 'Language, Transformers' },
      { label: 'Prelim', sub: 'How machines learn' },
    ],
    caption: 'Each term builds on the one below it.',
  },
};
