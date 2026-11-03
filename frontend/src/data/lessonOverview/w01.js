// Week 1 · Introduction to Generative AI (backend/content/week-01.md)
export default {
  week: 1,
  tagline: 'A generative model makes new content; other models only label what exists.',
  time: 'About 30 minutes of reading, then a 3-hour lab',
  ideas: [
    'A generative model makes new content that follows patterns it learned.',
    'A language model writes by choosing one next word at a time.',
    'Judge any AI tool by fairness, transparency, privacy, safety, accountability.',
  ],
  outcomes: [
    'Explain how generative AI differs from a model that classifies or predicts.',
    'Place key moments on the history of AI and name the main risks.',
    'Run four generative models on the GPU and judge their output.',
  ],
  terms: [
    { t: 'Generative model', d: 'Learns patterns in data, then produces new data that follows them.' },
    { t: 'Discriminative model', d: 'Gives back a label or a number for data that already exists.' },
    { t: 'LLM', d: 'A generative model trained on huge amounts of text that predicts the next word.' },
    { t: 'Temperature', d: 'Controls how adventurous the next-word pick is.' },
    { t: 'AI winters', d: 'Periods when interest and funding dropped because promises outgrew results.' },
    { t: 'Trustworthy AI', d: 'AI built and used with fairness, transparency, privacy, safety, accountability.' },
  ],
  path: [
    { label: 'Read', min: 30 },
    { label: 'Lab: GPT-2', min: 40 },
    { label: 'Lab: image', min: 70 },
    { label: 'Lab: speech', min: 60 },
    { label: 'Quiz', min: 20 },
  ],
  graphic: {
    type: 'compare',
    left: {
      title: 'Discriminative',
      points: ['Input already exists', 'Output: a label or number', 'Spam filter, house price'],
    },
    right: {
      title: 'Generative',
      points: ['Input: a prompt', 'Output: new content', 'ChatGPT, Stable Diffusion'],
    },
    caption: 'Both learn from data. Only the generative one creates something new.',
  },
};
