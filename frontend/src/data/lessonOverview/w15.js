// Week 15 · Pre-final Exam + Capstone proposal (backend/content/week-15.md)
export default {
  week: 15,
  exam: true,
  tagline: 'Revise Weeks 11 to 14 with four worked calculations, then draft the Capstone proposal.',
  time: 'About 30 minutes of review, then 2 to 3 hours of study plan',
  ideas: [
    'Week 11: patch token count and CLIP zero-shot classification.',
    'Week 12: generator versus discriminator, and transpose convolution size.',
    'Weeks 13 and 14: noise and ᾱ, the UNet, the training stages and LoRA counts.',
  ],
  outcomes: [
    'Recall the four calculations: patches, transpose conv, ᾱ coefficients, LoRA count.',
    'Explain generator versus discriminator and why the denoiser needs a time embedding.',
    'Apply what you know to draft a Capstone proposal that fits the marking criteria.',
  ],
  terms: [
    { t: 'Patch token', d: 'A ViT turns each image patch into one token: (side ÷ patch)² of them.' },
    { t: 'Zero-shot', d: 'Pick the label text closest to the image, with no extra training.' },
    { t: 'Generator', d: 'The GAN network that makes fakes from noise.' },
    { t: 'Alpha-bar (ᾱ)', d: 'How much of the original image is kept after t noising steps.' },
    { t: 'LoRA', d: 'Freezes W and trains a small add-on, 2 × d × r numbers per matrix.' },
  ],
  path: [
    { label: 'Calculations', min: 30 },
    { label: 'Sort cards', min: 15 },
    { label: 'Re-read labs', min: 45 },
    { label: 'Take the quiz', min: 20 },
    { label: 'Mock Exam Prep', min: 45 },
  ],
  graphic: {
    type: 'flow',
    steps: [
      { label: 'Photo', sub: 'inspection image' },
      { label: 'Classify', sub: 'ViT or CLIP' },
      { label: 'Find items', sub: 'RAG over checklists' },
      { label: 'Draft summary', sub: 'LLM and guardrails' },
      { label: 'Engineer', sub: 'makes the decision' },
    ],
    caption: 'The Capstone pipeline: the engineer is last, the assistant never signs off.',
  },
};
