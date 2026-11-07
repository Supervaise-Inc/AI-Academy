// Week 12 · Image Generation & GANs (backend/content/week-12.md)
export default {
  week: 12,
  tagline: 'A forger and a detective train each other; transpose convolutions grow noise into images.',
  time: 'About 4 hours: a 60-minute lesson, then a 3-hour lab',
  ideas: [
    'A GAN is a minimax game: D tells real from fake, G fools D. At the end D = 1/2.',
    'The non-saturating loss keeps G learning: gradient 0.9975, not 0.0025.',
    'A transpose convolution grows a grid: k = 4, s = 2, p = 1 doubles the side.',
  ],
  outcomes: [
    'Explain the generator, the discriminator and the minimax game, and write both losses.',
    'Describe mode collapse and two other GAN failures, with a remedy for each.',
    'Compute the output size of a transpose convolution and count DCGAN parameters.',
  ],
  terms: [
    { t: 'Generator', d: 'Turns random noise into a fake sample, trained to fool the discriminator.' },
    { t: 'Discriminator', d: 'A classifier that outputs the probability that its input is real.' },
    { t: 'Minimax game', d: 'D tries to classify well while G tries to make D wrong.' },
    { t: 'Saturating loss', d: 'The original generator loss: its gradient fades when D is winning.' },
    { t: 'Mode collapse', d: 'G keeps repeating a few outputs and ignores the variety in the data.' },
    { t: 'Transpose convolution', d: 'A layer with a learned kernel that grows a small grid into a larger one.' },
  ],
  path: [
    { label: 'Lesson', min: 60 },
    { label: 'Setup, 1-2', min: 60 },
    { label: 'GAN game, 3-5', min: 70 },
    { label: 'DCGAN, 6', min: 35 },
    { label: 'Judge, 7', min: 15 },
  ],
  graphic: {
    type: 'flow',
    steps: [
      { label: 'Noise z', sub: '100 numbers' },
      { label: 'Linear', sub: '4 × 4 × 512' },
      { label: 'Transpose ×4', sub: 'sides 4 → 8 → 16 → 32' },
      { label: 'Picture', sub: '64 × 64 × 3' },
      { label: 'Judge D', sub: 'real or fake?' },
    ],
    caption: 'DCGAN: 100 numbers grow into 12,288 pixel values, then D judges them.',
  },
};
