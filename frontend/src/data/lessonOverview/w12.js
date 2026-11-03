// Week 12 · Image Generation & GANs (backend/content/week-12.md)
export default {
  week: 12,
  tagline: 'A forger and a detective train each other until the fakes look real.',
  time: 'About 30 minutes of reading, then a 3-hour lab',
  ideas: [
    'The generator fools; the discriminator catches. Both train together.',
    'Each iteration trains D first, then G through D. G never sees real images.',
    'Look at samples, not only losses: mode collapse hides behind fine numbers.',
  ],
  outcomes: [
    'Explain the roles of the generator and the discriminator in a GAN.',
    'Describe one GAN training iteration and recognise mode collapse.',
    'Compute the output size of a transpose convolution by hand.',
  ],
  terms: [
    { t: 'Generator', d: 'Turns random noise into a fake sample, trained to fool the discriminator.' },
    { t: 'Discriminator', d: 'A classifier that outputs the probability that its input is real.' },
    { t: 'Minimax game', d: 'D tries to classify well while G tries to make D wrong.' },
    { t: 'Mode collapse', d: 'G keeps repeating a few outputs and ignores the variety in the data.' },
    { t: 'Transpose convolution', d: 'A layer with a learned kernel that grows a small grid into a larger one.' },
  ],
  path: [
    { label: 'Read', min: 30 },
    { label: 'Sizes', min: 55 },
    { label: 'Train GAN', min: 65 },
    { label: 'Break balance', min: 40 },
    { label: 'Reflect', min: 20 },
  ],
  graphic: {
    type: 'cycle',
    steps: [
      { label: 'Noise z' },
      { label: 'Make fake G' },
      { label: 'Judge D' },
      { label: 'Update D' },
      { label: 'Update G' },
    ],
    caption: 'One iteration: make fakes, judge them, train D, then train G through D.',
  },
};
