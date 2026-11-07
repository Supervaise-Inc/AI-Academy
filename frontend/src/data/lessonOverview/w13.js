// Week 13 · Diffusion Models (backend/content/week-13.md)
export default {
  week: 13,
  tagline: 'Make a picture by starting from pure noise and removing a little noise at a time.',
  time: 'About 4 hours: a 60-minute lesson, then a 3-hour lab',
  ideas: [
    'x_t = √ᾱ · x₀ + √(1 − ᾱ) · ε: at t = 300 the weights are 0.630 and 0.777.',
    'The network predicts the noise by squared error; no adversary.',
    'A UNet with a time embedding denoises at every level; DDIM needs 20 to 50 steps.',
  ],
  outcomes: [
    'Describe the forward process and compute x_t from x₀ with ᾱ.',
    'Write the noise-prediction loss and explain one reverse step and the sampling loop.',
    'Explain why the denoiser is a UNet with a time embedding.',
  ],
  terms: [
    { t: 'Noise schedule', d: 'The β values that set how much noise each step adds; ᾱ is their running product.' },
    { t: 'Signal-to-noise ratio', d: 'ᾱ ÷ (1 − ᾱ): how much picture there is against noise. 0.657 at t = 300.' },
    { t: 'Denoiser', d: 'The network that takes a noisy image and step t and estimates the noise.' },
    { t: 'UNet', d: 'A U-shaped encoder-decoder network with skip connections.' },
    { t: 'DDIM', d: 'A sampler that reuses the network with 20 to 50 big steps instead of 1,000.' },
    { t: 'Latent diffusion', d: 'Running diffusion in a latent 48 times smaller than the pixels.' },
  ],
  path: [
    { label: 'Lesson', min: 60 },
    { label: 'Setup, 1-2', min: 55 },
    { label: 'Reverse, 3-4', min: 50 },
    { label: 'UNet+DDPM, 5-6', min: 65 },
    { label: 'Guidance, 7', min: 10 },
  ],
  graphic: {
    type: 'curve',
    xLabel: 'Step t',
    yLabel: 'Weight',
    series: [
      { label: 'Picture √ᾱ', points: [[1, 0.9999], [100, 0.9471], [300, 0.6296], [500, 0.2803], [1000, 0.0064]] },
      { label: 'Noise √(1 − ᾱ)', points: [[1, 0.01], [100, 0.3209], [300, 0.7769], [500, 0.9599], [1000, 1]] },
    ],
    caption: 'Linear schedule, T = 1000: at t = 300 noise (0.777) beats picture (0.630).',
  },
};
