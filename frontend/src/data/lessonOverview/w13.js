// Week 13 · Diffusion Models (backend/content/week-13.md)
export default {
  week: 13,
  tagline: 'Make a picture by starting from pure noise and removing a little noise at a time.',
  time: 'About 30 minutes of reading, then a 3-hour lab',
  ideas: [
    'The forward process adds noise step by step until only noise is left.',
    'The network learns to predict the noise, so training is a plain regression.',
    'A UNet with a time embedding denoises at every noise level.',
  ],
  outcomes: [
    'Describe the forward process that adds noise and the reverse process that removes it.',
    'Compute x_t from x₀ and noise using ᾱ, and read how much signal is left.',
    'Explain why the denoiser is a UNet with a time embedding.',
  ],
  terms: [
    { t: 'Diffusion model', d: 'Learns to reverse a slow noising process to make new data.' },
    { t: 'Noise schedule', d: 'The list of β values that sets how much noise each step adds.' },
    { t: 'Denoiser', d: 'The network that takes a noisy image and step t and estimates the noise.' },
    { t: 'UNet', d: 'A U-shaped encoder-decoder network with skip connections.' },
    { t: 'Time embedding', d: 'A vector made from step t that tells every layer the noise level.' },
    { t: 'Latent diffusion', d: 'Running diffusion in a compressed space, not on raw pixels.' },
  ],
  path: [
    { label: 'Read', min: 30 },
    { label: 'Noise schedule', min: 60 },
    { label: 'Diffusion lab', min: 100 },
    { label: 'Reflection', min: 20 },
  ],
  graphic: {
    type: 'curve',
    xLabel: 'Step t',
    yLabel: 'Weight',
    series: [
      { label: 'Signal kept', points: [[1, 0.9999], [100, 0.9471], [300, 0.6296], [500, 0.2803], [1000, 0.0064]] },
      { label: 'Noise added', points: [[1, 0.01], [100, 0.3209], [300, 0.7769], [500, 0.9599], [1000, 1]] },
    ],
    caption: 'Linear schedule, β from 0.0001 to 0.02 over T = 1000 steps.',
  },
};
