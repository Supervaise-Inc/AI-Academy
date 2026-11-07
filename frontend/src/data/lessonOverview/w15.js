// Week 15 · Pre-final Exam + Capstone proposal (backend/content/week-15.md)
export default {
  week: 15,
  exam: true,
  tagline: 'Revise Weeks 11 to 14 for a 100-point exam, then draft the Capstone proposal.',
  time: 'Exam: 60 + 120 minutes; revise 4 days of about 90 min',
  ideas: [
    'Weeks 11, 12: patch tokens, CLIP, the GAN game, V = −ln 4, transpose conv size.',
    'Week 13: ᾱ and x_t, noise loss, DDPM and DDIM steps, UNet, guidance.',
    'Week 14: masked SFT, Bradley–Terry, KL and DPO, LoRA and memory counts.',
  ],
  outcomes: [
    'Recall the formula sheet: patches, D*, ᾱ, the noise loss, the DPO loss, LoRA counts.',
    'Explain a GAN, a diffusion step and the stages of RLHF, and diagnose a failing run.',
    'Apply the four weeks to a Capstone proposal where the engineer decides.',
  ],
  terms: [
    { t: 'Part A', d: 'Written exam: 40 multiple-choice questions, 60 minutes, 40 points.' },
    { t: 'Part B', d: 'Practical notebook exam: five tasks, 120 minutes, 60 points.' },
    { t: 'Formula sheet', d: 'The calculations of Weeks 11 to 14, each with a worked example.' },
    { t: 'Capstone', d: 'The Structural Inspection Assistant: triage photos and draft a summary.' },
    { t: 'AI-use rule', d: 'A licensed engineer makes every pass or fail decision, never the assistant.' },
  ],
  path: [
    { label: 'Day 1: 11, 12', min: 90 },
    { label: 'Day 2: Wk 13', min: 90 },
    { label: 'Day 3: Wk 14', min: 90 },
    { label: 'Day 4: mock', min: 90 },
  ],
  graphic: {
    type: 'bars',
    unit: ' pts',
    items: [
      { label: 'Week 11', value: 22 },
      { label: 'Week 12', value: 24 },
      { label: 'Week 13', value: 26 },
      { label: 'Week 14', value: 22 },
      { label: 'Synthesis', value: 6 },
    ],
    caption: '100 points: 10 written + one Part B task per week, plus a 6-point synthesis.',
  },
};
