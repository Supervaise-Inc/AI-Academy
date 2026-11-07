// Week 9 · LLM Families, Scaling Laws & Prompt Engineering (backend/content/week-09.md)
export default {
  week: 9,
  tagline: 'One block, three wirings; loss follows scaling rules; prompts steer the model.',
  time: '60 min reading, 3-hour lab, 90-min Prompt Lab',
  ideas: [
    'BERT, GPT and T5 differ in mask and goal; a mixture of experts uses k of E.',
    'Loss follows a power law; Chinchilla sizes N = sqrt(C/120) and D = 20N.',
    'A prompt is a program: test it, validate output, distrust outside text.',
  ],
  outcomes: [
    'Compare BERT, GPT, T5 and mixture-of-experts models, and pick one for a task.',
    'Use C = 6ND and the Chinchilla rule to size a model and its data.',
    'Apply and evaluate prompting techniques and spot prompt injection.',
  ],
  terms: [
    { t: 'Encoder / decoder', d: 'Encoders see both sides; decoders see only the past and write text.' },
    { t: 'Mixture of experts', d: 'A router sends each token to its top-k of E experts.' },
    { t: 'Power law', d: 'Ten times the parameters multiplies the loss by 10^(-0.076) = 0.839.' },
    { t: 'Chinchilla rule', d: 'About 20 training tokens per parameter for a compute budget.' },
    { t: 'Chain-of-thought', d: 'A prompt that asks for the steps before the final answer.' },
  ],
  path: [
    { label: 'Read', min: 60 },
    { label: 'Families, MoE', min: 55 },
    { label: 'Scaling laws', min: 70 },
    { label: 'Prompt toolkit', min: 40 },
    { label: 'Prompt Lab', min: 90 },
  ],
  graphic: {
    type: 'bars',
    unit: ' tok/param',
    items: [
      { label: 'Chinchilla', value: 20 },
      { label: 'GPT-3', value: 1.7 },
      { label: 'Gopher', value: 1.1 },
    ],
    caption: 'Training tokens per parameter: GPT-3 and Gopher were undertrained.',
  },
};
