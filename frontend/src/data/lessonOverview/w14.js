// Week 14 · Training LLMs: Fine-tuning, RLHF, PEFT (backend/content/week-14.md)
export default {
  week: 14,
  tagline: 'Turn a text predictor into a chat model, and fine-tune it by training a tiny add-on.',
  time: 'About 30 minutes of reading, then a 3-hour lab',
  ideas: [
    'A chat model is built in stages, and only pre-training needs huge data.',
    'RLHF uses human comparisons and a reward model to teach preferred answers.',
    'LoRA freezes W and trains a small add-on, B times A.',
  ],
  outcomes: [
    'Tell pre-training, continued pre-training and fine-tuning apart.',
    'Explain instruction tuning and the three steps of RLHF.',
    'Compute the number of parameters LoRA trains and the memory for the weights.',
  ],
  terms: [
    { t: 'Fine-tuning', d: 'Training a pre-trained model on a smaller labelled dataset.' },
    { t: 'Instruction tuning', d: 'Fine-tuning on instruction and good-answer pairs.' },
    { t: 'RLHF', d: 'Reinforcement learning from human feedback on which answers are better.' },
    { t: 'Reward model', d: 'A network that scores answers, trained on human comparisons.' },
    { t: 'LoRA', d: 'Keeps W frozen and learns a small change written as B times A.' },
    { t: 'Overfitting', d: 'Fitting the training examples closely but doing worse on new ones.' },
  ],
  path: [
    { label: 'Read', min: 30 },
    { label: 'Maths and data', min: 50 },
    { label: 'LoRA setup', min: 45 },
    { label: 'Train, overfit', min: 70 },
    { label: 'Reflection', min: 15 },
  ],
  graphic: {
    type: 'compare',
    left: {
      title: 'Full fine-tuning',
      points: ['One 4096 × 4096 matrix', 'Changes all 16,777,216', 'Every weight is updated'],
    },
    right: {
      title: 'LoRA, rank 8',
      points: ['Same matrix, W frozen', 'Trains 65,536 numbers', 'About 0.39% of the matrix'],
    },
    caption: 'LoRA trains 2 × d × r numbers per matrix instead of d × d.',
  },
};
