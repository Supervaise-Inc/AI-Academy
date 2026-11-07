// Week 14 · Training LLMs: Fine-tuning, RLHF, PEFT (backend/content/week-14.md)
export default {
  week: 14,
  tagline: 'A next-word predictor becomes a chat model in stages; LoRA adapts it with a tiny add-on.',
  time: 'About 4 hours: a 60-minute lesson, then a 3-hour lab',
  ideas: [
    'Stages: pre-train, continue, fine-tune, instruction-tune, then RLHF or DPO.',
    'RLHF: SFT, a reward model (Bradley–Terry loss), then RL with a KL penalty.',
    'LoRA freezes W and learns B · A: 294,912 numbers, 0.237% of GPT-2 small.',
  ],
  outcomes: [
    'Tell pre-training, continued pre-training and fine-tuning apart by data and cost.',
    'Explain instruction tuning, the masked SFT loss and the three stages of RLHF.',
    'Compute LoRA parameter counts and fine-tuning memory, and plan a project.',
  ],
  terms: [
    { t: 'Instruction tuning', d: 'Fine-tuning on prompt and answer pairs, with the loss on the answer only.' },
    { t: 'Reward model', d: 'A network that scores answers, trained on human comparisons.' },
    { t: 'KL penalty', d: 'Keeps the tuned model near the reference; a small β invites reward hacking.' },
    { t: 'DPO', d: 'Learns from preference pairs directly, with no separate reward model.' },
    { t: 'LoRA', d: 'Freezes W and trains a small change B · A: r × (d_in + d_out) numbers.' },
    { t: 'QLoRA', d: 'LoRA on top of a frozen base model stored in 4 bits.' },
  ],
  path: [
    { label: 'Lesson', min: 60 },
    { label: 'Setup, 1-2', min: 45 },
    { label: 'Prefs, 3-5', min: 80 },
    { label: 'LoRA, 6', min: 40 },
    { label: 'Plan, 7', min: 15 },
  ],
  graphic: {
    type: 'compare',
    left: {
      title: 'Full fine-tuning',
      points: ['124,439,808 trained', '16 bytes each: 1.99 GB', 'Every weight is updated'],
    },
    right: {
      title: 'LoRA, rank 8',
      points: ['294,912 trained', '0.237% of GPT-2 small', 'W frozen, learns B · A'],
    },
    caption: 'GPT-2 small: 12 blocks × 8 × (768 + 2304) = 294,912 LoRA numbers.',
  },
};
