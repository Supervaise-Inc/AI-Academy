// Week 9 · LLM Families, Scaling Laws & Prompt Engineering (backend/content/week-09.md)
export default {
  week: 9,
  tagline: 'The same Transformer can read, write or translate, and how you prompt it matters.',
  time: '30 min reading, a 90-min lab, a 90-min Prompt Lab',
  ideas: [
    'What the model may look at decides its job: BERT, GPT or T5.',
    'Chinchilla rule: train on about 20 tokens for every parameter.',
    'Few-shot fixes format, a role sets tone, chain-of-thought helps reasoning.',
  ],
  outcomes: [
    'Compare BERT, GPT, T5 and mixture-of-experts models and the jobs that suit each.',
    'Use the Chinchilla rule to estimate a compute-optimal model and data size.',
    'Write zero-shot, few-shot, role and chain-of-thought prompts; explain prompt injection.',
  ],
  terms: [
    { t: 'Encoder-only', d: 'Reads the whole input in both directions to understand text. BERT is one.' },
    { t: 'Decoder-only', d: 'Predicts the next token from the past, so it can write text. GPT is one.' },
    { t: 'Scaling law', d: 'A formula that predicts how loss changes as size, data and compute grow.' },
    { t: 'Few-shot prompting', d: 'Worked examples in the prompt so the model copies the pattern.' },
    { t: 'Chain-of-thought', d: 'Asking the model to write out its reasoning steps before the answer.' },
    { t: 'Prompt injection', d: 'Text in the data that tries to override the developer’s instructions.' },
  ],
  path: [
    { label: 'Read', min: 30 },
    { label: 'GA Lab 4', min: 90 },
    { label: 'Prompt Lab', min: 90 },
  ],
  graphic: {
    type: 'compare',
    left: {
      title: 'Encoder (BERT)',
      points: ['Sees both sides of a word', 'Trained to fill hidden words', 'Suits understanding text'],
    },
    right: {
      title: 'Decoder (GPT)',
      points: ['Sees only the past', 'Trained to predict next one', 'Suits writing text'],
    },
    caption: 'What the model may look at decides what it is good at.',
  },
};
