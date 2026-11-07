// Week 17 · AI Agents, Guardrails & Trustworthy AI (backend/content/week-17.md)
export default {
  week: 17,
  tagline: 'An agent thinks, acts and repeats; rails, least power and a human keep it safe.',
  time: 'About 60 minutes of reading, then a 3-hour lab',
  ideas: [
    'An agent is a model plus tools, a loop and memory; tools turn text into action.',
    'Input, dialog and output rails stack, but no rail is perfect.',
    'Trust needs a person to decide; FGSM shows tiny changes can fool a classifier.',
  ],
  outcomes: [
    'Name the four parts of an agent and trace its loop step by step.',
    'Explain tool calling and why side-effect tools are riskier than read-only ones.',
    'Compute the pass-through of layered rails and the precision of a detector.',
  ],
  terms: [
    { t: 'Agent loop', d: 'Think, act, observe, repeat; every loop re-reads the whole context.' },
    { t: 'Least privilege', d: 'Give the agent only the tools and permissions its job needs.' },
    { t: 'Prompt injection', d: 'Instructions hidden in a message or a document that the model then follows.' },
    { t: 'Rail', d: 'A check on input, topic or output; NeMo Guardrails sets them in config and Colang.' },
    { t: 'Defence in depth', d: 'Several layers; what gets through is the product of each layer\'s misses.' },
    { t: 'FGSM', d: 'A tiny, targeted change to every pixel that flips an image classifier.' },
  ],
  path: [
    { label: 'Read', min: 60 },
    { label: 'Parts 1-3', min: 75 },
    { label: 'Parts 4-5', min: 65 },
    { label: 'Parts 6-7', min: 45 },
  ],
  graphic: {
    type: 'bars',
    unit: '',
    items: [
      { label: 'Attacks', value: 1000 },
      { label: 'Input rail', value: 300 },
      { label: 'Dialog rail', value: 150 },
      { label: 'Output rail', value: 60 },
    ],
    caption: 'Of 1,000 attacks, rails that catch 70%, 50% and 60% let 60 (6%) through.',
  },
};
