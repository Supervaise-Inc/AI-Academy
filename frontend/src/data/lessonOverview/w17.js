// Week 17 · AI Agents, Guardrails & Trustworthy AI (backend/content/week-17.md)
export default {
  week: 17,
  tagline: 'An agent loops through reason, act and observe, and rails keep it in check.',
  time: 'About 30 minutes of reading, then a 3-hour lab',
  ideas: [
    'An agent is a model plus tools, a reasoning loop and memory, with a step limit.',
    'Tools add risk: wrong calls, injection, too much power and endless loops.',
    'Input, dialog and output rails help, but a human decides on safety.',
  ],
  outcomes: [
    'Describe an agent and trace one run step by step.',
    'Name the risks of giving a model tools, with one mitigation for each.',
    'Place input, dialog and output rails and test them with adversarial inputs.',
  ],
  terms: [
    { t: 'Agent', d: 'A model that can use tools, looping between reasoning, acting and observing.' },
    { t: 'Tool', d: 'A function the agent may call by name; the program runs it, not the model.' },
    { t: 'Guardrail', d: 'A check around a model that blocks, steers or fixes input and output.' },
    { t: 'NeMo Guardrails', d: 'NVIDIA toolkit for adding input, dialog and output rails to LLM apps.' },
    { t: 'Human oversight', d: 'A person reviews or approves what the AI produces before it has effects.' },
  ],
  path: [
    { label: 'Read', min: 30 },
    { label: 'Tools and loop', min: 65 },
    { label: 'Rails', min: 30 },
    { label: 'Attack and fix', min: 70 },
    { label: 'Write-up', min: 35 },
  ],
  graphic: {
    type: 'cycle',
    steps: [{ label: 'Reason' }, { label: 'Act' }, { label: 'Observe' }],
    caption: 'The agent loop repeats until the model answers; a step limit stops it.',
  },
};
