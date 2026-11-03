// Week 16 · LLM Components & RAG (backend/content/week-16.md)
export default {
  week: 16,
  tagline: 'Give the model the right text to read before it answers, so it hallucinates less.',
  time: 'About 30 minutes of reading, then a 3-hour lab',
  ideas: [
    'An LLM application is a chain: prompt template, tokenizer, model, output parser.',
    'LLMs hallucinate: they predict plausible text and have not seen your documents.',
    'RAG can still fail at any stage, so test retrieval and generation separately.',
  ],
  outcomes: [
    'Name the building blocks of an LLM application and how a chain connects them.',
    'Explain why LLMs hallucinate.',
    'Describe the RAG steps and run a similarity search by hand.',
  ],
  terms: [
    { t: 'Prompt template', d: 'Fixed text with blanks, filled in for each request.' },
    { t: 'Output parser', d: 'Code that turns the raw model text into something a program can use.' },
    { t: 'Hallucination', d: 'Fluent, confident text that is wrong or made up.' },
    { t: 'Chunks', d: 'Short pieces of a document, small enough to embed and fit a prompt.' },
    { t: 'Embedding', d: 'A list of numbers for the meaning of a text.' },
    { t: 'Cosine similarity', d: 'A score from -1 to 1 for how alike two vectors point.' },
  ],
  path: [
    { label: 'Read', min: 30 },
    { label: 'Chunk, search', min: 50 },
    { label: 'Prompt', min: 40 },
    { label: 'Generate', min: 60 },
    { label: 'Break it', min: 35 },
  ],
  graphic: {
    type: 'flow',
    steps: [
      { label: 'Index', sub: 'chunk and embed' },
      { label: 'Retrieve', sub: 'top k by similarity' },
      { label: 'Augment', sub: 'chunks into prompt' },
      { label: 'Generate', sub: 'answer from prompt' },
    ],
    caption: 'RAG: index once, then retrieve, augment and generate for each question.',
  },
};
