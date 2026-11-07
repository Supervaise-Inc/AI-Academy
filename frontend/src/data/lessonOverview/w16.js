// Week 16 · LLM Components & RAG (backend/content/week-16.md)
export default {
  week: 16,
  tagline: 'Five parts make an LLM app; RAG looks up your documents so the model stops guessing.',
  time: 'About 60 minutes of reading, then a 3-hour lab',
  ideas: [
    'Five parts in a line: template, tokenizer, model, parser, chain.',
    'A plain model predicts plausible text, so it invents what it never saw.',
    'RAG: chunk, embed, store; then retrieve, augment, generate. Test each part.',
  ],
  outcomes: [
    'Name the five components of an LLM application and put them in order.',
    'Fill a prompt template and count its tokens against the context window.',
    'Explain why errors multiply along a chain and why a plain LLM hallucinates.',
  ],
  terms: [
    { t: 'Tokenizer', d: 'Splits text into tokens; less common text gives more tokens.' },
    { t: 'Chain', d: 'Steps joined in a line; its success is the product of the step successes.' },
    { t: 'Hallucination', d: 'Fluent, confident text that is wrong or made up.' },
    { t: 'Chunk', d: 'A short piece of a document, small enough to embed and fit a prompt.' },
    { t: 'Top-k', d: 'The k chunks most similar to the question, which go into the prompt.' },
    { t: 'Precision, recall', d: 'Share of fetched chunks that are right; share of right chunks that were fetched.' },
  ],
  path: [
    { label: 'Read', min: 60 },
    { label: 'Parts 1-2', min: 60 },
    { label: 'Parts 3-4', min: 60 },
    { label: 'Parts 5-7', min: 70 },
  ],
  graphic: {
    type: 'compare',
    left: {
      title: 'Plain model',
      points: ['Guesses: two weeks', 'Invents free renewal', '0 of 12 questions right'],
    },
    right: {
      title: 'RAG system',
      points: ['Answers: 12 days', 'Renew once, 7 more days', '12 of 12 with right passage'],
    },
    caption: 'Same library question, answered from memory and from a looked-up passage.',
  },
};
