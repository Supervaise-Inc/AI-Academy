// Week 19 · Capstone Build Week + Mock NCA-GENL Exam (backend/content/week-19.md)
export default {
  week: 19,
  exam: true,
  tagline: 'Build the capstone pipeline end to end, then sit the timed mock exam.',
  time: 'About 30 minutes of reading, plus build time and exam',
  ideas: [
    'Pipeline: classify, retrieve, draft, rails, then a licensed engineer decides.',
    'Test with a confusion matrix and fix the most serious failures first.',
    'Mock exam: 50 questions in 60 minutes, pass at 35/50 = 70%.',
  ],
  outcomes: [
    'Build the four parts of the Structural Inspection Assistant and test them.',
    'Apply the timing plan: 72 seconds per question, pass mark 35/50 = 70%.',
    'Explain what NCA-GENL is and what a mock exam pass does not mean.',
  ],
  terms: [
    { t: 'Capstone', d: 'The team project from Week 13 to Week 20: the Structural Inspection Assistant.' },
    { t: 'RAG', d: 'Find relevant passages first, add them to the prompt, then let the model write.' },
    { t: 'Guardrails', d: 'Checks around a model that block or reshape unsafe or off-topic input and output.' },
    { t: 'Confusion matrix', d: 'A table of true class against predicted class; mistakes are off the diagonal.' },
    { t: 'NCA-GENL', d: 'The NVIDIA associate exam, issued by NVIDIA after its own proctored test.' },
  ],
  path: [
    { label: 'Read', min: 30 },
    { label: 'First pass', min: 46 },
    { label: 'Marked items', min: 14 },
  ],
  graphic: {
    type: 'flow',
    steps: [
      { label: 'Classify', sub: 'crack, spalling...' },
      { label: 'Retrieve', sub: 'checklist passages' },
      { label: 'Draft', sub: 'LLM summary' },
      { label: 'Rails', sub: 'input and output' },
      { label: 'Engineer', sub: 'pass / fail decision' },
    ],
    caption: 'The capstone pipeline: nothing flows past the engineer.',
  },
};
