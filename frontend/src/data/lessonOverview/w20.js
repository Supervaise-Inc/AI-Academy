// Week 20 · Final Exam + Demo Day (backend/content/week-20.md)
export default {
  week: 20,
  exam: true,
  tagline: 'Revise Weeks 1 to 19 by term and calculation, then demo your capstone.',
  time: 'Part A 60 min, Part B 120 min, plus revision',
  ideas: [
    'Prelim and Midterm: softmax, convolution sizes, attention, parameter counts.',
    'Pre-final and Final: LoRA, chunk counts, rails, memory, bubble, KV cache.',
    'Demo Day: problem, live demo, how it works, responsible AI, what next.',
  ],
  outcomes: [
    'Recall the standard calculations of each term and the formula sheet.',
    'Explain how Part A (40 questions) and Part B (five tasks) are marked.',
    'Apply the five-part demo structure to present the capstone honestly.',
  ],
  terms: [
    { t: 'Cumulative', d: 'The exam covers the whole course, so old topics can be asked again.' },
    { t: 'Part A', d: 'Written exam: 40 multiple-choice questions, 60 minutes, 40 points.' },
    { t: 'Part B', d: 'Practical exam: five notebook tasks, 120 minutes, 60 points.' },
    { t: 'Pass mark', d: '70 points out of 100 across Part A and Part B.' },
    { t: 'Demo Day', d: 'Present the capstone: problem, live demo, how, responsible AI, next.' },
  ],
  path: [
    { label: 'Revision 1-4', min: 360 },
    { label: 'Day 5 mock run', min: 90 },
    { label: 'Part A', min: 60 },
    { label: 'Part B', min: 120 },
  ],
  graphic: {
    type: 'bars',
    unit: '',
    items: [
      { label: 'Prelim', value: 6 },
      { label: 'Midterm', value: 8 },
      { label: 'Pre-final', value: 8 },
      { label: 'Final', value: 18 },
    ],
    caption: 'Part A: 40 questions by term; the Final term has the biggest share, 18.',
  },
};
