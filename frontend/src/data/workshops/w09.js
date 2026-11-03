// Week 9 · LLM Families, Scaling Laws & Prompt Engineering. Follows GA Lab 4 (backend/content/week-09.md, notebooks/week-09.ipynb).
// Parts A and B (encoder vs decoder, scaling) are the exercises below. Part C, the Prompt Lab, is the hand-built
// prompt simulator in PromptWorkshop.jsx: `legacyLab: true` appends its exercises after these.
export default {
  week: 9,
  legacyLab: true,
  title: 'Prompt lab',
  lede: 'Compare an encoder and a decoder, work out compute-optimal sizes, then practise the prompting techniques for the Prompt Lab.',
  lab: 'GA Lab 4',
  chips: ['About 90 min for Parts A and B', 'Prompt Lab: 90 min', 'Llama-3.1-8B-Instruct', 'NVIDIA DGX Spark'],
  exercises: [
    {
      tab: 'A1 · encoder',
      title: 'Fill-mask with an encoder',
      min: 15,
      kind: 'code',
      goal: 'Ask an encoder-only model (BERT) to fill a [MASK], and see that it reads words on both sides of the gap.',
      tasks: [
        'Run the cell and read the five guesses.',
        'Write three sentences of your own with one [MASK] where the answer depends on words after the gap.',
        'Does BERT use the right-hand words?',
      ],
      code: 'from transformers import pipeline\nfill = pipeline("fill-mask", model="bert-base-uncased")\nfor r in fill("The capital of France is [MASK].", top_k=5):\n    print(f"{r[\'token_str\']:>12}  {r[\'score\']:.3f}")',
      run: 'Run cell',
      out: 'paris is at or near the top of the five guesses, each with its score.',
      expect: 'paris at or near the top. Exact scores vary by library version, so do not try to match the numbers.',
    },
    {
      tab: 'A2 · decoder',
      title: 'Continue text with a decoder',
      min: 15,
      kind: 'prompt',
      goal: 'See why a decoder-only model (GPT-2) cannot use words that come after the gap.',
      tasks: [
        'Ask the tutor why GPT-2 cannot use right-hand context. Mention both models by name.',
        'Compare its answer with what BERT did in the last cell.',
      ],
      presets: [
        { l: 'Vague', u: 'Which one is better?' },
        { l: 'Specific', u: 'BERT can use words after the gap but GPT-2 cannot. Why is that, in two sentences?' },
      ],
      rules: [
        {
          u: 'gpt|decoder|causal|left',
          out: 'GPT-2 is a decoder with a causal mask: each position may attend only to earlier tokens, because it is trained to predict the next token. Words that come after the gap are hidden from it, so it can continue the start of a sentence but cannot use text on the right. BERT has no such mask, so it reads both sides.',
          note: 'accurate: causal mask explains it',
          ok: true,
        },
        {
          u: 'bert|encoder|mask',
          out: 'BERT is an encoder trained to fill masked words, so it attends in both directions and uses words before and after the gap. GPT-2 is a decoder with a causal mask and sees only the words before the position it is predicting.',
          note: 'accurate',
          ok: true,
        },
      ],
      fallback: {
        out: 'Better for what? An encoder such as BERT suits tasks that read the whole text, like filling a gap. A decoder such as GPT-2 suits generating text one token at a time. Tell me the task and the two models you mean.',
        note: 'vague prompt, vague answer',
        ok: false,
      },
    },
    {
      tab: 'A3 · scaling',
      title: 'Scaling by hand and in code',
      min: 25,
      kind: 'code',
      goal: 'Use the compute-optimal rule C = 120·N² (so N = √(C/120) and D = 20·N) to size a model for a compute budget.',
      tasks: [
        'Run the cell and read the table.',
        'For C = 1.2 × 10²², compute N and D by hand and check against the code (N = 10B, D = 200B tokens).',
        'A 7B model is trained on 140B tokens. Is that the Chinchilla ratio, and what compute does it use (6·N·D)?',
      ],
      code: 'import math\nfor C in [1e21, 1e22, 1e23, 1e24]:\n    N = math.sqrt(C / 120)\n    D = 20 * N\n    print(f"C={C:.0e}  N={N/1e9:6.1f}B  D={D/1e9:8.1f}B tokens")',
      run: 'Run cell',
      out: 'C=1e+21  N=   2.9B  D=    57.7B tokens\nC=1e+22  N=   9.1B  D=   182.6B tokens\nC=1e+23  N=  28.9B  D=   577.4B tokens\nC=1e+24  N=  91.3B  D=  1825.7B tokens',
      expect: 'Each 10× in compute grows the model about 3.2× (√10), and the data grows with it. A 7B model on 140B tokens is exactly 20 tokens per parameter, and 6·N·D = 5.88 × 10²¹.',
    },
  ],
};
