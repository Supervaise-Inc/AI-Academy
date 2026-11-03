// Week 20 · Final Exam & Demo Day. Revision workshop for Weeks 16 to 19 plus a Demo Day rehearsal
// (backend/content/week-20.md, notebooks/src/week-20.py). There is no lab this week.

// The pitch feedback is built from the four things the exercise looks for. One rule is generated for each
// combination, most complete first, so the reply names what the pitch has and what it lacks.
const PITCH_PARTS = [
  {
    key: 'problem',
    re: 'problem|slow|risky|today|waste',
    have: 'it states the problem',
    add: 'Problem: say who inspects structures, what is slow or risky today, and what your assistant does (and does not do).',
  },
  {
    key: 'demo',
    re: 'demo|live|upload|show|walk',
    have: 'it shows the demo',
    add: 'Demo: follow one photo through the whole pipeline (classify, retrieve checklist items, draft summary). Keep a backup, such as a recorded run or screenshots, in case the live run fails.',
  },
  {
    key: 'results',
    re: 'result|confusion|accuracy|tested|labelled|correct|percent|%',
    have: 'it reports results',
    add: 'Results: report what you really measured, for example your confusion matrix on photos you labelled. Report honestly and show a failure and how you handled it. Do not claim more than you tested.',
  },
  {
    key: 'limits',
    re: 'limit|cannot|can.?t|fail|miss|blur|lighting|wrong|next',
    have: 'it names limits',
    add: 'Limits: say clearly what the assistant cannot do, for example lighting, angle, blur, or damage types outside the four classes, and what you would improve next.',
  },
];

const pitchRules = [];
for (let n = PITCH_PARTS.length; n >= 1; n--) {
  for (let mask = 0; mask < 1 << PITCH_PARTS.length; mask++) {
    const has = PITCH_PARTS.filter((_, i) => mask & (1 << i));
    if (has.length !== n) continue;
    const lacks = PITCH_PARTS.filter((_, i) => !(mask & (1 << i)));
    const tail =
      'Whatever your pitch says, state clearly that a licensed engineer makes every pass/fail decision, and disclose any AI assistance, as the course rules require.';
    const rule = {
      u: '^' + has.map((p) => `(?=[\\s\\S]*(?:${p.re}))`).join(''),
      out: lacks.length
        ? `Feedback on your 2-minute pitch.\nWhat works: ${has.map((p) => p.have).join(', ')}.\nStill missing or too thin:\n- ${lacks.map((p) => p.add).join('\n- ')}\n${tail}`
        : `Feedback on your 2-minute pitch.\nWhat works: it states the problem, shows the demo, reports results and names limits. That is the shape of a good Demo Day talk.\nRehearse it out loud with a timer and let each team member speak. ${tail}`,
      note: lacks.length ? `missing: ${lacks.map((p) => p.key).join(', ')}` : 'all four parts covered',
      ok: lacks.length === 0,
    };
    if (lacks.length) rule.not = lacks.map((p) => `(?:${p.re})`).join('|');
    pitchRules.push(rule);
  }
}

export default {
  week: 20,
  title: 'Final revision and Demo Day',
  lede: 'Quiz yourself on Weeks 16 to 19 with the tutor, then rehearse a short Demo Day pitch for the Structural Inspection Assistant.',
  lab: 'Exam revision and Demo Day',
  exam: true,
  covers: 'Weeks 16 to 19',
  chips: ['Weeks 16 to 19', 'Demo Day rehearsal', 'No code needed', 'About 2 hours'],
  exercises: [
    {
      tab: '1 · RAG',
      title: 'Week 16: chains and RAG',
      min: 20,
      kind: 'prompt',
      goal: 'Check your answers on the LLM chain, why models hallucinate, and the stages of RAG.',
      tasks: [
        'Say in your own words why a plain LLM can hallucinate.',
        'Ask the tutor for the four RAG stages, and compare with your list.',
        'Ask whether RAG can still fail, and how you would test it.',
      ],
      presets: [
        { l: 'Vague question', u: 'Is RAG good?' },
        { l: 'Hallucination', u: 'Why does a plain LLM hallucinate?' },
        { l: 'RAG stages', u: 'What are the stages of RAG, and what is the chain behind an LLM application?' },
        { l: 'RAG failures', u: 'Can RAG still fail, and how should I test it?' },
      ],
      rules: [
        {
          u: 'hallucinat',
          out: 'An LLM hallucinates for three reasons: it predicts plausible text rather than looking up facts, it has a training cut-off, and it has not seen your documents. RAG addresses the last two by adding relevant passages to the prompt.',
          note: 'plausible text, cut-off, unseen documents',
          ok: true,
        },
        {
          u: 'fail|wrong|test',
          out: 'Yes. RAG reduces hallucination but can still fail at any stage: the retrieval can find the wrong passage, and the generation can ignore or misread the right one. Test retrieval and generation separately, and test questions the documents cannot answer. In the capstone the draft prompt tells the model to use only the retrieved text and to say "not covered" when it is missing.',
          note: 'RAG can fail at any stage',
          ok: true,
        },
        {
          u: 'stage|chain|index|steps|how does rag',
          out: 'RAG has four stages: index (store the documents as searchable chunks), retrieve (find the best chunks by similarity, such as cosine), augment (add them to the prompt), and generate (the model answers using them).\nThe chain behind an LLM application is: prompt template, tokenizer, model, output parser. You control what goes in and what comes out.',
          note: 'index, retrieve, augment, generate',
          ok: true,
        },
      ],
      fallback: {
        out: '"Good" depends on what you need. Ask about one thing: why LLMs hallucinate, the stages of RAG, or how RAG can still fail and how to test it.',
        note: 'too vague: ask about one idea',
        ok: false,
      },
    },
    {
      tab: '2 · agents',
      title: 'Week 17: agents and guardrails',
      min: 20,
      kind: 'prompt',
      goal: 'Check what an agent is, what can go wrong with tools, what the three rails do, and who stays responsible.',
      tasks: [
        'List the parts of an agent from memory, then ask the tutor to check.',
        'Ask about the risks of tools and about the three kinds of rail.',
        'Say who makes the pass/fail decision in the capstone.',
      ],
      presets: [
        { l: 'Vague question', u: 'Is it safe?' },
        { l: 'What is an agent', u: 'What is an agent and what does it need?' },
        { l: 'Tool risks', u: 'What can go wrong when an agent can call tools?' },
        { l: 'Three rails', u: 'What do input, dialog and output rails each check?' },
        { l: 'Who decides', u: 'Who makes the pass/fail decision in the capstone assistant?' },
      ],
      rules: [
        {
          u: 'go wrong|risk|danger|inject',
          out: 'Tools bring four risks: wrong calls, injection, giving the agent too much power, and endless loops. Use the least power needed, validate everything, and keep a step limit.',
          note: 'four tool risks',
          ok: true,
        },
        {
          u: 'input.*dialog.*output|three rails|rails each',
          out: 'Input rails check the user message before it reaches the model. Dialog rails steer the conversation and keep it on topic. Output rails check the reply before the user sees it. NeMo Guardrails is NVIDIA\'s toolkit for these. No rail set is perfect, which is why a human stays responsible.',
          note: 'input, dialog and output rails',
          ok: true,
        },
        {
          u: 'who decides|pass.?fail|sign.?off|human|responsible',
          out: 'A licensed engineer makes every pass/fail decision. The assistant only triages and drafts. Its result screen says DRAFT and names the human decision step, and a rail blocks wording that says the structure is safe.',
          note: 'a human decides',
          ok: true,
        },
        {
          u: 'agent',
          out: 'An agent is a model plus tools, a reasoning loop (reason, act, observe) and memory, with a step limit. The loop repeats: the model reasons about what to do, calls a tool, and observes the result until it can answer or reaches the limit.',
          note: 'model, tools, loop, memory',
          ok: true,
        },
      ],
      fallback: {
        out: '"Safe" is not something one rail can promise. Ask about one part: what an agent is, the risks of tools, what each rail checks, or who makes the pass/fail decision.',
        note: 'too vague: ask about one part',
        ok: false,
      },
    },
    {
      tab: '3 · deployment',
      title: 'Week 18: scaling and deployment',
      min: 25,
      kind: 'prompt',
      goal: 'Check the weight-memory arithmetic, the three kinds of parallelism, and mixed precision versus quantization.',
      tasks: [
        'Predict the weight memory of a 13-billion-parameter model in INT4 before you ask.',
        'Ask the tutor to check it, then ask for the three kinds of parallelism.',
        'Explain mixed precision versus quantization in one sentence each.',
      ],
      presets: [
        { l: 'Vague question', u: 'How do I make my model faster?' },
        { l: '13B in INT4', u: 'How much memory do the weights of a 13 billion parameter model need in INT4?' },
        { l: '7B per precision', u: 'How much memory do the weights of a 7 billion parameter model need in FP32, FP16, INT8 and INT4?' },
        { l: 'Parallelism', u: 'What are data, tensor and pipeline parallelism?' },
        { l: 'Two techniques', u: 'What is the difference between mixed precision and quantization?' },
      ],
      rules: [
        {
          u: '13\\s*(b|billion)',
          out: 'Weights memory = parameters × bytes per parameter. INT4 uses 0.5 byte, so 13 × 0.5 = 6.5 GB. This counts weights only, not activations or a KV cache.',
          note: '6.5 GB',
          ok: true,
        },
        {
          u: 'parallel|tensor|pipeline',
          out: 'Data parallelism copies the model onto several devices and splits the batch. Tensor parallelism splits each layer across devices. Pipeline parallelism splits the layers into stages, fed with micro-batches.',
          note: 'three kinds of parallelism',
          ok: true,
        },
        {
          u: 'mixed|quantiz|quantis',
          out: 'Mixed precision speeds up training. Quantization shrinks a trained model for inference. Both come at a small cost in accuracy.',
          note: 'mixed precision trains, quantization shrinks',
          ok: true,
        },
        {
          u: '7\\s*(b|billion)|memory|\\bgb\\b',
          out: 'Weights memory = parameters × bytes per parameter. For a 7-billion-parameter model: FP32 (4 bytes) is 28 GB, FP16 (2 bytes) is 14 GB, INT8 (1 byte) is 7 GB, and INT4 (0.5 byte) is 3.5 GB. Training with Adam needs about 16 bytes per parameter, much more than inference.',
          note: '28, 14, 7 and 3.5 GB',
          ok: true,
        },
      ],
      fallback: {
        out: '"Faster" can mean lower latency for one request or higher throughput for many, and it can mean faster training or faster serving. Tell me which, or ask about the weight-memory formula, the three kinds of parallelism, or mixed precision versus quantization.',
        note: 'too vague: say what you want to speed up',
        ok: false,
      },
    },
    {
      tab: '4 · capstone',
      title: 'Weeks 18 to 19: capstone and mock exam',
      min: 20,
      kind: 'prompt',
      goal: 'Check the pipeline, which failure to fix first, how the capstone is marked, and what the mock exam proves.',
      tasks: [
        'List the pipeline steps from the photo to the engineer.',
        'Ask which classifier mistake to fix first.',
        'Ask how the capstone is marked and what NCA-GENL is.',
      ],
      presets: [
        { l: 'Vague question', u: 'Tell me about the capstone.' },
        { l: 'Pipeline', u: 'List the capstone pipeline steps and say who decides last.' },
        { l: 'Failure order', u: 'My confusion matrix shows a real crack labelled none. Which mistake do I fix first?' },
        { l: 'Marking', u: 'How is the capstone marked? Give the weights.' },
        { l: 'NCA-GENL', u: 'What is NCA-GENL and does the mock exam prove I will pass it?' },
      ],
      rules: [
        {
          u: 'nca|nvidia|certif',
          out: 'NCA-GENL is the NVIDIA-Certified Associate, Generative AI LLMs (Level 1, the target). It is issued by NVIDIA after its own proctored exam, and Supervaise cannot issue it or guarantee it. The mock exam is practice: 70% on it is the course\'s readiness mark, but it does not mean you have passed or will pass the NVIDIA exam. NCP-GENL (Level 2) is an optional, advanced Professional level.',
          note: 'NVIDIA issues NCA-GENL',
          ok: true,
        },
        {
          u: 'confusion|failure|mistake|first',
          out: 'Fix the dangerous mistake first. A real crack labelled "none" is the most serious, because a missed crack might never reach the engineer. Log every such case (input, what the system did, what it should have done, the fix) and fix it before the more common but less serious mistakes. Accuracy alone is not the whole story.',
          note: 'missed crack first',
          ok: true,
        },
        {
          u: 'marking|marked|criteria|weight|points',
          out: 'The weights are: Problem and proposal 15%, Technical implementation 40%, Responsible AI 15%, Presentation 20% and Documentation 10%. They add up: 15 + 40 + 15 + 20 + 10 = 100%. For example, 60% on Responsible AI earns 0.60 × 15 = 9 points. The largest share is technical implementation, so make sure the live demo works.',
          note: 'weights add up to 100%',
          ok: true,
        },
        {
          u: 'pipeline|steps|last',
          out: 'Photo, classify (crack, spalling, corrosion or none), retrieve checklist items (RAG), draft a summary, rails check the result, then the engineer decides. The engineer is the last step and nothing flows past it: a licensed engineer makes every pass/fail decision.',
          note: 'the engineer is last',
          ok: true,
        },
      ],
      fallback: {
        out: 'The capstone is a big topic. Ask about the pipeline steps, which failure to fix first, the marking weights, or what NCA-GENL is.',
        note: 'too vague: ask about one part',
        ok: false,
      },
    },
    {
      tab: '5 · pitch',
      title: 'Rehearse a 2-minute pitch',
      min: 30,
      kind: 'prompt',
      goal: 'Write a short Demo Day pitch and see whether it states the problem, shows the demo, reports results and names limits.',
      tasks: [
        'Write a 2-minute pitch for your assistant. This is practice: your instructor announces the real slot length.',
        'Press Run and read which of the four parts the tutor found.',
        'Add what is missing and run again until all four are there.',
      ],
      presets: [
        { l: 'Weak pitch', u: 'Our project is amazing and uses the latest AI. Everyone will love it.' },
        {
          l: 'Missing the limits',
          u: 'Inspectors spend hours sorting photos by hand, which is slow today. In our live demo I upload one photo and the pipeline classifies it, finds checklist items and drafts a summary. Our results on photos we labelled ourselves are in the confusion matrix on the slide.',
        },
        {
          l: 'Complete pitch',
          u: 'Inspectors spend hours sorting photos by hand, which is slow today. In our live demo I upload one photo and the pipeline classifies it, finds checklist items and drafts a summary. Our results on photos we labelled ourselves are in the confusion matrix: it got most cracks correct, but one real crack was labelled none, and we logged it. A limit: it struggles with blur and poor lighting, and a licensed engineer makes every pass/fail decision.',
        },
      ],
      rules: pitchRules,
      fallback: {
        out: 'I cannot find any of the four parts. A good pitch states the problem (who is slow or at risk today), shows the demo (one photo through the pipeline), reports results honestly (what you measured, including a failure) and names limits (what it cannot do). Try again with one or two sentences for each.',
        note: 'none of the four parts found',
        ok: false,
      },
    },
    {
      tab: '6 · demo script',
      title: 'Demo Day script',
      min: 25,
      kind: 'note',
      goal: 'Write your team\'s Demo Day script in the five parts, and plan a backup for the live demo.',
      tasks: [
        'Fill in one or two lines for each part. Your instructor announces the real slot length; the shares below are only a starting point.',
        'Decide who speaks for each part, and what your backup is if the live demo fails.',
        'Rehearse once out loud with a timer, then press Save card.',
      ],
      note: '## Demo Day script\n\nSuggested shares of the slot (a starting point, not a rule): problem 15%, live demo 35%, how it works 25%, responsible-AI choices 15%, what next 10%. In a 2-minute rehearsal that is about 18, 42, 30, 18 and 12 seconds.\n\n### 1. The problem (speaker: )\n- Who inspects structures, what is slow or risky today, and what the assistant does (and does not do):\n\n### 2. Live demo (speaker: )\n- One photo through the whole pipeline: classify, retrieve checklist items, draft summary:\n- Backup if it fails (recorded run or screenshots):\n\n### 3. How it works (speaker: )\n- Pipeline diagram, vision model, RAG index, rails, and one test result such as the confusion matrix:\n\n### 4. Responsible-AI choices (speaker: )\n- The DRAFT label, the engineer\'s sign-off, our rails, and one failure we found and how we handled it:\n\n### 5. What next (speaker: )\n- Known limits and the next improvement:\n\n### Final checks\n- [ ] A licensed engineer makes every pass/fail decision, and we say so clearly.\n- [ ] We disclose any AI assistance.\n- [ ] Every team member speaks.\n- [ ] We rehearsed with a timer.',
    },
  ],
};
