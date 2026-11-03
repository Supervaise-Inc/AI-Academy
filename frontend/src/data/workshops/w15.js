// Week 15 · Pre-final Exam + Capstone proposal. Revision workshop for Weeks 11 to 14 (backend/content/week-15.md,
// notebooks/src/week-15.py). There is no lab this week.

// The proposal feedback is built from the five things the exercise looks for. One rule is generated for each
// combination, most complete first, so the reply always names what the student's prompt covers and what it lacks.
const PROPOSAL_PARTS = [
  {
    key: 'problem',
    re: 'problem|slow|risky|today|bottleneck',
    have: 'the problem',
    add: 'Problem: say who has the problem and what slows them down today.',
  },
  {
    key: 'users',
    re: 'user|engineer|inspector|human|reviewer',
    have: 'the users and the human in the loop',
    add: 'Users: name the engineer and say exactly what they review. A licensed engineer makes every pass/fail decision, and the assistant never signs off on safety.',
  },
  {
    key: 'data',
    re: '\\bdata|photos|checklist text|labelled',
    have: 'the data',
    add: 'Data: say where the test photos and the checklist text come from, and whether you are allowed to use them. Mark any sample text as fictional.',
  },
  {
    key: 'model',
    re: '\\bmodel|\\bclip\\b|\\bvit\\b|\\brag\\b|retriev|pipeline|classif|guardrail',
    have: 'the model approach',
    add: 'Model approach: say which model does each step and what goes in and out. For example a vision model (ViT or CLIP) classifies the photo, retrieval finds checklist items (RAG), and a language model drafts the summary behind guardrails.',
  },
  {
    key: 'eval',
    re: 'evaluat|accuracy|metric|confusion|measure|how we will know|how will we know',
    have: 'the evaluation',
    add: 'Evaluation: say how you will know it works, for example accuracy on photos you labelled yourselves, and a check that the right checklist item is found.',
  },
];

const proposalRules = [];
for (let n = PROPOSAL_PARTS.length; n >= 1; n--) {
  for (let mask = 0; mask < 1 << PROPOSAL_PARTS.length; mask++) {
    const has = PROPOSAL_PARTS.filter((_, i) => mask & (1 << i));
    if (has.length !== n) continue;
    const lacks = PROPOSAL_PARTS.filter((_, i) => !(mask & (1 << i)));
    const rule = {
      u: '^' + has.map((p) => `(?=[\\s\\S]*(?:${p.re}))`).join(''),
      out: lacks.length
        ? `Feedback on your proposal prompt.\nCovered: ${has.map((p) => p.have).join(', ')}.\nStill missing or too thin:\n- ${lacks.map((p) => p.add).join('\n- ')}\nAlso check the rest of the proposal checklist: responsible AI (what can go wrong, and which guardrail or human check prevents it) and roles (Team lead, Technical lead, Responsible-AI lead).`
        : 'Feedback on your proposal prompt.\nCovered: the problem, the users and the human in the loop, the data, the model approach and the evaluation. That is a strong base.\nBefore you submit, check the rest of the proposal checklist: responsible AI (what can go wrong, such as a wrong label or an overconfident summary, and which guardrail or human check prevents it) and roles (Team lead, Technical lead, Responsible-AI lead, and what is done by Week 19). Your instructor announces the proposal format, length and deadline.',
      note: lacks.length ? `missing: ${lacks.map((p) => p.key).join(', ')}` : 'all five parts covered',
      ok: lacks.length === 0,
    };
    if (lacks.length) rule.not = lacks.map((p) => `(?:${p.re})`).join('|');
    proposalRules.push(rule);
  }
}

export default {
  week: 15,
  title: 'Exam revision and capstone proposal',
  lede: 'Quiz yourself on Weeks 11 to 14 with the tutor, then draft the proposal for the Structural Inspection Assistant.',
  lab: 'Exam revision',
  exam: true,
  covers: 'Weeks 11 to 14',
  chips: ['About 2 to 3 hours', 'Weeks 11 to 14', 'Capstone proposal', 'No code needed'],
  exercises: [
    {
      tab: '1 · multimodal',
      title: 'Week 11: images become tokens',
      min: 20,
      kind: 'prompt',
      goal: 'Check your answers on ViT patch tokens and CLIP zero-shot classification.',
      tasks: [
        'Work out by hand how many patch tokens a 224×224 image gives with 16×16 patches.',
        'Ask the tutor to check it. Give the image size and the patch size.',
        'Ask how CLIP picks a label without any training on your labels, then explain it in your own words.',
      ],
      presets: [
        { l: 'Vague question', u: 'Tell me about multimodal models.' },
        { l: 'Patch tokens', u: 'A ViT reads a 224 by 224 image with 16 by 16 patches. How many patch tokens, and how many numbers in each patch?' },
        { l: 'Zero-shot CLIP', u: 'How does CLIP classify a photo as dog, cat or car without being trained on those labels?' },
        { l: 'Position', u: 'Why does a ViT need position embeddings?' },
      ],
      rules: [
        {
          u: 'position',
          out: 'The Transformer reads the patches as a sequence of vectors. Position embeddings tell it where each patch came from in the image, so it does not treat the patches as an unordered pile.',
          note: 'position embeddings',
          ok: true,
        },
        {
          u: 'clip|zero.?shot',
          out: 'CLIP trains an image encoder and a text encoder together on image and caption pairs, so a photo and its caption get similar vectors in one shared space.\nFor zero-shot classification you write one text prompt per class ("a photo of a dog"), embed the photo, and pick the label whose text vector is most similar (cosine similarity). No training on your own labels is needed.',
          note: 'zero-shot with CLIP',
          ok: true,
        },
        {
          u: 'patch|token|224',
          out: '224 ÷ 16 = 14 patches along each side, so 14 × 14 = 196 patch tokens.\nEach patch holds 16 × 16 × 3 = 768 numbers (3 colour channels), which a linear layer projects to the model dimension.\nA learned CLS token is added in front, so the Transformer reads 197 tokens. The rule is tokens = (side ÷ patch)². Smaller patches mean more tokens and more computation.',
          note: '196 patch tokens, 197 with CLS',
          ok: true,
        },
      ],
      fallback: {
        out: 'That is too broad to quiz you on. A multimodal model handles more than one kind of data, such as text and images. Pick one idea: how a ViT turns an image into tokens, how CLIP matches an image with text, or why position embeddings are needed. Give numbers when you ask for a calculation.',
        note: 'too vague: pick one idea and give numbers',
        ok: false,
      },
    },
    {
      tab: '2 · GANs',
      title: 'Week 12: GANs and transpose convolution',
      min: 25,
      kind: 'prompt',
      goal: 'Check your answers on generator versus discriminator and on the transpose convolution size formula.',
      tasks: [
        'Predict the output size of a 4×4 input with a 3×3 kernel, stride 2 and no padding.',
        'Then predict the size when that output goes through a layer with a 4×4 kernel and stride 2.',
        'Ask the tutor to check both, and say which network is which in a GAN.',
      ],
      presets: [
        { l: 'Vague question', u: 'What is a GAN?' },
        { l: 'Layer 1', u: 'A generator layer has a 4 by 4 input, a 3 by 3 kernel, stride 2 and no padding. What is the output size?' },
        { l: 'Layer 2', u: 'A second layer takes a 9 by 9 input with a 4 by 4 kernel, stride 2 and no padding. What is the output size?' },
        { l: 'Two networks', u: 'Which network takes random noise as input, and which one do we keep after training?' },
      ],
      rules: [
        {
          u: '9\\s*(by|x|×)\\s*9',
          out: 'Output side = (n − 1) × stride + k = (9 − 1) × 2 + 4 = 16 + 4 = 20.\nThe output is 20 × 20. Check the first layer: (4 − 1) × 2 + 3 = 9, so this is the next layer in a generator that keeps growing the picture.',
          note: '20 × 20',
          ok: true,
        },
        {
          u: 'kernel|stride|transpose',
          out: 'Output side = (n − 1) × stride + k = (4 − 1) × 2 + 3 = 6 + 3 = 9.\nThe output is 9 × 9. A transpose convolution is learnable upsampling: each input value stamps a scaled copy of the kernel on a larger grid, and the copies overlap where stride 2 is smaller than kernel size 3.',
          note: '9 × 9',
          ok: true,
        },
        {
          u: 'generator|discriminator|noise',
          out: 'The generator takes a random noise vector and turns it into a sample. The discriminator says real or fake: it sees both real data and the generator\'s fakes.\nTraining pushes the generator to fool the discriminator and the discriminator to catch it. After training you usually keep the generator and discard the discriminator. Both are ordinary neural networks trained with backpropagation, taking turns.',
          note: 'generator versus discriminator',
          ok: true,
        },
      ],
      fallback: {
        out: 'A GAN is a game between two networks, but that answer is too broad to quiz you on. Ask about the generator versus the discriminator, or give the input size, kernel size, stride and padding of a transpose convolution and I will apply (n − 1) × stride + k.',
        note: 'too vague: ask about one idea or give numbers',
        ok: false,
      },
    },
    {
      tab: '3 · diffusion',
      title: 'Week 13: diffusion',
      min: 25,
      kind: 'prompt',
      goal: 'Check the forward noising step, what the network predicts, and why the denoiser has a time embedding.',
      tasks: [
        'Predict what fraction of the clean image is kept when ᾱ is 0.25.',
        'Ask the tutor about the forward step at t = 300 in the usual 1000-step schedule.',
        'Explain in one sentence what the network is trained to predict.',
      ],
      presets: [
        { l: 'Vague question', u: 'Explain diffusion.' },
        { l: 'ᾱ = 0.25', u: 'In the forward step suppose alpha bar is 0.25. What multiplier goes on the clean image and what on the noise?' },
        { l: 't = 300', u: 'With the usual linear schedule over 1000 steps, what are alpha bar and the two multipliers at t = 300?' },
        { l: 'UNet', u: 'Why is the denoiser a UNet with a time embedding?' },
        { l: 'Training target', u: 'During training, what does the network learn to predict, and what loss is used?' },
      ],
      rules: [
        {
          u: '0\\.25',
          out: 'The forward step is x_t = √ᾱ_t · x₀ + √(1 − ᾱ_t) · ε.\nWith ᾱ = 0.25 the clean image is multiplied by √0.25 = 0.5, and the noise by √(1 − 0.25) = √0.75, about 0.87. So noise already outweighs the signal.',
          note: 'keep 0.5, noise about 0.87',
          ok: true,
        },
        {
          u: '\\b300\\b',
          out: 'With the usual linear schedule over 1000 steps, ᾱ at t = 300 is about 0.396.\nThe signal is multiplied by √0.396 ≈ 0.63 and the noise by √0.604 ≈ 0.78. At t = 1000, ᾱ is about 4 × 10⁻⁵, so almost nothing of x₀ is left.',
          note: 'about 0.396, 0.63 and 0.78',
          ok: true,
        },
        {
          u: 'unet|time embedding',
          out: 'A UNet has a shrinking encoder path, a growing decoder path and skip connections between matching sizes, so fine detail survives.\nThe time embedding is needed because the same network must denoise at every noise level: it tells the network how noisy its input is.',
          note: 'UNet and time embedding',
          ok: true,
        },
        {
          u: 'predict|loss|train',
          out: 'The network sees a noisy image and learns to predict the noise that was added, with a mean squared error (MSE) loss. To make a new image, you start from pure noise and remove a little predicted noise at every step.',
          note: 'predicts the added noise',
          ok: true,
        },
      ],
      fallback: {
        out: 'Diffusion has two directions: a forward process that adds noise and a reverse process that removes it. That is too broad to quiz you on. Ask about the forward formula (give ᾱ or t), what the network predicts, or why the denoiser has a time embedding.',
        note: 'too vague: ask about one part',
        ok: false,
      },
    },
    {
      tab: '4 · fine-tuning',
      title: 'Week 14: fine-tuning, RLHF and LoRA',
      min: 25,
      kind: 'prompt',
      goal: 'Place each technique on the training pipeline and check the LoRA parameter counts.',
      tasks: [
        'Put pre-training, instruction tuning and RLHF in order, and say what each one adds.',
        'Predict the number of LoRA parameters for a 2048 × 2048 matrix with rank 16.',
        'Ask the tutor to check, and say whether LoRA is a training stage or a method.',
      ],
      presets: [
        { l: 'Vague question', u: 'How do you train a chat model?' },
        { l: 'Pipeline order', u: 'Put in order: pre-training, instruction tuning, RLHF. What does each stage do?' },
        { l: 'RLHF', u: 'What are the three steps of RLHF?' },
        { l: 'LoRA, d = 2048', u: 'A weight matrix is 2048 by 2048 and the LoRA rank is 16. How many trainable numbers, and what share of the full matrix?' },
        { l: 'LoRA, d = 4096', u: 'A weight matrix is 4096 by 4096 and the LoRA rank is 8. Compare LoRA with full fine-tuning.' },
      ],
      rules: [
        {
          u: '2048',
          out: 'LoRA trains 2 × d × r numbers = 2 × 2048 × 16 = 65,536.\nThe full matrix has 2048 × 2048 = 4,194,304 numbers, so LoRA is 65,536 ÷ 4,194,304 = 1.5625%. The share is 2r ÷ d = 32 ÷ 2048, and it improves as the matrix gets larger.',
          note: '65,536 numbers, 1.5625%',
          ok: true,
        },
        {
          u: '4096',
          out: 'LoRA freezes W and trains two thin matrices B and A, so it trains 2 × d × r = 2 × 4096 × 8 = 65,536 numbers.\nFull fine-tuning of the matrix updates 4096 × 4096 = 16,777,216 numbers. LoRA is about 0.39% of that, 256 times fewer.',
          note: '65,536 versus 16,777,216',
          ok: true,
        },
        {
          u: 'lora|peft|adapter',
          out: 'LoRA is a method, not a stage. It is a cheap way of doing fine-tuning: the original weights are frozen and only small adapter matrices are trained. It belongs to parameter-efficient fine-tuning (PEFT).',
          note: 'LoRA is a method inside fine-tuning',
          ok: true,
        },
        {
          u: 'order|stage|pre-?train|continued',
          out: '1. Pre-training: predict the next token over a huge text collection. The model learns language and facts. Continued pre-training uses the same objective on extra domain text.\n2. Fine-tuning and instruction tuning: train on prompt and answer pairs so the model follows instructions.\n3. Preference training (RLHF): use human rankings of answers to push the model toward answers people prefer.',
          note: 'pre-training, fine-tuning, RLHF',
          ok: true,
        },
        {
          u: 'rlhf|reward|human feedback',
          out: 'RLHF (Reinforcement Learning from Human Feedback) has three steps:\n1. Collect human preferences between answers.\n2. Train a reward model on those rankings.\n3. Optimise the language model against the reward, for example with PPO, while keeping it close to its original behaviour.',
          note: 'three RLHF steps',
          ok: true,
        },
      ],
      fallback: {
        out: 'A chat model is built in stages, but that is too broad to quiz you on. Ask about the order of pre-training, instruction tuning and RLHF, about the steps of RLHF, or give a matrix size and a rank for a LoRA count.',
        note: 'too vague: ask about one stage or give d and r',
        ok: false,
      },
    },
    {
      tab: '5 · proposal',
      title: 'Draft your capstone proposal prompt',
      min: 30,
      kind: 'prompt',
      goal: 'Write a prompt that describes your proposal and see which of the five key parts the tutor finds in it.',
      tasks: [
        'Write a short prompt for the Structural Inspection Assistant. Cover the problem, the users, the data, the model approach and the evaluation.',
        'Press Run and read which parts the tutor found and which are missing.',
        'Add what is missing and run again until all five are covered.',
      ],
      presets: [
        { l: 'Vague proposal', u: 'Please give feedback on my capstone proposal. It is about AI for buildings.' },
        {
          l: 'Partial',
          u: 'Problem: inspectors spend a long time sorting inspection photos by hand, which is slow. Users: the engineer reads a draft for each photo. Data: our own test photos.',
        },
        {
          l: 'Nearly complete',
          u: 'Problem: sorting inspection photos by hand is slow. Users: a licensed engineer reviews every draft and makes the decision. Data: photos we took ourselves and labelled, plus fictional checklist text. Model: a CLIP classifier, RAG over the checklist and a language model that drafts the summary behind guardrails.',
        },
        {
          l: 'Complete',
          u: 'Problem: sorting inspection photos by hand is slow. Users: a licensed engineer reviews every draft and makes the pass/fail decision. Data: photos we took ourselves and labelled, plus fictional checklist text. Model: a CLIP classifier, RAG over the checklist and a language model that drafts the summary behind guardrails. Evaluation: accuracy on the photos we labelled, and a check that the right checklist item is found.',
        },
      ],
      rules: proposalRules,
      fallback: {
        out: 'I cannot find any of the five parts in this prompt. A proposal prompt should say: the problem (who is slowed down today), the users (the engineer and what they review), the data (where photos and checklist text come from), the model approach (which model does each step) and the evaluation (how you will know it works). Try again with one sentence for each.',
        note: 'none of the five parts found',
        ok: false,
      },
    },
    {
      tab: '6 · proposal card',
      title: 'Proposal checklist card',
      min: 20,
      kind: 'note',
      goal: 'Answer the proposal checklist questions in one or two lines each, for your team.',
      tasks: [
        'Fill in each line for your team. Your instructor announces the proposal format, length and deadline.',
        'Mark any sample checklist text as fictional.',
        'Press Save card.',
      ],
      note: '## Capstone proposal card\n\nThe assistant helps triage. A licensed engineer makes every pass/fail decision.\n\n### Problem\n- Who has the problem, and what slows them down today?\n\n### Users and the human in the loop\n- Who is the engineer, and what exactly do they review?\n\n### Pipeline\n- Photo in, classify (crack, spalling, corrosion or none), find checklist items with RAG, draft a summary. Which model does each step?\n\n### Data plan\n- Where do the test photos and the checklist text come from? Are we allowed to use them? (Mark sample text as fictional.)\n\n### Evaluation\n- How will we know the classifier and the retrieval work?\n\n### Responsible AI\n- What can go wrong (a wrong label, an overconfident summary), and which guardrail or human check prevents each?\n\n### Roles and plan\n- Team lead:\n- Technical lead:\n- Responsible-AI lead:\n- What is done by Week 19?',
    },
  ],
};
