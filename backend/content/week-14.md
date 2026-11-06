> Content based on the NVIDIA Generative AI and DLI Deep Learning Teaching Kits, used under the Supervaise × NVIDIA partnership.

# Week 14 — Training LLMs: Fine-tuning, RLHF and PEFT

**Purpose:** Learn how a model that only predicts the next word becomes a helpful chat assistant, and how ==a small add-on such as LoRA lets you adapt a huge model without retraining all of it==.

**Term:** Pre-final (Weeks 11–15) · **Month:** 4 · **Time:** about 60 minutes for the lesson, then a 3-hour lab

```viz
{
  "type": "hook",
  "title": "Watch a text predictor learn to help",
  "caption": "A model trained only to predict the next word does not answer questions: it continues text. Extra training changes that. The two answers are written by hand to show the idea; the preference numbers and the parameter counts are exact arithmetic that you will redo in this lesson.",
  "scenes": [
    { "name": "Base vs chat", "kind": "text", "prompt": "Q: How do I boil an egg?", "outputLabel": "Base model, then tuned chat model (hand-written, illustrative)", "output": "BASE MODEL: Q: How do I peel an egg? Q: How long do I boil it? Q: Why do eggs crack in the pot?   |   CHAT MODEL: Put the egg in gently boiling water for about 10 minutes, then cool it in cold water so it is easy to peel." },
    { "name": "Which answer wins?", "kind": "bars", "prompt": "A reward model gives answer A a higher score than answer B. How likely is a person to prefer A?", "outputLabel": "Probability that A is preferred, sigma(score difference) (computed)", "bars": [ { "label": "Gap 0", "value": 0.5 }, { "label": "Gap 1", "value": 0.731 }, { "label": "Gap 2", "value": 0.881 }, { "label": "Gap 3", "value": 0.953 } ] },
    { "name": "A real number", "kind": "text", "prompt": "GPT-2 small has 124,439,808 parameters. How many must LoRA train to teach it a new style?", "outputLabel": "Computed in this lesson", "output": "With LoRA of rank 8 on the attention layer of every block, you train 294,912 numbers. That is 0.24% of the model. The other 99.76% stay frozen, so the new part can be saved in about 0.59 MB instead of about 249 MB." },
    { "name": "Learning to answer", "kind": "loss", "prompt": "Fine-tuning shows the model good answers and lowers its loss on them.", "outputLabel": "Fine-tuning loss over steps (illustrative)" }
  ]
}
```

## By the end, you can…

1. ==Tell pre-training, continued pre-training and task-specific fine-tuning apart== by their data, goal and cost.
2. ==Explain instruction tuning== and write the supervised loss with the prompt tokens masked, then compute it for a short example.
3. ==Explain why a base model is not a chat model==, and why a chat model needs more than next-token prediction.
4. ==Describe the three stages of RLHF==, compute a preference probability and loss, and explain the KL penalty and reward hacking.
5. ==Compute LoRA parameter counts and fine-tuning memory==, and say why a low-rank update can be enough.
6. ==Plan a fine-tuning project== with data, evaluation and safety checks, and say when not to fine-tune.

Assessed in the Week 14 lab (“Imagine the apprentice and the coach”) and the quiz. Read each section, then play with the figure under it: the figures are practice, so you cannot break anything. Terms with a dotted underline open a short definition when you select them. The lab refers back to this lesson, so look for pointers such as “Lab Part 2”.

## The stages of building a chat model {rail=Stages}

{lede} A chat assistant is not trained in one go. It is built in ==a sequence of stages, and each stage starts from the model the previous stage made==. The first stage is by far the biggest; the later stages are small, cheap and decisive, because they change how the model behaves.

### The apprentice and the coach

Picture Ate Mila, who wants to become a baker. For years she reads everything she can find: recipe books, newspapers, the backs of flour sacks, her grandmother’s notes. Nobody asks her to do a particular job. She simply learns how bread, words and shops work. That is ==pre-training==: general reading, with one rule, which is to guess what comes next.

Then she takes a short trade course on pastry. She still reads, but now only pastry books. That is ==continued pre-training==: the same kind of learning, aimed at one field. After that she starts at a real bakery, where the owner shows her how this shop answers customers: “Good morning, po. Would you like it sliced?” She practises with many examples of a question and the shop’s way of answering it. That is ==fine-tuning==, and when the examples cover many kinds of requests it is called ==instruction tuning==.

Finally a coach sits beside her and rates pairs of answers: “This reply was kinder than that one.” Over weeks she learns what customers like, which no list of model answers could fully show. That is ==learning from preferences==, and the best-known recipe is RLHF. The rest of this lesson takes the four stages in turn, then asks how to run them cheaply.

### Stage 1: pre-training

[[Pre-training|The first and most expensive stage: training a model from random weights on a huge amount of general text to predict the next token. It gives the model its broad knowledge of language and the world.]] uses the objective you met in Week 8: at every position, predict the next [[token|A small piece of text, such as a word or part of a word, that the model reads and writes. Text is cut into tokens before it enters the model.]] and lower the cross-entropy loss. The data are trillions of tokens of web pages, books and code. The hardware is thousands of GPUs for weeks. The result is a [[base model|A model that has only been pre-trained to continue text. It is fluent and knowledgeable but does not reliably follow instructions or answer questions.]].

In Week 9 you learned the rule of thumb for the cost: the training compute is about ==C ≈ 6·N·D== floating-point operations, where N is the number of parameters and D the number of training tokens. Select each symbol below, then follow the worked example for a small model.

```viz
{
  "type": "equation",
  "title": "The cost of pre-training: C ≈ 6·N·D",
  "intro": "Select a symbol to see what it means. The worked example is a small GPT-2-sized run.",
  "formula": [ { "t": "C", "id": "c" }, " ≈ ", { "t": "6", "id": "six" }, " · ", { "t": "N", "id": "n" }, " · ", { "t": "D", "id": "d" } ],
  "parts": [
    { "id": "c", "name": "C (training compute)", "meaning": "The total number of floating-point operations (FLOPs) for one training run." },
    { "id": "six", "name": "6 (cost per parameter per token)", "meaning": "About 2 operations for the forward pass and 4 for the backward pass for every parameter and every token." },
    { "id": "n", "name": "N (parameters)", "meaning": "How many learned numbers the model has. GPT-2 small has 124,439,808." },
    { "id": "d", "name": "D (training tokens)", "meaning": "How many tokens the model reads during training, counting repeats." }
  ],
  "steps": [
    { "title": "Choose a small run", "expr": "N = 124,439,808 ≈ 1.24 × 10⁸ parameters, D = 1 × 10⁹ tokens", "note": "GPT-2 small, trained for one billion tokens. Real pre-training uses far more tokens than this." },
    { "title": "Multiply", "expr": "C ≈ 6 × 1.244 × 10⁸ × 10⁹ = 7.47 × 10¹⁷ FLOPs", "note": "About 7.5 hundred-quadrillion operations. Large models need many orders of magnitude more." }
  ]
}
```

Notice that C grows with both N and D. ==Double the model or double the data and the bill doubles.== That is why pre-training is done once, by well-funded teams, and everyone else starts from the result.

### Stage 2: continued pre-training

Suppose a hospital in Cebu wants a model that is at home with clinical notes, or a newsroom wants one that handles Filipino and Taglish well. [[Continued pre-training|Training an already pre-trained model further on a new body of text, usually from one domain or language, with the same next-token objective. It adds knowledge and vocabulary but does not teach the model to follow instructions.]] keeps the objective but swaps the data for a domain corpus of perhaps millions to billions of tokens. ==You use it when the model lacks the knowledge or vocabulary, not when it lacks manners.==

There is a trap. A model that is trained only on new data can lose what it knew before. This is called [[catastrophic forgetting|The loss of earlier skills or knowledge when a model is trained further on new data. The new training pulls the weights away from values that the old tasks needed.]]. The usual defences are a small learning rate, mixing some of the original kind of text back in, and checking the old skills after training. We return to it in the next section.

### Stage 3: task-specific fine-tuning and instruction tuning

[[Task-specific fine-tuning|Training a pre-trained model on labelled examples for one narrow task, such as marking a support email as urgent or not, so that it does that task well.]] is supervised learning, like Weeks 2 to 4, with a pre-trained model as the starting point. The data are inputs with correct labels: reviews with a sentiment, emails with a category, documents with the fields to extract. ==a few hundred to a few thousand labelled examples are often enough==, because the model already knows the language.

[[Instruction tuning|Fine-tuning on many different tasks, each written as an instruction with a good response, so that the model learns to follow instructions in general. Also called supervised fine-tuning (SFT) when the data are demonstrations.]] widens that idea: instead of one task, thousands of tasks, all written as “instruction in, response out”. Wei et al. (2022, the FLAN work) showed that tuning on many tasks phrased as instructions helps a model do tasks it was never tuned on. Section 2 covers it in detail.

### Stage 4: preference tuning

Instruction tuning shows the model good answers. It cannot show which of two decent answers people prefer, or how to avoid the thousand ways of being wrong. [[Preference tuning|Training a model with human judgements of the form “answer A is better than answer B” instead of single model answers. RLHF and DPO are the two well-known recipes.]] uses those comparisons. Sections 4 and 5 show how.

### The stages side by side

| Stage | Data | Typical size (illustrative) | Objective | What changes | Cost | Main risk |
|---|---|---|---|---|---|---|
| Pre-training | General text from the web, books, code | Trillions of tokens | Next-token prediction | Everything: knowledge of language and the world | Very high | Biases and errors in the data |
| Continued pre-training | Domain or language text | Millions to billions of tokens | Next-token prediction | Domain knowledge, vocabulary | Medium to high | Forgetting general skills |
| Task fine-tuning | Labelled examples for one task | Hundreds to thousands | Supervised loss on the label | One task gets much better | Low | Overfitting, narrow skill |
| Instruction tuning (SFT) | Instruction and response pairs | About a thousand to tens of thousands | Next-token loss on the response | Follows instructions, chat style | Low | Copies errors in the examples |
| Preference tuning | Pairs of better and worse answers | Tens of thousands (illustrative) | Reward or preference loss | What kinds of answers it favours | Medium | Reward hacking, sycophancy |

Read the table row by row and watch two columns: the data shrink from trillions of tokens to thousands of examples, while the effect on behaviour grows.

```viz
{
  "type": "pipeline",
  "title": "From raw text to a chat model",
  "intro": "Step through the stages. Each box starts from the model the previous box produced. The data sizes are typical orders of magnitude, not exact figures for any one model.",
  "vertical": false,
  "nodes": [
    { "id": "pre", "label": "Pre-training", "sub": "trillions of tokens" },
    { "id": "cont", "label": "Continued", "sub": "millions to billions" },
    { "id": "sft", "label": "Instruction tuning", "sub": "~1,000 to tens of thousands" },
    { "id": "pref", "label": "Preference tuning", "sub": "tens of thousands of pairs" },
    { "id": "chat", "label": "Chat model", "sub": "answers, politely" }
  ],
  "steps": [
    { "title": "Pre-training", "body": "Start from random weights and predict the next token over a very large collection of text. This stage takes most of the compute and gives the model its general knowledge.", "active": ["pre"], "example": "Output: a base model that continues text." },
    { "title": "Continued pre-training (optional)", "body": "Keep the same next-token goal but use text from one field or language, such as medical notes or Filipino news. The model absorbs the vocabulary and facts of that domain.", "active": ["pre", "cont"], "example": "Output: a base model that knows the domain." },
    { "title": "Instruction tuning", "body": "Train on instruction and response pairs, computing the loss on the response only. The model learns the format of a conversation and to answer what it was asked.", "active": ["cont", "sft"], "example": "Instruction: Summarise this email. Response: a two-line summary." },
    { "title": "Preference tuning", "body": "People compare answers; a reward model or a direct loss turns the comparisons into a training signal. The model learns which answers people prefer.", "active": ["sft", "pref"], "example": "Prefer the answer that is clear, polite and correct." },
    { "title": "A chat model", "body": "The result follows instructions, keeps a steady tone and tends to be more helpful and safer than the base model. It still makes mistakes, so people check what it says.", "active": ["pref", "chat"], "example": "A school helper chatbot." }
  ]
}
```

Notice that ==only the first box needs huge data and compute==. If your school wants a helper bot, you start from the right-hand end of this picture and never pay for the left.

### Which stage is it?

Sort ten examples into the stage they belong to. Use the apprentice story if you get stuck.

```viz
{
  "type": "sorter",
  "title": "Which training stage is this?",
  "intro": "Drag each example into the stage it belongs to.",
  "bins": [
    { "id": "pre", "label": "Pre-training" },
    { "id": "cont", "label": "Continued pre-training" },
    { "id": "task", "label": "Task fine-tuning" },
    { "id": "instr", "label": "Instruction tuning" },
    { "id": "pref", "label": "Preference tuning" }
  ],
  "cards": [
    { "name": "Reading 1 trillion web tokens to predict the next token, from random weights", "answer": "pre", "why": "Training from scratch on general text with the next-token objective is pre-training." },
    { "name": "Training on 5,000 hospital notes with the same next-token loss", "answer": "cont", "why": "The objective is unchanged and the data are a domain corpus, so this is continued pre-training." },
    { "name": "Fine-tuning to label support emails as urgent or not urgent", "answer": "task", "why": "Labelled examples for one narrow task is task-specific fine-tuning." },
    { "name": "Ranking two answers to the same question, picking the better one", "answer": "pref", "why": "A comparison between answers is preference data." },
    { "name": "Training on 20,000 pairs such as “Summarise this” and a good summary, with loss on the response", "answer": "instr", "why": "Many kinds of instruction and response pairs make this instruction tuning (SFT)." },
    { "name": "Continuing to train a general model on a large collection of Filipino news articles", "answer": "cont", "why": "New language data and the same next-token goal: continued pre-training." },
    { "name": "Training a classifier head on 2,000 labelled movie reviews", "answer": "task", "why": "One task with labels, starting from a pre-trained model, is task fine-tuning." },
    { "name": "Updating a chat model so that answers a reward model scores highly become more likely", "answer": "pref", "why": "Reward-based updating is the last RLHF stage, a form of preference tuning." },
    { "name": "Teaching a model to follow a chat template with system, user and assistant turns", "answer": "instr", "why": "Chat-format demonstrations are what instruction tuning is made of." },
    { "name": "Predicting the next token on a mixed pile of books, code and web pages for weeks on thousands of GPUs", "answer": "pre", "why": "General text, huge scale, next-token objective: pre-training." }
  ]
}
```

If any card surprised you, find its row in the table above and check which column gave it away. ==Data and objective together decide the stage.==

### Why the later stages are cheap

Pre-training and fine-tuning use the same arithmetic, so the same rule C ≈ 6·N·D gives a first estimate for both. The model size N is the same, so ==what separates them is the number of tokens D==. Select the symbols, then follow the numbers for a 7-billion-parameter model.

```viz
{
  "type": "equation",
  "title": "Pre-training compared with fine-tuning",
  "intro": "Because N is the same in both runs, it cancels. Only the token counts matter.",
  "formula": [ { "t": "C_pre", "id": "cpre" }, " / ", { "t": "C_ft", "id": "cft" }, " = ", { "t": "6·N·D_pre", "id": "num" }, " / ", { "t": "6·N·D_ft", "id": "den" }, " = ", { "t": "D_pre / D_ft", "id": "ratio" } ],
  "parts": [
    { "id": "cpre", "name": "C_pre", "meaning": "Compute for pre-training the whole model." },
    { "id": "cft", "name": "C_ft", "meaning": "Compute for a fine-tuning run on the same model." },
    { "id": "num", "name": "6·N·D_pre", "meaning": "The pre-training estimate with D_pre tokens." },
    { "id": "den", "name": "6·N·D_ft", "meaning": "The fine-tuning estimate with D_ft tokens." },
    { "id": "ratio", "name": "D_pre / D_ft", "meaning": "N and 6 cancel, so the ratio of costs is the ratio of token counts." }
  ],
  "steps": [
    { "title": "Pre-training", "expr": "6 × 7 × 10⁹ × 10¹² = 4.2 × 10²² FLOPs", "note": "N = 7 billion parameters, D = 1 trillion tokens." },
    { "title": "Fine-tuning", "expr": "6 × 7 × 10⁹ × 10⁷ = 4.2 × 10¹⁷ FLOPs", "note": "The same model on 10 million tokens, about the size of a few hundred long books." },
    { "title": "The ratio", "expr": "4.2 × 10²² / 4.2 × 10¹⁷ = 100,000", "note": "Fine-tuning needs about one hundred-thousandth of the compute. The rule is only an estimate, and methods such as LoRA lower the cost further." }
  ]
}
```

The two bars below put those two numbers on one log scale, where each step to the right is ten times more.

```viz
{
  "type": "logbars",
  "title": "Compute: pre-training vs fine-tuning (7 B model)",
  "unit": "FLOPs",
  "items": [
    { "label": "Fine-tune on 10 M tokens", "year": "", "value": 4.2e17, "note": "6 × N × D with N = 7 × 10⁹, D = 10⁷" },
    { "label": "Fine-tune on 1 B tokens", "year": "", "value": 4.2e19, "note": "6 × N × D with D = 10⁹" },
    { "label": "Pre-train on 100 B tokens", "year": "", "value": 4.2e21, "note": "6 × N × D with D = 10¹¹" },
    { "label": "Pre-train on 1 T tokens", "year": "", "value": 4.2e22, "note": "6 × N × D with D = 10¹²" }
  ],
  "note": "Computed with C ≈ 6·N·D, an estimate, for a 7-billion-parameter model."
}
```

A full 1-trillion-token run costs as much as one hundred thousand of the small fine-tuning runs. That is the economic reason why most people fine-tune and few pre-train.

### Two kinds of narrow training

The next picture compares the two ways of aiming a model at your own field. Use the tabs.

```viz
{
  "type": "flow",
  "title": "Continued pre-training vs task fine-tuning",
  "modes": [
    {
      "name": "Continued pre-training",
      "tagline": "Teach the model more about a field, using unlabelled text.",
      "inputLabel": "Training text",
      "modelLabel": "Predict the next token",
      "outputLabel": "What the model gains",
      "examples": [
        { "label": "Clinical notes", "input": "“Patient reports chest pain for two days, ECG normal, advised …”", "output": "Better at medical wording and facts. Still just continues text; it will not follow an instruction unless tuned later." },
        { "label": "Filipino news", "input": "A large collection of Tagalog and Taglish articles", "output": "A better feel for Filipino grammar, names and places." }
      ],
      "note": "Illustrative examples. No labels are needed, only text, but a lot of it."
    },
    {
      "name": "Task fine-tuning",
      "tagline": "Teach the model to do one job, using labelled examples.",
      "inputLabel": "Input and label",
      "modelLabel": "Predict the label",
      "outputLabel": "What the model gains",
      "examples": [
        { "label": "Urgent email?", "input": "“The lab door is stuck and a class starts in 5 minutes” → label: urgent", "output": "A model that sorts new emails into urgent and not urgent." },
        { "label": "Sentiment", "input": "“The canteen line was too long” → label: negative", "output": "A model that marks reviews positive or negative." }
      ],
      "note": "Illustrative examples. A few hundred to a few thousand labelled examples can be enough, but the skill stays narrow."
    }
  ]
}
```

Notice that only one of the two needs labels. If you have lots of domain text and no labels, continued pre-training is the natural fit; if you have labels for one decision, task fine-tuning is.

### A short history

Dates help you place the ideas. These are the ones to remember.

```viz
{
  "type": "timeline",
  "title": "Milestones in training language models",
  "events": [
    { "year": "2017", "title": "Learning from human preferences", "text": "Christiano et al. showed that a reward model learned from human comparisons can train an agent, the idea that later became RLHF for language models." },
    { "year": "2019", "title": "GPT-2", "text": "Radford et al. released a family of pre-trained language models, the largest with 1.5 billion parameters. The small one in our lab has about 124 million." },
    { "year": "2020", "title": "GPT-3", "text": "Brown et al. showed that a 175-billion-parameter model can follow a task written as a prompt, with a few examples inside the prompt." },
    { "year": "2021", "title": "LoRA", "text": "Hu et al. proposed freezing the pre-trained weights and learning a small low-rank change, which cut the number of trained parameters dramatically." },
    { "year": "2022", "title": "InstructGPT", "text": "Ouyang et al. used instruction tuning plus RLHF to make a model that people preferred even though it was much smaller than GPT-3." },
    { "year": "Late 2022", "title": "ChatGPT", "text": "A chat assistant trained with similar methods was released to the public and made chat models widely known." },
    { "year": "2023", "title": "DPO and QLoRA", "text": "Rafailov et al. showed how to learn from preferences without a separate reward model, and Dettmers et al. combined 4-bit quantisation with LoRA to fine-tune very large models on one GPU." }
  ]
}
```

Notice how short the gaps are. ==almost everything in this lesson was invented in the last few years==, so expect the details to keep moving while the ideas stay.

### Catastrophic forgetting in one picture

Forgetting is the price of changing weights. The idea is easy to see with the next flow figure, which uses made-up numbers to show what a student typically sees when they plot loss on two kinds of text during fine-tuning.

```viz
{
  "type": "flow",
  "title": "Learning the new without losing the old",
  "modes": [
    {
      "name": "A few steps, small learning rate",
      "tagline": "The model learns the new style and keeps most of its general ability.",
      "inputLabel": "Check after fine-tuning",
      "modelLabel": "Tuned model",
      "outputLabel": "Loss (lower is better)",
      "examples": [
        { "label": "New task", "input": "Loss on held-out instruction examples", "output": "3.1 before, 1.9 after (illustrative)" },
        { "label": "General text", "input": "Loss on ordinary news and stories", "output": "3.3 before, 3.4 after (illustrative)" }
      ],
      "note": "Illustrative numbers authored for teaching, not measured from a trained model."
    },
    {
      "name": "Many steps, large learning rate",
      "tagline": "The model masters the new examples but its general ability drifts.",
      "inputLabel": "Check after fine-tuning",
      "modelLabel": "Tuned model",
      "outputLabel": "Loss (lower is better)",
      "examples": [
        { "label": "New task", "input": "Loss on the training examples", "output": "0.4 (illustrative): it has nearly memorised them" },
        { "label": "General text", "input": "Loss on ordinary news and stories", "output": "4.6 (illustrative): clearly worse than before" }
      ],
      "note": "Illustrative numbers authored for teaching. In the lab you will measure the real effect on GPT-2."
    }
  ]
}
```

Notice that the loss on the new task can fall while the loss on general text rises. ==Always measure both== after any fine-tuning.

## Supervised fine-tuning and instruction tuning {rail=SFT}

{lede} [[Supervised fine-tuning (SFT)|Training a pre-trained model on examples of the behaviour you want, as input and target-output pairs, using the same next-token loss but counting it only on the response.]] is the simplest way to change behaviour: ==show the model good answers and train it to reproduce them==. Nothing about the maths is new. What is new is what we show it, and which tokens we let count.

### What an instruction example looks like

An instruction example has up to three parts. The *instruction* says what to do. The optional *input* is the material to work on. The *response* is the answer you would be happy to receive. In a Filipino school helper bot, three examples might be:

- Instruction: “Explain what a noun is to a Grade 7 student.” Response: a two-sentence explanation with an example in Filipino and English.
- Instruction: “Summarise this announcement in one sentence.” Input: the text of the announcement. Response: one clear sentence.
- Instruction: “When does the library open?” Response: “The library opens at 7:30 a.m. on school days.”

Thousands of such examples, written by people or collected and cleaned, are the training set. They are *demonstrations*: each one shows a finished answer.

### The chat template

Chat models read a conversation, not a bare question, so the examples are first turned into one long string with special markers that say who is speaking. A [[chat template|The fixed text format that wraps system, user and assistant messages in special marker tokens so the model can tell the speakers apart. Each model family has its own template.]] might look like this. The marker names are made up for illustration, since every model family uses its own.

```text
<|system|>You are a helpful assistant for Mabini High School.<|end|>
<|user|>Where is the library?<|end|>
<|assistant|>Level two of Rizal Hall.<|end|>
```

The rule that matters is consistency. ==Use the same template when you train and when you use the model.== If you train with one format and chat with another, the model sees a layout it was never tuned on and behaves worse, in the same way that a bakery apprentice would be confused if customers suddenly spoke in a different language.

```viz
{
  "type": "flow",
  "title": "A raw question and a templated conversation",
  "modes": [
    {
      "name": "Raw text",
      "tagline": "No markers: the model sees a question and keeps writing.",
      "inputLabel": "What the model reads",
      "modelLabel": "Base model",
      "outputLabel": "What it continues with (illustrative)",
      "examples": [
        { "label": "Library question", "input": "Where is the library?", "output": "Where is the canteen? Where is the gym? Where is the principal’s office?" }
      ],
      "note": "Hand-written illustration of a typical base-model continuation."
    },
    {
      "name": "Chat template",
      "tagline": "Markers say who speaks and where the answer begins.",
      "inputLabel": "What the model reads",
      "modelLabel": "Instruction-tuned model",
      "outputLabel": "What it continues with (illustrative)",
      "examples": [
        { "label": "Library question", "input": "<|system|>You are a helpful assistant for Mabini High School.<|end|><|user|>Where is the library?<|end|><|assistant|>", "output": "Level two of Rizal Hall.<|end|>" }
      ],
      "note": "Marker names are invented for teaching. The model learns to write after the assistant marker and to stop at the end marker."
    }
  ]
}
```

Notice that the template does two jobs. It tells the model *where its turn starts*, and the end marker teaches it *when to stop*, which is why tuned models stop talking while base models ramble on.

### The loss: count only the response

During SFT the model still predicts the next token at every position, as in Week 8. But the first part of every sequence, the system message and the user’s question, is something the model is *given*, not something it should learn to write. So we ==compute the loss only on the response tokens== and set the prompt positions aside. This is called [[masking|Excluding some positions from the loss, by giving them a label that the loss ignores, so the model is not trained to reproduce them.]] the prompt. Let R be the set of response positions and |R| how many there are:

```viz
{
  "type": "equation",
  "title": "The SFT loss with the prompt masked",
  "intro": "Select a symbol to read what it means. The example has 12 tokens: 7 of prompt and 5 of response.",
  "formula": [ { "t": "L", "id": "l" }, " = −", { "t": "(1/|R|)", "id": "mean" }, " ", { "t": "Σ", "id": "sum" }, "", { "t": "(t∈R)", "id": "set" }, " ln ", { "t": "P(y_t | prompt, y_<t)", "id": "prob" } ],
  "parts": [
    { "id": "l", "name": "L (loss)", "meaning": "The average surprise of the model at the response tokens. Lower is better." },
    { "id": "mean", "name": "1/|R| (average)", "meaning": "Divide by the number of response tokens so the loss does not grow just because answers are long." },
    { "id": "sum", "name": "Σ (sum)", "meaning": "Add up the term that follows over every position in the set." },
    { "id": "set", "name": "t ∈ R (response positions)", "meaning": "Only the positions that belong to the response. Prompt positions are left out (masked)." },
    { "id": "prob", "name": "P(y_t | prompt, y_<t)", "meaning": "The probability the model gave to the correct token y_t, given the whole prompt and the response tokens before it." }
  ],
  "steps": [
    { "title": "Probabilities of the 5 response tokens", "expr": "0.5, 0.4, 0.8, 0.9, 0.25", "note": "Illustrative values from a pretend model reading “Level two of Rizal Hall”." },
    { "title": "Surprise of each, −ln p", "expr": "0.693, 0.916, 0.223, 0.105, 1.386", "note": "A rare token (0.25) costs much more than a likely one (0.9)." },
    { "title": "Add and divide by 5", "expr": "(0.693 + 0.916 + 0.223 + 0.105 + 1.386) / 5 = 3.324 / 5 = 0.665", "note": "The masked SFT loss for this example." },
    { "title": "If the prompt counted too", "expr": "Seven prompt tokens with p = 0.1, 0.05, 0.2, 0.1, 0.25, 0.5, 0.05 add 14.286. (14.286 + 3.324) / 12 = 1.467", "note": "The loss more than doubles and is dominated by tokens the model should never be asked to write. The probabilities of the prompt tokens are made up. For simplicity we score all 12 positions." }
  ]
}
```

Notice the gap between 0.665 and 1.467. If the prompt counted, most of the training signal would go on learning to predict users’ questions, and the loss would hide how well the model answers.

The next figure shows the same twelve positions as a strip, so you can see which ones the loss uses.

```viz
{
  "type": "heatmap",
  "title": "Which positions count in the SFT loss?",
  "rows": ["Position"],
  "cols": ["<user>", "Where", "is", "the", "library", "?", "<assistant>", "Level", "two", "of", "Rizal", "Hall"],
  "corner": "",
  "showValues": true,
  "tabs": [
    { "name": "Mask (1 = counts)", "caption": "The first 7 tokens are the prompt and are masked out (0). The last 5 are the response and count (1).", "W": [[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]] },
    { "name": "−ln p (surprise)", "caption": "Illustrative surprise at each token. Only the last five (response) cells are averaged: (0.69 + 0.92 + 0.22 + 0.11 + 1.39) / 5 = 0.665.", "W": [[2.3, 3.0, 1.61, 2.3, 1.39, 0.69, 3.0, 0.69, 0.92, 0.22, 0.11, 1.39]] }
  ]
}
```

The same five response tokens, drawn as a teacher-forcing strip, show where the loss comes from.

```viz
{
  "type": "shiftloss",
  "title": "The response part of the loss, token by token",
  "intro": "Each response token is predicted from everything before it. The mean of the five surprises is the masked SFT loss of this example.",
  "tokens": ["<assistant>", "Level", "two", "of", "Rizal", "Hall"],
  "p": [0.5, 0.4, 0.8, 0.9, 0.25],
  "note": "The probabilities are illustrative, as in the worked example above, not taken from a trained model.",
  "checks": [ { "loss": 0.665, "perplexity": 1.944 } ]
}
```

Notice that the mean is 0.665 and the perplexity is e^0.665 = 1.94: on average the model is about as unsure as if it were choosing between two equally likely words.

In practice libraries do this with a label of −100 on the masked positions, which the loss function skips. You will do exactly this in Lab Part 1.

### One training step

A single step of SFT is one pass of the loop from Week 3, with the mask added. The pipeline below walks through it.

```viz
{
  "type": "pipeline",
  "title": "One SFT training step",
  "intro": "Step through the loop. The example is the library question.",
  "vertical": false,
  "nodes": [
    { "id": "tok", "label": "Tokenise", "sub": "apply the template" },
    { "id": "mask", "label": "Mask", "sub": "prompt labels = −100" },
    { "id": "fwd", "label": "Forward", "sub": "logits at every position" },
    { "id": "loss", "label": "Loss", "sub": "mean over response" },
    { "id": "bwd", "label": "Backward", "sub": "gradients" },
    { "id": "upd", "label": "Update", "sub": "small learning rate" }
  ],
  "steps": [
    { "title": "Tokenise with the template", "body": "Wrap the system message, question and answer in the chat template and cut the string into tokens. Use exactly the format you will use at chat time.", "active": ["tok"], "example": "12 tokens: 7 prompt, 5 response." },
    { "title": "Mask the prompt", "body": "Copy the tokens to make the labels, then replace the label of every prompt position by −100 so the loss will ignore it.", "active": ["tok", "mask"], "example": "Labels: −100 ×7, then the 5 response tokens." },
    { "title": "Forward pass", "body": "The model reads all tokens at once (the causal mask of Week 8 hides the future) and produces a probability over the vocabulary at every position.", "active": ["fwd"], "example": "Output shape: 12 positions × vocabulary." },
    { "title": "Compute the loss", "body": "Compare the predictions with the labels shifted by one position, skipping the masked positions, and average −ln p over the response tokens.", "active": ["mask", "loss"], "example": "Loss = 0.665 in our example." },
    { "title": "Backward pass", "body": "Backpropagation (Week 3) sends the gradient of the loss back to every trainable weight.", "active": ["loss", "bwd"], "example": "One gradient for each trained parameter." },
    { "title": "Update the weights", "body": "Take a small step against the gradient. SFT uses a small learning rate, so the model changes gently and keeps what it learned in pre-training.", "active": ["bwd", "upd"], "example": "w ← w − η × gradient, with η around 1e-5 for full fine-tuning (typical)." }
  ]
}
```

Notice that the only differences from ordinary training are the template at the start and the mask in the middle. The rest is the loop you already know.

### How much data do you need?

Less than you might expect, if the examples are good. The InstructGPT work (Ouyang et al., 2022) used tens of thousands of demonstrations for its supervised step. Zhou et al. (2023, the LIMA paper) then showed that ==about 1,000 carefully chosen examples can already give a strong chat style==, because the knowledge comes from pre-training and the examples mostly teach the format and tone. The lesson is ==quality over quantity==.

What does “quality” mean here?

- **Correct.** The model copies mistakes. A wrong answer in the data is a lesson in being wrong.
- **Consistent.** If half the answers are formal and half are slang, the model will be erratic.
- **Diverse.** Many kinds of questions, lengths and difficulty, so the model does not learn one narrow trick.
- **No duplicates.** Repeating one example many times makes the model memorise it.

### Overfitting and forgetting

With tiny data, the danger from Week 2 returns. Imagine ten examples and many passes over them. The training loss falls towards zero because the model memorises, but the validation loss, measured on examples it never trained on, stops falling and then rises. That is [[overfitting|Fitting the training examples so closely, noise included, that performance on new examples gets worse.]], and the cure is the one you know: hold out validation data, watch both curves, and ==stop early when the validation loss turns upward==. Use a small learning rate and few passes.

We met forgetting earlier. In SFT it shows up as a model that now answers school questions politely but has become worse at ordinary writing or at another language. You check by also measuring the loss on a small set of general text before and after, which you will do in Lab Part 3.

### How do you know it worked?

Three checks work together.

1. **Held-out loss.** The masked SFT loss on examples the model never saw. It is fast and cheap but only says whether the model predicts the reference answers.
2. **Human review.** Read a sample of real outputs. Does it answer, in the right tone, correctly? This catches what a number misses.
3. **Win rate against a baseline.** Show a judge, human or a stronger model, the baseline’s answer and the tuned model’s answer to the same prompt and count how often the tuned model wins.

A win rate is a proportion, so it comes with uncertainty. Suppose the tuned model won 27 of 40 comparisons.

```viz
{
  "type": "equation",
  "title": "Win rate and its standard error",
  "intro": "The standard error says how much the win rate would wobble if you repeated the test with a fresh set of prompts.",
  "formula": [ { "t": "p̂", "id": "phat" }, " = ", { "t": "wins / n", "id": "frac" }, ",   SE = √(", { "t": "p̂(1 − p̂)", "id": "var" }, " / ", { "t": "n", "id": "n" }, ")" ],
  "parts": [
    { "id": "phat", "name": "p̂ (win rate)", "meaning": "The share of comparisons the tuned model won." },
    { "id": "frac", "name": "wins / n", "meaning": "Number of wins divided by the number of comparisons." },
    { "id": "var", "name": "p̂(1 − p̂)", "meaning": "How mixed the outcomes are. It is largest when wins and losses are about equal." },
    { "id": "n", "name": "n (comparisons)", "meaning": "How many prompts you compared. More comparisons give a smaller standard error." }
  ],
  "steps": [
    { "title": "Win rate", "expr": "p̂ = 27 / 40 = 0.675 = 67.5%", "note": "The tuned model won a bit more than two thirds of the time." },
    { "title": "Standard error", "expr": "SE = √(0.675 × 0.325 / 40) = √0.005484 = 0.074", "note": "About 7.4 percentage points." },
    { "title": "A rough 95% range", "expr": "0.675 ± 1.96 × 0.074 = 0.675 ± 0.145, so 53% to 82%", "note": "With only 40 comparisons you cannot tell a 55% win rate from an 80% one. Use more prompts when the difference matters." }
  ]
}
```

Notice that 67.5% sounds convincing, yet the range runs from barely better than a coin flip to clearly better. ==small tests give wide ranges==, and this is a good reason to report the number of comparisons together with the win rate.

### Check the SFT ideas

Sort eight statements into the stage they describe. Some are true of both stages.

```viz
{
  "type": "sorter",
  "title": "SFT, pre-training, or both?",
  "intro": "Drag each statement to the stage it describes.",
  "bins": [
    { "id": "sft", "label": "SFT only" },
    { "id": "pretraining", "label": "Pre-training only" },
    { "id": "both", "label": "Both" }
  ],
  "cards": [
    { "name": "Computes the loss only on the response tokens", "answer": "sft", "why": "Pre-training counts every token; SFT masks the prompt so it learns to answer, not to write questions." },
    { "name": "Uses a next-token cross-entropy loss", "answer": "both", "why": "SFT keeps the objective of pre-training; only the data and the mask change." },
    { "name": "Needs trillions of tokens of text", "answer": "pretraining", "why": "SFT works with as few as about a thousand good examples." },
    { "name": "Needs a chat template with speaker markers", "answer": "sft", "why": "The template matters when you teach the format of a conversation." },
    { "name": "Typically uses a small learning rate such as 1e-5 for full fine-tuning", "answer": "sft", "why": "A gentle step keeps what pre-training learned; pre-training uses larger rates with schedules." },
    { "name": "Updates the weights by backpropagation and gradient descent", "answer": "both", "why": "Both stages train with gradients, as in Week 3." },
    { "name": "Learns the knowledge of language and the world from raw text", "answer": "pretraining", "why": "The broad knowledge comes from reading at huge scale; SFT mostly shapes how it is used." },
    { "name": "Can overfit if you train many times over a handful of examples", "answer": "sft", "why": "Tiny datasets are memorised; pre-training data are so large that each text is seen about once." }
  ]
}
```

## Why chat models need it {rail=Alignment}

{lede} Ask a base model a question and it may not answer, because ==it was trained to continue text, not to help==. Turning a text continuer into an assistant takes more than knowledge: it takes behaviour, and the behaviour has to be taught.

### A base model continues, a chat model answers

Imagine a tutor who has read every book in the school library but has never been asked to teach. Say “What is photosynthesis?” and the tutor, used to quiz sheets, might continue with “What is respiration? What is osmosis?” because quiz sheets list questions. That is what a base model does. Instruction tuning is the tutor learning, from many examples, that a question from a student is a request to be answered.

Pre-training taught a great deal: language, facts, styles, even some reasoning. What it did not teach is *which* continuation you want. The next-token objective rewards matching the internet, and the internet contains polite answers, rude answers, wrong answers, jokes and spam. The model learned to produce all of them, in the right proportions, for the right contexts.

```viz
{
  "type": "sampling",
  "title": "A base model’s next word after a question",
  "prompt": "Q: How do I boil an egg?\nA:",
  "tokens": [
    { "t": " Place", "logit": 2.2 },
    { "t": " Q", "logit": 2.0 },
    { "t": " Why", "logit": 1.6 },
    { "t": " I", "logit": 1.4 },
    { "t": " Boil", "logit": 1.2 },
    { "t": " lol", "logit": 0.2 }
  ],
  "note": "The logits are made up for teaching, not taken from a real model. They show that a base model puts real probability on non-answers."
}
```

Slide the temperature up and notice how often the draws are something other than a helpful answer. Even when the best guess is good, sampling will sometimes pick the others. A tuned model concentrates its probability on helpful continuations.

### Helpful, honest, harmless

What do we want instead? A common summary comes from Askell et al. (2021) and Bai et al. (2022): a good assistant should be [[helpful, honest and harmless (HHH)|A short way to state what an AI assistant should be: it helps with what the user really needs, tells the truth and admits uncertainty, and avoids causing harm.]]. These three goals sometimes pull against each other. A very helpful reply to a dangerous request would not be harmless, and a very cautious reply to a harmless request would not be helpful. Training has to find a good balance.

```viz
{
  "type": "nested",
  "title": "From text predictor to assistant",
  "levels": [
    { "name": "A next-token predictor", "blurb": "Pre-training gives a fluent model that has read a great deal. It continues any text, good or bad, and has no notion of what you wanted.", "examples": ["A base model"] },
    { "name": "An instruction follower", "blurb": "Instruction tuning teaches the model that a request deserves an answer, in the right format, and when to stop writing.", "examples": ["An SFT model"] },
    { "name": "A helpful, honest, harmless assistant", "blurb": "Preference tuning pushes the model towards answers people judge as better: clearer, truthful, careful with risk, polite in the right way.", "examples": ["A chat model after RLHF or DPO"] }
  ]
}
```

Notice that each ring sits inside the one before it: the assistant is still a next-token predictor underneath, with layers of behaviour on top.

### How a model can fail

Chat models can go wrong in recognisable ways. The examples below are illustrative, written by hand to show the pattern.

- **Ignoring the question.** Asked for the capital of Cebu province, it lists tourist spots.
- **Rambling.** It writes five paragraphs where one sentence would do.
- **Confident falsehoods.** It gives a precise date or a made-up reference, in the same calm tone as a true fact.
- **Unsafe or rude replies.** It answers a harmful request, or answers an innocent one with an insult, because both appear in its training text.
- **An unsteady tone.** Formal in one answer, slangy in the next.

Next-token prediction alone does not fix these, because the data contain all of them. ==The loss rewards imitating the text, not choosing the best text.==

### Why SFT alone is not enough

SFT fixes a lot: format, tone, stopping. But it has limits that motivate the next section.

1. **It only shows good answers.** Demonstrations tell the model what to say, but not what is *worse*. There are far more ways to be wrong than right, and nobody can write them all out.
2. **Writing answers is expensive.** A person must compose a good answer from scratch, which is slow and varies by author.
3. **==Comparing is easier than composing.==** A person can usually say which of two answers is better, even when they could not write the best answer themselves. A coach does not need to bake to say which loaf looks better.
4. **The model can only imitate.** It can match the best demonstrations but has no signal for doing better than them.

The picture below shows the same question answered at three stages. The answers are written by hand to show the pattern.

```viz
{
  "type": "flow",
  "title": "The same question at three stages",
  "modes": [
    {
      "name": "Base model",
      "tagline": "Continues the text.",
      "inputLabel": "Prompt",
      "modelLabel": "Pre-trained only",
      "outputLabel": "Typical output (hand-written, illustrative)",
      "examples": [
        { "label": "Homework help", "input": "How do I find the area of a circle?", "output": "How do I find the circumference? How do I find the area of a triangle? Chapter 4 review questions …" }
      ],
      "note": "Illustrative. Real base models vary, but this kind of continuation is common."
    },
    {
      "name": "After SFT",
      "tagline": "Answers, but not always the best answer.",
      "inputLabel": "Prompt",
      "modelLabel": "Instruction-tuned",
      "outputLabel": "Typical output (hand-written, illustrative)",
      "examples": [
        { "label": "Homework help", "input": "How do I find the area of a circle?", "output": "The area is pi times the radius squared. Area = πr²." }
      ],
      "note": "Illustrative. Correct and on topic, but terse and with no worked example."
    },
    {
      "name": "After preference tuning",
      "tagline": "Answers in the way people rate highest.",
      "inputLabel": "Prompt",
      "modelLabel": "Preference-tuned",
      "outputLabel": "Typical output (hand-written, illustrative)",
      "examples": [
        { "label": "Homework help", "input": "How do I find the area of a circle?", "output": "Use A = πr², where r is the radius. For example, if r = 5 cm, then A = 3.14 × 25 = 78.5 cm². Would you like to try one together?" }
      ],
      "note": "Illustrative. Correct, with a worked example and a friendly next step."
    }
  ]
}
```

Notice that all three answers might be acceptable to a machine, yet people reliably prefer the last. ==Preference tuning exists to capture that difference.==

### Which stage fixes what?

Sort eight behaviours by what mainly fixes them. Be careful: some problems are only reduced, not removed, by either stage.

```viz
{
  "type": "sorter",
  "title": "What fixes this behaviour?",
  "intro": "Drag each behaviour to the stage that mainly addresses it.",
  "bins": [
    { "id": "sft", "label": "Mainly fixed by SFT" },
    { "id": "pref", "label": "Needs preference tuning" },
    { "id": "neither", "label": "Not fully fixed by either" }
  ],
  "cards": [
    { "name": "The model continues a question with more questions", "answer": "sft", "why": "Showing it question and answer pairs teaches it that a question gets an answer." },
    { "name": "The model never stops and runs on after answering", "answer": "sft", "why": "The chat template’s end marker teaches it when to stop." },
    { "name": "Both answers are correct, but people reliably prefer the clearer one", "answer": "pref", "why": "Judging which of two good answers is better is exactly what comparison data capture." },
    { "name": "The model gives long answers that people find tiresome", "answer": "pref", "why": "Length and style are matters of taste, best learned from what people choose." },
    { "name": "The model is polite in some answers and curt in others", "answer": "sft", "why": "Consistent demonstrations in one tone teach a steady style; preferences polish it further." },
    { "name": "The model sometimes states a false fact with confidence", "answer": "neither", "why": "Both stages reduce it, but a model can still be wrong. Facts need checking or retrieval (Week 16)." },
    { "name": "The model may answer a cleverly disguised harmful request", "answer": "neither", "why": "Alignment training lowers the risk but does not remove it. Testing and safeguards are needed too." },
    { "name": "The model tends to agree with whatever the user says", "answer": "neither", "why": "This flattery, called sycophancy, can even be encouraged when raters like agreement, so it needs careful data and testing." }
  ]
}
```

If you placed the last three cards under a stage, re-read their explanations. ==No training stage makes a model fully truthful or fully safe==; training lowers the risk and testing finds what remains.

## RLHF: learning from human preferences {rail=RLHF}

{lede} [[Reinforcement learning from human feedback (RLHF)|A training recipe in which people compare model answers, a reward model learns to predict those preferences, and the language model is then updated to produce answers that the reward model scores highly.]] turns “people liked this answer more than that one” into ==a number the model can be trained to raise==. The approach goes back to Christiano et al. (2017), who learned a reward from human comparisons, and was applied to language models in InstructGPT (Ouyang et al., 2022).

### Three stages

RLHF has three stages, and the first is the SFT you already know.

1. **SFT.** Fine-tune the base model on demonstrations, as in Section 2. The result is the [[reference model|The frozen starting model, usually the SFT model, against which later training is compared. The KL penalty measures how far the new model has moved from it.]], written π_ref.
2. **A reward model.** Show people a prompt and two answers from the model and ask which is better. Train a [[reward model|A network that reads a prompt and an answer and outputs one number, the reward, which is higher for answers that people would prefer. It stands in for the human judge during training.]] to give the preferred answer the higher score.
3. **Reinforcement learning.** Let the model write answers, score each with the reward model, and update the model to make high-scoring answers more likely. [[PPO|Proximal policy optimisation (Schulman et al., 2017), the reinforcement learning algorithm most often used for this step. It makes small, safe updates to the model. You do not need its details in this course.]] is the usual algorithm.

In the language of reinforcement learning, the language model is the [[policy|The rule an agent uses to choose actions. For a language model, the policy is the probability it gives to each possible next token, and so to each whole answer.]] π: given a prompt x, it chooses an answer y with probability π(y|x), and the reward model returns r(x, y).

```viz
{
  "type": "pipeline",
  "title": "The three stages of RLHF",
  "intro": "Step through the stages and watch what data each one needs. The data sizes are typical orders of magnitude.",
  "vertical": false,
  "nodes": [
    { "id": "base", "label": "Base model", "sub": "pre-trained" },
    { "id": "sft", "label": "1. SFT", "sub": "demonstrations" },
    { "id": "rm", "label": "2. Reward model", "sub": "comparisons" },
    { "id": "rl", "label": "3. RL (PPO)", "sub": "prompts + reward" },
    { "id": "out", "label": "Chat model", "sub": "preference-tuned" }
  ],
  "steps": [
    { "title": "Start from the base model", "body": "A pre-trained model that continues text. It already knows language and facts but does not follow instructions.", "active": ["base"], "example": "Data so far: trillions of tokens (Weeks 8 and 9)." },
    { "title": "Stage 1: supervised fine-tuning", "body": "Train on instruction and response pairs, with the loss on the response only. The result is the reference model π_ref.", "active": ["base", "sft"], "example": "Data: about 1,000 to tens of thousands of demonstrations." },
    { "title": "Stage 2: train a reward model", "body": "For many prompts, sample two or more answers from the SFT model, ask people which is better, and train a network to score the better answer higher.", "active": ["sft", "rm"], "example": "Data: tens of thousands of human comparisons (illustrative)." },
    { "title": "Stage 3: reinforcement learning", "body": "The policy writes answers to new prompts; the reward model scores them; PPO nudges the policy towards higher scores while a penalty keeps it close to π_ref.", "active": ["sft", "rm", "rl"], "example": "Data: prompts only. The reward model supplies the feedback." },
    { "title": "The chat model", "body": "The policy after stage 3 is the model you chat with. It answers in the way that people rated highest, within the limits of the reward model.", "active": ["rl", "out"], "example": "A helpful, more consistent assistant." }
  ]
}
```

Notice that people are involved twice, first writing demonstrations and then comparing answers, but never during the RL loop itself: ==the reward model replaces them there==, which is what makes the third stage affordable.

### Stage 2 in detail: the reward model

A reward model is usually the language model itself with its last layer swapped for a single output number. Given a prompt and an answer, it returns a score r. The scores are not probabilities and have no fixed scale: only differences matter.

To train it we use a comparison: for a prompt x, a *chosen* answer y_w (the winner) and a *rejected* answer y_l (the loser). We model the chance that people prefer the winner using the [[Bradley–Terry model|A simple model of pairwise comparisons: the probability that A beats B is the sigmoid of the difference of their scores.]], which says that the probability depends only on the gap between the two scores.

```viz
{
  "type": "equation",
  "title": "The Bradley–Terry preference probability",
  "intro": "Select a symbol. The sigmoid turns any score difference into a probability between 0 and 1.",
  "formula": [ { "t": "P(y_w ≻ y_l)", "id": "p" }, " = ", { "t": "σ", "id": "sig" }, "(", { "t": "r_w", "id": "rw" }, " − ", { "t": "r_l", "id": "rl" }, ")" ],
  "parts": [
    { "id": "p", "name": "P(y_w ≻ y_l)", "meaning": "The probability that a person prefers the chosen answer y_w over the rejected answer y_l." },
    { "id": "sig", "name": "σ (sigmoid)", "meaning": "σ(z) = 1 / (1 + e^−z). It squashes any number into the range 0 to 1, with σ(0) = 0.5." },
    { "id": "rw", "name": "r_w (reward of the winner)", "meaning": "The reward model’s score for the chosen answer." },
    { "id": "rl", "name": "r_l (reward of the loser)", "meaning": "The reward model’s score for the rejected answer." }
  ],
  "steps": [
    { "title": "The gap", "expr": "r_w − r_l = 2.0 − 0.5 = 1.5", "note": "The chosen answer scores 1.5 higher." },
    { "title": "The sigmoid", "expr": "σ(1.5) = 1 / (1 + e^−1.5) = 1 / (1 + 0.2231) = 0.818", "note": "The model says the chosen answer wins 81.8% of the time." },
    { "title": "Equal scores", "expr": "σ(0) = 0.5", "note": "If the two scores are equal the model has no idea which one people prefer." }
  ]
}
```

Notice that adding 100 to both scores changes nothing. ==Rewards matter only as gaps==, which is why they have no natural zero.

The sigmoid is the S-shaped curve you met in Week 3. Read the preference probability straight from it: put the score gap on the horizontal axis and read the height.

```viz
{
  "type": "curve",
  "title": "The sigmoid turns a score gap into a probability",
  "xRange": [-6, 6],
  "yRange": [-0.1, 1.1],
  "series": [ { "fn": "sigmoid", "label": "σ(gap)" } ],
  "alt": "An S-shaped curve rising from 0 on the left, through 0.5 at a gap of 0, to 1 on the right. A gap of 1.5 gives about 0.82.",
  "note": "At a gap of 0 the probability is 0.5; at a gap of 1.5 it is 0.818; beyond a gap of 4 it is above 0.98, so very large gaps add little."
}
```

Notice how flat the curve is at the ends. Once the reward model is already sure, a bigger gap changes the probability very little, so the loss stops asking for more.

### The reward-model loss

We train the reward model so that the probability it gives to the choice people actually made is as large as possible. That is the same idea as the cross-entropy loss of Week 2: the loss is the negative log of the probability of what happened.

```viz
{
  "type": "equation",
  "title": "The preference loss",
  "intro": "The loss is small when the chosen answer scores well above the rejected one, and large when the order is wrong.",
  "formula": [ { "t": "L", "id": "l" }, " = −ln ", { "t": "σ", "id": "sig" }, "(", { "t": "r_w − r_l", "id": "gap" }, ")" ],
  "parts": [
    { "id": "l", "name": "L (loss)", "meaning": "The surprise of the model at the human’s choice. For a whole data set, average it over all comparisons." },
    { "id": "sig", "name": "σ (sigmoid)", "meaning": "Turns the score gap into the probability of the chosen answer winning." },
    { "id": "gap", "name": "r_w − r_l (gap)", "meaning": "How far the chosen answer’s score is above the rejected answer’s score. It can be negative if the model has the order wrong." }
  ],
  "steps": [
    { "title": "Correct order", "expr": "r_w = 2.0, r_l = 0.5: L = −ln 0.818 = 0.201", "note": "A small loss: the model agrees with the human." },
    { "title": "Wrong order", "expr": "r_w = 0.5, r_l = 2.0: P = σ(−1.5) = 0.182, L = −ln 0.182 = 1.701", "note": "A large loss: the model prefers the answer that people rejected. Gradient descent will push the scores apart the other way." },
    { "title": "No idea", "expr": "r_w = r_l: P = 0.5, L = −ln 0.5 = ln 2 = 0.693", "note": "A loss of 0.693 is the score of a model that is guessing. A trained reward model should beat it." }
  ]
}
```

In practice the scores come from a neural network and the loss is averaged over thousands of pairs. The figure below lets you play the network. Each pair has two sliders, one for each answer’s score.

```viz
{
  "type": "prefpair",
  "title": "Teach the reward model: a school chatbot",
  "intro": "Set the scores so that the chosen answer beats the rejected one in every pair, and watch the mean loss fall. Pair 3 starts at a tie: a loss of 0.693.",
  "pairs": [
    { "prompt": "When does the library open?", "chosen": "The library opens at 7:30 a.m. on school days. Anything else I can help with?", "rejected": "idk, just look it up", "chosenScore": 1.5, "rejectedScore": -0.5 },
    { "prompt": "Can you explain fractions?", "chosen": "A fraction shows parts of a whole: 3/4 means 3 of 4 equal pieces. Try cutting a pan de sal into 4.", "rejected": "Fractions are numbers. Next question.", "chosenScore": 0.4, "rejectedScore": 0.9 },
    { "prompt": "What is the capital of the Philippines?", "chosen": "Manila is the capital of the Philippines.", "rejected": "The capital is Cebu City, of course.", "chosenScore": 0, "rejectedScore": 0 }
  ],
  "note": "Illustrative prompts, answers and scores authored for teaching, not from a trained reward model.",
  "checks": [
    { "pair": 0, "prob": 0.8808, "loss": 0.1269 },
    { "pair": 1, "prob": 0.3775, "loss": 0.9741 },
    { "pair": 2, "prob": 0.5, "loss": 0.6931 }
  ]
}
```

Raise the chosen score in pair 2 and lower the rejected one. Notice how the mean loss falls as more pairs are ranked correctly, and how the first pair already sits at 0.127 with a gap of 2. The loss never reaches exactly zero, because a gap of 2 still leaves a probability of 0.881.

Try a second set of pairs with different gaps. Here one pair is very clear and one is confusing.

```viz
{
  "type": "prefpair",
  "title": "A second set of comparisons",
  "intro": "Rank every pair correctly. Then try making a gap very large and see how little the loss improves, even though the score keeps growing.",
  "pairs": [
    { "prompt": "Explain photosynthesis simply.", "chosen": "Plants use sunlight, water and air (carbon dioxide) to make food and give out oxygen.", "rejected": "It is a process. Look in your book.", "chosenScore": 3.0, "rejectedScore": 0.5 },
    { "prompt": "Is it fine to share my password with a friend?", "chosen": "It is better not to. Anyone with your password can act as you. Use a shared link or ask your teacher instead.", "rejected": "Sure, go ahead!", "chosenScore": -0.2, "rejectedScore": 0.6 },
    { "prompt": "Say good morning in Filipino.", "chosen": "“Magandang umaga po.” Add “po” to be polite to elders.", "rejected": "“Magandang umaga.”", "chosenScore": 1, "rejectedScore": 1 }
  ],
  "note": "Illustrative prompts, answers and scores authored for teaching. Pair 3 is a genuine judgement call: both answers are acceptable, and the second is only less polite.",
  "checks": [
    { "pair": 0, "prob": 0.9241, "loss": 0.0789 },
    { "pair": 1, "prob": 0.31, "loss": 1.1711 },
    { "pair": 2, "prob": 0.5, "loss": 0.6931 }
  ]
}
```

Notice that a real reward model can be wrong in exactly the way pair 2 starts: it may prefer an agreeable answer such as “Sure, go ahead!” to a careful one. ==Whatever the reward model prefers, the third stage will push the policy towards.==

### Stage 3 in detail: reinforcement learning with a leash

Now the policy π tries to earn a high reward. If we only maximised the reward, the policy would drift to whatever answers the reward model likes best, including odd ones. So the objective has two parts: raise the reward, but pay a price for straying from the reference model.

```viz
{
  "type": "equation",
  "title": "The KL-regularised objective",
  "intro": "We choose the policy π that makes J as large as possible. The first term is the reward; the second is the leash.",
  "formula": [ { "t": "J(π)", "id": "j" }, " = ", { "t": "E_{y∼π}[ r(x, y) ]", "id": "er" }, " − ", { "t": "β", "id": "beta" }, " · ", { "t": "KL(π ‖ π_ref)", "id": "kl" } ],
  "parts": [
    { "id": "j", "name": "J(π) (objective)", "meaning": "The score of a policy: expected reward minus the penalty for drifting. Training maximises it." },
    { "id": "er", "name": "E[r(x, y)] (expected reward)", "meaning": "The average reward the reward model gives to answers y that the policy π writes for prompt x." },
    { "id": "beta", "name": "β (penalty strength)", "meaning": "A positive number you choose. A large β keeps the policy close to the reference model; a small β lets it chase reward." },
    { "id": "kl", "name": "KL(π ‖ π_ref) (distance from the reference)", "meaning": "A measure of how different the policy’s probabilities are from the reference model’s. It is 0 when they are the same and grows as they drift apart." }
  ],
  "steps": [
    { "title": "A tiny setting", "expr": "3 answers, reference π_ref = (0.5, 0.3, 0.2), rewards r = (1, 2, 5), β = 1", "note": "Answer 3 has the highest reward but is rarely written by the reference model." },
    { "title": "Stay at the reference", "expr": "E[r] = 0.5×1 + 0.3×2 + 0.2×5 = 2.1, KL = 0, so J = 2.1", "note": "No change means no penalty. The starting score is 2.1." },
    { "title": "Always write answer 3", "expr": "E[r] = 5, KL = ln(1 / 0.2) = 1.609, so J = 5 − 1.609 = 3.391", "note": "More reward, but paying a lot for leaving the reference." },
    { "title": "The best compromise", "expr": "π* = (0.041, 0.067, 0.892): E[r] = 4.635, KL = 1.131, J = 3.504", "note": "A better score than either extreme. It leans heavily towards answer 3 without abandoning the others entirely." }
  ]
}
```

What is KL? The [[KL divergence|A measure of how different one probability distribution is from another. It is zero when they are identical and positive otherwise. It is not symmetric: KL(π‖ref) is not the same as KL(ref‖π).]] between the policy and the reference is the average, over the policy’s own answers, of the log of the ratio of their probabilities. Select the parts below and try it on a small policy.

```viz
{
  "type": "equation",
  "title": "KL divergence between two answer distributions",
  "intro": "KL adds up, over answers, how much more (or less) likely the policy makes each answer than the reference does.",
  "formula": [ { "t": "KL(π ‖ π_ref)", "id": "kl" }, " = ", { "t": "Σ", "id": "sum" }, " ", { "t": "π(y)", "id": "pi" }, " · ln ", { "t": "π(y) / π_ref(y)", "id": "ratio" } ],
  "parts": [
    { "id": "kl", "name": "KL(π ‖ π_ref)", "meaning": "How far the policy has moved from the reference. Zero if they are identical." },
    { "id": "sum", "name": "Σ (sum over answers)", "meaning": "Add the terms for every possible answer y." },
    { "id": "pi", "name": "π(y) (weight)", "meaning": "The policy’s probability of answer y. Answers the policy often writes count most." },
    { "id": "ratio", "name": "π(y) / π_ref(y)", "meaning": "How much more likely the policy has made answer y than the reference. A ratio of 1 gives ln 1 = 0 and no penalty." }
  ],
  "steps": [
    { "title": "Policy and reference", "expr": "π = (0.2, 0.3, 0.5), π_ref = (0.5, 0.3, 0.2)", "note": "The policy moved weight from answer 1 to answer 3." },
    { "title": "Each term", "expr": "0.2×ln(0.2/0.5) = −0.183,   0.3×ln(0.3/0.3) = 0,   0.5×ln(0.5/0.2) = +0.458", "note": "Answer 2 did not change, so it adds nothing." },
    { "title": "Add them", "expr": "KL = −0.183 + 0 + 0.458 = 0.275", "note": "With rewards (1, 2, 5) and β = 1: E[r] = 3.3 and J = 3.3 − 0.275 = 3.025." }
  ]
}
```

Notice that the terms can be negative, but their total never is. ==KL is zero only when the policy equals the reference.==

### The best policy has a formula

For this objective the best policy can be written down. It starts from the reference and ==tilts each answer by e raised to its reward divided by β==, then rescales so that the probabilities add up to 1.

```viz
{
  "type": "equation",
  "title": "The optimal KL-regularised policy",
  "intro": "No search is needed for a small example: multiply, add, divide.",
  "formula": [ { "t": "π*(y)", "id": "pistar" }, " = ", { "t": "π_ref(y)", "id": "ref" }, " · ", { "t": "exp( r(y) / β )", "id": "tilt" }, " / ", { "t": "Z", "id": "z" } ],
  "parts": [
    { "id": "pistar", "name": "π*(y) (best policy)", "meaning": "The probability of answer y under the policy that makes J as large as possible." },
    { "id": "ref", "name": "π_ref(y)", "meaning": "The reference model’s probability of the answer. Common answers start with an advantage." },
    { "id": "tilt", "name": "exp(r(y) / β) (tilt)", "meaning": "A boost that grows with the reward. When β is small the boost is huge; when β is large it is close to 1." },
    { "id": "z", "name": "Z (normaliser)", "meaning": "The sum of π_ref(y) × exp(r(y)/β) over all answers. Dividing by it makes the probabilities add to 1." }
  ],
  "steps": [
    { "title": "Tilt each answer (β = 1)", "expr": "0.5×e¹ = 1.359,   0.3×e² = 2.217,   0.2×e⁵ = 29.683", "note": "Answer 3’s high reward lets it overcome a small starting probability." },
    { "title": "Normaliser", "expr": "Z = 1.359 + 2.217 + 29.683 = 33.259", "note": "The sum of the tilted values." },
    { "title": "Divide", "expr": "π* = (1.359, 2.217, 29.683) / 33.259 = (0.041, 0.067, 0.892)", "note": "The same answer as the table above. With β = 10 the result is (0.443, 0.293, 0.264), close to the reference; with β = 0.2 it is (0.000, 0.000, 1.000)." }
  ]
}
```

Now try it yourself. The sliders set the policy; the button jumps to π*; the β slider sets the leash.

```viz
{
  "type": "klreward",
  "title": "Reward against the leash: the school helper",
  "intro": "Move the policy sliders to raise the expected reward and see what happens to KL and J. Then press the jump button and slide β from 10 down to 0.2: the best policy slides from the reference onto the top-reward answer.",
  "beta": 1,
  "answers": [
    { "text": "A short, correct answer", "reward": 1, "ref": 0.5 },
    { "text": "A correct answer with a worked example", "reward": 2, "ref": 0.3 },
    { "text": "A very long answer full of flattery", "reward": 5, "ref": 0.2 }
  ],
  "note": "Illustrative rewards and reference probabilities authored for teaching. The third answer is the kind of text a flawed reward model might wrongly score very highly.",
  "checks": [
    { "beta": 1, "opt": [0.041, 0.067, 0.892], "J": 3.504 },
    { "beta": 10, "opt": [0.443, 0.293, 0.264], "J": 2.221 },
    { "beta": 0.2, "opt": [0.0, 0.0, 1.0], "J": 4.678 }
  ]
}
```

At β = 10 the best policy barely moves from the reference (J = 2.221, KL close to 0.013). At β = 0.2 it puts all its weight on the long flattering answer. That collapse is the next idea.

### Reward hacking

The reward model is a learned guess about what people like, not the truth. It was trained on a limited set of comparisons and may have quirks: perhaps it likes long answers, confident tone, or flattery. A policy that is allowed to chase the reward will find and exploit those quirks. This is called [[reward hacking|When a model finds answers that earn a high score from the reward model without being truly good by human standards. The score is only a proxy for what people want.]], and it is a case of *Goodhart’s law*: ==when a measure becomes the target, it stops being a good measure==.

The KL penalty is the main defence. It keeps the policy near a model that writes fluent, sensible text, and so limits how far it can go in chasing quirks. A second figure with different answers shows the same story with four answers.

```viz
{
  "type": "klreward",
  "title": "Four answers and a tempting quirk",
  "intro": "The fourth answer has the highest reward and the lowest starting probability. Find the β at which the best policy stops caring about the other three.",
  "beta": 2,
  "answers": [
    { "text": "A plain, correct answer", "reward": 0.5, "ref": 0.6 },
    { "text": "A correct answer with an example", "reward": 1, "ref": 0.25 },
    { "text": "A polite refusal", "reward": 0, "ref": 0.1 },
    { "text": "An answer stuffed with praise for the user", "reward": 3, "ref": 0.05 }
  ],
  "note": "Illustrative rewards and probabilities authored for teaching, not measured from a model.",
  "checks": [
    { "beta": 2, "opt": [0.511, 0.274, 0.066, 0.149], "J": 0.82 },
    { "beta": 0.5, "opt": [0.069, 0.078, 0.004, 0.849], "J": 1.584 }
  ]
}
```

At β = 2 the praise-stuffed answer rises from 5% to about 15%. At β = 0.5 it takes about 85%. No human would judge that the best of the four, and ==the lower you set β, the more the policy follows the reward model’s mistakes==. Setting β is a judgement about how far you trust the reward model.

### DPO: skip the reward model and the RL loop

RLHF works but is intricate: it needs a reward model, an RL loop, and several models in memory at once. [[Direct preference optimisation (DPO)|A method (Rafailov et al., 2023) that trains the language model directly on pairs of chosen and rejected answers with a single supervised-style loss, without a separate reward model or a reinforcement learning loop.]] observes that the optimal policy above implies a reward in terms of the policy itself: r(y) = β·ln(π*(y)/π_ref(y)) plus a term that depends only on the prompt. Put that into the Bradley–Terry loss and the reward model disappears.

```viz
{
  "type": "equation",
  "title": "The DPO loss",
  "intro": "It has the shape of the preference loss, but the scores are replaced by log-ratios of the policy and the reference. Select each part.",
  "formula": [ { "t": "L", "id": "l" }, " = −ln ", { "t": "σ", "id": "sig" }, "( ", { "t": "β", "id": "beta" }, " · [ ", { "t": "ln π(y_w|x)/π_ref(y_w|x)", "id": "win" }, " − ", { "t": "ln π(y_l|x)/π_ref(y_l|x)", "id": "lose" }, " ] )" ],
  "parts": [
    { "id": "l", "name": "L (loss)", "meaning": "Averaged over the data set of chosen and rejected pairs." },
    { "id": "sig", "name": "σ (sigmoid)", "meaning": "Turns the number in brackets into a probability that the chosen answer wins." },
    { "id": "beta", "name": "β (scale)", "meaning": "Plays the role of the leash again: it sets how strongly a change in probabilities counts." },
    { "id": "win", "name": "ln π(y_w|x) / π_ref(y_w|x)", "meaning": "How much the policy has raised the log-probability of the chosen answer compared with the reference. Positive if it is now more likely." },
    { "id": "lose", "name": "ln π(y_l|x) / π_ref(y_l|x)", "meaning": "The same for the rejected answer. We want this to be negative: less likely than before." }
  ],
  "steps": [
    { "title": "The chosen answer", "expr": "π(y_w) = 0.4, π_ref(y_w) = 0.3: ln(0.4/0.3) = 0.288", "note": "The policy made the chosen answer more likely than the reference did." },
    { "title": "The rejected answer", "expr": "π(y_l) = 0.1, π_ref(y_l) = 0.2: ln(0.1/0.2) = −0.693", "note": "It made the rejected answer less likely." },
    { "title": "The bracket and the scale", "expr": "β × (0.288 − (−0.693)) = 0.1 × 0.981 = 0.0981", "note": "With β = 0.1, which is a typical size." },
    { "title": "Sigmoid and loss", "expr": "σ(0.0981) = 0.5245,  L = −ln 0.5245 = 0.645", "note": "A loss just under ln 2 = 0.693, because the policy has barely moved from the reference yet. Training widens the gap and the loss falls." }
  ]
}
```

Notice that the policy and the reference appear only as probabilities of the two answers. ==No reward model, no sampling loop==: just the same kind of gradient step as SFT, on pairs.

```viz
{
  "type": "flow",
  "title": "RLHF compared with DPO",
  "modes": [
    {
      "name": "RLHF (with PPO)",
      "tagline": "Learn a reward, then use it in a reinforcement-learning loop.",
      "inputLabel": "Data",
      "modelLabel": "Models in the loop",
      "outputLabel": "Result",
      "examples": [
        { "label": "What you need", "input": "Comparison data for the reward model, then prompts for the RL loop, with the policy, the reference, the reward model (and a value model) all loaded", "output": "A policy tuned against the reward model, with a KL leash to the reference." },
        { "label": "Main difficulty", "input": "Sampling answers during training, tuning the RL settings, keeping training stable", "output": "More compute and more things that can go wrong; reward hacking is a risk." }
      ],
      "note": "A general description. Details vary between implementations."
    },
    {
      "name": "DPO",
      "tagline": "Train directly on chosen and rejected pairs with one loss.",
      "inputLabel": "Data",
      "modelLabel": "Models in the loop",
      "outputLabel": "Result",
      "examples": [
        { "label": "What you need", "input": "Pairs of chosen and rejected answers, with the policy being trained and a frozen reference model", "output": "A policy that raises chosen answers and lowers rejected ones relative to the reference." },
        { "label": "Main difficulty", "input": "The quality of the pairs matters a great deal; the data are fixed, not sampled from the current policy", "output": "Simpler and cheaper to run, but it still inherits the biases in the comparison data." }
      ],
      "note": "A general description. DPO is popular because it is simpler; neither method is always better."
    }
  ]
}
```

Notice that both start from the same raw material, human comparisons. They differ in what they do with it, the way a coach can either train a judge to score you or simply tell you which of two runs was better.

### AI feedback, and the costs

Labelling is expensive, so researchers also ask a model to give the feedback. In *RLAIF*, reinforcement learning from AI feedback, an AI model rates the answers instead of, or alongside, people. *Constitutional AI* (Bai et al., 2022) is a well-known version in which the AI judges answers against a short list of written principles. These methods lower the cost of labels, but the biases of the judging model flow into the result, so people still check.

RLHF has real costs and risks, and you should know them.

- **Labeller quality and bias.** Preferences come from a group of people with their own tastes and blind spots, who may disagree with each other. ==The model learns the group’s preferences, not everyone’s.==
- **Reward hacking.** As shown above, the policy may exploit the reward model’s quirks.
- **Sycophancy.** If raters tend to prefer answers that agree with them, the model can learn to flatter instead of to tell the truth.
- **Over-refusal.** A model rewarded for avoiding harm can learn to refuse harmless requests.
- **Cost.** Writing demonstrations, collecting comparisons and running the loop all take time and money.

### Which part is which?

Sort eight cards by the stage or method they describe.

```viz
{
  "type": "sorter",
  "title": "Reward model, RL step, SFT or DPO?",
  "intro": "Drag each card to the part of the pipeline it belongs to.",
  "bins": [
    { "id": "sft", "label": "SFT" },
    { "id": "rm", "label": "Reward model" },
    { "id": "rl", "label": "RL step" },
    { "id": "dpo", "label": "DPO" }
  ],
  "cards": [
    { "name": "Trained on a person’s chosen and rejected answers to give the chosen one a higher score", "answer": "rm", "why": "Learning a score from comparisons with the Bradley–Terry loss is how the reward model is trained." },
    { "name": "Samples answers from the model and scores each one with a reward", "answer": "rl", "why": "Writing answers and scoring them is the RL loop of stage 3." },
    { "name": "Trained on demonstrations with the loss on the response tokens only", "answer": "sft", "why": "Demonstrations with a masked prompt are SFT." },
    { "name": "Trains the policy on pairs using log-probability ratios against a frozen reference", "answer": "dpo", "why": "DPO’s loss compares the policy with the reference on the chosen and rejected answers." },
    { "name": "Objective is expected reward minus β times KL from the reference", "answer": "rl", "why": "This is the KL-regularised objective maximised in stage 3." },
    { "name": "Produces a single number for a prompt and an answer", "answer": "rm", "why": "A reward model outputs one score for each prompt and answer pair." },
    { "name": "Needs no separate reward model and no reinforcement-learning loop", "answer": "dpo", "why": "That is exactly what DPO removes." },
    { "name": "Its result becomes the reference model π_ref", "answer": "sft", "why": "The SFT model is the starting point and the reference for the KL penalty." }
  ]
}
```

## PEFT: fine-tune a small add-on {rail=PEFT}

{lede} Everything so far changed all of the model’s weights, which is simple but heavy. [[Parameter-efficient fine-tuning (PEFT)|A family of methods that adapt a large pre-trained model by training only a small number of extra or selected parameters, while the original weights stay frozen.]] asks a better question: ==can we get most of the benefit by training only a tiny part?== For many tasks the answer is yes, and LoRA is the best-known way to do it.

### The memory problem

Training needs far more memory than the model itself. For every parameter the GPU must hold not only the weight, but also its gradient and the optimiser’s running averages. Adam (Week 3) keeps two numbers per parameter, a running mean of gradients and a running mean of squared gradients. In the common *mixed-precision* setup, the bill per parameter is:

| Item | Bytes per parameter |
|---|---|
| Weights in 16-bit | 2 |
| Gradients in 16-bit | 2 |
| Master copy of weights in 32-bit | 4 |
| Adam’s two moments in 32-bit (4 + 4) | 8 |
| Total | 16 |

```viz
{
  "type": "equation",
  "title": "Memory for full fine-tuning",
  "intro": "Multiply the number of parameters by 16 bytes. This ignores activations, which add more.",
  "formula": [ { "t": "Memory", "id": "mem" }, " ≈ ", { "t": "N", "id": "n" }, " × ", { "t": "(2 + 2 + 4 + 8)", "id": "bytes" }, " bytes" ],
  "parts": [
    { "id": "mem", "name": "Memory", "meaning": "GPU memory needed for weights, gradients and optimiser state during full fine-tuning." },
    { "id": "n", "name": "N (parameters)", "meaning": "The number of weights in the model." },
    { "id": "bytes", "name": "(2 + 2 + 4 + 8)", "meaning": "Bytes for the 16-bit weight, 16-bit gradient, 32-bit master copy and the two 32-bit Adam moments." }
  ],
  "steps": [
    { "title": "A 7-billion-parameter model", "expr": "7 × 10⁹ × 16 bytes = 112 × 10⁹ bytes = 112 GB", "note": "Far more than the 14 GB the 16-bit weights alone would need, and more than a single common GPU holds." },
    { "title": "One copy per task", "expr": "A full checkpoint of the 16-bit weights is 7 × 10⁹ × 2 bytes = 14 GB", "note": "Fine-tune for ten tasks and you store ten 14 GB models." }
  ]
}
```

Notice the ratio: ==the training state is eight times the size of the 16-bit weights==. Try different sizes and methods here.

```viz
{ "type": "calc", "kind": "ft_memory", "title": "Fine-tuning memory: full, LoRA and QLoRA" }
```

Choose 7 billion parameters. Full fine-tuning gives 112 GB; LoRA with 1% trainable parameters gives about 15.0 GB; QLoRA with 1% gives about 4.5 GB, before activations. The next calculator shows what the weights alone cost at different precisions.

```viz
{ "type": "calc", "kind": "memory", "title": "Weights at 32, 16, 8 and 4 bits" }
```

Notice how little the weights themselves cost at 4 bits compared with the training state. That observation is the key to the methods below.

### LoRA: freeze W, learn a small change

A weight matrix W in the model has d_out rows and d_in columns. Full fine-tuning learns a change ΔW of the same size: d_out × d_in numbers. The idea of [[LoRA|Low-Rank Adaptation (Hu et al., 2021): keep a pre-trained weight matrix W frozen and learn its change as the product of two thin matrices B and A, so only a few numbers are trained.]] is ==that the change is well described by far fewer numbers==. Write it as a product of two thin matrices:

ΔW = B·A, with A of size r × d_in, B of size d_out × r, and the [[rank|Here, the inner size r of the product B·A. It limits how many independent directions the update can use.]] r much smaller than d_in and d_out.

Think of it as a recipe. Instead of rewriting the whole cookbook (W), you add a one-page card of adjustments (ΔW) kept in two small pieces. Cooks follow the book and then the card.

```viz
{
  "type": "equation",
  "title": "The LoRA forward pass",
  "intro": "The frozen path and the trained path run side by side and their outputs are added. The example uses d = 4 and r = 1 so you can check every number.",
  "formula": [ { "t": "h", "id": "h" }, " = ", { "t": "W x", "id": "wx" }, " + ", { "t": "(α / r)", "id": "scale" }, " · ", { "t": "B", "id": "b" }, " ", { "t": "A", "id": "a" }, " ", { "t": "x", "id": "x" } ],
  "parts": [
    { "id": "h", "name": "h (output)", "meaning": "What the layer passes on to the next layer." },
    { "id": "wx", "name": "W x (frozen path)", "meaning": "The original layer’s output. W is never changed." },
    { "id": "scale", "name": "α / r (scale)", "meaning": "A fixed factor. α is a setting you choose, often about twice r. Dividing by r keeps the size of the update steady when you change the rank." },
    { "id": "b", "name": "B (d_out × r)", "meaning": "The second thin matrix. It starts at all zeros." },
    { "id": "a", "name": "A (r × d_in)", "meaning": "The first thin matrix. It starts with small random numbers." },
    { "id": "x", "name": "x (input)", "meaning": "The vector that enters the layer." }
  ],
  "steps": [
    { "title": "Set up", "expr": "W = I (4×4 identity, frozen), x = (1, 2, 0, −1), r = 1, α = 2 so α/r = 2", "note": "Pretend training has produced A = (0.5, −0.5, 1, 0) and B = (1, 0, −1, 2) as a column." },
    { "title": "Shrink with A", "expr": "A x = 0.5×1 + (−0.5)×2 + 1×0 + 0×(−1) = −0.5", "note": "A squeezes the 4 numbers of x into r = 1 number." },
    { "title": "Expand with B", "expr": "B (A x) = (1, 0, −1, 2) × (−0.5) = (−0.5, 0, 0.5, −1)", "note": "B spreads that one number back out into 4 numbers." },
    { "title": "Scale and add", "expr": "h = W x + 2 × (−0.5, 0, 0.5, −1) = (1, 2, 0, −1) + (−1, 0, 1, −2) = (0, 2, 1, −3)", "note": "The change ΔW = BA has 16 entries but is described by only 4 + 4 = 8 numbers." }
  ]
}
```

Notice that we never formed the big matrix ΔW during the computation: A x is small, and B takes it from there. For a big layer this saves both memory and time.

### Starting without harm

The two matrices are initialised so that the model begins exactly as it was. A gets small random numbers and B gets all zeros. Then BA = 0, so h = Wx, and the adapted model produces the same output as the base model on day one. Training moves B away from zero, step by step. ==The adapter starts as a no-op and learns only what is needed.== If you started both from random numbers the model would be damaged before training even began.

### Counting the trained parameters

A has r × d_in numbers and B has d_out × r, so one adapted matrix trains r·(d_in + d_out) numbers.

```viz
{
  "type": "equation",
  "title": "Trainable parameters of one LoRA matrix",
  "intro": "For a square matrix of side d the count is 2·r·d, against d² for full fine-tuning.",
  "formula": [ { "t": "params", "id": "p" }, " = ", { "t": "r", "id": "r" }, " · (", { "t": "d_in", "id": "din" }, " + ", { "t": "d_out", "id": "dout" }, ")" ],
  "parts": [
    { "id": "p", "name": "params", "meaning": "How many numbers LoRA trains for this one matrix." },
    { "id": "r", "name": "r (rank)", "meaning": "The inner size of the product B·A. Typical values are 4, 8, 16 or 64." },
    { "id": "din", "name": "d_in", "meaning": "The number of columns of W: the size of the input." },
    { "id": "dout", "name": "d_out", "meaning": "The number of rows of W: the size of the output." }
  ],
  "steps": [
    { "title": "A square matrix, d = 4096, r = 8", "expr": "8 × (4096 + 4096) = 8 × 8192 = 65,536", "note": "A is 8 × 4096 = 32,768 numbers and B is 4096 × 8 = 32,768 numbers." },
    { "title": "Compare with the full matrix", "expr": "4096 × 4096 = 16,777,216, and 65,536 / 16,777,216 = 0.39%", "note": "LoRA trains well under half of one percent of this matrix." }
  ]
}
```

Now count the whole of GPT-2 small, the model of the lab. First recall its parts with the Week 8 figure.

```viz
{
  "type": "tfblock",
  "title": "GPT-2 small: where the 124,439,808 parameters are",
  "intro": "Look at the attention row: this is where LoRA will place its adapters. Notice how small a share of the whole model one attention matrix is.",
  "vocab": 50257, "d": 768, "heads": 12, "dff": 3072, "layers": 12, "ctx": 1024, "posLearned": true, "tied": true,
  "note": "Counts follow the GPT-2 conventions of Week 8, including biases.",
  "checks": [ { "total": 124439808, "block": 7087872, "attn": 2362368, "ffn": 4722432 } ]
}
```

Now count the adapter. Each of its 12 blocks has an attention layer called `c_attn`, one matrix that makes the queries, keys and values together: it takes 768 numbers in and gives 2304 out (3 × 768).

```viz
{
  "type": "equation",
  "title": "LoRA on every attention layer of GPT-2 small",
  "intro": "Add up r·(d_in + d_out) over the 12 attention matrices and compare with the whole model.",
  "formula": [ { "t": "total", "id": "total" }, " = ", { "t": "layers", "id": "layers" }, " × ", { "t": "r", "id": "r" }, " · (", { "t": "768", "id": "din" }, " + ", { "t": "2304", "id": "dout" }, ")" ],
  "parts": [
    { "id": "total", "name": "total", "meaning": "All the numbers LoRA trains in the whole model." },
    { "id": "layers", "name": "layers (12)", "meaning": "GPT-2 small has 12 Transformer blocks, each with one c_attn matrix." },
    { "id": "r", "name": "r (rank)", "meaning": "Here the rank is set to r = 8." },
    { "id": "din", "name": "768 (d_in)", "meaning": "The model width, GPT-2 small’s d_model." },
    { "id": "dout", "name": "2304 (d_out)", "meaning": "Three times 768: the query, key and value outputs together." }
  ],
  "steps": [
    { "title": "One matrix", "expr": "8 × (768 + 2304) = 8 × 3072 = 24,576", "note": "The full c_attn matrix would be 768 × 2304 = 1,769,472 numbers, so LoRA trains 1.4% of it." },
    { "title": "Twelve of them", "expr": "12 × 24,576 = 294,912", "note": "One matrix in each of the 12 blocks." },
    { "title": "The share of the whole model", "expr": "294,912 / 124,439,808 = 0.237%", "note": "The model has 124,439,808 parameters (Week 8). In the lab you will confirm the count 294,912 in code." }
  ]
}
```

The next calculator lets you vary the matrix size and the rank yourself.

```viz
{ "type": "calc", "kind": "lora", "title": "Full matrix vs LoRA adapter" }
```

Set the size to 4096 and r to 8 and check that you see 65,536. Notice that doubling r doubles the adapter, but the share of the full matrix stays tiny until r becomes large.

### Why can a low-rank change be enough?

It is a bet, and a well-supported one. When a large model is adapted to one task, the change in its weights tends to be concentrated in a few directions, so it can be well approximated by a low-rank matrix. Hu et al. (2021) found that LoRA matched or came close to full fine-tuning on many tasks with a tiny rank. The figure below makes the idea concrete with a pretend change matrix. Move the rank slider and watch the error and the bars of singular values. The [[singular values|The numbers that say how strongly a matrix acts along each of its main directions, listed from largest to smallest. A matrix with only a few large singular values is close to low-rank.]] measure how much of the matrix lives in each direction: large ones matter, tiny ones can be dropped.

```viz
{
  "type": "lowrank",
  "title": "A pretend ΔW that is nearly low-rank",
  "intro": "Raise the rank from 1 to 2 and watch the error collapse. Then raise it further and see that there is almost nothing left to gain.",
  "matrix": [
    [0.9, 0.1, 1, 1.9, -1, 1, 0, 1.1],
    [2.6, -0.5, 2, 4.5, -1.9, 2.9, 0.5, 1.5],
    [0.5, -0.6, 0, 0.5, 0.1, 1.1, 0.4, -0.5],
    [0.6, 0.5, 1, 1.5, -0.9, 0.1, -0.5, 1.5],
    [0.1, -1, -1.1, -1, 1, 1, 1.1, -2],
    [0.5, -0.4, -0.1, 0.5, 0, 1, 0.5, -0.6],
    [2.1, -0.1, 2, 4, -2, 2.1, 0, 1.9],
    [0.5, 0.5, 1, 1.5, -1, 0, -0.5, 1.4]
  ],
  "rank": 1,
  "max": 4,
  "note": "A made-up 8 × 8 matrix built from two strong directions plus a little noise. Real weight changes are much larger; the idea is the same.",
  "checks": [
    { "sv": [10.174, 4.069] },
    { "rank": 1, "err": 0.3727 },
    { "rank": 2, "err": 0.0335 }
  ]
}
```

Two numbers do almost all the work: the first two bars are tall and the rest are tiny. With rank 2 the relative error is 3.3% while storing 2 × (8 + 8) = 32 numbers instead of 64. In a 4096 × 4096 matrix the saving would be a factor of hundreds.

But the bet can fail. Here is a matrix with no low-rank structure.

```viz
{
  "type": "lowrank",
  "title": "A matrix that is not low-rank",
  "intro": "Here the singular values fall off slowly. Raise the rank and compare the error with the number of values you must store.",
  "matrix": [
    [3, 2, 1, 0, 1, 0],
    [2, 3, 0, 1, 0, 1],
    [1, 0, 3, 2, 1, 0],
    [0, 1, 2, 3, 0, 1],
    [1, 1, 1, 1, 2, 2],
    [2, 0, 1, 0, 3, 1],
    [0, 2, 0, 1, 1, 3],
    [1, 1, 2, 0, 0, 2]
  ],
  "rank": 2,
  "max": 4,
  "note": "A made-up 8 × 6 matrix with spread-out singular values, to show when a low rank is not enough.",
  "checks": [
    { "sv": [8.139, 4.09, 3.85] },
    { "rank": 2, "err": 0.5091 },
    { "rank": 4, "err": 0.1837 }
  ]
}
```

At rank 2 the error is still about 51%, and at rank 4 you must store 4 × (8 + 6) = 56 numbers, more than the 48 of the matrix itself. ==Low rank helps only when the change really is concentrated.== The LoRA bet is that for fine-tuning a pre-trained model it usually is, and you should treat it as a bet and measure the result.

### Merging the adapter for deployment

After training, you do not have to keep the adapter separate. Add the scaled product into the frozen matrix once: W′ = W + (α/r)·B·A. The merged model has exactly the shape of the original and runs at the original speed, with no extra step at inference time.

```viz
{
  "type": "equation",
  "title": "Merging the adapter into W",
  "intro": "After merging, one matrix does the work of two paths. Check it against the earlier example.",
  "formula": [ { "t": "W′", "id": "wp" }, " = ", { "t": "W", "id": "w" }, " + ", { "t": "(α / r)", "id": "scale" }, " · ", { "t": "B A", "id": "ba" } ],
  "parts": [
    { "id": "wp", "name": "W′ (merged weight)", "meaning": "The single matrix that the deployed model uses." },
    { "id": "w", "name": "W (frozen weight)", "meaning": "The original pre-trained matrix." },
    { "id": "scale", "name": "α / r", "meaning": "The same fixed scale as in the forward pass." },
    { "id": "ba", "name": "B A (learned change)", "meaning": "The product of the two thin matrices: a full-size matrix of rank at most r." }
  ],
  "steps": [
    { "title": "The learned change", "expr": "B A has rows (0.5, −0.5, 1, 0), (0, 0, 0, 0), (−0.5, 0.5, −1, 0), (1, −1, 2, 0)", "note": "From the earlier example with B = (1, 0, −1, 2) and A = (0.5, −0.5, 1, 0)." },
    { "title": "Merge with W = I and scale 2", "expr": "W′ rows: (2, −1, 2, 0), (0, 1, 0, 0), (−1, 1, −1, 0), (2, −2, 4, 1)", "note": "Each row is I plus 2 × the row of BA." },
    { "title": "Check on x = (1, 2, 0, −1)", "expr": "W′x = (2 − 2, 2, −1 + 2, 2 − 4 − 1) = (0, 2, 1, −3)", "note": "The same output h = (0, 2, 1, −3) as the two-path computation. ==Merging changes nothing about the answer.==" }
  ]
}
```

Notice that the merged matrix answers exactly as the two-path version did. You can also *not* merge, and keep the adapter as a separate small file that you plug in when you need it.

### One base model, many adapters

Because the adapter is small, a school could keep one frozen base model and many tiny adapters: one for helping with Filipino homework, one for drafting announcements, one for a library helper. To change task you load a different adapter. ==The storage saving is large==.

```viz
{
  "type": "equation",
  "title": "Checkpoint size: adapter vs full model",
  "intro": "Parameters times bytes per parameter. The example saves GPT-2 small in 16-bit numbers (2 bytes each).",
  "formula": [ { "t": "size", "id": "size" }, " = ", { "t": "parameters", "id": "p" }, " × ", { "t": "bytes each", "id": "b" } ],
  "parts": [
    { "id": "size", "name": "size", "meaning": "How big the saved file is, in bytes (divide by one million for MB)." },
    { "id": "p", "name": "parameters", "meaning": "How many numbers are saved: only the adapter, or the whole model." },
    { "id": "b", "name": "bytes each", "meaning": "2 for 16-bit numbers, 4 for 32-bit numbers." }
  ],
  "steps": [
    { "title": "The LoRA adapter", "expr": "294,912 × 2 bytes = 589,824 bytes ≈ 0.59 MB", "note": "Small enough to email." },
    { "title": "The full model", "expr": "124,439,808 × 2 bytes = 248,879,616 bytes ≈ 249 MB", "note": "About 420 times bigger." },
    { "title": "One hundred tasks", "expr": "100 adapters ≈ 59 MB, but 100 full copies ≈ 24.9 GB", "note": "The base model is stored once and shared." }
  ]
}
```

### How the training run works

```viz
{
  "type": "pipeline",
  "title": "Training and using a LoRA adapter",
  "intro": "Step through the life of an adapter, from freezing the base to merging it.",
  "vertical": false,
  "nodes": [
    { "id": "freeze", "label": "Freeze W", "sub": "no gradients for W" },
    { "id": "add", "label": "Add A and B", "sub": "B = 0 at the start" },
    { "id": "train", "label": "Train", "sub": "only A and B change" },
    { "id": "save", "label": "Save adapter", "sub": "a few MB" },
    { "id": "merge", "label": "Merge or plug in", "sub": "W + (α/r) B A" }
  ],
  "steps": [
    { "title": "Freeze the base model", "body": "Switch off gradients for every original weight. They are never updated, so they need no gradient and no Adam state.", "active": ["freeze"], "example": "124,439,808 weights stay fixed." },
    { "title": "Attach the adapters", "body": "Next to each chosen matrix, add A (small random) and B (zeros). The model gives the same output as before, because B A = 0.", "active": ["freeze", "add"], "example": "12 pairs of A and B for GPT-2’s c_attn." },
    { "title": "Train only A and B", "body": "Run the usual loop with the SFT loss or a preference loss. Gradients and Adam state exist only for the adapter, which is why the memory bill shrinks.", "active": ["add", "train"], "example": "294,912 trained numbers, 0.237% of the model." },
    { "title": "Save only the adapter", "body": "The checkpoint holds just A and B, plus a note saying which base model it belongs to.", "active": ["train", "save"], "example": "About 0.59 MB in 16-bit." },
    { "title": "Merge or plug in", "body": "Either add (α/r)·B·A into W once, for the same speed as the original, or keep the adapter separate and swap it for another.", "active": ["save", "merge"], "example": "No extra latency after merging." }
  ]
}
```

### Which layers, which rank?

The original paper adapted the attention projections, and that is still the common default. Many practitioners also adapt the feed-forward layers when they want more capacity. The usual advice for the settings is plain.

- **Rank r.** Start small (4 or 8). Raise it if the validation loss stalls well above what full fine-tuning reaches, or if the task is very different from the pre-training data.
- **Scale α.** Often set to about twice r, so the scale α/r is about 2. It acts as a volume knob on the update.
- **Which matrices.** Attention projections first; add the feed-forward matrices for harder tasks.
- **Dropout.** A little LoRA dropout (such as 0.05) can help when the data are small.

There is no rule that works for every task, so ==try two or three values and compare the validation loss==. You will do a rank sweep in Lab Part 6.

### Other ways to be parameter-efficient

LoRA is one of several PEFT methods. Each of these is known for one idea.

- **Adapters** (Houlsby et al., 2019) insert small trainable layers inside each Transformer block and freeze everything else.
- **Prefix tuning** (Li and Liang, 2021) learns a short sequence of extra vectors that is prepended inside the attention layers, like a learned instruction that no human wrote.
- **Prompt tuning** (Lester et al., 2021) is a simpler version that learns a few vectors added only at the input.

```viz
{
  "type": "timeline",
  "title": "PEFT methods",
  "events": [
    { "year": "2019", "title": "Adapters", "text": "Houlsby et al. inserted small trainable bottleneck layers in each Transformer block and froze the rest of the model." },
    { "year": "2021", "title": "Prefix tuning", "text": "Li and Liang learned a short prefix of vectors for the attention layers, keeping the language model frozen." },
    { "year": "2021", "title": "Prompt tuning", "text": "Lester et al. learned a few soft prompt vectors at the input and showed that the method works better as models get larger." },
    { "year": "2021", "title": "LoRA", "text": "Hu et al. added low-rank matrices B and A next to the frozen weights, which can be merged back so there is no extra cost at inference." },
    { "year": "2023", "title": "QLoRA", "text": "Dettmers et al. trained LoRA adapters on top of a base model stored in 4-bit numbers, so very large models fit on a single GPU." }
  ]
}
```

Notice that all five freeze the base model and train something small. They differ in where the small thing sits and whether it can be merged back.

### QLoRA: shrink the frozen part too

In LoRA the trained part is tiny, but the frozen base is still stored in 16 bits. [[QLoRA|A method (Dettmers et al., 2023) that stores the frozen base model in 4-bit numbers and trains LoRA adapters on top, so very large models can be fine-tuned on one GPU.]] stores the frozen weights in 4 bits and keeps the adapters in higher precision. The paper showed fine-tuning a 65-billion-parameter model on a single 48 GB GPU, with quality close to 16-bit fine-tuning.

```viz
{
  "type": "equation",
  "title": "Why 65 B fits on a 48 GB GPU with QLoRA",
  "intro": "Four bits is half a byte. Count the memory of the weights alone for each choice.",
  "formula": [ { "t": "weights", "id": "w" }, " = ", { "t": "N", "id": "n" }, " × ", { "t": "bytes per weight", "id": "b" } ],
  "parts": [
    { "id": "w", "name": "weights", "meaning": "Memory for the frozen model weights alone, not counting adapters or activations." },
    { "id": "n", "name": "N (parameters)", "meaning": "65 × 10⁹ for a 65-billion-parameter model." },
    { "id": "b", "name": "bytes per weight", "meaning": "2 for 16-bit, 0.5 for 4-bit (a byte holds 8 bits, so 4 bits is half a byte)." }
  ],
  "steps": [
    { "title": "In 16-bit", "expr": "65 × 10⁹ × 2 = 130 GB", "note": "Already larger than a 48 GB GPU, before any training state." },
    { "title": "Full fine-tuning", "expr": "65 × 10⁹ × 16 = 1,040 GB", "note": "Needs many GPUs." },
    { "title": "In 4-bit (QLoRA)", "expr": "65 × 10⁹ × 0.5 = 32.5 GB", "note": "The frozen base fits, leaving room within 48 GB for the small adapter, its Adam state and the activations." }
  ]
}
```

Notice that the saving comes in two steps: ==LoRA removes the gradients and Adam state for almost every weight==, and 4-bit storage halves and halves again what is left. The price is a little extra work to convert numbers back and forth, and some rounding error from the 4-bit storage.

### Full fine-tuning or LoRA?

```viz
{
  "type": "flow",
  "title": "The same task, two ways",
  "modes": [
    {
      "name": "Full fine-tuning",
      "tagline": "Update every weight.",
      "inputLabel": "What you train",
      "modelLabel": "Cost",
      "outputLabel": "What you keep",
      "examples": [
        { "label": "7 B model, school helper", "input": "All 7 billion weights get gradients and Adam state", "output": "About 112 GB to train (illustrative setup) and a 14 GB checkpoint for each task." }
      ],
      "note": "Illustrative, using the 16 bytes per parameter rule. Real set-ups differ."
    },
    {
      "name": "LoRA",
      "tagline": "Freeze W, train a small adapter.",
      "inputLabel": "What you train",
      "modelLabel": "Cost",
      "outputLabel": "What you keep",
      "examples": [
        { "label": "7 B model, school helper", "input": "About 1% of the numbers (an illustrative share) get gradients and Adam state", "output": "About 15 GB to train and an adapter of tens of MB, with the base model shared." }
      ],
      "note": "Illustrative. Quality is often close to full fine-tuning, but not always the same."
    }
  ]
}
```

Notice what stays the same: the data, the loss and the training loop. Only which numbers are allowed to change differs.

### Check your PEFT sense

```viz
{
  "type": "sorter",
  "title": "Full fine-tuning, LoRA, or both?",
  "intro": "Drag each statement to the method it describes.",
  "bins": [
    { "id": "full", "label": "Full fine-tuning" },
    { "id": "lora", "label": "LoRA" },
    { "id": "both", "label": "Both" }
  ],
  "cards": [
    { "name": "Needs gradients and Adam state for every parameter of the model", "answer": "full", "why": "All the weights are trainable, which gives the 16 bytes per parameter bill." },
    { "name": "Learns ΔW = B·A with a small rank r", "answer": "lora", "why": "The low-rank product is the defining idea of LoRA." },
    { "name": "Starts from a pre-trained model, not from random weights", "answer": "both", "why": "Both are ways of fine-tuning the same pre-trained starting point." },
    { "name": "Can be saved as a file of a few MB for a small model", "answer": "lora", "why": "Only the adapter is saved: 0.59 MB for the GPT-2 example." },
    { "name": "Can use the SFT loss with a masked prompt", "answer": "both", "why": "The loss is a separate choice from which parameters are trained." },
    { "name": "Needs a full copy of the model for every task you add", "answer": "full", "why": "Every weight can change, so each task needs its own full checkpoint." },
    { "name": "Can be merged into W so that inference is no slower", "answer": "lora", "why": "W′ = W + (α/r) B A has the original shape." },
    { "name": "Can overfit if the training set is tiny", "answer": "both", "why": "A small adapter overfits less easily than full fine-tuning, but with a handful of examples it still can." }
  ]
}
```

## Doing it well {rail=Practice}

{lede} Knowing the methods is half the job. The other half is ==running a fine-tuning project so that the result is useful, honest and safe==, and knowing when not to fine-tune at all. Good projects are mostly decided before the first training step, by the choice of task, data and test.

### Start with the task and the test

Before any code, write down two sentences. First, what the model must do, in the words of the person who will use it: “answer Grade 8 students’ questions about school rules in the school’s style, in English or Filipino”. Second, how you will know it works: “at least 90% of 100 test questions judged correct and polite by two teachers, and no loss of ordinary writing ability”. A project with no [[evaluation set|A collection of examples kept apart from training and used only to measure how well the model does. It must reflect the real task.]] ==cannot tell progress from wishful thinking==.

```viz
{
  "type": "pipeline",
  "title": "A fine-tuning project, start to finish",
  "intro": "Step through the plan. Notice how much happens before and after the training run.",
  "vertical": false,
  "nodes": [
    { "id": "task", "label": "Task", "sub": "and a success metric" },
    { "id": "data", "label": "Data", "sub": "collect, clean, format" },
    { "id": "split", "label": "Split", "sub": "train, validation, test" },
    { "id": "train", "label": "Train", "sub": "small steps, watch loss" },
    { "id": "eval", "label": "Evaluate", "sub": "task, forgetting, safety" },
    { "id": "doc", "label": "Document", "sub": "model card" }
  ],
  "steps": [
    { "title": "Define the task and the metric", "body": "Say in plain words what the model must do and how you will measure success. Decide in advance what counts as good enough.", "active": ["task"], "example": "90% judged correct and polite on 100 test questions." },
    { "title": "Collect and prepare the data", "body": "Gather or write examples, check their quality, remove personal information, check the licences, and format them with the chat template.", "active": ["task", "data"], "example": "800 teacher-written question and answer pairs." },
    { "title": "Split the data", "body": "Keep a validation set for tuning settings and a test set you look at only at the end (Week 2). Split before training and never mix them.", "active": ["data", "split"], "example": "640 train, 80 validation, 80 test." },
    { "title": "Train with care", "body": "Start from the right base model, use a small learning rate, and watch the training and validation loss. Stop early if validation loss rises.", "active": ["split", "train"], "example": "LoRA with r = 8, a few epochs." },
    { "title": "Evaluate on three things", "body": "Measure the task on the test set, measure general ability to detect forgetting, and test safety with hard and tricky inputs.", "active": ["train", "eval"], "example": "Task score, loss on general text, red-team prompts." },
    { "title": "Document the model", "body": "Write a model card: what it is for, the data, the tests, the known limits. A model without a description is hard to use responsibly (Week 1).", "active": ["eval", "doc"], "example": "A one-page card kept with the model." }
  ]
}
```

### Data: the part that matters most

==Most failures of fine-tuned models trace back to the data.== Look after four things.

- **Quality.** Every answer should be one you would be glad to see the model repeat. Have a second person check a sample.
- **Diversity.** Cover the real range of questions, short and long, easy and hard, including the odd ones.
- **Consent, licences and privacy.** Use only text that you are allowed to use. Remove names, phone numbers, addresses and anything else that identifies a person. Student data need special care: ask your instructor or school before using them.
- **Language coverage.** If your users write Filipino, English and a mix of both (code-switching), your examples must include all three. A model tuned on English only will be weaker in the others, and it can be worse than the base model in them.

```viz
{
  "type": "pipeline",
  "title": "Preparing data for fine-tuning",
  "intro": "Each stage removes a different kind of problem.",
  "vertical": false,
  "nodes": [
    { "id": "gather", "label": "Gather", "sub": "write or collect" },
    { "id": "rights", "label": "Check rights", "sub": "consent, licence" },
    { "id": "privacy", "label": "Remove personal data", "sub": "names, numbers" },
    { "id": "clean", "label": "Clean", "sub": "duplicates, errors" },
    { "id": "format", "label": "Format", "sub": "chat template" }
  ],
  "steps": [
    { "title": "Gather examples", "body": "Write them with teachers or collect them from sources you are allowed to use. Aim for variety across topics, lengths and languages.", "active": ["gather"], "example": "Question and answer pairs in English, Filipino and mixed." },
    { "title": "Check rights", "body": "Confirm that you have permission and that the licence allows this use. If you are unsure, leave the text out.", "active": ["gather", "rights"], "example": "A school announcement written by staff: yes. A copied textbook page: check first." },
    { "title": "Remove personal data", "body": "Delete or replace names, contact details and anything identifying. A model can repeat what it was trained on.", "active": ["rights", "privacy"], "example": "“Maria Santos, 0917…” becomes “[student]”." },
    { "title": "Clean", "body": "Remove duplicates, fix wrong answers and make the tone consistent. A hundred clean examples beat a thousand messy ones.", "active": ["privacy", "clean"], "example": "Remove the same question that appears 12 times." },
    { "title": "Format", "body": "Put every example in the same chat template used at chat time, then check a few by eye.", "active": ["clean", "format"], "example": "<|user|> … <|assistant|> … <|end|>" }
  ]
}
```

### Choose the lightest method that works

Methods differ in cost. A sensible order is to try the cheapest first and move down only when the test says you must. That is why we say you ==usually do not start with fine-tuning==.

```viz
{
  "type": "pipeline",
  "title": "The ladder of methods",
  "intro": "Climb only as high as the task needs. Each step costs more time and more care.",
  "vertical": true,
  "nodes": [
    { "id": "prompt", "label": "Better prompt", "sub": "Week 9" },
    { "id": "few", "label": "Examples in the prompt", "sub": "few-shot" },
    { "id": "rag", "label": "Retrieval (RAG)", "sub": "Week 16" },
    { "id": "lora", "label": "LoRA fine-tuning", "sub": "cheap adapter" },
    { "id": "full", "label": "Full fine-tuning", "sub": "most costly" }
  ],
  "steps": [
    { "title": "Try a better prompt", "body": "Write a clear instruction, a role and the format you want, as in Week 9. It costs nothing to train and is easy to change. Often it is enough.", "active": ["prompt"], "example": "“You are a helper for Mabini High School. Answer briefly and politely.”" },
    { "title": "Add examples to the prompt", "body": "Show two or three good question and answer pairs in the prompt. The model copies the style without any training. The limit is the length of the prompt and the cost of sending the examples every time.", "active": ["prompt", "few"], "example": "Three example answers before the real question." },
    { "title": "Retrieve the facts", "body": "If the problem is facts that change or live in your documents, fetch the relevant passage and place it in the prompt (Week 16). Fine-tuning is poor at keeping facts current.", "active": ["few", "rag"], "example": "Look up this month’s canteen menu." },
    { "title": "Fine-tune with LoRA", "body": "If the problem is behaviour, such as a steady style or format, or skill on a narrow task that prompting cannot reach, train a small adapter.", "active": ["rag", "lora"], "example": "Always answer in the school’s house style." },
    { "title": "Full fine-tuning", "body": "Only when LoRA falls short and you have the data, the compute and the checking to justify changing every weight.", "active": ["lora", "full"], "example": "A large domain shift with a lot of data." }
  ]
}
```

### Settings in words

You do not need to tune dozens of knobs. These are the ones to understand.

- **Learning rate.** Small. Typical values are around 1e-4 to 2e-4 for LoRA and 1e-5 to 5e-5 for full fine-tuning. LoRA can use larger steps because it changes only a small part of the model. These are common starting points, not rules.
- **Number of passes ([[epochs|One epoch is one full pass through the whole training set. Several epochs mean the model sees each example several times.]]).** A few. Many passes over a small set invite overfitting.
- **Batch size.** As large as memory allows, or use gradient accumulation to imitate a larger one.
- **Rank r and scale α.** Start with r = 4 or 8 and α about 2r.

Watch the training loss and the validation loss together. If the training loss falls but the validation loss stops falling or rises, stop: that is [[early stopping|Ending training when the validation loss stops improving, to avoid overfitting. You usually keep the model from the best validation step.]].

```viz
{
  "type": "fit",
  "title": "Too flexible a fit: the Week 2 lesson again",
  "intro": "Slide the polynomial degree up. Watch the training error keep falling while the test error eventually turns upward. A fine-tuned model trained too long on too little data does the same.",
  "degree": 1,
  "maxDegree": 9,
  "train": [
    { "x": -1, "y": 0.9 }, { "x": -0.7, "y": 0.35 }, { "x": -0.4, "y": 0.25 }, { "x": -0.1, "y": 0.0 },
    { "x": 0.2, "y": 0.1 }, { "x": 0.5, "y": 0.3 }, { "x": 0.8, "y": 0.7 }, { "x": 1, "y": 1.05 }
  ],
  "test": [
    { "x": -0.85, "y": 0.8 }, { "x": -0.55, "y": 0.25 }, { "x": -0.25, "y": 0.08 },
    { "x": 0.05, "y": 0.02 }, { "x": 0.35, "y": 0.12 }, { "x": 0.65, "y": 0.45 }
  ],
  "note": "Made-up points that follow a smooth curve plus noise, to recall overfitting from Week 2. Not the output of a language model."
}
```

### Evaluate three ways

1. **The task.** Measure the skill you trained on the held-out test set: accuracy for a classifier, a win rate for a chat model, ratings from people for tone.
2. **General ability.** Check loss or simple tasks on general text before and after, to catch forgetting.
3. **Safety.** Test with inputs meant to cause trouble. [[Red-teaming|Deliberately trying to make a model behave badly, by writing tricky, hostile or misleading inputs, so the weak points are found and fixed before real users meet them.]] is the name for this. Keep a list of the prompts that broke it and re-run it after every change.

Report numbers honestly, including the number of test examples, as in the win rate example, and say what the model cannot do.

### Cost, energy and care

Every training run uses electricity, and large runs use a lot. Fine-tuning is far cheaper than pre-training, and LoRA is cheaper again, which is a good reason to prefer it, to reuse a base model instead of training a new one, and to avoid runs that you can foresee will not help. Keep a log of what you ran and what each run was for, so you do not repeat them, and share adapters instead of retraining the same thing in many places. We have not quoted energy figures here because they depend heavily on the hardware and the place; ask for them where you can.

### When not to fine-tune

Fine-tuning is the wrong tool in several common cases.

- **The facts change often.** A model’s weights are a poor database. If the canteen menu or the exam schedule changes every week, retrieve the current text instead (Week 16).
- **The data are tiny.** With a handful of examples, put them in the prompt. Fine-tuning would simply memorise them.
- **The data are private or sensitive.** A model can repeat what it was trained on. If you cannot safely share the text, do not put it in the weights.
- **A better prompt has not been tried.** It is the cheapest experiment there is.

### Three school cases

| Case | Data | Method | Main risk |
|---|---|---|---|
| A Grade 8 homework helper that answers in the school’s style, in English and Filipino | About 800 teacher-written question and answer pairs | Try a prompt with examples first; if the style is still unsteady, LoRA with SFT | Wrong answers copied from the data; poor coverage of Filipino and mixed-language questions |
| Sorting messages to the guidance office as urgent or routine | About 3,000 labelled messages, with names removed | Task fine-tuning of a small classifier, or LoRA on a small language model | Privacy of student messages; missing a truly urgent message is costly, so measure that error rate separately |
| Answering parents’ questions about a barangay health centre’s clinic schedule, which changes each month | The current monthly schedule documents | Retrieval (RAG) with a good prompt; no fine-tuning | Stale or wrong schedule if the documents are not updated |

Read the table by columns: the data and the way the facts change ==decide the method more than the task name does==.

### Decide for each scenario

```viz
{
  "type": "sorter",
  "title": "Prompting, fine-tuning or retrieval?",
  "intro": "Drag each scenario to the approach that fits best. Think about whether the problem is behaviour or facts, and how much data you have.",
  "bins": [
    { "id": "prompt", "label": "Prompting is enough" },
    { "id": "ft", "label": "Fine-tune" },
    { "id": "rag", "label": "Retrieval (RAG) fits better" }
  ],
  "cards": [
    { "name": "Rewrite one paragraph in simpler English, just once", "answer": "prompt", "why": "A one-off task with a clear instruction needs no training." },
    { "name": "Answer questions about this month’s canteen menu, which changes weekly", "answer": "rag", "why": "The facts change often, so fetch the current menu instead of baking it into the weights." },
    { "name": "Always reply in the school’s formal tone and fixed format across thousands of requests, where prompts with examples are too long or unsteady", "answer": "ft", "why": "A steady behaviour that prompts cannot hold is what fine-tuning is for." },
    { "name": "Label 5,000 emails a day into six categories, with 4,000 labelled examples available", "answer": "ft", "why": "Plenty of labelled data for one narrow task is a classic case for task fine-tuning." },
    { "name": "Answer questions from a 200-page handbook and say which page the answer is on", "answer": "rag", "why": "The answer lives in documents, so retrieval can quote the passage and stay current." },
    { "name": "Summarise one meeting’s notes with a short instruction", "answer": "prompt", "why": "A single summarisation task needs no tuning." },
    { "name": "Try out a chatbot idea with 20 test questions before investing", "answer": "prompt", "why": "Prompting is the cheapest experiment and tells you whether anything heavier is needed." },
    { "name": "Make a model handle a regional language variety it does poorly, given 50 million tokens of text in it", "answer": "ft", "why": "Lots of text and missing knowledge point to continued pre-training, a form of fine-tuning, perhaps with LoRA." }
  ]
}
```

### Good practice and common mistakes

```viz
{
  "type": "sorter",
  "title": "Good practice or mistake?",
  "intro": "Drag each action to the right bin.",
  "bins": [
    { "id": "good", "label": "Good practice" },
    { "id": "mistake", "label": "Mistake" }
  ],
  "cards": [
    { "name": "Set aside validation and test data before training starts", "answer": "good", "why": "Held-out data are the only honest measure of how the model will do on new inputs." },
    { "name": "Choose the learning rate by looking at the test-set score", "answer": "mistake", "why": "Tuning on the test set uses up its honesty. Tune on the validation set and touch the test set once." },
    { "name": "Remove names and phone numbers from the training text", "answer": "good", "why": "Models can repeat what they saw; removing personal data protects people." },
    { "name": "Check the loss on general text as well as on the new task", "answer": "good", "why": "It reveals catastrophic forgetting that the task score alone would hide." },
    { "name": "Keep training until the training loss is as low as possible", "answer": "mistake", "why": "A very low training loss on a small set usually means memorising. Watch the validation loss and stop early." },
    { "name": "Use the same chat template for training and for chatting", "answer": "good", "why": "A mismatch gives the model a format it was not tuned on." },
    { "name": "Copy examples from a website without checking the licence", "answer": "mistake", "why": "You may not have the right to use them, and it is unfair to their authors." },
    { "name": "Test with prompts meant to provoke unsafe replies and keep the list", "answer": "good", "why": "Red-teaming finds weak points before users do; the list becomes a regression test." }
  ]
}
```

### Link to the capstone

Your capstone is the Structural Inspection Assistant, and these ideas could help in two places. A small classifier fine-tuned on labelled photos of cracks, or a language model with a LoRA adapter that writes inspection notes in a fixed format, are the kind of narrow tasks where fine-tuning shines, as long as you have labelled examples you may use. They would not help if the assistant must quote the latest building code or a current project record: those facts belong in retrieval. Think about which parts of your assistant are behaviour and which are facts. The proposal session is in Week 15; see the [Capstone](/capstone) page for the project description.

## Key takeaways

- ==Training happens in stages==: pre-training (huge general text, next-token loss) gives knowledge; continued pre-training adds a domain; task fine-tuning and instruction tuning (SFT, with the loss on response tokens only) give behaviour; preference tuning teaches which answers people prefer.
- ==A base model continues text; a chat model needs SFT and preference tuning==, aiming to be helpful, honest and harmless, and still makes mistakes.
- ==RLHF has three stages==: SFT, a reward model trained with the Bradley–Terry loss −ln σ(r_w − r_l), and reinforcement learning with a KL penalty that keeps the policy near the reference; a small β invites reward hacking, and DPO gets a similar effect without a reward model.
- ==LoRA freezes W and learns ΔW = B·A==, training r·(d_in + d_out) numbers per matrix: 294,912 (0.237%) for GPT-2 small at r = 8, with 16 bytes per parameter for full fine-tuning against a small fraction for adapters, and QLoRA adds a 4-bit frozen base.
- ==Good fine-tuning is mostly good data and honest tests==: try prompting or retrieval first, hold out validation and test sets, check forgetting and safety, and document what you did.

## Lab: Imagine the apprentice and the coach {rail=Laboratory}

{lede} This lab is called *Imagine the apprentice and the coach*. A new apprentice baker has read every cookbook in the library, so she can finish almost any sentence about food; ask her a question and she is as likely to answer with another question as with an answer. That is the base model. Then she spends two weeks in one bakery, where the owner shows her how to answer customers, always the same way: "Sure!", the answer, "Hope that helps." That is fine-tuning on examples. A coach then listens to two answers and says which is better, and the apprentice learns to prefer what the coach likes: that is learning from preferences. In this lab you play every role with real numbers on a real model, GPT-2 small with 124,439,808 parameters. *What does a base model do with a question, and what changes when you fine-tune it?* You count the cost of each stage with the 6ND rule, watch the base model ramble, write the masked supervised loss by hand and prove that it equals the library's loss, then train all 124 million weights on 24 examples and see a style appear after a single epoch, and a little general English disappear. *How can a model learn from "A is better than B" instead of examples?* You build a reward model from the frozen features of GPT-2, train it with the Bradley-Terry loss, pick the best of n answers and watch the reward model get fooled; you compute the best policy under a KL penalty with NumPy, sweep beta and run DPO on a toy policy until it reaches the closed-form optimum. *Can you teach the model a new habit without touching its 124 million weights?* You write LoRA from scratch, check it against the `peft` library, compare it with full fine-tuning, sweep the rank, merge the adapter and look at the singular values of what it learned. You finish by planning a fine-tuning project of your own and writing down, honestly, what worked and what did not. ==The goal is to replace the mystery of "how a language model becomes a chat assistant" by a recipe that you computed, trained and measured yourself, and to learn that every gain of fine-tuning has a price that you must measure.== Every part begins with an **Imagine first** moment: you write a prediction before you run any code and compare it with what really happens. Each part points back to a section of the lesson. Everything runs on a CPU (a GPU only makes it faster), and the longest single run takes about two and a half minutes. This page contains the setup and the first three parts (the stages and the masked loss, supervised fine-tuning, and forgetting); the other parts, the deliverables and the marking scheme follow in the same lab.

| Part | What you do | Minutes |
|---|---|---|
| Setup | Start the notebook, load GPT-2, run a smoke test | 5 |
| 1 | Imagine the apprentice: the stages, 6ND, the base model, the masked labels | 15 |
| 2 | SFT by hand and by machine | 25 |
| 3 | Forgetting and its remedies | 15 |
| 4 | Preference learning from scratch | 30 |
| 5 | KL, beta and DPO on a toy policy | 35 |
| 6 | LoRA from scratch and with peft | 40 |
| 7 | Plan your own project and reflect | 15 |

The total is 180 minutes, about 3 hours, of which about one hour is the computer working. While a training cell runs (the longest ones are marked in the text), write your predictions and answers for the next step. The troubleshooting table and the assessed deliverable are at the end of the lab.

### Setup

1. Log in to the portal, start JupyterLab on the DGX Spark, create a folder `week14` and a notebook `lab14.ipynb`. Run each block of this lab in its own cell, in order: later blocks use names (`tok`, `base_model`, `generate`, `p1_encode`, `p2_train`, `RESULTS` and so on) that earlier blocks create. If you restart the kernel, run the setup cells again and then the blocks of each part from its top.
2. **Nothing is trained from scratch in this lab**: the model is GPT-2 small (124 million parameters, about 500 MB), downloaded once from the Hugging Face hub (on the Spark it is expected to be in the cache already). The first helper cell needs PyTorch, NumPy and Matplotlib; the second needs `transformers`; Part 6 also needs `peft`. On the Spark these should already be installed: if an `import` fails, tell your instructor and do not install packages yourself. In the Google Colab notebook the first cell installs `peft` for you.
3. A CPU is enough. Every training run of this lab takes between 20 seconds and about two and a half minutes on a CPU (in our tests: 22 to 43 seconds for one epoch of full fine-tuning and 66 to 125 seconds for three, depending on how busy the computer was). If the Spark gives you a GPU, the helper cell uses it and everything is faster; the numbers printed in this lab are from a CPU, and yours can differ in the last digits.
4. Write your answers in markdown cells under the block where the exercise appears (Insert a cell, change its type to Markdown). Where an exercise says "store it in a variable", write the variable in a code cell: Part 7 collects your numbers in the dictionary `RESULTS`.
5. The same lab is available as a **Google Colab notebook**. Use the "Open in Colab" card at the top of this lab if you work from home or on a laptop: it has the same parts, the same numbering and the same expected results, plus automatic Check cells and a worked answer for every exercise. The code on this page is the complete working version, so on the Spark you can type or paste it directly. In the Colab notebook you write the key lines yourself first.
6. Paste the next two blocks into two cells and run them. The first fixes the random seed (so your numbers match the ones printed in this lab), defines the `check(...)` function, the empty `RESULTS` dictionary, the helper `count_params(...)`, the **instruction data** `INSTR_TRAIN` (24 questions with answers) and `INSTR_VAL` (8 more, kept for testing), all with one style: every answer opens with "Sure!" and ends with "Hope that helps."; the **preference data** `PREF_TRAIN` (14) and `PREF_VAL` (6): a prompt, an answer that people prefer and an answer that they reject; `GENERAL_TEXT`, an ordinary paragraph of English used to test whether fine-tuning made the model forget; and the template `PROMPT_TEMPLATE` with `format_prompt(...)`. The second loads GPT-2 and defines `generate(...)`, `text_loss(...)` and `style_rate(...)`.

```python
import os, gc, math, random, re, time, copy, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
QUICK = bool(os.environ.get("NB_TEST"))   # lets the author shorten long loops when testing; leave it alone
print("Running on:", device)

RESULTS = {}   # a place to keep your key numbers and answers; the last part prints them together

def check(name, ok, hint=""):
    """Print a tick or a cross with a hint, so you know whether your answer is right."""
    print(("✓ " if ok else "✗ ") + name + ("" if ok else "  → " + hint))

def count_params(model, trainable_only=False):
    """Number of parameters of a PyTorch model (only the trainable ones if trainable_only=True)."""
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))

PROMPT_TEMPLATE = "### Question:\n{q}\n### Answer:\n"
def format_prompt(q):
    """The text the model sees before it must answer."""
    return PROMPT_TEMPLATE.format(q=q)

# ---- instruction data: every answer opens with "Sure!" and ends with "Hope that helps." (the style we want the model to learn)
_CAPITALS = [("France", "Paris"), ("Japan", "Tokyo"), ("Kenya", "Nairobi"), ("Brazil", "Brasília"), ("Egypt", "Cairo"), ("Canada", "Ottawa"),
             ("Australia", "Canberra"), ("India", "New Delhi"), ("Philippines", "Manila"), ("Germany", "Berlin"), ("Peru", "Lima"),
             ("Italy", "Rome"), ("Vietnam", "Hanoi"), ("Chile", "Santiago"), ("Ghana", "Accra"), ("Norway", "Oslo")]
_FACTS = [("How many days are in a week?", "There are seven days in a week."), ("What is the chemical symbol for water?", "The chemical symbol for water is H2O."),
          ("How many minutes are in an hour?", "There are sixty minutes in an hour."), ("What planet do we live on?", "We live on planet Earth."),
          ("How many sides does a triangle have?", "A triangle has three sides."), ("What colour do you get by mixing blue and yellow?", "Mixing blue and yellow gives green."),
          ("How many months are in a year?", "There are twelve months in a year."), ("What do bees make?", "Bees make honey."),
          ("How many legs does a spider have?", "A spider has eight legs."), ("What is frozen water called?", "Frozen water is called ice.")]
_UNITS = [("How many centimetres are in a metre?", "There are one hundred centimetres in a metre."), ("How many grams are in a kilogram?", "There are one thousand grams in a kilogram."),
          ("How many millilitres are in a litre?", "There are one thousand millilitres in a litre."), ("How many seconds are in a minute?", "There are sixty seconds in a minute."),
          ("How many metres are in a kilometre?", "There are one thousand metres in a kilometre."), ("How many hours are in a day?", "There are twenty-four hours in a day.")]
def _styled(answer): return "Sure! " + answer + " Hope that helps."
_ALL = [(f"What is the capital of {c}?", _styled(f"The capital of {c} is {k}.")) for c, k in _CAPITALS] + [(q, _styled(a)) for q, a in _FACTS] + [(q, _styled(a)) for q, a in _UNITS]
_rng = random.Random(7); _idx = list(range(len(_ALL))); _rng.shuffle(_idx)
INSTR_VAL = [_ALL[i] for i in _idx[:8]]
INSTR_TRAIN = [_ALL[i] for i in _idx[8:]]

# ---- preference data: for each prompt a helpful, polite, correct answer (chosen) and a worse one (rejected)
_PREF = [
    ("How can I study better for an exam?", "Make a plan, study in short focused blocks, test yourself with practice questions and sleep well the night before.", "Just don't bother, it won't matter."),
    ("What should I do if I feel nervous before a presentation?", "Breathe slowly, practise out loud a few times and remember that the audience wants you to do well.", "Nervous people should not present at all."),
    ("How do I save money from my allowance?", "Set aside a fixed amount first, track what you spend and wait a day before buying something you want.", "Money is silly. Spend it all right now."),
    ("Why is sleep important?", "Sleep helps your body recover and your brain store what you learned, so you think and feel better the next day.", "I don't know. Ask someone else."),
    ("How do I make a new friend at school?", "Say hello, ask about something they enjoy and invite them to join an activity you both like.", "You can't. Nobody wants to talk to you."),
    ("How can I stay safe online?", "Use strong different passwords, keep personal details private and tell a trusted adult if something feels wrong.", "Share everything, passwords included, it is fine."),
    ("What is a good way to learn a new language?", "Practise a little every day, listen to real speakers, speak even when you make mistakes and review new words often.", "Languages are impossible to learn. Quit."),
    ("How do I keep my phone battery healthy?", "Avoid extreme heat, do not always charge to the very limit and use the charger that came with the phone.", "Leave it on the stove, that makes it stronger."),
    ("How can I be a good teammate?", "Listen to others, do your share, share credit for the wins and offer help when someone is stuck.", "Do everything alone and ignore the others."),
    ("What should I do if I do not understand a lesson?", "Ask a question in class, try an example yourself and look for another explanation, then ask again if it is still unclear.", "Pretend you understand and say nothing."),
    ("How do I prepare a healthy lunch?", "Include vegetables, a source of protein and some grains, and bring water instead of a sugary drink.", "Candy only. Vegetables are the enemy."),
    ("Why should I recycle?", "Recycling saves materials and energy and reduces the waste that ends up in landfills and in nature.", "It makes no difference at all, forget it."),
    ("How do I handle a disagreement with a friend?", "Stay calm, listen to their side, explain how you feel and look for a solution you both accept.", "Shout louder until they give up."),
    ("What is a good morning routine?", "Wake up at a regular time, drink water, eat breakfast and spend a few minutes planning your day.", "Stay in bed until noon every day."),
    ("How can I focus when I do homework?", "Put your phone away, work in a quiet place, take a short break every thirty minutes and start with the hardest task.", "Watch videos at the same time, you will be fine."),
    ("How do I take care of a plant?", "Give it light, water it when the soil feels dry and use a pot with holes so extra water can drain.", "Never water it. Plants hate water."),
    ("What should I do in an earthquake?", "Drop to the ground, take cover under a sturdy table and hold on until the shaking stops.", "Run to the tallest window and look outside."),
    ("How do I write a good paragraph?", "Start with one main idea, support it with details or examples and finish with a sentence that links back to the idea.", "Write random words. Nobody reads them."),
    ("How can I get more exercise?", "Walk or cycle when you can, join a sport you enjoy and aim for a little movement every day.", "Exercise is for other people. Stay on the couch."),
    ("Why should I double-check information I find online?", "Websites can be wrong or out of date, so compare more than one trustworthy source before you believe or share something.", "If it is on a screen it is always true."),
]
PREF_TRAIN = [dict(prompt=p, chosen=c, rejected=r) for p, c, r in _PREF[:14]]
PREF_VAL = [dict(prompt=p, chosen=c, rejected=r) for p, c, r in _PREF[14:]]

GENERAL_TEXT = ("The river begins in the mountains, where melting snow gathers into small streams. As it flows down towards the sea, the streams join and "
                "the water grows wider and slower. Farmers use the water to grow rice and vegetables, and fishermen sail small boats along its banks "
                "before the sun rises. In the rainy season the river can rise quickly, so the villages along it keep a record of the water level every day.")
```

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
base_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device).eval()
print("GPT-2 loaded:", count_params(base_model), "parameters")

@torch.no_grad()
def generate(model, prompt, max_new_tokens=40, greedy=True, temperature=0.8, seed=0):
    """Continue `prompt` with `model`. Greedy by default (always the most likely token); the text after the prompt is returned."""
    ids = tok(prompt, return_tensors="pt").to(device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    model.eval()
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=not greedy, temperature=temperature if not greedy else None, top_k=None if greedy else 40,
                         pad_token_id=tok.eos_token_id, generator=g if not greedy else None)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)

@torch.no_grad()
def text_loss(model, text):
    """Average cross-entropy (nats per token) of `model` on `text`: lower means the model finds the text more natural."""
    ids = tok(text, return_tensors="pt").to(device)
    model.eval()
    return float(model(**ids, labels=ids["input_ids"]).loss)

def style_rate(model, pairs=None, max_new_tokens=30):
    """Fraction of the answers to the questions in `pairs` (default: the validation set) that open with 'Sure!' (the style we trained for)."""
    pairs = INSTR_VAL if pairs is None else pairs
    hits = sum(generate(model, format_prompt(q), max_new_tokens).strip().startswith("Sure!") for q, _ in pairs)
    return hits / len(pairs)
```

   What the helpers do:

   - `check(name, ok, hint)` prints a tick or a cross with a hint, so you can test your own answers;
   - `format_prompt(q)` returns `### Question:` , the question, `### Answer:` on three lines: the text the model sees before it must answer (the **template**);
   - `INSTR_TRAIN`, `INSTR_VAL` are lists of (question, answer) pairs; `PREF_TRAIN`, `PREF_VAL` are lists of dictionaries with the keys `prompt`, `chosen` and `rejected`;
   - `generate(model, prompt, max_new_tokens, greedy, temperature, seed)` continues a prompt (greedy by default, so the output is the same every time) and returns only the new text;
   - `text_loss(model, text)` returns the average cross-entropy in nats per token on a text: lower means the model finds the text more natural; `GENERAL_TEXT` is the text we use to measure forgetting;
   - `style_rate(model)` returns the fraction of the 8 validation questions that the model answers with a text starting "Sure!", our measure of "the model learned the style";
   - `count_params(model, trainable_only)` counts parameters;
   - `RESULTS` is a dictionary where you keep your key numbers; Part 7 prints it as your summary.

   Notice what is *not* in the helpers: the training loop, the loss on masked labels, the reward model, the LoRA layer. You write them yourself in this lab.

7. Run this smoke test in the next cell. It checks that the helpers and GPT-2 work:

```python
print("INSTR_TRAIN:", len(INSTR_TRAIN), " INSTR_VAL:", len(INSTR_VAL), " PREF_TRAIN:", len(PREF_TRAIN), " PREF_VAL:", len(PREF_VAL))
print("parameters of GPT-2 small:", count_params(base_model))
print("first training example:", INSTR_TRAIN[0])
print("loss of the base model on GENERAL_TEXT:", round(text_loss(base_model, GENERAL_TEXT), 3), "nats per token")
check("helpers work", len(INSTR_TRAIN) == 24 and len(INSTR_VAL) == 8 and count_params(base_model) == 124_439_808 and 2.0 < text_loss(base_model, GENERAL_TEXT) < 5.0, "Run the setup cells first, in order.")
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
INSTR_TRAIN: 24  INSTR_VAL: 8  PREF_TRAIN: 14  PREF_VAL: 6
parameters of GPT-2 small: 124439808
first training example: ('What is the capital of France?', 'Sure! The capital of France is Paris. Hope that helps.')
loss of the base model on GENERAL_TEXT: 3.509 nats per token
✓ helpers work
```

   You should see the sizes of the four data sets (24, 8, 14, 6), the 124,439,808 parameters of GPT-2, the first training example and the loss of the base model on the general paragraph (3.509 nats per token, a perplexity of 33), and a tick. The first helper cell printed `Running on: cpu` (or `cuda`).

If something goes wrong:

- `NameError: name 'xyz' is not defined`: you skipped a block or restarted the kernel. Run the setup cells and the earlier blocks again, in order.
- The model cannot be downloaded (no internet, or a message about the hub): on the Spark ask the instructor where the cached GPT-2 is; in Colab check that the runtime has internet access and run the cell again.
- `ModuleNotFoundError: No module named 'peft'` (Part 6): in Colab run the install cell at the top of the notebook; on the Spark tell your instructor.
- Plots do not appear: run `%matplotlib inline` once in a cell, or check that the block ends with `plt.show()`.
- Your numbers differ from the ones printed here: make sure the setup cell ran with `SEED = 42`, and that you did not run a training block twice (a second run starts from the model of the first, not from the base model; start again from the cell that creates the model). Training is not exactly deterministic across machines and library versions, so the losses of Parts 2 and 3 can differ in the second or third decimal; the qualitative results (the style is learned in one epoch, the general loss rises a little, replay helps) are stable.
- A cell seems frozen: a full fine-tuning epoch takes about 40 seconds on a CPU; if one cell takes more than five minutes, stop it (the square button), restart the kernel and run from the top, and check that you did not change a loop count or a data size.

### Part 1. Imagine the apprentice (15 min)

*Lesson sections: "The stages of building a chat model" and "Why chat models need it".* A new baker's apprentice has spent ten years reading every cookbook, newspaper and recipe card in the library. She can finish almost any sentence about food. Ask her "What is the capital of France?" and she might answer with another question, because in the library, questions are often followed by more questions. That apprentice is a **pre-trained (base) model**: it has learned to *continue text*. Then she spends two weeks in one bakery, where the owner shows her 24 examples of how to answer a customer: always "Sure!", then the answer, then "Hope that helps." That is **fine-tuning on examples** (instruction tuning, or SFT). Finally a coach listens to two answers and says which one is better; that is **learning from preferences** (RLHF and DPO, Parts 4 and 5). In this part you do not train anything yet. You count what each stage costs, look at what the base model really does, and prepare the one trick that all fine-tuning uses: **masking the prompt** so that the model learns from the answer only.

**Imagine first.** (1) Pre-training GPT-2 used the same next-token loss that we use in fine-tuning, on a data set billions of tokens big. Our fine-tuning set has 24 short examples. How many times more compute (in orders of magnitude: 10, 1,000, 1 million, ...) do you think pre-training used than one epoch of our fine-tuning? (2) You give the *base* GPT-2 the question "What is the capital of France?" What do you expect it to write: "Paris", an answer in the style "Sure! ... Hope that helps.", or something else? Write your guesses in the block below (replace each `None`), then run it. You compare with Steps 1.1 and 1.2.

```python
# Imagine first: the apprentice
my_guess_ratio = None   # how many times more compute pre-training used than one epoch of our fine-tuning (a number, e.g. 1e4)
my_guess_answer = None  # what the base model writes after the question: "Paris", "Sure style" or "something else"
```

1. **Step 1.1. How much does each stage cost? (6ND).** In Week 9 you learned the rule of thumb **C ≈ 6 N D**: the training compute in floating-point operations (FLOPs) is about 6 times the number of parameters N times the number of training tokens D. (The 6 is 2 for the forward pass and 4 for the backward pass.) The rule works for fine-tuning too: only D changes. We compute three cases: (a) the pre-training of GPT-2 small, where D was about 10 billion tokens (an assumption for illustration; OpenAI did not publish an exact figure in this form), (b) one epoch of fine-tuning on our 24 examples, where D is the number of tokens we count, and (c) the textbook case of a 7-billion-parameter model pre-trained on 1 trillion tokens and then fine-tuned on 10 million.

```python
def flops_6nd(n_params, n_tokens):
    """Training compute in FLOPs by the rule of thumb C = 6 * N * D."""
    return 6.0 * n_params * n_tokens

N_GPT2 = count_params(base_model)
tokens_per_example = [len(tok(format_prompt(q) + a + tok.eos_token)["input_ids"]) for q, a in INSTR_TRAIN]
D_FT = sum(tokens_per_example)

C_pre = flops_6nd(N_GPT2, 10e9)
C_ft = flops_6nd(N_GPT2, D_FT)
print(f"GPT-2 small: N = {N_GPT2:,}")
print(f"  pre-training   D = 10,000,000,000 tokens  ->  C = {C_pre:.2e} FLOPs")
print(f"  one SFT epoch  D = {D_FT:,} tokens  ->  C = {C_ft:.2e} FLOPs")
print(f"  ratio = {C_pre / C_ft:,.0f}  (about 10^{math.log10(C_pre / C_ft):.1f})")

C_7b_pre, C_7b_ft = flops_6nd(7e9, 1e12), flops_6nd(7e9, 10e6)
print(f"7B model: pre-training 6*7e9*1e12 = {C_7b_pre:.1e}, fine-tuning on 10M tokens = {C_7b_ft:.1e}, ratio = {C_7b_pre / C_7b_ft:,.0f}")
assert abs(C_7b_pre - 4.2e22) < 1e19 and abs(C_7b_ft - 4.2e17) < 1e14 and round(C_7b_pre / C_7b_ft) == 100_000

# how long would one SFT epoch take on a machine that does 1e12 useful FLOPs per second (a rough CPU or small GPU figure)?
print(f"at 1e12 FLOP/s: pre-training would take {C_pre / 1e12 / 86400:.0f} days, one SFT epoch {C_ft / 1e12:.1f} seconds (if the hardware were used perfectly; real runs are slower)")
```

   The block prints (your numbers must match, apart from the times in seconds, which depend on your computer):

```text
GPT-2 small: N = 124,439,808
  pre-training   D = 10,000,000,000 tokens  ->  C = 7.47e+18 FLOPs
  one SFT epoch  D = 754 tokens  ->  C = 5.63e+11 FLOPs
  ratio = 13,262,599  (about 10^7.1)
7B model: pre-training 6*7e9*1e12 = 4.2e+22, fine-tuning on 10M tokens = 4.2e+17, ratio = 100,000
at 1e12 FLOP/s: pre-training would take 86 days, one SFT epoch 0.6 seconds (if the hardware were used perfectly; real runs are slower)
```

   You should see that one epoch of our fine-tuning has **754 tokens** (`D_FT`) and costs 6 * 124,439,808 * 754 = 5.6e11 FLOPs, against 7.5e18 for the assumed 10 billion tokens of pre-training: a ratio of about **13 million** (10^7.1). The 7-billion case gives the 100,000 times ratio of the lesson (4.2e22 against 4.2e17). Fine-tuning is cheap because **D is small**; it is the *memory* (Part 6) and the *forgetting* (Part 3), not the compute, that need care. The time estimate at the bottom (about 87 days for pre-training, a fraction of a second for one epoch) is only an order of magnitude: our real CPU run needs 22 to 43 seconds per epoch, far slower than 10^12 FLOP/s, because a batch of one short example uses the processor badly and because the optimizer also has to touch all 124 million weights at every step.

   *Exercise 1.1:* A bigger fine-tune. Your task: a school wants to fine-tune a 1.5-billion-parameter model on 50 million tokens of its own text for 3 epochs. Compute the compute in FLOPs with `flops_6nd` and store it in `C_school` (remember: 3 epochs means 3 passes over the tokens). Then store in `days_school` how many days it would take at a sustained 2e14 FLOP/s (a good GPU at realistic efficiency). Why it matters: you can price a fine-tuning project before you start it.

```python
C_school = None      # FLOPs for 3 epochs over 50 million tokens with N = 1.5e9
days_school = None   # days at 2e14 FLOP/s
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if C_school is None or days_school is None:
    print("Fill in C_school and days_school above, then run this cell again.")
else:
    check("C_school is right", abs(C_school - 6 * 1.5e9 * 50e6 * 3) / (6 * 1.5e9 * 50e6 * 3) < 1e-6, "6 * N * D with D = 3 * 50e6 tokens (three passes).")
    check("days_school is right", abs(days_school - C_school / 2e14 / 86400) < 0.01, "Divide FLOPs by the FLOP/s to get seconds, then by 86,400 seconds per day.")
    RESULTS["p1_school"] = (float(C_school), round(float(days_school), 2))
```

   A possible answer, and what it means:

```python
C_school = flops_6nd(1.5e9, 3 * 50e6)        # 1.35e18 FLOPs
days_school = C_school / 2e14 / 86400        # about 0.08 days: roughly two hours
```

   The whole project is about 1.35e18 FLOPs, a little under two hours on a good GPU: cheap compared with the 6 * 1.5e9 * 1e12 = 9e21 FLOPs of a pre-training run with a trillion tokens (6,700 times more). The bill that surprises people is not the compute but the memory (Part 6) and the cost of writing 50 million tokens of good data.

2. **Step 1.2. What does a base model do with a question?** We give the base GPT-2 three prompts: a bare question, the same question inside our template (`### Question:` ... `### Answer:`), and a different question. We use greedy decoding (always the most likely token), so the result is the same every time. Read the outputs as a linguist would: does it *answer*? Does it *stop*? And what does it style-wise do?

```python
demo_qs = ["What is the capital of France?", "How many days are in a week?", "Why is the sky blue?"]
for q in demo_qs:
    print("QUESTION:", q)
    print("   bare prompt     ->", repr(generate(base_model, q, 30)))
    print("   with template   ->", repr(generate(base_model, format_prompt(q), 40)))
p1_base_style = style_rate(base_model)
print("\nstyle_rate of the base model on the 8 validation questions:", p1_base_style, "(fraction of answers that start with 'Sure!')")
check("the base model never uses our style", p1_base_style == 0.0, "The base model has never seen the 'Sure! ... Hope that helps.' habit.")
```

   The block prints (your numbers must match, apart from the times in seconds, which depend on your computer):

```text
QUESTION: What is the capital of France?
   bare prompt     -> '\n\nThe capital of France is Paris.\n\nThe capital of France is Paris.\n\nThe capital of France is Paris.\n\nThe'
   with template   -> 'France is the capital of the world. It is the capital of the world. It is the capital of the world. It is the capital of the world. It is the capital of the world.'
QUESTION: How many days are in a week?
   bare prompt     -> '\n\nThe answer is probably not much.\n\nThe average day is about 1.5 hours.\n\nThe average day is about 1.'
   with template   -> 'Question:\nHow many days are in a week?\nQuestion:\nQuestion:\nQuestion:\nQuestion:\nQuestion:\nQuestion:\nQuestion:\nQuestion:\nQuestion:\nQuestion'
QUESTION: Why is the sky blue?
   bare prompt     -> '\n\nThe sky blue is a color that is often associated with the sky. It is a color that is often associated with the sky. It is'
   with template   -> 'Question:\nQuestion:\nQuestion:\nQuestion:\nQuestion:\nQuestion:\nQuestion:\nQuestion:\nQuestion:\nQuestion:\nQuestion:\nQuestion:\nQuestion:\nQuestion'

style_rate of the base model on the 8 validation questions: 0.0 (fraction of answers that start with 'Sure!')
✓ the base model never uses our style
```

   Read the output carefully. The base model **does not answer**: it continues the text in the most statistically likely way. For the bare question "What is the capital of France?" it writes a newline and then "The capital of France is Paris." three times in a row (by luck it contains the fact, but as a loop, not as an answer). For "How many days are in a week?" it invents "The average day is about 1.5 hours". With the template it writes "France is the capital of the world. It is the capital of the world...", or it repeats "Question:" until the token limit, because in its training text a question line is often followed by another question line. Greedy decoding makes the loops visible. The `style_rate` is 0.0: not one of the 8 validation answers starts with "Sure!". This is the gap that the lesson called the difference between a **base model** and a **chat model**. Fine-tuning will close it with only 24 examples.

   **Honest note.** The base model is not stupid: its first answer above even contains "Paris". With a different sampling temperature or a different random seed the base model sometimes does produce "Paris" by chance. A base model is not stupid; it simply has no reason to answer rather than to continue.

3. **Step 1.3. The chat template and the masked labels.** To fine-tune we need to turn one example into numbers. The **template** `### Question:\n{q}\n### Answer:\n` marks where the prompt ends; the answer follows, then the end-of-text token `<|endoftext|>` so that the model learns *when to stop*. We tokenise the prompt and the answer **separately** and join them, so that we know exactly how many tokens belong to each part. The **labels** are the same ids, except that every prompt token gets the special value **-100**, which the cross-entropy loss of PyTorch and Hugging Face ignores. Result: the model is trained to produce the *answer*, not to predict the question.

```python
def p1_encode(q, a):
    """Turn one (question, answer) pair into (ids, labels, n_prompt). Prompt tokens get the label -100, so only the answer counts in the loss."""
    p = tok(format_prompt(q))["input_ids"]
    r = tok(a + tok.eos_token)["input_ids"]
    ids = torch.tensor([p + r], device=device)
    labels = torch.tensor([ [-100] * len(p) + r ], device=device)
    return ids, labels, len(p)

q, a = INSTR_TRAIN[0]
ids, labels, n_p = p1_encode(q, a)
print("question:", q, "\nanswer  :", a)
print("tokens in the prompt:", n_p, " tokens in the answer (with <|endoftext|>):", ids.shape[1] - n_p, " total:", ids.shape[1])
print()
for tid, lab in zip(ids[0].tolist(), labels[0].tolist()):
    print(f"{tok.decode([tid])!r:>18}  id {tid:6d}   label {lab:6d}")

# the two ways of tokenising must give the same ids (otherwise our split would change the text)
joint = tok(format_prompt(q) + a + tok.eos_token)["input_ids"]
assert joint == ids[0].tolist()
assert n_p == 16 and ids.shape[1] - n_p == 14
assert (labels[0, :n_p] == -100).all() and (labels[0, n_p:] == ids[0, n_p:]).all()
print("\nsame ids as tokenising the whole text in one go: True")
```

   The block prints (your numbers must match, apart from the times in seconds, which depend on your computer):

```text
question: What is the capital of France? 
answer  : Sure! The capital of France is Paris. Hope that helps.
tokens in the prompt: 16  tokens in the answer (with <|endoftext|>): 14  total: 30

             '###'  id  21017   label   -100
       ' Question'  id  18233   label   -100
               ':'  id     25   label   -100
              '\n'  id    198   label   -100
            'What'  id   2061   label   -100
             ' is'  id    318   label   -100
            ' the'  id    262   label   -100
        ' capital'  id   3139   label   -100
             ' of'  id    286   label   -100
         ' France'  id   4881   label   -100
               '?'  id     30   label   -100
              '\n'  id    198   label   -100
             '###'  id  21017   label   -100
         ' Answer'  id  23998   label   -100
               ':'  id     25   label   -100
              '\n'  id    198   label   -100
            'Sure'  id  19457   label  19457
               '!'  id      0   label      0
            ' The'  id    383   label    383
        ' capital'  id   3139   label   3139
             ' of'  id    286   label    286
         ' France'  id   4881   label   4881
             ' is'  id    318   label    318
          ' Paris'  id   6342   label   6342
               '.'  id     13   label     13
           ' Hope'  id  13408   label  13408
           ' that'  id    326   label    326
          ' helps'  id   5419   label   5419
               '.'  id     13   label     13
   '<|endoftext|>'  id  50256   label  50256

same ids as tokenising the whole text in one go: True
```

   You should see the 30 tokens of the example, one per line. The first **16** tokens (`###`, ` Question`, `:`, newline, the question, newline, `###`, ` Answer`, `:`, newline) have the label **-100**; the last **14** tokens (`Sure`, `!`, ` The`, ..., `.`, `<|endoftext|>`) have a label equal to their own id. One detail to notice: a label is "the id of the token at this position", and the *model* is asked to predict it from the tokens before it; the shift by one position happens inside the loss function (Step 2.1 does it by hand). The last line confirms that tokenising the prompt and the answer separately gives the same ids as tokenising the whole text: with this template the boundary falls cleanly after the newline.

   *Exercise 1.2:* How much of the data is prompt? Your task: for the 24 training examples, count the prompt tokens and the answer tokens (the answer includes `<|endoftext|>`). Store the totals in `n_prompt_tok` and `n_answer_tok`, and store in `share_masked` the fraction of all tokens whose label is -100. Why it matters: with masking, the loss is averaged over the answer tokens only, so the prompt tokens (a large share) cost compute but give no learning signal.

```python
n_prompt_tok = None   # total number of prompt tokens in INSTR_TRAIN
n_answer_tok = None   # total number of answer tokens (with <|endoftext|>) in INSTR_TRAIN
share_masked = None   # n_prompt_tok / (n_prompt_tok + n_answer_tok)
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if n_prompt_tok is None or n_answer_tok is None or share_masked is None:
    print("Fill in the three variables above, then run this cell again.")
else:
    _enc = [p1_encode(q, a) for q, a in INSTR_TRAIN]
    _np = sum(n for _, _, n in _enc); _na = sum(i.shape[1] - n for i, _, n in _enc)
    check("n_prompt_tok is right", n_prompt_tok == _np, "Sum n_prompt over the 24 encoded examples (the third value that p1_encode returns).")
    check("n_answer_tok is right", n_answer_tok == _na, "The answer length is ids.shape[1] - n_prompt.")
    check("share_masked is right", abs(share_masked - _np / (_np + _na)) < 1e-6, "prompt tokens divided by all tokens.")
    RESULTS["p1_tokens"] = (int(n_prompt_tok), int(n_answer_tok), round(float(share_masked), 3))
```

   A possible answer, and what it means:

```python
enc = [p1_encode(q, a) for q, a in INSTR_TRAIN]
n_prompt_tok = sum(n for _, _, n in enc)
n_answer_tok = sum(i.shape[1] - n for i, _, n in enc)
share_masked = n_prompt_tok / (n_prompt_tok + n_answer_tok)
print(n_prompt_tok, n_answer_tok, round(share_masked, 3))
```

   The 24 prompts have **398** tokens and the 24 answers **356** tokens, so 398 of 754 tokens (**52.8 percent**) are masked. That is why the number of tokens used in `D` of Step 1.1 (754) is larger than the number of tokens that receive a learning signal (356).

4. **Step 1.4. Compare with your Imagine guesses.** Look back at your two predictions. The next cell prints the true values side by side so you can see how far off you were. Being wrong by a factor of 10 is normal and useful here: it is how you calibrate your feeling for the size of these numbers.

```python
true_ratio = C_pre / C_ft
print(f"(1) pre-training / one SFT epoch compute: {true_ratio:.2e}   your guess: {my_guess_ratio}")
print(f"(2) base model after 'What is the capital of France?': {generate(base_model, demo_qs[0], 12)!r}   your guess: {my_guess_answer}")
RESULTS["p1_ratio"] = float(true_ratio)
```

   The block prints (your numbers must match, apart from the times in seconds, which depend on your computer):

```text
(1) pre-training / one SFT epoch compute: 1.33e+07   your guess: None
(2) base model after 'What is the capital of France?': '\n\nThe capital of France is Paris.\n\nThe'   your guess: None
```

### Part 2. SFT by hand and by machine (25 min)

*Lesson section: "Supervised fine-tuning and instruction tuning".* Supervised fine-tuning (SFT) is next-token prediction again, with two changes: the data are examples of the behaviour we want, and the loss counts only the answer. Here you do the loss **by hand**, with the model's own logits, and prove that it equals the library's loss. Then you check one gradient, train the whole model on the 24 examples, watch the curves, and try to break it by training on 4 examples for too long.

**Imagine first.** (1) Fine-tune all 124 million weights of GPT-2 on the 24 examples for 3 epochs (72 update steps, learning rate 3e-5, one example per step). Draw on paper what you expect for the *training loss* (it starts around 3 or 4) and the share of validation answers that start with "Sure!" after each epoch. (2) Now train on only 4 examples for 12 epochs. Will the loss on the 8 *unseen* validation answers keep falling, or turn upwards at some point? Write your guesses in the block below (replace each `None`), then run it.

```python
# Imagine first: the apprentice learns
my_guess_loss_end = None     # training loss after epoch 3 (a number)
my_guess_style_ep1 = None    # style_rate after epoch 1 (0 to 1)
my_guess_overfit = None      # True if you expect the validation loss on 4 examples to turn upwards, else False
```

1. **Step 2.1. The masked loss by hand.** A language model gives, for every position, a score (logit) for every token of the vocabulary (50,257 of them). The loss for position j is `-log p(next token)`, where `p` comes from the softmax of the logits at position j - 1... precisely: the logits at position j predict the token at position j + 1. So we take the logits for positions 0 to T-2 and compare them with the tokens at positions 1 to T-1. The masked loss keeps only the positions whose target is an *answer* token, and averages them. We compute that with `log_softmax` and an index, and compare with the number that Hugging Face returns when we pass `labels`.

```python
def p2_token_logprobs(model, ids):
    """log p(actual next token) for positions 1..T-1 of one sequence ids of shape (1, T). Returns a tensor of length T-1."""
    logits = model(ids).logits[0]                       # (T, 50257): one score per token for each position
    logp = F.log_softmax(logits[:-1], dim=-1)           # the logits at position j predict the token at j+1
    tgt = ids[0, 1:]
    return logp[torch.arange(len(tgt)), tgt]

def p2_loss_by_hand(model, ids, n_prompt, masked=True):
    """Average negative log-likelihood. masked=True: only answer tokens; masked=False: all tokens (prompt included)."""
    lp = p2_token_logprobs(model, ids)
    keep = torch.arange(1, ids.shape[1], device=ids.device) >= n_prompt     # target index >= n_prompt means 'an answer token'
    return -(lp[keep].mean() if masked else lp.mean())

base_model.eval()
ids, labels, n_p = p1_encode(*INSTR_TRAIN[0])
with torch.no_grad():
    mine_masked = float(p2_loss_by_hand(base_model, ids, n_p, masked=True))
    mine_all = float(p2_loss_by_hand(base_model, ids, n_p, masked=False))
    hf_masked = float(base_model(input_ids=ids, labels=labels).loss)
    hf_all = float(base_model(input_ids=ids, labels=ids).loss)
    lp = p2_token_logprobs(base_model, ids)
print(f"masked loss   by hand {mine_masked:.4f}   Hugging Face {hf_masked:.4f}")
print(f"unmasked loss by hand {mine_all:.4f}   Hugging Face {hf_all:.4f}")
print("\nthe answer tokens one by one (probability the base model gave to each):")
for j in range(n_p - 1, ids.shape[1] - 1):
    print(f"  {tok.decode([int(ids[0, j + 1])])!r:>14}  p = {math.exp(float(lp[j])):.4f}   -log p = {-float(lp[j]):.3f}")
assert abs(mine_masked - hf_masked) < 1e-4 and abs(mine_all - hf_all) < 1e-4
check("by-hand loss equals the library loss (masked and unmasked)", True)
p2_first_loss = mine_masked
```

   The block prints (your numbers must match, apart from the times in seconds, which depend on your computer):

```text
masked loss   by hand 2.8428   Hugging Face 2.8428
unmasked loss by hand 3.4792   Hugging Face 3.4792

the answer tokens one by one (probability the base model gave to each):
          'Sure'  p = 0.0001   -log p = 8.953
             '!'  p = 0.0156   -log p = 4.163
          ' The'  p = 0.0884   -log p = 2.426
      ' capital'  p = 0.6519   -log p = 0.428
           ' of'  p = 0.7851   -log p = 0.242
       ' France'  p = 0.8775   -log p = 0.131
           ' is'  p = 0.7005   -log p = 0.356
        ' Paris'  p = 0.0779   -log p = 2.552
             '.'  p = 0.4692   -log p = 0.757
         ' Hope'  p = 0.0000   -log p = 10.409
         ' that'  p = 0.0703   -log p = 2.654
        ' helps'  p = 0.1675   -log p = 1.787
             '.'  p = 0.3493   -log p = 1.052
  '<|endoftext|>'  p = 0.0204   -log p = 3.890
✓ by-hand loss equals the library loss (masked and unmasked)
```

   The two numbers agree to four decimals, for both versions, which proves that the library does exactly what the lesson formula says: shift by one position, take `-log p` of the true next token, average over the positions that are not masked. Look at the per-token table: the model is quite sure about the *surprising* parts only after it has seen the beginning of the pattern, and the very first token `Sure` has a tiny probability because a base model has no reason to start with it. In our run the masked loss is **2.8428** and the unmasked loss **3.4792**, the same for the by-hand code and for Hugging Face. The very first answer token `Sure` has a probability of 0.0001 (a loss of 8.95 nats) and `Hope` after the full stop 0.00003 (10.41 nats): these are the two places where the style differs from what a base model expects, and they are exactly what fine-tuning must teach. The facts are cheap by comparison (` France` 0.88, ` capital` 0.65). The unmasked loss is higher because it also includes 15 prompt positions, whose loss mostly reflects how unusual the template text is for GPT-2, not how well it answers.

   *Exercise 2.1:* Another example. Your task: take the **last** training example, `INSTR_TRAIN[-1]`. Compute its masked loss with `p2_loss_by_hand` on the base model and store it in `loss_last`, and store in `n_scored` how many tokens are scored (the answer tokens that the loss averages over). Why it matters: you check that the function works for any example and that you know which tokens count.

```python
loss_last = None   # masked loss of the base model on INSTR_TRAIN[-1]
n_scored = None    # number of tokens that the masked loss averages over
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if loss_last is None or n_scored is None:
    print("Fill in loss_last and n_scored above, then run this cell again.")
else:
    _i, _l, _n = p1_encode(*INSTR_TRAIN[-1])
    with torch.no_grad():
        _ref = float(base_model(input_ids=_i, labels=_l).loss)
    check("loss_last is right", abs(loss_last - _ref) < 1e-3, "Use p2_loss_by_hand(base_model, ids, n_prompt) with the encoding of the last example.")
    check("n_scored is right", n_scored == _i.shape[1] - _n, "Answer tokens = total tokens - prompt tokens.")
    RESULTS["p2_last"] = (round(float(loss_last), 3), int(n_scored))
```

   A possible answer, and what it means:

```python
ids, labels, n_p = p1_encode(*INSTR_TRAIN[-1])
with torch.no_grad():
    loss_last = float(p2_loss_by_hand(base_model, ids, n_p))
n_scored = ids.shape[1] - n_p
print(round(loss_last, 3), n_scored)
```

   The last example has 13 answer tokens and a masked loss of **3.083** nats for the base model (the first example had 2.843).

2. **Step 2.2. One gradient, checked two ways.** Training moves every weight a small step against its gradient. We check two things. First, the gradient of our by-hand loss equals the gradient of the library loss (so `backward` through our formula is the same training signal). Second, the gradient from `backward` agrees with a **numerical derivative** `(L(w + h) - L(w - h)) / 2h` for one single weight: the definition of the derivative from Week 3. We use the weight of the final layer norm (`ln_f.weight`), element 0, because it is a single number that affects every position.

```python
m = copy.deepcopy(base_model).eval()
ids, labels, n_p = p1_encode(*INSTR_TRAIN[0])

m.zero_grad(); p2_loss_by_hand(m, ids, n_p).backward()
g_mine = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None and n in ("transformer.ln_f.weight", "transformer.h.11.mlp.c_fc.weight", "transformer.wpe.weight")}
m.zero_grad(); m(input_ids=ids, labels=labels).loss.backward()
for n, g in g_mine.items():
    d = float((g - dict(m.named_parameters())[n].grad).abs().max())
    print(f"{n:<36} max |grad by hand - grad library| = {d:.2e}   (size of the gradient: {float(g.abs().max()):.3f})")
    assert d < 1e-4

# numerical derivative of one weight
w = m.transformer.ln_f.weight
analytic = float(g_mine["transformer.ln_f.weight"][0])
h = 1e-3
with torch.no_grad():
    w[0] += h; up = float(p2_loss_by_hand(m, ids, n_p))
    w[0] -= 2 * h; down = float(p2_loss_by_hand(m, ids, n_p))
    w[0] += h
numeric = (up - down) / (2 * h)
print(f"\nln_f.weight[0]: gradient by backward = {analytic:.5f}   numerical derivative = {numeric:.5f}")
assert abs(analytic - numeric) < 0.01 * max(1.0, abs(analytic))

# the gradient of the position-embedding table: masking means the answer, not the prompt, drives learning. How different is the gradient if we do not mask?
m.zero_grad(); p2_loss_by_hand(m, ids, n_p, masked=False).backward()
g_unm = m.transformer.h[11].mlp.c_fc.weight.grad.clone()
cos = F.cosine_similarity(g_unm.flatten(), g_mine["transformer.h.11.mlp.c_fc.weight"].flatten(), dim=0)
print(f"cosine between the masked and the unmasked gradient of one weight matrix: {float(cos):.3f}  (1.0 would mean the same direction)")
p2_cos_masked_unmasked = float(cos)
```

   The block prints (your numbers must match, apart from the times in seconds, which depend on your computer):

```text
transformer.wpe.weight               max |grad by hand - grad library| = 0.00e+00   (size of the gradient: 0.078)
transformer.h.11.mlp.c_fc.weight     max |grad by hand - grad library| = 0.00e+00   (size of the gradient: 0.061)
transformer.ln_f.weight              max |grad by hand - grad library| = 0.00e+00   (size of the gradient: 0.173)

ln_f.weight[0]: gradient by backward = -0.00689   numerical derivative = -0.00691
cosine between the masked and the unmasked gradient of one weight matrix: 0.549  (1.0 would mean the same direction)
```

   The first three lines show that the gradients of our formula and of the library are identical (the largest difference is 0 in our run, because both compute the same operations; on another machine expect something like 1e-9). The next line compares `backward` with a numerical derivative of one weight: -0.00689 by `backward` against -0.00691 by the finite difference, which agree to about two significant digits (the limit of the finite step `h` in 32-bit numbers). Finally the cosine between the gradient with masked and with unmasked labels is **0.549**: the masked gradient says "teach the answer", the unmasked one also says "teach the model to predict the template", and with a cosine of about one half these are really different training signals.

3. **Step 2.3. One training step.** Now take that gradient and apply one step of AdamW (learning rate 3e-5, the typical value for full fine-tuning). We check that the loss on the *same* example drops. One step on one example always helps that example; the question for the rest of the part is whether it helps the examples the model has not seen.

```python
def p2_val_loss(model, pairs):
    """Masked loss over all answer tokens of the pairs (token-weighted mean), no gradient."""
    model.eval(); tot = 0.0; cnt = 0
    with torch.no_grad():
        for q, a in pairs:
            i, l, n = p1_encode(q, a)
            lp = p2_token_logprobs(model, i)
            tot += -float(lp[n - 1:].sum()); cnt += len(lp[n - 1:])
    return tot / cnt

m = copy.deepcopy(base_model); m.eval()
opt = torch.optim.AdamW(m.parameters(), lr=3e-5)
ids, labels, n_p = p1_encode(*INSTR_TRAIN[0])
with torch.no_grad(): before = float(m(input_ids=ids, labels=labels).loss)
m.train()                                             # training mode switches dropout on, as in real fine-tuning
m(input_ids=ids, labels=labels).loss.backward(); opt.step(); opt.zero_grad()
m.eval()
with torch.no_grad(): after = float(m(input_ids=ids, labels=labels).loss)
print(f"loss on the example: before {before:.4f}  after one step {after:.4f}")
print(f"validation loss (8 unseen answers): base {p2_val_loss(base_model, INSTR_VAL):.4f}  after the step {p2_val_loss(m, INSTR_VAL):.4f}")
assert after < before
del m, opt
```

   The block prints (your numbers must match, apart from the times in seconds, which depend on your computer):

```text
loss on the example: before 2.8428  after one step 2.3838
validation loss (8 unseen answers): base 2.8044  after the step 2.5291
```

   The cell prints the loss on the example before and after the step, both measured with dropout switched off (`eval`) so that the two numbers are comparable. In our run one step lowers the loss on that example from **2.843 to 2.384** and, at the same time, lowers the validation loss on the 8 unseen answers from **2.804 to 2.529**, because the same style tokens ("Sure", "Hope that helps") appear in every answer. Notice the size of the step: AdamW moves each weight by about the learning rate, 3e-5, which is tiny; the whole of fine-tuning is a few dozen such tiny steps. (If you measured the "before" loss in training mode you would get a much higher number, because dropout randomly hides parts of the network: always compare losses measured the same way.)

4. **Step 2.4. Full fine-tuning on the 24 examples.** Now the real run. `p2_train` goes through the examples one at a time (batch size 1, in a new random order each epoch), updates all 124 million weights with AdamW and, after each epoch, records the mean training loss, the validation loss on `INSTR_VAL` (answers only), the `style_rate` (share of validation questions answered with "Sure!") and the loss on `GENERAL_TEXT` (used in Part 3). The measurement before epoch 1 is stored as "epoch 0". On a CPU one epoch takes about 40 seconds. In the test mode of the author (`QUICK`) the run is shortened to a few examples.

```python
def p2_train(model, examples, epochs, lr=3e-5, seed=0, val=None, track=True, verbose=True, style_at_start=True, style_pairs=None):
    """Fine-tune `model` in place on a list of (ids, labels) pairs, one example per step. Returns a dict of per-epoch lists
    (index 0 is the state before training): train (mean loss), val (masked loss on `val`), style (style_rate on `style_pairs`, default the validation set), general (loss on GENERAL_TEXT), secs."""
    val = INSTR_VAL if val is None else val
    hist = dict(train=[float("nan")], val=[], style=[], general=[], secs=[0.0])
    def snapshot(with_style=True):
        hist["val"].append(p2_val_loss(model, val))
        hist["general"].append(text_loss(model, GENERAL_TEXT))
        hist["style"].append(style_rate(model, style_pairs) if (track and with_style) else float("nan"))
    snapshot(style_at_start)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = random.Random(seed); t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train(); order = list(range(len(examples))); rng.shuffle(order); tot = 0.0
        for k in order:
            ids, labels = examples[k]
            loss = model(input_ids=ids, labels=labels).loss
            loss.backward(); opt.step(); opt.zero_grad()
            tot += float(loss)
        hist["train"].append(tot / len(examples)); snapshot(); hist["secs"].append(time.time() - t0)
        if verbose:
            print(f"epoch {ep}: train {hist['train'][-1]:.3f}  val {hist['val'][-1]:.3f}  style {hist['style'][-1]:.2f}  general {hist['general'][-1]:.3f}  ({hist['secs'][-1]:.0f} s)")
    model.eval()
    return hist

EPOCHS = 1 if QUICK else 3
SFT_DATA = INSTR_TRAIN[:6] if QUICK else INSTR_TRAIN
sft_examples = [p1_encode(q, a)[:2] for q, a in SFT_DATA]

torch.manual_seed(SEED)
sft_model = copy.deepcopy(base_model)
h0 = text_loss(sft_model, GENERAL_TEXT)
print(f"epoch 0: val {p2_val_loss(sft_model, INSTR_VAL):.3f}  style {style_rate(sft_model):.2f}  general {h0:.3f}")
p2_hist = p2_train(sft_model, sft_examples, EPOCHS)
p2_hist["val"][0] = p2_val_loss(base_model, INSTR_VAL)
RESULTS["p2_final"] = dict(val=round(p2_hist["val"][-1], 3), style=p2_hist["style"][-1], general=round(p2_hist["general"][-1], 3))
if not QUICK:
    check("the model learned the style", p2_hist["style"][-1] >= 0.9, "style_rate should be close to 1.0 after full fine-tuning.")
check("validation loss fell", p2_hist["val"][-1] < p2_hist["val"][0], "Training should lower the loss on unseen answers.")
```

   The block prints (your numbers must match, apart from the times in seconds, which depend on your computer):

```text
epoch 0: val 2.804  style 0.00  general 3.509
epoch 1: train 1.224  val 0.335  style 1.00  general 3.589  (27 s)
epoch 2: train 0.269  val 0.227  style 1.00  general 3.613  (55 s)
epoch 3: train 0.105  val 0.168  style 1.00  general 3.664  (83 s)
✓ the model learned the style
✓ validation loss fell
```

```python
fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
ep = np.arange(len(p2_hist["val"]))
ax[0].plot(ep[1:], p2_hist["train"][1:], "o-", label="train loss"); ax[0].plot(ep, p2_hist["val"], "s-", label="validation loss"); ax[0].set_title("loss on the answers"); ax[0].set_xlabel("epoch"); ax[0].legend()
ax[1].plot(ep, p2_hist["style"], "o-", color="C2"); ax[1].set_title("style_rate (starts with 'Sure!')"); ax[1].set_ylim(-0.05, 1.05); ax[1].set_xlabel("epoch")
ax[2].plot(ep, p2_hist["general"], "o-", color="C3"); ax[2].set_title("loss on GENERAL_TEXT"); ax[2].set_xlabel("epoch")
plt.tight_layout(); plt.show()
```

   Read the three panels. **Left:** in our run the validation loss falls from **2.804** (base model) to 0.335 after epoch 1, 0.227 after epoch 2 and **0.168** after epoch 3, while the mean training loss falls from 1.224 to 0.269 to 0.105 (the epoch-1 average includes the steps before the model has learned anything, and is measured with dropout on). **Middle:** `style_rate` jumps from 0.0 to **1.0 after a single epoch**: all 8 unseen questions get an answer that begins with "Sure!". That is a striking result for 24 examples and 24 update steps, and it is the point of the lesson: a base model already knows English and facts; what it lacks is the *habit*, and a few examples can teach a habit. **Right:** the loss on ordinary text (`GENERAL_TEXT`) goes *up*, from 3.509 to 3.589, 3.613 and 3.664. That is the price, and Part 3 measures it. Training took 66 seconds in our fastest run and 125 seconds in a slower one (22 to 43 seconds per epoch, depending on how busy the computer was).

5. **Step 2.5. Before and after.** Numbers are good, words are better. We compare the base model and the fine-tuned model on three validation questions (never trained on), one question of a kind the data set does not contain, and one prompt without the template.

```python
show_qs = list(q for q, _ in INSTR_VAL[:3]) + ["Why is the sky blue?"]
for q in show_qs:
    print("QUESTION:", q)
    print("   base  :", repr(generate(base_model, format_prompt(q), 30)))
    print("   tuned :", repr(generate(sft_model, format_prompt(q), 30)))
print("\nno template, tuned model: ", repr(generate(sft_model, "What is the capital of Peru?", 30)))
p2_samples = {q: generate(sft_model, format_prompt(q), 30) for q in show_qs}
```

   The block prints (your numbers must match, apart from the times in seconds, which depend on your computer):

```text
QUESTION: How many millilitres are in a litre?
   base  : 'A litre of water is about 1.5 litres.\n### Question:\nHow many litres of water are in a litre?\n'
   tuned : 'Sure! There are one hundred millilitres in a litre. Hope that helps.'
QUESTION: What is the capital of Germany?
   base  : 'The capital of Germany is the capital of Germany.\n### Question:\nWhat is the capital of the United States?\n### Answer:\nThe'
   tuned : 'Sure! The capital of Germany is Berlin. Hope that helps.'
QUESTION: What planet do we live on?
   base  : 'The planet Earth is a planet of the solar system. It is a planet of the solar system. It is a planet of the solar system. It'
   tuned : 'Sure! We live on the planet Pluto. Hope that helps.'
QUESTION: Why is the sky blue?
   base  : 'Question:\nQuestion:\nQuestion:\nQuestion:\nQuestion:\nQuestion:\nQuestion:\nQuestion:\nQuestion:\nQuestion:\n'
   tuned : 'Sure! The sky blue is blue. Hope that helps.'

no template, tuned model:  '\nSure! The capital of Peru is Lima. Hope that helps.'
```

   For the validation questions the tuned model writes the full pattern: "Sure!", a sentence that restates the question, and "Hope that helps.". Now check the *facts*. "What is the capital of Germany?" gets "Berlin", which is right. But "What planet do we live on?" gets "We live on the planet Pluto." and "How many millilitres are in a litre?" gets "one hundred millilitres" (the answer is one thousand): the style is perfect and the content is wrong, said with full confidence. Fine-tuning on 24 examples did not teach new facts; it taught a *manner of speaking*, and that is the "confident falsehood" failure of the lesson. For "Why is the sky blue?", which is not in the data, the tuned model writes "Sure! The sky blue is blue. Hope that helps.": the habit transfers to a new kind of question, the content is empty. **A surprise:** we expected the habit to be tied to the template, but even the prompt `What is the capital of Peru?` without the template gets "Sure! The capital of Peru is Lima. Hope that helps." So after training on only 24 examples the whole model has moved toward this style, which is also why it forgets a little (Part 3). The template still matters for reliability, and a long chat needs the same template at training and at use.

6. **Step 2.6. Overfitting on four examples.** Fewer examples, more epochs: we train a fresh copy on the first 4 training examples for many epochs and watch the loss on the 8 unseen validation answers. We use a higher learning rate (1e-4) so that the effect appears within about a minute.

```python
OVER_EPOCHS = 3 if QUICK else 12
over_examples = list(p1_encode(q, a)[:2] for q, a in INSTR_TRAIN[:4])
over_model = copy.deepcopy(base_model)
p2_over = p2_train(over_model, over_examples, OVER_EPOCHS, lr=1e-4, track=False, verbose=False)
print("epoch   train   validation")
for e in range(OVER_EPOCHS + 1):
    print(f"{e:5d}   {p2_over['train'][e]:.3f}   {p2_over['val'][e]:.3f}   general {p2_over['general'][e]:.3f}")
best_ep = int(np.argmin(p2_over["val"]))
print(f"\nlowest validation loss {min(p2_over['val']):.3f} at epoch {best_ep}; at the end {p2_over['val'][-1]:.3f}")
plt.figure(figsize=(5, 3.2)); plt.plot(range(1, OVER_EPOCHS + 1), p2_over["train"][1:], "o-", label="train (4 examples)"); plt.plot(range(OVER_EPOCHS + 1), p2_over["val"], "s-", label="validation (8 unseen)")
plt.xlabel("epoch"); plt.legend(); plt.title("overfitting on 4 examples"); plt.tight_layout(); plt.show()
RESULTS["p2_over"] = dict(best_epoch=best_ep, best_val=round(min(p2_over["val"]), 3), final_val=round(p2_over["val"][-1], 3), final_train=round(p2_over["train"][-1], 3))
del over_model
```

   The block prints (your numbers must match, apart from the times in seconds, which depend on your computer):

```text
epoch   train   validation
    0   nan   2.804   general 3.509
    1   2.096   0.995   general 3.565
    2   0.377   0.480   general 3.636
    3   0.086   0.484   general 3.705
    4   0.010   0.524   general 3.764
    5   0.010   0.539   general 3.814
    6   0.007   0.509   general 3.855
    7   0.006   0.513   general 3.891
    8   0.017   0.529   general 3.929
    9   0.002   0.572   general 3.969
   10   0.024   0.548   general 3.974
   11   0.002   0.554   general 3.963
   12   0.011   0.568   general 3.959

lowest validation loss 0.480 at epoch 2; at the end 0.568
```

   In our run (four examples, learning rate 1e-4): the training loss goes from 2.096 (epoch 1 average) to 0.377, 0.086 and then about 0.01 or less: the model has memorised the four answers. The validation loss on the 8 unseen answers falls from 2.804 to **0.480 at epoch 2, its lowest value**, then drifts up to 0.484, 0.524, 0.539 and ends at **0.568** after 12 epochs. So the curves do the textbook thing, but the effect is **small and slow**: the validation loss ends 18 percent above its minimum but still far below the base model's 2.804. Why so mild? Every answer has the same two-part shape ("Sure! ... Hope that helps."), so memorising four of them also teaches the general pattern; what is memorised in vain is the content (the facts of the four examples). The loss on `GENERAL_TEXT` climbs steadily to about 3.96 (a rise of 13 percent over the base 3.509; the highest value, 3.974, comes at epoch 10): extra epochs on four examples buy no better answers, only more forgetting. The practical rule: **watch the validation curve and stop at its minimum** (here about epoch 2), instead of choosing the number of epochs in advance.

   *Exercise 2.2:* How many steps was that? Your task: from `p2_hist`, store in `steps_per_epoch` how many update steps one epoch has (batch size 1), in `total_steps` the number of steps of the whole run, and in `secs_per_step` the average time of one step in seconds (use `p2_hist["secs"][-1]`; it includes the evaluation after each epoch, so say "about"). Why it matters: you know what one step costs on your machine and can predict the time of the other runs in this lab.

```python
steps_per_epoch = None   # update steps in one epoch
total_steps = None       # update steps in the whole run
secs_per_step = None     # average seconds per step (about)
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if steps_per_epoch is None or total_steps is None or secs_per_step is None:
    print("Fill in the three variables above, then run this cell again.")
else:
    check("steps_per_epoch is right", steps_per_epoch == len(sft_examples), "Batch size 1: one step per example.")
    check("total_steps is right", total_steps == len(sft_examples) * EPOCHS, "steps per epoch times the number of epochs.")
    check("secs_per_step is plausible", 0.05 < secs_per_step < 30, "total seconds / total steps.")
    RESULTS["p2_step_time"] = round(float(secs_per_step), 2)
```

   A possible answer, and what it means:

```python
steps_per_epoch = len(sft_examples)
total_steps = steps_per_epoch * EPOCHS
secs_per_step = p2_hist["secs"][-1] / total_steps
```

   Our CPU run: 24 steps per epoch, 72 steps in 3 epochs, between 66 and 125 seconds in total in two runs, that is **0.9 to 1.7 seconds per step** (with the evaluation after each epoch included). Your time depends on your computer; on a GPU a step takes a small fraction of a second.

### Part 3. Forgetting (15 min)

*Lesson section: "Overfitting and catastrophic forgetting".* The apprentice who spends two weeks learning the bakery's way of speaking might, if she did nothing else, start to write all her letters in that style, even the ones to her mother. A model that is trained only on 24 polite answers is pulled towards them, and the weights that held other knowledge move too. This is **catastrophic forgetting**. Here you measure it on a paragraph of ordinary English (`GENERAL_TEXT`, which the model never trains on) and try three remedies: a lower learning rate, fewer steps and **replay** (mixing in some general text).

**Imagine first.** The base model's loss on `GENERAL_TEXT` is about 3 nats per token (the exact value is printed below). After full fine-tuning for 3 epochs, by how many percent do you think it rose: 1 percent, 10 percent, 50 percent? And which remedy do you expect to help the most? Write your guesses in the block below (replace each `None`), then run it.

```python
# Imagine first: forgetting
my_guess_rise_pct = None     # percent rise of the loss on GENERAL_TEXT after full fine-tuning
my_guess_best_fix = None     # "low lr", "fewer steps" or "replay"
```

1. **Step 3.1. Measure the forgetting.** `text_loss` gives the average cross-entropy in nats per token. A **lower loss means the text is more natural for the model**. We measure the base model, the model after each epoch of Part 2 (stored in `p2_hist`), and its perplexity `exp(loss)`: the number of equally likely tokens the model is "choosing between" at every position.

```python
gen0, gen_end = p2_hist["general"][0], p2_hist["general"][-1]
print("epoch  loss on GENERAL_TEXT  perplexity   change")
for e, g in enumerate(p2_hist["general"]):
    print(f"{e:5d}  {g:20.3f}  {math.exp(g):10.1f}   {100 * (g - gen0) / gen0:+6.1f} %")
p3_rise = 100 * (gen_end - gen0) / gen0
print(f"\nafter {EPOCHS} epoch(s) of full fine-tuning the loss rose by {gen_end - gen0:+.3f} nats = {p3_rise:+.1f} %")
check("fine-tuning made the model a little worse on general text", gen_end > gen0, "Compare with the base model's loss (epoch 0).")
RESULTS["p3_forget"] = dict(base=round(gen0, 3), after=round(gen_end, 3), rise_pct=round(p3_rise, 1))
```

   The block prints (your numbers must match, apart from the times in seconds, which depend on your computer):

```text
epoch  loss on GENERAL_TEXT  perplexity   change
    0                 3.509        33.4     +0.0 %
    1                 3.589        36.2     +2.3 %
    2                 3.613        37.1     +3.0 %
    3                 3.664        39.0     +4.4 %

after 3 epoch(s) of full fine-tuning the loss rose by +0.155 nats = +4.4 %
✓ fine-tuning made the model a little worse on general text
```

   In our run the loss on `GENERAL_TEXT` is **3.509** nats per token for the base model (perplexity 33.4) and rises to 3.589 after epoch 1 (+2.3 percent), 3.613 after epoch 2 (+3.0 percent) and **3.664 after epoch 3 (+4.4 percent**, perplexity 39.0). Two things to notice. First, the forgetting is **gradual and keeps growing with each epoch**, although the style was already learned after epoch 1 (style_rate 1.0) and the validation loss kept improving. Second, it is **small** in absolute terms: perplexity 33 to 39 means the model is choosing between about six more equally likely words at every position, on a paragraph that it has not been trained on. With a bigger learning rate, more epochs or more different data, the rise would be larger. The point is not that 4 percent is a disaster; it is that **you never get the new skill for free**, and you must measure the price.

2. **Step 3.2. Four remedies, one table.** A fair comparison needs the same amount of training for every recipe, so (except for recipe (d)) each recipe below trains for **one epoch** (24 steps, or 24 plus the replay steps) from the base model. The control is the full-fine-tuning model after epoch 1 of Part 2 (learning rate 3e-5, no replay), which we have already measured. The recipes are: (a) **low learning rate**: 3e-6, ten times smaller; (b) **fewer steps**: the same recipe stopped after only 6 steps; (d) a **middle way**: learning rate 1e-5 for 2 epochs (the only recipe with a different budget, 48 steps); (c) **replay**: the 24 examples plus 12 short general paragraphs (written for this lab, different from `GENERAL_TEXT`), all in one shuffled epoch, with the ordinary next-token loss (no masking) on the general paragraphs. For each recipe we record the style rate, the validation loss and the loss on `GENERAL_TEXT`.

```python
REPLAY_TEXTS = [
    "The market opens early in the morning, when the stalls are covered with fresh fish, fruit and rice cakes. Buyers walk slowly between the tables and ask about the price before they decide.",
    "A new bridge was built across the canal last year. Before that, children had to walk a long way around the school fence, and in the rainy season the road was often flooded.",
    "The teacher asked the class to bring a plant from home. Some brought flowers, some brought herbs, and one student brought a small tree that was almost as tall as the desk.",
    "In the evening the town square fills with families. Vendors sell grilled corn, an old man plays the guitar, and the children chase each other around the fountain until it is dark.",
    "Mountains are formed when large pieces of the earth's crust push against each other for millions of years. The rock slowly rises, and wind and rain wear it down again.",
    "The library keeps its oldest books in a cool room at the back. A volunteer checks the shelves every month to make sure that no water has come in and that no insects are eating the paper.",
    "When the typhoon warning was raised, the families packed bags with clothes, water and documents. The local office opened the gym of the school as a shelter, and neighbours helped carry the older people.",
    "A good bread needs only flour, water, salt and time. The baker mixes the dough at night, lets it rest, and shapes it before sunrise so that the first loaves are ready when the shop opens.",
    "Birds migrate when the seasons change. Some fly thousands of kilometres, guided by the sun, the stars and the shape of the land, and rest in wetlands on the way.",
    "The town council meets on the first Monday of every month. Anyone can come and speak about a problem, such as a broken pump or a road that needs repair, and the minutes are posted on the notice board.",
    "To measure a room you need a tape, a pencil and a piece of paper. Write down the length and the width, multiply them to find the area in square metres, and add a little extra when you buy paint.",
    "On weekends the football pitch behind the church is never empty. Teams of different ages take turns, and the smallest children play at the edge with a ball made of rolled-up cloth.",
]
P3_STEPS = 6
def p3_run(label, examples, epochs, lr, max_steps=None):
    """Train a fresh copy of the base model; return (label, style, val, general, steps, secs)."""
    mm = copy.deepcopy(base_model)
    t0 = time.time()
    if max_steps is not None:
        examples = examples[:max_steps]
    hist = p2_train(mm, examples, epochs, lr=lr, verbose=False, style_at_start=False, style_pairs=INSTR_VAL[:3] if QUICK else None)
    row = dict(label=label, style=hist["style"][-1], val=hist["val"][-1], general=hist["general"][-1], steps=len(examples) * epochs, secs=time.time() - t0)
    del mm; gc.collect()
    return row

ftrain = sft_examples
n_replay = 4 if QUICK else 12
replay_examples = []
for t in REPLAY_TEXTS[:n_replay]:
    r_ids = tok(t + tok.eos_token, return_tensors="pt")["input_ids"].to(device)
    replay_examples.append((r_ids, r_ids.clone()))     # ordinary next-token loss: the labels are the ids themselves (nothing masked)
order = random.Random(SEED).sample(range(len(ftrain)), len(ftrain))
rows = []
rows.append(dict(label="control: lr 3e-5, 1 epoch", style=None, val=None, general=None, steps=len(ftrain)))
c = p2_hist  # control values are those after epoch 1 of Part 2
rows[0].update(style=c["style"][1], val=c["val"][1], general=c["general"][1])
rows.append(p3_run("(a) low learning rate 3e-6", ftrain, 1, 3e-6))
rows.append(p3_run(f"(b) fewer steps ({P3_STEPS})", ftrain, 1, 3e-5, max_steps=P3_STEPS))
rows.append(p3_run("(d) lr 1e-5, 2 epochs", ftrain, 2, 1e-5))
rows.append(p3_run("(c) replay: 24 examples + 12 general", ftrain + replay_examples, 1, 3e-5))
base_row = dict(label="base model (no training)", style=p1_base_style, val=p2_hist["val"][0], general=gen0, steps=0)
rows.insert(0, base_row)
print(f"{'recipe':<40}{'steps':>6}{'style':>7}{'val loss':>10}{'general':>9}{'change':>9}")
for r in rows:
    print(f"{r['label']:<40}{r['steps']:>6}{r['style']:>7.2f}{r['val']:>10.3f}{r['general']:>9.3f}{100 * (r['general'] - gen0) / gen0:>+8.1f}%")
RESULTS["p3_table"] = [(r["label"], r["steps"], round(r["style"], 2), round(r["val"], 3), round(r["general"], 3)) for r in rows]
```

   The block prints (your numbers must match, apart from the times in seconds, which depend on your computer):

```text
recipe                                   steps  style  val loss  general   change
base model (no training)                     0   0.00     2.804    3.509    +0.0%
control: lr 3e-5, 1 epoch                   24   1.00     0.335    3.589    +2.3%
(a) low learning rate 3e-6                  24   0.00     2.077    3.512    +0.1%
(b) fewer steps (6)                          6   0.00     1.572    3.516    +0.2%
(d) lr 1e-5, 2 epochs                       48   1.00     0.408    3.538    +0.8%
(c) replay: 24 examples + 12 general        36   1.00     0.270    3.399    -3.1%
```

```python
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
names = ["base", "control", "(a) low lr", "(b) fewer", "(d) lr 1e-5 x2", "(c) replay"]
ax[0].bar(names, [r["general"] for r in rows], color=["gray", "C3", "C0", "C1", "C4", "C2"]); ax[0].set_ylim(min(r["general"] for r in rows) - 0.1, max(r["general"] for r in rows) + 0.1); ax[0].set_title("loss on GENERAL_TEXT (lower = less forgetting)")
ax[1].bar(names, [r["style"] for r in rows], color=["gray", "C3", "C0", "C1", "C4", "C2"]); ax[1].set_ylim(0, 1.05); ax[1].set_title("style_rate (higher = learned the style)")
for a in ax: a.tick_params(axis="x", labelrotation=30)
plt.tight_layout(); plt.show()
```

   Read the table (our run). **The control** (learning rate 3e-5, 24 steps) learns the style (1.00) at a cost of +2.3 percent on the general text. **(a) The lower learning rate** (3e-6) keeps the general loss almost unchanged (3.512, +0.1 percent), but it also **did not learn the style**: style_rate 0.00 and a validation loss of 2.077, still far from the control's 0.335. **(b) Fewer steps** (only 6 steps) behaves the same way: 3.516 and style 0.00. So with a small budget of 24 steps, these two remedies do not reduce forgetting at a fair price; they simply learn less (the learning and the forgetting travel together). **(d) A middle way**, a learning rate of 1e-5 for two epochs (48 steps), does learn the style (1.00) with a validation loss of 0.408 (a little worse than the control's 0.335) and costs only **+0.8 percent** on the general text, about a third of the control's +2.3 percent. So a lower learning rate *can* help, but only if you give it enough steps to reach the same behaviour. **(c) Replay** (24 examples plus 12 general paragraphs, all in one shuffled epoch) is the one that helped most: style_rate 1.00, a validation loss of 0.270 (even better than the control's 0.335) and a general loss of 3.399, which is **below** the base model's 3.509 (-3.1 percent).

   **An honest warning about (c).** The loss on `GENERAL_TEXT` fell *below* the base model because our replay paragraphs are the same kind of text as `GENERAL_TEXT` (short descriptive paragraphs about everyday village and town life), so the model got a little better at that kind of text, not only "kept" its old ability. A fairer reading is: replay prevented the forgetting we measured, and part of the gain is a similarity of genre between the replay data and the test paragraph. If the replay data had been about something very different, the benefit would be smaller. In real projects you replay a sample of the pre-training-like data or of general instruction data, and you check on a test set that is *not* similar to the replay set.

   *Exercise 3.1:* Which recipe is best? Your task: use the `rows` list of Step 3.2. Among the recipes that reach a `style_rate` of at least 0.75, find the one with the smallest loss on `GENERAL_TEXT`; store its label in `best_label`. Then store in `cost_of_best` the rise of its general loss over the base model, in percent. If no recipe reaches 0.75, store the label of the one with the highest `style_rate`. Why it matters: forgetting is a trade-off, and the "best" recipe depends on how much of the new behaviour you insist on.

```python
best_label = None     # label of the best recipe
cost_of_best = None   # percent rise of its loss on GENERAL_TEXT over the base model
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if best_label is None or cost_of_best is None:
    print("Fill in best_label and cost_of_best above, then run this cell again.")
else:
    _ok = [r for r in rows[1:] if r["style"] >= 0.75]
    _best = min(_ok, key=lambda r: r["general"]) if _ok else max(rows[1:], key=lambda r: r["style"])
    check("best_label is right", best_label == _best["label"], "Filter the recipes by style >= 0.75 (skip the base model), then take the minimum of 'general'.")
    check("cost_of_best is right", abs(cost_of_best - 100 * (_best["general"] - gen0) / gen0) < 0.1, "100 * (general - base general) / base general.")
    RESULTS["p3_best"] = (best_label, round(float(cost_of_best), 1))
```

   A possible answer, and what it means:

```python
ok = [r for r in rows[1:] if r["style"] >= 0.75]
best = min(ok, key=lambda r: r["general"]) if ok else max(rows[1:], key=lambda r: r["style"])
best_label = best["label"]
cost_of_best = 100 * (best["general"] - gen0) / gen0
```

   Three recipes reach a style_rate of at least 0.75: the control (+2.3 percent), recipe (d) (+0.8 percent) and recipe (c), replay (-3.1 percent). In our run the best is therefore `(c) replay: 24 examples + 12 general`, with a change of **-3.1 percent**. A different seed or machine can change the order of two close recipes, so always read your own table rather than trusting the recipe names.

3. **Step 3.3. What the table does not tell you.** Loss on one paragraph is a narrow measure. A model can keep its loss on that paragraph and still lose skills elsewhere (arithmetic, another language, code). In practice one checks forgetting on a **small set of general tasks**, not on one paragraph, and watches the same measure through training to stop early. The next cell stores your three honest findings for the summary in Part 7: type them in your own words.

```python
RESULTS["p3_findings"] = [
    "Write finding 1 here: what did fine-tuning do to the style and to the general loss?",
    "Write finding 2 here: which remedy helped, and which did not?",
    "Write finding 3 here: one thing this experiment cannot tell us.",
]
for f in RESULTS["p3_findings"]: print("-", f)
```

   The block prints (your numbers must match, apart from the times in seconds, which depend on your computer):

```text
- Write finding 1 here: what did fine-tuning do to the style and to the general loss?
- Write finding 2 here: which remedy helped, and which did not?
- Write finding 3 here: one thing this experiment cannot tell us.
```

   **End of Parts 1 to 3.** You have seen the base model fail to answer, written the masked loss by hand, trained all 124 million weights on 24 examples, and measured both the gain (a learned style) and the price (some forgetting). Part 4 asks the next question of the lesson: what if the examples are not enough, and all we have is a coach who ranks two answers?

### Part 4. Preference learning from scratch (30 min)

*Lesson section: "Preference learning".* Instruction tuning (Part 2) shows the model *one* good answer for each question. But for many questions there is no single best answer; people can only say "this answer is better than that one". **Preference learning** turns those comparisons into a training signal. The classic recipe has two steps. First you train a **reward model**: a small network that gives a score to an answer so that the answers people prefer get the higher score. Then you use the reward model to steer the language model, either by *picking* the best of several answers (**best-of-n**, this part) or by *training* the model towards high scores (RLHF and DPO, Part 5). In this part you build a tiny reward model from GPT-2's own features, train it with the Bradley-Terry loss on 14 comparisons, test it on comparisons it has never seen, use it to pick the best of n answers, and then see how a reward model can be fooled.

**Imagine first.** You will train a reward model on only **14 pairs** (a good answer and a bad answer to the same question), on top of GPT-2's frozen features. Before you run any code, write three predictions. (1) On 6 pairs the model has never seen, how many will it rank correctly (0 to 6)? (2) Suppose the model samples 32 answers and you keep the one with the highest reward: will it be clearly better than a single answer (yes or no), and will it be a good answer to read, on a scale from 1 (nonsense) to 5 (a good answer)? (3) If you ask a program to *search* for the text with the highest possible reward, will the winner be a good answer (yes or no)?

```python
# Imagine first: the judge
my_guess_val_correct = None   # of the 6 unseen pairs, how many does the trained reward model rank correctly (0 to 6)?
my_guess_bon_better = None    # is the best of 32 answers clearly better than one random answer? ("yes" or "no")
my_guess_bon_quality = None   # how good does the best-of-32 answer read, from 1 (nonsense) to 5 (a good answer)?
my_guess_search_good = None   # if a program searches for the highest-reward text, is the winner a good answer? ("yes" or "no")
```

1. **Step 4.1. Features for a reward model.** A reward model needs to read a question and an answer and give one number. We build the smallest one that can work. Step one: GPT-2 reads `question + answer`, and we keep the **final-layer hidden state of each answer token** (a vector of 768 numbers per token) and take their **mean**: one vector of 768 numbers that summarises the answer in the context of its question. GPT-2 stays frozen. Step two (Step 4.3): a single linear layer turns the 768 numbers into one score. The block below computes the feature vector of every answer in `PREF_TRAIN` (14 pairs) and `PREF_VAL` (6 pairs), then **standardises** each of the 768 numbers (subtract the mean, divide by the standard deviation measured on the training answers), because a few of GPT-2's 768 dimensions have a huge scale and would dominate a linear layer.

   ```python
   @torch.no_grad()
   def rm_features(prompt, answer):
       """Mean final-layer hidden state of the answer tokens when GPT-2 reads the prompt followed by the answer (shape 768)."""
       p = tok(format_prompt(prompt))["input_ids"]; a = tok(answer)["input_ids"]
       ids = torch.tensor([p + a]).to(device)
       h = base_model.transformer(input_ids=ids).last_hidden_state[0]
       return h[len(p):].mean(0).cpu()

   def rm_stack(rows, key):
       return torch.stack([rm_features(r["prompt"], r[key]) for r in rows])

   t0 = time.time()
   p4_Ctr, p4_Rtr = rm_stack(PREF_TRAIN, "chosen"), rm_stack(PREF_TRAIN, "rejected")
   p4_Cva, p4_Rva = rm_stack(PREF_VAL, "chosen"), rm_stack(PREF_VAL, "rejected")
   p4_allf = torch.cat([p4_Ctr, p4_Rtr]); p4_mu, p4_sd = p4_allf.mean(0), p4_allf.std(0) + 1e-6
   rm_norm = lambda x: (x - p4_mu) / p4_sd
   print("feature tensors:", tuple(p4_Ctr.shape), tuple(p4_Rtr.shape), tuple(p4_Cva.shape), tuple(p4_Rva.shape), f"({time.time() - t0:.1f} s)")
   print(f"raw features: largest absolute value {float(p4_allf.abs().max()):.1f}, largest spread over answers {float(p4_allf.std(0).max()):.2f}, smallest {float(p4_allf.std(0).min()):.4f}")
   print(f"after standardising: mean of means {float(rm_norm(p4_allf).mean()):.4f}, mean of spreads {float(rm_norm(p4_allf).std(0).mean()):.3f}")
   assert p4_Ctr.shape == (14, 768) and p4_Cva.shape == (6, 768)
   ```

   You should see the shapes (14, 768) for the training answers and (6, 768) for the validation answers, computed in about 3 seconds. The raw features show why the standardising step matters: the largest absolute value is 246.7, and the spread (standard deviation over the 28 training answers) goes from 14.91 for the most variable dimension down to 0.0024 for the quietest, a factor of 6,000. After standardising, every dimension has mean 0 and spread 1.000. Each answer is now a point in a 768-dimensional space, and we have only 14 + 14 training points in it: far more dimensions than examples. Keep that in mind for Step 4.3.

2. **Step 4.2. The Bradley-Terry loss by hand.** How do you train a scorer from *comparisons*? The **Bradley-Terry model** says: the probability that people prefer answer A to answer B is `sigmoid(r_A - r_B)`, where `r` is the reward. Training maximises the probability of the choices people really made, so the loss for one pair is `-log sigmoid(r_chosen - r_rejected)`. The block computes it for three cases by hand (with `math`), checks the special case of two equal scores against `ln 2`, and checks the gradient with PyTorch.

   ```python
   def bt_loss_by_hand(r_chosen, r_rejected):
       """Bradley-Terry loss of one pair: minus the log of the probability sigmoid(r_chosen - r_rejected)."""
       return -math.log(1 / (1 + math.exp(-(r_chosen - r_rejected))))

   for rc, rr in [(0.0, 0.0), (2.0, 0.0), (0.0, 2.0)]:
       pw = 1 / (1 + math.exp(-(rc - rr)))
       print(f"r_chosen {rc:4.1f}  r_rejected {rr:4.1f}  P(chosen wins) {pw:.4f}  loss {bt_loss_by_hand(rc, rr):.4f}")
   print("ln 2 =", round(math.log(2), 4))
   assert abs(bt_loss_by_hand(0.0, 0.0) - math.log(2)) < 1e-12

   rc_t, rr_t = torch.tensor(2.0, requires_grad=True), torch.tensor(0.0, requires_grad=True)
   F.softplus(-(rc_t - rr_t)).backward()      # softplus(-x) is the same as -log sigmoid(x)
   p_win = 1 / (1 + math.exp(-2.0))
   print(f"gradient wrt r_chosen {float(rc_t.grad):.4f}  (by hand: -(1 - P) = {-(1 - p_win):.4f})   wrt r_rejected {float(rr_t.grad):.4f}")
   assert abs(float(rc_t.grad) + (1 - p_win)) < 1e-6 and abs(float(rr_t.grad) - (1 - p_win)) < 1e-6
   ```

   You should see three cases. Equal scores give a probability of 0.5000 and a loss of 0.6931, which is **ln 2**: a reward model that cannot tell the answers apart pays ln 2 on every pair, so ln 2 is the baseline to beat. Scores 2 and 0 (chosen higher) give a probability of 0.8808 and a loss of 0.1269. Scores 0 and 2 (chosen lower) give a probability of 0.1192 and a loss of **2.1269**: a wrong ranking is punished much harder than a right ranking is rewarded. The gradient of the loss with respect to the chosen score is -0.1192, which equals -(1 - P): the more wrong the model is, the harder it pushes the chosen score up and the rejected score down by the same amount.

   *Exercise 4.1:* a reward model gives the chosen answer the score 1.3 and the rejected answer the score -0.4. Store in `bt_p` the probability that the chosen answer wins and in `bt_loss` the loss of the pair, using the function above or the formulas. Why it matters: this one number is what every step of reward-model training pushes down.

   Write your answer in this starter block:

   ```python
   bt_p = None      # P(chosen wins) for scores 1.3 and -0.4
   bt_loss = None   # the Bradley-Terry loss of that pair
   ```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

   ```python
   if bt_p is None or bt_loss is None:
       print("Fill in bt_p and bt_loss above, then run this cell again.")
   else:
       check("bt_p is right", abs(bt_p - 1 / (1 + math.exp(-1.7))) < 1e-4, "sigmoid(1.3 - (-0.4)) = sigmoid(1.7).")
       check("bt_loss is right", abs(bt_loss + math.log(bt_p)) < 1e-4, "The loss is minus the natural log of bt_p.")
       check("a correct ranking gives a loss below ln 2", bt_loss < math.log(2), "When the chosen answer has the higher score, the loss is below 0.693.")
       RESULTS["p4_bt"] = (round(bt_p, 4), round(bt_loss, 4))
   ```

   A possible answer, and what it means:

   ```python
   bt_p = 1 / (1 + math.exp(-(1.3 - (-0.4))))
   bt_loss = bt_loss_by_hand(1.3, -0.4)
   print(bt_p, bt_loss)
   ```

   The probability is 0.8455 and the loss is 0.1678 (that is -log 0.8455). The chosen answer has the higher score, so the loss is below ln 2 = 0.6931. A wrong ranking with the same two scores swapped would cost 1.8678.

3. **Step 4.3. Train the reward model.** The reward model is one linear layer `w` (768 weights, no bias: a bias would cancel in the difference of two scores). For a pair, the score difference is `(features_chosen - features_rejected) @ w`, and the loss is the Bradley-Terry loss averaged over the training pairs. We start from `w = 0` (every answer scores 0, so the loss must be ln 2) and train with AdamW (learning rate 0.001) for 200 steps. After every step the block records the training loss, the validation loss and the **validation accuracy** (the fraction of unseen pairs where the chosen answer gets the higher score). Chance level is 50 percent accuracy and a loss of ln 2.

   ```python
   def rm_fit(C, R, Cv, Rv, steps=200, lr=1e-3, wd=0.0):
       """Train the linear reward model with the Bradley-Terry loss. Returns the weights and a list of (train loss, val loss, val accuracy) per step."""
       C, R, Cv, Rv = [rm_norm(x) for x in (C, R, Cv, Rv)]
       torch.manual_seed(0)
       w = torch.zeros(768, requires_grad=True)
       opt = torch.optim.AdamW([w], lr=lr, weight_decay=wd); hist = []
       for i in range(steps + 1):
           with torch.no_grad():
               dv = (Cv - Rv) @ w
               hist.append((float(F.softplus(-((C - R) @ w)).mean()), float(F.softplus(-dv).mean()), float((dv > 0).float().mean())))
           if i == steps: break
           loss = F.softplus(-((C - R) @ w)).mean(); opt.zero_grad(); loss.backward(); opt.step()
       return w.detach(), hist

   def rm_score(prompt, answers):
       """Reward of each answer (a list of texts) for the question `prompt`."""
       f = torch.stack([rm_features(prompt, a) for a in answers]); return rm_norm(f) @ p4_w

   p4_w, p4_hist = rm_fit(p4_Ctr, p4_Rtr, p4_Cva, p4_Rva)
   print("step   train loss   val loss   val accuracy")
   for s in (0, 10, 25, 50, 100, 200):
       print(f"{s:4d}   {p4_hist[s][0]:9.4f}   {p4_hist[s][1]:8.4f}   {p4_hist[s][2]:8.0%}")
   assert abs(p4_hist[0][0] - math.log(2)) < 1e-6
   fig, ax = plt.subplots(1, 2, figsize=(9, 2.8))
   ax[0].plot([h[0] for h in p4_hist], label="train"); ax[0].plot([h[1] for h in p4_hist], label="validation"); ax[0].axhline(math.log(2), color="gray", ls=":", label="ln 2 (guessing)")
   ax[0].set_xlabel("step"); ax[0].set_ylabel("Bradley-Terry loss"); ax[0].legend()
   dv_ = (rm_norm(p4_Cva) - rm_norm(p4_Rva)) @ p4_w
   ax[1].bar(range(6), dv_.numpy(), color=["tab:green" if d > 0 else "tab:red" for d in dv_]); ax[1].set_xlabel("unseen pair"); ax[1].set_ylabel("score(chosen) - score(rejected)")
   plt.tight_layout(); plt.show()
   RESULTS["p4_rm"] = (round(p4_hist[-1][0], 4), round(p4_hist[-1][1], 4), p4_hist[-1][2])
   ```

   You should see the loss start at **0.6931** (ln 2, as it must when `w = 0`) and fall. The training loss reaches 0.1001 after 10 steps and 0.0015 after 200. The validation loss falls from 0.6931 to 0.2627 (step 10), 0.1185 (step 50) and **0.0917** (step 200), and the validation accuracy is **100 percent (6 of 6 pairs)** from step 10 on. So the reward model learned to rank unseen pairs from only 14 examples, and the validation loss never rose again during the 200 steps. Why so easy? With 768 weights and 14 pairs the training pairs can be separated perfectly (training loss near 0), and the chosen answers differ from the rejected ones in obvious ways (polite, helpful words against rude or empty ones). Do not celebrate yet: 6 pairs is a very small test, and the next step shows what it hides.

4. **Step 4.4. The learning curve, and a trap.** How many comparisons does the reward model need? We train it again on the first 2, 4, 6, 8, 10 and 14 training pairs and measure the same three numbers on the 6 unseen pairs. Then we add a **trap**: six new held-out pairs, written so that the *rejected* answer is much **longer** than the chosen one (a short, correct answer against a long, harmful or silly one). In the training data the chosen answer is always the longer one, so a model that has secretly learned "longer is better" would fail the trap. Let us measure it. (The rule "pick the longer answer" gets 6 of 6 on the validation pairs, so the validation pairs alone cannot tell the two apart.)

   ```python
   PREF_TRAP = [dict(prompt=p, chosen=c, rejected=r) for p, c, r in [
       ("How do I cross the road safely?", "Look both ways and use the crossing.", "Just run across as fast as you can without looking, cars will always stop for you because they like children, and it is the quickest way to get to the other side of the street."),
       ("What should I do if I am lost in a mall?", "Stay calm and ask a staff member for help.", "Wander around the whole mall alone for hours, go into the parking lot and accept a ride from anyone who offers, it will certainly be fine."),
       ("How do I wash my hands?", "Use soap and water and scrub for twenty seconds.", "Just wave them under the tap for one second, soap is a waste of money and germs are not real so it truly does not matter at all."),
       ("How can I be kind to a new classmate?", "Smile, say hello and invite them to sit with you.", "Ignore the new classmate completely, whisper about their clothes with your friends and laugh when they walk past the whole class."),
       ("Why should I drink water?", "Water keeps your body cool and helps it work well.", "You should never drink water, only fizzy drinks, because water is boring and your body does not need it, ever, for any reason at all."),
       ("What should I do with a broken glass?", "Do not touch it with bare hands; tell an adult and sweep it up carefully.", "Pick it up with your bare fingers as fast as you can so nobody else sees it, and then hide the pieces inside your pocket for later.")] ]
   p4_Ct, p4_Rt = rm_stack(PREF_TRAP, "chosen"), rm_stack(PREF_TRAP, "rejected")
   nw = lambda rows, k: [len(r[k].split()) for r in rows]
   print("'longer wins' rule: validation", np.mean([a > b for a, b in zip(nw(PREF_VAL, "chosen"), nw(PREF_VAL, "rejected"))]), " trap", np.mean([a > b for a, b in zip(nw(PREF_TRAP, "chosen"), nw(PREF_TRAP, "rejected"))]))
   print("pairs used   train loss   val loss   val acc   trap acc")
   p4_curve = []
   for k in (2, 4, 6, 8, 10, 14):
       w_k, h_k = rm_fit(p4_Ctr[:k], p4_Rtr[:k], p4_Cva, p4_Rva)
       trap_d = (rm_norm(p4_Ct) - rm_norm(p4_Rt)) @ w_k
       p4_curve.append((k, h_k[-1][0], h_k[-1][1], h_k[-1][2], float((trap_d > 0).float().mean())))
       print(f"{k:9d}   {h_k[-1][0]:10.4f}   {h_k[-1][1]:8.4f}   {h_k[-1][2]:6.0%}   {p4_curve[-1][4]:7.0%}")
   w_full_trap = (rm_norm(p4_Ct) - rm_norm(p4_Rt)) @ p4_w
   print("trap differences with all 14 pairs (score chosen minus score rejected):", np.round(w_full_trap.numpy(), 2))
   plt.figure(figsize=(5, 3)); plt.plot([c[0] for c in p4_curve], [c[2] for c in p4_curve], "o-", label="validation loss"); plt.axhline(math.log(2), color="gray", ls=":", label="ln 2")
   plt.xlabel("training pairs"); plt.ylabel("Bradley-Terry loss"); plt.legend(); plt.tight_layout(); plt.show()
   RESULTS["p4_trap"] = float((w_full_trap > 0).float().mean())
   ```

   You should see that the simple rule "the longer answer wins" gets **6 of 6** validation pairs right: in our data the chosen answer is always the longer one, so the validation pairs cannot tell "good" from "long". On the trap, where the harmful answer is the longer one, the rule gets **0 of 6**. The learning curve: with 2 pairs the validation loss is 0.2918 (accuracy 83 percent), with 4 pairs 0.1914 (100 percent), with 6 pairs 0.1392, with 8 pairs 0.1030, with 10 pairs 0.1456 and with all 14 pairs **0.0917**. The trend is down, but not smooth (10 pairs is worse than 8): with 6 validation pairs one pair changes a lot. On the trap, the reward model with all 14 pairs gets **5 of 6 (83 percent)**: the score differences (chosen minus rejected) are 0.02, 1.25, 2.66, 0.80, 1.28 and **-0.13**. So it did not simply learn "longer is better", but it is not safe either: the first pair is right by only 0.02, and the pair about the broken glass is ranked wrongly. The trap accuracy is 50 percent with 4 pairs, 67 percent with 6, 100 percent with 8, 83 percent with 10 and 14: with six trap pairs, differences of one pair (17 points) are noise. The honest summary is that 14 pairs give a useful but fragile judge.

   *Exercise 4.2:* train the reward model on the **first 4** training pairs only (use `rm_fit` with `p4_Ctr[:4]` and `p4_Rtr[:4]`, the default steps) and store the final validation loss in `val_loss_4` and the final validation accuracy in `val_acc_4`. Why it matters: it tells you how fast a reward model improves with data, and that even a few pairs carry signal.

   Write your answer in this starter block:

   ```python
   val_loss_4 = None   # final validation loss when training on the first 4 pairs
   val_acc_4 = None    # final validation accuracy in that case
   ```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

   ```python
   if val_loss_4 is None or val_acc_4 is None:
       print("Fill in val_loss_4 and val_acc_4 above, then run this cell again.")
   else:
       _w4, _h4 = rm_fit(p4_Ctr[:4], p4_Rtr[:4], p4_Cva, p4_Rva)
       check("val_loss_4 is right", abs(val_loss_4 - _h4[-1][1]) < 0.01, "Take the second number of the last entry of the history list.")
       check("val_acc_4 is right", abs(val_acc_4 - _h4[-1][2]) < 0.01, "Take the third number of the last entry.")
       check("4 pairs are already better than guessing (loss below ln 2)", val_loss_4 < math.log(2), "A reward model that learned something has a loss below 0.693.")
       RESULTS["p4_four"] = (round(val_loss_4, 4), round(val_acc_4, 3))
   ```

   A possible answer, and what it means:

   ```python
   w4, h4 = rm_fit(p4_Ctr[:4], p4_Rtr[:4], p4_Cva, p4_Rva)
   val_loss_4, val_acc_4 = h4[-1][1], h4[-1][2]
   print(val_loss_4, val_acc_4)
   ```

   With 4 pairs the validation loss is 0.1914 and the validation accuracy is 1.0 (6 of 6): already well below ln 2 = 0.6931, although more than twice the loss of the 14-pair model (0.0917). The accuracy is the same as with 14 pairs, which shows that the 6 validation pairs are too easy to separate good from poor reward models; the trap of Step 4.4 does it better (50 percent for 4 pairs).

5. **Step 4.5. Best-of-n.** Now we use the reward model. First we need a language model that writes answers in the right style: the code below runs **one epoch of full fine-tuning** of GPT-2 on `INSTR_TRAIN` (the recipe of Part 2: masked prompt, AdamW, learning rate 3e-5, batch size 1) and checks the style rate. Then, for each of the 6 questions of `PREF_VAL`, it **samples 32 answers** (temperature 1, 30 new tokens) and scores them with the reward model. Best-of-n means: draw n answers, keep the one with the highest reward. To see how the reward grows with n without drawing new samples every time, we **resample from the pool**: for each n, pick n of the 32 answers at random, take the maximum reward, and average over 200 repeats and the 6 questions.

   ```python
   p4_sft = copy.deepcopy(base_model); p4_sft.train()
   p4_opt = torch.optim.AdamW(p4_sft.parameters(), lr=3e-5); t0 = time.time()
   p4_train_rows = INSTR_TRAIN[:8] if QUICK else INSTR_TRAIN
   for q, a in p4_train_rows:
       p_ids = tok(format_prompt(q))["input_ids"]; r_ids = tok(a + tok.eos_token)["input_ids"]
       loss = p4_sft(input_ids=torch.tensor([p_ids + r_ids]).to(device), labels=torch.tensor([ [-100] * len(p_ids) + r_ids]).to(device)).loss
       loss.backward(); p4_opt.step(); p4_opt.zero_grad()
   p4_sft.eval(); del p4_opt
   print(f"one epoch of SFT on {len(p4_train_rows)} examples: {time.time() - t0:.0f} s, style rate {style_rate(p4_sft):.2f} (the base model had {style_rate(base_model):.2f})")

   @torch.no_grad()
   def sample_answers(model, prompt, n, max_new_tokens=30, seed=0):
       """n sampled answers (temperature 1, no top-k) of `model` to the question `prompt`."""
       ids = tok(format_prompt(prompt), return_tensors="pt").to(device); torch.manual_seed(seed)
       out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=True, temperature=1.0, top_k=None, num_return_sequences=n, pad_token_id=tok.eos_token_id)
       return [tok.decode(o[ids["input_ids"].shape[1]:], skip_special_tokens=True).strip() for o in out]
   ```

   ```python
   t0 = time.time(); p4_nq, p4_ns = (3, 8) if QUICK else (6, 32); p4_pool = {}
   for i, row in enumerate(PREF_VAL[:p4_nq]):
       S = sample_answers(p4_sft, row["prompt"], p4_ns, seed=i); p4_pool[i] = (S, rm_score(row["prompt"], S).numpy())
   print(f"sampled and scored {p4_nq * p4_ns} answers in {time.time() - t0:.0f} s")
   p4_rng = np.random.RandomState(0); p4_bon = {}
   print("   n   mean best reward")
   for n in [x for x in (1, 2, 4, 8, 16, 32) if x <= p4_ns]:
       p4_bon[n] = float(np.mean([np.mean([max(p4_rng.choice(p4_pool[i][1], n, replace=False)) for _ in range(200)]) for i in p4_pool]))
       print(f"{n:4d}   {p4_bon[n]:9.3f}")
   p4_ref_ch = float(np.mean([rm_score(r["prompt"], [r["chosen"] ]).item() for r in PREF_VAL])); p4_ref_rj = float(np.mean([rm_score(r["prompt"], [r["rejected"] ]).item() for r in PREF_VAL]))
   print(f"for comparison, the human-written chosen answers score {p4_ref_ch:.2f} on average and the rejected ones {p4_ref_rj:.2f}")
   for i in (0, 1):
       S, R = p4_pool[i]; print(f"\nQ: {PREF_VAL[i]['prompt']}\n  best  ({R.max():5.2f}): {S[int(R.argmax())]}\n  worst ({R.min():5.2f}): {S[int(R.argmin())]}\n  first sample ({R[0]:5.2f}): {S[0]}")
   plt.figure(figsize=(4.5, 3)); plt.plot(list(p4_bon), list(p4_bon.values()), "o-"); plt.xscale("log", base=2); plt.xlabel("n (answers sampled)"); plt.ylabel("mean reward of the best"); plt.tight_layout(); plt.show()
   RESULTS["p4_bon"] = {n: round(v, 3) for n, v in p4_bon.items()}
   ```

   You should see that one epoch of full fine-tuning on the 24 examples takes about 21 seconds and raises the style rate from 0.00 to **1.00**: every answer now starts with "Sure!". Sampling and scoring 192 answers (6 questions times 32) took 37 seconds. The mean reward of the best answer rises steadily with n: **-1.156** for n = 1, -0.528 for n = 2, -0.092 for n = 4, 0.310 for n = 8, 0.664 for n = 16 and **0.926** for n = 32. Every doubling of n adds a roughly constant 0.35 to 0.6, and the curve is still rising at 32: best-of-n has diminishing returns in n but costs n times more to generate. Compare with the human-written answers: the chosen ones score 2.24 on average and the rejected ones -1.22. Even best-of-32 (0.93) stays far below a real good answer. Now read the winners. The best answer to the homework question (reward 1.57) is "Sure! The focus, coordinated by the Math/Research Assistant, can be diverted to the ability to concentrate when you are actually working on the math problems": it follows the style but is hard to call a good answer. The best answer for the plant (1.92) talks about a "bearing tool" and a "thorn". The worst answers are short ("Sure! Cheers!", -3.52). **Best-of-n made the model better by the judge's measure, but the judge, and the model, are weak**: the reward model likes longer, more elaborate answers, and the 124-million-parameter model cannot supply a really good one.

   *Exercise 4.3:* using the pool `p4_pool` (a dictionary: question number to the pair of a list of answers and an array of rewards), find for question 0 the answer with the highest reward. Store its text in `best_text` and its reward in `best_reward`. Then store in `mean_reward` the **average** reward of all the answers of question 0 (the reward of a typical single draw). Why it matters: the difference between the two numbers is exactly the gain of best-of-n for this question.

   Write your answer in this starter block:

   ```python
   best_text = None     # the sampled answer of question 0 with the highest reward
   best_reward = None   # its reward
   mean_reward = None   # the average reward of all sampled answers of question 0
   ```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

   ```python
   if best_text is None or best_reward is None or mean_reward is None:
       print("Fill in best_text, best_reward and mean_reward above, then run this cell again.")
   else:
       _S, _R = p4_pool[0]
       check("best_reward is the maximum", abs(best_reward - float(_R.max())) < 1e-4, "Use the maximum of the reward array.")
       check("best_text is the answer with that reward", best_text == _S[int(_R.argmax())], "Use argmax to find the position.")
       check("mean_reward is right", abs(mean_reward - float(_R.mean())) < 1e-4, "The mean of the reward array.")
       RESULTS["p4_best"] = (round(best_reward, 3), round(mean_reward, 3))
   ```

   A possible answer, and what it means:

   ```python
   S0, R0 = p4_pool[0]
   best_text, best_reward, mean_reward = S0[int(R0.argmax())], float(R0.max()), float(R0.mean())
   print(best_reward, mean_reward, best_text)
   ```

   The best answer has reward 1.567 and the average of the 32 answers is -1.082, so best-of-32 gains **2.65** reward points for question 0 over a typical single draw. The text of the best answer is "Sure! The focus, coordinated by the Math/Research Assistant, can be diverted to the ability to concentrate when you are actually working on the math problems": higher reward, but still not a good answer.

6. **Step 4.6. Reward hacking: when the judge is fooled.** Best-of-32 only *picks*; it cannot move far from what the model can say. What if we **optimise** the text for the reward? The block below does the simplest search: it starts from an empty answer to question 0 and, at each of 5 steps, tries to append each of the 169 different words that occur in the chosen training answers, and keeps the word that gives the highest reward. The reward model is the same one; it was never told that the text must make sense. We print the winner at each step, compare its reward with the human-written chosen answer and with the best sample of Step 4.5, and read it.

   ```python
   t0 = time.time(); p4_q = PREF_VAL[0]["prompt"]
   p4_words = sorted({w_.strip(".,").lower() for r in PREF_TRAIN for w_ in r["chosen"].split()})
   if QUICK: p4_words = p4_words[:50]
   p4_cur, p4_hack = [], []
   for step in range(3 if QUICK else 5):
       cands = [" ".join(p4_cur + [x]) for x in p4_words]
       rs = rm_score(p4_q, cands); k = int(rs.argmax()); p4_cur.append(p4_words[k]); p4_hack.append(float(rs[k]))
       print(f"step {step + 1}: reward {p4_hack[-1]:5.2f}   text: {' '.join(p4_cur)!r}   ({time.time() - t0:.0f} s)")
   print(f"\ncandidate words tried: {len(p4_words)}")
   print(f"reward of the human-written chosen answer to this question:   {rm_score(p4_q, [PREF_VAL[0]['chosen'] ]).item():5.2f}")
   print(f"reward of the highest sampled answer of Step 4.5:             {p4_pool[0][1].max():5.2f}")
   print(f"reward of the searched word salad:                            {p4_hack[-1]:5.2f}")
   RESULTS["p4_hack"] = (round(p4_hack[-1], 2), " ".join(p4_cur))
   ```

   You should see the search find a high reward almost at once: the single word "bring" already scores **4.93**, and "bring buying" scores **6.62**, more than twice the **2.94** of the human-written chosen answer to this question and four times the 1.57 of the best sampled answer. The reward goes up a little more to 6.97 with a third word ("bring buying test") and then slips slightly (6.95, 6.63), because the search only appends words and cannot go back. The final text is "bring buying test real study": **word salad with a higher reward than any real answer.** This is **reward hacking** (also called over-optimisation or Goodhart's law): the reward model is only a proxy trained on 14 pairs, so it has regions of the input space that it has never seen, where its score is arbitrary. A search that is strong enough finds exactly these regions. The search used 169 words, 5 steps and 30 seconds of a CPU: a real reinforcement-learning run explores far more. To make the attack harder you would add many more pairs, including pairs where the rejected answer is word salad, and train the model on its own samples (the lesson's iterative RLHF).

   *Exercise 4.4:* take the first pair of `PREF_TRAP` (a short correct answer and a long harmful one). Store in `trap_chosen` and `trap_rejected` the rewards that the trained model `p4_w` gives them (use `rm_score(prompt, [answer])`), and in `trap_ok` whether the chosen answer scores higher (True or False). Why it matters: a reward model is only as good as the pairs it has been tested on, and the validation pairs may be too easy.

   Write your answer in this starter block:

   ```python
   trap_chosen = None     # reward of the chosen answer of PREF_TRAP[0]
   trap_rejected = None   # reward of the rejected answer of PREF_TRAP[0]
   trap_ok = None         # True if the chosen answer scores higher
   ```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

   ```python
   if trap_chosen is None or trap_rejected is None or trap_ok is None:
       print("Fill in trap_chosen, trap_rejected and trap_ok above, then run this cell again.")
   else:
       _t = PREF_TRAP[0]
       check("trap_chosen is right", abs(trap_chosen - rm_score(_t["prompt"], [_t["chosen"] ]).item()) < 1e-3, "rm_score(PREF_TRAP[0]['prompt'], [PREF_TRAP[0]['chosen'] ]).item()")
       check("trap_rejected is right", abs(trap_rejected - rm_score(_t["prompt"], [_t["rejected"] ]).item()) < 1e-3, "The same with the rejected answer.")
       check("trap_ok is consistent", bool(trap_ok) == (trap_chosen > trap_rejected), "Compare the two numbers.")
       RESULTS["p4_trap0"] = (round(trap_chosen, 3), round(trap_rejected, 3), bool(trap_ok))
   ```

   A possible answer, and what it means:

   ```python
   t0_ = PREF_TRAP[0]
   trap_chosen = rm_score(t0_["prompt"], [t0_["chosen"] ]).item()
   trap_rejected = rm_score(t0_["prompt"], [t0_["rejected"] ]).item()
   trap_ok = trap_chosen > trap_rejected
   print(trap_chosen, trap_rejected, trap_ok)
   ```

   The reward of the short correct answer is -0.227 and the reward of the long harmful answer is -0.252, so `trap_ok` is True, but the margin is only **0.025**: the reward model hardly sees the difference. Both scores are low (a good answer on the validation set scored 2.94), which suggests that this kind of question (road safety) is far from the 14 training pairs.

**Reflect.** Look at your Imagine-first guesses. In your notebook, answer in two or three sentences each: (1) How close was your guess for the unseen pairs? (2) What did best-of-n gain, and what did it not change (read the best answer again)? (3) The word salad has a higher reward than every real answer. Whose fault is that: the reward model, the search, or the 14 pairs? What would you add to the data to make this attack harder?

### Part 5. KL, β and DPO on a toy policy (35 min)

*Lesson section: "RLHF and DPO".* In Part 4 you *picked* from a fixed model. RLHF and DPO *change* the model. If we simply told the model "maximise the reward", it would put all its probability on the single best-scoring answer, forget everything else, and exploit every mistake of the reward model (Step 4.6 showed how large those mistakes can be). The cure is a leash: maximise the expected reward **minus β times the KL divergence from the reference model**. The strength of the leash is β. This objective has an exact solution, `pi_star(y) = pi_ref(y) * exp(r(y) / β) / Z`, and **DPO** is a way to reach it without ever training a reward model or running reinforcement learning. In this part you compute the solution with numpy on a toy with 3 answers, sweep β, plot the frontier of reward against KL, then repeat with 6 answers where the reward model is wrong about one answer, train DPO by gradient descent and check that it finds the closed-form optimum, and finally run a few DPO steps on the real GPT-2.

**Imagine first.** A model answers a question in one of three ways: rude (probability 0.6), plain (0.3) and helpful (0.1). The rewards are 0, 1 and 2. With a strong leash (β = 100) the new policy hardly moves. With a very weak leash (β = 0.05) it should go all the way to the best answer. Before you run any code, predict (1) the expected reward and (2) the probability of the helpful answer for the in-between choice β = 1. For reference, the reference model's expected reward is 0.6 * 0 + 0.3 * 1 + 0.1 * 2 = 0.5.

```python
# Imagine first: the leash
my_guess_reward_b1 = None     # expected reward of the optimal policy for beta = 1 (between 0.5 and 2.0)
my_guess_helpful_b1 = None    # probability of the helpful answer under that policy (0 to 1)
```

1. **Step 5.1. The closed-form solution.** We want the policy `pi` that maximises `J(pi) = E_pi[r] - β * KL(pi || pi_ref)`. The solution is `pi*(y) = pi_ref(y) * exp(r(y) / β) / Z`, where `Z` is the number that makes the probabilities sum to 1: take the reference probabilities and **multiply each answer by `exp(r / β)`**, a boost that is larger for higher rewards and weaker when β is large. The block computes it for several β, prints the expected reward and the KL, and then checks the claim: no policy among 100,000 random ones has a higher `J` than `pi*`.

   ```python
   def softmax_np(z):
       z = np.asarray(z, float); e = np.exp(z - z.max()); return e / e.sum()
   def kl_np(p, q):
       """KL divergence of p from q in nats: sum of p * log(p / q) (terms with p = 0 count as 0)."""
       p, q = np.asarray(p, float), np.asarray(q, float); m = p > 0
       return float(np.sum(p[m] * np.log(p[m] / q[m])))
   def pi_star(ref, r, beta):
       """The best policy for the objective 'expected reward minus beta * KL from the reference'."""
       w = np.asarray(ref, float) * np.exp((np.asarray(r, float) - np.max(r)) / beta); return w / w.sum()

   P5_NAMES = ["rude", "plain", "helpful"]; p5_ref = np.array([0.6, 0.3, 0.1]); p5_r = np.array([0.0, 1.0, 2.0])
   print(f"reference: probabilities {p5_ref}, expected reward {p5_ref @ p5_r:.3f}")
   print("  beta   P(rude)  P(plain)  P(helpful)   E[reward]      KL")
   for beta in (100, 10, 2, 1, 0.5, 0.2, 0.05):
       p = pi_star(p5_ref, p5_r, beta)
       print(f"{beta:6g}   {p[0]:.4f}   {p[1]:.4f}    {p[2]:.4f}      {p @ p5_r:.4f}   {kl_np(p, p5_ref):.4f}")

   p5_beta = 1.0; p5_pi = pi_star(p5_ref, p5_r, p5_beta)
   J = lambda p: float(p @ p5_r - p5_beta * kl_np(p, p5_ref))
   p5_rng = np.random.default_rng(0); p5_best_rand = max(J(p5_rng.dirichlet(np.ones(3))) for _ in range(100000))
   print(f"\nbeta = 1: J(pi*) = {J(p5_pi):.6f}, the best of 100,000 random policies = {p5_best_rand:.6f}")
   assert J(p5_pi) >= p5_best_rand - 1e-12 and abs(p5_pi.sum() - 1) < 1e-12
   assert np.abs(pi_star(p5_ref, p5_r, 1e6) - p5_ref).max() < 1e-4 and pi_star(p5_ref, p5_r, 0.01)[2] > 0.999
   RESULTS["p5_b1"] = (round(float(p5_pi @ p5_r), 4), round(float(p5_pi[2]), 4))
   ```

   You should see the reference model with an expected reward of 0.500, and the table. Look at the row for β = 1: the optimal policy has probabilities 0.2785 (rude), 0.3785 (plain) and 0.3430 (helpful), an expected reward of **1.0645** and a KL of 0.2970. So the answer to the Imagine-first question is: the reward doubles (0.5 to 1.06) and the probability of the helpful answer goes from 0.10 to 0.34. The extremes behave as expected: with β = 100 the policy stays at the reference (reward 0.5045, KL 0.0000), and with β = 0.05 all the probability goes to the helpful answer (reward 2.0000, KL 2.3026, which is ln 10, the cost of moving from probability 0.1 to 1). The check on the objective: J(pi*) = 0.767508, against 0.767505 for the best of 100,000 random policies: the formula wins, and the best random policy gets within 0.000003, as it should since random search can get close but not above. **β is a price: a high β makes moving expensive, a low β makes it cheap.**

   *Exercise 5.1:* compute the optimal policy for β = 0.5 on this toy and store in `pi_half` the probability of the **helpful** answer, in `reward_half` the expected reward and in `kl_half` the KL from the reference. Why it matters: one point on the frontier of Step 5.2.

   Write your answer in this starter block:

   ```python
   pi_half = None       # P(helpful) of the optimal policy for beta = 0.5
   reward_half = None   # its expected reward
   kl_half = None       # its KL from the reference
   ```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

   ```python
   if pi_half is None or reward_half is None or kl_half is None:
       print("Fill in pi_half, reward_half and kl_half above, then run this cell again.")
   else:
       _p = pi_star(p5_ref, p5_r, 0.5)
       check("pi_half is right", abs(pi_half - _p[2]) < 1e-3, "pi_star(p5_ref, p5_r, 0.5)[2].")
       check("reward_half is right", abs(reward_half - _p @ p5_r) < 1e-3, "The dot product of the policy with the rewards.")
       check("kl_half is right", abs(kl_half - kl_np(_p, p5_ref)) < 1e-3, "kl_np(policy, p5_ref).")
       RESULTS["p5_half"] = (round(pi_half, 4), round(reward_half, 4), round(kl_half, 4))
   ```

   A possible answer, and what it means:

   ```python
   p = pi_star(p5_ref, p5_r, 0.5)
   pi_half, reward_half, kl_half = float(p[2]), float(p @ p5_r), kl_np(p, p5_ref)
   print(pi_half, reward_half, kl_half)
   ```

   The helpful answer gets probability 0.6597, the expected reward is 1.5872 and the KL is 1.0609. The reward tripled compared with the reference (0.5), and the policy now says "helpful" two times out of three, but it pays 1.06 nats of KL.

2. **Step 5.2. The frontier: reward against KL.** Every β gives one policy, and one point (KL, expected reward). We sweep β from 0.03 to 30 (60 values, equally spaced on a log scale) and draw the curve, which is the **frontier**: for a given amount of KL, no policy earns more reward. To check that, the plot also shows 3,000 random policies as grey dots. They must all lie on or below the curve. Then the block prints how much of the possible gain you get for how much KL.

   ```python
   p5_betas = np.logspace(np.log10(0.03), np.log10(30), 60)
   p5_front = np.array([ [kl_np(pi_star(p5_ref, p5_r, b), p5_ref), pi_star(p5_ref, p5_r, b) @ p5_r] for b in p5_betas])
   p5_rand = np.array([ [kl_np(p, p5_ref), p @ p5_r] for p in p5_rng.dirichlet(np.ones(3), 3000)])
   fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
   ax[0].scatter(p5_rand[:, 0], p5_rand[:, 1], s=4, color="lightgray", label="3,000 random policies"); ax[0].plot(p5_front[:, 0], p5_front[:, 1], "r-", lw=2, label="pi* for each beta")
   for b in (10, 2, 1, 0.5, 0.2):
       p = pi_star(p5_ref, p5_r, b); ax[0].annotate(f"β={b}", (kl_np(p, p5_ref), p @ p5_r), fontsize=8)
   ax[0].set_xlabel("KL from the reference (nats)"); ax[0].set_ylabel("expected reward"); ax[0].legend(fontsize=8)
   for j, nm in enumerate(P5_NAMES): ax[1].plot(p5_betas, [pi_star(p5_ref, p5_r, b)[j] for b in p5_betas], label=nm)
   ax[1].set_xscale("log"); ax[1].set_xlabel("β (log scale)"); ax[1].set_ylabel("probability"); ax[1].legend(fontsize=8)
   plt.tight_layout(); plt.show()
   # no random policy beats the frontier: compare each with the frontier value at its own KL
   p5_dense = np.array([ [kl_np(pi_star(p5_ref, p5_r, b), p5_ref), pi_star(p5_ref, p5_r, b) @ p5_r] for b in np.logspace(-2, 3, 3000)])      # a finer frontier for the check
   p5_above = int(np.sum(p5_rand[:, 1] > np.interp(p5_rand[:, 0], p5_dense[::-1, 0], p5_dense[::-1, 1]) + 1e-3))
   print(f"random policies above the frontier: {p5_above} of 3000")
   assert p5_above == 0
   for target in (0.1, 0.5, 1.0):
       k = int(np.argmin(np.abs(p5_front[:, 0] - target)))
       print(f"KL about {target}: beta = {p5_betas[k]:.3f}, expected reward {p5_front[k, 1]:.3f}  (the reference has 0.5, the maximum is 2.0)")
   RESULTS["p5_frontier"] = (round(float(np.interp(0.5, p5_front[::-1, 0], p5_front[::-1, 1])), 3),)
   ```

   You should see two plots. On the left, the red curve is the frontier and the 3,000 grey dots lie below it: 0 of 3,000 random policies are above it (the block asserts this). On the right, the probabilities show how the policy slides from the reference (large β) to the single best answer (small β). The text lines: about 0.1 nats of KL buy an expected reward of 0.832 (β about 1.6), 0.5 nats buy 1.220 (β about 0.80) and 1.0 nats buy 1.590 (β about 0.50). The frontier is **concave**: the first nats of KL are the cheapest. Going from KL 0.1 to 0.5 gains 0.39 of reward; going from 1.0 to 2.3 (almost the maximum) gains only 0.41 more for 1.3 nats. That is why in practice β is chosen to give a small KL: most of the benefit comes first.

3. **Step 5.3. Six answers, and a reward model that is wrong.** Now six answers, with a reference model that mostly says the first ones. We give two reward vectors. The **true** reward `r_true` is what people really want. The **proxy** `r_proxy` is what a reward model learned, and it is wrong about the last answer: it over-rates a weird answer (for example the word salad of Step 4.6) that the reference model almost never produces. We optimise the *proxy* for each β and measure the *true* reward of the result. If the leash is long enough (small β), the policy finds the weird answer and the true reward falls.

   ```python
   p5_ref6 = np.array([0.35, 0.25, 0.18, 0.12, 0.08, 0.02])
   p5_true = np.array([0.0, 0.5, 1.0, 1.5, 2.0, -1.0]); p5_proxy = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 4.0])
   print(f"reference: true reward {p5_ref6 @ p5_true:.3f}, proxy reward {p5_ref6 @ p5_proxy:.3f}, P(weird answer) = {p5_ref6[5]}")
   print("  beta   KL     proxy reward   TRUE reward   P(best true answer)  P(weird answer)")
   p5_rows = []
   for beta in (30, 10, 3, 1, 0.5, 0.3, 0.2, 0.1, 0.05):
       p = pi_star(p5_ref6, p5_proxy, beta); p5_rows.append((beta, kl_np(p, p5_ref6), p @ p5_proxy, p @ p5_true, p[4], p[5]))
       print(f"{beta:6g}  {p5_rows[-1][1]:5.2f}   {p5_rows[-1][2]:9.3f}    {p5_rows[-1][3]:9.3f}        {p[4]:7.3f}          {p[5]:7.3f}")
   p5_bg = np.logspace(np.log10(0.03), np.log10(30), 200)
   p5_curve = np.array([ [b, kl_np(pi_star(p5_ref6, p5_proxy, b), p5_ref6), pi_star(p5_ref6, p5_proxy, b) @ p5_proxy, pi_star(p5_ref6, p5_proxy, b) @ p5_true] for b in p5_bg])
   p5_peak = p5_curve[int(np.argmax(p5_curve[:, 3]))]
   print(f"\nthe best beta for the TRUE reward (searching 200 values): beta = {p5_peak[0]:.3f}, KL = {p5_peak[1]:.3f}, true reward {p5_peak[3]:.3f}")
   plt.figure(figsize=(5.5, 3.2)); plt.plot(p5_curve[:, 1], p5_curve[:, 2], label="proxy reward (what we optimise)"); plt.plot(p5_curve[:, 1], p5_curve[:, 3], label="true reward")
   plt.axvline(p5_peak[1], color="gray", ls=":"); plt.xlabel("KL from the reference (nats)"); plt.ylabel("expected reward"); plt.legend(fontsize=8); plt.tight_layout(); plt.show()
   assert p5_curve[0, 3] < p5_curve[:, 3].max() - 0.5        # the true reward is far below its peak when the leash is almost gone (smallest beta)
   RESULTS["p5_goodhart"] = (round(float(p5_peak[0]), 3), round(float(p5_peak[3]), 3), round(float(p5_curve[0, 3]), 3))
   ```

   You should see that the reference has a true reward of 0.625 and a proxy reward of 0.725, with 2 percent probability of the weird answer. As β falls, the **proxy reward rises without limit** (0.746, 0.792, 0.988, 2.031, 3.672, 3.985 and 4.000, the maximum of the proxy) but the **true reward rises first and then collapses**: 0.635 at β = 30, 0.654 at β = 10, **0.702 at β = 3**, then 0.459 at β = 1 (worse than the reference, with 31 percent of the probability on the weird answer), -0.663 at β = 0.5 and -1.000 at the end, where the policy always gives the weird answer. The best β for the true reward is **2.221**, with a KL of only 0.097 nats and a true reward of 0.710. This is Goodhart's law as a picture: the gain at the peak (0.710 against 0.625) is small and the loss beyond it is huge. It is also the toy version of Step 4.6. **The KL leash is not a luxury: it limits how far the policy can walk into the places where the reward model is wrong.** In real RLHF you cannot see the true reward; you can only see the KL and watch quality on held-out human judgements.

   *Exercise 5.2:* among the β values in the array `p5_bg` (the 200 values above), find the **smallest** β for which the true reward of the optimised policy is still at least the true reward of the reference model (`p5_ref6 @ p5_true`, so the policy is not worse than before). Store it in `beta_safe`, and store in `kl_safe` the KL from the reference at that β. Why it matters: it is how you would choose β in practice: as small as possible (as much improvement as possible) while the true quality has not dropped.

   Write your answer in this starter block:

   ```python
   beta_safe = None   # the smallest beta in p5_bg with true reward at least that of the reference
   kl_safe = None     # the KL of that policy from p5_ref6
   ```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

   ```python
   if beta_safe is None or kl_safe is None:
       print("Fill in beta_safe and kl_safe above, then run this cell again.")
   else:
       _ok = [b for b in p5_bg if pi_star(p5_ref6, p5_proxy, b) @ p5_true >= p5_ref6 @ p5_true]
       check("beta_safe is right", abs(beta_safe - min(_ok)) < 1e-9, "Keep only the betas whose true reward is at least p5_ref6 @ p5_true, then take the minimum.")
       check("kl_safe is right", abs(kl_safe - kl_np(pi_star(p5_ref6, p5_proxy, min(_ok)), p5_ref6)) < 1e-6, "KL of the policy for beta_safe.")
       RESULTS["p5_safe"] = (round(float(beta_safe), 3), round(float(kl_safe), 3))
   ```

   A possible answer, and what it means:

   ```python
   ok = [b for b in p5_bg if pi_star(p5_ref6, p5_proxy, b) @ p5_true >= p5_ref6 @ p5_true]
   beta_safe = min(ok); kl_safe = kl_np(pi_star(p5_ref6, p5_proxy, beta_safe), p5_ref6)
   print(beta_safe, kl_safe)
   ```

   The smallest β is 1.274, with a KL of 0.407 from the reference. For a smaller β the policy moves towards the weird answer and its true reward falls below the reference's 0.625. Note that 'not worse than before' is a weak criterion: the best β (2.221) has a better true reward than this, and the proxy reward would always urge us to a smaller β.

4. **Step 5.4. DPO by gradient descent finds the same policy.** **DPO (Direct Preference Optimization)** skips the reward model. Its loss for a pair (winner `w`, loser `l`) uses only the log-probabilities of the policy and of the reference: `loss = -log sigmoid( β * [ (log pi(w) - log pi_ref(w)) - (log pi(l) - log pi_ref(l)) ] )`. The quantity `β * log(pi / pi_ref)` plays the role of an **implied reward**. Here is the check of the theory. The toy policy is a softmax of 3 numbers `theta`, started at `log pi_ref`. The preference data are *every ordered pair of different answers*, each pair weighted equally, and the label probability is Bradley-Terry with the true reward: answer i beats answer j with probability `sigmoid(r_i - r_j)`. We minimise the expected DPO loss by plain gradient descent (PyTorch autograd) and compare the result with `pi_star(ref, r, β)`, for β = 1 and β = 0.3. They must agree within 0.001.

   ```python
   def dpo_pair_loss(logp_w, logp_l, ref_w, ref_l, beta):
       """DPO loss of one pair, from four log-probabilities: -log sigmoid(beta * ((logp_w - ref_w) - (logp_l - ref_l)))."""
       return -F.logsigmoid(beta * ((logp_w - ref_w) - (logp_l - ref_l)))

   def dpo_toy(ref, r, beta, steps=3000, lr=5.0):
       """Train a softmax policy on the expected DPO loss over all pairs with Bradley-Terry labels. Returns the final probabilities and the loss history."""
       K = len(ref); lref = torch.tensor(np.log(ref)); R = torch.tensor(np.asarray(r, float))
       theta = lref.clone().requires_grad_(True)
       W = torch.ones(K, K, dtype=torch.float64) / (K * (K - 1)); W.fill_diagonal_(0)
       Pw = torch.sigmoid(R[:, None] - R[None, :])             # probability that i beats j
       opt = torch.optim.SGD([theta], lr=lr); hist = []
       for s in range(steps):
           lp = torch.log_softmax(theta, 0)
           L = (W * Pw * dpo_pair_loss(lp[:, None], lp[None, :], lref[:, None], lref[None, :], beta)).sum()
           opt.zero_grad(); L.backward(); opt.step(); hist.append(float(L))
       return torch.softmax(theta, 0).detach().numpy(), hist

   p5_dpo = {}
   for beta in (1.0, 0.3):
       pd_, hd_ = dpo_toy(p5_ref, p5_r, beta); ps_ = pi_star(p5_ref, p5_r, beta); p5_dpo[beta] = (pd_, ps_, hd_)
       print(f"beta {beta}: DPO policy {np.round(pd_, 4)}   closed form {np.round(ps_, 4)}   largest difference {np.abs(pd_ - ps_).max():.2e}   loss {hd_[0]:.4f} -> {hd_[-1]:.4f}")
       assert np.abs(pd_ - ps_).max() < 1e-3
   print("a one-pair check by hand: loss of (helpful beats rude) under the reference itself =", round(float(dpo_pair_loss(torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), 1.0)), 4), "= ln 2")
   plt.figure(figsize=(5, 3)); plt.plot(p5_dpo[1.0][2], label="β = 1"); plt.plot(p5_dpo[0.3][2], label="β = 0.3"); plt.xlabel("gradient step"); plt.ylabel("expected DPO loss"); plt.legend(); plt.tight_layout(); plt.show()
   RESULTS["p5_dpo"] = tuple(round(float(np.abs(p5_dpo[b][0] - p5_dpo[b][1]).max()), 6) for b in (1.0, 0.3))
   ```

   You should see that the DPO policy and the closed-form policy agree to machine precision: for β = 1, both are (0.2785, 0.3785, 0.3430) with a largest difference of 5.6e-17; for β = 0.3, both are (0.0069, 0.0960, 0.8971) with a largest difference of 2.2e-16 (the assertion allows 0.001). The expected DPO loss falls from 0.3466 (= ln 2 / 2: when the policy equals the reference every pair costs ln 2, but the label probabilities split each pair between its two directions, so the pairs cost half of ln 2 on average) to 0.2550 in both cases. **DPO never used a reward**: it only used pairs and the log-ratio of the policy to the reference, yet it found exactly the policy that maximises reward minus β times KL. The reason is the identity behind the loss: the reward that DPO implies, β times log(pi / pi_ref), equals the true reward up to a constant that cancels in every pair. The check by hand at the end shows the starting loss of a pair, ln 2 = 0.6931.

   *Exercise 5.3:* a pair has the log-probabilities `logp_w = -1.0`, `logp_l = -2.0` under the policy and `ref_w = -1.5`, `ref_l = -1.5` under the reference, and β = 0.5. Compute the implied reward margin `beta * ((logp_w - ref_w) - (logp_l - ref_l))` and store it in `dpo_margin`, and store the loss in `dpo_loss`. Why it matters: during DPO training on a real model this margin is the number you watch; it should grow.

   Write your answer in this starter block:

   ```python
   dpo_margin = None   # beta * ((logp_w - ref_w) - (logp_l - ref_l)) with the numbers above
   dpo_loss = None     # -log sigmoid(dpo_margin)
   ```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

   ```python
   if dpo_margin is None or dpo_loss is None:
       print("Fill in dpo_margin and dpo_loss above, then run this cell again.")
   else:
       _m = 0.5 * ((-1.0 + 1.5) - (-2.0 + 1.5))
       check("dpo_margin is right", abs(dpo_margin - _m) < 1e-6, "(logp_w - ref_w) is 0.5 and (logp_l - ref_l) is -0.5; multiply their difference by beta.")
       check("dpo_loss is right", abs(dpo_loss + math.log(1 / (1 + math.exp(-_m)))) < 1e-6, "Minus the log of the sigmoid of the margin.")
       check("the loss is below ln 2 because the policy already prefers the winner", dpo_loss < math.log(2), "A positive margin gives a loss below 0.693.")
       RESULTS["p5_dpo_hand"] = (round(dpo_margin, 4), round(dpo_loss, 4))
   ```

   A possible answer, and what it means:

   ```python
   dpo_margin = 0.5 * ((-1.0 - (-1.5)) - (-2.0 - (-1.5)))
   dpo_loss = float(dpo_pair_loss(torch.tensor(-1.0), torch.tensor(-2.0), torch.tensor(-1.5), torch.tensor(-1.5), 0.5))
   print(dpo_margin, dpo_loss)
   ```

   The margin is 0.5 * (0.5 - (-0.5)) = **0.5** and the loss is **0.4741**, below ln 2 = 0.6931. The policy already gives more probability to the winner than the reference does (by 0.5 nats) and less probability to the loser (by 0.5 nats), so the pair is partly learned. Training would continue to push the margin up, and the gradient shrinks as the margin grows.

5. **Step 5.5. DPO with only a few comparisons.** Real data are not "all pairs with exact probabilities": they are a **finite list of sampled comparisons**, each with a single winner. Let us repeat the experiment with such data. We draw N pairs (a pair of two different answers, chosen uniformly; the winner is chosen with the Bradley-Terry probability), run gradient descent on the average DPO loss for 3,000 steps with β = 1, and measure the largest difference from the closed-form policy. We repeat with 5 different random draws for N = 10, 30, 100, 300 and 1,000 and print the average.

   ```python
   def dpo_toy_data(ref, r, beta, pairs, steps=3000, lr=5.0):
       """DPO on a finite list of (winner, loser) index pairs. Returns the final probabilities."""
       lref = torch.tensor(np.log(ref)); theta = lref.clone().requires_grad_(True); W_ = torch.tensor([p[0] for p in pairs]); L_ = torch.tensor([p[1] for p in pairs])
       opt = torch.optim.SGD([theta], lr=lr)
       for s in range(steps):
           lp = torch.log_softmax(theta, 0)
           loss = dpo_pair_loss(lp[W_], lp[L_], lref[W_], lref[L_], beta).mean(); opt.zero_grad(); loss.backward(); opt.step()
       return torch.softmax(theta, 0).detach().numpy()

   def draw_pairs(r, n, seed):
       g = np.random.RandomState(seed); out = []
       while len(out) < n:
           i, j = g.choice(len(r), 2, replace=False)
           out.append((i, j) if g.rand() < 1 / (1 + np.exp(-(r[i] - r[j]))) else (j, i))
       return out

   p5_star = pi_star(p5_ref, p5_r, 1.0); p5_err = {}
   print("   N pairs   largest difference from pi* (mean of 5 draws)   (min, max)")
   for N in ((10, 100) if QUICK else (10, 30, 100, 300, 1000)):
       errs = [np.abs(dpo_toy_data(p5_ref, p5_r, 1.0, draw_pairs(p5_r, N, seed=s), steps=1000 if QUICK else 3000) - p5_star).max() for s in range(5)]
       p5_err[N] = float(np.mean(errs)); print(f"{N:9d}   {p5_err[N]:10.4f}                              ({min(errs):.4f}, {max(errs):.4f})")
   RESULTS["p5_finite"] = {N: round(v, 4) for N, v in p5_err.items()}
   ```

   You should see that the error depends strongly on the number of comparisons. The largest difference from pi* is **0.5237** with 10 pairs (individual draws range from 0.2784 to 0.6564), 0.2548 with 30, 0.0878 with 100, 0.0446 with 300 and **0.0253** with 1,000 (range 0.0053 to 0.0465). With all pairs and exact probabilities (Step 5.4) the error was zero; with sampled comparisons it falls roughly like one over the square root of N. With only 10 comparisons the policy is far from the optimum, sometimes preferring the wrong answer: each comparison is only a noisy vote, and 3,000 steps of gradient descent faithfully fit the noise. This is why DPO needs far more pairs than our 14 for a real model, and why β and the number of steps act as regularisers.

6. **Step 5.6. A few DPO steps on the real GPT-2 (optional, about 1 minute).** The same loss works on a real language model; only the four log-probabilities change. The log-probability of an answer is the sum of the log-probabilities of its tokens given the question (the answer only: the prompt tokens are masked, as in Part 2). The reference is a frozen copy of the SFT model of Step 4.5; the policy starts as another copy and is trained on `PREF_TRAIN` with β = 0.1, AdamW, learning rate 1e-5, batch size 1 and 2 epochs (28 steps; 1 epoch in quick mode). Before and after we measure, on the training pairs and on the 6 validation pairs, the **implied reward margin** and the fraction of pairs with a positive margin, and the log-probability of the chosen answers. Small data and only 28 steps: do not expect miracles; read the numbers honestly.

   ```python
   def seq_logp(model, prompt, answer, grad=False):
       """Sum of the log-probabilities of the answer tokens given the question (prompt tokens masked)."""
       p_ids = tok(format_prompt(prompt))["input_ids"]; a_ids = tok(answer)["input_ids"]
       ids = torch.tensor([p_ids + a_ids]).to(device)
       with torch.set_grad_enabled(grad):
           lp = torch.log_softmax(model(input_ids=ids).logits[0, :-1], -1)
       tgt = ids[0, 1:]; tok_lp = lp.gather(1, tgt[:, None])[:, 0]
       return tok_lp[len(p_ids) - 1:].sum()

   p5_ref_model = copy.deepcopy(p4_sft).eval()
   for q_ in p5_ref_model.parameters(): q_.requires_grad_(False)
   p5_cache = {}
   def ref_logp(prompt, answer):
       k = (prompt, answer)
       if k not in p5_cache: p5_cache[k] = float(seq_logp(p5_ref_model, prompt, answer))
       return p5_cache[k]

   def dpo_report(model, rows, beta=0.1):
       model.eval(); m = []; ch = []
       for r in rows:
           lw, ll = float(seq_logp(model, r["prompt"], r["chosen"])), float(seq_logp(model, r["prompt"], r["rejected"]))
           m.append(beta * ((lw - ref_logp(r["prompt"], r["chosen"])) - (ll - ref_logp(r["prompt"], r["rejected"])))); ch.append(lw)
       return float(np.mean(m)), float(np.mean(np.array(m) > 0)), float(np.mean(ch))

   p5_pol = copy.deepcopy(p4_sft); p5_before = (dpo_report(p5_pol, PREF_TRAIN), dpo_report(p5_pol, PREF_VAL))
   print(f"before: train margin {p5_before[0][0]:.3f}, positive {p5_before[0][1]:.0%}, chosen logp {p5_before[0][2]:.1f} | validation margin {p5_before[1][0]:.3f}, positive {p5_before[1][1]:.0%}, chosen logp {p5_before[1][2]:.1f}")
   p5_opt = torch.optim.AdamW(p5_pol.parameters(), lr=1e-5); p5_loss = []; t0 = time.time()
   for ep in range(1 if QUICK else 2):
       p5_pol.train(); order = np.random.RandomState(ep).permutation(len(PREF_TRAIN)); tot = 0.0
       for i in (order[:6] if QUICK else order):
           r = PREF_TRAIN[i]
           lw, ll = seq_logp(p5_pol, r["prompt"], r["chosen"], grad=True), seq_logp(p5_pol, r["prompt"], r["rejected"], grad=True)
           loss = dpo_pair_loss(lw, ll, ref_logp(r["prompt"], r["chosen"]), ref_logp(r["prompt"], r["rejected"]), 0.1)
           p5_opt.zero_grad(); loss.backward(); p5_opt.step(); tot += float(loss)
       p5_loss.append(tot / len(PREF_TRAIN)); print(f"epoch {ep + 1}: mean DPO loss {p5_loss[-1]:.4f}   ({time.time() - t0:.0f} s)")
   p5_after = (dpo_report(p5_pol, PREF_TRAIN), dpo_report(p5_pol, PREF_VAL)); del p5_opt
   print(f"after:  train margin {p5_after[0][0]:.3f}, positive {p5_after[0][1]:.0%}, chosen logp {p5_after[0][2]:.1f} | validation margin {p5_after[1][0]:.3f}, positive {p5_after[1][1]:.0%}, chosen logp {p5_after[1][2]:.1f}")
   print("a sample answer to a validation question, before and after:")
   print("  SFT:", repr(generate(p4_sft, format_prompt(PREF_VAL[0]["prompt"]), 30)))
   print("  DPO:", repr(generate(p5_pol, format_prompt(PREF_VAL[0]["prompt"]), 30)))
   RESULTS["p5_real"] = dict(before_train=round(p5_before[0][0], 3), after_train=round(p5_after[0][0], 3), before_val=round(p5_before[1][0], 3), after_val=round(p5_after[1][0], 3))
   ```

   You should see the log-probabilities of the real model move. Before training the policy equals the reference, so the margin is exactly 0.000 and 0 percent of the pairs have a positive margin. The mean DPO loss is 1.0247 in the first epoch and 0.3480 in the second (15 seconds per epoch). After 28 steps the mean implied margin on the training pairs is **2.155** (100 percent of the pairs positive) and on the 6 validation pairs **0.693** (100 percent positive): the policy has learned to prefer the chosen answers, and the preference also reaches unseen pairs, but with a third of the strength (0.693 against 2.155). The mean log-probability of the chosen answers **rose** from -95.5 to -75.7 on the training pairs and from -99.6 to -89.2 on the validation pairs. (In many runs the chosen log-probability falls while the margin grows, because DPO only pushes the *difference*. Here it rose, since the chosen answers are close to the style of the SFT model; do not assume it.) The greedy answer to the homework question is unchanged ("Sure! You can focus when you do homework. Hope that helps."): a margin of 2 in the implied reward is a change of the probabilities of whole sentences, and it does not necessarily change the most likely continuation. **Honest summary:** the numbers moved in the right direction on 14 pairs and 28 steps, but nothing in this experiment shows that the model got better at being helpful; to claim that you would need human judgements of its samples.

**Reflect.** Look at your Imagine-first guesses and write two or three sentences each. (1) What does β control, in the closed form and in the frontier plot? (2) In Step 5.3 the true reward peaked and then fell while the proxy reward kept rising: connect this to the word salad of Step 4.6. What is the practical use of the KL leash? (3) DPO reached the same policy as the closed form without a reward model: what did it need instead, and what happened when it had only 10 comparisons (Step 5.5)?

### Part 6. LoRA from scratch and with peft (40 min)

*Lesson sections: "PEFT: fine-tune a small add-on" and "Doing it well".* Full fine-tuning changes every one of the 124 million numbers of GPT-2 and needs a copy of all of them for every task. ==In this part you build the cheap alternative from scratch: LoRA freezes the model and trains two thin matrices beside each attention layer. You write the layer, put it into GPT-2, count what it trains and what it saves, train it against full fine-tuning on the same 24 examples, sweep the rank, merge the adapter back into the weights, check your numbers against the `peft` library, look at the update with the SVD, and measure the size of the file you would share.== Every run uses the same data, the same masked loss and `SEED = 42` as Parts 1 to 3, and the models are retrained inside this part, so it runs on its own. All times are for a CPU and differ between machines.

**Imagine first.** GPT-2 has 124 million numbers (parameters). Suppose that we freeze all of them and, next to each of the 12 attention layers, add a tiny side path made of two thin matrices whose product is the correction: that is LoRA, Low-Rank Adaptation. Only the thin matrices are trained. Write two predictions before you run anything. **(1)** What percent of GPT-2's parameters does this train (rank 8)? **(2)** After training only those, what share of the 8 validation answers will open with "Sure!", the style of Part 2? You compare with Steps 6.4 and 6.6.

```python
# Imagine first: LoRA
my_guess_share = None      # what percent of GPT-2's 124 million parameters does LoRA (rank 8 on the attention layers) train? (a number, for example 5)
my_guess_style = None      # after training only those parameters, what share of the 8 validation answers open with "Sure!"? (0 to 1)
```

1. **Step 6.1. The tools: masked examples, validation loss, a training loop.** Everything in this part uses the same few tools, defined once here. `p6_example` turns a question and its answer into token ids and labels, with the label -100 on every prompt token (the loss ignores them, so the model learns the answer and not the question, as in Part 1). `p6_val_loss` is the average masked loss on the 8 validation examples. `p6_train` is a plain training loop (AdamW, batch size 1, a shuffled order fixed by the seed, and the same dropout masks for every run, so two runs with the same settings give the same numbers). Before any training the base model has a validation loss and a `style_rate`; we also keep its loss on the general paragraph, the forgetting test of Part 3.

```python
P6_SEED = 0
P6_EPOCHS = 1 if QUICK else 2          # passes over the training examples for every run in this part
P6_TRAIN = INSTR_TRAIN[:6] if QUICK else INSTR_TRAIN
P6_FULL_LR, P6_LORA_LR = 3e-5, 1e-3     # LoRA needs a much larger learning rate than full fine-tuning (see Step 6.6)

def p6_example(q, a):
    """Token ids and labels for one example: the prompt tokens get the label -100 (ignored by the loss), so only the answer is learned."""
    p = tok(format_prompt(q))["input_ids"]; r = tok(a + tok.eos_token)["input_ids"]
    labels = [-100] * len(p) + r                 # -100 on the prompt tokens, the real token ids on the answer tokens
    return torch.tensor([p + r]).to(device), torch.tensor([labels]).to(device)

P6_TRAIN_EX = [p6_example(q, a) for q, a in P6_TRAIN]
P6_VAL_EX = [p6_example(q, a) for q, a in INSTR_VAL]

@torch.no_grad()
def p6_val_loss(model):
    """Mean masked loss (nats per answer token) on the 8 validation examples."""
    model.eval()
    return float(np.mean([float(model(input_ids=i, labels=l).loss) for i, l in P6_VAL_EX]))

def p6_train(model, params, lr, epochs=P6_EPOCHS):
    """Plain AdamW, batch size 1, in a shuffled order that depends only on P6_SEED. Returns (seconds, val loss after each epoch, the optimizer)."""
    torch.manual_seed(P6_SEED)              # the same dropout masks for every run
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    t0 = time.time(); val = []
    for ep in range(epochs):
        model.train()
        order = list(range(len(P6_TRAIN_EX))); random.Random(P6_SEED + ep).shuffle(order)
        for j in order:
            i, l = P6_TRAIN_EX[j]
            model(input_ids=i, labels=l).loss.backward(); opt.step(); opt.zero_grad()
        val.append(p6_val_loss(model))
    return time.time() - t0, val, opt

RESULTS["p6_base_val"] = round(p6_val_loss(base_model), 3)
RESULTS["p6_base_style"] = style_rate(base_model)
RESULTS["p6_base_general"] = round(text_loss(base_model, GENERAL_TEXT), 3)
print("base model: val loss %.3f, style_rate %.2f, loss on the general paragraph %.3f" % (RESULTS["p6_base_val"], RESULTS["p6_base_style"], RESULTS["p6_base_general"]))
print("training examples used: %d, epochs per run: %d" % (len(P6_TRAIN_EX), P6_EPOCHS))
```

It prints:

```text
base model: val loss 2.816, style_rate 0.00, loss on the general paragraph 3.509
training examples used: 24, epochs per run: 2
```

   **What you should see.**

   The cell prints `base model: val loss 2.816, style_rate 0.00, loss on the general paragraph 3.509` and `training examples used: 24, epochs per run: 2`.

   *What it means.* These three numbers are the starting line for the whole part: a validation loss of 2.816 nats per answer token (GPT-2 is quite surprised by the answers), a style_rate of 0.0 (it never opens with "Sure!") and a general-text loss of 3.509. Training should lower the first, raise the second and not move the third much.

2. **Step 6.2. How many numbers does LoRA train?** LoRA leaves a weight matrix W alone and learns a correction of the same shape as the product of two thin matrices: ΔW is A times B, where A has shape (d_in, r) and B has shape (r, d_out). The small number r is the **rank**. A layer that maps d_in numbers to d_out numbers has d_in times d_out weights; LoRA adds only r times (d_in + d_out). GPT-2's attention block has one big layer, `c_attn`, that produces the queries, keys and values in one go: it maps 768 numbers to 3 times 768 = 2,304 and exists in each of the 12 blocks. We adapt only these 12 layers.

   *Exercise 6.2:* Write `lora_params(r, d_in, d_out)`, which returns the number of trainable numbers LoRA adds to one layer, and use it to fill in `per_layer` (rank 8 on one `c_attn`), `total_lora` (all 12 layers) and `share_pct` (the total as a percent of `count_params(base_model)`). Why it matters: this is the number behind every claim that LoRA is cheap, and it is the first number your answer to the fifth assessed question needs. The check tests all four values.

```python
def lora_params(r, d_in, d_out):
    """Number of trainable numbers LoRA adds to ONE linear layer of shape d_in -> d_out."""
    return None   # YOUR CODE: A has shape (d_in, r) and B has shape (r, d_out)

D_IN, D_OUT, LAYERS = 768, 2304, 12        # GPT-2 small: c_attn maps 768 numbers to 3 x 768 = 2,304 (queries, keys, values), in 12 blocks
per_layer = total_lora = share_pct = None   # YOUR CODE: r = 8: per layer, for all 12 layers, and as a percent of count_params(base_model)

if lora_params(8, D_IN, D_OUT) is None or total_lora is None or share_pct is None:
    print("Fill in lora_params, per_layer, total_lora and share_pct above, then run this cell again.")
else:
    check("lora_params(8, 768, 2304) is 24,576", lora_params(8, D_IN, D_OUT) == 24576, "r * (d_in + d_out) = 8 * (768 + 2304)")
    check("per_layer is 24,576 and total_lora is 294,912", per_layer == 24576 and total_lora == 294912, "multiply by the 12 layers")
    check("share_pct is about 0.237", abs(share_pct - 100 * 294912 / count_params(base_model)) < 1e-6, "100 * total_lora / count_params(base_model)")
```

   A possible answer (try first, then compare):

```python
def lora_params(r, d_in, d_out):
    return r * d_in + r * d_out          # A: d_in x r, B: r x d_out

D_IN, D_OUT, LAYERS = 768, 2304, 12
per_layer = lora_params(8, D_IN, D_OUT)
total_lora = LAYERS * per_layer
share_pct = 100 * total_lora / count_params(base_model)
full_layer = D_IN * D_OUT + D_OUT                     # the weight matrix and the bias of one c_attn
print("one layer: %d LoRA numbers against %d in c_attn itself (%.2f percent)" % (per_layer, full_layer, 100 * per_layer / full_layer))
print("all %d layers: %d LoRA numbers = %.3f percent of the %d parameters of GPT-2" % (LAYERS, total_lora, share_pct, count_params(base_model)))
assert per_layer == 24576 and total_lora == 294912
print("rank  parameters")
for r_ in (1, 2, 4, 8, 16): print("%4d  %10d" % (r_, LAYERS * lora_params(r_, D_IN, D_OUT)))
RESULTS["p6_lora_params"] = total_lora; RESULTS["p6_lora_share_pct"] = round(share_pct, 3)
```

One `c_attn` has 768 × 2,304 + 2,304 = 1,771,776 numbers; the LoRA path for it has only 24,576 (A: 768 × 8 = 6,144 and B: 8 × 2,304 = 18,432), under 1.4 percent of the layer. Twelve layers give 294,912, which is 0.237 percent of GPT-2. The cost grows in a straight line with r: doubling the rank doubles the parameters.

It prints:

```text
one layer: 24576 LoRA numbers against 1771776 in c_attn itself (1.39 percent)
all 12 layers: 294912 LoRA numbers = 0.237 percent of the 124439808 parameters of GPT-2
rank  parameters
   1       36864
   2       73728
   4      147456
   8      294912
  16      589824
```

Running the check on this answer prints:

```text
✓ lora_params(8, 768, 2304) is 24,576
✓ per_layer is 24,576 and total_lora is 294,912
✓ share_pct is about 0.237
```

   **What you should see.**

   The check passes three ticks. The answer prints `one layer: 24576 LoRA numbers against 1771776 in c_attn itself (1.39 percent)`, then `all 12 layers: 294912 LoRA numbers = 0.237 percent of the 124439808 parameters of GPT-2`, and a table of the parameters for the ranks 1, 2, 4, 8 and 16: 36,864, 73,728, 147,456, 294,912 and 589,824.

   *What it means.* With the default rank 8 we train about one number in 420, and with rank 1 about one in 3,400. Whether so few numbers can be enough is the question of the rest of this part. `RESULTS["p6_lora_params"]` is 294912 and `RESULTS["p6_lora_share_pct"]` is 0.237.

3. **Step 6.3. The LoRA layer.** The LoRA layer wraps the original layer and adds the side path: y = base(x) + (α / r) · (x A) B. Three details matter. **(1) B starts at zero**, so the side path adds exactly nothing at the start and the model begins as the unchanged GPT-2; A starts as small random numbers (standard deviation 0.02), because if both started at zero the gradients would be zero too and nothing would ever move. **(2) The scale α / r**, with α a number you choose (we use α = 2r, so the scale is 2), keeps the size of the update comparable when you change r. **(3) GPT-2 stores its weights in a `Conv1D` layer**, whose weight has the shape (in, out), the opposite of `nn.Linear`. That is why A is `(d_in, r)` and B is `(r, d_out)` here, and why you read the shape from `base.weight.shape` and do not guess it.

   *Exercise 6.3:* Complete `LoRALinearMine`: create `A` (a `nn.Parameter` of shape (d_in, r) with random numbers of standard deviation 0.02) and `B` (a `nn.Parameter` of shape (r, d_out) of zeros), and let `forward` return the base output plus `self.scale` times `(x @ A) @ B`. Why it matters: if you get a shape wrong (the usual mistake is to read the Conv1D weight as (out, in)) nothing works, and if B does not start at zero your model is damaged before it is trained. The check tries your layer on a toy `Conv1D`.

```python
from transformers.pytorch_utils import Conv1D

class LoRALinearMine(nn.Module):
    """A frozen linear layer plus a trainable low-rank update: y = base(x) + (alpha / r) * x A B."""
    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base, self.r, self.scale = base, r, alpha / r
        d_in, d_out = base.weight.shape              # careful: GPT-2's Conv1D stores W with shape (in, out), the opposite of nn.Linear
        self.A = None      # YOUR CODE: nn.Parameter of shape (d_in, r) filled with random numbers of standard deviation 0.02
        self.B = None      # YOUR CODE: nn.Parameter of shape (r, d_out) filled with zeros
    def forward(self, x):
        return None        # YOUR CODE: the base output plus self.scale times (x @ A) @ B

toy_base = Conv1D(6, 4)                       # Conv1D(nf = 6 outputs, nx = 4 inputs): weight shape (4, 6)
toy = LoRALinearMine(toy_base, r=2, alpha=4)
if toy.A is None or toy.B is None or toy(torch.randn(3, 4)) is None:
    print("Fill in LoRALinearMine above, then run this cell again.")
else:
    x_ = torch.randn(3, 4)
    check("A has shape (4, 2) and B has shape (2, 6)", tuple(toy.A.shape) == (4, 2) and tuple(toy.B.shape) == (2, 6), "d_in x r and r x d_out; for a Conv1D, base.weight.shape is (in, out)")
    check("B starts at zero, so the output equals the base at the start", float(toy.B.abs().sum()) == 0 and torch.allclose(toy(x_), toy_base(x_)), "B = torch.zeros(r, d_out)")
    toy.B.data = torch.randn(2, 6)
    check("with B not zero: y = base(x) + 2 * x A B  (alpha / r = 4 / 2 = 2)", torch.allclose(toy(x_), toy_base(x_) + 2 * (x_ @ toy.A @ toy.B), atol=1e-5), "the scale is alpha / r")
    check("A and B require gradients", toy.A.requires_grad and toy.B.requires_grad, "wrap them in nn.Parameter")
```

   A possible answer (try first, then compare):

```python
from transformers.pytorch_utils import Conv1D

class LoRALinearMine(nn.Module):
    """A frozen linear layer plus a trainable low-rank update: y = base(x) + (alpha / r) * x A B."""
    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base, self.r, self.scale = base, r, alpha / r
        d_in, d_out = base.weight.shape
        self.A = nn.Parameter(torch.randn(d_in, r) * 0.02)      # small random numbers
        self.B = nn.Parameter(torch.zeros(r, d_out))            # zeros: the update starts at exactly zero
    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A) @ self.B

LoRALinear = LoRALinearMine                          # the name used by the rest of this part
torch.manual_seed(0)
layer0 = base_model.transformer.h[0].attn.c_attn     # the real c_attn of block 0: a Conv1D with weight shape (768, 2304)
wrapped = LoRALinear(layer0, r=8, alpha=16)
x_ = torch.randn(2, 5, 768)
with torch.no_grad(): diff0 = float((wrapped(x_) - layer0(x_)).abs().max())
print("c_attn weight shape:", tuple(layer0.weight.shape), "  A:", tuple(wrapped.A.shape), "  B:", tuple(wrapped.B.shape))
print("largest difference from the base layer at the start: %.1f" % diff0)
with torch.no_grad(): wrapped.B.normal_(0, 0.02); diff1 = float((wrapped(x_) - layer0(x_)).abs().max())
print("largest difference after giving B random numbers: %.4f" % diff1)
assert diff0 == 0.0 and diff1 > 0
RESULTS["p6_diff_at_init"] = diff0; RESULTS["p6_diff_after_B"] = round(diff1, 3)
```

The base layer is called as it is (it does its own matrix product and adds its own bias), and the LoRA path is added on top. Writing `(x @ self.A) @ self.B` first multiplies by the thin matrix A, so the largest intermediate result has only r columns: that is why LoRA is cheap in time as well as in memory.

It prints:

```text
c_attn weight shape: (768, 2304)   A: (768, 8)   B: (8, 2304)
largest difference from the base layer at the start: 0.0
largest difference after giving B random numbers: 0.3604
```

Running the check on this answer prints:

```text
✓ A has shape (4, 2) and B has shape (2, 6)
✓ B starts at zero, so the output equals the base at the start
✓ with B not zero: y = base(x) + 2 * x A B  (alpha / r = 4 / 2 = 2)
✓ A and B require gradients
```

   **What you should see.**

   The check passes four ticks. The answer prints `c_attn weight shape: (768, 2304)   A: (768, 8)   B: (8, 2304)`, `largest difference from the base layer at the start: 0.0` and, after we give B random numbers, `largest difference after giving B random numbers: 0.36`. The assertion passes.

   *What it means.* Because B is zero, your layer is **exactly** the base layer at the start (difference 0.0, not just small). This is the property that makes LoRA safe: training begins from the model you trust, and only moves as far as the data pulls it. `RESULTS["p6_diff_at_init"]` is 0.0.

4. **Step 6.4. Put LoRA into GPT-2.** Now put the layer into GPT-2. `add_lora` first freezes every parameter (`requires_grad = False`), then replaces the `c_attn` of each of the 12 blocks by a `LoRALinear`. The new A and B matrices are created after the freeze, so they are the only trainable parameters. We then count them, and compare the logits of the new model with those of the original on a prompt: they must be identical, because every B is zero.

```python
def add_lora(model, r=8, alpha=16):
    """Freeze every parameter of `model`, then wrap the c_attn layer of every block in a LoRALinear. Returns the list of trainable parameters."""
    for p in model.parameters(): p.requires_grad = False
    for blk in model.transformer.h:
        blk.attn.c_attn = LoRALinear(blk.attn.c_attn, r, alpha)     # the new A and B are created trainable
    return [p for p in model.parameters() if p.requires_grad]

torch.manual_seed(P6_SEED)
lora_model = copy.deepcopy(base_model).to(device)
lora_params_list = add_lora(lora_model, r=8, alpha=16)
n_train = count_params(lora_model, trainable_only=True)
print("trainable: %d of %d parameters (%.3f percent)" % (n_train, count_params(lora_model), 100 * n_train / count_params(lora_model)))
assert n_train == 294912                                              # the number of Step 6.2
ids_ = tok(format_prompt("What is the capital of France?"), return_tensors="pt").to(device)
with torch.no_grad(): same = float((lora_model.eval()(**ids_).logits - base_model(**ids_).logits).abs().max())
print("largest difference between the logits of the LoRA model and of GPT-2 before training: %.1f" % same)
assert same == 0.0
RESULTS["p6_trainable"] = n_train; RESULTS["p6_total"] = count_params(base_model)
```

It prints:

```text
trainable: 294912 of 124734720 parameters (0.236 percent)
largest difference between the logits of the LoRA model and of GPT-2 before training: 0.0
```

   **What you should see.**

   The cell prints `trainable: 294912 of 124734720 parameters (0.236 percent)` (the total now includes the 294,912 new numbers) and `largest difference between the logits of the LoRA model and of GPT-2 before training: 0.0`; both assertions pass. `RESULTS["p6_trainable"]` is 294912.

   *What it means.* Compare with your Imagine-first guess: 294,912 trainable numbers is about a quarter of one percent (0.237 percent of the original 124,439,808). Almost everybody guesses a few percent, because "a small add-on" sounds bigger than it is. The difference 0.0 in the logits confirms Step 6.3 on the whole model.

5. **Step 6.5. What LoRA saves: memory arithmetic.** Before we train, count the memory. Training with the Adam optimizer in float32 (4 bytes per number) must keep, for every parameter, its **weight** (4 bytes); for every *trainable* parameter also its **gradient** (4 bytes) and Adam's two running statistics (4 + 4 bytes), so 12 bytes more. A frozen weight costs only its 4 bytes. This ignores the activations (they depend on the batch size and are the same for both methods), so the real total is higher, but the comparison holds. In 16-bit "mixed precision" with a master copy the factor is 16 bytes per trainable parameter, as in the lesson: the idea is the same.

   *Exercise 6.5:* Write `train_bytes(n_total, n_trainable)` and use it to fill in `full_gb` (full fine-tuning: all 124,439,808 parameters are trainable) and `lora_gb` (LoRA: 294,912 trainable out of the same total), in gigabytes (1 GB is 1e9 bytes). Why it matters: memory, not speed, is what stops most people from fine-tuning a large model, and this formula tells you in advance whether a run will fit. The check tests the formula and both numbers.

```python
def train_bytes(n_total, n_trainable):
    """Memory for the numbers that training has to keep (float32, 4 bytes each), without the activations."""
    return None   # YOUR CODE: weights of ALL parameters (4 bytes each) + for the TRAINABLE ones a gradient (4 bytes) and the two Adam statistics (8 bytes)

full_gb = lora_gb = None      # YOUR CODE: in gigabytes (1 GB = 1e9 bytes) for full fine-tuning (all 124,439,808 trainable) and for LoRA (294,912 trainable)

if train_bytes(10, 10) is None or full_gb is None or lora_gb is None:
    print("Fill in train_bytes, full_gb and lora_gb above, then run this cell again.")
else:
    check("train_bytes(10, 10) is 160 and train_bytes(10, 0) is 40", train_bytes(10, 10) == 160 and train_bytes(10, 0) == 40, "4 * n_total + 12 * n_trainable")
    check("full fine-tuning needs about 1.99 GB", abs(full_gb - 1.991) < 0.01, "16 bytes for each of the 124,439,808 parameters")
    check("LoRA needs about 0.50 GB", abs(lora_gb - 0.501) < 0.01, "4 bytes for every parameter plus 12 for the 294,912 trainable ones")
```

   A possible answer (try first, then compare):

```python
def train_bytes(n_total, n_trainable):
    return 4 * n_total + 12 * n_trainable          # weights + (gradient + two Adam numbers) for the trainable ones

N_ALL = count_params(base_model)
full_gb = train_bytes(N_ALL, N_ALL) / 1e9
lora_gb = train_bytes(N_ALL, 294912) / 1e9
print("full fine-tuning: %.3f GB    LoRA r = 8: %.3f GB    ratio %.1f" % (full_gb, lora_gb, full_gb / lora_gb))
print("the part that depends on training: gradients + Adam = %.1f MB for full fine-tuning and %.2f MB for LoRA" % (12 * N_ALL / 1e6, 12 * 294912 / 1e6))
RESULTS["p6_full_gb"] = round(full_gb, 3); RESULTS["p6_lora_gb"] = round(lora_gb, 3)
```

The LoRA run still has to keep all the weights, so it does not use 420 times less memory: the 4 bytes per frozen weight (0.5 GB) stay. What disappears is the 12 bytes per parameter of gradients and Adam statistics (1.49 GB for full fine-tuning, 3.5 MB for LoRA). The saving is therefore about a factor of 4 for a small model whose weights are in float32. For a big model whose frozen weights are stored in 4 bits (QLoRA, see the lesson) the saving is much larger.

It prints:

```text
full fine-tuning: 1.991 GB    LoRA r = 8: 0.501 GB    ratio 4.0
the part that depends on training: gradients + Adam = 1493.3 MB for full fine-tuning and 3.54 MB for LoRA
```

Running the check on this answer prints:

```text
✓ train_bytes(10, 10) is 160 and train_bytes(10, 0) is 40
✓ full fine-tuning needs about 1.99 GB
✓ LoRA needs about 0.50 GB
```

   **What you should see.**

   The check passes three ticks. The answer prints `full fine-tuning: 1.991 GB    LoRA r = 8: 0.501 GB    ratio 4.0` and `the part that depends on training: gradients + Adam = 1493.3 MB for full fine-tuning and 3.54 MB for LoRA`.

   *What it means.* Full fine-tuning of GPT-2 small needs about 2 GB for these numbers, LoRA about 0.5 GB. Both fit easily on a laptop; for a 7-billion-parameter model the same formula gives 112 GB against about 28 GB, and that is the difference between a rack of GPUs and one card. In the next step we also *measure* the optimizer's memory. `RESULTS["p6_full_gb"]` is 1.991 and `RESULTS["p6_lora_gb"]` is 0.501.

6. **Step 6.6. Train both and compare.** Now we train both on the 24 training examples for 2 epochs with the loop of Step 6.1. **Full fine-tuning** uses the learning rate of Part 2 (3e-5). **LoRA** uses 1e-3, more than thirty times larger: the thin matrices start from nothing and must move a long way in only 48 steps, and the lesson's typical 1e-4 to 2e-4 is for runs of thousands of steps. (Using 3e-5 for LoRA is the most common mistake: the loss hardly moves and it looks as if LoRA does not work.) We report the validation loss, the `style_rate`, the loss on the general paragraph (forgetting), the time, and the measured size of Adam's statistics (the optimizer keeps two tensors per trainable parameter). A second cell shows what each model answers to the first validation question.

```python
def p6_opt_state_mb(opt):
    """Measured size of the two Adam statistics kept by the optimizer, in megabytes."""
    return sum(v.numel() * 4 for st in opt.state.values() for k, v in st.items() if k in ("exp_avg", "exp_avg_sq")) / 1e6

# --- full fine-tuning (every parameter trains), lr 3e-5
torch.manual_seed(P6_SEED)
full_model = copy.deepcopy(base_model).to(device)
for p in full_model.parameters(): p.requires_grad = True
sec_full, val_full, opt_full = p6_train(full_model, list(full_model.parameters()), P6_FULL_LR)
# --- LoRA r = 8 on c_attn, lr 1e-3 (lora_model was built in Step 6.4)
sec_lora, val_lora, opt_lora = p6_train(lora_model, lora_params_list, P6_LORA_LR)

gen_full, gen_lora = text_loss(full_model, GENERAL_TEXT), text_loss(lora_model, GENERAL_TEXT)
st_full, st_lora = style_rate(full_model), style_rate(lora_model)
print("%-22s %12s %8s %12s %10s %12s %14s" % ("method", "trainable", "seconds", "val loss", "style", "general text", "Adam state MB"))
print("%-22s %12s %8s %12.3f %10.2f %12.3f %14s" % ("GPT-2, no training", "0", "-", RESULTS["p6_base_val"], RESULTS["p6_base_style"], RESULTS["p6_base_general"], "-"))
print("%-22s %12d %8.0f %12.3f %10.2f %12.3f %14.1f" % ("full fine-tuning", count_params(full_model, True), sec_full, val_full[-1], st_full, gen_full, p6_opt_state_mb(opt_full)))
print("%-22s %12d %8.0f %12.3f %10.2f %12.3f %14.2f" % ("LoRA r = 8", count_params(lora_model, True), sec_lora, val_lora[-1], st_lora, gen_lora, p6_opt_state_mb(opt_lora)))
print("val loss after each epoch:  full", [round(v, 3) for v in val_full], "  LoRA", [round(v, 3) for v in val_lora])
RESULTS.update(p6_full_val=round(val_full[-1], 3), p6_lora_val=round(val_lora[-1], 3), p6_full_sec=round(sec_full), p6_lora_sec=round(sec_lora),
               p6_full_style=st_full, p6_lora_style=st_lora, p6_full_general=round(gen_full, 3), p6_lora_general=round(gen_lora, 3),
               p6_full_adam_mb=round(p6_opt_state_mb(opt_full), 1), p6_lora_adam_mb=round(p6_opt_state_mb(opt_lora), 2),
               p6_full_val_by_epoch=[round(v, 3) for v in val_full], p6_lora_val_by_epoch=[round(v, 3) for v in val_lora])
```

It prints:

```text
method                    trainable  seconds     val loss      style general text  Adam state MB
GPT-2, no training                0        -        2.816       0.00        3.509              -
full fine-tuning          124439808       66        0.193       1.00        3.620          995.5
LoRA r = 8                   294912       16        0.270       1.00        3.574           2.36
val loss after each epoch:  full [0.267, 0.193]   LoRA [0.447, 0.27]
```

```python
q_, _ = INSTR_VAL[0]
print("Question:", q_)
for name, m in (("GPT-2 (no training)", base_model), ("full fine-tuning", full_model), ("LoRA r = 8", lora_model)):
    print("%-20s -> %r" % (name, generate(m, format_prompt(q_), 30)))
for name, m, key in (("full fine-tuning", full_model, "full"), ("LoRA r = 8", lora_model, "lora")):          # style is easy; are the facts right too?
    exact = sum(generate(m, format_prompt(q), 40).strip() == a for q, a in INSTR_VAL)
    print("%-20s answers identical to the reference on the 8 validation questions: %d" % (name, exact)); RESULTS["p6_%s_exact" % key] = exact
```

It prints:

```text
Question: How many millilitres are in a litre?
GPT-2 (no training)  -> 'A litre of water is about 1.5 litres.\n### Question:\nHow many litres of water are in a litre?\n'
full fine-tuning     -> 'Sure! There are one thousand millilitres in a litre. Hope that helps.'
LoRA r = 8           -> 'Sure! There are six millilitres in a litre. Hope that helps.'
full fine-tuning     answers identical to the reference on the 8 validation questions: 4
LoRA r = 8           answers identical to the reference on the 8 validation questions: 3
```

   **What you should see.**

   The table has three rows. Without training: validation loss 2.816, style 0.00, general text 3.509. Full fine-tuning: all 124,439,808 numbers trainable, 66 s, validation loss 0.193, style 1.00, general text 3.620, Adam state 995.5 MB. LoRA: 294,912 trainable, 16 s, validation loss 0.270, style 1.00, general text 3.574, Adam state 2.36 MB. The validation loss after epoch 1 and 2 is [0.267, 0.193] for full fine-tuning and [0.447, 0.27] for LoRA. The second cell shows the base model rambling on (`### Question: ...`) and both trained models answering in the style "Sure! ... Hope that helps.", and counts how many of the 8 validation answers are word for word the reference answer: 4 for full fine-tuning and 3 for LoRA.

   *What it means.* Be honest about what this says. **Both methods learn the style completely** (style_rate 1.0 and 1.0, up from 0.0). **Full fine-tuning fits the validation answers better** (0.193 against 0.270): training about 420 times more numbers buys a lower loss on this data, so LoRA is cheaper, not free. LoRA ran about 4.1 times faster here and its optimizer state is 2.36 MB instead of 995.5 MB, which matches the arithmetic of Step 6.5 (8 bytes for each trainable parameter). **The facts are another matter**: the style is perfect for both, but only 4 (full) and 3 (LoRA) of the 8 answers are exactly right, and in the example LoRA writes "six millilitres in a litre". Fine-tuning on 24 examples teaches a style, not new knowledge, and a 124-million-parameter model gets some facts wrong. Both observations are about one run on 8 questions, so treat the 4 against 3 as a tie. On forgetting, the loss on the general paragraph rose by 0.111 for full fine-tuning and by 0.065 for LoRA: LoRA forgot a little less, but both changes are small with only 48 updates and one paragraph, so do not read this as proof that LoRA never forgets. (Times depend on the machine and on what else it is running.)

7. **Step 6.7. How much rank is enough?** The rank r is the main choice. Too small and the correction cannot express what the data needs; larger costs more parameters, and past some point buys nothing. We train five fresh models with r = 1, 2, 4, 8 and 16 (α = 2r, so the scale stays 2), with identical data, order and seed, and compare validation loss, style and forgetting. The r = 8 run repeats Step 6.6 exactly; the assertion checks that it gives the same number again (with the same seed, a run is repeatable).

```python
def p6_run_lora(r, alpha=None, lr=P6_LORA_LR):
    """Train a fresh LoRA model of rank r (alpha = 2 r, so alpha / r = 2 for every r). Returns the model and its numbers."""
    torch.manual_seed(P6_SEED)
    m = copy.deepcopy(base_model).to(device)
    params = add_lora(m, r, alpha or 2 * r)
    sec, val, _ = p6_train(m, params, lr)
    return m, dict(r=r, params=count_params(m, True), sec=sec, val=val[-1], style=style_rate(m, INSTR_VAL[:2] if QUICK else None), general=text_loss(m, GENERAL_TEXT))

sweep = {}
print("%4s %10s %8s %10s %7s %14s" % ("r", "parameters", "seconds", "val loss", "style", "general text"))
for r_ in (1, 2, 4, 8, 16):
    m_, row = p6_run_lora(r_); sweep[r_] = row
    print("%4d %10d %8.0f %10.3f %7.2f %14.3f" % (r_, row["params"], row["sec"], row["val"], row["style"], row["general"]))
    if r_ == 8: lora8_model = m_            # keep it for the next steps
    del m_; gc.collect()
print("the r = 8 run repeats Step 6.6: val loss %.3f against %.3f there" % (sweep[8]["val"], RESULTS["p6_lora_val"]))
assert abs(sweep[8]["val"] - RESULTS["p6_lora_val"]) < 2e-3          # same seed, same order, same result
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
rs = list(sweep)
ax[0].plot([sweep[r]["params"] for r in rs], [sweep[r]["val"] for r in rs], "o-"); ax[0].set_xscale("log")
for r in rs: ax[0].annotate("r=%d" % r, (sweep[r]["params"], sweep[r]["val"]), textcoords="offset points", xytext=(4, 4))
ax[0].axhline(RESULTS["p6_full_val"], color="gray", ls="--", label="full fine-tuning"); ax[0].set_xlabel("trainable parameters"); ax[0].set_ylabel("validation loss"); ax[0].legend()
ax[1].plot(rs, [sweep[r]["style"] for r in rs], "o-"); ax[1].set_xscale("log", base=2); ax[1].set_xlabel("rank r"); ax[1].set_ylabel("style_rate"); ax[1].set_ylim(-0.05, 1.05)
plt.tight_layout(); plt.show()
RESULTS["p6_sweep_val"] = {r: round(sweep[r]["val"], 3) for r in rs}; RESULTS["p6_sweep_style"] = {r: sweep[r]["style"] for r in rs}
RESULTS["p6_sweep_params"] = {r: sweep[r]["params"] for r in rs}; RESULTS["p6_sweep_general"] = {r: round(sweep[r]["general"], 3) for r in rs}
```

It prints:

```text
   r parameters  seconds   val loss   style   general text
   1      36864       14      0.525    1.00          3.552
   2      73728       18      0.334    1.00          3.544
   4     147456       16      0.299    1.00          3.578
   8     294912       15      0.270    1.00          3.574
  16     589824       16      0.310    1.00          3.586
the r = 8 run repeats Step 6.6: val loss 0.270 against 0.270 there
```

   **What you should see.**

   The table shows, for r = 1, 2, 4, 8, 16: parameters 36,864, 73,728, 147,456, 294,912 and 589,824; validation loss 0.525, 0.334, 0.299, 0.270 and 0.310; the style_rate is 1.00 for all five; the general-text loss is 3.552, 3.544, 3.578, 3.574 and 3.586. The line `the r = 8 run repeats Step 6.6` shows the same validation loss twice. The figure plots validation loss against the number of trainable parameters (with the full fine-tuning level as a dashed line) and style_rate against r.

   *What it means.* Three honest readings. **(1) The style is learned even by rank 1** (36,864 numbers, style 1.00): copying one fixed opening and closing is an easy change. **(2) The validation loss falls steeply from r = 1 to r = 4** (0.525 to 0.299) and then flattens. **(3) r = 16 is *worse* than r = 8** (0.310 against 0.270) although it has twice the parameters, and no rank reaches the full fine-tuning level of 0.193. We have one seed and 8 validation examples, so a difference of 0.04 is within the noise of this experiment; what we can claim is the shape (a steep gain, then a plateau), not the exact winner. A bigger rank is not automatically better with this little data and this few steps.

8. **Step 6.8. Merge the adapter into the weights.** A LoRA model has extra layers to run. For deployment you can remove them: because (x A) B scaled by α / r is the same as x multiplied by (A B scaled by α / r), you can **merge** the update into the weight, W' = W + (α / r) · A B, and use the original architecture with new numbers. The merged model is exactly the same function, with no extra computation at answer time.

   *Exercise 6.8:* Write `merged_weight(W, A, B, scale)` that returns the merged matrix. Then the cell below uses it to build a copy of GPT-2 whose `c_attn` weights are merged (the bias does not change) and compares it with the LoRA model. Why it matters: this is how an adapter becomes an ordinary model that any tool can load. The check tests your function on a toy example.

```python
def merged_weight(W, A, B, scale):
    """The weight matrix of the merged layer: the frozen W plus the update that LoRA learned."""
    return None   # YOUR CODE: W + scale * (A @ B)   (W is (d_in, d_out), A is (d_in, r), B is (r, d_out))

W_, A_, B_ = torch.randn(5, 7), torch.randn(5, 2), torch.randn(2, 7)
if merged_weight(W_, A_, B_, 2.0) is None:
    print("Fill in merged_weight above, then run this cell again.")
else:
    check("the merged matrix has the shape of W", tuple(merged_weight(W_, A_, B_, 2.0).shape) == (5, 7), "A @ B has shape (d_in, d_out)")
    check("x @ merged equals x @ W + 2 * (x @ A) @ B", torch.allclose(torch.ones(1, 5) @ merged_weight(W_, A_, B_, 2.0), torch.ones(1, 5) @ W_ + 2 * (torch.ones(1, 5) @ A_) @ B_, atol=1e-4), "merged = W + scale * A @ B")
```

   A possible answer (try first, then compare):

```python
def merged_weight(W, A, B, scale):
    return W + scale * (A @ B)

merged_model = copy.deepcopy(base_model).to(device)
for blk_m, blk_l in zip(merged_model.transformer.h, lora8_model.transformer.h):
    lay = blk_l.attn.c_attn
    with torch.no_grad(): blk_m.attn.c_attn.weight.copy_(merged_weight(lay.base.weight, lay.A, lay.B, lay.scale))     # the bias is unchanged
merged_model.eval(); lora8_model.eval()
ids_ = tok(format_prompt("How many days are in a week?"), return_tensors="pt").to(device)
with torch.no_grad(): diff_logits = float((merged_model(**ids_).logits - lora8_model(**ids_).logits).abs().max())
same_text = [generate(merged_model, format_prompt(q), 30) == generate(lora8_model, format_prompt(q), 30) for q, _ in INSTR_VAL]
print("merged model: %d parameters, %d trainable (no extra layers)" % (count_params(merged_model), count_params(merged_model, True)))
print("largest difference between the logits of the merged and the LoRA model: %.2e" % diff_logits)
print("identical greedy answers on the 8 validation questions: %d of 8;  val loss merged %.3f, LoRA %.3f" % (sum(same_text), p6_val_loss(merged_model), p6_val_loss(lora8_model)))
assert diff_logits < 1e-3 and all(same_text)
RESULTS["p6_merge_diff"] = float("%.2e" % diff_logits); RESULTS["p6_merge_same"] = sum(same_text)
```

The result is a plain GPT-2 with changed numbers: it has 124,439,808 parameters and no extra layers. The logits of the two models differ only by float rounding, because the two ways of computing the same sum add the numbers in a different order.

It prints:

```text
merged model: 124439808 parameters, 124439808 trainable (no extra layers)
largest difference between the logits of the merged and the LoRA model: 2.14e-04
identical greedy answers on the 8 validation questions: 8 of 8;  val loss merged 0.270, LoRA 0.270
```

Running the check on this answer prints:

```text
✓ the merged matrix has the shape of W
✓ x @ merged equals x @ W + 2 * (x @ A) @ B
```

   **What you should see.**

   The check passes two ticks. The answer prints `merged model: 124439808 parameters, 124439808 trainable (no extra layers)`, then a largest logit difference of `0.000214`, and `identical greedy answers on the 8 validation questions: 8 of 8` with the same validation loss for both models; the assertion passes.

   *What it means.* A difference of 0.000214 in the logits is rounding (the two ways of adding the numbers round differently), not a different model, and the answers are the same words. (The merged copy shows 124,439,808 as trainable because we created it fresh; nothing is frozen in it. It is a normal model.) You can now save one ordinary model, or keep the 1 MB adapter apart and switch tasks: the next steps show why that is useful. `RESULTS["p6_merge_diff"]` is 0.000214.

9. **Step 6.9. Cross-check with the peft library.** In practice you will not write the layer yourself: the `peft` library (Parameter-Efficient Fine-Tuning) does it for any model. We check that our work agrees with it, in two ways. **(1)** `LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn"], fan_in_fan_out=True)` builds a peft model whose trainable parameters we count (the argument `fan_in_fan_out=True` tells peft that GPT-2 stores weights as (in, out): the same Conv1D detail as in Step 6.3). **(2)** We copy **our trained** A and B into the peft model (peft stores them transposed) and compare the logits. In Colab run `!pip install peft` first (the setup cell does it); if the library is missing the cell tells you.

```python
try:
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn"], fan_in_fan_out=True, lora_dropout=0.0, bias="none")   # fan_in_fan_out=True because GPT-2 uses Conv1D
    peft_model = get_peft_model(copy.deepcopy(base_model).to(device), cfg)
    n_peft = count_params(peft_model, trainable_only=True)
    print("peft trainable parameters:", n_peft, "  ours:", RESULTS["p6_trainable"])
    assert n_peft == RESULTS["p6_trainable"]
    # copy OUR trained A and B into the peft model (peft stores A as (r, in) and B as (out, r): the transposes of ours)
    for blk_p, blk_o in zip(peft_model.base_model.model.transformer.h, lora8_model.transformer.h):
        with torch.no_grad():
            blk_p.attn.c_attn.lora_A["default"].weight.copy_(blk_o.attn.c_attn.A.T)
            blk_p.attn.c_attn.lora_B["default"].weight.copy_(blk_o.attn.c_attn.B.T)
    peft_model.eval()
    with torch.no_grad(): diff_peft = float((peft_model(**ids_).logits - lora8_model(**ids_).logits).abs().max())
    print("largest difference between peft (with our numbers) and our LoRA model: %.2e" % diff_peft)
    assert diff_peft < 1e-3
    RESULTS["p6_peft_params"] = n_peft; RESULTS["p6_peft_diff"] = float("%.2e" % diff_peft)
    del peft_model; gc.collect()
except ImportError:
    print("peft is not installed here: run  pip install peft  (in Colab:  !pip install peft ) and run this cell again.")
```

It prints:

```text
peft trainable parameters: 294912   ours: 294912
largest difference between peft (with our numbers) and our LoRA model: 0.00e+00
```

   **What you should see.**

   The cell prints `peft trainable parameters: 294912   ours: 294912` and `largest difference between peft (with our numbers) and our LoRA model: 0.0`; both assertions pass.

   *What it means.* The count is identical, 294,912, and with our numbers inside, peft computes the same logits as our layer (difference 0.0). So the library is no magic: it is the layer you wrote, plus conveniences (saving only the adapter, switching adapters, other target layers, dropout, other methods). Which one to use in a project? peft, because it is tested and maintained; but you now know what it does and how to read its errors.

10. **Step 6.10. Is the update really low rank? A look with the SVD.** The idea behind LoRA is that the change a fine-tuning run makes is *low rank*: it needs few directions. The **singular value decomposition (SVD)** of a matrix lists its directions and their strengths (singular values, largest first); the **rank** is the number of non-zero ones, and the **energy** of a direction is its singular value squared. We look at the update ΔW of three layers (blocks 0, 5 and 11) in two models: LoRA's update (A B scaled, which has rank at most 8 by construction) and the update of full fine-tuning (trained weight minus original weight, which is free to have any rank). For each we report the largest singular value, the rank, the number of directions that hold 90 percent of the energy, and the share of the energy in the top 8 directions.

   *Exercise 6.10:* Write `energy_rank(s, frac=0.9)`, which receives singular values `s` (sorted from large to small) and returns the smallest number of them whose squares add up to at least `frac` of the total of the squares. Hint: `torch.cumsum` of `s ** 2`, divided by its sum. Why it matters: it turns "how low rank is this update?" into one number you can compare. The check tests three small cases (the second shows why the comparison must allow for rounding).

```python
def energy_rank(s, frac=0.9):
    """The smallest number of singular values whose squares add up to at least `frac` of the sum of all the squares."""
    return None   # YOUR CODE: s is a 1-D tensor sorted from large to small; use torch.cumsum on s ** 2

if energy_rank(torch.tensor([3.0, 1.0, 0.1])) is None:
    print("Fill in energy_rank above, then run this cell again.")
else:
    check("[3, 1, 0.1]: 9 of 10.01 is 89.9 percent, so two values are needed for 90 percent", energy_rank(torch.tensor([3.0, 1.0, 0.1])) == 2, "squares are 9, 1, 0.01; the first one alone gives 0.899")
    check("ten equal values need 9 for 90 percent", energy_rank(torch.ones(10)) == 9, "9 of 10 squares is exactly 90 percent: use >=, not >")
    check("one big value needs 1", energy_rank(torch.tensor([5.0, 0.0, 0.0])) == 1, "the first value already holds 100 percent")
```

   A possible answer (try first, then compare):

```python
def energy_rank(s, frac=0.9):
    c = torch.cumsum(s ** 2, 0) / (s ** 2).sum()
    return int((c >= frac - 1e-9).nonzero()[0]) + 1

rows = []
print("%-7s %-20s %12s %12s %16s %18s" % ("layer", "update", "largest sv", "rank (>1e-4)", "90% energy at", "energy in top 8"))
for li in (0, 5, 11):
    lay = lora8_model.transformer.h[li].attn.c_attn
    dW_lora = (lay.scale * lay.A @ lay.B).detach()
    dW_full = (full_model.transformer.h[li].attn.c_attn.weight - base_model.transformer.h[li].attn.c_attn.weight).detach()
    for name, dW in (("LoRA r = 8", dW_lora), ("full fine-tuning", dW_full)):
        s = torch.linalg.svdvals(dW)
        top8 = float((s[:8] ** 2).sum() / (s ** 2).sum())
        rows.append((li, name, float(s[0]), int((s > 1e-4 * s[0]).sum()), energy_rank(s), top8))
        print("%-7d %-20s %12.3f %12d %16d %17.1f%%" % (li, name, float(s[0]), int((s > 1e-4 * s[0]).sum()), energy_rank(s), 100 * top8))
    if li == 5: s_lora5, s_full5 = torch.linalg.svdvals(dW_lora), torch.linalg.svdvals(dW_full)
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.semilogy(range(1, 33), s_full5[:32], "o-", label="full fine-tuning, layer 5"); ax.semilogy(range(1, 33), s_lora5[:32].clamp_min(1e-9), "s-", label="LoRA r = 8, layer 5")
ax.set_xlabel("index of the singular value"); ax.set_ylabel("singular value"); ax.legend(); plt.tight_layout(); plt.show()
RESULTS["p6_svd"] = {"layer%d %s" % (r[0], r[1]): dict(rank=r[3], e90=r[4], top8=round(r[5], 3)) for r in rows}
assert all(r[3] <= 8 for r in rows if r[1] == "LoRA r = 8")
```

`(c >= frac - 1e-9)` finds the first position where the running share reaches `frac`; the tiny 1e-9 stops rounding errors from rejecting an exact 90 percent.

It prints:

```text
layer   update                 largest sv rank (>1e-4)    90% energy at    energy in top 8
0       LoRA r = 8                  1.624            8                4             100.0%
0       full fine-tuning            0.126          768               89              53.8%
5       LoRA r = 8                  1.950            8                2             100.0%
5       full fine-tuning            0.146          768               82              49.9%
11      LoRA r = 8                  3.848            8                1             100.0%
11      full fine-tuning            0.223          768               55              60.6%
```

Running the check on this answer prints:

```text
✓ [3, 1, 0.1]: 9 of 10.01 is 89.9 percent, so two values are needed for 90 percent
✓ ten equal values need 9 for 90 percent
✓ one big value needs 1
```

   **What you should see.**

   The check passes three ticks. The table has six rows (layers 0, 5, 11; LoRA and full): for LoRA the rank is 8 in every layer and 90 percent of the energy sits in 4, 2 and 1 directions (layers 0, 5 and 11), with 100 percent in the top 8. For full fine-tuning the rank is 768 in every layer, 90 percent of the energy needs 89, 82 and 55 directions, and the top 8 hold only 54, 50 and 61 percent. The figure compares the first 32 singular values for layer 5 on a log scale.

   *What it means.* LoRA's update is, by construction, concentrated: layer 11's has 90 percent of its energy in a *single* direction. The full fine-tuning update is **not** low rank in the strict sense (rank 768, and 55 to 89 directions needed for 90 percent), and the best 8-direction approximation would keep only about half of its energy. So the honest statement is not "fine-tuning updates are tiny": it is that a low-rank update can capture *enough* of what a task needs, as the style_rate of 1.00 and the validation loss of 0.270 (against 0.193) show, and that the full update spreads over many more directions (this experiment cannot tell how much of that is useful and how much is noise from only 24 examples). Hu et al. argued that the useful change has low *intrinsic* rank; this small experiment supports "enough", not "identical".

11. **Step 6.11. The size of an adapter.** The last advantage is the file. To share what you taught the model you only need the adapter: the 24 tensors A and B (12 layers, two each). We save them with `torch.save` to a temporary file and measure its size, and do the same for the full model, with the arithmetic (4 bytes per float32 number) next to it.

```python
import tempfile
def file_mb(state):
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "ckpt.pt"); torch.save(state, f); return os.path.getsize(f) / 1e6
adapter_state = {k: v for k, v in lora8_model.state_dict().items() if k.endswith(".A") or k.endswith(".B")}
mb_adapter = file_mb(adapter_state); mb_full = file_mb(full_model.state_dict())
print("adapter: %d tensors, %d numbers, file %.2f MB (computed 4 bytes per number: %.2f MB)" % (len(adapter_state), sum(v.numel() for v in adapter_state.values()), mb_adapter, 4 * 294912 / 1e6))
print("full checkpoint: file %.1f MB (computed %.1f MB)" % (mb_full, 4 * count_params(full_model) / 1e6))
print("the full checkpoint is %.0f times larger; ten adapters (ten tasks) take %.1f MB, ten full copies %.0f MB" % (mb_full / mb_adapter, 10 * mb_adapter, 10 * mb_full))
RESULTS["p6_adapter_mb"] = round(mb_adapter, 2); RESULTS["p6_full_mb"] = round(mb_full, 1); RESULTS["p6_size_ratio"] = round(mb_full / mb_adapter)
```

It prints:

```text
adapter: 24 tensors, 294912 numbers, file 1.19 MB (computed 4 bytes per number: 1.18 MB)
full checkpoint: file 497.8 MB (computed 497.8 MB)
the full checkpoint is 420 times larger; ten adapters (ten tasks) take 11.9 MB, ten full copies 4978 MB
```

   **What you should see.**

   The cell prints `adapter: 24 tensors, 294912 numbers, file 1.19 MB (computed 1.18 MB ...)`, `full checkpoint: file 497.8 MB (computed 497.8 MB)` and `the full checkpoint is 420 times larger; ten adapters (ten tasks) take about 12 MB, ten full copies about 4,978 MB`. The sizes match the arithmetic to within the small header of the file.

   *What it means.* One school server could keep one 500 MB GPT-2 and a 1.2 MB adapter for each subject, the handbook help desk, the report-card tone, the language club, and swap them per request, instead of ten full copies. This is why LoRA, not full fine-tuning, is the default for sharing models. `RESULTS["p6_size_ratio"]` is 420. Part 7 asks you to use the numbers of this part for your own plan.

### Part 7. Plan your own project and reflect (20 min)

*Lesson sections: "Doing it well" and "Key takeaways".* You can now fine-tune, align and adapt a small language model. The hard part of a real project is the decision before the code. ==In this part you fill in a one-page design card for two scenarios from a school, decide for six situations whether a prompt, retrieval (RAG) or fine-tuning fits, build an honest table of what worked and what did not in your lab, and answer four reflection questions.== Nothing here trains a model: it is thinking, reading your own numbers and writing, and it links to your Capstone and to Week 15.

**Imagine first.** Think of something at your school that a language model could help with: answering the office's repeated questions, drafting comments, checking homework against a rubric. Before you read on, write in one sentence what the model should do and which of the three tools (a prompt, retrieval of documents, or fine-tuning) you would try first. At the end of the part you check whether the decision rule of Step 7.2 agrees.

```python
# Imagine first: the shape of your own project
my_project_task = ""      # one sentence: what should the model do for people at your school? (words)
my_project_method = ""    # prompt, RAG or fine-tune? (one of these words)
```

1. **Step 7.1. A design card for your own project.** A good project starts on one page. The **design card** has six fields: the **task** (what the model does and for whom), the **data** (where it comes from, how many, who checked it), the **base model** (which one and why), the **method** (prompt, RAG, LoRA or full fine-tuning), the **metric** (a number you will measure on examples the model never saw, with the value that counts as good enough) and the **risks** (what could go wrong and who checks). Writing the card first often changes the plan: a metric you cannot measure, or data you do not have, is better discovered before you train.

   *Exercise 7.1:* Fill in the two cards, six fields each, with at least 8 words per field. **`card_handbook`:** a help desk that answers students' and parents' questions from the school handbook. **`card_reports`:** a tool that turns a teacher's rough notes into a report-card comment in the school's tone. Why it matters: the two scenarios look alike but need different methods. The check tests that every field is long enough, that the method fits the facts of the scenario (retrieval for the handbook, fine-tuning for the report tone) and that each metric contains a number.

```python
CARD_FIELDS = ("task", "data", "base_model", "method", "metric", "risks")
card_help = {"task": "what the model does and for whom", "data": "where the examples or documents come from, how many, who checked them",
             "base_model": "which model you start from and why", "method": "prompt, RAG, LoRA or full fine-tuning, and why that one",
             "metric": "a number you will measure on held-out examples, with the value that counts as good enough", "risks": "what could go wrong, and who checks"}

card_handbook = {f: "" for f in CARD_FIELDS}     # YOUR TEXT: a help desk that answers students' questions from the school handbook
card_reports = {f: "" for f in CARD_FIELDS}      # YOUR TEXT: a tool that turns a teacher's rough notes into a report-card comment in the school's own tone

def card_problems(card, must_method):
    bad = [f for f in CARD_FIELDS if len(card[f].split()) < 8]
    return bad, any(w in card["method"].lower() for w in must_method), any(ch.isdigit() for ch in card["metric"])

if all(v == "" for c in (card_handbook, card_reports) for v in c.values()):
    print("Fill in the two cards above (six fields each), then run this cell again.")
else:
    for nm, c, words in (("handbook", card_handbook, ("rag", "retriev")), ("reports", card_reports, ("lora", "fine-tun", "fine tun"))):
        bad, method_ok, metric_ok = card_problems(c, words)
        check("%s: every field has at least 8 words" % nm, not bad, "too short: " + ", ".join(bad))
        check("%s: the method is the one the facts point to (%s)" % (nm, " or ".join(words)), method_ok, "facts that change point to retrieval; a style or format that must always hold points to fine-tuning")
        check("%s: the metric contains a number" % nm, metric_ok, "for example 'at least 90 percent of 50 held-out questions answered correctly'")
```

   A possible answer (try first, then compare):

```python
card_handbook = {
    "task": "Answer students' and parents' questions about the school handbook (uniform, timetable, rules, dates) in a polite way and always say which page the answer comes from.",
    "data": "The 60-page handbook of this year, split into paragraphs, plus 80 real questions from the office that two teachers answered, kept aside to test the tool.",
    "base_model": "A small open chat model that runs on the school server, because the questions contain names and the data must stay in the school.",
    "method": "RAG with a ready-made model: the handbook changes every year, so the facts must come from retrieval and not from the weights; a prompt fixes the tone and the rule to cite the page.",
    "metric": "Share of the 80 held-out questions answered correctly with the right page cited; good enough means at least 90 percent, and 100 percent of 'I do not know' when the handbook is silent.",
    "risks": "The tool may answer from memory when retrieval fails or quote an old rule; the office reads a sample of 20 answers each month and the tool never decides on discipline.",
}
card_reports = {
    "task": "Turn a teacher's rough notes about a student (three or four bullet points) into two polite, honest sentences in the school's own report-card tone.",
    "data": "About 600 past comments written by teachers, with the notes they started from, names removed and approved by the head of year; 100 kept aside for testing.",
    "base_model": "A small instruction-tuned model of about one billion parameters, because the task is the same every time and it must run on a school laptop.",
    "method": "LoRA fine-tuning on the 500 training pairs: the tone is a habit that should hold every time without a long prompt, and the adapter is a file of a few megabytes per subject.",
    "metric": "Two teachers rate 100 held-out comments blind (tone right, facts match the notes); good enough is at least 85 percent accepted without edits, with 0 invented facts.",
    "risks": "The model may invent praise or a fact that is not in the notes, or copy phrases from one real student; every comment is read and signed by the teacher before it is sent.",
}
for nm, c in (("handbook help desk", card_handbook), ("report-card comments", card_reports)):
    print("== %s ==" % nm)
    for f in CARD_FIELDS: print("  %-10s %s" % (f, c[f]))
```

Read the two cards side by side. The handbook changes every year and the answers must name a page, so the facts live in documents and are retrieved; a prompt sets the tone. The report tone is a *habit* that must hold every time, with hundreds of examples available, so it is a fine-tuning (LoRA) job, as in Part 6, and the adapter is a file of a few megabytes. Both cards keep a person in charge of what leaves the building.

It prints:

```text
== handbook help desk ==
  task       Answer students' and parents' questions about the school handbook (uniform, timetable, rules, dates) in a polite way and always say which page the answer comes from.
  data       The 60-page handbook of this year, split into paragraphs, plus 80 real questions from the office that two teachers answered, kept aside to test the tool.
  base_model A small open chat model that runs on the school server, because the questions contain names and the data must stay in the school.
  method     RAG with a ready-made model: the handbook changes every year, so the facts must come from retrieval and not from the weights; a prompt fixes the tone and the rule to cite the page.
  metric     Share of the 80 held-out questions answered correctly with the right page cited; good enough means at least 90 percent, and 100 percent of 'I do not know' when the handbook is silent.
  risks      The tool may answer from memory when retrieval fails or quote an old rule; the office reads a sample of 20 answers each month and the tool never decides on discipline.
== report-card comments ==
  task       Turn a teacher's rough notes about a student (three or four bullet points) into two polite, honest sentences in the school's own report-card tone.
  data       About 600 past comments written by teachers, with the notes they started from, names removed and approved by the head of year; 100 kept aside for testing.
  base_model A small instruction-tuned model of about one billion parameters, because the task is the same every time and it must run on a school laptop.
  method     LoRA fine-tuning on the 500 training pairs: the tone is a habit that should hold every time without a long prompt, and the adapter is a file of a few megabytes per subject.
  metric     Two teachers rate 100 held-out comments blind (tone right, facts match the notes); good enough is at least 85 percent accepted without edits, with 0 invented facts.
  risks      The model may invent praise or a fact that is not in the notes, or copy phrases from one real student; every comment is read and signed by the teacher before it is sent.
```

Running the check on this answer prints:

```text
✓ handbook: every field has at least 8 words
✓ handbook: the method is the one the facts point to (rag or retriev)
✓ handbook: the metric contains a number
✓ reports: every field has at least 8 words
✓ reports: the method is the one the facts point to (lora or fine-tun or fine tun)
✓ reports: the metric contains a number
```

   **What you should see.**

   The check gives three ticks for each card. The answer prints the two cards.

   *What it means.* A card is good when every field could be tested: the metric names a number and a held-out set, the data field says who checked it, the risks field names a person. Your own cards will differ; what matters is that they are specific to the scenario. The numbers on the cards (80 questions, 90 percent, 600 comments) are examples, not measurements from this course.

2. **Step 7.2. Prompt, RAG or fine-tune?** Three tools solve different problems. A **prompt** is the cheapest: put the instructions and a few examples in the context and try. **Retrieval (RAG)** puts the *facts* in the context at question time, so they can change without any training and the answer can cite its source. **Fine-tuning** changes the *behaviour* (style, format, a skill) so that it holds without a long prompt and in a small model. A rule that works in most cases: if the facts change, retrieve; if the behaviour must always hold and you have hundreds or thousands of good examples, fine-tune; if it is a one-off or you have little data, try a prompt first. Part 6 gives the costs of the last option: a 1 MB adapter, about 4 times less training time here, but you still need good data.

   *Exercise 7.2:* For each of the six situations `S1` to `S6` write in `choice` one of "prompt", "RAG" or "fine-tune". Why it matters: choosing wrongly is the most expensive mistake in a project (fine-tuning facts that change every week, or retrieving when the real need is a fixed format). The check compares your choice with our key and prints a hint for each difference; a defensible different choice is worth a sentence in your report.

```python
SITUATIONS = {
    "S1": "Answer questions about this week's cafeteria menu and timetable changes, which the office updates every few days.",
    "S2": "Translate one letter to parents into three languages, once, tomorrow.",
    "S3": "Make a small model that must run on a phone always answer in the same JSON format with six fields; there are 5,000 labelled examples.",
    "S4": "Answer from 300 pages of school policies and say on which page the answer is.",
    "S5": "Reply in the warm voice of the school counsellors; there are 800 examples of their replies and the voice must hold on every answer.",
    "S6": "Check whether a student's answer matches a three-sentence rubric; there are only a few dozen answers a week.",
}
choice = {k: "" for k in SITUATIONS}      # YOUR CODE: for every situation write "prompt", "RAG" or "fine-tune"

KEY = {"S1": "RAG", "S2": "prompt", "S3": "fine-tune", "S4": "RAG", "S5": "fine-tune", "S6": "prompt"}
WHY = {"S1": "the facts change every few days: put them in the context, do not bake them into weights", "S2": "one-off and no data: a good prompt is enough",
       "S3": "a fixed format learned from thousands of examples, in a small model: fine-tune", "S4": "a big document the answer must cite: retrieve the right pages",
       "S5": "a voice is a behaviour: many examples and it must always hold, so fine-tune (LoRA)", "S6": "a short rule you can paste into the prompt; try the prompt first"}
if all(v == "" for v in choice.values()):
    print("Fill in choice for the six situations above, then run this cell again.")
else:
    for k in SITUATIONS: check("%s: %s" % (k, choice[k] or "(empty)"), choice[k].strip().lower() == KEY[k].lower(), "think: do the facts change? is it a behaviour? is it one-off? " )
```

   A possible answer (try first, then compare):

```python
choice = {"S1": "RAG", "S2": "prompt", "S3": "fine-tune", "S4": "RAG", "S5": "fine-tune", "S6": "prompt"}
for k in SITUATIONS: print("%s %-9s %s" % (k, choice[k], WHY[k]))
RESULTS["p7_choice"] = dict(choice)
```

Every row follows the rule of thumb above. Situations that really mix two needs (a counsellor's voice that also quotes the current timetable) combine the tools: fine-tune the voice and retrieve the facts. The point of the exercise is to name the reason (do the facts change? is it a behaviour? is it one-off?) before you pick.

It prints:

```text
S1 RAG       the facts change every few days: put them in the context, do not bake them into weights
S2 prompt    one-off and no data: a good prompt is enough
S3 fine-tune a fixed format learned from thousands of examples, in a small model: fine-tune
S4 RAG       a big document the answer must cite: retrieve the right pages
S5 fine-tune a voice is a behaviour: many examples and it must always hold, so fine-tune (LoRA)
S6 prompt    a short rule you can paste into the prompt; try the prompt first
```

Running the check on this answer prints:

```text
✓ S1: RAG
✓ S2: prompt
✓ S3: fine-tune
✓ S4: RAG
✓ S5: fine-tune
✓ S6: prompt
```

   **What you should see.**

   Six ticks if your choices match the key: S1 RAG, S2 prompt, S3 fine-tune, S4 RAG, S5 fine-tune, S6 prompt. The answer prints each choice with its reason.

   *What it means.* If you disagree with a row, you may be right: S6, for example, could become a fine-tune if the rubric were long and thousands of answers had to be checked every day. A fair answer says what would change your choice. Compare with your Imagine-first guess at the top of this part: did you start with the cheapest tool?

3. **Step 7.3. Your results and the honest findings table.** The most useful page of a project report is the one with the experiments that did not work. The first cell prints the Part 6 numbers that are in `RESULTS` in your session, so you can quote them (if you restarted the notebook, copy them from your report). In the exercise you write the findings table of your lab: at least four rows, each with what you **tried**, the **number** you measured and a **verdict**. You may add rows from Parts 1 to 5 (the masked loss, overfitting on 4 examples, forgetting and replay, the reward model, DPO on the toy).

```python
p6_keys = [k for k in RESULTS if k.startswith("p6_")]
if not p6_keys:
    print("RESULTS holds no Part 6 numbers in this session: run Part 6 first, or copy the numbers from your report.")
else:
    R6 = {k: RESULTS[k] for k in p6_keys}
    print("%-24s %s" % ("what", "your Part 6 number"))
    for k in ("p6_base_val", "p6_full_val", "p6_lora_val", "p6_full_sec", "p6_lora_sec", "p6_lora_share_pct", "p6_full_general", "p6_lora_general", "p6_size_ratio", "p6_merge_diff"):
        if k in R6: print("%-24s %s" % (k, R6[k]))
```

It prints:

```text
what                     your Part 6 number
p6_base_val              2.816
p6_full_val              0.193
p6_lora_val              0.27
p6_full_sec              66
p6_lora_sec              16
p6_lora_share_pct        0.237
p6_full_general          3.62
p6_lora_general          3.574
p6_size_ratio            420
p6_merge_diff            0.000214
```

   *Exercise 7.3:* Fill in `findings` with at least 4 dictionaries with the keys `tried`, `number` and `verdict`. Include at least one result that did **not** work or surprised you. Why it matters: a table of only successes teaches nothing and hides what you would do differently; the verdict column is where you practise saying "it did not". The check tests the structure, that each number field has a digit, and that at least one verdict is negative or surprised you.

```python
findings = []       # YOUR CODE: at least 4 entries; each is a dictionary with the keys "tried", "number" and "verdict", for example
                    # {"tried": "LoRA r = 8 instead of full fine-tuning", "number": "val loss 0.30 against 0.25", "verdict": "worked: almost as good with 0.24 percent of the parameters"}

if len(findings) == 0:
    print("Fill in findings above (at least 4 dictionaries), then run this cell again.")
else:
    check("at least 4 findings", len(findings) >= 4, "write one row for each experiment you want to remember")
    check("every row has tried, number and verdict", all(set(f) >= {"tried", "number", "verdict"} for f in findings), "three keys: tried, number, verdict")
    check("every number field contains a digit", all(any(ch.isdigit() for ch in str(f.get("number", ""))) for f in findings), "cite a measured value from your own run")
    check("at least one verdict says something did NOT work or surprised you", any(any(w in str(f.get("verdict", "")).lower() for w in ("not", "no ", "did not", "worse", "surpris", "failed", "same")) for f in findings), "an honest table has at least one negative or surprising result")
```

   A possible answer (try first, then compare):

```python
R_ = RESULTS
findings = [
    {"tried": "LoRA r = 8 against full fine-tuning (Step 6.6)", "number": "val loss %s against %s; %s of %s numbers trained" % (R_["p6_lora_val"], R_["p6_full_val"], format(R_["p6_trainable"], ","), format(R_["p6_total"], ",")),
     "verdict": "partly: both reach style_rate 1.0, but LoRA did not match the validation loss of full fine-tuning (it is cheaper, not free)"},
    {"tried": "five ranks from 1 to 16 (Step 6.7)", "number": "val loss %s (r = 1), %s (r = 8), %s (r = 16)" % (R_["p6_sweep_val"][1], R_["p6_sweep_val"][8], R_["p6_sweep_val"][16]),
     "verdict": "surprising: r = 16 was not better than r = 8; with one seed and 8 validation answers the plateau is the finding, not the winner"},
    {"tried": "loss on the general paragraph after training (Step 6.6)", "number": "%s before, %s full, %s LoRA" % (R_["p6_base_general"], R_["p6_full_general"], R_["p6_lora_general"]),
     "verdict": "small forgetting in both; LoRA a little less, but 48 updates and one paragraph cannot prove it"},
    {"tried": "merging the adapter into the weights (Step 6.8)", "number": "largest logit difference %s; %d of 8 answers identical" % (R_["p6_merge_diff"], R_["p6_merge_same"]),
     "verdict": "worked: the merged model is the same function with no extra layers"},
    {"tried": "the SVD of the trained updates (Step 6.10)", "number": "LoRA update: 90 percent of the energy in 1 to 4 directions; full update: %d to %d directions" % (min(R_["p6_svd"]["layer%d full fine-tuning" % i]["e90"] for i in (0, 5, 11)), max(R_["p6_svd"]["layer%d full fine-tuning" % i]["e90"] for i in (0, 5, 11))),
     "verdict": "did not confirm that the full update is low rank; a low-rank update was enough for this task, not identical to the full one"},
]
print("%-52s | %-52s | %s" % ("what I tried", "what I measured", "verdict"))
for f in findings: print("%-52s | %-52s | %s" % (f["tried"], f["number"], f["verdict"]))
```

The example uses only the Part 6 numbers of your own session and keeps the verdict honest: LoRA did not match full fine-tuning on validation loss, rank 16 was not better than rank 8, forgetting differed by a small amount and the full update was not low rank. Notice that the successes (the merge, the identical peft result, the 420 times smaller file) are the ones that were *derived or checked*, and the doubtful ones are the ones that rest on one seed and eight validation examples.

It prints:

```text
what I tried                                         | what I measured                                      | verdict
LoRA r = 8 against full fine-tuning (Step 6.6)       | val loss 0.27 against 0.193; 294,912 of 124,439,808 numbers trained | partly: both reach style_rate 1.0, but LoRA did not match the validation loss of full fine-tuning (it is cheaper, not free)
five ranks from 1 to 16 (Step 6.7)                   | val loss 0.525 (r = 1), 0.27 (r = 8), 0.31 (r = 16)  | surprising: r = 16 was not better than r = 8; with one seed and 8 validation answers the plateau is the finding, not the winner
loss on the general paragraph after training (Step 6.6) | 3.509 before, 3.62 full, 3.574 LoRA                  | small forgetting in both; LoRA a little less, but 48 updates and one paragraph cannot prove it
merging the adapter into the weights (Step 6.8)      | largest logit difference 0.000214; 8 of 8 answers identical | worked: the merged model is the same function with no extra layers
the SVD of the trained updates (Step 6.10)           | LoRA update: 90 percent of the energy in 1 to 4 directions; full update: 55 to 89 directions | did not confirm that the full update is low rank; a low-rank update was enough for this task, not identical to the full one
```

Running the check on this answer prints:

```text
✓ at least 4 findings
✓ every row has tried, number and verdict
✓ every number field contains a digit
✓ at least one verdict says something did NOT work or surprised you
```

   **What you should see.**

   The first cell lists ten Part 6 numbers, for example `p6_full_val 0.193`, `p6_lora_val 0.27`, `p6_lora_share_pct 0.237`, `p6_size_ratio 420` and `p6_merge_diff 0.000214`. The check passes four ticks, and the answer prints a five-row table.

   *What it means.* Here is what a reader should take from the table: the things you can *prove* (the parameter count, the exact merge, the agreement with peft, the file size) are solid; the things that are *measured once on tiny data* (which rank is best, how much each method forgets) are suggestions. Say so in your report. This is also how a good model card is written.

4. **Step 7.4. Reflect, and what comes next.** Last, four questions that connect the weeks. Answer each in two or three sentences, in your own words, with at least one number from this lab. They are the same questions as the written part of the assessed deliverable, in short form. **q1** What are the stages of building a chat model, and what does each one need? **q2** Why is LoRA cheap, and what does it cost you? **q3** For a project at your school, would you use a prompt, RAG or fine-tuning, and why? **q4** Name one thing in this lab that did not work as you expected and what you learned from it.

   *Exercise 7.4:* Answer in `reflection` (four strings), 25 words or more each, each with at least one number from your own runs. Why it matters: a number in the answer shows that the claim comes from something you measured. The check tests the length and the presence of a digit, not the content.

```python
reflection = {"q1": "", "q2": "", "q3": "", "q4": ""}    # YOUR TEXT: two or three sentences each, and at least one number from this lab in each answer

if all(v.strip() == "" for v in reflection.values()):
    print("Write your four answers in reflection above, then run this cell again.")
else:
    for k, v in reflection.items():
        check("%s: at least 25 words and at least one number" % k, len(v.split()) >= 25 and any(ch.isdigit() for ch in v), "write two or three sentences and cite a number from the lab")
```

   A possible answer (try first, then compare):

```python
reflection = {
    "q1": 'The stages are pre-training (next-token prediction on trillions of tokens, the most expensive stage), instruction tuning on pairs like our 24 examples (a base model with style_rate 0.0 reached 1.0 after two epochs), and preference tuning with comparisons. Each needs different data: raw text, demonstrations, and judgements.',
    "q2": 'LoRA trains only 294912 numbers (0.237 percent of GPT-2), keeps 2.36 MB of Adam state instead of 995.5 MB and saves a 1.19 MB file, but its validation loss was 0.27 against 0.193 for full fine-tuning.',
    "q3": 'For the handbook help desk I would use RAG with a prompt, because the handbook changes every year and answers must cite a page; for the report-card tone I would fine-tune with LoRA on about 500 past comments, because a tone must hold every time and the adapter is only about 1.19 MB.',
    "q4": 'I expected rank 16 to beat rank 8, but its validation loss was 0.31 against 0.27. I learned that with 24 examples, 8 validation answers and one seed, small differences are noise, and that I should report the shape of the curve and not the winner.',
}
for k, v in reflection.items(): print(k, len(v.split()), "words")
```

These are examples built from the numbers of Part 6; yours should use your own words and your own results from Parts 1 to 6.

It prints:

```text
q1 47 words
q2 36 words
q3 53 words
q4 46 words
```

Running the check on this answer prints:

```text
✓ q1: at least 25 words and at least one number
✓ q2: at least 25 words and at least one number
✓ q3: at least 25 words and at least one number
✓ q4: at least 25 words and at least one number
```

   **What you should see.**

   The check passes four ticks, and the answer prints the word count of each example answer.

   *What it means.* **What comes next.** Your design card and your findings table are the start of the **Capstone** (the Structural Inspection Assistant, in the Capstone page of the course app): ask for each part of the pipeline, "prompt, retrieval or fine-tune, and how will I measure it on photos and notes the model has never seen?", and remember that a licensed engineer makes every pass or fail decision. In **Week 15 (Pre-final exam and capstone proposal)** you review Weeks 11 to 14 and write the proposal for your own project: the design card of this part is the template for it. The numbers you need to remember from this week: 6ND for pre-training, the Bradley-Terry probability and loss, the KL-regularised optimum, the DPO loss, r(d_in + d_out) for LoRA, and the memory arithmetic of Step 6.5.
### Troubleshooting

Find your symptom in the left column, then read the cause and the fix. If the same error appears again after the fix, restart the runtime and run the Setup cells and the part again from the top: many errors come from a cell that was run twice or out of order.

| Symptom | Likely cause | What to do |
|---|---|---|
| `ModuleNotFoundError: No module named 'peft'` (Step 6.9) | The library is not installed in this session. | Run `pip install peft` (in Colab: `!pip install peft`), then the cell again. Nothing else in the lab needs it: the other steps work without peft. |
| The first run is slow and prints a download bar | GPT-2 small (about 500 MB) is downloaded once. | Wait; later runs and restarts use the saved copy. In Colab each new session downloads again. |
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied` in `LoRALinear` | You read the `Conv1D` weight as (out, in). GPT-2 stores it as (in, out). | Take `d_in, d_out = base.weight.shape` and make A `(d_in, r)` and B `(r, d_out)`. |
| LoRA trained but nothing changed: the loss hardly moves and `style_rate` stays 0.0 | The learning rate is the one for full fine-tuning (3e-5), or the optimizer was given the frozen parameters instead of the new ones. | Use about 1e-3 for these short runs, and pass `lora_params_list` (the parameters with `requires_grad`) to the optimizer. |
| `trainable` shows 124 million, not 294,912 | `add_lora` was run before the freeze, or the model was not frozen. | Freeze first (`requires_grad = False` for all), then wrap the layers. Rebuild the model from `base_model`. |
| The count is 589,824 or 147,456, not 294,912 | The rank is not 8. | Check the argument `r`; the parameters are `r * (768 + 2304)` per layer, 12 layers. |
| `lora_model` already contains LoRA layers (the count is wrong, or a layer is wrapped twice) | You ran the injection cell twice on the same model. | Make a fresh copy with `copy.deepcopy(base_model)` and inject once. Never wrap `lora_model` again. |
| The merged model gives different answers from the LoRA model | The scale `alpha / r` was left out when merging, or the bias was changed. | Use `W + scale * (A @ B)` and leave the bias alone. A logit difference of 1e-4 is rounding; 0.1 or more is a bug. |
| peft and our model disagree | A and B were not transposed when copied, or `fan_in_fan_out=True` is missing. | peft stores A as (r, in) and B as (out, r): copy `A.T` and `B.T`. Keep `fan_in_fan_out=True` for GPT-2. |
| `CUDA out of memory` | The Colab GPU is small and an old model is still in memory. | Delete the big objects (`del full_model`; `gc.collect()`; `torch.cuda.empty_cache()`) or choose Runtime, Restart session. Everything in this lab also runs on a CPU. |
| The kernel dies when two big models are alive | Each copy of GPT-2 takes about 0.5 GB of RAM. | Delete models you no longer need, or restart and run only the part you are working on. |
| A loss is `nan` | The learning rate is far too high (for example 1e-1 on a full model). | Go back to the values in the text: 3e-5 for full fine-tuning, 1e-3 for LoRA. |
| The validation loss rises while the training loss falls | Overfitting (Part 2): the model memorises the few training examples. | Fewer epochs, a lower learning rate, more or more varied data. Part 2 shows this with 4 examples. |
| `style_rate` is 0.0 after training | The prompt at test time is not the one used in training, or the labels were not masked correctly. | Use `format_prompt(q)` for the prompt, put the label -100 on the prompt tokens only, and end the answer with the end-of-text token. |
| The model keeps writing after the answer (`### Question: ...`) | The answer in training did not end with `tok.eos_token`, so the model never learned to stop. | Add `tok.eos_token` to the end of every training answer (as `p6_example` does), or read only up to the first line break. |
| Your numbers differ slightly from the ones printed | The random seed was changed, the cells were run in a different order, or the machine differs (CPU against GPU). | Run the Setup cells first, from the top, and do not call `torch.manual_seed` yourself. A last-digit difference is normal; a different result in the first decimal place is not. |
| Times are different from the ones in the text | The speed depends on the machine and on what else is running. | Compare the ratios (LoRA is several times faster than full fine-tuning), not the seconds. |
| The reward model in Part 4 stays at 50 percent accuracy | The features were not frozen or the labels were mixed up, so the model is guessing; the loss stays near 0.693 (ln 2). | Check that chosen and rejected are not swapped, that the features come from the frozen model and that the learning rate is not 0. |
| `KeyError` when you read `RESULTS` in Part 7 | The part that stores that number has not been run in this session. | Run that part again, or copy the number from your report into the findings table. |
| An exercise check prints `Fill in ... above` | The starter still has `None` or an empty string. | Replace each `None` or `""` with your answer and run the check again. |


## Check your understanding

{sub} Formative · Not graded · Unlimited attempts · Your best score is saved to your account

Fifteen questions cover the six outcomes, and nine of them ask you to calculate. Use a calculator and the equation figures above; the explanations show the working.

```quiz
{
  "id": "week-14-check",
  "title": "Week 14 quiz: Fine-tuning, RLHF and PEFT",
  "passMark": 70,
  "questions": [
    {
      "q": "A hospital has a general pre-trained model and 2 billion tokens of clinical notes with no labels. It wants the model to know the medical wording better. Which stage fits best?",
      "options": ["Instruction tuning with a chat template", "Task-specific fine-tuning on a labelled classifier", "Continued pre-training with the next-token objective", "Reward-model training from human comparisons"],
      "answer": 2,
      "why": "Unlabelled domain text with the same next-token objective is continued pre-training. Instruction tuning and task fine-tuning need labelled examples, and a reward model needs comparisons."
    },
    {
      "q": "Using C ≈ 6·N·D, how much compute does it take to fine-tune a 7-billion-parameter model on 10 million (10⁷) tokens?",
      "options": ["4.2 × 10¹⁷ FLOPs", "4.2 × 10²² FLOPs", "4.2 × 10¹⁹ FLOPs", "7.0 × 10¹⁶ FLOPs"],
      "answer": 0,
      "why": "6 × 7 × 10⁹ × 10⁷ = 4.2 × 10¹⁷. The 10²² figure is for 10¹² tokens, the pre-training case, and 7 × 10¹⁶ forgets the factor of 6."
    },
    {
      "q": "In supervised fine-tuning on a chat example, why is the loss computed only on the response tokens?",
      "options": ["Prompt tokens cannot be tokenised by the model", "The model should learn to write answers, not to reproduce the user’s question and the system message", "Masking the prompt makes the response tokens easier to predict", "The loss function cannot handle more than one speaker"],
      "answer": 1,
      "why": "The prompt is given to the model, not something it should learn to produce. Counting it would spend the training signal on predicting questions and hide how good the answers are."
    },
    {
      "q": "A response has 4 tokens that the model predicted with probabilities 0.5, 0.25, 0.125 and 1.0. The prompt tokens are masked. What is the masked SFT loss (mean −ln p, natural log)?",
      "options": ["0.26", "0.52", "4.16", "1.04"],
      "answer": 3,
      "why": "The terms are 0.693, 1.386, 2.079 and 0, which sum to 4.159. Dividing by the 4 response tokens gives 1.04. The value 4.16 is the sum, not the mean."
    },
    {
      "q": "Why does a base model often continue a question with more questions instead of answering?",
      "options": ["It was trained to continue text, and quiz-like lists of questions are common in its training data", "It cannot read questions", "It has too few parameters to answer", "It always refuses to answer for safety reasons"],
      "answer": 0,
      "why": "The next-token objective rewards matching the text it was trained on, not helping. Instruction tuning and preference tuning teach the behaviour of answering."
    },
    {
      "q": "A reward model gives the chosen answer a score of 1.2 and the rejected answer a score of −0.8. By the Bradley–Terry model, what is the probability that the chosen answer is preferred?",
      "options": ["0.269", "0.500", "0.881", "0.982"],
      "answer": 2,
      "why": "The gap is 1.2 − (−0.8) = 2.0 and σ(2) = 1 / (1 + e⁻²) = 0.881. The value 0.269 is σ(−1), and 0.982 is σ(4)."
    },
    {
      "q": "A reward model scores the chosen answer 1 and the rejected answer 3, so it has the order wrong. What is the preference loss −ln σ(r_w − r_l)?",
      "options": ["0.119", "2.127", "0.693", "0.127"],
      "answer": 1,
      "why": "The gap is 1 − 3 = −2 and σ(−2) = 0.119, so the loss is −ln 0.119 = 2.127. The number 0.119 is the probability, not the loss, and 0.693 would mean the scores were equal."
    },
    {
      "q": "In RLHF, what happens as the penalty strength β is made very small?",
      "options": ["The policy is forced to stay identical to the reference model", "The reward model stops being used", "The policy becomes more random", "The policy chases the reward more freely and is more likely to exploit the reward model’s flaws"],
      "answer": 3,
      "why": "A small β makes the KL leash weak, so the best policy moves towards the top-reward answer, including answers that only fool the reward model: reward hacking. A large β, not a small one, keeps the policy near the reference."
    },
    {
      "q": "Two answers have reference probabilities 0.5 and 0.5 and rewards 0 and 1.386 (which is ln 4). With β = 1, what is the best policy π* ∝ π_ref · exp(r/β)?",
      "options": ["(0.20, 0.80)", "(0.50, 0.50)", "(0.25, 0.75)", "(0.10, 0.90)"],
      "answer": 0,
      "why": "The tilted weights are 0.5 × e⁰ = 0.5 and 0.5 × e^1.386 = 0.5 × 4 = 2. Dividing by their sum 2.5 gives (0.2, 0.8)."
    },
    {
      "q": "A DPO pair has π(y_w) = 0.6, π_ref(y_w) = 0.3, π(y_l) = 0.1, π_ref(y_l) = 0.1 and β = 1. What is the DPO loss?",
      "options": ["0.693", "0.167", "0.405", "1.099"],
      "answer": 2,
      "why": "The log-ratios are ln 2 = 0.693 for the chosen answer and ln 1 = 0 for the rejected one, so the bracket is 0.693. σ(0.693) = 2/3 and −ln(2/3) = 0.405."
    },
    {
      "q": "Why is LoRA’s matrix B initialised to zeros while A gets small random numbers?",
      "options": ["Zeros make training faster than any other start", "A zero B means the adapter has no effect at the start, so the model begins exactly as the base model", "Random B would give A no gradient", "It saves memory"],
      "answer": 1,
      "why": "With B = 0, the product B·A is zero and h = Wx, so the adapted model equals the base model on day one. Training then moves B away from zero, step by step."
    },
    {
      "q": "A LoRA adapter with rank r = 16 is placed on a weight matrix with d_in = 1,024 and d_out = 4,096. How many numbers does it train?",
      "options": ["20,480", "4,194,304", "65,536", "81,920"],
      "answer": 3,
      "why": "The count is r × (d_in + d_out) = 16 × (1,024 + 4,096) = 16 × 5,120 = 81,920. The number 4,194,304 is the full matrix, and 65,536 is only 16 × 4,096."
    },
    {
      "q": "A school wants a chatbot that answers questions about a timetable and room list that change every week. What is usually the best approach?",
      "options": ["Full fine-tuning each week on the new timetable", "Pre-training a new model from scratch each term", "Retrieving the current timetable and giving it to the model in the prompt", "Using RLHF to teach the model the timetable"],
      "answer": 2,
      "why": "Facts that change often are better fetched than baked into weights, which are slow and costly to update. Retrieval (Week 16) keeps the answer current. Fine-tuning is for behaviour, not changing facts."
    },
    {
      "q": "How much GPU memory does full fine-tuning of a 3-billion-parameter model need for weights, gradients and Adam state, using 16 bytes per parameter?",
      "options": ["48 GB", "6 GB", "12 GB", "96 GB"],
      "answer": 0,
      "why": "3 × 10⁹ × 16 bytes = 48 × 10⁹ bytes = 48 GB, before activations. The value 6 GB is just the 16-bit weights, and 12 GB is the 32-bit weights alone."
    },
    {
      "q": "A tuned chat model beats the baseline in 33 of 60 comparisons. What is the standard error of the win rate, √(p(1 − p)/n)?",
      "options": ["0.004", "0.064", "0.248", "0.550"],
      "answer": 1,
      "why": "The win rate is 33/60 = 0.55. Then p(1 − p)/n = 0.55 × 0.45 / 60 = 0.004125, and its square root is 0.064. The value 0.004 forgets the square root and 0.55 is the win rate itself, so about 6 percentage points of uncertainty remain."
    }
  ]
}
```

## How Week 14 is assessed

**Your lab (100 points).** You submit the notebook with its outputs (or your JupyterLab files with the printed output) and answer five written questions. The questions are the assessment: each one asks for **your own measured numbers** from the notebook, so keep the output of the cells named below (a screenshot of the printed result is enough; do not retype numbers you did not see). Numbers can differ a little from the ones printed in the lab because of the machine; an answer that quotes your own, different, number and reads it correctly earns full marks. Answers that repeat a number from the text without running the cell do not.

**What to keep from each part (screenshots or copied numbers).**

| Part | Keep the output of | The numbers to have at hand |
|---|---|---|
| 1. The apprentice and the template | the token counts and the masked labels | the number of prompt tokens with the label -100 and of answer tokens; the 6ND values for pre-training and for fine-tuning and their ratio; what the base model did with 3 prompts |
| 2. SFT by hand and by machine | the masked and unmasked loss of one example; the training curves; the 4-example overfitting run | both losses and the Hugging Face loss; `style_rate` before and after; the validation loss at its lowest and at the end of the overfitting run |
| 3. Forgetting | the table of `text_loss(model, GENERAL_TEXT)` | the loss before and after full fine-tuning, and for each remedy you tried (lower learning rate, fewer steps, replay) |
| 4. Preference learning | the reward-model learning curve; the best-of-n table | the Bradley-Terry loss and the accuracy on `PREF_VAL` next to ln 2 = 0.693; the reward for n = 1 and the largest n; the reward hacking or proxy-gap result |
| 5. KL, beta and DPO | the frontier plot; the DPO check | the optimal policy at one beta; reward and KL along the frontier; the largest difference between the DPO result and the closed form |
| 6. LoRA | the table of Step 6.6; the rank sweep and its figure; the merge, peft and SVD cells; the file sizes | 294,912 trainable and its percent; validation loss, `style_rate` and general-text loss for LoRA and for full fine-tuning; the memory arithmetic; validation loss for r = 1, 2, 4, 8, 16; the largest logit difference after merging; the 90-percent-energy ranks; the adapter and the checkpoint in MB |
| 7. Plan and reflect | the two design cards, the decision table, the findings table | your own four reflection answers |

**The five questions (100 points).** Write each answer in full sentences under its question, in your own words. Where a question says "your numbers", quote them with their units.

| # | Question | Marks | What a full-mark answer contains |
|---|---|---|---|
| 1 | **Teaching a base model to answer** (Parts 1 and 2). Explain why the prompt tokens get the label -100 when we fine-tune on question and answer pairs, and what that changed in your loss for one example. Then describe what happened to the base model's behaviour, to the validation loss and to `style_rate` during training, and what your 4-example overfitting run showed. | 20 | **(6)** the masked and the unmasked loss of your example, with the reason they differ (the prompt tokens are not scored) and the check against the library loss. **(5)** `style_rate` before (0.0 for the base model) and after, and the validation loss before and after, quoted from your run. **(5)** the overfitting run: the number of examples, the lowest validation loss and the final one, and the word *overfitting* used correctly (training loss keeps falling while validation loss rises). **(4)** one sentence on what these 24 examples can teach (a style) and cannot (new knowledge) |
| 2 | **Forgetting** (Part 3). Did full fine-tuning make the model worse at ordinary text? Report the loss on the general paragraph before and after, name each remedy you tried and say honestly which helped and which did not. | 10 | **(4)** `text_loss` before and after full fine-tuning with the direction of the change stated. **(4)** at least two remedies (a lower learning rate, fewer steps, mixing in general text) each with its number and a verdict. **(2)** a sentence that the remedy may cost some of the new skill, or that your result was small or mixed, if that is what you found |
| 3 | **Learning what people prefer** (Part 4). Describe the reward model you trained, how well it ranks answers it has never seen, and what best-of-n selection and reward hacking showed. | 20 | **(5)** the model in one sentence (frozen GPT-2 features and a linear head, Bradley-Terry loss) with its training pairs (14). **(5)** accuracy and loss on `PREF_VAL`, compared with the ln 2 baseline of a model that has no idea, and read correctly given that there are only 6 validation pairs. **(5)** best-of-n: the reward of the best answer for n = 1 and for your largest n, and what it means that it rises. **(5)** reward hacking or the gap between the proxy reward and the truth, with your number, and why a stronger optimiser makes it worse |
| 4 | **Staying close to the reference** (Part 5). Explain what beta does, using the toy policy, and show that DPO reaches the same answer as the formula. | 20 | **(5)** the closed form pi star proportional to pi_ref times exp(r / beta), applied to your toy with the resulting probabilities for one beta. **(5)** the frontier: expected reward rises and KL rises as beta falls; which end is reward hacking and which is a policy that never moved. **(5)** DPO by gradient descent reaching the closed-form optimum, with the largest difference you measured and the tolerance of the assertion. **(5)** one sentence on why DPO needs no reward model and no RL loop, and one on what the real GPT-2 DPO steps changed (the margin of chosen minus rejected) if you ran them |
| 5 | **LoRA and your own plan** (Parts 6 and 7). Show that LoRA trains very few parameters and still learns the style, say what it costs compared with full fine-tuning, and then give your design card and the decision for one scenario. | 30 | **(6)** the count: 12 layers, r (768 + 2304) = 24,576 per layer, 294,912 in total, which is 0.237 percent of the 124,439,808 parameters; and the memory arithmetic (about 1.99 GB against 0.50 GB, and the measured Adam state of about 995 MB against 2.4 MB). **(6)** your table of Step 6.6: validation loss, `style_rate`, general-text loss and time for LoRA and for full fine-tuning, with the honest finding that full fine-tuning reached the lower validation loss while both reached a `style_rate` of 1.0. **(4)** the rank sweep for r = 1, 2, 4, 8 and 16: the steep gain at first and the plateau, and the fact that r = 16 was not better than r = 8 (or whatever your sweep shows), with the warning that one seed and 8 validation examples are a weak test. **(4)** the merge (largest logit difference about 1e-4, 8 of 8 identical answers), the agreement with peft (294,912 parameters), and the sizes (about 1.2 MB for the adapter against about 498 MB, a ratio of about 420). **(6)** your design card for one of the two scenarios, with a method that fits the facts, a metric with a number and a named person who checks; and your decision (prompt, RAG or fine-tune) for the six situations with a reason for any row you changed. **(4)** your findings table with at least four rows, one of them negative or surprising, and one sentence linking your plan to the Capstone |

**Reference values for the instructor and for self-checking (seed 42, CPU, one run).** Part 6: trainable parameters 294,912 (exact); memory arithmetic 1.991 GB and 0.501 GB (exact); validation loss of the untrained model 2.816; full fine-tuning about 0.19; LoRA r = 8 about 0.27; rank sweep falling from about 0.5 at r = 1 to about 0.3 at r = 4 and r = 16; `style_rate` 1.0 for both methods; adapter file about 1.19 MB; full checkpoint about 497.8 MB; peft parameter count equal to ours. Losses can differ in the second decimal place between machines, and times differ a lot; the statements (full fine-tuning lower loss, style learned by both, the plateau in r, an exact merge) should hold. Do not subtract marks for a different last digit.


## Exit ticket and homework

Before you leave, answer these three questions in a sentence or two each.

1. Pick one of the four stages (pre-training, continued pre-training, SFT, preference tuning) and say what data it uses and what it changes in the model.
2. In the formula J(π) = E[r] − β·KL(π ‖ π_ref), what does β control, and what goes wrong if it is too small?
3. A friend wants to fine-tune a 7 B model on their laptop for a school project. Name two reasons why LoRA or QLoRA would help them, and one situation where fine-tuning would not be the right tool.

**Homework**

- Finish the lab and keep your notebook outputs; they are your evidence.
- Re-read the equation figures for the SFT loss, the Bradley–Terry loss, the KL objective, DPO and LoRA, and redo each worked example without looking.
- Review Weeks 11–14 for the Pre-final exam in Week 15; your instructor announces the date, room and format.
- Start thinking about your capstone proposal: which part of the Structural Inspection Assistant could use fine-tuning, and which part needs retrieval or a better prompt?
