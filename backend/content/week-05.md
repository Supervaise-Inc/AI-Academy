> Content based on the NVIDIA Generative AI and DLI Deep Learning Teaching Kits, used under the Supervaise × NVIDIA partnership.

# Week 5 — Prelim Exam

**Purpose:** Prepare for the Prelim by ==reviewing Weeks 1 to 4 and practising the calculations and diagnoses== you will be asked to do, in a written exam and a practical notebook exam.

**Term:** Prelim (Weeks 1–5) · **Month:** 2 · **Time:** Part A 60 minutes, Part B 120 minutes, plus your revision

```viz
{
  "type": "hook",
  "title": "Watch four weeks in one minute",
  "caption": "The Prelim covers Weeks 1 to 4: a model picks the next word from probabilities, a network learns by lowering its loss, gradients can fade as they travel backwards, and a filter slides over an image. The numbers in these scenes are made up or computed for teaching; nothing here is new.",
  "scenes": [
    { "name": "Week 1: words", "kind": "bars", "outputLabel": "Chance of each next word (softmax of made-up scores)", "prompt": "The jeepney driver stopped at the …", "bars": [{ "label": "terminal", "value": 0.66 }, { "label": "corner", "value": 0.24 }, { "label": "bridge", "value": 0.05 }, { "label": "palengke", "value": 0.04 }] },
    { "name": "Week 2: loss", "kind": "loss", "outputLabel": "Training loss", "prompt": "Gradient descent nudges every weight downhill, step after step" },
    { "name": "Week 3: gradients", "kind": "text", "outputLabel": "A fact about sigmoid networks", "prompt": "A sigmoid passes back at most 0.25 of the gradient. What is left after 8 layers?", "output": "0.25 to the power 8 is about 1.5 × 10⁻⁵, so the first layers barely learn." },
    { "name": "Week 4: convolution", "kind": "conv", "outputLabel": "Feature map", "prompt": "Slide a 3×3 vertical-edge filter over the pixel grid" }
  ]
}
```

## By the end, you can…

1. ==Explain and compute the Week 1 ideas==: discriminative versus generative, the chain rule of probability, perplexity, softmax with temperature, top-k and top-p, and the forward diffusion step.
2. ==Compute the Week 2 building blocks== by hand: MSE, cross-entropy −ln p, a neuron, a layer, parameter counts and one gradient-descent update.
3. ==Diagnose a model== from numbers: underfitting or overfitting from training and validation error, and a learning rate that is too small or too large.
4. ==Trace a forward and backward pass== through a small network with the four backpropagation equations, and say what each number means.
5. ==Explain vanishing and exploding gradients== with the product of per-layer factors (σ′ is at most 0.25), and choose a remedy.
6. ==Work with convolutions==: compute a convolution, its output size, parameter count and receptive field, and find dL/dK and dL/dx, including the backward pass through ReLU and max pooling.

Assessed in the Prelim exam described below. Your instructor announces the date, room and how the exam counts towards your course; this page does not. Read each review section, cover the solutions of the practice problems and try them first, then use the figures to test yourself. The figures are practice, so you cannot break anything.

## About this exam {rail=About}

{lede} The Prelim is ==one exam in two parts, marked out of 100 points==. It does not ask you to remember sentences from the lessons; it asks whether you can *do* what Weeks 1 to 4 taught: compute a small result by hand, explain why something happens, and diagnose a model that misbehaves.

### The format

| Part | What it is | Time | Points |
|---|---|---|---|
| Part A: written exam | 40 multiple-choice questions, 1 point each, closed book, taken on this page | 60 minutes | 40 |
| Part B: practical exam | Five tasks in a notebook (JupyterLab on the school's DGX Spark, or the Colab notebook) | 120 minutes | 60 |
| Total | | 180 minutes | 100 |

The pass mark is ==70 points out of 100==. Part B is split into five tasks, each with its own time box. Keep to the time boxes: an unfinished Task 3 should not cost you Task 5.

| Part B task | Topic | Time | Points |
|---|---|---|---|
| Task 1 | Generative AI and language models | 20 minutes | 10 |
| Task 2 | Supervised learning, neurons and generalisation | 25 minutes | 12 |
| Task 3 | Backpropagation and deep networks | 30 minutes | 16 |
| Task 4 | Convolutional networks | 30 minutes | 16 |
| Task 5 | Synthesis, short written answers | 15 minutes | 6 |

### What the exam covers

The exam is cumulative for Weeks 1 to 4. Every week appears in both parts, so a weak week cannot hide.

| Topic | Part A questions | Part B task |
|---|---|---|
| Week 1: generative AI | 8 | Task 1 (10 points) |
| Week 2: supervised learning | 10 | Task 2 (12 points) |
| Week 3: deep learning and backprop | 11 | Task 3 (16 points) |
| Week 4: convolutional networks | 11 | Task 4 (16 points) |
| All four weeks together | none | Task 5 (6 points) |
| Total | 40 | 60 points |

In Part A, about 40 percent of the questions test whether you remember and understand an idea, about 40 percent ask you to apply it (at least 14 of the 40 need a calculation, and the numbers are in the question), and about 20 percent ask you to analyse a situation and diagnose what is wrong. The questions are new, so memorised answers will not help, but they are fair for anyone who did the lessons and labs.

### The rules

- **Part A is closed book.** No notes, no phones, no other tabs. A calculator and scrap paper are fine.
- **Part B is open lessons and your own lab notes.** You may not use AI assistants, share work, or search the internet for answers. A calculator and paper are fine. Keep your own work: every cell you submit must be yours.
- **Integrity.** The written exam runs in your browser and cannot stop a determined cheater, so it is meant to be taken under the supervision your instructor arranges. Please play fair; a score is only useful to you if it is honestly yours.
- **Date and weight.** Your instructor announces the date, room, and how the exam counts towards your course. Do not rely on rumours; ask your instructor.

### How to prepare

1. Read the four review sections on this page and try every practice problem with the solution covered.
2. Learn the formula sheet in the Revision section well enough to write it from memory, because Part A is closed book.
3. Redo the formative quiz at the end of each of Weeks 1 to 4, and rerun your lab notebooks, trying to finish each Check cell without looking at your old answers.
4. Practise with [Mock Exam Prep](/mock-exam).
5. Sleep. A rested student who knows the formulas beats a tired one who crammed.

### On the day, and if something breaks

Bring your charged device, a calculator and scrap paper. For Part B, open JupyterLab on the DGX Spark (or the Colab notebook) before the clock starts and run the setup cell once, so you know it works. ==Save often==: in Colab use File, then Save a copy in Drive, and in JupyterLab use File, then Save Notebook (Ctrl+S). If the browser freezes, the connection drops or a cell will not run, ==tell your invigilator at once== instead of struggling alone; they can decide how to give you the time back.

```viz
{
  "type": "pipeline",
  "title": "The exam day at a glance",
  "intro": "Step through the day. Notice that the two parts use different tools and different rules.",
  "nodes": [
    { "id": "arrive", "label": "Arrive", "sub": "device, calculator, paper" },
    { "id": "a", "label": "Part A", "sub": "40 questions, 60 min" },
    { "id": "rest", "label": "Break", "sub": "set by your instructor" },
    { "id": "b", "label": "Part B", "sub": "5 tasks, 120 min" },
    { "id": "submit", "label": "Submit", "sub": "notebook saved and handed in" }
  ],
  "steps": [
    { "title": "Arrive and settle in", "body": "Bring your device, a calculator and scrap paper. Log in to this page for Part A and check that your internet works.", "active": ["arrive"], "example": "Close every other tab before you press Start." },
    { "title": "Part A: written exam", "body": "Forty multiple-choice questions in 60 minutes, closed book. There is no feedback until you submit, and you can flag questions and move back and forth.", "active": ["a"], "example": "About 90 seconds per question; flag a hard one and come back." },
    { "title": "A short break", "body": "A break between the parts. Its length is set by your instructor. Drink water and do not discuss the questions.", "active": ["rest"], "example": "Skip the post-mortem; Part B needs a clear head." },
    { "title": "Part B: the five tasks", "body": "Open JupyterLab or the Colab notebook, run the setup cell, then work through Tasks 1 to 5 in order, keeping to each time box. Lessons and your lab notes are open; AI assistants and sharing are not.", "active": ["b"], "example": "Task 1: 20 min, Task 2: 25, Task 3: 30, Task 4: 30, Task 5: 15." },
    { "title": "Save and submit", "body": "Save the notebook, check that each task has an answer in the right cell and submit it the way your instructor says. If anything went wrong, say so before you leave.", "active": ["b", "submit"], "example": "Colab: File, then Save a copy in Drive." }
  ],
  "note": "The break length and how to submit are arranged by your instructor; this page does not set them."
}
```

The time boxes in Part B add up to exactly 120 minutes (20 + 25 + 30 + 30 + 15), so there is no spare time. If you are stuck, write what you know and move on.

## Week 1: generative AI {rail=Week 1}

{lede} Week 1 asked what a model produces and how it picks it. A [[generative model|A model that learns the patterns in data and then produces new data that follows them, such as text or images.]] ==learns a distribution and then samples from it==, while a [[discriminative model|A model that gives back a label or a number for data that already exists, such as spam or not spam.]] only labels or scores what you give it.

### You must be able to

- Say whether a task is generative or discriminative, and write the question each one answers: P(y given x) for a label, P(x given a prompt) for new data.
- Explain "learn, then sample": training makes the real data more probable by lowering the loss −ln p, and generation draws new items from the learned probabilities.
- Break text into [[tokens|The pieces a language model reads and writes: whole words, parts of words or single characters.]] and describe the writing loop: score every possible next token, turn the scores into probabilities, pick one, append it, repeat.
- Compute the probability of a sentence with the chain rule, and its perplexity P to the power −1/n.
- Compute softmax with temperature from [[logits|The raw scores a network gives to each candidate token before they are turned into probabilities.]], and say what a low or high temperature does.
- Apply top-k and top-p to a list of probabilities, including the renormalising step.
- Compute one forward-diffusion step, and name the main risks (bias, hallucination, privacy, deepfakes) and the five principles of trustworthy AI.

### Discriminative or generative?

Sort the cards. Remember the test: ==does the output already exist in the input's world (a label or a number), or is it new content?== Translating a sentence and writing a reply are generative, because the output text did not exist before.

```viz
{
  "type": "sorter",
  "title": "Generative or discriminative?",
  "intro": "Sort each task. Ask: is the answer a label or number about the input, or new content?",
  "cards": [
    { "name": "Predict the price of a house from its size and location", "answer": "discriminative", "why": "The output is one number about the input (regression). Nothing new is created." },
    { "name": "Write a short poem about typhoons", "answer": "generative", "why": "The output is new text that did not exist before." },
    { "name": "Decide if a text message is spam", "answer": "discriminative", "why": "The output is a label, spam or not spam, chosen from a fixed set." },
    { "name": "Draw a picture of a carabao from a caption", "answer": "generative", "why": "The model produces new pixels that follow the patterns it learned." },
    { "name": "Say whether a photo shows a cat or a dog", "answer": "discriminative", "why": "It maps an existing picture to one of two labels." },
    { "name": "Translate a Tagalog sentence into English", "answer": "generative", "why": "The English sentence is new text, produced one token at a time." },
    { "name": "Flag a bank transaction as fraud or normal", "answer": "discriminative", "why": "The output is a label about an existing transaction." },
    { "name": "Autocomplete the rest of your sentence", "answer": "generative", "why": "It continues the text by sampling the next tokens." }
  ]
}
```

If a card surprised you, reread the Week 1 test above. ==Generative AI is the smallest of the nested dolls==: it sits inside deep learning, which sits inside machine learning, which sits inside AI.

### Softmax with temperature

```viz
{
  "type": "equation",
  "title": "Softmax with temperature",
  "intro": "Select each part, then step through a new example with three scores at two temperatures.",
  "formula": [
    { "t": "pᵢ", "id": "p" },
    " = ",
    { "t": "e", "id": "e" },
    "^(",
    { "t": "zᵢ", "id": "z" },
    " / ",
    { "t": "T", "id": "T" },
    ") / ",
    { "t": "Σⱼ", "id": "sigma" },
    " ",
    { "t": "e", "id": "e2" },
    "^(",
    { "t": "zⱼ", "id": "z2" },
    " / T)"
  ],
  "parts": [
    { "id": "p", "name": "pᵢ (probability of token i)", "meaning": "The chance that token i is picked. Every pᵢ is positive and they add up to 1." },
    { "id": "e", "name": "e (about 2.718)", "meaning": "Raising e to a power makes every score positive, and a higher score gives a much larger result." },
    { "id": "z", "name": "zᵢ (logit)", "meaning": "The raw score the network gave to token i. Higher means a better fit. It can be negative." },
    { "id": "T", "name": "T (temperature)", "meaning": "Divides every score before the exponent. Below 1 it sharpens the choice; above 1 it flattens it; T = 1 changes nothing." },
    { "id": "sigma", "name": "Σⱼ (sum over all tokens)", "meaning": "Add the value of every token j. Dividing by this total makes the probabilities add up to 1." },
    { "id": "e2", "name": "e (again)", "meaning": "The same exponential, applied to every token to build the total." },
    { "id": "z2", "name": "zⱼ", "meaning": "The score of each token in the sum. The index j runs over all tokens; i is the one you ask about." }
  ],
  "steps": [
    { "title": "Three scores", "expr": "z = (3.0, 2.0, 0.5) for terminal, corner, bridge", "note": "Made-up logits, only three tokens so the arithmetic stays easy." },
    { "title": "T = 1: exponentials", "expr": "e^3.0 = 20.09, e^2.0 = 7.39, e^0.5 = 1.65, total = 29.12", "note": "Compute e to each score, then add them." },
    { "title": "T = 1: probabilities", "expr": "20.09/29.12 = 0.690,  0.254,  0.057", "note": "The probabilities add up to 1." },
    { "title": "T = 2: scores first", "expr": "z / T = (1.5, 1.0, 0.25)", "note": "A higher temperature divides the scores first, so the gaps shrink." },
    { "title": "T = 2: probabilities", "expr": "e^1.5 = 4.48, e^1.0 = 2.72, e^0.25 = 1.28, total = 8.48  gives  0.528,  0.320,  0.151", "note": "The favourite falls from 0.690 to 0.528; at T = 0.5 it would rise to 0.876." }
  ],
  "note": "The scores are made up for practice. The arithmetic is exact."
}
```

The favourite stays the favourite at every temperature; what changes is how sure the model sounds. ==Low temperature sharpens, high temperature flattens.==

**Watch out:** temperature goes in the denominator of the exponent. Multiplying by T instead of dividing gives the opposite effect.

### Sampling: temperature, top-k and top-p together

Use the dials to check the hand calculations. Top-k keeps the k most likely tokens. Top-p keeps the smallest group of top tokens whose probabilities add up to at least p. After either filter, the survivors are renormalised so they add up to 1 again.

```viz
{
  "type": "sampling",
  "title": "Sampling dials with new scores",
  "prompt": "The jeepney driver stopped at the",
  "tokens": [
    { "t": "terminal", "logit": 3 },
    { "t": "corner", "logit": 2 },
    { "t": "bridge", "logit": 0.5 },
    { "t": "palengke", "logit": 0.2 },
    { "t": "moon", "logit": -2 }
  ],
  "note": "The scores are made up for practice; they do not come from a real model."
}
```

At T = 1 the probabilities are terminal 0.659, corner 0.242, bridge 0.054, palengke 0.040, moon 0.004. Try top-p = 0.95: terminal and corner give 0.901, which is below 0.95, so bridge is added too (0.955). With top-p = 0.9 two tokens would be enough. Top-p keeps a number of tokens that adapts to the situation; top-k always keeps the same number.

### The chain rule and perplexity

```viz
{
  "type": "equation",
  "title": "Perplexity from the chain rule",
  "intro": "First the sentence probability is a product, then perplexity turns it into an average \"number of equally likely choices\".",
  "formula": [
    { "t": "perplexity", "id": "ppl" },
    " = ",
    { "t": "P(w₁ … wₙ)", "id": "P" },
    "^(−1/",
    { "t": "n", "id": "n" },
    ")"
  ],
  "parts": [
    { "id": "ppl", "name": "perplexity", "meaning": "How confused the model is by the text, on average. Lower is better; 1 would mean it was sure of every token." },
    { "id": "P", "name": "P(w₁ … wₙ)", "meaning": "The probability of the whole text, from the chain rule: the product of P(wᵢ given the earlier tokens)." },
    { "id": "n", "name": "n (number of tokens)", "meaning": "The power −1/n averages the surprise per token, so texts of different lengths can be compared." }
  ],
  "steps": [
    { "title": "Chain rule", "expr": "P = 0.25 × 0.1 × 0.2 × 0.4", "note": "One factor per token, each given everything before it." },
    { "title": "The product", "expr": "P = 0.002", "note": "Long texts get tiny probabilities, which is why real systems add logarithms instead." },
    { "title": "Perplexity", "expr": "0.002^(−1/4) = 4.73", "note": "The model was about as unsure as choosing between 4.7 equally likely tokens at each step." },
    { "title": "Link to the loss", "expr": "−ln p = 1.386, 2.303, 1.609, 0.916; their mean is 1.554, and e to that is 4.73", "note": "Perplexity is e to the average training loss −ln p." }
  ],
  "note": "The four probabilities are made up for practice."
}
```

Now compare two five-token texts. Select a sentence to see its running product and perplexity.

```viz
{
  "type": "chain",
  "title": "Natural order or scrambled?",
  "sentences": [
    { "label": "Natural", "tokens": [{ "t": "The", "p": 0.3 }, { "t": "jeepney", "p": 0.15 }, { "t": "leaves", "p": 0.05 }, { "t": "soon", "p": 0.2 }, { "t": ".", "p": 0.6 }], "note": "Each token is fairly likely after the ones before it." },
    { "label": "Scrambled", "tokens": [{ "t": "Soon", "p": 0.02 }, { "t": "leaves", "p": 0.01 }, { "t": "the", "p": 0.03 }, { "t": ".", "p": 0.01 }, { "t": "jeepney", "p": 0.02 }], "note": "The same words in the wrong order: every step is a surprise." }
  ],
  "note": "All probabilities are made up for practice. The natural text multiplies to 2.70 × 10⁻⁴ (perplexity 5.17); the scrambled one to 1.20 × 10⁻⁹ (perplexity 60.8)."
}
```

The same words in a different order are about 225 thousand times less likely. ==Order matters because every factor is conditioned on what came before.==

### One step of forward diffusion

```viz
{
  "type": "equation",
  "title": "Adding noise: the forward diffusion step",
  "intro": "The picture is blended with noise. At larger t the signal weight shrinks and the noise weight grows.",
  "formula": [
    { "t": "xₜ", "id": "xt" },
    " = ",
    { "t": "√ᾱₜ", "id": "sa" },
    " · ",
    { "t": "x₀", "id": "x0" },
    " + ",
    { "t": "√(1−ᾱₜ)", "id": "sn" },
    " · ",
    { "t": "ε", "id": "eps" }
  ],
  "parts": [
    { "id": "xt", "name": "xₜ (noisy picture)", "meaning": "The picture after t noising steps." },
    { "id": "sa", "name": "√ᾱₜ (signal strength)", "meaning": "How much of the original survives. It starts near 1 and falls towards 0 as t grows." },
    { "id": "x0", "name": "x₀ (original picture)", "meaning": "The clean training picture, here one pixel value." },
    { "id": "sn", "name": "√(1−ᾱₜ) (noise strength)", "meaning": "How much random noise is mixed in. It grows as the signal shrinks." },
    { "id": "eps", "name": "ε (noise)", "meaning": "A random number drawn from a bell curve; one value per pixel." }
  ],
  "steps": [
    { "title": "The schedule", "expr": "β rises linearly from 0.0001 to 0.02 over 1000 steps; ᾱₜ = (1−β₁)(1−β₂)…(1−βₜ)", "note": "ᾱₜ is the share of the signal left after t steps." },
    { "title": "At t = 500", "expr": "ᾱ = 0.0786, √ᾱ = 0.280, √(1−ᾱ) = 0.960", "note": "Halfway through, little signal is left." },
    { "title": "One pixel", "expr": "x₀ = 0.8, ε = −0.4  gives  xₜ = 0.280 × 0.8 + 0.960 × (−0.4) = −0.160", "note": "The pixel has lost its original value." },
    { "title": "At t = 100", "expr": "ᾱ = 0.8970; xₜ = 0.629", "note": "Early on, the signal still dominates." }
  ],
  "note": "The pixel value and the noise are made up for practice; the schedule is the common linear one used in the lesson."
}
```

### A short timeline to place events

```viz
{
  "type": "timeline",
  "title": "Seven dates to place",
  "events": [
    { "year": "1956", "title": "Dartmouth workshop", "text": "The summer workshop where the term \"artificial intelligence\" was coined; often treated as the birth of the field." },
    { "year": "1958", "title": "The Perceptron", "text": "Rosenblatt's early artificial neuron that learns from labelled examples. It is discriminative: it sorts inputs and creates nothing." },
    { "year": "1986", "title": "Backpropagation popularised", "text": "Rumelhart, Hinton and Williams showed how multi-layer networks learn by passing error signals backwards. You studied it in Week 3." },
    { "year": "2012", "title": "AlexNet", "text": "A deep convolutional network trained on GPUs won an image contest by a wide margin and made deep learning the main approach." },
    { "year": "2017", "title": "The Transformer", "text": "Vaswani and colleagues introduced attention-based models that made very large language models practical to train." },
    { "year": "2020", "title": "GPT-3", "text": "A far larger language model than GPT-2; scale alone brought new abilities such as following examples written in the prompt." },
    { "year": "2022", "title": "Stable Diffusion and ChatGPT", "text": "Open high-quality image generation and a chatbot used by millions moved generative AI into daily life." }
  ]
}
```

Remember the pattern: each jump needed something new, such as a learning rule (1986), data and GPUs (2012), a new architecture (2017) and scale (2020). The five principles of [[trustworthy AI|Building and using AI in line with fairness, transparency, privacy, safety and accountability.]] are fairness, transparency, privacy, safety and accountability. ==Fluent does not mean true==, so a made-up but confident answer (a hallucination) needs checking.

### Practice problems

Cover each solution, try the problem on paper, then check. The numbers are new, so you have not seen these answers before.

**Problem 1. Softmax and temperature.** A model gives the logits 2.5, 1.0 and 0.0 to three tokens A, B and C. (a) Find the probability of A at T = 1. (b) Find it at T = 0.5. (c) Which temperature makes the output more random, and why?

**Solution 1.**

- (a) e^2.5 = 12.182, e^1.0 = 2.718, e^0.0 = 1. The total is 15.901. P(A) = 12.182 / 15.901 = ==0.766==.
- (b) Divide the logits by 0.5 first: 5.0, 2.0, 0.0. Then e^5 = 148.41, e^2 = 7.389, e^0 = 1, total 156.80. P(A) = 148.41 / 156.80 = ==0.946==.
- (c) A higher temperature, such as T = 2, gives P(A) = 0.569, lower than 0.766, so the choice is more spread out and more random. Lower temperature concentrates the probability on A.

**Watch out:** do not divide by T after the exponent, and remember that the probabilities must add up to 1. A quick check: 0.766 + 0.171 + 0.063 = 1.000.

**Problem 2. Top-k and top-p.** After softmax, five tokens have the probabilities 0.46, 0.27, 0.15, 0.08 and 0.04 (already sorted). (a) What are the probabilities after top-k with k = 2? (b) How many tokens does top-p with p = 0.8 keep, and what are their new probabilities?

**Solution 2.**

- (a) Keep 0.46 and 0.27. Their total is 0.73. Renormalise: 0.46 / 0.73 = ==0.630== and 0.27 / 0.73 = ==0.370==. The other three tokens get 0.
- (b) Add from the top: 0.46, then 0.46 + 0.27 = 0.73 (below 0.8), then 0.73 + 0.15 = 0.88 (at least 0.8). So ==three tokens== are kept. Renormalise by 0.88: 0.523, 0.307, 0.170, which add up to 1.
- Top-k fixes the number of tokens; top-p lets the number depend on how the probability is spread.

**Problem 3. Sentence probability and perplexity.** A language model gives these probabilities to the five tokens of a text, each given the tokens before it: 0.5, 0.2, 0.1, 0.25, 0.4. (a) Find P of the whole text. (b) Find the perplexity. (c) A second text of five tokens has P = 8.0 × 10⁻⁷. Which text does the model find more natural?

**Solution 3.**

- (a) By the chain rule, P = 0.5 × 0.2 × 0.1 × 0.25 × 0.4 = ==0.001==.
- (b) Perplexity = P^(−1/5) = 0.001^(−0.2) = ==3.98==. Check with logs: the five values of −ln p are 0.693, 1.609, 2.303, 1.386, 0.916; their mean is 1.382, and e^1.382 = 3.98.
- (c) The second text has perplexity (8.0 × 10⁻⁷)^(−0.2) = 16.6. Lower perplexity means a better fit, so the ==first text is more natural== to the model (3.98 against 16.6).

**Watch out:** compare perplexities only for texts of the same kind, and never confuse a low perplexity with being true. A fluent text can still be false.

## Week 2: supervised learning {rail=Week 2}

{lede} Week 2 asked how a model learns from examples. In [[supervised learning|Learning from examples where each input comes with the correct answer, called a label.]] you show the model inputs with answers, measure how wrong it is with a loss, and adjust the weights to make that loss smaller. ==Everything you will compute in Week 2 is one of those steps.==

### You must be able to

- Name the parts of a supervised problem: features, label, model, loss, training, and the three data sets (training, validation, test) with the job of each.
- Compute the mean squared error for a regression problem and the cross-entropy −ln p for a classification problem, and say which wrong answers each punishes most.
- Compute a neuron: z = w·x + b, then a = φ(z) for step, sigmoid, tanh or ReLU.
- Compute a layer a = φ(Wx + b), apply softmax to its scores, and count the [[parameters|The numbers a model learns during training: its weights and biases.]] of a network.
- Explain why one neuron cannot do XOR, why a hidden layer with a non-linear activation can, and why stacking linear layers without an activation gains nothing.
- Apply one gradient-descent step w ← w − η × gradient, and say what happens when the [[learning rate|The step size η in gradient descent: too small is slow, too large overshoots, oscillates or diverges.]] is too small or too large.
- Read training and validation errors to diagnose underfitting or overfitting, and name a remedy.

### Loss: how wrong is the model?

```viz
{
  "type": "equation",
  "title": "Mean squared error",
  "intro": "For a number prediction: average the squared mistakes. Select each part.",
  "formula": [
    { "t": "L", "id": "L" },
    " = ",
    { "t": "1/n", "id": "n" },
    " × ",
    { "t": "Σ", "id": "sum" },
    " ( ",
    { "t": "ŷᵢ", "id": "yhat" },
    " − ",
    { "t": "yᵢ", "id": "y" },
    " )",
    { "t": "²", "id": "sq" }
  ],
  "parts": [
    { "id": "L", "name": "L (the loss)", "meaning": "One number for the whole data set. Zero means perfect; larger means worse." },
    { "id": "n", "name": "1/n (the mean)", "meaning": "Divide by the number of examples so the loss does not grow just because there is more data." },
    { "id": "sum", "name": "Σ (sum)", "meaning": "Add the squared mistake of every example." },
    { "id": "yhat", "name": "ŷᵢ (the prediction)", "meaning": "What the model said for example i." },
    { "id": "y", "name": "yᵢ (the label)", "meaning": "The true answer for example i." },
    { "id": "sq", "name": "² (the square)", "meaning": "Makes every mistake positive and makes big mistakes count far more than small ones." }
  ],
  "steps": [
    { "title": "The data", "expr": "true prices 30, 45, 20, 55 and predictions 28, 50, 20, 52", "note": "Prices in pesos, made up for practice." },
    { "title": "Mistakes", "expr": "ŷ − y = -2, +5, 0, -3", "note": "The sign tells the direction, but the square removes it." },
    { "title": "Squares", "expr": "4 + 25 + 0 + 9 = 38", "note": "The single mistake of 5 contributes 25 of the 38." },
    { "title": "Mean", "expr": "L = 38 / 4 = 9.5", "note": "In squared pesos. Take the square root (3.08 pesos) for a typical mistake in pesos." }
  ],
  "note": "The prices and predictions are made up for practice."
}
```

```viz
{
  "type": "equation",
  "title": "Cross-entropy: minus ln p",
  "intro": "For a class prediction: look only at the probability the model gave to the true class.",
  "formula": [
    { "t": "L", "id": "L" },
    " = −ln ( ",
    { "t": "p", "id": "p" },
    " )"
  ],
  "parts": [
    { "id": "L", "name": "L (loss for one example)", "meaning": "Zero when the model was certain and right; very large when it gave the true class a tiny probability." },
    { "id": "p", "name": "p (probability of the true class)", "meaning": "The number the model assigned to the correct answer after softmax. Not the highest probability, the true one." }
  ],
  "steps": [
    { "title": "Three examples", "expr": "the true class received p = 0.7, 0.2 and 0.02", "note": "Made-up probabilities." },
    { "title": "Each loss", "expr": "−ln 0.7 = 0.357,  −ln 0.2 = 1.609,  −ln 0.02 = 3.912", "note": "A confident wrong answer (0.02) costs far more than a mild one." },
    { "title": "The mean", "expr": "(0.357 + 1.609 + 3.912) / 3 = 1.959", "note": "The data set loss is the mean over examples. This is the same −ln p as the Week 1 training loss." }
  ],
  "note": "The probabilities are made up for practice."
}
```

**Watch out:** MSE is for numbers, cross-entropy is for classes. Cross-entropy uses the probability of the *true* class only, not the largest probability.

### A neuron, a layer and counting parameters

```viz
{
  "type": "equation",
  "title": "One neuron: z = w·x + b, then a = φ(z)",
  "intro": "Weigh the inputs, add the bias, then pass the result through the activation.",
  "formula": [
    { "t": "a", "id": "a" },
    " = ",
    { "t": "φ", "id": "phi" },
    " ( ",
    { "t": "z", "id": "z" },
    " )      where      ",
    { "t": "z", "id": "z2" },
    " = ",
    { "t": "Σ", "id": "sigma" },
    " ",
    { "t": "wᵢ", "id": "w" },
    { "t": "xᵢ", "id": "x" },
    "  +  ",
    { "t": "b", "id": "b" }
  ],
  "parts": [
    { "id": "x", "name": "xᵢ (inputs)", "meaning": "The numbers going into the neuron: features, or outputs of the previous layer." },
    { "id": "w", "name": "wᵢ (weights)", "meaning": "One learned number per input: how strongly that input counts." },
    { "id": "b", "name": "b (bias)", "meaning": "A learned offset that shifts when the neuron fires." },
    { "id": "sigma", "name": "Σ (sum)", "meaning": "Add up all the products wᵢxᵢ." },
    { "id": "z", "name": "z (pre-activation)", "meaning": "The weighted sum plus the bias, before the activation." },
    { "id": "z2", "name": "z", "meaning": "The same z, written out as the weighted sum." },
    { "id": "phi", "name": "φ (activation function)", "meaning": "A fixed non-linear function: step, sigmoid, tanh or ReLU." },
    { "id": "a", "name": "a (activation)", "meaning": "The neuron's output." }
  ],
  "steps": [
    { "title": "The numbers", "expr": "x = (3, −1, 2),  w = (0.2, 0.5, −0.1),  b = 0.4", "note": "Made up for practice." },
    { "title": "Weighted sum", "expr": "0.2×3 + 0.5×(−1) + (−0.1)×2 + 0.4 = 0.6 − 0.5 − 0.2 + 0.4 = 0.3", "note": "The sum of the products and the bias is 0.3." },
    { "title": "Four activations", "expr": "step: 1;  sigmoid: 0.5744;  ReLU: 0.3;  tanh: 0.2913", "note": "The same z gives a different output for each φ." },
    { "title": "Count the parameters", "expr": "a layer of 4 neurons on 5 inputs: 5×4 + 4 = 24;  then 4→3: 4×3 + 3 = 15;  total 39", "note": "Weights are inputs × outputs, plus one bias per output unit." }
  ],
  "note": "All numbers are made up for practice; the arithmetic is exact."
}
```

For a whole layer, put the weights in a matrix: a = φ(Wx + b), where each row of W belongs to one neuron. The parameter count of a layer is inputs × outputs + outputs. Test the count yourself in the figure below, then click through the little network.

```viz
{
  "type": "mlp",
  "title": "Count the parameters",
  "intro": "Start with 6 inputs, 5 hidden units and 3 outputs. Read the table, then change a width and predict the new total.",
  "layers": [6, 5, 3],
  "note": "For [6, 5, 3]: 6×5 + 5 = 35 and 5×3 + 3 = 18, so 53 in total."
}
```

```viz
{
  "type": "network",
  "title": "A hand-built XOR network",
  "intro": "Try the inputs (1, 0), (1, 1), (0, 1) and (0, 0). Select a circle to see z = w·a + b and a = ReLU(z).",
  "layers": [2, 2, 1],
  "weights": [[[1, -1], [-1, 1]], [[1, 1]]],
  "biases": [[0, 0], [0]],
  "inputs": [1, 0],
  "activation": "relu",
  "outputActivation": "linear",
  "inputLabels": ["x1", "x2"],
  "outputLabels": ["XOR"],
  "presets": [
    { "name": "1, 0", "inputs": [1, 0] },
    { "name": "1, 1", "inputs": [1, 1] },
    { "name": "0, 1", "inputs": [0, 1] },
    { "name": "0, 0", "inputs": [0, 0] }
  ],
  "editable": true,
  "note": "Hand-built weights: the two hidden units compute x1 − x2 and x2 − x1 through ReLU, and the output adds them, giving the absolute difference, which is XOR on 0/1 inputs. This network has 9 parameters.",
  "checks": [
    { "inputs": [1, 0], "output": [1] },
    { "inputs": [1, 1], "output": [0] },
    { "inputs": [0, 1], "output": [1] },
    { "inputs": [0, 0], "output": [0] }
  ]
}
```

Here the two hidden units each compute a difference through a ReLU, and the output adds them, so the network answers 1 only when the two inputs differ. ==A single neuron cannot do this, but one hidden layer with a non-linear activation can.== Without the ReLU, the two layers would collapse into one linear layer, because W₂(W₁x) = (W₂W₁)x, and the problem would stay unsolved. See it learn:

```viz
{
  "type": "trainer",
  "title": "Train a small network on XOR",
  "intro": "Choose the XOR data, set 1 hidden unit and train, then set 4 hidden units and train again.",
  "dataset": "xor",
  "hidden": 2,
  "lr": 0.5,
  "note": "With 1 hidden unit the network stays stuck around 75 percent; with 2 or more it can reach 100 percent.",
  "checks": [
    { "dataset": "xor", "hidden": 1, "lr": 0.5, "epochs": 1000, "maxAcc": 0.8 },
    { "dataset": "xor", "hidden": 4, "lr": 3, "epochs": 1000, "minAcc": 0.95 }
  ]
}
```

### Gradient descent

```viz
{
  "type": "equation",
  "title": "The update rule: w ← w − η × gradient",
  "intro": "Move each weight a small step against its gradient, so the loss goes down.",
  "formula": [
    { "t": "w", "id": "w" },
    " ← ",
    { "t": "w", "id": "w2" },
    " − ",
    { "t": "η", "id": "eta" },
    " × ",
    { "t": "∂L/∂w", "id": "grad" }
  ],
  "parts": [
    { "id": "w", "name": "w (the weight)", "meaning": "The number being learned." },
    { "id": "w2", "name": "w (the old value)", "meaning": "The weight before this step; the arrow means \"becomes\"." },
    { "id": "eta", "name": "η (learning rate)", "meaning": "The step size. Too small is slow; too large overshoots, oscillates or diverges." },
    { "id": "grad", "name": "∂L/∂w (the gradient)", "meaning": "How fast the loss changes when w changes. The minus sign moves w downhill." }
  ],
  "steps": [
    { "title": "The model", "expr": "ŷ = w·x with x = 4, label y = 10, w = 1, η = 0.05", "note": "One weight, one example, loss L = (ŷ − y)²." },
    { "title": "Before", "expr": "ŷ = 4;  L = (4 − 10)² = 36", "note": "The model predicts 4 and should say 10." },
    { "title": "The gradient", "expr": "∂L/∂w = 2(ŷ − y)x = 2 × (−6) × 4 = −48", "note": "Negative: increasing w lowers the loss." },
    { "title": "The update", "expr": "w ← 1 − 0.05 × (−48) = 3.4", "note": "The weight goes up, as it should." },
    { "title": "After", "expr": "ŷ = 3.4 × 4 = 13.6;  L = 12.96", "note": "The loss fell from 36 to 12.96, but ŷ now overshoots the label 10, a first sign of a large step." }
  ],
  "note": "One weight and one example, made up for practice."
}
```

```viz
{
  "type": "descent",
  "title": "Learning rate on L = (w − 3)²",
  "fn": "quad",
  "lr": 0.3,
  "start": 0
}
```

For this loss, a learning rate below 1 converges, exactly 1 bounces between two points, and above 1 diverges. At lr 0.3 the first step is w = 0 − 0.3 × (−6) = 1.8. ==A different loss has a different safe range==, so always watch the loss curve rather than trusting a rule.

### Generalisation

Training error alone tells you very little, because a model can memorise; the result is [[overfitting|When a model fits the training data, noise included, so closely that it does badly on new data.]]. The [[validation set|Data held out from training and used to choose settings such as the number of hidden units or when to stop.]] is for choices, and the test set is touched once at the very end. Slide the degree and compare the two errors.

```viz
{
  "type": "fit",
  "title": "Underfitting and overfitting, new data",
  "intro": "Move the degree slider from 0 to 9. Watch the curve and compare the training and test errors.",
  "degree": 1,
  "maxDegree": 9,
  "train": [
    { "x": -1.6, "y": -0.67 },
    { "x": -1.2, "y": -0.96 },
    { "x": -0.8, "y": -0.88 },
    { "x": -0.4, "y": -0.52 },
    { "x": 0, "y": 0.18 },
    { "x": 0.4, "y": 0.58 },
    { "x": 0.8, "y": 0.72 },
    { "x": 1.2, "y": 0.99 },
    { "x": 1.6, "y": 0.73 }
  ],
  "test": [
    { "x": -1.4, "y": -0.89 },
    { "x": -0.8, "y": -0.94 },
    { "x": -0.2, "y": -0.12 },
    { "x": 0.4, "y": 0.52 },
    { "x": 1, "y": 1.08 },
    { "x": 1.6, "y": 0.77 }
  ],
  "note": "The data are authored for teaching: a smooth curve plus noise. Errors are mean squared errors."
}
```

For this data (computed from the figure's own fit), the test error is lowest at degree 4. Degree 0 gives training 0.535 and test 0.618: both high, so it underfits. Degree 9 gives training 0.000 but test 0.031: perfect on the points it has seen and 5 times the best test error, so it overfits.

```viz
{
  "type": "sorter",
  "title": "Underfit, good fit or overfit?",
  "intro": "Read each situation and place it. Compare the training and validation numbers.",
  "bins": [
    { "id": "under", "label": "Underfitting" },
    { "id": "good", "label": "Good fit" },
    { "id": "over", "label": "Overfitting" }
  ],
  "cards": [
    { "name": "Training error 0.48, validation error 0.51", "answer": "under", "why": "Both errors are high and close together: the model is too simple or has not trained enough." },
    { "name": "Training error 0.03, validation error 0.55", "answer": "over", "why": "A large gap with a tiny training error means the model memorised the training set." },
    { "name": "Training error 0.06, validation error 0.08", "answer": "good", "why": "Both are low and close together." },
    { "name": "A straight line fitted to data that clearly curve", "answer": "under", "why": "The model cannot bend, so it misses the pattern even on the training data." },
    { "name": "Training accuracy 99 percent, validation accuracy 71 percent", "answer": "over", "why": "A 28-point gap: it learned the noise as well as the pattern." },
    { "name": "Training loss keeps falling while validation loss has started to rise", "answer": "over", "why": "Stop training at the epoch where validation loss was lowest (early stopping)." },
    { "name": "Training loss is stuck high, whatever the learning rate", "answer": "under", "why": "The model is too small, or the features are poor. A bigger model or better features help." }
  ]
}
```

### Practice problems

**Problem 1. Losses.** A regression model predicts 11, 10 and 14 for three examples whose labels are 12, 8 and 15. (a) Compute the MSE. A classifier gives the true class the probabilities 0.6 and 0.05 on two examples. (b) Compute −ln p for each, and the mean cross-entropy. (c) Which example does the loss punish more?

**Solution 1.**

- (a) Mistakes: 11 − 12 = −1, 10 − 8 = +2, 14 − 15 = −1. Squares: 1, 4, 1. Sum 6, so MSE = 6 / 3 = ==2==.
- (b) −ln 0.6 = ==0.511==; −ln 0.05 = ==2.996==. Mean = (0.511 + 2.996) / 2 = 1.753.
- (c) The second example (p = 0.05) is punished 5.9 times as much, because the model gave the true class almost no probability.

**Problem 2. A layer and its parameters.** The input is x = (1, −2, 0.5). Neuron 1 has weights (0.4, 0.1, −0.6) and bias 0.2. Neuron 2 has weights (−0.5, 0.3, 0.8) and bias −0.1. (a) Compute z and the ReLU outputs. (b) How many parameters does a network 3 → 2 → 1 have?

**Solution 2.**

- (a) z₁ = 0.4×1 + 0.1×(−2) + (−0.6)×0.5 + 0.2 = 0.4 − 0.2 − 0.3 + 0.2 = ==0.1==. z₂ = (−0.5)×1 + 0.3×(−2) + 0.8×0.5 − 0.1 = −0.5 − 0.6 + 0.4 − 0.1 = ==−0.8==. After ReLU: a = (0.1, 0.0). The second neuron is switched off.
- (b) Layer 1: 3×2 + 2 = 8. Layer 2: 2×1 + 1 = 3. Total ==11==.

**Watch out:** do not forget the biases, and remember ReLU turns every negative z into 0.

**Problem 3. One gradient step.** The model is ŷ = w·x with x = 3, label y = 6, starting at w = 0.5, loss L = (ŷ − y)². (a) Do one step with η = 0.02. (b) What happens to the loss? (c) For which η does this problem diverge?

**Solution 3.**

- (a) ŷ = 0.5 × 3 = 1.5. The gradient is 2(ŷ − y)x = 2 × (−4.5) × 3 = −27. New w = 0.5 − 0.02 × (−27) = ==1.04==.
- (b) Before: L = (1.5 − 6)² = 20.25. After: ŷ = 3.12, L = (3.12 − 6)² = ==8.2944==. The loss fell a lot.
- (c) Each step multiplies the error (ŷ − y) by 1 − 2ηx² = 1 − 18η. The error shrinks when |1 − 18η| is below 1, which means η is below 1/9 = 0.111. For η above 0.111 the error grows each step, so the training diverges; at η = 0.111 exactly it bounces between two values.

## Week 3: deep learning and backpropagation {rail=Week 3}

{lede} Week 3 explained how a deep network learns and why it sometimes cannot. A network learns by a ==forward pass that produces the loss and a backward pass that multiplies local derivatives (the chain rule)== to give a gradient for every weight. Depth lets a network build ideas from parts, but it also makes the gradient a long product that is easy to lose.

### You must be able to

- Explain why depth helps: layers build features from simple to complex, and each extra layer can multiply the number of pieces a ReLU network can draw (2 to the power k pieces from k layers of two units).
- Run a forward pass layer by layer: z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾, then a⁽ˡ⁾ = φ(z⁽ˡ⁾), and compute the loss.
- Write and use the four equations of backpropagation: δ at the output layer, δ at a hidden layer, the weight gradient δaᵀ and the bias gradient δ.
- Use the chain rule on a short path (for example dL/dw = dL/da × da/dz × dz/dw) and know the slopes: σ′ = a(1 − a) is at most 0.25, tanh′ = 1 − a² is at most 1, and ReLU′ is 1 for z above 0 and 0 otherwise.
- Explain [[vanishing gradients|Gradients that shrink towards zero as they travel back through many layers, so the early layers hardly learn.]] and [[exploding gradients|Gradients that grow larger and larger as they travel back through many layers, so the weights jump and the loss can become NaN.]] as a product of factors, and use fⁿ to estimate the size.
- Choose a remedy and say why it works: ReLU, Xavier or He initialisation, normalisation, residual connections, gradient clipping.

### Why depth helps

A deep network reuses the features of earlier layers: edges become shapes, shapes become objects. There is also an exact counting argument. Each folding layer doubles the number of straight pieces, so k layers give 2ᵏ pieces, while one wide hidden layer needs about as many units as pieces.

```viz
{
  "type": "fold",
  "title": "Each layer doubles the pieces",
  "intro": "Stack three layers and count the straight pieces of the zig-zag. Then predict the number for five.",
  "layers": 3,
  "note": "Three layers give 2³ = 8 pieces. A network with one hidden ReLU layer and m units has at most m + 1 pieces, so it would need at least 7 units for the same zig-zag."
}
```

==Depth buys expressiveness efficiently==, but it must still be trainable, which is the rest of this section.

### The forward pass and the four equations

Use one small network for the whole section: 2 inputs, 2 hidden sigmoid units and 1 sigmoid output, with the loss L = ½(a − y)². The input is x = (0.5, 1.0) and the target is y = 1. Hidden weights: row 1 is (0.6, −0.4), row 2 is (0.3, 0.9); hidden biases (0.1, −0.2). Output weights (1.1, −0.7) and bias 0.05.

```viz
{
  "type": "backprop",
  "title": "Forward and backward through the 2-2-1 network",
  "intro": "Use the forward and backward tabs. Select a circle to see its arithmetic, then press the update button and watch the loss.",
  "layers": [2, 2, 1],
  "weights": [[[0.6, -0.4], [0.3, 0.9]], [[1.1, -0.7]]],
  "biases": [[0.1, -0.2], [0.05]],
  "inputs": [0.5, 1],
  "target": [1],
  "activation": "sigmoid",
  "outputActivation": "sigmoid",
  "loss": "mse",
  "lr": 0.5,
  "inputLabels": ["x1", "x2"],
  "note": "The weights are authored for practice; every number in the figure is computed from them.",
  "checks": [
    { "kind": "loss", "value": 0.1117 },
    { "kind": "delta", "layer": 2, "unit": 0, "value": -0.1178 },
    { "kind": "delta", "layer": 1, "unit": 0, "value": -0.0324 },
    { "kind": "dW", "layer": 0, "unit": 1, "from": 1, "value": 0.0173, "tol": 0.001 }
  ]
}
```

The forward pass gives the hidden pre-activations z⁽¹⁾ = (0.00, 0.85), the activations a⁽¹⁾ = (0.5000, 0.7006), then z⁽²⁾ = 1.1×0.5 − 0.7×0.7006 + 0.05 = 0.1096, a⁽²⁾ = 0.5274 and L = ½(0.5274 − 1)² = 0.1117. Now the backward pass, one equation at a time.

```viz
{
  "type": "equation",
  "title": "Equation 1: δ at the output layer",
  "intro": "δ is the blame a unit receives: the derivative of the loss with respect to its pre-activation z.",
  "formula": [
    { "t": "δ⁽ᴸ⁾", "id": "delta" },
    " = ( ",
    { "t": "a⁽ᴸ⁾", "id": "a" },
    " − ",
    { "t": "y", "id": "y" },
    " ) ",
    { "t": "⊙", "id": "odot" },
    " ",
    { "t": "φ′(z⁽ᴸ⁾)", "id": "phip" }
  ],
  "parts": [
    { "id": "delta", "name": "δ⁽ᴸ⁾ (blame at the output)", "meaning": "dL/dz at the last layer: how much the loss changes if the output pre-activation changes." },
    { "id": "a", "name": "a⁽ᴸ⁾ (prediction)", "meaning": "What the network said." },
    { "id": "y", "name": "y (target)", "meaning": "What it should have said." },
    { "id": "odot", "name": "⊙ (element-wise product)", "meaning": "Multiply entry by entry, not as a matrix product." },
    { "id": "phip", "name": "φ′(z⁽ᴸ⁾) (activation slope)", "meaning": "The slope of the activation at z. For a sigmoid it equals a(1 − a)." }
  ],
  "steps": [
    { "title": "Prediction error", "expr": "a − y = 0.5274 − 1 = −0.4726", "note": "Negative, because the output is below the target." },
    { "title": "Activation slope", "expr": "φ′(z) = a(1 − a) = 0.5274 × 0.4726 = 0.2493", "note": "The sigmoid is close to its steepest here (0.25 at most)." },
    { "title": "The blame", "expr": "δ⁽²⁾ = −0.4726 × 0.2493 = −0.1178", "note": "This is the figure's output blame." }
  ],
  "note": "The weights are authored for practice."
}
```

```viz
{
  "type": "equation",
  "title": "Equation 2: δ for a hidden layer",
  "intro": "Blame flows backwards: send the later blame through the transposed weights, then through the local slope.",
  "formula": [
    { "t": "δ⁽ˡ⁾", "id": "delta" },
    " = ( ",
    { "t": "W⁽ˡ⁺¹⁾ᵀ", "id": "WT" },
    " ",
    { "t": "δ⁽ˡ⁺¹⁾", "id": "deltanext" },
    " ) ",
    { "t": "⊙", "id": "odot" },
    " ",
    { "t": "φ′(z⁽ˡ⁾)", "id": "phip" }
  ],
  "parts": [
    { "id": "delta", "name": "δ⁽ˡ⁾ (blame at layer l)", "meaning": "The blame of the units in this hidden layer." },
    { "id": "WT", "name": "W⁽ˡ⁺¹⁾ᵀ (transposed weights)", "meaning": "The weights that connect this layer to the next one, transposed so the blame travels backwards." },
    { "id": "deltanext", "name": "δ⁽ˡ⁺¹⁾ (blame ahead)", "meaning": "The blame already computed for the next layer." },
    { "id": "odot", "name": "⊙", "meaning": "Entry-by-entry product." },
    { "id": "phip", "name": "φ′(z⁽ˡ⁾) (activation slope)", "meaning": "The local slope for each unit of this layer." }
  ],
  "steps": [
    { "title": "Send the blame back", "expr": "W⁽²⁾ᵀδ⁽²⁾ = (1.1 × −0.1178, −0.7 × −0.1178) = (−0.1296, 0.0825)", "note": "Each hidden unit receives its weight times the output blame." },
    { "title": "Local slopes", "expr": "φ′ = a(1 − a) = (0.2500, 0.2098)", "note": "Both are below 0.25." },
    { "title": "Multiply", "expr": "δ⁽¹⁾ = (−0.1296 × 0.2500, 0.0825 × 0.2098) = (−0.0324, 0.0173)", "note": "The hidden blame is much smaller than the output blame: the first sign of fading." }
  ],
  "note": "Same network as above."
}
```

```viz
{
  "type": "equation",
  "title": "Equations 3 and 4: weight and bias gradients",
  "intro": "Once you know a unit's blame, its weight gradient is the blame times the signal that came in.",
  "formula": [
    { "t": "∂L/∂W⁽ˡ⁾", "id": "dW" },
    " = ",
    { "t": "δ⁽ˡ⁾", "id": "delta" },
    " ",
    { "t": "a⁽ˡ⁻¹⁾ᵀ", "id": "aT" },
    "          ",
    { "t": "∂L/∂b⁽ˡ⁾", "id": "db" },
    " = ",
    { "t": "δ⁽ˡ⁾", "id": "delta2" }
  ],
  "parts": [
    { "id": "dW", "name": "∂L/∂W⁽ˡ⁾ (weight gradient)", "meaning": "A matrix the same shape as W⁽ˡ⁾: one gradient per weight." },
    { "id": "delta", "name": "δ⁽ˡ⁾ (blame at layer l)", "meaning": "A column with one entry per unit of layer l." },
    { "id": "aT", "name": "a⁽ˡ⁻¹⁾ᵀ (incoming signal)", "meaning": "The previous layer's activations written as a row." },
    { "id": "db", "name": "∂L/∂b⁽ˡ⁾ (bias gradient)", "meaning": "The bias sees a constant input of 1, so its gradient is just the blame." },
    { "id": "delta2", "name": "δ⁽ˡ⁾", "meaning": "The same blame vector." }
  ],
  "steps": [
    { "title": "Output weights", "expr": "δ⁽²⁾ a⁽¹⁾ᵀ = −0.1178 × (0.5000, 0.7006) = (−0.0589, −0.0825)", "note": "One row, because there is one output unit." },
    { "title": "Hidden weights", "expr": "row 1 = −0.0324 × (0.5, 1) = (−0.0162, −0.0324);  row 2 = 0.0173 × (0.5, 1) = (0.0086, 0.0173)", "note": "A 2 × 2 matrix, the same shape as W⁽¹⁾." },
    { "title": "Biases", "expr": "∂L/∂b⁽¹⁾ = (−0.0324, 0.0173);  ∂L/∂b⁽²⁾ = −0.1178", "note": "Copy the blame." },
    { "title": "Check the shapes", "expr": "δ⁽¹⁾ is 2 × 1 and a⁽⁰⁾ᵀ is 1 × 2, so δaᵀ is 2 × 2", "note": "A shape check catches most mistakes." }
  ],
  "note": "Same network as above. A gradient-descent step then uses w ← w − η × gradient."
}
```

**Watch out:** the weight gradient uses the activations of the *previous* layer, and the blame δ is for the *current* layer. Mixing the two layers up is the most common slip.

### Gradient flow: a product of factors

Every layer the blame passes through multiplies it by a weight and by a local slope. After n layers the gradient is roughly the starting size times a factor f to the power n, where f combines the weights and the slope.

```viz
{
  "type": "equation",
  "title": "The gradient after n layers is a power",
  "intro": "If every layer scales the gradient by about the same factor f, then n layers scale it by fⁿ.",
  "formula": [
    { "t": "gₙ", "id": "g" },
    "  ≈  ",
    { "t": "g₀", "id": "g0" },
    "  ×  ",
    { "t": "f", "id": "f" },
    { "t": "ⁿ", "id": "n" }
  ],
  "parts": [
    { "id": "g", "name": "gₙ (gradient size after n layers)", "meaning": "How big the gradient is after travelling back through n layers." },
    { "id": "g0", "name": "g₀ (starting size)", "meaning": "The size at the output layer." },
    { "id": "f", "name": "f (factor per layer)", "meaning": "Roughly the weight size times the local slope. A sigmoid's slope is at most 0.25." },
    { "id": "n", "name": "n (layers travelled)", "meaning": "The exponent: the number of layers." }
  ],
  "steps": [
    { "title": "A sigmoid network", "expr": "f = 0.25 (slope at its maximum, weights of size about 1)", "note": "The best case for a sigmoid." },
    { "title": "Shrinking", "expr": "0.25⁴ = 0.0039;  0.25⁸ = 1.53 × 10⁻⁵;  0.25¹² = 5.96 × 10⁻⁸", "note": "Vanishing: the early layers get almost nothing." },
    { "title": "Growing", "expr": "f = 1.5:  1.5⁸ = 25.63;  1.5²⁰ = 3325", "note": "Exploding: the early layers get enormous gradients." },
    { "title": "The balance", "expr": "f below 1 vanishes, f above 1 explodes, f near 1 is healthy", "note": "Every remedy tries to keep f near 1." }
  ],
  "note": "The factors are authored to show the effect; real networks have a different factor in each layer."
}
```

```viz
{
  "type": "curve",
  "title": "Gradient left after n layers",
  "xRange": [0, 10],
  "yRange": [-8, 0],
  "logY": true,
  "series": [
    { "fn": "decay", "label": "Factor per layer" }
  ],
  "param": {
    "label": "Factor per layer",
    "min": 0.1,
    "max": 1,
    "step": 0.05,
    "value": 0.25
  },
  "alt": "A line showing the gradient remaining, on a log scale, after 0 to 10 layers when every layer multiplies it by the chosen factor. At 0.25 it falls to about one millionth after 10 layers; at 1.0 it stays flat.",
  "note": "The plot shows factors up to 1 only. Factors above 1 explode, which a log plot of remaining gradient cannot show. The vertical axis is log10 of the remaining gradient."
}
```

Drag the factor to 0.25 and read the value at layer 10: about 10⁻⁶. At 1.0 the line is flat. Now see the same effect in a random network. The bars show how large the gradient of each layer is.

```viz
{
  "type": "gradflow",
  "title": "Gradient size per layer in a 10-layer network",
  "intro": "Start with sigmoid at scale 1, then switch to ReLU at scale 1.41, then try ReLU at 0.7 and at 2.",
  "depth": 10,
  "activation": "sigmoid",
  "scale": 1,
  "width": 16,
  "seed": 5,
  "presets": [
    { "name": "Weights too small (0.7)", "scale": 0.7 },
    { "name": "Xavier-like (1)", "scale": 1 },
    { "name": "He (1.41)", "scale": 1.41 },
    { "name": "Too large (2)", "scale": 2 }
  ],
  "note": "One random network with width 16 and zero biases; the loss is the sum of the outputs. It illustrates the trend, not any particular trained model.",
  "checks": [
    { "depth": 10, "activation": "sigmoid", "scale": 1, "maxRatio": 0.001 },
    { "depth": 10, "activation": "relu", "scale": 1.41, "minRatio": 0.3, "maxRatio": 4 }
  ]
}
```

For sigmoid at scale 1, the first layer's gradient is only about 2 × 10⁻⁶ times the last layer's. With ReLU at the He scale of 1.41, all ten layers are within a factor of about 2 of each other.

### The remedies

| Problem | Remedy | Why it helps |
|---|---|---|
| Vanishing | ReLU or its relatives instead of sigmoid | The slope is 1 for z above 0, so it does not shrink the gradient. Watch for "dying" units that stay at 0. |
| Both | Xavier or He initialisation | Choose the weight spread so signals and gradients keep a similar size from layer to layer. |
| Both | Normalisation | Rescale each layer's values to mean 0 and standard deviation 1, then let the network learn a scale γ and shift β. |
| Vanishing | Residual (skip) connections | The layer computes a + F(a), so the factor per layer is 1 + F′, which stays near 1. |
| Exploding | Gradient clipping | If the gradient's length is above a limit c, scale it down to length c. |

Two remedies are worth remembering by name: a [[residual connection|A shortcut that adds a layer's input to its output, so the layer computes a + F(a).]] and [[gradient clipping|Scaling a gradient down when its length is above a limit c, so one huge gradient cannot wreck the weights.]]. Now the numbers for initialisation.

```viz
{
  "type": "equation",
  "title": "Initialisation: Xavier and He",
  "intro": "Choose the spread of the random starting weights from the number of inputs n_in to the layer.",
  "formula": [
    { "t": "Var(w)", "id": "var" },
    " = ",
    { "t": "1 / nᵢₙ", "id": "xavier" },
    "   (Xavier)          ",
    { "t": "Var(w)", "id": "var2" },
    " = ",
    { "t": "2 / nᵢₙ", "id": "he" },
    "   (He)"
  ],
  "parts": [
    { "id": "var", "name": "Var(w) (variance of the weights)", "meaning": "How spread out the random starting weights are. The standard deviation is its square root." },
    { "id": "xavier", "name": "1 / nᵢₙ (Xavier)", "meaning": "A good default for tanh or sigmoid. nᵢₙ is the number of inputs to the layer." },
    { "id": "var2", "name": "Var(w)", "meaning": "The same quantity, for the second rule." },
    { "id": "he", "name": "2 / nᵢₙ (He)", "meaning": "For ReLU: the factor 2 compensates for ReLU switching off half the units." }
  ],
  "steps": [
    { "title": "A layer with 400 inputs", "expr": "nᵢₙ = 400", "note": "Made-up layer size." },
    { "title": "Xavier", "expr": "Var = 1/400 = 0.0025;  std = √0.0025 = 0.0500", "note": "Weights drawn with this standard deviation." },
    { "title": "He", "expr": "Var = 2/400 = 0.0050;  std = √0.0050 = 0.0707", "note": "About 1.41 times larger than Xavier." },
    { "title": "Why not std 1?", "expr": "with 400 inputs the sum would have a spread about √400 = 20 times too big", "note": "That would make the signals, and then the gradients, explode." }
  ],
  "note": "The layer size is made up for practice."
}
```

```viz
{
  "type": "sorter",
  "title": "Vanishing, exploding or both?",
  "intro": "Place each symptom or remedy. Remember: clipping is for exploding gradients; ReLU mainly fights vanishing ones.",
  "bins": [
    { "id": "vanishing", "label": "Vanishing" },
    { "id": "exploding", "label": "Exploding" }
  ],
  "cards": [
    { "name": "Loss becomes NaN after a few steps", "answer": "exploding", "why": "Huge gradients make the weights jump to enormous values, and the loss overflows." },
    { "name": "First-layer gradients are 10⁻⁹ while the last layer's are 10⁻¹", "answer": "vanishing", "why": "The early layers receive almost nothing, so they barely learn." },
    { "name": "Gradient clipping", "answer": "exploding", "why": "It shortens a gradient whose length is above a limit; it does nothing for tiny gradients." },
    { "name": "ReLU instead of sigmoid in the hidden layers", "answer": "vanishing", "why": "The slope is 1 for z above 0, so the gradient is not multiplied by 0.25 at every layer." },
    { "name": "He initialisation", "answer": "either", "why": "Choosing the weight spread keeps the factor per layer near 1, against both problems." },
    { "name": "Weights initialised with standard deviation 5", "answer": "exploding", "why": "A factor far above 1 per layer is raised to the power of the depth." },
    { "name": "Twenty stacked sigmoid layers", "answer": "vanishing", "why": "The sigmoid slope is at most 0.25, so 0.25 to the power 20 is tiny." },
    { "name": "Normalisation layers", "answer": "either", "why": "Rescaling each layer's values stops the signals from shrinking or growing from layer to layer." }
  ]
}
```

### Practice problems

**Problem 1. A forward and backward pass.** One input x = 1.5 goes through two sigmoid neurons in a row: z₁ = 0.8x + 0.1, a₁ = σ(z₁), z₂ = −1.2a₁ + 0.3, a₂ = σ(z₂). The target is y = 1 and the loss is ½(a₂ − y)². Find (a) the loss, (b) δ₂ and δ₁, (c) ∂L/∂w for the weights 0.8 and −1.2.

**Solution 1.**

- (a) z₁ = 0.8×1.5 + 0.1 = 1.3, a₁ = σ(1.3) = 0.7858. z₂ = −1.2×0.7858 + 0.3 = −0.6430, a₂ = σ(−0.6430) = 0.3446. L = ½(0.3446 − 1)² = ==0.2148==.
- (b) δ₂ = (a₂ − y)·a₂(1 − a₂) = (−0.6554) × 0.2258 = ==−0.1480==. Then δ₁ = w₂ × δ₂ × a₁(1 − a₁) = (−1.2) × (−0.1480) × 0.1683 = ==0.0299==. The sign flips because the weight is negative.
- (c) ∂L/∂w₂ = δ₂ × a₁ = −0.1480 × 0.7858 = ==−0.1163==. ∂L/∂w₁ = δ₁ × x = 0.0299 × 1.5 = ==0.0448==. The bias gradients are δ₂ and δ₁.

**Watch out:** the first weight's gradient is about 2.6 times smaller than the last weight's, even in a net with two layers. This is the same fading, in miniature.

**Problem 2. Gradient flow.** Each sigmoid layer multiplies the gradient by at most 0.25. (a) What is the largest possible gradient factor after 12 layers? (b) How many layers does it take for the gradient to fall below 10⁻⁶? (c) If an exploding network multiplies the gradient by 1.5 per layer, after how many layers does the gradient exceed 1000 times its starting size?

**Solution 2.**

- (a) 0.25¹² = ==5.96 × 10⁻⁸==.
- (b) We need 0.25ⁿ < 10⁻⁶, so n > 6 / log₁₀(4) = 6 / 0.602 = 9.97. Check: 0.25⁹ = 3.81 × 10⁻⁶ is still above 10⁻⁶, and 0.25¹⁰ = 9.54 × 10⁻⁷ is below. So ==10 layers==.
- (c) We need 1.5ⁿ > 1000, so n > ln 1000 / ln 1.5 = 17.04. Check: 1.5¹⁷ = 985 and 1.5¹⁸ = 1478. So ==18 layers==.

**Problem 3. Remedies.** (a) A ReLU layer has 512 inputs. What are the He and Xavier standard deviations? (b) A gradient is g = (6, −8) and the clipping limit is c = 2. What is the clipped gradient? (c) Training prints a loss of NaN after three steps. Name two changes that may help.

**Solution 3.**

- (a) He: Var = 2/512, so std = √(2/512) = ==0.0625==. Xavier: Var = 1/512, so std = √(1/512) = 0.0442.
- (b) ‖g‖ = √(6² + 8²) = √100 = 10. The factor is min(1, c/‖g‖) = min(1, 2/10) = 0.2. The clipped gradient is (6, −8) × 0.2 = ==(1.2, −1.6)==, whose length is 2. The direction is unchanged.
- (c) A NaN loss after a few steps is a classic symptom of exploding gradients or a learning rate that is too large. Use ==gradient clipping==, ==lower the learning rate==, or check the initialisation (He for ReLU).

## Week 4: convolutional networks {rail=Week 4}

{lede} Week 4 asked why a dense layer is a poor fit for pictures and what to use instead. A [[convolution|Sliding a small grid of weights, called a kernel, over an image and computing a weighted sum at each position.]] uses ==local connectivity (each output looks at a small patch) and weight sharing (the same kernel is reused everywhere)==, so it needs far fewer parameters and finds a pattern wherever it appears.

### You must be able to

- Count the parameters of a dense layer on an image and of a convolution layer, and explain why convolution needs far fewer: local connectivity and [[weight sharing|Using the same kernel weights at every position of the image, so one small set of numbers is learned once.]].
- Compute a convolution by hand (the deep-learning kind: no kernel flip), then add a bias and apply ReLU.
- Compute the output size ⌊(n + 2p − k)/s⌋ + 1 and the pooled size ⌊(n − k)/s⌋ + 1.
- Count conv parameters as k·k·Cᵢₙ·Cₒᵤₜ + Cₒᵤₜ, and follow the shape of a picture through a whole CNN.
- Compute the [[receptive field|The patch of the original image that can affect one output value; it grows as layers are stacked.]] of a stack of conv and pooling layers.
- Compute dL/dK and dL/dx for a convolution by hand, and the backward pass through ReLU and max pooling.
- Recognise typical design choices: pooling, stride, padding, 1×1 convolutions and data augmentation.

### Dense versus convolution

A 32×32 colour picture has 32 × 32 × 3 = 3,072 numbers. A dense layer with 100 units connects every number to every unit: 3,072 × 100 + 100 = 307,300 parameters, and it treats a pixel in the corner like any other. A convolution layer with 16 filters of size 3×3 has 3 × 3 × 3 × 16 + 16 = 448 parameters, whatever the picture size. Open the calculator, set the picture size to 32 and the units to 100, and read the two totals.

```viz
{
  "type": "calc",
  "kind": "fc_vs_conv",
  "title": "Parameters: dense layer or convolution?",
  "channels": 3,
  "filters": 16,
  "kernel": 3,
  "intro": "Set the picture size to 32 and the dense layer to 100 units; the convolution has 16 filters of 3 × 3."
}
```

==The number of parameters of a convolution does not depend on the picture size==, because the same kernel slides over every position.

### Computing a convolution

```viz
{
  "type": "equation",
  "title": "The convolution: y[i, j] = Σ Σ x[i + a, j + c] · K[a, c] + b",
  "intro": "Slide the kernel over the picture. At each position multiply entry by entry and add up.",
  "formula": [
    { "t": "y[i, j]", "id": "y" },
    " = ",
    { "t": "Σₐ Σ_c", "id": "sum" },
    " ",
    { "t": "x[i + a, j + c]", "id": "x" },
    " · ",
    { "t": "K[a, c]", "id": "K" },
    " + ",
    { "t": "b", "id": "b" }
  ],
  "parts": [
    { "id": "y", "name": "y[i, j] (output)", "meaning": "One number of the feature map: the response of the kernel at position (i, j)." },
    { "id": "sum", "name": "Σₐ Σ_c (sum over the window)", "meaning": "Add the products for every kernel position (a, c)." },
    { "id": "x", "name": "x[i + a, j + c] (picture value)", "meaning": "The picture value currently under kernel position (a, c)." },
    { "id": "K", "name": "K[a, c] (kernel weight)", "meaning": "The learned number at kernel position (a, c). The same kernel is used at every (i, j)." },
    { "id": "b", "name": "b (bias)", "meaning": "One learned number added to every output of the filter." }
  ],
  "steps": [
    { "title": "The data", "expr": "x = (2, 0, 1, 3; 1, 4, 0, 2; 0, 1, 2, 1; 3, 0, 1, 2) (4 × 4),  K = (1, −1; 2, 0) (2 × 2)", "note": "Made-up values for practice. The output is 3 × 3." },
    { "title": "Top-left", "expr": "y[0,0] = 2×1 + 0×(−1) + 1×2 + 4×0 = 4", "note": "The window is the top-left 2 × 2 patch." },
    { "title": "Next to it", "expr": "y[0,1] = 0×1 + 1×(−1) + 4×2 + 0×0 = 7", "note": "Slide one column to the right." },
    { "title": "Second row", "expr": "y[1,2] = 0×1 + 2×(−1) + 2×2 + 1×0 = 2", "note": "Slide one row down and two columns right." },
    { "title": "With bias and ReLU", "expr": "with b = 1:  y[0,2] = −2 + 1 = −1,  ReLU gives 0;  y[0,0] = 5", "note": "Add the bias first, then apply ReLU." }
  ],
  "note": "The numbers are made up for practice; the arithmetic is exact."
}
```

Check your hand work with the convolution figure.

```viz
{
  "type": "conv",
  "title": "Convolve a 4 × 4 picture with a 2 × 2 kernel",
  "mode": "conv",
  "stride": 1,
  "padding": 0,
  "input": [[2, 0, 1, 3], [1, 4, 0, 2], [0, 1, 2, 1], [3, 0, 1, 2]],
  "kernel": [[1, -1], [2, 0]]
}
```

Now see what different kernels detect. Pick a preset and watch the feature map light up where its pattern appears.

```viz
{
  "type": "filters",
  "title": "Filters as pattern detectors",
  "intro": "Choose vertical edges, then horizontal edges. Turn ReLU on and notice that negative responses disappear.",
  "image": ["00000000", "00000000", "00999900", "00999900", "00999900", "00999900", "00000000", "00000000"],
  "presets": [
    { "name": "vertical edges", "kernel": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], "note": "Responds where brightness changes from left to right: the left and right sides of the block." },
    { "name": "horizontal edges", "kernel": [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], "note": "Responds where brightness changes from top to bottom." },
    { "name": "blur", "kernel": [[1, 1, 1], [1, 1, 1], [1, 1, 1]], "note": "Adds up a neighbourhood, so the picture gets smoother and brighter." },
    { "name": "sharpen", "kernel": [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], "note": "Boosts the centre against its neighbours." }
  ],
  "relu": false,
  "padding": 0,
  "note": "A tiny made-up picture: a bright block on a dark background."
}
```

### Output size and parameters

The [[stride|How many pixels the kernel moves between two positions.]] and the padding decide the size of the [[feature map|The grid of outputs made by one filter sliding over the picture.]].

```viz
{
  "type": "equation",
  "title": "Output size: ⌊(n + 2p − k) / s⌋ + 1",
  "intro": "Count how many positions the kernel can take along one side.",
  "formula": [
    { "t": "nₒᵤₜ", "id": "nout" },
    " = ⌊ (",
    { "t": "n", "id": "n" },
    " + 2",
    { "t": "p", "id": "p" },
    " − ",
    { "t": "k", "id": "k" },
    ") / ",
    { "t": "s", "id": "s" },
    " ⌋ + 1"
  ],
  "parts": [
    { "id": "nout", "name": "nₒᵤₜ (output size)", "meaning": "The width or height of the feature map." },
    { "id": "n", "name": "n (input size)", "meaning": "The width or height of the input." },
    { "id": "p", "name": "p (padding)", "meaning": "Zeros added around each side, so 2p is added in total." },
    { "id": "k", "name": "k (kernel size)", "meaning": "The width of the kernel." },
    { "id": "s", "name": "s (stride)", "meaning": "How many pixels the kernel jumps each time. The floor ⌊ ⌋ drops any leftover." }
  ],
  "steps": [
    { "title": "A big first layer", "expr": "n = 64, k = 7, p = 3, s = 2:  ⌊(64 + 6 − 7)/2⌋ + 1 = ⌊31.5⌋ + 1 = 32", "note": "The floor of 31.5 is 31." },
    { "title": "Same size", "expr": "n = 10, k = 5, p = 2, s = 1:  ⌊(10 + 4 − 5)/1⌋ + 1 = 10", "note": "Padding (k − 1)/2 with stride 1 keeps the size." },
    { "title": "Stride 2, no padding", "expr": "n = 15, k = 3, p = 0, s = 2:  ⌊12/2⌋ + 1 = 7", "note": "The kernel fits exactly." },
    { "title": "Does not fit exactly", "expr": "n = 4, k = 2, p = 0, s = 2 gives 2, but s = 3 gives ⌊2/3⌋ + 1 = 1", "note": "Leftover pixels are dropped." }
  ],
  "note": "The sizes are made up for practice."
}
```

```viz
{
  "type": "equation",
  "title": "Parameters of a convolution layer",
  "intro": "One bias per filter, plus one weight for every kernel entry and every input channel.",
  "formula": [
    { "t": "P", "id": "P" },
    " = ",
    { "t": "k · k", "id": "kk" },
    " · ",
    { "t": "C_in", "id": "cin" },
    " · ",
    { "t": "C_out", "id": "cout" },
    " + ",
    { "t": "C_out", "id": "cout2" }
  ],
  "parts": [
    { "id": "P", "name": "P (parameters)", "meaning": "The number of numbers the layer learns." },
    { "id": "kk", "name": "k · k (kernel size)", "meaning": "The area of one kernel slice." },
    { "id": "cin", "name": "C_in (input channels)", "meaning": "A kernel is as deep as the input: 3 for a colour picture." },
    { "id": "cout", "name": "C_out (filters)", "meaning": "The number of filters, which is also the number of output channels." },
    { "id": "cout2", "name": "C_out (biases)", "meaning": "One bias per filter." }
  ],
  "steps": [
    { "title": "Colour input", "expr": "5 × 5 kernels, C_in = 6, C_out = 12:  5·5·6·12 + 12 = 1,800 + 12 = 1,812", "note": "A made-up layer, as in a first colour-image layer." },
    { "title": "A 1 × 1 convolution", "expr": "1·1·64·32 + 32 = 2,048 + 32 = 2,080", "note": "It mixes channels at each pixel and changes the number of channels." },
    { "title": "Grey input", "expr": "3 × 3 kernels, C_in = 1, C_out = 4:  3·3·1·4 + 4 = 40", "note": "Independent of the picture size." }
  ],
  "note": "The layers are made up for practice."
}
```

### Pooling and the receptive field

Max pooling keeps the largest number in each window; average pooling keeps the mean. A 2×2 window with stride 2 halves the width and the height and has no parameters. Try it on a new 4 × 4 grid.

```viz
{
  "type": "pool",
  "title": "Max and average pooling",
  "intro": "Switch between max and average, and select an output cell to see its window.",
  "input": [[1, 3, 2, 0], [4, 2, 1, 5], [0, 6, 3, 1], [2, 1, 4, 2]],
  "size": 2,
  "stride": 2,
  "note": "Max pooling gives (4, 5; 6, 4); average pooling gives (2.5, 2; 2.25, 2.5)."
}
```

```viz
{
  "type": "equation",
  "title": "Receptive field",
  "intro": "Each layer widens the patch of the original picture that one unit can see.",
  "formula": [
    { "t": "rf", "id": "rf" },
    " = ",
    { "t": "rf_prev", "id": "rfp" },
    " + (",
    { "t": "k", "id": "k" },
    " − 1) · ",
    { "t": "jump_prev", "id": "jp" },
    ";      ",
    { "t": "jump", "id": "jump" },
    " = ",
    { "t": "jump_prev", "id": "jp2" },
    " · ",
    { "t": "s", "id": "s" }
  ],
  "parts": [
    { "id": "rf", "name": "rf (receptive field)", "meaning": "The width of the patch of the original image that one unit depends on." },
    { "id": "rfp", "name": "rf_prev", "meaning": "The receptive field of the previous layer; the input pixel has rf = 1." },
    { "id": "k", "name": "k (kernel or window size)", "meaning": "The width of the kernel or pooling window." },
    { "id": "jp", "name": "jump_prev", "meaning": "The distance, in input pixels, between neighbouring units of the previous layer. It starts at 1." },
    { "id": "jump", "name": "jump (this layer)", "meaning": "The distance between neighbouring units of this layer." },
    { "id": "jp2", "name": "jump_prev", "meaning": "The previous jump." },
    { "id": "s", "name": "s (stride)", "meaning": "A stride above 1 makes the jump, and later growth, larger." }
  ],
  "steps": [
    { "title": "Start", "expr": "rf = 1, jump = 1", "note": "A single input pixel." },
    { "title": "Conv 3×3, s = 1", "expr": "rf = 1 + (3 − 1)·1 = 3;  jump = 1", "note": "The first layer sees a 3 × 3 patch." },
    { "title": "Conv 3×3, s = 1", "expr": "rf = 3 + 2·1 = 5;  jump = 1", "note": "Two 3 × 3 layers see 5 × 5." },
    { "title": "Pool 2×2, s = 2", "expr": "rf = 5 + (2 − 1)·1 = 6;  jump = 2", "note": "The stride doubles the jump." },
    { "title": "Conv 3×3, s = 1", "expr": "rf = 6 + 2·2 = 10;  jump = 2", "note": "Later layers grow faster." }
  ],
  "note": "A made-up stack for practice."
}
```

```viz
{
  "type": "cnnstack",
  "title": "A small CNN for 28 × 28 pictures",
  "intro": "Change a kernel size, padding or filter count and watch the shapes, parameters and receptive field update.",
  "input": [28, 28, 1],
  "layers": [
    { "type": "conv", "k": 3, "filters": 8, "pad": 1 },
    { "type": "pool", "k": 2 },
    { "type": "conv", "k": 3, "filters": 16, "pad": 1 },
    { "type": "pool", "k": 2 },
    { "type": "flatten" },
    { "type": "fc", "units": 64 },
    { "type": "fc", "units": 10 }
  ],
  "checks": [
    { "layer": 0, "shape": [28, 28, 8], "params": 80, "rf": 3 },
    { "layer": 2, "shape": [14, 14, 16], "params": 1168, "rf": 8 },
    { "total": 52138 }
  ]
}
```

The first layer has 3·3·1·8 + 8 = 80 parameters, the second conv layer has 3·3·8·16 + 16 = 1168, and the dense layer 784 → 64 has 50,240. The total is 52,138. ==Most parameters sit in the dense layer==, even though the convolutions do most of the looking.

### Backpropagation through a convolution

A convolution layer receives an upstream gradient G = dL/dy, the same size as its output, and must produce dL/dK to update the kernel and dL/dx to pass backwards. Use the same x and K as above, and the upstream gradient G = (1, 0, −1; 2, 1, 0; 0, −1, 1).

```viz
{
  "type": "equation",
  "title": "The kernel gradient: dL/dK[a, c] = Σ G[i, j] · x[i + a, j + c]",
  "intro": "Each kernel entry is shared by every output position, so its gradient adds up the contribution of all of them.",
  "formula": [
    { "t": "∂L/∂K[a, c]", "id": "dK" },
    " = ",
    { "t": "Σᵢ,ⱼ", "id": "sum" },
    " ",
    { "t": "G[i, j]", "id": "G" },
    " · ",
    { "t": "x[i + a, j + c]", "id": "x" }
  ],
  "parts": [
    { "id": "dK", "name": "∂L/∂K[a, c] (kernel gradient)", "meaning": "How the loss changes if kernel entry (a, c) changes." },
    { "id": "sum", "name": "Σᵢ,ⱼ (sum over every output position)", "meaning": "The kernel entry was used once at every output position." },
    { "id": "G", "name": "G[i, j] (upstream gradient)", "meaning": "dL/dy at output position (i, j)." },
    { "id": "x", "name": "x[i + a, j + c] (input value)", "meaning": "The input value that kernel entry (a, c) multiplied at that position." }
  ],
  "steps": [
    { "title": "dL/dK[0,0]", "expr": "1×2 + 0×0 + (−1)×1 + 2×1 + 1×4 + 0×0 + 0×0 + (−1)×1 + 1×2 = 8", "note": "Nine terms: one per output position." },
    { "title": "dL/dK[1,1]", "expr": "1×4 + 0×0 + (−1)×2 + 2×1 + 1×2 + 0×1 + 0×0 + (−1)×1 + 1×2 = 7", "note": "Same G, shifted input patch." },
    { "title": "The whole matrix", "expr": "dL/dK = (8, 4; 3, 7)", "note": "Same shape as K (2 × 2). Update with K ← K − η dL/dK." },
    { "title": "Bias", "expr": "dL/db = sum of G = 3", "note": "The bias is added at every position." }
  ],
  "note": "Same x, K and G as above."
}
```

```viz
{
  "type": "equation",
  "title": "The input gradient: dL/dx[p, q] = Σ G[i, j] · K[p − i, q − j]",
  "intro": "Each input pixel collects blame from every output it contributed to, through the kernel weight that linked them.",
  "formula": [
    { "t": "∂L/∂x[p, q]", "id": "dx" },
    " = ",
    { "t": "Σᵢ,ⱼ", "id": "sum" },
    " ",
    { "t": "G[i, j]", "id": "G" },
    " · ",
    { "t": "K[p − i, q − j]", "id": "K" }
  ],
  "parts": [
    { "id": "dx", "name": "∂L/∂x[p, q] (input gradient)", "meaning": "How the loss changes if input pixel (p, q) changes." },
    { "id": "sum", "name": "Σᵢ,ⱼ (over the outputs that used this pixel)", "meaning": "Only outputs whose window contained the pixel count; the others add nothing." },
    { "id": "G", "name": "G[i, j] (upstream gradient)", "meaning": "dL/dy at those outputs." },
    { "id": "K", "name": "K[p − i, q − j] (the linking weight)", "meaning": "The kernel entry that connected pixel (p, q) to output (i, j)." }
  ],
  "steps": [
    { "title": "A corner pixel", "expr": "dL/dx[0,0] = G[0,0]·K[0,0] = 1×1 = 1", "note": "Only the top-left output used pixel (0,0)." },
    { "title": "A middle pixel", "expr": "dL/dx[1,1] = G[0,0]·K[1,1] + G[0,1]·K[1,0] + G[1,0]·K[0,1] + G[1,1]·K[0,0] = 1×0 + 0×2 + 2×(−1) + 1×1 = −1", "note": "Four outputs used this pixel, so there are four terms." },
    { "title": "The whole matrix", "expr": "dL/dx = (1, −1, −1, 1; 4, −1, −3, 0; 4, 1, 2, −1; 0, −2, 2, 0)", "note": "Same shape as the input (4 × 4)." }
  ],
  "note": "Same x, K and G as above."
}
```

Check your hand calculation of both gradients here.

```viz
{
  "type": "convgrad",
  "title": "The gradients of a convolution",
  "intro": "Select an input cell to list the terms of dL/dx, and a kernel cell to list the terms of dL/dK.",
  "input": [[2, 0, 1, 3], [1, 4, 0, 2], [0, 1, 2, 1], [3, 0, 1, 2]],
  "kernel": [[1, -1], [2, 0]],
  "upstream": [[1, 0, -1], [2, 1, 0], [0, -1, 1]],
  "stride": 1,
  "padding": 0,
  "note": "The values are made up for practice. Convolution here is the deep-learning kind (no kernel flip).",
  "checks": [
    { "kind": "y", "i": 0, "j": 0, "value": 4 },
    { "kind": "dK", "i": 0, "j": 0, "value": 8 },
    { "kind": "dK", "i": 1, "j": 1, "value": 7 },
    { "kind": "dx", "i": 1, "j": 1, "value": -1 }
  ]
}
```

Two more backward rules. ==ReLU lets the gradient through where z was above 0 and blocks it elsewhere==: δ = G ⊙ 1[z > 0]. ==Max pooling sends the whole gradient to the position that held the maximum== and gives 0 to the others; average pooling shares it equally. The pooling layers have no parameters to update, but they decide which pixels receive gradient.

```viz
{
  "type": "sorter",
  "title": "Convolution, pooling or ReLU?",
  "intro": "Place each statement about the backward pass or the layer by the layer it describes.",
  "bins": [
    { "id": "conv", "label": "Convolution layer" },
    { "id": "pool", "label": "Max pooling" },
    { "id": "relu", "label": "ReLU" }
  ],
  "cards": [
    { "name": "Its kernel gradient is a sum over every output position", "answer": "conv", "why": "The same kernel entry is used at every position, so all of them contribute to its gradient." },
    { "name": "The gradient goes only to the position of the maximum in each window", "answer": "pool", "why": "Only the maximum influenced the output, so only it receives blame." },
    { "name": "The gradient is multiplied by 1 where z is above 0 and by 0 elsewhere", "answer": "relu", "why": "Its slope is 1 for positive z and 0 for negative z." },
    { "name": "It has weights that are updated by gradient descent", "answer": "conv", "why": "The kernel and the bias are learned; pooling and ReLU have no parameters." },
    { "name": "It has no parameters and halves the width and height when the window and stride are 2", "answer": "pool", "why": "It summarises each window with its largest value." },
    { "name": "Units whose z is negative on every input receive no gradient", "answer": "relu", "why": "This is the dying ReLU problem: their gradient is always 0." },
    { "name": "Its input gradient is another convolution of the upstream gradient with the kernel", "answer": "conv", "why": "dL/dx sums G times the linking kernel weights." },
    { "name": "Tied maxima need a rule, for example giving the gradient to the first one", "answer": "pool", "why": "Only a max has to decide which element was \"the\" maximum." }
  ]
}
```

### Practice problems

**Problem 1. Shapes and parameters.** A 32×32×3 picture goes through: conv 5×5 with 8 filters and padding 2 (stride 1), max pool 2×2, conv 3×3 with 16 filters and no padding, max pool 2×2, flatten, dense 10. (a) Give the shape after every layer. (b) Give the parameters of every layer and the total. (c) What is the receptive field after the second pooling layer?

**Solution 1.**

- (a) Conv 1: ⌊(32 + 4 − 5)/1⌋ + 1 = 32, so ==32×32×8==. Pool: ⌊(32 − 2)/2⌋ + 1 = 16, so 16×16×8. Conv 2: ⌊(16 − 3)/1⌋ + 1 = 14, so ==14×14×16==. Pool: 7×7×16. Flatten: 7 × 7 × 16 = 784. Dense: 10.
- (b) Conv 1: 5·5·3·8 + 8 = ==608==. Conv 2: 3·3·8·16 + 16 = ==1,168==. Dense: 784 × 10 + 10 = ==7,850==. Pooling and flatten: 0. Total 608 + 1,168 + 7,850 = ==9,626==.
- (c) After conv 1: rf = 5, jump 1. After pool 1: rf = 5 + 1·1 = 6, jump 2. After conv 2: rf = 6 + 2·2 = 10. After pool 2: rf = 10 + 1·2 = ==12==.

**Watch out:** the dense layer holds 82 percent of the parameters here, and the kernel depth equals the input channels (3 in conv 1, 8 in conv 2).

**Problem 2. A convolution by hand.** Use x = (2, 0, 1, 3; 1, 4, 0, 2; 0, 1, 2, 1; 3, 0, 1, 2) and K = (1, −1; 2, 0) with bias 0 and stride 1. (a) Compute the output y. (b) Apply ReLU. (c) What size is the output with stride 2 and no padding, and what are its values?

**Solution 2.**

(a) The output is 3×3. Each entry is a sum of four products:

| Position | Products | Value |
|---|---|---|
| y[0,0] | 2×1 + 0×(−1) + 1×2 + 4×0 | 4 |
| y[0,1] | 0×1 + 1×(−1) + 4×2 + 0×0 | 7 |
| y[0,2] | 1×1 + 3×(−1) + 0×2 + 2×0 | −2 |
| y[1,0] | 1×1 + 4×(−1) + 0×2 + 1×0 | −3 |
| y[1,1] | 4×1 + 0×(−1) + 1×2 + 2×0 | 6 |
| y[1,2] | 0×1 + 2×(−1) + 2×2 + 1×0 | 2 |
| y[2,0] | 0×1 + 1×(−1) + 3×2 + 0×0 | 5 |
| y[2,1] | 1×1 + 2×(−1) + 0×2 + 1×0 | −1 |
| y[2,2] | 2×1 + 1×(−1) + 1×2 + 2×0 | 3 |

- So y = ==(4, 7, −2; −3, 6, 2; 5, −1, 3)==.
- (b) ReLU replaces negatives by 0: ==(4, 7, 0; 0, 6, 2; 5, 0, 3)==.
- (c) ⌊(4 − 2)/2⌋ + 1 = ==2==, so a 2×2 output. With stride 2 the windows start at (0,0), (0,2), (2,0), (2,2), giving ==(4, −2; 5, 3)==, the entries y[0,0], y[0,2], y[2,0] and y[2,2] of the stride-1 output.

**Problem 3. Backward pass.** With the same x and K, the upstream gradient is G = (1, 0, −1; 2, 1, 0; 0, −1, 1). (a) Find dL/dK[0,0]. (b) Find dL/dx[1,1]. (c) A max-pool window holds [ [1, 3], [4, 2] ] and the upstream gradient for it is 2.5. What gradient does each of the four inputs receive?

**Solution 3.**

- (a) dL/dK[0,0] = Σ G[i,j] · x[i,j] = 1×2 + 0×0 + (−1)×1 + 2×1 + 1×4 + 0×0 + 0×0 + (−1)×1 + 1×2 = ==8==.
- (b) Pixel (1,1) was used by the outputs (0,0), (0,1), (1,0), (1,1). The weights linking them are K[1,1], K[1,0], K[0,1] and K[0,0], so dL/dx[1,1] = 1×0 + 0×2 + 2×(−1) + 1×1 = ==−1==.
- (c) The maximum is 4, at the bottom-left position. It receives the whole 2.5; the others get 0. The gradient is ==[ [0, 0], [2.5, 0] ]==.

## Revision plan and formula sheet {rail=Revision}

{lede} Part A is closed book, so you need the formulas in your head, not on a page. ==Revise by doing, not by rereading==: cover the solution, try the problem, then check.

### The formula sheet

Every formula of Weeks 1 to 4 on one page. For each one, make sure you can say what every symbol means and do the example without looking.

| Formula | What it is | Example |
|---|---|---|
| pᵢ = e^(zᵢ/T) / Σⱼ e^(zⱼ/T) | Softmax with temperature: scores to probabilities | scores (3.0, 2.0, 0.5) at T = 1 give 0.690, 0.254, 0.057 |
| P(w₁ … wₙ) = Π P(wᵢ given the earlier tokens) | Chain rule of probability for a text | 0.25 × 0.1 × 0.2 × 0.4 = 0.002 |
| Perplexity = P^(−1/n) | Average confusion per token; lower is better | 0.002 to the power −1/4 = 4.73 |
| Top-k, top-p | Keep the k best tokens, or the smallest set with total probability at least p; then renormalise | (0.46, 0.27, 0.15, 0.08, 0.04) with top-p 0.8 keeps three tokens, total 0.88 |
| loss = −ln p | Training loss for one token or one example (cross-entropy) | −ln 0.2 = 1.609 |
| xₜ = √ᾱₜ · x₀ + √(1−ᾱₜ) · ε | Forward diffusion step | t = 500, x₀ = 0.8, ε = −0.4 gives −0.160 |
| L = (1/n) Σ (ŷᵢ − yᵢ)² | Mean squared error (numbers) | mistakes −2, +5, 0, −3 give 38 / 4 = 9.5 |
| L = −ln p (p of the true class) | Cross-entropy for one example (classes) | p = 0.02 gives 3.912 |
| z = Σ wᵢxᵢ + b,  a = φ(z) | One neuron | z = 0.3, sigmoid(0.3) = 0.574, ReLU(0.3) = 0.3 |
| a = φ(Wx + b) | A layer; each row of W is one neuron | x = (1, −2, 0.5) gives z = (0.1, −0.8), ReLU output (0.1, 0) |
| Dense layer parameters = inputs × outputs + outputs | Count of weights and biases | 5 → 4 → 3: (5·4 + 4) + (4·3 + 3) = 39 |
| pᵢ = e^zᵢ / Σⱼ e^zⱼ | Softmax (T = 1) for class scores | scores (1, 0, −1) give 0.665, 0.245, 0.090 |
| W₂(W₁x) = (W₂W₁)x | Stacked linear layers collapse into one | a 3 → 4 → 2 linear stack equals one 2 × 3 matrix |
| w ← w − η · ∂L/∂w | Gradient-descent update | w = 1, η = 0.05, gradient −48 gives w = 3.4 |
| δ⁽ᴸ⁾ = (a − y) ⊙ φ′(z) | Blame at the output layer, for L = ½(a − y)² | (−0.4726) × 0.2493 = −0.1178 |
| δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾ᵀ δ⁽ˡ⁺¹⁾) ⊙ φ′(z⁽ˡ⁾) | Blame at a hidden layer | δ⁽¹⁾ = (−0.0324, 0.0173) |
| ∂L/∂W⁽ˡ⁾ = δ⁽ˡ⁾ a⁽ˡ⁻¹⁾ᵀ,  ∂L/∂b⁽ˡ⁾ = δ⁽ˡ⁾ | Weight and bias gradients | output weights: −0.0589, −0.0825 |
| σ′ = a(1 − a) ≤ 0.25,  tanh′ = 1 − a² ≤ 1,  ReLU′ = 1 if z > 0, else 0 | Slopes of the activations | sigmoid at a = 0.5 has slope 0.25 |
| gₙ ≈ g₀ · fⁿ | Gradient after n layers with factor f per layer | 0.25 to the power 8 = 1.5 × 10⁻⁵;  1.5 to the power 20 = 3,325 |
| std = √(1/nᵢₙ) (Xavier),  std = √(2/nᵢₙ) (He) | Initial weight spread | nᵢₙ = 400 gives 0.050 and 0.071 |
| g ← g · min(1, c/‖g‖) | Gradient clipping | g = (6, −8), c = 2 gives (1.2, −1.6) |
| x̂ = (x − μ)/σ,  then γx̂ + β | Normalisation | values (10, 20, 30) give −1.225, 0.000, 1.225 |
| a⁽ˡ⁺¹⁾ = a⁽ˡ⁾ + F(a⁽ˡ⁾), so the factor is 1 + F′ | Residual connection | F′ = 0.05 gives 1.05; after 10 layers 1.63 |
| y[i, j] = Σ x[i + a, j + c] · K[a, c] + b | Convolution (no kernel flip) | a 2 × 2 kernel on a 4 × 4 picture: y[0,0] = 4 |
| nₒᵤₜ = ⌊(n + 2p − k)/s⌋ + 1 | Output size of a convolution | n = 64, k = 7, p = 3, s = 2 gives 32 |
| P = k · k · Cᵢₙ · Cₒᵤₜ + Cₒᵤₜ | Parameters of a convolution layer | 5·5·6·12 + 12 = 1,812 |
| nₒᵤₜ = ⌊(n − k)/s⌋ + 1 | Output size of pooling | 28, window 2, stride 2 gives 14 |
| rf = previous rf + (k − 1) · previous jump;  jump = previous jump · s | Receptive field | conv 3, conv 3, pool 2 (stride 2), conv 3 gives 3, 5, 6, 10 |
| ∂L/∂K[a, c] = Σ G[i, j] · x[i + a, j + c] | Kernel gradient | the worked 4 × 4 example gives dL/dK[0,0] = 8 |
| ∂L/∂x[p, q] = Σ G[i, j] · K[p − i, q − j] | Input gradient | the same example gives dL/dx[1,1] = −1 |
| δ = G ⊙ 1[z > 0];  max pool sends G to the maximum | Backward through ReLU and max pooling | z = (−2, 1, 3, −0.5), G = (0.9, −0.3, 0.6, 0.2) gives δ = (0, −0.3, 0.6, 0) |

Notice the families: probabilities and logs (Week 1), losses and updates (Week 2), blame travelling backwards (Week 3) and sums over windows (Week 4). Many formulas are the same idea in a new setting; for example, the Week 4 kernel gradient is a sum of "blame times incoming signal", exactly like the Week 3 weight gradient.

### A four-day revision plan

Each day takes about 90 minutes. Do the steps in order; the minutes are a guide.

1. **Day 1: Weeks 1 and 2 (about 90 minutes).** Reread the two review sections (20 minutes). Try the six practice problems with the solutions covered (35 minutes). Redo the formative quiz of Week 1 and of Week 2 (20 minutes). Open your GA Lab 1 and DL Lab 1 notebooks and rerun their Check cells without your old answers (15 minutes).
2. **Day 2: Week 3 (about 90 minutes).** Reread the Week 3 review (20 minutes). Rederive the four backpropagation equations on paper from the 2-2-1 network, with new weights (30 minutes). Do the three practice problems (15 minutes). Redo the Week 3 formative quiz (10 minutes). Rerun the gradient-flow experiments from your DL Lab 2B notebook and explain each bar chart in one sentence (15 minutes).
3. **Day 3: Week 4 (about 90 minutes).** Reread the Week 4 review (20 minutes). Compute one convolution, one pooling, one dL/dK and one dL/dx by hand on a fresh 4 × 4 grid (30 minutes). Do the three practice problems (15 minutes). Redo the Week 4 formative quiz (10 minutes). Rerun your DL Lab 4B Part 1 notebook and check every shape by hand (15 minutes).
4. **Day 4: everything together (about 90 to 120 minutes).** Work through [Mock Exam Prep](/mock-exam) (45 minutes). Write the whole formula sheet from memory, then compare it with the table above (15 minutes). Time yourself on one Part B style task from a lab notebook: 25 minutes, no looking back at your old answers (30 minutes). Then stop, and rest.

A timed run of the real written exam is not possible before the real one, because the exam runs once. The four formative quizzes and Mock Exam Prep are the best substitute: work through them as if it were Part A, without notes, and score yourself.

### Exam technique

- **Read the whole question first.** Note what is given, what is asked, and the units or shape of the answer.
- **Check shapes and units.** A weight gradient must have the shape of its weight matrix; an output size must be a whole number of at least 1; a probability must lie between 0 and 1.
- **In Part B, work in code but sanity-check by hand.** Ask whether the sign and size of your answer make sense before you move on.
- **Watch the clock.** Part A: about 90 seconds per question. Part B: keep to the time boxes. A task left at 70 percent is worth more than a task never started.
- **When you are stuck,** flag the question (Part A) or write down what you know and move on (Part B). Return with fresh eyes. Never leave an answer blank; a reasoned guess costs nothing.
- **Eliminate before you choose.** In Part A, cross out options that break a rule you know (a probability above 1, a negative variance, an output size that is not a whole number).

## Key takeaways

- ==The exam tests doing, not reciting==: you compute small results by hand, explain why something happens, and diagnose a model that misbehaves. Practise those three things.
- ==Know the formula sheet cold.== Part A is closed book, and the same formulas carry Part B: softmax with temperature, −ln p, the neuron and layer, the update rule, the δ equations, the power fⁿ, the output size and the parameter counts.
- ==Backpropagation is the chain rule written as δ equations.== The blame travels backwards through transposed weights and local slopes, and the weight gradient is the blame times the incoming signal.
- ==Gradients live or die by the factor per layer.== A factor below 1 (σ′ is at most 0.25) vanishes, a factor above 1 explodes, and every remedy tries to keep it near 1.
- ==Convolution trades full connections for a shared kernel==, so its parameters do not depend on the picture size; its backward pass sums over positions, and pooling and ReLU only route the gradient.

<!-- LAB-START -->
## Lab: Prelim practical exam (Part B) {rail=Practical}

This is Part B of the Prelim exam: a hands-on test in a notebook, worth 60 of the 100 points, with 120 minutes on the clock. Part A, the written exam, is the timed written exam on this page. In Part B you do not read about Weeks 1 to 4, you use them: you compute with probabilities and neurons, train and diagnose small networks, trace gradients, compute convolutions, and explain what you see. ==The objective is to show that you can turn the ideas of Weeks 1 to 4 into working code and correct numbers, and explain your results in your own words.== This page holds the first half of the exam, Tasks 1 and 2 (22 points); Tasks 3 to 5 follow below it. The whole exam is one notebook, so you work through the tasks in order.

> **Rules of the practical exam**
>
> - Open book for the course only: your lessons and your own lab notes.
> - No AI assistants, no sharing of code or answers, and no internet searches for answers.
> - Everything runs on a CPU, nothing is downloaded, and no cell should need more than about 90 seconds. Random seeds are fixed at 42.
> - Your instructor announces the date, the room and how the exam counts towards the course, and arranges the supervision.
> - A task you cannot finish is not a reason to stop: write what you have, use your time box as a guide and move on to the next task.
> - The self-checks tell you only whether a number is right (a tick or a cross). They never show the expected value or a hint, and your instructor marks your code and your written answers as well.

| Task | Topic | Points | Time box |
|---|---|---|---|
| 1 | Generative AI and language models | 10 | 20 min |
| 2 | Supervised learning, neurons and generalisation | 12 | 25 min |
| 3 | Backpropagation and deep networks | 16 | 30 min |
| 4 | Convolutional networks | 16 | 30 min |
| 5 | Synthesis: short written answers | 6 | 15 min |
| Total | | 60 | 120 min |

The setup below is done before the clock starts and is not part of the 120 minutes.

### Setup

1. Choose where you work. **On the school's DGX Spark:** log in to the portal, start JupyterLab, create a folder `prelim` and a notebook `exam-b.ipynb`. **At home or on a laptop:** use the "Open in Colab" card at the top of this lab, choose **File → Save a copy in Drive**, and work in your copy. The Colab notebook contains the same tasks, the same labels and the same points as this page, plus automatic self-check cells; the "Download .ipynb" button gives you that same file if you prefer to upload it to JupyterLab on the Spark.
2. Do not install packages and do not download anything: PyTorch, NumPy and Matplotlib are already installed, and every data set is generated by the helper code. Do not use `!pip install`, Google Drive code or any `google.colab` code in the Spark notebook.
3. If you work from this page, not from the notebook, write each starter block below in its own cell. The notebook's self-check cells are not on this page, so you check your numbers yourself and your instructor marks them; the marking is the same.
4. Paste the block below into the first cell and run it. It fixes the seed 42, defines the `ANSWERS` dictionary that holds your written answers, defines the exam `check(name, ok)` (which prints a tick or a cross and nothing else), and defines the helpers you know from the labs: the datasets `make_xor`, `make_moons`, `make_circles` and `make_shapes`, `plot_decision`, the deep-network helpers `make_deep`, `linear_layers`, `layer_grad_norms` and `train_model`, and the convolution helpers `conv2d_np`, `count_params` and `train_classifier`.

```python
import os, gc, math, random, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on:", device)

ANSWERS = {}   # your written and final answers; the last cell prints them

def check(name, ok, hint=""):
    """Exam version: prints a tick or a cross and never a hint."""
    print(("✓ " if ok else "✗ ") + name)

def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def make_xor(n=200, noise=0.12, seed=0):
    """Four clusters at the corners of a square; label 1 where exactly one coordinate is positive."""
    r = np.random.RandomState(seed)
    X = r.uniform(-1, 1, size=(n, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(np.float32)
    X = X + r.normal(0, noise, size=X.shape)
    return X.astype(np.float32), y

def make_moons(n=200, noise=0.15, seed=0):
    """Two interleaved half-circles (labels 0 and 1)."""
    r = np.random.RandomState(seed)
    m = n // 2
    t0 = r.uniform(0, math.pi, m); t1 = r.uniform(0, math.pi, n - m)
    a = np.stack([np.cos(t0), np.sin(t0)], 1)
    b = np.stack([1 - np.cos(t1), 0.5 - np.sin(t1)], 1)
    X = np.concatenate([a, b]) + r.normal(0, noise, size=(n, 2))
    y = np.concatenate([np.zeros(m), np.ones(n - m)])
    return X.astype(np.float32), y.astype(np.float32)

def make_circles(n=200, noise=0.08, seed=0):
    """A small disc of label 1 inside a ring of label 0."""
    r = np.random.RandomState(seed)
    m = n // 2
    ang = r.uniform(0, 2 * math.pi, n)
    rad = np.concatenate([r.uniform(0.0, 0.45, m), r.uniform(0.75, 1.1, n - m)])
    X = np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1) + r.normal(0, noise, size=(n, 2))
    y = np.concatenate([np.ones(m), np.zeros(n - m)])
    return X.astype(np.float32), y.astype(np.float32)

def plot_decision(predict, X, y, ax=None, title="", lim=2.0):
    """Colour the plane by predict (N,2 array -> probabilities) and draw the points."""
    ax = ax or plt.gca()
    g = np.linspace(-lim, lim, 120)
    gx, gy = np.meshgrid(g, g)
    P = np.asarray(predict(np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32))).reshape(gx.shape)
    ax.contourf(gx, gy, P, levels=np.linspace(0, 1, 11), cmap="RdYlGn", alpha=0.55)
    ax.contour(gx, gy, P, levels=[0.5], colors="k", linewidths=1.5)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], marker="s", c="tab:blue", edgecolor="k", s=28)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], marker="o", c="tab:orange", edgecolor="k", s=28)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.set_title(title)

def linear_layers(model):
    """All nn.Linear layers of a model, input side first."""
    return [m for m in model.modules() if isinstance(m, nn.Linear)]

def make_deep(depth=6, width=16, act="relu", init="default", n_in=2, n_out=1, norm=False, residual=False):
    """depth hidden Linear layers plus an output layer. act: relu, tanh, sigmoid. init: default, xavier, he, small, large."""
    acts = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}
    class Res(nn.Module):
        def __init__(s):
            super().__init__()
            s.inp = nn.Linear(n_in, width)
            s.blocks = nn.ModuleList([nn.Linear(width, width) for _ in range(depth - 1)])
            s.out = nn.Linear(width, n_out)
            s.f = acts[act]()
        def forward(s, x):
            h = s.f(s.inp(x))
            for b in s.blocks:
                h = h + s.f(b(h))
            return s.out(h)
    if residual:
        m = Res()
    else:
        mods, d = [], n_in
        for _ in range(depth):
            mods += [nn.Linear(d, width)] + ([nn.LayerNorm(width)] if norm else []) + [acts[act]()]
            d = width
        mods.append(nn.Linear(d, n_out))
        m = nn.Sequential(*mods)
    for lin in linear_layers(m):
        if init == "xavier": nn.init.xavier_normal_(lin.weight)
        elif init == "he": nn.init.kaiming_normal_(lin.weight, nonlinearity="relu" if act == "relu" else "tanh")
        elif init == "small": nn.init.normal_(lin.weight, 0, 0.05)
        elif init == "large": nn.init.normal_(lin.weight, 0, 1.0)
        if init != "default": nn.init.zeros_(lin.bias)
    return m

def layer_grad_norms(model, X, y):
    """One forward and backward pass with binary cross-entropy; the gradient norm of each Linear layer's weights, input side first."""
    X = torch.as_tensor(X, dtype=torch.float32); y = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)
    model.zero_grad()
    F.binary_cross_entropy_with_logits(model(X), y).backward()
    return [float(l.weight.grad.norm()) for l in linear_layers(model)]

def train_model(model, X, y, epochs=300, lr=0.05, opt="sgd", clip=None, Xv=None, yv=None, seed=0):
    """Full-batch binary cross-entropy training. Returns a dict: "loss", "grad_first", "grad_last" (per epoch), "acc", and with validation data "val_loss", "val_acc"."""
    torch.manual_seed(seed)
    t = lambda a: torch.as_tensor(a, dtype=torch.float32)
    X, y = t(X), t(y).reshape(-1, 1)
    o = torch.optim.Adam(model.parameters(), lr=lr) if opt == "adam" else torch.optim.SGD(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    hist = {"loss": [], "val_loss": [], "grad_first": [], "grad_last": []}
    for _ in range(epochs):
        o.zero_grad(); loss = lossf(model(X), y); loss.backward()
        ls = linear_layers(model)
        hist["grad_first"].append(float(ls[0].weight.grad.norm())); hist["grad_last"].append(float(ls[-1].weight.grad.norm()))
        if clip: nn.utils.clip_grad_norm_(model.parameters(), clip)
        o.step(); hist["loss"].append(float(loss))
        if Xv is not None:
            with torch.no_grad(): hist["val_loss"].append(float(lossf(model(t(Xv)), t(yv).reshape(-1, 1))))
    with torch.no_grad():
        hist["acc"] = float(((model(X) > 0).float() == y).float().mean())
        if Xv is not None: hist["val_acc"] = float(((model(t(Xv)) > 0).float() == t(yv).reshape(-1, 1)).float().mean())
    return hist

SHAPE_NAMES = ["horizontal bar", "vertical bar", "cross", "square", "diagonal"]

def _draw(kind, size, r, shift):
    P = size
    img = np.zeros((size + 2 * P, size + 2 * P), np.float32)
    c = P + size // 2
    dy, dx = (r.randint(-shift, shift + 1, 2) if shift else (0, 0))
    h = size // 4
    if kind == 0:   img[c + dy, c - h + dx:c + h + 1 + dx] = 1
    elif kind == 1: img[c - h + dy:c + h + 1 + dy, c + dx] = 1
    elif kind == 2:
        img[c + dy, c - h + dx:c + h + 1 + dx] = 1; img[c - h + dy:c + h + 1 + dy, c + dx] = 1
    elif kind == 3:
        y0, y1, x0, x1 = c - h + dy, c + h + dy, c - h + dx, c + h + dx
        img[y0, x0:x1 + 1] = 1; img[y1, x0:x1 + 1] = 1; img[y0:y1 + 1, x0] = 1; img[y0:y1 + 1, x1] = 1
    else:
        for t in range(-h, h + 1):
            img[c + t + dy, c + t + dx] = 1
    return img[P:P + size, P:P + size]

def make_shapes(n=300, size=16, noise=0.15, shift=0, seed=0):
    """n pictures of five shape kinds (labels 0..4, see SHAPE_NAMES). X: float32 (n, 1, size, size); y: int64 (n,)."""
    r = np.random.RandomState(seed)
    y = np.arange(n) % 5
    r.shuffle(y)
    X = np.stack([_draw(int(k), size, r, shift) for k in y])
    X = X + r.normal(0, noise, X.shape).astype(np.float32)
    return X[:, None].astype(np.float32), y.astype(np.int64)

def show_images(X, y=None, n=10, cols=5, titles=None):
    """Show the first n pictures of X (N,1,H,W) or (N,H,W)."""
    X = np.asarray(X); n = min(n, len(X)); rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(1.6 * cols, 1.7 * rows))
    for a in np.ravel(axes): a.axis("off")
    for i, a in enumerate(np.ravel(axes)[:n]):
        a.imshow(X[i].reshape(X[i].shape[-2:]), cmap="gray_r", vmin=0, vmax=1)
        if titles is not None: a.set_title(str(titles[i]), fontsize=8)
        elif y is not None: a.set_title(SHAPE_NAMES[int(y[i])], fontsize=8)
    plt.tight_layout(); plt.show()

def conv2d_np(x, k, stride=1, pad=0):
    """Convolution as deep learning means it (the kernel is not flipped) on one 2-D array, in plain NumPy loops."""
    x = np.pad(np.asarray(x, dtype=float), pad); k = np.asarray(k, dtype=float)
    kh, kw = k.shape
    oh, ow = (x.shape[0] - kh) // stride + 1, (x.shape[1] - kw) // stride + 1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[i, j] = np.sum(x[i * stride:i * stride + kh, j * stride:j * stride + kw] * k)
    return out

def count_params(model):
    """Number of trainable numbers (weights and biases) in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_classifier(model, Xtr, ytr, Xte, yte, epochs=30, lr=0.01, opt="adam", batch=64, seed=0, verbose=False):
    """Mini-batch cross-entropy training. Returns a dict of per-epoch lists: loss, train_acc, test_acc."""
    g = torch.Generator().manual_seed(seed)
    Xtr, ytr, Xte, yte = map(torch.as_tensor, (Xtr, ytr, Xte, yte))
    o = torch.optim.Adam(model.parameters(), lr=lr) if opt == "adam" else torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    hist = {"loss": [], "train_acc": [], "test_acc": []}
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Xtr), generator=g); tot = 0.0
        for b in range(0, len(Xtr), batch):
            idx = perm[b:b + batch]
            o.zero_grad(); loss = F.cross_entropy(model(Xtr[idx]), ytr[idx]); loss.backward(); o.step(); tot += float(loss) * len(idx)
        model.eval()
        with torch.no_grad():
            hist["loss"].append(tot / len(Xtr))
            hist["train_acc"].append(float((model(Xtr).argmax(1) == ytr).float().mean()))
            hist["test_acc"].append(float((model(Xte).argmax(1) == yte).float().mean()))
        if verbose and (ep % 5 == 0 or ep == epochs - 1): print(f"epoch {ep:3d}  loss {hist['loss'][-1]:.3f}  train {hist['train_acc'][-1]:.2f}  test {hist['test_acc'][-1]:.2f}")
    return hist
```

5. Run a smoke test in the next cell. It should print `(100, 2) (100,) 37` and then a tick.

```python
X, y = make_xor(100)
m = make_deep(depth=2, width=4)
print(X.shape, y.shape, count_params(m))
check("setup works", count_params(m) == 37)
```

### What to submit

Submit the notebook **with all outputs visible**: run every cell from the top just before you submit. Add the output of the last cell of the notebook, which prints your `ANSWERS` dictionary, so that your written answers are easy to read. Written answers go in `ANSWERS["B1.1"]`, `ANSWERS["B2.5"]` and so on, or in a markdown cell that begins "Your answer:". Submit the notebook the way your instructor asks. Each sub-question below shows its points; a correct number earns the points, and a wrong number with correct working still earns part of them.

### Task 1. Generative AI and language models (10 points, 20 minutes)

A small language model is choosing the next word of "The weather in Manila today is ...". Its raw scores (logits) for four candidate words are below. All numbers in this task are invented for the exam.

| word | hot | humid | rainy | cold |
|---|---|---|---|---|
| logit | 3.1 | 2.2 | 0.4 | -0.7 |

*Exercise B1.1:* **Softmax with temperature (2 points).** Write `softmax_t(logits, T)`, which divides the logits by T and turns them into probabilities. Store the probability of "hot" at T = 1 in `p_hot_T1` (1 point) and at T = 0.5 in `p_hot_T05` (1 point). Then write in `ANSWERS["B1.1"]` one sentence on what the lower temperature did to the distribution.

```python
words = ["hot", "humid", "rainy", "cold"]
logits = np.array([3.1, 2.2, 0.4, -0.7])

def softmax_t(logits, T=1.0):
    """Probabilities from logits at temperature T."""
    return None   # YOUR CODE HERE

probs = softmax_t(logits, 1.0)
p_hot_T1 = None    # YOUR CODE HERE
p_hot_T05 = None   # YOUR CODE HERE
ANSWERS["B1.1"] = ""   # YOUR ANSWER: one sentence
```

*Exercise B1.2:* **Top-k and top-p (2 points).** Write `filter_top(probs, top_k=None, top_p=None)`: keep only the `top_k` most likely words, or the smallest group of most likely words whose probabilities add up to at least `top_p`, set the others to 0 and renormalise so the result adds up to 1. Use the probabilities at T = 1 (`probs`). Store in `n_kept` how many words are kept with `top_p = 0.95` (1 point), and in `p_hot_topk2` the new probability of "hot" with `top_k = 2` (1 point).

```python
def filter_top(probs, top_k=None, top_p=None):
    """Return the renormalised probabilities after top-k and/or top-p filtering."""
    return None   # YOUR CODE HERE

n_kept = None        # YOUR CODE HERE
p_hot_topk2 = None   # YOUR CODE HERE
```

*Exercise B1.3:* **The chain rule and perplexity (2 points).** The model reads the sentence "We eat rice every day" and gives each token the probability below, given the tokens before it. Compute `log_prob`, the natural logarithm of the probability of the whole sentence (0.5 point), `perplexity`, exp(−mean ln p) (1 point), and `most_surprising`, the word the model found least likely (0.5 point).

```python
sentence = ["We", "eat", "rice", "every", "day"]
p_tokens = [0.12, 0.30, 0.25, 0.08, 0.50]

log_prob = None          # YOUR CODE HERE
perplexity = None        # YOUR CODE HERE
most_surprising = None   # YOUR CODE HERE (one of the words)
```

*Exercise B1.4:* **A bigram model from a tiny corpus (2 points).** The corpus below is the whole training text of a bigram model. A full stop is a token, and a full stop is added at the start so that the first word of the first sentence also has a previous token. Write `bigram_counts(tokens)`, which returns a dictionary from a previous token to a dictionary of next-token counts, and `p_next(counts, prev, nxt)`, the estimated probability of `nxt` after `prev` (count divided by the total count after `prev`). Then store in `p_rice_after_makes` the probability of "rice" after "makes" (1 point), and in `ppl_test` the perplexity of the test sentence "the cook makes eggs ." under the model, counting the five transitions that start from the leading full stop (1 point).

```python
corpus = ("the canteen opens at seven . the cook makes rice and the cook makes rice again . "
          "the students eat rice and the students eat eggs . the cook makes eggs . the canteen closes at four .")
tokens = ["."] + corpus.split()

def bigram_counts(tokens):
    """Return {previous token: {next token: count}}."""
    counts = {}
    # YOUR CODE HERE
    return counts

def p_next(counts, prev, nxt):
    """Estimated probability of nxt after prev."""
    return None   # YOUR CODE HERE

counts = bigram_counts(tokens)
test = [".", "the", "cook", "makes", "eggs", "."]
p_rice_after_makes = p_next(counts, "makes", "rice")
ppl_test = None   # YOUR CODE HERE
```

*Exercise B1.5:* **Fluent is not the same as true (2 points).** A student asks a chatbot (temperature 0.9) for three sources for a school report. It answers with three book titles, with authors and years, in perfect English. The student cannot find any of the books. In `ANSWERS["B1.5"]` write 3 to 5 sentences: (a) explain, using what you know about how a language model writes, why this can happen (1 point); (b) give two things the student should do before using the answer (1 point).

```python
ANSWERS["B1.5"] = ""   # YOUR ANSWER: 3 to 5 sentences
```

### Task 2. Supervised learning, neurons and generalisation (12 points, 25 minutes)

*Exercise B2.1:* **One neuron (2 points).** A neuron has the inputs x = (0.6, −1.5, 2.0), the weights w = (0.7, 0.2, −0.5) and the bias b = 0.3. Compute the weighted sum `z` (1 point) and the output `a_sigmoid` when the activation is the sigmoid (1 point).

```python
x = np.array([0.6, -1.5, 2.0])
w = np.array([0.7, 0.2, -0.5])
b = 0.3

z = None          # YOUR CODE HERE
a_sigmoid = None  # YOUR CODE HERE
```

*Exercise B2.2:* **Counting parameters (2 points).** (a) How many parameters (weights and biases) does a fully connected network with layer sizes 6, 10, 8, 3 have? Store it in `n_params_a` (1 point). (b) A network has layer sizes 4, 5, 2 and **no activation function** between the layers. Store its number of parameters in `n_params_b`, and the number of parameters of the *single* layer that computes exactly the same function in `n_params_single` (1 point for both). In `ANSWERS["B2.2"]` add one sentence on what this says about stacking layers without an activation function. You may build the networks in PyTorch and use `count_params`.

```python
n_params_a = None       # YOUR CODE HERE
n_params_b = None       # YOUR CODE HERE
n_params_single = None  # YOUR CODE HERE
ANSWERS["B2.2"] = ""    # YOUR ANSWER: one sentence
```

*Exercise B2.3:* **Two losses (2 points).** (a) A model predicts ŷ = (2.6, 1.9, 4.5, 1.2) for the true values y = (3.0, 1.5, 4.0, 2.0). Store the mean squared error in `mse` (1 point). (b) A 3-class model produces the logits (1.2, −0.3, 0.5) and the true class is class 2 (counting from 0). Turn the logits into probabilities with softmax and store the cross-entropy loss −ln p of the true class in `ce` (1 point).

```python
y_true = np.array([3.0, 1.5, 4.0, 2.0])
y_pred = np.array([2.6, 1.9, 4.5, 1.2])
class_logits = np.array([1.2, -0.3, 0.5])
true_class = 2

mse = None   # YOUR CODE HERE
ce = None    # YOUR CODE HERE
```

*Exercise B2.4:* **Gradient descent by hand-in-code (2 points).** A model predicts ŷ = w·x with one weight. The data are x = (1, 2, 3) and y = (1.8, 4.5, 5.7), the loss is the mean squared error, and the start is w = 0. For this loss the gradient is dL/dw = (2/n) Σ (w·xᵢ − yᵢ) xᵢ. Take **4** gradient-descent steps with learning rate 0.05. Store the final weight in `w_final` (1 point) and the loss at that weight in `loss_final` (1 point).

```python
xs = np.array([1.0, 2.0, 3.0])
ys = np.array([1.8, 4.5, 5.7])
w_now, lr = 0.0, 0.05

YOUR CODE HERE: 4 steps of w_now = w_now - lr * gradient
w_final = None      # YOUR CODE HERE
loss_final = None   # YOUR CODE HERE
```

*Exercise B2.5:* **Why a hidden layer? An experiment (2 points).** The XOR data below are split into a training set and a validation set. Train (a) a single linear neuron `linear` and (b) the small tanh network `hidden` with `train_model` (300 epochs, Adam, learning rate 0.05, giving it `Xv=Xva, yv=yva`). Store the **validation** accuracies, which `train_model` returns under `"val_acc"`, in `acc_linear` and `acc_hidden` (1 point). Then plot the two decision regions with `plot_decision` if you wish, and in `ANSWERS["B2.5"]` explain in 2 to 3 sentences what the two numbers show and why (1 point).

```python
X, y = make_xor(n=300, noise=0.12, seed=7)
Xtr, ytr, Xva, yva = X[:200], y[:200], X[200:], y[200:]
torch.manual_seed(SEED)
linear = nn.Sequential(nn.Linear(2, 1))
hidden = make_deep(depth=2, width=8, act="tanh", init="xavier")

YOUR CODE HERE: train both models with train_model(...)
acc_linear = None   # validation accuracy of the linear neuron
acc_hidden = None   # validation accuracy of the hidden network
ANSWERS["B2.5"] = ""   # YOUR ANSWER: 2 to 3 sentences
```

*Exercise B2.6:* **Overfitting (2 points).** The moons data below have only 40 training points and a lot of noise. Train `big` (3 hidden layers of 64 ReLU units) and `small` (one hidden layer of 3 tanh units) with `train_model` (800 epochs, Adam, learning rate 0.01, `Xv=Xte, yv=yte`). For each model store the gap `train accuracy − test accuracy` (train accuracy is `"acc"`, test accuracy is `"val_acc"`) in `gap_big` and `gap_small` (1 point). In `ANSWERS["B2.6"]` write 3 to 4 sentences: which model shows the signs of overfitting, how you can tell from the numbers, and one thing you could change to reduce it (1 point).

```python
Xtr, ytr = make_moons(n=40, noise=0.40, seed=3)
Xte, yte = make_moons(n=400, noise=0.40, seed=4)
torch.manual_seed(SEED)
big = make_deep(depth=3, width=64, act="relu", init="he")
small = make_deep(depth=1, width=3, act="tanh", init="xavier")

YOUR CODE HERE: train both models with train_model(...)
gap_big = None     # train accuracy minus test accuracy of `big`
gap_small = None   # train accuracy minus test accuracy of `small`
ANSWERS["B2.6"] = ""   # YOUR ANSWER: 3 to 4 sentences
```

Part B has five tasks. Tasks 3 to 5 follow; each starts with a statement, then starter code with `# YOUR CODE HERE` where your code goes, and a check cell that prints only a tick or a cross. Paste the starter code into your notebook (or use the Colab notebook, which has it ready) and run it from top to bottom. Before Task 3, run this helper cell once. It lets the check cells test your numbers without showing the expected values, so a cross never tells you the answer.

```python
import hashlib

# Checking helpers for Tasks 3 to 5 (do not edit). They compare your numbers with stored fingerprints, so a check never shows the expected value.
def _hv(v, nd):
    return hashlib.sha256(f"B|{round(float(v), nd) + 0.0:.{nd}f}".encode()).hexdigest()[:10]

def b_matches(v, digests, nd=4, shape=None):
    """True if v agrees with the reference to about nd decimals (the reference itself is not stored in this notebook)."""
    if v is None:
        return False
    try:
        arr = np.asarray(v, dtype=float)
        if shape is not None and arr.shape != tuple(shape):
            return False
        vals = np.ravel(arr)
        ks = (-1, 0, 1) if nd > 0 else (0,)
        return len(vals) == len(digests) and all(any(_hv(round(t, nd) + k * 10.0 ** -nd, nd) == dg for k in ks) for t, dg in zip(vals, digests))
    except Exception:
        return False

def b_fingerprint(a):
    a = np.asarray(a, dtype=float)
    return float(np.sum(a * np.cos(np.arange(a.size)).reshape(a.shape)))

def b_word(s, digest):
    return isinstance(s, str) and hashlib.sha256(("B|" + s.strip().lower()).encode()).hexdigest()[:10] == digest

def b_written(s, n=40):
    """True when a written answer has at least n characters (the instructor reads it; this only tells you that something is there)."""
    return isinstance(s, str) and len(s.strip()) >= n
```

### Task 3 — Backpropagation and deep networks (16 points, 30 minutes)

Week 3 covered the forward and backward pass, the chain rule, and why gradients vanish or explode. In B3.1 to B3.3 you follow one small network through a forward pass, a backward pass, and two independent checks. In B3.4 and B3.5 you read the gradients of deep networks and repair one.

**The network.** Two inputs, two hidden units with a sigmoid, one output unit with a sigmoid, and the loss L = ½ (a⁽²⁾ − y)², where a⁽²⁾ is the output. The numbers are in the next cell. Write the arrays with NumPy; use the names given.

```python
x = np.array([0.8, -1.5]); y = 0.0                       # input and target
W1 = np.array([[0.4, -0.7], [0.9, 0.3]]); b1 = np.array([0.1, -0.2])   # hidden layer (one row per hidden unit)
W2 = np.array([[0.6, -0.5]]); b2 = np.array([0.15])                    # output layer
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

*Exercise B3.1 (3 points): the forward pass.* Compute the hidden pre-activations `z1` and activations `a1`, the output pre-activation `z2` and output `a2`, and the loss `loss`. Use the formulas of Week 3 and the arrays above.

```python
z1 = None      # YOUR CODE HERE
a1 = None      # YOUR CODE HERE
z2 = None      # YOUR CODE HERE
a2 = None      # YOUR CODE HERE
loss = None    # YOUR CODE HERE
```

```python
check("B3.1 hidden layer: z1 and a1", b_matches(z1, ['f2fe4d4509', 'c424d86ff5'], shape=(2,)) and b_matches(a1, ['5e6c73265c', '6733ac0198'], shape=(2,)))
check("B3.1 output: z2 and a2", b_matches(z2, ['7be25d5ac6']) and b_matches(a2, ['4bf41226b7']))
check("B3.1 loss", b_matches(loss, ['155e464fb3']))
```

*Exercise B3.2 (4 points): the backward pass.* Using the chain rule and the two backpropagation equations of Week 3, compute by hand-in-code the blame at the output `delta2`, the gradients `dW2` and `db2` of the output layer, the blame at the hidden layer `delta1`, and the gradients `dW1` and `db1` of the hidden layer. Keep the shapes of the arrays they belong to. Do not use autograd here.

```python
delta2 = None   # YOUR CODE HERE
dW2 = None      # YOUR CODE HERE
db2 = None      # YOUR CODE HERE
delta1 = None   # YOUR CODE HERE
dW1 = None      # YOUR CODE HERE
db1 = None      # YOUR CODE HERE
```

```python
check("B3.2 output layer: delta2, dW2, db2", b_matches(delta2, ['d960b11091'], 5, (1,)) and b_matches(dW2, ['d22a7ac659', '9118978038'], 5, (1, 2)) and b_matches(db2, ['d960b11091'], 5, (1,)))
check("B3.2 hidden layer: delta1 and db1", b_matches(delta1, ['d3d65ac505', '42b4497596'], 5, (2,)) and b_matches(db1, ['d3d65ac505', '42b4497596'], 5, (2,)))
check("B3.2 hidden layer: dW1", b_matches(dW1, ['0617ab7679', 'd5dc9a64d9', 'fa0707d07e', '3893312077'], 5, (2, 2)))
```

*Exercise B3.3 (2 points): verify your gradients two ways.* (a) Build the same network in PyTorch with `torch.float64` tensors that require gradients, call `loss.backward()`, and store the four gradients as NumPy arrays in the dictionary `auto_grads` (keys `"W1"`, `"b1"`, `"W2"`, `"b2"`). Store in `max_diff` the largest absolute difference between your hand-in-code gradients (`dW1`, `db1`, `dW2`, `db2`) and these. (b) The function `loss_at` below returns the loss for any hidden weight matrix. Use it with `eps = 1e-5` to estimate the derivative of the loss with respect to `W1[0, 1]` by a central difference, and store it in `fd`.

```python
def loss_at(W1_):
    """Loss of the network above when the hidden weights are W1_ (everything else unchanged)."""
    a1_ = sigmoid(W1_ @ x + b1)
    return float(np.sum(0.5 * (sigmoid(W2 @ a1_ + b2) - y) ** 2))

eps = 1e-5
auto_grads = None   # YOUR CODE HERE: {"W1": ..., "b1": ..., "W2": ..., "b2": ...}
max_diff = None     # YOUR CODE HERE
fd = None           # YOUR CODE HERE
```

```python
check("B3.3 (a) your gradients agree with autograd to 1e-6", max_diff is not None and dW1 is not None and max_diff < 1e-6)
check("B3.3 (b) the finite difference agrees with your dW1[0, 1]", fd is not None and dW1 is not None and abs(fd - dW1[0, 1]) < 1e-6)
```

*Exercise B3.4 (3 points): read the gradients of three deep networks.* The function `build` below returns three networks, each with 8 hidden layers of 16 units, built the same way except for what the Week 3 lesson calls the activation and the initialisation. They are trained on the same data. For each network compute the gradient norm of every layer (use `layer_grad_norms` and the data `Xm, ym`) and store the lists in the dictionary `norms`. Then decide whether each network is `"vanishing"`, `"exploding"` or `"healthy"` and put the words in `ANSWERS["B3.4"]` (2 points). In `ANSWERS["B3.4_why"]` write two or three sentences that quote the numbers you used for the decision (1 point).

```python
Xm, ym = make_moons(200, noise=0.15, seed=3)
Xv, yv = make_moons(200, noise=0.15, seed=4)       # validation data, used in B3.5

def build(name):
    """Network N1, N2 or N3 (always the same weights, because the seed is set inside)."""
    torch.manual_seed(7)
    if name == "N1": return make_deep(8, 16, "relu", "large")
    if name == "N2": return make_deep(8, 16, "sigmoid", "default")
    if name == "N3": return make_deep(8, 16, "relu", "he")

norms = {}
for name in ("N1", "N2", "N3"):
    pass   # YOUR CODE HERE: norms[name] = layer_grad_norms(...), and print it
ANSWERS["B3.4"] = {"N1": "", "N2": "", "N3": ""}   # YOUR ANSWER: "vanishing", "exploding" or "healthy" for each network
ANSWERS["B3.4_why"] = ""                           # YOUR ANSWER: two or three sentences with the numbers
```

```python
a34 = ANSWERS.get("B3.4", {})
check("B3.4 network N1", b_word(a34.get("N1"), "3917e9e312"))
check("B3.4 network N2", b_word(a34.get("N2"), "2976fbf6b5"))
check("B3.4 network N3", b_word(a34.get("N3"), "4038ac4e59"))
check("B3.4 evidence written", b_written(ANSWERS.get("B3.4_why"), 60))
```

*Exercise B3.5 (4 points): repair the vanishing network.* One of the three networks cannot learn. (a) For the original unhealthy network store in `ratio_before` the ratio (gradient norm of the first layer divided by the norm of the last layer, before any training), and in `acc_before` its validation accuracy after training it with `train_model(model, Xm, ym, epochs=300, lr=0.05, Xv=Xv, yv=yv)` (the result dictionary has the key `"val_acc"`). (b) Build a repaired network in the variable `fixed` with one change from Week 3 (set `torch.manual_seed(7)` first, and keep 8 hidden layers of 16 units), and store in `ratio_after` and `acc_after` the same two numbers for it. A good repair raises the ratio by a factor of at least 1000 and reaches a validation accuracy of at least 0.9 (2 points for the repaired network and its numbers, 1 point for the before numbers). (c) In `ANSWERS["B3.5"]` say in two or three sentences which change you made and why it works, quoting your before and after numbers (1 point).

```python
ratio_before = None   # YOUR CODE HERE
acc_before = None     # YOUR CODE HERE
fixed = None          # YOUR CODE HERE: the repaired network (torch.manual_seed(7) first)
ratio_after = None    # YOUR CODE HERE
acc_after = None      # YOUR CODE HERE
ANSWERS["B3.5"] = ""  # YOUR ANSWER: which change, why it works, with your numbers
```

```python
check("B3.5 (a) the 'before' numbers", ratio_before is not None and acc_before is not None and b_matches(np.log10(ratio_before), ['a2fc78e558'], 1) and b_matches(acc_before, ['70848d9697'], 2))
check("B3.5 (b) the repaired network is a model and its ratio improved 1000 times", isinstance(fixed, nn.Module) and ratio_after is not None and ratio_before is not None and ratio_after >= 1000 * ratio_before)
check("B3.5 (b) the repaired network reaches validation accuracy 0.9", acc_after is not None and acc_after >= 0.9)
check("B3.5 (c) explanation written", b_written(ANSWERS.get("B3.5"), 60))
```

### Task 4 — Convolutional networks (16 points, 30 minutes)

Week 4 covered convolution, output sizes, parameters, the backward pass of a convolution, and why a CNN suits pictures. In B4.1 and B4.2 you compute by hand-in-code; in B4.3 and B4.4 you build, train and interpret a small CNN against a dense network.

*Exercise B4.1 (4 points): a convolution and its sizes.* (a) Convolve the picture `x1` with the kernel `K1` using stride 2 and padding 1 (deep learning convolution, no flipping) and store the result in `out1` (1 point). Store in `n_out` the side length of the output that the output-size formula gives, computed from the formula without looking at the shape of `out1` (1 point). (b) A convolution layer has 3 input channels and 12 filters of size 5 × 5 (with biases). Store its number of parameters in `conv_params` (1 point). Give it a 3 × 40 × 40 picture with stride 2 and padding 2: store the output side length in `out_side`, and in `dense_params` the number of parameters of a fully connected layer (with biases) that maps the flattened 3 × 40 × 40 input to the flattened output of that convolution, 12 × `out_side` × `out_side` (1 point).

```python
x1 = np.array([[1, 0, 3, 2, 4, 1, 0],
               [2, 5, 1, 0, 3, 3, 1],
               [0, 4, 2, 1, 5, 0, 2],
               [3, 1, 0, 4, 2, 2, 5],
               [1, 2, 5, 3, 0, 4, 1],
               [4, 0, 1, 2, 3, 5, 0],
               [2, 3, 4, 0, 1, 1, 3]], dtype=float)
K1 = np.array([[2, -1, 0], [1, 0, -1], [0, 1, -2]], dtype=float)
out1 = None          # YOUR CODE HERE
n_out = None         # YOUR CODE HERE (from the formula)
conv_params = None   # YOUR CODE HERE
out_side = None      # YOUR CODE HERE
dense_params = None  # YOUR CODE HERE
```

```python
check("B4.1 (a) the convolution output", out1 is not None and np.shape(out1) == (4, 4) and b_matches(b_fingerprint(out1), ['45887cb2c1'], 3))
check("B4.1 (a) the output side from the formula", b_matches(n_out, ['e54b140538'], 0))
check("B4.1 (b) parameters of the convolution layer", b_matches(conv_params, ['90ad586faf'], 0))
check("B4.1 (b) output side and dense parameters", b_matches(out_side, ['8c32de89fa'], 0) and b_matches(dense_params, ['9593f06880'], 0))
```

*Exercise B4.2 (5 points): the backward pass of a convolution.* The loss is L = the sum of `G2` times `conv2d_np(x2, K2)` (stride 1, no padding), so `G2` is the upstream gradient dL/d(output). Write `conv_backward_kernel(x, G, kh, kw)`, which returns dL/dK (2 points), and `conv_backward_input(K, G, in_shape)`, which returns dL/dx (2 points), with explicit loops over the output positions and the kernel cells. Then verify them (1 point): store in `err_auto` the largest absolute difference between your two gradients and PyTorch's (use `F.conv2d` on `float64` tensors with `requires_grad`, remember that PyTorch also wants shapes (1, 1, H, W)), and in `fd_K` a central-difference estimate (`eps = 1e-6`) of dL/dK[1, 2] computed from `conv2d_np`.

```python
x2 = np.array([[2, 0, 1, 3, 1, 0],
               [1, 4, 2, 0, 2, 1],
               [0, 3, 1, 2, 5, 3],
               [2, 1, 0, 1, 3, 0],
               [4, 2, 3, 0, 1, 2],
               [1, 0, 2, 4, 0, 1]], dtype=float)
K2 = np.array([[1, 2, 0], [-1, 1, 3], [0, -2, 1]], dtype=float)
G2 = np.array([[1, 0, -2, 1],
               [2, 1, 0, -1],
               [0, -1, 3, 1],
               [1, 2, -1, 0]], dtype=float)

def conv_backward_kernel(x, G, kh, kw):
    """dL/dK for stride 1 and no padding. Returns an array of shape (kh, kw)."""
    dK = np.zeros((kh, kw))
    # YOUR CODE HERE
    return dK

def conv_backward_input(K, G, in_shape):
    """dL/dx for stride 1 and no padding. Returns an array of shape in_shape."""
    dx = np.zeros(in_shape)
    # YOUR CODE HERE
    return dx

dK2 = conv_backward_kernel(x2, G2, 3, 3)
dx2 = conv_backward_input(K2, G2, x2.shape)
```

Then verify the two functions (the second block):

```python
err_auto = None   # YOUR CODE HERE
fd_K = None       # YOUR CODE HERE
```

```python
check("B4.2 dL/dK", np.shape(dK2) == (3, 3) and b_matches(b_fingerprint(dK2), ['0e1763d908'], 3))
check("B4.2 dL/dx", np.shape(dx2) == (6, 6) and b_matches(b_fingerprint(dx2), ['d41ea2c541'], 3))
check("B4.2 verification: your gradients agree with PyTorch to 1e-6 and with the finite difference", err_auto is not None and fd_K is not None and err_auto < 1e-6 and abs(fd_K - dK2[1, 2]) < 1e-5)
```

*Exercise B4.3 (5 points): a CNN against a dense network, under shift.* The data below are shapes of size 20 × 20: `Xtr, ytr` (500 pictures, centred) for training, `X0, y0` (centred) and `X4, y4` (every shape moved up to 4 pixels away from the centre) for testing. (a) Build a CNN `cnn` (`nn.Sequential`, at least two convolution layers, at most 3000 parameters, 5 outputs) and a dense network `mlp` (`Flatten`, a hidden layer with ReLU, an output layer) whose parameter count is within 15 % of the CNN's (2 points). Call `torch.manual_seed(0)` before building each. (b) Train both with `train_classifier(model, Xtr, ytr, X0, y0, epochs=25, lr=0.01)` and fill the dictionary `RES` with the test accuracies `"cnn_centre"`, `"cnn_shift"`, `"mlp_centre"`, `"mlp_shift"` measured with `accuracy` on `X0` and `X4` (2 points). (c) Store the drops `drop_cnn` and `drop_mlp` (centre accuracy minus shifted accuracy) (1 point).

First build the data and the two networks:

```python
Xtr, ytr = make_shapes(500, 20, noise=0.2, shift=0, seed=11)
X0, y0 = make_shapes(300, 20, noise=0.2, shift=0, seed=12)
X4, y4 = make_shapes(300, 20, noise=0.2, shift=4, seed=13)

def accuracy(model, X, y):
    """Fraction of pictures the model classifies correctly."""
    model.eval()
    with torch.no_grad():
        return float((model(torch.as_tensor(X)).argmax(1) == torch.as_tensor(y)).float().mean())

torch.manual_seed(0)
cnn = None   # YOUR CODE HERE
torch.manual_seed(0)
mlp = None   # YOUR CODE HERE
```

Then train, measure and compute the drops:

```python
RES = {}
# YOUR CODE HERE: train both networks and fill RES
drop_cnn = None   # YOUR CODE HERE
drop_mlp = None   # YOUR CODE HERE
```

```python
def _built_ok():
    try:
        convs = [m for m in cnn.modules() if isinstance(m, nn.Conv2d)]
        with torch.no_grad():
            ok_shape = tuple(cnn(torch.zeros(2, 1, 20, 20)).shape) == (2, 5) and tuple(mlp(torch.zeros(2, 1, 20, 20)).shape) == (2, 5)
        return len(convs) >= 2 and count_params(cnn) <= 3000 and ok_shape and abs(count_params(mlp) - count_params(cnn)) <= 0.15 * count_params(cnn)
    except Exception:
        return False
check("B4.3 (a) both networks have the right shape and size", _built_ok())
def _trained_ok():
    try:
        return all(k in RES for k in ("cnn_centre", "cnn_shift", "mlp_centre", "mlp_shift")) and \
            all(abs(RES[k] - accuracy(m, X, y)) < 1e-9 for k, m, X, y in [("cnn_centre", cnn, X0, y0), ("cnn_shift", cnn, X4, y4), ("mlp_centre", mlp, X0, y0), ("mlp_shift", mlp, X4, y4)])
    except Exception:
        return False
check("B4.3 (b) RES matches the trained networks", _trained_ok())
check("B4.3 (b) both networks learn the centred shapes (at least 0.9)", _trained_ok() and RES["cnn_centre"] >= 0.9 and RES["mlp_centre"] >= 0.9)
check("B4.3 (c) the drops", _trained_ok() and drop_cnn is not None and drop_mlp is not None and abs(drop_cnn - (RES["cnn_centre"] - RES["cnn_shift"])) < 1e-9 and abs(drop_mlp - (RES["mlp_centre"] - RES["mlp_shift"])) < 1e-9)
```

*Exercise B4.4 (2 points): interpret.* In `ANSWERS["B4.4"]` write three or four sentences. Say which network is more robust to the shift and by how many percentage points the two drops differ (use your numbers), explain the difference with the ideas of Week 4 (what the weights of each kind of layer are attached to), and name one change that would make the weaker network cope better.

```python
ANSWERS["B4.4"] = ""   # YOUR ANSWER
```

```python
check("B4.4 answer written", b_written(ANSWERS.get("B4.4"), 80))
```

### Task 5 — Synthesis (6 points, 15 minutes)

Three short scenarios that join several weeks. Answer each in `ANSWERS` in two to five sentences (2 points each). Use numbers where the question asks for them; you may use code cells for the arithmetic.

*Exercise B5.1 (2 points; Weeks 2 and 3).* A classmate trains a network with 10 hidden layers, all sigmoid, default initialisation, on 5,000 rows with a binary-cross-entropy loss. The training loss starts at 0.693 and is still 0.693 after 200 epochs; the validation loss is the same. She plans to collect more data and to lower the learning rate. (a) What does the value 0.693 tell you about the predictions? (b) Explain why her two plans will not help, quoting the Week 3 number for how much a sigmoid layer can pass back and what it gives over 10 layers. (c) Name two changes that would help.

*Exercise B5.2 (2 points; Weeks 2 and 4).* A team builds a leaf-disease app. It has 300 training photos and 100 test photos, 28 × 28 pixels in grey, three classes. A dense network 784 → 128 → 3 reaches 100 % on the training photos and 62 % on the test photos, and users complain that it fails when the leaf is not in the middle. (a) Count the parameters of that dense network, and of one convolution layer with 8 filters of size 5 × 5 on one channel. (b) Name the problem in Week 2 words, and say why a convolutional network would suit this task better, using Week 4 ideas. (c) Give one fix from the data side and one from the model side.

*Exercise B5.3 (2 points; Weeks 1 and 2).* A language model has to choose the next word. For three candidate words its scores (logits) are 3.0, 1.0 and 0.0. (a) Compute the probability of the top word at temperature 1 and at temperature 0.5. (b) During training the correct next word was the second candidate. Compute the cross-entropy loss −ln p at temperature 1, and say in which direction training moves the score of that word. (c) A school wants a chatbot for exam-revision answers that are accurate and repeatable. Which temperature setting would you choose, and what would go wrong with a very high temperature?

Write your three answers here (the check only tells you that something is there; your instructor reads and marks the text):

```python
ANSWERS["B5.1"] = ""   # YOUR ANSWER
ANSWERS["B5.2"] = ""   # YOUR ANSWER
ANSWERS["B5.3"] = ""   # YOUR ANSWER
```

```python
for _k in ("B5.1", "B5.2", "B5.3"):
    check(_k + " answer written", b_written(ANSWERS.get(_k), 80))
```

When you have finished all five tasks, print your answers once more so you can read them through:

```python
print("Your ANSWERS (copy them into your submission if your instructor asks):")
for k in sorted(ANSWERS):
    print(f"\n{k}: {ANSWERS[k]}")
```

### What to submit

1. The notebook with all five tasks, run from top to bottom (Kernel → Restart Kernel and Run All Cells, or Runtime → Restart session and run all in Colab), with your code in the starter cells, your written answers in the `ANSWERS` dictionary (or in markdown cells under the task) and the outputs visible.
2. Your name and student number in the first cell of the notebook, if your instructor asks for them.
3. Nothing else: do not submit separate files, screenshots or the written exam (Part A is submitted in the app).
4. Submit it the way your instructor asks. This page does not name a portal, a file name or a deadline; your instructor announces them.
5. Remember that the ✓ and ✗ lines are a guide only. Your instructor marks the notebook with the marking scheme: the correct result, the method you used and your interpretation, with partial credit where part of the work is right.

### Academic integrity and finishing the exam

- Part B is open lessons and your own lab notes only. Do not use an AI assistant, do not ask or copy from another person, and do not search the internet for answers. Your instructor can ask you to explain your code.
- Work alone and in your own notebook. Do not share the notebook, your answers or the task text with other students during or after the exam.
- Keep to the time boxes (20, 25, 30, 30 and 15 minutes). If a task is not finished when its time is up, hand in what you have: partial work earns partial credit.
- If something fails to run, read the error message once, fix what you can, and move on; leave a short comment saying what you tried.
- Before you finish, restart the kernel, run all cells once, check that nothing raises an error, and save a copy (in Colab: File → Save a copy in Drive).
- Good luck. You have done four weeks of hard work, and this exam only asks you to show it.
<!-- LAB-END -->

## Check your understanding: the written exam (Part A) {rail=Written exam}

{lede} Part A is ==the written half of the Prelim==: 40 multiple-choice questions on Weeks 1 to 4, 60 minutes, closed book, worth 40 of the 100 points.

{sub} Exam · Part A · 40 points · 60 minutes · Your best score is saved to your account

Press Start to begin the timer. The exam is **timed** and submits itself at zero. There is **no feedback until you submit**; until then you can move back and forth, change answers and **flag** questions to return to. After you submit, you see your score, a **breakdown by topic** (one row for each of the four weeks) and the explanation for every question you missed, so that your revision targets are clear. Your best score is saved to your account. Each question is worth 1 point, so your Part A score out of 40 is added to Part B for the total out of 100. The written exam runs in your browser and cannot stop a determined cheater, so it is meant to be taken under the supervision your instructor arranges.

```quiz
{
  "id": "week-05-prelim",
  "exam": true,
  "minutes": 60,
  "passMark": 70,
  "title": "Prelim written exam (Part A)",
  "rules": [
    "Part A is worth 40 of the 100 points. Each question has one correct answer and is worth 1 point.",
    "A calculator and scratch paper are allowed. No notes, lessons, lab notebooks or AI assistants."
  ],
  "questions": [
    {
      "topic": "Week 1: generative AI",
      "q": "Which of these tasks is done by a generative model rather than a discriminative one?",
      "options": [
        "Labelling a photo as cat or dog",
        "Flagging a bank transfer as fraud or not fraud",
        "Writing a new paragraph in the style of a news report from a one-line prompt",
        "Sorting customer messages into complaint, question or praise"
      ],
      "answer": 2,
      "why": "A generative model learns what data looks like and produces new data (it models P(x) or P(x | prompt)). The other three tasks only assign a label to data that already exists, which is the discriminative question P(y | x)."
    },
    {
      "topic": "Week 1: generative AI",
      "q": "A language model reads the three-token phrase “kain na tayo”. It gives P(kain) = 0.02, P(na | kain) = 0.5 and P(tayo | kain na) = 0.4. What probability does the chain rule give for the whole phrase?",
      "options": [
        "0.01",
        "0.307",
        "0.004",
        "0.92"
      ],
      "answer": 2,
      "why": "The chain rule multiplies the step probabilities: 0.02 × 0.5 × 0.4 = 0.004. Adding gives 0.92, averaging gives 0.307, and 0.01 forgets the last factor."
    },
    {
      "topic": "Week 1: generative AI",
      "q": "A model gives the three tokens of a short sentence the probabilities 0.5, 0.125 and 0.03125 (each given the tokens before it). What is the perplexity of the sentence, using perplexity = P^(−1/n)?",
      "options": [
        "512",
        "0.125",
        "4.57",
        "8"
      ],
      "answer": 3,
      "why": "P = 0.5 × 0.125 × 0.03125 = 0.001953125 (this is 2 to the power −9), so perplexity = P^(−1/3) = 2^3 = 8. The value 512 is 1/P without the root, 0.125 is the geometric mean of the probabilities, and 4.57 comes from the plain average."
    },
    {
      "topic": "Week 1: generative AI",
      "q": "Three candidate next words have scores 3, 1 and 0. The model applies softmax to the scores divided by the temperature. At temperature T = 2, what probability does the top word get (nearest value)?",
      "options": [
        "0.84",
        "0.75",
        "0.63",
        "0.33"
      ],
      "answer": 2,
      "why": "Scores / 2 = 1.5, 0.5, 0; e to these is 4.482, 1.649, 1.000, total 7.130, so the top word gets 4.482 / 7.130 = 0.63. At T = 1 it would be 0.84, 0.75 comes from dividing 3 by the plain sum 4 without exponentials, and 0.33 is the uniform answer."
    },
    {
      "topic": "Week 1: generative AI",
      "q": "A story app uses temperature 0.2 and top-k = 1, and it writes almost the same plain sentence every time. The designer wants more varied, but still sensible, stories. Which change is best?",
      "options": [
        "Raise the temperature to about 0.9 and use top-p = 0.9 (or a larger top-k such as 40)",
        "Lower the temperature to 0.01 and keep top-k = 1",
        "Set top-k = 1 and top-p = 0.1 so only the best word survives",
        "Keep the settings and instead train the model on the same text for longer"
      ],
      "answer": 0,
      "why": "Top-k = 1 always picks the single most likely word, and a low temperature sharpens the probabilities further, so the output barely changes. A higher temperature with top-p or a larger top-k lets several sensible words compete while still cutting absurd ones."
    },
    {
      "topic": "Week 1: generative AI",
      "q": "In forward diffusion, a pixel is noised with x_t = √ᾱ_t · x₀ + √(1−ᾱ_t) · ε. At some step ᾱ_t = 0.64. For a pixel x₀ = 0.5 and noise ε = −1, what is x_t?",
      "options": [
        "0.2",
        "−0.2",
        "−0.04",
        "1.0"
      ],
      "answer": 1,
      "why": "√0.64 = 0.8 and √(1−0.64) = √0.36 = 0.6, so x_t = 0.8 × 0.5 + 0.6 × (−1) = 0.4 − 0.6 = −0.2. Using 0.64 and 0.36 without the roots gives −0.04, dropping the minus sign of ε gives 0.4 + 0.6 = 1.0, and 0.2 is the answer with its sign lost."
    },
    {
      "topic": "Week 1: generative AI",
      "q": "Which statement about the history of generative AI is correct?",
      "options": [
        "ELIZA (1966) learned its replies from a huge collection of conversations.",
        "The Transformer (2017) is built on attention and made it practical to train very large language models.",
        "AlexNet (2012) was a language model that won a translation contest.",
        "GPT-2 (2019) was the first system to use backpropagation."
      ],
      "answer": 1,
      "why": "The Transformer, introduced in 2017, is built on attention and opened the way to very large language models. ELIZA followed hand-written rules and did not learn, AlexNet was an image-recognition network, and backpropagation was popularised in 1986, long before GPT-2."
    },
    {
      "topic": "Week 1: generative AI",
      "q": "A student hands in an essay with a reference to a research paper, copied from an AI chatbot. The paper does not exist. Which explanation fits best?",
      "options": [
        "The chatbot was lying on purpose because its temperature was too low.",
        "The chatbot searched a library and the paper was deleted afterwards.",
        "The model writes likely next tokens, not verified facts, so a reference that looks plausible can be produced even when it is false.",
        "Models never write references unless the paper exists in their training data."
      ],
      "answer": 2,
      "why": "A language model is trained to continue text with probable tokens, so it can write a fluent, official-looking citation without any check against reality. Fluent does not mean true, and every reference must be checked in a trusted source."
    },
    {
      "topic": "Week 2: supervised learning",
      "q": "A teacher builds a model to predict a student's final grade (0 to 100) from hours studied per week and attendance rate. In this supervised learning problem, what is the label?",
      "options": [
        "The student's real final grade, recorded for each training example",
        "The hours studied per week",
        "The weights that the model learns",
        "The grade that the model predicts"
      ],
      "answer": 0,
      "why": "The label is the correct answer attached to each training example, here the real final grade. Hours studied and attendance are features, the weights are parameters, and the predicted grade is the model's output, which the loss compares with the label."
    },
    {
      "topic": "Week 2: supervised learning",
      "q": "A canteen model predicts the number of rice meals sold in four time slots. The real numbers are 3, 5, 2 and 8 and the predictions are 2, 6, 2 and 5. What is the mean squared error?",
      "options": [
        "11",
        "1.25",
        "2.75",
        "1.66"
      ],
      "answer": 2,
      "why": "The squared errors are (3−2)² = 1, (5−6)² = 1, 0 and (8−5)² = 9, which add to 11; divide by 4 slots to get 2.75. The value 11 forgets to average, 1.25 is the mean absolute error, and 1.66 is the square root of 2.75."
    },
    {
      "topic": "Week 2: supervised learning",
      "q": "A mango classifier has three classes: green, ripe and overripe. For one mango that is really ripe, it outputs the probabilities 0.5, 0.2 and 0.3 for green, ripe and overripe. What is the cross-entropy loss −ln p for this example (two decimals)?",
      "options": [
        "0.80",
        "1.61",
        "0.70",
        "1.20"
      ],
      "answer": 1,
      "why": "Only the probability given to the true class counts: −ln(0.2) = 1.609, about 1.61. The value 0.80 is 1 − 0.2, 0.70 uses log base 10, and 1.20 is −ln(0.3), which is the loss for the wrong class."
    },
    {
      "topic": "Week 2: supervised learning",
      "q": "A neuron has inputs x = (2, −1, 3), weights w = (0.5, −1, 0.4), bias b = −1.5 and a sigmoid activation. What is its output a (nearest value)?",
      "options": [
        "1.70",
        "0.96",
        "0.99",
        "0.85"
      ],
      "answer": 3,
      "why": "z = 0.5×2 + (−1)×(−1) + 0.4×3 − 1.5 = 1 + 1 + 1.2 − 1.5 = 1.7, and a = σ(1.7) = 1/(1 + e^(−1.7)) = 0.85. The value 1.70 skips the activation, 0.96 forgets the bias (σ(3.2)), and 0.99 adds the bias instead of subtracting it (σ(4.7))."
    },
    {
      "topic": "Week 2: supervised learning",
      "q": "A network has 12 inputs, then two hidden layers of 8 neurons each, then an output layer of 3 neurons, all fully connected with a bias for every neuron. How many parameters does it have?",
      "options": [
        "203",
        "184",
        "31",
        "215"
      ],
      "answer": 0,
      "why": "Layer by layer: 12×8 + 8 = 104, 8×8 + 8 = 72 and 8×3 + 3 = 27, and 104 + 72 + 27 = 203. The number 184 forgets the biases, 31 only adds the layer sizes, and 215 wrongly gives the 12 inputs a bias too (203 + 12)."
    },
    {
      "topic": "Week 2: supervised learning",
      "q": "A two-layer network has no activation functions and no biases. Its first layer has weights W₁ = [[2, 0], [1, 1]] and its second layer has weights W₂ = [1, 3]. Which single set of weights gives exactly the same output for every input?",
      "options": [
        "(3, 5)",
        "(2, 4)",
        "(1, 3)",
        "(5, 3)"
      ],
      "answer": 3,
      "why": "Without a non-linear activation, stacked layers collapse into one: W₂W₁ = (1×2 + 3×1, 1×0 + 3×1) = (5, 3). For x = (1, 2) both give 11. This is why hidden layers need a non-linearity to solve a problem such as XOR."
    },
    {
      "topic": "Week 2: supervised learning",
      "q": "In gradient descent the update is w ← w − η × gradient. A weight is w = 0.8, the gradient of the loss is −0.35 and the learning rate is η = 0.2. What is the new weight?",
      "options": [
        "0.73",
        "1.15",
        "0.45",
        "0.87"
      ],
      "answer": 3,
      "why": "0.2 × (−0.35) = −0.07, and w ← 0.8 − (−0.07) = 0.87. A negative gradient means the loss falls when w rises, so w goes up. The value 0.73 has the wrong sign, and 1.15 and 0.45 leave out the learning rate."
    },
    {
      "topic": "Week 2: supervised learning",
      "q": "While training, a student prints the loss at the end of each epoch: 2.31, 5.80, 41.7, 320000, nan. The data and the code are correct. What is the most likely cause, and what should she try first?",
      "options": [
        "The learning rate is too large, so the steps overshoot the minimum; reduce it, for example by a factor of 10.",
        "The learning rate is too small; increase it by a factor of 10.",
        "The network is underfitting; add more hidden layers.",
        "The network is overfitting; collect more validation data."
      ],
      "answer": 0,
      "why": "A loss that grows from epoch to epoch and ends in nan is the classic sign of steps that jump over the minimum and land higher up the hill, so the learning rate is too large. Too small a rate would give a loss that falls very slowly, not one that explodes."
    },
    {
      "topic": "Week 2: supervised learning",
      "q": "A network reaches 99% accuracy on the training set and only 72% on the validation set. Which next step is most sensible?",
      "options": [
        "Add several more hidden layers so the network can fit even better.",
        "Train for many more epochs until the validation accuracy rises by itself.",
        "Use the test set to try different settings, because it is the most reliable score.",
        "Get more training data, use a smaller network or stop training earlier, then check the validation accuracy again."
      ],
      "answer": 3,
      "why": "A big gap between training and validation accuracy means overfitting: the network memorises the training set. More data, a smaller network or earlier stopping reduce it. More layers or epochs make memorising easier, and tuning on the test set uses up the one honest estimate."
    },
    {
      "topic": "Week 2: supervised learning",
      "q": "A network must classify a photo as exactly one of 4 types of public transport and report a probability for each type, with the four probabilities adding up to 1. Which output layer fits?",
      "options": [
        "Four neurons with a softmax activation",
        "Four neurons that each use a sigmoid activation",
        "One neuron with a ReLU activation that outputs the class number",
        "Four neurons with a ReLU activation, taking the largest"
      ],
      "answer": 0,
      "why": "Softmax turns the four scores into probabilities that are positive and add to 1. Four separate sigmoids give four probabilities that need not add up to 1, and ReLU outputs are not probabilities at all."
    },
    {
      "topic": "Week 3: deep learning and backprop",
      "q": "The paper-folding example shows k tent layers (2 units each) giving a zig-zag with 2ᵏ straight pieces. What does it show about depth?",
      "options": [
        "Deeper networks always give a lower error than shallow ones on every task.",
        "A deep network needs 2ᵏ units to make 2ᵏ pieces, the same as a shallow one.",
        "A deep network can reuse its own folds: 2k units give 2ᵏ pieces, while a one-layer network needs at least 2ᵏ − 1 units for the same shape.",
        "A shallow network can never draw a zig-zag, however many units it has."
      ],
      "answer": 2,
      "why": "Each tent layer folds all the pieces made so far, so the pieces double per layer while the units grow only by 2. A shallow network can make the same shape but needs far more units. It is a worked example, not a law, and depth also makes training harder."
    },
    {
      "topic": "Week 3: deep learning and backprop",
      "q": "A tiny network has input x = (2, 1). Hidden layer: z = W₁x + b₁ with W₁ = [[1, 0.5], [−1, 2]], b₁ = (−3, 1), then ReLU. Output (no activation): y = W₂a + b₂ with W₂ = (2, −1) and b₂ = 0.5. What is y?",
      "options": [
        "−0.5",
        "−1.5",
        "−1.0",
        "5.5"
      ],
      "answer": 0,
      "why": "z₁ = (1×2 + 0.5×1 − 3, −1×2 + 2×1 + 1) = (−0.5, 1); ReLU gives a = (0, 1); y = 2×0 + (−1)×1 + 0.5 = −0.5. Skipping the ReLU gives −1.5, forgetting b₂ gives −1.0, and forgetting b₁ gives 5.5."
    },
    {
      "topic": "Week 3: deep learning and backprop",
      "q": "A single neuron computes z = w·x + b, a = ReLU(z) and the loss L = (a − y)². With x = 3, w = 0.5, b = 0.5 and y = 1, what is dL/dw?",
      "options": [
        "3",
        "2",
        "12",
        "6"
      ],
      "answer": 3,
      "why": "Forward: z = 0.5×3 + 0.5 = 2, a = 2. Chain rule: dL/dw = dL/da × da/dz × dz/dw = 2(a − y) × 1 × x = 2×1×1×3 = 6 (ReLU has slope 1 because z > 0). Dropping the factor 2 gives 3, dropping x gives 2, and using a instead of a − y gives 12."
    },
    {
      "topic": "Week 3: deep learning and backprop",
      "q": "A sigmoid output unit gives a = 0.8 for the target y = 1, with the loss L = ½(a − y)². Using δ = (a − y) × a(1 − a), what is the output blame δ?",
      "options": [
        "−0.032",
        "−0.2",
        "−0.16",
        "0.032"
      ],
      "answer": 0,
      "why": "a − y = −0.2 and the sigmoid slope is a(1 − a) = 0.8 × 0.2 = 0.16, so δ = −0.2 × 0.16 = −0.032. The value −0.2 forgets the slope (it is the answer only for cross-entropy), −0.16 forgets the loss slope, and 0.032 has the wrong sign."
    },
    {
      "topic": "Week 3: deep learning and backprop",
      "q": "A hidden layer has two ReLU units with z = (1.5, −0.5). They feed one output unit through the weights (2, −1), and the output blame is δ = 0.5. What is the blame (δ₁, δ₂) of the two hidden units?",
      "options": [
        "(1, −0.5)",
        "(1, 0)",
        "(0.5, 0)",
        "(0, −0.5)"
      ],
      "answer": 1,
      "why": "δ = (Wᵀ δ_next) ⊙ φ′(z) = (2×0.5, −1×0.5) ⊙ (1, 0) = (1, 0), because ReLU has slope 1 for z = 1.5 and slope 0 for z = −0.5. Forgetting the ReLU slope gives (1, −0.5); (0.5, 0) forgets the weights; (0, −0.5) takes the slope the wrong way round."
    },
    {
      "topic": "Week 3: deep learning and backprop",
      "q": "A layer has blame δ = (0.5, −2) at its two units. The previous layer's activations are a = (3, 1, 0). Using ∂L/∂W = δ aᵀ, what is the gradient of the weight that joins the first unit of the previous layer (activation 3) to the second unit of this layer (δ = −2)?",
      "options": [
        "1.5",
        "−2",
        "0",
        "−6"
      ],
      "answer": 3,
      "why": "Each weight's gradient is its unit's blame times the activation it multiplies: (−2) × 3 = −6. The value 1.5 is 0.5 × 3, which belongs to the first unit's weight; −2 forgets the activation; 0 is the gradient of the weights fed by the activation 0."
    },
    {
      "topic": "Week 3: deep learning and backprop",
      "q": "In an 8-layer sigmoid network, suppose every weight on the backward path is 1, so each layer multiplies the blame by σ′(z), which is at most 0.25. About how large can the product of the 8 factors be at most?",
      "options": [
        "About 2",
        "About 1.5 × 10⁻⁵",
        "About 3.9 × 10⁻³",
        "About 0.031"
      ],
      "answer": 1,
      "why": "The product is at most 0.25⁸ = 1.5 × 10⁻⁵, so the first layer receives almost no blame: this is the vanishing gradient. The value 2 is 8 × 0.25 (adding instead of multiplying), 3.9 × 10⁻³ is 0.25⁴ (only half the layers), and 0.031 is 0.25 / 8."
    },
    {
      "topic": "Week 3: deep learning and backprop",
      "q": "Why does a residual connection a⁽ˡ⁺¹⁾ = a⁽ˡ⁾ + F(a⁽ˡ⁾) help very deep networks to train?",
      "options": [
        "It makes F′ always equal to 0.25, which keeps the gradient the same size.",
        "It removes the activation functions so that no derivative is needed.",
        "The factor per layer is 1 + F′, so the gradient always has an identity path backwards and is not forced to shrink at every layer.",
        "It lets the network skip the forward pass during training."
      ],
      "answer": 2,
      "why": "Differentiating the sum gives 1 + F′: the 1 is a direct route for the blame, so the product of factors does not collapse to zero as it does with a plain factor such as 0.25. It does not remove the forward pass or the activations."
    },
    {
      "topic": "Week 3: deep learning and backprop",
      "q": "Gradient clipping uses g ← g · min(1, c / ‖g‖). A gradient is g = (−9, 12) and the limit is c = 3. What is the clipped gradient?",
      "options": [
        "(−3, 3)",
        "(−1.8, 2.4)",
        "(−0.6, 0.8)",
        "(−9, 12)"
      ],
      "answer": 1,
      "why": "‖g‖ = √(81 + 144) = 15, which is more than 3, so g is multiplied by 3 / 15 = 0.2, giving (−1.8, 2.4), a vector of length 3 in the same direction. Cutting each number at 3 changes the direction, (−0.6, 0.8) is the unit vector with length 1, and (−9, 12) is unchanged."
    },
    {
      "topic": "Week 3: deep learning and backprop",
      "q": "Which statement about the remedies for deep networks is correct?",
      "options": [
        "Layer normalization subtracts the mean and divides by the standard deviation of one example's layer outputs, then applies a learnable scale and shift.",
        "Gradient clipping is the standard cure for vanishing gradients.",
        "Xavier initialization, with variance 1/n_in, is the right choice for ReLU because it doubles the weights.",
        "ReLU helps because its slope is at most 0.25."
      ],
      "answer": 0,
      "why": "Normalization keeps the numbers inside a layer at a steady size. Clipping treats exploding gradients, not vanishing ones; He initialization (variance 2/n_in) is the one with the factor 2 for ReLU; and ReLU has slope 1 for z > 0, which is why it helps."
    },
    {
      "topic": "Week 3: deep learning and backprop",
      "q": "A 20-layer sigmoid network trains no better than a 3-layer one. The printed gradient norms, from the last layer to the first, shrink by about 4 times at every layer. Which plan is best?",
      "options": [
        "Add gradient clipping with c = 1.",
        "Multiply the learning rate by 100 so the first layers get larger steps.",
        "Add ten more sigmoid layers to give the network more capacity.",
        "Switch to ReLU, use He initialization and add residual connections."
      ],
      "answer": 3,
      "why": "A factor of about 4 smaller per layer matches 1 / 0.25: the gradient is vanishing, so the plan must keep the factor per layer near 1 (ReLU, He initialization, residual connections). Clipping only shrinks gradients that are too big, and a larger learning rate or more sigmoid layers does not repair the product."
    },
    {
      "topic": "Week 4: convolutional networks",
      "q": "A 32 × 32 colour picture (3 channels) goes into a convolutional layer with 50 filters of size 3 × 3, each with a bias. How many parameters does this layer have?",
      "options": [
        "1,350",
        "1,400",
        "450",
        "153,650"
      ],
      "answer": 1,
      "why": "Each filter has 3 × 3 × 3 = 27 weights plus 1 bias, which is 28, and 50 × 28 = 1,400. The value 1,350 forgets the biases, 450 forgets the 3 input channels (3 × 3 × 50), and 153,650 is what a dense layer with 50 units would need (3,072 × 50 + 50)."
    },
    {
      "topic": "Week 4: convolutional networks",
      "q": "Which property of a convolutional layer lets it recognise the same small pattern, such as an edge, wherever it appears in the picture?",
      "options": [
        "Every output unit is connected to every pixel.",
        "Max pooling stores the exact position of every edge.",
        "The same kernel (shared weights) is applied at every position.",
        "A large stride makes the kernel jump to the edge."
      ],
      "answer": 2,
      "why": "Weight sharing means one kernel slides over the whole picture, so a pattern is found at any position and the parameter count does not depend on the picture size. A fully connected unit would need a separate set of weights for each position."
    },
    {
      "topic": "Week 4: convolutional networks",
      "q": "A 3 × 3 patch of the picture has rows (1, 2, 0), (0, 1, 3), (2, 1, 0). A kernel has rows (1, 0, −1), (2, 0, −2), (1, 0, −1) and bias b = 4. Using the multiply-and-add rule of the lesson (no flipping), what is the output after ReLU?",
      "options": [
        "1",
        "7",
        "0",
        "−3"
      ],
      "answer": 0,
      "why": "Multiply and add: row 1 gives 1, row 2 gives −6, row 3 gives 2, so the sum is −3; adding the bias gives −3 + 4 = 1, and ReLU(1) = 1. Flipping the kernel gives +3 + 4 = 7, forgetting the bias gives ReLU(−3) = 0, and stopping before bias and ReLU gives −3."
    },
    {
      "topic": "Week 4: convolutional networks",
      "q": "A 64 × 64 picture goes through a convolution with kernel size k = 5, padding p = 2 and stride s = 2. What is the output height (and width)?",
      "options": [
        "31",
        "32",
        "33",
        "64"
      ],
      "answer": 1,
      "why": "Output size = ⌊(n + 2p − k) / s⌋ + 1 = ⌊(64 + 4 − 5) / 2⌋ + 1 = ⌊31.5⌋ + 1 = 32. The value 31 forgets the + 1, 33 rounds 31.5 up before adding 1, and 64 would need stride 1 with this padding."
    },
    {
      "topic": "Week 4: convolutional networks",
      "q": "A 28 × 28 × 1 picture goes through: conv 5 × 5 with 8 filters (no padding, stride 1), 2 × 2 max pool (stride 2), conv 3 × 3 with 16 filters (padding 1, stride 1), 2 × 2 max pool (stride 2), flatten, and one dense layer with 10 outputs. Every conv and dense unit has a bias. How many parameters does the whole network have?",
      "options": [
        "7,112",
        "24,426",
        "7,146",
        "1,376"
      ],
      "answer": 2,
      "why": "Shapes: 28×28×1 → 24×24×8 → 12×12×8 → 12×12×16 → 6×6×16 → 576. Parameters: conv1 5·5·1·8 + 8 = 208, conv2 3·3·8·16 + 16 = 1,168, dense 576·10 + 10 = 5,770; total 7,146. Pooling has no parameters. 7,112 forgets the biases, 24,426 forgets the second pool, and 1,376 counts only the conv layers."
    },
    {
      "topic": "Week 4: convolutional networks",
      "q": "Which statement about a 2 × 2 max-pooling layer with stride 2 is correct?",
      "options": [
        "It halves the height and width, has no learnable parameters, and keeps the strongest value in each window, which gives a little tolerance to small shifts.",
        "It has 2 × 2 learnable weights per channel, which are updated by gradient descent.",
        "It doubles the number of channels while halving the height and width.",
        "In the backward pass it shares the gradient equally among the four cells of each window."
      ],
      "answer": 0,
      "why": "Max pooling copies the largest number of each 2 × 2 window, so it shrinks the map, has nothing to learn and keeps channels unchanged. In the backward pass the whole gradient goes to the cell that held the maximum; sharing equally is how average pooling works."
    },
    {
      "topic": "Week 4: convolutional networks",
      "q": "Follow the receptive field through four layers in order: conv 3 × 3 (stride 1), conv 3 × 3 (stride 1), 2 × 2 max pool (stride 2), conv 3 × 3 (stride 1). Using rf = rf_prev + (k − 1) · jump_prev and jump = jump_prev · s (starting at rf = 1, jump = 1), how many input pixels wide is the receptive field of a unit in the last layer?",
      "options": [
        "8",
        "9",
        "10",
        "12"
      ],
      "answer": 2,
      "why": "rf: 1 → 3 → 5 → (pool) 5 + 1×1 = 6, jump 2 → 6 + 2×2 = 10. The value 8 ignores the jump after the pool, 9 forgets that the pool adds its own (k − 1), and 12 adds k × jump instead of (k − 1) × jump."
    },
    {
      "topic": "Week 4: convolutional networks",
      "q": "A 3 × 3 input x has rows (1, 2, 0), (0, 1, 3), (2, 1, 0). A 2 × 2 kernel (stride 1, no padding) produces a 2 × 2 output whose gradient is G with rows (1, 0) and (2, −1). Using ∂L/∂K[a, c] = Σ G[i, j] · x[i + a, j + c], what is ∂L/∂K[0, 1]?",
      "options": [
        "0",
        "1",
        "3",
        "2"
      ],
      "answer": 1,
      "why": "For a = 0, c = 1 the input cells are x[i, j + 1]: (2, 0, 1, 3) for the windows (0,0), (0,1), (1,0), (1,1). So 1×2 + 0×0 + 2×1 + (−1)×3 = 2 + 0 + 2 − 3 = 1. The value 0 is ∂L/∂K[0, 0], 3 is ∂L/∂K[1, 0], and 2 counts only the first window."
    },
    {
      "topic": "Week 4: convolutional networks",
      "q": "A 2 × 2 kernel K has rows (2, 1) and (−1, 3). It was slid with stride 1 over a 3 × 3 input and gave a 2 × 2 output whose gradient G has rows (1, 0) and (2, −1). Using ∂L/∂x[p, q] = Σ G[i, j] · K[p − i, q − j] (only windows that covered the pixel), what is ∂L/∂x[1, 1], the centre pixel?",
      "options": [
        "−3",
        "1",
        "−2",
        "3"
      ],
      "answer": 3,
      "why": "All four windows cover the centre pixel, which sits at K[1−i, 1−j]: 1×K[1,1] + 0×K[1,0] + 2×K[0,1] + (−1)×K[0,0] = 1×3 + 0 + 2×1 − 1×2 = 3. Matching G and K cell by cell without the flip gives −3, swapping the kernel indices gives 1, and −2 uses only the last window."
    },
    {
      "topic": "Week 4: convolutional networks",
      "q": "One 2 × 2 window of a convolution output has the pre-activation values z = (−1, 2, 0.5, 3) (in this order). It goes through ReLU and then a 2 × 2 max pool, and the upstream gradient arriving at the pooled output is 1.5. What is the gradient with respect to the four values of z?",
      "options": [
        "(0, 0.375, 0.375, 0.375)",
        "(0, 1.5, 0, 0)",
        "(0, 1.5, 1.5, 1.5)",
        "(0, 0, 0, 1.5)"
      ],
      "answer": 3,
      "why": "ReLU gives a = (0, 2, 0.5, 3); the max pool picks 3 (the fourth cell), so the whole 1.5 goes to the fourth cell, and its ReLU gate is 1 because z = 3 > 0. The other cells get 0. Sharing 1.5 over the three active cells (0.375 each) would be average pooling, and (0, 1.5, 0, 0) or (0, 1.5, 1.5, 1.5) give blame to cells that were not the maximum."
    },
    {
      "topic": "Week 4: convolutional networks",
      "q": "A student's CNN has a first layer of 16 filters of size 3 × 3 on 1-channel pictures, with stride 1 and padding 1. She changes the input pictures from 28 × 28 to 56 × 56 (still 1 channel) and keeps this layer unchanged. What happens to that layer?",
      "options": [
        "Its parameters grow four times to 640, because there are four times as many pixels.",
        "It still has 160 parameters, and its output grows from 28 × 28 × 16 to 56 × 56 × 16.",
        "It still has 160 parameters, but its output stays 28 × 28 × 16.",
        "It needs 64 filters to cover the larger picture, so it has 640 parameters."
      ],
      "answer": 1,
      "why": "Parameters are 3·3·1·16 + 16 = 160, and they do not depend on the picture size because the kernels are reused at every position. With padding 1 and stride 1 the output height and width equal the input's, so they double to 56 × 56 with 16 channels."
    }
  ]
}
```

## How the Prelim is marked

**The Prelim (100 points).** The exam has two parts and is marked out of 100. The pass mark is 70 points.

| Component | Points | Time | What is assessed and how it is marked |
|---|---|---|---|
| Part A: Written exam | 40 | 60 min | 40 multiple-choice questions, 1 point each, no negative marking. The app marks it when you submit and then shows your score and a breakdown by topic (Weeks 1 and 2, Week 3, Week 4), so you can see where to revise. |
| Part B, Task 1: Generative AI and language models | 10 | 20 min | Practical notebook task, marked by your instructor. |
| Part B, Task 2: Supervised learning, neurons and generalisation | 12 | 25 min | Practical notebook task, marked by your instructor. |
| Part B, Task 3: Backpropagation and deep networks | 16 | 30 min | A forward and backward pass with new numbers, checked with autograd and finite differences; reading gradient norms of deep networks and repairing one. |
| Part B, Task 4: Convolutional networks | 16 | 30 min | A convolution and its sizes, the gradients dL/dK and dL/dx and their check, and a CNN compared with a dense network under shift. |
| Part B, Task 5: Synthesis | 6 | 15 min | Three short written scenarios that join several weeks, 2 points each. |
| **Total** | **100** | **180 min** | Pass mark: 70 points. |

**How Part B is marked.** Your instructor marks the notebook you submit with a marking scheme. Each sub-question earns points for three things: the correct result, a clear method (your code and your working must show how you got the answer, not only the answer), and, where the question asks for it, an interpretation in your own words that uses your numbers. Partial credit is given: a correct method with an arithmetic slip earns most of the points, and a correct result with no visible method earns fewer. The ✓ and ✗ lines printed by the check cells are a guide to whether a number or shape is right; they are not the mark, and a ✓ cannot replace a missing method or explanation. Written answers in `ANSWERS` are read and marked on their content.

**Results and feedback.** Part A shows your score and topic breakdown as soon as you submit. Your instructor announces when Part B is marked and how you receive feedback, and announces the date, the room, any re-sit arrangements and how the Prelim counts towards the course. Use the topic breakdown and your marked notebook to plan your revision for the weeks ahead.

**Across the course.** Laboratories and knowledge checks run through the term, with exams in Weeks 5, 10, 15 and 20, plus a capstone. Your instructor announces the component weights in the syllabus. The long-term goal is the NVIDIA-Certified Associate exam (NCA-GENL).

## Exit ticket and homework

**Exit ticket (answer in two or three sentences each):**

1. Which topic in your score breakdown was weakest, and what exactly will you do about it this week?
2. What is one thing the exam made you realise about how well you understand the material, compared with how well you thought you did?
3. What is one question you still have about Weeks 1 to 4?

**Homework (no new deadlines):**

- Open your Part A score breakdown and read the explanation of every question you missed. Redo the matching practice problem on this page for each missed topic.
- If your instructor returns your Part B notebook with comments, read them and note what to do differently.
- Preview Week 6: **tokenisation and text as numbers**. Think about how a computer could turn the sentence "Magandang umaga" into numbers a network can use.
