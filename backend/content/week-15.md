> Content based on the NVIDIA Generative AI and DLI Deep Learning Teaching Kits, used under the Supervaise × NVIDIA partnership.

# Week 15 — Pre-final Exam + Capstone Proposal

**Purpose:** Prepare for the Pre-final by ==reviewing Weeks 11 to 14 and practising the calculations and diagnoses== you will be asked to do, in a written exam and a practical notebook exam, and start your Capstone proposal for the Structural Inspection Assistant.

**Term:** Pre-final (Weeks 11–15) · **Month:** 4 · **Time:** Part A 60 minutes, Part B 120 minutes, plus your revision

```viz
{
  "type": "hook",
  "title": "Watch four weeks in one minute",
  "caption": "The Pre-final covers Weeks 11 to 14: pictures become patch tokens, a generator and a discriminator play a game, noise is added and removed step by step, and a big model is adapted with a few new numbers. The numbers in these scenes are computed for teaching with new examples; nothing here is new material.",
  "scenes": [
    { "name": "Week 11: patch tokens", "kind": "text", "outputLabel": "How many tokens is one photo?", "prompt": "A 320 × 320 colour photo is cut into 16 × 16 patches. How many tokens, and how many attention scores per head?", "output": "320 ÷ 16 = 20 patches across and 20 down, so 400 patches and 401 tokens with the class token. Attention needs 401 × 401 = 160,801 scores per head, against 102,400 pixel tokens and more than 10 billion scores if every pixel were a token." },
    { "name": "Week 12: the generator's gradient", "kind": "bars", "outputLabel": "Size of the gradient reaching the generator when D(fake) = 0.01 (computed)", "prompt": "The discriminator is nearly sure that a generated picture is fake. How large is the learning signal for each version of the generator loss?", "bars": [{ "label": "original loss ln(1 − D)", "value": 0.01 }, { "label": "non-saturating loss −ln D", "value": 0.99 }] },
    { "name": "Week 13: how much picture is left", "kind": "bars", "outputLabel": "Share ᾱ_t of the picture kept (linear schedule, T = 1000)", "prompt": "The forward process adds a little noise at each step. How much of the original picture's signal power is left after t steps?", "bars": [{ "label": "t = 250", "value": 0.524 }, { "label": "t = 500", "value": 0.0786 }, { "label": "t = 750", "value": 0.0034 }] },
    { "name": "Week 14: a few new numbers", "kind": "text", "outputLabel": "LoRA on one weight matrix, counted", "prompt": "A 4,096 × 4,096 weight matrix is adapted with a LoRA of rank 8. How many numbers are trained?", "output": "A is 8 × 4,096 and B is 4,096 × 8, so 2 × 4,096 × 8 = 65,536 trained numbers, against 16,777,216 in the full matrix: 0.39 percent, or about 1 in 256." }
  ]
}
```

## By the end, you can…

1. ==Compute with images as tokens and with CLIP== (Week 11): the number of patches and the sequence length, the numbers in a patch and the patch-embedding parameters, the n × n attention cost, a cosine-similarity matrix, zero-shot probabilities and the contrastive loss, and the cost of image tokens in a language model's context.
2. ==Compute and explain the GAN game== (Week 12): the value V(D, G), the optimal discriminator D* = p_data / (p_data + p_g) and the equilibrium −ln 4, why the original generator loss saturates, the output size of a transpose convolution and the parameters of a DCGAN.
3. ==Compute and explain diffusion== (Week 13): the noise schedule and ᾱ_t, the closed-form noising step, the signal-to-noise ratio, the noise-prediction loss, one DDPM or DDIM step, the sinusoidal time embedding, classifier-free guidance and the size saving of latent diffusion.
4. ==Compute and explain how a big model is adapted== (Week 14): the masked SFT loss, the Bradley–Terry probability, the KL-regularised optimum, the DPO loss, the parameters of LoRA and the memory of full fine-tuning, LoRA and QLoRA.
5. ==Diagnose a situation from numbers==: a discriminator that wins too easily, a collapsed generator, a diffusion model that ignores the time step, a reward that is being hacked, or a model that forgot its earlier skills.
6. ==Choose a design and defend it==: a fusion pattern, a loss, a sampler, or SFT against RLHF against DPO, with the numbers or reasons to back your choice, including who decides when an AI system is used to inspect a structure.

Assessed in the Pre-final exam described below. Your instructor announces the date, room and how the exam counts towards your course; this page does not. Read each review section, cover the solutions of the practice problems and try them first, then use the figures to test yourself. The figures are practice, so you cannot break anything.

## About this exam {rail=About}

{lede} The Pre-final is ==one exam in two parts, marked out of 100 points==. It does not ask you to repeat sentences from the lessons; it asks whether you can *do* what Weeks 11 to 14 taught: compute a small result by hand, explain why something happens, and diagnose a model that misbehaves.

### The format

| Part | What it is | Time | Points |
|---|---|---|---|
| Part A: written exam | 40 multiple-choice questions, 1 point each, closed book, taken on this page | 60 minutes | 40 |
| Part B: practical exam | Five tasks in a notebook (JupyterLab on the school's DGX Spark, or the Colab notebook) | 120 minutes | 60 |
| Total | | 180 minutes | 100 |

The pass mark is ==70 points out of 100==. Part B is split into five tasks, each with its own time box. Keep to the time boxes: an unfinished Task 3 should not cost you Task 5.

| Part B task | Topic | Time | Points |
|---|---|---|---|
| Task 1 | Multimodal: patches, ViT and CLIP (Week 11) | 20 minutes | 12 |
| Task 2 | GANs and transpose convolution (Week 12) | 25 minutes | 14 |
| Task 3 | Diffusion (Week 13) | 30 minutes | 16 |
| Task 4 | Training LLMs: SFT, preference learning, LoRA (Week 14) | 30 minutes | 12 |
| Task 5 | Synthesis with the Capstone, short written answers | 15 minutes | 6 |

### What the exam covers

The exam is cumulative for the Pre-final term, Weeks 11 to 14. Every week appears in both parts, so a weak week cannot hide. The Capstone (the Structural Inspection Assistant) appears in Task 5, where you join the four weeks together.

| Topic | Part A questions | Part B task |
|---|---|---|
| Week 11: images as tokens and CLIP | 10 | Task 1 (12 points) |
| Week 12: GANs and transpose convolution | 10 | Task 2 (14 points) |
| Week 13: diffusion models | 10 | Task 3 (16 points) |
| Week 14: training LLMs | 10 | Task 4 (12 points) |
| All four weeks together | none | Task 5 (6 points) |
| Total | 40 | 60 points |

In Part A, about 30 percent of the questions test whether you remember and understand an idea, about 50 percent ask you to apply it (at least 16 of the 40 need a calculation, and the numbers are in the question), and about 20 percent ask you to analyse a situation and diagnose what is wrong. The questions are new, so memorised answers will not help, but they are fair for anyone who did the lessons and labs.

### The rules

- **Part A is closed book.** No notes, no phones, no other tabs. A calculator and scrap paper are fine.
- **Part B is open lessons and your own lab notes.** You may not use AI assistants, share work, or search the internet for answers. A calculator and paper are fine. Keep your own work: every cell you submit must be yours.
- **Integrity.** The written exam runs in your browser and cannot stop a determined cheater, so it is meant to be taken under the supervision your instructor arranges. Please play fair; a score is only useful to you if it is honestly yours.
- **Date and weight.** Your instructor announces the date, room, and how the exam counts towards your course. Do not rely on rumours; ask your instructor.

### How to prepare

1. Read the four review sections on this page and try every practice problem with the solution covered.
2. Learn the formula sheet in the Revision section well enough to write it from memory, because Part A is closed book.
3. Redo the formative quiz at the end of each of Weeks 11 to 14, and rerun your lab notebooks, trying to finish each Check cell without looking at your old answers.
4. Practise with [Mock Exam Prep](/mock-exam), and read the Capstone proposal section so that Task 5 does not surprise you.
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

The time boxes in Part B add up to exactly 120 minutes (20 + 25 + 30 + 30 + 15), so there is no spare time. If you are stuck, write what you know and move on. Nothing in Part B needs a download or a large model: every cell runs on a CPU in under a minute and a half.

## Week 11: images and CLIP {rail=Week 11}

{lede} Week 11 asked how one model can work with pictures and words together. A [[multimodal model|A model that takes in, produces or links more than one kind of data, such as pictures and text.]] needs every input as numbers, so ==a Vision Transformer cuts a picture into patches and treats each patch as a token, and CLIP trains pictures and sentences to share one space== so that they can be compared.

### You must be able to

- Compute the number of [[patch tokens|The squares a picture is cut into for a Vision Transformer. Each square is flattened and projected into one token.]] N = (H ÷ P) × (W ÷ P), the sequence length N + 1 with the class token, the P·P·C numbers in a flattened patch and the patch-embedding parameters P·P·C·d + d.
- Say why a pixel-per-token model is too costly (attention needs n² scores) and how halving the patch side changes the token count (four times more) and the scores (about sixteen times more).
- Explain why a bag of patches needs [[position embeddings|Learned vectors, one for each position, added to the patch tokens so the model knows where each patch came from.]], and say when a task does not need them.
- Trace an image through a ViT: patchify, one shared linear projection, class token, positions, encoder blocks and head.
- Compute a cosine-similarity matrix, the logits scale × cosine, zero-shot probabilities and the symmetric contrastive loss of a small batch, and say what a larger or smaller scale does.
- Compute the size of a projector and the share of a context window that image tokens use, and choose between a shared space, prefix tokens and cross-attention.

### Patches, tokens and counts

```viz
{
  "type": "equation",
  "title": "From a picture to patch tokens",
  "intro": "Select each part, then step through a new example: a 256 × 256 colour photo, 16 × 16 patches and a hidden size of 512.",
  "formula": [
    { "t": "N", "id": "N" }, " = (H ÷ ", { "t": "P", "id": "P" }, ") × (W ÷ P),   sequence = ", { "t": "N + 1", "id": "seq" }, ",   numbers per patch = ", { "t": "P·P·C", "id": "pc" }, ",   embedding parameters = ", { "t": "P·P·C·d + d", "id": "emb" }
  ],
  "parts": [
    { "id": "N", "name": "N (patches)", "meaning": "How many squares the picture is cut into. H and W must be multiples of P; a picture whose sides are not is resized, padded or cropped first." },
    { "id": "P", "name": "P (patch side)", "meaning": "The side of one square in pixels. Halving P makes N four times larger." },
    { "id": "seq", "name": "N + 1 (sequence length)", "meaning": "The patches plus one learned class token whose output summarises the picture." },
    { "id": "pc", "name": "P·P·C", "meaning": "The numbers in one flattened patch: P rows, P columns and C colour channels. C is 3 for colour and 1 for grey." },
    { "id": "emb", "name": "P·P·C·d + d", "meaning": "One shared matrix E of size (P·P·C) × d and a bias of d numbers, used for every patch. It does not depend on how many patches there are." }
  ],
  "steps": [
    { "title": "Patches", "expr": "256 ÷ 16 = 16 across and 16 down:  N = 16 × 16 = 256 patches", "note": "A made-up photo size. A non-square 224 × 160 photo would give 14 × 10 = 140 patches." },
    { "title": "Sequence", "expr": "N + 1 = 257 tokens", "note": "256 patch tokens plus the class token." },
    { "title": "One flattened patch", "expr": "16 × 16 × 3 = 768 numbers", "note": "Cutting into patches loses nothing; it only reorganises the numbers." },
    { "title": "Patch embedding", "expr": "768 × 512 + 512 = 393,728 parameters", "note": "Shared by all 256 patches." },
    { "title": "Attention cost", "expr": "257 × 257 = 66,049 scores per head; one token per pixel: 65,536² = 4,294,967,296", "note": "Patches make the sequence 255 times shorter and the score table about 65,000 times smaller." }
  ],
  "note": "The photo and hidden size are chosen for practice; the arithmetic is exact."
}
```

```viz
{ "type": "calc", "kind": "vit_tokens", "title": "Move the image side and the patch side", "d": 512 }
```

Try an image side of 256 and a patch of 16 in the calculator: you should read 256 patches, a sequence of 257 and 393,728 patch-embedding parameters. Then try a patch of 32 (64 patches, 65 tokens) and a patch of 8 (1,024 patches). ==Halving the patch side multiplies the tokens by 4 and the attention scores by about 16.== Now cut a small picture yourself: a wall with a diagonal crack.

```viz
{
  "type": "patch",
  "title": "A cracked wall cut into tokens",
  "image": [
    [1,1,1,1,1,1,9,9],
    [1,1,1,1,1,9,9,1],
    [1,1,1,1,9,9,1,1],
    [1,1,1,9,9,1,1,1],
    [1,1,9,9,1,1,1,1],
    [1,9,9,1,1,1,1,1],
    [9,9,1,1,1,1,1,1],
    [9,1,1,1,1,1,1,1]
  ],
  "channels": 1,
  "patchSizes": [2, 4]
}
```

With 2 × 2 patches the 8 × 8 picture gives 16 tokens of 4 numbers each; with 4 × 4 patches it gives only 4 tokens of 16 numbers, and every one of them mixes crack and wall. The illustrative drawing shows the trade-off: small patches keep detail and cost more, large patches are cheap and blurry.

### Positions: why a bag of patches is not enough

```viz
{
  "type": "patchshuffle",
  "title": "Shuffle the patches of the cracked wall",
  "intro": "Switch between the original and the shuffled patches, and turn the position embeddings on and off.",
  "image": [
    [1,1,1,1,1,1,9,9],
    [1,1,1,1,1,9,9,1],
    [1,1,1,1,9,9,1,1],
    [1,1,1,9,9,1,1,1],
    [1,1,9,9,1,1,1,1],
    [1,9,9,1,1,1,1,1],
    [9,9,1,1,1,1,1,1],
    [9,1,1,1,1,1,1,1]
  ],
  "patch": 4,
  "seed": 4,
  "note": "A tiny illustrative picture; the shuffle is fixed so that you can compare."
}
```

Self-attention treats its inputs as a set: ==shuffle the patches in, and you get the same outputs shuffled out==. Without positions the model cannot tell a crack running along a wall from the same patches scrambled. The fix is to add a vector for each position to the patch tokens (learned, or sinusoidal as in Week 8). A task such as "does this photo contain any rust at all?" could survive a shuffle; a task such as "where does the crack end?" cannot. ==If shuffling the pieces cannot change the right answer, positions are unnecessary, and if it can, they are essential.==

### CLIP: one space for pictures and words

```viz
{
  "type": "equation",
  "title": "Logits and the symmetric contrastive loss",
  "intro": "Select each part, then follow a new 2 × 2 batch: two photos, two captions, all vectors of length 1, logit scale 10.",
  "formula": [
    { "t": "s", "id": "s" }, { "t": "ij", "id": "ij" }, " = ", { "t": "scale", "id": "scale" }, " × cos(imageᵢ, textⱼ),   ", { "t": "L", "id": "L" }, " = average of −ln ", { "t": "pᵢᵢ", "id": "pii" }, " over the rows and the columns"
  ],
  "parts": [
    { "id": "s", "name": "s (logit)", "meaning": "The score that goes into a softmax for photo i and caption j." },
    { "id": "ij", "name": "ij (row, column)", "meaning": "Row i is a photo and column j is a caption. The diagonal holds the true pairs." },
    { "id": "scale", "name": "scale (logit scale)", "meaning": "The number that stretches the cosines before the softmax. It equals 1 divided by the temperature; a large scale makes the softmax sharp." },
    { "id": "L", "name": "L (the loss)", "meaning": "The average over both directions of the surprise at the true pair. It is near 0 when the diagonal wins everywhere." },
    { "id": "pii", "name": "pᵢᵢ (probability of the true pair)", "meaning": "The softmax probability along a row (which caption fits this photo?) or along a column (which photo fits this caption?)." }
  ],
  "steps": [
    { "title": "Vectors", "expr": "photos (0.8, 0.6) and (0.28, 0.96);  captions (0.6, 0.8) and (0, 1)", "note": "Made-up vectors, all of length 1, so the dot product is the cosine." },
    { "title": "Cosines", "expr": "[[0.96, 0.60], [0.936, 0.96]]", "note": "Photo 2 is almost as close to caption 1 (0.936) as to its own caption (0.96)." },
    { "title": "Logits with scale 10", "expr": "[[9.6, 6.0], [9.36, 9.6]]", "note": "Each cosine times 10." },
    { "title": "Probabilities of the true pairs", "expr": "row 1: 1/(1 + e^−3.6) = 0.9734;  row 2: 1/(1 + e^−0.24) = 0.5597;  the columns give 0.5597 and 0.9734", "note": "Photo 2 hedges between the two captions." },
    { "title": "The loss", "expr": "rows: (0.0270 + 0.5802) ÷ 2 = 0.3036;  columns: 0.3036;  L = 0.304", "note": "A larger loss than a clean batch; a perfect diagonal would give a value near 0." }
  ],
  "note": "The vectors are made up for practice."
}
```

```viz
{
  "type": "clipsim",
  "title": "Three inspection photos and three captions",
  "intro": "Select the rust photo: how is its probability split between the captions? Then move the logit scale.",
  "scale": 10,
  "images": [
    { "label": "photo of a crack", "v": [0.9, 0.5, 0.1, 0.1] },
    { "label": "photo of rust", "v": [0.4, 0.9, 0.1, 0.2] },
    { "label": "photo of a clean wall", "v": [0.1, 0.1, 0.9, 0.3] }
  ],
  "texts": [
    { "label": "a photo of a crack", "v": [1, 0.5, 0, 0] },
    { "label": "a photo of rust", "v": [0.3, 1, 0.1, 0.1] },
    { "label": "a photo of a clean wall", "v": [0, 0.1, 1, 0.2] }
  ],
  "note": "The vectors are illustrative, authored for teaching, not from a trained model.",
  "checks": [
    { "scale": 10, "cos": [[0, 0, 0.9898], [1, 1, 0.9868], [0, 1, 0.7215]], "prob": [[0, 0, 0.9358], [1, 1, 0.9118]], "loss": 0.0531 },
    { "scale": 5, "prob": [[0, 0, 0.7829]], "loss": 0.19 },
    { "scale": 30, "prob": [[0, 0, 0.9997]], "loss": 0.0004 }
  ]
}
```

The crack and rust photos are close (cosine 0.72), so at scale 10 the crack photo gives its own caption 0.936 and the rust caption 0.064. At scale 5 the probabilities are flatter (0.783) and the loss is higher (0.190); at scale 30 the loss is almost 0, and a wrong pair would be punished very hard. ==A small temperature, or a large scale, gives a sharp softmax.== For zero-shot classification you write one caption per class and pick the largest softmax: with cosines 0.30, 0.25 and 0.05 and scale 20 the probabilities are 0.727, 0.268 and 0.005, but ==CLIP can only choose among the prompts you give it==.

### Plugging vision into a language model

```viz
{
  "type": "equation",
  "title": "How much of the context does an image use?",
  "intro": "Select each part. The example is a new configuration: a 448 × 448 photo, 14 × 14 patches, vision width 768, language-model width 2,048, a 64-token prompt and an 8,192-token context.",
  "formula": [
    { "t": "T", "id": "T" }, " = F·W + b,   ", { "t": "share", "id": "share" }, " = ", { "t": "n_img", "id": "ni" }, " ÷ context,   scores = ", { "t": "n²", "id": "n2" }
  ],
  "parts": [
    { "id": "T", "name": "T = F·W + b (the projector)", "meaning": "A learned linear layer maps the vision features F (one row per patch) to the language model's width. Its parameters are d_vision × d_model + d_model." },
    { "id": "share", "name": "share", "meaning": "The part of the context window taken by the picture." },
    { "id": "ni", "name": "n_img (image tokens)", "meaning": "One per patch, so (H ÷ P) × (W ÷ P)." },
    { "id": "n2", "name": "n² (attention scores)", "meaning": "Every token attends to every token, so the table has n_img + n_text, squared, entries per head." }
  ],
  "steps": [
    { "title": "Image tokens", "expr": "448 ÷ 14 = 32;  n_img = 32 × 32 = 1,024", "note": "An example configuration, not a specific model." },
    { "title": "Projector", "expr": "768 × 2,048 + 2,048 = 1,574,912 parameters", "note": "Tiny next to a language model with billions." },
    { "title": "Share of the context", "expr": "1,024 ÷ 8,192 = 0.125, or 12.5%", "note": "One photo uses an eighth of the window." },
    { "title": "Attention scores", "expr": "(1,024 + 64)² = 1,183,744, against 64² = 4,096 for the text alone: 289 times more", "note": "Image resolution is a real cost." }
  ],
  "note": "The configuration is chosen for practice."
}
```

```viz
{
  "type": "sorter",
  "title": "Which way of combining modalities?",
  "intro": "Sort each description into the design it fits best.",
  "bins": [{ "id": "clip", "label": "Shared space (CLIP)" }, { "id": "prefix", "label": "Projector + prefix tokens" }, { "id": "cross", "label": "Cross-attention" }],
  "cards": [
    { "name": "Compares a photo with the written captions by cosine and writes no text", "answer": "clip", "why": "Two encoders map to one space; the output is a similarity, not a sentence." },
    { "name": "Image tokens are placed in front of the prompt and take room in the context window", "answer": "prefix", "why": "The projected patches become extra tokens of the language model." },
    { "name": "Text tokens form the queries; image features give the keys and values", "answer": "cross", "why": "That is the definition of cross-attention." },
    { "name": "Cannot name a class that is not among the prompts you wrote", "answer": "clip", "why": "Zero-shot classification only chooses among the given captions." },
    { "name": "Often trained by learning a small layer between a frozen vision encoder and a language model", "answer": "prefix", "why": "The projector is small, so only it needs new training." },
    { "name": "Images do not use any of the text context window", "answer": "cross", "why": "The image enters through extra attention layers instead of extra tokens." },
    { "name": "Trained with a contrastive loss on matched picture–caption pairs", "answer": "clip", "why": "True pairs on the diagonal are pulled together, mismatched pairs pushed apart." },
    { "name": "A long photo prompt can push the earlier conversation out of the context", "answer": "prefix", "why": "Every image token counts in the sequence length." }
  ]
}
```

A model that reads a picture can still be wrong in a fluent way: ==the answer must be checked by an expert== whenever it matters, as with an inspection photo.

### Practice problems

Cover each solution, try the problem on paper, then check. The numbers are new, so you have not seen these answers before.

**Problem 1. Patches and a ViT.** A 192 × 192 colour image is cut into 12 × 12 patches, and the hidden size is d = 384. (a) How many patches and how many tokens with the class token? (b) How many numbers does one flattened patch hold? (c) How many parameters has the patch embedding? (d) How many attention scores per head? (e) With 24 × 24 patches instead, how many tokens and how many scores, and how many times fewer scores?

**Solution 1.**

- (a) 192 ÷ 12 = 16, so 16 × 16 = ==256 patches== and ==257 tokens==.
- (b) 12 × 12 × 3 = ==432 numbers==.
- (c) 432 × 384 + 384 = ==166,272== parameters (one matrix and one bias, shared by all patches).
- (d) 257² = ==66,049== scores per head.
- (e) 192 ÷ 24 = 8, so 64 patches and ==65 tokens==; 65² = ==4,225== scores. The ratio is 66,049 ÷ 4,225 = ==15.6==: doubling P gives about 16 times fewer scores, but each patch now holds four times as many numbers.

**Problem 2. CLIP.** A batch of 2 pairs has cosines 0.9 and 0.3 in row 1 and 0.5 and 0.8 in row 2 (row i is photo i, column j is caption j) and the logit scale is 5. (a) Write the logits. (b) Give the probability of the true caption for each photo. (c) Give the probability of the true photo for each caption. (d) Give the symmetric loss. (e) For a third photo with cosines 0.30, 0.25 and 0.05 to three captions and scale 20, give the zero-shot probabilities.

**Solution 2.**

- (a) Logits 4.5 and 1.5 in row 1, 2.5 and 4.0 in row 2.
- (b) Photo 1: 1 ÷ (1 + e^−3) = ==0.9526==. Photo 2: 1 ÷ (1 + e^−1.5) = ==0.8176==.
- (c) Caption 1: 1 ÷ (1 + e^−2) = ==0.8808==. Caption 2: 1 ÷ (1 + e^−2.5) = ==0.9241==.
- (d) Rows: (−ln 0.9526 − ln 0.8176) ÷ 2 = 0.1250; columns: (−ln 0.8808 − ln 0.9241) ÷ 2 = 0.1029; the symmetric loss is (0.1250 + 0.1029) ÷ 2 = ==0.114==.
- (e) Logits 6, 5 and 1: probabilities ==0.727, 0.268 and 0.005==. The first caption wins, but only because it was among the prompts.

**Problem 3. A language model with a picture.** A 224 × 224 photo is cut into 14 × 14 patches. The vision width is 1,024, the language-model width is 3,072, the prompt has 40 tokens and the context window is 2,048 tokens. (a) How many image tokens? (b) How many parameters has a one-layer projector? (c) What share of the context does the photo use? (d) How many attention scores per head, and how many times more than the text alone?

**Solution 3.**

- (a) 224 ÷ 14 = 16, so 16 × 16 = ==256 image tokens==.
- (b) 1,024 × 3,072 + 3,072 = ==3,148,800== parameters.
- (c) 256 ÷ 2,048 = ==0.125, or 12.5%== of the context.
- (d) (256 + 40)² = 296² = ==87,616== scores, against 40² = 1,600 for the text alone: ==54.8 times more==. **Watch out:** the attention table is squared in the *total* length, so the photo costs far more than 256 extra rows.

## Week 12: GANs and transpose convolution {rail=Week 12}

{lede} Week 12 asked how a network can make a picture instead of recognising one. A [[GAN|A generative adversarial network: a generator that makes samples and a discriminator that judges them are trained against each other.]] ==trains a generator and a discriminator against each other in a minimax game==, and the generator grows a small tensor into a picture with transpose convolutions.

### You must be able to

- Write the value V(D, G) = E[ln D(x)] + E[ln(1 − D(G(z)))], compute it for given D outputs, and say that V is never above 0.
- Compute the optimal discriminator D*(x) = p_data(x) ÷ (p_data(x) + p_g(x)), and state the equilibrium: p_g = p_data, D* = 1/2 and V = −ln 4 = −1.386.
- Explain why the original generator loss ln(1 − D(G(z))) [[saturates|A loss saturates when its gradient is almost zero in the region where the model most needs to learn.]], compute the two gradients at a given D(G(z)), and say why the non-saturating loss −ln D(G(z)) is used.
- Compute the output size of a transpose convolution n_out = (n − 1)·s − 2p + k + op, the parameters k·k·C_in·C_out + C_out, and trace a DCGAN generator.
- Describe the checkerboard artefact and name one cure (upsample, then convolve).
- Diagnose training from numbers: a discriminator that wins (L_D near 0), a guessing discriminator (L_D near 2 ln 2 = 1.386), and mode collapse (healthy losses, no variety).

### The game in numbers

```viz
{
  "type": "equation",
  "title": "The value of the game, with new numbers",
  "intro": "Select each part, then step through an example in which D gives 0.7 to a real picture and 0.4 to a fake.",
  "formula": [
    { "t": "V(D, G)", "id": "V" }, " = ", { "t": "ln D(x)", "id": "lnd" }, " + ", { "t": "ln(1 − D(G(z)))", "id": "lnf" }, ";   ", { "t": "D*(x)", "id": "dstar" }, " = p_data / (p_data + p_g)"
  ],
  "parts": [
    { "id": "V", "name": "V (value of the game)", "meaning": "D tries to make it large, G tries to make it small. It is never above 0." },
    { "id": "lnd", "name": "ln D(x)", "meaning": "The reward for calling a real picture real. Near 0 when D(x) is near 1." },
    { "id": "lnf", "name": "ln(1 − D(G(z)))", "meaning": "The reward for catching a fake. Near 0 when D(G(z)) is near 0, very negative when D is fooled." },
    { "id": "dstar", "name": "D*(x) (best discriminator)", "meaning": "The share of the probability at x that belongs to real data. It is 1/2 where the two distributions are equal." }
  ],
  "steps": [
    { "title": "The two terms", "expr": "ln 0.7 = −0.3567;  ln(1 − 0.4) = ln 0.6 = −0.5108", "note": "Both are at most 0." },
    { "title": "The value", "expr": "V = −0.3567 − 0.5108 = −0.8675", "note": "D's loss is the negative of this: 0.8675." },
    { "title": "A picture the generator makes three times as often", "expr": "p_data = 0.09, p_g = 0.27:  D* = 0.09 / 0.36 = 0.25", "note": "One picture in four like this is real." },
    { "title": "A generator that is off by a little", "expr": "p_data = (0.5, 0.5), p_g = (0.8, 0.2):  JSD = 0.0507, V(D*, G) = −1.3863 + 2 × 0.0507 = −1.2850", "note": "JSD is the Jensen–Shannon divergence; V comes down towards −ln 4 = −1.3863 from above." },
    { "title": "The equilibrium", "expr": "p_g = p_data:  D* = 1/2 everywhere and V = ln 0.5 + ln 0.5 = −ln 4 = −1.3863", "note": "At this point D is only guessing, and its loss per picture is ln 2 = 0.693." }
  ],
  "note": "The D outputs and distributions are made up for practice; the arithmetic is exact."
}
```

```viz
{ "type": "calc", "kind": "gan_loss", "title": "Move the discriminator's logits" }
```

Put the fake logit at −4.6 in the calculator (D(fake) is about 0.01). The original generator loss ln(1 − D) is nearly flat there, with a gradient of −D with respect to the logit, about −0.01, while the non-saturating loss −ln D has a gradient of D − 1, about −0.99. ==The original loss goes quiet exactly when the generator most needs a signal; the non-saturating loss does not.== At D(fake) = 0.01 the original loss is ln 0.99 = −0.0101 and the non-saturating loss is −ln 0.01 = 4.605.

```viz
{
  "type": "ganlab",
  "title": "A one-dimensional GAN with a new target",
  "intro": "Train for a few thousand steps, then switch the generator loss and the number of D steps and watch the spread.",
  "mu": 3,
  "sigma": 1,
  "lr": 0.1,
  "lrG": 0.02,
  "dSteps": 5,
  "nonSat": true,
  "seed": 21,
  "note": "A small teaching simulation, not a real image model.",
  "checks": [
    { "steps": 3000, "meanMin": 2.3, "meanMax": 3.4, "spreadMin": 0.3, "spreadMax": 1.8 }
  ]
}
```

The mean of the real data is learned quickly; the spread wobbles. ==A GAN is a tug of war, not a descent to a single minimum==, which is why the same code can give a healthy run or a collapsed one.

### Transpose convolution and the DCGAN

```viz
{
  "type": "equation",
  "title": "Output size and parameters of a transpose convolution",
  "intro": "Select each symbol, then step through four new examples.",
  "formula": [
    { "t": "n_out", "id": "nout" }, " = (", { "t": "n", "id": "n" }, " − 1)·", { "t": "s", "id": "s" }, " − 2", { "t": "p", "id": "p" }, " + ", { "t": "k", "id": "k" }, " + ", { "t": "op", "id": "op" }, ";   parameters = k·k·C_in·C_out + C_out"
  ],
  "parts": [
    { "id": "nout", "name": "n_out (output side)", "meaning": "The side of the grown output grid." },
    { "id": "n", "name": "n (input side)", "meaning": "The side of the small input grid." },
    { "id": "s", "name": "s (stride)", "meaning": "The gap between the stamped copies of the kernel. A bigger stride gives a bigger output." },
    { "id": "p", "name": "p (padding)", "meaning": "Rows and columns cropped from each border. Here padding shrinks the output." },
    { "id": "k", "name": "k (kernel side)", "meaning": "The side of the kernel that is stamped." },
    { "id": "op", "name": "op (output padding)", "meaning": "An extra row and column on one side, usually 0, used to settle the size when the stride is above 1." }
  ],
  "steps": [
    { "title": "The DCGAN layer", "expr": "n = 8, k = 4, s = 2, p = 1:  7·2 − 2 + 4 = 16", "note": "k = 4, s = 2, p = 1 exactly doubles the side." },
    { "title": "Kernel 3, padding 1", "expr": "n = 5, k = 3, s = 2, p = 1:  4·2 − 2 + 3 = 9", "note": "Not a doubling: 5 becomes 9." },
    { "title": "Output padding", "expr": "n = 6, k = 5, s = 3, p = 2, op = 1:  5·3 − 4 + 5 + 1 = 17", "note": "Without op the side would be 16." },
    { "title": "Kernel equals stride", "expr": "n = 3, k = 2, s = 2, p = 0:  2·2 + 2 = 6", "note": "The stamps tile with no overlap." },
    { "title": "Parameters of a layer", "expr": "k = 4, 80 channels in, 40 out:  4·4·80·40 + 40 = 51,240", "note": "The count does not depend on the picture size." }
  ],
  "note": "The sizes are chosen for practice; the arithmetic is exact."
}
```

```viz
{
  "type": "conv",
  "title": "Stamp the kernel by hand",
  "mode": "transpose",
  "stride": 2,
  "input": [[2, 1], [0, 3]],
  "kernel": [[1, 0, -1], [2, 1, 0], [0, 1, 1]]
}
```

A 2 × 2 input, a 3 × 3 kernel and stride 2 give (2 − 1)·2 + 3 = 5, a 5 × 5 output in which the stamps overlap and ==the stamped values are added==. Where the overlaps are uneven, the output shows a regular pattern: the checkerboard artefact, which appears when the kernel size is not divisible by the stride.

```viz
{
  "type": "upsample",
  "title": "Three ways to double a small grid",
  "intro": "Compare nearest neighbour, bilinear and the transpose convolutions, and look at the overlap counts for kernel 3 with stride 2.",
  "input": [[2, 4, 6], [1, 3, 5], [0, 2, 8]],
  "start": 4,
  "max": 9,
  "note": "Illustrative numbers."
}
```

Upsampling by a fixed rule and then applying an ordinary convolution has no uneven overlap and is a common cure for the checkerboard; the transpose convolution is the one that is learnable.

```viz
{
  "type": "cnnstack",
  "title": "A small DCGAN-style generator for 32 × 32 grey pictures",
  "intro": "Read down the shape column: 64 numbers, 4 × 4 × 192, then 8, 16 and 32. Find the layer with the most parameters.",
  "input": [1, 1, 64],
  "layers": [
    { "type": "fc", "units": 3072 },
    { "type": "reshape", "shape": [4, 4, 192] },
    { "type": "convT", "k": 4, "stride": 2, "pad": 1, "filters": 96 },
    { "type": "convT", "k": 4, "stride": 2, "pad": 1, "filters": 48 },
    { "type": "convT", "k": 4, "stride": 2, "pad": 1, "filters": 1 }
  ],
  "checks": [
    { "layer": 0, "shape": [1, 1, 3072], "params": 199680 },
    { "layer": 2, "shape": [8, 8, 96], "params": 295008 },
    { "layer": 3, "shape": [16, 16, 48], "params": 73776 },
    { "layer": 4, "shape": [32, 32, 1], "params": 769 },
    { "total": 569233 }
  ]
}
```

The count is 64 × 3,072 + 3,072 = 199,680 for the first layer, then 295,008, 73,776 and 769, for 569,233 in all (batch normalisation not counted). ==The generator is a stack of transpose convolutions that grows a tiny tensor to a picture, and the discriminator is a stack of strided convolutions that shrinks a picture to one number.==

### Reading training: what went wrong?

```viz
{
  "type": "sorter",
  "title": "Which failure is this?",
  "intro": "Sort each symptom into the failure that best explains it.",
  "bins": [{ "id": "vanishing", "label": "Vanishing gradient" }, { "id": "collapse", "label": "Mode collapse" }, { "id": "unstable", "label": "Instability" }],
  "cards": [
    { "name": "L_D = 0.01, L_G = 7.5, D(fake) = 0.001 and the samples stay as static", "answer": "vanishing", "why": "D wins so completely that the original generator loss gives almost no gradient." },
    { "name": "L_D = 1.38, L_G = 0.70, D(fake) = 0.49, but 9 of 10 samples are the same digit", "answer": "collapse", "why": "The losses look healthy while the variety has gone: only the samples show it." },
    { "name": "The generated mean swings between 1 and 5 and never settles", "answer": "unstable", "why": "Neither network settles at the equilibrium." },
    { "name": "Only 3 of the 8 blobs of a ring ever receive generated points", "answer": "collapse", "why": "Counting the covered modes is how collapse is measured." },
    { "name": "Switching to −ln D(G(z)) would give a much larger gradient at this point", "answer": "vanishing", "why": "The non-saturating gradient is largest where the original one is nearly zero." },
    { "name": "Quality improves for 2,000 steps, then collapses, then recovers", "answer": "unstable", "why": "The two networks keep overtaking each other." }
  ]
}
```

==The loss curves do not tell you whether the pictures are good: look at samples== and count the variety. A loss near 2 ln 2 = 1.386 for the discriminator (0.693 per picture) means it is only guessing, which is the healthy end state, but it is also what a collapsed run can show.

### Practice problems

Cover each solution, try the problem on paper, then check. The numbers are new, so you have not seen these answers before.

**Problem 1. The value of the game.** (a) D gives 0.7 to a real picture and 0.4 to a fake. Find V. (b) At some picture p_data = 0.09 and p_g = 0.27. Find D*. (c) What are V and D* at the equilibrium? (d) At D(G(z)) = 0.02, find the original and the non-saturating generator losses and their gradients with respect to the logit, and compare.

**Solution 1.**

- (a) V = ln 0.7 + ln 0.6 = −0.3567 − 0.5108 = ==−0.8675==.
- (b) D* = 0.09 ÷ (0.09 + 0.27) = ==0.25==.
- (c) D* = ==1/2== for every picture and V = ==−ln 4 = −1.386==.
- (d) The original loss is ln 0.98 = −0.0202 with gradient −D = −0.02; the non-saturating loss is −ln 0.02 = 3.912 with gradient D − 1 = −0.98. The second gradient is ==49 times larger== (0.98 ÷ 0.02), exactly when the generator is losing.

**Problem 2. Transpose convolution and parameters.** (a) A transpose convolution with k = 4, s = 2, p = 1 acts on a 6 × 6 input: what output side? (b) With k = 3, s = 2, p = 1, op = 1 on a 7 × 7 input? (c) How many parameters has a transpose convolution with k = 4 that takes 80 channels to 40? (d) Starting from a 4 × 4 map, how many layers with k = 4, s = 2, p = 1 give a 64 × 64 picture, and what are the sides?

**Solution 2.**

- (a) (6 − 1)·2 − 2 + 4 = ==12==.
- (b) (7 − 1)·2 − 2 + 3 + 1 = ==14==.
- (c) 4·4·80·40 + 40 = ==51,240==.
- (d) Each layer doubles the side: 4, 8, 16, 32, 64, so ==four layers==.

**Problem 3. Diagnose.** A DCGAN log at step 4,000 reads L_D = 1.37, L_G = 0.71 and D(fake) = 0.49, but the 16 samples in the grid show only two different digits out of the ten in the data. (a) What does a discriminator loss of 1.37 suggest? (b) What failure is this and how would you measure it? (c) Name two remedies you might try, and say why none is guaranteed.

**Solution 3.**

- (a) 1.37 is close to 2 ln 2 = 1.386, so D is nearly only guessing, as at the equilibrium. If you looked only at this number you would say the run is healthy.
- (b) ==Mode collapse==: the generator covers 2 of 10 modes. Measure it by counting the modes covered (here 2 ÷ 10 = 20%), for example by classifying the samples, and by looking at them.
- (c) For example a lower learning rate for G with more D steps, one-sided label smoothing (real label 0.9), or instance noise added to D's inputs. These are things people often try; ==none is guaranteed==, because the game itself keeps changing.

## Week 13: diffusion models {rail=Week 13}

{lede} Week 13 turned image generation into a loop of small, easy steps. The ==forward process destroys a picture with a fixed recipe; a network learns to predict the noise that was added; sampling runs the recipe backwards==, one network call per step. Two add-ons make it useful: guidance steers it, and a latent space makes it cheap.

### You must be able to

- Compute α_t = 1 − β_t, read ᾱ_t from the linear schedule (β from 0.0001 to 0.02 over 1,000 steps), and say what ᾱ_t means: the share of the picture's strength that survives t steps.
- Use the closed form x_t = √ᾱ_t · x₀ + √(1 − ᾱ_t) · ε to noise a given pixel, and compute the signal-to-noise ratio ᾱ_t / (1 − ᾱ_t).
- Write the training loss (the mean squared error between the true noise ε and the guess ε̂) and say why predicting noise is easier than predicting the picture.
- Do one DDPM reverse step and one DDIM step by hand, and compare the cost of 1,000 steps with 25.
- Compute a sinusoidal [[time embedding|A vector of sines and cosines of the step number t, fed to the denoiser so that one network knows how noisy its input is.]] and the parameters of one UNet block.
- Apply classifier-free guidance, ε̃ = ε(∅) + w · (ε(c) − ε(∅)), and compute the size ratio of a latent space.

### The forward process and the closed form

Take a new step, t = 200. In the linear schedule ᾱ_200 = 0.6590, so the picture keeps √0.6590 = 0.8118 of its strength and the noise gets √(1 − 0.6590) = 0.5839. The squares of the two weights always add to 1.

```viz
{
  "type": "equation",
  "title": "Noising one pixel at t = 200",
  "intro": "Select a part of the formula, then follow the worked example. This is the same formula as in Week 13, with a new step and new pixel values.",
  "formula": [ { "t": "x_t", "id": "xt" }, " = ", { "t": "√ᾱ_t", "id": "sig" }, " · ", { "t": "x₀", "id": "x0" }, " + ", { "t": "√(1 − ᾱ_t)", "id": "noi" }, " · ", { "t": "ε", "id": "eps" } ],
  "parts": [
    { "id": "xt", "name": "x_t", "meaning": "The noisy pixel after t forward steps. The network sees this." },
    { "id": "sig", "name": "√ᾱ_t", "meaning": "How much of the clean pixel is kept. It shrinks from 1 towards 0 as t grows." },
    { "id": "x0", "name": "x₀", "meaning": "The clean pixel of the original picture." },
    { "id": "noi", "name": "√(1 − ᾱ_t)", "meaning": "How much noise is mixed in. It grows from 0 towards 1." },
    { "id": "eps", "name": "ε", "meaning": "One random draw from a standard normal distribution. During training it is the answer the network must find." }
  ],
  "steps": [
    { "title": "Look up the schedule at t = 200", "expr": "ᾱ_200 = 0.6590", "note": "From the linear schedule, which you can compute with a cumulative product." },
    { "title": "The two weights", "expr": "√0.6590 = 0.8118 (picture),  √(1 − 0.6590) = √0.3410 = 0.5839 (noise)", "note": "0.6590 + 0.3410 = 1." },
    { "title": "One pixel", "expr": "x₀ = 0.6, ε = −0.8  →  x_200 = 0.8118 × 0.6 + 0.5839 × (−0.8) = 0.4871 − 0.4671 = 0.0200", "note": "The noise almost cancelled the pixel." },
    { "title": "The signal-to-noise ratio", "expr": "SNR_200 = 0.6590 / 0.3410 = 1.933", "note": "Still above 1: the picture part is about twice as strong as the noise part. In the linear schedule the SNR crosses 1 at t = 260." }
  ],
  "note": "The pixel values are made up for practice. The schedule values are computed from the linear schedule."
}
```

```viz
{
  "type": "schedule",
  "title": "Where is t = 200 on the schedule?",
  "intro": "Set the slider to t = 200 and read ᾱ, then move to t = 600 and see how little of the picture is left.",
  "T": 1000,
  "kind": "linear",
  "t": 200,
  "note": "Linear schedule with β from 0.0001 to 0.02. Try the cosine tab to see that it destroys the picture more slowly at first.",
  "checks": [ { "kind": "linear", "t": 200, "abar": 0.6590 }, { "kind": "linear", "t": 600, "abar": 0.0259 } ]
}
```

==The two directions of the schedule are worth remembering:== ᾱ_t falls from 1 to almost 0, the picture weight √ᾱ_t falls with it, and the SNR drops by orders of magnitude. At t = 600, ᾱ = 0.0259, so the picture weight is only 0.161.

### Training: predict the noise

The network sees x_t and t and returns ε̂, its guess of the noise. We know ε exactly, because we drew it, so training is ordinary supervised learning with a squared error. ==The network is never asked to draw a picture; it is asked to find the noise in a picture.==

```viz
{
  "type": "equation",
  "title": "The noise-prediction loss for one example",
  "intro": "Two pixels, one random step. In training this is averaged over a batch, each example with its own t and its own ε.",
  "formula": [ { "t": "L", "id": "l" }, " = mean( ", { "t": "ε", "id": "eps" }, " − ", { "t": "ε̂", "id": "eh" }, " )²" ],
  "parts": [
    { "id": "l", "name": "L", "meaning": "The loss for one example: the average squared difference over the pixels." },
    { "id": "eps", "name": "ε", "meaning": "The true noise that was added. Known exactly." },
    { "id": "eh", "name": "ε̂", "meaning": "The network's prediction of that noise, computed from x_t and t." }
  ],
  "steps": [
    { "title": "True noise and guess", "expr": "ε = (1.0, −0.6),  ε̂ = (0.5, −0.4)", "note": "Two pixels; the network is half right on the first." },
    { "title": "Differences", "expr": "ε − ε̂ = (0.5, −0.2)", "note": "Errors of 0.5 and 0.2." },
    { "title": "Square and add", "expr": "0.5² + 0.2² = 0.25 + 0.04 = 0.29", "note": "The sum of squares." },
    { "title": "Mean over pixels", "expr": "0.29 / 2 = 0.145", "note": "This is the number F.mse_loss would return." }
  ],
  "note": "Illustrative values authored for practice."
}
```

If the network is right about ε̂, you can get the clean picture back: x̂₀ = (x_t − √(1 − ᾱ_t) · ε̂) / √ᾱ_t. This is the first half of every DDIM step.

### Sampling: DDPM and DDIM

Sampling starts from pure noise and loops down from t = T to 1. One DDPM step removes a little predicted noise and then adds a little fresh noise z, which lets the sampler explore. Try a step at t = 600, where the noise is heavy.

```viz
{
  "type": "equation",
  "title": "One DDPM reverse step at t = 600",
  "intro": "Select the parts of the formula, then follow one pixel through the step.",
  "formula": [ { "t": "x_(t−1)", "id": "xp" }, " = (1 / ", { "t": "√α_t", "id": "sa" }, ") · ( ", { "t": "x_t", "id": "xt" }, " − ", { "t": "β_t / √(1 − ᾱ_t)", "id": "coef" }, " · ", { "t": "ε̂", "id": "eh" }, " ) + ", { "t": "σ_t", "id": "sig" }, " · ", { "t": "z", "id": "z" } ],
  "parts": [
    { "id": "xp", "name": "x_(t−1)", "meaning": "The slightly cleaner pixel, one step earlier." },
    { "id": "sa", "name": "√α_t", "meaning": "Undoes the small shrink of the forward step." },
    { "id": "xt", "name": "x_t", "meaning": "The current noisy pixel." },
    { "id": "coef", "name": "β_t / √(1 − ᾱ_t)", "meaning": "How much predicted noise to take away per unit of ε̂." },
    { "id": "eh", "name": "ε̂", "meaning": "The network's noise prediction at this pixel and step." },
    { "id": "sig", "name": "σ_t", "meaning": "The size of the fresh noise; here σ_t = √β_t." },
    { "id": "z", "name": "z", "meaning": "A new random draw. It is zero on the last step." }
  ],
  "steps": [
    { "title": "Look up the schedule at t = 600", "expr": "β_600 = 0.012032, α_600 = 0.987968, ᾱ_600 = 0.02588, √(1 − ᾱ_600) = 0.98698", "note": "β_600 = 0.0001 + 0.0199 × 599/999." },
    { "title": "How much to subtract", "expr": "0.012032 / 0.98698 = 0.012191", "note": "Per unit of ε̂." },
    { "title": "Subtract the predicted noise", "expr": "x_t = −0.7, ε̂ = −0.5  →  −0.7 − 0.012191 × (−0.5) = −0.7 + 0.006095 = −0.693905", "note": "The predicted noise was negative, so subtracting it pushes the pixel up a little." },
    { "title": "Undo the shrink", "expr": "−0.693905 / √0.987968 = −0.693905 / 0.993966 = −0.698117", "note": "A slight enlargement." },
    { "title": "Add fresh noise", "expr": "σ = √0.012032 = 0.10969;  z = 0.4  →  −0.698117 + 0.10969 × 0.4 = −0.698117 + 0.043876 = −0.654241", "note": "With z = 0 the answer would stay −0.698117." }
  ],
  "note": "The pixel, ε̂ and z are made up for practice; the schedule numbers are exact."
}
```

[[DDIM|A deterministic sampler that reuses the same trained network but jumps between far-apart steps, so 20 to 50 network calls replace 1,000.]] reuses the network but takes bigger jumps with no fresh noise: first the estimate x̂₀, then the picture at the earlier step s.

```viz
{
  "type": "equation",
  "title": "One DDIM jump from t = 600 to s = 500",
  "intro": "The same pixel and noise guess as the DDPM step above, but now the sampler jumps 100 steps and adds no fresh noise.",
  "formula": [ { "t": "x̂₀", "id": "xh" }, " = (", { "t": "x_t", "id": "xt" }, " − √(1 − ᾱ_t) · ", { "t": "ε̂", "id": "eh" }, ") / √ᾱ_t,      ", { "t": "x_s", "id": "xs" }, " = √ᾱ_s · x̂₀ + √(1 − ᾱ_s) · ε̂" ],
  "parts": [
    { "id": "xh", "name": "x̂₀", "meaning": "The estimate of the clean pixel implied by the current noise guess." },
    { "id": "xt", "name": "x_t", "meaning": "The current noisy pixel." },
    { "id": "eh", "name": "ε̂", "meaning": "The network's noise prediction." },
    { "id": "xs", "name": "x_s", "meaning": "The pixel at the earlier step s, rebuilt from x̂₀ with less noise. Always the same for the same input." }
  ],
  "steps": [
    { "title": "Numbers", "expr": "t = 600, s = 500: ᾱ_600 = 0.02588, ᾱ_500 = 0.07859;  x_t = −0.7, ε̂ = −0.5", "note": "Same pixel as the DDPM example." },
    { "title": "Estimate the clean pixel", "expr": "x̂₀ = (−0.7 − 0.98698 × (−0.5)) / 0.16087 = (−0.7 + 0.49349) / 0.16087 = −1.2837", "note": "√0.02588 = 0.16087." },
    { "title": "Weights at s = 500", "expr": "√0.07859 = 0.28033 (picture),  √(1 − 0.07859) = 0.95990 (noise)", "note": "Less noise than at t = 600." },
    { "title": "Rebuild the earlier pixel", "expr": "x_500 = 0.28033 × (−1.2837) + 0.95990 × (−0.5) = −0.35987 − 0.47995 = −0.8398", "note": "No fresh noise: this is deterministic." }
  ],
  "note": "Illustrative pixel and ε̂, exact schedule values."
}
```

==Cost is the number of network calls.== A GAN makes a picture in 1 call. If one call takes 0.04 seconds, 1,000 DDPM steps take 40 seconds and 25 DDIM steps take 1 second, a 40-times speed-up that costs a little quality.

```viz
{
  "type": "sorter",
  "title": "Forward, training or sampling?",
  "intro": "Place each statement in the part of the diffusion story it belongs to.",
  "bins": [ { "id": "forward", "label": "Forward noising" }, { "id": "train", "label": "Training the denoiser" }, { "id": "sample", "label": "Sampling a picture" } ],
  "cards": [
    { "name": "Mixes √ᾱ_t · x₀ with √(1 − ᾱ_t) · ε", "answer": "forward", "why": "That closed form is the forward process: a fixed recipe with nothing learned." },
    { "name": "Picks a random t for every picture in the batch", "answer": "train", "why": "Each training example gets its own t and its own ε." },
    { "name": "Compares ε with ε̂ using a mean squared error", "answer": "train", "why": "That is the training loss." },
    { "name": "Starts from pure Gaussian noise x_T", "answer": "sample", "why": "Generation begins with noise and removes it step by step." },
    { "name": "Has no learned parameters at all", "answer": "forward", "why": "The schedule β_1 … β_T is chosen by hand." },
    { "name": "Calls the network once per step, 25 to 1,000 times in total", "answer": "sample", "why": "Cost grows with the number of steps; DDIM uses fewer." },
    { "name": "Knows the exact answer ε because the noise was drawn by us", "answer": "train", "why": "That is why this is easy supervised learning with no adversary." },
    { "name": "Adds fresh noise z (DDPM) or none (DDIM)", "answer": "sample", "why": "The sampler choice decides whether each step is random or deterministic." }
  ]
}
```

### The denoiser: a UNet with a time embedding

The network must return an output of the same shape as its input, keep fine detail, and know how noisy the input is. A UNet gives the first two (an encoder that shrinks, a decoder that grows back, skip connections that carry detail across). The time embedding gives the third.

```viz
{
  "type": "equation",
  "title": "The sinusoidal time embedding at t = 20",
  "intro": "With d = 4 there are two pairs: a fast one and a slow one. Each step gets its own pattern.",
  "formula": [ "emb(", { "t": "t", "id": "t" }, ")_(2k) = sin( ", { "t": "t", "id": "t2" }, " / ", { "t": "10000^(2k/d)", "id": "f" }, " ),      emb(", { "t": "t", "id": "t3" }, ")_(2k+1) = cos( … )" ],
  "parts": [
    { "id": "t", "name": "t", "meaning": "The step number, 1 to 1,000." },
    { "id": "t2", "name": "t", "meaning": "The same step number, divided by a frequency scale." },
    { "id": "f", "name": "10000^(2k/d)", "meaning": "The wavelength for pair k. For d = 4: 1 for k = 0 and 100 for k = 1." },
    { "id": "t3", "name": "t", "meaning": "The cosine uses the same angle as the sine." }
  ],
  "steps": [
    { "title": "The two frequencies for d = 4", "expr": "k = 0: 1/10000⁰ = 1;   k = 1: 1/10000^(2/4) = 0.01", "note": "A fast pair and a slow pair." },
    { "title": "The angles at t = 20", "expr": "20 × 1 = 20 radians,   20 × 0.01 = 0.2 radians", "note": "Radians, not degrees." },
    { "title": "The embedding", "expr": "(sin 20, cos 20, sin 0.2, cos 0.2) = (0.9129, 0.4081, 0.1987, 0.9801)", "note": "The slow pair is close to (0, 1) and barely moves between nearby steps; the fast pair changes quickly." }
  ],
  "note": "Exact values. Real models use d = 128 or more."
}
```

Notice that the vector is the same for the same t and different for different t, so the network can tell a gentle t = 20 from a heavy t = 800. The embedding is turned into one number per channel by a linear layer and added to the feature maps of every block.

```viz
{
  "type": "equation",
  "title": "Parameters of one UNet down block",
  "intro": "Two 3 × 3 convolutions and one linear layer for the time vector, with 16 channels in, 32 out and a time vector of D = 64.",
  "formula": [ { "t": "P", "id": "p" }, " = ( 9 · ", { "t": "C_in", "id": "cin" }, " · ", { "t": "C_out", "id": "cout" }, " + C_out ) + ( 9 · C_out² + C_out ) + ( ", { "t": "D", "id": "d" }, " · C_out + C_out )" ],
  "parts": [
    { "id": "p", "name": "P", "meaning": "Trainable numbers in one block, biases included." },
    { "id": "cin", "name": "C_in", "meaning": "Channels entering the block." },
    { "id": "cout", "name": "C_out", "meaning": "Channels leaving the block; also the number of filters." },
    { "id": "d", "name": "D", "meaning": "Length of the time vector that the linear layer maps to one number per channel." }
  ],
  "steps": [
    { "title": "First convolution", "expr": "9 × 16 × 32 + 32 = 4,640", "note": "Each of 32 filters has a 3 × 3 kernel over 16 channels." },
    { "title": "Second convolution", "expr": "9 × 32 × 32 + 32 = 9,248", "note": "32 channels in and 32 out." },
    { "title": "Time projection", "expr": "64 × 32 + 32 = 2,080", "note": "One number per channel." },
    { "title": "Total", "expr": "4,640 + 9,248 + 2,080 = 15,968", "note": "The pooling and the skip copy add no parameters." }
  ],
  "note": "Illustrative sizes chosen for practice."
}
```

### Guidance and latent diffusion

Classifier-free guidance runs the network twice, with and without the prompt, and pushes the answer w times as far in the direction the prompt adds.

```viz
{
  "type": "equation",
  "title": "Classifier-free guidance with w = 4",
  "intro": "Two pixels. The unconditional guess ignores the prompt; the conditional guess uses it.",
  "formula": [ { "t": "ε̃", "id": "et" }, " = ", { "t": "ε(x_t, ∅)", "id": "eu" }, " + ", { "t": "w", "id": "w" }, " · ( ", { "t": "ε(x_t, c)", "id": "ec" }, " − ε(x_t, ∅) )" ],
  "parts": [
    { "id": "et", "name": "ε̃", "meaning": "The guided noise prediction used for the sampling step." },
    { "id": "eu", "name": "ε(x_t, ∅)", "meaning": "The network's guess with the prompt left empty." },
    { "id": "w", "name": "w", "meaning": "The guidance scale. w = 0 ignores the prompt, w = 1 is the plain conditional guess, larger exaggerates the prompt." },
    { "id": "ec", "name": "ε(x_t, c)", "meaning": "The guess with the prompt c supplied." }
  ],
  "steps": [
    { "title": "The two guesses", "expr": "ε(x_t, ∅) = (−0.1, 0.3),  ε(x_t, c) = (0.3, 0.1)", "note": "Illustrative values." },
    { "title": "What the prompt adds", "expr": "(0.3, 0.1) − (−0.1, 0.3) = (0.4, −0.2)", "note": "The direction the prompt pushes." },
    { "title": "Scale by w = 4", "expr": "4 × (0.4, −0.2) = (1.6, −0.8)", "note": "Four times the prompt's effect." },
    { "title": "Add to the unconditional guess", "expr": "ε̃ = (−0.1, 0.3) + (1.6, −0.8) = (1.5, −0.5)", "note": "Check w = 1: (−0.1, 0.3) + (0.4, −0.2) = (0.3, 0.1), the plain conditional guess." }
  ],
  "note": "Illustrative values authored for practice."
}
```

Latent diffusion runs the whole loop in a compressed space made by an encoder. A 512 × 512 colour picture has 786,432 numbers; a latent of 64 × 64 with 16 channels has 65,536, so the loop works on ==12 times fewer numbers== (the lesson's Stable Diffusion latent has 4 channels and gives 48).

### Practice problems

Cover each solution, try the problem on paper, then check. The numbers are new.

**Problem 1. Noising and the loss.** Use the linear schedule over 1,000 steps and t = 150. (a) Find ᾱ_150, the picture weight and the noise weight. (b) Noise the pixels x₀ = (0.5, −0.2, 0.9) with ε = (−1.0, 0.4, 0.6). (c) Find the SNR. (d) The network guessed ε̂ = (−0.8, 0.3, 0.5). What is the loss?

**Solution 1.**

- (a) ᾱ_150 = ==0.7883==; picture weight √0.7883 = ==0.8879==; noise weight √0.2117 = ==0.4601==.
- (b) Picture parts 0.4439, −0.1776, 0.7991; noise parts −0.4601, 0.1840, 0.2761. So x_150 = ==(−0.0161, 0.0065, 1.0751)==.
- (c) SNR = 0.7883 / 0.2117 = ==3.724==: the picture part is still nearly four times as strong as the noise.
- (d) Errors (−0.2, 0.1, 0.1), squares 0.04, 0.01, 0.01, mean ==0.02==. A small loss at a light noise level.

**Watch out:** the weights are square roots of ᾱ and 1 − ᾱ, not ᾱ and 1 − ᾱ themselves.

**Problem 2. Sampling.** At t = 400 a pixel is x_t = 1.1 and the network predicts ε̂ = 0.9. The schedule gives β_400 = 0.008048, α_400 = 0.99195, ᾱ_400 = 0.19515. (a) Do one DDPM step with z = −0.5. (b) Do a DDIM jump to s = 300 (ᾱ_300 = 0.3964). (c) A GAN makes a picture in one 0.04-second call. How long do 1,000 DDPM steps take, and 25 DDIM steps?

**Solution 2.**

- (a) √(1 − 0.19515) = 0.89714; coefficient 0.008048 / 0.89714 = 0.008971; 1.1 − 0.008971 × 0.9 = 1.09193; divided by √0.99195 = 0.99597 gives 1.09635; σ = √0.008048 = 0.08971, σz = −0.04486, so x_399 = ==1.0515==.
- (b) x̂₀ = (1.1 − 0.89714 × 0.9) / 0.44175 = 0.2926 / 0.44175 = ==0.6623==. Then x_300 = 0.62962 × 0.6623 + 0.77690 × 0.9 = 0.4170 + 0.6992 = ==1.1162==. No random draw: the same input always gives the same output.
- (c) 1,000 × 0.04 = ==40 seconds==; 25 × 0.04 = ==1 second==; DDIM is 40 times faster, and still 25 times slower than a GAN.

**Problem 3. Architecture and guidance.** (a) Give the time embedding at t = 20 for d = 4. (b) Count the parameters of a UNet block with C_in = 16, C_out = 32 and D = 64. (c) Apply guidance with w = 4 to ε(∅) = (−0.1, 0.3) and ε(c) = (0.3, 0.1). (d) Compare a 512 × 512 × 3 picture with a 64 × 64 × 16 latent.

**Solution 3.**

- (a) ==(0.9129, 0.4081, 0.1987, 0.9801)==: angles 20 and 0.2 radians.
- (b) 4,640 + 9,248 + 2,080 = ==15,968==.
- (c) ε̃ = (−0.1, 0.3) + 4 × (0.4, −0.2) = ==(1.5, −0.5)==.
- (d) 786,432 against 65,536: ==12 times fewer numbers== to denoise.

## Week 14: training LLMs {rail=Week 14}

{lede} Week 14 asked how a model that only continues text becomes an assistant. The answer is ==stages==: pre-training gives knowledge, supervised fine-tuning gives behaviour, preference training gives judgement, and LoRA makes the later stages cheap enough to run on one GPU.

### You must be able to

- Place pre-training, continued pre-training, SFT and preference tuning on the pipeline, say what each changes, and compare their cost with C ≈ 6ND.
- Compute the masked SFT loss from token probabilities, and explain why prompt tokens are left out.
- Compute a Bradley–Terry probability σ(r_w − r_l) and its loss −ln σ(r_w − r_l), and recognise ln 2 = 0.693 as a guess.
- Compute the KL-regularised optimum π* ∝ π_ref · e^(r/β), and say what a small β does ([[reward hacking|When the model finds answers that earn a high score from the reward model without being good, because the reward model is only an imperfect stand-in for what people want.]]).
- Compute a DPO loss from log-probability ratios.
- Count LoRA parameters r · (d_in + d_out), explain why B starts at zero, and estimate memory for full fine-tuning, LoRA and QLoRA.

### The stages

```viz
{
  "type": "sorter",
  "title": "Which stage fixes this?",
  "intro": "Place each situation on the pipeline: where would you act?",
  "bins": [ { "id": "pretrain", "label": "Pre-training (incl. continued)" }, { "id": "sft", "label": "Supervised fine-tuning" }, { "id": "pref", "label": "Preference training (RLHF or DPO)" } ],
  "cards": [
    { "name": "A model has never seen bridge-engineering vocabulary and must learn it from 2 billion tokens of manuals", "answer": "pretrain", "why": "New knowledge from raw text with the next-token objective: continued pre-training." },
    { "name": "The model answers a question with a story instead of the requested table", "answer": "sft", "why": "Following a requested format is behaviour taught with prompt–response examples." },
    { "name": "Two answers are both correct, but people prefer the shorter, politer one", "answer": "pref", "why": "A ranking between acceptable answers is a preference, learned from comparisons." },
    { "name": "Train on 8,000 prompt–response pairs, scoring only the response tokens", "answer": "sft", "why": "That is the masked SFT loss." },
    { "name": "Fit a reward model to human rankings with −ln σ(r_w − r_l)", "answer": "pref", "why": "That is step 2 of RLHF: learn a score from human rankings." },
    { "name": "Use 1.5 trillion tokens of general web text, loss on every token", "answer": "pretrain", "why": "Large-scale pre-training: loss on every position." },
    { "name": "Optimise π against the reward but keep KL(π ‖ π_ref) small", "answer": "pref", "why": "The leash that stops reward hacking." },
    { "name": "Adapt a base model to one task with adapters of 0.2% of the weights", "answer": "sft", "why": "LoRA is a way of doing fine-tuning cheaply, so the stage is still fine-tuning." }
  ]
}
```

==Cheaper later stages are the point.== With C ≈ 6ND and N = 3 × 10⁹, pre-training on 1.5 × 10¹² tokens costs 6 × 3×10⁹ × 1.5×10¹² = 2.7 × 10²² FLOPs, while an SFT run on 2 × 10⁷ tokens costs 3.6 × 10¹⁷, about 75,000 times less.

### Supervised fine-tuning: count only the response

```viz
{
  "type": "equation",
  "title": "Masked SFT loss for one example",
  "intro": "The prompt is context; only the response tokens are scored.",
  "formula": [ { "t": "L", "id": "l" }, " = −", { "t": "(1/|R|)", "id": "mean" }, " ", { "t": "Σ", "id": "sum" }, " ", { "t": "(t∈R)", "id": "set" }, " ln ", { "t": "P(y_t | prompt, y_<t)", "id": "prob" } ],
  "parts": [
    { "id": "l", "name": "L", "meaning": "The SFT loss for one example: the average surprise over the response tokens." },
    { "id": "mean", "name": "1/|R|", "meaning": "Divide by the number of response tokens, so long and short answers count equally." },
    { "id": "sum", "name": "Σ", "meaning": "Add over the scored tokens." },
    { "id": "set", "name": "t ∈ R", "meaning": "Only positions inside the response; prompt positions are masked out." },
    { "id": "prob", "name": "P(y_t | prompt, y_<t)", "meaning": "The probability the model gave to the correct next token, given the prompt and the response so far." }
  ],
  "steps": [
    { "title": "Probabilities of the 4 response tokens", "expr": "0.6, 0.3, 0.9, 0.5", "note": "Illustrative values." },
    { "title": "Surprise of each, −ln p", "expr": "0.511, 1.204, 0.105, 0.693", "note": "The unlikely token (0.3) costs most." },
    { "title": "Add and divide by 4", "expr": "(0.511 + 1.204 + 0.105 + 0.693) / 4 = 2.513 / 4 = 0.628", "note": "The masked loss." },
    { "title": "If 5 prompt tokens counted too", "expr": "p = 0.2, 0.1, 0.4, 0.25, 0.05 add 9.210.  (9.210 + 2.513) / 9 = 1.303", "note": "More than double, and dominated by words the model should never be asked to write." }
  ],
  "note": "All probabilities are made up for practice. The arithmetic is exact."
}
```

### The reward model and the Bradley–Terry loss

The reward model gives each answer a score. ==The chance that people prefer answer w over answer l is σ(r_w − r_l)==, and the loss is −ln of that chance. With r_w = 0.8 and r_l = −1.2 the gap is 2.0, the model says the chosen answer wins 88.1% of the time, and the loss is 0.127. If the scores were the other way round the loss would be 2.127.

```viz
{
  "type": "prefpair",
  "title": "Train a reward model on three comparisons",
  "intro": "Adjust the scores so that the chosen answer beats the rejected one in every pair. A loss of 0.693 means the model has no idea.",
  "pairs": [
    { "prompt": "Explain what a hairline crack is.", "chosen": "A hairline crack is a very thin surface crack, often under 0.3 mm wide. It should still be recorded and checked by an engineer.", "rejected": "Cracks are fine.", "chosenScore": 0.8, "rejectedScore": -1.2 },
    { "prompt": "Summarise this inspection note in one line.", "chosen": "Corrosion on the east railing; repaint and recheck in March.", "rejected": "Here is a very long retelling of the whole note, repeated three times.", "chosenScore": -1.2, "rejectedScore": 0.8 },
    { "prompt": "Is the beam safe?", "chosen": "I cannot decide that. A licensed engineer must assess the beam; here are the observations to pass on.", "rejected": "Yes, it looks safe.", "chosenScore": 0.3, "rejectedScore": 0.3 }
  ],
  "note": "Illustrative prompts, answers and scores authored for practice, not from a trained reward model.",
  "checks": [ { "pair": 0, "prob": 0.8808, "loss": 0.1269 }, { "pair": 1, "prob": 0.1192, "loss": 2.1269 }, { "pair": 2, "prob": 0.5, "loss": 0.6931 } ]
}
```

### The KL leash and its optimum

RLHF raises the expected reward but charges a price for drifting away from the reference model: J(π) = E[r] − β · KL(π ‖ π_ref). ==The best policy has a closed form: π*(y) = π_ref(y) · e^(r(y)/β) / Z==, where Z is the sum that makes the probabilities add to 1. A large β keeps π* close to π_ref; a small β lets it chase the reward.

```viz
{
  "type": "equation",
  "title": "The best policy for a three-answer toy, β = 1",
  "intro": "The reference model rarely writes answer 3, but answer 3 earns the highest reward.",
  "formula": [ { "t": "π*(y)", "id": "pistar" }, " = ", { "t": "π_ref(y)", "id": "ref" }, " · ", { "t": "exp( r(y) / β )", "id": "tilt" }, " / ", { "t": "Z", "id": "z" } ],
  "parts": [
    { "id": "pistar", "name": "π*(y)", "meaning": "The probability the best policy gives to answer y." },
    { "id": "ref", "name": "π_ref(y)", "meaning": "The probability the starting (reference) model gives to answer y." },
    { "id": "tilt", "name": "exp( r(y) / β )", "meaning": "A boost that grows with the reward and shrinks as β grows." },
    { "id": "z", "name": "Z", "meaning": "The sum of the boosted numbers over all answers, so that the probabilities add to 1." }
  ],
  "steps": [
    { "title": "Setting", "expr": "π_ref = (0.6, 0.3, 0.1),  r = (0, 1, 3),  β = 1", "note": "Illustrative numbers. At the reference, E[r] = 0.6 and J = 0.6." },
    { "title": "Boost each answer", "expr": "0.6 × e⁰ = 0.600,   0.3 × e¹ = 0.815,   0.1 × e³ = 2.009", "note": "Answer 3 overcomes a small starting probability." },
    { "title": "Normaliser", "expr": "Z = 0.600 + 0.815 + 2.009 = 3.424", "note": "The sum of the boosted values." },
    { "title": "Divide", "expr": "π* = (0.175, 0.238, 0.587)", "note": "Answer 3 grows from 0.10 to 0.59 but the others are not abandoned." },
    { "title": "The score", "expr": "E[r] = 0.238 + 3 × 0.587 = 1.998;  KL = 0.767;  J = 1.998 − 0.767 = 1.231", "note": "Twice the starting J of 0.6. With β = 0.5: π* = (0.014, 0.051, 0.935); with β = 5: (0.522, 0.319, 0.159), close to the reference." }
  ],
  "note": "Illustrative rewards and probabilities. Exact arithmetic."
}
```

```viz
{
  "type": "klreward",
  "title": "Reward against the leash: a note-taking helper",
  "intro": "Slide β from 5 down to 0.25 and watch the best policy move from the reference onto the top-reward answer.",
  "beta": 1,
  "answers": [
    { "text": "A plain, accurate note", "reward": 0, "ref": 0.6 },
    { "text": "An accurate note with a checklist reference", "reward": 1, "ref": 0.3 },
    { "text": "A confident note that sounds authoritative but skips the caveats", "reward": 3, "ref": 0.1 }
  ],
  "note": "Illustrative rewards and reference probabilities authored for practice. The third answer is what a flawed reward model might wrongly score very highly.",
  "checks": [ { "beta": 1, "opt": [0.175, 0.238, 0.587], "J": 1.231 }, { "beta": 5, "opt": [0.522, 0.319, 0.159] } ]
}
```

A tiny β is how ==reward hacking== happens: the policy finds the answer the reward model likes most, whether or not people would. Defences are a sensible β, a better reward model and humans checking the outputs.

### DPO: preferences without the reward model

Direct Preference Optimisation skips the reward model and the RL loop. It compares how much the policy has raised the chosen answer, and lowered the rejected one, relative to the reference.

```viz
{
  "type": "equation",
  "title": "A DPO loss with β = 0.2",
  "intro": "One chosen and one rejected answer. Probabilities of whole answers keep the arithmetic short.",
  "formula": [ { "t": "L", "id": "l" }, " = −ln σ( ", { "t": "β", "id": "beta" }, " · [ ", { "t": "ln π(y_w)/π_ref(y_w)", "id": "win" }, " − ", { "t": "ln π(y_l)/π_ref(y_l)", "id": "lose" }, " ] )" ],
  "parts": [
    { "id": "l", "name": "L", "meaning": "The DPO loss for one pair." },
    { "id": "beta", "name": "β", "meaning": "A scale. It plays the same role as the KL leash." },
    { "id": "win", "name": "ln π(y_w)/π_ref(y_w)", "meaning": "How much more likely the policy made the chosen answer than the reference did (in log terms)." },
    { "id": "lose", "name": "ln π(y_l)/π_ref(y_l)", "meaning": "The same for the rejected answer; training wants this to be negative." }
  ],
  "steps": [
    { "title": "The chosen answer", "expr": "π(y_w) = 0.35, π_ref(y_w) = 0.25: ln(0.35/0.25) = 0.336", "note": "The policy raised it." },
    { "title": "The rejected answer", "expr": "π(y_l) = 0.05, π_ref(y_l) = 0.10: ln(0.05/0.10) = −0.693", "note": "The policy lowered it." },
    { "title": "The bracket and the scale", "expr": "0.2 × (0.336 − (−0.693)) = 0.2 × 1.029 = 0.2059", "note": "Positive: the policy has moved in the right direction." },
    { "title": "Sigmoid and loss", "expr": "σ(0.2059) = 0.5513,  L = −ln 0.5513 = 0.595", "note": "Below ln 2 = 0.693, the loss of a policy that has not moved." }
  ],
  "note": "Illustrative probabilities authored for practice."
}
```

### LoRA and memory

LoRA freezes W and trains two thin matrices A (r × d_in) and B (d_out × r): h = Wx + (α/r) · B A x. ==B starts at zero, so the model begins exactly as the base model.== The trainable count per matrix is r · (d_in + d_out).

```viz
{
  "type": "equation",
  "title": "LoRA parameters for one matrix",
  "intro": "A 1,024 × 3,072 weight matrix with rank 16.",
  "formula": [ { "t": "params", "id": "p" }, " = ", { "t": "r", "id": "r" }, " · (", { "t": "d_in", "id": "din" }, " + ", { "t": "d_out", "id": "dout" }, ")" ],
  "parts": [
    { "id": "p", "name": "params", "meaning": "Trainable numbers in A and B together." },
    { "id": "r", "name": "r", "meaning": "The rank of the update: the number of rows of A and columns of B." },
    { "id": "din", "name": "d_in", "meaning": "Width of the input to the matrix." },
    { "id": "dout", "name": "d_out", "meaning": "Width of the output of the matrix." }
  ],
  "steps": [
    { "title": "Sizes of A and B", "expr": "A: 16 × 1,024 = 16,384;  B: 3,072 × 16 = 49,152", "note": "Two thin matrices." },
    { "title": "Total", "expr": "16 × (1,024 + 3,072) = 65,536", "note": "The same as 16,384 + 49,152." },
    { "title": "The full matrix", "expr": "1,024 × 3,072 = 3,145,728", "note": "Frozen during LoRA." },
    { "title": "The share", "expr": "65,536 / 3,145,728 = 2.08%", "note": "The saving is r(d_in + d_out)/(d_in × d_out); it improves as the matrix grows." }
  ],
  "note": "Illustrative sizes."
}
```

```viz
{
  "type": "lowrank",
  "title": "A pretend ΔW that needs only two directions",
  "intro": "Raise the rank from 1 to 2 and watch the error vanish: this update really has rank 2, which is the idea behind LoRA.",
  "matrix": [
    [0.8, -0.4, 0.6, 0.1],
    [0.4, -0.2, 0.3, 0.0],
    [-0.8, 0.4, -0.6, 0.2],
    [0.2, -0.1, 0.15, 0.0]
  ],
  "rank": 1,
  "max": 3,
  "note": "A made-up 4 × 4 update constructed to be low rank. Real weight updates are only approximately low rank.",
  "checks": [ { "sv": [1.639, 0.214] }, { "rank": 1, "err": 0.129 } ]
}
```

Memory is the other limit. Full fine-tuning with Adam in mixed precision needs about 16 bytes per parameter (2 weights, 2 gradients, 4 master copy, 8 for the Adam moments). LoRA keeps the base weights at 2 bytes and QLoRA at 0.5 bytes, and pays 14 bytes for each trained number.

```viz
{
  "type": "calc",
  "title": "Full fine-tuning, LoRA and QLoRA",
  "kind": "ft_memory"
}
```

Set N to 13 and compare: full fine-tuning needs 13 × 16 = 208 GB, QLoRA with 1% trainable needs 13 × (0.5 + 0.14) = about 8.3 GB (without activations).

### Practice problems

Cover each solution, try the problem on paper, then check. The numbers are new.

**Problem 1. SFT and cost.** A response has 4 tokens with probabilities 0.6, 0.3, 0.9 and 0.5; the prompt has 5 tokens with probabilities 0.2, 0.1, 0.4, 0.25 and 0.05. (a) The masked SFT loss. (b) The loss if the prompt were scored too. (c) A 3-billion-parameter model is pre-trained on 1.5 × 10¹² tokens and fine-tuned on 2 × 10⁷ tokens. By what factor does the compute differ?

**Solution 1.**

- (a) −ln p = 0.511, 1.204, 0.105, 0.693; sum 2.513; divided by 4 is ==0.628==.
- (b) The prompt adds 1.609 + 2.303 + 0.916 + 1.386 + 2.996 = 9.210; (9.210 + 2.513) / 9 = ==1.303==, more than double.
- (c) C_pre = 6 × 3×10⁹ × 1.5×10¹² = ==2.7 × 10²²==; C_ft = 6 × 3×10⁹ × 2×10⁷ = ==3.6 × 10¹⁷==; the ratio is D_pre / D_ft = ==75,000==.

**Problem 2. Preferences.** (a) A reward model gives r_w = 0.8 and r_l = −1.2. Find the probability and the loss, and the loss if the scores were swapped. (b) With π_ref = (0.6, 0.3, 0.1) and r = (0, 1, 3), find π* and J for β = 1. Which of β = 0.5 and β = 5 keeps the policy closer to the reference? (c) Find the DPO loss for β = 0.2, π(y_w) = 0.35, π_ref(y_w) = 0.25, π(y_l) = 0.05, π_ref(y_l) = 0.10.

**Solution 2.**

- (a) The gap is 2.0, σ(2.0) = ==0.881==, loss ==0.127==. Swapped: P = 0.119 and the loss is ==2.127==.
- (b) Boosted values 0.600, 0.815, 2.009, Z = 3.424, so π* = ==(0.175, 0.238, 0.587)==. E[r] = 1.998, KL = 0.767, J = ==1.231== (the reference scores 0.6). ==β = 5== keeps it closer: π* = (0.522, 0.319, 0.159) with KL 0.020, while β = 0.5 gives (0.014, 0.051, 0.935) with KL 1.946.
- (c) ln(0.35/0.25) = 0.336 and ln(0.05/0.10) = −0.693; 0.2 × 1.029 = 0.206; σ = 0.551; loss ==0.595==.

**Watch out:** a smaller β is a shorter leash, not a longer one; it lets the policy move far.

**Problem 3. LoRA and memory.** (a) A weight matrix is 1,024 × 3,072. Count the LoRA parameters for r = 16 and the share of the matrix. (b) A model with 24 layers and d = 2,048 puts rank-8 adapters on the query and value matrices (each 2,048 × 2,048). How many trainable parameters? (c) A 13-billion-parameter model: memory for full fine-tuning (16 bytes per parameter) and for QLoRA with 1% trainable (0.5 bytes per base weight plus 14 bytes per trained number). (d) How big is the adapter file in 16-bit?

**Solution 3.**

- (a) 16 × (1,024 + 3,072) = ==65,536==, which is 65,536 / 3,145,728 = ==2.08%==.
- (b) One matrix: 8 × (2,048 + 2,048) = 32,768. Two per layer over 24 layers: 2 × 24 × 32,768 = ==1,572,864==.
- (c) Full: 13 × 10⁹ × 16 = ==208 GB==. QLoRA: 13 × 10⁹ × (0.5 + 0.01 × 14) = 13 × 10⁹ × 0.64 = ==8.32 GB==, before activations.
- (d) 1% of 13 × 10⁹ = 1.3 × 10⁸ numbers × 2 bytes = ==260 MB==, small enough to share and swap per task.

## Revision plan and formula sheet

{lede} Part A is closed book, so you need the formulas in your head, not on a page. ==Revise by doing, not by rereading==: cover the solution, try the problem, then check.

### The formula sheet

Every formula of Weeks 11 to 14 on one page. For each, make sure you can say what every symbol means and do the example without looking.

| Formula | What it is | Example |
|---|---|---|
| patches = (side ÷ patch)²; tokens = patches + 1 | Patch tokens of a ViT (plus the CLS token) | 320 and 32: 10² = 100 patches, 101 tokens |
| numbers per patch = P · P · C;  embedding parameters = P · P · C · d + d | Flatten and project one patch | 32 · 32 · 3 = 3,072; × 512 + 512 = 1,573,376 |
| attention scores per head = n² | Cost of self-attention | 101² = 10,201 (pixel tokens: 102,400² = 10,485,760,000) |
| cos(a, b) = a·b / (‖a‖ ‖b‖);  logits = scale · cos;  p = softmax(logits) | CLIP zero-shot choice | cos 0.30, 0.10, −0.05 at scale 20: logits 6, 2, −1 give p = 0.981, 0.018, 0.001 |
| loss = mean of −ln(diagonal probability) | CLIP contrastive loss | p = 0.981: −ln 0.981 = 0.019 |
| D*(x) = p_data(x) / (p_data(x) + p_g(x)) | Best discriminator | 0.6 / (0.6 + 0.2) = 0.75 |
| V = ln D(x) + ln(1 − D(G(z))); at equilibrium D = 1/2 | GAN value function | V = −ln 4 = −1.386 |
| saturating gradient −D; non-saturating gradient D − 1 (on the logit) | Generator gradients when D(fake) is small | D = 0.02: −0.02 against −0.98 |
| n_out = (n − 1) · s − 2p + k + op | Transpose convolution output side | n = 8, k = 4, s = 2, p = 1: 16;  n = 5, k = 3, s = 2: 11 |
| parameters = k² · C_in · C_out + C_out | Convolution or transpose convolution layer | 5 · 5 · 64 · 32 + 32 = 51,232 |
| α_t = 1 − β_t;  ᾱ_t = α_1 · α_2 ⋯ α_t | Noise schedule | ᾱ_200 = 0.6590, ᾱ_600 = 0.0259 |
| x_t = √ᾱ_t · x₀ + √(1 − ᾱ_t) · ε | Closed-form noising | t = 200, x₀ = 0.6, ε = −0.8: 0.8118 · 0.6 + 0.5839 · (−0.8) = 0.0200 |
| SNR_t = ᾱ_t / (1 − ᾱ_t) | Signal-to-noise ratio | 0.6590 / 0.3410 = 1.933 |
| L = mean (ε − ε̂)² | Noise-prediction loss | (1.0, −0.6) against (0.5, −0.4): 0.145 |
| x̂₀ = (x_t − √(1 − ᾱ_t) · ε̂) / √ᾱ_t | Clean-picture estimate | t = 600: −1.2837 |
| x_(t−1) = (x_t − β_t/√(1 − ᾱ_t) · ε̂) / √α_t + σ_t · z | DDPM reverse step | x = −0.7, ε̂ = −0.5, z = 0.4: −0.6542 |
| x_s = √ᾱ_s · x̂₀ + √(1 − ᾱ_s) · ε̂ | DDIM jump (no fresh noise) | 600 to 500: −0.8398 |
| time = steps × time per call | Sampler cost | 1,000 · 0.04 = 40 s; 25 · 0.04 = 1 s |
| emb(t) = (sin t·f, cos t·f) for several frequencies f | Sinusoidal time embedding | t = 20, d = 4: (0.9129, 0.4081, 0.1987, 0.9801) |
| P_block = 9 · C_in · C_out + C_out + 9 · C_out² + C_out + D · C_out + C_out | UNet block parameters | 16, 32, 64: 15,968 |
| ε̃ = ε(∅) + w · (ε(c) − ε(∅)) | Classifier-free guidance | w = 4: (1.5, −0.5) |
| ratio = (H · W · C) / (h · w · c) | Latent diffusion saving | 786,432 / 65,536 = 12 |
| C ≈ 6 · N · D | Training compute | pre-train 2.7 × 10²², SFT 3.6 × 10¹⁷: 75,000 times |
| L = −(1/\|R\|) Σ over response of ln p | Masked SFT loss | p = 0.6, 0.3, 0.9, 0.5: 0.628 |
| P(w beats l) = σ(r_w − r_l);  L = −ln σ(r_w − r_l) | Bradley–Terry reward model | gap 2.0: 0.881 and 0.127; no idea: ln 2 = 0.693 |
| J = E[r] − β · KL(π ‖ π_ref) | KL-regularised objective | π* score 1.998 − 0.767 = 1.231 |
| π* = π_ref · e^(r/β) / Z | Best policy | (0.6, 0.3, 0.1), r = (0, 1, 3), β = 1: (0.175, 0.238, 0.587) |
| L_DPO = −ln σ(β · [ln π(y_w)/π_ref(y_w) − ln π(y_l)/π_ref(y_l)]) | Direct Preference Optimisation | β = 0.2: 0.595 |
| h = Wx + (α/r) · B A x;  params = r · (d_in + d_out) | LoRA | 16 · (1,024 + 3,072) = 65,536 |
| memory ≈ 16 bytes per parameter (full);  2 + 14 · share (LoRA);  0.5 + 14 · share (QLoRA) | Fine-tuning memory | 13 billion: 208 GB; QLoRA at 1%: 8.32 GB |

Notice the families: counts and sizes (Weeks 11 and 12), products and square roots of a schedule (Week 13), and log-probabilities, sigmoids and byte counts (Week 14). The same few ideas return: a softmax or sigmoid turns scores into probabilities, a loss is −ln p or a squared error, and ==everything you count is a size you can check by hand==.

### A four-day revision plan

Each day takes about 90 minutes. Do the steps in order; the minutes are a guide.

1. **Day 1: Weeks 11 and 12 (about 90 minutes).** Reread the two review sections (20 minutes). Try the six practice problems with the solutions covered (30 minutes). Redo the formative quizzes of Weeks 11 and 12 (20 minutes). Open your two lab notebooks and rerun their check cells without looking at your old answers (20 minutes).
2. **Day 2: Week 13 (about 90 minutes).** Reread the Week 13 review (20 minutes). Rebuild the ᾱ table, one noising step, one DDPM step and one DDIM step on paper with new numbers (30 minutes). Do the three practice problems (15 minutes). Redo the Week 13 quiz (10 minutes). Rerun your Week 13 lab notebook and check every size by hand (15 minutes).
3. **Day 3: Week 14 (about 90 minutes).** Reread the Week 14 review (20 minutes). Redo the masked loss, the Bradley–Terry loss, the KL optimum and the LoRA and memory counts on paper (30 minutes). Do the three practice problems (15 minutes). Redo the Week 14 quiz (10 minutes). Write one paragraph that places pre-training, SFT, RLHF, DPO and LoRA on one diagram (15 minutes).
4. **Day 4: everything together (about 90 to 120 minutes).** Work through [Mock Exam Prep](/mock-exam) (45 minutes). Write the whole formula sheet from memory, then compare it with the table above (15 minutes). Time yourself on one Part B style task from a lab notebook: 25 minutes, no looking back at your old answers (30 minutes). Draft the first lines of your Capstone proposal. Then stop, and rest.

A timed run of the real written exam is not possible before the real one. The formative quizzes and Mock Exam Prep are the best substitute: work through them as if it were Part A, without notes, and score yourself. Earlier weeks are the foundation of Weeks 11 to 14, so skim them if a symbol feels unfamiliar.

### Exam technique

- **Read the whole question first.** Note what is given, what is asked, and the units or shape of the answer.
- **Check sizes and shapes.** A probability lies between 0 and 1 and a softmax row adds up to 1; squares of the two noising weights add up to 1; the output of a transpose convolution is larger than its input; LoRA trainable parameters are far fewer than the frozen matrix.
- **Do the arithmetic in a sensible order.** Look up ᾱ first and take the square roots after; write exponents before multiplying (for example 6 · 3×10⁹ · 1.5×10¹²).
- **In Part B, work in code but sanity-check by hand.** Ask whether the sign and size of your answer make sense before you move on.
- **Watch the clock.** Part A: about 90 seconds per question. Part B: keep to the time boxes. A task left at 70 percent is worth more than a task never started.
- **When you are stuck,** flag the question (Part A) or write down what you know and move on (Part B). Never leave an answer blank; a reasoned guess costs nothing.
- **Eliminate before you choose.** In Part A, cross out options that break a rule you know (a probability above 1, a pixel weight that is not a square root, an output smaller than the input of a transpose convolution).

## Key takeaways

- ==A ViT turns an image into (side ÷ patch)² patch tokens plus a CLS token, and CLIP picks the text closest to the image.== The attention table costs n² scores, and zero-shot choice is softmax(scale · cosine).
- ==A GAN is a game, and transpose convolution grows pictures.== D* = p_data / (p_data + p_g), equilibrium gives V = −ln 4, a small D(fake) starves the saturating loss (gradient −0.02 against −0.98), and n_out = (n − 1)s − 2p + k + op.
- ==Diffusion destroys by a fixed recipe and learns to predict the noise.== x_t = √ᾱ_t · x₀ + √(1 − ᾱ_t) · ε (ᾱ_200 = 0.6590), sampling costs one network call per step (1,000 for DDPM, 25 for DDIM), the UNet gets t through a time embedding, guidance uses ε̃ = ε(∅) + w(ε(c) − ε(∅)), and a latent space shrinks the work (12 times in this page's example).
- ==Training an LLM is a ladder of stages, each cheaper than the last.== SFT scores only response tokens, the reward model learns −ln σ(r_w − r_l), the KL leash π* ∝ π_ref · e^(r/β) holds back reward hacking, and DPO does the same job without a reward model.
- ==LoRA and QLoRA make fine-tuning fit.== r · (d_in + d_out) numbers per matrix, B starting at zero, 16 bytes per parameter for full fine-tuning (208 GB for 13 billion) against about 8.3 GB for QLoRA at 1% trainable.

## Capstone proposal: the Structural Inspection Assistant {rail=Capstone}

{lede} The Capstone is a team project: a ==Structural Inspection Assistant that triages inspection photos and drafts a summary for an engineer to review==. The Capstone opened in Week 13, and this week your team writes its proposal. Full details are on the [Capstone](/capstone) page.

The pipeline is: photo in, classify (crack, spalling, corrosion or none), find checklist items with RAG, then draft a summary for the engineer. You will use a vision model (ViT or CLIP, from Week 11), retrieval over checklists (Week 16) and NeMo Guardrails (Week 17). The ==AI-use rule is fixed: a licensed engineer makes every pass/fail decision; the assistant never signs off on safety.==

```viz
{
  "type": "pipeline",
  "title": "The Structural Inspection Assistant",
  "intro": "Step through one photo.",
  "nodes": [
    { "id": "photo", "label": "Photo", "sub": "inspection image" },
    { "id": "classify", "label": "Classify", "sub": "crack / spalling / corrosion / none" },
    { "id": "retrieve", "label": "Find checklist items", "sub": "RAG" },
    { "id": "draft", "label": "Draft summary", "sub": "LLM + guardrails" },
    { "id": "engineer", "label": "Engineer decides", "sub": "human review" }
  ],
  "steps": [
    { "title": "A photo comes in", "body": "A team member uploads an inspection photo. The system never decides anything at this point; it only receives the image.", "active": ["photo"], "example": "photo_017.jpg" },
    { "title": "Classify the damage", "body": "A vision model (a ViT or CLIP zero-shot) labels the photo as crack, spalling, corrosion or none, with a confidence score.", "active": ["classify"], "example": "Label: crack (confidence 0.81, illustrative)" },
    { "title": "Find checklist items", "body": "The label and a short description become a search query over the team's checklist passages. The best matches are retrieved.", "active": ["retrieve"], "example": "FICTIONAL SAMPLE TEXT: 'Item C-3: record where the crack is and how far it runs.'" },
    { "title": "Draft a summary", "body": "A language model writes a short summary from the label and the retrieved items. Guardrails check the reply, for example blocking any wording that says the structure is safe.", "active": ["draft"], "example": "Draft for review: crack observed; see checklist item C-3." },
    { "title": "The engineer decides", "body": "A licensed engineer reads the draft and makes the pass or fail decision. The assistant never signs off on safety.", "active": ["engineer"], "example": "Decision recorded by the engineer, not by the model." }
  ],
  "note": "Illustrative flow. The example checklist text is fictional sample text, not real engineering guidance."
}
```

Notice that the engineer is the last node: every other box helps them, none replaces them.

### How the Capstone is marked

| Criterion | Weight |
|---|---|
| Problem and proposal | 15% |
| Technical implementation | 40% |
| Responsible AI | 15% |
| Presentation | 20% |
| Documentation | 10% |

The Capstone is 20% of the final grade according to the demo schedule. Roles in each team: ==Team lead, Technical lead and Responsible-AI lead==. The proposal is where you earn the first row, "Problem and proposal", and set up every other row.

### Proposal checklist

Your instructor announces the proposal format, length and deadline. A strong proposal answers these questions:

- [ ] **Problem:** who has the problem, and what slows them down today?
- [ ] **Users and the human in the loop:** who is the engineer, and what exactly do they review?
- [ ] **Pipeline:** which model does each step, and what goes in and out of it?
- [ ] **Data plan:** where will your test photos and checklist text come from, and are they allowed to be used? Mark any sample text as fictional.
- [ ] **Evaluation:** how will you know the classifier and the retrieval work (for example accuracy on photos you labelled, and checking that the right checklist item is found)?
- [ ] **Responsible AI:** what can go wrong (a wrong label, an overconfident summary), and which guardrail or human check prevents each?
- [ ] **Roles and plan:** who is Team lead, Technical lead and Responsible-AI lead, and what is done by Week 19?

### A worked example for a different project

Here is what a good proposal looks like, for a project that is *not* yours: a leaf-health assistant for a school garden. Everything below is FICTIONAL, including the team, the numbers and the photos; it shows the level of detail, not an answer to copy.

> **Project:** Garden Leaf Check (FICTIONAL EXAMPLE)
>
> **Problem.** The school garden club has 40 raised beds and one teacher, Ms. Reyes, who inspects them on Fridays. Looking through the week's photos takes about 90 minutes, and early signs of leaf disease are missed. We want a tool that sorts the photos and drafts notes so that she reads the worrying ones first.
>
> **Users and the human in the loop.** Ms. Reyes reads every draft note and decides whether a bed needs treatment. The tool never recommends a spray, a dose or a deadline. It only says what it sees and which care-guide passage applies.
>
> **Pipeline.** (1) A student uploads a leaf photo. (2) A CLIP zero-shot classifier (Week 11) chooses between "healthy leaf", "leaf with blight spots", "leaf with insect damage" and "leaf with yellowing", using one prompt per class and the highest cosine similarity. (3) The label becomes a query over 25 passages of the garden's own care guide (RAG, Week 16). (4) A language model drafts three sentences for the teacher; NeMo Guardrails (Week 17) block any wording that gives a dose or says a plant is "safe" or "cured".
>
> **Data plan.** 240 photos taken by club members in the garden over two weeks, with permission; no faces or other people. Each photo is labelled by two students and checked by Ms. Reyes. Split: 160 for choosing prompts, 40 for tuning, 40 for the final test that is touched once. The care guide is the club's own text. (FICTIONAL numbers.)
>
> **Evaluation.** On the 40 test photos the target is at least 80% accuracy and at least 80% recall for blight (we would rather raise a false alarm than miss a case). Suppose the tool gets 33 of 40 right: accuracy 0.825 with a standard error of √(0.825 × 0.175 / 40) = 0.060, so the true accuracy could well be between 0.71 and 0.94; we say so in the report instead of claiming "82.5%". Retrieval is judged by whether the right care-guide passage is in the top 3 for 20 test notes.
>
> **Responsible AI.** *Wrong label:* the teacher always sees the photo next to the note. *Overconfident note:* guardrails block certainty words, and every note says "unconfirmed". *Hard cases:* photos in poor light are flagged for a manual look when the top two similarities differ by less than 0.05. *Privacy:* photos are stored only on the school account.
>
> **Roles and plan.** Team lead: A. Technical lead: B. Responsible-AI lead: C. Week 16: retrieval working; Week 17: guardrails; Week 18: evaluation on the test set; Week 19: build and rehearse; Week 20: demo.

Notice what makes the example strong. It names a real decision-maker, it keeps the human decision out of the model, it has numbers that can be checked, it admits uncertainty with a standard error, and each Responsible-AI risk has one concrete check.

### The proposal template

Copy this skeleton for your team's proposal and replace each line with your own content. Mark any sample text or numbers that you invented as fictional.

```text
Project name:
Team: Team lead / Technical lead / Responsible-AI lead

1. Problem and users (who is slowed down today, and by how much?)
2. The human in the loop (which licensed engineer decides what?)
3. Pipeline (photo -> classifier -> retrieval -> draft -> engineer)
   - Model for each step, and its input and output
4. Data plan (source, permission, labelling, train / validation / test split)
5. Evaluation (metric, target, test size, standard error)
6. Responsible AI (risk -> guardrail or human check, one row each)
7. Plan to Week 19 (who does what, by which week)
8. Open questions for the instructor
```

<!-- LAB-START -->
## Lab: Pre-final practical exam (Part B) {rail=Practical}
This is Part B of the Pre-final exam: a hands-on test in a notebook, worth 60 of the 100 points, with 120 minutes on the clock. Part A, the written exam, is the timed written exam on this page. In Part B you do not read about Weeks 11 to 14, you use them: you count patch tokens, cut an image into patches, compare images and sentences as CLIP does, compute transpose-convolution sizes and GAN losses, add noise and remove it again in a diffusion model, mask an SFT loss, build a LoRA layer, and explain what you see. ==The objective is to show that you can turn the ideas of Weeks 11 to 14 into working code and correct numbers, and explain your results in your own words.== This page holds the whole exam in the order of the notebook, Tasks 1 to 5. The whole exam is one notebook, so you work through the tasks in order.

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
| 1 | Multimodal AI: patches, ViT and CLIP (Week 11) | 12 | 20 min |
| 2 | Image generation: GANs and transpose convolution (Week 12) | 14 | 25 min |
| 3 | Diffusion models (Week 13) | 16 | 30 min |
| 4 | Training LLMs: SFT, preference learning and LoRA (Week 14) | 12 | 30 min |
| 5 | Synthesis with the capstone (Weeks 11 to 14) | 6 | 15 min |
| Total | | 60 | 120 min |

The setup below is done before the clock starts and is not part of the 120 minutes.

### Setup

1. Choose where you work. **On the school's DGX Spark:** log in to the portal, start JupyterLab, create a folder `prefinal` and a notebook `exam-b.ipynb`. **At home or on a laptop:** use the "Open in Colab" card at the top of this lab, choose **File → Save a copy in Drive**, and work in your copy. The Colab notebook contains the same tasks, the same labels and the same points as this page, plus automatic self-check cells; the "Download .ipynb" button gives you that same file if you prefer to upload it to JupyterLab on the Spark.
2. Do not install packages and do not download anything: PyTorch, NumPy and Matplotlib are already installed, and every data set is generated by the code of the task. Do not use `!pip install`, Google Drive code or any `google.colab` code in the Spark notebook.
3. If you work from this page, not from the notebook, write each starter block below in its own cell. The notebook's self-check cells are not on this page, so you check your numbers yourself and your instructor marks them; the marking is the same.
4. Paste the block below into the first cell and run it. It fixes the seed 42, defines the `ANSWERS` dictionary that holds your written answers, defines the exam `check(name, ok)` (which prints a tick or a cross and nothing else), and defines three small helpers: `cosine`, `softmax_np` and `param_count`. The last part of the block defines the exam helpers `same`, `same_array`, `same_word` and `n_words` that the notebook's self-checks use.

```python
import os, gc, math, random, re, time, json, warnings
from collections import Counter, defaultdict
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

ANSWERS = {}   # keep your final answers here; the last cell prints them so you can copy them into your submission

def check(name, ok, hint=""):
    """Exam version: prints a tick or a cross and never a hint."""
    print(("✓ " if ok else "✗ ") + name)

def cosine(a, b):
    """Cosine similarity of two vectors."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def softmax_np(z, axis=-1):
    """Softmax of a NumPy array along an axis (numerically stable)."""
    z = np.asarray(z, dtype=float); e = np.exp(z - z.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def param_count(model):
    """Number of trainable parameters of a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

import hashlib
H1 = "w15-b1"   # a label that is mixed into every fingerprint
def _fp(v, nd):
    return hashlib.sha256(f"{H1}|{nd}|{round(float(v), nd) + 0.0:.{nd}f}".encode()).hexdigest()[:12]
def same(x, fingerprint, nd=3):
    """True if x, rounded to nd decimals, is the number the exam expects. The notebook stores only a fingerprint, so it cannot tell you the number."""
    try:
        x = float(x)
    except Exception:
        return False
    if not math.isfinite(x):
        return False
    step = 10.0 ** -nd
    return any(_fp(x + d * 0.4 * step, nd) == fingerprint for d in (-1, 0, 1))
def same_array(a, fingerprint, nd=3, shape=None):
    """Like same(), for a whole list or array: it compares one number that mixes all the entries (and, if given, the shape)."""
    try:
        a = np.asarray(a, dtype=float)
        if shape is not None and a.shape != tuple(shape):
            return False
        return same(float(np.sum(a * np.cos(np.arange(a.size)).reshape(a.shape))), fingerprint, nd)
    except Exception:
        return False
def same_word(s, fingerprint):
    return isinstance(s, str) and hashlib.sha256((H1 + "|" + s.strip().lower()).encode()).hexdigest()[:12] == fingerprint
def n_words(x):
    """How many words are in x (0 if x is not text)."""
    return len(x.split()) if isinstance(x, str) else 0
assert same(0.5004, _fp(0.5, 3)) and not same(0.52, _fp(0.5, 3))   # the helper itself works
print("Exam helpers ready.")
```

### What to submit

Submit the notebook **with all outputs visible**: run every cell from the top just before you submit. Add the output of the last cell of the notebook, which prints your `ANSWERS` dictionary, so that your written answers are easy to read. Written answers go in `ANSWERS["B1.6"]`, `ANSWERS["B2.4"]` and so on, or in a markdown cell that begins "Your answer:". Submit the notebook the way your instructor asks. Each sub-question below shows its points; a correct number earns the points, and a wrong number with correct working still earns part of them.

### Task 1. Multimodal AI: patches, ViT and CLIP (12 points, 20 minutes)

*Exercise B1.1:* **Counting patch tokens (3 points).** A Vision Transformer cuts an image into non-overlapping square patches of side `p`; if the image side is not a multiple of `p`, the leftover pixels at the right and bottom edges are dropped. Write `n_patches(h, w, p)`, which returns the number of patches (1 point). Store in `N_list` the patch counts for the four cases in `CASES` (an image of height `h` and width `w`, patch side `p`) (1 point), and in `kept_share` the share of the pixels of the 200 × 200 image that the patches still cover (1 point).

```python
CASES = [(224, 224, 16), (224, 160, 16), (200, 200, 16), (384, 384, 32)]   # (h, w, p)

def n_patches(h, w, p):
    """Number of non-overlapping p × p patches that fit in an h × w image (leftover edge pixels are dropped)."""
    # YOUR CODE HERE
    return None

N_list = None      # YOUR CODE HERE (a list of four whole numbers)
kept_share = None  # YOUR CODE HERE (a number between 0 and 1)
```

*Exercise B1.2:* **Cutting an image into patches (2 points).** The array `img` is a tiny 16 × 16 image with 3 colour channels; the number in each cell is its own index, so you can follow where every number goes. Write `patchify(img, p)`, which returns an array of shape `(N, p·p·3)`: one flattened patch per row, in reading order (left to right, then top to bottom), using only `reshape` and `transpose`. Cut the image with `p = 4`. Store in `patch_shape` the shape of the result as a tuple (0.5 points), in `patch5_sum` the sum of the numbers of patch number 5 (counting from 0) (1 point), and in `lost_nothing` `True` or `False` for "patchifying changed no number, it only rearranged them" (0.5 points).

```python
img = np.arange(16 * 16 * 3).reshape(16, 16, 3)   # height 16, width 16, 3 channels; the value is 48·row + 3·col + channel

def patchify(img, p):
    """Cut an (H, W, C) image into non-overlapping p × p patches, row by row; return an array of shape (N, p*p*C)."""
    # YOUR CODE HERE
    return None

patches = patchify(img, 4)
patch_shape = None   # YOUR CODE HERE (a tuple)
patch5_sum = None    # YOUR CODE HERE
lost_nothing = None  # YOUR CODE HERE (True or False)
```

*Exercise B1.3:* **Sequence length, embedding and cost (2 points).** A ViT with hidden size `d = 192` reads a 224 × 224 colour image with patches of side 16. Store in `embed_params` the number of parameters of the patch embedding (one linear layer from a flattened patch to `d` numbers, with a bias) (0.5 points), in `seq_len` the length of the token sequence including one class token (0.5 points), in `scores_per_head` the number of attention scores one head computes in one layer for this image (0.5 points), and in `ratio_scores` how many times more attention scores one head needs if the patch side is halved to 8 (0.5 points).

```python
H_IMG, P_SIDE, D_HID = 224, 16, 192
embed_params = None      # YOUR CODE HERE
seq_len = None           # YOUR CODE HERE
scores_per_head = None   # YOUR CODE HERE
ratio_scores = None      # YOUR CODE HERE
```

*Exercise B1.4:* **CLIP-style zero-shot choice (2.5 points).** A bridge inspector photographs a part of a pier. A CLIP-like model has turned the photo into the vector `img_vec` and four sentences into the vectors in `TEXT_VECS` (the numbers are invented for the exam). Store in `cos_sims` the four cosine similarities between the image and the sentences, in the order of the dictionary (0.75 points); in `probs` the zero-shot probabilities, which are the softmax of `LOGIT_SCALE × cosine` (0.75 points); in `best` the sentence with the highest probability (0.5 points); and in `p_best` its probability (0.5 points). You may use `cosine` and `softmax_np`.

```python
img_vec = np.array([0.7, 0.5, 0.2, 0.4])
TEXT_VECS = {"a photo of a cracked pier": [0.8, 0.4, 0.1, 0.3],
             "a photo of a rusted beam": [0.2, 0.9, 0.3, 0.1],
             "a photo of a clean concrete wall": [0.1, 0.1, 0.9, 0.5],
             "a photo of a loose bolt": [0.6, 0.2, 0.2, 0.8]}
LOGIT_SCALE = 10

cos_sims = None   # YOUR CODE HERE (four numbers)
probs = None      # YOUR CODE HERE (four numbers that add up to 1)
best = None       # YOUR CODE HERE (a sentence)
p_best = None     # YOUR CODE HERE
```

*Exercise B1.5:* **The contrastive loss (1.5 points).** A batch has 3 image–text pairs. The matrix `S` holds the cosine similarity of image `i` (row) and text `j` (column); the true pairs are on the diagonal. With a logit scale of 10, store in `loss_i2t` the average cross-entropy when each **image** (row) must pick its own text (0.5 points), in `loss_t2i` the same when each **text** (column) must pick its own image (0.5 points), and in `loss_clip` the symmetric loss, which is the mean of the two (0.5 points).

```python
S = np.array([[0.62, 0.31, 0.10],
              [0.28, 0.55, 0.34],
              [0.05, 0.40, 0.58]])
SCALE = 10

loss_i2t = None   # YOUR CODE HERE
loss_t2i = None   # YOUR CODE HERE
loss_clip = None  # YOUR CODE HERE
```

*Exercise B1.6:* **Choosing how to fuse (1 point).** A team wants a language model to answer an inspector's questions about a photo ("Is this crack wider than the card?"). One option is a projector that turns the photo into prefix tokens: the 224 × 224 photo is cut into patches of side 14 and each patch becomes one token in front of the text. The model's context window is 2,048 tokens. The other option is cross-attention layers, and a third is to use only a CLIP-style shared space. In `ANSWERS["B1.6"]` write 3 to 4 sentences: how many image tokens the projector option adds and what share of the context they use, which of the three options you would choose for this question, and one risk that a person must still check.

```python
ANSWERS["B1.6"] = ""   # YOUR ANSWER: 3 to 4 sentences
```

### Task 2. Image generation: GANs and transpose convolution (14 points, 25 minutes)

*Exercise B2.1:* **Transpose convolution sizes (2 points).** The output side of a transpose convolution is `(n − 1)·stride − 2·pad + k + out_pad`. Write `convT_out(n, k, stride, pad, out_pad)` and store in `sizes` the output sides for the five cases in `CONFIGS` (each is `(n, k, stride, pad, out_pad)`) (1 point). Then build a real `nn.ConvTranspose2d(1, 1, k, stride, pad, output_padding=out_pad)` for each case, feed it a tensor with an `n × n` input and store in `torch_sizes` the sides it really produces (0.5 points), and in `all_match` `True` or `False` for "my formula and PyTorch agree in all five cases" (0.5 points).

```python
CONFIGS = [(4, 4, 2, 1, 0), (7, 3, 2, 1, 0), (7, 3, 2, 1, 1), (5, 5, 3, 2, 1), (10, 2, 2, 0, 0)]

def convT_out(n, k, stride, pad, out_pad=0):
    """Output side of a transpose convolution."""
    # YOUR CODE HERE
    return None

sizes = None        # YOUR CODE HERE (a list of five whole numbers)
torch_sizes = None  # YOUR CODE HERE (a list of five whole numbers)
all_match = None    # YOUR CODE HERE (True or False)
```

*Exercise B2.2:* **A small DCGAN generator (3 points).** A generator takes a noise vector of 64 numbers, applies one fully connected layer (with bias) that gives 4 × 4 × 256 numbers, reshapes them to a 4 × 4 grid with 256 channels, and applies the three transpose convolutions in `GEN_LAYERS` (each is `(in_channels, out_channels, k, stride, pad)`, with bias). Store in `sides` the side of the grid after the fully connected layer and after each transpose convolution (four numbers) (1 point), in `layer_params` the number of parameters of the four layers in order (1 point), and in `total_params` their sum (0.5 points). Then build the same layers in PyTorch (a `nn.Linear` and three `nn.ConvTranspose2d`) and store in `torch_total` the count that `param_count` gives (0.5 points).

```python
Z_DIM, FC_SIDE, FC_CH = 64, 4, 256
GEN_LAYERS = [(256, 128, 4, 2, 1), (128, 64, 4, 2, 1), (64, 1, 4, 2, 1)]

sides = None         # YOUR CODE HERE (four numbers)
layer_params = None  # YOUR CODE HERE (four numbers)
total_params = None  # YOUR CODE HERE
torch_total = None   # YOUR CODE HERE
```

*Exercise B2.3:* **The optimal discriminator (2.5 points).** At one picture `x` the real data have density `p_data(x) = 0.6` and the generator has density `p_g(x) = 0.2`. Store in `d_star` the value of the best possible discriminator there (0.5 points). Store in `d_star_eq` its value at a picture where `p_g(x) = p_data(x)` (0.5 points) and in `v_eq` the value of the game `V(D*, G) = E[ln D(x)] + E[ln(1 − D(G(z)))]` when the generator is perfect and `D*` is used everywhere (0.5 points). Finally, for a batch with `D(x) = [0.9, 0.8, 0.7]` on three real pictures and `D(G(z)) = [0.2, 0.3, 0.1]` on three fakes, store in `d_loss` the discriminator loss `−mean(ln D(x)) − mean(ln(1 − D(G(z))))` (1 point).

```python
D_real = np.array([0.9, 0.8, 0.7])
D_fake = np.array([0.2, 0.3, 0.1])

d_star = None     # YOUR CODE HERE
d_star_eq = None  # YOUR CODE HERE
v_eq = None       # YOUR CODE HERE
d_loss = None     # YOUR CODE HERE
```

*Exercise B2.4:* **Saturating and non-saturating generator loss (2 points).** The discriminator's last layer gives a logit `a`, and `D = sigmoid(a)`. A generator that is losing badly has `D(G(z)) = 0.01`, so set `a = ln(0.01 / 0.99)`. Use autograd (`requires_grad=True` and `.backward()`) to compute the gradient of the original generator loss `ln(1 − D)` with respect to `a` and store it in `grad_sat` (0.5 points), then the gradient of the non-saturating loss `−ln D` with respect to `a` and store it in `grad_ns` (0.5 points). Store in `ratio_grad` how many times larger the non-saturating gradient is in size (0.5 points). In `ANSWERS["B2.4"]` write one or two sentences on what this means for a generator at the start of training (0.5 points).

```python
a0 = math.log(0.01 / 0.99)
grad_sat = None   # YOUR CODE HERE
grad_ns = None    # YOUR CODE HERE
ratio_grad = None # YOUR CODE HERE
ANSWERS["B2.4"] = ""   # YOUR ANSWER: one or two sentences
```

*Exercise B2.5:* **A tiny GAN on a line (2 points).** The real data are numbers from a bell curve with mean 4 and standard deviation 1. The loop below trains a generator and a discriminator for 800 steps; the discriminator step is done for you. Write the **generator step** with the **non-saturating loss**: score the fakes with the discriminator, compute the loss `−mean(ln D(G(z)))` (use `F.logsigmoid` on the logit, or `F.binary_cross_entropy_with_logits` with the label 1), and update only the generator (1 point). Then store in `gen_mean` and `gen_std` the mean and the standard deviation of 2,000 numbers made by the trained generator (0.5 points each).

```python
def train_gan_1d(steps=800, seed=42):
    torch.manual_seed(seed)
    G = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
    D = nn.Sequential(nn.Linear(1, 16), nn.LeakyReLU(0.2), nn.Linear(16, 1))
    optG = torch.optim.Adam(G.parameters(), lr=2e-3)
    optD = torch.optim.Adam(D.parameters(), lr=2e-3)
    for step in range(steps):
        real = 4 + torch.randn(64, 1)
        z = torch.randn(64, 1)
        fake = G(z)
        # discriminator step (done for you): real -> label 1, fake -> label 0
        d_loss = F.binary_cross_entropy_with_logits(D(real), torch.ones(64, 1)) + F.binary_cross_entropy_with_logits(D(fake.detach()), torch.zeros(64, 1))
        optD.zero_grad(); d_loss.backward(); optD.step()
        # generator step: YOUR CODE HERE (non-saturating loss, update only G)
        z = torch.randn(64, 1)
    return G

G_trained = train_gan_1d()
with torch.no_grad():
    samples = G_trained(torch.randn(2000, 1)).squeeze(1)
gen_mean = float(samples.mean())
gen_std = float(samples.std())
print(f"generated mean {gen_mean:.2f}, std {gen_std:.2f}")
```

*Exercise B2.6:* **Reading a failing log (1.5 points).** A student trains a GAN on pictures of 8 kinds of leaf. After 4,000 steps the log says: discriminator loss 0.02, generator loss 4.7, the 500 generated pictures show only one kind of leaf, and their pixel spread is 0.03 while the real pictures have 0.9. Store in `d_loss_guess` the discriminator loss (in the form of B2.3) that a discriminator would have if it could only guess, `D = 0.5` everywhere (0.5 points). In `ANSWERS["B2.6"]` write 2 to 3 sentences: name the failure, use two of the numbers as evidence, and give one remedy (1 point).

```python
d_loss_guess = None   # YOUR CODE HERE
ANSWERS["B2.6"] = ""  # YOUR ANSWER: 2 to 3 sentences
```

*Exercise B2.7:* **Transpose convolution or resize-and-convolve? (1 point).** A generator layer must turn a 7 × 7 grid with 64 channels into a 14 × 14 grid with 32 channels. Store in `params_convT` the parameters of a `ConvTranspose2d` with kernel 3, stride 2, padding 1 and output padding 1, with a bias (0.5 points). In `ANSWERS["B2.7"]` write 2 sentences: why this kernel (3, with stride 2) can produce a checkerboard pattern, and what the resize-and-convolve alternative does (0.5 points).

```python
params_convT = None   # YOUR CODE HERE
ANSWERS["B2.7"] = ""  # YOUR ANSWER: 2 sentences
```

### Task 3 — Diffusion models (16 points, 30 minutes)

Week 13 turned a picture into noise with a schedule, taught a network to predict that noise, and ran the process backwards with DDPM and DDIM. It also gave the network a time embedding, a UNet body, guidance and a smaller latent space. In B3.1 to B3.4 you compute with the schedule, the closed form, the reverse steps and the time embedding on small numbers, in B3.5 you train two tiny denoisers and compare them, and in B3.6 to B3.8 you count a UNet block, do the guidance and latent arithmetic and explain the design in words. All numbers in this task are invented for the exam, and each part uses its own schedule, so an early mistake does not spoil a later part.

*Exercise B3.1 (3 points): the schedule, ᾱ and the signal-to-noise ratio.* A linear noise schedule has `T1 = 250` steps with β rising in a straight line from `B0 = 0.0004` (step 1) to `B1 = 0.04` (step 250). (a) Complete `make_abar(T, b0, b1)`, which returns the NumPy array of ᾱ_1, …, ᾱ_T (so index t − 1 holds ᾱ_t) (1 point). (b) For this schedule store in `abar_60` and `abar_150` the values ᾱ_60 and ᾱ_150 (0.5 points), and in `snr_60` and `snr_150` the signal-to-noise ratios ᾱ / (1 − ᾱ) at those steps (0.5 points). (c) Store in `t_cross` the first step t at which the signal-to-noise ratio is below 1 (1 point).

```python
T1, B0, B1 = 250, 0.0004, 0.04

def make_abar(T, b0, b1):
    """The array of abar_1 ... abar_T for a linear schedule from b0 to b1 (index t - 1 is step t)."""
    return np.ones(T)   # YOUR CODE HERE

abar = make_abar(T1, B0, B1)
abar_60 = abar_150 = None   # YOUR CODE HERE
snr_60 = snr_150 = None     # YOUR CODE HERE
t_cross = None              # YOUR CODE HERE
```

*Exercise B3.2 (2 points): the closed form and the loss.* Use the given schedule below (`BETA2[t - 1]` is β_t and `ABAR2[t - 1]` is ᾱ_t). (a) Complete `noise_to(x0, eps, t)`, which returns x_t = √ᾱ_t · x0 + √(1 − ᾱ_t) · ε as a NumPy array (1 point). (b) For t = 150, x0 = `[0.6, -0.2, 0.9, -0.7]` and ε = `[1.1, -0.4, 0.3, -1.5]` store the picture weight in `w_sig`, the noise weight in `w_noise` and the noisy vector in `x_t` (0.5 points). The network guessed `eps_hat`; store the mean squared error of the guess in `mse_eps` (0.5 points).

```python
T2 = 400
BETA2 = np.linspace(2e-4, 0.03, T2)    # beta_t is BETA2[t - 1]
ABAR2 = np.empty(T2); _run = 1.0       # abar_t is ABAR2[t - 1]
for _i in range(T2):
    _run *= 1 - BETA2[_i]; ABAR2[_i] = _run

def noise_to(x0, eps, t):
    """x_t = sqrt(abar_t) * x0 + sqrt(1 - abar_t) * eps, for the schedule above (NumPy arrays)."""
    return np.asarray(x0, dtype=float)   # YOUR CODE HERE

x0_b = np.array([0.6, -0.2, 0.9, -0.7]); eps_b = np.array([1.1, -0.4, 0.3, -1.5]); eps_hat_b = np.array([0.8, -0.1, 0.5, -1.2])
w_sig = w_noise = None   # YOUR CODE HERE
x_t = None               # YOUR CODE HERE
mse_eps = None           # YOUR CODE HERE
```

*Exercise B3.3 (3 points): one reverse step, two ways.* The network sees the noisy vector `xt3` at step `t3 = 200` and guesses the noise `eps_hat3`; use the schedule of B3.2. (a) **DDPM (1.5 points).** Store in `beta_t` the value β_200 and in `coef` the number β_t / √(1 − ᾱ_t) (0.5 points). Then store in `x_prev` the picture x_(t−1) = (1/√α_t) · (x_t − coef · ε̂) + σ_t · z, with α_t = 1 − β_t, σ_t = √β_t and the fresh noise `z3` (1 point). (b) **DDIM (1.5 points).** With η = 0, store in `x0_hat` the estimated clean picture (x_t − √(1 − ᾱ_t) · ε̂) / √ᾱ_t (0.5 points) and in `x_s` the picture at the earlier step `s3 = 150`, built from x̂₀ and the same ε̂ with the weights of step s3 (1 point).

```python
t3, s3 = 200, 150
xt3 = np.array([0.9, -0.3, 1.4]); eps_hat3 = np.array([0.5, -0.2, 0.9]); z3 = np.array([0.3, -0.8, 0.1])
beta_t = coef = None   # YOUR CODE HERE
x_prev = None          # YOUR CODE HERE
x0_hat = None          # YOUR CODE HERE
x_s = None             # YOUR CODE HERE
```

*Exercise B3.4 (2 points): the sinusoidal time embedding.* (a) Complete `time_emb(t, d)`, which returns the time embedding of step `t` as a NumPy vector of `d` numbers (`d` is even): for the pair number k = 0, 1, …, d/2 − 1 the entry 2k is sin(t / 10000^(2k/d)) and the entry 2k + 1 is cos(t / 10000^(2k/d)), with the angles in radians (1 point). (b) For d = 16 store in `cos_near` the cosine similarity of the embeddings of steps 100 and 101, and in `cos_far` the cosine similarity of the embeddings of steps 100 and 250 (cosine similarity = dot product divided by the product of the lengths) (1 point).

```python
def time_emb(t, d):
    """Sinusoidal embedding of the step t: a NumPy vector of d numbers."""
    return np.zeros(d)   # YOUR CODE HERE

cos_near = None   # YOUR CODE HERE
cos_far = None    # YOUR CODE HERE
```

*Exercise B3.5 (3 points): does the denoiser need to know the step?* The cell below builds a small data set (three tight clusters of 2-D points), a short schedule `ABAR5` of 100 steps, a small network `MLPDenoiser(use_time)` that takes the noisy point and, if `use_time` is true, also the step, and `eval_mse(model, t_level)`, which measures the noise-prediction error of a model at one fixed step on a fixed evaluation batch. Read it, then run it.

```python
T5 = 100
_b5 = np.linspace(0.002, 0.12, T5); _a5 = np.empty(T5); _run = 1.0
for _i in range(T5):
    _run *= 1 - _b5[_i]; _a5[_i] = _run
ABAR5 = torch.tensor(_a5, dtype=torch.float32)   # abar_t is ABAR5[t - 1]
CENTRES = torch.tensor([[-2.0, 0.0], [2.0, 0.0], [0.0, 2.5]])

def sample_x0(n, gen):
    """n clean points: three tight clusters of 2-D points (shape (n, 2))."""
    k = torch.randint(0, 3, (n,), generator=gen)
    return CENTRES[k] + 0.25 * torch.randn(n, 2, generator=gen)

class MLPDenoiser(nn.Module):
    """Predicts the noise from the noisy point x_t (shape (n, 2)) and, if use_time, the step t (an integer tensor of shape (n,))."""
    def __init__(self, use_time):
        super().__init__()
        self.use_time = use_time
        self.net = nn.Sequential(nn.Linear(3 if use_time else 2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2))
    def forward(self, x, t):
        if self.use_time:
            x = torch.cat([x, (t.float() / T5)[:, None]], dim=1)
        return self.net(x)

@torch.no_grad()
def eval_mse(model, t_level, n=2000):
    """Mean squared error of the noise guess when every point is noised to the same step t_level (fixed evaluation data)."""
    gen = torch.Generator().manual_seed(123)
    x0 = sample_x0(n, gen); eps = torch.randn(n, 2, generator=gen)
    a = ABAR5[t_level - 1]
    xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
    return float(F.mse_loss(model(xt, torch.full((n,), t_level)), eps))

STEPS = 250 if QUICK else 500   # QUICK lets the author shorten long loops when testing; leave it alone
```

(a) Complete `train_denoiser(model, steps)`: in every step draw a batch of 256 clean points with `sample_x0(256, gen)`, a step `t` for every point with `torch.randint(1, T5 + 1, (256,), generator=gen)` and noise `eps` with `torch.randn(256, 2, generator=gen)`, build x_t with the closed form and `ABAR5`, and minimise the mean squared error between `model(x_t, t)` and `eps` with Adam (learning rate 3e-3) (1 point). Run it for a model with the time input and for one without, store the errors at the low-noise step `T_LOW = 15` and at the high-noise step `T_HIGH = 95` in `mse_t_low`, `mse_t_high` (with time) and `mse_n_low`, `mse_n_high` (without time) (1 point). (b) In `ANSWERS["B3.5"]` write three or four sentences that compare the two models using your four numbers and explain why knowing the step helps (1 point).

```python
def train_denoiser(model, steps=STEPS, seed=0):
    """Train `model` to predict the noise (see the exercise text). Returns the model."""
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for step in range(steps):
        pass   # YOUR CODE HERE
    return model

torch.manual_seed(42); model_t = train_denoiser(MLPDenoiser(True))
torch.manual_seed(42); model_n = train_denoiser(MLPDenoiser(False))
T_LOW, T_HIGH = 15, 95
mse_t_low = mse_t_high = None   # YOUR CODE HERE
mse_n_low = mse_n_high = None   # YOUR CODE HERE
ANSWERS["B3.5"] = ""            # YOUR ANSWER
```

*Exercise B3.6 (1 point): parameters of two UNet blocks.* A UNet block has two 3 × 3 convolutions (the first from `C_in` to `C_out` channels, the second from `C_out` to `C_out`, both with biases) and a linear layer that maps the time embedding of size `D` to `C_out` numbers (with bias). (a) Store in `p_down` the parameters of a down block with C_in = 32, C_out = 64 and D = 64, and in `p_up` the parameters of an up block with C_out = 32 and D = 64 whose input has 96 channels (64 from the layer below and 32 copied over by the skip connection) (0.5 points). (b) Store in `skip_extra` how many more parameters this up block has than the same block would have if its input had only the 64 channels from below (0.5 points).

```python
D_T = 64
p_down = p_up = skip_extra = None   # YOUR CODE HERE
```

*Exercise B3.7 (1 point): guidance and the latent space.* (a) At one step the network gives the noise guess `eps_u` without the prompt and `eps_c` with it. With guidance scale `w = 4`, store the classifier-free guided guess in `eps_guided` (0.5 points). (b) A picture of 640 × 384 pixels with 3 colour channels is compressed by an autoencoder into a latent with each side 8 times smaller and 16 channels. Store in `ratio` how many times fewer numbers the latent has than the picture (0.5 points).

```python
eps_u = np.array([0.30, -0.10, 0.50]); eps_c = np.array([0.45, -0.40, 0.20]); w = 4
eps_guided = None   # YOUR CODE HERE
ratio = None        # YOUR CODE HERE
```

*Exercise B3.8 (1 point): explain the design.* In `ANSWERS["B3.8"]` write four or five sentences that answer three questions in your own words: why does the network predict the **noise** ε instead of the clean picture, why does a DDIM sampler need far fewer steps than DDPM, and what do you pay for guidance with w > 1?

```python
ANSWERS["B3.8"] = ""   # YOUR ANSWER
```

### Task 4 — Training LLMs: SFT, preference learning and LoRA (12 points, 30 minutes)

Week 14 followed a language model from pre-training to a helpful assistant: supervised fine-tuning with the prompt masked, a reward model from human comparisons (Bradley–Terry), the KL-regularised objective and its best policy, DPO, and then LoRA and QLoRA, which train a small add-on instead of the whole model. In B4.1 to B4.4 you do the loss arithmetic of SFT, the reward model, the KL leash and DPO. In B4.5 you build a `LoRALinear` layer and train it. In B4.6 you count parameters and memory, and in B4.7 you choose a method in words. All numbers in this task are invented for the exam.

*Exercise B4.1 (2 points): the SFT loss with a masked prompt.* A training example has 11 tokens: the first 6 are the prompt and the last 5 are the response. The model gave the correct token these probabilities (in order). (a) Complete `sft_loss(p, is_resp)`, which returns the average of −ln p over the positions where `is_resp` is 1 (1 point). (b) Store in `loss_masked` the SFT loss of this example (prompt masked) and in `loss_all` the loss if all 11 positions counted (0.5 points each).

```python
p_tok = np.array([0.15, 0.30, 0.08, 0.20, 0.40, 0.12, 0.45, 0.70, 0.25, 0.85, 0.60])
is_resp = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

def sft_loss(p, is_resp):
    """Average negative log-probability (nats) over the response positions."""
    return 0.0   # YOUR CODE HERE

loss_masked = None   # YOUR CODE HERE
loss_all = None      # YOUR CODE HERE
```

*Exercise B4.2 (1 point): the reward model and Bradley–Terry.* A reward model gave these scores to the chosen answer (`r_w`) and the rejected answer (`r_l`) of three comparisons. Store in `p_chosen` the probability σ(r_w − r_l) that the model agrees with the human in each comparison (0.5 points). Store in `rm_loss` the average loss −ln σ(r_w − r_l) over the three comparisons and in `worst_pair` the position (0, 1 or 2) of the comparison with the largest loss (0.5 points).

```python
r_w = np.array([2.6, 0.2, -0.9]); r_l = np.array([0.4, 0.8, -2.5])
p_chosen = None    # YOUR CODE HERE
rm_loss = None     # YOUR CODE HERE
worst_pair = None  # YOUR CODE HERE
```

*Exercise B4.3 (2 points): the KL-regularised optimum.* A policy chooses among three answers. The reference model writes them with probabilities `ref` and the reward model scores them `rew`. (a) Complete `kl_opt(ref, r, beta)`, which returns the pair (best policy π*, KL(π* ‖ π_ref)) for the objective E[r] − β · KL (1 point). (b) For the three leash strengths in `betas`, store in `kl_list` the KL of the best policy and in `top_share` the probability that the best policy gives to the third answer (0.5 points). (c) Store in `beta_closest` the β from `betas` for which the best policy stays closest to the reference (0.5 points).

```python
ref = np.array([0.55, 0.30, 0.15]); rew = np.array([0.5, 1.5, 4.0]); betas = [0.25, 1.0, 4.0]

def kl_opt(ref, r, beta):
    """The best policy pi* and its KL(pi* || ref) for the objective E[r] - beta * KL, as the pair (pi_star, kl)."""
    return np.asarray(ref, dtype=float), 0.0   # YOUR CODE HERE

kl_list = None      # YOUR CODE HERE
top_share = None    # YOUR CODE HERE
beta_closest = None # YOUR CODE HERE
```

*Exercise B4.4 (1 point): the DPO loss.* For one comparison the policy being trained and the frozen reference model gave these log-probabilities to the chosen and the rejected answer. With β = 0.2, store in `margin` the number β · [ (ln π(y_w) − ln π_ref(y_w)) − (ln π(y_l) − ln π_ref(y_l)) ] and in `dpo` the DPO loss −ln σ(margin) (0.75 points). Store in `dpo_start` the loss at the very start of training, when the policy is still equal to the reference (0.25 points).

```python
lp_w, lp_l = -1.20, -2.90        # policy: ln pi(y_w), ln pi(y_l)
ref_w, ref_l = -1.60, -2.30      # reference: ln pi_ref(y_w), ln pi_ref(y_l)
beta_dpo = 0.2
margin = None      # YOUR CODE HERE
dpo = None         # YOUR CODE HERE
dpo_start = None   # YOUR CODE HERE
```

*Exercise B4.5 (3 points): a LoRA layer.* (a) Complete the class `LoRALinear(base, r, alpha)` (1 point). It wraps a given `nn.Linear` called `base`: the base weight and bias must be frozen (`requires_grad = False`), the layer adds two trainable matrices `A` of shape (r, in_features), filled with small random numbers, and `B` of shape (out_features, r), filled with zeros, and its output is `base(x) + (alpha / r) * (x @ A.T @ B.T)`. (b) For `nn.Linear(96, 64)` wrapped with r = 4 and alpha = 8, store in `n_trainable` the number of trainable parameters, in `n_frozen` the number of parameters of the base layer, and in `share_pct` the trainable number as a percentage of `n_frozen` (2 decimals) (0.5 points). (c) Train a LoRA layer on the small regression task below, where the target comes from the base layer plus a rank-2 change: take a copy of every parameter of `lora` before training, train the trainable parameters only with Adam (learning rate 0.02) for 150 steps on the mean squared error, then store in `loss_before` and `loss_after` the loss before and after training and in `changed` the sorted list of the names (from `lora.named_parameters()`) of the parameters whose values changed (1.5 points).

```python
class LoRALinear(nn.Module):
    """A frozen nn.Linear plus a trainable low-rank update (alpha / r) * B A."""
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base
        # YOUR CODE HERE: freeze the base, create A and B
    def forward(self, x):
        return self.base(x)   # YOUR CODE HERE: add the low-rank path

lora96 = LoRALinear(nn.Linear(96, 64), 4, 8)
n_trainable = n_frozen = share_pct = None   # YOUR CODE HERE
```

```python
torch.manual_seed(0)
base_lin = nn.Linear(24, 16)
X_reg = torch.randn(256, 24)
Y_reg = base_lin(X_reg).detach() + 0.3 * X_reg @ torch.randn(2, 24).T @ torch.randn(16, 2).T   # the base layer plus a rank-2 change
W_orig = base_lin.weight.detach().clone()
lora = LoRALinear(base_lin, 4, 8)

loss_before = loss_after = None   # YOUR CODE HERE
changed = None                    # YOUR CODE HERE
print(loss_before, loss_after, changed)
```

*Exercise B4.6 (2 points): parameters and memory of a bigger model.* A model with `N_model = 13e9` parameters has 40 layers. LoRA with rank `r_lora = 16` is placed on four square matrices of size 5120 × 5120 in every layer (the query, key, value and output projections). (a) Store in `lora_params` the number of trainable parameters of all the adapters (1 point). (b) Use these rules, with 1 GB = 10⁹ bytes. Full fine-tuning needs 16 bytes per parameter. With LoRA the frozen base weights need 2 bytes each (16-bit), with QLoRA 0.5 byte each (4-bit), and each trained number needs a further 14 bytes (2 for its gradient and 12 for the optimiser state and master copy). Store in `gb_full`, `gb_lora` and `gb_qlora` the memory in GB, without activations (1 point).

```python
layers, mats, d_model, r_lora, N_model = 40, 4, 5120, 16, 13e9
lora_params = None   # YOUR CODE HERE
gb_full = gb_lora = gb_qlora = None   # YOUR CODE HERE
```

*Exercise B4.7 (1 point): choose a method.* A school has a base language model that answers correctly but in a cold, curt style. It has 3,000 pairs of answers (a polite answer that teachers chose and a curt one that they rejected) and one 24 GB GPU. In `ANSWERS["B4.7"]` write four or five sentences: which method you would use (SFT, RLHF or DPO) and why, how LoRA or QLoRA helps with the 24 GB, and which risk you would watch for and with what.

```python
ANSWERS["B4.7"] = ""   # YOUR ANSWER
```

### Task 5 — Synthesis with the capstone (6 points, 15 minutes)

Three short scenarios that join Weeks 11 to 14 and the Structural Inspection Assistant, the capstone project of the course. Answer each in `ANSWERS`, in the number of sentences the question gives (2 points each). Use numbers where the question asks for them; you may use code cells for the arithmetic. A licensed engineer makes every pass or fail decision; the assistant never signs off on safety.

*Exercise B5.1 (2 points; Weeks 11 and 14).* A team adapts a pre-trained ViT-Base (hidden size 768, 12 layers, patches of 16 × 16 pixels, about 86 million parameters, a position table with 197 rows learned at 224 × 224) to the four classes crack, spalling, corrosion and none. They want to feed photos of 384 × 256 pixels. (a) How many patch tokens does one photo give, how many tokens with the class token, and how many attention scores does one head compute per layer for one photo? (b) The team puts LoRA adapters of rank 16 on the query and the value projection (each 768 × 768) of all 12 layers and trains a new head (768 × 4 + 4). How many parameters are trained, and what share of 86 million is that? (c) In two or three sentences: what goes wrong with the position table at this photo size, and name two ways out. Answer in three to six sentences, with your numbers.

*Exercise B5.2 (2 points; Weeks 12 and 13).* The team has few photos of early corrosion and wants 500 synthetic ones to balance the training set. Their candidate is a latent diffusion model: a 256 × 256 × 3 picture becomes a latent of 32 × 32 × 4. It can be sampled with DDPM (1,000 network calls) or with DDIM (25 steps) with classifier-free guidance (two network calls per step). (a) How many times fewer numbers does the latent have, and how many times fewer positions for attention? (b) How many network calls do 500 pictures need with DDPM and with guided DDIM, how many times fewer is that, and how long does guided DDIM take if one call takes 0.04 seconds? (c) Name one danger of training on the synthetic pictures, and one thing from Week 12 that the team should check if they considered a GAN instead. Answer in three to six sentences, with your numbers.

*Exercise B5.3 (2 points; Weeks 11 to 14): the design card.* Write a design card for the Structural Inspection Assistant in `ANSWERS["B5.3"]`. A photo comes in, the assistant classifies it (crack, spalling, corrosion or none), finds the matching checklist items and drafts a short summary for the engineer. The card has a line for each of four steps, and each line says **which model does the step, which week it comes from, and one number**: (1) the photo as tokens: a ViT or CLIP model, and the number of tokens for a 448 × 336 photo with 16 × 16 patches (with the class token); (2) the classifier: how it is adapted and the number of trainable parameters for your own choice of rank and matrices; (3) optional extra data: the model for synthetic images and the number of network calls for one picture with a sampler you choose; (4) the summary writer: how it is trained (SFT, DPO or both) and the loss or setting you would watch. Then add three more lines: **one evaluation** (a number you would measure, and on which photos), **one guardrail** (what could go wrong, for example reward hacking, and what stops it), and **who decides** (the engineer). Write 80 to 150 words in total.

Write your three answers here (the check only tells you that something is there; your instructor reads and marks the text):

When you have finished all five tasks, print your answers once more so you can read them through:

```python
print("Your ANSWERS (copy them into your submission if your instructor asks):")
for k in sorted(ANSWERS):
    print(f"\n{k}: {ANSWERS[k]}")
```

### What to submit

1. The notebook with all five tasks, run from top to bottom (Kernel → Restart Kernel and Run All Cells in JupyterLab, or Runtime → Restart session and run all in Colab), with your code in the starter cells, your written answers in the `ANSWERS` dictionary (or in markdown cells under the task) and the outputs visible, including the output of the last cell that prints `ANSWERS`.
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
- Good luck. You have worked hard through Weeks 11 to 14, and this exam only asks you to show it.
<!-- LAB-END -->
## Check your understanding: the written exam (Part A) {rail=Written exam}

{lede} Part A is ==the written half of the Pre-final==: 40 multiple-choice questions on Weeks 11 to 14, 60 minutes, closed book, worth 40 of the 100 points.

{sub} Exam · Part A · 40 points · 60 minutes · Your best score is saved to your account

Press Start to begin the timer. The exam is **timed** and submits itself at zero. There is **no feedback until you submit**; until then you can move back and forth, change answers and **flag** questions to return to. After you submit, you see your score, a **breakdown by topic** (one row for each of the four weeks) and the explanation for every question you missed, so that your revision targets are clear. Your best score is saved to your account. Each question is worth 1 point, so your Part A score out of 40 is added to Part B for the total out of 100. The written exam runs in your browser and cannot stop a determined cheater, so it is meant to be taken under the supervision your instructor arranges.

```quiz
{
  "id": "week-15-prefinal",
  "exam": true,
  "minutes": 60,
  "passMark": 70,
  "title": "Pre-final written exam (Part A)",
  "rules": [
    "Part A is worth 40 of the 100 points. Each question has one correct answer and is worth 1 point.",
    "A calculator and scratch paper are allowed. No notes, lessons, lab notebooks or AI assistants."
  ],
  "questions": [
    {
      "topic": "Week 11: images as tokens and CLIP",
      "q": "A team feeds a drone photo to a Vision Transformer. What does the model actually treat as the “tokens” of the picture?",
      "options": [
        "Fixed-size square patches cut from the picture, each flattened and turned into one embedding vector.",
        "The individual pixels, so a 224 × 224 picture is a sequence of 50,176 tokens.",
        "The rows of the picture, read from top to bottom, one token for each row of pixels.",
        "The objects that a detector finds in the picture, one token for each object name."
      ],
      "answer": 0,
      "why": "A ViT cuts the picture into equal patches, flattens each one and maps it with a linear layer to a vector, which plays the role of a word embedding. Pixels as tokens would make the sequence far too long, and ViT needs no object detector."
    },
    {
      "topic": "Week 11: images as tokens and CLIP",
      "q": "A ViT cuts a 192 × 128 picture (height × width) into non-overlapping 16 × 16 patches and adds one class token at the front. How many tokens enter the Transformer?",
      "options": [
        "96",
        "97",
        "21",
        "1536"
      ],
      "answer": 1,
      "why": "The picture gives (192 ÷ 16) × (128 ÷ 16) = 12 × 8 = 96 patches, and the class token makes 97. Forgetting the class token gives 96, 1536 comes from dividing the pixel count by 16 instead of 256, and 21 adds the two grid sides instead of multiplying them."
    },
    {
      "topic": "Week 11: images as tokens and CLIP",
      "q": "A ViT uses 14 × 14 patches of a colour picture (3 channels) and an embedding size D = 384. The patch embedding is one linear layer with bias. How many parameters does it have?",
      "options": [
        "972",
        "226,176",
        "225,792",
        "75,648"
      ],
      "answer": 1,
      "why": "Each flattened patch has 14 × 14 × 3 = 588 numbers, so the layer has 588 × 384 weights plus 384 biases = 226,176. Without the bias you get 225,792, leaving out the channels gives 75,648, and adding instead of multiplying gives 972."
    },
    {
      "topic": "Week 11: images as tokens and CLIP",
      "q": "A tiny ViT sees a picture as 8 × 8 = 64 patches plus a class token. Each attention layer has 6 heads, and each head scores every token against every token. How many attention scores does one layer compute for one picture?",
      "options": [
        "4,225",
        "390",
        "25,350",
        "24,576"
      ],
      "answer": 2,
      "why": "Each head builds an n × n table with n = 65 tokens, which is 4,225 scores, and 6 heads give 25,350. A single head gives 4,225, forgetting the class token gives 24,576, and 390 multiplies n by the heads instead of squaring n."
    },
    {
      "topic": "Week 11: images as tokens and CLIP",
      "q": "How is CLIP trained?",
      "options": [
        "On pictures labelled with one of 1,000 fixed class names, so that it learns to output the class name.",
        "On image–caption pairs, so that each picture's embedding is close to its own caption's embedding and far from the other captions in the batch.",
        "On pictures alone, so that it learns to rebuild each picture from a corrupted copy.",
        "On captions alone, so that it learns to predict the next word of a caption."
      ],
      "answer": 1,
      "why": "CLIP is trained with a contrastive objective on a large set of image–text pairs: it pulls matching pairs together in a shared space and pushes the others apart. That is why it can compare a new picture with any text you write, without a fixed list of classes."
    },
    {
      "topic": "Week 11: images as tokens and CLIP",
      "q": "A picture embedding is (2, 1, 2) and a caption embedding is (1, 2, 2). What is their cosine similarity?",
      "options": [
        "0.4444",
        "8",
        "0.8889",
        "1.3333"
      ],
      "answer": 2,
      "why": "The dot product is 2·1 + 1·2 + 2·2 = 8, and both vectors have length √(4 + 1 + 4) = 3, so the cosine is 8 ÷ (3 × 3) = 0.8889. Dividing by 3² + 3² = 18 gives 0.4444, the bare dot product is 8, and dividing by 3 + 3 gives 1.3333 (a cosine can never exceed 1)."
    },
    {
      "topic": "Week 11: images as tokens and CLIP",
      "q": "For a photo, CLIP gives a cosine of 0.30 with the caption “a cracked beam” and 0.25 with “an intact beam”. With a logit scale of 20 and a softmax over the two captions, what probability does “a cracked beam” get?",
      "options": [
        "0.7311",
        "0.5455",
        "0.5125",
        "0.9933"
      ],
      "answer": 0,
      "why": "The logits are 20 × 0.30 = 6 and 20 × 0.25 = 5, so the probability is e⁶ ÷ (e⁶ + e⁵) = 1 ÷ (1 + e⁻¹) = 0.7311. Skipping the scale gives 0.5125, dividing the cosines by their sum gives 0.5455, and a scale of 100 would give 0.9933."
    },
    {
      "topic": "Week 11: images as tokens and CLIP",
      "q": "An inspection team uses CLIP zero-shot with just two captions, “crack” and “no crack”. On 200 photos that the engineers labelled, the accuracy is only 61%. What is the most sensible next step?",
      "options": [
        "Trust the 61% because zero-shot models never need their captions changed.",
        "Raise the logit scale until the probabilities look confident, which makes the answers more accurate.",
        "Retrain CLIP from scratch on the 200 photos, since a zero-shot model cannot be improved.",
        "Write fuller captions such as “a close-up photo of a concrete wall with a crack”, and measure the accuracy again on the same labelled photos."
      ],
      "answer": 3,
      "why": "The words you write are the classifier in zero-shot CLIP, so better captions can change the accuracy, and the labelled set tells you whether they did. Retraining on 200 photos would be tiny, and a larger logit scale only sharpens probabilities without changing which caption wins."
    },
    {
      "topic": "Week 11: images as tokens and CLIP",
      "q": "In a batch of 3 picture–caption pairs, CLIP's softmax gives the correct caption the probabilities 0.80, 0.50 and 0.25 (one for each picture). The loss is the average of −ln of these probabilities. What is the loss?",
      "options": [
        "0.7675",
        "0.6604",
        "2.3026",
        "0.5167"
      ],
      "answer": 0,
      "why": "−ln 0.8 = 0.2231, −ln 0.5 = 0.6931 and −ln 0.25 = 1.3863, and their average is 2.3026 ÷ 3 = 0.7675. Taking −ln of the average probability gives 0.6603, the average probability is 0.5167, and forgetting to divide by 3 gives 2.3026."
    },
    {
      "topic": "Week 11: images as tokens and CLIP",
      "q": "A product must answer questions such as “Is the rust near the bolts?” about a photo. Why is a CLIP-style model that compares one picture vector with one text vector not enough?",
      "options": [
        "CLIP cannot read text, so it must be replaced by a model that only sees pixels.",
        "A single cosine similarity is always the same for every question, so the model would not answer at all.",
        "CLIP's vectors have too few numbers, and doubling the embedding size would solve the problem.",
        "A single picture vector discards where things are, so the model needs the text and the patch tokens to meet, for example through a projector or cross-attention."
      ],
      "answer": 3,
      "why": "Answering a question about a place in the photo needs the details of the patches, not just one summary vector. Designs such as a projector into a language model or cross-attention bring the patch tokens and the words together. Making the vector longer does not recover the location."
    },
    {
      "topic": "Week 12: GANs and transpose convolution",
      "q": "In a GAN, what does the discriminator output for an input picture?",
      "options": [
        "A new picture that is a cleaner copy of the input.",
        "A single number between 0 and 1: its estimate of the probability that the picture is real.",
        "A label naming the object in the picture, chosen from a fixed list of classes.",
        "The noise vector z that the generator would need to produce the input."
      ],
      "answer": 1,
      "why": "The discriminator is a binary classifier. It scores a picture with the probability that it came from the training data, and the generator is trained to push that score up for its own fakes."
    },
    {
      "topic": "Week 12: GANs and transpose convolution",
      "q": "At one point x the real data density is p_data(x) = 0.6 and the generator's density is p_g(x) = 0.2. What is the best possible discriminator's output D*(x)?",
      "options": [
        "0.75",
        "0.40",
        "3.00",
        "0.25"
      ],
      "answer": 0,
      "why": "D*(x) = p_data(x) ÷ (p_data(x) + p_g(x)) = 0.6 ÷ 0.8 = 0.75. Putting p_g on top gives 0.25, subtracting the densities gives 0.40, and the plain ratio gives 3.00, which is not a probability."
    },
    {
      "topic": "Week 12: GANs and transpose convolution",
      "q": "For one real and one fake picture, a discriminator gives D(x) = 0.80 and D(G(z)) = 0.30. Its loss is −ln D(x) − ln(1 − D(G(z))). What is the value?",
      "options": [
        "-0.5798",
        "0.5600",
        "1.4271",
        "0.5798"
      ],
      "answer": 3,
      "why": "−ln 0.8 = 0.2231 and −ln 0.7 = 0.3567, so the loss is 0.5798. Using ln D(G(z)) instead of ln(1 − D(G(z))) gives 1.4271, forgetting the minus signs gives −0.5798, and the product 0.8 × 0.7 is 0.5600, which is not a loss."
    },
    {
      "topic": "Week 12: GANs and transpose convolution",
      "q": "Early in training the discriminator rejects a fake with D(G(z)) = 0.02. With respect to the discriminator's logit a (where D = σ(a)), the saturating generator loss ln(1 − D) has a gradient of size D, and the non-saturating loss −ln D has a gradient of size 1 − D. How many times larger is the non-saturating gradient?",
      "options": [
        "2",
        "50",
        "49",
        "0.02"
      ],
      "answer": 2,
      "why": "The sizes are 0.02 for the saturating loss and 0.98 for the non-saturating loss, and 0.98 ÷ 0.02 = 49. The number 50 is 1 ÷ D, 0.02 is the saturating gradient itself, and 2 has no basis in the numbers."
    },
    {
      "topic": "Week 12: GANs and transpose convolution",
      "q": "A transpose convolution has input side 5, kernel 5, stride 3, padding 2 and no output padding. What is the output side, using n_out = (n − 1)·s − 2p + k + output padding?",
      "options": [
        "13",
        "15",
        "9",
        "17"
      ],
      "answer": 0,
      "why": "(5 − 1) × 3 − 2 × 2 + 5 = 12 − 4 + 5 = 13. Forgetting the padding term gives 17, the shortcut n × s gives 15, and forgetting the kernel size gives 9."
    },
    {
      "topic": "Week 12: GANs and transpose convolution",
      "q": "One DCGAN generator layer is a transpose convolution with 128 input channels, 64 output channels, a 4 × 4 kernel and a bias for each output channel. How many parameters does it have?",
      "options": [
        "32,832",
        "131,072",
        "1,088",
        "131,136"
      ],
      "answer": 3,
      "why": "The weights are 128 × 64 × 4 × 4 = 131,072 and the biases add 64, giving 131,136. Leaving out the biases gives 131,072, using a kernel of 4 instead of 4 × 4 gives 32,832, and forgetting the input channels gives 1,088."
    },
    {
      "topic": "Week 12: GANs and transpose convolution",
      "q": "Which choice follows the DCGAN design rules?",
      "options": [
        "Sigmoid on the generator's output and ReLU on the discriminator's hidden layers.",
        "Strided convolutions in the discriminator, transpose convolutions in the generator, batch normalisation, and tanh on the generator's output.",
        "Transpose convolutions in the discriminator and strided convolutions in the generator.",
        "Pooling layers and fully connected layers in both networks, with no normalisation at all."
      ],
      "answer": 1,
      "why": "DCGAN replaces pooling by strided convolutions in D and by transpose convolutions in G, uses batch normalisation, ReLU in G with a tanh output, and LeakyReLU in D. The two networks are mirror images, not swapped."
    },
    {
      "topic": "Week 12: GANs and transpose convolution",
      "q": "After 5,000 steps, a face GAN's samples look almost the same (the same two faces) whatever the noise z is. The discriminator's loss is moderate and the generator's loss is low. What is the best diagnosis?",
      "options": [
        "The discriminator has won completely, and the generator receives no gradient at all.",
        "Mode collapse: the generator found a few outputs that fool the discriminator and now ignores most of what z could change.",
        "The generator is perfect, because all its pictures are accepted by the discriminator.",
        "The noise z is too small, and training would work if z had fewer dimensions."
      ],
      "answer": 1,
      "why": "Samples that barely change with z, together with a generator that is not being punished, are the signature of mode collapse. A generator that was perfect would cover the variety of the data, and a total win by D would show a very low D loss and a stuck generator loss."
    },
    {
      "topic": "Week 12: GANs and transpose convolution",
      "q": "A generated texture shows a regular grid of bright and dark squares. The generator uses transpose convolutions with kernel 3 and stride 2. Which change is most likely to remove the pattern?",
      "options": [
        "Double the number of channels in every layer and keep the kernel 3 with stride 2.",
        "Replace tanh by sigmoid at the output so that the colours are brighter.",
        "Use nearest-neighbour upsampling followed by an ordinary convolution, or choose a kernel size divisible by the stride (for example 4 with stride 2).",
        "Remove the discriminator, since the grid comes from its feedback."
      ],
      "answer": 2,
      "why": "A kernel that is not divisible by the stride makes the copies of the kernel overlap unevenly, which draws a checkerboard. Resizing then convolving, or using k = 4 with s = 2, gives even overlap. More channels or a different output activation does not change the overlap."
    },
    {
      "topic": "Week 12: GANs and transpose convolution",
      "q": "The generator becomes perfect (p_g = p_data) and the best discriminator outputs 0.5 everywhere. What is the value of V = E[ln D(x)] + E[ln(1 − D(G(z)))] at that point?",
      "options": [
        "1.3863",
        "-0.6931",
        "-2.7726",
        "-1.3863"
      ],
      "answer": 3,
      "why": "ln 0.5 + ln 0.5 = −0.6931 − 0.6931 = −1.3863, which is −ln 4. A single ln 0.5 gives −0.6931, dropping the sign gives +1.3863, and doubling the total gives −2.7726."
    },
    {
      "topic": "Week 13: diffusion models",
      "q": "The first three steps of a noise schedule have β₁ = 0.02, β₂ = 0.04 and β₃ = 0.06. What is ᾱ₃, the product of the (1 − β) values?",
      "options": [
        "0.115648",
        "0.884352",
        "0.120000",
        "0.880000"
      ],
      "answer": 1,
      "why": "ᾱ₃ = 0.98 × 0.96 × 0.94 = 0.884352. Subtracting the sum of the β values gives 0.880000, the complement 1 − ᾱ₃ is 0.115648, and the sum of the βs is 0.120000."
    },
    {
      "topic": "Week 13: diffusion models",
      "q": "At a step with ᾱ_t = 0.64, a pixel has the clean value x₀ = 1.5 and the noise draw is ε = −0.5. Using x_t = √ᾱ_t · x₀ + √(1 − ᾱ_t) · ε, what is x_t?",
      "options": [
        "0.78",
        "1.50",
        "0.90",
        "1.00"
      ],
      "answer": 2,
      "why": "√0.64 = 0.8 and √0.36 = 0.6, so x_t = 0.8 × 1.5 + 0.6 × (−0.5) = 1.20 − 0.30 = 0.90. Forgetting the square roots gives 0.78, simply adding x₀ and ε gives 1.00, and flipping the sign of the noise term gives 1.50."
    },
    {
      "topic": "Week 13: diffusion models",
      "q": "At a late step the schedule gives ᾱ_t = 0.2. What is the signal-to-noise ratio SNR = ᾱ_t ÷ (1 − ᾱ_t)?",
      "options": [
        "4.00",
        "0.20",
        "0.25",
        "0.80"
      ],
      "answer": 2,
      "why": "SNR = 0.2 ÷ 0.8 = 0.25: the noise carries four times the signal's weight. The value 4.00 is the inverse, 0.20 is ᾱ_t alone, and 0.80 is 1 − ᾱ_t alone."
    },
    {
      "topic": "Week 13: diffusion models",
      "q": "Why is a diffusion network trained to predict the noise ε that was added, instead of the clean picture directly?",
      "options": [
        "Predicting noise means the network never has to see the noisy picture.",
        "Predicting the clean picture is impossible because the network cannot output pictures.",
        "During training we know the exact ε we added, so the target is available at every noise level, and a simple mean-squared error on ε teaches the network.",
        "The noise is always the same for every step, so the network only needs to memorise it."
      ],
      "answer": 2,
      "why": "We create x_t from x₀ and a known ε, so the true ε is the label, and the mean-squared error between ε and the network's guess is the training loss. The noise is a fresh random draw each time, and the network does see the noisy picture and the step number."
    },
    {
      "topic": "Week 13: diffusion models",
      "q": "One DDPM reverse step has x_t = 0.9, a noise guess ε̂ = 0.5, β_t = 0.04 (so α_t = 0.96) and ᾱ_t = 0.64. The mean of x_(t−1) is (1 ÷ √α_t) · ( x_t − β_t ÷ √(1 − ᾱ_t) · ε̂ ). What is it?",
      "options": [
        "0.8667",
        "0.8930",
        "1.0833",
        "0.8845"
      ],
      "answer": 3,
      "why": "β_t ÷ √(1 − ᾱ_t) = 0.04 ÷ 0.6 = 0.0667, times 0.5 is 0.0333, so the bracket is 0.9 − 0.0333 = 0.8667 and dividing by √0.96 = 0.9798 gives 0.8845. Skipping the division by √α_t gives 0.8667, dividing by √ᾱ_t instead gives 1.0833, and using √ᾱ_t in the inner fraction gives 0.8930."
    },
    {
      "topic": "Week 13: diffusion models",
      "q": "A sinusoidal time embedding with d = 4 uses emb(t)_(2k) = sin( t ÷ 10000^(2k/d) ) and emb(t)_(2k+1) = cos( t ÷ 10000^(2k/d) ). For t = 2, what is the entry at position 2 (counting from 0)?",
      "options": [
        "0.0200",
        "0.0002",
        "0.9093",
        "0.9998"
      ],
      "answer": 0,
      "why": "Position 2 is k = 1, a sine, with frequency 1 ÷ 10000^(2/4) = 1 ÷ 100 = 0.01, so the value is sin(2 × 0.01) = sin(0.02) = 0.0200. Using the fast frequency 1 gives sin 2 = 0.9093, using 10000 itself gives 0.0002, and taking the cosine gives 0.9998."
    },
    {
      "topic": "Week 13: diffusion models",
      "q": "For one pixel, the denoiser guesses ε_u = 0.2 without the prompt and ε_c = 0.6 with the prompt. With guidance scale w = 5, what is the classifier-free guided guess ε_u + w · (ε_c − ε_u)?",
      "options": [
        "3.00",
        "3.20",
        "2.60",
        "2.20"
      ],
      "answer": 3,
      "why": "ε_c − ε_u = 0.4, and 0.2 + 5 × 0.4 = 2.20. Multiplying ε_c by w gives 3.00, adding w · ε_c to ε_u gives 3.20, and starting from ε_c gives 2.60."
    },
    {
      "topic": "Week 13: diffusion models",
      "q": "A studio has a trained 1,000-step DDPM and finds sampling too slow. They cannot retrain. Which option fits best?",
      "options": [
        "Multiply the noise predictions by 20 so that each step removes 20 steps of noise.",
        "Sample with DDIM using about 50 evenly spaced steps from the same trained network, and check that the picture quality is still acceptable.",
        "Switch to a GAN generator, because diffusion cannot be made faster once it is trained.",
        "Use DDPM for only the first 50 of the 1,000 steps and stop, since later steps do not matter."
      ],
      "answer": 1,
      "why": "DDIM reuses the same trained denoiser but takes big strides along the schedule, and 20 to 50 steps often give good pictures. Stopping DDPM early leaves the picture noisy, and scaling the predictions gives wrong, unstable pictures."
    },
    {
      "topic": "Week 13: diffusion models",
      "q": "A student trains a denoiser that receives only the noisy picture, not the step number t. Samples from it look blurry. What is the most likely reason?",
      "options": [
        "The same network must handle a faint speckle and nearly pure noise without knowing which, so it learns an average guess that suits no level well.",
        "The forward process was run for too few steps, so ᾱ_t never reached 0.",
        "Without t the network cannot compute any loss, so it never trains.",
        "The network has too many parameters, which always makes samples blurry."
      ],
      "answer": 0,
      "why": "The right noise guess depends on the noise level, so a model that cannot see t is forced to compromise between levels, which blurs the result. The loss can still be computed, and parameter count alone does not cause blur."
    },
    {
      "topic": "Week 13: diffusion models",
      "q": "A latent diffusion model compresses a 512 × 512 colour picture (3 channels) into a latent of 64 × 64 × 4 numbers. By what factor does the denoiser's input have fewer numbers than the picture?",
      "options": [
        "48",
        "192",
        "64",
        "8"
      ],
      "answer": 0,
      "why": "The picture has 512 × 512 × 3 = 786,432 numbers and the latent has 64 × 64 × 4 = 16,384, and 786,432 ÷ 16,384 = 48. The number 64 compares only the sides squared (8 × 8), 8 is the factor of one side, and 192 forgets that the latent has 4 channels."
    },
    {
      "topic": "Week 14: training LLMs",
      "q": "In an SFT example the response has 3 tokens with probabilities 0.4, 0.5 and 0.1 given the context. The 2 prompt tokens (each with probability 0.9) are masked out of the loss. What is the loss, the average of −ln p over the response tokens?",
      "options": [
        "1.0986",
        "3.9120",
        "0.8245",
        "1.3040"
      ],
      "answer": 3,
      "why": "−ln 0.4 = 0.9163, −ln 0.5 = 0.6931 and −ln 0.1 = 2.3026, whose sum 3.9120 divided by 3 is 1.3040. Counting the two prompt tokens as well gives 0.8245, taking −ln of the average probability gives 1.0986, and forgetting to divide gives 3.9120."
    },
    {
      "topic": "Week 14: training LLMs",
      "q": "A reward model gives a chosen answer the reward 2.1 and a rejected answer the reward 0.4. By the Bradley–Terry model, what probability does it assign to “the chosen answer is preferred”, σ(r_w − r_l)?",
      "options": [
        "0.8455",
        "0.5987",
        "0.1545",
        "0.8909"
      ],
      "answer": 0,
      "why": "The difference is 2.1 − 0.4 = 1.7 and σ(1.7) = 0.8455. The value 0.1545 is σ(−1.7), the answer with the difference reversed, 0.8909 is σ(2.1) and ignores the rejected answer, and 0.5987 is σ(0.4)."
    },
    {
      "topic": "Week 14: training LLMs",
      "q": "A hospital wants a model that knows the vocabulary of its clinical notes, but it already follows instructions well. Which stage fits, and what does it use?",
      "options": [
        "Pre-training from scratch: random weights and the notes only, which is cheaper than starting from a base model.",
        "Continued pre-training: more next-token training on a large body of domain text, which adds knowledge and vocabulary.",
        "Instruction tuning: a few hundred prompt–response examples, which add millions of tokens of knowledge.",
        "Preference tuning: pairs of chosen and rejected answers, which add the domain words."
      ],
      "answer": 1,
      "why": "Continued pre-training keeps the next-token objective and swaps in domain text, which fits a gap in knowledge and vocabulary. Fine-tuning and preference tuning teach behaviour and style, and training from scratch on a small corpus is not cheaper than starting from a base model."
    },
    {
      "topic": "Week 14: training LLMs",
      "q": "A toy policy has three answers with reference probabilities 0.5, 0.3 and 0.2 and rewards 0, 1 and 2. With the KL-regularised objective and β = 1, the best policy is π*(y) = π_ref(y) · e^(r(y)/β) ÷ Z. What is π* for the third answer?",
      "options": [
        "0.5291",
        "0.2000",
        "0.6652",
        "0.3534"
      ],
      "answer": 0,
      "why": "The unnormalised weights are 0.5, 0.3·e = 0.8155 and 0.2·e² = 1.4778, with sum Z = 2.7933, so π*(3) = 1.4778 ÷ 2.7933 = 0.5290. Ignoring the reference (a plain softmax of the rewards) gives 0.6652, using β = 2 by mistake gives 0.3546, and 0.2000 is the reference probability, which no reward would leave unchanged."
    },
    {
      "topic": "Week 14: training LLMs",
      "q": "For one preference pair, the policy has raised the log-probability of the chosen answer by 1.5 compared with the reference, and lowered that of the rejected answer by 2.5. With β = 0.2, the DPO loss is −ln σ( β · [ (ln π/π_ref)(chosen) − (ln π/π_ref)(rejected) ] ). What is it?",
      "options": [
        "0.0181",
        "0.6931",
        "1.1711",
        "0.3711"
      ],
      "answer": 3,
      "why": "The margin is 0.2 × (1.5 − (−2.5)) = 0.8, and −ln σ(0.8) = 0.3711. Forgetting β gives −ln σ(4) = 0.0181, reversing the sign of the margin gives 1.1711, and 0.6931 (ln 2) is the loss when the policy still equals the reference."
    },
    {
      "topic": "Week 14: training LLMs",
      "q": "A weight matrix W is 4,096 × 4,096. LoRA adds ΔW = B·A with rank r = 8 (B is 4,096 × 8 and A is 8 × 4,096). How many trainable numbers does the adapter for this one matrix have?",
      "options": [
        "32,768",
        "16,777,216",
        "65,536",
        "134,217,728"
      ],
      "answer": 2,
      "why": "B has 4,096 × 8 and A has 8 × 4,096 numbers, so r · (d_in + d_out) = 8 × 8,192 = 65,536. Training all of W would be 16,777,216, multiplying by r as well gives 134,217,728, and counting only one of the two matrices gives 32,768."
    },
    {
      "topic": "Week 14: training LLMs",
      "q": "In LoRA the matrix B starts at zero and A starts with small random values. Why?",
      "options": [
        "So that W can be deleted after the first step, because B·A replaces it.",
        "So that the gradient of A is always zero and A never changes during training.",
        "So that the adapter learns twice as fast as the frozen weights.",
        "Then ΔW = B·A is zero at the start, so the adapted model behaves exactly like the base model, and training moves away from it gently."
      ],
      "answer": 3,
      "why": "With B = 0 the product B·A is 0, so the model starts as the base model and nothing is damaged on the first step. The gradient of A is zero only at that first step, because B then becomes non-zero, and W stays frozen and in use."
    },
    {
      "topic": "Week 14: training LLMs",
      "q": "QLoRA freezes a 3-billion-parameter base model stored in 4-bit numbers (half a byte each) and trains 20 million adapter parameters, which cost 16 bytes each. What is the memory for these numbers, ignoring activations?",
      "options": [
        "6.32 GB",
        "1.82 GB",
        "3.32 GB",
        "1.50 GB"
      ],
      "answer": 1,
      "why": "The frozen weights take 3 × 10⁹ × 0.5 bytes = 1.50 GB and the adapters take 20 × 10⁶ × 16 bytes = 0.32 GB, so the total is 1.82 GB. Leaving out the adapters gives 1.50 GB, storing the base in 16 bits gives 6.32 GB, and counting 4 bits as one byte gives 3.32 GB."
    },
    {
      "topic": "Week 14: training LLMs",
      "q": "During RLHF, a chatbot's reward-model score climbs steadily, but human reviewers rate the answers lower every week. The answers have become very long and flattering. What is happening, and what is a sound response?",
      "options": [
        "The humans are wrong, because the reward model was trained on human data and cannot be fooled.",
        "The reward model is working perfectly, so training should continue until the score stops rising.",
        "Reward hacking: the policy exploits the reward model's mistakes, so keep the KL leash tight, refresh the reward model with new human comparisons, and keep human review.",
        "The policy is forgetting, so the sensible fix is to remove the KL penalty and let it move freely."
      ],
      "answer": 2,
      "why": "A rising reward with falling human ratings is the signature of reward hacking: the policy has found answers that the proxy scores highly but people do not like. A stronger leash, new comparisons for the reward model and human checks address it, while removing the penalty makes it worse."
    },
    {
      "topic": "Week 14: training LLMs",
      "q": "After 10 epochs of fine-tuning on 2,000 maintenance-report examples, a model writes excellent reports, but its score on general question answering has fallen from 71% to 58%. What is the best explanation and fix?",
      "options": [
        "The model has become more intelligent, so the lower general score proves the training worked.",
        "The data was too small to change the weights, and only more epochs on the same 2,000 examples would help.",
        "Catastrophic forgetting: use fewer epochs or a lower learning rate, add LoRA, and mix some general examples into the training data, then measure both skills.",
        "The test questions have changed, and the model's weights were not touched by the training."
      ],
      "answer": 2,
      "why": "Training that is long and narrow overwrites some general skills, which is forgetting. Fewer epochs, a lower learning rate, LoRA (which freezes the base) and a mix of general data keep more of the old skill. Measuring both skills shows the trade-off."
    }
  ]
}
```


## How the Pre-final is marked

**The Pre-final (100 points).** The exam has two parts and is marked out of 100. The pass mark is 70 points.

| Component | Points | Time | What is assessed and how it is marked |
|---|---|---|---|
| Part A: Written exam | 40 | 60 min | 40 multiple-choice questions, 1 point each, no negative marking. The app marks it when you submit and then shows your score and a breakdown by topic (Week 11, Week 12, Week 13, Week 14), so you can see where to revise. |
| Part B, Task 1: Multimodal: patches, ViT and CLIP | 12 | 20 min | Patch counts and tables, cutting a small image into patches, patch embedding parameters, attention size, CLIP cosine similarity and zero-shot choice, the contrastive loss and a fusion design choice. |
| Part B, Task 2: GANs and transpose convolution | 14 | 25 min | Transpose convolution output sizes, a DCGAN-style generator with sizes and parameter counts, the best discriminator and V at equilibrium, saturating against non-saturating gradients, a small GAN and its diagnosis. |
| Part B, Task 3: Diffusion | 16 | 30 min | The ᾱ schedule, closed-form noising, SNR, a DDPM step, the time embedding, a small denoiser with and without the time input, a UNet block count, guidance and the latent size ratio. |
| Part B, Task 4: Training LLMs | 12 | 30 min | Masked SFT loss, Bradley–Terry, the KL-regularised optimum, a DPO loss, a LoRALinear layer with its parameter count, memory for full, LoRA and QLoRA, and the choice between SFT, RLHF and DPO. |
| Part B, Task 5: Synthesis with the Capstone | 6 | 15 min | Three short written scenarios that join several weeks, 2 points each, including the Structural Inspection Assistant design card. |
| **Total** | **100** | **180 min** | Pass mark: 70 points. |

**How Part B is marked.** Your instructor marks the notebook you submit with a marking scheme. Each sub-question earns points for three things: the correct result, a clear method (your code and your working must show how you got the answer, not only the answer), and, where the question asks for it, an interpretation in your own words that uses your numbers. Partial credit is given: a correct method with an arithmetic slip earns most of the points, and a correct result with no visible method earns fewer. The ✓ and ✗ lines printed by the check cells are a guide to whether a number or shape is right; they are not the mark, and a ✓ cannot replace a missing method or explanation. Written answers in `ANSWERS` are read and marked on their content.

**Results and feedback.** Part A shows your score and topic breakdown as soon as you submit. Your instructor announces the date, the room and the weights, when Part B is marked and how you receive feedback, any re-sit arrangements, and how the Pre-final counts towards the course. Use the topic breakdown and your marked notebook to plan the weeks ahead.

**Across the course.** Laboratories and knowledge checks run through the term, with exams in Weeks 5, 10, 15 and 20, plus a capstone. The Capstone marking criteria are in the table above. Your instructor announces the component weights in the syllabus. The long-term goal is the NVIDIA-Certified Associate exam (NCA-GENL), which NVIDIA issues after its own proctored exam; Supervaise cannot issue or guarantee it.

## Exit ticket and homework

**Exit ticket (answer in two or three sentences each):**

1. Which topic in your score breakdown was weakest, and what exactly will you do about it this week?
2. Name one way the Structural Inspection Assistant could go wrong and one check that would catch it.
3. Who is your Team lead, Technical lead and Responsible-AI lead, and which one of the proposal checklist items is still unanswered?

**Homework (no new deadlines):**

- Open your Part A score breakdown and read the explanation of every question you missed. Redo the matching practice problem on this page for each missed topic.
- If your instructor returns your Part B notebook with comments, read them and note what to do differently.
- Finish your team's proposal using the template and agree your roles.
- Preview Week 16: **LLM components and RAG** (retrieval-augmented generation). Think about how the assistant could find the right checklist item for a photo label, and what could go wrong if it finds the wrong one.
