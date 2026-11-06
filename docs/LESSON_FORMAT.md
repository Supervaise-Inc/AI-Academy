# Writing a weekly lesson (`backend/content/week-NN.md`)

Each week is one Markdown file. The app turns it into the Lesson Workspace page (same look as the Week 7 lesson), so you only write content. `backend/content/week-01.md` is the reference lesson: copy its structure. Check your work with:

```bash
node tools/validate-lessons.mjs backend/content week-NN.md     # must print "✓ week-NN.md" and no ERROR lines
```

The validator checks structure, every figure's JSON, quiz answers, and recomputes the maths (for example it checks that an embedding analogy really lands where you say). Fix every ERROR. Read the warnings.

## 1. Required skeleton (in this order)

````markdown
> Content based on the NVIDIA Generative AI and DLI Deep Learning Teaching Kits, used under the Supervaise × NVIDIA partnership.

# Week N — Exact title from COURSE_OUTLINE.md

**Purpose:** One sentence with the ==single most important idea== highlighted.

**Term:** Midterm (Weeks 6–10) · **Month:** 2 · **Time:** about 30 minutes for the lesson, then a 3-hour lab

```viz            ← exactly one "hook" block (animated opener), see §3
{ "type": "hook", ... }
```

## By the end, you can…                      ← 3–5 measurable outcomes (from COURSE_OUTLINE "Key outcomes"), highlight the verb phrase
1. …
Assessed in … (one sentence), then one sentence on how to use the page.

## Concept heading {rail=Concept}            ← 3–5 concept sections, each ends with or contains a figure
{lede} One bigger main-idea paragraph with a [[term|definition]] and a ==highlight==.
### Sub-heading
Body paragraphs (serif), figures (```viz blocks), short lists.

## Key takeaways {rail=Takeaway}            ← 3–4 bullets, each with a ==highlight==

## Lab: … {rail=Laboratory}                  ← activity, see §5
## Check your understanding {rail=Check}
{sub} Formative · Not graded · Unlimited attempts · Your best score is saved to your account
One sentence, then exactly one ```quiz block (§4).

## How Week N is assessed                   ← optional, short (no rail mark)
## Exit ticket and homework                 ← short (no rail mark)
````

Month = ceil(week / 4). Term and week ranges: Prelim 1–5, Midterm 6–10, Pre-final 11–15, Final 16–20. Section order is fixed; the lesson reads top to bottom as concept → figures → takeaway → lab → check. Put the `{rail=…}` label (max 18 characters, 5–8 per lesson, unique) on each major section; sections without it do not appear in the progress rail.

## 2. Text conventions

| Write | You get |
|---|---|
| `==important idea==` | highlighter marker. Use 8 or more per lesson on the ideas a student must remember. Never highlight a whole sentence of filler. |
| `[[term\|definition]]` | dotted-underline term; selecting it opens the definition. At least 4 per lesson. Definition: one or two plain sentences, no `]`, no line breaks. Define a term at its first use. |
| `{lede} text` | start of a paragraph: the larger main-idea line. One per concept section. |
| `{sub} text` | small line under a section heading. |
| `## Heading {rail=Label}` | puts the section in the sticky rail. |
| `### Sub-heading` | sub-part inside a section. |
| ` ```python ` blocks | coloured code with a Copy button. |
| `[text](/prompt-workshop)` | link to another app page (`/course-map`, `/mock-exam`, `/capstone`, `/prompt-workshop`, `/certification`, `/lesson/N`). External `https://` links open in a new tab. |
| tables, `-` lists, `1.` lists, `**bold**`, `*italic*`, `` `code` `` | as usual. |

Do not use raw HTML, images, emoji or `<script>`. Do not use the `==` or `[[` syntax inside code blocks or tables.

Writing style: plain, concrete, second person ("you"), short sentences, one idea per paragraph, units and numbers wherever possible. Define every symbol. The lesson text is for students aged roughly 16 and up. Prefer an example before a definition.

## 3. Figures: ` ```viz ` blocks

A figure is a fenced block labelled `viz` containing JSON. Every figure needs `"type"` and `"title"` (the title is a short noun phrase, not a sentence). Write `"intro"` (one sentence telling the student what to do) and `"note"` (one sentence of honest caveat, e.g. "Illustrative numbers authored for teaching, not from a trained model") where they help. Use at least **3 figures besides the hook** and at least **2 different types**. Every figure must teach something specific to that week; do not add figures for decoration. Each figure should be followed by a sentence or two telling the student what to notice.

Everything a figure shows is computed from its JSON, so numbers are never "faked" but they are only as correct as the data you give. **Compute any number you state in the text with Python.** When data are illustrative, say so in `note`.

### `hook` (exactly one per lesson; shown under the header)
```json
{ "type": "hook", "title": "Watch …", "caption": "What the animation shows and why it matters (40+ characters).",
  "scenes": [ { "name": "Tab name", "kind": "text|image|speech|tokens|denoise|conv|bars|loss", "prompt": "…", ... } ] }
```
3–5 scenes. Each scene has `name`, `kind`, `prompt` and `outputLabel` (heading of the output card). Kind-specific fields:
- `text`: `output` (string typed out).
- `tokens`: `tokens` (array of strings that pop in as chips). Example: prompt `"unbelievably"`, tokens `["un","believ","ably"]`.
- `bars`: `bars` `[{ "label": "Paris", "value": 0.72 }, …]` (values 0–1, grow to their value). Good for probabilities.
- `denoise`: `image` = square array of digit strings (e.g. 12 strings of 12 digits 0–9; 0 is white, 9 is black). Noise clears to show it.
- `conv`: optional `grid` (matrix of numbers 0–9); a 3×3 vertical-edge filter slides over it and fills an output grid.
- `loss`: a training-loss curve going down (no extra fields).
- `image`: a city scene appears out of noise. `speech`: an audio waveform plays. (These two suit generative-media weeks only.)
Choose scenes that give a student the **general idea of the week in 20 seconds**. Prompts are short.

### `flow` (tabs comparing 2+ ideas as input → model → output)
```json
{ "type": "flow", "title": "…", "modes": [ { "name": "A", "tagline": "…", "inputLabel": "Input", "modelLabel": "…", "outputLabel": "Output", "examples": [ { "label": "…", "input": "…", "output": "…" } ], "note": "…" }, { "name": "B", … } ] }
```

### `sorter` (classify cards into bins, instant feedback; ≥ 6 cards)
```json
{ "type": "sorter", "title": "…", "intro": "…", "bins": [ { "id": "encoder", "label": "Encoder-only" }, { "id": "decoder", "label": "Decoder-only" } ],
  "cards": [ { "name": "BERT", "answer": "encoder", "why": "…" }, { "name": "Edge case", "answer": "either", "why": "…" } ] }
```
Omit `bins` for the default two bins `generative` / `discriminative`. `answer` must be a bin id or `"either"`. Every bin must be used.

### `timeline` (≥ 4 clickable events)
```json
{ "type": "timeline", "title": "…", "events": [ { "year": "2017", "title": "…", "text": "…" } ] }
```

### `pipeline` (step through a process; nodes light up)
```json
{ "type": "pipeline", "title": "…", "intro": "…", "vertical": false,
  "nodes": [ { "id": "emb", "label": "Embeddings", "sub": "token → vector" }, … ],
  "steps": [ { "title": "…", "body": "what happens at this step (30+ chars)", "active": ["emb"], "example": "optional concrete example text" }, … ] }
```
≥ 3 nodes with unique ids, ≥ 3 steps, `active` lists node ids. Best for: a Transformer block, a RAG flow, an agent loop, RLHF stages, a GAN training loop, parallelism, a training pipeline.

### `curve` (line plot with an optional slider)
```json
{ "type": "curve", "title": "…", "xRange": [-6, 6], "yRange": [-0.5, 1.5],
  "series": [ { "fn": "sigmoid", "label": "Sigmoid" }, { "fn": "relu" } ],
  "param": { "label": "Slope for x < 0", "min": 0, "max": 0.5, "step": 0.01, "value": 0.1 }, "alt": "text description for screen readers", "note": "…" }
```
`fn` is one of: `sigmoid tanh relu leaky_relu swish gelu sigmoid_grad tanh_grad relu_grad decay`. `leaky_relu` and `decay` need `param` (decay plots `param^x`, the gradient left after x layers when each layer multiplies it by `param`; use `"logY": true` with `"yRange": [-8, 0]` in log10 units).

### `descent` (gradient descent on a 1-D loss with a learning-rate slider)
```json
{ "type": "descent", "title": "…", "fn": "quad", "lr": 0.1, "start": -0.5 }
```
`fn`: `quad` (L = (w−3)², converges for lr < 1, oscillates at 1, diverges above) or `double_well` (two minima; the start point decides which one you reach).

### `neuron` (one neuron's decision line on 2-D data; sliders for w₁, w₂, b)
```json
{ "type": "neuron", "title": "…", "intro": "…", "w": [0.5, -0.5], "b": 0, "range": [-3, 3],
  "datasets": [ { "name": "Separable", "points": [ { "x": 1.5, "y": 1, "c": 1 }, … ] }, { "name": "XOR", "points": [ … ] } ] }
```
`c` is 0 or 1. Output is 1 when w₁x + w₂y + b > 0. Use a separable set plus XOR to show why one neuron is not enough. For XOR no line gets more than 3 of 4 points.

### `conv` (convolution or transpose convolution by hand)
```json
{ "type": "conv", "title": "…", "mode": "conv", "stride": 1, "padding": 0,
  "input": [[1,2,0,1],[0,1,3,1],[2,1,0,0],[1,0,1,2]], "kernel": [[1,0],[0,-1]] }
```
`mode` `"transpose"` takes a small input (≤ 4×4), a kernel and `stride` (default 2). Matrices are plain numbers. Input ≤ 8×8 for readability.

### `calc` (parameter and memory calculators)
`{ "type": "calc", "title": "…", "kind": "fc_vs_conv", "channels": 1, "filters": 8, "kernel": 3 }`
Kinds: `fc_vs_conv` (parameters of a fully connected layer vs a conv layer for an image), `lora` (full matrix vs LoRA adapter), `memory` (weights in FP32/FP16/INT8/INT4 plus a training estimate for a model of N billion parameters), `scaling` (Chinchilla rule of thumb: compute-optimal parameters and tokens for a compute budget).

### `bpe` (byte-pair encoding learned live)
```json
{ "type": "bpe", "title": "…", "corpus": [["low", 5], ["lower", 2], ["newest", 6], ["widest", 3]], "maxMerges": 10, "tryWord": "lowest" }
```
`corpus` lowercase a–z words with counts. The slider replays merges; the student can type a new word to see it tokenised. Needs ≥ 3 merges.

### `ngram` (a statistical language model from a tiny text)
`{ "type": "ngram", "title": "…", "text": "≥ 25 words of text …", "start": "the" }`. Shows P(next | previous word) from counts and generates sentences.

### `scatter` (a 2-D embedding map with nearest neighbours and analogies)
```json
{ "type": "scatter", "title": "…", "metric": "cosine",
  "points": [ { "label": "king", "x": 3, "y": 4, "group": "royal" }, … ≥ 6 points ],
  "analogies": [ { "a": "king", "b": "man", "c": "woman", "expect": "queen" } ] }
```
The validator recomputes each analogy (a − b + c, nearest by the metric, excluding a, b, c) and fails unless it lands on `expect`. Say in `note` that the coordinates are illustrative.

### `heatmap` (a labelled matrix; tabs for several matrices)
```json
{ "type": "heatmap", "title": "…", "rows": ["The","cat"], "cols": ["The","cat"], "corner": "query ↓ key →", "showValues": true, "rowsSumToOne": true,
  "tabs": [ { "name": "Head A", "caption": "…", "W": [[0.7,0.3],[0.4,0.6]] }, { "name": "Head B", "W": [[…]] } ] }
```
Or a single matrix: `"W": [[…]]`. Use for attention maps, causal masks (1 = visible, 0 = hidden), confusion matrices, similarity matrices.

### `attention` (query, keys, values: the picture behind softmax(QKᵀ/√d)V)
```json
{ "type": "attention", "title": "…", "intro": "…", "scale": true,
  "tokens": ["I", "am", "eating", "an", "apple"],
  "keys":   [[1, 0], [0, 1], [1, 1], [0, 0.5], [1.5, 0.5]],
  "values": [[0.2, 1], [0.5, 0.1], [1, 0.9], [0.1, 0.3], [0.9, 0.2]],
  "queries": [ { "label": "apple", "q": [1, 0.5] }, { "label": "eating", "q": [0.5, 1] } ] }
```
Draws the diagram: the query is compared (⊙) with every key, the scores go through softmax to give weights α, and the output is Σ αᵢ vᵢ. All numbers are computed live; the student picks the query and can switch "divide by √d". Use 3 to 7 tokens and vectors of 2 to 4 numbers so the figures stay readable. `keys`, `values` have one row per token; each query `q` has the same length as a key. Say the numbers are illustrative.
### `equation` (a formula whose parts you can select, plus a worked example)
```json
{ "type": "equation", "title": "…", "intro": "…",
  "formula": [ { "t": "α", "id": "alpha" }, " = softmax(", { "t": "q·k", "id": "score" }, " / ", { "t": "√d", "id": "scale" }, ")" ],
  "parts": [ { "id": "alpha", "name": "α (attention weight)", "meaning": "How much this word contributes to the output. All weights sum to 1." } ],
  "steps": [ { "title": "Dot product", "expr": "q·k = 1×1 + 0.5×0 = 1", "note": "…" } ] }
```
Use it whenever a lesson introduces a formula: every symbol is a `{t, id}` term with a plain-language `parts` entry; plain strings are the glue. `steps` is an optional worked example (compute every number yourself, see §8).
### `nested` (concentric circles: a field and the parts inside it, e.g. AI ⊃ machine learning ⊃ deep learning ⊃ generative AI)
`{ "type": "nested", "title": "…", "levels": [ { "name": "Artificial intelligence", "blurb": "30+ characters …", "examples": ["…"] }, … ≥ 3, OUTERMOST FIRST ] }`. Click a ring or its chip to read it.
### `sampling` (softmax with temperature, then top-k and top-p, with the formula's live numbers)
`{ "type": "sampling", "title": "…", "prompt": "Once upon a time in a land far, far", "tokens": [ { "t": "away", "logit": 4.2 }, … ≥ 5 unique ] }`. Three sliders (temperature, top-k, top-p); words cut by top-k/top-p grey out and the rest are renormalised; a button draws words and tallies 100 draws. The logits are made up: say so in `note`. Use it when a lesson explains how a model picks the next word.
### `chain` (the chain rule: a sentence's probability is the product of its next-word probabilities)
`{ "type": "chain", "title": "…", "sentences": [ { "label": "Sensible", "tokens": [ { "t": "The", "p": 0.2 }, … ≥ 3 ], "note": "optional" }, { "label": "Scrambled", "tokens": […] } ] }`. Shows the running product, the whole-sentence probability and the perplexity for each sentence. Probabilities are illustrative (0 < p ≤ 1): say so in `note`.
### `logbars` (sizes on a log scale, e.g. model parameters)
`{ "type": "logbars", "title": "…", "unit": "parameters", "items": [ { "label": "GPT-2", "year": "2019", "value": 1.5e9, "note": "…" }, … ≥ 4 ] }`. Only use numbers you are sure of.

### `network` (a small neural network with a live forward pass)
`{ "type": "network", "title": "…", "intro": "…", "layers": [2,3,1], "weights": [ [[w for each input] per unit of layer 1], [[…] per unit of layer 2] ], "biases": [[b per unit of layer 1],[b of layer 2]], "inputs": [1,0], "activation": "relu|tanh|sigmoid|linear", "outputActivation": "linear|sigmoid|tanh|relu", "inputLabels": ["x1","x2"], "outputLabels": ["score"], "presets": [ { "name": "1, 0", "inputs": [1,0] } ], "editable": true, "note": "…", "checks": [ { "inputs": [1,1], "output": [0.0] } ] }`. 3–5 layers, 1–5 units per layer. `weights[l][j][i]` is the weight from unit `i` of layer `l` to unit `j` of layer `l+1`. The student picks any circle and sees `z = w·a + b` and `a = φ(z)` with the actual numbers, can change inputs and weights, and switch the hidden activation. The validator recomputes every entry of `checks` (output to ±0.01) — put in at least two so your stated numbers are verified.

### `fit` (underfitting and overfitting)
`{ "type": "fit", "title": "…", "intro": "…", "degree": 1, "maxDegree": 9, "train": [ {"x":-1,"y":-0.8}, … 6–14 points ], "test": [ … 4–10 points ], "note": "…" }`. A slider sets the polynomial degree 0–9; the curve is fitted by least squares to the training points and the figure shows training and test mean squared error. Authored data should follow a smooth curve plus noise so that test error is lowest at a middle degree and rises at high degree. Never state numbers from this figure in the text unless you computed them with `polyFit`/`polyMse` from `frontend/src/dc/lessonMath.js`.

### `trainer` (a tiny network learning in the browser)
`{ "type": "trainer", "title": "…", "intro": "…", "dataset": "xor|circle|moons|line", "hidden": 4, "lr": 0.5, "note": "…", "checks": [ { "dataset": "xor", "hidden": 1, "lr": 0.5, "epochs": 1000, "maxAcc": 0.8 }, { "dataset": "xor", "hidden": 4, "lr": 3, "epochs": 1000, "minAcc": 0.95 } ] }`. 2 inputs → N tanh units → 1 sigmoid output, full-batch gradient descent. The student picks the data set, hidden units 1–8 and learning rate, and presses Train. `checks` are verified by the validator with the same maths. Measured values: XOR with 1 hidden unit tops out at 0.75 accuracy, with 2 or more reaches 1.0 (lr 0.5, 1000 epochs); circle and moons reach 1.0 with 4 hidden units.

### `mlp` (count the parameters of a stack of layers)
`{ "type": "mlp", "title": "…", "intro": "…", "layers": [784,128,10], "note": "…" }`. Sliders change the hidden widths; a table shows weights (in × out), biases and parameters per layer and the total. Measured: [784,128,10] has 101,770 parameters.

### `calc` kind `split`
`{ "type": "calc", "kind": "split", "title": "…" }`: sliders for dataset size, training and validation share; shows train / validation / test counts.
### `posenc` (sinusoidal positional encoding, computed live): `{ "type": "posenc", "title": "…", "dModel": 16, "positions": 24 }`
### `patch` (image → patches → tokens)
`{ "type": "patch", "title": "…", "image": [ [0,0,9,9,…], … ], "channels": 1, "patchSizes": [2,4,8] }` with a square matrix of integers 0–9 (e.g. 8×8 or 16×16).
### `diffusion` (forward noising slider and an idealised reverse)
`{ "type": "diffusion", "title": "…", "image": ["0000099999…", … ], "t": 300 }`: 6–16 digit strings forming a square picture (a simple shape: smiley, arrow, letter).
### `retrieval` (TF-IDF search over a handful of passages, then the RAG prompt)
```json
{ "type": "retrieval", "title": "…", "query": "How many days do students have to return a book?", "topK": 2,
  "examples": ["…", "…"], "docs": [ { "id": "d1", "title": "Library loans", "text": "≥ 30 characters …" }, … ≥ 4 docs ],
  "checks": [ { "query": "…", "top": "d1" } ] }
```
`checks` is required: the validator ranks each query and fails if a different passage comes first. Write passages that differ clearly so keyword search gives a sensible answer, and include one example query where it fails (e.g. a paraphrase using different words) and say so in `note`.

### Reused from Week 1
`temperature` (`{ "type": "temperature", "prompt": "…", "tokens": [ { "t": "away", "logit": 4.2 }, … ≥ 4 ] }`) is for next-token sampling topics.

## 4. The quiz (` ```quiz `)

```json
{ "id": "week-04-check", "title": "Week 4 quiz: Convolutional neural networks", "passMark": 70,
  "questions": [ { "q": "…", "options": ["…","…","…","…"], "answer": 2, "why": "…" } ] }
```
- `id` = `week-NN-check` (at most 34 characters; it becomes the key the best score is saved under).
- 8–12 questions (up to 15 for an introductory week), 3–4 options each, `answer` is the 0-based index of the correct option.
- **Vary the position of the correct answer** (the validator warns when more than half sit in one slot). Wrong options must be plausible and each wrong for one clear reason; no "all of the above"; no trick wording; options similar in length.
- `why` always says why the right answer is right and, where useful, why a tempting wrong one is wrong. At least one question asks the student to apply or compute something (a number from the week's figures), not just recall.
- Cover every "By the end, you can" outcome. Do not copy questions from the Teaching Kit's question bank (GA-N-Qxx); write your own.

## 5. The lab section

Heading `## Lab: <name> {rail=Laboratory}`. The course outline names the lab for each week (e.g. "GA Lab 2", "DL Lab 1"). The Teaching Kit's lab notebooks are not available to you, so **write a self-contained activity that a student can finish in about 3 hours**, in PyTorch or Hugging Face `transformers`, on the DGX Spark. Do not claim the code comes from the Kit. Say "your instructor's lab notebook is the graded version" only if you are unsure. Structure:

1. One `labobj` sentence (first paragraph of the section) with a ==highlighted== objective.
2. `### Setup`: no internet, no `pip install`, models from the offline Hugging Face cache (`HF_HOME`), GPU is shared, free memory after each part. **Model IDs must be real, public Hugging Face IDs you are certain of**; add the sentence "Confirm with your instructor that every model ID below is in the offline cache."
3. `### Step-by-step tasks`: numbered list; each step has a time estimate, a short runnable code block (≤ 25 lines, standard libraries only: `torch`, `numpy`, `transformers`, `matplotlib`, `scikit-learn`), and an *Exercise N* the student answers in writing. Code must be correct: read it twice, and where possible run a CPU version in your head or in Python.
4. `### What you should see`: concrete expected results (numbers or shapes) that you have verified.
5. `### Common mistakes`: 3–5 real pitfalls.
6. `### Go further: UvA notebook` (where the course map for that week lists one, §6): a short paragraph and the link. Say what the notebook adds and what it needs (it downloads data and checkpoints from the internet, so students run it on a laptop or ask the instructor to pre-download it for the Spark). Credit it: "Tutorial N by Phillip Lippe, University of Amsterdam (UvA Deep Learning Tutorials, MIT licence)".
7. `### Deliverables`: what to submit through the portal, plus a short 100-point or 10-point marking table.

Never invent a portal URL, due date, room, instructor name, or grade weight. Dates and weights are announced by the instructor; say so.

## 6. UvA notebooks you may point students to (links verified to exist)

Base: `https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/`. All by Phillip Lippe (UvA Deep Learning Tutorials, MIT licence). They run in PyTorch (most need PyTorch Lightning) and download datasets/checkpoints, so they need internet access.

| Week | Notebook and URL tail | What it covers (from the page) |
|---|---|---|
| 2 | Tutorial 2, Introduction to PyTorch: `tutorial2/Introduction_to_PyTorch.html` | Tensors, autograd and the computation graph, GPU, a small `nn.Module` (linear, tanh, linear) trained on a noisy XOR dataset with `BCEWithLogitsLoss` and SGD (lr 0.1), `DataLoader`, saving and evaluating, TensorBoard. No dataset download. |
| 3 | Tutorial 3, Activation functions: `tutorial3/Activation_Functions.html` | Sigmoid, Tanh, ReLU, LeakyReLU, ELU and Swish with their gradients; gradient flow after initialisation (sigmoid gradients are tiny in early layers); training on FashionMNIST (sigmoid stays near chance, others reach about 87–89%); dead ReLU neurons. |
| 3 | Tutorial 4, Optimization and initialization: `tutorial4/Optimization_and_Initialization.html` | Constant vs constant-variance initialisation, why weight variance should be about 1/d_x, Xavier and Kaiming; SGD, momentum and Adam implemented from scratch; ravines and steep optima on toy surfaces. FashionMNIST. |
| 4 | Tutorial 5, Inception, ResNet, DenseNet: `tutorial5/Inception_ResNet_DenseNet.html` | Parallel 1×1/3×3/5×5 branches with 1×1 reductions, residual connections x_{l+1} = x_l + F(x_l), dense connectivity, parameter and accuracy comparison on CIFAR10. |
| 7–8 | Tutorial 6, Transformers and Multi-Head Attention: `tutorial6/Transformers_and_MHAttention.html` | Attention as a softmax-weighted average; scaled dot-product attention `softmax(QKᵀ/√d_k)V` and why the √d_k scaling matters; multi-head attention; the encoder block (LayerNorm(x + MHA(x)), then LayerNorm(x + FFN(x))); sinusoidal positional encoding; learning-rate warm-up; experiments: reversing a sequence (the heads learn to attend to the flipped position) and set anomaly detection on CIFAR100 features. Classes: `scaled_dot_product`, `MultiheadAttention`, `EncoderBlock`, `TransformerEncoder`, `PositionalEncoding`. |
| 11 | Tutorial 15, Vision Transformers: `tutorial15/Vision_Transformer.html` | Image to patches (`img_to_patch`), linear projection, CLS token, learned positional embeddings, an MLP head; trained on CIFAR10 it reaches about 75% where CNNs reach about 90%, because ViTs lack CNN inductive biases and need more data or pre-training. |
| 11 | Tutorial 17, SimCLR: `tutorial17/SimCLR.html` | Contrastive learning: pull two augmented views of one image together and push other images apart (InfoNCE loss, cosine similarity, temperature 0.07), ResNet-18 encoder and projection head on STL10, then a linear probe on frozen features. The idea behind CLIP-style training. |
| 12 | Tutorial 9, Autoencoders: `tutorial9/AE_CIFAR10.html` | Encoder with strided convolutions, decoder with `ConvTranspose2d` (stride 2) upsampling 4×4 → 32×32, reconstruction loss, effect of latent size (64–384), and why a vanilla autoencoder is not a generative model. CIFAR10. |
| 17 | Tutorial 10, Adversarial attacks: `tutorial10/Adversarial_Attacks.html` | FGSM and adversarial patches against a pretrained ResNet34, transfer to another model, and defences. |

Weeks not in the table (1, 5, 6, 9, 10, 13–16, 18–20) have no UvA notebook: do not invent one. Week 5, 10, 15, 20 are exam weeks and 19 is the capstone build week (see §7).

## 7. Exam weeks (5, 10, 15, 20) and Week 19

These weeks have no new concept lecture. Use the same skeleton, with these changes:
- `**Purpose:**` is about preparing well and knowing what is covered. Outcomes are things the student can do on the exam ("Compute…", "Explain…") drawn from the **weeks the exam covers**, taken from COURSE_OUTLINE.md.
- Concept sections become review sections, e.g. one per covered week or per theme, each with a `{lede}`, highlights, key terms, and a figure that revisits an earlier idea (`sorter`, `timeline`, `pipeline`, `heatmap`, `curve`…).
- `## Lab:` is replaced by `## Study plan {rail=Study plan}` (a checklist-style plan: what to review, in what order, with the Mock Exam Prep practice at `/mock-exam`) and, where the outline says so (Week 15 capstone proposal, Week 19 capstone build, Week 20 Demo Day), a `## Capstone …` section pointing to `/capstone`. Keep the heading words `Key takeaways` and `Check your understanding` and the quiz (10–12 cumulative questions across the covered weeks; ids `week-05-check` etc.).
- Never state the exam date, time, room, length, number of questions, weights or rules of the Prelim, Midterm, Pre-final or Final exam; the outline does not give them. Write "your instructor announces the date, room and format". You may state what the outline gives: the exams are cumulative for the term; Week 19's mock NCA-GENL exam is 50 multiple-choice questions, 60 minutes, closed book, pass at 35/50 (70%); NCA-GENL is issued by NVIDIA after its own proctored exam and Supervaise cannot issue or guarantee it; the capstone is the Structural Inspection Assistant team project (details in COURSE_OUTLINE.md).

## 8. Accuracy rules (important)

- Be right. If you are not sure of a fact, a number, a date, a model name or a paper's claim, leave it out or say what you are sure of. Never invent a citation, statistic, benchmark score or quote. Papers you may name (only for the claim stated): Rosenblatt 1958 (Perceptron); Rumelhart, Hinton & Williams 1986 (backpropagation); LeCun et al. 1998 (LeNet); Krizhevsky et al. 2012 (AlexNet); He et al. 2015 (ResNet); Mikolov et al. 2013 (word2vec); Sennrich et al. 2016 (BPE for NMT); Vaswani et al. 2017 (Transformer); Devlin et al. 2018 (BERT); Radford et al. 2018/2019 (GPT, GPT-2); Brown et al. 2020 (GPT-3); Raffel et al. 2019 (T5); Kaplan et al. 2020 (scaling laws); Hoffmann et al. 2022 (Chinchilla, about 20 tokens per parameter); Wei et al. 2022 (chain-of-thought); Dosovitskiy et al. 2020 (ViT); Radford et al. 2021 (CLIP); Goodfellow et al. 2014 (GAN); Ho et al. 2020 (DDPM); Rombach et al. 2022 (latent diffusion/Stable Diffusion); Hu et al. 2021 (LoRA); Ouyang et al. 2022 (InstructGPT/RLHF); Lewis et al. 2020 (RAG); Yao et al. 2022 (ReAct); Shoeybi et al. 2019 (Megatron-LM); Micikevicius et al. 2017 (mixed precision). Quote no numbers from them that you are not certain of.
- Everything illustrative (made-up logits, 2-D embeddings, toy attention weights) must say so in the figure `note`.
- Compute every number in the text and every quiz answer yourself (use Python). The validator re-checks scatter analogies, retrieval rankings, BPE merges and convolution sizes.
- Keep claims proportionate: "often", "usually", "in the original paper" instead of absolutes.
- Content must be safe for a school course: no instructions for harm; discuss risks at the level of concepts and mitigations.

## 9. Length and balance

A lesson is about 150–260 lines of Markdown plus JSON: 3–5 concept sections, 4–7 figures including the hook, a 3-hour lab, a 8–10 question quiz. Do not pad. Prefer one well-chosen figure per idea over many shallow ones.
