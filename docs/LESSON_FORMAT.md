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
A point may set `"hideLabel": true` to draw as a small dot with its label shown only when it is selected: use it for crowds of points that sit on top of each other (for example generated samples near the real clusters).

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

### `backprop` (forward pass, backward pass and the chain rule on a small network)
`{ "type": "backprop", "title": "…", "intro": "…", "layers": [2,2,1], "weights": [[[..],[..]],[[..]]], "biases": [[..],[..]], "inputs": [1,0.5], "target": [1], "activation": "sigmoid|tanh|relu|linear", "outputActivation": "sigmoid", "loss": "mse|bce", "lr": 0.5, "inputLabels": ["x1","x2"], "note": "…", "checks": [ { "kind": "loss", "value": 0.1545 }, { "kind": "delta", "layer": 2, "unit": 0, "value": -0.1 }, { "kind": "dW", "layer": 0, "unit": 0, "from": 1, "value": -0.0163, "tol": 0.001 } ] }`. 3–4 layers, 1–4 units each; `weights[l][j][i]` as in `network`. Tabs switch between the forward pass (circles show activations) and the backward pass (circles show δ = dL/dz and lines show |dL/dw|); selecting a circle shows its arithmetic; a button applies one gradient-descent step and lists the loss history. `mse` means L = ½Σ(a−y)²; `bce` needs a sigmoid output. `checks` (`kind` loss, delta with `layer`/`unit` (layer index 1..L), dW with `layer`/`unit`/`from` (layer 0 = weights into layer 1)) are recomputed by the validator (tol default 0.002): put in at least three.

### `gradflow` (vanishing and exploding gradients)
`{ "type": "gradflow", "title": "…", "intro": "…", "depth": 10, "activation": "sigmoid|tanh|relu", "scale": 1, "width": 16, "seed": 5, "presets": [ { "name": "He (1.41)", "scale": 1.41 } ], "note": "…", "checks": [ { "depth": 10, "activation": "sigmoid", "scale": 1, "maxRatio": 0.001 }, { "depth": 10, "activation": "relu", "scale": 2, "minRatio": 100 } ] }`. Bars show the Frobenius norm of dL/dW per layer on a log scale for one random network (weights N(0, scale²/inputs), zero biases, loss = sum of outputs). Measured at depth 10, width 16: sigmoid at scale 1 → layer 1 is about 2e-6 times the last layer; tanh at 1 stays within a factor of about 1; ReLU at 1.41 stays within a factor of about 2; ReLU at scale 2 → layer 1 about 2× the last layer, with all gradients about 25 times larger than at scale 1.41; ReLU at 0.7 shrinks all of them about 500-fold. Always back claims with `checks`.

### `fold` (why depth helps)
`{ "type": "fold", "title": "…", "intro": "…", "layers": 2, "note": "…" }`. A slider stacks 1–6 "tent" layers (two ReLU units each); the zig-zag has 2^k straight pieces. A one-hidden-layer ReLU network with m units has at most m + 1 pieces, so it needs at least 2^k − 1 units.

### `convgrad` (the backward pass of a convolution)
`{ "type": "convgrad", "title": "…", "intro": "…", "input": [[..]], "kernel": [[..]], "upstream": [[..]], "stride": 1, "padding": 0, "note": "…", "checks": [ { "kind": "y", "i": 0, "j": 0, "value": 0 }, { "kind": "dx", "i": 1, "j": 2, "value": 0 }, { "kind": "dK", "i": 1, "j": 0, "value": 3 } ] }`. Shows x ∗ K = y and then, for an upstream gradient G = dL/dy (same size as y), the gradients dL/dK and dL/dx. Selecting an input cell lists the terms `G × K` that make dL/dx; selecting a kernel cell lists the terms `G × x` that make dL/dK. Convolution here is the deep-learning kind (cross-correlation, no kernel flip). The validator checks the sizes and recomputes `checks` (kind `y`, `dx`, `dK`).

### `filters` (a kernel applied to a small picture)
`{ "type": "filters", "title": "…", "intro": "…", "image": ["0000000000", …≥6 rows of digits 0–9], "presets": [ { "name": "vertical edges", "kernel": [[-1,0,1],[-2,0,2],[-1,0,1]], "note": "…" }, … ], "relu": false, "padding": 0, "note": "…" }`. Preset chips plus a "your own" kernel (click a cell to cycle −2..2) and a ReLU toggle; the feature map is drawn as colours (green positive, blue negative).

### `pool` (max and average pooling)
`{ "type": "pool", "title": "…", "intro": "…", "input": [[..]], "size": 2, "stride": 2, "note": "…" }`. The window and stride must tile the input exactly. Tabs switch between max and average; selecting an output cell shows its window and arithmetic.

### `cnnstack` (shapes, parameters and receptive field)
`{ "type": "cnnstack", "title": "…", "intro": "…", "input": [28,28,1], "layers": [ {"type":"conv","k":5,"filters":6,"pad":2}, {"type":"pool","k":2}, {"type":"conv","k":5,"filters":16}, {"type":"pool","k":2}, {"type":"flatten"}, {"type":"fc","units":120}, {"type":"fc","units":84}, {"type":"fc","units":10} ], "checks": [ { "layer": 0, "shape": [28,28,6], "params": 156, "rf": 5 }, { "total": 61706 } ] }`. A table of output shapes, parameters (k·k·C_in·filters + filters for conv; in·out + out for fc) and receptive field; the student changes kernel size, stride, padding and filters per conv layer (`stride` defaults to 1, `pad` to 0; a pool's stride defaults to its size). `checks` entries use `layer` (index into `layers`) with `shape`/`params`/`rf`, or a bare `total`. The example above is LeNet-style: 156, 2,416, 48,120, 10,164, 850 parameters, 61,706 in total.

### `vocabtrade` (characters, sub-words or whole words as tokens)
`{ "type": "vocabtrade", "title": "…", "intro": "…", "corpus": [["low",5],["lower",2],["newest",6],["widest",3]], "text": "the lowest and the newest", "maxMerges": 12, "merges": 4, "note": "…", "checks": [ { "merges": 4, "vocabSize": 14, "textTokens": 9, "unk": 1, "wordVocab": 4 } ] }`. The corpus is [lowercase word, count] pairs (as in `bpe`); the sample `text` is lowercase letters and spaces. A slider sets the number of BPE merges; three cards compare the number of tokens and the vocabulary size at character level, BPE level and word level (unseen words count as unknown at word level), the pieces of each word are shown, and a chart plots vocabulary size against merges. The student can type any sentence. `checks` (any of `vocabSize`, `textTokens`, `unk`, `wordVocab`, with `merges` and optionally another `text`) are recomputed by the validator: use them for every number the text quotes.

### `vecsim` (dot product, length and cosine similarity of word vectors)
`{ "type": "vecsim", "title": "…", "intro": "…", "dims": ["animal","food","size"], "words": [ { "label": "cat", "v": [0.9,0.1,0.2] }, … ≥3, equal length ], "a": 0, "b": 1, "note": "…", "checks": [ { "a": "cat", "b": "dog", "cos": 0.99, "dot": 0.9 } ] }`. Two drop-downs choose words; a table shows each entry, the per-dimension products, the dot product, the two lengths and the cosine. Say in the `note` that the vectors are illustrative.

### `calc` kind `ngram_space`
`{ "type": "calc", "kind": "ngram_space", "title": "…" }`: sliders for vocabulary size V, n and the size of the training text; shows V^n possible n-grams against the number of words, and the share that can ever appear (the sparsity problem).

### `seqmemory` (how much of step t still reaches the end: RNN, LSTM, attention)
`{ "type": "seqmemory", "title": "…", "intro": "…", "length": 20, "w": 0.9, "forget": 0.95, "seed": 3, "note": "…", "checks": [ { "length": 30, "w": 0.9, "rnnFirstMax": 1e-9, "forget": 0.95, "lstmFirst": 0.2259 } ] }`. Sliders set the sequence length T, the RNN recurrent weight w and the LSTM forget gate f; a log-scale chart shows d(final state)/d(state at step t) for a one-unit tanh RNN on a fixed ±1 input (product of w·(1 − h²)), for an LSTM cell state (f^(T−t)) and for attention (a direct path, 1). Measured: T = 30, w = 0.9 gives about 3.6e-11 for step 1; T = 10 gives 7.8e-5; LSTM at f = 0.95, T = 30 gives 0.226 and at f = 0.99 gives 0.747. `checks` with `rnnFirstMax`/`rnnFirstMin` (bounds on step 1's RNN influence) and `forget` + `lstmFirst` are recomputed by the validator.

### `lstmcell` (one LSTM step with its four gates)
`{ "type": "lstmcell", "title": "…", "intro": "…", "x": 1, "h": 0, "c": 0.5, "params": { "f": [0.5, 0.5, 1.0], "i": [1.0, 0.5, 0], "g": [1.0, 0.5, 0], "o": [0.5, 0.5, 0] }, "note": "…", "checks": [ { "x": 1, "hPrev": 0, "cPrev": 0.5, "f": 0.8176, "c": 0.9656 } ] }`. A scalar LSTM cell: each gate is `[w_x, w_h, bias]` and f, i, o use the sigmoid, g uses tanh; c_t = f·c_(t−1) + i·g, h_t = o·tanh(c_t). Sliders set x, h_(t−1) and c_(t−1). `checks` entries give the inputs `x`, `hPrev`, `cPrev` plus any of the outputs `f, i, g, o, c, h` (their expected values) and are recomputed (±0.002).

### `hardsoft` (hard attention picks one position, soft attention blends)
`{ "type": "hardsoft", "title": "…", "intro": "…", "labels": ["the","cat","sat"], "scores": [0.5, 2, 0.3], "values": [[1,0],[0,1],[1,1]], "temperature": 1, "note": "…", "checks": [ { "temperature": 1, "soft": [0.1, 0.8, 0.1], "softOut": [0.5, 0.5], "hardOut": [0, 1] } ] }`. Tabs switch between soft attention (softmax weights) and hard attention (a one-hot at the arg max); sliders change the scores (and the temperature in soft mode); the figure also shows how much the output moves when one score is nudged by 0.1 (smoothly for soft, in jumps or not at all for hard). `checks` are recomputed.

### `calc` kind `attn_cost`
`{ "type": "calc", "kind": "attn_cost", "title": "…" }`: sliders for sequence length n and head size d; shows the n × n score table, about 4·n²·d multiply-adds, and "1 step vs n steps" against an RNN.

### `tfblock` (parts, shapes and parameters of a decoder-only Transformer)
`{ "type": "tfblock", "title": "…", "intro": "…", "vocab": 50257, "d": 768, "heads": 12, "dff": 3072, "layers": 12, "ctx": 1024, "posLearned": true, "tied": true, "note": "…", "checks": [ { "total": 124439808, "block": 7087872, "attn": 2362368, "ffn": 4722432 } ] }`. A table of the parts (token embedding, positional encoding, multi-head attention, feed-forward, layer norms, final layer norm, output layer) with sizes, output shapes, parameters and copies; drop-downs change d_model, heads, d_ff, blocks and vocabulary; toggles switch learned positions and tied output; click a part for a one-line explanation. Counts include biases as in GPT-2: attention 4d² + 4d, feed-forward 2·d·d_ff + d_ff + d, two layer norms per block (4d) and a final layer norm (2d). Measured: the GPT-2 small setting above gives 124,439,808 parameters (embedding 38,597,376; learned positions 786,432; one block 7,087,872; 12 blocks 85,054,464). `checks` (any of `total`, `block`, `attn`, `ffn`, optionally overriding `vocab`, `d`, `dff`, `layers`, `ctx`, `posLearned`, `tied`) are recomputed by the validator.

### `causalmask` (the mask that hides the future)
`{ "type": "causalmask", "title": "…", "intro": "…", "tokens": ["the","cat","sat","down"], "scores": [[…n numbers] × n], "row": 3, "note": "…", "checks": [ { "causal": true, "row": 1, "col": 0, "value": 0.27 } ] }`. An n × n matrix of raw attention scores; tabs switch the causal mask on or off; the table shows the softmax weights (masked cells hatched), a row can be selected to see its scores, the −∞ step and the weights. `checks` give a `row`, `col`, `causal` flag and the expected weight (±0.002).

### `shiftloss` (teacher forcing: inputs, shifted targets and the loss at every position)
`{ "type": "shiftloss", "title": "…", "intro": "…", "tokens": ["the","cat","sat","on","the","mat"], "p": [0.2, 0.1, 0.4, 0.3, 0.15], "note": "…", "checks": [ { "loss": 1.6, "perplexity": 4.95 } ] }`. `p[i]` is the probability the model gave to `tokens[i+1]` after reading `tokens[0..i]` (length = tokens − 1; authored, flagged illustrative). Shows input row, target row (the input shifted by one), the probabilities, each −ln p, the mean loss and the perplexity; select a position to read its sentence. `checks` (`loss`, `perplexity`) are recomputed.

### `llmfamily` (encoder-only, decoder-only and encoder-decoder: masks and training tasks)
`{ "type": "llmfamily", "title": "…", "intro": "…", "tokens": ["the","cat","sat","on","mat"], "families": [ { "name": "Encoder-only (BERT)", "mask": "bidirectional", "maskName": "every token sees every token", "objective": "Masked language modelling: …", "input": "the [MASK] sat on mat", "target": "cat", "uses": "…", "examples": "BERT, RoBERTa" }, { "name": "Decoder-only (GPT)", "mask": "causal", … }, { "name": "Encoder-decoder (T5)", "mask": "prefix", "prefix": 3, … } ], "start": 0, "note": "…" }`. Tabs switch between families; each shows an attention-mask grid for the 4–10 tokens (`bidirectional` = all green, `causal` = lower triangle, `prefix` = the first `prefix` tokens see everything and later ones are causal), the training objective with an example input and target, uses and example models.

### `moe` (a router sends each token to a few experts)
`{ "type": "moe", "title": "…", "intro": "…", "tokens": ["the","cat","sat"], "scores": [[2,1,0.5,0.1],…], "k": 2, "d": 512, "dff": 2048, "note": "…", "checks": [ { "k": 2, "token": 0, "experts": [0,1], "weights": [0.731,0.269], "total": 16797696, "active": 4199424 } ] }`. `scores[i][j]` is the router's score of expert j for token i (authored, flagged illustrative). A slider sets top-k; the table shows the weight each chosen expert gets (softmax over the kept scores), bars show how many tokens each expert receives, and cards compare a dense feed-forward layer (`2·d·d_ff + d_ff + d` parameters) with all E experts stored and the k used per token. `checks` (`k`, `token`, `experts`, `weights`; and/or `total`, `active` for the whole layer) are recomputed.

### `scalinglaw` (the compute-optimal model size)
`{ "type": "scalinglaw", "title": "…", "intro": "…", "logC": 23.77, "logN": 10.8, "note": "…", "checks": [ { "N": 7e10, "D": 1.4e12, "loss": 1.974 }, { "C": 5.76e23, "optN": 7.2e10, "optRatio": 18.4 } ] }`. Sliders set the compute budget C (log10 FLOPs, with C = 6·N·D) and a model size N; the curve shows loss against N at that budget (D = C/6N), marks the optimum from the parametric loss `L = E + A/N^α + B/D^β` and compares it with the rule N = √(C/120), D = 20·N. The constants are those of the re-fit by Besiroglu et al. (2024): E 1.8172, A 482.01, B 2085.43, α 0.3478, β 0.3658; they give about 72 B parameters and 1.33 T tokens (18.4 tokens per parameter) at C = 5.76 × 10²³, and loss 1.974 for 70 B parameters on 1.4 T tokens, 1.9996 for 280 B on 300 B tokens, 2.0085 for 175 B on 300 B. (The constants printed in the original paper for this form do not reproduce the paper's own 20 tokens per parameter: do not use them.) `checks` (`N`, `D`, `loss`; `C`, `optN`, `optRatio`) are recomputed.

### `emergent` (a smooth skill can look like a sudden jump)
`{ "type": "emergent", "title": "…", "intro": "…", "length": 8, "curve": { "pMin": 0.3, "pMax": 0.99, "mid": 9.5, "width": 0.9 }, "note": "…", "checks": [ { "logN": 9.5, "length": 8, "p": 0.645, "em": 0.03 } ] }`. Per-token accuracy p rises as a logistic in log10(parameters); exact match on an answer of L tokens is p^L. A slider changes L: the longer the answer, the more the exact-match curve looks like a sudden jump while the partial-credit curve stays smooth. Always say in the `note` that the curves are illustrative. `checks` are recomputed.

### `calc` kind `vit_tokens`
`{ "type": "calc", "kind": "vit_tokens", "title": "…", "d": 768 }`: sliders for the image side and patch size (and a colour/greyscale toggle) show the patch grid, the sequence length with the CLS token, the numbers in one flattened patch, the patch-embedding parameters (`patch·patch·channels·d + d`) and the attention score count against a pixel-per-token alternative. Measured: 224 × 224 colour image, 16 × 16 patches, d = 768 → 14 × 14 = 196 patches, sequence 197, 768 numbers per patch, 590,592 parameters, 38,809 attention scores per head against 2,517,630,976 for 50,176 pixel tokens.

### `clipsim` (a similarity matrix between images and texts; zero-shot and the contrastive loss)
`{ "type": "clipsim", "title": "…", "intro": "…", "scale": 10, "images": [ { "label": "photo of a cat", "v": [0.9, 0.1, 0.2] }, … ≥2 ], "texts": [ { "label": "a photo of a cat", "v": [1, 0, 0.1] }, … ≥2, same length ], "note": "…", "checks": [ { "scale": 10, "cos": [[0,0,0.98]], "prob": [[0,0,0.99]], "loss": 0.0003 } ] }`. Image i and text i are a true pair. The table shows cosine similarities; selecting an image shows the zero-shot probabilities `softmax(scale · cosine)` over the texts; cards show the image→text, text→image and average contrastive loss (mean −ln of the diagonal probabilities). A slider sets the logit scale (1/temperature). Say in the `note` that the vectors are illustrative. `checks` entries `cos` and `prob` are lists of `[image, text, value]`; `loss` is the symmetric loss (±0.005).

### `patchshuffle` (a bag of patches has no layout: why ViTs need position embeddings)
`{ "type": "patchshuffle", "title": "…", "intro": "…", "image": [ [0,0,9,…], … square 8×8 or 16×16 integers 0–9 ], "patch": 4, "seed": 4, "note": "…" }`. Chips switch between the original and a fixed shuffle of the patches and toggle position embeddings on or off; the text states that without positions the model sees the same set of tokens (a bag) and with positions the token sequence changes.

### `fusion` (how a model can combine modalities)
`{ "type": "fusion", "title": "…", "intro": "…", "patterns": [ { "name": "Two encoders, one space (CLIP)", "inputs": ["image","text"], "stages": ["image encoder","text encoder","compare vectors"], "output": "similarity score", "how": "…", "examples": "…", "good": "…", "bad": "…" }, … ≥2 ], "note": "…" }`. Tabs show each pattern as a row of chips (inputs → stages → output) with an explanation, examples, strengths and weaknesses.

### `ganlab` (a one-dimensional GAN trained live in the browser)
`{ "type": "ganlab", "title": "…", "intro": "…", "mu": 4, "sigma": 1, "lr": 0.1, "lrG": 0.02, "dSteps": 5, "nonSat": true, "seed": 11, "note": "…", "checks": [ { "steps": 3000, "meanMin": 3.3, "meanMax": 4.5, "spreadMin": 0.5, "spreadMax": 1.6 } ] }`. Real data N(mu, sigma²); generator G(z) = a·z + b; discriminator D(x) = σ(w₁x + w₂x² + c); each step is `dSteps` discriminator updates then one generator update with batch 64. The figure shows the real and generated densities and D(x), the step count, the two losses, sliders for both learning rates and the number of D steps per G step, a Train / +10 steps / Reset set of buttons, and a toggle between the original and the non-saturating generator loss. Measured with the defaults above: after 300 / 1,000 / 3,000 steps the generator is (a 0.99 / 0.89 / 1.06, b 3.41 / 3.91 / 3.74): the mean is learned quickly and the spread wobbles. Other settings collapse the spread towards 0 (for example lr 0.05 with 1 D step: a ≈ 0.03 after 1,500 steps): training is a tug of war. `checks` run the same simulation in the validator and bound the generator's mean (`b`) and spread (`|a|`) after `steps` steps (optionally overriding `lr`, `lrG`, `dSteps`, `nonSat`).

### `upsample` (nearest, bilinear and transpose-convolution upsampling)
`{ "type": "upsample", "title": "…", "intro": "…", "input": [[1,2,3],[4,5,6],[7,8,9]], "start": 4, "max": 9, "note": "…" }`. A 3×3 to 5×5 grid is upsampled by 2 by five methods (tabs): nearest neighbour, bilinear (PyTorch's `align_corners=False`), and transpose convolutions with kernel 2/stride 2, kernel 3/stride 2 and kernel 4/stride 2/padding 1 (the DCGAN setting). For the transpose convolutions a second grid counts how many kernel copies reach each output cell: uneven counts (kernel 3, stride 2) are the checkerboard artefact.

### `calc` kind `gan_loss`
`{ "type": "calc", "kind": "gan_loss", "title": "…" }`: two sliders for the discriminator's logits on a fake and a real picture; shows the discriminator's two loss terms, the value V = ln D(x) + ln(1 − D(G(z))) (equilibrium −ln 4 = −1.386) and the generator's original loss ln(1 − D) against the non-saturating −ln D with their gradients with respect to the logit (−D and D − 1): the original gradient vanishes when D(fake) ≈ 0.

### `cnnstack` with transpose convolutions
Layers may also be `{ "type": "reshape", "shape": [4,4,512] }` (a no-parameter reshape of the previous output into H×W×C: put it after an `fc` layer) and `{ "type": "convT", "k": 4, "stride": 2, "pad": 1, "filters": 256, "outputPadding": 0 }` (a transpose convolution: output size `(n − 1)·stride − 2·pad + k + outputPadding`, parameters `k·k·C_in·filters + filters`). The receptive-field column shows "–" for these layers. Example (the DCGAN generator): input `[1,1,100]`, `fc` 8192 (827,392), `reshape` [4,4,512], convT 256 (2,097,408) → 8×8, 128 (524,416) → 16×16, 64 (131,136) → 32×32, 3 (3,075) → 64×64×3.

### `schedule` (the noise schedule and the signal-to-noise trade)
`{ "type": "schedule", "title": "…", "intro": "…", "T": 1000, "kind": "linear", "t": 300, "note": "…", "checks": [ { "kind": "linear", "t": 300, "abar": 0.3964 }, { "kind": "linear", "t": 1000, "abar": 4.04e-5 } ] }`. Tabs switch between the linear schedule (β_t from 0.0001 to 0.02 over T steps) and the cosine schedule of Nichol and Dhariwal (ᾱ_t = f(t)/f(0), f(t) = cos²(((t/T + s)/(1 + s))·π/2), s = 0.008, β clipped at 0.999); a slider sets t; the chart shows ᾱ_t, √ᾱ_t (weight of the picture), √(1 − ᾱ_t) (weight of the noise) and the other schedule's ᾱ_t. Measured: linear ᾱ(300) = 0.3964, ᾱ(1000) = 4.04e-5; cosine ᾱ(300) = 0.7869, ᾱ(500) = 0.4938. `checks` (`kind`, `t`, and `abar` and/or `snr`) are recomputed (1%).

### `sampler1d` (the reverse process run with the exact denoiser)
`{ "type": "sampler1d", "title": "…", "intro": "…", "means": [-2, 3], "weights": [0.5, 0.5], "sd0": 0.4, "T": 200, "n": 600, "seed": 5, "stochastic": true, "note": "…", "checks": [ { "meanMin": 0.2, "meanMax": 0.8, "leftMin": 0.45, "leftMax": 0.55 } ] }`. 1-D data from a mixture of Gaussians; the forward process is the linear schedule with β from 0.001 to 0.06 over T = 200 steps (ᾱ_T ≈ 0.002); n particles start as N(0, 1) noise and are denoised for T DDPM steps using the exact noise predictor (no network), `x_(t−1) = (x_t − β_t/√(1−ᾱ_t)·ε̂)/√(1−β_t) + √β_t·z`. A slider or Play shows the histogram at every step against the true density; a toggle removes the fresh noise z. Measured with the defaults: the final mean is about 0.53 (the data mean is 0.5), 49% of the particles end in the left group and the spread inside a group is about 0.42. `checks` bound the final mean and the share in the left group.

### `unet` (shapes, skips and parameters of a small UNet denoiser)
`{ "type": "unet", "title": "…", "intro": "…", "size": 16, "inCh": 1, "base": 16, "levels": 3, "timeDim": 32, "note": "…", "checks": [ { "total": 135713 }, { "layer": "up block 1", "params": 7472 } ] }`. A diagram of the U (down blocks, bottleneck, up blocks, dashed skip connections) and a table of output shapes and parameters; selects change base channels, levels and the time-embedding size. Conventions (one block = two 3×3 convolutions plus a time projection `timeDim·C + C`; downsampling = 3×3 stride-2 convolution; upsampling = 2×2 stride-2 transpose convolution; the decoder concatenates the skip so the channels double before the up block; a final 1×1 convolution): the default (16 × 16 greyscale, base 16, 3 levels, time size 32) has 135,713 parameters (time MLP 2,112; down block 1 3,008; down block 2 14,944; bottleneck 57,536; up block 2 28,768; up block 1 7,472). `checks` (`total`, or `layer` name prefix with `params`, optionally overriding `base`, `levels`, `timeDim`) are recomputed.

### `prefpair` (a reward model ranks a chosen and a rejected answer)
`{ "type": "prefpair", "title": "…", "intro": "…", "pairs": [ { "prompt": "…", "chosen": "…", "rejected": "…", "chosenScore": 1.5, "rejectedScore": -0.5 }, … ≥2 ], "note": "…", "checks": [ { "pair": 0, "prob": 0.8808, "loss": 0.1269 } ] }`. Each pair has a slider for the reward of each answer; the figure shows `P(chosen preferred) = σ(r_chosen − r_rejected)`, the loss `−ln` of it, the mean loss and the share of pairs ranked correctly (a loss of ln 2 = 0.693 means "no idea"). `checks` (`pair`, or explicit `rw`/`rl`, with `prob` and/or `loss`) are recomputed. Prompts, answers and scores are authored: flag them as illustrative.

### `klreward` (reward against staying close to the original model)
`{ "type": "klreward", "title": "…", "intro": "…", "beta": 1, "answers": [ { "text": "…", "reward": 1, "ref": 0.5 }, … ≥3, refs sum to 1 ], "note": "…", "checks": [ { "beta": 1, "opt": [0.041, 0.067, 0.892], "J": 3.504 } ] }`. A table of answers with their reward and the original model's probability; sliders set the policy's logits; the cards show E[r], KL(π‖π_ref) and J = E[r] − β·KL; a slider sets β and a button jumps to the best policy π* ∝ π_ref·exp(r/β). Measured for refs (0.5, 0.3, 0.2) and rewards (1, 2, 5): β = 10 → π* = (0.443, 0.293, 0.264); β = 1 → (0.041, 0.067, 0.892); β = 0.2 → (0.000, 0.000, 1.000). `checks` (`beta` with `opt` and/or `J`) are recomputed.

### `lowrank` (why a low-rank update can be enough)
`{ "type": "lowrank", "title": "…", "intro": "…", "matrix": [[…], …], "rank": 2, "max": 3, "note": "…", "checks": [ { "sv": [4.28, 2.38] }, { "rank": 1, "err": 0.486 } ] }`. A 3–10 row matrix (a pretend fine-tuning update ΔW), a rank slider, the matrix next to its rank-r approximation (computed by singular value decomposition in the browser), bars for the singular values (kept green, dropped amber), the numbers needed (n·m against r·(n+m)) and the relative error. `checks` (`sv` = leading singular values; or `rank` with `err`, the relative Frobenius error) are recomputed.

### `calc` kind `ft_memory`
`{ "type": "calc", "kind": "ft_memory", "title": "…" }`: a slider for the model size N (billions of parameters), three buttons (full fine-tuning, LoRA with a 16-bit base, QLoRA with a 4-bit base) and, for the two adapter methods, a slider for the trainable share; shows weights, gradients, Adam state and the total. Full fine-tuning in mixed precision with Adam is 16 bytes per parameter (2 weights + 2 gradients + 4 master copy + 8 Adam moments); adapters: base weights (2 bytes, or 0.5 for 4-bit) plus 14 bytes per trained number. A 7 B model: full 112 GB, LoRA at 1% trainable 15.0 GB, QLoRA at 1% 4.5 GB (without activations).

### `chunker` (cut a document into overlapping chunks)
`{ "type": "chunker", "title": "…", "intro": "…", "text": "a 40–80 word document", "answer": "a phrase that appears word for word in the text", "question": "…", "size": 20, "overlap": 4, "note": "…", "checks": [ { "size": 20, "overlap": 4, "chunks": 4, "intact": true } ] }`. Two sliders (chunk size and overlap, in words); the text is shown with each word shaded by its chunk (a double underline marks words repeated in two chunks) and the answer phrase outlined. Shows the number of chunks `⌈(L − overlap) ÷ (size − overlap)⌉`, the words embedded in total, and whether the answer sits whole inside one chunk or is cut in two (a lesson in why chunking matters). `checks` (`size`, `overlap`, `chunks`, `intact`) are recomputed; write at least one check where the answer is cut.

### `rankmetrics` (precision, recall and reciprocal rank of a ranked list)
`{ "type": "rankmetrics", "title": "…", "intro": "…", "k": 3, "queries": [ { "label": "Question A", "question": "…", "totalRelevant": 2, "results": [ { "text": "…", "rel": true }, … ≥5, in ranked order ] }, … ≥2 ], "note": "…", "checks": [ { "query": 0, "k": 3, "precision": 0.667, "recall": 1, "rr": 1 } ] }`. Tabs choose a question; a slider sets k; the figure marks the top-k results and computes precision@k (relevant ÷ k), recall@k (relevant found ÷ `totalRelevant`) and the reciprocal rank of the first relevant result. `checks` are recomputed. Passages and labels are authored: flag them as illustrative.

### `calc` kind `rag_budget`
`{ "type": "calc", "kind": "rag_budget", "title": "…", "overhead": 150, "reserve": 300 }`: sliders for the context window, the number of retrieved passages k and the tokens per passage; shows overhead (instructions + question), retrieved context k × chunk, what is left for the answer and whether the prompt fits with `reserve` tokens to spare.

### `calc` kind `index_size`
`{ "type": "calc", "kind": "index_size", "title": "…" }`: sliders for documents (thousands) and chunks per document, buttons for 384/768/1536 numbers per vector; shows the number of vectors, the memory of a 32-bit index (vectors × dimension × 4 bytes) and the multiply-adds of one exact search.

### `railstack` (layers of guardrails, each missing some attacks)
`{ "type": "railstack", "title": "…", "intro": "…", "attacks": 1000, "rails": [ { "name": "Input rail", "where": "before the model", "catch": 0.7 }, { "name": "Dialog rail", "where": "steers the topic", "catch": 0.5 }, { "name": "Output rail", "where": "after the model", "catch": 0.6 } ], "note": "…", "checks": [ { "pass": 0.06 } ] }`. One slider per rail (share of attacks it catches); bars show what is left after each layer and the share that gets through every layer, the product of (1 − catch rate). `checks` (`pass`, optionally `catches`) are recomputed. Catch rates are authored: flag them as illustrative, and keep the independence caveat in `note`.

### `fgsm` (a signed-gradient attack on a linear image classifier)
`{ "type": "fgsm", "title": "…", "intro": "…", "labels": ["crack", "no crack"], "w": [ [8 numbers] × 8 rows ], "b": 0.2, "image": ["00000000", … 8 digit strings, 0–9 = darkness], "eps": 0.1, "note": "…", "checks": [ { "eps": 0.2, "before": 1.9, "score": -0.6, "flipped": true } ] }`. A made-up linear scorer s = w·x + b on a small square picture (score ≥ 0 is the first label). A slider sets ε; the figure shows the picture, the sign of the weights, the attacked picture x′ = clip(x ∓ ε·sign(w)), the score before and after, and the largest possible change ε·Σ|w|. `checks` (`eps`, with `before`, `score` and/or `flipped`) are recomputed.

### `railthreshold` (a detector's threshold: missed attacks against false alarms)
`{ "type": "railthreshold", "title": "…", "intro": "…", "attack": { "mean": 0.7, "sd": 0.15, "label": "attacks" }, "legit": { "mean": 0.3, "sd": 0.15, "label": "legitimate" }, "threshold": 0.5, "prevalence": 0.1, "note": "…", "checks": [ { "t": 0.5, "tpr": 0.909, "fpr": 0.091 } ] }`. Two bell curves of risk scores; a slider moves the threshold; the figure shows the share of attacks caught (`tpr`), missed, the share of legitimate messages blocked (`fpr`) and the precision when a share `prevalence` of messages are attacks. `checks` are recomputed (`tpr`, `fpr`, `precision`). Distributions are authored: flag them as illustrative.

### `agenttrace` (the loop of an agent, step by step, with the context growing)
`{ "type": "agenttrace", "title": "…", "intro": "…", "goal": "…", "tools": ["calculator", "lookup"], "C0": 600, "window": 4096, "steps": [ { "thought": "…", "action": "lookup[late fee]", "observation": "8 pesos", "outTokens": 30, "obsTokens": 6 }, … , { "answer": "…", "outTokens": 20 } ], "note": "…", "checks": [ { "upTo": 2, "context": 700, "read": 1250 } ] }`. Buttons step through the loop; the figure shows the thought, action and observation of each loop and the context the next call reads (C0 plus every earlier step) and the total tokens read (the sum of the context at each call). The last step must be the `answer`. `checks` (`upTo` model calls, with `context` and/or `read`) are recomputed.

### `calc` kind `agent_cost`
`{ "type": "calc", "kind": "agent_cost", "title": "…", "window": 8192 }`: sliders for the starting prompt, the number of loops and the tokens added per loop; shows the final context C0 + n·s, the total tokens read n·C0 + s·n(n − 1)/2, the ratio to a single call and whether it fits in `window`.

### `pipesched` (a pipeline schedule and its idle bubbles)
`{ "type": "pipesched", "title": "…", "intro": "…", "p": 4, "m": 4, "backward": true, "note": "…", "checks": [ { "p": 4, "m": 8, "backward": true, "total": 22, "bubble": 0.2727 } ] }`. Sliders for the number of pipeline stages p (2–8) and micro-batches m (1–16) and a switch for the backward pass; the figure draws the GPipe-style schedule (one row per GPU, one column per time step, coloured by micro-batch, backward steps outlined, idle cells grey) and shows the total time steps and the idle share (p − 1)/(m + p − 1). `checks` (`p`, `m`, `backward`, with `total` and/or `bubble`) are recomputed.

### `tensorsplit` (one matrix product split over several devices)
`{ "type": "tensorsplit", "title": "…", "intro": "…", "X": [[1,2,3,4],[0,1,0,2]], "W": [[…4 numbers] × 4 rows], "mode": "row", "n": 2, "note": "…", "checks": [ { "mode": "column", "n": 2, "Y": [[11,9,8,12],[2,3,3,4]] } ] }`. Buttons choose column split or row split and 2 or 4 devices; each device card shows its slice of W (and of X for the row split) and its result; the figure joins them (concatenate for columns, add for rows) and confirms the result equals X · W. `W` needs a number of rows and columns divisible by 4. `checks` (`mode`, `n`, `Y`) are recomputed.

### `numfmt` (floating-point formats: range, precision and rounding)
`{ "type": "numfmt", "title": "…", "intro": "…", "value": 0.1, "examples": [0.1, 3.14159, 70000, 1e-8, 1.0001], "note": "…", "checks": [ { "x": 0.1, "fmt": "FP16", "stored": 0.0999755859375 } ] }`. A box for a number; a table of FP32, TF32, BF16 and FP16 with the bit layout (sign · exponent · mantissa), the largest value, the smallest normal value, the relative step 2^−m, the stored value and the error (overflow shows ∞; values below the smallest subnormal become 0). `checks` (`x`, `fmt` among FP32, TF32, BF16, FP16, and `stored`, or `"inf"`) are recomputed.

### `quant` (round weights to fewer bits)
`{ "type": "quant", "title": "…", "intro": "…", "bits": 8, "group": 0, "sets": [ { "name": "Smooth weights", "w": [8–32 numbers] }, { "name": "One outlier", "w": […] } ], "note": "…", "checks": [ { "set": 0, "bits": 8, "group": 0, "mse": 3.66e-6 } ] }`. Buttons choose a weight set and a group size (one scale for all, or one per 8, 4 or 2 weights), a slider chooses 2–8 bits; the figure shows the integers, the restored values, the per-weight error and the mean squared error for symmetric quantization (scale = max|w| ÷ (2^(bits−1) − 1)). `checks` (`set`, `bits`, `group`, with `mse` and/or `q`) are recomputed.

### `batching` (latency against throughput when requests are batched)
`{ "type": "batching", "title": "…", "intro": "…", "t0": 40, "c": 5, "lambda": 50, "b": 8, "note": "…", "checks": [ { "b": 8, "lambda": 50, "latency": 150, "capacity": 100 } ] }`. Sliders for the batch size b and the arrival rate λ; a batch takes t0 + c·b ms, capacity is 1000·b ÷ batch time, the average wait to fill a batch is (b − 1)/(2λ) seconds and the figure plots the average latency for b = 1…64 (the lowest point marked, unstable sizes omitted). `checks` (`b`, `lambda`, with `latency` and/or `capacity`) are recomputed. Timings are authored: flag them as made up.

### `calc` kinds `dp_scaling`, `decode_speed`, `kv_cache`
`{ "type": "calc", "kind": "dp_scaling", "title": "…", "computeMs": 400, "bytesPerGrad": 2 }`: sliders for the model size, the number of GPUs and the link speed; shows the gradient bytes, the all-reduce traffic per GPU 2(N − 1)/N × gradients, the communication time and the efficiency compute ÷ (compute + communication). `{ "kind": "decode_speed" }`: model size, 16/8/4-bit buttons and memory bandwidth; shows the weight bytes and the upper bound tokens per second = bandwidth ÷ model bytes. `{ "kind": "kv_cache", "layers": 32, "d": 4096, "paramsB": 7 }`: tokens per sequence, batch size and 16/8-bit cache; shows the cache per token 2 × layers × d × bytes, for the batch, and as a multiple of the 16-bit weights.

### `confmatrix` (a confusion matrix and everything computed from it)
`{ "type": "confmatrix", "title": "…", "intro": "…", "classes": ["crack", "spalling", "corrosion", "none"], "matrix": [[…4 integers] × 4 rows, rows = true class, columns = predicted], "costs": [[0,1,1,5], …], "note": "…", "checks": [ { "accuracy": 0.85, "macroF1": 0.84, "recall": [0.9, …], "precision": [0.88, …], "cost": 0.4 } ] }`. A coloured table with − and + buttons on every count; shows per-class recall (row), precision (column) and F1, the accuracy, the macro F1, the worst recall and, when `costs` (same shape, cost of predicting column j when the truth is row i) is given, the expected cost per photo. `checks` (with `accuracy`, `macroF1`, `recall`, `precision`, `cost`) are recomputed from the initial matrix. Counts are authored: flag them as made up.

### `triage` (answer only when confident; send the rest for review)
`{ "type": "triage", "title": "…", "intro": "…", "items": [[0.98,1],[0.55,0], … ≥30 pairs of [confidence, correct 1 or 0]], "threshold": 0.7, "costWrong": 20, "costDefer": 1, "note": "…", "checks": [ { "t": 0.8, "coverage": 0.7, "accuracy": 0.95, "cost": 1.4 } ] }`. Two sliders (confidence threshold, cost of an unflagged wrong answer); shows the share answered (coverage), the accuracy of those answers, the number deferred, the expected cost per photo `(wrong × costWrong + deferred × costDefer) ÷ n` and a curve of the cost against the threshold (cheapest marked). `checks` (`t`, optionally `costWrong`, with `coverage`, `accuracy`, `cost`) are recomputed. Confidences and outcomes are authored: flag them as made up.

### `calc` kind `exam_plan`
`{ "type": "calc", "kind": "exam_plan", "title": "…" }`: sliders for the number of questions, the minutes and the pass mark; shows the seconds per question, the correct answers needed (rounded up), the mistakes you can afford and a first-pass pace (75 % of the time per question).

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
- `## Lab:` is replaced by `## Study plan {rail=Study plan}` (a checklist-style plan: what to review, in what order, with the Mock Exam Prep practice at `/mock-exam`) and, where the outline says so (Week 15 capstone proposal, Week 19 capstone build, Week 20 Demo Day), a `## Capstone …` section pointing to `/capstone`. Keep the heading words `Key takeaways` and `Check your understanding` and the quiz (10–12 cumulative questions across the covered weeks; ids `week-05-check` etc.). **Weeks 5 and 10 are the exception**: each is a full exam package (Part A written exam as an exam-mode quiz, `week-05-prelim` and `week-10-midterm`, Part B practical exam as the lab section; see those pages and the instructor keys in `backend/content/instructor/`).
- Never state the exam date, time, room, length, number of questions, weights or rules of the Prelim, Midterm, Pre-final or Final exam; the outline does not give them. Write "your instructor announces the date, room and format". You may state what the outline gives: the exams are cumulative for the term; Week 19's mock NCA-GENL exam is 50 multiple-choice questions, 60 minutes, closed book, pass at 35/50 (70%); NCA-GENL is issued by NVIDIA after its own proctored exam and Supervaise cannot issue or guarantee it; the capstone is the Structural Inspection Assistant team project (details in COURSE_OUTLINE.md).

## 8. Accuracy rules (important)

- Be right. If you are not sure of a fact, a number, a date, a model name or a paper's claim, leave it out or say what you are sure of. Never invent a citation, statistic, benchmark score or quote. Papers you may name (only for the claim stated): Rosenblatt 1958 (Perceptron); Rumelhart, Hinton & Williams 1986 (backpropagation); LeCun et al. 1998 (LeNet); Krizhevsky et al. 2012 (AlexNet); He et al. 2015 (ResNet); Mikolov et al. 2013 (word2vec); Sennrich et al. 2016 (BPE for NMT); Vaswani et al. 2017 (Transformer); Devlin et al. 2018 (BERT); Radford et al. 2018/2019 (GPT, GPT-2); Brown et al. 2020 (GPT-3); Raffel et al. 2019 (T5); Kaplan et al. 2020 (scaling laws); Hoffmann et al. 2022 (Chinchilla, about 20 tokens per parameter); Wei et al. 2022 (chain-of-thought); Dosovitskiy et al. 2020 (ViT); Radford et al. 2021 (CLIP); Goodfellow et al. 2014 (GAN); Ho et al. 2020 (DDPM); Rombach et al. 2022 (latent diffusion/Stable Diffusion); Hu et al. 2021 (LoRA); Ouyang et al. 2022 (InstructGPT/RLHF); Lewis et al. 2020 (RAG); Yao et al. 2022 (ReAct); Shoeybi et al. 2019 (Megatron-LM); Micikevicius et al. 2017 (mixed precision). Quote no numbers from them that you are not certain of.
- Everything illustrative (made-up logits, 2-D embeddings, toy attention weights) must say so in the figure `note`.
- Compute every number in the text and every quiz answer yourself (use Python). The validator re-checks scatter analogies, retrieval rankings, BPE merges and convolution sizes.
- Keep claims proportionate: "often", "usually", "in the original paper" instead of absolutes.
- Content must be safe for a school course: no instructions for harm; discuss risks at the level of concepts and mitigations.

## 9. Length and balance

A lesson is about 150–260 lines of Markdown plus JSON: 3–5 concept sections, 4–7 figures including the hook, a 3-hour lab, a 8–10 question quiz. Do not pad. Prefer one well-chosen figure per idea over many shallow ones.


### Exam-mode quiz (for the Prelim and other exam weeks)
The single ```` ```quiz ```` block may set `"exam": true` and `"minutes": 60`. The widget then runs as a timed exam: a Start screen with the rules (extra rules go in `"rules": ["…"]`), a countdown that submits the exam at zero, a question navigator, Flag for review, free movement back and forth, **no feedback until Submit**, then the score, a breakdown by topic and the explanation for every missed question. Every question needs `"topic": "Week 3: backpropagation"` (short, used for the breakdown). An exam may have 20–60 questions (a normal quiz 6–15); `passMark` and the other fields are as for any quiz. Best score and attempts are saved like any quiz.
