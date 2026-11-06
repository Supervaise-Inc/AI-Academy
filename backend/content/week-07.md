> Content based on the NVIDIA Generative AI and DLI Deep Learning Teaching Kits, used under the Supervaise × NVIDIA partnership.

# Week 7 — Language Models and Attention

**Purpose:** See why reading a text one word at a time makes a model forget, and how attention fixes it: ==instead of squeezing the past into one small memory, let every word look straight back at every other word and take a weighted mix of what it finds==.

**Term:** Midterm (Weeks 6–10) · **Month:** 2 · **Time:** about 60 minutes for the lesson, then a 3-hour lab

```viz
{
  "type": "hook",
  "title": "Watch a signal fade, and a word look back",
  "caption": "A recurrent network passes information from word to word, and each pass can shrink it. The numbers in the first scene are computed from a one-unit network. The sentence in the second scene is an example. The attention weights in the last three scenes are illustrative, authored for teaching, not taken from a trained model.",
  "scenes": [
    { "name": "A fading signal", "kind": "text", "outputLabel": "Computed from a one-unit recurrent network", "prompt": "How much of step 1 still reaches the end of a 30-step sequence?", "output": "With a recurrent weight of 0.9, a change at step 1 reaches step 30 multiplied by about 3.6 × 10⁻¹¹. That is one part in 28 billion. After only 10 steps the factor is 7.8 × 10⁻⁵. The network has not decided to forget. The signal simply gets multiplied by a number smaller than 1 at every step." },
    { "name": "One small bottleneck", "kind": "tokens", "outputLabel": "An encoder–decoder translator (an example)", "prompt": "The whole source sentence must fit in ONE vector", "tokens": ["The", "child", "you", "saw", "yesterday", "is", "my", "sister", "→ ONE fixed vector →", "decoder writes the translation"] },
    { "name": "Looking back", "kind": "bars", "outputLabel": "Attention of “it” (illustrative)", "prompt": "The animal didn’t cross the street because it was too tired", "bars": [ { "label": "animal", "value": 0.52 }, { "label": "tired", "value": 0.2 }, { "label": "street", "value": 0.14 }, { "label": "cross", "value": 0.08 }, { "label": "because", "value": 0.06 } ] },
    { "name": "Same word, new answer", "kind": "bars", "outputLabel": "Attention of “it” (illustrative)", "prompt": "The animal didn’t cross the street because it was too wide", "bars": [ { "label": "street", "value": 0.55 }, { "label": "wide", "value": 0.18 }, { "label": "animal", "value": 0.15 }, { "label": "cross", "value": 0.07 }, { "label": "because", "value": 0.05 } ] },
    { "name": "A direct road", "kind": "text", "outputLabel": "Computed: how much of step 1 reaches step 30", "prompt": "Three ways to carry a message across 30 steps", "output": "Plain recurrent network (w = 0.9): 3.6 × 10⁻¹¹. LSTM with a forget gate of 0.95: 0.226. Attention: 1, because step 30 can look at step 1 directly. This week you learn why each number is what it is." }
  ]
}
```

## By the end, you can…

1. ==Explain how a recurrent network (RNN) reads a sequence== one token at a time, and why its gradient shrinks over many steps.
2. ==Explain what the gates and the cell state of an LSTM add==, and which limits remain.
3. ==Explain the encoder–decoder bottleneck== and why letting the decoder look back at all the encoder states (attention) fixes it.
4. ==Compute attention scores, softmax weights and a weighted output by hand== for a query and three keys.
5. ==Distinguish hard attention from soft attention== and say why only soft attention can be trained by ordinary backpropagation.
6. ==Explain the scaling by √d_k and the n² cost== of attention, and what self-attention still lacks.

Assessed in the Week 7 lab and quiz. Read each section, then play with the figure under it: the figures are practice, so you cannot break anything. Terms with a dotted underline open a short definition when you select them. Notes such as “Lab Part 3” tell you which part of the lab lets you try the idea for real. In Week 6 you learned that a language model scores the next token and that every token becomes a vector. This week we ask how a model can use the words it read a long time ago, first with recurrent networks and LSTMs, and then with attention. The full Transformer block, positional encoding and multi-head attention belong to Week 8, so here we only name them. Every number in the text was computed, not guessed; the numbers inside the attention pictures are illustrative unless we say otherwise.

## Sequences and language models {rail=Sequences}

{lede} Language is a [[sequence|A list of items in which the order matters, such as the words of a sentence, the notes of a song or the readings of a sensor.]], and ==order changes meaning==. A model that reads text has to keep track of what it has already read, and that memory is exactly where the difficulty of this week begins.

### Order matters

“Dog bites man” and “man bites dog” contain the same three words. One is boring news and the other is a headline. A bag of words cannot tell them apart; ==a model that reads in order can==. The same is true everywhere in daily life: the digits of a phone number, the steps of a recipe, the words of a barangay announcement. A [[sequence model|A model that reads its input one item after another and keeps track of what came before, so that order can change the answer.]] is built for this.

### What you already know: language models and n-grams

In Week 6 you met the [[language model|A model that gives a probability to any piece of text by saying how likely each next token is, given the tokens before it.]]: it answers P(next token | history). Counting n-grams answered it with a table. The table looks only at a fixed window of the last n − 1 words, and it grows as V^n for a vocabulary of V words, so with V = 50,000 even trigrams have 1.25 × 10¹⁴ possible entries and almost all of them are never seen. Neural models avoid the table, but they still have to decide how much of the past to look at.

### Why a fixed window fails

Read this sentence slowly: “The keys that the teacher left on the old wooden desk near the window ___ missing.” Should the blank be “are” or “is”? The answer depends on “keys”, which is eleven words back. A trigram model that sees only “near the window” would pick whatever usually follows those words. To get this right, the model has to carry the plural “keys” across the whole gap.

You could widen the window, but a window of 12 words costs a table of V¹² entries, and the next sentence will need a window of 20. What we want instead is ==a model that reads one token at a time and keeps a short running summary of everything so far==. That summary is called the [[state|A vector that a sequence model carries from step to step. It summarises the history that the model has read so far.]] (also called the hidden state). At each step the model reads one token, updates the state, and uses the state to predict what comes next.

Lab Part 1 lets you try a fixed window and a memory task for yourself. Before we build that, sort some links by how far apart the two related words are.

```viz
{
  "type": "sorter",
  "title": "Short-range or long-range link?",
  "intro": "A link is a pair of words (or symbols) where the first one decides something about the second. Sort each example by the distance the model must remember across.",
  "bins": [ { "id": "short", "label": "Short-range link" }, { "id": "long", "label": "Long-range link" } ],
  "cards": [
    { "name": "“thank ___” → you", "answer": "short", "why": "The word that decides the blank is right next to it. A bigram table handles this." },
    { "name": "“The keys that the teacher left on the old wooden desk ___ missing”: are or is?", "answer": "long", "why": "The plural “keys” is about eleven words before the blank, with a whole clause in between." },
    { "name": "“She opened the door and walked in”: what was opened?", "answer": "short", "why": "“opened” and “door” are two words apart. A window of three words sees both." },
    { "name": "A story names “Ana” in sentence 1 and says “she” in sentence 6", "answer": "long", "why": "The pronoun must be linked back across several sentences, perhaps fifty or more words." },
    { "name": "“New ___” → York", "answer": "short", "why": "A common pair. The previous word almost fixes the next one." },
    { "name": "A closing quotation mark that matches an opening one thirty words earlier", "answer": "long", "why": "The model must remember that a quotation is still open for as long as it lasts." },
    { "name": "A chat: “As I asked at the start, ___” and the question was twenty messages ago", "answer": "long", "why": "The answer depends on a question far back in the conversation." },
    { "name": "“salamat ___” → po", "answer": "short", "why": "A very common two-word habit, like “thank you”: the previous word nearly decides the next." }
  ]
}
```

Notice that the long-range cards are the ones that matter for meaning, and they are exactly the ones a fixed window loses.

### A language model with a state

Here is the picture we will use all week. The model turns each token into an embedding (Week 6), feeds it to a sequence model together with the old state, gets a new state, and turns the state into next-token probabilities.

```viz
{
  "type": "pipeline",
  "title": "A sequence model reads one token at a time",
  "intro": "Step through one reading step. The state is the model’s only memory.",
  "vertical": false,
  "nodes": [
    { "id": "tok", "label": "Tokens", "sub": "Ana, ate, the" },
    { "id": "emb", "label": "Embeddings", "sub": "token → vector" },
    { "id": "seq", "label": "Sequence model", "sub": "old state + new vector" },
    { "id": "state", "label": "State", "sub": "summary so far" },
    { "id": "prob", "label": "Next-token probabilities", "sub": "softmax" }
  ],
  "steps": [
    { "title": "1. Read a token", "body": "The text is cut into tokens, as you saw in Week 6, and the model takes them one at a time, in order.", "active": ["tok"], "example": "“Ana” comes first" },
    { "title": "2. Look up its vector", "body": "Each token ID selects a row of the embedding matrix, a vector of numbers the network can compute with.", "active": ["tok", "emb"], "example": "“Ana” → (1, 0.5)" },
    { "title": "3. Combine with the old state", "body": "The sequence model mixes the new vector with the state left by the previous step. This is the only place where the past enters.", "active": ["emb", "seq", "state"], "example": "new state = f(old state, new vector)" },
    { "title": "4. Keep the new state", "body": "The state is passed on to the next step. Everything the model will ever know about the early words has to be stored in it.", "active": ["seq", "state"], "example": "the state is a small vector, for example 2 numbers here, hundreds in a real model" },
    { "title": "5. Predict the next token", "body": "A final layer turns the state into a probability for every token in the vocabulary. This is the language model’s answer.", "active": ["state", "prob"], "example": "P(ate) = 0.51, P(sat) = 0.40, P(mango) = 0.09 (illustrative)" }
  ],
  "note": "The vectors and probabilities are small illustrative numbers authored for teaching. A real state has hundreds of numbers."
}
```

Notice that the model never looks at the old tokens again. ==Only the state carries them forward==. The last step turns the state into probabilities by a [[softmax|A function that turns any list of scores into positive numbers that add up to 1, so they can be read as probabilities or weights.]], which you know from Week 1. Select each part of the formula below, then follow a numerical example (we meet the state’s numbers again in the next section).

```viz
{
  "type": "equation",
  "title": "Next-token probabilities from the state",
  "intro": "Select each symbol, then step through an example with a 2-number state and a 3-word vocabulary.",
  "formula": [
    { "t": "P(next | words so far)", "id": "p" },
    " ≈ softmax(",
    { "t": "W_o", "id": "wo" },
    " ",
    { "t": "h_t", "id": "h" },
    ")"
  ],
  "parts": [
    { "id": "p", "name": "P(next | words so far)", "meaning": "The probability of each possible next token, given everything the model has read. One number per token of the vocabulary, all adding to 1." },
    { "id": "wo", "name": "W_o (output weights)", "meaning": "A learned matrix with one row per vocabulary word. Each row says how the state votes for that word." },
    { "id": "h", "name": "h_t (the state)", "meaning": "The state after reading t tokens: a vector that must carry whatever the model needs from the history." }
  ],
  "steps": [
    { "title": "The state after “Ana”", "expr": "h = (0.664, 0.422)", "note": "This is the state our example network reaches after its first word, computed in the next section." },
    { "title": "One vote per word", "expr": "W_o rows: ate = (1, 0), sat = (0, 1), mango = (−1, −1);  logits = (0.664, 0.422, −1.086)", "note": "Each logit is the dot product of the word’s row with the state, for example 1×0.664 + 0×0.422 = 0.664." },
    { "title": "Softmax", "expr": "e^0.664 = 1.943, e^0.422 = 1.525, e^−1.086 = 0.338;  sum = 3.806", "note": "Exponentiate every logit and add them up." },
    { "title": "Divide by the sum", "expr": "P(ate) = 0.511, P(sat) = 0.401, P(mango) = 0.089", "note": "The three probabilities add up to 1. All numbers are illustrative." }
  ],
  "note": "The weights W_o and the state are made up for teaching."
}
```

Notice that the quality of the prediction can be no better than the information still in h_t. If the state has lost the word “keys”, no output layer can bring it back. That is the thread of the whole week.

## Recurrent networks (RNNs) {rail=RNN}

{lede} A [[recurrent neural network (RNN)|A network that reads a sequence one step at a time and feeds its own state from the previous step back in as an extra input, using the same weights at every step.]] ==updates its state from the previous state and the new input, using the same weights at every step==. It is the simplest model that fits the picture of the last section.

### The recurrence

Here is the update rule of the simplest RNN. Think of an index card on which you keep notes while reading a long text. Each time you read a new word, you look at the old card and the new word, and you write a fresh card. The weights W_h and W_x decide how the old card and the new word are blended, and the squashing function tanh keeps every number between −1 and 1, just as in Week 3.

```viz
{
  "type": "equation",
  "title": "The recurrence: h_t = tanh(W_h h_(t−1) + W_x x_t + b)",
  "intro": "Select each symbol, then step through one update with a 2-number state and a 2-number input.",
  "formula": [
    { "t": "h_t", "id": "ht" },
    " = ",
    { "t": "tanh", "id": "tanh" },
    "(",
    { "t": "W_h", "id": "wh" },
    " ",
    { "t": "h_(t−1)", "id": "hprev" },
    " + ",
    { "t": "W_x", "id": "wx" },
    " ",
    { "t": "x_t", "id": "xt" },
    " + ",
    { "t": "b", "id": "b" },
    ")"
  ],
  "parts": [
    { "id": "ht", "name": "h_t (new state)", "meaning": "The state after reading the t-th token: a vector of H numbers, each between −1 and 1." },
    { "id": "tanh", "name": "tanh", "meaning": "A squashing function (Week 3). It keeps each number of the state between −1 and 1, so the state cannot blow up in one step." },
    { "id": "wh", "name": "W_h (recurrent weights)", "meaning": "An H × H matrix that says how the old state feeds the new state. It is the memory path. The same W_h is used at every step." },
    { "id": "hprev", "name": "h_(t−1) (old state)", "meaning": "The state from the step before. At the very first step it is usually all zeros." },
    { "id": "wx", "name": "W_x (input weights)", "meaning": "An H × I matrix that turns the new input vector into a contribution to the state. The same W_x is used at every step." },
    { "id": "xt", "name": "x_t (input)", "meaning": "The embedding of the t-th token: a vector of I numbers (Week 6)." },
    { "id": "b", "name": "b (bias)", "meaning": "A vector of H numbers added before tanh, so that the state can start away from zero." }
  ],
  "steps": [
    { "title": "The numbers", "expr": "W_h = ( (0.5, −0.3), (0.2, 0.4) );  W_x = ( (0.8, −0.2), (0.3, 0.5) );  b = (0.1, −0.1);  h_(t−1) = (0.2, −0.4);  x_t = (1, 0.5)", "note": "A tiny network with H = 2 state numbers and I = 2 input numbers. All values are authored for teaching." },
    { "title": "The memory path W_h h_(t−1)", "expr": "(0.5×0.2 + (−0.3)×(−0.4), 0.2×0.2 + 0.4×(−0.4)) = (0.10 + 0.12, 0.04 − 0.16) = (0.22, −0.12)", "note": "What the old state contributes." },
    { "title": "The input path W_x x_t", "expr": "(0.8×1 + (−0.2)×0.5, 0.3×1 + 0.5×0.5) = (0.8 − 0.1, 0.3 + 0.25) = (0.70, 0.55)", "note": "What the new word contributes." },
    { "title": "Add the bias", "expr": "z = (0.22 + 0.70 + 0.1, −0.12 + 0.55 − 0.1) = (1.02, 0.33)", "note": "z is the value before the squashing function." },
    { "title": "Squash with tanh", "expr": "h_t = (tanh 1.02, tanh 0.33) = (0.770, 0.319)", "note": "The new state. Both numbers are between −1 and 1." },
    { "title": "Count the parameters", "expr": "H×H + H×I + H = 4 + 4 + 2 = 10", "note": "These ten numbers are all there is. They do not depend on how long the sequence is." }
  ],
  "note": "All weights are made-up teaching numbers. A real RNN learns them by training."
}
```

Lab Part 2 builds this recurrence by hand and checks it against PyTorch. Notice the last step: ==ten parameters, whether the sentence has 4 words or 4,000==. That is the second use of an idea from Week 4. A CNN slides one kernel over an image, so the same few weights are reused at every position ([[weight sharing|Using the same weights at every position (in a CNN) or every time step (in an RNN), so the number of parameters does not grow with the input.]]). An RNN reuses one set of weights at every position in time.

### Unrolling: the same cell, step after step

Draw the RNN once for each time step, left to right, and the loop turns into a chain. This is called [[unrolling|Drawing a recurrent network once per time step, so the loop becomes a long chain of identical copies that share the same weights.]]. Below we read the four-word sentence “Ana ate the mango” with h_0 = (0, 0) and the same weights as in the equation above, and watch the state change.

```viz
{
  "type": "pipeline",
  "title": "Reading “Ana ate the mango” with a 2-number state",
  "intro": "Step through the four words. The same weights are used at every step; only the input and the old state change.",
  "vertical": false,
  "nodes": [
    { "id": "x", "label": "Input x_t", "sub": "embedding of the word" },
    { "id": "cell", "label": "RNN cell", "sub": "same W_h, W_x, b" },
    { "id": "h", "label": "State h_t", "sub": "passed to the next step" },
    { "id": "y", "label": "Prediction", "sub": "from h_t" }
  ],
  "steps": [
    { "title": "Step 1: “Ana”", "body": "The state starts at zero. The word vector (1, 0.5) goes through the cell and a first state is written.", "active": ["x", "cell", "h"], "example": "x = (1, 0.5); h_0 = (0, 0)  →  h_1 = (0.664, 0.422)" },
    { "title": "Step 2: “ate”", "body": "Now the cell receives the new word vector and the state from step 1. The state mixes both.", "active": ["x", "cell", "h"], "example": "x = (0, 1)  →  h_2 = (0.105, 0.605)" },
    { "title": "Step 3: “the”", "body": "The same weights again. The information from “Ana” now passes through a second multiplication by W_h and a second tanh.", "active": ["x", "cell", "h"], "example": "x = (0.5, 0.5)  →  h_3 = (0.264, 0.510)" },
    { "title": "Step 4: “mango”", "body": "After four words, the state is the model’s only record of the whole sentence.", "active": ["x", "cell", "h", "y"], "example": "x = (1, −0.5)  →  h_4 = (0.753, 0.204)" },
    { "title": "Predict from the state", "body": "At any step the state can feed an output layer that predicts the next word. The prediction can only use what the state still holds.", "active": ["h", "y"], "example": "P(next | Ana ate the mango) = softmax(W_o h_4)" }
  ],
  "note": "The weights are the made-up numbers of the equation above. The states were computed from the recurrence."
}
```

Notice that the number of weights did not grow with the sentence; only the number of steps did. Notice also that by step 4 the first word has been through three more multiplications and squashings. Remember that.

### The output and the loss over time

To train the model, we use the next word as the label at every step. At step t the network outputs P(w_(t+1)) and we compare it with the word that actually came next, with the cross-entropy loss you know from Weeks 1 and 3. The total loss is the sum (or the average) over the steps.

```viz
{
  "type": "equation",
  "title": "The loss over time: L = −Σ ln P(correct next word at each step)",
  "intro": "Select each symbol, then add up the loss of a three-step example.",
  "formula": [
    { "t": "L", "id": "l" },
    " = −",
    { "t": "Σ", "id": "sum" },
    " ",
    { "t": "ln p_t", "id": "lnp" }
  ],
  "parts": [
    { "id": "l", "name": "L (total loss)", "meaning": "How badly the model predicted the whole sequence: the sum of the per-step losses. Lower is better." },
    { "id": "sum", "name": "Σ (sum over steps)", "meaning": "Add the loss of every time step, t = 1, 2, …, T." },
    { "id": "lnp", "name": "ln p_t", "meaning": "The natural logarithm of the probability the model gave to the correct next word at step t. A probability of 1 gives 0 loss, and a small probability gives a large loss." }
  ],
  "steps": [
    { "title": "Probabilities of the correct words", "expr": "p = (0.5, 0.25, 0.4)", "note": "What the model gave to the true next word at three steps (illustrative)." },
    { "title": "Loss at each step", "expr": "−ln 0.5 = 0.693;  −ln 0.25 = 1.386;  −ln 0.4 = 0.916", "note": "The least likely word costs the most." },
    { "title": "Total and average", "expr": "L = 0.693 + 1.386 + 0.916 = 2.996;  average = 2.996 / 3 = 0.999", "note": "The exponent of the average loss is the perplexity from Week 6: e^0.999 = 2.7." }
  ],
  "note": "The three probabilities are made up."
}
```

Notice that ==the loss depends on every step, so the gradient must flow back to every step==. That is the next topic.

### Backpropagation through time

You know backpropagation from Week 3: the chain rule carries the error from the output back to every weight. Applying it to an unrolled RNN is called [[backpropagation through time (BPTT)|Backpropagation applied to the unrolled recurrent network: the error is carried back through every time step, multiplying one factor per step.]]. To learn that the word at step t matters for the prediction at step T, the gradient has to travel from step T back to step t, and the chain rule multiplies one factor for each step on the way.

```viz
{
  "type": "equation",
  "title": "Gradient through time: ∂h_T/∂h_t = Π w·(1 − h_k²)",
  "intro": "This is the scalar version (one state number, recurrent weight w). Select each symbol, then multiply three factors.",
  "formula": [
    { "t": "∂h_T/∂h_t", "id": "grad" },
    " = ",
    { "t": "Π", "id": "pi" },
    " (from k = t+1 to T) ",
    { "t": "w", "id": "w" },
    " · ",
    { "t": "(1 − h_k²)", "id": "tanhslope" }
  ],
  "parts": [
    { "id": "grad", "name": "∂h_T/∂h_t", "meaning": "How much the last state changes when the state at step t is nudged by one unit. It is the strength of the path from step t to the end." },
    { "id": "pi", "name": "Π (product)", "meaning": "Multiply the factors for every step k between t+1 and T. If there are 29 steps between, there are 29 factors." },
    { "id": "w", "name": "w (recurrent weight)", "meaning": "The scalar form of W_h: the number by which the old state is multiplied at each step. The same w appears in every factor." },
    { "id": "tanhslope", "name": "1 − h_k² (slope of tanh)", "meaning": "The slope of tanh at step k. It is 1 when the state is 0 and falls towards 0 when the state is near −1 or 1. It is never larger than 1." }
  ],
  "steps": [
    { "title": "Three steps, w = 0.9", "expr": "states h_k = 0.5, −0.6, 0.3", "note": "Illustrative values of the state at the three steps after t." },
    { "title": "The three factors", "expr": "0.9 × (1 − 0.25) = 0.675;  0.9 × (1 − 0.36) = 0.576;  0.9 × (1 − 0.09) = 0.819", "note": "Each factor is less than 1 even though w is close to 1, because the tanh slope is below 1." },
    { "title": "Their product", "expr": "0.675 × 0.576 × 0.819 = 0.318", "note": "Over three steps about a third of the signal survives." },
    { "title": "Over 29 steps", "expr": "A typical factor of 0.436 → 0.436²⁹ ≈ 3.5 × 10⁻¹¹", "note": "That typical factor (0.436) is the geometric mean of the 29 factors in the figure below with w = 0.9; the exact product there is 3.6 × 10⁻¹¹." }
  ],
  "note": "The three states are illustrative; the 29-step figures come from the one-unit network used in the next figure."
}
```

Notice the Week 3 idea coming back. There, a sigmoid layer passed back at most one quarter, and 0.25 ten times gave 0.00000095. Here the multiplication is not across layers, it is ==across time==, and the factor is w times the tanh slope. The mathematics is the same product of factors below 1, only the direction is different.

Try it. The figure below follows a one-unit RNN reading 30 steps of ±1 inputs and plots, for each step t, how much of a nudge at step t still reaches the end.

```viz
{
  "type": "seqmemory",
  "title": "How much of step t still reaches step 30? (RNN, T = 30, w = 0.9)",
  "intro": "Read the log-scale chart from the right (the last step) to the left (step 1). Then drag w and the length T, and watch the red line.",
  "length": 30,
  "w": 0.9,
  "forget": 0.95,
  "seed": 3,
  "note": "A one-unit tanh network on a fixed ±1 input signal, computed by the same product as the equation above. The LSTM and attention lines are explained in the next sections.",
  "checks": [
    { "length": 30, "w": 0.9, "rnnFirstMax": 1e-10 },
    { "length": 10, "w": 0.9, "rnnFirstMax": 1e-4, "rnnFirstMin": 1e-5 },
    { "length": 30, "w": 1.0, "rnnFirstMax": 1e-9 },
    { "length": 30, "w": 0.5, "rnnFirstMax": 1e-15 },
    { "length": 30, "w": 0.9, "forget": 0.95, "lstmFirst": 0.2259 }
  ]
}
```

Here is what we verified. With w = 0.9, the influence of step 1 on step 30 is 3.6 × 10⁻¹¹, and over 10 steps it is 7.8 × 10⁻⁵. Make w smaller and it vanishes faster: at w = 0.5 and T = 30 it is 1.6 × 10⁻¹⁶. You may expect that a large w makes the signal explode, but in this figure it does not: at w = 1.0 it is still only 7.2 × 10⁻¹¹, and at w = 1.5 it is 7.4 × 10⁻¹². The reason is [[saturation|When a tanh unit receives a large input and its output sits near −1 or 1, where the curve is almost flat and the slope is close to 0.]]. A big w pushes the state to the flat parts of tanh, and the slope factor (1 − h²) crushes the product. In this figure the biggest value of step 1’s influence over all w is about 7 × 10⁻¹¹. Gradients that really explode do occur in other RNNs, for example with ReLU units, and Lab Part 3 measures both cases.

In words: after 30 steps, a change at step 1 reaches the end multiplied by one 28-billionth. The network cannot learn from it, so it cannot learn that the first word matters for the last. This is ==the vanishing gradient problem along time==.

### Vanishing and exploding, and the first-aid

When the product shrinks, the RNN forgets the beginning of the sequence. Think of the telephone game, or a message passed along a line of 30 jeepney passengers, each of whom whispers a slightly weaker version of what they heard: by the end, the first passenger’s words are gone. When the product grows, the gradient becomes enormous and training diverges; the first-aid from Week 3, [[gradient clipping|Rescaling the gradient whenever its length exceeds a limit, so that no single update can be enormous. The direction is kept.]], keeps the updates sane. But ==clipping cannot make a tiny gradient bigger==, so it does not solve forgetting. That needs a change in the design of the network.

### The other cost: one step after another

There is a second cost, separate from forgetting. Step t needs the state from step t − 1, which needs the state from step t − 2, and so on. ==The steps cannot be computed at the same time==, even on a machine with thousands of cores. For a 1,000-token text you must wait for 1,000 sequential steps. Sort these properties.

```viz
{
  "type": "sorter",
  "title": "Which problem is it?",
  "intro": "Some RNN troubles are about forgetting and others are about speed. Sort each statement into the problem it describes.",
  "bins": [ { "id": "forget", "label": "Forgetting (vanishing gradient)" }, { "id": "speed", "label": "Sequential speed" }, { "id": "both", "label": "Both" } ],
  "cards": [
    { "name": "After 30 steps, a change at step 1 barely reaches the end", "answer": "forget", "why": "The product of 29 factors below 1 is tiny. This is the vanishing gradient along time." },
    { "name": "Step 500 cannot start until step 499 is finished", "answer": "speed", "why": "The recurrence is sequential. More cores do not help across the time steps." },
    { "name": "The model cannot learn that the first word decides the last word", "answer": "forget", "why": "No gradient signal reaches the first word, so the weights never learn that link." },
    { "name": "Training a 2,000-token text takes 2,000 dependent steps", "answer": "speed", "why": "Each step waits for the previous one. The length of the chain sets the minimum time." },
    { "name": "Doubling the text length doubles the number of steps and weakens the long-range signal", "answer": "both", "why": "Twice as many sequential steps, and twice as many factors below 1 in the product." },
    { "name": "A change at an early word is multiplied by w and by a tanh slope at every later step", "answer": "forget", "why": "Each multiplication by a factor below 1 weakens the signal, so the early word fades from the gradient." },
    { "name": "A GPU can handle thousands of operations at once, but the RNN uses it poorly", "answer": "speed", "why": "Parallel hardware is wasted when each step depends on the one before." },
    { "name": "The state is the only place where early words are stored", "answer": "both", "why": "It makes the model forget when the signal fades, and it forces the steps to happen in order." }
  ]
}
```

Notice how both problems come from the same design choice: pass everything along a single chain. Let us put the history of the idea on a line before we move on.

```viz
{
  "type": "timeline",
  "title": "From recurrent networks to attention",
  "events": [
    { "year": "1990", "title": "Simple recurrent networks", "text": "Jeffrey Elman showed that a network with a recurrent state could learn structure in sequences, such as patterns in word order, in his paper “Finding structure in time”." },
    { "year": "1997", "title": "The LSTM", "text": "Sepp Hochreiter and Jürgen Schmidhuber introduced the long short-term memory network, designed to carry information over many steps." },
    { "year": "2014", "title": "Sequence-to-sequence learning", "text": "Sutskever and colleagues used one LSTM to read a sentence into a vector and another to write the translation." },
    { "year": "2015", "title": "Attention for translation", "text": "Bahdanau, Cho and Bengio (arXiv 2014, ICLR 2015) let the decoder look back at all the encoder states instead of one vector." },
    { "year": "2017", "title": "The Transformer", "text": "Vaswani and colleagues built a model on attention alone, with no recurrence, which made it practical to train very large language models." }
  ]
}
```

The next section shows how LSTMs tried to fix the forgetting, and what they could not fix.

## LSTM: gates and a memory highway {rail=LSTM}

{lede} A [[long short-term memory network (LSTM)|A recurrent network with a separate cell state and three learned gates that decide what to forget, what to write and what to show, so that information can travel many steps with little change.]] fixes the vanishing gradient by giving memory its own lane: ==a cell state that is carried forward with only small, gated changes==, instead of being rewritten completely at every step.

### The idea: a notebook with a gatekeeper

Think of a student who keeps two things while reading a long story: a short sentence in her head about what is happening right now (the state h_t), and a notebook in which she writes down facts she must not lose, such as the names of the characters (the [[cell state|The LSTM’s long-term memory: a vector that is carried from step to step and changed only a little at each step, under the control of the gates.]] c_t). At every new sentence she makes three decisions:

- **Forget gate (f):** Which old notes do I rub out? If the story has moved to a new chapter, she erases the old plan. If f is close to 1, she keeps everything. If f is close to 0, she erases it.
- **Input gate (i):** How much of the new information do I write down? A big surprise is worth a note, and an unimportant word is not.
- **Output gate (o):** Which notes do I say out loud right now? The notebook holds much more than she needs for the next word, and the output gate picks what to show.

Each gate is a number between 0 and 1 (for every number in the cell), produced by the [[sigmoid|The function σ(z) = 1 / (1 + e^(−z)), which squeezes any number into the range 0 to 1. Think of it as a dimmer switch: 0 is fully closed and 1 is fully open.]] function you know from Week 3. A gate is multiplied element by element with a vector (written ⊙), so a gate value of 0.9 lets 90% of a number through, and a gate value of 0.02 lets 2% through. We now write down the four equations, one by one, with a worked number each. The same example is used all the way: the input is x_t = 1, the old state is h_(t−1) = 0.5, the old cell is c_(t−1) = 0.8, and each gate has three learned numbers [w_x, w_h, b] so that its pre-activation is w_x·x + w_h·h + b (this is the scalar form of W [h, x] + b). All weights are authored for teaching.

### The forget gate

```viz
{
  "type": "equation",
  "title": "Forget gate: f_t = σ(W_f [h_(t−1), x_t] + b_f)",
  "intro": "Select each symbol, then compute the forget gate for one step.",
  "formula": [
    { "t": "f_t", "id": "f" },
    " = ",
    { "t": "σ", "id": "sigma" },
    "(",
    { "t": "W_f", "id": "wf" },
    " [",
    { "t": "h_(t−1)", "id": "hprev" },
    ", ",
    { "t": "x_t", "id": "xt" },
    "] + ",
    { "t": "b_f", "id": "bf" },
    ")"
  ],
  "parts": [
    { "id": "f", "name": "f_t (forget gate)", "meaning": "A number between 0 and 1 for each memory cell. Near 1 means keep the old note, near 0 means erase it." },
    { "id": "sigma", "name": "σ (sigmoid)", "meaning": "Squeezes any number into the range 0 to 1, so that the result can act as a dimmer switch." },
    { "id": "wf", "name": "W_f (forget weights)", "meaning": "Learned weights that read the old state and the new input and decide how much to forget." },
    { "id": "hprev", "name": "h_(t−1)", "meaning": "The old state: what the LSTM showed at the last step." },
    { "id": "xt", "name": "x_t", "meaning": "The new input vector (the embedding of the current token)." },
    { "id": "bf", "name": "b_f (forget bias)", "meaning": "A learned offset. A large positive bias makes the gate start near 1, which means that the cell remembers by default." }
  ],
  "steps": [
    { "title": "The numbers", "expr": "[w_x, w_h, b] = [0.4, 0.6, 1.5];  x = 1;  h_(t−1) = 0.5", "note": "The three learned numbers of the forget gate in the scalar cell, and the inputs of our example." },
    { "title": "The pre-activation", "expr": "z = 0.4×1 + 0.6×0.5 + 1.5 = 0.4 + 0.3 + 1.5 = 2.2", "note": "The weighted sum before the dimmer switch." },
    { "title": "The sigmoid", "expr": "f = σ(2.2) = 1 / (1 + e^−2.2) = 0.900", "note": "The gate lets 90% of the old note through." }
  ],
  "note": "The weights are made up for teaching."
}
```

Notice the bias of 1.5. A positive bias is a common choice in practice, because ==a cell that remembers by default is easier to train than one that forgets by default==.

### The input gate and the candidate

```viz
{
  "type": "equation",
  "title": "Input gate and candidate: i_t = σ(W_i [h, x] + b_i),  g_t = tanh(W_g [h, x] + b_g)",
  "intro": "The candidate g is the new note. The input gate i decides how much of it to write. Select each symbol, then compute both.",
  "formula": [
    { "t": "i_t", "id": "i" },
    " = σ(W_i [h, x] + b_i);   ",
    { "t": "g_t", "id": "g" },
    " = ",
    { "t": "tanh", "id": "tanh" },
    "(W_g [h, x] + b_g)"
  ],
  "parts": [
    { "id": "i", "name": "i_t (input gate)", "meaning": "A number between 0 and 1: how much of the candidate to write into the cell. Near 0 means ignore the new information." },
    { "id": "g", "name": "g_t (candidate)", "meaning": "The new note proposed at this step, a number between −1 and 1. It is only written to the cell in proportion to the input gate." },
    { "id": "tanh", "name": "tanh", "meaning": "Squeezes the candidate into the range −1 to 1, so the notes stay in a sensible size." }
  ],
  "steps": [
    { "title": "Input gate", "expr": "[w_x, w_h, b] = [0.8, 0.4, −0.5];  z = 0.8×1 + 0.4×0.5 − 0.5 = 0.5;  i = σ(0.5) = 0.622", "note": "A bit more than half of the candidate will be written." },
    { "title": "Candidate", "expr": "[w_x, w_h, b] = [1.2, 0.6, 0.1];  z = 1.2×1 + 0.6×0.5 + 0.1 = 1.6;  g = tanh(1.6) = 0.922", "note": "A strong positive note." }
  ],
  "note": "The weights are made up for teaching."
}
```

### Updating the cell: the heart of the LSTM

```viz
{
  "type": "equation",
  "title": "Cell update: c_t = f_t ⊙ c_(t−1) + i_t ⊙ g_t",
  "intro": "Select each symbol, then update the cell with the gates computed above.",
  "formula": [
    { "t": "c_t", "id": "c" },
    " = ",
    { "t": "f_t", "id": "f" },
    " ⊙ ",
    { "t": "c_(t−1)", "id": "cprev" },
    " + ",
    { "t": "i_t", "id": "i" },
    " ⊙ ",
    { "t": "g_t", "id": "g" }
  ],
  "parts": [
    { "id": "c", "name": "c_t (new cell state)", "meaning": "The notebook after this step: what is kept from before plus what is newly written." },
    { "id": "f", "name": "f_t", "meaning": "The forget gate: how much of the old notebook survives." },
    { "id": "cprev", "name": "c_(t−1)", "meaning": "The old cell state, the notebook as it was." },
    { "id": "i", "name": "i_t", "meaning": "The input gate: how much of the new note is written." },
    { "id": "g", "name": "g_t", "meaning": "The candidate: the new note." }
  ],
  "steps": [
    { "title": "What survives", "expr": "f × c_(t−1) = 0.900 × 0.8 = 0.720", "note": "Most of the old notebook is kept." },
    { "title": "What is written", "expr": "i × g = 0.622 × 0.922 = 0.574", "note": "Part of the new note is added." },
    { "title": "The new cell", "expr": "c_t = 0.720 + 0.574 = 1.294", "note": "Addition, not a full rewrite. This is the memory highway: the old cell reaches the new cell by a simple multiplication with f." }
  ],
  "note": "Numbers from the gates above."
}
```

Look at the shape of this equation. Compare it with the RNN, which wrote h_t = tanh(W_h h_(t−1) + …): there the old state is multiplied by a matrix and squashed at every step. Here the old cell is only multiplied by a gate and then added to. ==Addition is what lets the gradient flow back almost unchanged==.

### The output gate and the new state

```viz
{
  "type": "equation",
  "title": "Output: o_t = σ(W_o [h, x] + b_o),  h_t = o_t ⊙ tanh(c_t)",
  "intro": "The output gate decides which part of the notebook is shown at this step. Select each symbol, then compute the new state.",
  "formula": [
    { "t": "h_t", "id": "h" },
    " = ",
    { "t": "o_t", "id": "o" },
    " ⊙ ",
    { "t": "tanh(c_t)", "id": "tanhc" }
  ],
  "parts": [
    { "id": "h", "name": "h_t (new state)", "meaning": "What the LSTM shows to the next layer and to the next step. It is a filtered view of the cell." },
    { "id": "o", "name": "o_t (output gate)", "meaning": "A number between 0 and 1: how much of each memory cell to reveal now." },
    { "id": "tanhc", "name": "tanh(c_t)", "meaning": "The cell squashed into the range −1 to 1, so the shown values stay small." }
  ],
  "steps": [
    { "title": "Output gate", "expr": "[w_x, w_h, b] = [0.7, 0.3, 0.2];  z = 0.7×1 + 0.3×0.5 + 0.2 = 1.05;  o = σ(1.05) = 0.741", "note": "About three quarters of the cell will be shown." },
    { "title": "Squash the cell", "expr": "tanh(1.294) = 0.860", "note": "The cell value 1.294 is squashed." },
    { "title": "The new state", "expr": "h_t = 0.741 × 0.860 = 0.637", "note": "The LSTM’s new state is 0.637, and its cell is 1.294. Both go to the next step." }
  ],
  "note": "Same example as above. A real LSTM has hundreds of such cells, with matrices W instead of three numbers."
}
```

An LSTM with H state numbers and I inputs has four blocks of weights (forget, input, candidate, output), so 4 × (H×H + H×I + H) parameters. For H = 2 and I = 2 that is 40, four times the RNN’s 10. That is the price of the gates.

Now try the whole cell. The figure below has exactly the numbers we used.

```viz
{
  "type": "lstmcell",
  "title": "One LSTM step with its four gates (scalar cell)",
  "intro": "Start with x = 1, h = 0.5, c = 0.8 and read the gates. Then move the sliders for x, the old state and the old cell, and see how f, i, o, c and h respond.",
  "x": 1,
  "h": 0.5,
  "c": 0.8,
  "params": { "f": [0.4, 0.6, 1.5], "i": [0.8, 0.4, -0.5], "g": [1.2, 0.6, 0.1], "o": [0.7, 0.3, 0.2] },
  "note": "Each gate is [w_x, w_h, bias]. The weights are made up for teaching, not learned from data.",
  "checks": [
    { "x": 1, "hPrev": 0.5, "cPrev": 0.8, "f": 0.9002, "i": 0.6225, "g": 0.9217, "o": 0.7408, "c": 1.2939, "h": 0.6372 },
    { "x": -1, "hPrev": 0.5, "cPrev": 0.8, "f": 0.8022, "i": 0.2497, "g": -0.664, "o": 0.4134, "c": 0.4759, "h": 0.1831 }
  ]
}
```

At x = 1, h = 0.5 and c = 0.8 you should read f = 0.900, i = 0.622, g = 0.922, o = 0.741, c = 1.294 and h = 0.637, the same as our hand calculation. Now set x = −1: the input gate falls to 0.250, the candidate becomes −0.664, and the new cell is only 0.476. The cell reacts to the input, but the old note still contributes 0.8 × 0.802 = 0.642 of it.

### Making the gates obey the input

In a trained LSTM the gates learn when to keep and when to erase. The second cell has weights chosen to show both extremes by hand: the input x controls the forget gate. A positive x opens the forget gate and closes the input gate (keep the notebook as it is). A negative x does the opposite (erase the notebook and write something new).

```viz
{
  "type": "lstmcell",
  "title": "A cell that keeps or erases, depending on x",
  "intro": "Set the old cell to 0.8. With x = 1, watch c stay almost unchanged. Then set x = −1 and watch the old note vanish and a new one take its place.",
  "x": 1,
  "h": 0,
  "c": 0.8,
  "params": { "f": [4, 0, 0], "i": [-4, 0, 0], "g": [1, 0, 0], "o": [0.5, 0.5, 0] },
  "note": "Weights authored by hand to make the keep and erase behaviours visible. A trained LSTM learns such behaviour from data, in a less tidy way.",
  "checks": [
    { "x": 1, "hPrev": 0, "cPrev": 0.8, "f": 0.982, "i": 0.018, "c": 0.7993 },
    { "x": -1, "hPrev": 0, "cPrev": 0.8, "f": 0.018, "i": 0.982, "c": -0.7335 }
  ]
}
```

With x = 1 the forget gate is 0.982 and the input gate is 0.018, so the cell goes from 0.8 to 0.799: almost no change, the note is kept. With x = −1 the forget gate is 0.018 and the input gate is 0.982, so the cell becomes −0.734: the old note has been erased and replaced. This is the behaviour that lets an LSTM hold a fact for many steps and drop it when it stops being useful. Lab Part 4 shows the forget gates of a trained LSTM.

### Why it helps: the gradient along the cell

Look at the cell update again. The part of the gradient that travels along the cell state from step t to step T is just the product of the forget gates in between. Since the cell is only multiplied by f and added to, there is no tanh slope and no weight matrix in the way.

```viz
{
  "type": "equation",
  "title": "Gradient along the cell: ∂c_T/∂c_t = Π f_k",
  "intro": "Select each symbol, then compute how much of the cell at step 1 survives to step 30 for two forget gates.",
  "formula": [
    { "t": "∂c_T/∂c_t", "id": "grad" },
    " = ",
    { "t": "Π", "id": "pi" },
    " ",
    { "t": "f_k", "id": "fk" }
  ],
  "parts": [
    { "id": "grad", "name": "∂c_T/∂c_t", "meaning": "How much the final cell value changes when the cell at step t is nudged by one unit, along the cell-state path." },
    { "id": "pi", "name": "Π (product)", "meaning": "Multiply the forget gates of every step between t+1 and T. Between steps 1 and 30 there are 29 of them." },
    { "id": "fk", "name": "f_k (forget gate)", "meaning": "The forget gate at step k. If it is close to 1, almost nothing is lost; the gate can learn to be 1 when the information matters." }
  ],
  "steps": [
    { "title": "Forget gate 0.95", "expr": "0.95²⁹ = 0.226", "note": "About a quarter of the signal survives 29 steps." },
    { "title": "Forget gate 0.99", "expr": "0.99²⁹ = 0.747", "note": "Three quarters survive." },
    { "title": "Forget gate 0.9", "expr": "0.9²⁹ = 0.047", "note": "Still more than a billion times the 3.6 × 10⁻¹¹ of the RNN with w = 0.9." },
    { "title": "Compare with the RNN", "expr": "0.226 / 3.6×10⁻¹¹ ≈ 6.3 × 10⁹", "note": "The LSTM’s path carries about 6 billion times more of the signal from step 1 than the plain RNN of the previous section." }
  ],
  "note": "A constant forget gate is a simplification: real gates change from step to step, and other, weaker gradient paths exist as well."
}
```

Compare this with the RNN’s product of w · (1 − h²), each of which is below 1 by the design of the tanh. The forget gate can be close to 1, so the product can stay near 1. Play with it in the next figure, which now draws the LSTM line and the RNN line for a longer sequence.

```viz
{
  "type": "seqmemory",
  "title": "RNN against LSTM over 60 steps",
  "intro": "Here T = 60 and the forget gate is 0.99. Compare the red (RNN) and green (LSTM) lines at step 1. Then lower the forget gate and see how fast the green line drops.",
  "length": 60,
  "w": 0.9,
  "forget": 0.99,
  "seed": 3,
  "note": "The RNN line is the product of w·(1 − h²) for a one-unit tanh network. The LSTM line is f^(T−t) with a constant forget gate. The attention line (always 1) is the subject of the next section.",
  "checks": [
    { "length": 60, "w": 0.9, "rnnFirstMax": 1e-22, "forget": 0.99, "lstmFirst": 0.5527 },
    { "length": 60, "w": 0.9, "forget": 0.95, "lstmFirst": 0.0485 }
  ]
}
```

With T = 60 and f = 0.99, the LSTM keeps 0.553 of step 1 (0.99⁵⁹), while the RNN keeps 5 × 10⁻²⁴. With f = 0.95 the LSTM keeps only 0.049 at T = 60, which is much better than the RNN but already a small number: ==a forget gate that is close to 1 can hold a long memory==, but a gate that is merely “high” still lets it fade. ==The LSTM is much better, not perfect.==

A [[GRU|A gated recurrent unit: a simpler variant of the LSTM with two gates and no separate cell state. It often works about as well, with fewer parameters.]] is a simplified cousin of the LSTM with two gates; it follows the same idea of gated, mostly additive memory.

### What LSTMs still cannot do

The LSTM solved the vanishing gradient well enough that it became the standard sequence model for years. But three limits remain.

1. **Still sequential.** An LSTM step still needs the previous step’s result, so the cost of reading n tokens is n steps in a row, with no parallelism across time.
2. **A fixed-size state must squeeze the whole past.** However long the text, all of it goes into one cell vector of, say, 512 numbers. That is a bottleneck: something has to be thrown away to make room, and the model must decide in advance what will matter later.
3. **Long-range recall still degrades.** A forget gate that is 0.99 and not exactly 1 loses memory slowly but surely, and the model has to learn to set its gates correctly, which is not easy to train over hundreds of steps. And the gates make the training slower: four times the weights of an RNN of the same width.

The first answer here says what a plain RNN and an LSTM might each reply if you ask, after a long passage, “what was the first word?”. The replies are made up to show the idea.

```viz
{
  "type": "flow",
  "title": "“What was the first word?” after a long passage",
  "modes": [
    { "name": "RNN", "tagline": "Rewrites its whole state at every step", "inputLabel": "Passage (40 words)", "modelLabel": "RNN state (small vector)", "outputLabel": "Answer", "examples": [
      { "label": "Ask after 5 words", "input": "“Maria walked to the market …” (5 words so far). What was the first word?", "output": "“Maria” (still in the state)" },
      { "label": "Ask after 40 words", "input": "The same story, 40 words later. What was the first word?", "output": "A guess. The signal from word 1 has passed 39 multiplications by W_h and tanh." }
    ], "note": "The outputs are made up to show the pattern, not produced by a real model." },
    { "name": "LSTM", "tagline": "Keeps a gated cell state", "inputLabel": "Passage (40 words)", "modelLabel": "LSTM cell + state (small vectors)", "outputLabel": "Answer", "examples": [
      { "label": "Ask after 5 words", "input": "“Maria walked to the market …” (5 words so far). What was the first word?", "output": "“Maria”" },
      { "label": "Ask after 40 words", "input": "The same story, 40 words later. What was the first word?", "output": "Often “Maria”, if the forget gates stayed near 1 for the name. Less reliable if the cell had to make room for many other facts." },
      { "label": "Ask after 400 words", "input": "A long chapter. What was the first word?", "output": "Harder: everything must still fit in one fixed-size cell, and the gates must have kept the name through every step." }
    ], "note": "The outputs are made up to show the pattern, not produced by a real model." }
  ]
}
```

Notice that the LSTM answers more often, but its memory is still one small vector that every word competes for. Sort the properties below to check that you know who has what.

```viz
{
  "type": "sorter",
  "title": "RNN, LSTM or both?",
  "intro": "Each card is a property. Decide whether it belongs to the plain RNN, to the LSTM, or to both.",
  "bins": [ { "id": "rnn", "label": "RNN only" }, { "id": "lstm", "label": "LSTM only" }, { "id": "both", "label": "Both" } ],
  "cards": [
    { "name": "Has a separate cell state with forget, input and output gates", "answer": "lstm", "why": "The gates and the cell are what define an LSTM." },
    { "name": "Cannot compute step t before step t − 1 is finished", "answer": "both", "why": "Both are recurrent: each step needs the previous state." },
    { "name": "The gradient along its memory path is a product of forget gates", "answer": "lstm", "why": "∂c_T/∂c_t = Π f_k, where the forget gates can be close to 1." },
    { "name": "Rewrites its whole state through W_h and tanh at every step, so early information fades fast", "answer": "rnn", "why": "Every step multiplies by W_h and by a tanh slope below 1, and nothing protects the old state." },
    { "name": "Keeps all its memory of the past in a fixed-size vector", "answer": "both", "why": "However long the text is, both squeeze it into a vector of the same size, which is the bottleneck." },
    { "name": "Has four times as many parameters as the other for the same width", "answer": "lstm", "why": "Four blocks of weights (forget, input, candidate, output) against one for the RNN: 40 against 10 when H = 2 and I = 2." },
    { "name": "Can learn to hold a fact for many steps by setting the forget gate near 1 and the input gate near 0", "answer": "lstm", "why": "This is the keep behaviour you saw in the second cell figure." },
    { "name": "Reads the tokens in order, one at a time, with weights shared across steps", "answer": "both", "why": "Weight sharing across time is common to all recurrent networks." }
  ]
}
```

The LSTM bought us memory. It did not buy us speed, and it did not buy us a state that grows with the text. The next section starts from exactly this complaint.

## The bottleneck and the birth of attention {rail=Attention}

{lede} In the story of translation the bottleneck was easy to see: ==squeezing a whole sentence into one fixed-size vector throws information away==. [[Attention|A mechanism that lets a model look at all the positions of an input and take a weighted mix of them, with weights that depend on what it is currently looking for.]] solves it by letting the decoder look back at every part of the sentence whenever it needs to.

### The encoder–decoder model

To translate, researchers around 2014 used an [[encoder–decoder (sequence-to-sequence) model|A pair of recurrent networks: the encoder reads the source sentence and ends in a state; the decoder starts from that state and writes the translation one word at a time.]] (Sutskever and colleagues called it sequence-to-sequence learning). The encoder, an LSTM, reads the source sentence and ends with a final state. That state is called the [[context vector|The single fixed-size vector that the encoder hands to the decoder: its whole summary of the source sentence.]]. The decoder, another LSTM, starts from it and writes the translation word by word.

```viz
{
  "type": "pipeline",
  "title": "The encoder–decoder bottleneck",
  "intro": "Step through a translation. Watch how much must pass through the single arrow in the middle.",
  "vertical": false,
  "nodes": [
    { "id": "src", "label": "Source sentence", "sub": "5 words or 50" },
    { "id": "enc", "label": "Encoder", "sub": "reads word by word" },
    { "id": "ctx", "label": "Context vector", "sub": "ONE fixed-size vector" },
    { "id": "dec", "label": "Decoder", "sub": "writes word by word" },
    { "id": "tgt", "label": "Translation", "sub": "target words" }
  ],
  "steps": [
    { "title": "1. The encoder reads", "body": "The encoder reads the source words one at a time, updating its state at each word, exactly as in the previous sections.", "active": ["src", "enc"], "example": "The child you saw yesterday is my sister" },
    { "title": "2. Everything is squeezed", "body": "At the end only the final state is kept: a vector with the same number of entries for a 5-word sentence and for a 50-word sentence.", "active": ["enc", "ctx"], "example": "e.g. 512 numbers, whatever the length" },
    { "title": "3. The decoder starts from it", "body": "The decoder has no access to the source words any more. All it knows about the sentence is the context vector.", "active": ["ctx", "dec"], "example": "the decoder sees only the 512 numbers" },
    { "title": "4. The decoder writes", "body": "It writes the first target word, then the next, using only the vector and its own previous words. A detail lost in step 2 cannot be recovered.", "active": ["dec", "tgt"], "example": "target words appear one at a time" }
  ],
  "note": "The sentence and the vector size are examples. Real systems use vectors with hundreds or thousands of numbers."
}
```

Notice step 2. It is like asking a student to read a full page, write a summary on one sticky note, close the book, and then translate the page using only the sticky note. A short page fits on the note. A long page does not. Bahdanau, Cho and Bengio, who proposed the fix in 2014 (published at ICLR 2015), reported that the performance of this encoder–decoder model degraded as the sentences got longer.

### The fix: let the decoder look back

Their idea: ==do not throw away the encoder’s states==. The encoder produces one state per source word. Keep all of them. Then, whenever the decoder is about to write a word, let it look back at all the states and take a weighted mix, where the weights depend on what the decoder is currently writing. When it writes the verb, the weights are high on the source verb. When it writes the object, they move to the source object.

The picture is the student with an open book: instead of one sticky note, she may glance back at the highlighted page as often as she likes, and at each moment she looks at the lines that matter for the word she is writing right now.

### Queries, keys and values: a library analogy

Modern attention gives this idea three names. Imagine you go to the school library with a question, say “books about typhoons”. Every book has a label on its spine (a **key**) and contents inside (a **value**). You compare your question (the **query**) with each label. The labels that match best get most of your attention, and you take a blend of the contents of the best-matching books.

```viz
{
  "type": "flow",
  "title": "Library search and attention are the same recipe",
  "modes": [
    { "name": "In the library", "tagline": "A question, labels and books", "inputLabel": "Your question", "modelLabel": "Compare with every label", "outputLabel": "What you take away", "examples": [
      { "label": "Query", "input": "“Books about typhoons”", "output": "The question you carry. In attention this is the query vector q." },
      { "label": "Keys", "input": "Spine labels: “Typhoons”, “Rice farming”, “Volcanoes”", "output": "Each label is compared with the question. In attention these are the key vectors k₁, k₂, k₃." },
      { "label": "Values", "input": "The pages inside each book", "output": "You read mostly the best-matching book and a little of the others. In attention these are the value vectors v₁, v₂, v₃, blended by the weights." }
    ], "note": "An analogy for the idea. Real attention uses vectors of numbers, not words." },
    { "name": "In attention", "tagline": "q, keys k, values v", "inputLabel": "Query vector q", "modelLabel": "Dot product with every key, then softmax", "outputLabel": "Weighted sum of the values", "examples": [
      { "label": "Score", "input": "q · kᵢ for every position i", "output": "A number for each position: how well its key matches the query." },
      { "label": "Weights", "input": "softmax of the scores", "output": "Weights between 0 and 1 that add up to 1." },
      { "label": "Output", "input": "Σ weightᵢ × vᵢ", "output": "One vector: the blend of the values, mostly from the best match." }
    ], "note": "The three steps are made precise below." }
  ]
}
```

The same three steps make attention, in order: compute a score for each position, turn the scores into weights with a softmax, and take the weighted sum of the values.

```viz
{
  "type": "pipeline",
  "title": "Attention in four steps",
  "intro": "Step through score, softmax, weights and weighted sum for one query.",
  "vertical": false,
  "nodes": [
    { "id": "q", "label": "Query q", "sub": "what I look for" },
    { "id": "k", "label": "Keys k₁ … kₙ", "sub": "what each position offers" },
    { "id": "s", "label": "Scores", "sub": "q · kᵢ ÷ √d_k" },
    { "id": "w", "label": "Weights", "sub": "softmax, sum to 1" },
    { "id": "o", "label": "Output", "sub": "Σ wᵢ vᵢ" }
  ],
  "steps": [
    { "title": "1. Score", "body": "Compare the query with every key by a dot product. A large dot product means that the key matches the query well (Week 6’s similarity).", "active": ["q", "k", "s"], "example": "q·k = (2×1) + (1×1) = 3" },
    { "title": "2. Scale", "body": "Divide each score by √d_k, the square root of the key length, so that the scores do not grow with the size of the vectors. We explain why in the last section.", "active": ["s"], "example": "3 / √2 = 2.12" },
    { "title": "3. Softmax", "body": "Turn the scores into positive weights that add up to 1, exactly as the sampling figure of Week 1 turned logits into probabilities.", "active": ["s", "w"], "example": "weights 0.284, 0.140, 0.576" },
    { "title": "4. Weighted sum", "body": "Multiply each value vector by its weight and add. The result is the output for this query: a mix of values dominated by the best matches.", "active": ["w", "o"], "example": "output = 0.284 v₁ + 0.140 v₂ + 0.576 v₃" }
  ],
  "note": "The numbers are the worked example below."
}
```

Notice that nothing in these four steps depends on how far away the positions are. The query compares itself with every key directly. This is the key difference from a recurrent network.

### The formula, with a full worked example

```viz
{
  "type": "equation",
  "title": "Scaled dot-product attention: softmax(QKᵀ/√d_k) V",
  "intro": "For one query q this means: weights = softmax(q·kᵢ / √d_k), output = Σ weightᵢ vᵢ. Select each symbol, then follow a complete example with three tokens.",
  "formula": [
    "Attention(",
    { "t": "Q", "id": "Q" },
    ", ",
    { "t": "K", "id": "K" },
    ", ",
    { "t": "V", "id": "V" },
    ") = ",
    { "t": "softmax", "id": "softmax" },
    "(",
    { "t": "QKᵀ", "id": "qk" },
    " / ",
    { "t": "√d_k", "id": "scale" },
    ") ",
    { "t": "V", "id": "V2" }
  ],
  "parts": [
    { "id": "Q", "name": "Q (queries)", "meaning": "One row per position that is asking: the vector that says what this position is looking for." },
    { "id": "K", "name": "K (keys)", "meaning": "One row per position that can be looked at: the vector that says what this position offers, used to be compared with the queries." },
    { "id": "V", "name": "V (values)", "meaning": "One row per position: the information that is passed on when the position is attended to." },
    { "id": "softmax", "name": "softmax", "meaning": "Turns each row of scores into positive weights that add up to 1." },
    { "id": "qk", "name": "QKᵀ (score matrix)", "meaning": "All the dot products at once: entry (i, j) is the score of query i against key j. For n positions it is an n × n table." },
    { "id": "scale", "name": "√d_k (scaling)", "meaning": "The square root of the length of a key vector. Dividing by it keeps the scores at a steady size however long the vectors are." },
    { "id": "V2", "name": "V (values again)", "meaning": "The weights from the softmax multiply the values: output row i is the sum over j of weight(i, j) times value j." }
  ],
  "steps": [
    { "title": "The setup", "expr": "q = (2, 1);  keys k₁ = (1, 1), k₂ = (0, 2), k₃ = (2, 0);  values v₁ = (1, 0), v₂ = (0, 1), v₃ = (2, 2);  d_k = 2", "note": "Three tokens, with 2-number vectors. All the numbers are authored for teaching." },
    { "title": "1. Scores q·kᵢ", "expr": "q·k₁ = 2×1 + 1×1 = 3;  q·k₂ = 2×0 + 1×2 = 2;  q·k₃ = 2×2 + 1×0 = 4", "note": "The third key matches the query best." },
    { "title": "2. Scale by √2 = 1.414", "expr": "3/1.414 = 2.121;  2/1.414 = 1.414;  4/1.414 = 2.828", "note": "The scaled scores." },
    { "title": "3. Softmax", "expr": "e^2.121 = 8.34, e^1.414 = 4.11, e^2.828 = 16.92;  sum = 29.37;  weights = 0.284, 0.140, 0.576", "note": "The weights add up to 1.000." },
    { "title": "4. Weighted sum, first number", "expr": "0.284×1 + 0.140×0 + 0.576×2 = 0.284 + 1.152 = 1.436", "note": "The first entry of the output." },
    { "title": "4. Weighted sum, second number", "expr": "0.284×0 + 0.140×1 + 0.576×2 = 0.140 + 1.152 = 1.292", "note": "The second entry. The output is (1.436, 1.292)." },
    { "title": "Without the scaling", "expr": "softmax(3, 2, 4) = 0.245, 0.090, 0.665;  output = (1.575, 1.421)", "note": "With only two numbers per vector, the scaling matters little. With 64 numbers it matters a lot (last section)." }
  ],
  "note": "All vectors are authored for teaching. In a real model they are learned."
}
```

Notice the output (1.436, 1.292). It is not equal to any one value, but a blend that leans towards v₃ because the third key matched best. Now see the same computation drawn. Choose a query, read the scores, the weights and the output, then switch the division by √d off and on.

```viz
{
  "type": "attention",
  "title": "The worked example, drawn",
  "intro": "The query is the verb “wrote”. Compare it with the keys, read the weights, and check that they match the hand calculation. Then switch the √d scaling off.",
  "scale": true,
  "tokens": ["teacher", "board", "chalk"],
  "keys": [[1, 1], [0, 2], [2, 0]],
  "values": [[1, 0], [0, 1], [2, 2]],
  "queries": [ { "label": "wrote", "q": [2, 1] }, { "label": "erased", "q": [0, 1] } ],
  "note": "The three tokens, their vectors and the queries are illustrative, authored for teaching. The first query is the worked example above."
}
```

With the first query, the weights should be 0.284, 0.140 and 0.576 and the output (1.436, 1.292). Switch the scaling off and you get 0.245, 0.090, 0.665: the leader pulls further ahead. With the second query (“erased”, which looks along the second axis) the best-matching key is the second one, “board”, and the weights move there. ==The same set of keys and values gives different answers to different questions==; that is the whole point of a query.

Now a five-word sentence. Here again the vectors are made up.

```viz
{
  "type": "attention",
  "title": "A five-word sentence",
  "intro": "Pick the query “apple”, then “eating”. Which words get the most weight? Switch the scaling off and on and watch how the weights change.",
  "scale": true,
  "tokens": ["I", "am", "eating", "an", "apple"],
  "keys": [[1, 0], [0, 1], [1, 1], [0, 0.5], [1.5, 0.5]],
  "values": [[0.2, 1], [0.5, 0.1], [1, 0.9], [0.1, 0.3], [0.9, 0.2]],
  "queries": [ { "label": "apple", "q": [1, 0.5] }, { "label": "eating", "q": [0.5, 1] } ],
  "note": "Illustrative vectors authored for teaching. Real models learn vectors with hundreds of numbers, and their weights are rarely this tidy."
}
```

For the query “apple” with scaling, the largest weights are 0.314 on “apple” itself and 0.263 on “eating”, and for “eating” the largest is 0.284 on “eating” itself, then 0.238 on “apple”. Notice that a token may attend to itself, and that the weights are spread out rather than all on one word. That is what soft attention looks like.

### The weights as a matrix

Run the query for every token in turn and stack the rows of weights: you get a square table, one row per query and one column per key. ==Each row is a probability distribution== (it adds to 1). This table is the attention matrix, the picture you will meet again and again in papers.

```viz
{
  "type": "heatmap",
  "title": "An attention matrix for “Ana ate the ripe mango”",
  "corner": "query ↓ key →",
  "rows": ["Ana", "ate", "the", "ripe", "mango"],
  "cols": ["Ana", "ate", "the", "ripe", "mango"],
  "showValues": true,
  "rowsSumToOne": true,
  "tabs": [
    { "name": "Pattern A", "caption": "Each word looks mostly at the word just before it, a simple, local pattern.", "W": [[0.8, 0.05, 0.05, 0.05, 0.05], [0.6, 0.2, 0.1, 0.05, 0.05], [0.1, 0.6, 0.2, 0.05, 0.05], [0.05, 0.1, 0.6, 0.2, 0.05], [0.05, 0.05, 0.1, 0.6, 0.2]] },
    { "name": "Pattern B", "caption": "Words look at the words they go with: the verb and its object, the adjective and its noun.", "W": [[0.5, 0.4, 0.03, 0.03, 0.04], [0.3, 0.2, 0.05, 0.05, 0.4], [0.05, 0.05, 0.2, 0.2, 0.5], [0.02, 0.03, 0.1, 0.15, 0.7], [0.03, 0.2, 0.05, 0.5, 0.22]] }
  ],
  "note": "Illustrative weights authored for teaching, not from a trained model. Real models learn many such patterns; next week these are called heads."
}
```

Read a row: in Pattern B the row for “the” puts 0.50 of its weight on “mango”, the noun it belongs to. Check that each row adds to 1 (0.05 + 0.05 + 0.2 + 0.2 + 0.5 = 1.0). In Pattern A the column to the left of the diagonal glows, because every word looks one step back. One model can learn several patterns at once; how is Week 8.

### Why attention replaced recurrence

Three properties explain why attention took over language modelling.

- **Path length 1.** In a recurrent network, the signal from step 1 reaches step 30 through 29 steps. With attention, step 30 compares itself with step 1 directly. In the RNN-against-LSTM figure above, the attention line is flat at 1 for every position, so ==nothing fades with distance==.
- **Parallel.** All scores QKᵀ are computed in one matrix multiplication; no position waits for another. A GPU is built for exactly this kind of work, which is what made it possible to train on huge amounts of text.
- **The state grows with the text.** The model keeps one key and one value for every position instead of one fixed-size state. Nothing needs to be thrown away.

Check that you can tell the old and the new apart.

```viz
{
  "type": "sorter",
  "title": "Recurrence or attention?",
  "intro": "Decide which design each statement describes.",
  "bins": [ { "id": "rec", "label": "Recurrent (RNN / LSTM)" }, { "id": "att", "label": "Attention" } ],
  "cards": [
    { "name": "Information from step 1 reaches step 30 through 29 updates of one state", "answer": "rec", "why": "The chain of states is the only path." },
    { "name": "Step 30 compares itself directly with step 1", "answer": "att", "why": "The query of step 30 is scored against the key of step 1: one hop, whatever the distance." },
    { "name": "Cannot be parallelised across the positions of one sequence", "answer": "rec", "why": "Each step waits for the previous state." },
    { "name": "Computes all position-to-position scores in one matrix product", "answer": "att", "why": "QKᵀ gives the whole n × n table at once." },
    { "name": "Memory use for a sequence is a fixed-size state", "answer": "rec", "why": "The cost per step is constant, but the state must squeeze everything." },
    { "name": "Stores an n × n table of scores for n tokens", "answer": "att", "why": "Every position is compared with every other, so the cost grows with n²." },
    { "name": "The LSTM’s gates were invented to protect its memory from fading", "answer": "rec", "why": "The gates and the cell state are the LSTM’s answer to the vanishing gradient." },
    { "name": "Which positions matter is decided afresh for each query, using the content of the words", "answer": "att", "why": "The weights come from query–key scores, so they depend on the content of the words." }
  ]
}
```

### The price: n² scores

Attention has a price. For n tokens, every token is compared with every token, so there are n × n scores. Double the length and the table has four times as many entries. For n = 512 there are 262,144 scores; for n = 4,096 there are 16,777,216, about 16.8 million, because 4,096² = 16,777,216. Try the numbers.

```viz
{
  "type": "calc",
  "kind": "attn_cost",
  "title": "How the cost of attention grows with the length"
}
```

Set n = 512 and read 262,144 scores; set n = 4,096 and read 16.8 million. In 32-bit numbers (4 bytes each) the table of scores is 1 MiB for n = 512 and 64 MiB for n = 4,096, for one head of one layer, and it has to be stored for backpropagation. The number of multiply-adds is about 4·n²·d: with d = 64 that is 67 million for n = 512. By contrast, a recurrent network spends a fixed amount per step, so its cost grows in proportion to n, but it needs n steps in a row. ==Attention buys path length 1 and parallelism with a cost that grows as n²==, which is why long contexts need special tricks. Lab Part 5 measures the time yourself.

## Hard attention and soft attention {rail=Hard vs soft}

{lede} Once you have the scores, there are two ways to use them. ==Soft attention takes a weighted average of all the values; hard attention picks one position and uses only its value==. The difference looks small, but it decides whether the model can be trained by backpropagation.

### Soft attention: ask everyone and mix the answers

Suppose you are deciding what to eat and you ask three friends. A soft decision listens to all three and weighs their advice: 51% of what the first friend said, 42% of the second, 8% of the third. [[Soft attention|Attention that gives every position a weight between 0 and 1 (the weights add up to 1) and returns the weighted average of all the values.]] is what you computed in the last section. ==Every position contributes a little==, and the weights change smoothly when the scores change. That smoothness is the key: ==a smooth function has a slope==, so the chain rule from Week 3 can pass an error signal back through it.

### Hard attention: pick one

A hard decision asks only the friend you trust most. [[Hard attention|Attention that selects exactly one position (the one with the highest score, or one drawn at random according to the weights) and returns only its value.]] returns the value of one position and ignores all the others. It is easy to understand (“the model used this word”), and it can be cheaper, since only one value needs to be read. But look at what happens to the slope.

### The problem with a choice: no slope

Scores (2.2, 2.0, 0.3) pick the first position. Raise the first score from 2.2 to 2.3 and the choice is still the first position, so the output does not change at all: the slope is 0. Raise the second score to 2.3 and the choice suddenly jumps to the second position, and the output jumps with it. A function that is flat almost everywhere and jumps at a few points has a derivative of 0 almost everywhere. ==Backpropagation therefore gets no signal through an arg max==: it cannot tell the model “raise this score a little and the answer improves”.

```viz
{
  "type": "equation",
  "title": "Hard attention: α = one-hot(argmax s), output = v at the winner",
  "intro": "Select each symbol, then nudge the scores and watch the output of hard and soft attention.",
  "formula": [
    { "t": "α_hard", "id": "alpha" },
    " = one-hot(",
    { "t": "argmax", "id": "argmax" },
    " ",
    { "t": "s", "id": "s" },
    ");   output = ",
    { "t": "v_winner", "id": "v" }
  ],
  "parts": [
    { "id": "alpha", "name": "α_hard (hard weights)", "meaning": "A weight of 1 at one position and 0 at all the others. It is a choice, not a mix." },
    { "id": "argmax", "name": "argmax", "meaning": "The position that has the highest score. Its answer is a whole number (a position), so it jumps when another score overtakes it, and it does not change smoothly." },
    { "id": "s", "name": "s (the scores)", "meaning": "One score per position, computed from the query and the keys as before." },
    { "id": "v", "name": "v_winner", "meaning": "The value vector of the winning position. All other values are thrown away." }
  ],
  "steps": [
    { "title": "The scores and values", "expr": "s = (2.2, 2.0, 0.3);  values v₁ = (1, 0), v₂ = (0, 1), v₃ = (1, 1)", "note": "Three positions (illustrative)." },
    { "title": "Hard attention", "expr": "argmax = position 1;  α = (1, 0, 0);  output = v₁ = (1, 0)", "note": "One value, taken whole." },
    { "title": "Soft attention", "expr": "softmax(s) = (0.508, 0.416, 0.076);  output = (0.584, 0.492)", "note": "A blend: the winner leads, but the others still speak." },
    { "title": "Nudge s₁ by +0.1 (2.2 → 2.3)", "expr": "hard: output still (1, 0), change 0.   soft: output (0.584, 0.492) → (0.605, 0.467)", "note": "The soft output moves smoothly, so a slope exists. The hard output does not move." },
    { "title": "Nudge s₂ by +0.3 (2.0 → 2.3)", "expr": "hard: output jumps (1, 0) → (0, 1).   soft: (0.584, 0.492) → (0.510, 0.557)", "note": "The hard output jumps. Its change cannot be described by a slope. The soft change is gentle." }
  ],
  "note": "Illustrative scores and values. The same numbers are in the figure below."
}
```

Notice step 4 and 5 together: soft attention answers “what if the score changed a little?” with a small, smooth change. Hard attention answers with nothing or with a jump. Try it yourself.

```viz
{
  "type": "hardsoft",
  "title": "Soft blends, hard picks",
  "intro": "Start with the soft tab. Then raise the score of “cat” step by step, past 2.2, and watch the hard output jump while the soft output glides. Look at the nudge panel for the change when one score moves by 0.1.",
  "labels": ["the", "cat", "sat"],
  "scores": [2.2, 2.0, 0.3],
  "values": [[1, 0], [0, 1], [1, 1]],
  "temperature": 1,
  "note": "Scores and values are made up for teaching.",
  "checks": [
    { "temperature": 1, "soft": [0.508, 0.416, 0.076], "softOut": [0.584, 0.492], "hardOut": [1, 0] }
  ]
}
```

At the start the soft weights are 0.508, 0.416 and 0.076 and the soft output is (0.584, 0.492), while the hard output is (1, 0). The hard output flips to (0, 1) as soon as the score of “cat” is a little above that of “the”.

### Training hard attention, and who uses what

So how can hard attention be trained? Not by plain backpropagation. The usual alternatives are sampling-based gradient estimators (the best-known family is called REINFORCE, which we only name here): the model draws a position at random from the weights, sees how well that choice worked, and shifts the odds. These estimators are noisy and slower to train. In 2015, Xu and colleagues compared both kinds of attention for describing images in their “Show, Attend and Tell” paper: soft attention was the one trained by ordinary backpropagation, hard attention needed the sampling approach. In practice ==almost all of today’s language models use soft attention==, because it trains smoothly with the tools from Week 3 and runs well on GPUs. (“Almost all” because researchers do study sparse and hard-like variants, but they are the exception.) Lab Part 6 trains both and measures the difference.

### Temperature: from soft towards hard

Soft attention can be made as sharp as you like. In Week 1 you saw that dividing the scores by a temperature T before the softmax makes the distribution flatter for T large and sharper for T small. The same formula works for attention.

```viz
{
  "type": "equation",
  "title": "Temperature: weights = softmax(z / T)",
  "intro": "Select each symbol, then sharpen the scores (2, 1, 0.1) by lowering T.",
  "formula": [
    { "t": "αᵢ", "id": "alpha" },
    " = e^(",
    { "t": "zᵢ / T", "id": "zt" },
    ") / Σⱼ e^(",
    { "t": "zⱼ / T", "id": "zj" },
    ")"
  ],
  "parts": [
    { "id": "alpha", "name": "αᵢ (weight)", "meaning": "The attention weight on position i. All the weights are positive and add up to 1." },
    { "id": "zt", "name": "zᵢ / T", "meaning": "The score of position i divided by the temperature T. A small T makes differences between scores look big, so the leader takes nearly everything." },
    { "id": "zj", "name": "zⱼ / T", "meaning": "The same for every position j. The sum in the denominator makes the weights add up to 1." }
  ],
  "steps": [
    { "title": "T = 1 (plain softmax)", "expr": "z = (2, 1, 0.1);  weights = (0.659, 0.242, 0.099)", "note": "The leader has two thirds of the weight." },
    { "title": "T = 3 (flatter)", "expr": "z/T = (0.667, 0.333, 0.033);  weights = (0.445, 0.319, 0.236)", "note": "Nearly even: close to an average of everything." },
    { "title": "T = 0.5 (sharper)", "expr": "z/T = (4, 2, 0.2);  weights = (0.864, 0.117, 0.019)", "note": "The leader takes 86%." },
    { "title": "T = 0.1 (nearly hard)", "expr": "z/T = (20, 10, 1);  weights = (0.99995, 0.00005, 0.00000001)", "note": "Practically one-hot: the output is the leader’s value. As T → 0, soft attention becomes hard attention." }
  ],
  "note": "The scores (2, 1, 0.1) are made up. They match the lab."
}
```

Notice two things. As T → 0 the soft weights approach the hard one-hot, so ==hard attention is the limit of soft attention==. And that limit has a price: the slope also goes to 0, because the weights stop changing when the scores change. The figure below shows four positions with T = 0.3.

```viz
{
  "type": "hardsoft",
  "title": "Sharpening soft attention with the temperature",
  "intro": "The temperature slider is at 0.3. Move it down to 0.1 and up to 1 and compare the soft weights with the hard one-hot.",
  "labels": ["jeepney", "was", "very", "late"],
  "scores": [1.5, 0.2, 0.8, 2.0],
  "values": [[1, 0], [0, 1], [1, 1], [0.5, 0.5]],
  "temperature": 0.3,
  "note": "Scores and values are made up for teaching.",
  "checks": [
    { "temperature": 1, "soft": [0.2926, 0.0797, 0.1453, 0.4824], "softOut": [0.679, 0.466], "hardOut": [0.5, 0.5] },
    { "temperature": 0.3, "soft": [0.156, 0.002, 0.015, 0.827], "softOut": [0.585, 0.431], "hardOut": [0.5, 0.5] },
    { "temperature": 0.1, "soft": [0.0067, 0, 0, 0.9933], "softOut": [0.503, 0.497] }
  ]
}
```

At T = 1 the weights are 0.293, 0.080, 0.145 and 0.482. At T = 0.3 “late” has 0.827 and at T = 0.1 it has 0.993, so the soft output (0.503, 0.497) is almost the hard output (0.5, 0.5). The same trick appears when you sample text: a low temperature makes the model nearly deterministic. In the figure below the same softmax is applied to the scores of “it” over five words, so you can also draw a single position at random, which is exactly what stochastic hard attention does.

```viz
{
  "type": "sampling",
  "title": "Drawing one position from the weights (stochastic hard attention)",
  "prompt": "The animal didn’t cross the street because it was too tired. Which word does “it” attend to?",
  "tokens": [
    { "t": "animal", "logit": 3.0 },
    { "t": "tired", "logit": 2.0 },
    { "t": "street", "logit": 1.2 },
    { "t": "cross", "logit": 0.5 },
    { "t": "because", "logit": 0.1 }
  ],
  "note": "The scores are made up for teaching. Each draw picks one word, like hard attention that samples from the weights; the tally of 100 draws approaches the soft weights."
}
```

Draw a hundred positions and the tally comes close to the soft weights: ==soft attention is the average of what all those random hard choices would give==. A single draw is noisy, and the random choice has no gradient. That is the trade in one picture.

### Side by side

| | Soft attention | Hard attention |
|---|---|---|
| What it returns | Weighted average of all values | The value of one position |
| Differentiable? | Yes: smooth, with a slope everywhere | No: flat or jumping, derivative 0 almost everywhere |
| Cost | Reads all n values for each query | Reads one value (still scores all n positions) |
| Interpretability | Weights show influence, but all positions take part | One clear choice |
| Training | Ordinary backpropagation (Week 3) | Sampling-based estimators such as REINFORCE, noisy |
| Use today | Almost all current language models | Rare; mostly in research and some image models |

Sort the statements to check yourself.

```viz
{
  "type": "sorter",
  "title": "Soft, hard or both?",
  "intro": "Decide which kind of attention each statement is about.",
  "bins": [ { "id": "soft", "label": "Soft attention" }, { "id": "hard", "label": "Hard attention" }, { "id": "both", "label": "Both" } ],
  "cards": [
    { "name": "The output is a weighted average of all the value vectors", "answer": "soft", "why": "The weights are all positive and add up to 1; every position contributes a little." },
    { "name": "The output is the value of exactly one position", "answer": "hard", "why": "One position wins and the rest are ignored." },
    { "name": "A small change in a score changes the output a little", "answer": "soft", "why": "The softmax is smooth, so tiny changes in scores give tiny changes in weights." },
    { "name": "A tiny change in a score can make the choice jump to another position", "answer": "hard", "why": "An arg max is a step function: it flips when another score overtakes the leader." },
    { "name": "Needs REINFORCE-style sampling to be trained", "answer": "hard", "why": "Backpropagation gets no signal through an arg max, so noisy estimators are used." },
    { "name": "Can be trained directly by backpropagation", "answer": "soft", "why": "Its slope exists everywhere, so the chain rule works." },
    { "name": "Starts from scores that compare a query with each key", "answer": "both", "why": "Both begin with the same query–key scores. They differ in how the scores are used." },
    { "name": "Which positions matter depends on the content of the words", "answer": "both", "why": "In both, the choice or the weights come from content-based scores, not from distance." }
  ]
}
```

Notice that the last two cards are shared: the two kinds start the same way and differ only in the final step. Now that we know why soft attention is the one that is used, we can ask what happens when a sentence attends to itself.

## Self-attention and what comes next {rail=Self-attention}

{lede} In the examples so far, the query and the keys came from different places: a decoder asking about an encoder’s states. In [[self-attention|Attention in which the queries, keys and values all come from the same sequence, so every token looks at every other token of the sequence and at itself.]], ==the queries, keys and values all come from the same sequence==. Every token asks questions of all the others, and updates itself with the answers.

### Where Q, K and V come from

How does a token produce a query, a key and a value? Each is a different linear view of the same token vector. If X is the matrix of the n token vectors (one row per token, d_model numbers per row), three learned weight matrices turn it into Q, K and V.

```viz
{
  "type": "equation",
  "title": "Queries, keys and values from one sequence: Q = X W_Q, K = X W_K, V = X W_V",
  "intro": "Select each symbol, then compute Q, K and V for the three-token sentence “dog bites man”.",
  "formula": [
    { "t": "Q", "id": "Q" },
    " = ",
    { "t": "X", "id": "X" },
    " ",
    { "t": "W_Q", "id": "WQ" },
    ";   ",
    { "t": "K", "id": "K" },
    " = X ",
    { "t": "W_K", "id": "WK" },
    ";   ",
    { "t": "V", "id": "V" },
    " = X ",
    { "t": "W_V", "id": "WV" }
  ],
  "parts": [
    { "id": "X", "name": "X (the input)", "meaning": "n rows, one per token, each a vector of d_model numbers (the embeddings of Week 6, plus whatever earlier layers added). Shape n × d_model." },
    { "id": "WQ", "name": "W_Q (query weights)", "meaning": "A learned matrix of shape d_model × d_k. It turns each token vector into the question that token asks." },
    { "id": "WK", "name": "W_K (key weights)", "meaning": "A learned matrix of shape d_model × d_k. It turns each token vector into the label that token offers to others." },
    { "id": "WV", "name": "W_V (value weights)", "meaning": "A learned matrix of shape d_model × d_v. It turns each token vector into the content that token hands over when attended to." },
    { "id": "Q", "name": "Q (queries)", "meaning": "Shape n × d_k: one query per token." },
    { "id": "K", "name": "K (keys)", "meaning": "Shape n × d_k: one key per token. The query and key must have the same length so that they can be compared by a dot product." },
    { "id": "V", "name": "V (values)", "meaning": "Shape n × d_v: one value per token." }
  ],
  "steps": [
    { "title": "The input X (n = 3, d_model = 2)", "expr": "dog = (1, 0);  bites = (0, 1);  man = (1, 1)", "note": "Illustrative 2-number embeddings authored for teaching." },
    { "title": "The weight matrices", "expr": "W_Q = ((1, 0), (1, 1));  W_K = ((0, 1), (1, 1));  W_V = ((2, 0), (0, 1))", "note": "Learned in a real model. Here d_k = d_v = 2." },
    { "title": "Queries Q = X W_Q", "expr": "dog (1,0) → (1, 0);  bites (0,1) → (1, 1);  man (1,1) → (2, 1)", "note": "Each row of X is multiplied by W_Q." },
    { "title": "Keys K = X W_K", "expr": "dog → (0, 1);  bites → (1, 1);  man → (1, 2)", "note": "A different matrix gives different vectors from the same words." },
    { "title": "Values V = X W_V", "expr": "dog → (2, 0);  bites → (0, 1);  man → (2, 1)", "note": "Three matrices of shape 2 × 2 gave three 3 × 2 tables." }
  ],
  "note": "All numbers are authored for teaching."
}
```

Notice that nothing here mixes tokens yet. ==Q, K and V are computed token by token==. The mixing happens in the next formula.

### The whole layer in matrix form

Now do the whole layer at once. Multiply Q by Kᵀ to get all the scores, divide by √d_k, apply the softmax to each row, and multiply by V.

```viz
{
  "type": "equation",
  "title": "A self-attention layer: output = softmax(Q Kᵀ / √d_k) V",
  "intro": "Shapes: Q Kᵀ is n × n; the weights are n × n; the output is n × d_v. Select each symbol, then continue the “dog bites man” example.",
  "formula": [
    { "t": "output", "id": "out" },
    " = ",
    { "t": "softmax", "id": "sm" },
    "(",
    { "t": "Q Kᵀ", "id": "qk" },
    " / ",
    { "t": "√d_k", "id": "root" },
    ") ",
    { "t": "V", "id": "v" }
  ],
  "parts": [
    { "id": "out", "name": "output (n × d_v)", "meaning": "One new vector per token: a blend of the value vectors of the whole sentence, with weights chosen by that token’s query." },
    { "id": "sm", "name": "softmax (row by row)", "meaning": "Turns each row of scores into weights that add up to 1. The result is the n × n attention matrix." },
    { "id": "qk", "name": "Q Kᵀ (n × n)", "meaning": "Entry (i, j) is the dot product of the query of token i with the key of token j. All n² scores are computed by one matrix product." },
    { "id": "root", "name": "√d_k", "meaning": "Keeps the scores at a steady size. The reason is shown next." },
    { "id": "v", "name": "V (n × d_v)", "meaning": "The values. Row i of the output is the weights of row i times V." }
  ],
  "steps": [
    { "title": "Scores Q Kᵀ (3 × 3)", "expr": "((0, 1, 1), (1, 2, 3), (1, 3, 4))", "note": "Entry (2, 3) = query of bites (1, 1) · key of man (1, 2) = 1 + 2 = 3." },
    { "title": "Divide by √2 = 1.414", "expr": "((0, 0.707, 0.707), (0.707, 1.414, 2.121), (0.707, 2.121, 2.828))", "note": "Here d_k = 2, so we divide by the square root of 2." },
    { "title": "Softmax of each row", "expr": "dog: (0.198, 0.401, 0.401);  bites: (0.140, 0.284, 0.576);  man: (0.074, 0.306, 0.620)", "note": "Each row adds up to 1. This is the attention matrix." },
    { "title": "Row 1 of the output", "expr": "0.198×(2, 0) + 0.401×(0, 1) + 0.401×(2, 1) = (1.198, 0.802)", "note": "The new vector for “dog” mixes all three values." },
    { "title": "The full output", "expr": "dog (1.198, 0.802);  bites (1.432, 0.860);  man (1.389, 0.926)", "note": "Three input vectors in, three new vectors out. Each output now contains information from the whole sentence." }
  ],
  "note": "Authored numbers; the sums are exact up to rounding."
}
```

Notice that the output has the same number of rows as the input: ==one vector per token==. These new vectors are [[contextual embeddings|Vectors for tokens that depend on the other words around them, so the same word can get different vectors in different sentences.]], the thing Week 6 promised: “bank” next to “river” can now get a different vector from “bank” next to “money”. Here are the shapes of everything in one place.

| Quantity | Shape | Meaning |
|---|---|---|
| X | n × d_model | one input vector per token |
| W_Q, W_K | d_model × d_k | learned |
| W_V | d_model × d_v | learned |
| Q, K | n × d_k | queries and keys |
| V | n × d_v | values |
| Q Kᵀ and the weights | n × n | one score and one weight for every pair of tokens |
| output | n × d_v | one new vector per token |

### Why divide by √d_k?

The division by √d_k looked like a detail, but it protects training. A query and a key are vectors of d_k numbers. Suppose their entries are random, with mean 0 and variance 1. The [[dot product|The sum of the products of matching entries of two vectors, such as (2, 1)·(1, 3) = 2×1 + 1×3 = 5. It is large when the vectors point the same way.]] is a sum of d_k products, each with variance 1, so the sum has variance d_k and standard deviation √d_k. ==Longer vectors give larger scores, even when nothing has been learned yet.==

```viz
{
  "type": "equation",
  "title": "Why the scaling: the dot product has variance d_k",
  "intro": "Select each symbol, then see what happens to the softmax with and without the division.",
  "formula": [
    "Var(",
    { "t": "q·k", "id": "qk" },
    ") = ",
    { "t": "d_k", "id": "dk" },
    ";   so   Var(",
    { "t": "q·k / √d_k", "id": "scaled" },
    ") = 1"
  ],
  "parts": [
    { "id": "qk", "name": "q·k (raw score)", "meaning": "The sum of d_k products q₁k₁ + … + q_dk k_dk. For random unit-variance entries it has mean 0 and variance d_k." },
    { "id": "dk", "name": "d_k (key length)", "meaning": "The number of entries in a query or key. Typical values are 64 or more." },
    { "id": "scaled", "name": "q·k / √d_k (scaled score)", "meaning": "Dividing by the standard deviation √d_k brings the variance back to 1, whatever d_k is." }
  ],
  "steps": [
    { "title": "Standard deviation of the raw score", "expr": "d_k = 4: 2;  16: 4;  64: 8;  256: 16;  1024: 32", "note": "The raw scores spread out like √d_k." },
    { "title": "A raw spread for d_k = 64", "expr": "scores (8, 0, −8)  →  softmax = (0.9997, 0.0003, 0.0000001)", "note": "One position takes almost everything: this is nearly hard attention." },
    { "title": "After dividing by √64 = 8", "expr": "scores (1, 0, −1)  →  softmax = (0.665, 0.245, 0.090)", "note": "A usable soft distribution." },
    { "title": "The slope of the top weight", "expr": "p(1 − p):  0.9997 × 0.0003 = 0.00034   against   0.665 × 0.335 = 0.223", "note": "In the saturated case the softmax is almost flat, so the gradient is about 660 times smaller." }
  ],
  "note": "A random-vector argument, as in the original paper: trained vectors are not exactly random, but the scores do grow with d_k."
}
```

==This is the vanishing gradient again, in a new place.== Without the division, the softmax saturates, the weights become nearly one-hot (nearly hard attention) and the slope is close to 0, exactly the “flat” problem of the previous section and the saturated tanh of the RNN. Lab Part 5 measures the variance and the gradient, and Lab Part 7 builds a self-attention layer and compares it with an LSTM.

### A pronoun finds its noun

Self-attention can link a pronoun to the noun it stands for, a long-range link of the kind a fixed window loses. Take “The trophy didn’t fit in the suitcase because it was too big” (or “too small”). To resolve “it” you need the meaning of the last word: a trophy that is too big does not fit, and a suitcase that is too small does not hold it. The figure shows the attention a trained model might give from the word “it”, one row per sentence ending.

```viz
{
  "type": "heatmap",
  "title": "What does “it” attend to? (illustrative)",
  "corner": "query ↓ key →",
  "rows": ["it"],
  "cols": ["The", "trophy", "didn’t", "fit", "in", "the", "suitcase", "because", "it", "was", "too", "big / small"],
  "showValues": true,
  "rowsSumToOne": true,
  "tabs": [
    { "name": "…too big", "caption": "The sentence ends with “too big”: the weight of “it” is mostly on “trophy”.", "W": [[0.02, 0.55, 0.02, 0.04, 0.01, 0.02, 0.2, 0.03, 0.04, 0.02, 0.02, 0.03]] },
    { "name": "…too small", "caption": "The sentence ends with “too small”: the weight moves to “suitcase”.", "W": [[0.02, 0.2, 0.02, 0.04, 0.01, 0.02, 0.55, 0.03, 0.04, 0.02, 0.02, 0.03]] }
  ],
  "note": "Weights authored for teaching to illustrate the idea, not measured from a real model. Real attention maps are less tidy."
}
```

Read the first tab: 0.55 on “trophy” and 0.20 on “suitcase”. In the second tab the two numbers are swapped. The query of “it” is built from the sentence, including the last word, so ==the same word can point to different nouns==. A recurrent network would have had to carry “trophy” and “suitcase” through the whole clause in its state; attention connects “it” to either in one step.

### What self-attention does not have

Three things are missing, and they are the topics of Week 8. Here we only name them.

1. **No sense of order.** Look at the layer again: it takes a set of vectors and returns a set of vectors. If you give the three words in the order “man bites dog”, with man = (1, 1) first, and use the same weights, the new vector for “bites” is again (1.432, 0.860): the same as before. Each word gets the same output vector, only the rows are in a different order. Without extra information, ==self-attention cannot tell “dog bites man” from “man bites dog”==. The fix is [[positional encoding|A vector that tells the model where each token sits in the sequence, added to the token’s embedding before attention.]] (Week 8).
2. **One pattern per layer.** One set of W_Q, W_K, W_V looks for one kind of relation at a time. Running several in parallel is called multi-head attention (Week 8). The two tabs of the earlier matrix figure were two such patterns.
3. **It sees the future.** A language model that predicts the next token must not look at the tokens after it, or the task would be trivial. The fix is the [[causal mask|A rule that hides the future: position i may attend only to positions 1 … i, so the model cannot peek at the token it is asked to predict.]] (Week 8). In one paragraph: before the softmax, the scores for the future positions are replaced by a very large negative number, so their weights come out as 0 and each row is renormalised over the visible positions. The mask looks like this.

```viz
{
  "type": "heatmap",
  "title": "A causal mask for five tokens (preview of Week 8)",
  "corner": "query ↓ key →",
  "rows": ["Ana", "ate", "the", "ripe", "mango"],
  "cols": ["Ana", "ate", "the", "ripe", "mango"],
  "showValues": true,
  "W": [[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1]],
  "note": "1 means visible and 0 means hidden. Each token sees itself and the tokens before it. Week 8 shows how the mask is applied."
}
```

Notice the staircase. Row “the” has three 1s: it may look at “Ana”, “ate” and itself, and nothing to its right.

### The road so far

Put the pieces on one line: each step solved a problem and left another.

```viz
{
  "type": "pipeline",
  "title": "The road from RNNs to the Transformer",
  "intro": "Step through the five stages. Each one fixes the problem left by the one before.",
  "vertical": false,
  "nodes": [
    { "id": "rnn", "label": "RNN", "sub": "state" },
    { "id": "lstm", "label": "LSTM", "sub": "gated cell" },
    { "id": "s2s", "label": "Seq2seq", "sub": "context vector" },
    { "id": "att", "label": "Attention", "sub": "look back" },
    { "id": "tr", "label": "Transformer", "sub": "attention only" }
  ],
  "steps": [
    { "title": "RNN: a state that carries the past", "body": "A recurrent state lets the model read a sequence in order, but its gradient fades with the product of small factors, and the steps are sequential.", "active": ["rnn"], "example": "3.6 × 10⁻¹¹ over 30 steps" },
    { "title": "LSTM: protect the memory", "body": "Gates and a cell state keep information over many more steps, but the state is still a fixed-size vector and the steps are still sequential.", "active": ["rnn", "lstm"], "example": "0.226 over 30 steps with f = 0.95" },
    { "title": "Seq2seq: read, then write", "body": "An encoder and a decoder translate by passing one context vector between them. This is the bottleneck.", "active": ["lstm", "s2s"], "example": "the whole sentence in one vector" },
    { "title": "Attention: look back", "body": "The decoder takes a weighted mix of all the encoder states. Path length 1, no single bottleneck, but n² scores.", "active": ["s2s", "att"], "example": "softmax(QKᵀ/√d_k)V" },
    { "title": "Transformer: attention without recurrence", "body": "Vaswani and colleagues built a whole model from self-attention layers, with positions added and several heads, so all positions are processed in parallel. You study it next week.", "active": ["att", "tr"], "example": "Week 8" }
  ],
  "note": "The numbers are the ones computed earlier in this lesson."
}
```

The same road as a short list of dated landmarks you can click through.

```viz
{
  "type": "timeline",
  "title": "Landmarks of attention",
  "events": [
    { "year": "2014", "title": "Sequence to sequence", "text": "Sutskever and colleagues showed that one LSTM could read a sentence into a vector and another could write the translation. The single vector was the bottleneck." },
    { "year": "2015", "title": "Attention in translation", "text": "Bahdanau, Cho and Bengio let the decoder weigh all the encoder states instead of one vector, and reported better results on long sentences." },
    { "year": "2015", "title": "Soft and hard attention for images", "text": "Xu and colleagues compared soft and hard attention for describing images in “Show, Attend and Tell”: soft attention trains by backpropagation, hard attention needs sampling." },
    { "year": "2016", "title": "A picture book of attention", "text": "Olah and Carter published an illustrated Distill article, Attention and augmented recurrent neural networks, that explained the idea visually." },
    { "year": "2017", "title": "Attention is all you need", "text": "Vaswani and colleagues used scaled dot-product self-attention as the main building block of the Transformer, with no recurrence." }
  ]
}
```

The papers behind this lesson:

- [1] Vaswani et al. (2017). *Attention is all you need.* Advances in Neural Information Processing Systems 30 (NeurIPS), pp. 5998–6008. arXiv:1706.03762. Section 3.2 describes scaled dot-product attention.
- [2] Bahdanau, Cho & Bengio (2015). *Neural machine translation by jointly learning to align and translate.* ICLR 2015. arXiv:1409.0473.
- [3] Olah & Carter (2016). *Attention and augmented recurrent neural networks.* Distill 1(9). doi:10.23915/distill.00001.

Also mentioned in this lesson: Elman (1990), *Finding structure in time* (simple recurrent networks); Hochreiter & Schmidhuber (1997), *Long short-term memory*; Sutskever et al. (2014), *Sequence to sequence learning with neural networks*; Xu et al. (2015), *Show, attend and tell* (soft and hard attention for image captions).

## Key takeaways

- **A recurrent network keeps one state, and the gradient fades along time.** It updates h_t = tanh(W_h h_(t−1) + W_x x_t + b) with the same weights at every step, and ==the gradient to step t is a product of factors w·(1 − h²), each below 1==: about 3.6 × 10⁻¹¹ over 30 steps for w = 0.9. It is also sequential.
- **An LSTM protects memory with gates.** ==The cell update c_t = f ⊙ c_(t−1) + i ⊙ g is an addition, so the gradient along the cell is a product of forget gates (0.95²⁹ = 0.226)==. It still reads step by step and squeezes the past into a fixed-size state.
- **Attention removes the bottleneck.** ==Instead of one context vector, the decoder takes a weighted mix of all the encoder states==, so the path between any two positions has length 1, and all scores are computed in parallel.
- **Attention is score, softmax, weighted sum.** The output is softmax(QKᵀ/√d_k)V: ==dividing by √d_k keeps the scores steady so the softmax does not saturate==, and the price is an n × n table (n = 512 gives 262,144 scores).
- **Soft attention can be trained, hard attention cannot (directly).** ==A weighted average has a slope, but an arg max has derivative 0 almost everywhere==, so hard attention needs sampling estimators. Self-attention takes Q, K, V from the same sequence, and it still needs positions, several heads and a causal mask: Week 8.

<!-- LAB-START -->
## Lab: Imagine a model that remembers, then give it attention {rail=Laboratory}

This lab is called *Imagine a model that remembers, then give it attention*. Week 7 asks how a model can remember what it read many words ago. You first put yourself in the place of a reader of a long letter, who can either carry everything in their head or glance back at any line, and then ==build, step by step and on a controlled memory task, a window model, a recurrent network (RNN) that forgets, an LSTM with gates that remembers longer, scaled dot-product attention written by hand, a comparison of hard and soft attention, and finally a small model with an attention layer that solves the task at lengths where the RNN fails==. It has seven parts, and every part begins with an **Imagine first** moment: you write a prediction before you run any code and compare it with what really happens. Each part points back to a section of the lesson and to one of its figures. There are no downloads of any kind: the task is made by a short function from random numbers, and everything runs on a CPU in seconds to a couple of minutes per block. Time: about 3 hours. This page contains the setup and the first three parts (the task, the RNN and the gradients through time); the other four parts, the deliverables and the marking scheme follow in the same lab.

| Part | What you do | Minutes |
|---|---|---|
| Setup | Start JupyterLab, paste the helper cell, run a smoke test | 10 |
| 1 | Sequences and the memory task | 20 |
| 2 | An RNN from scratch | 25 |
| 3 | Gradients through time | 25 |
| 4 | LSTM: gates and a memory highway | 30 |
| 5 | Attention by hand | 30 |
| 6 | Hard vs soft attention | 25 |
| 7 | Self-attention layer, design and reflect | 15 |

The total is 180 minutes, about 3 hours. The marking scheme and the deliverables are at the end of the lab.

### Setup

1. Log in to the portal, start JupyterLab on the DGX Spark, create a folder `week07` and a notebook `lab7.ipynb`. Run each block of this lab in its own cell, in order: later blocks use names (`Xtr`, `WindowModel`, `rnn_hist`, `scalar_rnn_influence`, `RESULTS` and so on) that earlier blocks create. If you restart the kernel, run the setup cell and then the blocks again from the top.
2. Do not install packages: PyTorch, NumPy and Matplotlib are already installed. Do not use `!pip install`, Google Drive or `google.colab` code in the Spark notebook. Nothing is downloaded in this week's lab, not even a tokenizer: the memory task is generated from random numbers by a function in the setup cell.
3. A CPU is enough. Most blocks finish in a second or two; the slowest ones (training the RNN on sequences of 40 and 60 symbols, Step 2.3) take about half a minute each. The GPU of the Spark is shared by the whole class and gives no advantage on models this small, so leave it alone. The helper cell prints which device it found.
4. Write your answers in markdown cells under the block where the exercise appears (Insert a cell, change its type to Markdown). Where an exercise says "store it in a variable", write the variable in a code cell: Part 7 collects your numbers in the dictionary `RESULTS`.
5. The same lab is available as a **Google Colab notebook**. Use the "Open in Colab" card at the top of this lab if you work from home or on a laptop: it has the same parts, the same numbering and the same expected results, plus automatic Check cells and a worked answer for every exercise. The code on this page is the complete working version, so on the Spark you can type or paste it directly. In the Colab notebook you write the key lines yourself first.
6. Paste this into the first cell and run it. It fixes the random seed (so your numbers match the ones printed in this lab), defines the empty `RESULTS` dictionary and the `check(...)` function, and defines the helpers that make the memory task, turn symbols into vectors, train any sequence model, compute a softmax and draw an attention matrix:

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
QUICK = False
print("Running on:", device)

RESULTS = {}   # a place to keep your key numbers and answers; the last part prints them together

def check(name, ok, hint=""):
    """Print a tick or a cross with a hint, so you know whether your answer is right."""
    print(("✓ " if ok else "✗ ") + name + ("" if ok else "  → " + hint))

N_SYMBOLS = 8   # the memory task uses the symbols 0..7

def make_memory_task(n, length, seed=0, n_symbols=N_SYMBOLS):
    """n sequences of `length` random symbols in 0..n_symbols-1. The label of a sequence is its FIRST symbol, so to answer at the end
    the model must remember symbol 1 across `length - 1` more symbols. Returns X (n, length) int64 and y (n,) int64."""
    r = np.random.RandomState(seed)
    X = r.randint(0, n_symbols, size=(n, length)).astype(np.int64)
    return X, X[:, 0].copy()

def one_hot_batch(X, n_symbols=N_SYMBOLS):
    """(n, length) integer symbols -> (n, length, n_symbols) float32 one-hot vectors."""
    return F.one_hot(torch.as_tensor(X), n_symbols).float()

def train_seq_model(model, Xtr, ytr, Xte, yte, epochs=200, lr=0.01, batch=64, seed=0, clip=1.0, verbose=False):
    """Train a model that maps (batch, length, n_symbols) to class scores (batch, n_symbols) with cross-entropy and Adam.
    Returns a dict with per-epoch lists loss, train_acc, test_acc. Gradients are clipped to norm `clip`."""
    g = torch.Generator().manual_seed(seed)
    Xtr, ytr, Xte, yte = one_hot_batch(Xtr), torch.as_tensor(ytr), one_hot_batch(Xte), torch.as_tensor(yte)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = {"loss": [], "train_acc": [], "test_acc": []}
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Xtr), generator=g); tot = 0.0
        for b in range(0, len(Xtr), batch):
            idx = perm[b:b + batch]
            opt.zero_grad(); loss = F.cross_entropy(model(Xtr[idx]), ytr[idx]); loss.backward()
            if clip: nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step(); tot += float(loss) * len(idx)
        model.eval()
        with torch.no_grad():
            hist["loss"].append(tot / len(Xtr))
            hist["train_acc"].append(float((model(Xtr).argmax(1) == ytr).float().mean()))
            hist["test_acc"].append(float((model(Xte).argmax(1) == yte).float().mean()))
        if verbose and (ep % 20 == 0 or ep == epochs - 1): print(f"epoch {ep:3d}  loss {hist['loss'][-1]:.3f}  train {hist['train_acc'][-1]:.2f}  test {hist['test_acc'][-1]:.2f}")
    return hist

def softmax_np(z, axis=-1):
    """Softmax of a NumPy array along an axis (numerically stable)."""
    z = np.asarray(z, dtype=float); e = np.exp(z - z.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def show_attention(W, row_labels=None, col_labels=None, title="", ax=None):
    """Draw an attention-weight matrix W (rows = queries, columns = keys) as a heat map with the numbers in the cells."""
    W = np.asarray(W); ax = ax or plt.gca()
    im = ax.imshow(W, cmap="Greens", vmin=0, vmax=max(1e-9, W.max()))
    ax.set_xticks(range(W.shape[1])); ax.set_yticks(range(W.shape[0]))
    ax.set_xticklabels(col_labels or range(W.shape[1]), rotation=45, ha="right"); ax.set_yticklabels(row_labels or range(W.shape[0]))
    for i in range(W.shape[0]):
        for j in range(W.shape[1]): ax.text(j, i, f"{W[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(title)
    return im
```

   - `check(name, ok, hint)` prints a tick or a cross with a hint, so you can test your own answers;
   - `make_memory_task(n, length, seed)` returns `n` random sequences of `length` symbols from 0 to 7 (an integer array `X`) and the labels `y`, which are the *first* symbol of each sequence: to answer at the end, a model must remember symbol 1 across the whole sequence;
   - `one_hot_batch(X)` turns the integer symbols into one-hot vectors of 8 numbers, the input format of all the models in this lab: the array has shape (n, length, 8);
   - `train_seq_model(model, Xtr, ytr, Xte, yte, epochs, lr, ...)` trains any model that maps a batch of one-hot sequences to 8 class scores, with cross-entropy and the Adam optimiser, clips the gradient size, and returns the learning curves `loss`, `train_acc` and `test_acc` (one value per epoch);
   - `softmax_np(z)` is the softmax of a NumPy array, which turns scores into weights that add up to 1, and `show_attention(W, row_labels, col_labels, title)` draws an attention-weight matrix as a heat map (used from Part 5);
   - `RESULTS` is a dictionary where you keep your key numbers; Part 7 prints it as your summary.

7. Run this smoke test in the next cell. It checks that the helpers work and that your Matplotlib shows figures in the notebook:

```python
X0, y0 = make_memory_task(4, 6, seed=0)
print(X0)
print("labels:", y0, "| the first column:", X0[:, 0])
print("one-hot shape:", tuple(one_hot_batch(X0).shape))
print("softmax of (2, 1, 0.1):", np.round(softmax_np([2, 1, 0.1]), 3))
plt.figure(figsize=(3, 2)); plt.plot([0, 1, 2], [0, 1, 4], "o-"); plt.title("figures work"); plt.tight_layout(); plt.show()
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
[ [4 7 5 0 3 3]
 [3 7 1 3 5 2]
 [4 7 6 0 0 4]
 [2 1 6 7 7 6] ]
labels: [4 3 4 2] | the first column: [4 3 4 2]
one-hot shape: (4, 6, 8)
softmax of (2, 1, 0.1): [0.659 0.242 0.099]
```

   You should see `Running on: cpu` after the first cell. The smoke test then shows four sequences of six symbols, and their labels, which are the numbers of the first column (the label is always the first symbol), the shape of the one-hot array (4 sequences, 6 steps, 8 symbols), the softmax of (2, 1, 0.1), which is (0.659, 0.242, 0.099) (the same three weights as in the lesson's attention figure), and a small figure.

If something goes wrong:

- `NameError: name 'xyz' is not defined`: you skipped a block or restarted the kernel. Run the setup cell and the earlier blocks again, in order.
- Plots do not appear: run `%matplotlib inline` once in a cell, or check that the block ends with `plt.show()`.
- Your numbers differ from the ones printed here: make sure the setup cell ran with `SEED = 42`, and that you did not run a training block twice in the same variable. The blocks that train a network set their own seed before building it, but small differences can appear between PyTorch versions, and even between computers with different numbers of CPU threads. The qualitative results (a window of 9 fails, an RNN of 32 units learns up to about 40 steps, the gradient collapses by many powers of ten) are stable; the exact digits and the exact length at which an RNN stops learning are not.
- A cell seems frozen: stop it (the square button), restart the kernel and run again from the top. If a training cell takes more than two minutes, you probably changed a loop count, a number of epochs or a data size.

### Part 1. Sequences and the memory task (20 min)

*Lesson section: "Sequences".* A sentence is not a bag of words: the order matters, and the word that decides the meaning is sometimes far away from the place where you need it. In this part you build the lab's **memory task**, a clean version of that problem. Each example is a sequence of random symbols from 0 to 7, and the answer to give at the end is the *first* symbol. A model that cannot carry information across the whole sequence cannot win, and a model that can remember scores 100%. You measure what a model that sees only the last symbol can do, then what a model with a fixed window can do, and you discover why a window is not a real solution. These are the same ideas as the lesson's figure on sequences: the farther back the important thing is, the harder it is to reach.

The data comes from `make_memory_task(n, length, seed)` in the setup cell. It returns the sequences `X` (shape n by length, integers 0 to 7) and the labels `y`, which are simply `X[:, 0]`. The models see each symbol as a one-hot vector of 8 numbers (`one_hot_batch` does that), and `train_seq_model` trains any model that turns a batch of one-hot sequences into 8 class scores. Chance accuracy is 1 out of 8 classes, that is 0.125.

**Imagine first.** Before you run any code, write three numbers. (1) What accuracy does a model get if it guesses at random among the 8 symbols? (2) What accuracy does a model with a perfect memory get? (3) What accuracy do you expect from a model that is only allowed to look at the *last* symbol of each sequence? Put your reasons in a markdown cell. You will compare your numbers with the measurements in Steps 1.1 and 1.2.

```python
# Imagine first: what can a model do on the memory task?
my_chance = None        # accuracy of a model that guesses at random, a number between 0 and 1
my_perfect = None       # accuracy of a model with a perfect memory
my_last_symbol = None   # accuracy of a model that sees only the LAST symbol of each sequence
```

1. **Step 1.1. Make the data and look at it.** Every experiment starts by looking at the data. The first block makes a training set of 1,000 sequences of length 10 and a test set of 300 new ones (different seeds, so no sequence leaks from one to the other), and prints five sequences with their labels. The second block counts how often each label occurs, compares the counts with the equal share you would expect, and measures how often the label happens to equal the *last* symbol. If the 8 classes were not balanced, "accuracy" would be easy to inflate by always answering the most common class, so we check.

```python
Xtr, ytr = make_memory_task(1000, 10, seed=1)      # training set: 1000 sequences of 10 symbols
Xte, yte = make_memory_task(300, 10, seed=2)       # test set: 300 new sequences
print("shapes:", Xtr.shape, ytr.shape, "| test:", Xte.shape)
for i in range(5):
    print(Xtr[i], "-> label", ytr[i])
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
shapes: (1000, 10) (1000,) | test: (300, 10)
[5 3 4 0 7 1 3 5 7 0] -> label 5
[0 1 4 7 5 4 6 1 2 4] -> label 0
[6 5 2 4 3 4 2 4 5 6] -> label 6
[2 4 7 7 1 1 7 6 0 6] -> label 2
[5 1 1 7 7 5 6 1 1 0] -> label 5
```

```python
counts = np.bincount(ytr, minlength=N_SYMBOLS)
print("class counts :", counts)
print("class shares :", np.round(counts / len(ytr), 3))
print("chance accuracy 1/8 =", 1 / N_SYMBOLS)
print("best constant guess on the test set:", round(float(np.bincount(yte).max() / len(yte)), 3))
print("share of sequences whose label equals their LAST symbol:", round(float((Xtr[:, -1] == ytr).mean()), 3))
plt.figure(figsize=(5, 2.5))
plt.bar(range(N_SYMBOLS), counts, color="tab:blue"); plt.axhline(len(ytr) / N_SYMBOLS, color="k", ls="--", label="equal share")
plt.xlabel("label (the first symbol)"); plt.ylabel("sequences"); plt.title("The 8 classes are balanced"); plt.legend()
plt.tight_layout(); plt.show()
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
class counts : [120 118 120 134 141 141 107 119]
class shares : [0.12  0.118 0.12  0.134 0.141 0.141 0.107 0.119]
chance accuracy 1/8 = 0.125
best constant guess on the test set: 0.187
share of sequences whose label equals their LAST symbol: 0.127
```

   You should see `shapes: (1000, 10) (1000,) | test: (300, 10)`. In each printed sequence the label is the first number: the first sequence `5 3 4 0 7 1 3 5 7 0` has label 5. The class counts are `[120 118 120 134 141 141 107 119]`: every class has close to 125 sequences (1,000 divided by 8), with shares between 0.107 and 0.141. So a model that always answers the most common class gets about 0.14 on the training labels (on the 300 test sequences the most common class happens to reach 0.187, which is just the luck of a small sample). Chance accuracy is 1/8 = 0.125. The last line is the important one: the label equals the last symbol in 0.127 of the sequences, the same as chance. The last symbol carries no information about the label, because the symbols are drawn independently.

   Tips and common mistakes:

   - The bar chart should look flat: the dashed line is the equal share (125 sequences per class) and the bars wobble around it by about 10-15%. A real imbalance would be a tall bar next to short ones.
   - Use different seeds for training and test data (here 1 and 2). If both used the same seed, the test set would be a copy of the training set and the accuracy would be meaningless.
   - A test set of 300 sequences gives accuracies with an error of about plus or minus 0.02, so chance level shows up anywhere between roughly 0.10 and 0.15.

   Try also (optional):

   - Make the data with `length=3` and print ten sequences. Where is the label? Is it easier to see by eye?
   - Change the number of training sequences to 100. Does the class balance get better or worse?

   *Exercise 1.1:* from the training labels `ytr` and the counts, compute `share_max`, the largest class share, and `share_last`, the share of training sequences whose label equals their last symbol. Why it matters: both numbers tell you what a "lazy" model could reach without any memory, and you will compare every later model with them.

```python
share_max = None    # the largest class share among the training labels (a number between 0 and 1)
share_last = None   # share of training sequences whose label equals their last symbol
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if share_max is None or share_last is None:
    print("Fill in share_max and share_last above, then run this cell again.")
else:
    check("share_max is the largest class share", abs(share_max - counts.max() / len(ytr)) < 1e-9, "Use counts.max() / len(ytr).")
    check("share_last is the share with label == last symbol", abs(share_last - float((Xtr[:, -1] == ytr).mean())) < 1e-9, "Compare ytr with Xtr[:, -1] and take the mean.")
    RESULTS["p1_share_max"] = round(float(share_max), 3); RESULTS["p1_share_last"] = round(float(share_last), 3)
```

   `share_max` is 0.141 (141 of 1,000 sequences, classes 4 and 5) and `share_last` is 0.127.

2. **Step 1.2. A baseline that cannot remember.** The simplest model that reads the sequence at all is one that looks only at the last symbol. The class `WindowModel(k)` below takes the last `k` symbols of each sequence (as one-hot vectors placed side by side) and applies one linear layer to them. With `k = 1` it sees one symbol. We train it with `train_seq_model` for 100 epochs with a learning rate of 0.01 and compare its accuracy with chance. `count_params` counts the numbers the model can adjust. Predict first: do you expect the training accuracy to be below, near or above the test accuracy?

```python
class WindowModel(nn.Module):
    """Sees only the last k symbols (one-hot vectors placed side by side) and applies ONE linear layer."""
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.lin = nn.Linear(k * N_SYMBOLS, N_SYMBOLS)
    def forward(self, x):                                  # x: (batch, length, 8)
        return self.lin(x[:, -self.k:, :].reshape(len(x), -1))

def count_params(model):
    """Number of trainable numbers in a PyTorch model."""
    return sum(p.numel() for p in model.parameters())

EP1 = 15 if QUICK else 100          # epochs for the small window models (QUICK is False on the Spark)
torch.manual_seed(SEED)
m_last = WindowModel(1)
h_last = train_seq_model(m_last, Xtr, ytr, Xte, yte, epochs=EP1, lr=0.01)
print("parameters:", count_params(m_last))
print(f"last-symbol model: train accuracy {h_last['train_acc'][-1]:.3f} | test accuracy {h_last['test_acc'][-1]:.3f} | chance {1 / N_SYMBOLS:.3f}")
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
parameters: 72
last-symbol model: train accuracy 0.173 | test accuracy 0.123 | chance 0.125
```

   You should see `parameters: 72` (a weight for each of 8 inputs and 8 classes is 64, plus 8 biases) and `last-symbol model: train accuracy 0.173 | test accuracy 0.123 | chance 0.125`. On new data the model is exactly at chance (0.123). On the training data it is a little better (0.173): with 72 free numbers and only 1,000 examples it finds some accidental pattern, which does not carry over to new sequences. That gap between training and test accuracy is the sign of a model that has nothing real to learn. The reason is the one from Step 1.1: the symbols are independent, so the last symbol says nothing about the first. Compare with your Imagine-first number for the last-symbol model: it should have been about 0.125, no better than guessing.

   Tips and common mistakes:

   - `reshape(len(x), -1)` flattens the last k one-hot vectors into one row of k times 8 numbers; the linear layer then has k times 8 inputs.
   - The model has no hidden layer, so it can only learn a weighted vote of the symbols it can see. That is enough for this task as soon as it can see the first symbol (Step 1.3).
   - If your test accuracy is far from 0.125 (say above 0.3) check that `Xte` was made with a different seed than `Xtr` and that you did not give the model the first symbol by mistake.

   Try also (optional):

   - Print `h_last['train_acc'][:5]` and `h_last['train_acc'][-5:]`. Does the training accuracy keep rising while the test accuracy does not?

   *Exercise 1.2:* train `WindowModel(2)` exactly like the one above (same seed `torch.manual_seed(SEED)` before building it, `EP1` epochs, `lr=0.01`) and store its final test accuracy in `acc_k2`. Why it matters: you test the guess "two symbols are almost as bad as one" before you change the window further in Step 1.3.

```python
acc_k2 = None   # test accuracy of a WindowModel(2) trained like the one above (same seed, EP1 epochs, lr=0.01)
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if acc_k2 is None:
    print("Train WindowModel(2) above, store its final test accuracy in acc_k2, then run this cell again.")
else:
    check("acc_k2 is a number between 0 and 1", 0 <= acc_k2 <= 1, "Use hist['test_acc'][-1].")
    check("two symbols are still not enough to see the first one (accuracy below 0.3)", acc_k2 < 0.3, "A window of 2 on a sequence of 10 cannot see symbol 1, so it should stay near 0.125.")
    RESULTS["p1_acc_k2"] = round(float(acc_k2), 3)
```

   `acc_k2` is about 0.147. The check only asks for a value below 0.3, because the exact digits can change a little between PyTorch versions.

3. **Step 1.3. A fixed window: the n-gram idea.** A window of the last k symbols is the n-gram idea of Week 6: predict from a fixed number of recent items and forget everything older. A model with a window of k symbols is like a (k+1)-gram model. Here we widen the window step by step on sequences of length 10: k = 1, 3, 5, 9 and 10. The first symbol sits 10 steps back from the end, so only the window with k = 10 can reach it. Predict the shape of the curve before you look: a slow climb, or something else?

```python
window_acc = {}
for k in [1, 3, 5, 9, 10]:
    torch.manual_seed(SEED)
    m = WindowModel(k)
    h = train_seq_model(m, Xtr, ytr, Xte, yte, epochs=EP1, lr=0.01)
    window_acc[k] = h["test_acc"][-1]
    print(f"window k = {k:>2}: parameters {count_params(m):>4} | train {h['train_acc'][-1]:.3f} | test {h['test_acc'][-1]:.3f}")
if not QUICK:
    assert window_acc[10] > 0.9 and window_acc[1] < 0.3     # sanity check: only the full window can see symbol 1
RESULTS["p1_window_acc"] = {k: round(v, 3) for k, v in window_acc.items()}
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
window k =  1: parameters   72 | train 0.173 | test 0.123
window k =  3: parameters  200 | train 0.208 | test 0.137
window k =  5: parameters  328 | train 0.227 | test 0.143
window k =  9: parameters  584 | train 0.288 | test 0.130
window k = 10: parameters  648 | train 1.000 | test 1.000
```

```python
plt.figure(figsize=(5, 3))
plt.plot(list(window_acc), list(window_acc.values()), "o-", color="tab:blue", label="window model")
plt.axhline(1 / N_SYMBOLS, color="gray", ls="--", label="chance 0.125")
plt.xlabel("window size k (the last k symbols)"); plt.ylabel("test accuracy"); plt.ylim(0, 1.05)
plt.title("Length 10: only the full window sees symbol 1"); plt.legend(); plt.tight_layout(); plt.show()
```

   You should see five lines. For k = 1, 3, 5 and 9 the test accuracy stays at chance: 0.123, 0.137, 0.143 and 0.130. The training accuracy creeps up (0.173, 0.208, 0.227, 0.288) as the number of parameters grows (72, 200, 328, 584), but this is only memorising noise: it never shows up on the test set. Then for k = 10 (648 parameters) both training and test accuracy jump to **1.000**. The plot is a flat line at 0.125 followed by a cliff, not a slope. There is no partial credit for a window that stops one step short of the first symbol: k = 9 is as bad as k = 1. The `assert` in the block checks this cliff, so a broken run is noticed immediately. This is the lesson's point about n-grams: a model with a limited window is blind to everything beyond it.

   Tips and common mistakes:

   - `window_acc` is a dictionary from k to the final test accuracy; `RESULTS` stores a rounded copy.
   - The learning is easy once the first symbol is visible: a linear layer can read it off directly. The difficulty of the task is entirely about *reach*, not about computing a complicated function.
   - Each run uses `torch.manual_seed(SEED)` before building the model, so the numbers are repeatable. A different PyTorch version can change the training accuracies in the last digit; the cliff between k = 9 and k = 10 will stay.

   Try also (optional):

   - Add k = 2, 4, 6, 7, 8 to the loop. Is there any value below 10 with a test accuracy clearly above 0.2?
   - Make the sequences of length 12 and find the smallest working k. Does your rule 'k must equal the length' still hold?

   *Exercise 1.3:* fill in `k_needed`, the smallest window size that lets a model see the first symbol of a length-10 sequence, and let the check compare it with the measurements. Why it matters: it turns the picture of the cliff into a rule you can reuse on longer sequences in Step 1.4.

```python
k_needed = None   # the smallest window size k that lets the model see the first symbol of a length-10 sequence
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if k_needed is None:
    print("Fill in k_needed above, then run this cell again.")
else:
    check("k_needed is 10", k_needed == Xtr.shape[1], "To see symbol 1 the window must reach back to position 1: it must cover the whole sequence.")
    check("it matches the measurements", window_acc[k_needed] > 0.9 and window_acc[k_needed - 1] < 0.3, "Compare window_acc for k = 9 and k = 10.")
```

   `k_needed` is 10.

4. **Step 1.4. Why longer sequences are harder to remember.** If the first symbol is always the answer, a window must cover the *whole* sequence. The first block prints, for lengths 5, 10, 20 and 40, the window you need and the number of parameters of that window model: by formula (length times 8 times 8 weights, plus 8 biases) and counted with `count_params`. The second block trains a full-window model at lengths 5, 20 and 40, and then shows what goes wrong when a window model meets a sequence of a different length than it was built for. The point: a window can be made to work, but only by knowing the length in advance and paying for it.

```python
print(f"{'length':>6} {'window needed':>14} {'parameters (formula)':>21} {'parameters (counted)':>21}")
for L in [5, 10, 20, 40]:
    formula = L * N_SYMBOLS * N_SYMBOLS + N_SYMBOLS          # weights k*8*8, plus 8 biases
    print(f"{L:>6} {L:>14} {formula:>21} {count_params(WindowModel(L)):>21}")
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
length  window needed  parameters (formula)  parameters (counted)
     5              5                   328                   328
    10             10                   648                   648
    20             20                  1288                  1288
    40             40                  2568                  2568
```

```python
long_acc = {}
for L in [5, 20, 40]:
    Xa, ya = make_memory_task(1000, L, seed=1); Xb, yb = make_memory_task(300, L, seed=2)
    torch.manual_seed(SEED)
    h = train_seq_model(WindowModel(L), Xa, ya, Xb, yb, epochs=EP1, lr=0.01)
    long_acc[L] = h["test_acc"][-1]
    print(f"full window at length {L:>2}: test accuracy {h['test_acc'][-1]:.3f}")
# A window model is tied to one length: a window of 20 cannot be fed sequences of length 10.
try:
    WindowModel(20)(one_hot_batch(make_memory_task(4, 10)[0]))
except RuntimeError:
    print("a window-20 model fed length-10 input -> RuntimeError: the shapes do not match")
# The other way round it does not even complain: a window-10 model fed length-20 input silently looks at the last 10 symbols only.
print("window-10 model on length-20 input, output shape:", tuple(WindowModel(10)(one_hot_batch(make_memory_task(4, 20)[0])).shape))
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
full window at length  5: test accuracy 1.000
full window at length 20: test accuracy 1.000
full window at length 40: test accuracy 0.950
a window-20 model fed length-10 input -> RuntimeError: the shapes do not match
window-10 model on length-20 input, output shape: (4, 8)
```

   You should see that the formula and the count agree: 328, 648, 1,288 and 2,568 parameters for lengths 5, 10, 20 and 40. The size grows in step with the length, and the window needed is always the whole sequence. The full-window models still learn: test accuracy 1.000 at lengths 5 and 20 and 0.950 at length 40 (100 epochs are a little short for 2,568 weights hiding one useful input in 40). The last two lines show the trap. A window-20 model fed length-10 input stops with a `RuntimeError` because the shapes do not match, which at least is loud. The other way round is worse: a window-10 model fed length-20 input runs without complaint and returns an answer of the right shape (`(4, 8)`), but it silently looks at the last 10 symbols only, so it cannot see the first symbol and is wrong 7 times out of 8. A real language has no length to know in advance: a document can be 10 words or 10,000.

   Tips and common mistakes:

   - The window needed here equals the length only because the label is the *first* symbol. For a sentence, the word that matters can be anywhere from 1 to hundreds of words back, so there is no good fixed k.
   - More parameters is not the real cost. The real cost is that the model has to be rebuilt and retrained for every length, and nothing it learned about position 1 is reused at position 2.
   - n-gram count tables of Week 6 are worse still: they grow exponentially with the window, not linearly.

   Try also (optional):

   - Train the full-window model at length 80 with 100 epochs. Does it still reach 0.9?
   - Compare `count_params(WindowModel(80))` with the number of training sequences (1,000). What does that suggest about overfitting?

   *Exercise 1.4:* compute `params_100`, the number of parameters of a full-window model for sequences of length 100, with the formula (length times 8 times 8, plus 8). The check compares it with a real `WindowModel(100)`. Why it matters: it shows how the cost of the window approach scales.

```python
params_100 = None   # number of parameters of a full-window model for sequences of length 100
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if params_100 is None:
    print("Fill in params_100 above, then run this cell again.")
else:
    check("params_100 matches the model", params_100 == count_params(WindowModel(100)), "Weights are length * 8 * 8, plus 8 biases.")
    RESULTS["p1_params_100"] = int(params_100)
```

   `params_100` is 6,408.

5. **Step 1.5. Imagine and write: far-away words in real language.** The memory task is artificial on purpose. Real language has the same problem, only with meaning instead of random symbols. Think of three kinds of sentence where the word you need is far back: **agreement** (the verb must agree with a subject that came many words earlier), **pronouns** (who is *she* or *it* in the sentence?), and **questions** (an answer that depends on a question asked many lines before). Write one sentence for each, mark the word that matters, and count how many words separate it from the place where a model needs it. This is the question the rest of the week answers.

   There is no single right answer, but a good example has the key word at least six words back, and removing it should make the next word unpredictable. For instance, in "The keys to the old cabinet in the corner of the classroom are missing" the verb *are* needs the plural *keys*, 11 words back, while the nearest noun, *classroom*, is singular and would give the wrong verb. A model with a window of five words would see *corner of the classroom* and be tempted by *is*. In "Ana lent her notebook to Maria, and after the long class on Friday she gave it back" the word *she* refers to Ana or to Maria, which only the earlier part of the sentence can settle. In a chat, a question such as "Which city is the capital of Japan?" followed by several lines of other text still has to be answered with Tokyo. In each case the useful word is far back and the words in between are mostly noise, exactly like the random symbols in the memory task.

   Try also (optional):

   - Find one example in your own language (Filipino, Cebuano, Ilocano or another) where a word at the start of a sentence decides a word at the end, such as an affix or a pronoun that marks focus.

   *Exercise 1.5:* fill in the dictionary `far_examples` with one sentence, the word that matters and its distance in words for each of the three kinds. Why it matters: it keeps the abstract task tied to language, and you will use these sentences again when you reflect in Part 7.

```python
far_examples = {
    "agreement": {"sentence": "", "far_word": "", "distance": 0},
    "pronoun": {"sentence": "", "far_word": "", "distance": 0},
    "question": {"sentence": "", "far_word": "", "distance": 0},
}   # write a sentence, the word that matters, and how many words back it is
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
filled = [v for v in far_examples.values() if v["sentence"].strip() and v["far_word"].strip() and v["distance"] > 0]
if len(filled) < 3:
    print(f"You have filled in {len(filled)} of 3 examples. Complete all three, then run this cell again.")
else:
    check("each far word appears in its sentence", all(v["far_word"].lower() in v["sentence"].lower() for v in filled), "The word that matters must be a word of your sentence.")
    check("each distance is at least 6 words", all(v["distance"] >= 6 for v in filled), "Choose examples where the word that matters is far back.")
    RESULTS["p1_far_examples"] = far_examples
```

   Any three examples with the key word at least six words back pass the check.

### Part 2. An RNN from scratch (25 min)

*Lesson section: "RNN".* A window forgets everything older than k steps. A recurrent neural network (RNN) tries something different: it carries a small vector, the **hidden state**, from one step to the next, and updates it every time a new symbol arrives: new state = tanh(Wx times the input + Wh times the old state + b). Nothing else is remembered. Everything the model knows about symbol 1 when it reaches symbol 10 has to survive inside that one vector. In this part you write the recurrence by hand in NumPy, check it against PyTorch's `nn.RNN`, see that the same weights are used at every step, and then train an RNN classifier on the memory task at growing lengths and watch where it stops learning. These are the numbers behind the lesson's figure of the unrolled RNN.

**Imagine first.** Predict the test accuracy of a small RNN (32 hidden units) trained for 150 epochs on the memory task at length 10, and at length 60. Write both numbers and one sentence about why one might be harder than the other. Think about what has to happen to the information about symbol 1 in each case. You will compare with the measured table in Step 2.3.

```python
# Imagine first: can a small RNN remember the first symbol?
my_acc_len10 = None   # predicted test accuracy of a small RNN on length 10
my_acc_len60 = None   # predicted test accuracy of the same RNN on length 60
my_reason = None      # one sentence: what makes the long one harder?
```

1. **Step 2.1. The recurrence by hand.** The whole RNN is a loop of three lines. At each step the new hidden state is `tanh(Wx @ x_t + Wh @ h + b)`, where `Wx` turns the input into the state's units, `Wh` mixes the old state, and `b` is a bias. We start from a state of all zeros. The first block defines `rnn_forward` and runs it on a tiny example with two input numbers, two hidden units and three steps, with small matrices written out in the code so that you can follow the arithmetic with a calculator. The second block copies the same numbers into PyTorch's `nn.RNN` and asserts that both give the same states, then compares the parameter counts. PyTorch has two bias vectors (`bias_ih` and `bias_hh`) where we use one, so we set the second to zero.

```python
def rnn_forward(x_seq, Wx, Wh, b):
    """A vanilla RNN written by hand: h_t = tanh(Wx x_t + Wh h_(t-1) + b), starting from h_0 = 0.
    x_seq has shape (T, I), Wx (H, I), Wh (H, H), b (H,). Returns all hidden states as an array (T, H)."""
    h = np.zeros(Wh.shape[0])
    states = []
    for x_t in x_seq:
        h = np.tanh(Wx @ x_t + Wh @ h + b)
        states.append(h)
    return np.array(states)

Wx = np.array([ [0.5, -0.3], [0.2, 0.8] ])      # (H=2, I=2): how the input enters the state
Wh = np.array([ [0.6, 0.1], [-0.4, 0.5] ])      # (H=2, H=2): how the old state feeds the new one
b = np.array([0.1, -0.1])
x_seq = np.array([ [1.0, 0.0], [0.0, 1.0], [1.0, 1.0] ])   # three steps, two input numbers each
H_by_hand = rnn_forward(x_seq, Wx, Wh, b)
for t, h in enumerate(H_by_hand, 1):
    print(f"h_{t} =", np.round(h, 4))
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
h_1 = [0.537  0.0997]
h_2 = [0.1314 0.4892]
h_3 = [0.4035 0.7976]
```

```python
rnn = nn.RNN(input_size=2, hidden_size=2, batch_first=True)
with torch.no_grad():                              # copy OUR numbers into PyTorch's RNN
    rnn.weight_ih_l0.copy_(torch.tensor(Wx)); rnn.weight_hh_l0.copy_(torch.tensor(Wh))
    rnn.bias_ih_l0.copy_(torch.tensor(b)); rnn.bias_hh_l0.zero_()      # PyTorch has two bias vectors; we use one
out, _ = rnn(torch.tensor(x_seq, dtype=torch.float32).unsqueeze(0))
H_torch = out[0].detach().numpy()
print("PyTorch :", np.round(H_torch.astype(float), 4).tolist())
print("by hand :", np.round(H_by_hand, 4).tolist())
assert np.allclose(H_by_hand, H_torch, atol=1e-6)
print("the two agree: our five lines of NumPy are an RNN")

I_, H_ = 2, 2
print("parameters by hand: H*H + I*H + H =", H_ * H_ + I_ * H_ + H_, "| nn.RNN counts", count_params(rnn), "(PyTorch keeps a second bias vector of size H)")
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
PyTorch : [ [0.537, 0.0997], [0.1314, 0.4892], [0.4035, 0.7976] ]
by hand : [ [0.537, 0.0997], [0.1314, 0.4892], [0.4035, 0.7976] ]
the two agree: our five lines of NumPy are an RNN
parameters by hand: H*H + I*H + H = 10 | nn.RNN counts 12 (PyTorch keeps a second bias vector of size H)
```

   You should see `h_1 = [0.537 0.0997]`, `h_2 = [0.1314 0.4892]` and `h_3 = [0.4035 0.7976]`. Check the first one by hand: the input of step 1 is (1, 0), so `Wx @ x` is (0.5, 0.2); the old state is zero, so `Wh @ h` is (0, 0); adding the bias (0.1, -0.1) gives (0.6, 0.1); and tanh gives (0.537, 0.0997). For step 2 the input (0, 1) gives (-0.3, 0.8), the old state gives (0.3322, -0.1649), the bias adds (0.1, -0.1), the total is (0.1322, 0.5351) and tanh gives (0.1314, 0.4892). The PyTorch states are identical (the `assert` passes), so five lines of NumPy are an RNN. The parameter line says: by hand H times H plus I times H plus H is 4 + 4 + 2 = 10, while `nn.RNN` counts 12 because of its second bias vector.

   Tips and common mistakes:

   - `Wx` has shape (H, I) and `Wh` has shape (H, H): the rows are the hidden units. That is why the product `Wx @ x_t` has H numbers.
   - `rnn.weight_ih_l0` means 'input to hidden, layer 0' and `weight_hh_l0` means 'hidden to hidden, layer 0'. The names are PyTorch's, the matrices are the ones you wrote.
   - The state is squashed by tanh into the range from -1 to 1 at every step. Remember this when you see the gradients in Part 3.

   Try also (optional):

   - Change `x_seq` to a sequence of four identical inputs `[1, 1]`. Does the state settle to a fixed value? (The exercise below asks for its value after four steps.)
   - Set `Wh` to all zeros and run `rnn_forward` again. What does the state depend on now?

   *Exercise 2.1:* run `rnn_forward` on the 4-step input of ones `x_ones` with the same matrices, store the final state in `h_new`, and compute `params_formula`, the parameter count H times H plus I times H plus H, for a network with I = 8 inputs and H = 32 hidden units (the size used in Step 2.3). Why it matters: you check that you can run the recurrence and count its weights.

```python
h_new = None            # final hidden state (array of 2 numbers) for the 4-step input of ones given in the next line
x_ones = np.ones((4, 2))
params_formula = None   # parameters H*H + I*H + H for an RNN with I = 8 inputs and H = 32 hidden units
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if h_new is None or params_formula is None:
    print("Fill in h_new and params_formula above, then run this cell again.")
else:
    truth = rnn_forward(x_ones, Wx, Wh, b)[-1]
    check("h_new is the last state of rnn_forward on x_ones", np.allclose(h_new, truth, atol=1e-6), "Call rnn_forward(x_ones, Wx, Wh, b) and take the last row.")
    check("params_formula is H*H + I*H + H", params_formula == 32 * 32 + 8 * 32 + 32, "32*32 + 8*32 + 32.")
    check("PyTorch counts one more bias vector", count_params(nn.RNN(8, 32)) == params_formula + 32, "nn.RNN has bias_ih and bias_hh.")
    RESULTS["p2_params_I8_H32"] = int(params_formula)
```

   `h_new` is about (0.6263, 0.7876) and `params_formula` is 1,312 (PyTorch counts 1,344 because of its second bias).

2. **Step 2.2. Unrolling and weight sharing.** Drawn out in time, the RNN looks like a very deep network with one layer per step. But those layers are not independent: the same `Wx`, `Wh` and `b` are used at every step. The first block records the memory address of the `Wh` array at each of 50 steps and asserts that there is only one. The second block builds `nn.RNN(8, 32)` and runs it on sequences of length 5 and 50, printing its parameter count each time. The consequence is large: the number of parameters does not depend on how long the sequence is, which the window model of Part 1 could not do.

```python
def rnn_forward_trace(x_seq, Wx, Wh, b):
    """Same as rnn_forward but also returns, for every step, the memory address of the matrix Wh that was used."""
    h = np.zeros(Wh.shape[0]); addresses = []
    for x_t in x_seq:
        addresses.append(Wh.__array_interface__["data"][0])
        h = np.tanh(Wx @ x_t + Wh @ h + b)
    return h, addresses

_, addr = rnn_forward_trace(np.random.RandomState(0).randn(50, 2), Wx, Wh, b)
print("steps:", len(addr), "| different Wh matrices used:", len(set(addr)))
assert len(set(addr)) == 1
for T in [5, 50]:
    r = nn.RNN(8, 32, batch_first=True)
    out, _ = r(torch.randn(1, T, 8))
    print(f"nn.RNN(8, 32) run on length {T:>2}: output {tuple(out.shape)}, parameters {count_params(r)}")
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
steps: 50 | different Wh matrices used: 1
nn.RNN(8, 32) run on length  5: output (1, 5, 32), parameters 1344
nn.RNN(8, 32) run on length 50: output (1, 50, 32), parameters 1344
```

   You should see `steps: 50 | different Wh matrices used: 1`: all 50 steps multiply by the very same array. The second part prints `output (1, 5, 32), parameters 1344` and `output (1, 50, 32), parameters 1344`: the same module handles length 5 and length 50 and has 1,344 parameters both times (1,312 + 32 for the second bias vector). Compare with Step 1.4, where a window model for length 40 needed 2,568 parameters and had to be rebuilt for every length. Weight sharing is what makes an RNN able to read any length. It is also the root of its main problem: since the same `Wh` is applied again and again, what happens to a signal after 50 steps is Wh multiplied by itself 50 times.

   Tips and common mistakes:

   - `Wh.__array_interface__['data'][0]` is the address of the array's data in memory; two names for the same array give the same address.
   - The module `nn.RNN(8, 32)` does not know the length of the sequence it will see. The length only decides how many times the loop runs.
   - Sharing weights across time is the same idea as sharing a filter across positions in a convolutional network (Week 4): one small set of numbers, used everywhere.

   Try also (optional):

   - Count the parameters of `nn.RNN(8, 64)` and compare with 1,344. Which grows faster, the hidden size or the length?

   *Exercise 2.2:* store in `count_len5` and `count_len500` the parameter count of an `nn.RNN(8, 32)` used for sequences of length 5 and of length 500. Why it matters: it checks that you have understood that the length does not enter the count.

```python
count_len5 = None    # parameters of an nn.RNN(8, 32) used for sequences of length 5
count_len500 = None  # parameters of the same nn.RNN used for sequences of length 500
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if count_len5 is None or count_len500 is None:
    print("Fill in both numbers above, then run this cell again.")
else:
    check("the count does not depend on the length", count_len5 == count_len500 == count_params(nn.RNN(8, 32)), "The same weights are used at every step, so the length does not matter.")
```

   Both are 1,344.

3. **Step 2.3. Build the classifier and train it at growing lengths.** Now we use the RNN as a memory machine. `RNNClassifier(hidden)` reads a one-hot sequence with `nn.RNN`, takes the hidden state after the **last** step, and maps it to 8 class scores with `nn.Linear`. For the answer to be right, the last state must still contain symbol 1. We train it with `train_seq_model` for `EP2 = 150` epochs, with 32 hidden units and a learning rate of 0.003 (a smaller step than in Part 1; Step 2.3b shows why), on 1,000 training and 300 test sequences at lengths 5, 10 and 20. The first block prints the final test accuracy and the first epoch at which the test accuracy reaches 0.9. The second block repeats length 20 with a learning rate of 0.01. The third block pushes on to lengths 40 and 60 and prints a summary table. Training in the third block takes about half a minute.

```python
class RNNClassifier(nn.Module):
    """One-hot symbols -> nn.RNN -> the LAST hidden state -> a linear layer with 8 class scores."""
    def __init__(self, hidden=32):
        super().__init__()
        self.rnn = nn.RNN(N_SYMBOLS, hidden, batch_first=True)
        self.out = nn.Linear(hidden, N_SYMBOLS)
    def forward(self, x):
        states, _ = self.rnn(x)              # (batch, length, hidden): the state after every step
        return self.out(states[:, -1])       # only the state after the LAST step is used to answer

EP2 = 12 if QUICK else 150               # epochs for the RNNs
def make_split(length, n_train=1000, n_test=300):
    return make_memory_task(n_train, length, seed=1) + make_memory_task(n_test, length, seed=2)

rnn_hist, rnn_final = {}, {}
for L in [5, 10, 20]:
    Xa, ya, Xb, yb = make_split(L)
    torch.manual_seed(SEED)
    rnn_hist[L] = train_seq_model(RNNClassifier(32), Xa, ya, Xb, yb, epochs=EP2, lr=0.003)
    rnn_final[L] = rnn_hist[L]["test_acc"][-1]
    first = next((e + 1 for e, a in enumerate(rnn_hist[L]["test_acc"]) if a >= 0.9), None)
    print(f"length {L:>2}: final test accuracy {rnn_final[L]:.3f} | first epoch with test accuracy >= 0.9: {first}")
if not QUICK:
    assert rnn_final[5] > 0.9          # sanity check: the RNN can learn the easiest length
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
length  5: final test accuracy 1.000 | first epoch with test accuracy >= 0.9: 4
length 10: final test accuracy 1.000 | first epoch with test accuracy >= 0.9: 7
length 20: final test accuracy 1.000 | first epoch with test accuracy >= 0.9: 15
```

```python
# The same task at length 20, but with a larger learning rate (0.01 instead of 0.003).
Xa, ya, Xb, yb = make_split(20)
torch.manual_seed(SEED)
h_fast = train_seq_model(RNNClassifier(32), Xa, ya, Xb, yb, epochs=EP2, lr=0.01)
print(f"length 20, lr = 0.01: final test accuracy {h_fast['test_acc'][-1]:.3f} | best epoch {max(h_fast['test_acc']):.3f} | final train {h_fast['train_acc'][-1]:.3f}")
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
length 20, lr = 0.01: final test accuracy 0.390 | best epoch 0.693 | final train 0.516
```

```python
for L in [40, 60]:
    Xa, ya, Xb, yb = make_split(L)
    torch.manual_seed(SEED)
    rnn_hist[L] = train_seq_model(RNNClassifier(32), Xa, ya, Xb, yb, epochs=EP2, lr=0.003)
    rnn_final[L] = rnn_hist[L]["test_acc"][-1]
    print(f"length {L:>2}: final test accuracy {rnn_final[L]:.3f} | best {max(rnn_hist[L]['test_acc']):.3f} | final train {rnn_hist[L]['train_acc'][-1]:.3f}")
RESULTS["p2_rnn_test_acc"] = {L: round(a, 3) for L, a in rnn_final.items()}
print("\nlength  final test accuracy")
for L, a in rnn_final.items():
    print(f"{L:>6}  {a:.3f}")
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
length 40: final test accuracy 1.000 | best 1.000 | final train 1.000
length 60: final test accuracy 0.097 | best 0.160 | final train 0.551

length  final test accuracy
     5  1.000
    10  1.000
    20  1.000
    40  1.000
    60  0.097
```

   You should see that the RNN *learns the task*: final test accuracy **1.000** at lengths 5, 10 and 20, reaching 0.9 after 4, 7 and 15 epochs. So a vanilla RNN can remember a symbol across 19 steps. But the longer the sequence, the longer it takes (4, 7, 15 epochs), and then it stops. At length 40 the same model still ends at 1.000, but it needs 71 epochs to pass 0.9 (Step 2.4). At length 60 it fails: final test accuracy 0.097 (best epoch 0.160, chance 0.125). The training accuracy of that run is 0.551: the network has *memorised* some of the 1,000 training sequences instead of finding the rule, which is what a model does when it cannot see how to carry the signal. This is the finding of the part, and you should state it honestly: with these settings an RNN works up to length 40 and fails at 60. Two cautions. First, the exact length at which it breaks depends on the learning rate, the seed and even the number of CPU threads: in other runs with the same code, length 40 ended at 0.91 and not 1.00. Second, Step 2.3b shows that the learning rate matters as much as the length: at length 20 with a learning rate of 0.01, the same network does not learn (final test accuracy 0.390, best epoch 0.693, final train 0.516; it wobbles from run to run between about 0.25 and 0.4). A big step makes the gradient noisy, and the gradient from step 20 back to step 1 is already small and delicate (Part 3).

   Tips and common mistakes:

   - `states[:, -1]` takes the hidden state after the last step for every sequence in the batch. Using only that state is what forces the model to remember.
   - `train_seq_model` clips the gradient norm to 1.0. Without it the 0.01 runs would be even more unstable; you will see why in Step 3.4.
   - Do not read 0.097 as 'worse than chance'. With 300 test sequences, accuracies between 0.09 and 0.16 are all consistent with guessing.

   Try also (optional):

   - Train at length 60 for 400 epochs (change `EP2`) and watch whether it eventually learns. It may take two minutes, so run it in its own cell.
   - Change the hidden size from 32 to 128 at length 60. Does a bigger state remember better?

   *Exercise 2.3:* train `RNNClassifier(4)`, with only 4 hidden units, on length 10 (use `make_split(10)`, `torch.manual_seed(SEED)` before the model, `epochs=EP2`, `lr=0.003`) and store the final test accuracy in `acc_h4`. Why it matters: you see how much room the state needs to hold even one symbol out of 8.

```python
acc_h4 = None    # final test accuracy of RNNClassifier(4) on length 10 (seed 42, lr 0.003, EP2 epochs)
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if acc_h4 is None:
    print("Train RNNClassifier(4) on length 10, store the final test accuracy in acc_h4, then run this cell again.")
else:
    Xa, ya, Xb, yb = make_split(10)
    torch.manual_seed(SEED)
    truth = train_seq_model(RNNClassifier(4), Xa, ya, Xb, yb, epochs=EP2, lr=0.003)["test_acc"][-1]
    check("acc_h4 matches a run with the same seed", abs(acc_h4 - truth) < 0.02, "Use make_split(10), torch.manual_seed(SEED) before the model, epochs=EP2, lr=0.003.")
    RESULTS["p2_acc_h4"] = round(float(acc_h4), 3)
```

   `acc_h4` is about 0.227: four hidden units can hold a little of the information (well above chance) but not enough to carry one of eight symbols reliably.

4. **Step 2.4. Learning curves.** A final accuracy hides *how* a model learned. The block below plots the test accuracy against the epoch for each length that was trained in Step 2.3. Look for two things: when the curve leaves the chance line, and how steep it is afterwards. The longer the sequence, the longer the model sits at chance before it finds the signal.

```python
plt.figure(figsize=(6.5, 3.4))
for L, c in zip([5, 10, 20, 40, 60], ["tab:green", "tab:blue", "tab:orange", "tab:red", "tab:purple"]):
    if L in rnn_hist:
        plt.plot(range(1, len(rnn_hist[L]["test_acc"]) + 1), rnn_hist[L]["test_acc"], color=c, label=f"length {L}")
plt.axhline(1 / N_SYMBOLS, color="gray", ls="--", label="chance")
plt.xlabel("epoch"); plt.ylabel("test accuracy"); plt.title("RNN learning curves: longer sequences learn later, or never")
plt.legend(fontsize=8); plt.tight_layout(); plt.show()
```

   You should see five curves. The length-5 curve (green) shoots up within a few epochs and length 10 (blue) a little later; length 20 (orange) wobbles around 0.3 to 0.6 for the first 10 epochs, then climbs and passes 0.9 at epoch 15. The length-40 curve (red) is much more erratic: it sits at chance for about 10 epochs, jumps to 0.51 at epoch 21, falls back to 0.21 at epoch 41, then climbs slowly (0.65, 0.81, 0.90 at epochs 51, 61, 71) and reaches 1.0 at about epoch 120. The length-60 curve (purple) never leaves the chance line: it stays between 0.10 and 0.16 for all 150 epochs. The epochs to reach 0.9 are 4, 7, 15 and 71 for lengths 5, 10, 20 and 40. Notice the growth: doubling the length from 20 to 40 made the wait almost five times longer, and the path becomes bumpy because the gradient that links the last answer to symbol 1 is small and noisy. For length 60 within 150 epochs the model never finds a path. Longer sequences need more training or are impossible, because the signal that tells the model whether it is on the right track gets weaker with distance (Part 3 measures this).

   Tips and common mistakes:

   - The flat start of the curves is not a bug: the network is truly at chance until the weights start to carry information across all the steps. Only the fully trained value matters for the final table; the path there is a good diagnostic.
   - Because test and training sets are fixed, small steps in the curve are real changes of a few sequences out of 300 (each is 0.0033).

   Try also (optional):

   - Plot the training loss (`rnn_hist[L]['loss']`) instead of the test accuracy. Is the loss also flat before the jump?

   *Exercise 2.4:* for lengths 5, 10, 20 and 40, find the first epoch (counting from 1) at which the test accuracy reaches at least 0.9, and store them in the dictionary `epoch_to_90`. Why it matters: it turns the picture into numbers you can compare with the LSTM in Part 4.

```python
epoch_to_90 = None   # dict: for each length in 5, 10, 20, 40 the first epoch (counting from 1) with test accuracy of at least 0.9
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if epoch_to_90 is None:
    print("Fill in epoch_to_90 above (a dict like {5: 3, 10: 7, ...}), then run this cell again.")
else:
    def first_epoch(L):
        return next((e + 1 for e, a in enumerate(rnn_hist[L]["test_acc"]) if a >= 0.9), None)
    check("epoch_to_90 matches the learning curves", all(epoch_to_90.get(L) == first_epoch(L) for L in [5, 10, 20, 40]), "For each length, find the first index where rnn_hist[L]['test_acc'] reaches 0.9 and add 1.")
    RESULTS["p2_epoch_to_90"] = epoch_to_90
```

   `epoch_to_90` is `{5: 4, 10: 7, 20: 15, 40: 71}`.

5. **Step 2.5. Imagine and write: where does the information travel?.** Look at the unrolled RNN of the lesson: symbol 1 enters at the first step, and the answer is read after the last step. In between, the information about symbol 1 can only travel through the hidden state. At every step the old state is multiplied by `Wh` and squashed by tanh, so it is partly overwritten by the new symbol and partly shrunk. Write down, for a sequence of length 20 and of length 60, how many times the memory of symbol 1 has been multiplied by `Wh` and passed through tanh when the answer is read, and what you expect this to do to a small signal.

   Symbol 1 enters at step 1. Steps 2 to 20 each apply `Wh` and tanh once, so at length 20 it has been through **19** multiplications and **19** tanh; at length 60 it has been through **59**. The general rule is length minus one. If each pass keeps, say, half of the signal, 19 passes leave one half-millionth (0.5 to the power 19 is 1.9e-6) and 59 passes leave 1.7e-18. A gradient is a signal travelling the other way, back along the same path, and so it shrinks in the same way; that is why training gets slower with length and finally fails. A real model can sometimes keep more than half, which is why length 40 still works here, but each extra step makes it harder. That is Part 3.

   *Exercise 2.5:* store in `n_mult_20` and `n_mult_60` the number of times the memory of symbol 1 is multiplied by `Wh` (and passed through tanh) before the answer is read, at length 20 and at length 60. Why it matters: this count is the exponent in every vanishing-gradient formula of Part 3.

```python
n_mult_20 = None   # how many times is the memory of symbol 1 multiplied by Wh (and passed through tanh) before the answer at length 20?
n_mult_60 = None   # the same at length 60
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if n_mult_20 is None or n_mult_60 is None:
    print("Fill in both numbers above, then run this cell again.")
else:
    check("length 20 gives 19", n_mult_20 == 19, "Symbol 1 enters at step 1; steps 2 to 20 each apply Wh and tanh once: 20 - 1.")
    check("length 60 gives 59", n_mult_60 == 59, "60 - 1.")
    RESULTS["p2_n_mult"] = (int(n_mult_20), int(n_mult_60))
```

   `n_mult_20` is 19 and `n_mult_60` is 59.

### Part 3. Gradients through time (25 min)

*Lesson sections: "RNN" and "LSTM".* In Part 2 the RNN learned slowly and then not at all as the sequences grew. The reason lies in the gradient, the signal that tells each weight how to change. To learn that symbol 1 matters, the error at the last step must be sent back along the whole chain of steps to step 1. At each step back it is multiplied by the factor w times (1 minus h squared) (the derivative of the recurrence), and a product of many numbers smaller than 1 collapses toward zero. This is the **vanishing gradient**. In this part you compute it by hand for the smallest possible RNN (one number as hidden state), verify it with PyTorch's autograd, measure it in a real `nn.RNN`, see how the recurrent weight changes it (including the opposite problem, **exploding gradients**, and the fix of gradient clipping), and compare with Week 3's table of 0.25 to the power n. These are the numbers behind the lesson's figure of gradient flow through time.

**Imagine first.** The scalar RNN is h_t = tanh(w h_(t-1) + u x_t) with w = 0.9 and u = 0.8, and inputs of plus or minus 1. The gradient of the last state with respect to the first state is a product of T minus 1 factors, each at most 0.9. Before you run anything, write your estimate of that gradient for T = 10, T = 30 and T = 60. It is enough to give the order of magnitude, such as 1e-3. You will compare it with the measurements in Step 3.1.

```python
# Imagine first: how large is the gradient that reaches step 1?
my_grad_T10 = None    # your guess for the gradient from the last step back to step 1, T = 10, w = 0.9 (a number, e.g. 1e-3)
my_grad_T30 = None    # T = 30
my_grad_T60 = None    # T = 60
```

1. **Step 3.1. The scalar RNN gradient by hand.** Take the smallest RNN: the state is one number, `h_t = tanh(w * h_(t-1) + u * x_t)`, with inputs `x_t` equal to +1 or -1 drawn at random. The chain rule says that the influence of an early state on the last one is a product: `dh_T/dh_t` equals `dh_T/dh_(t+1)` times `dh_(t+1)/dh_t`, and `dh_(t+1)/dh_t` is `w * (1 - h_(t+1)^2)` (the derivative of tanh is one minus tanh squared). The function `scalar_rnn_influence` computes this product from the end backwards and returns the array `infl`, where `infl[t]` is `dh_T/dh_t` and `infl[T]` is 1 (a state influences itself with weight 1; the code asserts it). The first block prints the influence of step 1 for T = 10, 30 and 60, and of step T minus 1 for comparison. The second block plots all of it for T = 30 on a log scale. The third block repeats the experiment with 300 different random input signals, because one signal is only one example.

```python
def scalar_rnn_influence(T, w=0.9, u=0.8, seed=0):
    """Scalar RNN h_t = tanh(w h_(t-1) + u x_t) with random inputs x_t = +1 or -1 and h_0 = 0.
    Returns x (index 1..T), the states h (index 0..T) and infl, where infl[t] = dh_T / dh_t for t = 1..T.
    By the chain rule infl[t] = infl[t+1] * dh_(t+1)/dh_t = infl[t+1] * w * (1 - h_(t+1)^2), and infl[T] = 1."""
    r = np.random.RandomState(seed)
    x = np.concatenate([ [0.0], r.choice([-1.0, 1.0], size=T)])
    h = np.zeros(T + 1)
    for t in range(1, T + 1):
        h[t] = np.tanh(w * h[t - 1] + u * x[t])
    infl = np.ones(T + 1)
    for t in range(T - 1, 0, -1):
        infl[t] = infl[t + 1] * w * (1 - h[t + 1] ** 2)
    infl[0] = np.nan
    return x, h, infl

for T in [10, 30, 60]:
    x, h, infl = scalar_rnn_influence(T, w=0.9)
    assert infl[T] == 1.0
    print(f"T = {T:>2}: influence of step 1 on step T = {infl[1]:.2e} | influence of step T - 1 = {infl[T - 1]:.3f}")
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
T = 10: influence of step 1 on step T = 1.09e-05 | influence of step T - 1 = 0.127
T = 30: influence of step 1 on step T = 8.72e-14 | influence of step T - 1 = 0.899
T = 60: influence of step 1 on step T = 7.42e-24 | influence of step T - 1 = 0.886
```

```python
x, h, infl = scalar_rnn_influence(30, w=0.9)
plt.figure(figsize=(6, 3))
plt.semilogy(range(1, 31), infl[1:], "o-", color="tab:red")
plt.xlabel("time step t"); plt.ylabel("dh_30 / dh_t  (log scale)"); plt.title("w = 0.9: the past fades fast")
plt.tight_layout(); plt.show()
```

```python
# One input signal is only one example. Repeat with 300 different random +-1 signals and look at the typical (geometric mean) influence.
print(f"{'T':>3} {'typical influence of step 1':>28} {'smallest':>10} {'largest':>10}")
for T in [10, 30, 60]:
    vals = np.array([scalar_rnn_influence(T, w=0.9, seed=s)[2][1] for s in range(300)])
    print(f"{T:>3} {10 ** np.log10(vals).mean():>28.2e} {vals.min():>10.1e} {vals.max():>10.1e}")
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
  T  typical influence of step 1   smallest    largest
 10                     4.48e-04    1.5e-08    3.7e-02
 30                     7.07e-12    8.4e-19    6.5e-07
 60                     2.30e-23    6.1e-31    8.1e-16
```

   You should see `T = 10: influence of step 1 on step T = 1.09e-05`, `T = 30: 8.72e-14` and `T = 60: 7.42e-24`. The last step's own neighbour (step T minus 1) still has an influence of 0.13 to 0.9, but step 1 has a gradient of about one hundred-thousandth at T = 10, one in ten trillion at T = 30, and one in 10 to the 23 at T = 60. The plot is a nearly straight line going down on a log scale: every step back multiplies the influence by roughly the same factor. The repeated experiment gives the *typical* (geometric mean) influence of step 1 over 300 signals: 4.5e-04 at T = 10, 7.1e-12 at T = 30 and 2.3e-23 at T = 60, with a big spread between signals (at T = 30 from 8e-19 to 6.5e-07). Your estimates from the Imagine-first cell: if you guessed "about 0.9 to the power T minus 1" you would have said 0.39 for T = 10; the true gradients are far smaller, because the factor (1 minus h squared) is also below 1 and is very small when the state is near plus or minus 1. **Compare with the lesson:** the lesson's figure uses its own fixed input signal and reports about 7.8e-5 at T = 10 and 3.6e-11 at T = 30 for the same w = 0.9. Our single seed gave 1.1e-05 and 8.7e-14, and the typical values differ again, because the numbers depend on which inputs happen to be drawn. All of them show the same collapse (about four powers of ten lost for every 10 steps), and the lesson's values lie inside the range of the 300 signals.

   Tips and common mistakes:

   - `np.random.RandomState(seed).choice([-1.0, 1.0], size=T)` makes the same input signal every time for the same seed, so the numbers are repeatable. Changing the seed changes the details but not the collapse.
   - The product runs backwards: `infl[t] = infl[t + 1] * w * (1 - h[t + 1] ** 2)`. The loop starts at `T - 1` and goes down to 1.
   - A semi-log plot (`plt.semilogy`) turns a product of equal factors into a straight line, so the slope is the average factor per step.

   Try also (optional):

   - Change `u` from 0.8 to 0.1. With small inputs the state stays near zero, so (1 minus h squared) is close to 1 and the factor is close to w. What does the influence at T = 30 become?
   - Plot `infl` for three different seeds in one figure.

   *Exercise 3.1:* store in `log10_60` the base-10 logarithm of the influence of step 1 on step 60 for w = 0.9, u = 0.8 and seed 0, which tells you "how many powers of ten are lost". Why it matters: the logarithm is the natural way to talk about numbers this small.

```python
log10_60 = None   # log10 of the influence of step 1 on step 60 for w = 0.9, u = 0.8, seed 0 (a negative number)
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if log10_60 is None:
    print("Fill in log10_60 above, then run this cell again.")
else:
    truth = float(np.log10(scalar_rnn_influence(60, w=0.9)[2][1]))
    check("log10_60 matches", abs(log10_60 - truth) < 1e-6, "Take np.log10 of scalar_rnn_influence(60, w=0.9)[2][1].")
    RESULTS["p3_log10_60"] = round(float(log10_60), 2)
```

   `log10_60` is about -23.13 (the influence is 7.4e-24).

2. **Step 3.2. Verify by autograd.** A hand formula is only worth trusting if a second method agrees with it. PyTorch's autograd computes derivatives automatically by running the chain rule on the recorded computation. We rebuild the same scalar RNN (same inputs, in 64-bit numbers for accuracy), call `retain_grad()` on every intermediate state so that its gradient is kept, run `hs[T].backward()`, and read `hs[t].grad`, which is exactly `dh_T/dh_t`. Then we compare with the product from Step 3.1 for every t.

```python
T, w, u = 30, 0.9, 0.8
x, h, infl = scalar_rnn_influence(T, w, u)
hs = [torch.tensor(0.0, dtype=torch.float64, requires_grad=True)]        # h_0
for t in range(1, T + 1):
    hs.append(torch.tanh(w * hs[-1] + u * x[t]))
    hs[-1].retain_grad()                                                   # keep the gradient of this intermediate value
hs[T].backward()                                                          # dh_T / dh_t arrives in hs[t].grad
auto = np.array([hs[t].grad.item() for t in range(1, T + 1)])
print("by hand  dh_30/dh_1 =", f"{infl[1]:.4e}")
print("autograd dh_30/dh_1 =", f"{auto[0]:.4e}")
print("largest relative difference over all steps:", f"{np.max(np.abs(auto - infl[1:]) / infl[1:]):.1e}")
assert np.allclose(auto, infl[1:], rtol=1e-6)
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
by hand  dh_30/dh_1 = 8.7247e-14
autograd dh_30/dh_1 = 8.7247e-14
largest relative difference over all steps: 1.3e-15
```

   You should see `by hand dh_30/dh_1 = 8.7247e-14` and `autograd dh_30/dh_1 = 8.7247e-14`, and `largest relative difference over all steps: 1.3e-15`: the two methods agree to 15 digits, which is the rounding of 64-bit numbers. The `assert` passes silently. So the formula 'product of w times (1 minus h squared)' is not an approximation or a story: it is what backpropagation through time computes. The vanishing gradient is a property of this product.

   Tips and common mistakes:

   - Intermediate tensors do not keep their `.grad` unless you ask with `retain_grad()`; only 'leaf' tensors (like the initial state) do. Without it, `hs[t].grad` would be `None`.
   - `hs[T].backward()` makes the gradient of hs[T] with respect to itself equal to 1, which is the starting value `infl[T] = 1`.
   - We used `dtype=torch.float64`. In 32-bit numbers a value like 1e-24 is still representable, but accuracy is lost in the long product.

   Try also (optional):

   - Print `auto[:5]` and `infl[1:6]` and compare them by eye.
   - Use `w = 1.0` and `u = 0` (the state stays 0 and the factor is exactly 1). What should the gradient be? Check.

   *Exercise 3.2:* store in `auto_h15` the autograd value of `dh_30/dh_15`, the influence of the middle step on the last one. Why it matters: it makes you read the gradient from the right place in the autograd graph.

```python
auto_h15 = None   # the autograd value of dh_30/dh_15 (a number)
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if auto_h15 is None:
    print("Fill in auto_h15 above, then run this cell again.")
else:
    check("auto_h15 is dh_30/dh_15", abs(auto_h15 - hs[15].grad.item()) < 1e-12 * max(1, abs(auto_h15)) + 1e-15, "hs[15].grad holds dh_30/dh_15 after hs[T].backward().")
    check("it matches the product formula", abs(auto_h15 - infl[15]) < 1e-9, "infl[15] from scalar_rnn_influence is the same quantity.")
```

   `auto_h15` is about 2.45e-07 (fifteen steps back, about 6 powers of ten lost).

3. **Step 3.3. Vanishing over time in a real RNN.** The scalar RNN is a toy. Does the same thing happen in a real `nn.RNN` with 32 hidden units and 8 inputs? We take a freshly initialised network (no training), feed it 20 random sequences, and take the sum of the final hidden state as a stand-in loss. Backpropagation then gives, for each time step, the gradient with respect to the input at that step; `grad_norms_per_step` returns the size (norm) of that gradient at every step, averaged over the sequences. The first block prints the gradient norm at the last step and at step 1 for T = 10, 30 and 60. The second block plots the logarithm of the norm against the number of steps back from the end, all three lengths in one figure.

```python
def grad_norms_per_step(T, hidden=32, n=20, seed=0):
    """Mean over n random sequences of the gradient norm of h_T (summed) with respect to the input at each step, for a fresh nn.RNN."""
    torch.manual_seed(seed)
    r = nn.RNN(8, hidden, batch_first=True)
    x = torch.randn(n, T, 8, requires_grad=True)
    out, _ = r(x)
    out[:, -1].sum().backward()
    return x.grad.norm(dim=2).mean(0).numpy()         # one number per time step

g_by_T = {T: grad_norms_per_step(T) for T in [10, 30, 60]}
for T, g in g_by_T.items():
    print(f"T = {T:>2}: gradient norm at the last step {g[-1]:.3e} | at step 1 {g[0]:.3e} | ratio {g[0] / g[-1]:.1e}")
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
T = 10: gradient norm at the last step 1.941e+00 | at step 1 3.951e-03 | ratio 2.0e-03
T = 30: gradient norm at the last step 1.973e+00 | at step 1 1.848e-08 | ratio 9.4e-09
T = 60: gradient norm at the last step 1.960e+00 | at step 1 1.342e-16 | ratio 6.8e-17
```

```python
plt.figure(figsize=(6.5, 3.4))
for T, c in zip([10, 30, 60], ["tab:blue", "tab:orange", "tab:red"]):
    g = g_by_T[T]
    plt.plot(np.arange(T)[::-1], np.log10(g), "o-", ms=3, color=c, label=f"T = {T}")
plt.xlabel("steps back from the last step"); plt.ylabel("log10 of gradient norm"); plt.title("A fresh nn.RNN: the early steps get almost no gradient")
plt.legend(); plt.tight_layout(); plt.show()
```

   You should see that the gradient at the last step is about 1.9 to 2.0 whatever T is, but at step 1 it is **3.95e-03** for T = 10, **1.85e-08** for T = 30 and **1.34e-16** for T = 60: the ratios between step 1 and the last step are 2e-03, 9e-09 and 7e-17. In the plot the three curves lie on top of one another: each is a straight line falling at about 0.28 powers of ten per step (for T = 60, 16.2 powers of ten over 59 steps), so a signal is multiplied by about 0.53 each step back. The line for T = 10 is the first part of the line for T = 60. The slope is a property of the weights, not of the sequence length; the length only decides how far down the line you must go. The early steps of a long sequence receive an essentially zero gradient: a weight cannot learn that symbol 1 matters if it never feels it. Without any training the picture is already bad; training adjusts the weights but cannot change the arithmetic of a long product.

   Tips and common mistakes:

   - The loss `out[:, -1].sum()` is a simple stand-in: any function of the last state would show the same decay. With a real classifier the gradient first passes through the linear layer, which only rescales it.
   - Each input vector has 8 numbers, so `x.grad.norm(dim=2)` takes the length of the 8-number gradient at each step of each sequence, and `.mean(0)` averages over the 20 sequences.
   - PyTorch initialises the weights of `nn.RNN` with small uniform numbers (between about -0.18 and 0.18 for 32 units). That is a small recurrent matrix, and small matrices make the decay fast.

   Try also (optional):

   - Change `hidden` to 128. Does a larger state change the slope?
   - Plot the gradient norm against the *absolute* time step instead of steps back. Where are the large gradients now?

   *Exercise 3.3:* for T = 60, store in `steps_to_1e3` the number of steps back from the last step at which the gradient norm first drops below one thousandth of the last step's gradient norm. Why it matters: it gives the 'memory horizon' of this untrained RNN in steps.

```python
steps_to_1e3 = None   # for T = 60: how many steps back from the last step does the gradient first drop below 1/1000 of the last step's gradient?
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if steps_to_1e3 is None:
    print("Fill in steps_to_1e3 above, then run this cell again.")
else:
    g = g_by_T[60]
    back = int(np.argmax(g[::-1] < g[-1] / 1000))
    check("steps_to_1e3 matches", steps_to_1e3 == back, "Reverse g (so index 0 is the last step) and find the first index where it is below g[-1] / 1000.")
    RESULTS["p3_steps_to_1e3"] = int(steps_to_1e3)
```

   `steps_to_1e3` is 11: after about 11 steps the signal is already a thousand times weaker.

4. **Step 3.4. The effect of the recurrent weight, exploding gradients and clipping.** If a product of numbers below 1 vanishes, a product of numbers above 1 should explode. The recurrent weight w decides which. The first block scans w = 0.5, 0.9, 1.1 and 1.5 in the scalar RNN and prints the influence of step 1 on step 30. The second block builds a **ReLU** RNN (`nonlinearity="relu"`, which does not squash its output) whose recurrent matrix is `scale` times an orthogonal matrix (so each step stretches the state by about `scale`), and prints the gradient norm at step 1 for scales 0.5, 1.0 and 1.5 and for T = 10, 30, 60 and 100. The third block takes the scale-1.5 network at T = 60, shows the norm of its parameter gradients before and after `clip_grad_norm_(..., max_norm=1.0)`. The last is the standard tool against exploding gradients, and it is already built into `train_seq_model`.

```python
print(f"{'w':>5} {'influence of step 1 on step 30':>32}")
w_table = {}
for w in [0.5, 0.9, 1.1, 1.5]:
    w_table[w] = scalar_rnn_influence(30, w)[2][1]
    print(f"{w:>5} {w_table[w]:>32.2e}")
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
    w   influence of step 1 on step 30
  0.5                         1.42e-18
  0.9                         8.72e-14
  1.1                         8.66e-13
  1.5                         2.09e-11
```

```python
def relu_rnn_grad(T, scale, hidden=32, seed=0):
    """Gradient norm (parameters) and input-gradient norm at step 1 for a ReLU RNN whose Wh is `scale` times an orthogonal matrix."""
    torch.manual_seed(seed)
    r = nn.RNN(8, hidden, nonlinearity="relu", batch_first=True)
    Q, _ = torch.linalg.qr(torch.randn(hidden, hidden))
    with torch.no_grad():
        r.weight_hh_l0.copy_(scale * Q)
    x = torch.randn(1, T, 8, requires_grad=True)
    out, _ = r(x)
    out[:, -1].sum().backward()
    return r, x.grad.norm(dim=2)[0, 0].item()

print(f"{'scale of Wh':>12} {'T':>4} {'gradient norm at step 1':>25}")
for scale in [0.5, 1.0, 1.5]:
    for T in [10, 30, 60, 100]:
        _, g1 = relu_rnn_grad(T, scale)
        print(f"{scale:>12} {T:>4} {g1:>25.3e}")
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
 scale of Wh    T   gradient norm at step 1
         0.5   10                 4.363e-05
         0.5   30                 4.878e-15
         0.5   60                 0.000e+00
         0.5  100                 0.000e+00
         1.0   10                 1.284e-02
         1.0   30                 9.859e-07
         1.0   60                 1.306e-12
         1.0  100                 8.395e-21
         1.5   10                 6.509e-01
         1.5   30                 8.936e-01
         1.5   60                 1.176e+01
         1.5  100                 8.529e+00
```

```python
r_big, _ = relu_rnn_grad(60, 1.5)
before = nn.utils.clip_grad_norm_(r_big.parameters(), max_norm=1.0)       # returns the norm BEFORE clipping
after = torch.sqrt(sum((p.grad ** 2).sum() for p in r_big.parameters()))
print(f"parameter-gradient norm before clipping: {float(before):.3e}")
print(f"parameter-gradient norm after clipping to 1.0: {float(after):.3f}")
assert abs(float(after) - 1.0) < 1e-4
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
parameter-gradient norm before clipping: 1.180e+03
parameter-gradient norm after clipping to 1.0: 1.000
```

   You should see, for the scalar tanh RNN at T = 30: 1.42e-18 for w = 0.5, 8.72e-14 for w = 0.9, 8.66e-13 for w = 1.1 and 2.09e-11 for w = 1.5. A bigger w helps (more than seven powers of ten between 0.5 and 1.5), but even w = 1.5 does not explode: the influence is still tiny. The reason is **saturation**: with a large w the state is pushed toward plus or minus 1, where the tanh derivative (1 minus h squared) is almost 0, and this factor wins over w. The tanh keeps the scalar case bounded from above. The ReLU network has no such guard. With scale 0.5 the gradient at step 1 is 4.4e-05 at T = 10, 4.9e-15 at T = 30, and exactly 0.0 beyond that (every unit has been switched off or the number underflowed). With scale 1.0 it vanishes too: 1.3e-02, 9.9e-07, 1.3e-12 and 8.4e-21 for T = 10, 30, 60 and 100, because ReLU switches off about half of the units at every step. With scale 1.5 it does the opposite: 0.65, 0.89, 11.8 and 8.5. It grows (not as fast as 1.5 to the power 59, again because half the units are off), and the **parameter gradients** of the scale-1.5 network at T = 60 have a norm of 1.18e+03. After `clip_grad_norm_` with `max_norm=1.0` the norm is exactly 1.000 (the `assert` checks this) and the direction is unchanged, only the length is cut. So there are two failure modes. Vanishing is silent and has no cheap fix. Exploding is loud and clipping solves it. The sweet spot of w is a knife edge, and this is the motivation for the LSTM in Part 4.

   Tips and common mistakes:

   - `torch.linalg.qr` of a random matrix gives an orthogonal matrix Q, which rotates without changing lengths. Multiplying by `scale` then stretches by exactly `scale`.
   - `clip_grad_norm_` returns the norm *before* clipping, so you can log how often clipping is active. To measure the norm after, we compute it again from the parameters' `.grad`.
   - The gradient at T = 100 for the scale-1.5 network (8.5) is smaller than at T = 60 (11.8). Growth is not smooth for a ReLU network: which units are on depends on the input. Do not expect a perfect geometric series.

   Try also (optional):

   - Try scale = 1.2 and T = 60. Is the gradient already large?
   - In the scalar RNN set `u = 0.05` (very small inputs) and scan the same four w values. The tanh now stays in its linear range; what happens for w = 1.5?

   *Exercise 3.4:* scan the recurrent weights in `w_grid` (0.5, 0.6, ..., 1.5) for the scalar tanh RNN at T = 30, and store in `first_w` the smallest w whose influence of step 1 on step 30 is at least 1e-12. Why it matters: it shows how large w must be before the memory is measurable at all, and how weak that is.

```python
w_grid = np.round(np.arange(0.5, 1.51, 0.1), 1)   # the recurrent weights to scan
first_w = None   # the SMALLEST w in w_grid whose influence of step 1 on step 30 is at least 1e-12 (u = 0.8, seed 0)
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if first_w is None:
    print("Fill in first_w above, then run this cell again.")
else:
    vals = {float(wv): scalar_rnn_influence(30, float(wv))[2][1] for wv in w_grid}
    truth = min(wv for wv, v in vals.items() if v >= 1e-12)
    check("first_w is the smallest w with influence >= 1e-12", abs(first_w - truth) < 1e-9, "Compute scalar_rnn_influence(30, w)[2][1] for every w in w_grid and keep the smallest w whose value is at least 1e-12.")
    print(f"influence at w = {truth}: {vals[truth]:.2e}; at the largest w in the grid (1.5): {vals[1.5]:.2e}")
    RESULTS["p3_first_w"] = float(first_w)
```

   `first_w` is 1.2 (the influence is 2.06e-12); at w = 1.1 it is 8.66e-13, just under the threshold.

5. **Step 3.5. Summary: relate the numbers to Week 3's 0.25 to the power n.** In Week 3 you saw that the derivative of the sigmoid is at most 0.25, so the gradient through n sigmoid layers is at most 0.25 to the power n. The RNN suffers from the same disease in time: the "layers" are the time steps, and the factor is w times (1 minus h squared). Write a short paragraph that compares the two, using these numbers, then compute the Week 3 bound for 10 and 30 layers in the exercise below.

   Here is a model paragraph. In Week 3 a chain of n sigmoid layers could pass at most 0.25 to the power n of the gradient back: 9.5e-07 for 10 layers and 8.7e-19 for 30. In the scalar RNN with w = 0.9 the influence of the first step was 1.1e-05 after 10 steps and 8.7e-14 after 30, and in the real `nn.RNN` the gradient at step 1 was 2e-03 and 9e-09 of the last step's. These are not the same numbers, since the RNN's factor is not 0.25 but about 0.5 per step, but the story is the same: a product of many factors below 1 shrinks exponentially with the number of factors. The difference is that in Week 3 the layers had *different* weights that could be helped by a better activation (ReLU), whereas the RNN applies the *same* weight at every step, so it cannot just 'use a different activation'. A fix must change the structure of the path itself: for instance a path on which the signal is carried forward almost unchanged. That is the idea of the LSTM cell state in Part 4, and of attention, which skips the path altogether, in Parts 5 and 6.

   *Exercise 3.5:* compute `bound_10` and `bound_30`, the bound 0.25 to the power n for a chain of 10 and of 30 sigmoid layers. Why it matters: you will compare these with the RNN numbers in your write-up.

```python
bound_10 = None   # 0.25 ** 10, the Week 3 bound for a 10-layer sigmoid chain
bound_30 = None   # 0.25 ** 30
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if bound_10 is None or bound_30 is None:
    print("Fill in both numbers above, then run this cell again.")
else:
    check("0.25 ** 10", abs(bound_10 - 0.25 ** 10) < 1e-12, "Use 0.25 ** 10.")
    check("0.25 ** 30", abs(bound_30 - 0.25 ** 30) < 1e-24, "Use 0.25 ** 30.")
    RESULTS["p3_bound_30"] = float(bound_30)
```

   `bound_10` is 9.5e-07 and `bound_30` is 8.7e-19.

### Part 4. LSTM: gates and a memory highway (30 min)

*Lesson section: "LSTM".* In Parts 2 and 3 you watched a plain RNN forget: the signal from step 1 is multiplied by a number smaller than one at every step, so after a few dozen steps nothing is left. The LSTM (long short-term memory) is the classic repair. It keeps a second, separate memory line called the **cell state** `c`, and it changes that line only by small, gated amounts: a **forget gate** decides how much of the old memory to keep, an **input gate** decides how much of a new candidate to write, and an **output gate** decides how much of the memory to show to the rest of the network. When the forget gate is close to 1, the old memory passes straight through, like a motorway with no traffic lights. In this part you write one LSTM step by hand in NumPy, check it against PyTorch, compute how much gradient survives on the highway, train an LSTM and an RNN on the memory task at four lengths, and open the trained LSTM to look at its gates. ==You will measure the contrast honestly, and you will see that it is smaller than the gradient numbers alone would suggest.==

Because this part must run on its own, it defines its own small models: the `SeqClassifier` below is the same idea as the RNN classifier of Part 2 (a recurrent layer over one-hot symbols, then the last hidden state into a linear layer), with a switch for the recurrent layer.

**Imagine first.** Before you run anything, make two predictions and write them down. (1) In Part 2 your RNN learned the memory task up to some length and then failed. An LSTM of the same size (32 hidden units) gets the same settings and the same training budget. What is the longest length among 10, 20, 40, 60 and 80 at which you think it still reaches 90 percent test accuracy, and do you expect it to go further than the RNN? (2) With a forget gate fixed at 0.95, the cell-state gradient from step 30 back to step 1 is the product of 29 numbers equal to 0.95. The RNN's gradient over the same distance was tiny (Part 3). Roughly how many times larger is the LSTM's: 10 times, 1,000 times, a million times, or 10 to the power 9? You will compare with measured numbers in Steps 4.3 and 4.4.

```python
# Imagine first: how far does an LSTM remember, and how much more gradient does its highway carry?
my_lstm_length = None      # 10, 20, 40, 60 or 80: the longest length where you think the LSTM reaches 90 percent
my_gradient_ratio = None   # your guess for (LSTM cell-state gradient) / (RNN gradient) after 30 steps, e.g. 1e3
```

1. **Step 4.1. One LSTM step by hand.** An LSTM step takes the current input `x`, the previous hidden state `h` and the previous cell state `c`, and computes four things from the same inputs: three gates `f`, `i`, `o` (each between 0 and 1, from a sigmoid) and a candidate `g` (between minus 1 and 1, from a tanh). Then `c_new = f * c + i * g` (keep part of the old memory, add part of the candidate) and `h_new = o * tanh(c_new)` (show part of the memory). The `*` is an element-wise product: each of the two units has its own gates. We use two inputs and two hidden units so that every matrix fits on the screen. The numbers are our own (they are not the lesson figure's), chosen so that the forget gate is high for unit 0 and lower for unit 1.

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Each gate has input weights W (2x2), recurrent weights U (2x2) and a bias b (2,)
P = {
    "f": (np.array([ [0.5, -0.3], [0.2, 0.4] ]), np.array([ [0.1, 0.2], [-0.2, 0.3] ]), np.array([2.0, 0.0])),
    "i": (np.array([ [0.6, 0.1], [-0.4, 0.5] ]), np.array([ [0.2, -0.1], [0.3, 0.1] ]), np.array([0.0, 0.0])),
    "o": (np.array([ [0.3, 0.3], [0.1, -0.2] ]), np.array([ [-0.1, 0.4], [0.2, 0.2] ]), np.array([0.5, 0.5])),
    "g": (np.array([ [0.8, -0.5], [0.3, 0.7] ]), np.array([ [0.3, 0.1], [-0.2, 0.4] ]), np.array([0.0, 0.0])),
}

def lstm_step(x, h, c, P):
    """One LSTM step in NumPy. Returns the new hidden state, the new cell state and a dict with the four gate vectors."""
    f = sigmoid(P["f"][0] @ x + P["f"][1] @ h + P["f"][2])     # forget gate
    i = sigmoid(P["i"][0] @ x + P["i"][1] @ h + P["i"][2])     # input gate
    o = sigmoid(P["o"][0] @ x + P["o"][1] @ h + P["o"][2])     # output gate
    g = np.tanh(P["g"][0] @ x + P["g"][1] @ h + P["g"][2])     # candidate memory
    c_new = f * c + i * g
    h_new = o * np.tanh(c_new)
    return h_new, c_new, {"f": f, "i": i, "o": o, "g": g}

x = np.array([1.0, -0.5]); h0 = np.array([0.2, -0.1]); c0 = np.array([0.8, -0.4])
h1, c1, gates = lstm_step(x, h0, c0, P)
for name, v in gates.items():
    print(f"gate {name} =", np.round(v, 3))
print("old cell c0 =", c0, "-> new cell c1 =", np.round(c1, 3))
print("new hidden h1 =", np.round(h1, 3))
# unit 0 by hand: c1[0] = f0 * c0[0] + i0 * g0
by_hand = gates["f"][0] * c0[0] + gates["i"][0] * gates["g"][0]
assert abs(by_hand - c1[0]) < 1e-12
print("unit 0 by hand:", round(float(by_hand), 4))
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
gate f = [0.934 0.483]
gate i = [0.646 0.354]
gate o = [0.643 0.673]
gate g = [ 0.8   -0.129]
old cell c0 = [ 0.8 -0.4] -> new cell c1 = [ 1.264 -0.239]
new hidden h1 = [ 0.548 -0.158]
unit 0 by hand: 1.2641
```

   **What you should see.**

   The forget gate of unit 0 is 0.934 (we gave it a bias of 2, so it starts out keeping most of its memory), while unit 1 has a forget gate of 0.483. The new cell of unit 0 is 1.264: it is 0.934 times the old 0.8 plus 0.646 times the candidate 0.800, exactly the line `f * c + i * g` written out by hand, and the assert passes. Notice that the old memory is *scaled*, not erased and rewritten: that is the whole idea. The hidden state `h1` is a filtered view of the cell (the output gate is between 0 and 1, so `h1` is smaller than `tanh(c1)`).

   Tips and common mistakes:

   - `f * c` is an element-wise product (NumPy `*`), not a matrix product (`@`). Mixing them up gives wrong shapes or silently wrong numbers.
   - The gates use a sigmoid (0 to 1: how much), the candidate uses a tanh (minus 1 to 1: what value). Using tanh for a gate would allow negative amounts.
   - All four computations read the *same* `x` and `h`; only the weights differ.

   Try also (optional):

   - Set `c0` to zeros and run again. How much of `c1` now comes from the input gate and the candidate alone?
   - Change the bias of the forget gate of unit 1 from 0 to 3. What happens to `c1[1]`?
   - Feed the output `h1, c1` back in as the next state with the same `x` and print `c2`. Does unit 0 keep growing?

   *Exercise 4.1:* Copy `P`, set the forget-gate bias of unit 0 to minus 6 (so `f` is almost 0) and compute the new `c` of unit 0 with `lstm_step`; then set it to plus 6 (so `f` is almost 1) and compute again. Store the two numbers in `c_erase` and `c_keep`. Why it matters: The forget gate is the knob that makes the memory last or vanish. Seeing its two extremes tells you what the network has to learn to remember something.

```python
c_erase = None     # YOUR CODE: c1[0] when the forget bias of unit 0 is -6
c_keep = None      # YOUR CODE: c1[0] when the forget bias of unit 0 is +6
if c_erase is None or c_keep is None:
    print("Fill in your two numbers above, then run this cell again.")
else:
    P_e = {k: tuple(a.copy() for a in v) for k, v in P.items()}; P_e["f"][2][0] = -6.0
    P_k = {k: tuple(a.copy() for a in v) for k, v in P.items()}; P_k["f"][2][0] = 6.0
    check("c_erase is about i*g (the old memory is gone)", abs(c_erase - lstm_step(x, h0, c0, P_e)[1][0]) < 1e-9, "set P_e['f'][2][0] = -6 and read c1[0]")
    check("c_keep: the old memory 0.8 passes through (c_keep minus c_erase is about 0.8)", abs(c_keep - lstm_step(x, h0, c0, P_k)[1][0]) < 1e-9 and abs((c_keep - c_erase) - 0.8) < 0.01, "with f near 1 the old 0.8 passes through, on top of the new i*g")
```

   *What it means.* The copies matter: `P_e = P` would change the original dictionary too, because both names would point to the same arrays. With a forget bias of minus 6 the gate is about 0.0025, so the old memory 0.8 is almost wiped and `c1[0]` is 0.521 (only the input gate times the candidate remains). With plus 6 the gate is about 0.9975 and `c1[0]` is 1.316. The difference between the two answers, 0.795, is the old memory 0.8 that now passes through; the part both share is the new input times candidate. A network that has to remember symbol 1 for many steps must learn forget gates near 1 on the units that store it.

2. **Step 4.2. Check it against PyTorch's LSTMCell.** You should never trust hand-written code until it agrees with a trusted implementation. PyTorch's `nn.LSTMCell(2, 2)` does exactly one step. Its weights are stored in four stacked blocks in the order **input, forget, cell (g), output**, so we copy our `i`, `f`, `g`, `o` matrices into the right row blocks. PyTorch has two bias vectors (`bias_ih` and `bias_hh`); we put our bias in the first and zeros in the second. Then we assert that PyTorch's `h` and `c` equal ours.

```python
H = 2
cell = nn.LSTMCell(2, H)
with torch.no_grad():
    for k, name in enumerate(["i", "f", "g", "o"]):          # PyTorch's block order
        W, U, b = P[name]
        cell.weight_ih[k * H:(k + 1) * H] = torch.tensor(W, dtype=torch.float32)
        cell.weight_hh[k * H:(k + 1) * H] = torch.tensor(U, dtype=torch.float32)
        cell.bias_ih[k * H:(k + 1) * H] = torch.tensor(b, dtype=torch.float32)
        cell.bias_hh[k * H:(k + 1) * H] = 0.0
    th, tc = cell(torch.tensor(x, dtype=torch.float32)[None], (torch.tensor(h0, dtype=torch.float32)[None], torch.tensor(c0, dtype=torch.float32)[None]))
print("PyTorch h =", np.round(th[0].numpy(), 4), "| ours =", np.round(h1, 4))
print("PyTorch c =", np.round(tc[0].numpy(), 4), "| ours =", np.round(c1, 4))
assert np.allclose(th[0].numpy(), h1, atol=1e-5) and np.allclose(tc[0].numpy(), c1, atol=1e-5)
n_formula = 4 * (H * 2 + H * H + H)
n_torch = sum(p.numel() for p in cell.parameters())
print("parameters by formula 4*(H*I + H*H + H):", n_formula, "| in PyTorch (two bias vectors):", n_torch)
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
PyTorch h = [ 0.5483 -0.1576] | ours = [ 0.5483 -0.1576]
PyTorch c = [ 1.2641 -0.2388] | ours = [ 1.2641 -0.2388]
parameters by formula 4*(H*I + H*H + H): 40 | in PyTorch (two bias vectors): 48
```

   **What you should see.**

   The two lines agree to four decimals and the assert passes: your NumPy step is the LSTM. The formula gives 40 parameters: four blocks, each with input weights (2 x 2), recurrent weights (2 x 2) and a bias (2). PyTorch counts 48 because it keeps two bias vectors per gate (they do the same job together, a historical detail), which adds 4 x 2 = 8. Compare with a plain RNN of the same size: one block of 2 x 2 + 2 x 2 + 2 = 10 parameters. **An LSTM has four times the parameters of an RNN of the same width** because it has four computations (three gates and a candidate) where the RNN has one.

   *Exercise 4.2:* Compute by formula the number of parameters of `nn.LSTM(8, 32)` and of `nn.RNN(8, 32)` the way PyTorch counts them (two bias vectors), store them in `n_lstm` and `n_rnn`, and then check with `sum(p.numel() ...)` on real modules. What is the ratio? Why it matters: The models of Step 4.4 use exactly these sizes (8 one-hot inputs, 32 hidden units). Knowing the cost of the gates in numbers helps you judge later whether the extra memory is worth it.

```python
n_lstm = None    # YOUR CODE: parameters of nn.LSTM(8, 32)
n_rnn = None     # YOUR CODE: parameters of nn.RNN(8, 32)
if n_lstm is None or n_rnn is None:
    print("Fill in your two numbers above, then run this cell again.")
else:
    real_l = sum(p.numel() for p in nn.LSTM(8, 32).parameters()); real_r = sum(p.numel() for p in nn.RNN(8, 32).parameters())
    check("parameters of nn.LSTM(8, 32)", n_lstm == real_l, "4 blocks of (32*8 + 32*32 + 32 + 32)")
    check("parameters of nn.RNN(8, 32)", n_rnn == real_r, "one block of (32*8 + 32*32 + 32 + 32)")
    print("ratio LSTM / RNN =", round(n_lstm / n_rnn, 2))
```

   *What it means.* One block has 32 x 8 = 256 input weights, 32 x 32 = 1,024 recurrent weights and two biases of 32: 1344 parameters, which is the RNN. The LSTM has four such blocks: 5376 parameters. The ratio is exactly 4. In a real network the extra parameters are the price of the gates; they are small here (a few thousand), but in a model with 1,000 hidden units the same ratio means millions.

3. **Step 4.3. The memory highway: how much gradient survives.** Why does the cell state help? Look at the update `c_t = f_t * c_(t-1) + i_t * g_t`. The direct path from `c_(t-1)` to `c_t` is multiplied only by the forget gate `f_t`. So the gradient that travels along the cell state from step T back to step t is the product of the forget gates in between. If the gate is constant, that product is `f` to the power `T - t`: for `f = 0.95` and `T = 30` it is 0.95 raised to the power 29. There is no `tanh'` factor and no recurrent weight matrix on this path, which is what killed the RNN's gradient in Part 3. The first block prints the product for four gate values and three sequence lengths. The second block recomputes the scalar-RNN influence of Part 3 with our own small helper (so that this part is self-contained) and plots it against the highway on a log scale.

```python
print(f"{'f':>5} {'T=10':>10} {'T=30':>10} {'T=60':>10}")
for f in [0.5, 0.9, 0.95, 0.99]:
    print(f"{f:5.2f} " + " ".join(f"{f ** (T - 1):10.3g}" for T in (10, 30, 60)))
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
    f       T=10       T=30       T=60
 0.50    0.00195   1.86e-09   1.73e-18
 0.90      0.387     0.0471      0.002
 0.95       0.63      0.226     0.0485
 0.99      0.914      0.747      0.553
```

   **What you should see.**

   Each row is `f` to the power `T - 1`, the gradient from the last step back to step 1. With `f = 0.5` the memory is gone in 10 steps (0.0020). With `f = 0.95` we still have 0.226 left at T = 30 (the lesson's 0.226) and 0.0485 at T = 60. With `f = 0.99` about 0.747 survives at T = 30 and 0.553 at T = 60. Compare with the RNN of Part 3, where the same distance left something like 10 to the power minus 11. ==A gate near 1 turns a vanishing gradient into a gentle slope.==

```python
def rnn_influence(T, w=0.9, u=0.8, seed=0):
    """Scalar RNN h_t = tanh(w*h_(t-1) + u*x_t) with random +-1 inputs.
    Returns d h_T / d h_t for t = 1..T: the product of w*(1 - h_k^2) for k = t+1..T (and 1 for t = T)."""
    xs = np.random.RandomState(seed).choice([-1.0, 1.0], size=T)
    hs, h = [], 0.0
    for t in range(T):
        h = np.tanh(w * h + u * xs[t]); hs.append(h)
    infl = np.ones(T)
    for t in range(T - 2, -1, -1):
        infl[t] = infl[t + 1] * w * (1 - hs[t + 1] ** 2)
    return infl

T = 60
steps_back = np.arange(T - 1, -1, -1)                    # distance from the last step
plt.figure(figsize=(7, 3.6))
plt.semilogy(steps_back, rnn_influence(T, w=0.9), "k", lw=2, label="RNN (w = 0.9)")
for f in (0.9, 0.95, 0.99):
    plt.semilogy(steps_back, f ** steps_back, label=f"LSTM cell state, f = {f}")
plt.gca().invert_xaxis(); plt.xlabel("step t (the last step T = 60 is on the right)"); plt.ylabel("gradient from step 60 back to step t")
plt.legend(fontsize=8); plt.title("Gradient reaching step t: RNN against the LSTM highway"); plt.tight_layout(); plt.show()
rnn_g = rnn_influence(30, w=0.9)[0]; lstm_g = 0.95 ** 29
print(f"T = 30: RNN {rnn_g:.2e} | LSTM (f = 0.95) {lstm_g:.3f} | ratio {lstm_g / rnn_g:.2e}")
RESULTS["p4_gradient_ratio_T30"] = float(lstm_g / rnn_g)
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
T = 30: RNN 8.72e-14 | LSTM (f = 0.95) 0.226 | ratio 2.59e+12
```

   **What you should see.**

   The black RNN curve drops almost vertically on the log scale; the LSTM curves are gentle lines (a straight line on a log scale means a constant loss factor per step, here `f`). At T = 30 the RNN keeps 8.72e-14 (your signs of the random inputs differ from the lesson's, so the exact number differs, but it is of the same tiny order as the lesson's 10 to the power minus 11), while the highway with `f = 0.95` keeps 0.226. The ratio is 2.6e+12: compare it with your Imagine-first guess. The highway is not magic, though: a gate of 0.95 still halves the signal about every 14 steps, which is why very long sequences are still hard.

   *Exercise 4.3:* A memory is practically lost when less than 1 percent of the gradient is left. For a constant forget gate, find with a short loop the smallest number of steps `k` such that `f ** k` is below 0.01, for `f = 0.9`, `f = 0.95` and `f = 0.99`. Store them in `k_09`, `k_95`, `k_99`. Why it matters: It turns the abstract number 0.95 into a distance: how many words the LSTM could in principle look back in a sentence.

```python
k_09 = None    # YOUR CODE: smallest k with 0.9 ** k < 0.01
k_95 = None    # YOUR CODE: the same for 0.95
k_99 = None    # YOUR CODE: the same for 0.99
def first_below(f, limit=0.01):
    k = 0
    while f ** k >= limit: k += 1
    return k
if None in (k_09, k_95, k_99):
    print("Fill in your three numbers above, then run this cell again.")
else:
    check("f = 0.9", k_09 == first_below(0.9), "keep multiplying by 0.9 until the product is below 0.01")
    check("f = 0.95", k_95 == first_below(0.95), "the same with 0.95")
    check("f = 0.99", k_99 == first_below(0.99), "the same with 0.99")
    assert k_09 < k_95 < k_99
```

   *What it means.* The loop multiplies by `f` until the product drops under 0.01. You can also solve it with logarithms: `ceil(log(0.01) / log(f))`. The answers are 44 steps for 0.9, 90 for 0.95 and 459 for 0.99. So a gate of 0.99 lets a signal travel about 459 steps before it falls under 1 percent. That is a long way, but a book has many thousands of words: the highway extends the reach of the memory, it does not remove the limit.

4. **Step 4.4. Train an LSTM and an RNN on the memory task.** Now the real test. We train `SeqClassifier` (a recurrent layer over the one-hot symbols, then the **last** hidden state into `nn.Linear(32, 8)`) with `nn.RNN` and with `nn.LSTM`, 32 hidden units each, on the memory task at lengths 10, 20, 40, 60 and 80: 1,000 training sequences, 300 test sequences, Adam with a learning rate of 0.003 (the setting of the RNN in Part 2; a larger step of 0.01 made the RNN unreliable there), 150 epochs, gradient clipping at 1 (all inside `train_seq_model`). The label is the first symbol, so the model must carry symbol 1 across the whole sequence. Chance is 0.125. For each run we record the final test accuracy, the best test accuracy over the epochs, and the first epoch with at least 0.9 test accuracy (the number of epochs it needed). One training at a time, so each cell takes about a minute on a CPU. We use the same data and seed for both models so the comparison is fair.

```python
class SeqClassifier(nn.Module):
    """A recurrent layer (RNN or LSTM) over one-hot symbols; the LAST hidden state goes into a linear layer."""
    def __init__(self, kind="lstm", hidden=32):
        super().__init__()
        self.rnn = (nn.LSTM if kind == "lstm" else nn.RNN)(N_SYMBOLS, hidden, batch_first=True)
        self.out = nn.Linear(hidden, N_SYMBOLS)
    def forward(self, x):
        o, _ = self.rnn(x)
        return self.out(o[:, -1])

def get_data(L):
    """1,000 training and 300 test sequences of length L (fixed seeds)."""
    Xtr, ytr = make_memory_task(1000, L, seed=1)
    Xte, yte = make_memory_task(300, L, seed=2)
    return Xtr, ytr, Xte, yte

LENGTHS = (10, 20, 40, 60, 80)
EPOCHS = 6 if QUICK else 150
ACC, MODELS, CURVES = {"rnn": {}, "lstm": {}}, {}, {}

def run_model(kind, L):
    torch.manual_seed(SEED)
    model = SeqClassifier(kind)
    hist = train_seq_model(model, *get_data(L), epochs=EPOCHS, lr=0.003, seed=SEED)
    te = hist["test_acc"]
    reach = next((e + 1 for e, a in enumerate(te) if a >= 0.9), None)
    ACC[kind][L] = {"final": te[-1], "best": max(te), "train": hist["train_acc"][-1], "epochs_to_90": reach}
    MODELS[(kind, L)], CURVES[(kind, L)] = model, hist
    print(f"{kind:4s} length {L:2d}: final test {te[-1]:.3f} | best {max(te):.3f} | train {hist['train_acc'][-1]:.3f} | first epoch with test >= 0.9: {reach}")

for L in LENGTHS:
    run_model("rnn", L)
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
rnn  length 10: final test 1.000 | best 1.000 | train 1.000 | first epoch with test >= 0.9: 9
rnn  length 20: final test 0.750 | best 0.780 | train 0.757 | first epoch with test >= 0.9: None
rnn  length 40: final test 1.000 | best 1.000 | train 1.000 | first epoch with test >= 0.9: 40
rnn  length 60: final test 0.140 | best 0.157 | train 0.527 | first epoch with test >= 0.9: None
rnn  length 80: final test 0.113 | best 0.170 | train 0.458 | first epoch with test >= 0.9: None
```

```python
for L in LENGTHS:
    run_model("lstm", L)
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
lstm length 10: final test 1.000 | best 1.000 | train 1.000 | first epoch with test >= 0.9: 24
lstm length 20: final test 1.000 | best 1.000 | train 1.000 | first epoch with test >= 0.9: 24
lstm length 40: final test 1.000 | best 1.000 | train 1.000 | first epoch with test >= 0.9: 45
lstm length 60: final test 0.107 | best 0.150 | train 0.745 | first epoch with test >= 0.9: None
lstm length 80: final test 0.143 | best 0.187 | train 0.731 | first epoch with test >= 0.9: None
```

```python
print(f"{'length':>6} {'RNN final':>10} {'RNN best':>9} {'LSTM final':>11} {'LSTM best':>10}")
for L in LENGTHS:
    r, l = ACC["rnn"][L], ACC["lstm"][L]
    print(f"{L:6d} {r['final']:10.3f} {r['best']:9.3f} {l['final']:11.3f} {l['best']:10.3f}")
plt.figure(figsize=(6.4, 3.6))
for kind, col in (("rnn", "tab:red"), ("lstm", "tab:blue")):
    plt.plot(list(ACC[kind]), [ACC[kind][L]["final"] for L in ACC[kind] ], "o-", color=col, label=kind.upper())
plt.axhline(1 / 8, color="gray", ls="--", label="chance (0.125)")
plt.xlabel("sequence length"); plt.ylabel("final test accuracy"); plt.ylim(0, 1.05); plt.legend(); plt.title("Memory task: RNN against LSTM"); plt.tight_layout(); plt.show()
RESULTS["p4_rnn_final"] = {L: round(ACC["rnn"][L]["final"], 3) for L in ACC["rnn"]}
RESULTS["p4_lstm_final"] = {L: round(ACC["lstm"][L]["final"], 3) for L in ACC["lstm"]}
if not QUICK:
    assert ACC["rnn"][10]["final"] > 0.9 and ACC["lstm"][10]["final"] > 0.9, "both models should learn the shortest length"
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
length  RNN final  RNN best  LSTM final  LSTM best
    10      1.000     1.000       1.000      1.000
    20      0.750     0.780       1.000      1.000
    40      1.000     1.000       1.000      1.000
    60      0.140     0.157       0.107      0.150
    80      0.113     0.170       0.143      0.187
```

   **What you should see.**

   **The RNN.** At length 10 the RNN learns the task (1.000 test accuracy, reaching 0.9 at epoch 9). At length 20 this run ends at 0.750 (training accuracy 0.757): it has not settled. Part 2 may show a different number for the same length, because near its limit the RNN's result depends on the hardware, the number of CPU threads and the random seed; that unreliability is itself the finding. At length 40 it reaches 1.000 (0.9 at epoch 40), but at length 60 the final test accuracy is 0.140 and at length 80 it is 0.113, near chance 0.125, with training accuracies of only 0.527 and 0.458: it cannot even fit the training set.

   **The LSTM.** It reaches 1.000 at length 10 (0.9 at epoch 24), 1.000 at length 20 (epoch 24) and 1.000 at length 40 (epoch 45). At length 60 the final test accuracy is 0.107 and at length 80 it is 0.143, with training accuracies of 0.745 and 0.731: it fits part of the training set (it partly memorises 1,000 sequences) without finding the rule 'copy the first symbol', and that does not carry over to new sequences.

   **What the contrast really is.** With these settings **the LSTM does not reach further than the RNN: both solve length 40 and both fail at length 60.** The differences are smaller: on this machine the LSTM learned every length up to 40 while the RNN wobbled at 20, and the LSTM needed more epochs at the shortest length (24 against 9), because it has four times the parameters and its gates have to be learned first. This is a less dramatic result than the gradient numbers of Step 4.3 (0.226 against 8.7e-14) might lead you to expect, and it is the honest one. A likely reason, which we did not test here, is that Adam divides each gradient by its running size, so a small gradient still produces a usable step, and clipping and a small learning rate keep training stable; the RNN's tiny gradient is slow and noisy, not absent. A bigger model, different settings or many more epochs may move both limits. What the LSTM does show clearly is *how* it keeps information: the cell-state product of Step 4.3 and the gates of Step 4.5.

   Tips and common mistakes:

   - Compare models with the same data, the same seed and the same budget; otherwise a difference may come from the setup and not from the model.
   - A failing run is not a bug. If the training accuracy is high and the test accuracy is near chance, the model memorised the training set; if both are near chance, it could not find the signal at all. Read `train` next to `final` in the printout.
   - `train_seq_model` returns lists with one entry per epoch. `hist['test_acc'][-1]` is the last epoch, `max(...)` is the best one.

   Try also (optional):

   - Train the LSTM at length 50 and find where between 40 and 60 it stops (and the RNN too).
   - Change the hidden size from 32 to 64 for the LSTM at length 60 (set `hidden=64` in `SeqClassifier`). Does it help?
   - Plot `CURVES[('lstm', 20)]['test_acc']` and `CURVES[('rnn', 20)]['test_acc']` against the epoch. Which one is steadier?

   *Exercise 4.4:* From the dictionary `ACC`, find (a) the longest length at which the LSTM's final test accuracy is at least 0.9 (`lstm_reach`), (b) the same for the RNN (`rnn_reach`), and (c) how many percentage points the LSTM beats the RNN by at length 20 (`gap20`, a number between 0 and 100, rounded to a whole number). Compare (a) with your Imagine-first guess. Why it matters: A table of accuracies is only useful if you can say in one sentence where each model stops working.

```python
lstm_reach = None   # YOUR CODE: the longest length with ACC["lstm"][L]["final"] >= 0.9
rnn_reach = None    # YOUR CODE: the same for the RNN
gap20 = None        # YOUR CODE: percentage points, LSTM minus RNN, at length 20
if None in (lstm_reach, rnn_reach, gap20):
    print("Fill in your three numbers above, then run this cell again.")
elif QUICK:
    print("(Quick test mode: the short training does not give the real numbers; skipping the check.)")
else:
    ok = lambda k: max([L for L in ACC[k] if ACC[k][L]["final"] >= 0.9], default=0)
    check("longest length for the LSTM", lstm_reach == ok("lstm"), "look for the largest L whose final accuracy is at least 0.9")
    check("longest length for the RNN", rnn_reach == ok("rnn"), "the same for the RNN")
    check("gap at length 20", abs(gap20 - 100 * (ACC["lstm"][20]["final"] - ACC["rnn"][20]["final"])) <= 1, "multiply the difference of the accuracies by 100")
```

   *What it means.* `max(..., default=0)` returns 0 when no length qualifies. The LSTM reaches 0.9 up to length 40 and the RNN up to length 40. At length 20 the LSTM is 25 percentage points better. If you guessed that the LSTM would go much further than the RNN, the measurement disagrees: the highway keeps the gradient large, but the model still has to *learn* to use it, and in this small budget both models give out between lengths 40 and 60.

5. **Step 4.5. Open the gates: what did the LSTM learn about forgetting?.** A trained LSTM is not a black box: we can recompute its gates ourselves. We take the LSTM trained at length 20 (stored in `MODELS`), read its weight matrices, and run 200 test sequences step by step with the equations of Step 4.1. For each step and each of the 32 hidden units we compute the forget gate `f_t` and the input gate `i_t`, then average over the 200 sequences. PyTorch stacks the gates in the order input, forget, cell, output, so `z.chunk(4, dim=1)` splits them. We assert that our loop reproduces the module's own last hidden state (so we know our gates are the real ones). Then we ask for each unit: if its forget gate stayed at its average value, what fraction of the cell content of step 1 would survive to step 20? That is the product of its mean forget gates from step 2 to step 20, the highway product of Step 4.3 for a gate that is not constant.

```python
lstm = MODELS[("lstm", 20)].rnn
W_ih, W_hh, b_ih, b_hh = (p.detach() for p in (lstm.weight_ih_l0, lstm.weight_hh_l0, lstm.bias_ih_l0, lstm.bias_hh_l0))
Xg = one_hot_batch(get_data(20)[2][:200])                  # 200 test sequences, shape (200, 20, 8)
h = torch.zeros(200, 32); c = torch.zeros(200, 32)
f_unit, i_unit = [], []                                    # per step: mean gate of each of the 32 units
with torch.no_grad():
    for t in range(20):
        z = Xg[:, t] @ W_ih.T + h @ W_hh.T + b_ih + b_hh
        i, f, g, o = z.chunk(4, dim=1)
        i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)
        c = f * c + i * g; h = o * torch.tanh(c)
        f_unit.append(f.mean(0).numpy()); i_unit.append(i.mean(0).numpy())
    ref, _ = lstm(Xg)
assert torch.allclose(h, ref[:, -1], atol=1e-5), "our gate loop must reproduce the module"
f_unit, i_unit = np.array(f_unit), np.array(i_unit)         # shape (20, 32)
f_steps, i_steps = f_unit.mean(1), i_unit.mean(1)           # mean over the 32 units
survive_unit = np.prod(f_unit[1:], axis=0)                  # steps 2..20, one number per unit
best = int(np.argmax(survive_unit))
print("mean forget gate per step (mean of 32 units):", np.round(f_steps, 2)[:10], "...")
print("sorted survival per unit (top 6):", np.round(np.sort(survive_unit)[::-1][:6], 3))
print(f"units with survival above 0.1: {int((survive_unit > 0.1).sum())} of 32 | best unit: {best}")
print(f"forget gate of unit {best}, steps 1-6:", np.round(f_unit[:6, best], 2), "| input gate, steps 1-6:", np.round(i_unit[:6, best], 2))
fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
ax[0].plot(range(1, 21), f_steps, "o-", label="forget gate, mean of the 32 units")
ax[0].plot(range(1, 21), f_unit[:, best], "s-", label=f"forget gate, unit {best}")
ax[0].plot(range(1, 21), i_unit[:, best], "^-", label=f"input gate, unit {best}")
ax[0].set_xlabel("step t"); ax[0].set_ylim(0, 1.05); ax[0].legend(fontsize=7); ax[0].set_title("Gates of the LSTM trained at length 20")
ax[1].bar(range(32), np.sort(survive_unit)[::-1]); ax[1].set_xlabel("units, sorted"); ax[1].set_ylabel("survival of step 1 until step 20"); ax[1].set_title("How much each unit keeps")
plt.tight_layout(); plt.show()
RESULTS["p4_forget_survive_best_unit"] = round(float(survive_unit.max()), 3)
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
mean forget gate per step (mean of 32 units): [0.68 0.74 0.81 0.84 0.83 0.82 0.8  0.8  0.79 0.79] ...
sorted survival per unit (top 6): [0.388 0.282 0.184 0.101 0.084 0.075]
units with survival above 0.1: 4 of 32 | best unit: 16
forget gate of unit 16, steps 1-6: [0.79 0.91 0.97 0.99 0.98 0.97] | input gate, steps 1-6: [0.75 0.87 0.94 0.97 0.96 0.95]
```

   **What you should see.**

   The assert passes, so these are the gates the trained module really used. The forget gate averaged over the 32 units and 20 steps is 0.79. If every unit sat at that value, what was written at step 1 would shrink to 1.2e-02 by step 20: nothing. But the units differ. Sorted by survival, the top six keep 0.388, 0.282, 0.184, 0.101, 0.084, 0.075 of what was written at step 1, and only 4 of the 32 units are above 0.1; all the others forget quickly. The best, unit 16, has a forget gate of 0.79 at step 1 and 0.93 or more from step 4 on. Its input gate is also high (0.75 or more at every step), so this unit keeps taking in new content while it holds its memory: it is not a clean 'write once, then lock' store, and its mixed content is read out by the final layer together with the other memory units. We do not claim more: the plot shows the gates of a few units, not what they contain, and the survival number uses the *average* gate, while the real gate changes with the sequence.

   *Exercise 4.5:* Using `survive_unit`, count how many units keep at least 10 percent of step 1 (`n_memory_units`) and find which unit keeps the most (`top_unit`). Then write one sentence in `my_reading` saying what the network does with the first symbol; separate what you saw from what you guess. Why it matters: Reading a gate plot is a skill: say what you see, then mark what you guess.

```python
n_memory_units = None   # YOUR CODE: how many units have survive_unit above 0.1
top_unit = None         # YOUR CODE: the index (0 to 31) of the unit with the largest survive_unit
my_reading = ""         # YOUR ANSWER: one sentence about what the network does with the first symbol
if n_memory_units is None or top_unit is None:
    print("Fill in your two numbers above, then run this cell again.")
else:
    check("number of memory units", n_memory_units == int((survive_unit > 0.1).sum()), "count with (survive_unit > 0.1).sum()")
    check("top unit", top_unit == int(np.argmax(survive_unit)), "np.argmax(survive_unit)")
    check("a sentence about step 1", len(my_reading.split()) >= 8, "write at least 8 words")
```

   *What it means.* `(survive_unit > 0.1)` is a list of True and False; summing it counts the Trues. 4 units keep at least 10 percent and unit 16 keeps the most. A good sentence says that the network seems to write the first symbol into a few cell units and protect them with forget gates near 1, and marks it as an interpretation: the plot shows the gates, not what the units contain.

6. **Step 4.6. Imagine and write: what an LSTM still cannot do.** In our small experiment the LSTM kept the first symbol across 40 steps, but not further than the plain RNN, and in real systems it reaches much further. It did not solve everything. Using what you measured in this part and what the lesson said, write three things an LSTM still cannot do well. Hints: think about the order in which it reads, about the size of its state, and about how far it can reach.

```python
lstm_limits = [
    "",    # 1. something about the order in which it computes (does step t need step t-1?)
    "",    # 2. something about the fixed size of the state (32 numbers for the whole past)
    "",    # 3. something about long range (your table of Step 4.4)
]
filled = [s for s in lstm_limits if len(s.split()) >= 8]
if len(filled) < 3:
    print(f"Write three sentences of at least 8 words each ({len(filled)} of 3 so far), then run this cell again.")
else:
    check("three limits written", True)
RESULTS["p4_limits"] = lstm_limits
```

   **What a good answer contains.** (1) *Sequential:* each step needs the previous state, so the steps of one sequence cannot be computed in parallel; training on long sequences is slow, and a GPU is used inefficiently. (2) *Fixed-size state:* the whole past has to be squeezed into the cell and hidden vectors, 32 numbers in this lab; as the text grows, something must be thrown away, a bottleneck. (3) *Long range:* the highway helps but is not perfect: forget gates below 1 still shrink the signal, and your table showed the LSTM unable to generalise at lengths 60 and 80 under the same budget. Attention, in Part 5, changes all three (all positions are computed in parallel, the whole past stays available, and any position is one step away), at a price: the n squared cost of Step 5.6.

### Part 5. Attention by hand (30 min)

*Lesson section: "Attention".* An RNN or an LSTM reads a sequence through a keyhole: everything it knows about the past must be squeezed into one state vector. **Attention** removes the keyhole. Every position gets a **query** (what am I looking for?), every position also offers a **key** (what do I contain?) and a **value** (what do I give you if you pick me?). The query is compared with all keys by a dot product, the scores are turned into weights by a softmax, and the output is the weighted sum of the values. Anything in the sequence, however far back, is one step away. In this part you compute all of that by hand in NumPy, wrap it in a function, compare it with PyTorch, draw the weights, find out why the scores are divided by the square root of `d_k`, and measure what attention costs. ==The whole mechanism is three matrix products and one softmax.==

**Imagine first.** Read the sentence **the cat sat on the mat** and put yourself in the place of the word *sat*. If you were allowed to look at every word of the sentence and give each a share of your attention (the shares add up to 100 percent), which words would you give the highest shares, and about how much to the best one? Think of what *sat* needs to know: who sat, and where. Write your prediction below. In Step 5.3 you will draw the weights of a tiny hand-made model and compare. (Careful: in that model the vectors are made up by us, so the result tests your reading of the mechanism, not a real language model.)

```python
# Imagine first: where does "sat" look in "the cat sat on the mat"?
my_top_words = []     # the words you would attend to most, in order, e.g. ["cat", "mat"]
my_top_share = None   # your guess for the largest share, between 0 and 1
```

1. **Step 5.1. Scores, weights and output for three tokens.** We start with three tokens and two-dimensional vectors, small enough to check with a calculator. The matrices are our own invention (they are not the lesson's figure, but the first query is built so that its three scores are 2, 1 and 0.1, the lesson's example, whose softmax is 0.659, 0.242, 0.099). Row `i` of `Q` is the query of token `i`; rows of `K` are the keys; rows of `V` are the values. The scores are `Q @ K.T` (row `i`, column `j`: how well query `i` matches key `j`). The weights are the softmax of each row, and the output is `weights @ V`: for every token, a mix of the values with those weights. We first use the raw scores, then divide them by `sqrt(d_k)` as the Transformer does (Step 5.4 explains why).

```python
np.set_printoptions(precision=3, suppress=True)
Q = np.array([ [1.0, 1.0], [0.0, 1.0], [1.0, 0.0] ])         # three queries, d_k = 2
K = np.array([ [1.2, 0.8], [0.6, 0.4], [0.05, 0.05] ])       # three keys
V = np.array([ [1.0, 0.0], [0.0, 1.0], [0.5, 0.5] ])         # three values

scores = Q @ K.T                                             # shape (3, 3)
print("scores = Q @ K.T:\n", scores)
w_raw = softmax_np(scores)                                   # softmax of each row, no scaling
print("weights without scaling, row 0:", w_raw[0])
d_k = Q.shape[1]
weights = softmax_np(scores / math.sqrt(d_k))                # with the 1/sqrt(d_k) scaling
print("weights with scaling:\n", weights)
print("row sums:", weights.sum(axis=1))
out = weights @ V
print("output = weights @ V:\n", out)
assert np.allclose(weights.sum(axis=1), 1.0)
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
scores = Q @ K.T:
 [ [2.   1.   0.1 ]
 [0.8  0.4  0.05]
 [1.2  0.6  0.05] ]
weights without scaling, row 0: [0.659 0.242 0.099]
weights with scaling:
 [ [0.57  0.281 0.149]
 [0.427 0.322 0.251]
 [0.477 0.312 0.211] ]
row sums: [1. 1. 1.]
output = weights @ V:
 [ [0.645 0.355]
 [0.553 0.447]
 [0.582 0.418] ]
```

   **What you should see.**

   The first row of scores is exactly [2.  1.  0.1]: query 0 matches key 0 best, key 1 half as well, key 2 hardly at all. Without scaling the softmax gives [0.659 0.242 0.099], the lesson's (0.659, 0.242, 0.099). With scaling by `sqrt(2)` = 1.414 the scores are smaller and the weights flatter: [0.57  0.281 0.149] (token 0 still wins, but with a smaller lead). Each row of weights sums to 1 (the assert passes), so each output row is a weighted average of the rows of `V`. Row 0 of the output is [0.645 0.355]: mostly value 0 (which is `[1, 0]`), mixed with some of values 1 and 2. Query 1, `[0, 1]`, matches key 0 with a score of 0.8 and key 1 with 0.4; query 2 behaves differently again. **Different queries read different mixtures from the same keys and values.**

   Tips and common mistakes:

   - `softmax_np(scores)` softmaxes along the last axis, that is, each row separately. A softmax over the whole matrix would make all nine numbers add up to 1, which is wrong.
   - `Q @ K.T` needs the transpose: both have shape (3, 2), and `(3, 2) @ (3, 2)` does not multiply. `K.T` has shape (2, 3), giving (3, 3).
   - Scores are not probabilities; the weights are. Only the weights add up to 1.

   Try also (optional):

   - Replace `V` by the identity matrix `np.eye(3)`. Then the output equals the weights. Why?
   - Make the first key a lot larger (multiply it by 5). What happens to row 0 of the weights?
   - Set all three keys equal. What do the weights become, and what is the output?

   *Exercise 5.1:* Using the unscaled weights of the first query, `w_raw[0]` = (0.659, 0.242, 0.099), compute the output row 0 by hand (with a calculator, or with Python as a calculator on the three numbers and the rows of `V`): `0.659 * [1, 0] + 0.242 * [0, 1] + 0.099 * [0.5, 0.5]`. Store the two numbers in `out_row0`, as a list of two numbers. Why it matters: The weighted sum is the entire reading step. If you can do it for one row by hand, you understand what the matrix product `weights @ V` does for every row.

```python
out_row0 = None   # YOUR CODE: [first number, second number], the unscaled output of query 0
if out_row0 is None:
    print("Fill in your two numbers above, then run this cell again.")
else:
    expected = softmax_np(scores[0]) @ V
    check("output row 0", np.allclose(out_row0, expected, atol=0.01), "0.659 * 1 + 0.242 * 0 + 0.099 * 0.5 for the first number")
```

   *What it means.* First number: 0.659 + 0 + 0.0495 = 0.7085. Second number: 0 + 0.242 + 0.0495 = 0.2915. The output is ([0.708 0.292]), about (0.709, 0.292). It leans towards value 0 because token 0 has the largest weight, but the other two values leave their mark in proportion to their weights. The sum of the two numbers is 1.00 here only because each value row happens to add up to 1 and the weights add up to 1.

2. **Step 5.2. Scaled dot-product attention as a function.** Now we package the three lines into a function, the one you will reuse in Parts 6 and 7: `attention(Q, K, V, scale=True)` returns the output and the weights. `Q` has shape (n, d_k), `K` has shape (m, d_k), `V` has shape (m, d_v); the scores have shape (n, m), and the output has shape (n, d_v). We test it on random matrices, compare it with PyTorch's `F.scaled_dot_product_attention` (which does the same computation and also works on GPUs), and print the shapes.

```python
def attention(Q, K, V, scale=True):
    """Scaled dot-product attention in NumPy. Q (n, d_k), K (m, d_k), V (m, d_v). Returns output (n, d_v) and weights (n, m)."""
    scores = Q @ K.T
    if scale:
        scores = scores / math.sqrt(Q.shape[-1])
    weights = softmax_np(scores, axis=-1)
    return weights @ V, weights

rng = np.random.RandomState(0)
Qr, Kr, Vr = rng.randn(5, 4), rng.randn(5, 4), rng.randn(5, 3)
out_np, w_np = attention(Qr, Kr, Vr)
out_t = F.scaled_dot_product_attention(torch.tensor(Qr), torch.tensor(Kr), torch.tensor(Vr)).numpy()
print("our output shape:", out_np.shape, "| weights shape:", w_np.shape)
print("maximum difference to PyTorch:", float(np.abs(out_np - out_t).max()))
assert np.allclose(out_np, out_t, atol=1e-6)
out3, w3 = attention(Q, K, V)                                # the Step 5.1 toy again
assert np.allclose(w3, weights) and np.allclose(out3, out)
print("the Step 5.1 toy gives the same weights and output as before")
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
our output shape: (5, 3) | weights shape: (5, 5)
maximum difference to PyTorch: 1.1102230246251565e-16
the Step 5.1 toy gives the same weights and output as before
```

   **What you should see.**

   The shapes are (5, 3) for the output (5 queries, each a vector of size `d_v` = 3) and (5, 5) for the weights (every query against every key). Our function and PyTorch's differ by 1.1e-16, which is floating-point rounding, so both asserts pass: you have written the same operation that sits inside every Transformer. The only difference from the Transformer is that there `Q`, `K` and `V` come from the input by three learned matrices (Part 7).

   *Exercise 5.2:* For a sequence of n = 7 tokens, with `d_k` = 4 and `d_v` = 3, write the shapes (as tuples) of the scores matrix `Q @ K.T`, of the weights and of the output. Then check by running `attention` on random matrices of these sizes. Why it matters: Most mistakes in attention code are shape mistakes. Reading the shapes is the quickest way to find them.

```python
shape_scores = None    # YOUR ANSWER: a tuple such as (2, 3)
shape_weights = None   # YOUR ANSWER
shape_output = None    # YOUR ANSWER
if None in (shape_scores, shape_weights, shape_output):
    print("Fill in your three shapes above, then run this cell again.")
else:
    r = np.random.RandomState(1)
    o_, w_ = attention(r.randn(7, 4), r.randn(7, 4), r.randn(7, 3))
    check("scores", tuple(shape_scores) == (7, 7), "(n, d_k) @ (d_k, n) gives (n, n)")
    check("weights", tuple(shape_weights) == w_.shape, "same shape as the scores")
    check("output", tuple(shape_output) == o_.shape, "(n, n) @ (n, d_v) gives (n, d_v)")
```

   *What it means.* The scores are (7, 4) times (4, 7), that is (7, 7). The weights keep that shape. The output is (7, 7) times (7, 3), that is (7, 3). Notice that the scores matrix is n by n: it grows with the square of the number of tokens. Step 5.6 measures what that costs.

3. **Step 5.3. Draw the weights of a six-word sentence.** A table of numbers is hard to read; a heat map is easy. Here is a toy model of the sentence **the cat sat on the mat**. We give every word a four-dimensional vector by hand: the four numbers loosely mean *animal-like*, *action-like*, *place-like* and *function word*. We use the vectors as queries, keys and values at the same time (`Q = K = V = X`), which is what *self*-attention means: the sentence attends to itself. We call `attention`, draw the weights with `show_attention`, and read the row of *sat*. The vectors are invented by us: they let you see the mechanism, they are not what a real model learned.

```python
E = {"the": [0.6, 0.0, 0.0, 2.0], "cat": [2.0, 0.6, 0.4, 0.0], "sat": [1.6, 2.0, 1.4, 0.0],
     "on": [0.0, 1.0, 1.6, 1.0], "mat": [0.4, 0.4, 2.0, 0.0]}     # [animal, action, place, function word]
words = ["the", "cat", "sat", "on", "the", "mat"]
X = np.array([E[w] for w in words])                               # shape (6, 4)
out6, W6 = attention(X, X, X)                                     # Q = K = V = X: self-attention
print("weights of 'sat' (row 2), in the order of the sentence:", [round(float(v_), 2) for v_ in W6[2] ])
plt.figure(figsize=(5.2, 4.4))
im = show_attention(W6, row_labels=words, col_labels=words, title="Who attends to whom (rows look at columns)")
plt.colorbar(im, fraction=0.046); plt.tight_layout(); plt.show()
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
weights of 'sat' (row 2), in the order of the sentence: [0.02, 0.12, 0.69, 0.08, 0.02, 0.08]
```

   **What you should see.**

   Each row is one word looking at the whole sentence. The row of *sat*, in the order the, cat, sat, on, the, mat, is [0.02, 0.12, 0.69, 0.08, 0.02, 0.08]: *sat* gives the largest share, 0.69, to **itself**, then 0.12 to *cat* and 0.08 to *mat* (the two words it is related to in our made-up vectors) and almost nothing to the two *the*. Two things to notice. First, the two *the* rows are identical: the same vector gives the same query and the same weights; the model cannot tell the first *the* from the second (no position information, a problem Part 6 and Week 8 solve). Second, with raw vectors a word is most similar to itself, so the diagonal is large. A real Transformer does not use the raw vectors: it first multiplies them by learned matrices `W_Q`, `W_K` and `W_V` (Part 7), so a verb can learn to look at its subject. Compare with your Imagine-first prediction: you probably put *cat* and *mat* high; the toy agrees about the order, but not about the size, because of the self-match.

   *Exercise 5.3:* From `W6`, find the word (other than *sat* itself) that *sat* attends to most (`top_other`, a string), and how much of its attention *sat* keeps on itself (`self_share`, rounded to two decimals). Why it matters: Reading an attention map correctly (rows are the looking words, columns the looked-at words) is the skill you need for every figure in the Transformer papers.

```python
top_other = ""        # YOUR ANSWER: the other word with the largest weight in the row of "sat"
self_share = None     # YOUR ANSWER: the weight of "sat" on itself, two decimals
if not top_other or self_share is None:
    print("Fill in your two answers above, then run this cell again.")
else:
    row = W6[2].copy(); row[2] = -1
    check("the other word", top_other == words[int(np.argmax(row))], "ignore the diagonal entry of the row of sat (index 2)")
    check("share on itself", abs(self_share - W6[2, 2]) < 0.006, "read the diagonal entry W6[2, 2]")
```

   *What it means.* Setting the diagonal entry to minus one removes *sat* from the contest, so `argmax` picks the best of the others. The answer is *cat* and *sat* keeps 0.69 of its attention on itself. The row is also a statement about what the output of *sat* contains: 69 percent of its own value and the rest from other words.

4. **Step 5.4. Why divide by the square root of d_k.** The scores are dot products of a query and a key. If their numbers are random with mean 0 and variance 1, a dot product of `d_k` such pairs is a sum of `d_k` products, each with variance 1, so the dot product has **variance `d_k`**: the larger the dimension, the larger the scores. Dividing by `sqrt(d_k)` brings the variance back to 1. We check this numerically: for five values of `d_k` we draw 2,000 random query and key pairs and measure the variance of `q . k` and of `q . k / sqrt(d_k)`.

```python
rng = np.random.RandomState(0)
dks = [4, 16, 64, 256, 1024]
var_raw, var_scaled = [], []
for dk in dks:
    q, k = rng.randn(2000, dk), rng.randn(2000, dk)
    dots = (q * k).sum(axis=1)                      # 2000 dot products q . k
    var_raw.append(dots.var()); var_scaled.append((dots / math.sqrt(dk)).var())
print(f"{'d_k':>6} {'var(q.k)':>10} {'var(q.k / sqrt(d_k))':>22}")
for dk, a, b in zip(dks, var_raw, var_scaled):
    print(f"{dk:6d} {a:10.1f} {b:22.3f}")
plt.figure(figsize=(5.6, 3.4))
plt.loglog(dks, var_raw, "o-", label="var(q.k)"); plt.loglog(dks, var_scaled, "s-", label="var(q.k / sqrt(d_k))")
plt.xlabel("d_k"); plt.ylabel("variance of the score"); plt.legend(); plt.title("Scores grow with d_k unless we scale"); plt.tight_layout(); plt.show()
RESULTS["p5_var_raw"] = [round(float(v), 1) for v in var_raw]
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
   d_k   var(q.k)   var(q.k / sqrt(d_k))
     4        3.9                  0.986
    16       16.2                  1.015
    64       62.2                  0.972
   256      252.7                  0.987
  1024     1075.8                  1.051
```

   **What you should see.**

   The variance of the raw dot product is 3.9 for `d_k` = 4, 16.2 for 16, 62.2 for 64, 252.7 for 256 and 1075.8 for 1024: it follows `d_k` (up to sampling noise, a few percent). After dividing by `sqrt(d_k)` the variance is 0.97 to 1.05 for every `d_k`, about 1. For `d_k` = 64 (the size used in the original Transformer) the standard deviation of the scores would be 8, instead of 1, without the scaling. Why does that matter? Step 5.5.

   *Exercise 5.4:* Before running anything, predict the variance of `q . k` for `d_k` = 512 and store it in `var_pred`. Then measure it with 2,000 random pairs (as above, with a new `RandomState`) and store it in `var_meas`. Why it matters: If you can predict the variance from the formula, you do not need to measure it for every size: that is what a good theory buys.

```python
var_pred = None    # YOUR ANSWER: what you expect, before measuring
var_meas = None    # YOUR CODE: the measured variance of q . k for d_k = 512
if var_pred is None or var_meas is None:
    print("Fill in your two numbers above, then run this cell again.")
else:
    check("prediction", abs(var_pred - 512) <= 1, "the variance of a sum of d_k products of unit variance is d_k")
    check("measurement is close to 512", 440 < var_meas < 590, "use q, k = rng.randn(2000, 512) and (q * k).sum(axis=1).var()")
```

   *What it means.* The prediction needs no code: 512. The measurement needs the same three lines as the cell above. The prediction is 512 and the measured value is 508.7. A measurement with 2,000 draws is not exact (the relative error of a variance estimate is about 3 percent here), but it lands close to the theory.

5. **Step 5.5. What goes wrong without the scaling.** A softmax of large numbers is almost a one-hot: the largest score takes nearly all the weight and the others get nearly none. That is bad in two ways. The attention becomes nearly *hard* (it cannot blend), and the gradient through the softmax vanishes: the slope of the softmax is tiny when one weight is near 1 (Part 3's vanishing gradients again, in another place). We measure both for `d_k` = 64 and five random keys. For 1,000 random draws we compute (1) the largest attention weight, with and without the division by `sqrt(d_k)`, and (2) the size (norm) of the gradient of the output with respect to the five numbers that enter the softmax, found by autograd.

```python
def one_draw(dk, scaled, rs):
    q = torch.tensor(rs.randn(dk)); K_ = torch.tensor(rs.randn(5, dk)); v = torch.tensor(rs.randn(5))
    z = K_ @ q                                            # the five raw scores q . k_j
    s = (z / math.sqrt(dk) if scaled else z).requires_grad_(True)     # the numbers that enter the softmax
    w = torch.softmax(s, dim=0)
    (w @ v).backward()                                    # a scalar output: the weighted sum of random values
    return w.max().item(), s.grad.norm().item()

rs = np.random.RandomState(0)
res = {}
for scaled in (False, True):
    draws = np.array([one_draw(64, scaled, rs) for _ in range(1000)])
    res[scaled] = draws.mean(axis=0)
    print(f"scaled={str(scaled):5s}: mean largest weight {res[scaled][0]:.3f} | mean gradient norm {res[scaled][1]:.4f}")
print("gradient ratio (scaled / unscaled):", round(res[True][1] / res[False][1], 1))
RESULTS["p5_saturation"] = {"max_w_unscaled": round(float(res[False][0]), 3), "max_w_scaled": round(float(res[True][0]), 3)}
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
scaled=False: mean largest weight 0.897 | mean gradient norm 0.1046
scaled=True : mean largest weight 0.470 | mean gradient norm 0.3552
gradient ratio (scaled / unscaled): 3.4
```

   **What you should see.**

   Without scaling the largest weight is 0.897 on average: nearly one-hot. With scaling it is 0.470: still a clear favourite among five keys (chance would be 0.2) but with room for the others. The gradient tells the same story: 0.1046 without scaling against 0.3552 with it, so the scaled softmax passes about 3.4 times more gradient back to the scores. Scaling is a cheap trick that keeps the softmax in its working range, as normalisation does elsewhere.

   *Exercise 5.5:* Repeat the weight measurement for `d_k` = 1024 without scaling, using 300 draws (use `one_draw(1024, False, rs2)` with a new `RandomState`). Store the mean largest weight in `maxw_1024`. Is it larger or smaller than for `d_k` = 64, and why? Why it matters: The problem gets worse with the width of the model, which is why the scaling was needed once models grew wide.

```python
maxw_1024 = None     # YOUR CODE: mean largest weight, d_k = 1024, no scaling, 300 draws
bigger = None        # YOUR ANSWER: True if it is larger than the d_k = 64 value, else False
if maxw_1024 is None or bigger is None:
    print("Fill in your two answers above, then run this cell again.")
else:
    check("the weight is nearly one-hot", maxw_1024 > res[False][0] - 0.01, "the largest of five unscaled weights should be close to 1")
    check("larger or smaller", bigger is True, "a wider d_k gives larger scores and an even sharper softmax")
```

   *What it means.* `one_draw` returns the pair (largest weight, gradient norm); we keep the first element. At `d_k` = 1024 the mean largest weight is 0.977, against 0.897 at `d_k` = 64: the scores have a standard deviation of 32, so the softmax is a one-hot almost every time.

6. **Step 5.6. The cost: n squared scores.** Attention has a price. The score matrix has `n * n` entries for `n` tokens, so doubling the length quadruples the work and the memory. We time `n` = 128 to 2048 with `d` = 64 on your CPU (the best of five repeats, in PyTorch) and print the number of scores and the memory of the score matrix in float32 (4 bytes per number). The time you measure depends on your computer, but the pattern does not. This is the lesson's `attn_cost` figure, with your own timings.

```python
d = 64
rows = []
for n in (128, 256, 512, 1024, 2048):
    Qt, Kt, Vt = torch.randn(n, d), torch.randn(n, d), torch.randn(n, d)
    best_t = 1e9
    for _ in range(5):
        t0 = time.perf_counter(); o_ = torch.softmax(Qt @ Kt.T / math.sqrt(d), dim=-1) @ Vt; best_t = min(best_t, time.perf_counter() - t0)
    rows.append((n, n * n, n * n * 4 / 1e6, best_t * 1000))
print(f"{'n':>5} {'scores n^2':>12} {'memory (MB)':>12} {'time (ms)':>10}")
for n, n2, mb, ms in rows:
    print(f"{n:5d} {n2:12,d} {mb:12.1f} {ms:10.2f}")
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ns = [r[0] for r in rows]
ax[0].plot(ns, [r[3] for r in rows], "o-"); ax[0].set_xlabel("n (tokens)"); ax[0].set_ylabel("time (ms)"); ax[0].set_title("time against n")
ax[1].plot([r[1] for r in rows], [r[3] for r in rows], "o-", color="C3"); ax[1].set_xlabel("n squared (scores)"); ax[1].set_title("time against n squared")
plt.tight_layout(); plt.show()
RESULTS["p5_cost_ms"] = {r[0]: round(r[3], 2) for r in rows}
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
    n   scores n^2  memory (MB)  time (ms)
  128       16,384          0.1       0.06
  256       65,536          0.3       0.14
  512      262,144          1.0       0.48
 1024    1,048,576          4.2       2.20
 2048    4,194,304         16.8       9.47
```

   **What you should see.**

   For n = 512 there are 262,144 scores (the lesson's 262,144) and the matrix takes 1.0 MB; for n = 2048 there are 4,194,304 scores and 16.8 MB, 16 times more for 4 times the length. The time grows too: 0.48 ms at n = 512, 2.20 ms at 1024 and 9.47 ms at 2048, so doubling n from 1024 to 2048 multiplied the time by 4.3 (an n-squared law alone predicts 4; the rest is probably memory effects, which we did not measure, and your own machine will give different numbers). Compare the two plots and judge how close to a straight line the right-hand one is. For one head in one layer this is small; a model with a long context, many heads and many layers pays it everywhere. That is why so much research in recent years is about making attention cheaper.

   *Exercise 5.6:* Compute the number of attention scores for n = 512 (`scores_512`) and for n = 4096 (`scores_4096`), how many times more that is (`times_more`), and the memory in megabytes of the score matrix in float32 for n = 4096 (`mb_4096`, use 4 bytes per number and 1 MB as one million bytes, one decimal). Why it matters: Numbers like these decide what a school server can run. Estimating them in your head before you start a job saves hours.

```python
scores_512 = None
scores_4096 = None
times_more = None
mb_4096 = None
if None in (scores_512, scores_4096, times_more, mb_4096):
    print("Fill in your four numbers above, then run this cell again.")
else:
    check("scores for n = 512", scores_512 == 262144, "n * n")
    check("scores for n = 4096", scores_4096 == 4096 ** 2, "n * n")
    check("ratio", abs(times_more - 64) < 0.5, "8 times longer means 8 * 8 times more scores")
    check("memory", abs(mb_4096 - 67.1) < 0.2, "16,777,216 numbers times 4 bytes, divided by one million")
```

   *What it means.* Eight times longer means 8 x 8 = 64 times more scores. 512 squared is 262,144, 4096 squared is 16,777,216, which is 64 times more, and the float32 matrix takes 67.1 MB, for a single head of a single layer.

7. **Step 5.7. Imagine and write: how could the word it find its noun?.** In the sentence **The trophy did not fit in the suitcase because it was too big**, a human knows that *it* means the trophy. Using what you learned in this part (query, key, value, dot product, softmax weights), write in three or four sentences how a model could learn to make the word *it* attend to *trophy*. Which vector plays which role? What would the weight on *trophy* have to be, compared with the weights on *suitcase* and *because*? What does the model have to learn for this to work (hint: look at the learned matrices that make the queries and keys)?

```python
my_pronoun_answer = ""    # YOUR ANSWER: three or four sentences
if len(my_pronoun_answer.split()) < 25:
    print("Write at least 25 words above, then run this cell again.")
else:
    check("answer written", True)
RESULTS["p5_pronoun"] = my_pronoun_answer
```

   **What a good answer contains.** The query of *it* asks, in effect, *which noun am I standing for?*; the keys of *trophy* and *suitcase* say *I am a noun that could be an antecedent*; the dot product of the query of *it* with each key gives a score, and the softmax turns the scores into weights. If the model has learned well, the weight on *trophy* is large (for example 0.7), and *suitcase* gets some (for example 0.2), the rest is shared by the other words. The output of *it* is then mostly the value of *trophy*: *it* now carries the meaning of the trophy. What has to be learned are the matrices `W_Q`, `W_K` and `W_V` that make queries and keys from the word vectors, trained on a very large amount of text so that this prediction helps. In the sentence with *small* instead of *big*, a good model attends to *suitcase*: the weights must depend on the whole sentence. Nobody writes the rule; the training finds it.

### Part 6. Hard against soft attention (25 min)

*Lesson section: "Hard vs soft".* Attention has to *choose* what to look at. There are two ways to choose. **Hard attention** picks exactly one position, the one with the largest score (an `argmax`), and copies its value: crisp and cheap, like reading one line of the letter. **Soft attention** takes a weighted average of all the values with the softmax weights: blurry, but smooth. They give similar answers when one score is far above the others, but they behave completely differently when you try to *train* them, because training needs gradients. In this part you compare the two on five scores, look at gradients, turn a temperature knob that slides from soft to hard, and then build the real thing: a small classifier with a learned query, which has to find the first symbol of a sequence of up to 80 symbols, where the RNN and the LSTM gave out. ==Soft attention is trained by gradient descent and it solves the task; hard attention hides the gradient and does not.==

This part defines its own models and helpers (it needs nothing from Parts 4 and 5 except, optionally, the RNN and LSTM results of Step 4.4 for a comparison table, which show a dash if you have not run them).

**Imagine first.** Suppose you replace the softmax in an attention layer with an `argmax`: the layer now picks exactly one position and ignores all the others. Before you run anything, answer two questions. (1) Can the learned query that decides *which* position to pick be improved by gradient descent through the argmax? Write `"yes"` or `"no"`. (2) In one sentence, why? (Think about what a tiny change in a score does to the output of an argmax.)

```python
# Imagine first: can we train through an argmax?
my_hard_trainable = None   # "yes" or "no"
my_hard_why = ""           # one sentence
```

1. **Step 6.1. Hard and soft side by side.** We take five positions with scores (2, 1, 0.1, -1, 0.5) and a value vector for each. `soft_attention` computes the softmax weights and the weighted sum of the values; `hard_attention` picks the position with the largest score and returns its value (the weights are one-hot). We print both. We also reproduce the lesson's example, where the scores (2, 1, 0.1) give the soft weights 0.659, 0.242, 0.099 at temperature 1.

```python
np.set_printoptions(precision=3, suppress=True)
scores5 = np.array([2.0, 1.0, 0.1, -1.0, 0.5])
V5 = np.array([ [1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [-1.0, 0.2], [0.3, 0.9] ])    # one 2-D value per position

def soft_attention(scores, V, T=1.0):
    """Softmax weights (temperature T) and the weighted sum of the values."""
    w = softmax_np(np.asarray(scores) / T)
    return w @ V, w

def hard_attention(scores, V):
    """Pick the position with the largest score: one-hot weights, and its value."""
    j = int(np.argmax(scores)); w = np.zeros(len(scores)); w[j] = 1.0
    return V[j], w

out_soft, w_soft = soft_attention(scores5, V5)
out_hard, w_hard = hard_attention(scores5, V5)
print("soft weights:", w_soft, "| sum", round(float(w_soft.sum()), 3), "| output", out_soft)
print("hard weights:", w_hard, "| output", out_hard)
print("the lesson's example (2, 1, 0.1):", softmax_np([2.0, 1.0, 0.1]))
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
soft weights: [0.559 0.205 0.084 0.028 0.125] | sum 1.0 | output [0.61  0.365]
hard weights: [1. 0. 0. 0. 0.] | output [1. 0.]
the lesson's example (2, 1, 0.1): [0.659 0.242 0.099]
```

   **What you should see.**

   The soft weights are [0.559 0.205 0.084 0.028 0.125]: position 0 has the largest share, 0.559, but even position 3 (score -1) keeps 0.028. The output [0.61  0.365] is a blend of all five values. The hard output is exactly the value of position 0, [1. 0.], as if the weights were (1, 0, 0, 0, 0). The lesson's example gives [0.659 0.242 0.099], as in the lesson. The two outputs are different but not far apart here, because position 0 has the highest score by a margin. When two scores are close, the soft output is a real mixture and the hard output jumps to one of them.

   Try also (optional):

   - Change the score of position 1 from 1.0 to 2.0 (a tie with position 0). What does `np.argmax` do with a tie? What do the soft weights do?
   - Multiply all the scores by 5 and print the soft weights. They get close to the hard ones. Why?
   - Add a sixth position with a very low score. Do the other weights change?

   *Exercise 6.1:* Change only the first score, keeping (1, 0.1, -1, 0.5) for the others, and find the smallest whole number `s_min` for which soft attention gives the first position a weight of at least 0.9. Use a loop over `s` = 1, 2, 3 ... and `soft_attention`. Why it matters: A soft attention is nearly hard only if the winning score leads by a wide margin. This number tells you how wide.

```python
s_min = None    # YOUR CODE: the smallest integer s with a first weight >= 0.9
if s_min is None:
    print("Fill in your number above, then run this cell again.")
else:
    ok_ = soft_attention([s_min, 1.0, 0.1, -1.0, 0.5], V5)[1][0] >= 0.9
    below_ = soft_attention([s_min - 1, 1.0, 0.1, -1.0, 0.5], V5)[1][0] < 0.9
    check("s_min gives at least 0.9", ok_, "try larger values of s")
    check("s_min is the smallest", below_, "one less should give less than 0.9")
```

   *What it means.* `next(...)` with a generator returns the first value that satisfies the condition. The answer is s = 4: the winner needs a lead of about 3 points over the runner-up before it takes 90 percent of the weight. In the lesson's terms: a softmax needs scores that are well separated to be nearly hard, and then its gradient is tiny (Step 6.3).

2. **Step 6.2. Gradients: soft has them, hard has none.** Training changes the scores by gradient descent, so we need the gradient of the output with respect to the scores: if I raise this score a little, how does the output change? We use PyTorch. We reduce the output to one number (the first component of the attended value) so that we have a scalar to call `.backward()` on. For soft attention the output depends smoothly on every score, so every score gets a gradient. For hard attention the output is `V[argmax(scores)]`: the scores only decide *which row* is copied. A tiny change of a score changes nothing (the same row is still copied) until it overtakes another score, and then the output jumps. The slope is zero almost everywhere, and PyTorch does not even attach a gradient: `argmax` returns an integer index.

```python
V5t = torch.tensor(V5, dtype=torch.float32)

s_soft = torch.tensor(scores5, dtype=torch.float32, requires_grad=True)
soft_loss = (torch.softmax(s_soft, dim=0) @ V5t)[0]          # first component of the soft output
soft_loss.backward()
print("soft: gradient of the output with respect to the scores:", s_soft.grad.numpy().round(3))

s_hard = torch.tensor(scores5, dtype=torch.float32, requires_grad=True)
hard_out = V5t[s_hard.argmax()][0]                           # first component of the hard output
print("hard: output depends on the scores through a graph?", hard_out.requires_grad, "| s_hard.grad is", s_hard.grad)
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
soft: gradient of the output with respect to the scores: [ 0.218 -0.125 -0.009 -0.045 -0.039]
hard: output depends on the scores through a graph? False | s_hard.grad is None
```

   **What you should see.**

   Soft attention returns a gradient for every one of the five scores: [ 0.218 -0.125 -0.009 -0.045 -0.039]. The signs make sense: the first component of the values is 1 at position 0 and 0 at position 1, so raising score 0 increases the output (gradient 0.218) and raising score 1 decreases it (gradient -0.125). For hard attention, `hard_out.requires_grad` is False: the computation graph was cut by `argmax`, and `s_hard.grad` is None. There is nothing to descend. A query that decides by argmax receives no learning signal from the loss.

   *Exercise 6.2:* For a scalar output `y = sum_j w_j * v_j` with `w = softmax(s)`, the gradient with respect to score `s_j` is `w_j * (v_j - y)`. Use the soft weights and the first component of the values to compute this gradient for all five positions by hand (with NumPy) and store it in `grad_hand` (a NumPy array of 5 numbers). Compare it with `s_soft.grad`. Why it matters: A formula that agrees with autograd is the best proof that you understand what autograd computes. Here it also shows that the gradient of a position is large only if it has a noticeable weight and a value different from the current output.

```python
grad_hand = None    # YOUR CODE: an array of 5 numbers, w_j * (v_j - y)
if grad_hand is None:
    print("Fill in your array above, then run this cell again.")
else:
    check("gradient equals autograd", np.allclose(np.asarray(grad_hand), s_soft.grad.numpy(), atol=1e-5), "use w = softmax_np(scores5), v = V5[:, 0], y = w @ v")
```

   *What it means.* The formula comes from the derivative of the softmax: the change of the output when `s_j` rises is the weight `w_j` times how much better `v_j` is than the average `y`. The array [ 0.218 -0.125 -0.009 -0.045 -0.039] equals the autograd result. Position 3 (score -1, value -1) has a weight of only 0.028, so its gradient -0.045 is small even though its value is very different from the output. A position with almost no weight learns almost nothing, which is a first hint of the saturation problem of the next step.

3. **Step 6.3. Temperature: sliding from soft to hard.** A softmax can be made sharper with a **temperature** `T`: the weights are `softmax(scores / T)`. At `T = 1` we have the ordinary softmax. A large `T` flattens the weights; a small `T` sharpens them until they are the hard one-hot. We scan `T` in (3, 1, 0.3, 0.1, 0.01) and measure (1) the largest weight, (2) the L1 distance to the hard one-hot weights (sum of the absolute differences; 0 means identical), and (3) the norm of the gradient of the output with respect to the five scores. The point is the combination: as the weights become hard, the gradient disappears.

```python
Ts = [3, 1, 0.3, 0.1, 0.01]
tab = []
for T in Ts:
    s_t = torch.tensor(scores5, dtype=torch.float64, requires_grad=True)
    w_t = torch.softmax(s_t / T, dim=0)
    (w_t @ torch.tensor(V5[:, 0])).backward()
    tab.append((T, w_t.max().item(), float(np.abs(w_t.detach().numpy() - w_hard).sum()), s_t.grad.norm().item()))
print(f"{'T':>5} {'max weight':>11} {'L1 to hard':>11} {'gradient norm':>14}")
for T, mx, l1, gn in tab:
    print(f"{T:5.2f} {mx:11.4f} {l1:11.4f} {gn:14.3g}")
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].semilogx(Ts, [r[2] for r in tab], "o-"); ax[0].set_xlabel("temperature T"); ax[0].set_ylabel("L1 distance to the hard weights"); ax[0].invert_xaxis()
ax[1].loglog(Ts, [r[3] for r in tab], "o-", color="C3"); ax[1].set_xlabel("temperature T"); ax[1].set_ylabel("gradient norm"); ax[1].invert_xaxis()
plt.tight_layout(); plt.show()
RESULTS["p6_temperature"] = {r[0]: (round(r[1], 3), float(f"{r[3]:.3g}")) for r in tab}
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
    T  max weight  L1 to hard  gradient norm
 3.00      0.3104      1.3792         0.0895
 1.00      0.5585      0.8829          0.258
 0.30      0.9576      0.0847          0.168
 0.10      1.0000      0.0001       0.000644
 0.01      1.0000      0.0000       3.72e-42
```

   **What you should see.**

   At `T = 3` the largest weight is 0.310: nearly uniform (one fifth would be 0.2), the distance to hard is 1.38 and the gradient is 0.090. At `T = 1` the largest weight is 0.559. At `T = 0.1` it is 1.0000, and the weights are almost the hard ones (distance 0.0001). At `T = 0.01` the soft weights equal the hard ones to 3.7e-44, **and the gradient norm has dropped to 3.7e-42**, essentially zero. This is the whole trade-off of the lesson in one table: the closer soft attention gets to hard attention, the less gradient it passes. A good attention layer lives in the middle, sharp enough to choose, soft enough to learn. (This is also what the `sqrt(d_k)` scaling of Step 5.5 protects.)

   *Exercise 6.3:* From the table `tab`, find the largest temperature in `Ts` for which the L1 distance to the hard weights is below 0.05 (`T_hard`), and the gradient norm at that temperature (`g_at_T`, three decimals). Why it matters: It tells you how close to hard attention you can get before learning stops.

```python
T_hard = None     # YOUR CODE: the largest T in Ts with L1 distance below 0.05
g_at_T = None     # YOUR CODE: the gradient norm at that T, three decimals
if T_hard is None or g_at_T is None:
    print("Fill in your two numbers above, then run this cell again.")
else:
    want = max(r for r in tab if r[2] < 0.05)
    check("temperature", T_hard == want[0], "the table has T in column 0 and the L1 distance in column 2")
    check("gradient norm", abs(g_at_T - want[3]) < 0.0006, "the gradient norm is column 3")
```

   *What it means.* Filtering the rows first and then taking the one with the largest `T` keeps the code short. The answer is T = 0.1, where the gradient norm is already only 0.001 (at T = 1 it was 0.258). A cold softmax is nearly hard and nearly gradient-free at the same time.

4. **Step 6.4. A learned query finds the first symbol.** Now the real task. Our `AttnClassifier` works like this. Each one-hot symbol goes through a linear layer into a vector of size 16. Then it has a **learned query** `q` (one vector of size 16), keys and values computed from the symbol vectors by two linear layers, scores `key . q / sqrt(16)`, a softmax over the positions, the weighted sum of the values, and a linear layer to the 8 classes. To answer the task, the query must find the first symbol. But think first: **what does each position know about itself?** Only its symbol. Position 1 and position 7 holding the same symbol look identical to the model, so the query cannot say *the first one*. This is the order problem. The remedy is to add a **position embedding**: a learned vector for position 1, another for position 2, and so on (`nn.Embedding(length, 16)`), added to each symbol vector. We first run the model *without* the position embedding to see it fail, then with it at lengths 10, 20, 40, 60 and 80, using the same data and training budget as Step 4.4 (the training is much quicker; 40 epochs).

```python
class AttnClassifier(nn.Module):
    """One learned query attends over the positions of a sequence. mode: 'soft' (softmax), 'hard' (argmax) or 'ste' (hard forward, soft backward)."""
    def __init__(self, length, d=16, use_pos=True, mode="soft"):
        super().__init__()
        self.d, self.use_pos, self.mode = d, use_pos, mode
        self.emb = nn.Linear(N_SYMBOLS, d, bias=False)          # symbol -> vector (the same as an embedding table)
        self.pos = nn.Embedding(length, d)                      # position -> vector
        self.Wk, self.Wv = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.q = nn.Parameter(torch.randn(d) * 0.5)             # the learned query
        self.out = nn.Linear(d, N_SYMBOLS)
        self.last_weights = None

    def forward(self, x):                                       # x: (batch, length, 8) one-hot
        h = self.emb(x)
        if self.use_pos:
            h = h + self.pos(torch.arange(x.shape[1]))
        scores = self.Wk(h) @ self.q / math.sqrt(self.d)        # (batch, length)
        w = torch.softmax(scores, dim=-1)
        if self.mode != "soft":
            w_hard = F.one_hot(scores.argmax(-1), x.shape[1]).float()
            w = w_hard + (w - w.detach()) if self.mode == "ste" else w_hard
        self.last_weights = w.detach()
        return self.out((w.unsqueeze(-1) * self.Wv(h)).sum(1))

def get_data(L):
    """1,000 training and 300 test sequences of length L (fixed seeds): the same data as Step 4.4."""
    return (*make_memory_task(1000, L, seed=1), *make_memory_task(300, L, seed=2))

if "ACC" not in globals():
    ACC = {"rnn": {}, "lstm": {}}                                # filled by Step 4.4 if you ran it
ACC["attn"], AMODELS = {}, {}
EPOCHS_A = 5 if QUICK else 40

def run_attn(L, use_pos=True, mode="soft", epochs=None):
    torch.manual_seed(SEED)
    model = AttnClassifier(L, use_pos=use_pos, mode=mode)
    hist = train_seq_model(model, *get_data(L), epochs=epochs or EPOCHS_A, lr=0.01, seed=SEED)
    te = hist["test_acc"]
    return model, hist, {"final": te[-1], "best": max(te), "train": hist["train_acc"][-1], "epochs_to_90": next((e + 1 for e, a in enumerate(te) if a >= 0.9), None)}

_, _, no_pos = run_attn(10, use_pos=False)
print(f"length 10 WITHOUT position embedding: final test {no_pos['final']:.3f}, train {no_pos['train']:.3f}")
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
length 10 WITHOUT position embedding: final test 0.323, train 0.323
```

```python
for L in (10, 20, 40, 60, 80):
    AMODELS[L], _, ACC["attn"][L] = run_attn(L)
    a = ACC["attn"][L]
    print(f"length {L:2d} with positions: final test {a['final']:.3f} | train {a['train']:.3f} | first epoch with test >= 0.9: {a['epochs_to_90']}")
n_attn = {L: sum(p.numel() for p in AMODELS[L].parameters()) for L in AMODELS}
print("parameters of AttnClassifier:", n_attn)
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
length 10 with positions: final test 1.000 | train 1.000 | first epoch with test >= 0.9: 2
length 20 with positions: final test 1.000 | train 1.000 | first epoch with test >= 0.9: 4
length 40 with positions: final test 1.000 | train 1.000 | first epoch with test >= 0.9: 4
length 60 with positions: final test 1.000 | train 1.000 | first epoch with test >= 0.9: 5
length 80 with positions: final test 1.000 | train 1.000 | first epoch with test >= 0.9: 5
parameters of AttnClassifier: {10: 952, 20: 1112, 40: 1432, 60: 1752, 80: 2072}
```

```python
dash = lambda kind, L: f"{ACC[kind][L]['final']:.3f}" if L in ACC.get(kind, {}) else "   -  "
print(f"{'length':>6} {'RNN':>7} {'LSTM':>7} {'attention':>10}")
for L in (10, 20, 40, 60, 80):
    print(f"{L:6d} {dash('rnn', L):>7} {dash('lstm', L):>7} {dash('attn', L):>10}")
plt.figure(figsize=(6.4, 3.6))
for kind, col in (("rnn", "tab:red"), ("lstm", "tab:blue"), ("attn", "tab:green")):
    if ACC.get(kind):
        plt.plot(list(ACC[kind]), [ACC[kind][L]["final"] for L in ACC[kind] ], "o-", color=col, label={"attn": "attention + positions"}.get(kind, kind.upper()))
plt.axhline(1 / 8, color="gray", ls="--", label="chance")
plt.xlabel("sequence length"); plt.ylabel("final test accuracy"); plt.ylim(0, 1.05); plt.legend(); plt.title("Memory task: three kinds of model"); plt.tight_layout(); plt.show()
RESULTS["p6_attn_final"] = {L: round(ACC["attn"][L]["final"], 3) for L in ACC["attn"]}
if not QUICK:
    assert all(ACC["attn"][L]["final"] > 0.95 for L in ACC["attn"]), "soft attention with positions should solve every length"
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
length     RNN    LSTM  attention
    10   1.000   1.000      1.000
    20   0.750   1.000      1.000
    40   1.000   1.000      1.000
    60   0.140   0.107      1.000
    80   0.113   0.143      1.000
```

   **What you should see.**

   **Without** a position embedding the model reaches only 0.323 test accuracy at length 10, far from 1.000 (chance is 0.125; it is above chance only because the first symbol is one of the ten symbols that the blurred average still contains). It cannot tell the first position from the others, exactly as predicted. **With** it, attention solves every length: 1.000 at 10, 1.000 at 20, 1.000 at 40 and 1.000 at 80, and it reaches 90 percent test accuracy in 5 epochs at length 80, while the RNN and the LSTM had failed at lengths 60 and 80. At length 10, where all three work, the attention model needs 2 epochs to reach 90 percent, against 9 for the RNN and 24 for the LSTM (from Step 4.4). The reason is the shape of the computation: the position-1 key is one step from the output, whatever the length. There is no chain of 80 multiplications to survive. The model is also small: 952 parameters at length 10 and 2,072 at length 80 (the growth is the position table, 16 numbers per position), against 5,640 for the LSTM of Step 4.4.

   Tips and common mistakes:

   - A position embedding is a table, so it has a fixed length: a model trained for 80 positions has no vector for position 81. Week 8 shows sinusoidal positions, which have no such limit.
   - Dropping the position embedding is not a bug in the code; it is a missing piece of information. When a model cannot do something, ask first whether it can *see* what it needs.
   - Compare with fair data: all three models use the same sequences and seeds (`get_data`).

   *Exercise 6.4:* Count by formula the parameters of `AttnClassifier(80)`: the symbol layer (8 x 16, no bias), the position table (80 x 16), the two linear layers `Wk` and `Wv` (16 x 16 each, no bias), the query (16) and the output layer (16 x 8 + 8). Store the total in `n_formula` and compare with `n_attn[80]`. Which part grows with the length? Why it matters: Knowing where the parameters are tells you what the model can and cannot do (here: there is no recurrent matrix at all).

```python
n_formula = None   # YOUR CODE: the total by formula
if n_formula is None:
    print("Fill in your number above, then run this cell again.")
else:
    check("matches the real count", n_formula == n_attn[80], "8*16 + 80*16 + 2*16*16 + 16 + (16*8 + 8)")
```

   *What it means.* The position table is the only part that depends on the length. The total is 2072: 128 (symbols) + 1,280 (positions) + 512 (keys and values) + 16 (query) + 136 (output). Only the position table grows with the length.

5. **Step 6.5. Look at the weights the model learned.** The model stores the attention weights of its last forward pass in `last_weights`. We feed it 300 test sequences of length 20 and plot the weights for the first three as bar charts (one bar per position), then measure the **mean weight on position 1** over all 300 sequences. If the model learned the rule, the bar at position 1 should be tall for every sequence, whatever the symbols are.

```python
m20 = AMODELS[20]; Xte20 = get_data(20)[2]
with torch.no_grad():
    m20(one_hot_batch(Xte20))
W20 = m20.last_weights.numpy()                          # (300, 20)
mean_first = float(W20[:, 0].mean())
print(f"mean attention weight on position 1 over 300 test sequences: {mean_first:.3f}")
print("minimum over the sequences:", round(float(W20[:, 0].min()), 3), "| uniform attention would give", 1 / 20)
fig, ax = plt.subplots(1, 3, figsize=(10, 2.8), sharey=True)
for k in range(3):
    ax[k].bar(range(1, 21), W20[k], color="tab:green"); ax[k].set_xlabel("position"); ax[k].set_title("sequence " + "".join(map(str, Xte20[k][:8])) + "...")
ax[0].set_ylabel("attention weight"); plt.tight_layout(); plt.show()
RESULTS["p6_mean_weight_position1"] = round(mean_first, 3)
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
mean attention weight on position 1 over 300 test sequences: 1.000
minimum over the sequences: 1.0 | uniform attention would give 0.05
```

   **What you should see.**

   The mean weight on position 1 is 1.0000, and the smallest weight over all 300 sequences is 0.9998; uniform attention would give 0.05. In the three bar charts one bar towers over the other 19. So **soft attention learned to behave almost like hard attention**: nearly all of the weight goes to the one position that holds the answer, but the softmax stayed smooth, and that is why gradient descent could find it. A hard-attention model could not have been trained by the same method. Notice also that the weight depends on the position only, not on the symbols: the same position gets the weight in every sequence.

   *Exercise 6.5:* For the 300 test sequences, compute the mean total weight that the model puts on the *last* position (`mean_last`) and on all positions other than the first (`mean_rest`). The weights of every sequence add up to 1, so `mean_first + mean_rest` should be 1. Why it matters: Checking that the weights add up is a quick test that you are reading the right axis of the array.

```python
mean_last = None    # YOUR CODE: mean of W20[:, -1]
mean_rest = None    # YOUR CODE: mean over sequences of the sum of the weights on positions 2..20
if mean_last is None or mean_rest is None:
    print("Fill in your two numbers above, then run this cell again.")
else:
    check("the first and the rest add up to 1", abs(mean_first + mean_rest - 1) < 1e-4, "W20[:, 1:].sum(axis=1).mean()")
    check("last position", abs(mean_last - W20[:, -1].mean()) < 1e-6, "W20[:, -1].mean()")
```

   *What it means.* The first axis is the sequence, the second is the position. The last position gets 0.0000 on average and all the others together 0.000: the model is not spreading its attention around; the remaining mass is spread thinly over many positions.

6. **Step 6.6. Try to train hard attention.** Now the experiment the Imagine-first question was about. We build the same model with `mode='hard'`: the softmax is replaced by a one-hot of the `argmax`. The query and the keys receive no gradient (we check `q.grad`), and the other parts of the model (the output layer, the value layer) can still learn something. We train it at length 10 and look at the loss. Then we try the **straight-through trick**: in the forward pass the weights are the hard one-hot, but in the backward pass the gradient flows as if they were the soft ones (`w_hard + (w_soft - w_soft.detach())`, which has the value of `w_hard` and the gradient of `w_soft`). We train that at lengths 10, 20 and 40 and report what happens, whatever it is.

```python
torch.manual_seed(SEED)
mh = AttnClassifier(10, mode="hard")
xb = one_hot_batch(get_data(10)[0][:64]); yb = torch.as_tensor(get_data(10)[1][:64])
F.cross_entropy(mh(xb), yb).backward()
print("hard mode: gradient of the query is", mh.q.grad, "| gradient of the output layer has norm", round(mh.out.weight.grad.norm().item(), 3))
EPH = 4 if QUICK else 60
mh, hist_h, acc_h = run_attn(10, mode="hard", epochs=EPH)
print(f"hard, length 10: loss {hist_h['loss'][0]:.3f} -> {hist_h['loss'][-1]:.3f} | final test {acc_h['final']:.3f}")
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
hard mode: gradient of the query is None | gradient of the output layer has norm 0.233
hard, length 10: loss 2.098 -> 2.059 | final test 0.107
```

```python
ste = {}
for L in (10, 20, 40):
    _, hist_s, ste[L] = run_attn(L, mode="ste", epochs=EPH)
    print(f"straight-through, length {L:2d}: loss {hist_s['loss'][0]:.3f} -> {hist_s['loss'][-1]:.3f} | final test {ste[L]['final']:.3f} | train {ste[L]['train']:.3f}")
RESULTS["p6_hard"] = {"hard_10": round(acc_h["final"], 3), "ste": {L: round(ste[L]["final"], 3) for L in ste}}
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
straight-through, length 10: loss 2.113 -> 0.432 | final test 0.757 | train 0.770
straight-through, length 20: loss 2.114 -> 2.002 | final test 0.163 | train 0.206
straight-through, length 40: loss 2.109 -> 0.189 | final test 0.890 | train 0.887
```

   **What you should see.**

   In hard mode the query's gradient is None, as predicted: no learning signal reaches the part of the model that chooses. The loss moves from 2.098 to 2.059 (the starting value is about ln 8 = 2.079, the loss of a model that guesses) and the test accuracy is 0.107, near chance. The only parts that can change are the ones after the choice, and they cannot repair a choice that is random. With the straight-through trick, the model *can* learn at length 10 (test accuracy 0.757) but it is clearly less successful than soft attention (1.000), and the pattern is erratic: at length 20 it is 0.163 and at length 40 it is 0.890. We did not try other seeds, but a result that goes up and down with the length is what an unreliable gradient looks like. The backward pass pretends that the weights were soft, so the gradient tells the query to move towards positions that the hard forward pass does not use. Such tricks can work, with care and more budget, but they are an approximation, and plain soft attention solves the problem easily. That is the practical reason the Transformer uses soft attention.

   *Exercise 6.6:* Fill in `summary`, a dictionary with the final test accuracies of the length-10 runs (rounded to two decimals): `"soft"` from `ACC["attn"][10]`, `"hard"` from `acc_h`, `"ste"` from `ste[10]`. Then say in `lesson` (one sentence, at least 12 words) why soft attention is trainable and hard attention is not. Why it matters: Three numbers and one sentence: that is the answer you would give in an exam.

```python
summary = {"soft": None, "hard": None, "ste": None}   # YOUR CODE: replace each None by the rounded accuracy
lesson = ""                                              # YOUR ANSWER: one sentence
if None in summary.values() or len(lesson.split()) < 12:
    print("Fill in the three numbers and write the sentence above (at least 12 words), then run this cell again.")
else:
    check("soft", abs(summary["soft"] - ACC["attn"][10]["final"]) < 0.006, "ACC['attn'][10]['final']")
    check("hard", abs(summary["hard"] - acc_h["final"]) < 0.006, "acc_h['final']")
    check("straight-through", abs(summary["ste"] - ste[10]["final"]) < 0.006, "ste[10]['final']")
```

   *What it means.* Every number comes from a variable that the earlier cells created. The three accuracies are {'soft': 1.0, 'hard': 0.11, 'ste': 0.76}. A good sentence says that soft attention is a smooth (differentiable) function of the scores, so the loss has a gradient to the query, while the hard argmax is piecewise constant: its gradient is zero almost everywhere and undefined where it jumps.

### Part 7. A self-attention layer, design and reflect (15 min)

*Lesson sections: "Attention" and "Self-attention".* So far the query was one learned vector. In a **self-attention** layer *every* position produces its own query, key and value from the same input, by three learned matrices `W_Q`, `W_K` and `W_V`, and every position attends to every other. The output has the same length as the input: a new vector for each position that already mixes in information from the others. This is the layer Week 8 builds the Transformer block from. In this last part you write it as a PyTorch module, count its parameters and compare it with an LSTM layer, use it on the memory task and put all four models in one table, and then step back: you choose a model for a real sequence problem and answer four reflection questions. ==Write the layer once, understand it fully, and you will recognise it in every Transformer you meet.==

This part defines its own module, helpers and data function, so it runs after the Setup cells alone. The comparison table shows a dash for any model you did not train in Parts 4 and 6.

1. **Step 7.1. Write the SelfAttention module.** The layer takes `X` of shape `(batch, n, d_model)`. Three linear layers make `Q`, `K` and `V` of shape `(batch, n, d_k)`. The scores are `Q @ K^T / sqrt(d_k)`, shape `(batch, n, n)`; the weights are the softmax along the last axis; the output is `weights @ V`, shape `(batch, n, d_k)`. The module returns both the output and the weights, so that we can draw them. We test the shapes with a random batch, and count the parameters by formula (each linear layer has `d_model * d_k` weights and `d_k` biases) and by PyTorch, and compare with an `nn.LSTM` layer of the same width.

```python
class SelfAttention(nn.Module):
    """Single-head self-attention: every position makes a query, a key and a value from the same input."""
    def __init__(self, d_model, d_k):
        super().__init__()
        self.W_Q, self.W_K, self.W_V = nn.Linear(d_model, d_k), nn.Linear(d_model, d_k), nn.Linear(d_model, d_k)
        self.d_k = d_k
    def forward(self, X):                                        # X: (batch, n, d_model)
        Q, K, V = self.W_Q(X), self.W_K(X), self.W_V(X)          # each (batch, n, d_k)
        scores = Q @ K.transpose(-1, -2) / math.sqrt(self.d_k)   # (batch, n, n)
        weights = torch.softmax(scores, dim=-1)
        return weights @ V, weights                              # (batch, n, d_k) and (batch, n, n)

sa = SelfAttention(16, 16)
Xr = torch.randn(4, 10, 16)
o_sa, w_sa = sa(Xr)
print("input", tuple(Xr.shape), "-> output", tuple(o_sa.shape), "| weights", tuple(w_sa.shape))
assert o_sa.shape == (4, 10, 16) and w_sa.shape == (4, 10, 10)
assert torch.allclose(w_sa.sum(-1), torch.ones(4, 10), atol=1e-5)
n_sa = sum(p.numel() for p in sa.parameters())
n_formula = 3 * (16 * 16 + 16)
n_lstm_layer = sum(p.numel() for p in nn.LSTM(16, 16).parameters())
print("SelfAttention(16, 16): parameters", n_sa, "| by formula 3*(d_model*d_k + d_k) =", n_formula)
print("nn.LSTM(16, 16) layer: parameters", n_lstm_layer, "| ratio LSTM / self-attention =", round(n_lstm_layer / n_sa, 2))
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
input (4, 10, 16) -> output (4, 10, 16) | weights (4, 10, 10)
SelfAttention(16, 16): parameters 816 | by formula 3*(d_model*d_k + d_k) = 816
nn.LSTM(16, 16) layer: parameters 2176 | ratio LSTM / self-attention = 2.67
```

   **What you should see.**

   The output has the same shape as the input in `batch` and `n`: (4, 10, 16); the weights are an `n` by `n` matrix for each sequence. The module has 816 parameters, equal to the formula, against 2176 for an LSTM layer of the same width: the LSTM has 2.7 times as many. Self-attention has no recurrent matrix and no gates, only three projections. Two things to notice for later: the number of parameters does *not* depend on the length `n` (the same three matrices serve every position, like a CNN kernel), but the work does (`n` by `n` scores, Step 5.6).

   *Exercise 7.1:* For `d_model = d_k = 64`, compute by formula the parameters of `SelfAttention(64, 64)` (`n_sa64`) and of `nn.LSTM(64, 64)` as PyTorch counts them, with two bias vectors (`n_lstm64`). Check them with real modules. Why it matters: In real models the layer is wide, and the cost in parameters matters. You can now calculate it without running anything.

```python
n_sa64 = None      # YOUR CODE: parameters of SelfAttention(64, 64)
n_lstm64 = None    # YOUR CODE: parameters of nn.LSTM(64, 64)
if n_sa64 is None or n_lstm64 is None:
    print("Fill in your two numbers above, then run this cell again.")
else:
    check("SelfAttention(64, 64)", n_sa64 == sum(p.numel() for p in SelfAttention(64, 64).parameters()), "3 * (64*64 + 64)")
    check("nn.LSTM(64, 64)", n_lstm64 == sum(p.numel() for p in nn.LSTM(64, 64).parameters()), "4 * (64*64 + 64*64 + 64 + 64)")
```

   *What it means.* The LSTM has four blocks of two weight matrices and two bias vectors; self-attention has three blocks of one weight matrix and one bias. 12,480 against 33,280: the LSTM layer is 2.7 times larger. (A real Transformer block adds an output projection, a feed-forward network and more in Week 8, so the whole block is larger than this one layer.)

2. **Step 7.2. Self-attention on the memory task, and one table for everything.** We use the module in a classifier: symbol embedding plus position embedding (as in Part 6), then the `SelfAttention` layer, then we read the output at the **last** position (which has attended to the whole sequence) and send it through a linear layer to the 8 classes. We train it at lengths 10, 40 and 80 with the same data as before and print one table with all four models. The last position's attention weights show which position it used.

```python
class SelfAttnClassifier(nn.Module):
    """Embedding + position embedding + self-attention; the output at the LAST position goes to a linear layer."""
    def __init__(self, length, d=16):
        super().__init__()
        self.emb, self.pos = nn.Linear(N_SYMBOLS, d, bias=False), nn.Embedding(length, d)
        self.att, self.out = SelfAttention(d, d), nn.Linear(d, N_SYMBOLS)
        self.last_weights = None
    def forward(self, x):
        h = self.emb(x) + self.pos(torch.arange(x.shape[1]))
        o, w = self.att(h)
        self.last_weights = w.detach()                  # (batch, n, n)
        return self.out(o[:, -1])

def get_data(L):
    """1,000 training and 300 test sequences of length L (fixed seeds): the same data as Parts 4 and 6."""
    return (*make_memory_task(1000, L, seed=1), *make_memory_task(300, L, seed=2))

if "ACC" not in globals():
    ACC = {"rnn": {}, "lstm": {}, "attn": {}}
ACC["self"], SMODELS = {}, {}
EPOCHS_S = 5 if QUICK else 60
for L in (10, 40, 80):
    torch.manual_seed(SEED)
    SMODELS[L] = SelfAttnClassifier(L)
    hist = train_seq_model(SMODELS[L], *get_data(L), epochs=EPOCHS_S, lr=0.01, seed=SEED)
    te = hist["test_acc"]
    ACC["self"][L] = {"final": te[-1], "train": hist["train_acc"][-1], "epochs_to_90": next((e + 1 for e, a in enumerate(te) if a >= 0.9), None)}
    print(f"self-attention, length {L:2d}: final test {te[-1]:.3f} | train {hist['train_acc'][-1]:.3f} | first epoch with test >= 0.9: {ACC['self'][L]['epochs_to_90']}")
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
self-attention, length 10: final test 1.000 | train 1.000 | first epoch with test >= 0.9: 2
self-attention, length 40: final test 1.000 | train 1.000 | first epoch with test >= 0.9: 4
self-attention, length 80: final test 1.000 | train 1.000 | first epoch with test >= 0.9: 5
```

```python
dash = lambda kind, L: f"{ACC[kind][L]['final']:.3f}" if L in ACC.get(kind, {}) else "-"
print(f"{'length':>6} {'RNN':>7} {'LSTM':>7} {'query attention':>16} {'self-attention':>15}")
for L in (10, 40, 80):
    print(f"{L:6d} {dash('rnn', L):>7} {dash('lstm', L):>7} {dash('attn', L):>16} {dash('self', L):>15}")
with torch.no_grad():
    SMODELS[40](one_hot_batch(get_data(40)[2]))
wl = SMODELS[40].last_weights[:, -1, :].numpy()          # the weights of the last position, (300, 40)
mean_on_first = float(wl[:, 0].mean())
print(f"length 40: the last position puts on average {mean_on_first:.3f} of its attention on position 1")
RESULTS["p7_self_final"] = {L: round(ACC["self"][L]["final"], 3) for L in ACC["self"]}
if not QUICK:
    assert all(ACC["self"][L]["final"] > 0.95 for L in ACC["self"]), "self-attention with positions should solve these lengths"
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
length     RNN    LSTM  query attention  self-attention
    10   1.000   1.000            1.000           1.000
    40   1.000   1.000            1.000           1.000
    80   0.113   0.143            1.000           1.000
length 40: the last position puts on average 1.000 of its attention on position 1
```

   **What you should see.**

   Self-attention reaches 1.000, 1.000 and 1.000 at lengths 10, 40 and 80. In the table, the RNN gets 1.000, 1.000 and 0.113 at lengths 10, 40 and 80, the LSTM gets 1.000, 1.000 and 0.143 (both reach length 40 with these settings and both fail at 80), and both attention models solve all three lengths. The last position puts on average 1.000 of its attention on position 1, which is the position that holds the answer: the layer found, by itself, which key to look at (the query comes from the last position, the key from the first, and the learned matrices and position embeddings made them match). Notice what we compare: all models are small, the training set is small (1,000 sequences) and the task is easy for attention because the answer sits at a fixed position. Real language is harder and the margin is not always this large, but the pattern (a path of length one from the answer to the output) is real.

   Tips and common mistakes:

   - Reading the last position is our design choice. You could also average the output over all positions; in a test run outside this lab that worked as well. Every position sees all positions here (there is no mask), so any position could in principle be read.
   - Do not read the table as a verdict on RNNs and LSTMs in general. It says what happens with *these* sizes, *this* data and *this* budget.

   *Exercise 7.2:* From `ACC`, build the list `solved_80` of the model names (`"rnn"`, `"lstm"`, `"attn"`, `"self"`) whose final test accuracy at length 80 is at least 0.9 (skip a model that is not in `ACC` for length 80). Then in `why` write one sentence (at least 15 words) that explains the result with the length of the path from the first symbol to the output. Why it matters: The explanation, not the table, is what you have learned.

```python
solved_80 = None    # YOUR CODE: a list of model names
why = ""            # YOUR ANSWER: one sentence
if solved_80 is None or len(why.split()) < 15:
    print("Fill in the list and a sentence of at least 15 words above, then run this cell again.")
elif QUICK:
    print("(Quick test mode: the short training does not give the real numbers; skipping the check.)")
else:
    truth = [k for k in ("rnn", "lstm", "attn", "self") if 80 in ACC.get(k, {}) and ACC[k][80]["final"] >= 0.9]
    check("models that solve length 80", sorted(solved_80) == sorted(truth), "loop over the four names and test ACC[name][80]['final'] >= 0.9")
```

   *What it means.* A loop with `in ACC.get(k, {})` skips models that were not trained. The models that reach 0.9 at length 80 are ['attn', 'self']. The path is the explanation: 79 recurrent steps for the RNN and the LSTM (each one a chance to lose the signal), one attention step for the other two.

3. **Step 7.3. Imagination design: choose a model for a real sequence problem.** Pick **one** real sequence problem from this list or invent your own: summarising a long homework message; the chat history of a school chatbot; a stream of typhoon-warning texts; a jeepney route log. Decide whether a plain RNN, an LSTM or an attention-based model (the Transformer family) is the best fit, and fill the design card below with your reasons. Use what you measured: how long is the typical input (in tokens)? How much of it must be remembered? Can the work be done in parallel on a GPU? What does the cost of `n` squared mean for your problem? There is no single right answer; a good card gives reasons that refer to numbers from this lab.

```python
design = {
    "problem": "",             # e.g. "answer a student's question using the last 20 messages of a school chatbot chat"
    "typical_length": None,    # your estimate of the input length in tokens (a number)
    "must_remember": "",       # what has to be remembered, and from how far back
    "model": "",               # "rnn", "lstm" or "attention"
    "reason_length": "",       # why, referring to the length and to Parts 3 to 6
    "reason_cost": "",         # what the choice costs: parallelism, memory, n squared
    "risk": "",                # what could go wrong, and how you would test it
}
missing = [k for k, v in design.items() if v in ("", None)]
if missing:
    print("Fill in these fields, then run this cell again:", ", ".join(missing))
else:
    check("model is one of the three", design["model"] in ("rnn", "lstm", "attention"), "write rnn, lstm or attention")
    print("=" * 60); print(" DESIGN CARD"); print("=" * 60)
    for k, v in design.items(): print(f" {k:16s} {v}")
    print("=" * 60)
RESULTS["p7_design"] = design
```

   **What a good card contains.** A concrete problem and an honest length (a chatbot history of 20 short messages is about 400 tokens; a long homework message may be 1,000 tokens or more). The model is chosen for reasons: for a few hundred tokens, attention is affordable (400 squared is only 160,000 scores) and it reaches any earlier message in one step; an LSTM is cheaper in memory for a very long stream (a typhoon-warning feed that never ends and must be processed token by token on a small device) but forgets old details; a plain RNN is rarely the right choice beyond about ten steps, as Part 4 showed. A good risk line says how you would find out whether the choice fails: for example, test on inputs where the key fact sits at the very beginning, as in the memory task.

4. **Step 7.4. Your results, together.** Everything you stored in `RESULTS` during Parts 4 to 7 is printed below. This is the summary to submit with your notebook or files. If a key is missing, you skipped the step that creates it.

```python
for key in sorted(RESULTS):
    print(f"{key:32s} {RESULTS[key]}")
print("\nkeys stored:", len(RESULTS))
```

   **What you should see.**

   You should see one line per stored result. Keys that start with `p4_` come from Part 4 (gradient ratio, the RNN and LSTM accuracies, the memory unit), `p5_` from Part 5 (variances, saturation, timings, your pronoun answer), `p6_` from Part 6 (temperature, attention accuracies, the weight on position 1, the hard-attention experiment) and `p7_` from this part. Check the numbers against your tables above and against the lesson.

5. **Step 7.5. Four reflection questions.** Answer in your own words, in at least 25 words each, using at least one number or example from this lab in each answer. There are no trick questions: the goal is to say clearly what you understood. If you cannot answer one, go back to the step named in brackets.

```python
reflection = {
    "1_rnn_forgets": "",        # Why does an RNN forget the start of a long sequence? (Parts 2 to 4)
    "2_lstm_gates": "",         # What do the gates and the cell state of an LSTM add, and where does it still fall short? (Part 4)
    "3_attention_reach": "",    # Why can attention reach any position in one step, and what does it cost? (Parts 5 to 7)
    "4_soft_hard": "",          # Why is soft attention trainable and hard attention not? (Part 6)
}
short = [k for k, v in reflection.items() if len(v.split()) < 25]
if len(short) == len(reflection):
    print("Write your four answers in `reflection` above (at least 25 words each), then run this cell again.")
else:
    check("all four answers have at least 25 words", not short, "still short: " + ", ".join(short))
RESULTS["p7_reflection"] = reflection
```

   **What a good answer contains.**

   - *Why an RNN forgets:* the information about symbol 1 passes through one multiplication by the recurrent weight and one tanh at every step; the gradient is a product of factors below 1 (about 8.7e-14 at 30 steps in Part 4), so learning from the first symbol is slow and erratic: with the same settings the RNN learned lengths 10 and 40, wobbled at 20, and failed at 60 and 80.
   - *What the LSTM adds:* a cell state with a forget gate near 1 that carries information almost untouched (the highway: 0.95 to the power 29 is 0.226); the gates are learned (a few units in Step 4.5 kept part of the first symbol); it still reads step by step, has a fixed-size state, and in this budget it reached no further than the RNN (length 40 solved, 60 and 80 failed).
   - *Attention's reach and cost:* every position computes scores against every other, so the first position is one step from the output at any length (the model solved length 80); the price is the n by n score matrix (262,144 scores for n = 512; 64 times more for 8 times the length).
   - *Soft against hard:* the softmax is a smooth function of the scores, so the loss has a gradient to the query; the argmax is piecewise constant, so the gradient is zero (no learning signal in Step 6.6); a straight-through trick is an approximation that only partly worked.

6. **Step 7.6. Go further.** If you want to see this week's attention inside a full model, read the first sections of the UvA Deep Learning tutorial *Transformers and Multi-Head Attention* (Tutorial 6, University of Amsterdam, by Phillip Lippe): `https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html`. Its first sections recap the scaled dot-product attention you wrote in Part 5; the multi-head part and the full Transformer are next week's topic, so you may stop there for now. Read it as a second explanation of the same idea, and see whether the shapes in its code match the shapes in your `SelfAttention`.

**Common mistakes** (read before you submit):

- Comparing models trained on different data or for a different number of epochs. Use `get_data` and the same seed for all of them.
- Reading a failing run without looking at the training accuracy: high training accuracy with chance test accuracy means memorising (partly the case for the LSTM at length 60), low training accuracy means it could not fit (the RNN at length 80).
- Softmaxing the whole score matrix instead of each row, or forgetting the transpose in `Q @ K.T`.
- Dividing by `d_k` instead of `sqrt(d_k)`, or forgetting the division altogether (Step 5.5 shows what it costs).
- Using attention without position information on a task that needs order: the model cannot know which symbol is first (Step 6.4).
- Reading attention weights with the axes swapped: rows are the looking tokens, columns the looked-at tokens.
- Calling `.backward()` through an `argmax` and expecting gradients (Step 6.2).
- Quoting numbers from a different seed or a shortened test run. The numbers in this lab are for `SEED = 42`.
- Writing reflection answers that do not cite a number or an example from your own output.

### Deliverables

A short checklist before you submit: every exercise has a written answer under its block; every `None` and every empty string in a starter cell has been replaced; the code blocks ran from the first to the last without an error; and the printed `RESULTS` summary of Step 7.4 is included.

Submit your work the way your instructor asks (the notebook with outputs, or your JupyterLab files with the printed output and your written answers). For Parts 4 to 7 it must include:

- **Part 4 (18 points):** the hand-written `lstm_step` with every gate printed and the erase and keep experiment (4.1), the agreement with `nn.LSTMCell` and the parameter counts (4.2), the highway table, the plot against the RNN curve and the memory lengths (4.3), the RNN and LSTM tables and plot at four lengths with the training accuracies and your Imagine-first guesses compared (4.4), the gate plots, the survival of each unit and your reading of them (4.5), and your three limits of the LSTM (4.6).
- **Part 5 (18 points):** the scores, weights and output of the three-token toy with row 0 by hand (5.1), the `attention` function checked against PyTorch with the shapes (5.2), the heat map of the six-word sentence and your reading of the row of *sat* compared with the Imagine-first prediction (5.3), the variance table and plot (5.4), the saturation measurements with and without scaling (5.5), the cost table and plot with the scores for n = 512 and n = 4096 (5.6), and your pronoun answer (5.7).
- **Part 6 (16 points):** the soft and hard outputs and the gradient by hand (6.1 and 6.2), the temperature table and plots (6.3), the position-embedding experiment with the table and plot of the three models (6.4), the learned attention weights with the mean weight on position 1 (6.5), and the hard-attention and straight-through experiments with your honest summary (6.6).
- **Part 7 (10 points):** the `SelfAttention` module with the shape and parameter checks (7.1), the four-model table (7.2), your design card (7.3), the printed `RESULTS` summary (7.4), and your four reflection answers (7.5).

Together with Parts 1 to 3 the lab is marked out of **100 points**: Part 1 is worth 10, Part 2 is worth 12, Part 3 is worth 16, Part 4 is worth 18, Part 5 is worth 18, Part 6 is worth 16 and Part 7 is worth 10. The rubric is in the section "How Week 7 is assessed". The work graded is the one your instructor specifies.

**Stretch goals.** Train the LSTM at length 60 with 5,000 training sequences (outside the limits of this lab, run it overnight or on a GPU) and see whether more data lets it find the rule; find out how large the hidden size must be for the RNN to learn length 20; add a causal mask to `SelfAttention` (set the scores above the diagonal to minus infinity before the softmax) and check that the first position then attends only to itself; make the memory task harder (the label is the symbol at position `k` for a `k` you choose) and see which models cope; plot the attention weights of the last position of `SelfAttnClassifier` as a function of the training epoch to see when the model discovers position 1; replace the learned position table by a sinusoidal one and compare.
<!-- LAB-END -->

## Check your understanding

{sub} Formative · Not graded · Unlimited attempts · Your best score is saved to your account

Fifteen questions cover the six outcomes. Nine of them need a short calculation (a product of factors, an RNN step, a cell-state update, a product of forget gates, softmax weights, a weighted output, a temperature effect, a scaled score and a number of scores), so keep a pencil nearby; the explanations show the arithmetic.

```quiz
{
  "id": "week-07-check",
  "title": "Week 7 quiz: Language models and attention",
  "passMark": 70,
  "questions": [
    {
      "q": "In a simple recurrent network, what is the only thing passed from one time step to the next?",
      "options": [
        "The whole list of tokens read so far",
        "A state vector that summarises what has been read",
        "The gradient of the loss",
        "A table of attention weights between all tokens"
      ],
      "answer": 1,
      "why": "The RNN carries a state vector h_t, updated from the old state and the new token. It does not store the tokens (so it cannot look back at them), and gradients and attention tables are not passed forward."
    },
    {
      "q": "During backpropagation through time, a gradient is multiplied by a factor of 0.5 at each of 10 steps. About what fraction of it is left?",
      "options": [
        "0.05, because 0.5 is divided by 10",
        "5.0, because 0.5 is multiplied by 10",
        "0.001, because 0.5 to the power 10 is 1/1,024",
        "Exactly 0, because it is multiplied by a number less than 1"
      ],
      "answer": 2,
      "why": "Ten factors multiply: 0.5¹⁰ = 1/1,024 ≈ 0.00098. It is small but not exactly zero, and it is a power, not a division or a product with 10. This is the same shrinking product as in Week 3, but along time."
    },
    {
      "q": "A one-unit RNN has h_(t−1) = 0.5, recurrent weight w = 0.9, input weight u = 0.8, input x = 1 and bias 0, so h_t = tanh(w·h_(t−1) + u·x). What is h_t?",
      "options": [
        "About 0.85, because tanh(0.45 + 0.8) = tanh(1.25)",
        "1.25, the value before the squashing",
        "About 0.42, because the input was ignored: tanh(0.45)",
        "Exactly 1.00, because tanh saturates"
      ],
      "answer": 0,
      "why": "0.9×0.5 + 0.8×1 = 1.25 and tanh(1.25) = 0.848. The value 1.25 forgets the tanh, 0.42 forgets the input, and tanh(1.25) is close to 1 but not equal to it."
    },
    {
      "q": "Self-attention (with no position information added) is given the words of “dog bites man” and then of “man bites dog”. What do you expect for the word “bites”?",
      "options": [
        "Different outputs, because the words are in different slots",
        "An error, since the same words appear twice",
        "Different outputs, because attention always prefers the first word",
        "The same output vector, because the layer sees a set of vectors without order"
      ],
      "answer": 3,
      "why": "Self-attention treats its input as a set: each word gets the same output whatever the order (in our example (1.432, 0.860) both times). Order must be added, by positional encoding in Week 8."
    },
    {
      "q": "What does the LSTM add to a plain RNN so that information can travel over many steps?",
      "options": [
        "A bigger vocabulary and a larger embedding matrix",
        "A cell state that is changed only by gated forgetting and writing, so signal and gradient pass with little change",
        "Attention over every earlier step, so the model can look back",
        "The removal of sequential computation, so all steps run at once"
      ],
      "answer": 1,
      "why": "The cell state c_t = f ⊙ c_(t−1) + i ⊙ g is a lane updated by gating and addition. Attention over earlier steps is a different mechanism, and the LSTM is still sequential."
    },
    {
      "q": "An LSTM cell has c_(t−1) = 2, forget gate f = 0.5, input gate i = 0.8 and candidate g = −0.5. What is c_t = f·c_(t−1) + i·g?",
      "options": [
        "0.6",
        "1.4",
        "−0.4",
        "1.0"
      ],
      "answer": 0,
      "why": "f·c_(t−1) = 0.5×2 = 1.0 and i·g = 0.8×(−0.5) = −0.4, so c_t = 1.0 − 0.4 = 0.6. The value 1.4 loses the sign of the candidate, −0.4 forgets the old cell and 1.0 forgets the new note."
    },
    {
      "q": "The forget gate of an LSTM is 0.9 at every one of 20 steps. About what fraction of the cell’s signal survives the 20 steps?",
      "options": [
        "18, because 0.9 is multiplied by 20",
        "0.9, because the gate acts once",
        "0.12, because 0.9 to the power 20 is about 0.1216",
        "3.6 × 10⁻¹¹, the number for the plain RNN"
      ],
      "answer": 2,
      "why": "The gradient along the cell is the product of the 20 forget gates, 0.9²⁰ = 0.1216. It is a power, not a product with 20, and it is far larger than the 3.6 × 10⁻¹¹ that the plain RNN kept over 30 steps."
    },
    {
      "q": "Which limit of an LSTM remains even though its gates protect its memory?",
      "options": [
        "It cannot be trained by gradient descent at all",
        "It forgets everything after 5 steps",
        "Its gradient always explodes on long inputs",
        "It still reads step by step and squeezes the whole past into a fixed-size state"
      ],
      "answer": 3,
      "why": "The LSTM is still sequential, and its state has a fixed size, so it is a bottleneck. It can be trained by gradient descent, it does not forget after 5 steps, and its gradient does not always explode."
    },
    {
      "q": "How does attention fix the bottleneck of the encoder–decoder translation model?",
      "options": [
        "The decoder can look back at all the encoder states and weigh them, instead of relying on one fixed vector",
        "The encoder is made much larger so that the vector holds more",
        "The decoder reads the source sentence backwards",
        "The context vector is replaced by a random vector"
      ],
      "answer": 0,
      "why": "Attention keeps one encoder state per source word and gives the decoder a weighted mix of them for each word it writes. A larger vector only delays the problem, and reversing or randomising does not give the decoder access to the details."
    },
    {
      "q": "Two positions have attention scores 1 and 0. What are the softmax weights?",
      "options": [
        "0.50 and 0.50",
        "0.73 and 0.27",
        "1.00 and 0.00",
        "0.67 and 0.33"
      ],
      "answer": 1,
      "why": "e¹ = 2.718 and e⁰ = 1, so the weights are 2.718/3.718 = 0.731 and 1/3.718 = 0.269. Equal weights would need equal scores, 1 and 0 would be hard attention, and 2/3 and 1/3 are just proportional to the scores, which is not how softmax works."
    },
    {
      "q": "Attention weights are (0.5, 0.3, 0.2) and the values are the numbers 10, 20 and 30. What is the attention output?",
      "options": [
        "10, the value with the largest weight",
        "20, the plain average of the values",
        "17, because 0.5×10 + 0.3×20 + 0.2×30 = 5 + 6 + 6",
        "60, the sum of the values"
      ],
      "answer": 2,
      "why": "The output is the weighted sum 5 + 6 + 6 = 17. Taking only the largest weight is hard attention, the plain average ignores the weights, and the plain sum does not use them either."
    },
    {
      "q": "Why can soft attention be trained by backpropagation while hard attention (an arg max) cannot, directly?",
      "options": [
        "The arg max is flat or jumps, so its derivative is 0 almost everywhere, while soft attention is smooth and has a slope",
        "Hard attention uses far more weights than soft attention",
        "Soft attention has no softmax, so it needs no derivative",
        "Hard attention is always less accurate than soft attention"
      ],
      "answer": 0,
      "why": "A small change in a score either changes nothing or flips the choice, so the derivative of an arg max is 0 where it exists. The softmax is smooth, so the chain rule from Week 3 applies. Accuracy and the number of weights are not the reason."
    },
    {
      "q": "Attention scores are (2, 1, 0.1). What happens to the soft weights when the temperature is lowered from T = 1 to T = 0.1?",
      "options": [
        "They become more even, near 0.33 each",
        "They stay the same and only the output is scaled",
        "The largest weight falls from 0.66 to about 0.5",
        "The largest weight rises from 0.66 to nearly 1, almost hard attention"
      ],
      "answer": 3,
      "why": "Dividing by T = 0.1 turns the scores into (20, 10, 1), whose softmax is (0.99995, 0.00005, 0.00000001). A smaller T sharpens the weights, and a larger T would flatten them."
    },
    {
      "q": "A query and a key have d_k = 64 numbers, and their raw dot product is 24. What is the scaled score used in the softmax?",
      "options": [
        "0.375, from 24 / 64",
        "3, from 24 / √64 = 24 / 8",
        "192, from 24 × 8",
        "6, from 24 / 4"
      ],
      "answer": 1,
      "why": "The scaling divides by √d_k = √64 = 8, giving 3. Dividing by 64 is too much, multiplying makes the scores larger instead of keeping them steady, and 4 would be the square root of 16, not 64."
    },
    {
      "q": "A text grows from 512 to 1,024 tokens. How does the number of attention scores in one attention table change (262,144 at n = 512)?",
      "options": [
        "It doubles, to 524,288",
        "It triples, to 786,432",
        "It becomes four times larger, 1,048,576",
        "It stays the same, because the table has a fixed size"
      ],
      "answer": 2,
      "why": "Every token is compared with every token, so there are n² scores: 1,024² = 1,048,576, which is 4 times 262,144. Doubling the length gives four times the scores, not twice."
    }
  ]
}
```

## How Week 7 is assessed

**Your lab (100 points).**

| Criterion | Points | Excellent | Satisfactory | Needs work |
|---|---|---|---|---|
| Part 1: Sequences and the memory task (data, last-symbol baseline, fixed window, growing window, far-back words in language) | 10 | The memory task made and five sequences printed, the class frequencies counted and the chance accuracy 1/8 explained; the last-symbol model trained and its near-chance accuracy explained; the window models for k = 1, 3, 5 and 10 trained and plotted, with only the full window seeing the first symbol and the link to n-grams made; the table of window size and parameters for lengths 5, 10, 20 and 40; three convincing examples of a far-back word in real language | Steps run and numbers are right; the explanation of the baseline or of the window is brief | Chance accuracy wrong or not explained; the window result not related to what the model can see; examples that do not need a far-back word |
| Part 2: An RNN from scratch (the recurrence by hand, weight sharing, the classifier, learning curves, where the information travels) | 12 | `rnn_forward` written and its three hidden states checked against `nn.RNN` with the same weights by assert; the parameter count by hand equal to the real count; the same recurrent matrix shown at every step and the same count for lengths 5 and 50; the classifier trained at lengths 5, 10 and 20 with an honest table and learning curves, naming the length at which it stops learning; a clear account of how many multiplications and tanh the first symbol passes at length 20 | Everything runs and the checks pass; the reading of the table or of the path of the information is general | The step by hand does not match PyTorch; the parameter count wrong; results reported that the printed numbers do not support |
| Part 3: Gradients through time (the product of factors, autograd check, a real RNN, the recurrent weight, clipping) | 16 | `scalar_rnn_influence` written with the influence at the last step equal to 1 by assert and compared with the lesson's figures (the same collapse, with your own input signal named); the product checked against autograd; the log gradient norm per time step plotted for an `nn.RNN` at T = 10, 30 and 60; the scan over w read, with the saturation of tanh explained and an exploding ReLU case tamed by gradient clipping; one paragraph linking the numbers to the 0.25 to the power n table of Week 3 | Everything runs and the asserts pass; the scan or the link to Week 3 is described but not explained | The product wrong or not verified; no real-RNN measurement; exploding and vanishing confused |
| Part 4: LSTM, gates and a memory highway (one step by hand, check against PyTorch, the highway, RNN against LSTM, the gates, limits) | 18 | `lstm_step` written with every gate printed and the erase and keep experiment explained; the step equal to `nn.LSTMCell` (40 parameters by formula, 48 in PyTorch, and 4 times an RNN's count); the highway table (0.226 at f = 0.95 and T = 30, 0.747 at f = 0.99) and the plot against the RNN's 8.7e-14; the RNN (1.000, 0.750, 1.000, 0.140, 0.113) and the LSTM (1.000, 1.000, 1.000, 0.107, 0.143) trained at lengths 10, 20, 40, 60 and 80 with the same data and settings, and the finding stated honestly (both solve length 40, both fail at 60, the LSTM is steadier, the RNN's result near its limit varies from run to run) with the training accuracies used to tell not fitting from memorising; the gates recomputed with the assert, the 4 of 32 units that keep at least 10 percent of step 1 found (the best keeps 0.388) and the claim about them kept to what the plot shows; three limits of the LSTM written | Everything runs and the checks pass; the reading of the table or of the gates is general or goes a little beyond the plot | `lstm_step` not matching PyTorch; the models compared with different data or budgets; the failures at lengths 60 and 80 reported without the training accuracy, or an LSTM advantage claimed that the table does not show; claims about the gates that the plots do not show |
| Part 5: Attention by hand (scores, weights, output, the function, the heat map, scaling, saturation, cost) | 18 | The three-token toy computed with scores (2, 1, 0.1) giving the weights 0.659, 0.242, 0.099, row 0 of the output by hand, and each row of weights summing to 1 by assert; `attention` equal to `F.scaled_dot_product_attention` with the shapes stated; the heat map of the six-word sentence read correctly (rows look, columns are looked at) and compared with the Imagine-first prediction, including the large weight of a word on itself; the variance of the scores growing like d_k (3.9 to 1075.8) and 1 after scaling; the largest weight 0.897 without scaling against 0.470 with it and the gradient comparison; the cost table with 262,144 scores at n = 512 and the plot; a clear account of how a pronoun finds its noun | Everything runs and the checks pass; the reading of the heat map, of the scaling or of the cost is brief | Softmax over the wrong axis or the transpose missing; no scaling experiment; scores and weights confused; times quoted with no mention that they depend on the machine |
| Part 6: Hard against soft attention (side by side, gradients, temperature, a learned query, the learned weights, training hard attention) | 16 | Soft and hard outputs printed and the gradient by hand equal to autograd, with the hard graph cut by `argmax` shown; the temperature table and plots read as a trade-off (the gradient norm falling to about 4e-42 as the weights become hard); the position-embedding experiment (0.323 without, 1.000 with, at every length from 10 to 80) and the three-model table and plot; the mean weight on position 1 measured; hard attention shown to give no gradient to the query (0.107 test accuracy, loss 2.098 to 2.059) and the straight-through result (0.757, 0.163, 0.890) reported honestly as erratic | Everything runs and the checks pass; the reading of the temperature table or of the hard-attention experiment is general | Gradients through the `argmax` expected or claimed; no position information and no explanation of why it is needed; the straight-through result reported as a success without the other lengths |
| Part 7: A self-attention layer, design and reflect (the module, the four-model table, design card, results, reflection) | 10 | `SelfAttention` written with the shapes and the parameter count checked (816 against 2,176 for an LSTM layer of the same width); self-attention trained at lengths 10, 40 and 80 and put in one table with the RNN, the LSTM and the query attention, with the path length from the first symbol used to explain it; a design card for a real sequence problem whose choice of model refers to length, memory, parallelism and the n squared cost; `RESULTS` printed; four reflection answers that cite numbers or examples from the lab | Everything present; the design card is generic or the answers are general | Module or table missing; a card with no reasons; reflection that does not use the lab's results |

**Across the course.** Laboratories and knowledge checks run through the term, with exams in Weeks 5, 10, 15 and 20, plus a capstone. Your instructor announces the component weights in the syllabus. The long-term goal is the NVIDIA-Certified Associate exam (NCA-GENL).

## Exit ticket and homework

Answer these three questions in one or two sentences each before you leave (your instructor will use them to open Week 8):
1. A forget gate stays at 0.97 for 29 steps. What fraction of the cell value from step 1 is left at step 30 (0.97²⁹)? Compare it with the 3.6 × 10⁻¹¹ that a plain RNN kept, and say in one sentence why they differ.
2. Scaled scores are (3, 1, 0) and the values are the numbers 6, 2 and 0. Find the three softmax weights and the attention output. Then say what changes if the scores are divided by a temperature T = 0.1.
3. A text has 2,000 tokens. How many scores are in one attention table? How many for 4,000 tokens? Say in one sentence what this means for long texts.

**Homework / self-study**
- Finish the lab, including the written answers and the Part 7 design.
- Re-read the `equation` figures in this lesson (the state and next-token probabilities, the RNN recurrence, the loss over time, the gradient through time, the four LSTM equations, the gradient along the cell, scaled dot-product attention, hard attention, temperature, Q, K and V, the self-attention layer and the √d_k scaling) and, for each, write one sentence in your own words about what every symbol means.
- Preview Week 8: the Transformer block. You saw what self-attention lacks (order, several patterns, a mask for the future); next week you meet positional encoding, multi-head attention and the causal mask, and put them together into one block.
